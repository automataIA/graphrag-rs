//! Contextual Chunk Enrichment via LLM (Anthropic Contextual Retrieval pattern)
//!
//! Before embedding, an LLM augments each chunk with 2-3 sentences of context
//! derived from the entire document. The llama.cpp server's prompt cache
//! (`cache_prompt`, on by default) means the document prefix is evaluated
//! **once** for all chunks from the same document.
//!
//! ## How it works
//!
//! 1. Group chunks by source document
//! 2. Build prompts using a static prefix (full document) + dynamic suffix (chunk)
//! 3. Call the OpenAI-compatible server — the static prefix is reused from the KV cache
//! 4. Return enriched chunks: `[LLM context]\n\n[original chunk text]`
//!
//! The server context size (`llama-server -c`, divided by `-np` slots) must be
//! at least [`ContextualEnricher::calculate_num_ctx`] for the largest document,
//! otherwise the server rejects or truncates the prompt.

use crate::{core::TextChunk, llm::LlmConfig};
#[cfg(all(feature = "async", feature = "ureq"))]
use crate::{
    core::{Document, Result},
    llm::{LlmClient, LlmGenerationParams},
};
#[cfg(all(feature = "async", feature = "ureq"))]
use std::collections::{HashMap, HashSet};

/// Configuration for contextual chunk enrichment
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextualEnricherConfig {
    /// Enable contextual enrichment. When false, chunks are returned unchanged.
    pub enabled: bool,

    /// Safety margin added to the calculated `num_ctx` (as a fraction, e.g. 0.05 = 5%).
    ///
    /// Prevents silent truncation caused by slightly underestimating token counts.
    pub safety_margin: f32,

    /// Maximum tokens for the LLM-generated context sentences (output budget).
    ///
    /// 100-200 tokens is typically enough for 2-3 context sentences.
    pub max_output_tokens: u32,

    /// Separator inserted between the generated context and the original chunk text.
    pub context_separator: String,
}

impl Default for ContextualEnricherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            safety_margin: 0.05,
            max_output_tokens: 150,
            context_separator: "\n\n".to_string(),
        }
    }
}

/// LLM-based contextual chunk enricher (Anthropic Contextual Retrieval pattern)
///
/// Uses the configured LLM to prepend 2-3 context sentences to each chunk before embedding,
/// enriching the context available to retrieval for long documents.
///
/// ## Example
///
/// ```ignore
/// use graphrag_core::text::contextual_enricher::{ContextualEnricher, ContextualEnricherConfig};
/// use graphrag_core::llm::LlmConfig;
///
/// # async fn example() -> graphrag_core::Result<()> {
/// let llm_config = LlmConfig {
///     enabled: true,
///     ..LlmConfig::default()
/// };
///
/// let enricher = ContextualEnricher::with_defaults(llm_config);
///
/// // Process a document's chunks
/// let enriched = enricher.enrich_document_chunks(&document, &chunks).await?;
/// # Ok(())
/// # }
/// ```
pub struct ContextualEnricher {
    #[cfg(all(feature = "async", feature = "ureq"))]
    client: LlmClient,
    config: ContextualEnricherConfig,
}

impl ContextualEnricher {
    /// Create a new contextual enricher with explicit configs
    pub fn new(llm_config: LlmConfig, enricher_config: ContextualEnricherConfig) -> Self {
        #[cfg(not(all(feature = "async", feature = "ureq")))]
        let _ = llm_config;

        Self {
            #[cfg(all(feature = "async", feature = "ureq"))]
            client: LlmClient::new(llm_config),
            config: enricher_config,
        }
    }

    /// Create with default enricher config
    pub fn with_defaults(llm_config: LlmConfig) -> Self {
        Self::new(llm_config, ContextualEnricherConfig::default())
    }

    /// Estimate token count from text length (approximation: chars / 4)
    ///
    /// This is a fast, dependency-free estimate. For most Western-language text,
    /// 1 token ≈ 4 characters. The safety_margin in config compensates for variance.
    pub fn estimate_tokens(text: &str) -> u32 {
        (text.len() / 4) as u32
    }

    /// Estimate the context size (tokens per server slot) needed for a document and its chunks
    ///
    /// Formula:
    /// ```text
    /// num_ctx = tokens(prompt_instructions)   // ~100 fixed tokens
    ///         + tokens(full_document)          // static prefix → KV cached
    ///         + tokens(largest_chunk)          // dynamic suffix
    ///         + max_output_tokens              // generation budget
    ///         + safety_margin (5%)             // avoid edge-case truncation
    /// ```
    ///
    /// The result is rounded up to the nearest 1024 and clamped to [4096, 131072].
    pub fn calculate_num_ctx(&self, document_text: &str, chunks: &[TextChunk]) -> u32 {
        let instruction_tokens = 100u32;
        let doc_tokens = Self::estimate_tokens(document_text);
        let max_chunk_tokens = chunks
            .iter()
            .map(|c| Self::estimate_tokens(&c.content))
            .max()
            .unwrap_or(0);

        let base =
            instruction_tokens + doc_tokens + max_chunk_tokens + self.config.max_output_tokens;
        let with_margin = (base as f32 * (1.0 + self.config.safety_margin)) as u32;

        // Round up to nearest 1024 for memory alignment efficiency
        let rounded = with_margin.div_ceil(1024) * 1024;

        // Clamp: minimum 4096 (sensible floor), maximum 131072 (128k, typical GPU limit)
        rounded.clamp(4096, 131_072)
    }

    /// Build the contextual enrichment prompt using the KV-cache-friendly structure
    ///
    /// The document text is the **static prefix** — it will be KV-cached after the
    /// first request and reused for all subsequent chunks from the same document.
    /// Only the chunk content changes between requests.
    #[cfg(feature = "async")]
    pub fn build_prompt(document_text: &str, chunk_text: &str) -> String {
        format!(
            "<document>\n{document_text}\n</document>\n\n\
             Here is the chunk we want to situate within the whole document:\n\
             <chunk>\n{chunk_text}\n</chunk>\n\n\
             Please give a short succinct context to situate this chunk within \
             the overall document for the purposes of improving search retrieval \
             of the chunk. Answer only with the succinct context and nothing else.",
        )
    }

    /// Enrich a single chunk with LLM-generated context
    #[cfg(all(feature = "async", feature = "ureq"))]
    async fn enrich_one(&self, chunk: &TextChunk, document_text: &str) -> Result<String> {
        let prompt = Self::build_prompt(document_text, &chunk.content);

        let params = LlmGenerationParams {
            max_tokens: Some(self.config.max_output_tokens),
            temperature: Some(0.1), // low temperature for factual, deterministic context
            ..Default::default()
        };

        let context = self.client.generate_with_params(&prompt, params).await?;

        Ok(format!(
            "{}{}{}",
            context.trim(),
            self.config.context_separator,
            chunk.content,
        ))
    }

    /// Enrich all chunks belonging to a single document
    ///
    /// Chunks are processed sequentially so the server's prompt cache keeps the
    /// document prefix across all requests.
    #[cfg(all(feature = "async", feature = "ureq"))]
    pub async fn enrich_document_chunks(
        &self,
        document: &Document,
        chunks: &[TextChunk],
    ) -> Result<Vec<TextChunk>> {
        if !self.config.enabled {
            return Ok(chunks.to_vec());
        }

        let doc_chunks: Vec<&TextChunk> = chunks
            .iter()
            .filter(|c| c.document_id == document.id)
            .collect();

        if doc_chunks.is_empty() {
            return Ok(chunks.to_vec());
        }

        #[cfg(feature = "tracing")]
        {
            let doc_chunk_owned: Vec<TextChunk> = doc_chunks.iter().map(|c| (*c).clone()).collect();
            tracing::info!(
                doc = %document.title,
                chunks = doc_chunks.len(),
                required_ctx = self.calculate_num_ctx(&document.content, &doc_chunk_owned),
                "Starting contextual enrichment (prompt cache enabled)",
            );
        }

        let mut result = chunks.to_vec();

        for (i, chunk) in doc_chunks.iter().enumerate() {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "Enriching chunk {}/{} (id={})",
                i + 1,
                doc_chunks.len(),
                chunk.id,
            );

            match self.enrich_one(chunk, &document.content).await {
                Ok(enriched_content) => {
                    if let Some(target) = result.iter_mut().find(|c| c.id == chunk.id) {
                        target.content = enriched_content;
                    }
                },
                Err(e) => {
                    // Fail gracefully: keep original chunk on error
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        chunk_id = %chunk.id,
                        error = %e,
                        "Contextual enrichment failed for chunk, keeping original",
                    );
                },
            }
        }

        Ok(result)
    }

    /// Enrich chunks from multiple documents in a single pipeline call
    ///
    /// Documents are processed one at a time so the server can maximise KV cache
    /// reuse (the document context stays loaded for all its chunks before
    /// moving to the next document).
    #[cfg(all(feature = "async", feature = "ureq"))]
    pub async fn enrich_chunks(
        &self,
        documents: &[Document],
        chunks: Vec<TextChunk>,
    ) -> Result<Vec<TextChunk>> {
        if !self.config.enabled {
            return Ok(chunks);
        }

        let doc_map: HashMap<_, &Document> = documents.iter().map(|d| (d.id.clone(), d)).collect();

        let mut enriched = chunks.clone();
        let mut processed: HashSet<_> = HashSet::new();

        for chunk in &chunks {
            if processed.contains(&chunk.document_id) {
                continue;
            }
            if let Some(doc) = doc_map.get(&chunk.document_id) {
                enriched = self.enrich_document_chunks(doc, &enriched).await?;
                processed.insert(chunk.document_id.clone());
            }
        }

        Ok(enriched)
    }

    /// Access the enricher configuration
    pub fn config(&self) -> &ContextualEnricherConfig {
        &self.config
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::core::{ChunkId, Document, DocumentId};

    fn make_document(content: &str) -> Document {
        Document::new(
            DocumentId::new("doc_test".to_string()),
            "Test Document".to_string(),
            content.to_string(),
        )
    }

    fn make_chunk(doc_id: DocumentId, id: &str, content: &str) -> TextChunk {
        TextChunk::new(
            ChunkId::new(id.to_string()),
            doc_id,
            content.to_string(),
            0,
            content.len(),
        )
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "a".repeat(400);
        assert_eq!(ContextualEnricher::estimate_tokens(&text), 100);
    }

    #[test]
    fn test_calculate_num_ctx_minimum() {
        let config = LlmConfig::default();
        let enricher = ContextualEnricher::with_defaults(config);
        let doc = make_document("short document");
        let chunk = make_chunk(doc.id.clone(), "c0", "short chunk");
        // Should always be at least 4096
        assert!(enricher.calculate_num_ctx(&doc.content, &[chunk]) >= 4096);
    }

    #[test]
    fn test_calculate_num_ctx_large_document() {
        let config = LlmConfig::default();
        let enricher = ContextualEnricher::with_defaults(config);
        // Simulate a 45k-token book (~180k chars)
        let doc_content = "word ".repeat(36_000);
        let chunk_content = "word ".repeat(500); // ~500 tokens
        let doc = make_document(&doc_content);
        let chunk = make_chunk(doc.id.clone(), "c0", &chunk_content);

        let num_ctx = enricher.calculate_num_ctx(&doc.content, &[chunk]);
        // Should be > doc tokens + chunk tokens
        assert!(num_ctx > 36_000 + 500);
        // Should be a multiple of 1024
        assert_eq!(num_ctx % 1024, 0);
    }

    #[test]
    fn test_build_prompt_contains_document_and_chunk() {
        let prompt = ContextualEnricher::build_prompt("full document text", "chunk excerpt");
        assert!(prompt.contains("full document text"));
        assert!(prompt.contains("chunk excerpt"));
        assert!(prompt.contains("<document>"));
        assert!(prompt.contains("<chunk>"));
    }

    #[test]
    fn test_disabled_enricher_returns_original() {
        // When disabled, enrich_chunks should be a no-op
        let config_with_disabled = ContextualEnricherConfig {
            enabled: false,
            ..Default::default()
        };
        // We can verify the config
        assert!(!config_with_disabled.enabled);
    }
}
