//! Late Chunking — context-preserving embeddings for RAG
//!
//! Standard RAG embeds each chunk in isolation, losing cross-chunk context.
//! Late Chunking (Jina AI, 2024) fixes this by encoding the **whole document**
//! first and then extracting per-chunk embeddings via span pooling:
//!
//! ```text
//! Standard:  chunk₁ → embed₁   chunk₂ → embed₂   (context-blind)
//! Late:       [chunk₁ | chunk₂ | …] → model → pool spans → embed₁, embed₂
//! ```
//!
//! Each chunk's embedding "sees" the entire document during the attention pass,
//! giving it +5-10% retrieval accuracy over standard chunking.
//!
//! ## Two usage modes
//!
//! 1. **`LateChunkingStrategy`** — a [`ChunkingStrategy`] that splits text and
//!    records precise byte spans. Use this when you will pass the chunks to a
//!    late-chunking-aware embedding provider separately.
//!
//! 2. **`JinaLateChunkingClient`** — calls the Jina embeddings API with
//!    `late_chunking=true` to get document-context-aware embeddings directly.
//!
//! ## Model context limits
//!
//! | Model                  | Max tokens | Notes                          |
//! |------------------------|------------|-------------------------------|
//! | Jina v3 (default)      | 8 192      | Good for most documents        |
//! | gte-Qwen2-7B-instruct  | 32 768     | Better quality, needs more GPU |
//!
//! For documents exceeding the limit use [`LateChunkingStrategy::split_into_sections`]
//! to pre-divide the document and apply late chunking section-by-section.

use crate::{
    core::{ChunkId, ChunkingStrategy, DocumentId, GraphRAGError, TextChunk},
    text::chunking::HierarchicalChunker,
};
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique late-chunking chunk IDs
static LATE_CHUNK_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configuration for the late chunking strategy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LateChunkingConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,

    /// Chunk overlap in characters
    pub chunk_overlap: usize,

    /// Maximum document size in tokens before splitting into sections.
    ///
    /// - `8192` for Jina v3 (default)
    /// - `32768` for gte-Qwen2-7B-instruct
    pub max_doc_tokens: u32,

    /// Annotate each chunk's `position_in_document` metadata field.
    ///
    /// Embedding providers can use this to apply position-aware pooling.
    pub annotate_positions: bool,
}

impl Default for LateChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
            max_doc_tokens: 8192, // Jina v3 default
            annotate_positions: true,
        }
    }
}

/// Context-aware chunking strategy for use with late-chunking embedding models
///
/// Splits text using [`HierarchicalChunker`] and records precise byte-offset
/// spans in each chunk's metadata. A late-chunking embedding provider
/// (Jina API or a local `candle` model) can then use these spans to extract
/// per-chunk representations from a single full-document forward pass.
///
/// # Examples
///
/// ```rust
/// use graphrag_core::text::late_chunking::{LateChunkingStrategy, LateChunkingConfig};
/// use graphrag_core::core::{ChunkingStrategy, DocumentId};
///
/// let strategy = LateChunkingStrategy::with_defaults(DocumentId::new("doc-1".to_string()));
/// let chunks = strategy.chunk("First paragraph.\n\nSecond paragraph.");
///
/// for chunk in &chunks {
///     // position_in_document ∈ [0.0, 1.0] — used by embedding provider for pooling
///     assert!(chunk.metadata.position_in_document.is_some());
/// }
/// ```
pub struct LateChunkingStrategy {
    config: LateChunkingConfig,
    document_id: DocumentId,
    inner: HierarchicalChunker,
}

impl LateChunkingStrategy {
    /// Create a new late chunking strategy with explicit config
    pub fn new(config: LateChunkingConfig, document_id: DocumentId) -> Self {
        Self {
            inner: HierarchicalChunker::new().with_min_size(50),
            config,
            document_id,
        }
    }

    /// Create with default config (8192 token limit, 512-char chunks)
    pub fn with_defaults(document_id: DocumentId) -> Self {
        Self::new(LateChunkingConfig::default(), document_id)
    }

    /// Set the maximum document token limit (choose based on embedding model)
    ///
    /// - `8192`  → Jina v3
    /// - `32768` → gte-Qwen2-7B-instruct
    pub fn with_max_doc_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_doc_tokens = max_tokens;
        self
    }

    /// Estimate token count from character count (1 token ≈ 4 chars)
    pub fn estimate_tokens(text: &str) -> u32 {
        (text.len() / 4) as u32
    }

    /// Returns `true` if the document fits within the model's context window
    pub fn fits_in_context(&self, text: &str) -> bool {
        Self::estimate_tokens(text) <= self.config.max_doc_tokens
    }

    /// Split an oversized document into sections that fit within the context window
    ///
    /// Sections are formed by grouping paragraphs (double-newline boundaries)
    /// until the next paragraph would exceed the limit. Each section can be
    /// embedded independently with late chunking applied within it.
    pub fn split_into_sections(&self, text: &str) -> Vec<String> {
        if self.fits_in_context(text) {
            return vec![text.to_string()];
        }

        let max_chars = (self.config.max_doc_tokens * 4) as usize;
        let mut sections: Vec<String> = Vec::new();
        let mut current = String::new();

        for paragraph in text.split("\n\n") {
            let needed = current.len() + if current.is_empty() { 0 } else { 2 } + paragraph.len();
            if needed > max_chars && !current.is_empty() {
                sections.push(current.trim().to_string());
                current = String::new();
            }
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(paragraph);
        }

        if !current.trim().is_empty() {
            sections.push(current.trim().to_string());
        }

        sections
    }
}

impl ChunkingStrategy for LateChunkingStrategy {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        let raw_chunks =
            self.inner
                .chunk_text(text, self.config.chunk_size, self.config.chunk_overlap);
        let doc_len = text.len().max(1);
        let mut chunks = Vec::with_capacity(raw_chunks.len());
        let mut current_pos: usize = 0;

        for chunk_content in raw_chunks {
            if chunk_content.trim().is_empty() {
                current_pos += chunk_content.len();
                continue;
            }

            let chunk_id = ChunkId::new(format!(
                "{}_lc_{}",
                self.document_id,
                LATE_CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst),
            ));

            let start = current_pos;
            let end = start + chunk_content.len();
            let mut chunk = TextChunk::new(
                chunk_id,
                self.document_id.clone(),
                chunk_content.clone(),
                start,
                end,
            );

            // Record relative position so the embedding layer knows the span
            if self.config.annotate_positions {
                chunk.metadata.position_in_document = Some(start as f32 / doc_len as f32);
            }

            chunks.push(chunk);
            current_pos = end;
        }

        chunks
    }
}

/// Jina AI embeddings client with native late chunking support
///
/// Calls the Jina embeddings API with `late_chunking=true`. The API encodes
/// the concatenated inputs as a single sequence and returns per-input embeddings
/// where each embedding reflects the full-document context.
///
/// For fully **local** operation (no API key), configure Ollama with
/// `rjmalagon/gte-qwen2-7b-instruct` — it provides excellent 32k-context
/// embeddings without native late chunking but with a far larger context window.
///
/// # Examples
///
/// ```rust,no_run
/// use graphrag_core::text::late_chunking::JinaLateChunkingClient;
///
/// # async fn example(chunks: &[graphrag_core::TextChunk]) -> graphrag_core::Result<()> {
/// let client = JinaLateChunkingClient::new("jina_xxxx".to_string());
/// let embeddings = client.embed_with_late_chunking(chunks).await?;
/// assert_eq!(embeddings.len(), chunks.len());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct JinaLateChunkingClient {
    api_key: String,
    /// Model name (default: `"jina-embeddings-v3"`)
    model: String,
}

impl JinaLateChunkingClient {
    const ENDPOINT: &'static str = "https://api.jina.ai/v1/embeddings";

    /// Create a new client with a Jina API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "jina-embeddings-v3".to_string(),
        }
    }

    /// Override the embedding model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Embed chunks using Jina's late chunking API
    ///
    /// Sends all chunk contents in a single request with `late_chunking: true`.
    /// The Jina API encodes them as one sequence so each chunk's embedding
    /// incorporates the full document context.
    ///
    /// Returns one embedding vector per chunk, in the same order as the input.
    #[cfg(feature = "ureq")]
    pub async fn embed_with_late_chunking(
        &self,
        chunks: &[TextChunk],
    ) -> crate::Result<Vec<Vec<f32>>> {
        let inputs: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

        let body = serde_json::json!({
            "model": self.model,
            "input": inputs,
            "late_chunking": true,
        });

        let agent = ureq::AgentBuilder::new().build();
        let response = agent
            .post(Self::ENDPOINT)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| GraphRAGError::Generation {
                message: format!("Jina API request failed: {e}"),
            })?;

        let json: serde_json::Value =
            response
                .into_json()
                .map_err(|e| GraphRAGError::Generation {
                    message: format!("Failed to parse Jina API response: {e}"),
                })?;

        let data = json["data"]
            .as_array()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Invalid Jina API response: missing 'data' array".to_string(),
            })?;

        let embeddings = data
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DocumentId;

    #[test]
    fn test_late_chunking_produces_chunks_with_position() {
        let strategy = LateChunkingStrategy::with_defaults(DocumentId::new("test-doc".to_string()));

        let text = "First paragraph about machine learning.\n\n\
             Second paragraph about deep learning.\n\n\
             Third paragraph about neural networks.";

        let chunks = strategy.chunk(text);
        assert!(!chunks.is_empty());

        // Every chunk should have a position annotation
        for chunk in &chunks {
            assert!(
                chunk.metadata.position_in_document.is_some(),
                "chunk {} missing position metadata",
                chunk.id
            );
        }
    }

    #[test]
    fn test_chunk_ids_have_lc_suffix() {
        let strategy = LateChunkingStrategy::with_defaults(DocumentId::new("doc".to_string()));
        let chunks = strategy.chunk("Some text to chunk into pieces here.");
        for chunk in &chunks {
            assert!(
                chunk.id.0.contains("_lc_"),
                "Expected '_lc_' in ID: {}",
                chunk.id
            );
        }
    }

    #[test]
    fn test_fits_in_context() {
        let config = LateChunkingConfig {
            max_doc_tokens: 10,
            ..Default::default()
        };
        let strategy = LateChunkingStrategy::new(config, DocumentId::new("d".to_string()));

        assert!(strategy.fits_in_context("tiny")); // 4 chars → 1 token
        assert!(!strategy.fits_in_context(&"x".repeat(100))); // 100 chars → 25 tokens
    }

    #[test]
    fn test_split_into_sections_short_doc() {
        let strategy = LateChunkingStrategy::with_defaults(DocumentId::new("d".to_string()));
        let text = "Short document.";
        let sections = strategy.split_into_sections(text);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0], text);
    }

    #[test]
    fn test_split_into_sections_long_doc() {
        let config = LateChunkingConfig {
            max_doc_tokens: 5, // 20 chars max
            ..Default::default()
        };
        let strategy = LateChunkingStrategy::new(config, DocumentId::new("d".to_string()));

        // Each paragraph is ~15 chars, exceeding the 20-char section limit when combined
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        let sections = strategy.split_into_sections(text);
        // Should be split into multiple sections
        assert!(
            sections.len() > 1,
            "Expected multiple sections, got {}",
            sections.len()
        );
        // All content should be present
        let combined = sections.join(" ");
        assert!(combined.contains("Paragraph one"));
        assert!(combined.contains("Paragraph two"));
        assert!(combined.contains("Paragraph three"));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(LateChunkingStrategy::estimate_tokens(&"a".repeat(400)), 100);
        assert_eq!(LateChunkingStrategy::estimate_tokens(""), 0);
    }
}
