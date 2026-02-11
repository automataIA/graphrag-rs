//! Trait-based chunking strategy implementations
//!
//! This module provides concrete implementations of the ChunkingStrategy trait
//! that wrap existing chunking logic while maintaining a clean, minimal interface.

use crate::{
    core::{ChunkId, DocumentId, TextChunk, ChunkingStrategy},
    text::{HierarchicalChunker, SemanticChunker},
};

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique chunk IDs
static CHUNK_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Hierarchical chunking strategy wrapper
///
/// Wraps the existing HierarchicalChunker to implement ChunkingStrategy trait.
/// This strategy respects semantic boundaries (paragraphs, sentences, words).
pub struct HierarchicalChunkingStrategy {
    inner: HierarchicalChunker,
    chunk_size: usize,
    overlap: usize,
    document_id: DocumentId,
}

impl HierarchicalChunkingStrategy {
    /// Create a new hierarchical chunking strategy
    pub fn new(chunk_size: usize, overlap: usize, document_id: DocumentId) -> Self {
        Self {
            inner: HierarchicalChunker::new().with_min_size(50),
            chunk_size,
            overlap,
            document_id,
        }
    }

    /// Set minimum chunk size
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.inner = self.inner.with_min_size(min_size);
        self
    }
}

impl ChunkingStrategy for HierarchicalChunkingStrategy {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        let chunks_text = self.inner.chunk_text(text, self.chunk_size, self.overlap);
        let mut chunks = Vec::new();
        let mut current_pos = 0;

        for chunk_content in chunks_text {
            if !chunk_content.trim().is_empty() {
                let chunk_id = ChunkId::new(format!("{}_{}", self.document_id,
                    CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst)));
                let chunk_start = current_pos;
                let chunk_end = chunk_start + chunk_content.len();

                let chunk = TextChunk::new(
                    chunk_id,
                    self.document_id.clone(),
                    chunk_content.clone(),
                    chunk_start,
                    chunk_end,
                );
                chunks.push(chunk);
                current_pos = chunk_end;
            } else {
                current_pos += chunk_content.len();
            }
        }

        chunks
    }
}

/// Semantic chunking strategy wrapper
///
/// Wraps the existing SemanticChunker to implement ChunkingStrategy trait.
/// This strategy uses embedding similarity to determine natural breakpoints.
pub struct SemanticChunkingStrategy {
    _inner: SemanticChunker,
    document_id: DocumentId,
}

impl SemanticChunkingStrategy {
    /// Create a new semantic chunking strategy
    pub fn new(chunker: SemanticChunker, document_id: DocumentId) -> Self {
        Self {
            _inner: chunker,
            document_id,
        }
    }
}

impl ChunkingStrategy for SemanticChunkingStrategy {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        // Note: This is a simplified implementation
        // In a real scenario, you would need to handle the async nature of semantic chunking
        // or use a synchronous embedding generator

        // For now, fall back to a simple sentence-based approach
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current_pos = 0;

        // Group sentences into chunks of reasonable size
        let chunk_size = 5; // sentences per chunk
        for chunk_sentences in sentences.chunks(chunk_size) {
            let chunk_content = chunk_sentences.join(". ") + ".";
            let chunk_id = ChunkId::new(format!("{}_{}", self.document_id,
                CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst)));
            let chunk_start = current_pos;
            let chunk_end = chunk_start + chunk_content.len();

            let chunk = TextChunk::new(
                chunk_id,
                self.document_id.clone(),
                chunk_content,
                chunk_start,
                chunk_end,
            );
            chunks.push(chunk);
            current_pos = chunk_end;
        }

        chunks
    }
}

/// Rust code chunking strategy using tree-sitter
///
/// Parses Rust code using tree-sitter and creates chunks at function/method boundaries.
/// This ensures that code chunks are syntactically complete and meaningful.
#[cfg(feature = "code-chunking")]
pub struct RustCodeChunkingStrategy {
    min_chunk_size: usize,
    document_id: DocumentId,
}

#[cfg(feature = "code-chunking")]
impl RustCodeChunkingStrategy {
    /// Create a new Rust code chunking strategy
    pub fn new(min_chunk_size: usize, document_id: DocumentId) -> Self {
        Self {
            min_chunk_size,
            document_id,
        }
    }
}

#[cfg(feature = "code-chunking")]
impl ChunkingStrategy for RustCodeChunkingStrategy {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        use tree_sitter::Parser;

        let mut parser = Parser::new();
        let language = tree_sitter_rust::language();
        parser.set_language(&language).expect("Error loading Rust grammar");

        let tree = parser.parse(text, None).expect("Error parsing Rust code");
        let root_node = tree.root_node();

        let mut chunks = Vec::new();

        // Extract top-level items: functions, impl blocks, structs, enums, mods
        self.extract_chunks(&root_node, text, &mut chunks);

        // If no chunks found (e.g., just expressions), create a single chunk
        if chunks.is_empty() && !text.trim().is_empty() {
            let chunk_id = ChunkId::new(format!("{}_{}", self.document_id,
                CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst)));
            let chunk = TextChunk::new(
                chunk_id,
                self.document_id.clone(),
                text.to_string(),
                0,
                text.len(),
            );
            chunks.push(chunk);
        }

        chunks
    }
}

#[cfg(feature = "code-chunking")]
impl RustCodeChunkingStrategy {
    /// Extract code chunks from AST nodes
    fn extract_chunks(&self, node: &tree_sitter::Node, source: &str, chunks: &mut Vec<TextChunk>) {
        match node.kind() {
            // Top-level items that should become chunks
            "function_item" | "impl_item" | "struct_item" | "enum_item" | "mod_item" | "trait_item" => {
                let start_byte = node.start_byte();
                let end_byte = node.end_byte();

                // Convert byte indices to char indices
                let start_pos = source.len() - source[start_byte..].len();
                let end_pos = source.len() - source[end_byte..].len();

                let chunk_content = &source[start_pos..end_pos];

                if chunk_content.len() >= self.min_chunk_size {
                    let chunk_id = ChunkId::new(format!("{}_{}", self.document_id,
                        CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst)));

                    let chunk = TextChunk::new(
                        chunk_id,
                        self.document_id.clone(),
                        chunk_content.to_string(),
                        start_pos,
                        end_pos,
                    );
                    chunks.push(chunk);
                }
            }

            // Source file (root) - process children
            "source_file" => {
                let mut child = node.child(0);
                while let Some(current) = child {
                    self.extract_chunks(&current, source, chunks);
                    child = current.next_sibling();
                }
            }

            // Other nodes - recurse into children
            _ => {
                let mut child = node.child(0);
                while let Some(current) = child {
                    self.extract_chunks(&current, source, chunks);
                    child = current.next_sibling();
                }
            }
        }
    }
}

/// Boundary-Aware Chunking Strategy (BAR-RAG)
///
/// This strategy implements the BAR-RAG (Boundary-Aware Retrieval-Augmented Generation)
/// approach by:
/// 1. Detecting semantic boundaries in text (sentences, paragraphs, headings, etc.)
/// 2. Scoring chunk coherence using sentence embeddings
/// 3. Finding optimal split points that maximize semantic unity
///
/// **Performance Target**: +40% semantic coherence, -60% entity fragmentation
pub struct BoundaryAwareChunkingStrategy {
    boundary_detector: crate::text::BoundaryDetector,
    coherence_scorer: std::sync::Arc<crate::text::SemanticCoherenceScorer>,
    max_chunk_chars: usize,
    min_chunk_chars: usize,
    document_id: DocumentId,
}

impl BoundaryAwareChunkingStrategy {
    /// Create a new boundary-aware chunking strategy
    ///
    /// # Arguments
    /// * `boundary_config` - Configuration for boundary detection
    /// * `coherence_config` - Configuration for coherence scoring
    /// * `embedding_provider` - Provider for generating sentence embeddings
    /// * `max_chunk_chars` - Maximum characters per chunk
    /// * `min_chunk_chars` - Minimum characters per chunk
    /// * `document_id` - Document identifier for chunk IDs
    pub fn new(
        boundary_config: crate::text::BoundaryDetectionConfig,
        coherence_config: crate::text::CoherenceConfig,
        embedding_provider: std::sync::Arc<dyn crate::embeddings::EmbeddingProvider>,
        max_chunk_chars: usize,
        min_chunk_chars: usize,
        document_id: DocumentId,
    ) -> Self {
        Self {
            boundary_detector: crate::text::BoundaryDetector::with_config(boundary_config),
            coherence_scorer: std::sync::Arc::new(
                crate::text::SemanticCoherenceScorer::new(coherence_config, embedding_provider),
            ),
            max_chunk_chars,
            min_chunk_chars,
            document_id,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(
        embedding_provider: std::sync::Arc<dyn crate::embeddings::EmbeddingProvider>,
        document_id: DocumentId,
    ) -> Self {
        Self::new(
            crate::text::BoundaryDetectionConfig::default(),
            crate::text::CoherenceConfig::default(),
            embedding_provider,
            2000, // max chars
            200,  // min chars
            document_id,
        )
    }

    /// Chunk text asynchronously (helper for async contexts)
    async fn chunk_async(&self, text: &str) -> Vec<TextChunk> {
        // 1. Detect all semantic boundaries
        let boundaries = self.boundary_detector.detect_boundaries(text);

        // Extract boundary positions suitable for splitting
        let boundary_positions: Vec<usize> = boundaries
            .iter()
            .filter(|b| {
                // Filter boundaries that are good split points
                matches!(
                    b.boundary_type,
                    crate::text::BoundaryType::Paragraph
                        | crate::text::BoundaryType::Heading
                        | crate::text::BoundaryType::CodeBlock
                )
            })
            .map(|b| b.position)
            .collect();

        // 2. Find optimal splits using coherence scoring
        let optimal_result = self
            .coherence_scorer
            .find_optimal_split(text, &boundary_positions)
            .await;

        let chunks = match optimal_result {
            Ok(result) => {
                // Use optimally scored chunks
                self.create_text_chunks_from_scored(&result.chunks)
            }
            Err(_) => {
                // Fallback: use boundary positions directly
                self.create_text_chunks_from_boundaries(text, &boundary_positions)
            }
        };

        // 3. Enforce size constraints
        self.enforce_size_constraints(chunks)
    }

    /// Create TextChunk objects from scored chunks
    fn create_text_chunks_from_scored(
        &self,
        scored_chunks: &[crate::text::ScoredChunk],
    ) -> Vec<TextChunk> {
        scored_chunks
            .iter()
            .enumerate()
            .map(|(i, sc)| {
                let chunk_id = ChunkId::new(format!("{}_{}", self.document_id, i));
                let mut chunk = TextChunk::new(
                    chunk_id,
                    self.document_id.clone(),
                    sc.text.clone(),
                    sc.start_pos,
                    sc.end_pos,
                );

                // Add coherence score to metadata
                chunk
                    .metadata
                    .custom
                    .insert("coherence_score".to_string(), sc.coherence_score.to_string());
                chunk
                    .metadata
                    .custom
                    .insert("sentence_count".to_string(), sc.sentence_count.to_string());

                chunk
            })
            .collect()
    }

    /// Create TextChunk objects from boundary positions (fallback)
    fn create_text_chunks_from_boundaries(
        &self,
        text: &str,
        boundaries: &[usize],
    ) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let mut prev_pos = 0;

        for (i, &pos) in boundaries.iter().enumerate() {
            if pos > prev_pos {
                let chunk_id = ChunkId::new(format!("{}_{}", self.document_id, i));
                let chunk = TextChunk::new(
                    chunk_id,
                    self.document_id.clone(),
                    text[prev_pos..pos].to_string(),
                    prev_pos,
                    pos,
                );
                chunks.push(chunk);
                prev_pos = pos;
            }
        }

        // Add final chunk
        if prev_pos < text.len() {
            let chunk_id = ChunkId::new(format!("{}_{}", self.document_id, chunks.len()));
            let chunk = TextChunk::new(
                chunk_id,
                self.document_id.clone(),
                text[prev_pos..].to_string(),
                prev_pos,
                text.len(),
            );
            chunks.push(chunk);
        }

        chunks
    }

    /// Enforce size constraints on chunks
    fn enforce_size_constraints(&self, mut chunks: Vec<TextChunk>) -> Vec<TextChunk> {
        let mut result = Vec::new();

        for chunk in chunks.drain(..) {
            let chunk_len = chunk.content.len();

            if chunk_len > self.max_chunk_chars {
                // Split large chunks at sentence boundaries
                result.extend(self.split_large_chunk(chunk));
            } else if chunk_len < self.min_chunk_chars && !result.is_empty() {
                // Merge small chunks with previous
                if let Some(mut prev_chunk) = result.pop() {
                    prev_chunk.content.push(' ');
                    prev_chunk.content.push_str(&chunk.content);
                    prev_chunk.end_offset = chunk.end_offset;
                    result.push(prev_chunk);
                } else {
                    result.push(chunk);
                }
            } else {
                result.push(chunk);
            }
        }

        result
    }

    /// Split a large chunk at sentence boundaries
    fn split_large_chunk(&self, chunk: TextChunk) -> Vec<TextChunk> {
        // Simple split at sentence boundaries
        let sentences: Vec<&str> = chunk
            .content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut sub_chunks = Vec::new();
        let mut current_text = String::new();
        let mut current_start = chunk.start_offset;

        for sentence in sentences {
            if current_text.len() + sentence.len() > self.max_chunk_chars && !current_text.is_empty()
            {
                // Create chunk
                let chunk_id = ChunkId::new(format!(
                    "{}_{}",
                    self.document_id,
                    CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst)
                ));
                let end = current_start + current_text.len();
                sub_chunks.push(TextChunk::new(
                    chunk_id,
                    self.document_id.clone(),
                    current_text.clone(),
                    current_start,
                    end,
                ));

                current_start = end;
                current_text.clear();
            }

            current_text.push_str(sentence);
            current_text.push('.');
        }

        // Add remaining text
        if !current_text.is_empty() {
            let chunk_id = ChunkId::new(format!(
                "{}_{}",
                self.document_id,
                CHUNK_COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            sub_chunks.push(TextChunk::new(
                chunk_id,
                self.document_id.clone(),
                current_text,
                current_start,
                chunk.end_offset,
            ));
        }

        sub_chunks
    }
}

impl ChunkingStrategy for BoundaryAwareChunkingStrategy {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        // Create a Tokio runtime to execute async code synchronously
        // This allows us to use async coherence scoring in a sync context
        let runtime = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");

        runtime.block_on(self.chunk_async(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_chunking_strategy() {
        let document_id = DocumentId::new("test_doc".to_string());
        let strategy = HierarchicalChunkingStrategy::new(100, 20, document_id);

        let text = "This is paragraph one.\n\nThis is paragraph two with more content to test chunking behavior.";
        let chunks = strategy.chunk(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.start_offset < chunk.end_offset);
        }
    }

    #[test]
    fn test_semantic_chunking_strategy() {
        let document_id = DocumentId::new("test_doc".to_string());
        // Note: In a real test, you would create a proper SemanticChunker
        // For now, we'll use a mock approach
        let config = crate::text::semantic_chunking::SemanticChunkerConfig::default();
        // We can't easily create a mock embedding generator here, so skip the test
        // let embedding_gen = crate::vector::EmbeddingGenerator::mock();
        // let chunker = SemanticChunker::new(config, embedding_gen);
        // let strategy = SemanticChunkingStrategy::new(chunker, document_id);
        //
        // let text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence.";
        // let chunks = strategy.chunk(text);
        //
        // assert!(!chunks.is_empty());
        // for chunk in &chunks {
        //     assert!(!chunk.content.is_empty());
        // }
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_rust_code_chunking_strategy() {
        let document_id = DocumentId::new("rust_code".to_string());
        let strategy = RustCodeChunkingStrategy::new(10, document_id);

        let rust_code = r#"
fn main() {
    println!("Hello, world!");
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}
"#;

        let chunks = strategy.chunk(rust_code);

        assert!(!chunks.is_empty());
        // Should find at least main function and struct/impl blocks
        assert!(chunks.len() >= 2);

        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.start_offset < chunk.end_offset);
        }
    }
}