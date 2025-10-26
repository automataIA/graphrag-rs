//! LazyGraphRAG Pipeline
//!
//! Complete end-to-end pipeline for LazyGraphRAG (Microsoft Research, 2025)
//! that achieves 0.1% of full GraphRAG indexing cost and 700x cheaper query costs.
//!
//! ## Key Features
//!
//! - **No LLM for Indexing**: Uses noun phrase extraction instead of LLM entity extraction
//! - **Fast Construction**: Builds concept graph from co-occurrence without summarization
//! - **Efficient Queries**: Iterative deepening search with bidirectional index
//! - **Zero Prior Summarization**: Works directly on text chunks
//!
//! ## Pipeline Stages
//!
//! 1. **Document Processing**: Load and chunk documents
//! 2. **Concept Extraction**: Extract concepts using NLP patterns (no LLM)
//! 3. **Graph Construction**: Build co-occurrence graph from concepts
//! 4. **Index Building**: Create bidirectional entity-chunk index
//! 5. **Query Processing**: Refine queries and retrieve using iterative deepening
//!
//! ## Example
//!
//! ```rust
//! use graphrag_core::lightrag::lazy_pipeline::{LazyGraphRAGPipeline, LazyPipelineConfig};
//!
//! let config = LazyPipelineConfig::default();
//! let mut pipeline = LazyGraphRAGPipeline::new(config);
//!
//! // Index documents
//! pipeline.index_document("doc1", "Text about machine learning and neural networks...");
//! pipeline.index_document("doc2", "More text about deep learning...");
//!
//! // Build the concept graph
//! pipeline.build_graph();
//!
//! // Query the graph
//! let results = pipeline.query("machine learning applications");
//! println!("Found {} relevant chunks", results.chunk_count());
//! ```

use crate::core::{ChunkId, TextChunk};
use crate::entity::BidirectionalIndex;
use crate::lightrag::concept_graph::{
    ConceptExtractor, ConceptExtractorConfig, ConceptGraph, ConceptGraphBuilder,
};
use crate::lightrag::iterative_deepening::{IterativeDeepeningSearch, SearchConfig, SearchResults};
use crate::lightrag::query_refinement::{QueryRefiner, QueryRefinementConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for LazyGraphRAG pipeline
#[derive(Debug, Clone)]
pub struct LazyPipelineConfig {
    /// Concept extraction configuration
    pub concept_extraction: ConceptExtractorConfig,

    /// Query refinement configuration
    pub query_refinement: QueryRefinementConfig,

    /// Search configuration
    pub search: SearchConfig,

    /// Chunk size for text splitting
    pub chunk_size: usize,

    /// Chunk overlap
    pub chunk_overlap: usize,

    /// Enable bidirectional indexing
    pub use_bidirectional_index: bool,
}

impl Default for LazyPipelineConfig {
    fn default() -> Self {
        Self {
            concept_extraction: ConceptExtractorConfig::default(),
            query_refinement: QueryRefinementConfig::default(),
            search: SearchConfig::default(),
            chunk_size: 512,
            chunk_overlap: 128,
            use_bidirectional_index: true,
        }
    }
}

/// LazyGraphRAG pipeline implementation
pub struct LazyGraphRAGPipeline {
    config: LazyPipelineConfig,
    concept_extractor: ConceptExtractor,
    graph_builder: ConceptGraphBuilder,
    concept_graph: Option<ConceptGraph>,
    bidirectional_index: Option<BidirectionalIndex>,
    query_refiner: QueryRefiner,
    search_engine: IterativeDeepeningSearch,
    chunks: HashMap<ChunkId, TextChunk>,
    document_count: usize,
}

impl LazyGraphRAGPipeline {
    /// Create a new LazyGraphRAG pipeline with configuration
    pub fn new(config: LazyPipelineConfig) -> Self {
        let concept_extractor = ConceptExtractor::with_config(config.concept_extraction.clone());
        let graph_builder = ConceptGraphBuilder::new();
        let query_refiner = QueryRefiner::new(config.query_refinement.clone());
        let search_engine = IterativeDeepeningSearch::new(config.search.clone());

        Self {
            config,
            concept_extractor,
            graph_builder,
            concept_graph: None,
            bidirectional_index: None,
            query_refiner,
            search_engine,
            chunks: HashMap::new(),
            document_count: 0,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(LazyPipelineConfig::default())
    }

    /// Index a document by extracting concepts and building the graph
    ///
    /// This processes the document, extracts concepts, and adds them to the graph builder.
    /// Call `build_graph()` after indexing all documents to finalize the graph.
    pub fn index_document(&mut self, document_id: &str, text: &str) {
        // Chunk the document
        let chunks = self.chunk_text(text, document_id);

        // Extract concepts from each chunk
        for chunk in chunks {
            let concepts = self.concept_extractor.extract_concepts(&chunk.content);

            // Add to graph builder
            self.graph_builder
                .add_document_concepts(document_id, concepts.clone());
            self.graph_builder
                .add_chunk_concepts(&chunk.id.as_str(), concepts);

            // Store chunk
            self.chunks.insert(chunk.id.clone(), chunk);
        }

        self.document_count += 1;
    }

    /// Build the concept graph from all indexed documents
    ///
    /// This finalizes the graph construction and creates the bidirectional index.
    /// Must be called after indexing all documents and before querying.
    pub fn build_graph(&mut self) {
        // Build concept graph
        let graph_builder = std::mem::replace(&mut self.graph_builder, ConceptGraphBuilder::new());
        self.concept_graph = Some(graph_builder.build());

        // Build bidirectional index if enabled
        if self.config.use_bidirectional_index {
            let mut index = BidirectionalIndex::new();

            // Add all concept-chunk mappings to the index
            if let Some(ref graph) = self.concept_graph {
                for (concept_text, concept) in &graph.concepts {
                    let entity_id = crate::core::EntityId::new(self.normalize_concept(concept_text));

                    for chunk_id in &concept.chunk_ids {
                        index.add_mapping(&entity_id, chunk_id);
                    }
                }
            }

            self.bidirectional_index = Some(index);
        }
    }

    /// Query the concept graph using iterative deepening search
    ///
    /// Returns search results with relevant chunks and search statistics.
    pub fn query(&self, query: &str) -> SearchResults {
        let graph = match &self.concept_graph {
            Some(g) => g,
            None => {
                // Return empty results if graph not built
                return SearchResults::new(query.to_string());
            }
        };

        let index = match &self.bidirectional_index {
            Some(i) => i,
            None => {
                // Return empty results if index not built
                return SearchResults::new(query.to_string());
            }
        };

        self.search_engine.search(query, graph, index)
    }

    /// Get chunks from search results
    pub fn get_chunks(&self, search_results: &SearchResults) -> Vec<TextChunk> {
        search_results
            .chunk_ids
            .iter()
            .filter_map(|chunk_id| self.chunks.get(chunk_id).cloned())
            .collect()
    }

    /// Get the concept graph
    pub fn get_concept_graph(&self) -> Option<&ConceptGraph> {
        self.concept_graph.as_ref()
    }

    /// Get the bidirectional index
    pub fn get_bidirectional_index(&self) -> Option<&BidirectionalIndex> {
        self.bidirectional_index.as_ref()
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> PipelineStatistics {
        let graph_stats = self.concept_graph.as_ref().map(|g| GraphStatistics {
            concept_count: g.concept_count(),
            relation_count: g.relation_count(),
        });

        let index_stats = self.bidirectional_index.as_ref().map(|i| i.get_statistics());

        PipelineStatistics {
            document_count: self.document_count,
            chunk_count: self.chunks.len(),
            graph_statistics: graph_stats,
            index_statistics: index_stats,
        }
    }

    /// Chunk text into smaller pieces
    fn chunk_text(&self, text: &str, document_id: &str) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let text_len = text.len();

        if text_len == 0 {
            return chunks;
        }

        let mut start = 0;
        let mut chunk_index = 0;

        while start < text_len {
            let end = (start + self.config.chunk_size).min(text_len);
            let chunk_text = text[start..end].to_string();

            let chunk = TextChunk::new(
                ChunkId::new(format!("{}_{}", document_id, chunk_index)),
                crate::core::DocumentId::new(document_id.to_string()),
                chunk_text,
                start,
                end,
            );

            chunks.push(chunk);

            // Move to next chunk with overlap
            if end >= text_len {
                break;
            }

            start = end - self.config.chunk_overlap;
            chunk_index += 1;
        }

        chunks
    }

    /// Normalize concept for entity ID
    fn normalize_concept(&self, concept: &str) -> String {
        concept
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .replace(' ', "_")
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatistics {
    /// Number of documents indexed
    pub document_count: usize,

    /// Number of chunks created
    pub chunk_count: usize,

    /// Concept graph statistics
    pub graph_statistics: Option<GraphStatistics>,

    /// Bidirectional index statistics
    pub index_statistics: Option<crate::entity::IndexStatistics>,
}

/// Concept graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Number of concepts in graph
    pub concept_count: usize,

    /// Number of relationships in graph
    pub relation_count: usize,
}

impl SearchResults {
    pub(crate) fn new(query: String) -> Self {
        Self {
            query,
            depth_reached: 0,
            total_chunks: 0,
            total_concepts_explored: 0,
            depth_results: Vec::new(),
            chunk_ids: Vec::new(),
            stop_reason: crate::lightrag::iterative_deepening::StopReason::MaxDepthReached,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_pipeline_creation() {
        let config = LazyPipelineConfig::default();
        let pipeline = LazyGraphRAGPipeline::new(config);

        assert_eq!(pipeline.document_count, 0);
        assert_eq!(pipeline.chunks.len(), 0);
    }

    #[test]
    fn test_index_and_build() {
        let mut pipeline = LazyGraphRAGPipeline::default();

        // Index a simple document
        pipeline.index_document(
            "test_doc",
            "Machine Learning is a subset of Artificial Intelligence. \
             Neural Networks are used in Deep Learning.",
        );

        assert_eq!(pipeline.document_count, 1);
        assert!(!pipeline.chunks.is_empty());

        // Build the graph
        pipeline.build_graph();

        assert!(pipeline.concept_graph.is_some());
        assert!(pipeline.bidirectional_index.is_some());
    }

    #[test]
    fn test_query_before_build() {
        let pipeline = LazyGraphRAGPipeline::default();

        // Query without building graph should return empty results
        let results = pipeline.query("machine learning");

        assert_eq!(results.total_chunks, 0);
        assert_eq!(results.chunk_ids.len(), 0);
    }

    #[test]
    fn test_pipeline_statistics() {
        let mut pipeline = LazyGraphRAGPipeline::default();

        pipeline.index_document("doc1", "Test document with some content");
        pipeline.build_graph();

        let stats = pipeline.get_statistics();

        assert_eq!(stats.document_count, 1);
        assert!(stats.chunk_count > 0);
        assert!(stats.graph_statistics.is_some());
    }

    #[test]
    fn test_chunking() {
        let config = LazyPipelineConfig {
            chunk_size: 10,
            chunk_overlap: 2,
            ..Default::default()
        };

        let pipeline = LazyGraphRAGPipeline::new(config);

        let text = "This is a test document";
        let chunks = pipeline.chunk_text(text, "test_doc");

        assert!(!chunks.is_empty());
        assert!(chunks[0].content.len() <= 10);
    }
}
