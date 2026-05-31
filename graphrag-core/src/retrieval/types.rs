//! Public data types of the retrieval pipeline.
//!
//! Phase 4 split: `RetrievalConfig`, `SearchResult`, `ResultType`, `QueryAnalysis`,
//! `QueryType`, `QueryIntent`, `QueryAnalysisResult`, `QueryResult`,
//! `RetrievalStatistics` were extracted verbatim from `retrieval/mod.rs`.
//! Re-exported via `pub use types::*;` in `mod.rs` so external paths are unchanged.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use crate::core::{ChunkId, EntityId, KnowledgeGraph};

/// Configuration parameters for the retrieval system
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum number of results to return
    pub top_k: usize,
    /// Minimum similarity score threshold for results (typically -1.0 to 1.0)
    pub similarity_threshold: f32,
    /// Maximum depth for graph relationship expansion
    pub max_expansion_depth: usize,
    /// Weight for entity-based results in scoring (0.0 to 1.0)
    pub entity_weight: f32,
    /// Weight for chunk-based results in scoring (0.0 to 1.0)
    pub chunk_weight: f32,
    /// Weight for graph-based results in scoring (0.0 to 1.0)
    pub graph_weight: f32,
    /// Enable concept-based chunk filtering (requires lazygraphrag feature)
    #[cfg(feature = "lazygraphrag")]
    pub use_concept_filtering: bool,
    /// Top-K concepts to select for filtering (requires lazygraphrag feature)
    #[cfg(feature = "lazygraphrag")]
    pub concept_top_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            similarity_threshold: 0.7,
            max_expansion_depth: 2,
            entity_weight: 0.4,
            chunk_weight: 0.4,
            graph_weight: 0.2,
            #[cfg(feature = "lazygraphrag")]
            use_concept_filtering: false,
            #[cfg(feature = "lazygraphrag")]
            concept_top_k: 20,
        }
    }
}

/// A search result containing relevant information
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique identifier for this result
    pub id: String,
    /// Content or description of the result
    pub content: String,
    /// Relevance score (higher is better)
    pub score: f32,
    /// Type of result (entity, chunk, graph path, etc.)
    pub result_type: ResultType,
    /// Names of entities associated with this result
    pub entities: Vec<String>,
    /// IDs of source chunks this result is derived from
    pub source_chunks: Vec<String>,
}

/// Type of search result indicating the retrieval strategy used
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResultType {
    /// Result from entity-based retrieval
    Entity,
    /// Result from text chunk retrieval
    Chunk,
    /// Result from graph path traversal
    GraphPath,
    /// Result from hierarchical document summarization
    HierarchicalSummary,
    /// Result from combining multiple retrieval strategies
    Hybrid,
}

// ============================================================================
// QUERY ANALYSIS - Adaptive retrieval strategy
// ============================================================================

/// Query analysis results to determine optimal retrieval strategy
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Type of query based on content analysis
    pub query_type: QueryType,
    /// Key entities detected in the query
    pub key_entities: Vec<String>,
    /// Conceptual terms extracted from the query
    pub concepts: Vec<String>,
    /// Inferred user intent from the query
    pub intent: QueryIntent,
    /// Query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
}

/// Classification of query types for adaptive retrieval strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// Queries focused on specific entities
    EntityFocused,
    /// Abstract concept queries requiring broader context
    Conceptual,
    /// Specific fact retrieval queries
    Factual,
    /// Open-ended exploration queries
    Exploratory,
    /// Queries about relationships between entities
    Relationship,
}

/// User intent classification for result presentation
#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    /// User wants a high-level summary or overview
    Overview,
    /// User wants detailed, specific information
    Detailed,
    /// User wants to compare multiple items
    Comparative,
    /// User wants to understand cause-effect relationships
    Causal,
    /// User wants time-based or chronological information
    Temporal,
}

/// Query analysis result with additional metadata for adaptive retrieval
#[derive(Debug, Clone)]
pub struct QueryAnalysisResult {
    /// Detected query type
    pub query_type: QueryType,
    /// Confidence score for the detected query type (0.0 to 1.0)
    pub confidence: f32,
    /// Keywords extracted and matched from the query
    pub keywords_matched: Vec<String>,
    /// Recommended retrieval strategies based on analysis
    pub suggested_strategies: Vec<String>,
    /// Overall query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
}

/// Query result with hierarchical summary
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Original query string
    pub query: String,
    /// List of search results
    pub results: Vec<SearchResult>,
    /// Optional generated summary of all results
    pub summary: Option<String>,
    /// Additional metadata about the query execution
    pub metadata: HashMap<String, String>,
}

/// Statistics about the retrieval system
#[derive(Debug)]
pub struct RetrievalStatistics {
    /// Number of vectors indexed in the system
    pub indexed_vectors: usize,
    /// Dimensionality of the vector embeddings
    pub vector_dimension: usize,
    /// Whether the vector index has been built
    pub index_built: bool,
    /// Current retrieval configuration
    pub config: RetrievalConfig,
}

impl RetrievalStatistics {
    /// Print retrieval statistics
    #[allow(dead_code)]
    pub fn print(&self) {
        #[cfg(feature = "tracing")]
        tracing::info!("Retrieval System Statistics:");
        #[cfg(feature = "tracing")]
        tracing::info!("  Indexed vectors: {}", self.indexed_vectors);
        #[cfg(feature = "tracing")]
        tracing::info!("  Vector dimension: {}", self.vector_dimension);
        #[cfg(feature = "tracing")]
        tracing::info!("  Index built: {}", self.index_built);
        #[cfg(feature = "tracing")]
        tracing::info!("  Configuration:");
        #[cfg(feature = "tracing")]
        tracing::info!("    Top K: {}", self.config.top_k);
        #[cfg(feature = "tracing")]
        tracing::info!(
            "    Similarity threshold: {:.2}",
            self.config.similarity_threshold
        );
        #[cfg(feature = "tracing")]
        tracing::info!(
            "    Max expansion depth: {}",
            self.config.max_expansion_depth
        );
        #[cfg(feature = "tracing")]
        tracing::info!("    Entity weight: {:.2}", self.config.entity_weight);
        #[cfg(feature = "tracing")]
        tracing::info!("    Chunk weight: {:.2}", self.config.chunk_weight);
        #[cfg(feature = "tracing")]
        tracing::info!("    Graph weight: {:.2}", self.config.graph_weight);
    }
}
