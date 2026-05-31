//! # GraphRAG Core
//!
//! Portable core library for GraphRAG - works on both native and WASM platforms.
//!
//! This is the foundational crate that provides:
//! - Knowledge graph construction and management
//! - Entity extraction and linking
//! - Vector embeddings and similarity search
//! - Graph algorithms (PageRank, community detection)
//! - Retrieval systems (semantic, keyword, hybrid)
//! - Caching and optimization
//!
//! ## Platform Support
//!
//! - **Native**: Full feature set with optional CUDA/Metal GPU acceleration
//! - **WASM**: Browser-compatible with Voy vector search and Candle embeddings
//!
//! ## Feature Flags
//!
//! - `wasm`: Enable WASM compatibility (uses Voy instead of HNSW)
//! - `cuda`: Enable NVIDIA GPU acceleration via Candle
//! - `metal`: Enable Apple Silicon GPU acceleration
//! - `webgpu`: Enable WebGPU acceleration for browser (via Burn)
//! - `pagerank`: Enable PageRank-based retrieval
//! - `lightrag`: Enable LightRAG optimizations (6000x token reduction)
//! - `caching`: Enable intelligent LLM response caching
//!
//! ## Quick Start
//!
//! ```rust
//! use graphrag_core::{GraphRAG, Config};
//!
//! # fn example() -> graphrag_core::Result<()> {
//! let config = Config::default();
//! let mut graphrag = GraphRAG::new(config)?;
//! graphrag.initialize()?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
// Note: WASM with wasm-bindgen DOES use std, so we don't disable it

// ================================
// MODULE DECLARATIONS
// ================================

// Core modules (always available)
/// Configuration management and loading
pub mod config;
/// Core traits and types
pub mod core;
/// Entity extraction and management
pub mod entity;
/// Text generation and LLM interactions (async feature only)
#[cfg(feature = "async")]
pub mod generation;
/// Graph data structures and algorithms
pub mod graph;
/// Retrieval strategies and implementations
pub mod retrieval;
/// Storage backends and persistence
#[cfg(any(
    feature = "memory-storage",
    feature = "persistent-storage",
    feature = "async"
))]
pub mod storage;
/// Text processing and chunking
pub mod text;
/// Vector operations and embeddings
pub mod vector;

/// Builder pattern implementations
pub mod builder;
/// Embedding generation and providers
pub mod embeddings;
/// Natural language processing utilities
pub mod nlp;
/// Ollama LLM integration
pub mod ollama;
/// Persistence layer for knowledge graphs (workspace management always available)
pub mod persistence;
/// Query processing and execution
pub mod query;
/// Text summarization capabilities
pub mod summarization;

// Pipeline modules
/// Data processing pipelines
pub mod pipeline;

// Advanced features (feature-gated)
#[cfg(feature = "parallel-processing")]
pub mod parallel;

#[cfg(feature = "lightrag")]
/// LightRAG dual-level retrieval optimization
pub mod lightrag;

/// Composable pipeline executor for build-graph operations
pub mod pipeline_executor;

// Utility modules
/// Reranking utilities for improving search result quality
pub mod reranking;

/// Monitoring, benchmarking, and performance tracking
pub mod monitoring;

/// RAG answer evaluation and criticism
pub mod critic;

/// Evaluation framework for query results and pipeline validation
pub mod evaluation;

/// Graph optimization (weight optimization, DW-GRPO)
#[cfg(feature = "async")]
pub mod optimization;

/// API endpoints and handlers
#[cfg(feature = "api")]
pub mod api;

/// Inference module for model predictions
pub mod inference;

/// Multi-document corpus processing
#[cfg(feature = "corpus-processing")]
pub mod corpus;

// Feature-gated modules
#[cfg(feature = "async")]
/// Async GraphRAG implementation
pub mod async_graphrag;

#[cfg(feature = "async")]
/// Async processing pipelines
pub mod async_processing;

#[cfg(feature = "caching")]
/// Caching utilities for LLM responses
pub mod caching;

#[cfg(feature = "function-calling")]
/// Function calling capabilities for LLMs
pub mod function_calling;

#[cfg(feature = "incremental")]
/// Incremental graph updates
pub mod incremental;

#[cfg(feature = "rograg")]
/// ROGRAG (Robustly Optimized GraphRAG) implementation
pub mod rograg;

// Future utility modules (optional, not currently needed):
// pub mod automatic_entity_linking;  // Advanced entity linking
// pub mod phase_saver;               // Phase state persistence

// ================================
// PUBLIC API EXPORTS
// ================================

/// Prelude module containing the most commonly used types
///
/// Import everything you need with a single line:
/// ```rust
/// use graphrag_core::prelude::*;
/// ```
///
/// This includes:
/// - `GraphRAG` - The main orchestrator
/// - `Config` - Configuration management
/// - `GraphRAGBuilder` - Fluent configuration builder
/// - Core types: `Document`, `Entity`, `Relationship`, `TextChunk`
/// - Error handling: `Result`, `GraphRAGError`
pub mod prelude {
    // Main entry point
    pub use crate::GraphRAG;

    // Configuration & Builders
    pub use crate::builder::GraphRAGBuilder;
    pub use crate::builder::TypedBuilder;
    pub use crate::config::Config;

    // Error handling
    pub use crate::core::{GraphRAGError, Result};

    // Core data types
    pub use crate::core::{
        ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, KnowledgeGraph,
        Relationship, TextChunk,
    };

    // Search results and explained answers
    pub use crate::retrieval::SearchResult;
    pub use crate::retrieval::{ExplainedAnswer, ReasoningStep, SourceReference, SourceType};

    // Pipeline executor
    pub use crate::pipeline_executor::{PipelineExecutor, PipelineReport};

    // Config deserialization helper
    pub use crate::config::setconfig::SetConfig;
}

// Re-export core types
pub use crate::config::Config;
pub use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, ErrorContext, ErrorSeverity,
    ErrorSuggestion, GraphRAGError, KnowledgeGraph, Relationship, Result, TextChunk,
};

// Re-export core traits (async feature only)
#[cfg(feature = "async")]
pub use crate::core::traits::{
    Embedder, EntityExtractor, GraphStore, LanguageModel, Retriever, Storage, VectorStore,
};

// Storage exports (when storage features are enabled)
#[cfg(feature = "memory-storage")]
pub use crate::storage::MemoryStorage;

// Re-export builder (GraphRAGBuilder exists, ConfigPreset and LLMProvider not yet implemented)
pub use crate::builder::GraphRAGBuilder;
// Note: GraphRAG struct is already public (defined at line 247)
// Note: builder::GraphRAG is a placeholder - the real implementation is the main GraphRAG struct

// Feature-gated exports
#[cfg(feature = "lightrag")]
pub use crate::lightrag::{
    DualLevelKeywords, DualLevelRetriever, DualRetrievalConfig, DualRetrievalResults,
    KeywordExtractor, KeywordExtractorConfig, MergeStrategy, SemanticSearcher,
};

#[cfg(feature = "pagerank")]
pub use crate::graph::pagerank::{PageRankConfig, PersonalizedPageRank};

#[cfg(feature = "leiden")]
pub use crate::graph::leiden::{HierarchicalCommunities, LeidenCommunityDetector, LeidenConfig};

#[cfg(feature = "cross-encoder")]
pub use crate::reranking::cross_encoder::{
    ConfidenceCrossEncoder, CrossEncoder, CrossEncoderConfig, RankedResult, RerankingStats,
};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::pagerank_retrieval::{PageRankRetrievalSystem, ScoredResult};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::hipporag_ppr::{Fact, HippoRAGConfig, HippoRAGRetriever};


// ================================
// MAIN GRAPHRAG SYSTEM
// ================================
//
// The `GraphRAG` orchestrator type lives in `graphrag.rs`. It is re-exported
// here so external callers continue to import it via `graphrag_core::GraphRAG`.

mod graphrag;
pub use graphrag::GraphRAG;
