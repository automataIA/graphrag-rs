//! LightRAG: Simple and Fast Retrieval-Augmented Generation
//!
//! This module implements LightRAG's dual-level retrieval optimization,
//! achieving 6000x token reduction compared to traditional GraphRAG.
//!
//! Key innovation: Extract keywords at TWO levels:
//! - High-level: Broader topics, themes, concepts (abstract)
//! - Low-level: Specific entities, attributes, details (concrete)
//!
//! Reference: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (EMNLP 2025)
//! Paper: arXiv:2410.05779
//!
//! ## LazyGraphRAG
//!
//! This module also includes LazyGraphRAG components:
//! - Concept-based retrieval (no prior summarization)
//! - Noun phrase extraction (no LLM required)
//! - 0.1% of full GraphRAG indexing cost
//! - 700x cheaper query costs

pub mod dual_retrieval;
pub mod keyword_extraction;
pub mod graph_indexer;

#[cfg(feature = "lazygraphrag")]
pub mod concept_graph;

#[cfg(feature = "lazygraphrag")]
pub mod query_refinement;

#[cfg(feature = "lazygraphrag")]
pub mod iterative_deepening;

#[cfg(feature = "lazygraphrag")]
pub mod lazy_pipeline;

pub use dual_retrieval::{
    DualLevelRetriever, DualRetrievalConfig, DualRetrievalResults,
    MergeStrategy, SemanticSearcher,
};
pub use keyword_extraction::{
    KeywordExtractor, KeywordExtractorConfig, DualLevelKeywords,
};
pub use graph_indexer::{GraphIndexer, ExtractionResult, ExtractedEntity, ExtractedRelationship};

#[cfg(feature = "lazygraphrag")]
pub use query_refinement::{QueryRefiner, QueryRefinementConfig, RefinedQuery};

#[cfg(feature = "lazygraphrag")]
pub use iterative_deepening::{IterativeDeepeningSearch, SearchConfig, SearchResults, DepthResults, StopReason};

#[cfg(feature = "lazygraphrag")]
pub use lazy_pipeline::{LazyGraphRAGPipeline, LazyPipelineConfig, PipelineStatistics, GraphStatistics};
