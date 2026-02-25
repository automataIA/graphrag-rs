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
pub mod graph_indexer;
pub mod keyword_extraction;

#[cfg(feature = "lazygraphrag")]
pub mod concept_graph;

#[cfg(feature = "lazygraphrag")]
pub mod concept_selection;

#[cfg(feature = "lazygraphrag")]
pub mod query_concepts;

#[cfg(feature = "lazygraphrag")]
pub mod query_refinement;

#[cfg(feature = "lazygraphrag")]
pub mod iterative_deepening;

#[cfg(feature = "lazygraphrag")]
pub mod lazy_pipeline;

pub use dual_retrieval::{
    DualLevelRetriever, DualRetrievalConfig, DualRetrievalResults, MergeStrategy, SemanticSearcher,
};
pub use graph_indexer::{ExtractedEntity, ExtractedRelationship, ExtractionResult, GraphIndexer};
pub use keyword_extraction::{DualLevelKeywords, KeywordExtractor, KeywordExtractorConfig};

#[cfg(feature = "lazygraphrag")]
pub use query_refinement::{QueryRefinementConfig, QueryRefiner, RefinedQuery};

#[cfg(feature = "lazygraphrag")]
pub use iterative_deepening::{
    DepthResults, IterativeDeepeningSearch, SearchConfig, SearchResults, StopReason,
};

#[cfg(feature = "lazygraphrag")]
pub use lazy_pipeline::{
    GraphStatistics, LazyGraphRAGPipeline, LazyPipelineConfig, PipelineStatistics,
};

#[cfg(feature = "lazygraphrag")]
pub use concept_selection::{
    ConceptRanker, ConceptRankingStats, ConceptSelectionConfig, RankedConcept,
};

#[cfg(feature = "lazygraphrag")]
pub use query_concepts::{MatchedConcept, MatchingStats, QueryConceptMatcher, QueryMatchConfig};
