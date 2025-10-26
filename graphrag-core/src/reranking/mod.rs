//! Reranking module for improving retrieval accuracy
//!
//! This module provides various reranking strategies to refine initial
//! retrieval results. Reranking typically uses more expensive but more
//! accurate models to score query-document pairs.

/// Confidence-based reranking utilities
pub mod confidence;
/// Cross-encoder reranking implementation
pub mod cross_encoder;

pub use cross_encoder::{
    CrossEncoder, CrossEncoderConfig, ConfidenceCrossEncoder,
    RankedResult, RerankingStats,
};
