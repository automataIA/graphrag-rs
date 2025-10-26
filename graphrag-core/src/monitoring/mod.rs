//! Monitoring and benchmarking utilities for GraphRAG
//!
//! This module provides tools for measuring and tracking GraphRAG performance:
//! - Benchmarking system for quality evaluation
//! - Performance metrics tracking
//! - Cost and token usage monitoring

/// Benchmarking system for quality improvements
pub mod benchmark;

pub use benchmark::{
    BenchmarkConfig, BenchmarkDataset, BenchmarkQuery, BenchmarkRunner,
    BenchmarkSummary, LatencyMetrics, QualityMetrics, QueryBenchmark,
    TokenMetrics,
};
