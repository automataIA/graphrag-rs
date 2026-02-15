//! Monitoring and benchmarking utilities for GraphRAG
//!
//! This module provides tools for measuring and tracking GraphRAG performance:
//! - Benchmarking system for quality evaluation
//! - Performance metrics tracking
//! - Cost and token usage monitoring

/// Benchmarking system for quality improvements
pub mod benchmark;

/// Metrics collection system for monitoring
#[cfg(feature = "dashmap")]
pub mod metrics_collector;

pub use benchmark::{
    BenchmarkConfig, BenchmarkDataset, BenchmarkQuery, BenchmarkRunner,
    BenchmarkSummary, LatencyMetrics, QualityMetrics, QueryBenchmark,
    TokenMetrics,
};

#[cfg(feature = "dashmap")]
pub use metrics_collector::{MetricsCollector, HistogramStats};
