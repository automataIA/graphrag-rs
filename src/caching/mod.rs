//! High-performance LLM response caching system
//!
//! This module provides a transparent caching layer for Language Model operations
//! using the moka cache library for high-performance concurrent caching with TTL support.
//!
//! ## Features
//!
//! - **Transparent caching**: Drop-in replacement for existing LLM clients
//! - **High performance**: Sub-millisecond response times for cached queries
//! - **Cost optimization**: Dramatic reduction in LLM API calls (6x+ cost reduction)
//! - **Multiple eviction policies**: LRU, LFU, TTL with configurable parameters
//! - **Intelligent cache keys**: Content-based hashing for optimal hit rates
//! - **Comprehensive monitoring**: Cache statistics and health metrics
//! - **Cache warming**: Preload frequently accessed content
//! - **Thread-safe**: Fully concurrent operations with lock-free performance
//!
//! ## Usage
//!
//! ```rust
//! use graphrag_rs::caching::{CachedLLMClient, CacheConfig, EvictionPolicy};
//! use graphrag_rs::generation::MockLLM;
//!
//! # async fn example() -> graphrag_rs::Result<()> {
//! // Create cache configuration
//! let cache_config = CacheConfig::builder()
//!     .max_capacity(10_000)
//!     .ttl_seconds(3600)
//!     .eviction_policy(EvictionPolicy::LRU)
//!     .enable_statistics(true)
//!     .build();
//!
//! // Wrap any LLM with caching
//! let base_llm = Box::new(MockLLM::new()?);
//! let cached_llm = CachedLLMClient::new(base_llm, cache_config).await?;
//!
//! // Use exactly like any other LLM
//! let response1 = cached_llm.complete("What is AI?").await?; // Cache miss
//! let response2 = cached_llm.complete("What is AI?").await?; // Cache hit!
//!
//! // Monitor cache performance
//! let stats = cached_llm.cache_statistics();
//! println!("Cache hit rate: {:.2}%", stats.hit_rate() * 100.0);
//! # Ok(())
//! # }
//! ```

pub mod cache_config;
pub mod cache_key;
pub mod client;
pub mod stats;
pub mod warming;

pub use cache_config::{CacheConfig, CacheConfigBuilder, EvictionPolicy};
pub use cache_key::{CacheKey, CacheKeyGenerator};
pub use client::CachedLLMClient;
pub use stats::{CacheStatistics, CacheHealth, CacheMetrics};
pub use warming::{CacheWarmer, WarmingStrategy, WarmingConfig};

use crate::core::GraphRAGError;

/// Re-export the LanguageModel trait for convenience
pub use crate::core::traits::{LanguageModel, GenerationParams, ModelInfo};

/// Cache-specific error types
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("Cache initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Cache key generation failed: {0}")]
    KeyGenerationFailed(String),

    #[error("Cache operation failed: {0}")]
    OperationFailed(String),

    #[error("Cache warming failed: {0}")]
    WarmingFailed(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl From<CacheError> for GraphRAGError {
    fn from(err: CacheError) -> Self {
        GraphRAGError::Generation {
            message: format!("Cache error: {err:?}"),
        }
    }
}

/// Result type for cache operations
pub type CacheResult<T> = std::result::Result<T, CacheError>;

/// Cache entry metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheEntry {
    /// The cached response
    pub response: String,
    /// When this entry was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// How many times this entry has been accessed
    pub access_count: u64,
    /// Last access time
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Optional metadata tags
    pub metadata: std::collections::HashMap<String, String>,
}

impl CacheEntry {
    pub fn new(response: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            response,
            created_at: now,
            access_count: 1,
            last_accessed: now,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = chrono::Utc::now();
    }

    pub fn age(&self) -> chrono::Duration {
        chrono::Utc::now() - self.created_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry() {
        let mut entry = CacheEntry::new("test response".to_string());
        assert_eq!(entry.response, "test response");
        assert_eq!(entry.access_count, 1);

        entry.access();
        assert_eq!(entry.access_count, 2);
        assert!(entry.age().num_seconds() >= 0);
    }
}