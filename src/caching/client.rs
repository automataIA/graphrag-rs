//! High-performance cached LLM client implementation

use super::stats::SharedCacheStatistics;
use super::{
    CacheConfig, CacheEntry, CacheError, CacheHealth, CacheKey, CacheKeyGenerator, CacheMetrics,
    CacheResult, CacheStatistics, CacheWarmer, WarmingConfig,
};
use crate::core::traits::{GenerationParams, LanguageModel, ModelInfo};
use crate::core::Result;
use moka::future::Cache;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// High-performance cached LLM client that wraps any LanguageModel implementation
pub struct CachedLLMClient<T: LanguageModel> {
    /// The underlying LLM client
    inner: Arc<T>,
    /// High-performance concurrent cache
    cache: Cache<String, CacheEntry>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache key generator
    key_generator: Arc<RwLock<CacheKeyGenerator>>,
    /// Cache statistics
    stats: SharedCacheStatistics,
    /// Cache warming configuration
    warming_config: Option<WarmingConfig>,
}

impl<T: LanguageModel + Send + Sync> CachedLLMClient<T> {
    /// Create a new cached LLM client
    pub async fn new(inner: T, config: CacheConfig) -> CacheResult<Self> {
        config.validate()?;

        let cache = Self::build_cache(&config).await?;
        let key_generator = Arc::new(RwLock::new(CacheKeyGenerator::new()));
        let stats = Arc::new(CacheStatistics::new());

        Ok(Self {
            inner: Arc::new(inner),
            cache,
            config,
            key_generator,
            stats,
            warming_config: None,
        })
    }

    /// Create a new cached LLM client with warming
    pub async fn with_warming(
        inner: T,
        config: CacheConfig,
        warming_config: WarmingConfig,
    ) -> CacheResult<Self> {
        let mut client = Self::new(inner, config).await?;
        client.warming_config = Some(warming_config);
        Ok(client)
    }

    /// Build the moka cache with the given configuration
    async fn build_cache(config: &CacheConfig) -> CacheResult<Cache<String, CacheEntry>> {
        let mut cache_builder = Cache::builder();

        // Set capacity
        cache_builder = cache_builder.max_capacity(config.max_capacity);

        // Set initial capacity for performance
        if let Some(initial) = config.initial_capacity {
            cache_builder = cache_builder.initial_capacity(initial as usize);
        }

        // Set TTL
        cache_builder = cache_builder.time_to_live(config.ttl_duration());

        // Set cleanup interval for expired entries
        cache_builder = cache_builder.time_to_idle(config.cleanup_interval());

        // Build the cache
        let cache = cache_builder.build();

        Ok(cache)
    }

    /// Execute a query with caching
    async fn execute_with_cache<F, Fut>(&self, cache_key: CacheKey, operation: F) -> Result<String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<String>>,
    {
        let start_time = Instant::now();

        // Try to get from cache first
        if let Some(mut entry) = self.cache.get(&cache_key.key_hash).await {
            // Update access statistics
            entry.access();

            // Update cache with new access info
            self.cache
                .insert(cache_key.key_hash.clone(), entry.clone())
                .await;

            // Record cache hit
            let time_saved = start_time.elapsed();
            self.stats.record_hit(time_saved);

            return Ok(entry.response);
        }

        // Cache miss - execute the operation
        self.stats.record_miss();

        let response = operation().await?;

        // Validate response size
        if response.len() > self.config.max_entry_size {
            self.stats.record_error();
            return Err(crate::core::GraphRAGError::Generation {
                message: format!(
                    "Response size ({} bytes) exceeds maximum cache entry size ({} bytes)",
                    response.len(),
                    self.config.max_entry_size
                ),
            });
        }

        // Create cache entry
        let entry = CacheEntry::new(response.clone());
        let entry_size = self.estimate_entry_size(&entry);

        // Insert into cache
        self.cache.insert(cache_key.key_hash, entry).await;

        // Record insertion
        self.stats.record_insertion(entry_size);

        Ok(response)
    }

    /// Estimate the memory size of a cache entry
    fn estimate_entry_size(&self, entry: &CacheEntry) -> usize {
        // Rough estimation: response size + metadata overhead
        entry.response.len()
            + entry
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + 100 // Fixed overhead for timestamps and counters
    }

    /// Generate cache key for a request
    async fn generate_cache_key(
        &self,
        prompt: &str,
        params: Option<&GenerationParams>,
    ) -> CacheResult<CacheKey> {
        let key_gen = self.key_generator.read().await;
        let model_info = self.inner.model_info();
        key_gen.generate_key_with_params(prompt, params, Some(&model_info.name))
    }

    /// Check if a query is already cached
    pub async fn is_cached(&self, prompt: &str, params: Option<&GenerationParams>) -> bool {
        if let Ok(cache_key) = self.generate_cache_key(prompt, params).await {
            self.cache.get(&cache_key.key_hash).await.is_some()
        } else {
            false
        }
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> CacheMetrics {
        self.stats.snapshot()
    }

    /// Get cache health status
    pub fn cache_health(&self) -> CacheHealth {
        let metrics = self.cache_statistics();
        CacheHealth::evaluate(metrics, self.config.max_capacity)
    }

    /// Clear the entire cache
    pub async fn clear_cache(&self) {
        self.cache.invalidate_all();
        // Reset statistics but keep current tracking
        // Don't reset current_size as invalidate_all will trigger eviction events
    }

    /// Remove a specific entry from cache
    pub async fn invalidate(
        &self,
        prompt: &str,
        params: Option<&GenerationParams>,
    ) -> CacheResult<bool> {
        let cache_key = self.generate_cache_key(prompt, params).await?;

        if let Some(entry) = self.cache.get(&cache_key.key_hash).await {
            let entry_size = self.estimate_entry_size(&entry);
            self.cache.invalidate(&cache_key.key_hash).await;
            self.stats.record_eviction(entry_size);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get cache utilization (0.0 to 1.0)
    pub fn cache_utilization(&self) -> f64 {
        let current_size = self.cache.entry_count();
        if self.config.max_capacity == 0 {
            0.0
        } else {
            (current_size as f64 / self.config.max_capacity as f64).min(1.0)
        }
    }

    /// Warm the cache if warming is configured
    pub async fn warm_cache(&self) -> CacheResult<()> {
        if let Some(ref warming_config) = self.warming_config {
            let warmer = CacheWarmer::new(warming_config.clone())?;
            let results = warmer.warm_cache(self).await?;
            results.print();
            Ok(())
        } else {
            Err(CacheError::Configuration(
                "Cache warming not configured".to_string(),
            ))
        }
    }

    /// Update cache key generation strategy
    pub async fn update_key_strategy(&self, new_generator: CacheKeyGenerator) {
        let mut key_gen = self.key_generator.write().await;
        *key_gen = new_generator;
    }

    /// Get the underlying LLM client
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Export cache contents for persistence (if configured)
    pub async fn export_cache(&self) -> CacheResult<Vec<(String, CacheEntry)>> {
        let entries = Vec::new();

        // Note: moka doesn't provide direct iteration over all entries
        // This is a simplified version - in production you might want to
        // implement a separate persistence layer

        // For now, we return an empty vector as moka doesn't expose entry iteration
        // In a real implementation, you'd maintain a separate index or use a different cache
        // that supports iteration

        Ok(entries)
    }

    /// Print cache statistics
    pub fn print_statistics(&self) {
        self.stats.print();
    }

    /// Print cache health report
    pub fn print_health(&self) {
        let health = self.cache_health();
        health.print();
    }

    /// Get detailed cache information
    pub fn cache_info(&self) -> CacheInfo {
        let metrics = self.cache_statistics();
        let health = self.cache_health();

        CacheInfo {
            config: self.config.clone(),
            metrics,
            health,
            entry_count: self.cache.entry_count(),
            weighted_size: self.cache.weighted_size(),
        }
    }
}

/// Comprehensive cache information
#[derive(Debug)]
pub struct CacheInfo {
    pub config: CacheConfig,
    pub metrics: CacheMetrics,
    pub health: CacheHealth,
    pub entry_count: u64,
    pub weighted_size: u64,
}

impl CacheInfo {
    pub fn print(&self) {
        println!("Cache Information:");
        println!("  Configuration:");
        println!("    Max capacity: {}", self.config.max_capacity);
        println!("    TTL: {}s", self.config.ttl_seconds);
        println!("    Eviction policy: {:?}", self.config.eviction_policy);
        println!("    Statistics enabled: {}", self.config.enable_statistics);
        println!("  Current state:");
        println!("    Entry count: {}", self.entry_count);
        println!("    Weighted size: {}", self.weighted_size);
        println!(
            "    Utilization: {:.1}%",
            (self.entry_count as f64 / self.config.max_capacity as f64 * 100.0).min(100.0)
        );
        println!();
        self.health.print();
    }
}

// Implement LanguageModel trait for the cached client
impl<T: LanguageModel + Send + Sync> LanguageModel for CachedLLMClient<T> {
    type Error = crate::core::GraphRAGError;

    fn complete(&self, prompt: &str) -> Result<String> {
        // For sync trait, we need to use a blocking approach
        // In practice, you might want to use async version or handle this differently
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let cache_key = self.generate_cache_key(prompt, None).await.map_err(|e| {
                    crate::core::GraphRAGError::Generation {
                        message: format!("Cache key generation failed: {e}"),
                    }
                })?;

                self.execute_with_cache(cache_key, || async { self.inner.complete(prompt) })
                    .await
            })
        })
    }

    fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let cache_key = self
                    .generate_cache_key(prompt, Some(&params))
                    .await
                    .map_err(|e| crate::core::GraphRAGError::Generation {
                        message: format!("Cache key generation failed: {e}"),
                    })?;

                self.execute_with_cache(cache_key, || async {
                    self.inner.complete_with_params(prompt, params.clone())
                })
                .await
            })
        })
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn model_info(&self) -> ModelInfo {
        let mut info = self.inner.model_info();
        info.name = format!("Cached({})", info.name);
        info
    }
}

// Async versions of the LanguageModel methods
impl<T: LanguageModel + Send + Sync> CachedLLMClient<T> {
    /// Async version of complete
    pub async fn complete_async(&self, prompt: &str) -> Result<String> {
        let cache_key = self.generate_cache_key(prompt, None).await.map_err(|e| {
            crate::core::GraphRAGError::Generation {
                message: format!("Cache key generation failed: {e}"),
            }
        })?;

        self.execute_with_cache(cache_key, || async { self.inner.complete(prompt) })
            .await
    }

    /// Async version of complete_with_params
    pub async fn complete_with_params_async(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let cache_key = self
            .generate_cache_key(prompt, Some(&params))
            .await
            .map_err(|e| crate::core::GraphRAGError::Generation {
                message: format!("Cache key generation failed: {e}"),
            })?;

        self.execute_with_cache(cache_key, || async {
            self.inner.complete_with_params(prompt, params.clone())
        })
        .await
    }
}

impl<T: LanguageModel> Clone for CachedLLMClient<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            cache: self.cache.clone(),
            config: self.config.clone(),
            key_generator: Arc::clone(&self.key_generator),
            stats: Arc::clone(&self.stats),
            warming_config: self.warming_config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::GenerationParams;
    use crate::generation::MockLLM;

    #[tokio::test]
    async fn test_cached_client_creation() {
        let mock_llm = MockLLM::new().unwrap();
        let config = CacheConfig::default();
        let client = CachedLLMClient::new(mock_llm, config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_cache_hit_miss() {
        let mock_llm = MockLLM::new().unwrap();
        let config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, config).await.unwrap();

        let prompt = "What is AI?";

        // First call should be a cache miss
        let response1 = client.complete_async(prompt).await.unwrap();
        let stats1 = client.cache_statistics();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);

        // Second call should be a cache hit
        let response2 = client.complete_async(prompt).await.unwrap();
        let stats2 = client.cache_statistics();
        assert_eq!(stats2.cache_misses, 1);
        assert_eq!(stats2.cache_hits, 1);

        // Responses should be identical
        assert_eq!(response1, response2);
    }

    #[tokio::test]
    async fn test_cache_with_params() {
        let mock_llm = MockLLM::new().unwrap();
        let config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, config).await.unwrap();

        let prompt = "What is AI?";
        let params1 = GenerationParams {
            temperature: Some(0.7),
            ..Default::default()
        };
        let params2 = GenerationParams {
            temperature: Some(0.9),
            ..Default::default()
        };

        // Different parameters should result in different cache entries
        let _response1 = client
            .complete_with_params_async(prompt, params1)
            .await
            .unwrap();
        let _response2 = client
            .complete_with_params_async(prompt, params2)
            .await
            .unwrap();

        let stats = client.cache_statistics();
        assert_eq!(stats.cache_misses, 2); // Both should be misses
        assert_eq!(stats.cache_hits, 0);
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let mock_llm = MockLLM::new().unwrap();
        let config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, config).await.unwrap();

        let prompt = "What is AI?";

        // Cache the response
        let _response1 = client.complete_async(prompt).await.unwrap();
        assert!(client.is_cached(prompt, None).await);

        // Invalidate the entry
        let was_cached = client.invalidate(prompt, None).await.unwrap();
        assert!(was_cached);
        assert!(!client.is_cached(prompt, None).await);
    }

    #[tokio::test]
    async fn test_cache_utilization() {
        let mock_llm = MockLLM::new().unwrap();
        let mut config = CacheConfig::development();
        config.max_capacity = 10; // Small cache for testing
        let client = CachedLLMClient::new(mock_llm, config).await.unwrap();

        assert_eq!(client.cache_utilization(), 0.0);

        // Add some entries
        for i in 0..5 {
            let prompt = format!("Query {i}");
            let _ = client.complete_async(&prompt).await.unwrap();
        }

        let utilization = client.cache_utilization();
        assert!(utilization > 0.0);
        assert!(utilization <= 1.0);
    }

    #[tokio::test]
    async fn test_cache_health() {
        let mock_llm = MockLLM::new().unwrap();
        let config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, config).await.unwrap();

        // Generate some cache activity
        for i in 0..10 {
            let prompt = format!("Query {i}");
            let _ = client.complete_async(&prompt).await.unwrap();
        }

        let health = client.cache_health();
        assert!(matches!(
            health.status,
            super::super::stats::HealthStatus::Healthy | super::super::stats::HealthStatus::Warning
        ));
    }

    #[test]
    fn test_language_model_trait() {
        let mock_llm = MockLLM::new().unwrap();
        let config = CacheConfig::development();

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let client = CachedLLMClient::new(mock_llm, config).await.unwrap();

            // Test sync trait methods
            assert!(client.is_available());

            let model_info = client.model_info();
            assert!(model_info.name.contains("Cached"));

            let response = client.complete("Test prompt").unwrap();
            assert!(!response.is_empty());
        });
    }
}
