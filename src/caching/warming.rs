//! Cache warming strategies for improved performance

use super::{CacheError, CacheResult, CachedLLMClient};
use crate::core::traits::{GenerationParams, LanguageModel};
use std::time::Duration;

/// Cache warming strategies
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WarmingStrategy {
    /// Warm cache with predefined common queries
    PredefinedQueries,
    /// Warm cache based on query patterns from logs
    LogBasedPatterns,
    /// Warm cache with synthetic variations of common queries
    SyntheticVariations,
    /// Warm cache with frequently accessed content
    FrequencyBased,
    /// Custom warming with user-provided queries
    Custom,
}

/// Configuration for cache warming
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarmingConfig {
    /// Warming strategy to use
    pub strategy: WarmingStrategy,

    /// Maximum number of queries to warm
    pub max_queries: usize,

    /// Delay between warming requests to avoid overloading
    pub delay_between_requests: Duration,

    /// Whether to warm cache in background
    pub background_warming: bool,

    /// Custom queries for warming (used with Custom strategy)
    pub custom_queries: Vec<String>,

    /// Parameters to use for warming requests
    pub warming_params: Option<GenerationParams>,

    /// Whether to continue warming on errors
    pub continue_on_error: bool,

    /// Maximum errors before stopping warming
    pub max_errors: usize,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            strategy: WarmingStrategy::PredefinedQueries,
            max_queries: 50,
            delay_between_requests: Duration::from_millis(100),
            background_warming: true,
            custom_queries: Vec::new(),
            warming_params: None,
            continue_on_error: true,
            max_errors: 5,
        }
    }
}

impl WarmingConfig {
    /// Create a new warming config builder
    pub fn builder() -> WarmingConfigBuilder {
        WarmingConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> CacheResult<()> {
        if self.max_queries == 0 {
            return Err(CacheError::Configuration(
                "max_queries must be greater than 0".to_string(),
            ));
        }

        if matches!(self.strategy, WarmingStrategy::Custom) && self.custom_queries.is_empty() {
            return Err(CacheError::Configuration(
                "custom_queries must not be empty when using Custom strategy".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for warming configuration
pub struct WarmingConfigBuilder {
    config: WarmingConfig,
}

impl WarmingConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: WarmingConfig::default(),
        }
    }

    pub fn strategy(mut self, strategy: WarmingStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn max_queries(mut self, max: usize) -> Self {
        self.config.max_queries = max;
        self
    }

    pub fn delay_between_requests(mut self, delay: Duration) -> Self {
        self.config.delay_between_requests = delay;
        self
    }

    pub fn background_warming(mut self, enabled: bool) -> Self {
        self.config.background_warming = enabled;
        self
    }

    pub fn custom_queries(mut self, queries: Vec<String>) -> Self {
        self.config.custom_queries = queries;
        self
    }

    pub fn warming_params(mut self, params: GenerationParams) -> Self {
        self.config.warming_params = Some(params);
        self
    }

    pub fn continue_on_error(mut self, enabled: bool) -> Self {
        self.config.continue_on_error = enabled;
        self
    }

    pub fn max_errors(mut self, max: usize) -> Self {
        self.config.max_errors = max;
        self
    }

    pub fn build(self) -> WarmingConfig {
        self.config
    }

    pub fn build_validated(self) -> CacheResult<WarmingConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for WarmingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache warming implementation
pub struct CacheWarmer {
    config: WarmingConfig,
}

impl CacheWarmer {
    /// Create a new cache warmer
    pub fn new(config: WarmingConfig) -> CacheResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Warm the cache using the configured strategy
    pub async fn warm_cache<T>(&self, client: &CachedLLMClient<T>) -> CacheResult<WarmingResults>
    where
        T: LanguageModel + Send + Sync,
    {
        let queries = self.generate_warming_queries().await?;
        let mut results = WarmingResults::new();

        println!(
            "Starting cache warming with {} queries using strategy {:?}",
            queries.len(),
            self.config.strategy
        );

        let mut error_count = 0;

        for (i, query) in queries.iter().enumerate() {
            if i >= self.config.max_queries {
                break;
            }

            let start_time = std::time::Instant::now();

            match self.warm_single_query(client, query).await {
                Ok(was_cached) => {
                    let duration = start_time.elapsed();
                    results.successful_queries += 1;

                    if was_cached {
                        results.cache_hits += 1;
                    } else {
                        results.cache_misses += 1;
                    }

                    results.total_time += duration;

                    if !was_cached {
                        println!(
                            "  Warmed query {}/{}: {} ({:.2}ms)",
                            i + 1,
                            queries.len().min(self.config.max_queries),
                            Self::truncate_query(query, 50),
                            duration.as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    results.failed_queries += 1;
                    error_count += 1;

                    eprintln!(
                        "Failed to warm query '{}': {}",
                        Self::truncate_query(query, 50),
                        e
                    );

                    if !self.config.continue_on_error || error_count >= self.config.max_errors {
                        return Err(CacheError::WarmingFailed(format!(
                            "Too many errors during warming: {error_count}"
                        )));
                    }
                }
            }

            // Add delay between requests
            if i < queries.len() - 1 && !self.config.delay_between_requests.is_zero() {
                // Simple synchronous delay since we don't have tokio time
                std::thread::sleep(self.config.delay_between_requests);
            }
        }

        results.calculate_statistics();

        println!(
            "Cache warming completed: {} successful, {} failed, {:.2}% cache hit rate",
            results.successful_queries,
            results.failed_queries,
            results.cache_hit_rate * 100.0
        );

        Ok(results)
    }

    /// Warm cache in the background
    pub async fn warm_cache_background<T>(
        &self,
        client: CachedLLMClient<T>,
    ) -> CacheResult<tokio::task::JoinHandle<CacheResult<WarmingResults>>>
    where
        T: LanguageModel + Send + Sync + 'static,
    {
        let warmer = Self::new(self.config.clone())?;

        let handle = tokio::spawn(async move { warmer.warm_cache(&client).await });

        Ok(handle)
    }

    /// Generate warming queries based on the configured strategy
    async fn generate_warming_queries(&self) -> CacheResult<Vec<String>> {
        match self.config.strategy {
            WarmingStrategy::PredefinedQueries => Ok(self.get_predefined_queries()),
            WarmingStrategy::LogBasedPatterns => self.get_log_based_queries().await,
            WarmingStrategy::SyntheticVariations => Ok(self.get_synthetic_variations()),
            WarmingStrategy::FrequencyBased => self.get_frequency_based_queries().await,
            WarmingStrategy::Custom => Ok(self.config.custom_queries.clone()),
        }
    }

    /// Get predefined common queries for warming
    fn get_predefined_queries(&self) -> Vec<String> {
        vec![
            "What is artificial intelligence?".to_string(),
            "Explain machine learning".to_string(),
            "What are neural networks?".to_string(),
            "Define deep learning".to_string(),
            "What is natural language processing?".to_string(),
            "Explain computer vision".to_string(),
            "What is reinforcement learning?".to_string(),
            "Define data science".to_string(),
            "What is big data?".to_string(),
            "Explain cloud computing".to_string(),
            "What is blockchain?".to_string(),
            "Define cybersecurity".to_string(),
            "What is the Internet of Things?".to_string(),
            "Explain quantum computing".to_string(),
            "What is edge computing?".to_string(),
            "Define DevOps".to_string(),
            "What is microservices architecture?".to_string(),
            "Explain containerization".to_string(),
            "What is Kubernetes?".to_string(),
            "Define API".to_string(),
            "What is REST?".to_string(),
            "Explain GraphQL".to_string(),
            "What is a database?".to_string(),
            "Define SQL".to_string(),
            "What is NoSQL?".to_string(),
            "Explain version control".to_string(),
            "What is Git?".to_string(),
            "Define continuous integration".to_string(),
            "What is test-driven development?".to_string(),
            "Explain agile methodology".to_string(),
            "What is software architecture?".to_string(),
            "Define design patterns".to_string(),
            "What is functional programming?".to_string(),
            "Explain object-oriented programming".to_string(),
            "What is a compiler?".to_string(),
            "Define algorithm".to_string(),
            "What is data structure?".to_string(),
            "Explain time complexity".to_string(),
            "What is space complexity?".to_string(),
            "Define recursion".to_string(),
            "What is sorting?".to_string(),
            "Explain searching algorithms".to_string(),
            "What is a hash table?".to_string(),
            "Define binary tree".to_string(),
            "What is a graph?".to_string(),
            "Explain dynamic programming".to_string(),
            "What is greedy algorithm?".to_string(),
            "Define divide and conquer".to_string(),
            "What is backtracking?".to_string(),
            "Explain memoization".to_string(),
        ]
    }

    /// Get queries based on log patterns (placeholder implementation)
    async fn get_log_based_queries(&self) -> CacheResult<Vec<String>> {
        // This would analyze actual query logs in a real implementation
        // For now, return enhanced predefined queries
        Ok(self.get_predefined_queries())
    }

    /// Generate synthetic variations of common queries
    fn get_synthetic_variations(&self) -> Vec<String> {
        let base_queries = vec![
            "What is",
            "Explain",
            "Define",
            "How does",
            "Why is",
            "When should",
            "Where is",
            "Who invented",
        ];

        let topics = vec![
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks",
            "blockchain",
            "cloud computing",
            "quantum computing",
            "data science",
            "software engineering",
            "cybersecurity",
        ];

        let mut queries = Vec::new();
        for base in &base_queries {
            for topic in &topics {
                queries.push(format!("{base} {topic}?"));
                if queries.len() >= self.config.max_queries {
                    break;
                }
            }
            if queries.len() >= self.config.max_queries {
                break;
            }
        }

        queries
    }

    /// Get frequently accessed queries (placeholder implementation)
    async fn get_frequency_based_queries(&self) -> CacheResult<Vec<String>> {
        // This would analyze actual usage patterns in a real implementation
        // For now, return predefined queries with frequency weighting
        let mut queries = self.get_predefined_queries();

        // Simulate frequency-based ordering (most common first)
        queries.truncate(self.config.max_queries.min(20));
        Ok(queries)
    }

    /// Warm a single query
    async fn warm_single_query<T>(
        &self,
        client: &CachedLLMClient<T>,
        query: &str,
    ) -> CacheResult<bool>
    where
        T: LanguageModel + Send + Sync,
    {
        let params = self.config.warming_params.as_ref();

        // Check if already cached
        let was_cached = client.is_cached(query, params).await;

        if !was_cached {
            // Execute the query to warm the cache
            match params {
                Some(p) => {
                    client.complete_with_params(query, p.clone()).map_err(|e| {
                        CacheError::WarmingFailed(format!("Query execution failed: {e}"))
                    })?;
                }
                None => {
                    client.complete(query).map_err(|e| {
                        CacheError::WarmingFailed(format!("Query execution failed: {e}"))
                    })?;
                }
            }
        }

        Ok(was_cached)
    }

    /// Truncate query for display
    fn truncate_query(query: &str, max_len: usize) -> String {
        if query.len() <= max_len {
            query.to_string()
        } else {
            format!("{}...", &query[..max_len.saturating_sub(3)])
        }
    }
}

/// Results from cache warming operation
#[derive(Debug, Clone)]
pub struct WarmingResults {
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_time: Duration,
    pub cache_hit_rate: f64,
    pub avg_time_per_query: Duration,
}

impl WarmingResults {
    pub fn new() -> Self {
        Self {
            successful_queries: 0,
            failed_queries: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_time: Duration::ZERO,
            cache_hit_rate: 0.0,
            avg_time_per_query: Duration::ZERO,
        }
    }

    pub fn calculate_statistics(&mut self) {
        let total_queries = self.successful_queries;

        if total_queries > 0 {
            self.cache_hit_rate = self.cache_hits as f64 / total_queries as f64;
            self.avg_time_per_query = self.total_time / total_queries as u32;
        }
    }

    pub fn print(&self) {
        println!("Cache Warming Results:");
        println!("  Successful queries: {}", self.successful_queries);
        println!("  Failed queries: {}", self.failed_queries);
        println!(
            "  Cache hits: {} ({:.1}%)",
            self.cache_hits,
            self.cache_hit_rate * 100.0
        );
        println!("  Cache misses: {}", self.cache_misses);
        println!("  Total time: {:.2}s", self.total_time.as_secs_f64());
        println!(
            "  Avg time per query: {:.2}ms",
            self.avg_time_per_query.as_secs_f64() * 1000.0
        );
    }
}

impl Default for WarmingResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warming_config() {
        let config = WarmingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.strategy, WarmingStrategy::PredefinedQueries);
        assert_eq!(config.max_queries, 50);
    }

    #[test]
    fn test_warming_config_builder() {
        let config = WarmingConfig::builder()
            .strategy(WarmingStrategy::Custom)
            .max_queries(10)
            .custom_queries(vec!["test query".to_string()])
            .build_validated()
            .unwrap();

        assert_eq!(config.strategy, WarmingStrategy::Custom);
        assert_eq!(config.max_queries, 10);
        assert_eq!(config.custom_queries.len(), 1);
    }

    #[test]
    fn test_warming_config_validation() {
        let config = WarmingConfig { max_queries: 0, ..Default::default() };
        assert!(config.validate().is_err());

        let config = WarmingConfig {
            strategy: WarmingStrategy::Custom,
            custom_queries: Vec::new(),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_cache_warmer_creation() {
        let config = WarmingConfig::default();
        let warmer = CacheWarmer::new(config);
        assert!(warmer.is_ok());
    }

    #[tokio::test]
    async fn test_predefined_queries() {
        let config = WarmingConfig::builder()
            .strategy(WarmingStrategy::PredefinedQueries)
            .build();

        let warmer = CacheWarmer::new(config).unwrap();
        let queries = warmer.generate_warming_queries().await.unwrap();

        assert!(!queries.is_empty());
        assert!(queries.len() >= 10);
        assert!(queries
            .iter()
            .any(|q| q.contains("artificial intelligence")));
    }

    #[tokio::test]
    async fn test_synthetic_variations() {
        let config = WarmingConfig::builder()
            .strategy(WarmingStrategy::SyntheticVariations)
            .max_queries(20)
            .build();

        let warmer = CacheWarmer::new(config).unwrap();
        let queries = warmer.generate_warming_queries().await.unwrap();

        assert!(!queries.is_empty());
        assert!(queries.len() <= 20);
    }

    #[test]
    fn test_warming_results() {
        let mut results = WarmingResults::new();
        results.successful_queries = 10;
        results.cache_hits = 3;
        results.total_time = Duration::from_secs(5);

        results.calculate_statistics();

        assert_eq!(results.cache_hit_rate, 0.3);
        assert_eq!(results.avg_time_per_query, Duration::from_millis(500));
    }
}
