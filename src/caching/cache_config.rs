//! Cache configuration and policy management

use super::{CacheError, CacheResult};
use std::time::Duration;

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict items that haven't been accessed recently
    LRU,
    /// Least Frequently Used - evict items with the lowest access count
    LFU,
    /// Time To Live - evict items after a fixed time period
    TTL,
    /// Hybrid approach combining LRU and access frequency
    Adaptive,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

/// Cache configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub max_capacity: u64,

    /// Time-to-live for cache entries in seconds
    pub ttl_seconds: u64,

    /// Eviction policy to use
    pub eviction_policy: EvictionPolicy,

    /// Whether to enable detailed statistics collection
    pub enable_statistics: bool,

    /// Whether to enable cache warming
    pub enable_warming: bool,

    /// Initial capacity hint for performance
    pub initial_capacity: Option<u64>,

    /// Cleanup interval for expired entries (seconds)
    pub cleanup_interval_seconds: u64,

    /// Maximum size of individual cache entries (bytes)
    pub max_entry_size: usize,

    /// Whether to compress large cache entries
    pub enable_compression: bool,

    /// Compression threshold (bytes)
    pub compression_threshold: usize,

    /// Cache persistence options
    pub persistence: PersistenceConfig,
}

/// Cache persistence configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistenceConfig {
    /// Whether to enable cache persistence
    pub enabled: bool,

    /// Directory to store cache files
    pub directory: Option<String>,

    /// How often to save cache to disk (seconds)
    pub save_interval_seconds: u64,

    /// Whether to load cache from disk on startup
    pub load_on_startup: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            directory: None,
            save_interval_seconds: 300, // 5 minutes
            load_on_startup: false,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 10_000,
            ttl_seconds: 3600, // 1 hour
            eviction_policy: EvictionPolicy::LRU,
            enable_statistics: true,
            enable_warming: false,
            initial_capacity: Some(1_000),
            cleanup_interval_seconds: 300, // 5 minutes
            max_entry_size: 1024 * 1024, // 1MB
            enable_compression: false,
            compression_threshold: 1024 * 10, // 10KB
            persistence: PersistenceConfig::default(),
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration builder
    pub fn builder() -> CacheConfigBuilder {
        CacheConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> CacheResult<()> {
        if self.max_capacity == 0 {
            return Err(CacheError::Configuration(
                "max_capacity must be greater than 0".to_string()
            ));
        }

        if self.ttl_seconds == 0 {
            return Err(CacheError::Configuration(
                "ttl_seconds must be greater than 0".to_string()
            ));
        }

        if self.cleanup_interval_seconds == 0 {
            return Err(CacheError::Configuration(
                "cleanup_interval_seconds must be greater than 0".to_string()
            ));
        }

        if self.max_entry_size == 0 {
            return Err(CacheError::Configuration(
                "max_entry_size must be greater than 0".to_string()
            ));
        }

        if self.enable_compression && self.compression_threshold == 0 {
            return Err(CacheError::Configuration(
                "compression_threshold must be greater than 0 when compression is enabled".to_string()
            ));
        }

        if self.persistence.enabled && self.persistence.directory.is_none() {
            return Err(CacheError::Configuration(
                "persistence directory must be specified when persistence is enabled".to_string()
            ));
        }

        Ok(())
    }

    /// Get TTL as Duration
    pub fn ttl_duration(&self) -> Duration {
        Duration::from_secs(self.ttl_seconds)
    }

    /// Get cleanup interval as Duration
    pub fn cleanup_interval(&self) -> Duration {
        Duration::from_secs(self.cleanup_interval_seconds)
    }

    /// Create a production-ready configuration
    pub fn production() -> Self {
        Self {
            max_capacity: 100_000,
            ttl_seconds: 7200, // 2 hours
            eviction_policy: EvictionPolicy::Adaptive,
            enable_statistics: true,
            enable_warming: true,
            initial_capacity: Some(10_000),
            cleanup_interval_seconds: 600, // 10 minutes
            max_entry_size: 5 * 1024 * 1024, // 5MB
            enable_compression: true,
            compression_threshold: 50 * 1024, // 50KB
            persistence: PersistenceConfig {
                enabled: true,
                directory: Some("./cache".to_string()),
                save_interval_seconds: 1800, // 30 minutes
                load_on_startup: true,
            },
        }
    }

    /// Create a development configuration with smaller limits
    pub fn development() -> Self {
        Self {
            max_capacity: 1_000,
            ttl_seconds: 1800, // 30 minutes
            eviction_policy: EvictionPolicy::LRU,
            enable_statistics: true,
            enable_warming: false,
            initial_capacity: Some(100),
            cleanup_interval_seconds: 60, // 1 minute
            max_entry_size: 1024 * 1024, // 1MB
            enable_compression: false,
            compression_threshold: 1024 * 10, // 10KB
            persistence: PersistenceConfig::default(),
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            max_capacity: 50_000,
            ttl_seconds: 14400, // 4 hours
            eviction_policy: EvictionPolicy::LFU,
            enable_statistics: false, // Disable for max performance
            enable_warming: true,
            initial_capacity: Some(25_000),
            cleanup_interval_seconds: 1800, // 30 minutes
            max_entry_size: 10 * 1024 * 1024, // 10MB
            enable_compression: true,
            compression_threshold: 100 * 1024, // 100KB
            persistence: PersistenceConfig::default(),
        }
    }
}

/// Builder for cache configuration
pub struct CacheConfigBuilder {
    config: CacheConfig,
}

impl CacheConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }

    pub fn max_capacity(mut self, capacity: u64) -> Self {
        self.config.max_capacity = capacity;
        self
    }

    pub fn ttl_seconds(mut self, seconds: u64) -> Self {
        self.config.ttl_seconds = seconds;
        self
    }

    pub fn ttl_duration(mut self, duration: Duration) -> Self {
        self.config.ttl_seconds = duration.as_secs();
        self
    }

    pub fn eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.config.eviction_policy = policy;
        self
    }

    pub fn enable_statistics(mut self, enabled: bool) -> Self {
        self.config.enable_statistics = enabled;
        self
    }

    pub fn enable_warming(mut self, enabled: bool) -> Self {
        self.config.enable_warming = enabled;
        self
    }

    pub fn initial_capacity(mut self, capacity: u64) -> Self {
        self.config.initial_capacity = Some(capacity);
        self
    }

    pub fn cleanup_interval_seconds(mut self, seconds: u64) -> Self {
        self.config.cleanup_interval_seconds = seconds;
        self
    }

    pub fn max_entry_size(mut self, size: usize) -> Self {
        self.config.max_entry_size = size;
        self
    }

    pub fn enable_compression(mut self, enabled: bool) -> Self {
        self.config.enable_compression = enabled;
        self
    }

    pub fn compression_threshold(mut self, threshold: usize) -> Self {
        self.config.compression_threshold = threshold;
        self
    }

    pub fn persistence(mut self, config: PersistenceConfig) -> Self {
        self.config.persistence = config;
        self
    }

    pub fn build(self) -> CacheConfig {
        self.config
    }

    pub fn build_validated(self) -> CacheResult<CacheConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for CacheConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default();
        assert_eq!(config.max_capacity, 10_000);
        assert_eq!(config.ttl_seconds, 3600);
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
        assert!(config.enable_statistics);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = CacheConfig::builder()
            .max_capacity(5000)
            .ttl_seconds(1800)
            .eviction_policy(EvictionPolicy::LFU)
            .enable_statistics(false)
            .build();

        assert_eq!(config.max_capacity, 5000);
        assert_eq!(config.ttl_seconds, 1800);
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
        assert!(!config.enable_statistics);
    }

    #[test]
    fn test_config_validation() {
        let config = CacheConfig { max_capacity: 0, ..Default::default() };
        assert!(config.validate().is_err());

        let config = CacheConfig { ttl_seconds: 0, ..Default::default() };
        assert!(config.validate().is_err());

        let config = CacheConfig {
            enable_compression: true,
            compression_threshold: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_preset_configs() {
        let prod_config = CacheConfig::production();
        assert!(prod_config.validate().is_ok());
        assert_eq!(prod_config.max_capacity, 100_000);
        assert!(prod_config.enable_warming);

        let dev_config = CacheConfig::development();
        assert!(dev_config.validate().is_ok());
        assert_eq!(dev_config.max_capacity, 1_000);
        assert!(!dev_config.enable_warming);

        let perf_config = CacheConfig::high_performance();
        assert!(perf_config.validate().is_ok());
        assert_eq!(perf_config.eviction_policy, EvictionPolicy::LFU);
        assert!(!perf_config.enable_statistics);
    }
}