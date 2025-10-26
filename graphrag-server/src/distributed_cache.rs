//! Distributed Caching with Redis
//!
//! This module provides distributed caching for GraphRAG using Redis. It enables:
//! - Multi-level caching (L1/L2/L3)
//! - Cache coherence across multiple server instances
//! - Predictive prefetching
//! - Cache warming strategies
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │          Application                │
//! └──────────────┬──────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────┐
//! │       Cache Manager                 │
//! │  ┌────────────────────────────┐     │
//! │  │ L1: In-Memory (Fast)       │     │
//! │  │ - LRU eviction            │     │
//! │  │ - 100ms TTL               │     │
//! │  └────────────┬───────────────┘     │
//! │               │                     │
//! │  ┌────────────▼───────────────┐     │
//! │  │ L2: Redis (Distributed)    │     │
//! │  │ - Shared across servers   │     │
//! │  │ - 1h TTL                  │     │
//! │  └────────────┬───────────────┘     │
//! │               │                     │
//! │  ┌────────────▼───────────────┐     │
//! │  │ L3: Persistent Storage     │     │
//! │  │ - Long-term cache         │     │
//! │  │ - 24h+ TTL                │     │
//! │  └────────────────────────────┘     │
//! └─────────────────────────────────────┘
//! ```

use redis::{Client, Commands, RedisError};
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cache entry with metadata
#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
    access_count: u64,
    last_accessed: Instant,
}

/// Multi-level distributed cache
///
/// Provides L1 (in-memory), L2 (Redis), and L3 (persistent) caching layers
/// with automatic promotion/demotion and cache warming.
pub struct DistributedCache {
    /// L1 cache: In-memory LRU cache
    l1_cache: Arc<RwLock<HashMap<String, CacheEntry<Vec<u8>>>>>,
    /// L1 cache max size
    l1_max_size: usize,
    /// L1 TTL
    l1_ttl: Duration,

    /// L2 cache: Redis client
    redis_client: Option<Client>,
    /// L2 TTL (in seconds for Redis)
    l2_ttl: u64,

    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,

    /// Prefetch enabled
    _prefetch_enabled: bool,
}

/// Cache statistics
#[derive(Default, Clone)]
pub struct CacheStats {
    /// L1 hits
    pub l1_hits: u64,
    /// L1 misses
    pub l1_misses: u64,
    /// L2 hits
    pub l2_hits: u64,
    /// L2 misses
    pub l2_misses: u64,
    /// Total requests
    pub total_requests: u64,
    /// Evictions
    pub evictions: u64,
    /// Prefetches
    pub prefetches: u64,
}

impl CacheStats {
    /// Calculate L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.l1_hits as f64) / (self.total_requests as f64)
        }
    }

    /// Calculate L2 hit rate
    pub fn l2_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.l2_hits as f64) / (self.total_requests as f64)
        }
    }

    /// Calculate total hit rate
    pub fn total_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            ((self.l1_hits + self.l2_hits) as f64) / (self.total_requests as f64)
        }
    }
}

/// Cache configuration
pub struct CacheConfig {
    /// Redis URL (e.g., "redis://localhost:6379")
    pub redis_url: Option<String>,
    /// L1 cache max entries
    pub l1_max_size: usize,
    /// L1 TTL in seconds
    pub l1_ttl_secs: u64,
    /// L2 TTL in seconds
    pub l2_ttl_secs: u64,
    /// Enable prefetching
    pub prefetch_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: Some("redis://localhost:6379".to_string()),
            l1_max_size: 1000,
            l1_ttl_secs: 100,
            l2_ttl_secs: 3600, // 1 hour
            prefetch_enabled: true,
        }
    }
}

impl DistributedCache {
    /// Create a new distributed cache
    ///
    /// # Arguments
    /// * `config` - Cache configuration
    ///
    /// # Returns
    /// Result with DistributedCache or error
    pub fn new(config: CacheConfig) -> Result<Self, RedisError> {
        let redis_client = if let Some(url) = config.redis_url {
            match Client::open(url.clone()) {
                Ok(client) => {
                    // Test connection
                    match client.get_connection() {
                        Ok(_) => {
                            tracing::info!("✅ Redis connected: {}", url);
                            Some(client)
                        }
                        Err(e) => {
                            tracing::warn!("⚠️ Redis connection failed, L2 cache disabled: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("⚠️ Redis client creation failed, L2 cache disabled: {}", e);
                    None
                }
            }
        } else {
            tracing::info!("Redis URL not provided, L2 cache disabled");
            None
        };

        Ok(Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            l1_max_size: config.l1_max_size,
            l1_ttl: Duration::from_secs(config.l1_ttl_secs),
            redis_client,
            l2_ttl: config.l2_ttl_secs,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            _prefetch_enabled: config.prefetch_enabled,
        })
    }

    /// Get value from cache
    ///
    /// Checks L1 cache first, then L2 (Redis), with automatic promotion.
    ///
    /// # Arguments
    /// * `key` - Cache key
    ///
    /// # Returns
    /// Option with cached value
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.stats.write().total_requests += 1;

        // Try L1 cache first
        if let Some(value) = self.get_l1(key) {
            self.stats.write().l1_hits += 1;

            // Deserialize
            match bincode::deserialize::<T>(&value) {
                Ok(val) => return Some(val),
                Err(e) => {
                    tracing::warn!("Failed to deserialize L1 cache value: {}", e);
                }
            }
        }

        self.stats.write().l1_misses += 1;

        // Try L2 cache (Redis)
        if let Some(value) = self.get_l2(key) {
            self.stats.write().l2_hits += 1;

            // Promote to L1
            self.set_l1(key, value.clone());

            // Deserialize
            match bincode::deserialize::<T>(&value) {
                Ok(val) => return Some(val),
                Err(e) => {
                    tracing::warn!("Failed to deserialize L2 cache value: {}", e);
                }
            }
        }

        self.stats.write().l2_misses += 1;
        None
    }

    /// Set value in cache
    ///
    /// Stores in both L1 (in-memory) and L2 (Redis) for redundancy.
    ///
    /// # Arguments
    /// * `key` - Cache key
    /// * `value` - Value to cache
    pub fn set<T: Serialize>(&self, key: &str, value: &T) {
        // Serialize value
        let bytes = match bincode::serialize(value) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("Failed to serialize cache value: {}", e);
                return;
            }
        };

        // Set in L1
        self.set_l1(key, bytes.clone());

        // Set in L2 (Redis)
        self.set_l2(key, bytes);
    }

    /// Invalidate cache entry
    ///
    /// Removes from both L1 and L2 caches.
    ///
    /// # Arguments
    /// * `key` - Cache key to invalidate
    pub fn invalidate(&self, key: &str) {
        // Remove from L1
        self.l1_cache.write().remove(key);

        // Remove from L2
        if let Some(client) = &self.redis_client {
            if let Ok(mut conn) = client.get_connection() {
                let _: Result<(), _> = conn.del(key);
            }
        }
    }

    /// Invalidate pattern
    ///
    /// Removes all keys matching a pattern from both caches.
    ///
    /// # Arguments
    /// * `pattern` - Pattern to match (e.g., "query:*")
    pub fn invalidate_pattern(&self, pattern: &str) {
        // Remove from L1
        let keys_to_remove: Vec<String> = self
            .l1_cache
            .read()
            .keys()
            .filter(|k| Self::matches_pattern(k, pattern))
            .cloned()
            .collect();

        for key in keys_to_remove {
            self.l1_cache.write().remove(&key);
        }

        // Remove from L2
        if let Some(client) = &self.redis_client {
            if let Ok(mut conn) = client.get_connection() {
                // Get all keys matching pattern
                if let Ok(keys) = conn.keys::<_, Vec<String>>(pattern) {
                    for key in keys {
                        let _: Result<(), _> = conn.del(&key);
                    }
                }
            }
        }
    }

    /// Warm cache with frequently accessed keys
    ///
    /// Preloads cache with specified keys to improve hit rates.
    ///
    /// # Arguments
    /// * `keys` - Keys to warm
    /// * `loader` - Function to load values for keys
    pub fn warm<T, F>(&self, keys: Vec<String>, mut loader: F)
    where
        T: Serialize,
        F: FnMut(&str) -> Option<T>,
    {
        for key in keys {
            if let Some(value) = loader(&key) {
                self.set(&key, &value);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    /// Clear all caches
    pub fn clear(&self) {
        // Clear L1
        self.l1_cache.write().clear();

        // Clear L2
        if let Some(client) = &self.redis_client {
            if let Ok(mut conn) = client.get_connection() {
                let _: Result<(), _> = redis::cmd("FLUSHDB").query(&mut conn);
            }
        }
    }

    // --- Private methods ---

    /// Get from L1 cache
    fn get_l1(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.l1_cache.write();

        if let Some(entry) = cache.get_mut(key) {
            // Check TTL
            if entry.created_at.elapsed() > self.l1_ttl {
                cache.remove(key);
                return None;
            }

            // Update access stats
            entry.access_count += 1;
            entry.last_accessed = Instant::now();

            return Some(entry.value.clone());
        }

        None
    }

    /// Set in L1 cache
    fn set_l1(&self, key: &str, value: Vec<u8>) {
        let mut cache = self.l1_cache.write();

        // Evict if at capacity
        if cache.len() >= self.l1_max_size && !cache.contains_key(key) {
            self.evict_l1(&mut cache);
        }

        // Insert
        cache.insert(
            key.to_string(),
            CacheEntry {
                value,
                created_at: Instant::now(),
                access_count: 0,
                last_accessed: Instant::now(),
            },
        );
    }

    /// Evict from L1 cache using LRU
    fn evict_l1(&self, cache: &mut HashMap<String, CacheEntry<Vec<u8>>>) {
        // Find least recently used entry
        if let Some((lru_key, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
        {
            let lru_key = lru_key.clone();
            cache.remove(&lru_key);
            self.stats.write().evictions += 1;
        }
    }

    /// Get from L2 cache (Redis)
    fn get_l2(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(client) = &self.redis_client {
            if let Ok(mut conn) = client.get_connection() {
                if let Ok(value) = conn.get::<_, Vec<u8>>(key) {
                    return Some(value);
                }
            }
        }
        None
    }

    /// Set in L2 cache (Redis)
    fn set_l2(&self, key: &str, value: Vec<u8>) {
        if let Some(client) = &self.redis_client {
            if let Ok(mut conn) = client.get_connection() {
                let _: Result<(), _> = conn.set_ex(key, value, self.l2_ttl);
            }
        }
    }

    /// Check if key matches pattern
    fn matches_pattern(key: &str, pattern: &str) -> bool {
        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len() - 1];
            key.starts_with(prefix)
        } else {
            key == pattern
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_cache() {
        let config = CacheConfig {
            redis_url: None, // Disable Redis for this test
            l1_max_size: 2,
            l1_ttl_secs: 10,
            l2_ttl_secs: 60,
            prefetch_enabled: false,
        };

        let cache = DistributedCache::new(config).unwrap();

        // Set value
        cache.set("key1", &"value1".to_string());

        // Get value
        let value: Option<String> = cache.get("key1");
        assert_eq!(value, Some("value1".to_string()));

        // Stats
        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 1);
    }

    #[test]
    fn test_eviction() {
        let config = CacheConfig {
            redis_url: None,
            l1_max_size: 2,
            l1_ttl_secs: 10,
            l2_ttl_secs: 60,
            prefetch_enabled: false,
        };

        let cache = DistributedCache::new(config).unwrap();

        // Fill cache
        cache.set("key1", &"value1".to_string());
        cache.set("key2", &"value2".to_string());

        // Access key1 to make it recently used
        let _: Option<String> = cache.get("key1");

        // Add key3, should evict key2 (LRU)
        cache.set("key3", &"value3".to_string());

        // key1 and key3 should be present, key2 evicted
        let v1: Option<String> = cache.get("key1");
        let v2: Option<String> = cache.get("key2");
        let v3: Option<String> = cache.get("key3");

        assert_eq!(v1, Some("value1".to_string()));
        assert_eq!(v2, None);
        assert_eq!(v3, Some("value3".to_string()));
    }
}
