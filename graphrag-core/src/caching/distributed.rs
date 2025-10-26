//! Distributed caching with Redis
//!
//! Implements multi-level caching with Redis for horizontal scaling:
//! - L1: In-memory LRU cache (fastest)
//! - L2: Redis cache (distributed)
//! - L3: Persistent storage (fallback)

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use parking_lot::RwLock;

#[cfg(feature = "redis_storage")]
use redis::{Commands, Connection};

use crate::core::GraphRAGError;

type Result<T> = std::result::Result<T, GraphRAGError>;

/// Cache entry with TTL and access tracking
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached value
    pub value: T,
    /// Timestamp when this entry was created
    pub created_at: Instant,
    /// Timestamp when this entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed
    pub access_count: u64,
    /// Optional time-to-live for automatic expiration
    pub ttl: Option<Duration>,
}

impl<T: Clone> CacheEntry<T> {
    /// Create a new cache entry with the given value and optional TTL
    pub fn new(value: T, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }

    /// Check if this entry has expired based on its TTL
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Access the entry, updating access time and count, and return a clone of the value
    pub fn access(&mut self) -> T {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        self.value.clone()
    }
}

/// L1 Cache: In-memory LRU cache
pub struct L1Cache<K, V> {
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    max_size: usize,
    default_ttl: Option<Duration>,
}

impl<K, V> L1Cache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create a new L1 (in-memory) cache with the given maximum size and default TTL
    pub fn new(max_size: usize, default_ttl: Option<Duration>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::with_capacity(max_size))),
            max_size,
            default_ttl,
        }
    }

    /// Get a value from the cache, returning None if not found or expired
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write();
        if let Some(entry) = cache.get_mut(key) {
            if entry.is_expired() {
                cache.remove(key);
                None
            } else {
                Some(entry.access())
            }
        } else {
            None
        }
    }

    /// Put a value into the cache, evicting the oldest entry if at capacity
    pub fn put(&self, key: K, value: V) {
        let mut cache = self.cache.write();

        // Evict oldest entries if at capacity
        if cache.len() >= self.max_size && !cache.contains_key(&key) {
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(key, CacheEntry::new(value, self.default_ttl));
    }

    /// Invalidate (remove) a specific entry from the cache
    pub fn invalidate(&self, key: &K) {
        self.cache.write().remove(key);
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Get the current number of entries in the cache
    pub fn size(&self) -> usize {
        self.cache.read().len()
    }

    /// Get cache statistics including size, capacity, and access count
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read();
        let total_accesses: u64 = cache.values().map(|e| e.access_count).sum();
        CacheStats {
            size: cache.len(),
            capacity: self.max_size,
            total_accesses,
        }
    }
}

/// L2 Cache: Redis distributed cache
#[cfg(feature = "redis_storage")]
pub struct L2Cache {
    client: redis::Client,
    key_prefix: String,
    default_ttl: Option<Duration>,
}

#[cfg(feature = "redis_storage")]
impl L2Cache {
    /// Create a new L2 (Redis) cache with the given connection URL, key prefix, and default TTL
    pub fn new(url: &str, key_prefix: String, default_ttl: Option<Duration>) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to connect to Redis: {}", e),
        })?;

        Ok(Self {
            client,
            key_prefix,
            default_ttl,
        })
    }

    /// Generate a prefixed key for Redis storage
    fn prefixed_key(&self, key: &str) -> String {
        format!("{}:{}", self.key_prefix, key)
    }

    /// Get a value from Redis cache by key
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut conn = self.get_connection()?;
        let prefixed = self.prefixed_key(key);

        conn.get(&prefixed).map_err(|e| GraphRAGError::Storage {
            message: format!("Redis GET failed: {}", e),
        })
    }

    /// Put a value into Redis cache with optional TTL
    pub fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let mut conn = self.get_connection()?;
        let prefixed = self.prefixed_key(key);

        if let Some(ttl) = self.default_ttl {
            conn.set_ex::<_, _, ()>(&prefixed, value, ttl.as_secs())
                .map_err(|e| GraphRAGError::Storage {
                    message: format!("Redis SETEX failed: {}", e),
                })?;
        } else {
            conn.set::<_, _, ()>(&prefixed, value).map_err(|e| GraphRAGError::Storage {
                message: format!("Redis SET failed: {}", e),
            })?;
        }

        Ok(())
    }

    /// Invalidate (remove) a specific entry from Redis cache
    pub fn invalidate(&self, key: &str) -> Result<()> {
        let mut conn = self.get_connection()?;
        let prefixed = self.prefixed_key(key);

        conn.del::<_, ()>(&prefixed).map_err(|e| GraphRAGError::Storage {
            message: format!("Redis DEL failed: {}", e),
        })?;

        Ok(())
    }

    /// Clear all entries with the configured key prefix from Redis cache
    pub fn clear(&self) -> Result<()> {
        let mut conn = self.get_connection()?;
        let pattern = format!("{}:*", self.key_prefix);

        // Get all keys matching pattern
        let keys: Vec<String> = conn.keys(&pattern).map_err(|e| GraphRAGError::Storage {
            message: format!("Redis KEYS failed: {}", e),
        })?;

        // Delete all keys
        if !keys.is_empty() {
            conn.del::<_, ()>(&keys).map_err(|e| GraphRAGError::Storage {
                message: format!("Redis DEL failed: {}", e),
            })?;
        }

        Ok(())
    }

    /// Get a connection to the Redis server
    fn get_connection(&self) -> Result<Connection> {
        self.client.get_connection().map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to get Redis connection: {}", e),
        })
    }
}

/// Multi-level cache combining L1 (memory) and L2 (Redis)
pub struct DistributedCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + ToString,
    V: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    l1: L1Cache<K, V>,
    #[cfg(feature = "redis_storage")]
    #[allow(dead_code)]
    l2: Option<L2Cache>,
    #[cfg(not(feature = "redis_storage"))]
    #[allow(dead_code)]
    l2: Option<()>,
    stats: Arc<RwLock<DistributedCacheStats>>,
}

impl<K, V> DistributedCache<K, V>
where
    K: Eq + std::hash::Hash + Clone + ToString,
    V: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Create a new distributed cache with L1 (memory) and optional L2 (Redis) tiers
    pub fn new(
        l1_size: usize,
        l1_ttl: Option<Duration>,
        #[cfg(feature = "redis_storage")] redis_url: Option<&str>,
        #[cfg(not(feature = "redis_storage"))] _redis_url: Option<&str>,
        _l2_ttl: Option<Duration>,
    ) -> Result<Self> {
        let l1 = L1Cache::new(l1_size, l1_ttl);

        #[cfg(feature = "redis_storage")]
        let l2 = if let Some(url) = redis_url {
            Some(L2Cache::new(url, "graphrag".to_string(), _l2_ttl)?)
        } else {
            None
        };

        #[cfg(not(feature = "redis_storage"))]
        let l2 = None;

        Ok(Self {
            l1,
            l2,
            stats: Arc::new(RwLock::new(DistributedCacheStats::default())),
        })
    }

    /// Get value from cache (checks L1 then L2)
    pub fn get(&self, key: &K) -> Option<V> {
        // Try L1 first
        if let Some(value) = self.l1.get(key) {
            self.stats.write().l1_hits += 1;
            return Some(value);
        }

        self.stats.write().l1_misses += 1;

        // Try L2 (Redis) if available
        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            if let Ok(Some(bytes)) = l2.get(&key.to_string()) {
                if let Ok(value) = bincode::deserialize::<V>(&bytes) {
                    self.stats.write().l2_hits += 1;

                    // Populate L1 cache
                    self.l1.put(key.clone(), value.clone());

                    return Some(value);
                }
            }
            self.stats.write().l2_misses += 1;
        }

        None
    }

    /// Put value into cache (writes to both L1 and L2)
    pub fn put(&self, key: K, value: V) -> Result<()> {
        // Write to L1
        self.l1.put(key.clone(), value.clone());

        // Write to L2 (Redis) if available
        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            let bytes = bincode::serialize(&value).map_err(|e| GraphRAGError::Storage {
                message: format!("Serialization failed: {}", e),
            })?;
            l2.put(&key.to_string(), &bytes)?;
        }

        Ok(())
    }

    /// Invalidate key from all cache levels
    pub fn invalidate(&self, key: &K) -> Result<()> {
        self.l1.invalidate(key);

        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            l2.invalidate(&key.to_string())?;
        }

        Ok(())
    }

    /// Clear all cache levels
    pub fn clear(&self) -> Result<()> {
        self.l1.clear();

        #[cfg(feature = "redis_storage")]
        if let Some(l2) = &self.l2 {
            l2.clear()?;
        }

        Ok(())
    }

    /// Get comprehensive cache statistics
    pub fn stats(&self) -> DistributedCacheStats {
        let mut stats = self.stats.read().clone();
        let l1_stats = self.l1.stats();
        stats.l1_size = l1_stats.size;
        stats.l1_capacity = l1_stats.capacity;
        stats
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Current number of entries in the cache
    pub size: usize,
    /// Maximum capacity of the cache
    pub capacity: usize,
    /// Total number of accesses across all entries
    pub total_accesses: u64,
}

/// Distributed cache statistics
#[derive(Debug, Clone, Default)]
pub struct DistributedCacheStats {
    /// Number of cache hits in L1 (in-memory) cache
    pub l1_hits: u64,
    /// Number of cache misses in L1 (in-memory) cache
    pub l1_misses: u64,
    /// Current size of L1 cache
    pub l1_size: usize,
    /// Maximum capacity of L1 cache
    pub l1_capacity: usize,
    /// Number of cache hits in L2 (Redis) cache
    pub l2_hits: u64,
    /// Number of cache misses in L2 (Redis) cache
    pub l2_misses: u64,
}

impl DistributedCacheStats {
    /// Calculate the overall cache hit rate across both L1 and L2
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total_requests = total_hits + self.l1_misses + self.l2_misses;
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }

    /// Calculate the L1 cache hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total == 0 {
            0.0
        } else {
            self.l1_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_cache() {
        let cache = L1Cache::new(3, Some(Duration::from_secs(60)));

        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");

        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.size(), 3);

        // Test eviction
        cache.put("key4", "value4");
        assert_eq!(cache.size(), 3);
    }

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new("value", Some(Duration::from_millis(10)));
        assert!(!entry.is_expired());

        std::thread::sleep(Duration::from_millis(15));
        assert!(entry.is_expired());
    }
}
