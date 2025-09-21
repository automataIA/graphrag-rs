//! Embedding cache for neural transformers

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cache entry for embeddings
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub embedding: Vec<f32>,
    pub timestamp: u64,
    pub access_count: usize,
}

impl CacheEntry {
    pub fn new(embedding: Vec<f32>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            embedding,
            timestamp,
            access_count: 1,
        }
    }

    pub fn access(&mut self) -> &Vec<f32> {
        self.access_count += 1;
        &self.embedding
    }

    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.timestamp)
    }
}

/// LRU Cache for embeddings with configurable size and TTL
pub struct EmbeddingCache {
    cache: HashMap<String, CacheEntry>,
    access_order: Vec<String>,
    max_size: usize,
    ttl_seconds: Option<u64>,
    hits: usize,
    misses: usize,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
            ttl_seconds: None,
            hits: 0,
            misses: 0,
        }
    }

    pub fn with_ttl(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
            ttl_seconds: Some(ttl_seconds),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&self, key: &str) -> Option<CacheEntry> {
        // Note: This is a read-only get that doesn't update access stats
        // Use get_and_update for write operations
        if let Some(ttl) = self.ttl_seconds {
            if let Some(entry) = self.cache.get(key) {
                if entry.age_seconds() < ttl {
                    return Some(entry.clone());
                } else {
                    return None;
                }
            }
        }

        self.cache.get(key).cloned()
    }

    pub fn get_and_update(&mut self, key: &str) -> Option<CacheEntry> {
        // Clean expired entries if TTL is set
        if self.ttl_seconds.is_some() {
            self.clean_expired();
        }

        if let Some(mut entry) = self.cache.get(key).cloned() {
            // Update access order (move to end)
            self.update_access_order(key);

            // Update access count
            entry.access_count += 1;
            self.cache.insert(key.to_string(), entry.clone());

            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: String, embedding: Vec<f32>) {
        let entry = CacheEntry::new(embedding);

        // Remove if already exists (for LRU update)
        if self.cache.contains_key(&key) {
            self.remove_from_access_order(&key);
        }

        // Check if we need to evict
        while self.cache.len() >= self.max_size && !self.access_order.is_empty() {
            self.evict_lru();
        }

        // Insert new entry
        self.cache.insert(key.clone(), entry);
        self.access_order.push(key);
    }

    pub fn contains_key(&self, key: &str) -> bool {
        if let Some(ttl) = self.ttl_seconds {
            if let Some(entry) = self.cache.get(key) {
                return entry.age_seconds() < ttl;
            }
        }
        self.cache.contains_key(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        self.remove_from_access_order(key);
        self.cache.remove(key)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.max_size
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }

    fn update_access_order(&mut self, key: &str) {
        self.remove_from_access_order(key);
        self.access_order.push(key.to_string());
    }

    fn remove_from_access_order(&mut self, key: &str) {
        if let Some(pos) = self.access_order.iter().position(|x| x == key) {
            self.access_order.remove(pos);
        }
    }

    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.access_order.first().cloned() {
            self.cache.remove(&lru_key);
            self.access_order.remove(0);
        }
    }

    fn clean_expired(&mut self) {
        if let Some(ttl) = self.ttl_seconds {
            let mut expired_keys = Vec::new();

            for (key, entry) in &self.cache {
                if entry.age_seconds() >= ttl {
                    expired_keys.push(key.clone());
                }
            }

            for key in expired_keys {
                self.remove(&key);
            }
        }
    }

    /// Get cache statistics for monitoring
    pub fn get_detailed_stats(&self) -> CacheStats {
        let total_requests = self.hits + self.misses;
        let hit_rate = if total_requests > 0 {
            self.hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let total_embeddings: usize = self.cache.values().map(|entry| entry.embedding.len()).sum();
        let avg_embedding_size = if self.cache.is_empty() {
            0.0
        } else {
            total_embeddings as f64 / self.cache.len() as f64
        };

        CacheStats {
            size: self.cache.len(),
            capacity: self.max_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate,
            total_requests,
            avg_embedding_size,
            ttl_seconds: self.ttl_seconds,
        }
    }

    /// Optimize cache by removing least frequently used items
    pub fn optimize(&mut self) {
        if self.cache.len() < self.max_size {
            return;
        }

        // Collect keys to remove
        let mut items: Vec<_> = self.cache.iter().collect();
        items.sort_by(|a, b| a.1.access_count.cmp(&b.1.access_count));

        // Remove bottom 25% least frequently used
        let remove_count = self.cache.len() / 4;
        let keys_to_remove: Vec<String> = items.iter()
            .take(remove_count)
            .map(|(key, _)| key.to_string())
            .collect();

        for key in keys_to_remove {
            self.remove(&key);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
    pub total_requests: usize,
    pub avg_embedding_size: f64,
    pub ttl_seconds: Option<u64>,
}

impl CacheStats {
    pub fn print(&self) {
        println!("📊 Embedding Cache Statistics:");
        println!("   Size: {}/{}", self.size, self.capacity);
        println!("   Hit rate: {:.2}% ({}/{})", self.hit_rate * 100.0, self.hits, self.total_requests);
        println!("   Avg embedding size: {:.1} dims", self.avg_embedding_size);
        if let Some(ttl) = self.ttl_seconds {
            println!("   TTL: {ttl} seconds");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = EmbeddingCache::new(3);

        // Test insertion and retrieval
        let embedding1 = vec![1.0, 2.0, 3.0];
        cache.insert("key1".to_string(), embedding1.clone());

        let retrieved = cache.get("key1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().embedding, embedding1);

        // Test cache miss
        let missing = cache.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EmbeddingCache::new(2); // Small capacity

        // Fill cache
        cache.insert("key1".to_string(), vec![1.0]);
        cache.insert("key2".to_string(), vec![2.0]);

        // Both should be present
        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_some());

        // Insert third item, should evict least recently used
        cache.insert("key3".to_string(), vec![3.0]);

        // key1 should be evicted (was accessed first)
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_access_count_tracking() {
        let mut cache = EmbeddingCache::new(5);

        cache.insert("key1".to_string(), vec![1.0]);

        // Access multiple times
        let entry1 = cache.get("key1").unwrap();
        assert_eq!(entry1.access_count, 2); // 1 from insert + 1 from get

        let entry2 = cache.get("key1").unwrap();
        assert_eq!(entry2.access_count, 3); // Previous + 1
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = EmbeddingCache::new(10);

        // Generate some hits and misses
        cache.insert("key1".to_string(), vec![1.0, 2.0]);
        cache.get("key1"); // hit
        cache.get("key2"); // miss
        cache.get("key1"); // hit

        let (hits, misses) = cache.stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
        assert!((cache.hit_rate() - 2.0/3.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_with_ttl() {
        let mut cache = EmbeddingCache::with_ttl(5, 1); // 1 second TTL

        cache.insert("key1".to_string(), vec![1.0]);

        // Should be available immediately
        assert!(cache.contains_key("key1"));

        // Simulate passage of time by manually setting old timestamp
        if let Some(entry) = cache.cache.get_mut("key1") {
            entry.timestamp = 0; // Very old timestamp
        }

        // Should now be expired
        assert!(!cache.contains_key("key1"));
    }

    #[test]
    fn test_cache_optimization() {
        let mut cache = EmbeddingCache::new(4);

        // Insert items with different access patterns
        cache.insert("frequent".to_string(), vec![1.0]);
        cache.insert("rare1".to_string(), vec![2.0]);
        cache.insert("rare2".to_string(), vec![3.0]);
        cache.insert("medium".to_string(), vec![4.0]);

        // Access "frequent" multiple times
        for _ in 0..5 {
            cache.get("frequent");
        }

        // Access "medium" a few times
        for _ in 0..2 {
            cache.get("medium");
        }

        // rare1 and rare2 are accessed only once (from insert)

        let initial_size = cache.len();
        cache.optimize();

        // Should have removed some least frequently used items
        assert!(cache.len() < initial_size);

        // "frequent" should still be there
        assert!(cache.contains_key("frequent"));
    }

    #[test]
    fn test_detailed_stats() {
        let mut cache = EmbeddingCache::new(10);

        cache.insert("key1".to_string(), vec![1.0, 2.0, 3.0]); // 3 dims
        cache.insert("key2".to_string(), vec![4.0, 5.0]); // 2 dims

        cache.get("key1"); // hit
        cache.get("missing"); // miss

        let stats = cache.get_detailed_stats();

        assert_eq!(stats.size, 2);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_requests, 2);
        assert!((stats.avg_embedding_size - 2.5).abs() < 0.001); // (3+2)/2 = 2.5
    }
}