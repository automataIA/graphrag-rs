//! Persistent Caching Layer for GraphRAG
//!
//! Provides disk-based caching for:
//! - Query results
//! - Embeddings
//! - Entity extractions
//! - Graph structures

use crate::{Result, GraphRAGError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};

/// Persistent cache implementation with TTL and size limits
#[derive(Debug, Clone)]
pub struct PersistentCache {
    cache_dir: PathBuf,
    config: CacheConfig,
    metadata: CacheMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Enable compression for cached data
    pub enable_compression: bool,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1_073_741_824, // 1GB
            default_ttl: Duration::from_secs(86400 * 7), // 7 days
            enable_compression: true,
            eviction_policy: EvictionPolicy::LRU,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,    // Least Recently Used
    LFU,    // Least Frequently Used
    FIFO,   // First In First Out
    TTL,    // Time To Live based
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    total_size: usize,
    entry_count: usize,
    last_cleanup: SystemTime,
    hit_count: usize,
    miss_count: usize,
    entries: HashMap<String, EntryMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntryMetadata {
    key: String,
    size: usize,
    created_at: SystemTime,
    last_accessed: SystemTime,
    access_count: usize,
    ttl: Duration,
}

impl PersistentCache {
    /// Create a new persistent cache
    pub fn new(cache_dir: impl AsRef<Path>, config: CacheConfig) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir)?;

        // Load or create metadata
        let metadata_path = cache_dir.join("metadata.json");
        let metadata = if metadata_path.exists() {
            let data = fs::read_to_string(&metadata_path)?;
            serde_json::from_str(&data)?
        } else {
            CacheMetadata {
                total_size: 0,
                entry_count: 0,
                last_cleanup: SystemTime::now(),
                hit_count: 0,
                miss_count: 0,
                entries: HashMap::new(),
            }
        };

        let cache = Self {
            cache_dir,
            config,
            metadata,
        };

        // Start cleanup thread if enabled
        if cache.config.auto_cleanup {
            cache.start_cleanup_thread();
        }

        Ok(cache)
    }

    /// Get a value from the cache
    pub fn get<T: for<'de> Deserialize<'de>>(&mut self, key: &str) -> Result<Option<T>> {
        let hash = self.hash_key(key);
        let file_path = self.cache_dir.join(format!("{}.cache", hash));

        if !file_path.exists() {
            self.metadata.miss_count += 1;
            return Ok(None);
        }

        // Check if entry has expired
        if let Some(entry) = self.metadata.entries.get_mut(&hash) {
            if self.is_expired(entry) {
                self.remove_entry(&hash)?;
                self.metadata.miss_count += 1;
                return Ok(None);
            }

            // Update access metadata
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;
            self.metadata.hit_count += 1;

            // Read and deserialize data
            let data = fs::read(&file_path)?;
            let decompressed = if self.config.enable_compression {
                self.decompress(&data)?
            } else {
                data
            };

            let value = serde_json::from_slice(&decompressed)?;
            Ok(Some(value))
        } else {
            self.metadata.miss_count += 1;
            Ok(None)
        }
    }

    /// Put a value into the cache
    pub fn put<T: Serialize>(&mut self, key: &str, value: &T, ttl: Option<Duration>) -> Result<()> {
        let hash = self.hash_key(key);
        let file_path = self.cache_dir.join(format!("{}.cache", hash));

        // Serialize value
        let serialized = serde_json::to_vec(value)?;
        let data = if self.config.enable_compression {
            self.compress(&serialized)?
        } else {
            serialized
        };

        // Check if we need to evict entries
        if self.metadata.total_size + data.len() > self.config.max_size_bytes {
            self.evict_entries(data.len())?;
        }

        // Write to disk
        fs::write(&file_path, &data)?;

        // Update metadata
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        let entry = EntryMetadata {
            key: key.to_string(),
            size: data.len(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            ttl,
        };

        self.metadata.entries.insert(hash, entry);
        self.metadata.total_size += data.len();
        self.metadata.entry_count += 1;

        self.save_metadata()?;
        Ok(())
    }

    /// Remove a value from the cache
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        let hash = self.hash_key(key);
        self.remove_entry(&hash)
    }

    /// Clear the entire cache
    pub fn clear(&mut self) -> Result<()> {
        for entry_hash in self.metadata.entries.keys().cloned().collect::<Vec<_>>() {
            self.remove_entry(&entry_hash)?;
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_size: self.metadata.total_size,
            entry_count: self.metadata.entry_count,
            hit_count: self.metadata.hit_count,
            miss_count: self.metadata.miss_count,
            hit_rate: if self.metadata.hit_count + self.metadata.miss_count > 0 {
                self.metadata.hit_count as f64 /
                (self.metadata.hit_count + self.metadata.miss_count) as f64
            } else {
                0.0
            },
        }
    }

    fn hash_key(&self, key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key);
        format!("{:x}", hasher.finalize())
    }

    fn is_expired(&self, entry: &EntryMetadata) -> bool {
        if let Ok(elapsed) = entry.created_at.elapsed() {
            elapsed > entry.ttl
        } else {
            true
        }
    }

    fn remove_entry(&mut self, hash: &str) -> Result<bool> {
        if let Some(entry) = self.metadata.entries.remove(hash) {
            let file_path = self.cache_dir.join(format!("{}.cache", hash));
            if file_path.exists() {
                fs::remove_file(file_path)?;
            }
            self.metadata.total_size -= entry.size;
            self.metadata.entry_count -= 1;
            self.save_metadata()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn evict_entries(&mut self, required_space: usize) -> Result<()> {
        let mut entries_to_evict = Vec::new();
        let mut freed_space = 0;

        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                // Sort by last accessed time
                let mut entries: Vec<_> = self.metadata.entries.iter().collect();
                entries.sort_by_key(|(_k, v)| v.last_accessed);

                for (hash, entry) in entries {
                    if freed_space >= required_space {
                        break;
                    }
                    entries_to_evict.push(hash.clone());
                    freed_space += entry.size;
                }
            }
            EvictionPolicy::LFU => {
                // Sort by access count
                let mut entries: Vec<_> = self.metadata.entries.iter().collect();
                entries.sort_by_key(|(_k, v)| v.access_count);

                for (hash, entry) in entries {
                    if freed_space >= required_space {
                        break;
                    }
                    entries_to_evict.push(hash.clone());
                    freed_space += entry.size;
                }
            }
            EvictionPolicy::FIFO => {
                // Sort by creation time
                let mut entries: Vec<_> = self.metadata.entries.iter().collect();
                entries.sort_by_key(|(_k, v)| v.created_at);

                for (hash, entry) in entries {
                    if freed_space >= required_space {
                        break;
                    }
                    entries_to_evict.push(hash.clone());
                    freed_space += entry.size;
                }
            }
            EvictionPolicy::TTL => {
                // Remove expired entries first
                for (hash, entry) in &self.metadata.entries {
                    if self.is_expired(entry) {
                        entries_to_evict.push(hash.clone());
                        freed_space += entry.size;
                    }
                }
            }
        }

        for hash in entries_to_evict {
            self.remove_entry(&hash)?;
        }

        Ok(())
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simple compression using zlib
        use flate2::write::ZlibEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::ZlibDecoder;
        use std::io::Read;

        let mut decoder = ZlibDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.cache_dir.join("metadata.json");
        let data = serde_json::to_string_pretty(&self.metadata)?;
        fs::write(metadata_path, data)?;
        Ok(())
    }

    fn start_cleanup_thread(&self) {
        // In production, use tokio or async-std for async cleanup
        // For now, simplified synchronous version
    }

    /// Manual cleanup of expired entries
    pub fn cleanup(&mut self) -> Result<usize> {
        let mut removed = 0;
        let expired: Vec<_> = self.metadata.entries.iter()
            .filter(|(_k, v)| self.is_expired(v))
            .map(|(k, _)| k.clone())
            .collect();

        for hash in expired {
            if self.remove_entry(&hash)? {
                removed += 1;
            }
        }

        self.metadata.last_cleanup = SystemTime::now();
        self.save_metadata()?;
        Ok(removed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_size: usize,
    pub entry_count: usize,
    pub hit_count: usize,
    pub miss_count: usize,
    pub hit_rate: f64,
}

/// Specialized cache for embeddings
pub struct EmbeddingCache {
    cache: PersistentCache,
}

impl EmbeddingCache {
    pub fn new(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let mut config = CacheConfig::default();
        config.default_ttl = Duration::from_secs(86400 * 30); // 30 days for embeddings

        Ok(Self {
            cache: PersistentCache::new(cache_dir, config)?,
        })
    }

    pub fn get_embedding(&mut self, text: &str) -> Result<Option<Vec<f32>>> {
        self.cache.get(&format!("emb:{}", text))
    }

    pub fn put_embedding(&mut self, text: &str, embedding: &[f32]) -> Result<()> {
        self.cache.put(&format!("emb:{}", text), &embedding.to_vec(), None)
    }
}

/// Specialized cache for query results
pub struct QueryCache {
    cache: PersistentCache,
}

impl QueryCache {
    pub fn new(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let mut config = CacheConfig::default();
        config.default_ttl = Duration::from_secs(3600); // 1 hour for queries

        Ok(Self {
            cache: PersistentCache::new(cache_dir, config)?,
        })
    }

    pub fn get_result(&mut self, query: &str) -> Result<Option<String>> {
        self.cache.get(&format!("query:{}", query))
    }

    pub fn put_result(&mut self, query: &str, result: &str) -> Result<()> {
        self.cache.put(&format!("query:{}", query), &result.to_string(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persistent_cache() {
        let dir = tempdir().unwrap();
        let mut cache = PersistentCache::new(dir.path(), CacheConfig::default()).unwrap();

        // Test put and get
        cache.put("key1", &"value1", None).unwrap();
        let value: Option<String> = cache.get("key1").unwrap();
        assert_eq!(value, Some("value1".to_string()));

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.hit_count, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let dir = tempdir().unwrap();
        let mut config = CacheConfig::default();
        config.max_size_bytes = 100; // Very small cache

        let mut cache = PersistentCache::new(dir.path(), config).unwrap();

        // Add entries that exceed cache size
        for i in 0..10 {
            cache.put(&format!("key{}", i), &format!("value{}", i), None).unwrap();
        }

        // Cache should have evicted some entries
        assert!(cache.metadata.entry_count < 10);
    }

    #[test]
    fn test_embedding_cache() {
        let dir = tempdir().unwrap();
        let mut cache = EmbeddingCache::new(dir.path()).unwrap();

        let embedding = vec![0.1, 0.2, 0.3];
        cache.put_embedding("test text", &embedding).unwrap();

        let retrieved = cache.get_embedding("test text").unwrap();
        assert_eq!(retrieved, Some(embedding));
    }
}