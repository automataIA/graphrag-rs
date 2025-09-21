//! Key-value storage abstraction for LightRAG

use std::collections::HashMap;
use crate::core::Result;
#[cfg(feature = "redis-storage")]
use crate::core::GraphRAGError;

/// Key-value storage trait for LightRAG
pub trait KVStore: Send + Sync {
    /// Store an entity with key-value data
    fn store(&mut self, key: &str, value: KVData) -> Result<()>;

    /// Get data by key
    fn get(&self, key: &str) -> Result<Option<KVData>>;

    /// Batch get multiple keys
    fn batch_get(&self, keys: &[String]) -> Result<Vec<KVData>>;

    /// Search for data matching a query
    fn search(&self, query: &str) -> Result<Vec<KVData>>;

    /// Delete data by key
    fn delete(&mut self, key: &str) -> Result<bool>;

    /// List all keys
    fn list_keys(&self) -> Result<Vec<String>>;

    /// Clear all data
    fn clear(&mut self) -> Result<()>;
}

/// Key-value data types
#[derive(Debug, Clone)]
pub enum KVData {
    Entity(EntityData),
    Relation(RelationData),
    Concept(ConceptData),
}

impl KVData {
    pub fn get_id(&self) -> String {
        match self {
            KVData::Entity(e) => format!("entity_{}", e.name),
            KVData::Relation(r) => format!("relation_{}_{}", r.source_entity, r.target_entity),
            KVData::Concept(c) => format!("concept_{}", c.theme.replace(' ', "_")),
        }
    }
}

/// Entity data for key-value storage
#[derive(Debug, Clone)]
pub struct EntityData {
    pub name: String,
    pub entity_type: String,
    pub attributes: Vec<String>,
    pub mentions: Vec<EntityMention>,
}

/// Relationship data for key-value storage
#[derive(Debug, Clone)]
pub struct RelationData {
    pub source_entity: String,
    pub target_entity: String,
    pub relation_type: String,
    pub description: String,
    pub confidence: f32,
}

/// Concept data for key-value storage
#[derive(Debug, Clone)]
pub struct ConceptData {
    pub theme: String,
    pub summary: String,
    pub related_entities: Vec<String>,
    pub confidence: f32,
}

/// Entity mention for tracking occurrences
#[derive(Debug, Clone)]
pub struct EntityMention {
    pub text: String,
    pub context: String,
    pub position: usize,
}

/// In-memory implementation of KVStore
#[derive(Debug)]
pub struct MemoryKVStore {
    data: HashMap<String, KVData>,
}

impl MemoryKVStore {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl Default for MemoryKVStore {
    fn default() -> Self {
        Self::new()
    }
}

impl KVStore for MemoryKVStore {
    fn store(&mut self, key: &str, value: KVData) -> Result<()> {
        self.data.insert(key.to_string(), value);
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<KVData>> {
        Ok(self.data.get(key).cloned())
    }

    fn batch_get(&self, keys: &[String]) -> Result<Vec<KVData>> {
        let mut results = Vec::new();
        for key in keys {
            if let Some(data) = self.data.get(key) {
                results.push(data.clone());
            }
        }
        Ok(results)
    }

    fn search(&self, query: &str) -> Result<Vec<KVData>> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for data in self.data.values() {
            let matches = match data {
                KVData::Entity(e) => {
                    e.name.to_lowercase().contains(&query_lower)
                        || e.entity_type.to_lowercase().contains(&query_lower)
                        || e.attributes.iter().any(|attr| attr.to_lowercase().contains(&query_lower))
                }
                KVData::Relation(r) => {
                    r.source_entity.to_lowercase().contains(&query_lower)
                        || r.target_entity.to_lowercase().contains(&query_lower)
                        || r.relation_type.to_lowercase().contains(&query_lower)
                        || r.description.to_lowercase().contains(&query_lower)
                }
                KVData::Concept(c) => {
                    c.theme.to_lowercase().contains(&query_lower)
                        || c.summary.to_lowercase().contains(&query_lower)
                        || c.related_entities.iter().any(|entity| entity.to_lowercase().contains(&query_lower))
                }
            };

            if matches {
                results.push(data.clone());
            }
        }

        Ok(results)
    }

    fn delete(&mut self, key: &str) -> Result<bool> {
        Ok(self.data.remove(key).is_some())
    }

    fn list_keys(&self) -> Result<Vec<String>> {
        Ok(self.data.keys().cloned().collect())
    }

    fn clear(&mut self) -> Result<()> {
        self.data.clear();
        Ok(())
    }
}

/// Redis adapter for KVStore (placeholder for future implementation)
#[cfg(feature = "redis-storage")]
#[allow(dead_code)]
pub struct RedisKVStore {
    client: redis::Client,
    key_prefix: String,
}

#[cfg(feature = "redis-storage")]
impl RedisKVStore {
    pub fn new(url: &str, key_prefix: String) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| GraphRAGError::Storage {
            message: format!("Failed to connect to Redis: {e}"),
        })?;

        Ok(Self { client, key_prefix })
    }

    #[allow(dead_code)]
    fn prefixed_key(&self, key: &str) -> String {
        format!("{}:{}", self.key_prefix, key)
    }
}

#[cfg(feature = "redis-storage")]
impl KVStore for RedisKVStore {
    fn store(&mut self, _key: &str, _value: KVData) -> Result<()> {
        // Implementation would serialize value and store in Redis
        // This is a placeholder for future implementation
        todo!("Redis KVStore implementation pending")
    }

    fn get(&self, _key: &str) -> Result<Option<KVData>> {
        // Implementation would deserialize from Redis
        todo!("Redis KVStore implementation pending")
    }

    fn batch_get(&self, _keys: &[String]) -> Result<Vec<KVData>> {
        // Implementation would use Redis MGET
        todo!("Redis KVStore implementation pending")
    }

    fn search(&self, _query: &str) -> Result<Vec<KVData>> {
        // Implementation would use Redis SCAN with pattern matching
        todo!("Redis KVStore implementation pending")
    }

    fn delete(&mut self, _key: &str) -> Result<bool> {
        // Implementation would use Redis DEL
        todo!("Redis KVStore implementation pending")
    }

    fn list_keys(&self) -> Result<Vec<String>> {
        // Implementation would use Redis KEYS or SCAN
        todo!("Redis KVStore implementation pending")
    }

    fn clear(&mut self) -> Result<()> {
        // Implementation would use Redis FLUSHDB with prefix
        todo!("Redis KVStore implementation pending")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_kv_store() {
        let mut store = MemoryKVStore::new();

        let entity_data = EntityData {
            name: "John Doe".to_string(),
            entity_type: "person".to_string(),
            attributes: vec!["engineer".to_string(), "AI researcher".to_string()],
            mentions: vec![],
        };

        let kv_data = KVData::Entity(entity_data);

        // Test store and get
        store.store("john_doe", kv_data.clone()).unwrap();
        let retrieved = store.get("john_doe").unwrap();
        assert!(retrieved.is_some());

        // Test search
        let results = store.search("engineer").unwrap();
        assert_eq!(results.len(), 1);

        // Test delete
        let deleted = store.delete("john_doe").unwrap();
        assert!(deleted);

        let not_found = store.get("john_doe").unwrap();
        assert!(not_found.is_none());
    }
}