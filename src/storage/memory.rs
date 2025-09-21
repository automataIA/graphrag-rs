//! In-memory storage implementation
//!
//! This module provides a simple in-memory storage backend that implements
//! the Storage trait. It's useful for testing and development.

use crate::core::{
    traits::Storage, ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, Result,
    TextChunk,
};
use std::collections::HashMap;

/// In-memory storage implementation
#[derive(Debug, Clone)]
pub struct MemoryStorage {
    entities: HashMap<String, Entity>,
    documents: HashMap<String, Document>,
    chunks: HashMap<String, TextChunk>,
    next_id: usize,
}

impl MemoryStorage {
    /// Create a new empty memory storage
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            documents: HashMap::new(),
            chunks: HashMap::new(),
            next_id: 1,
        }
    }

    /// Generate the next unique ID
    fn next_id(&mut self) -> String {
        let id = self.next_id.to_string();
        self.next_id += 1;
        id
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get document count
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.entities.clear();
        self.documents.clear();
        self.chunks.clear();
        self.next_id = 1;
    }

    /// Check if an entity exists
    pub fn has_entity(&self, id: &str) -> bool {
        self.entities.contains_key(id)
    }

    /// Check if a document exists
    pub fn has_document(&self, id: &str) -> bool {
        self.documents.contains_key(id)
    }

    /// Check if a chunk exists
    pub fn has_chunk(&self, id: &str) -> bool {
        self.chunks.contains_key(id)
    }

    /// Get all entity IDs
    pub fn get_entity_ids(&self) -> Vec<String> {
        self.entities.keys().cloned().collect()
    }

    /// Get all document IDs
    pub fn get_document_ids(&self) -> Vec<String> {
        self.documents.keys().cloned().collect()
    }

    /// Get all chunk IDs
    pub fn get_chunk_ids(&self) -> Vec<String> {
        self.chunks.keys().cloned().collect()
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for MemoryStorage {
    type Entity = Entity;
    type Document = Document;
    type Chunk = TextChunk;
    type Error = GraphRAGError;

    fn store_entity(&mut self, mut entity: Self::Entity) -> Result<String> {
        // Generate ID if not set
        if entity.id.0.is_empty() {
            entity.id = EntityId::new(format!("entity_{}", self.next_id()));
        }

        let id = entity.id.0.clone();
        self.entities.insert(id.clone(), entity);
        Ok(id)
    }

    fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
        Ok(self.entities.get(id).cloned())
    }

    fn store_document(&mut self, mut document: Self::Document) -> Result<String> {
        // Generate ID if not set
        if document.id.0.is_empty() {
            document.id = DocumentId::new(format!("doc_{}", self.next_id()));
        }

        let id = document.id.0.clone();
        self.documents.insert(id.clone(), document);
        Ok(id)
    }

    fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>> {
        Ok(self.documents.get(id).cloned())
    }

    fn store_chunk(&mut self, mut chunk: Self::Chunk) -> Result<String> {
        // Generate ID if not set
        if chunk.id.0.is_empty() {
            chunk.id = ChunkId::new(format!("chunk_{}", self.next_id()));
        }

        let id = chunk.id.0.clone();
        self.chunks.insert(id.clone(), chunk);
        Ok(id)
    }

    fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>> {
        Ok(self.chunks.get(id).cloned())
    }

    fn list_entities(&self) -> Result<Vec<String>> {
        Ok(self.entities.keys().cloned().collect())
    }

    fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        for entity in entities {
            let id = self.store_entity(entity)?;
            ids.push(id);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Entity, EntityId};

    #[test]
    fn test_memory_storage_entities() {
        let mut storage = MemoryStorage::new();

        // Test storing an entity
        let entity = Entity::new(
            EntityId::new("test_entity".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        let id = storage.store_entity(entity.clone()).unwrap();
        assert_eq!(id, "test_entity");
        assert_eq!(storage.entity_count(), 1);

        // Test retrieving the entity
        let retrieved = storage.retrieve_entity(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Test Entity");
        assert_eq!(retrieved.entity_type, "Person");
        assert_eq!(retrieved.confidence, 0.9);

        // Test entity exists
        assert!(storage.has_entity(&id));
        assert!(!storage.has_entity("nonexistent"));
    }

    #[test]
    fn test_memory_storage_auto_id_generation() {
        let mut storage = MemoryStorage::new();

        // Test storing an entity with empty ID
        let entity = Entity::new(
            EntityId::new("".to_string()),
            "Auto ID Entity".to_string(),
            "Person".to_string(),
            0.8,
        );

        let id = storage.store_entity(entity).unwrap();
        assert!(id.starts_with("entity_"));
        assert_eq!(storage.entity_count(), 1);

        // Verify we can retrieve it
        let retrieved = storage.retrieve_entity(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Auto ID Entity");
    }

    #[test]
    fn test_memory_storage_batch_operations() {
        let mut storage = MemoryStorage::new();

        let entities = vec![
            Entity::new(
                EntityId::new("entity1".to_string()),
                "Entity 1".to_string(),
                "Person".to_string(),
                0.9,
            ),
            Entity::new(
                EntityId::new("entity2".to_string()),
                "Entity 2".to_string(),
                "Organization".to_string(),
                0.8,
            ),
        ];

        let ids = storage.store_entities_batch(entities).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(storage.entity_count(), 2);

        // Test listing entities
        let all_ids = storage.list_entities().unwrap();
        assert_eq!(all_ids.len(), 2);
        assert!(all_ids.contains(&"entity1".to_string()));
        assert!(all_ids.contains(&"entity2".to_string()));
    }

    #[test]
    fn test_memory_storage_clear() {
        let mut storage = MemoryStorage::new();

        // Add some data
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test".to_string(),
            "Person".to_string(),
            0.9,
        );
        storage.store_entity(entity).unwrap();

        assert_eq!(storage.entity_count(), 1);

        // Clear and verify
        storage.clear();
        assert_eq!(storage.entity_count(), 0);
        assert_eq!(storage.document_count(), 0);
        assert_eq!(storage.chunk_count(), 0);
    }
}
