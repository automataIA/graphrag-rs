//! Storage layer for GraphRAG
//!
//! This module provides abstractions and implementations for storing
//! knowledge graph data, vectors, and metadata.

use crate::core::{Document, Entity, Result, TextChunk};
#[cfg(feature = "async")]
use crate::core::{traits::Storage, GraphRAGError};
use std::collections::HashMap;

/// In-memory storage implementation for development and testing
#[derive(Debug, Default, Clone)]
pub struct MemoryStorage {
    documents: HashMap<String, Document>,
    entities: HashMap<String, Entity>,
    chunks: HashMap<String, TextChunk>,
    metadata: HashMap<String, String>,
}

impl MemoryStorage {
    /// Create a new memory storage instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a document
    pub fn store_document(&mut self, id: String, document: Document) -> Result<()> {
        self.documents.insert(id, document);
        Ok(())
    }

    /// Retrieve a document by ID
    pub fn get_document(&self, id: &str) -> Option<&Document> {
        self.documents.get(id)
    }

    /// Store an entity
    pub fn store_entity(&mut self, id: String, entity: Entity) -> Result<()> {
        self.entities.insert(id, entity);
        Ok(())
    }

    /// Retrieve an entity by ID
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Get all documents
    pub fn all_documents(&self) -> Vec<&Document> {
        self.documents.values().collect()
    }

    /// Get all entities
    pub fn all_entities(&self) -> Vec<&Entity> {
        self.entities.values().collect()
    }

    /// Store a chunk
    pub fn store_chunk(&mut self, id: String, chunk: TextChunk) -> Result<()> {
        self.chunks.insert(id, chunk);
        Ok(())
    }

    /// Retrieve a chunk by ID
    pub fn get_chunk(&self, id: &str) -> Option<&TextChunk> {
        self.chunks.get(id)
    }

    /// Get all chunks
    pub fn all_chunks(&self) -> Vec<&TextChunk> {
        self.chunks.values().collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.documents.clear();
        self.entities.clear();
        self.chunks.clear();
        self.metadata.clear();
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        StorageStats {
            document_count: self.documents.len(),
            entity_count: self.entities.len(),
            chunk_count: self.chunks.len(),
            metadata_count: self.metadata.len(),
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of documents stored
    pub document_count: usize,
    /// Number of entities stored
    pub entity_count: usize,
    /// Number of chunks stored
    pub chunk_count: usize,
    /// Number of metadata entries
    pub metadata_count: usize,
}

// Implement the Storage trait for MemoryStorage (only when async feature is enabled)
#[cfg(feature = "async")]
impl Storage for MemoryStorage {
    type Entity = Entity;
    type Document = Document;
    type Chunk = TextChunk;
    type Error = GraphRAGError;

    fn store_entity(&mut self, entity: Self::Entity) -> Result<String> {
        let id = entity.id.to_string();
        self.entities.insert(id.clone(), entity);
        Ok(id)
    }

    fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
        Ok(self.entities.get(id).cloned())
    }

    fn store_document(&mut self, document: Self::Document) -> Result<String> {
        let id = document.id.to_string();
        self.documents.insert(id.clone(), document);
        Ok(id)
    }

    fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>> {
        Ok(self.documents.get(id).cloned())
    }

    fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String> {
        let id = chunk.id.to_string();
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
        let ids: Vec<String> = entities.iter().map(|e| e.id.to_string()).collect();
        for entity in entities {
            self.entities.insert(entity.id.to_string(), entity);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn test_memory_storage() {
        let mut storage = MemoryStorage::new();

        let doc = Document {
            id: crate::core::DocumentId::new("doc1".to_string()),
            title: "Test Document".to_string(),
            content: "Test content".to_string(),
            metadata: IndexMap::new(),
            chunks: Vec::new(),
        };

        storage.store_document("doc1".to_string(), doc).unwrap();
        assert_eq!(storage.stats().document_count, 1);

        let retrieved = storage.get_document("doc1");
        assert!(retrieved.is_some());
    }
}
