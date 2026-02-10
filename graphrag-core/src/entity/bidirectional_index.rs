//! Bidirectional Entity-Chunk Index
//!
//! This module provides efficient bidirectional lookups between entities and chunks,
//! essential for LazyGraphRAG and E2GraphRAG query refinement and concept expansion.
//!
//! ## Key Features
//!
//! - **Fast lookups**: O(1) access in both directions
//! - **Many-to-many relationships**: One entity can appear in multiple chunks, one chunk can contain multiple entities
//! - **Incremental updates**: Add/remove mappings without rebuilding the entire index
//! - **Memory efficient**: Uses IndexMap for predictable iteration order
//!
//! ## Use Cases
//!
//! 1. **Query Expansion**: Given entities in a query, find all relevant chunks
//! 2. **Context Retrieval**: Given a chunk, find all related entities
//! 3. **Concept Graph Building**: Track concept co-occurrence across chunks
//! 4. **Iterative Deepening**: Expand search by traversing entity-chunk relationships
//!
//! ## Example
//!
//! ```rust
//! use graphrag_core::entity::bidirectional_index::BidirectionalIndex;
//! use graphrag_core::core::{EntityId, ChunkId};
//!
//! let mut index = BidirectionalIndex::new();
//!
//! let entity_id = EntityId::new("entity_1".to_string());
//! let chunk_id = ChunkId::new("chunk_1".to_string());
//!
//! // Add mapping
//! index.add_mapping(&entity_id, &chunk_id);
//!
//! // Query by entity
//! let chunks = index.get_chunks_for_entity(&entity_id);
//! assert_eq!(chunks.len(), 1);
//!
//! // Query by chunk
//! let entities = index.get_entities_for_chunk(&chunk_id);
//! assert_eq!(entities.len(), 1);
//! ```

use crate::core::{ChunkId, Entity, EntityId};
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bidirectional index for fast entity-chunk lookups
///
/// This structure maintains two indexes:
/// 1. Entity → Chunks: Given an entity, find all chunks it appears in
/// 2. Chunk → Entities: Given a chunk, find all entities it contains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidirectionalIndex {
    /// Maps entity IDs to the chunks they appear in
    entity_to_chunks: IndexMap<EntityId, IndexSet<ChunkId>>,

    /// Maps chunk IDs to the entities they contain
    chunk_to_entities: IndexMap<ChunkId, IndexSet<EntityId>>,

    /// Total number of entity-chunk mappings
    mapping_count: usize,
}

impl BidirectionalIndex {
    /// Create a new empty bidirectional index
    pub fn new() -> Self {
        Self {
            entity_to_chunks: IndexMap::new(),
            chunk_to_entities: IndexMap::new(),
            mapping_count: 0,
        }
    }

    /// Create a bidirectional index from a collection of entities
    ///
    /// This is useful for building the index from extracted entities
    pub fn from_entities(entities: &[Entity]) -> Self {
        let mut index = Self::new();

        for entity in entities {
            for mention in &entity.mentions {
                index.add_mapping(&entity.id, &mention.chunk_id);
            }
        }

        index
    }

    /// Add a mapping between an entity and a chunk
    ///
    /// This is idempotent - adding the same mapping multiple times has no effect
    pub fn add_mapping(&mut self, entity_id: &EntityId, chunk_id: &ChunkId) {
        // Add to entity → chunks index
        let chunks = self.entity_to_chunks.entry(entity_id.clone()).or_default();
        let was_new = chunks.insert(chunk_id.clone());

        // Add to chunk → entities index
        let entities = self.chunk_to_entities.entry(chunk_id.clone()).or_default();
        entities.insert(entity_id.clone());

        // Update mapping count only if it was a new mapping
        if was_new {
            self.mapping_count += 1;
        }
    }

    /// Add multiple mappings for an entity
    pub fn add_entity_mappings(&mut self, entity_id: &EntityId, chunk_ids: &[ChunkId]) {
        for chunk_id in chunk_ids {
            self.add_mapping(entity_id, chunk_id);
        }
    }

    /// Add multiple mappings for a chunk
    pub fn add_chunk_mappings(&mut self, chunk_id: &ChunkId, entity_ids: &[EntityId]) {
        for entity_id in entity_ids {
            self.add_mapping(entity_id, chunk_id);
        }
    }

    /// Remove a specific mapping between an entity and a chunk
    ///
    /// Returns true if the mapping existed and was removed
    pub fn remove_mapping(&mut self, entity_id: &EntityId, chunk_id: &ChunkId) -> bool {
        let mut removed = false;

        // Remove from entity → chunks index
        if let Some(chunks) = self.entity_to_chunks.get_mut(entity_id) {
            if chunks.swap_remove(chunk_id) {
                removed = true;

                // Clean up empty entries
                if chunks.is_empty() {
                    self.entity_to_chunks.swap_remove(entity_id);
                }
            }
        }

        // Remove from chunk → entities index
        if let Some(entities) = self.chunk_to_entities.get_mut(chunk_id) {
            entities.swap_remove(entity_id);

            // Clean up empty entries
            if entities.is_empty() {
                self.chunk_to_entities.swap_remove(chunk_id);
            }
        }

        if removed {
            self.mapping_count = self.mapping_count.saturating_sub(1);
        }

        removed
    }

    /// Remove all mappings for an entity
    ///
    /// Returns the number of mappings removed
    pub fn remove_entity(&mut self, entity_id: &EntityId) -> usize {
        let mut removed_count = 0;

        if let Some(chunks) = self.entity_to_chunks.swap_remove(entity_id) {
            removed_count = chunks.len();

            // Remove from chunk → entities index
            for chunk_id in chunks {
                if let Some(entities) = self.chunk_to_entities.get_mut(&chunk_id) {
                    entities.swap_remove(entity_id);

                    if entities.is_empty() {
                        self.chunk_to_entities.swap_remove(&chunk_id);
                    }
                }
            }
        }

        self.mapping_count = self.mapping_count.saturating_sub(removed_count);
        removed_count
    }

    /// Remove all mappings for a chunk
    ///
    /// Returns the number of mappings removed
    pub fn remove_chunk(&mut self, chunk_id: &ChunkId) -> usize {
        let mut removed_count = 0;

        if let Some(entities) = self.chunk_to_entities.swap_remove(chunk_id) {
            removed_count = entities.len();

            // Remove from entity → chunks index
            for entity_id in entities {
                if let Some(chunks) = self.entity_to_chunks.get_mut(&entity_id) {
                    chunks.swap_remove(chunk_id);

                    if chunks.is_empty() {
                        self.entity_to_chunks.swap_remove(&entity_id);
                    }
                }
            }
        }

        self.mapping_count = self.mapping_count.saturating_sub(removed_count);
        removed_count
    }

    /// Get all chunks that contain a specific entity
    ///
    /// Returns an empty vector if the entity is not found
    pub fn get_chunks_for_entity(&self, entity_id: &EntityId) -> Vec<ChunkId> {
        self.entity_to_chunks
            .get(entity_id)
            .map(|chunks| chunks.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all entities in a specific chunk
    ///
    /// Returns an empty vector if the chunk is not found
    pub fn get_entities_for_chunk(&self, chunk_id: &ChunkId) -> Vec<EntityId> {
        self.chunk_to_entities
            .get(chunk_id)
            .map(|entities| entities.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Check if a specific entity-chunk mapping exists
    pub fn has_mapping(&self, entity_id: &EntityId, chunk_id: &ChunkId) -> bool {
        self.entity_to_chunks
            .get(entity_id)
            .map(|chunks| chunks.contains(chunk_id))
            .unwrap_or(false)
    }

    /// Get the number of chunks an entity appears in
    pub fn get_entity_chunk_count(&self, entity_id: &EntityId) -> usize {
        self.entity_to_chunks
            .get(entity_id)
            .map(|chunks| chunks.len())
            .unwrap_or(0)
    }

    /// Get the number of entities in a chunk
    pub fn get_chunk_entity_count(&self, chunk_id: &ChunkId) -> usize {
        self.chunk_to_entities
            .get(chunk_id)
            .map(|entities| entities.len())
            .unwrap_or(0)
    }

    /// Get all entity IDs in the index
    pub fn get_all_entities(&self) -> Vec<EntityId> {
        self.entity_to_chunks.keys().cloned().collect()
    }

    /// Get all chunk IDs in the index
    pub fn get_all_chunks(&self) -> Vec<ChunkId> {
        self.chunk_to_entities.keys().cloned().collect()
    }

    /// Get the total number of unique entities
    pub fn entity_count(&self) -> usize {
        self.entity_to_chunks.len()
    }

    /// Get the total number of unique chunks
    pub fn chunk_count(&self) -> usize {
        self.chunk_to_entities.len()
    }

    /// Get the total number of entity-chunk mappings
    pub fn mapping_count(&self) -> usize {
        self.mapping_count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.mapping_count == 0
    }

    /// Clear all mappings from the index
    pub fn clear(&mut self) {
        self.entity_to_chunks.clear();
        self.chunk_to_entities.clear();
        self.mapping_count = 0;
    }

    /// Get co-occurring entities for a given entity
    ///
    /// Returns entities that appear in the same chunks, along with their co-occurrence count
    pub fn get_co_occurring_entities(&self, entity_id: &EntityId) -> HashMap<EntityId, usize> {
        let mut co_occurrences: HashMap<EntityId, usize> = HashMap::new();

        // Get all chunks this entity appears in
        if let Some(chunks) = self.entity_to_chunks.get(entity_id) {
            // For each chunk, get all entities in that chunk
            for chunk_id in chunks {
                if let Some(entities) = self.chunk_to_entities.get(chunk_id) {
                    for other_entity_id in entities {
                        // Skip the entity itself
                        if other_entity_id != entity_id {
                            *co_occurrences.entry(other_entity_id.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        co_occurrences
    }

    /// Get entities that appear in multiple chunks (common entities)
    ///
    /// Returns entities sorted by the number of chunks they appear in (descending)
    pub fn get_common_entities(&self, min_chunk_count: usize) -> Vec<(EntityId, usize)> {
        let mut common_entities: Vec<_> = self
            .entity_to_chunks
            .iter()
            .filter_map(|(entity_id, chunks)| {
                if chunks.len() >= min_chunk_count {
                    Some((entity_id.clone(), chunks.len()))
                } else {
                    None
                }
            })
            .collect();

        // Sort by chunk count descending
        common_entities.sort_by(|a, b| b.1.cmp(&a.1));

        common_entities
    }

    /// Get chunks that contain multiple entities (dense chunks)
    ///
    /// Returns chunks sorted by the number of entities they contain (descending)
    pub fn get_dense_chunks(&self, min_entity_count: usize) -> Vec<(ChunkId, usize)> {
        let mut dense_chunks: Vec<_> = self
            .chunk_to_entities
            .iter()
            .filter_map(|(chunk_id, entities)| {
                if entities.len() >= min_entity_count {
                    Some((chunk_id.clone(), entities.len()))
                } else {
                    None
                }
            })
            .collect();

        // Sort by entity count descending
        dense_chunks.sort_by(|a, b| b.1.cmp(&a.1));

        dense_chunks
    }

    /// Merge another index into this one
    ///
    /// Useful for combining indices from multiple documents
    pub fn merge(&mut self, other: &BidirectionalIndex) {
        for (entity_id, chunks) in &other.entity_to_chunks {
            for chunk_id in chunks {
                self.add_mapping(entity_id, chunk_id);
            }
        }
    }

    /// Get statistics about the index
    pub fn get_statistics(&self) -> IndexStatistics {
        let avg_chunks_per_entity = if self.entity_count() > 0 {
            self.mapping_count as f64 / self.entity_count() as f64
        } else {
            0.0
        };

        let avg_entities_per_chunk = if self.chunk_count() > 0 {
            self.mapping_count as f64 / self.chunk_count() as f64
        } else {
            0.0
        };

        IndexStatistics {
            total_entities: self.entity_count(),
            total_chunks: self.chunk_count(),
            total_mappings: self.mapping_count(),
            avg_chunks_per_entity,
            avg_entities_per_chunk,
        }
    }
}

impl Default for BidirectionalIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the bidirectional index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total number of unique entities
    pub total_entities: usize,

    /// Total number of unique chunks
    pub total_chunks: usize,

    /// Total number of entity-chunk mappings
    pub total_mappings: usize,

    /// Average number of chunks per entity
    pub avg_chunks_per_entity: f64,

    /// Average number of entities per chunk
    pub avg_entities_per_chunk: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, EntityId, EntityMention};

    #[test]
    fn test_basic_operations() {
        let mut index = BidirectionalIndex::new();

        let entity1 = EntityId::new("entity_1".to_string());
        let entity2 = EntityId::new("entity_2".to_string());
        let chunk1 = ChunkId::new("chunk_1".to_string());
        let chunk2 = ChunkId::new("chunk_2".to_string());

        // Add mappings
        index.add_mapping(&entity1, &chunk1);
        index.add_mapping(&entity1, &chunk2);
        index.add_mapping(&entity2, &chunk1);

        // Test entity → chunks lookup
        let chunks = index.get_chunks_for_entity(&entity1);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.contains(&chunk1));
        assert!(chunks.contains(&chunk2));

        // Test chunk → entities lookup
        let entities = index.get_entities_for_chunk(&chunk1);
        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&entity1));
        assert!(entities.contains(&entity2));

        // Test counts
        assert_eq!(index.entity_count(), 2);
        assert_eq!(index.chunk_count(), 2);
        assert_eq!(index.mapping_count(), 3);
    }

    #[test]
    fn test_idempotent_add() {
        let mut index = BidirectionalIndex::new();

        let entity = EntityId::new("entity_1".to_string());
        let chunk = ChunkId::new("chunk_1".to_string());

        index.add_mapping(&entity, &chunk);
        index.add_mapping(&entity, &chunk);
        index.add_mapping(&entity, &chunk);

        assert_eq!(index.mapping_count(), 1);
        assert_eq!(index.get_chunks_for_entity(&entity).len(), 1);
    }

    #[test]
    fn test_removal() {
        let mut index = BidirectionalIndex::new();

        let entity1 = EntityId::new("entity_1".to_string());
        let entity2 = EntityId::new("entity_2".to_string());
        let chunk1 = ChunkId::new("chunk_1".to_string());
        let chunk2 = ChunkId::new("chunk_2".to_string());

        index.add_mapping(&entity1, &chunk1);
        index.add_mapping(&entity1, &chunk2);
        index.add_mapping(&entity2, &chunk1);

        // Remove specific mapping
        assert!(index.remove_mapping(&entity1, &chunk1));
        assert_eq!(index.mapping_count(), 2);

        // Remove entity
        let removed = index.remove_entity(&entity1);
        assert_eq!(removed, 1);
        assert_eq!(index.mapping_count(), 1);

        // Only entity2 → chunk1 should remain
        assert_eq!(index.entity_count(), 1);
        assert_eq!(index.chunk_count(), 1);
    }

    #[test]
    fn test_from_entities() {
        let entity1 = Entity::new(
            EntityId::new("entity_1".to_string()),
            "Entity 1".to_string(),
            "PERSON".to_string(),
            0.9,
        )
        .with_mentions(vec![
            EntityMention {
                chunk_id: ChunkId::new("chunk_1".to_string()),
                start_offset: 0,
                end_offset: 8,
                confidence: 0.9,
            },
            EntityMention {
                chunk_id: ChunkId::new("chunk_2".to_string()),
                start_offset: 10,
                end_offset: 18,
                confidence: 0.9,
            },
        ]);

        let index = BidirectionalIndex::from_entities(&[entity1]);

        assert_eq!(index.entity_count(), 1);
        assert_eq!(index.chunk_count(), 2);
        assert_eq!(index.mapping_count(), 2);
    }

    #[test]
    fn test_co_occurrence() {
        let mut index = BidirectionalIndex::new();

        let entity1 = EntityId::new("entity_1".to_string());
        let entity2 = EntityId::new("entity_2".to_string());
        let entity3 = EntityId::new("entity_3".to_string());
        let chunk1 = ChunkId::new("chunk_1".to_string());
        let chunk2 = ChunkId::new("chunk_2".to_string());

        // entity1 and entity2 co-occur in both chunks
        index.add_mapping(&entity1, &chunk1);
        index.add_mapping(&entity1, &chunk2);
        index.add_mapping(&entity2, &chunk1);
        index.add_mapping(&entity2, &chunk2);

        // entity3 co-occurs with entity1 only in chunk1
        index.add_mapping(&entity3, &chunk1);

        let co_occurrences = index.get_co_occurring_entities(&entity1);
        assert_eq!(co_occurrences.get(&entity2), Some(&2)); // co-occurs in 2 chunks
        assert_eq!(co_occurrences.get(&entity3), Some(&1)); // co-occurs in 1 chunk
    }

    #[test]
    fn test_common_entities() {
        let mut index = BidirectionalIndex::new();

        let entity1 = EntityId::new("entity_1".to_string());
        let entity2 = EntityId::new("entity_2".to_string());
        let chunk1 = ChunkId::new("chunk_1".to_string());
        let chunk2 = ChunkId::new("chunk_2".to_string());
        let chunk3 = ChunkId::new("chunk_3".to_string());

        // entity1 appears in 3 chunks
        index.add_mapping(&entity1, &chunk1);
        index.add_mapping(&entity1, &chunk2);
        index.add_mapping(&entity1, &chunk3);

        // entity2 appears in 1 chunk
        index.add_mapping(&entity2, &chunk1);

        let common = index.get_common_entities(2);
        assert_eq!(common.len(), 1);
        assert_eq!(common[0].0, entity1);
        assert_eq!(common[0].1, 3);
    }

    #[test]
    fn test_merge() {
        let mut index1 = BidirectionalIndex::new();
        let mut index2 = BidirectionalIndex::new();

        let entity1 = EntityId::new("entity_1".to_string());
        let entity2 = EntityId::new("entity_2".to_string());
        let chunk1 = ChunkId::new("chunk_1".to_string());
        let chunk2 = ChunkId::new("chunk_2".to_string());

        index1.add_mapping(&entity1, &chunk1);
        index2.add_mapping(&entity2, &chunk2);

        index1.merge(&index2);

        assert_eq!(index1.entity_count(), 2);
        assert_eq!(index1.chunk_count(), 2);
        assert_eq!(index1.mapping_count(), 2);
    }

    #[test]
    fn test_statistics() {
        let mut index = BidirectionalIndex::new();

        let entity1 = EntityId::new("entity_1".to_string());
        let entity2 = EntityId::new("entity_2".to_string());
        let chunk1 = ChunkId::new("chunk_1".to_string());
        let chunk2 = ChunkId::new("chunk_2".to_string());

        index.add_mapping(&entity1, &chunk1);
        index.add_mapping(&entity1, &chunk2);
        index.add_mapping(&entity2, &chunk1);

        let stats = index.get_statistics();
        assert_eq!(stats.total_entities, 2);
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.total_mappings, 3);
        assert_eq!(stats.avg_chunks_per_entity, 1.5);
        assert_eq!(stats.avg_entities_per_chunk, 1.5);
    }
}
