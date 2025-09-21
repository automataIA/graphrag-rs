//! Core data structures and abstractions for GraphRAG
//!
//! This module contains the fundamental types, traits, and error handling
//! that power the GraphRAG system.

pub mod error;
pub mod registry;
pub mod traits;

#[cfg(test)]
pub mod test_traits;

// Re-export key items for convenience
pub use error::{ErrorContext, ErrorSeverity, GraphRAGError, Result};
pub use registry::{RegistryBuilder, ServiceConfig, ServiceContext, ServiceRegistry};
pub use traits::*;

use indexmap::IndexMap;
use petgraph::{graph::NodeIndex, Graph};
use std::collections::HashMap;

// PageRank-related imports are only available when the feature is enabled
#[cfg(feature = "pagerank")]
use sprs::CsMat;

/// Type alias for adjacency matrix build result to reduce type complexity
/// Only available when pagerank feature is enabled
#[cfg(feature = "pagerank")]
type AdjacencyMatrixResult = (
    CsMat<f64>,
    HashMap<EntityId, usize>,
    HashMap<usize, EntityId>,
);

/// Unique identifier for documents
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DocumentId(pub String);

impl DocumentId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for DocumentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<DocumentId> for String {
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

/// Unique identifier for entities
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EntityId(pub String);

impl EntityId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for EntityId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<EntityId> for String {
    fn from(id: EntityId) -> Self {
        id.0
    }
}

/// Unique identifier for text chunks
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ChunkId(pub String);

impl ChunkId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ChunkId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<ChunkId> for String {
    fn from(id: ChunkId) -> Self {
        id.0
    }
}

/// A document in the system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub title: String,
    pub content: String,
    pub metadata: IndexMap<String, String>,
    pub chunks: Vec<TextChunk>,
}

/// A chunk of text from a document
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextChunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub embedding: Option<Vec<f32>>,
    pub entities: Vec<EntityId>,
}

/// An entity extracted from text
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub entity_type: String,
    pub confidence: f32,
    pub mentions: Vec<EntityMention>,
    pub embedding: Option<Vec<f32>>,
}

/// A mention of an entity in text
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityMention {
    pub chunk_id: ChunkId,
    pub start_offset: usize,
    pub end_offset: usize,
    pub confidence: f32,
}

/// Relationship between entities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Relationship {
    pub source: EntityId,
    pub target: EntityId,
    pub relation_type: String,
    pub confidence: f32,
    pub context: Vec<ChunkId>,
}

/// Knowledge graph containing entities and their relationships
#[derive(Debug)]
pub struct KnowledgeGraph {
    graph: Graph<Entity, Relationship>,
    entity_index: HashMap<EntityId, NodeIndex>,
    documents: IndexMap<DocumentId, Document>,
    chunks: IndexMap<ChunkId, TextChunk>,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            entity_index: HashMap::new(),
            documents: IndexMap::new(),
            chunks: IndexMap::new(),
        }
    }

    /// Add a document to the knowledge graph
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let document_id = document.id.clone();

        // Add chunks to the index
        for chunk in &document.chunks {
            self.chunks.insert(chunk.id.clone(), chunk.clone());
        }

        // Store the document
        self.documents.insert(document_id, document);

        Ok(())
    }

    /// Add an entity to the knowledge graph
    pub fn add_entity(&mut self, entity: Entity) -> Result<NodeIndex> {
        let entity_id = entity.id.clone();
        let node_index = self.graph.add_node(entity);
        self.entity_index.insert(entity_id, node_index);
        Ok(node_index)
    }

    /// Add a relationship between entities
    pub fn add_relationship(&mut self, relationship: Relationship) -> Result<()> {
        let source_idx = self.entity_index.get(&relationship.source).ok_or_else(|| {
            crate::GraphRAGError::GraphConstruction {
                message: format!("Source entity {} not found", relationship.source),
            }
        })?;

        let target_idx = self.entity_index.get(&relationship.target).ok_or_else(|| {
            crate::GraphRAGError::GraphConstruction {
                message: format!("Target entity {} not found", relationship.target),
            }
        })?;

        self.graph.add_edge(*source_idx, *target_idx, relationship);
        Ok(())
    }

    /// Add a chunk to the knowledge graph
    pub fn add_chunk(&mut self, chunk: TextChunk) -> Result<()> {
        self.chunks.insert(chunk.id.clone(), chunk);
        Ok(())
    }

    /// Get an entity by ID
    pub fn get_entity(&self, id: &EntityId) -> Option<&Entity> {
        let node_idx = self.entity_index.get(id)?;
        self.graph.node_weight(*node_idx)
    }

    /// Get a document by ID
    pub fn get_document(&self, id: &DocumentId) -> Option<&Document> {
        self.documents.get(id)
    }

    /// Get a chunk by ID
    pub fn get_chunk(&self, id: &ChunkId) -> Option<&TextChunk> {
        self.chunks.get(id)
    }

    /// Get a mutable reference to an entity by ID
    pub fn get_entity_mut(&mut self, id: &EntityId) -> Option<&mut Entity> {
        let node_idx = self.entity_index.get(id)?;
        self.graph.node_weight_mut(*node_idx)
    }

    /// Get a mutable reference to a chunk by ID
    pub fn get_chunk_mut(&mut self, id: &ChunkId) -> Option<&mut TextChunk> {
        self.chunks.get_mut(id)
    }

    /// Get all entities
    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.graph.node_weights()
    }

    /// Get all entities (mutable)
    pub fn entities_mut(&mut self) -> impl Iterator<Item = &mut Entity> {
        self.graph.node_weights_mut()
    }

    /// Get all documents
    pub fn documents(&self) -> impl Iterator<Item = &Document> {
        self.documents.values()
    }

    /// Get all documents (mutable)
    pub fn documents_mut(&mut self) -> impl Iterator<Item = &mut Document> {
        self.documents.values_mut()
    }

    /// Get all chunks
    pub fn chunks(&self) -> impl Iterator<Item = &TextChunk> {
        self.chunks.values()
    }

    /// Get all chunks (mutable)
    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut TextChunk> {
        self.chunks.values_mut()
    }

    /// Get neighbors of an entity
    pub fn get_neighbors(&self, entity_id: &EntityId) -> Vec<(&Entity, &Relationship)> {
        use petgraph::visit::EdgeRef;

        if let Some(&node_idx) = self.entity_index.get(entity_id) {
            self.graph
                .edges(node_idx)
                .filter_map(|edge| {
                    let target_entity = self.graph.node_weight(edge.target())?;
                    Some((target_entity, edge.weight()))
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all relationships in the graph
    pub fn get_all_relationships(&self) -> Vec<&Relationship> {
        self.graph.edge_weights().collect()
    }

    /// Save knowledge graph to JSON file with optimized format for entities and relationships
    pub fn save_to_json(&self, file_path: &str) -> Result<()> {
        use std::fs;

        // Create optimized JSON structure based on 2024 best practices
        let mut json_data = json::JsonValue::new_object();

        // Add metadata
        json_data["metadata"] = json::object! {
            "format_version" => "2.0",
            "created_at" => chrono::Utc::now().to_rfc3339(),
            "total_entities" => self.entities().count(),
            "total_relationships" => self.get_all_relationships().len(),
            "total_chunks" => self.chunks().count(),
            "total_documents" => self.documents().count()
        };

        // Add entities with enhanced information
        let mut entities_array = json::JsonValue::new_array();
        for entity in self.entities() {
            let mut entity_obj = json::object! {
                "id" => entity.id.to_string(),
                "name" => entity.name.clone(),
                "type" => entity.entity_type.clone(),
                "confidence" => entity.confidence,
                "mentions_count" => entity.mentions.len()
            };

            // Add mentions with chunk references
            let mut mentions_array = json::JsonValue::new_array();
            for mention in &entity.mentions {
                mentions_array
                    .push(json::object! {
                        "chunk_id" => mention.chunk_id.to_string(),
                        "start_offset" => mention.start_offset,
                        "end_offset" => mention.end_offset,
                        "confidence" => mention.confidence
                    })
                    .unwrap();
            }
            entity_obj["mentions"] = mentions_array;

            // Add embedding if present
            if let Some(ref embedding) = entity.embedding {
                entity_obj["has_embedding"] = true.into();
                entity_obj["embedding_dimension"] = embedding.len().into();
                // Store only first few dimensions for debugging (full embedding too large for JSON)
                let sample_embedding: Vec<f32> = embedding.iter().take(5).cloned().collect();
                entity_obj["embedding_sample"] = sample_embedding.into();
            } else {
                entity_obj["has_embedding"] = false.into();
            }

            entities_array.push(entity_obj).unwrap();
        }
        json_data["entities"] = entities_array;

        // Add relationships with detailed information
        let mut relationships_array = json::JsonValue::new_array();
        for relationship in self.get_all_relationships() {
            let rel_obj = json::object! {
                "source_id" => relationship.source.to_string(),
                "target_id" => relationship.target.to_string(),
                "relation_type" => relationship.relation_type.clone(),
                "confidence" => relationship.confidence,
                "context_chunks" => relationship.context.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
            };
            relationships_array.push(rel_obj).unwrap();
        }
        json_data["relationships"] = relationships_array;

        // Add chunks information
        let mut chunks_array = json::JsonValue::new_array();
        for chunk in self.chunks() {
            let mut chunk_obj = json::object! {
                "id" => chunk.id.to_string(),
                "document_id" => chunk.document_id.to_string(),
                "content_length" => chunk.content.len(),
                "start_offset" => chunk.start_offset,
                "end_offset" => chunk.end_offset,
                "entity_count" => chunk.entities.len()
            };

            // Add content preview (first 200 chars, respecting UTF-8 boundaries)
            let content_preview = if chunk.content.len() > 200 {
                let mut end = 200;
                while end > 0 && !chunk.content.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &chunk.content[..end])
            } else {
                chunk.content.clone()
            };
            chunk_obj["content_preview"] = content_preview.into();

            // Add embedding info
            chunk_obj["has_embedding"] = chunk.embedding.is_some().into();
            if let Some(ref embedding) = chunk.embedding {
                chunk_obj["embedding_dimension"] = embedding.len().into();
            }

            chunks_array.push(chunk_obj).unwrap();
        }
        json_data["chunks"] = chunks_array;

        // Add documents information
        let mut documents_array = json::JsonValue::new_array();
        for document in self.documents() {
            let mut meta_obj = json::JsonValue::new_object();
            for (key, value) in &document.metadata {
                meta_obj[key] = value.clone().into();
            }

            let doc_obj = json::object! {
                "id" => document.id.to_string(),
                "title" => document.title.clone(),
                "content_length" => document.content.len(),
                "chunks_count" => document.chunks.len(),
                "metadata" => meta_obj
            };
            documents_array.push(doc_obj).unwrap();
        }
        json_data["documents"] = documents_array;

        // Save to file
        fs::write(file_path, json_data.dump())?;
        println!("Knowledge graph saved to {file_path}");

        Ok(())
    }

    /// Find entities by name (case-insensitive partial match)
    pub fn find_entities_by_name(&self, name: &str) -> impl Iterator<Item = &Entity> {
        let name_lower = name.to_lowercase();
        self.entities()
            .filter(move |entity| entity.name.to_lowercase().contains(&name_lower))
    }

    /// Get entity by ID (string version for compatibility)
    pub fn get_entity_by_id(&self, id: &str) -> Option<&Entity> {
        let entity_id = EntityId::new(id.to_string());
        self.get_entity(&entity_id)
    }

    /// Get entity relationships
    pub fn get_entity_relationships(&self, entity_id: &str) -> impl Iterator<Item = &Relationship> {
        let entity_id = EntityId::new(entity_id.to_string());
        if let Some(&node_idx) = self.entity_index.get(&entity_id) {
            self.graph
                .edges(node_idx)
                .map(|edge| edge.weight())
                .collect::<Vec<_>>()
                .into_iter()
        } else {
            Vec::new().into_iter()
        }
    }

    /// Find relationship path between two entities (simplified BFS)
    pub fn find_relationship_path(
        &self,
        entity1: &str,
        entity2: &str,
        _max_depth: usize,
    ) -> Vec<String> {
        let entity1_id = EntityId::new(entity1.to_string());
        let entity2_id = EntityId::new(entity2.to_string());

        let node1 = self.entity_index.get(&entity1_id);
        let node2 = self.entity_index.get(&entity2_id);

        if let (Some(&start), Some(&end)) = (node1, node2) {
            // Simple path finding - just check direct connections for now
            use petgraph::visit::EdgeRef;
            for edge in self.graph.edges(start) {
                if edge.target() == end {
                    return vec![edge.weight().relation_type.clone()];
                }
            }
        }

        Vec::new() // No path found or entities don't exist
    }

    /// Build PageRank calculator from current graph structure
    /// Only available when pagerank feature is enabled
    #[cfg(feature = "pagerank")]
    pub fn build_pagerank_calculator(
        &self,
    ) -> Result<crate::graph::pagerank::PersonalizedPageRank> {
        let config = crate::graph::pagerank::PageRankConfig::default();
        let (adjacency_matrix, node_mapping, reverse_mapping) = self.build_adjacency_matrix()?;

        Ok(crate::graph::pagerank::PersonalizedPageRank::new(
            config,
            adjacency_matrix,
            node_mapping,
            reverse_mapping,
        ))
    }

    /// Build adjacency matrix for PageRank calculations
    /// Only available when pagerank feature is enabled
    #[cfg(feature = "pagerank")]
    fn build_adjacency_matrix(&self) -> Result<AdjacencyMatrixResult> {
        let nodes: Vec<EntityId> = self.entities().map(|e| e.id.clone()).collect();
        let node_mapping: HashMap<EntityId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let reverse_mapping: HashMap<usize, EntityId> = nodes
            .iter()
            .enumerate()
            .map(|(i, id)| (i, id.clone()))
            .collect();

        // Build sparse adjacency matrix from relationships
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for relationship in self.get_all_relationships() {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_mapping.get(&relationship.source),
                node_mapping.get(&relationship.target),
            ) {
                row_indices.push(from_idx);
                col_indices.push(to_idx);
                values.push(relationship.confidence as f64);
            }
        }

        let matrix = if row_indices.is_empty() {
            // Create an empty matrix if no relationships
            sprs::CsMat::zero((nodes.len(), nodes.len()))
        } else {
            // Build using triplet matrix first, then convert to CSR
            let mut triplet_mat = sprs::TriMat::new((nodes.len(), nodes.len()));
            for ((row, col), val) in row_indices
                .into_iter()
                .zip(col_indices.into_iter())
                .zip(values.into_iter())
            {
                triplet_mat.add_triplet(row, col, val);
            }
            triplet_mat.to_csr()
        };

        Ok((matrix, node_mapping, reverse_mapping))
    }

    /// Count entities in the graph
    pub fn entity_count(&self) -> usize {
        self.entities().count()
    }

    /// Count relationships in the graph
    pub fn relationship_count(&self) -> usize {
        self.get_all_relationships().len()
    }

    /// Count documents in the graph
    pub fn document_count(&self) -> usize {
        self.documents().count()
    }

    /// Get all relationships as an iterator
    pub fn relationships(&self) -> impl Iterator<Item = &Relationship> {
        self.graph.edge_weights()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    /// Create a new document
    pub fn new(id: DocumentId, title: String, content: String) -> Self {
        Self {
            id,
            title,
            content,
            metadata: IndexMap::new(),
            chunks: Vec::new(),
        }
    }

    /// Add metadata to the document
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Add chunks to the document
    pub fn with_chunks(mut self, chunks: Vec<TextChunk>) -> Self {
        self.chunks = chunks;
        self
    }
}

impl TextChunk {
    /// Create a new text chunk
    pub fn new(
        id: ChunkId,
        document_id: DocumentId,
        content: String,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Self {
            id,
            document_id,
            content,
            start_offset,
            end_offset,
            embedding: None,
            entities: Vec::new(),
        }
    }

    /// Add an embedding to the chunk
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add entities to the chunk
    pub fn with_entities(mut self, entities: Vec<EntityId>) -> Self {
        self.entities = entities;
        self
    }
}

impl Entity {
    /// Create a new entity
    pub fn new(id: EntityId, name: String, entity_type: String, confidence: f32) -> Self {
        Self {
            id,
            name,
            entity_type,
            confidence,
            mentions: Vec::new(),
            embedding: None,
        }
    }

    /// Add mentions to the entity
    pub fn with_mentions(mut self, mentions: Vec<EntityMention>) -> Self {
        self.mentions = mentions;
        self
    }

    /// Add an embedding to the entity
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}
