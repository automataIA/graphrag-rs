//! Core data structures and abstractions for GraphRAG
//!
//! This module contains the fundamental types, traits, and error handling
//! that power the GraphRAG system.

pub mod error;
pub mod metadata;

// Registry requires async feature (uses storage)
#[cfg(feature = "async")]
pub mod registry;

// Traits module requires async feature
#[cfg(feature = "async")]
pub mod traits;

// Ollama adapters require both async and ollama features
#[cfg(all(feature = "async", feature = "ollama"))]
pub mod ollama_adapters;

// Entity extraction adapters require async feature
#[cfg(feature = "async")]
pub mod entity_adapters;

// Retrieval system adapters require async feature
#[cfg(feature = "async")]
pub mod retrieval_adapters;

// Test utilities for mock implementations
#[cfg(feature = "async")]
pub mod test_utils;

#[cfg(test)]
pub mod test_traits;

// Re-export key items for convenience
pub use error::{ErrorContext, ErrorSeverity, ErrorSuggestion, GraphRAGError, Result};
pub use metadata::ChunkMetadata;

#[cfg(feature = "async")]
pub use registry::{RegistryBuilder, ServiceConfig, ServiceContext, ServiceRegistry};

// Traits require async feature
#[cfg(feature = "async")]
pub use traits::*;

/// Core trait for text chunking strategies
///
/// This trait provides a simple interface for different chunking approaches.
/// Implementations can range from simple text splitters to sophisticated
/// AST-based code chunking strategies.
///
/// # Examples
///
/// ```rust
/// use graphrag_core::{ChunkingStrategy, TextChunk};
///
/// struct SimpleChunker;
///
/// impl ChunkingStrategy for SimpleChunker {
///     fn chunk(&self, text: &str) -> Vec<TextChunk> {
///         // Simple implementation
///         vec![]
///     }
/// }
/// ```
pub trait ChunkingStrategy: Send + Sync {
    /// Chunk text into pieces following the strategy's logic
    ///
    /// # Arguments
    /// * `text` - The input text to chunk
    ///
    /// # Returns
    /// A vector of TextChunk objects representing the chunks
    fn chunk(&self, text: &str) -> Vec<TextChunk>;
}

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
    /// Creates a new DocumentId from a string
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
    /// Creates a new EntityId from a string
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
    /// Creates a new ChunkId from a string
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
    /// Unique identifier for the document
    pub id: DocumentId,
    /// Title of the document
    pub title: String,
    /// Full text content of the document
    pub content: String,
    /// Additional metadata key-value pairs
    pub metadata: IndexMap<String, String>,
    /// Text chunks extracted from the document
    pub chunks: Vec<TextChunk>,
}

/// A chunk of text from a document
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextChunk {
    /// Unique identifier for the chunk
    pub id: ChunkId,
    /// ID of the parent document
    pub document_id: DocumentId,
    /// Text content of the chunk
    pub content: String,
    /// Starting character offset in the original document
    pub start_offset: usize,
    /// Ending character offset in the original document
    pub end_offset: usize,
    /// Optional vector embedding for the chunk
    pub embedding: Option<Vec<f32>>,
    /// List of entity IDs mentioned in this chunk
    pub entities: Vec<EntityId>,
    /// Semantic metadata for the chunk (chapter, keywords, summary, etc.)
    pub metadata: ChunkMetadata,
}

/// An entity extracted from text
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    /// Unique identifier for the entity
    pub id: EntityId,
    /// Name or label of the entity
    pub name: String,
    /// Type or category of the entity (e.g., "person", "organization")
    pub entity_type: String,
    /// Confidence score for the entity extraction (0.0-1.0)
    pub confidence: f32,
    /// List of locations where this entity is mentioned
    pub mentions: Vec<EntityMention>,
    /// Optional vector embedding for the entity
    pub embedding: Option<Vec<f32>>,

    // Temporal fields (Phase 1.2 - Advanced GraphRAG)
    /// First time this entity was mentioned (Unix timestamp)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub first_mentioned: Option<i64>,
    /// Last time this entity was mentioned (Unix timestamp)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_mentioned: Option<i64>,
    /// Temporal validity range (when the entity exists in the real world)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temporal_validity: Option<crate::graph::temporal::TemporalRange>,
}

/// A mention of an entity in text
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityMention {
    /// ID of the chunk containing this mention
    pub chunk_id: ChunkId,
    /// Starting character offset of the mention in the chunk
    pub start_offset: usize,
    /// Ending character offset of the mention in the chunk
    pub end_offset: usize,
    /// Confidence score for this specific mention (0.0-1.0)
    pub confidence: f32,
}

/// Relationship between entities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Relationship {
    /// Source entity ID for the relationship
    pub source: EntityId,
    /// Target entity ID for the relationship
    pub target: EntityId,
    /// Type of relationship (e.g., "works_for", "located_in")
    pub relation_type: String,
    /// Confidence score for the relationship (0.0-1.0)
    pub confidence: f32,
    /// Chunk IDs providing context for this relationship
    pub context: Vec<ChunkId>,

    /// Optional embedding vector for semantic similarity matching
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub embedding: Option<Vec<f32>>,

    // Temporal and causal fields (Phase 1.2 - Advanced GraphRAG)
    /// Type of temporal relationship (Before, During, Caused, etc.)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temporal_type: Option<crate::graph::temporal::TemporalRelationType>,
    /// Temporal range when this relationship was valid
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temporal_range: Option<crate::graph::temporal::TemporalRange>,
    /// Strength of causal relationship (0.0-1.0), only meaningful for causal temporal types
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub causal_strength: Option<f32>,
}

impl Relationship {
    /// Create a new relationship
    pub fn new(
        source: EntityId,
        target: EntityId,
        relation_type: String,
        confidence: f32,
    ) -> Self {
        Self {
            source,
            target,
            relation_type,
            confidence,
            context: Vec::new(),
            embedding: None,
            // Temporal fields default to None (backward compatible)
            temporal_type: None,
            temporal_range: None,
            causal_strength: None,
        }
    }

    /// Add context chunks to the relationship
    pub fn with_context(mut self, context: Vec<ChunkId>) -> Self {
        self.context = context;
        self
    }

    /// Set temporal type for this relationship
    pub fn with_temporal_type(
        mut self,
        temporal_type: crate::graph::temporal::TemporalRelationType,
    ) -> Self {
        self.temporal_type = Some(temporal_type);
        // Auto-set causal strength based on type if not already set
        if self.causal_strength.is_none() && temporal_type.is_causal() {
            self.causal_strength = Some(temporal_type.default_strength());
        }
        self
    }

    /// Set temporal range for this relationship
    pub fn with_temporal_range(mut self, start: i64, end: i64) -> Self {
        self.temporal_range = Some(crate::graph::temporal::TemporalRange::new(start, end));
        self
    }

    /// Set causal strength for this relationship
    pub fn with_causal_strength(mut self, strength: f32) -> Self {
        self.causal_strength = Some(strength.clamp(0.0, 1.0));
        self
    }

    /// Set embedding vector for this relationship (Phase 2.2)
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Knowledge graph containing entities and their relationships
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    graph: Graph<Entity, Relationship>,
    entity_index: HashMap<EntityId, NodeIndex>,
    documents: IndexMap<DocumentId, Document>,
    chunks: IndexMap<ChunkId, TextChunk>,

    /// Hierarchical organization of relationships (Phase 3.1)
    #[cfg(feature = "async")]
    pub relationship_hierarchy: Option<crate::graph::hierarchical_relationships::RelationshipHierarchy>,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            entity_index: HashMap::new(),
            documents: IndexMap::new(),
            chunks: IndexMap::new(),
            #[cfg(feature = "async")]
            relationship_hierarchy: None,
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

    /// Load knowledge graph from JSON file
    pub fn load_from_json(file_path: &str) -> Result<Self> {
        use std::fs;

        // Read and parse JSON
        let json_str = fs::read_to_string(file_path)?;
        let json_data = json::parse(&json_str)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to parse JSON: {}", e),
            })?;

        let mut kg = KnowledgeGraph::new();

        // Load entities
        if json_data["entities"].is_array() {
            for entity_obj in json_data["entities"].members() {
                let id = EntityId::new(entity_obj["id"].as_str().unwrap_or("").to_string());
                let name = entity_obj["name"].as_str().unwrap_or("").to_string();
                let entity_type = entity_obj["type"].as_str().unwrap_or("").to_string();
                let confidence = entity_obj["confidence"].as_f32().unwrap_or(0.0);

                // Parse mentions
                let mut mentions = Vec::new();
                if entity_obj["mentions"].is_array() {
                    for mention_obj in entity_obj["mentions"].members() {
                        let mention = EntityMention {
                            chunk_id: ChunkId::new(mention_obj["chunk_id"].as_str().unwrap_or("").to_string()),
                            start_offset: mention_obj["start_offset"].as_usize().unwrap_or(0),
                            end_offset: mention_obj["end_offset"].as_usize().unwrap_or(0),
                            confidence: mention_obj["confidence"].as_f32().unwrap_or(0.0),
                        };
                        mentions.push(mention);
                    }
                }

                let entity = Entity {
                    id,
                    name,
                    entity_type,
                    confidence,
                    mentions,
                    embedding: None, // Embeddings not stored in JSON
                    first_mentioned: None,
                    last_mentioned: None,
                    temporal_validity: None,
                };

                kg.add_entity(entity)?;
            }
        }

        // Load relationships
        if json_data["relationships"].is_array() {
            for rel_obj in json_data["relationships"].members() {
                let source = EntityId::new(rel_obj["source_id"].as_str().unwrap_or("").to_string());
                let target = EntityId::new(rel_obj["target_id"].as_str().unwrap_or("").to_string());
                let relation_type = rel_obj["relation_type"].as_str().unwrap_or("").to_string();
                let confidence = rel_obj["confidence"].as_f32().unwrap_or(0.0);

                let mut context = Vec::new();
                if rel_obj["context_chunks"].is_array() {
                    for chunk_id in rel_obj["context_chunks"].members() {
                        if let Some(chunk_id_str) = chunk_id.as_str() {
                            context.push(ChunkId::new(chunk_id_str.to_string()));
                        }
                    }
                }

                let relationship = Relationship {
                    source,
                    target,
                    relation_type,
                    confidence,
                    context,
                    embedding: None,
                    temporal_type: None,
                    temporal_range: None,
                    causal_strength: None,
                };

                // Ignore errors if entities don't exist
                let _ = kg.add_relationship(relationship);
            }
        }

        // Load chunks with full content
        if json_data["chunks"].is_array() {
            for chunk_obj in json_data["chunks"].members() {
                let id = ChunkId::new(chunk_obj["id"].as_str().unwrap_or("").to_string());
                let document_id = DocumentId::new(chunk_obj["document_id"].as_str().unwrap_or("").to_string());
                let start_offset = chunk_obj["start_offset"].as_usize().unwrap_or(0);
                let end_offset = chunk_obj["end_offset"].as_usize().unwrap_or(0);

                // Get full content
                let content = chunk_obj["content"].as_str().unwrap_or("").to_string();

                // Load entities list
                let mut entities = Vec::new();
                if chunk_obj["entities"].is_array() {
                    for entity_id in chunk_obj["entities"].members() {
                        if let Some(entity_id_str) = entity_id.as_str() {
                            entities.push(EntityId::new(entity_id_str.to_string()));
                        }
                    }
                }

                let chunk = TextChunk {
                    id,
                    document_id,
                    content,
                    start_offset,
                    end_offset,
                    embedding: None, // Embeddings not stored in JSON
                    entities,
                    metadata: ChunkMetadata::default(),
                };
                kg.add_chunk(chunk)?;
            }
        }

        // Load documents with full content
        if json_data["documents"].is_array() {
            for doc_obj in json_data["documents"].members() {
                let id = DocumentId::new(doc_obj["id"].as_str().unwrap_or("").to_string());
                let title = doc_obj["title"].as_str().unwrap_or("").to_string();
                let content = doc_obj["content"].as_str().unwrap_or("").to_string();

                // Parse metadata
                let mut metadata = IndexMap::new();
                if doc_obj["metadata"].is_object() {
                    for (key, value) in doc_obj["metadata"].entries() {
                        metadata.insert(key.to_string(), value.as_str().unwrap_or("").to_string());
                    }
                }

                let document = Document {
                    id,
                    title,
                    content,
                    metadata,
                    chunks: vec![], // Chunks are stored separately in the graph
                };
                kg.add_document(document)?;
            }
        }

        Ok(kg)
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
            if let Some(embedding) = &entity.embedding {
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

        // Add chunks information with FULL content for persistence
        let mut chunks_array = json::JsonValue::new_array();
        for chunk in self.chunks() {
            let mut chunk_obj = json::object! {
                "id" => chunk.id.to_string(),
                "document_id" => chunk.document_id.to_string(),
                "content" => chunk.content.clone(),  // Full content for persistence
                "start_offset" => chunk.start_offset,
                "end_offset" => chunk.end_offset
            };

            // Add entities list
            let entities_list: Vec<String> = chunk.entities.iter()
                .map(|e| e.to_string())
                .collect();
            chunk_obj["entities"] = entities_list.into();

            // Add embedding info
            chunk_obj["has_embedding"] = chunk.embedding.is_some().into();
            if let Some(embedding) = &chunk.embedding {
                chunk_obj["embedding_dimension"] = embedding.len().into();
            }

            chunks_array.push(chunk_obj).unwrap();
        }
        json_data["chunks"] = chunks_array;

        // Add documents information with FULL content for persistence
        let mut documents_array = json::JsonValue::new_array();
        for document in self.documents() {
            let mut meta_obj = json::JsonValue::new_object();
            for (key, value) in &document.metadata {
                meta_obj[key] = value.clone().into();
            }

            let doc_obj = json::object! {
                "id" => document.id.to_string(),
                "title" => document.title.clone(),
                "content" => document.content.clone(),  // Full content for persistence
                "metadata" => meta_obj
            };
            documents_array.push(doc_obj).unwrap();
        }
        json_data["documents"] = documents_array;

        // Save to file
        fs::write(file_path, json_data.dump())?;
        tracing::info!("Knowledge graph saved to {file_path}");

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

    /// Clear all entities and relationships while preserving documents and chunks
    ///
    /// This is useful for rebuilding the graph from scratch without reloading documents.
    pub fn clear_entities_and_relationships(&mut self) {
        self.graph.clear();
        self.entity_index.clear();
        // Note: documents and chunks are preserved
    }

    /// Calculate cosine similarity between two embedding vectors
    ///
    /// # Arguments
    ///
    /// * `a` - First embedding vector
    /// * `b` - Second embedding vector
    ///
    /// # Returns
    ///
    /// Cosine similarity score (range: -1.0 to 1.0, typically 0.0 to 1.0 for embeddings)
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Calculate temporal relevance score for a temporal range
    ///
    /// Gives higher scores to:
    /// - Recent events (closer to current time)
    /// - Events with known temporal bounds
    ///
    /// # Arguments
    ///
    /// * `range` - The temporal range to score
    ///
    /// # Returns
    ///
    /// Temporal relevance boost (0.0-0.3 range)
    fn calculate_temporal_relevance(range: &crate::graph::temporal::TemporalRange) -> f32 {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Get current timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Calculate recency: how close is the event to now?
        let mid_point = (range.start + range.end) / 2;
        let years_ago = ((now - mid_point) / (365 * 24 * 3600)).abs();

        // Decay function: more recent = higher score
        // Events in the past 10 years get max boost (0.3)
        // Older events get gradually less boost
        let recency_boost = if years_ago <= 10 {
            0.3
        } else if years_ago <= 50 {
            0.3 * (1.0 - (years_ago - 10) as f32 / 40.0)
        } else if years_ago <= 200 {
            0.1 * (1.0 - (years_ago - 50) as f32 / 150.0)
        } else {
            0.05 // Ancient events still get small boost for having temporal info
        };

        recency_boost.max(0.0)
    }

    /// Calculate dynamic weight for a relationship based on query context (Phase 2.2)
    ///
    /// Combines multiple factors:
    /// - Base weight: relationship.confidence
    /// - Semantic boost: similarity between relationship and query embeddings
    /// - Temporal boost: relevance of temporal range
    /// - Conceptual boost: matching query concepts in relationship type
    ///
    /// # Arguments
    ///
    /// * `relationship` - The relationship to weight
    /// * `query_embedding` - Optional embedding vector for the query
    /// * `query_concepts` - Concepts extracted from the query
    ///
    /// # Returns
    ///
    /// Dynamic weight value (typically 0.0-2.0, can be higher with strong matches)
    pub fn dynamic_weight(
        &self,
        relationship: &Relationship,
        query_embedding: Option<&[f32]>,
        query_concepts: &[String],
    ) -> f32 {
        let base_weight = relationship.confidence;

        // Semantic boost: similarity between relationship and query
        let semantic_boost = if let (Some(rel_emb), Some(query_emb)) =
            (relationship.embedding.as_deref(), query_embedding)
        {
            Self::cosine_similarity(rel_emb, query_emb).max(0.0)
        } else {
            0.0
        };

        // Temporal boost: relationships with recent or relevant temporal ranges
        let temporal_boost = if let Some(tr) = &relationship.temporal_range {
            Self::calculate_temporal_relevance(tr)
        } else {
            0.0
        };

        // Conceptual boost: relationships whose type matches query concepts
        let concept_boost = query_concepts
            .iter()
            .filter(|c| relationship.relation_type.to_lowercase().contains(&c.to_lowercase()))
            .count() as f32 * 0.15; // 15% boost per matching concept

        // Causal boost: causal relationships get extra weight
        let causal_boost = if let Some(strength) = relationship.causal_strength {
            strength * 0.2 // Up to 20% boost for strong causal links
        } else {
            0.0
        };

        // Combine all factors
        base_weight * (1.0 + semantic_boost + temporal_boost + concept_boost + causal_boost)
    }

    /// Convert KnowledgeGraph to petgraph format for Leiden clustering
    /// Returns a graph with entity names as nodes and relationship confidences as edge weights
    /// Only available when leiden feature is enabled
    #[cfg(feature = "leiden")]
    pub fn to_leiden_graph(&self) -> petgraph::Graph<String, f32, petgraph::Undirected> {
        let mut graph = Graph::new_undirected();
        let mut node_map = HashMap::new();

        // Add nodes (entities) - use entity name as node label
        for entity in self.entities() {
            let idx = graph.add_node(entity.name.clone());
            node_map.insert(entity.id.clone(), idx);
        }

        // Add edges (relationships) with confidence as weight
        for rel in self.get_all_relationships() {
            if let (Some(&src), Some(&tgt)) = (
                node_map.get(&rel.source),
                node_map.get(&rel.target)
            ) {
                graph.add_edge(src, tgt, rel.confidence);
            }
        }

        graph
    }

    /// Detect hierarchical communities in the entity graph using Leiden algorithm
    /// Only available when leiden feature is enabled
    ///
    /// # Arguments
    /// * `config` - Leiden algorithm configuration
    ///
    /// # Returns
    /// HierarchicalCommunities structure with community assignments at each level
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::{KnowledgeGraph, graph::LeidenConfig};
    ///
    /// let graph = KnowledgeGraph::new();
    /// // ... build graph ...
    ///
    /// let config = LeidenConfig {
    ///     max_cluster_size: 10,
    ///     resolution: 1.0,
    ///     ..Default::default()
    /// };
    ///
    /// let communities = graph.detect_hierarchical_communities(config).unwrap();
    /// ```
    #[cfg(feature = "leiden")]
    pub fn detect_hierarchical_communities(
        &self,
        config: crate::graph::leiden::LeidenConfig,
    ) -> Result<crate::graph::leiden::HierarchicalCommunities> {
        use crate::graph::leiden::LeidenCommunityDetector;

        // Convert to Leiden-compatible graph format
        let leiden_graph = self.to_leiden_graph();

        // Create detector and run clustering
        let detector = LeidenCommunityDetector::new(config);
        let mut communities = detector.detect_communities(&leiden_graph)?;

        // Enrich with entity metadata
        communities.entity_mapping = Some(self.build_entity_mapping());

        Ok(communities)
    }

    /// Build mapping from entity names to entity metadata
    /// Used to enrich hierarchical communities with entity information
    #[cfg(feature = "leiden")]
    fn build_entity_mapping(&self) -> HashMap<String, crate::graph::leiden::EntityMetadata> {
        use crate::graph::leiden::EntityMetadata;

        self.entities()
            .map(|entity| {
                let metadata = EntityMetadata {
                    id: entity.id.to_string(),
                    name: entity.name.clone(),
                    entity_type: entity.entity_type.clone(),
                    confidence: entity.confidence,
                    mention_count: entity.mentions.len(),
                };
                (entity.name.clone(), metadata)
            })
            .collect()
    }

    /// Build hierarchical organization of relationships (Phase 3.1)
    ///
    /// Creates multi-level clusters of relationships using recursive community detection
    /// and LLM-generated summaries for efficient retrieval.
    ///
    /// # Arguments
    /// * `num_levels` - Number of hierarchy levels to create (default: 3)
    /// * `ollama_client` - Optional Ollama client for generating cluster summaries
    ///
    /// # Returns
    /// Result containing the built hierarchy or an error
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::{KnowledgeGraph, ollama::OllamaClient};
    ///
    /// let mut graph = KnowledgeGraph::new();
    /// // ... build graph ...
    ///
    /// let ollama = OllamaClient::new("http://localhost:11434", "llama3.2");
    /// graph.build_relationship_hierarchy(3, Some(ollama)).await.unwrap();
    /// ```
    #[cfg(feature = "async")]
    pub async fn build_relationship_hierarchy(
        &mut self,
        num_levels: usize,
        ollama_client: Option<crate::ollama::OllamaClient>,
    ) -> Result<()> {
        use crate::graph::hierarchical_relationships::HierarchyBuilder;

        let builder = HierarchyBuilder::from_graph(self)
            .with_num_levels(num_levels);

        let builder = if let Some(client) = ollama_client {
            builder.with_ollama_client(client)
        } else {
            builder
        };

        let hierarchy = builder.build().await?;
        self.relationship_hierarchy = Some(hierarchy);

        Ok(())
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
            metadata: ChunkMetadata::default(),
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

    /// Add metadata to the chunk
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
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
            // Temporal fields default to None (backward compatible)
            first_mentioned: None,
            last_mentioned: None,
            temporal_validity: None,
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

    /// Set temporal validity range for this entity
    pub fn with_temporal_validity(mut self, start: i64, end: i64) -> Self {
        self.temporal_validity = Some(crate::graph::temporal::TemporalRange::new(start, end));
        self
    }

    /// Set mention timestamps for this entity
    pub fn with_mention_times(mut self, first: i64, last: i64) -> Self {
        self.first_mentioned = Some(first);
        self.last_mentioned = Some(last);
        self
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    #[test]
    fn test_entity_with_temporal_fields() {
        let entity = Entity::new(
            EntityId::new("socrates".to_string()),
            "Socrates".to_string(),
            "PERSON".to_string(),
            0.9,
        )
        .with_temporal_validity(-470 * 365 * 24 * 3600, -399 * 365 * 24 * 3600) // 470-399 BC
        .with_mention_times(1000, 2000);

        assert_eq!(entity.name, "Socrates");
        assert!(entity.temporal_validity.is_some());
        assert!(entity.first_mentioned.is_some());
        assert!(entity.last_mentioned.is_some());

        let validity = entity.temporal_validity.unwrap();
        assert_eq!(validity.start, -470 * 365 * 24 * 3600);
        assert_eq!(validity.end, -399 * 365 * 24 * 3600);
    }

    #[test]
    fn test_entity_temporal_serialization() {
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test Entity".to_string(),
            "TEST".to_string(),
            0.8,
        )
        .with_temporal_validity(100, 200);

        let json = serde_json::to_string(&entity).unwrap();
        let deserialized: Entity = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "Test Entity");
        assert!(deserialized.temporal_validity.is_some());

        let validity = deserialized.temporal_validity.unwrap();
        assert_eq!(validity.start, 100);
        assert_eq!(validity.end, 200);
    }

    #[test]
    fn test_relationship_with_temporal_type() {
        let rel = Relationship::new(
            EntityId::new("socrates".to_string()),
            EntityId::new("plato".to_string()),
            "TAUGHT".to_string(),
            0.9,
        )
        .with_temporal_type(crate::graph::temporal::TemporalRelationType::Caused)
        .with_temporal_range(100, 200);

        assert!(rel.temporal_type.is_some());
        assert!(rel.temporal_range.is_some());
        assert!(rel.causal_strength.is_some());

        let temporal_type = rel.temporal_type.unwrap();
        assert_eq!(temporal_type, crate::graph::temporal::TemporalRelationType::Caused);

        // Auto-set causal strength for causal types
        let strength = rel.causal_strength.unwrap();
        assert_eq!(strength, 0.9); // Default strength for Caused
    }

    #[test]
    fn test_relationship_with_causal_strength() {
        let rel = Relationship::new(
            EntityId::new("a".to_string()),
            EntityId::new("b".to_string()),
            "INFLUENCED".to_string(),
            0.8,
        )
        .with_temporal_type(crate::graph::temporal::TemporalRelationType::Enabled)
        .with_causal_strength(0.75);

        assert!(rel.causal_strength.is_some());
        assert_eq!(rel.causal_strength.unwrap(), 0.75);
    }

    #[test]
    fn test_relationship_temporal_serialization() {
        let rel = Relationship::new(
            EntityId::new("source".to_string()),
            EntityId::new("target".to_string()),
            "CAUSED".to_string(),
            0.9,
        )
        .with_temporal_type(crate::graph::temporal::TemporalRelationType::Caused)
        .with_temporal_range(100, 200)
        .with_causal_strength(0.95);

        let json = serde_json::to_string(&rel).unwrap();
        let deserialized: Relationship = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.relation_type, "CAUSED");
        assert!(deserialized.temporal_type.is_some());
        assert!(deserialized.temporal_range.is_some());
        assert!(deserialized.causal_strength.is_some());

        let temporal_type = deserialized.temporal_type.unwrap();
        assert_eq!(temporal_type, crate::graph::temporal::TemporalRelationType::Caused);

        let range = deserialized.temporal_range.unwrap();
        assert_eq!(range.start, 100);
        assert_eq!(range.end, 200);

        assert_eq!(deserialized.causal_strength.unwrap(), 0.95);
    }

    #[test]
    fn test_entity_backward_compatibility() {
        // Test that entities without temporal fields still work
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test".to_string(),
            "TEST".to_string(),
            0.9,
        );

        assert!(entity.first_mentioned.is_none());
        assert!(entity.last_mentioned.is_none());
        assert!(entity.temporal_validity.is_none());

        // Serialization should skip None fields
        let json = serde_json::to_string(&entity).unwrap();
        assert!(!json.contains("first_mentioned"));
        assert!(!json.contains("last_mentioned"));
        assert!(!json.contains("temporal_validity"));
    }

    #[test]
    fn test_relationship_backward_compatibility() {
        // Test that relationships without temporal fields still work
        let rel = Relationship::new(
            EntityId::new("a".to_string()),
            EntityId::new("b".to_string()),
            "RELATED_TO".to_string(),
            0.8,
        );

        assert!(rel.temporal_type.is_none());
        assert!(rel.temporal_range.is_none());
        assert!(rel.causal_strength.is_none());

        // Serialization should skip None fields
        let json = serde_json::to_string(&rel).unwrap();
        assert!(!json.contains("temporal_type"));
        assert!(!json.contains("temporal_range"));
        assert!(!json.contains("causal_strength"));
    }
}
