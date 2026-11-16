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

// LLM adapters require both async and llm features
#[cfg(all(feature = "async", feature = "llm"))]
pub mod llm_adapters;

// Entity extraction adapters require async feature
#[cfg(feature = "async")]
pub mod entity_adapters;

// Retrieval system adapters require async feature
#[cfg(feature = "async")]
pub mod retrieval_adapters;

// Test utilities for mock implementations
#[cfg(feature = "async")]
pub mod test_utils;

#[cfg(all(test, feature = "async"))]
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
/// ```ignore
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

    /// Builds the canonical entity ID from an entity type and a surface name.
    ///
    /// Shared by every extractor so the same entity always maps to the same ID:
    /// `lowercase(type) + "_" + lowercase(name)` with whitespace runs collapsed to
    /// a single `_`. Punctuation and articles are intentionally preserved.
    pub fn from_type_and_name(entity_type: &str, name: &str) -> Self {
        let norm = |s: &str| {
            s.to_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join("_")
        };
        Self(format!("{}_{}", norm(entity_type), norm(name)))
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

/// Prefix of the marker (`[detto da <chi>]`) that tags a relationship description as only
/// said by a character (epistemic status). Single source of truth: `said_description`
/// builds the marker from it and the merge in `add_relationship` detects it with it.
pub const SAID_MARKER_PREFIX: &str = "[detto da ";

/// True when `description` starts with the "said" marker ([`SAID_MARKER_PREFIX`]).
pub fn is_said_description(description: Option<&str>) -> bool {
    description.is_some_and(|d| d.starts_with(SAID_MARKER_PREFIX))
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
    /// Free-text description; for `OTHER` relations it keeps the original
    /// extractor label so no information is lost (see [`crate::ontology`])
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,

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
    pub fn new(source: EntityId, target: EntityId, relation_type: String, confidence: f32) -> Self {
        Self {
            description: None,
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
    pub relationship_hierarchy:
        Option<crate::graph::hierarchical_relationships::RelationshipHierarchy>,
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
    ///
    /// Idempotent upsert: if an entity with the same ID already exists, the new
    /// one is merged into it (mentions deduped by chunk/offsets, max confidence,
    /// existing name/type kept, embedding filled if missing) and the existing
    /// `NodeIndex` is returned. Chunks referenced by the mentions also get the
    /// entity ID recorded in their `entities` list.
    pub fn add_entity(&mut self, entity: Entity) -> Result<NodeIndex> {
        let entity_id = entity.id.clone();

        for mention in &entity.mentions {
            if let Some(chunk) = self.chunks.get_mut(&mention.chunk_id) {
                if !chunk.entities.contains(&entity_id) {
                    chunk.entities.push(entity_id.clone());
                }
            }
        }

        if let Some(&node_index) = self.entity_index.get(&entity_id) {
            if let Some(existing) = self.graph.node_weight_mut(node_index) {
                for m in entity.mentions {
                    let dup = existing.mentions.iter().any(|e| {
                        e.chunk_id == m.chunk_id
                            && e.start_offset == m.start_offset
                            && e.end_offset == m.end_offset
                    });
                    if !dup {
                        existing.mentions.push(m);
                    }
                }
                if entity.confidence > existing.confidence {
                    existing.confidence = entity.confidence;
                }
                if existing.embedding.is_none() {
                    existing.embedding = entity.embedding;
                }
            }
            return Ok(node_index);
        }

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

        let (source_idx, target_idx) = (*source_idx, *target_idx);

        // Upsert on (source, target, relation_type): merge instead of duplicating.
        let existing_edge = self
            .graph
            .edges_connecting(source_idx, target_idx)
            .find(|e| e.weight().relation_type == relationship.relation_type)
            .map(|e| {
                use petgraph::visit::EdgeRef;
                e.id()
            });

        if let Some(edge_id) = existing_edge {
            if let Some(existing) = self.graph.edge_weight_mut(edge_id) {
                for c in relationship.context {
                    if !existing.context.contains(&c) {
                        existing.context.push(c);
                    }
                }
                if relationship.confidence > existing.confidence {
                    existing.confidence = relationship.confidence;
                }
                if existing.embedding.is_none() {
                    existing.embedding = relationship.embedding;
                }
                // A narrated description replaces a "said" one, whatever the arrival order.
                // Known limit: said + incoming None/blank keeps the said text.
                let replace = existing.description.is_none()
                    || (is_said_description(existing.description.as_deref())
                        && relationship
                            .description
                            .as_deref()
                            .is_some_and(|d| !d.trim().is_empty())
                        && !is_said_description(relationship.description.as_deref()));
                if replace {
                    existing.description = relationship.description;
                }
            }
            return Ok(());
        }

        self.graph.add_edge(source_idx, target_idx, relationship);
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
        let json_data = json::parse(&json_str).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse JSON: {e}"),
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
                            chunk_id: ChunkId::new(
                                mention_obj["chunk_id"].as_str().unwrap_or("").to_string(),
                            ),
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
                    description: rel_obj["description"].as_str().map(str::to_string),
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
                let document_id =
                    DocumentId::new(chunk_obj["document_id"].as_str().unwrap_or("").to_string());
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
            let mut rel_obj = json::object! {
                "source_id" => relationship.source.to_string(),
                "target_id" => relationship.target.to_string(),
                "relation_type" => relationship.relation_type.clone(),
                "confidence" => relationship.confidence,
                "context_chunks" => relationship.context.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
            };
            if let Some(description) = &relationship.description {
                rel_obj["description"] = description.as_str().into();
            }
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
            let entities_list: Vec<String> = chunk.entities.iter().map(|e| e.to_string()).collect();
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
        #[cfg(feature = "tracing")]
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
            for ((row, col), val) in row_indices.into_iter().zip(col_indices).zip(values) {
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

    /// Merge entities that share the same normalized name but differ in type.
    ///
    /// Extractors derive ids as `<type>_<name>`, so the same surface form tagged with
    /// different types (`person_burattino`, `object_burattino`, ...) ends up as several
    /// nodes. Names are compared trimmed, whitespace-collapsed and lowercased; different
    /// names are never merged. Per group the entity with the most mentions wins (tie:
    /// higher confidence, then first seen) and keeps its id, name and type; mentions are
    /// unioned (deduped by chunk/offsets), confidence is the max. Relationships of the
    /// absorbed entities are re-pointed to the winner; self-loops created by the merge
    /// are dropped and identical `(source, target, relation_type)` edges are merged
    /// (contexts unioned, first non-empty description kept). Chunk `entities` lists are
    /// remapped too. Returns the number of entities absorbed.
    ///
    /// Run it before entity embeddings are computed: the surviving entity keeps its own
    /// embedding and absorbed ones are discarded.
    pub fn merge_same_name_entities(&mut self) -> usize {
        // Lowercase, collapse whitespace and drop one leading article
        // ("il babbo" == "babbo", "l'albero" == "albero"), never the whole name.
        const ARTICLES: &[&str] = &[
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "the", "a", "an",
        ];
        const ELIDED: &[&str] = &["l'", "l’", "un'", "un’"];
        let normalize = |n: &str| {
            let name = n
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            if let Some((first, rest)) = name.split_once(' ') {
                if ARTICLES.contains(&first) {
                    return rest.to_string();
                }
            }
            for prefix in ELIDED {
                if let Some(rest) = name.strip_prefix(prefix) {
                    if !rest.is_empty() {
                        return rest.to_string();
                    }
                }
            }
            name
        };

        let entities: Vec<Entity> = self.graph.node_weights().cloned().collect();
        let mut groups: IndexMap<String, Vec<usize>> = IndexMap::new();
        for (i, e) in entities.iter().enumerate() {
            groups.entry(normalize(&e.name)).or_default().push(i);
        }
        if groups.values().all(|g| g.len() < 2) {
            return 0;
        }

        let mut remap: HashMap<EntityId, EntityId> = HashMap::new();
        let mut merged_by_winner: HashMap<usize, Entity> = HashMap::new();
        let mut absorbed_idx: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for group in groups.values().filter(|g| g.len() > 1) {
            let mut winner = group[0];
            for &i in &group[1..] {
                let (w, c) = (&entities[winner], &entities[i]);
                if (c.mentions.len(), c.confidence) > (w.mentions.len(), w.confidence) {
                    winner = i;
                }
            }
            let mut merged = entities[winner].clone();
            for &i in group.iter().filter(|&&i| i != winner) {
                let loser = &entities[i];
                for m in &loser.mentions {
                    let dup = merged.mentions.iter().any(|e| {
                        e.chunk_id == m.chunk_id
                            && e.start_offset == m.start_offset
                            && e.end_offset == m.end_offset
                    });
                    if !dup {
                        merged.mentions.push(m.clone());
                    }
                }
                merged.confidence = merged.confidence.max(loser.confidence);
                if merged.embedding.is_none() {
                    merged.embedding = loser.embedding.clone();
                }
                remap.insert(loser.id.clone(), merged.id.clone());
                absorbed_idx.insert(i);
            }
            merged_by_winner.insert(winner, merged);
        }

        let relationships: Vec<Relationship> = self.graph.edge_weights().cloned().collect();

        for chunk in self.chunks.values_mut() {
            let mut remapped: Vec<EntityId> = Vec::with_capacity(chunk.entities.len());
            for id in chunk.entities.drain(..) {
                let id = remap.get(&id).cloned().unwrap_or(id);
                if !remapped.contains(&id) {
                    remapped.push(id);
                }
            }
            chunk.entities = remapped;
        }

        self.clear_entities_and_relationships();
        for (i, entity) in entities.into_iter().enumerate() {
            if absorbed_idx.contains(&i) {
                continue;
            }
            let entity = merged_by_winner.remove(&i).unwrap_or(entity);
            // IDs are unique per node, so this never merges again.
            let _ = self.add_entity(entity);
        }
        for mut rel in relationships {
            if let Some(to) = remap.get(&rel.source) {
                rel.source = to.clone();
            }
            if let Some(to) = remap.get(&rel.target) {
                rel.target = to.clone();
            }
            if rel.source == rel.target {
                continue;
            }
            let _ = self.add_relationship(rel);
        }
        absorbed_idx.len()
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
            .filter(|c| {
                relationship
                    .relation_type
                    .to_lowercase()
                    .contains(&c.to_lowercase())
            })
            .count() as f32
            * 0.15; // 15% boost per matching concept

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
            if let (Some(&src), Some(&tgt)) = (node_map.get(&rel.source), node_map.get(&rel.target))
            {
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
    /// * `llm_client` - Optional LLM client for generating cluster summaries
    ///
    /// # Returns
    /// Result containing the built hierarchy or an error
    ///
    /// # Example
    /// ```ignore
    /// use graphrag_core::{KnowledgeGraph, llm::{LlmClient, LlmConfig}};
    ///
    /// let mut graph = KnowledgeGraph::new();
    /// // ... build graph ...
    ///
    /// let llm = LlmClient::new(LlmConfig { enabled: true, ..LlmConfig::default() });
    /// graph.build_relationship_hierarchy(3, Some(llm)).await.unwrap();
    /// ```
    #[cfg(feature = "async")]
    pub async fn build_relationship_hierarchy(
        &mut self,
        num_levels: usize,
        llm_client: Option<crate::llm::LlmClient>,
    ) -> Result<()> {
        use crate::graph::hierarchical_relationships::HierarchyBuilder;

        let builder = HierarchyBuilder::from_graph(self).with_num_levels(num_levels);

        let builder = if let Some(client) = llm_client {
            builder.with_llm_client(client)
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
        assert_eq!(
            temporal_type,
            crate::graph::temporal::TemporalRelationType::Caused
        );

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
        assert_eq!(
            temporal_type,
            crate::graph::temporal::TemporalRelationType::Caused
        );

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

    // ---- graph consistency regression tests ----

    fn mention(chunk: &str, start: usize, end: usize) -> EntityMention {
        EntityMention {
            chunk_id: ChunkId::new(chunk.to_string()),
            start_offset: start,
            end_offset: end,
            confidence: 0.8,
        }
    }

    fn ent(id: &str, mentions: Vec<EntityMention>, confidence: f32) -> Entity {
        Entity::new(
            EntityId::new(id.to_string()),
            id.to_string(),
            "person".to_string(),
            confidence,
        )
        .with_mentions(mentions)
    }

    #[test]
    fn test_add_entity_is_idempotent_upsert() {
        let mut kg = KnowledgeGraph::new();
        let a = kg
            .add_entity(ent("person_x", vec![mention("c1", 0, 3)], 0.5))
            .unwrap();
        let b = kg
            .add_entity(ent(
                "person_x",
                vec![mention("c1", 0, 3), mention("c2", 5, 8)],
                0.9,
            ))
            .unwrap();

        assert_eq!(a, b);
        assert_eq!(kg.entity_count(), 1);
        let e = kg
            .get_entity(&EntityId::new("person_x".to_string()))
            .unwrap();
        assert_eq!(e.mentions.len(), 2, "mentions merged and deduped");
        assert_eq!(e.confidence, 0.9, "max confidence kept");
    }

    /// Merge two relationships (a, b, KNOWS) with the given descriptions; returns the result.
    fn merge_descriptions(first: Option<&str>, second: Option<&str>) -> Relationship {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(ent("a", vec![], 0.5)).unwrap();
        kg.add_entity(ent("b", vec![], 0.5)).unwrap();
        for (d, ctx, conf) in [(first, "c1", 0.4), (second, "c2", 0.9)] {
            let mut r = Relationship::new(
                EntityId::new("a".to_string()),
                EntityId::new("b".to_string()),
                "KNOWS".to_string(),
                conf,
            );
            r.context = vec![ChunkId::new(ctx.to_string())];
            r.description = d.map(str::to_string);
            kg.add_relationship(r).unwrap();
        }
        assert_eq!(kg.relationship_count(), 1);
        kg.get_all_relationships()[0].clone()
    }

    #[test]
    fn test_merge_narrated_replaces_said_description() {
        let said = "[detto da Volpe] x";
        let cases = [
            (Some(said), Some("narrato"), Some("narrato")),
            (Some("narrato"), Some(said), Some("narrato")),
            (Some(said), Some("[detto da Gatto] y"), Some(said)),
            (Some("uno"), Some("due"), Some("uno")),
            (Some(said), None, Some(said)),
            (Some(said), Some("  "), Some(said)),
            (None, Some(said), Some(said)),
            (None, Some("narrato"), Some("narrato")),
        ];
        for (first, second, expected) in cases {
            let r = merge_descriptions(first, second);
            assert_eq!(r.description.as_deref(), expected, "{first:?} + {second:?}");
            let ctx: Vec<_> = r.context.iter().map(|c| c.0.as_str()).collect();
            assert_eq!(ctx, vec!["c1", "c2"]);
            assert_eq!(r.confidence, 0.9);
        }
    }

    #[test]
    fn test_add_relationship_is_upsert_merging_context() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(ent("a", vec![], 0.5)).unwrap();
        kg.add_entity(ent("b", vec![], 0.5)).unwrap();
        let mk = |ctx: &[&str], conf: f32, ty: &str| {
            let mut r = Relationship::new(
                EntityId::new("a".to_string()),
                EntityId::new("b".to_string()),
                ty.to_string(),
                conf,
            );
            r.context = ctx.iter().map(|c| ChunkId::new(c.to_string())).collect();
            r
        };
        kg.add_relationship(mk(&["c1", "c2"], 0.4, "KNOWS"))
            .unwrap();
        kg.add_relationship(mk(&["c2", "c3"], 0.9, "KNOWS"))
            .unwrap();
        kg.add_relationship(mk(&["c1"], 0.5, "LOVES")).unwrap();

        assert_eq!(kg.relationship_count(), 2, "different type stays separate");
        let knows = kg
            .get_all_relationships()
            .into_iter()
            .find(|r| r.relation_type == "KNOWS")
            .unwrap();
        let ctx: Vec<_> = knows.context.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(ctx, vec!["c1", "c2", "c3"]);
        assert_eq!(knows.confidence, 0.9);
    }

    #[test]
    fn test_add_entity_records_id_in_chunk_entities() {
        let mut kg = KnowledgeGraph::new();
        kg.add_chunk(TextChunk::new(
            ChunkId::new("c1".to_string()),
            DocumentId::new("d".to_string()),
            "text".to_string(),
            0,
            4,
        ))
        .unwrap();
        kg.add_entity(ent("person_x", vec![mention("c1", 0, 2)], 0.5))
            .unwrap();
        kg.add_entity(ent("person_x", vec![mention("c1", 2, 4)], 0.5))
            .unwrap();
        let chunk = kg.get_chunk(&ChunkId::new("c1".to_string())).unwrap();
        assert_eq!(chunk.entities, vec![EntityId::new("person_x".to_string())]);
    }

    #[test]
    fn test_json_roundtrip_preserves_counts_without_duplication() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity(ent("a", vec![mention("c1", 0, 1)], 0.5))
            .unwrap();
        kg.add_entity(ent("b", vec![mention("c1", 2, 3)], 0.5))
            .unwrap();
        let mut r = Relationship::new(
            EntityId::new("a".to_string()),
            EntityId::new("b".to_string()),
            "KNOWS".to_string(),
            0.7,
        );
        r.context = vec![ChunkId::new("c1".to_string())];
        kg.add_relationship(r).unwrap();

        let path = std::env::temp_dir().join(format!(
            "graphrag_roundtrip_{}_{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let path_str = path.to_str().unwrap();
        kg.save_to_json(path_str).unwrap();
        let loaded = KnowledgeGraph::load_from_json(path_str).unwrap();
        // Save the reloaded graph and reload again: still stable.
        loaded.save_to_json(path_str).unwrap();
        let loaded2 = KnowledgeGraph::load_from_json(path_str).unwrap();
        let _ = std::fs::remove_file(&path);

        for g in [&loaded, &loaded2] {
            assert_eq!(g.entity_count(), 2);
            assert_eq!(g.relationship_count(), 1);
            assert_eq!(g.get_all_relationships()[0].context.len(), 1);
        }
    }

    #[test]
    fn test_entity_id_from_type_and_name_is_canonical() {
        assert_eq!(
            EntityId::from_type_and_name("PERSON", "  John   Doe ").0,
            "person_john_doe"
        );
        // Punctuation and articles are preserved.
        assert_eq!(
            EntityId::from_type_and_name("Concept", "The Good, Truth").0,
            "concept_the_good,_truth"
        );
    }
}

#[cfg(test)]
mod merge_same_name_tests {
    use super::*;

    fn ent(ty: &str, name: &str, chunk: &str, offsets: &[usize], conf: f32) -> Entity {
        let mut e = Entity::new(
            EntityId::from_type_and_name(ty, name),
            name.to_string(),
            ty.to_string(),
            conf,
        );
        e.mentions = offsets
            .iter()
            .map(|&o| EntityMention {
                chunk_id: ChunkId::new(chunk.to_string()),
                start_offset: o,
                end_offset: o + 5,
                confidence: conf,
            })
            .collect();
        e
    }

    fn graph() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        for c in ["c0", "c1"] {
            g.add_chunk(TextChunk::new(
                ChunkId::new(c.to_string()),
                DocumentId::new("d".to_string()),
                "testo".to_string(),
                0,
                5,
            ))
            .unwrap();
        }
        g
    }

    fn rel(s: &str, t: &str, ty: &str, ctx: &str) -> Relationship {
        let mut r = Relationship::new(
            EntityId::new(s.to_string()),
            EntityId::new(t.to_string()),
            ty.to_string(),
            0.8,
        );
        r.context = vec![ChunkId::new(ctx.to_string())];
        r
    }

    #[test]
    fn merges_same_name_keeps_type_with_most_mentions_and_unions_mentions() {
        let mut g = graph();
        g.add_entity(ent("OBJECT", "Burattino", "c0", &[1], 0.9))
            .unwrap();
        g.add_entity(ent("PERSON", "burattino", "c0", &[1, 20, 40], 0.7))
            .unwrap();
        g.add_entity(ent("CONCEPT", " Burattino ", "c1", &[3], 0.95))
            .unwrap();
        g.add_entity(ent("PERSON", "Geppetto", "c1", &[9], 0.9))
            .unwrap();

        assert_eq!(g.merge_same_name_entities(), 2);
        assert_eq!(g.entities().count(), 2);
        let w = g
            .get_entity(&EntityId::new("person_burattino".to_string()))
            .unwrap();
        assert_eq!(w.entity_type, "PERSON");
        // 3 own + c0@1 (dup of own c0@1, deduped) + c1@3
        assert_eq!(w.mentions.len(), 4);
        assert!((w.confidence - 0.95).abs() < 1e-6);
        assert!(g
            .get_entity(&EntityId::new("object_burattino".to_string()))
            .is_none());
        // different name untouched
        assert!(g
            .get_entity(&EntityId::new("person_geppetto".to_string()))
            .is_some());
        // chunk entity lists remapped, no stale ids
        let c0 = g.get_chunk(&ChunkId::new("c0".to_string())).unwrap();
        assert_eq!(
            c0.entities,
            vec![EntityId::new("person_burattino".to_string())]
        );
    }

    #[test]
    fn tie_keeps_first_seen_unless_confidence_higher() {
        let mut g = graph();
        g.add_entity(ent("OBJECT", "Acqua", "c0", &[1], 0.8))
            .unwrap();
        g.add_entity(ent("LOCATION", "Acqua", "c1", &[2], 0.8))
            .unwrap();
        g.merge_same_name_entities();
        assert!(g
            .get_entity(&EntityId::new("object_acqua".to_string()))
            .is_some());

        let mut g = graph();
        g.add_entity(ent("OBJECT", "Acqua", "c0", &[1], 0.8))
            .unwrap();
        g.add_entity(ent("LOCATION", "Acqua", "c1", &[2], 0.9))
            .unwrap();
        g.merge_same_name_entities();
        assert!(g
            .get_entity(&EntityId::new("location_acqua".to_string()))
            .is_some());
    }

    #[test]
    fn relationships_are_remapped_deduped_and_self_loops_dropped() {
        let mut g = graph();
        g.add_entity(ent("OBJECT", "Burattino", "c0", &[1], 0.9))
            .unwrap();
        g.add_entity(ent("PERSON", "Burattino", "c0", &[1, 20], 0.9))
            .unwrap();
        g.add_entity(ent("PERSON", "Geppetto", "c1", &[9], 0.9))
            .unwrap();
        g.add_relationship(rel("object_burattino", "person_geppetto", "KNOWS", "c0"))
            .unwrap();
        let mut with_desc = rel("person_burattino", "person_geppetto", "KNOWS", "c1");
        with_desc.description = Some("figlio".to_string());
        g.add_relationship(with_desc).unwrap();
        g.add_relationship(rel("object_burattino", "person_burattino", "SAME", "c0"))
            .unwrap();

        g.merge_same_name_entities();
        let rels: Vec<_> = g.relationships().collect();
        assert_eq!(rels.len(), 1, "self-loop dropped, duplicates merged");
        assert_eq!(rels[0].source.0, "person_burattino");
        assert_eq!(rels[0].target.0, "person_geppetto");
        assert_eq!(rels[0].context.len(), 2);
        assert_eq!(rels[0].description.as_deref(), Some("figlio"));
    }

    #[test]
    fn no_op_without_same_name_entities() {
        let mut g = graph();
        g.add_entity(ent("PERSON", "Geppetto", "c0", &[1], 0.9))
            .unwrap();
        g.add_entity(ent("PERSON", "Pinocchio", "c1", &[2], 0.9))
            .unwrap();
        assert_eq!(g.merge_same_name_entities(), 0);
        assert_eq!(g.entities().count(), 2);
    }

    #[test]
    fn leading_article_does_not_split_same_entity() {
        let mut g = graph();
        g.add_entity(ent("OBJECT", "albero", "c0", &[1, 9], 0.9))
            .unwrap();
        g.add_entity(ent("LOCATION", "l'albero", "c1", &[2], 0.9))
            .unwrap();
        g.add_entity(ent("PERSON", "babbo", "c0", &[3, 20], 0.9))
            .unwrap();
        g.add_entity(ent("PERSON", "Il babbo", "c1", &[4], 0.9))
            .unwrap();
        g.add_entity(ent("ANIMAL", "La Volpe", "c0", &[5, 30], 0.9))
            .unwrap();
        g.add_entity(ent("ANIMAL", "Volpe", "c1", &[6], 0.9))
            .unwrap();
        assert_eq!(g.merge_same_name_entities(), 3);
        let mut names: Vec<String> = g.entities().map(|e| e.name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["La Volpe", "albero", "babbo"]);
    }

    #[test]
    fn article_alone_or_inside_name_is_kept() {
        let mut g = graph();
        g.add_entity(ent("PERSON", "Il Gatto", "c0", &[1], 0.9))
            .unwrap();
        g.add_entity(ent("CONCEPT", "il", "c1", &[2], 0.9)).unwrap();
        g.add_entity(ent("OBJECT", "gatto il grande", "c1", &[3], 0.9))
            .unwrap();
        assert_eq!(g.merge_same_name_entities(), 0);
        assert_eq!(g.entities().count(), 3);
    }
}
