use crate::{
    core::{Document, Entity, KnowledgeGraph, Relationship},
    entity::EntityExtractor,
    text::TextProcessor,
    Result,
};

#[cfg(feature = "parallel-processing")]
use rayon::prelude::*;

use std::collections::HashMap;

// Incremental updates require async feature
#[cfg(feature = "async")]
pub mod incremental;

pub mod analytics;
pub mod embeddings;
pub mod temporal;
pub mod traversal;

// PageRank module is only available when the feature is enabled
#[cfg(feature = "pagerank")]
pub mod pagerank;

// Leiden community detection (feature-gated)
#[cfg(feature = "leiden")]
pub mod leiden;

#[cfg(feature = "async")]
pub use incremental::{
    ConflictStrategy, IncrementalGraphManager, IncrementalStatistics,
};

pub use analytics::{
    Community, CentralityScores, Path, GraphAnalytics,
};

pub use embeddings::{
    EmbeddingConfig, EmbeddingGraph, Node2Vec, GraphSAGE,
    GraphSAGEConfig, Aggregator,
};

pub use temporal::{
    TemporalGraph, TemporalEdge, Snapshot, TemporalQuery,
    TemporalAnalytics, EvolutionMetrics,
};

pub use traversal::{
    GraphTraversal, TraversalConfig, TraversalResult,
};

// PageRank exports are only available when the feature is enabled
#[cfg(feature = "pagerank")]
pub use pagerank::{MultiModalScores, PageRankConfig, PersonalizedPageRank, ScoreWeights};

// Leiden exports are only available when the feature is enabled
#[cfg(feature = "leiden")]
pub use leiden::{
    EntityMetadata, HierarchicalCommunities, LeidenConfig, LeidenCommunityDetector,
};

/// Graph builder for constructing knowledge graphs from documents
pub struct GraphBuilder {
    text_processor: TextProcessor,
    entity_extractor: EntityExtractor,
    similarity_threshold: f32,
    max_connections: usize,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        min_confidence: f32,
        similarity_threshold: f32,
        max_connections: usize,
    ) -> Result<Self> {
        Ok(Self {
            text_processor: TextProcessor::new(chunk_size, chunk_overlap)?,
            entity_extractor: EntityExtractor::new(min_confidence)?,
            similarity_threshold,
            max_connections,
        })
    }

    /// Build a knowledge graph from a collection of documents
    pub fn build_graph(&mut self, documents: Vec<Document>) -> Result<KnowledgeGraph> {
        let mut graph = KnowledgeGraph::new();

        // Process documents (in parallel if feature enabled)
        #[cfg(feature = "parallel-processing")]
        let processed_docs: Result<Vec<_>> = documents
            .into_par_iter()
            .map(|doc| self.process_document(doc))
            .collect();

        #[cfg(not(feature = "parallel-processing"))]
        let processed_docs: Result<Vec<_>> = documents
            .into_iter()
            .map(|doc| self.process_document(doc))
            .collect();

        let processed_docs = processed_docs?;

        // Add all documents and their chunks to the graph
        for (doc, chunks, entities) in processed_docs {
            let updated_doc = doc.with_chunks(chunks);

            // Add entities to the graph
            let mut entity_map = HashMap::new();
            for entity in entities {
                let node_idx = graph.add_entity(entity.clone())?;
                entity_map.insert(entity.id.clone(), node_idx);
            }

            // Extract and add relationships
            for chunk in &updated_doc.chunks {
                let chunk_entities: Vec<Entity> = chunk
                    .entities
                    .iter()
                    .filter_map(|id| graph.get_entity(id).cloned())
                    .collect();

                let relationships = self
                    .entity_extractor
                    .extract_relationships(&chunk_entities, chunk)?;

                for (source_id, target_id, relation_type) in relationships {
                    let relationship = Relationship {
                        source: source_id,
                        target: target_id,
                        relation_type,
                        confidence: 0.8, // Default confidence for extracted relationships
                        context: vec![chunk.id.clone()],
                    };

                    graph.add_relationship(relationship)?;
                }
            }

            graph.add_document(updated_doc)?;
        }

        // Build semantic similarity connections
        self.build_semantic_connections(&mut graph)?;

        Ok(graph)
    }

    /// Process a single document
    fn process_document(
        &self,
        document: Document,
    ) -> Result<(Document, Vec<crate::core::TextChunk>, Vec<Entity>)> {
        // Chunk the document
        let chunks = self.text_processor.chunk_text(&document)?;

        // Extract entities from chunks (in parallel if feature enabled)
        #[cfg(feature = "parallel-processing")]
        let entities_per_chunk: Result<Vec<_>> = chunks
            .par_iter()
            .map(|chunk| self.entity_extractor.extract_from_chunk(chunk))
            .collect();

        #[cfg(not(feature = "parallel-processing"))]
        let entities_per_chunk: Result<Vec<_>> = chunks
            .iter()
            .map(|chunk| self.entity_extractor.extract_from_chunk(chunk))
            .collect();

        let entities_per_chunk = entities_per_chunk?;

        // Flatten and deduplicate entities
        let mut all_entities = Vec::new();
        let mut entity_to_chunks = HashMap::new();

        for (chunk_idx, entities) in entities_per_chunk.into_iter().enumerate() {
            for entity in entities {
                let entity_id = entity.id.clone();

                // Track which chunks contain this entity
                entity_to_chunks
                    .entry(entity_id.clone())
                    .or_insert_with(Vec::new)
                    .push(chunk_idx);

                all_entities.push(entity);
            }
        }

        // Deduplicate entities and merge mentions
        let deduplicated_entities = self.deduplicate_and_merge_entities(all_entities)?;

        // Update chunks with entity references
        let mut updated_chunks = chunks;
        for (entity_id, chunk_indices) in entity_to_chunks {
            for &chunk_idx in &chunk_indices {
                if chunk_idx < updated_chunks.len() {
                    updated_chunks[chunk_idx].entities.push(entity_id.clone());
                }
            }
        }

        Ok((document, updated_chunks, deduplicated_entities))
    }

    /// Deduplicate entities and merge their mentions
    fn deduplicate_and_merge_entities(&self, entities: Vec<Entity>) -> Result<Vec<Entity>> {
        let mut entity_map: HashMap<String, Entity> = HashMap::new();

        for entity in entities {
            let key = format!("{}_{}", entity.entity_type, entity.name.to_lowercase());

            match entity_map.get_mut(&key) {
                Some(existing) => {
                    // Merge mentions
                    existing.mentions.extend(entity.mentions);
                    // Take the highest confidence
                    if entity.confidence > existing.confidence {
                        existing.confidence = entity.confidence;
                    }
                }
                None => {
                    entity_map.insert(key, entity);
                }
            }
        }

        Ok(entity_map.into_values().collect())
    }

    /// Build semantic similarity connections between entities
    fn build_semantic_connections(&mut self, graph: &mut KnowledgeGraph) -> Result<()> {
        let entities: Vec<Entity> = graph.entities().cloned().collect();

        // For entities with embeddings, find similar entities
        for (i, entity1) in entities.iter().enumerate() {
            if let Some(embedding1) = &entity1.embedding {
                let mut similarities = Vec::new();

                for (j, entity2) in entities.iter().enumerate() {
                    if i != j {
                        if let Some(embedding2) = &entity2.embedding {
                            let similarity = self.cosine_similarity(embedding1, embedding2);
                            if similarity > self.similarity_threshold {
                                similarities.push((j, similarity));
                            }
                        }
                    }
                }

                // Sort by similarity and take top connections
                similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                similarities.truncate(self.max_connections);

                // Add semantic similarity relationships
                for (j, similarity) in similarities {
                    let entity2 = &entities[j];
                    let relationship = Relationship {
                        source: entity1.id.clone(),
                        target: entity2.id.clone(),
                        relation_type: "SEMANTICALLY_SIMILAR".to_string(),
                        confidence: similarity,
                        context: Vec::new(),
                    };

                    graph.add_relationship(relationship)?;
                }
            }
        }

        Ok(())
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Add embeddings to entities
    pub fn add_entity_embeddings(
        &mut self,
        graph: &mut KnowledgeGraph,
        embedding_fn: impl Fn(&str) -> Result<Vec<f32>>,
    ) -> Result<()> {
        // This would typically call an embedding service
        // For now, we'll create placeholder embeddings

        for entity in graph.entities() {
            if entity.embedding.is_none() {
                let _embedding = embedding_fn(&entity.name)?;
                // Note: In a real implementation, you'd need to update the entity in the graph
                // This requires a mutable reference to the entity, which is not available here
                // You'd need to redesign the graph structure to allow updating entities
            }
        }

        Ok(())
    }

    /// Analyze graph statistics
    pub fn analyze_graph(&self, graph: &KnowledgeGraph) -> GraphStatistics {
        let entity_count = graph.entities().count();
        let document_count = graph.documents().count();
        let chunk_count = graph.chunks().count();

        let entity_types: HashMap<String, usize> =
            graph.entities().fold(HashMap::new(), |mut acc, entity| {
                *acc.entry(entity.entity_type.clone()).or_insert(0) += 1;
                acc
            });

        GraphStatistics {
            entity_count,
            document_count,
            chunk_count,
            entity_types,
            average_entities_per_chunk: if chunk_count > 0 {
                entity_count as f32 / chunk_count as f32
            } else {
                0.0
            },
        }
    }
}

/// Statistics about the constructed graph
#[derive(Debug)]
pub struct GraphStatistics {
    /// Total number of entities
    pub entity_count: usize,
    /// Total number of documents
    pub document_count: usize,
    /// Total number of chunks
    pub chunk_count: usize,
    /// Count of entities by type
    pub entity_types: HashMap<String, usize>,
    /// Average number of entities per chunk
    pub average_entities_per_chunk: f32,
}

impl GraphStatistics {
    /// Print graph statistics
    #[allow(dead_code)]
    pub fn print(&self) {
        tracing::info!("Graph Statistics:");
        tracing::info!("  Documents: {}", self.document_count);
        tracing::info!("  Chunks: {}", self.chunk_count);
        tracing::info!("  Entities: {}", self.entity_count);
        tracing::info!(
            "  Average entities per chunk: {:.2}",
            self.average_entities_per_chunk
        );
        tracing::info!("  Entity types:");
        for (entity_type, count) in &self.entity_types {
            tracing::info!("    {entity_type}: {count}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Document, DocumentId};

    #[test]
    fn test_graph_building() {
        let mut builder = GraphBuilder::new(500, 100, 0.7, 0.8, 5).unwrap();

        let documents = vec![
            Document::new(
                DocumentId::new("doc1".to_string()),
                "Test Document 1".to_string(),
                "John Smith works at Acme Corp. The company is based in New York.".to_string(),
            ),
            Document::new(
                DocumentId::new("doc2".to_string()),
                "Test Document 2".to_string(),
                "Jane Doe is a professor at MIT. She lives in Boston.".to_string(),
            ),
        ];

        let graph = builder.build_graph(documents).unwrap();
        let stats = builder.analyze_graph(&graph);

        assert!(stats.entity_count > 0);
        assert_eq!(stats.document_count, 2);
        assert!(stats.chunk_count >= 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let builder = GraphBuilder::new(500, 100, 0.7, 0.8, 5).unwrap();

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];

        assert!((builder.cosine_similarity(&vec1, &vec2) - 1.0).abs() < 0.001);
        assert!((builder.cosine_similarity(&vec1, &vec3) - 0.0).abs() < 0.001);
    }
}
