//! Collection processing pipeline and corpus-level analysis

use crate::core::{Result, Entity};
use crate::corpus::document_manager::{DocumentCollection, DocumentMetadata};
use crate::corpus::knowledge_graph::CorpusKnowledgeGraph;
use crate::lightrag::graph_indexer::GraphIndexer;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct ProcessingPipeline {
    pub entity_extraction: bool,
    pub relation_extraction: bool,
    pub concept_extraction: bool,
    pub cross_document_linking: bool,
    pub quality_filtering: bool,
    pub batch_size: usize,
    pub parallel_processing: bool,
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self {
            entity_extraction: true,
            relation_extraction: true,
            concept_extraction: true,
            cross_document_linking: true,
            quality_filtering: true,
            batch_size: 10,
            parallel_processing: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusStats {
    pub documents_processed: usize,
    pub total_entities_extracted: usize,
    pub total_relations_extracted: usize,
    pub total_concepts_extracted: usize,
    pub unique_entities_after_linking: usize,
    pub cross_document_entity_ratio: f32,
    pub processing_time_ms: u64,
    pub avg_entities_per_document: f32,
    pub avg_relations_per_document: f32,
    pub quality_scores: QualityMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub entity_confidence_avg: f32,
    pub relation_confidence_avg: f32,
    pub concept_coherence_avg: f32,
    pub cross_document_consistency: f32,
    pub overall_quality_score: f32,
}

impl CorpusStats {
    pub fn update_from_processing(&mut self, collection: &DocumentCollection, knowledge_graph: &CorpusKnowledgeGraph) {
        self.documents_processed = collection.documents.len();
        self.unique_entities_after_linking = knowledge_graph.global_entities.len();

        // Calculate ratios
        if self.total_entities_extracted > 0 {
            self.cross_document_entity_ratio = knowledge_graph.stats.cross_document_entities as f32 / self.total_entities_extracted as f32;
        }

        if self.documents_processed > 0 {
            self.avg_entities_per_document = self.total_entities_extracted as f32 / self.documents_processed as f32;
            self.avg_relations_per_document = self.total_relations_extracted as f32 / self.documents_processed as f32;
        }

        // Update quality metrics
        self.update_quality_metrics(knowledge_graph);
    }

    fn update_quality_metrics(&mut self, knowledge_graph: &CorpusKnowledgeGraph) {
        // Calculate average confidence scores
        let entity_confidences: Vec<f32> = knowledge_graph.global_entities.values()
            .map(|e| e.confidence_score)
            .collect();

        if !entity_confidences.is_empty() {
            self.quality_scores.entity_confidence_avg = entity_confidences.iter().sum::<f32>() / entity_confidences.len() as f32;
        }

        let relation_confidences: Vec<f32> = knowledge_graph.global_relations.values()
            .map(|r| r.confidence)
            .collect();

        if !relation_confidences.is_empty() {
            self.quality_scores.relation_confidence_avg = relation_confidences.iter().sum::<f32>() / relation_confidences.len() as f32;
        }

        // Calculate cross-document consistency
        self.quality_scores.cross_document_consistency = self.cross_document_entity_ratio;

        // Overall quality score (weighted combination)
        self.quality_scores.overall_quality_score =
            self.quality_scores.entity_confidence_avg * 0.3 +
            self.quality_scores.relation_confidence_avg * 0.3 +
            self.quality_scores.cross_document_consistency * 0.4;
    }

    pub fn print(&self) {
        tracing::info!("Corpus Processing Statistics");

        tracing::info!(
            documents_processed = self.documents_processed,
            total_entities = self.total_entities_extracted,
            total_relations = self.total_relations_extracted,
            total_concepts = self.total_concepts_extracted,
            unique_entities_after_linking = self.unique_entities_after_linking,
            cross_document_entity_ratio = format!("{:.1}%", self.cross_document_entity_ratio * 100.0),
            avg_entities_per_doc = format!("{:.1}", self.avg_entities_per_document),
            avg_relations_per_doc = format!("{:.1}", self.avg_relations_per_document),
            processing_time_s = format!("{:.1}s", self.processing_time_ms as f32 / 1000.0),
            "Corpus statistics"
        );

        tracing::info!(
            entity_confidence_avg = format!("{:.1}%", self.quality_scores.entity_confidence_avg * 100.0),
            relation_confidence_avg = format!("{:.1}%", self.quality_scores.relation_confidence_avg * 100.0),
            cross_document_consistency = format!("{:.1}%", self.quality_scores.cross_document_consistency * 100.0),
            overall_quality_score = format!("{:.1}%", self.quality_scores.overall_quality_score * 100.0),
            "Quality metrics"
        );
    }
}

pub struct CollectionProcessor {
    pipeline: ProcessingPipeline,
    graph_indexer: GraphIndexer,
    stats: CorpusStats,
}

impl CollectionProcessor {
    pub fn new() -> Result<Self> {
        let entity_types = vec!["person".to_string(), "organization".to_string(), "location".to_string(), "other".to_string()];
        Ok(Self {
            pipeline: ProcessingPipeline::default(),
            graph_indexer: GraphIndexer::new(entity_types, 3)?,
            stats: CorpusStats::default(),
        })
    }

    pub fn with_pipeline(pipeline: ProcessingPipeline) -> Result<Self> {
        let entity_types = vec!["person".to_string(), "organization".to_string(), "location".to_string(), "other".to_string()];
        Ok(Self {
            pipeline,
            graph_indexer: GraphIndexer::new(entity_types, 3)?,
            stats: CorpusStats::default(),
        })
    }

    /// Extract entities from all documents in the collection
    pub async fn extract_all_entities(&mut self, collection: &DocumentCollection) -> Result<HashMap<String, Vec<Entity>>> {
        tracing::info!(pipeline = ?self.pipeline, "Extracting entities from all documents");

        let start_time = std::time::Instant::now();
        let mut document_entities = HashMap::new();
        let mut total_entities = 0;

        // Process documents in batches
        let documents: Vec<&DocumentMetadata> = collection.documents.values().collect();
        let chunks = documents.chunks(self.pipeline.batch_size);

        for (batch_idx, chunk) in chunks.enumerate() {
            tracing::debug!(
                batch_number = batch_idx + 1,
                document_count = chunk.len(),
                "Processing batch"
            );

            if self.pipeline.parallel_processing {
                // Process batch in parallel (simulated for now)
                for doc in chunk {
                    let entities = self.extract_document_entities_internal(doc).await?;
                    total_entities += entities.len();
                    document_entities.insert(doc.id.clone(), entities);
                }
            } else {
                // Sequential processing
                for doc in chunk {
                    let entities = self.extract_document_entities_internal(doc).await?;
                    total_entities += entities.len();
                    document_entities.insert(doc.id.clone(), entities);
                }
            }
        }

        // Update statistics
        let processing_time = start_time.elapsed();
        self.stats.documents_processed = collection.documents.len();
        self.stats.total_entities_extracted = total_entities;
        self.stats.total_relations_extracted = 0; // Not tracking relations in this phase
        self.stats.total_concepts_extracted = 0; // Not tracking concepts in this phase
        self.stats.processing_time_ms = processing_time.as_millis() as u64;

        tracing::info!(
            total_entities = total_entities,
            processing_time_s = format!("{:.1}s", processing_time.as_secs_f32()),
            avg_entities_per_doc = format!("{:.1}", total_entities as f32 / collection.documents.len() as f32),
            "Entity extraction complete"
        );

        Ok(document_entities)
    }

    /// Extract entities from a single document
    pub async fn extract_document_entities(&mut self, document: &DocumentMetadata) -> Result<Vec<Entity>> {
        self.extract_document_entities_internal(document).await
    }

    /// Internal entity extraction implementation
    async fn extract_document_entities_internal(&mut self, document: &DocumentMetadata) -> Result<Vec<Entity>> {
        tracing::debug!(document_title = %document.title, "Processing document");

        // Use LightRAG graph indexer for entity extraction
        let extraction_result = self.graph_indexer.extract_from_text(&document.content)?;

        let mut entities = Vec::new();

        // Convert extracted entities to corpus-level entities
        for entity in extraction_result.entities {
            let corpus_entity = Entity {
                id: format!("{}_{}", document.id, entity.id).into(),
                name: entity.name,
                entity_type: entity.entity_type,
                confidence: entity.confidence,
                mentions: Vec::new(),
                embedding: None,
            };

            entities.push(corpus_entity);
        }

        // Apply quality filtering if enabled
        if self.pipeline.quality_filtering {
            entities = self.apply_quality_filter(entities);
        }

        tracing::debug!(entity_count = entities.len(), "Extracted entities");

        Ok(entities)
    }

    /// Apply quality filtering to extracted entities
    fn apply_quality_filter(&self, entities: Vec<Entity>) -> Vec<Entity> {
        let min_confidence = 0.3; // Minimum confidence threshold
        let min_name_length = 2; // Minimum name length

        entities.into_iter()
            .filter(|entity| {
                entity.confidence >= min_confidence &&
                entity.name.len() >= min_name_length &&
                !entity.name.trim().is_empty()
            })
            .collect()
    }

    /// Extract concepts from document collection
    pub async fn extract_concepts(&mut self, collection: &DocumentCollection) -> Result<Vec<ConceptCluster>> {
        tracing::info!("Extracting concepts from collection");

        let mut concepts = Vec::new();

        // Extract concepts from each document
        for doc in collection.documents.values() {
            let doc_concepts = self.extract_document_concepts(doc).await?;
            concepts.extend(doc_concepts);
        }

        // Cluster similar concepts
        let clustered_concepts = self.cluster_concepts(concepts).await?;

        self.stats.total_concepts_extracted = clustered_concepts.len();
        tracing::info!(cluster_count = clustered_concepts.len(), "Extracted concept clusters");

        Ok(clustered_concepts)
    }

    /// Extract concepts from a single document
    async fn extract_document_concepts(&self, document: &DocumentMetadata) -> Result<Vec<DocumentConcept>> {
        // Simple concept extraction based on noun phrases and key terms
        let text = &document.content;
        let mut concepts = Vec::new();

        // Extract common patterns that might be concepts
        let concept_patterns = [
            r"\b[A-Z][a-z]+ [a-z]+\b",  // Title case phrases
            r"\b[A-Z][A-Z]+ [A-Z]+\b",  // Acronym phrases
            r"\b\w+ system\b",          // System concepts
            r"\b\w+ process\b",         // Process concepts
            r"\b\w+ method\b",          // Method concepts
        ];

        let mut concept_id = 0;
        for pattern in &concept_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for mat in regex.find_iter(text) {
                    let concept_text = mat.as_str().to_string();

                    concepts.push(DocumentConcept {
                        id: format!("concept_{}_{}", document.id, concept_id),
                        text: concept_text,
                        document_id: document.id.clone(),
                        frequency: 1, // Would calculate actual frequency
                        context: text.chars().skip(mat.start().saturating_sub(50))
                                     .take(100).collect(),
                        confidence: 0.7,
                    });

                    concept_id += 1;
                }
            }
        }

        Ok(concepts)
    }

    /// Cluster similar concepts across documents
    async fn cluster_concepts(&self, concepts: Vec<DocumentConcept>) -> Result<Vec<ConceptCluster>> {
        let mut clusters = Vec::new();
        let mut concept_groups: HashMap<String, Vec<DocumentConcept>> = HashMap::new();

        // Simple clustering by exact text match (could be enhanced with semantic similarity)
        for concept in concepts {
            let normalized_text = concept.text.to_lowercase().trim().to_string();
            concept_groups.entry(normalized_text).or_default().push(concept);
        }

        // Create clusters from groups
        let mut cluster_id = 0;
        for (text, group_concepts) in concept_groups {
            if !group_concepts.is_empty() {
                let cluster = ConceptCluster {
                    id: format!("cluster_{cluster_id}"),
                    canonical_text: text,
                    member_concepts: group_concepts,
                    document_frequency: 1, // Would calculate actual frequency
                    total_mentions: 1,
                    confidence: 0.8,
                };

                clusters.push(cluster);
                cluster_id += 1;
            }
        }

        Ok(clusters)
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &CorpusStats {
        &self.stats
    }

    /// Configure processing pipeline
    pub fn set_pipeline(&mut self, pipeline: ProcessingPipeline) {
        self.pipeline = pipeline;
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CorpusStats::default();
    }
}

#[derive(Debug, Clone)]
pub struct DocumentConcept {
    pub id: String,
    pub text: String,
    pub document_id: String,
    pub frequency: usize,
    pub context: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ConceptCluster {
    pub id: String,
    pub canonical_text: String,
    pub member_concepts: Vec<DocumentConcept>,
    pub document_frequency: usize,
    pub total_mentions: usize,
    pub confidence: f32,
}