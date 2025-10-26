//! # Multi-Document Processing Module
//!
//! This module provides corpus-level analysis and cross-document entity resolution
//! for GraphRAG-RS, implementing the Multi-Document Processing capabilities outlined
//! in Phase 3 of the implementation plan.

pub mod document_manager;
pub mod entity_linker;
pub mod knowledge_graph;
pub mod collection_processor;

pub use document_manager::{DocumentManager, DocumentMetadata, DocumentCollection};
pub use entity_linker::{CrossDocumentEntityLinker, EntityCluster, LinkingStrategy};
pub use knowledge_graph::{CorpusKnowledgeGraph, GlobalEntity, GlobalRelation};
pub use collection_processor::{CollectionProcessor, ProcessingPipeline, CorpusStats};

use crate::core::Result;
use std::path::Path;

/// Main corpus coordinator that orchestrates multi-document processing
pub struct CorpusProcessor {
    document_manager: DocumentManager,
    entity_linker: CrossDocumentEntityLinker,
    knowledge_graph: CorpusKnowledgeGraph,
    collection_processor: CollectionProcessor,
    stats: CorpusStats,
}

impl CorpusProcessor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            document_manager: DocumentManager::new()?,
            entity_linker: CrossDocumentEntityLinker::new()?,
            knowledge_graph: CorpusKnowledgeGraph::new()?,
            collection_processor: CollectionProcessor::new()?,
            stats: CorpusStats::default(),
        })
    }

    /// Process a complete document collection
    pub async fn process_collection(&mut self, collection_path: &Path) -> Result<CorpusProcessingResult> {
        // Load and index documents
        let collection = self.document_manager.load_collection(collection_path).await?;

        // Extract entities from all documents
        let document_entities = self.collection_processor.extract_all_entities(collection).await?;

        // Perform cross-document entity linking
        let entity_clusters = self.entity_linker.link_entities(document_entities).await?;

        // Build corpus-level knowledge graph
        let global_graph = self.knowledge_graph.build_from_clusters(entity_clusters, collection).await?;

        // Update statistics
        self.stats.update_from_processing(collection, &global_graph);

        Ok(CorpusProcessingResult {
            documents_processed: collection.documents.len(),
            entities_linked: global_graph.global_entities.len(),
            relations_identified: global_graph.global_relations.len(),
            knowledge_graph: global_graph,
        })
    }

    /// Add a single document to an existing corpus
    pub async fn add_document(&mut self, document_path: &Path) -> Result<()> {
        let metadata = self.document_manager.add_document(document_path).await?;

        // Extract entities from new document
        let entities = self.collection_processor.extract_document_entities(&metadata).await?;

        // Link with existing entities
        self.entity_linker.link_new_document_entities(entities).await?;

        // Update knowledge graph incrementally
        self.knowledge_graph.integrate_new_document(&metadata).await?;

        Ok(())
    }

    /// Query the corpus-level knowledge graph
    pub async fn query_corpus(&self, query: &str) -> Result<Vec<GlobalEntity>> {
        self.knowledge_graph.query(query).await
    }

    /// Get corpus-level statistics
    pub fn get_stats(&self) -> &CorpusStats {
        &self.stats
    }

    /// Export corpus knowledge graph
    pub async fn export_knowledge_graph(&self, output_path: &Path) -> Result<()> {
        self.knowledge_graph.export(output_path).await
    }
}

/// Result of corpus processing
#[derive(Debug, Clone)]
pub struct CorpusProcessingResult {
    pub documents_processed: usize,
    pub entities_linked: usize,
    pub relations_identified: usize,
    pub knowledge_graph: CorpusKnowledgeGraph,
}

impl CorpusProcessingResult {
    pub fn print_summary(&self) {
        tracing::info!(
            documents_processed = self.documents_processed,
            entities_linked = self.entities_linked,
            relations_identified = self.relations_identified,
            clustering_efficiency = format!("{:.1}%", self.get_clustering_efficiency() * 100.0),
            "Corpus processing summary"
        );
    }

    fn get_clustering_efficiency(&self) -> f32 {
        if self.documents_processed == 0 {
            return 0.0;
        }

        // Rough estimate: assume average 10 entities per document
        let estimated_raw_entities = self.documents_processed * 10;
        if estimated_raw_entities == 0 {
            return 0.0;
        }

        1.0 - (self.entities_linked as f32 / estimated_raw_entities as f32)
    }
}