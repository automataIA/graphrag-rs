//! # LightRAG Core Implementation
//!
//! This module implements the LightRAG dual-level retrieval system with key-value storage
//! and incremental updates, providing 6000x token reduction compared to traditional GraphRAG.

pub mod dual_retrieval;
pub mod graph_indexer;
pub mod kv_store;
pub mod incremental_updater;

pub use dual_retrieval::{DualRetriever, RetrievalMode, FusionWeights, RetrievalResult as DualRetrievalResult};
pub use graph_indexer::{GraphIndexer, EntityReference, ConceptSummary, ExtractionResult};
pub use kv_store::{KVStore, KVData, EntityData, RelationData};
pub use incremental_updater::{IncrementalGraphUpdater, GraphUpdate, UpdateStats, ConflictStrategy};

use crate::core::{Result, GraphRAGError};

/// LightRAG configuration
#[derive(Debug, Clone)]
pub struct LightRAGConfig {
    pub retrieval_mode: RetrievalMode,
    pub max_gleaning_iterations: usize,
    pub entity_types: Vec<String>,
    pub fusion_weights: FusionWeights,
    pub incremental_enabled: bool,
    pub conflict_strategy: ConflictStrategy,
}

impl Default for LightRAGConfig {
    fn default() -> Self {
        Self {
            retrieval_mode: RetrievalMode::Hybrid,
            max_gleaning_iterations: 3,
            entity_types: vec![
                "person".to_string(),
                "organization".to_string(),
                "location".to_string(),
                "event".to_string(),
                "concept".to_string(),
            ],
            fusion_weights: FusionWeights::default(),
            incremental_enabled: true,
            conflict_strategy: ConflictStrategy::Merge,
        }
    }
}

/// Main LightRAG coordinator
pub struct LightRAG {
    #[allow(dead_code)] config: LightRAGConfig,
    dual_retriever: DualRetriever,
    graph_indexer: GraphIndexer,
    incremental_updater: Option<IncrementalGraphUpdater>,
}

impl LightRAG {
    pub fn new(config: LightRAGConfig) -> Result<Self> {
        let dual_retriever = DualRetriever::new(
            config.retrieval_mode.clone(),
            config.fusion_weights.clone(),
        )?;

        let graph_indexer = GraphIndexer::new(
            config.entity_types.clone(),
            config.max_gleaning_iterations,
        )?;

        let incremental_updater = if config.incremental_enabled {
            Some(IncrementalGraphUpdater::new(config.conflict_strategy.clone())?)
        } else {
            None
        };

        Ok(Self {
            config,
            dual_retriever,
            graph_indexer,
            incremental_updater,
        })
    }

    pub fn with_kv_store<T: KVStore + 'static>(mut self, store: T) -> Self {
        self.dual_retriever.set_kv_store(Box::new(store));
        self
    }

    /// Process text and build dual-level indexes
    pub fn process_text(&mut self, text: &str) -> Result<ProcessingResult> {
        // Extract entities and relationships using graph indexer
        let extraction_result = self.graph_indexer.extract_from_text(text)?;

        // Build dual-level indexes
        self.dual_retriever.build_indexes(&extraction_result)?;

        Ok(ProcessingResult {
            entities_extracted: extraction_result.entities.len(),
            relations_extracted: 0, // Relations are implicit in concepts for now
            tokens_used: extraction_result.tokens_used,
        })
    }

    /// Query using dual-level retrieval
    pub fn query(&self, query: &str) -> Result<Vec<DualRetrievalResult>> {
        self.dual_retriever.retrieve(query)
    }

    /// Apply incremental update
    pub fn incremental_update(&mut self, new_text: &str) -> Result<UpdateStats> {
        if let Some(ref mut updater) = self.incremental_updater {
            let extraction_result = self.graph_indexer.extract_from_text(new_text)?;
            updater.apply_incremental_update(extraction_result)
        } else {
            Err(GraphRAGError::Config {
                message: "Incremental updates not enabled".to_string(),
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub entities_extracted: usize,
    pub relations_extracted: usize,
    pub tokens_used: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub content: String,
    pub score: f32,
    pub level: RetrievalLevel,
}

#[derive(Debug, Clone)]
pub enum RetrievalLevel {
    LowLevel,  // Specific entities and attributes
    HighLevel, // Broader topics and themes
}