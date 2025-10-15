//! # GraphRAG Core
//!
//! Portable core library for GraphRAG - works on both native and WASM platforms.
//!
//! This is the foundational crate that provides:
//! - Knowledge graph construction and management
//! - Entity extraction and linking
//! - Vector embeddings and similarity search
//! - Graph algorithms (PageRank, community detection)
//! - Retrieval systems (semantic, keyword, hybrid)
//! - Caching and optimization
//!
//! ## Platform Support
//!
//! - **Native**: Full feature set with optional CUDA/Metal GPU acceleration
//! - **WASM**: Browser-compatible with Voy vector search and Candle embeddings
//!
//! ## Feature Flags
//!
//! - `wasm`: Enable WASM compatibility (uses Voy instead of HNSW)
//! - `cuda`: Enable NVIDIA GPU acceleration via Candle
//! - `metal`: Enable Apple Silicon GPU acceleration
//! - `webgpu`: Enable WebGPU acceleration for browser (via Burn)
//! - `pagerank`: Enable PageRank-based retrieval
//! - `lightrag`: Enable LightRAG optimizations (6000x token reduction)
//! - `caching`: Enable intelligent LLM response caching
//!
//! ## Quick Start
//!
//! ```rust
//! use graphrag_core::{GraphRAG, Config};
//!
//! # fn example() -> graphrag_core::Result<()> {
//! let config = Config::default();
//! let mut graphrag = GraphRAG::new(config)?;
//! graphrag.initialize()?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
// Note: WASM with wasm-bindgen DOES use std, so we don't disable it

// ================================
// MODULE DECLARATIONS
// ================================

// Core modules (always available)
/// Configuration management and loading
pub mod config;
/// Core traits and types
pub mod core;
/// Text processing and chunking
pub mod text;
/// Vector operations and embeddings
pub mod vector;
/// Graph data structures and algorithms
pub mod graph;
/// Entity extraction and management
pub mod entity;
/// Retrieval strategies and implementations
pub mod retrieval;
/// Text generation and LLM interactions (async feature only)
#[cfg(feature = "async")]
pub mod generation;
/// Storage backends and persistence
#[cfg(any(feature = "memory-storage", feature = "persistent-storage", feature = "async"))]
pub mod storage;
/// Query processing and execution
pub mod query;
/// Builder pattern implementations
pub mod builder;
/// Text summarization capabilities
pub mod summarization;
/// Ollama LLM integration
pub mod ollama;
/// Natural language processing utilities
pub mod nlp;
/// Embedding generation and providers
pub mod embeddings;

// Pipeline modules
/// Data processing pipelines
pub mod pipeline;

// Advanced features (feature-gated)
#[cfg(feature = "parallel-processing")]
pub mod parallel;

#[cfg(feature = "lightrag")]
/// LightRAG dual-level retrieval optimization
pub mod lightrag;

// Utility modules
/// Reranking utilities for improving search result quality
pub mod reranking;

/// Monitoring, benchmarking, and performance tracking
pub mod monitoring;

/// Evaluation framework for query results and pipeline validation
pub mod evaluation;

/// API endpoints and handlers
#[cfg(feature = "api")]
pub mod api;

/// Inference module for model predictions
pub mod inference;

/// Multi-document corpus processing
#[cfg(feature = "corpus-processing")]
pub mod corpus;

// Feature-gated modules
#[cfg(feature = "async")]
/// Async GraphRAG implementation
pub mod async_graphrag;

#[cfg(feature = "async")]
/// Async processing pipelines
pub mod async_processing;

#[cfg(feature = "caching")]
/// Caching utilities for LLM responses
pub mod caching;

#[cfg(feature = "function-calling")]
/// Function calling capabilities for LLMs
pub mod function_calling;

#[cfg(feature = "incremental")]
/// Incremental graph updates
pub mod incremental;

#[cfg(feature = "rograg")]
/// ROGRAG (Robustly Optimized GraphRAG) implementation
pub mod rograg;

// TODO: Implement remaining utility modules
// pub mod automatic_entity_linking;
// pub mod phase_saver;

// ================================
// PUBLIC API EXPORTS
// ================================

/// Prelude module containing the most commonly used types
pub mod prelude {
    // pub use crate::GraphRAG;
    // pub use crate::builder::{GraphRAGBuilder, ConfigPreset, LLMProvider};
    pub use crate::config::Config;
    pub use crate::core::{
        Document, DocumentId, Entity, EntityId, KnowledgeGraph,
        GraphRAGError, Result,
    };
}

// Re-export core types
pub use crate::config::Config;
pub use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention,
    ErrorContext, ErrorSeverity, GraphRAGError, KnowledgeGraph,
    Relationship, Result, TextChunk,
};

// Re-export core traits (async feature only)
#[cfg(feature = "async")]
pub use crate::core::traits::{
    Embedder, EntityExtractor, GraphStore, LanguageModel,
    Retriever, Storage, VectorStore,
};

// Storage exports (when storage features are enabled)
#[cfg(feature = "memory-storage")]
pub use crate::storage::MemoryStorage;

// TODO: Re-export builder when implemented
// pub use crate::builder::{ConfigPreset, GraphRAGBuilder, LLMProvider};

// TODO: Re-export main system when implemented
// pub use crate::GraphRAG;

// Feature-gated exports
#[cfg(feature = "lightrag")]
pub use crate::lightrag::{
    DualLevelRetriever, DualRetrievalConfig, DualRetrievalResults,
    KeywordExtractor, KeywordExtractorConfig, DualLevelKeywords,
    MergeStrategy, SemanticSearcher,
};

#[cfg(feature = "pagerank")]
pub use crate::graph::pagerank::{
    PageRankConfig, PersonalizedPageRank,
};

#[cfg(feature = "leiden")]
pub use crate::graph::leiden::{
    HierarchicalCommunities, LeidenConfig, LeidenCommunityDetector,
};

#[cfg(feature = "cross-encoder")]
pub use crate::reranking::cross_encoder::{
    CrossEncoder, CrossEncoderConfig, ConfidenceCrossEncoder,
    RankedResult, RerankingStats,
};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::pagerank_retrieval::{
    PageRankRetrievalSystem, ScoredResult,
};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::hipporag_ppr::{
    HippoRAGConfig, HippoRAGRetriever, Fact,
};

// ================================
// MAIN GRAPHRAG SYSTEM
// ================================

/// Main GraphRAG system
///
/// This is the primary entry point for using GraphRAG. It orchestrates
/// all components: knowledge graph, retrieval, generation, and caching.
///
/// # Examples
///
/// ```rust
/// use graphrag_core::{GraphRAG, Config};
///
/// # fn example() -> graphrag_core::Result<()> {
/// let config = Config::default();
/// let mut graphrag = GraphRAG::new(config)?;
/// graphrag.initialize()?;
///
/// // Add documents
/// graphrag.add_document_from_text("Your document text")?;
///
/// // Build knowledge graph
/// graphrag.build_graph()?;
///
/// // Query
/// let answer = graphrag.ask("Your question?")?;
/// println!("Answer: {}", answer);
/// # Ok(())
/// # }
/// ```
pub struct GraphRAG {
    config: Config,
    knowledge_graph: Option<KnowledgeGraph>,
    retrieval_system: Option<retrieval::RetrievalSystem>,
    #[cfg(feature = "parallel-processing")]
    #[allow(dead_code)]
    parallel_processor: Option<parallel::ParallelProcessor>,
}

impl GraphRAG {
    /// Create a new GraphRAG instance with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            knowledge_graph: None,
            retrieval_system: None,
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
        })
    }

    // TODO: Implement builder when GraphRAGBuilder module exists
    // /// Create a builder for configuring GraphRAG
    // pub fn builder() -> GraphRAGBuilder {
    //     GraphRAGBuilder::new()
    // }

    /// Initialize the GraphRAG system
    pub fn initialize(&mut self) -> Result<()> {
        self.knowledge_graph = Some(KnowledgeGraph::new());
        self.retrieval_system = Some(retrieval::RetrievalSystem::new(&self.config)?);
        Ok(())
    }

    /// Add a document from text content
    pub fn add_document_from_text(&mut self, text: &str) -> Result<()> {
        use crate::text::TextProcessor;
        use indexmap::IndexMap;

        // Use UUID for doc ID (works in both native and WASM)
        let doc_id = DocumentId::new(
            format!("doc_{}", uuid::Uuid::new_v4().simple())
        );

        let document = Document {
            id: doc_id,
            title: "Document".to_string(),
            content: text.to_string(),
            metadata: IndexMap::new(),
            chunks: Vec::new(),
        };

        let text_processor = TextProcessor::new(
            self.config.text.chunk_size,
            self.config.text.chunk_overlap
        )?;
        let chunks = text_processor.chunk_text(&document)?;

        let document_with_chunks = Document {
            chunks,
            ..document
        };

        self.add_document(document_with_chunks)
    }

    /// Add a document to the system
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let graph = self.knowledge_graph.as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        graph.add_document(document)
    }

    /// Build the knowledge graph from added documents
    ///
    /// This method implements dynamic pipeline selection based on the configured approach:
    /// - **Semantic** (config.approach = "semantic"): Uses LLM-based entity extraction with gleaning
    ///   for high-quality results. Requires Ollama to be enabled.
    /// - **Algorithmic** (config.approach = "algorithmic"): Uses pattern-based entity extraction
    ///   (regex + capitalization) for fast, resource-efficient processing.
    /// - **Hybrid** (config.approach = "hybrid"): Combines both approaches with weighted fusion.
    ///
    /// The selection is controlled by `config.approach` and mapped from TomlConfig's [mode] section.
    pub fn build_graph(&mut self) -> Result<()> {
        let graph = self.knowledge_graph.as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let chunks: Vec<_> = graph.chunks().cloned().collect();

        // PHASE 1: Extract and add all entities
        // Pipeline selection based on config.approach (semantic/algorithmic/hybrid)
        // - Semantic: config.entities.use_gleaning = true (LLM-based with iterative refinement)
        // - Algorithmic: config.entities.use_gleaning = false (pattern-based extraction)
        // - Hybrid: config.entities.use_gleaning = true (uses LLM + pattern fusion)
        if self.config.entities.use_gleaning && self.config.ollama.enabled {
            // LLM-based extraction with gleaning
            #[cfg(feature = "async")]
            {
                use crate::entity::GleaningEntityExtractor;
                use crate::ollama::OllamaClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM-based entity extraction with gleaning (max_rounds: {})",
                    self.config.entities.max_gleaning_rounds
                );

                // Create Ollama client
                let client = OllamaClient::new(self.config.ollama.clone());

                // Create base extractor with custom entity types and min confidence
                let base_extractor = crate::entity::EntityExtractor::new(
                    self.config.entities.min_confidence
                )?;

                // Create gleaning config from our config
                let gleaning_config = crate::entity::GleaningConfig {
                    max_gleaning_rounds: self.config.entities.max_gleaning_rounds,
                    completion_threshold: 0.8,
                    entity_confidence_threshold: self.config.entities.min_confidence as f64,
                    use_llm_completion_check: true,
                };

                // Create gleaning extractor
                let extractor = GleaningEntityExtractor::new(base_extractor, gleaning_config)
                    .with_llm_client(client);

                // Extract entities using async gleaning
                // We need to use a runtime to run async code in sync context
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| GraphRAGError::Config {
                        message: format!("Failed to create tokio runtime: {}", e),
                    })?;

                for chunk in &chunks {
                    let (entities, relationships) = rt.block_on(extractor.extract_with_gleaning(chunk))?;

                    // Add extracted entities
                    for entity in entities {
                        graph.add_entity(entity)?;
                    }

                    // Add extracted relationships
                    for relationship in relationships {
                        if let Err(e) = graph.add_relationship(relationship) {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                "Failed to add relationship: {} -> {} ({}). Error: {}",
                                e.to_string().split("entity ").nth(1).unwrap_or("unknown"),
                                e.to_string().split("entity ").nth(2).unwrap_or("unknown"),
                                "relationship",
                                e
                            );
                        }
                    }
                }
            }

            #[cfg(not(feature = "async"))]
            {
                return Err(GraphRAGError::Config {
                    message: "Gleaning extraction requires 'async' feature to be enabled".to_string(),
                });
            }
        } else {
            // Pattern-based extraction (regex + capitalization)
            use crate::entity::EntityExtractor;

            #[cfg(feature = "tracing")]
            tracing::info!("Using pattern-based entity extraction");

            let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

            for chunk in &chunks {
                let entities = extractor.extract_from_chunk(chunk)?;
                for entity in entities {
                    graph.add_entity(entity)?;
                }
            }

            // PHASE 2: Extract and add relationships between entities (for pattern-based only)
            // Gleaning extractor already extracts relationships in Phase 1
            // Only proceed if graph construction config enables relationship extraction
            if self.config.graph.extract_relationships {
            let all_entities: Vec<_> = graph.entities().cloned().collect();

            for chunk in &chunks {
                // Get entities that appear in this chunk
                let chunk_entities: Vec<_> = all_entities
                    .iter()
                    .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                    .cloned()
                    .collect();

                if chunk_entities.len() < 2 {
                    continue; // Need at least 2 entities for relationships
                }

                // Extract relationships
                let relationships = extractor.extract_relationships(&chunk_entities, chunk)?;

                // Add relationships to graph
                for (source_id, target_id, relation_type) in relationships {
                    let relationship = Relationship {
                        source: source_id.clone(),
                        target: target_id.clone(),
                        relation_type: relation_type.clone(),
                        confidence: self.config.graph.relationship_confidence_threshold,
                        context: vec![chunk.id.clone()],
                    };

                    // Log errors for debugging relationship extraction issues
                    if let Err(_e) = graph.add_relationship(relationship) {
                        #[cfg(feature = "tracing")]
                        tracing::debug!(
                            "Failed to add relationship: {} -> {} ({}). Error: {}",
                            source_id,
                            target_id,
                            relation_type,
                            _e
                        );
                    }
                }
            }
            }  // End of extract_relationships check
        }  // End of pattern-based extraction

        Ok(())
    }

    /// Query the system for relevant information
    pub fn ask(&mut self, query: &str) -> Result<String> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        let results = self.query_internal(query)?;
        Ok(results.join("\n"))
    }

    /// Internal query method
    fn query_internal(&mut self, query: &str) -> Result<Vec<String>> {
        let retrieval = self.retrieval_system.as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        let graph = self.knowledge_graph.as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Add embeddings to graph if not already present
        retrieval.add_embeddings_to_graph(graph)?;

        // Use hybrid query for real semantic search
        let search_results = retrieval.hybrid_query(query, graph)?;

        // Convert search results to strings
        let result_strings: Vec<String> = search_results
            .into_iter()
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();

        Ok(result_strings)
    }

    /// Check if system is initialized
    pub fn is_initialized(&self) -> bool {
        self.knowledge_graph.is_some() && self.retrieval_system.is_some()
    }

    /// Check if documents have been added
    pub fn has_documents(&self) -> bool {
        if let Some(graph) = &self.knowledge_graph {
            graph.chunks().count() > 0
        } else {
            false
        }
    }

    /// Check if graph has been built
    pub fn has_graph(&self) -> bool {
        if let Some(graph) = &self.knowledge_graph {
            graph.entities().count() > 0
        } else {
            false
        }
    }

    /// Get a reference to the knowledge graph
    pub fn knowledge_graph(&self) -> Option<&KnowledgeGraph> {
        self.knowledge_graph.as_ref()
    }

    /// Get entity details by ID
    pub fn get_entity(&self, entity_id: &str) -> Option<&Entity> {
        if let Some(graph) = &self.knowledge_graph {
            graph.entities().find(|e| e.id.0 == entity_id)
        } else {
            None
        }
    }

    /// Get all relationships involving an entity
    pub fn get_entity_relationships(&self, entity_id: &str) -> Vec<&Relationship> {
        if let Some(graph) = &self.knowledge_graph {
            let entity_id_obj = EntityId::new(entity_id.to_string());
            graph.relationships()
                .filter(|r| r.source == entity_id_obj || r.target == entity_id_obj)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get chunk by ID
    pub fn get_chunk(&self, chunk_id: &str) -> Option<&TextChunk> {
        if let Some(graph) = &self.knowledge_graph {
            graph.chunks().find(|c| c.id.0 == chunk_id)
        } else {
            None
        }
    }

    /// Query using PageRank-based retrieval (when pagerank feature is enabled)
    #[cfg(feature = "pagerank")]
    pub fn ask_with_pagerank(&mut self, query: &str) -> Result<Vec<retrieval::pagerank_retrieval::ScoredResult>> {
        use crate::retrieval::pagerank_retrieval::PageRankRetrievalSystem;

        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        let graph = self.knowledge_graph.as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let pagerank_system = PageRankRetrievalSystem::new(10);
        pagerank_system.search_with_pagerank(query, graph, Some(5))
    }

    /// Get a mutable reference to the knowledge graph
    pub fn knowledge_graph_mut(&mut self) -> Option<&mut KnowledgeGraph> {
        self.knowledge_graph.as_mut()
    }

    /// Ensure system is initialized
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.is_initialized() {
            self.initialize()
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphrag_creation() {
        let config = Config::default();
        let graphrag = GraphRAG::new(config);
        assert!(graphrag.is_ok());
    }

    // TODO: Enable when GraphRAGBuilder is fully implemented
    // #[test]
    // fn test_builder_pattern() {
    //     let graphrag = GraphRAG::builder()
    //         .with_preset(ConfigPreset::Basic)
    //         .build();
    //     assert!(graphrag.is_ok());
    // }
}
