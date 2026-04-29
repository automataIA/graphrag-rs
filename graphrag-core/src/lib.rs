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
/// Entity extraction and management
pub mod entity;
/// Text generation and LLM interactions (async feature only)
#[cfg(feature = "async")]
pub mod generation;
/// Graph data structures and algorithms
pub mod graph;
/// Retrieval strategies and implementations
pub mod retrieval;
/// Storage backends and persistence
#[cfg(any(
    feature = "memory-storage",
    feature = "persistent-storage",
    feature = "async"
))]
pub mod storage;
/// Text processing and chunking
pub mod text;
/// Vector operations and embeddings
pub mod vector;

/// Builder pattern implementations
pub mod builder;
/// Embedding generation and providers
pub mod embeddings;
/// Natural language processing utilities
pub mod nlp;
/// Ollama LLM integration
pub mod ollama;
pub mod openai;
pub mod chat;
/// Persistence layer for knowledge graphs (workspace management always available)
pub mod persistence;
/// Query processing and execution
pub mod query;
/// Text summarization capabilities
pub mod summarization;

// Pipeline modules
/// Data processing pipelines
pub mod pipeline;

// Advanced features (feature-gated)
#[cfg(feature = "parallel-processing")]
pub mod parallel;

#[cfg(feature = "lightrag")]
/// LightRAG dual-level retrieval optimization
pub mod lightrag;

/// Composable pipeline executor for build-graph operations
pub mod pipeline_executor;

// Utility modules
/// Reranking utilities for improving search result quality
pub mod reranking;

/// Monitoring, benchmarking, and performance tracking
pub mod monitoring;

/// RAG answer evaluation and criticism
pub mod critic;

/// Evaluation framework for query results and pipeline validation
pub mod evaluation;

/// Graph optimization (weight optimization, DW-GRPO)
#[cfg(feature = "async")]
pub mod optimization;

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

// Future utility modules (optional, not currently needed):
// pub mod automatic_entity_linking;  // Advanced entity linking
// pub mod phase_saver;               // Phase state persistence

// ================================
// PUBLIC API EXPORTS
// ================================

/// Prelude module containing the most commonly used types
///
/// Import everything you need with a single line:
/// ```rust
/// use graphrag_core::prelude::*;
/// ```
///
/// This includes:
/// - `GraphRAG` - The main orchestrator
/// - `Config` - Configuration management
/// - `GraphRAGBuilder` - Fluent configuration builder
/// - Core types: `Document`, `Entity`, `Relationship`, `TextChunk`
/// - Error handling: `Result`, `GraphRAGError`
pub mod prelude {
    // Main entry point
    pub use crate::GraphRAG;

    // Configuration & Builders
    pub use crate::builder::GraphRAGBuilder;
    pub use crate::builder::TypedBuilder;
    pub use crate::config::Config;

    // Error handling
    pub use crate::core::{GraphRAGError, Result};

    // Core data types
    pub use crate::core::{
        ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, KnowledgeGraph,
        Relationship, TextChunk,
    };

    // Search results and explained answers
    pub use crate::retrieval::SearchResult;
    pub use crate::retrieval::{ExplainedAnswer, ReasoningStep, SourceReference, SourceType};

    // Pipeline executor
    pub use crate::pipeline_executor::{PipelineExecutor, PipelineReport};

    // Config deserialization helper
    pub use crate::config::setconfig::SetConfig;
}

// Re-export core types
pub use crate::config::Config;
pub use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, ErrorContext, ErrorSeverity,
    ErrorSuggestion, GraphRAGError, KnowledgeGraph, Relationship, Result, TextChunk,
};

// Re-export core traits (async feature only)
#[cfg(feature = "async")]
pub use crate::core::traits::{
    Embedder, EntityExtractor, GraphStore, LanguageModel, Retriever, Storage, VectorStore,
};

// Storage exports (when storage features are enabled)
#[cfg(feature = "memory-storage")]
pub use crate::storage::MemoryStorage;

// Re-export builder (GraphRAGBuilder exists, ConfigPreset and LLMProvider not yet implemented)
pub use crate::builder::GraphRAGBuilder;
// Note: GraphRAG struct is already public (defined at line 247)
// Note: builder::GraphRAG is a placeholder - the real implementation is the main GraphRAG struct

// Feature-gated exports
#[cfg(feature = "lightrag")]
pub use crate::lightrag::{
    DualLevelKeywords, DualLevelRetriever, DualRetrievalConfig, DualRetrievalResults,
    KeywordExtractor, KeywordExtractorConfig, MergeStrategy, SemanticSearcher,
};

#[cfg(feature = "pagerank")]
pub use crate::graph::pagerank::{PageRankConfig, PersonalizedPageRank};

#[cfg(feature = "leiden")]
pub use crate::graph::leiden::{HierarchicalCommunities, LeidenCommunityDetector, LeidenConfig};

#[cfg(feature = "cross-encoder")]
pub use crate::reranking::cross_encoder::{
    ConfidenceCrossEncoder, CrossEncoder, CrossEncoderConfig, RankedResult, RerankingStats,
};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::pagerank_retrieval::{PageRankRetrievalSystem, ScoredResult};

#[cfg(feature = "pagerank")]
pub use crate::retrieval::hipporag_ppr::{Fact, HippoRAGConfig, HippoRAGRetriever};

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
    query_planner: Option<query::planner::QueryPlanner>,
    critic: Option<critic::Critic>,
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
            query_planner: None,
            critic: None,
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
        })
    }

    /// Create a Zero-Config local GraphRAG instance
    /// Uses: Candle (MiniLM) for embeddings, Memory/LanceDB for storage, Ollama for LLM
    pub fn default_local() -> Result<Self> {
        let mut config = Config::default();
        // Configure for local use
        config.ollama.enabled = true;
        // config.storage.type = StorageType::LanceDB; // Future

        Self::new(config)
    }

    /// Create a builder for configuring GraphRAG
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::GraphRAG;
    ///
    /// # fn example() -> graphrag_core::Result<()> {
    /// let graphrag = GraphRAG::builder()
    ///     .with_output_dir("./workspace")
    ///     .with_chunk_size(512)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> crate::builder::GraphRAGBuilder {
        crate::builder::GraphRAGBuilder::new()
    }

    /// Initialize the GraphRAG system.
    ///
    /// When `auto_save.enabled = true` and a `base_dir` is configured, attempts to
    /// load an existing graph from the workspace on disk before starting fresh.
    /// This means a second run reuses the previously built graph automatically.
    pub fn initialize(&mut self) -> Result<()> {
        // Try to restore from workspace if persistent storage is configured
        let loaded = self.try_load_from_workspace();

        if !loaded {
            self.knowledge_graph = Some(KnowledgeGraph::new());
        }

        self.retrieval_system = Some(retrieval::RetrievalSystem::new(&self.config)?);

        if let Some(client) =
            chat::ChatClient::from_config(&self.config.ollama, &self.config.openai)
        {
            self.query_planner = Some(query::planner::QueryPlanner::new(client));
        }

        Ok(())
    }

    /// Attempt to load the knowledge graph from a workspace on disk.
    /// Returns `true` if the graph was loaded successfully, `false` otherwise.
    fn try_load_from_workspace(&mut self) -> bool {
        if !self.config.auto_save.enabled {
            return false;
        }
        let base_dir = match &self.config.auto_save.base_dir {
            Some(d) => d.clone(),
            None => return false,
        };
        let workspace_name = self
            .config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");

        let manager = match persistence::WorkspaceManager::new(&base_dir) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Could not open workspace base dir '{}': {}", base_dir, e);
                return false;
            },
        };

        if !manager.workspace_exists(workspace_name) {
            return false;
        }

        match manager.load_graph(workspace_name) {
            Ok(graph) => {
                tracing::info!(
                    "Loaded graph from workspace '{}' ({} entities, {} relationships)",
                    workspace_name,
                    graph.entity_count(),
                    graph.relationship_count(),
                );
                self.knowledge_graph = Some(graph);
                true
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to load graph from workspace '{}': {}",
                    workspace_name,
                    e
                );
                false
            },
        }
    }

    /// Save the current knowledge graph to the configured workspace on disk.
    /// No-op when `auto_save.enabled = false` or `base_dir` is not set.
    pub fn save_to_workspace(&self) -> Result<()> {
        if !self.config.auto_save.enabled {
            return Ok(());
        }
        let base_dir = match &self.config.auto_save.base_dir {
            Some(d) => d,
            None => return Ok(()),
        };
        let workspace_name = self
            .config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let manager = persistence::WorkspaceManager::new(base_dir)?;
        manager.save_graph(graph, workspace_name)?;

        tracing::info!(
            "Saved graph to workspace '{}' in '{}' ({} entities, {} relationships)",
            workspace_name,
            base_dir,
            graph.entity_count(),
            graph.relationship_count(),
        );
        Ok(())
    }

    /// Add a document from text content
    pub fn add_document_from_text(&mut self, text: &str) -> Result<()> {
        use crate::text::TextProcessor;
        use indexmap::IndexMap;

        // Use UUID for doc ID (works in both native and WASM)
        let doc_id = DocumentId::new(format!("doc_{}", uuid::Uuid::new_v4().simple()));

        let document = Document {
            id: doc_id,
            title: "Document".to_string(),
            content: text.to_string(),
            metadata: IndexMap::new(),
            chunks: Vec::new(),
        };

        let text_processor =
            TextProcessor::new(self.config.text.chunk_size, self.config.text.chunk_overlap)?;
        let chunks = text_processor.chunk_text(&document)?;

        let document_with_chunks = Document { chunks, ..document };

        self.add_document(document_with_chunks)
    }

    /// Add a document to the system
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        graph.add_document(document)
    }

    /// Clear all entities and relationships from the knowledge graph
    ///
    /// This method preserves documents and text chunks but removes all extracted entities and relationships.
    /// Useful for rebuilding the graph from scratch without reloading documents.
    pub fn clear_graph(&mut self) -> Result<()> {
        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Clearing knowledge graph (preserving documents and chunks)");

        graph.clear_entities_and_relationships();
        Ok(())
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
    #[cfg(feature = "async")]
    pub async fn build_graph(&mut self) -> Result<()> {
        use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

        // When running inside a TUI, suppress indicatif output to avoid corrupting
        // ratatui's raw-mode terminal (the default draw target writes to stderr).
        let suppress = self.config.suppress_progress_bars;
        let make_pb = move |total: u64, style: ProgressStyle| -> ProgressBar {
            let pb = ProgressBar::new(total).with_style(style);
            if suppress {
                pb.set_draw_target(ProgressDrawTarget::hidden());
            }
            pb
        };

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let chunks: Vec<_> = graph.chunks().cloned().collect();
        let total_chunks = chunks.len();

        // PHASE 1: Extract and add all entities
        // Pipeline selection based on config.approach (semantic/algorithmic/hybrid)
        // - Semantic: config.entities.use_gleaning = true (LLM-based with iterative refinement)
        // - Algorithmic: config.entities.use_gleaning = false (pattern-based extraction)
        // - Hybrid: config.entities.use_gleaning = true (uses LLM + pattern fusion)

        // DEBUG: Log current configuration state
        #[cfg(feature = "tracing")]
        tracing::info!(
            "build_graph() - Config state: approach='{}', use_gleaning={}, ollama.enabled={}",
            self.config.approach,
            self.config.entities.use_gleaning,
            self.config.ollama.enabled
        );

        if self.config.entities.use_gleaning && self.config.chat_enabled() {
            // LLM-based extraction with gleaning
            #[cfg(feature = "async")]
            {
                use crate::entity::GleaningEntityExtractor;
                use crate::chat::ChatClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM-based entity extraction with gleaning (max_rounds: {})",
                    self.config.entities.max_gleaning_rounds
                );

                // Pick chat backend (ollama or openai). Skip extraction if neither is enabled.
                let client = match ChatClient::from_config(&self.config.ollama, &self.config.openai) {
                    Some(c) => c,
                    None => {
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            "Skipping LLM gleaning: neither config.ollama.enabled nor config.openai.enabled"
                        );
                        return Ok(());
                    },
                };

                // Create gleaning config from our config
                let gleaning_config = crate::entity::GleaningConfig {
                    max_gleaning_rounds: self.config.entities.max_gleaning_rounds,
                    completion_threshold: 0.8,
                    entity_confidence_threshold: self.config.entities.min_confidence as f64,
                    use_llm_completion_check: true,
                    entity_types: if self.config.entities.entity_types.is_empty() {
                        vec![
                            "PERSON".to_string(),
                            "ORGANIZATION".to_string(),
                            "LOCATION".to_string(),
                        ]
                    } else {
                        self.config.entities.entity_types.clone()
                    },
                    temperature: 0.1,
                    max_tokens: 1500,
                };

                // Create gleaning extractor with LLM client
                let extractor = GleaningEntityExtractor::new(client.clone(), gleaning_config);

                // Create relationship extractor for triple validation (if enabled)
                let rel_extractor = if self.config.entities.enable_triple_reflection {
                    Some(crate::entity::LLMRelationshipExtractor::new(Some(
                        &self.config.ollama,
                    ))?)
                } else {
                    None
                };

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-")
                );
                pb.set_message("Extracting entities with LLM");

                // Extract entities using async gleaning
                for (idx, chunk) in chunks.iter().enumerate() {
                    pb.set_message(format!(
                        "Chunk {}/{} (gleaning with {} rounds)",
                        idx + 1,
                        total_chunks,
                        self.config.entities.max_gleaning_rounds
                    ));

                    #[cfg(feature = "tracing")]
                    tracing::info!("Processing chunk {}/{} (LLM)", idx + 1, total_chunks);

                    let (entities, relationships) = extractor.extract_with_gleaning(chunk).await?;

                    // Build entity ID to name mapping for validation
                    let entity_map: std::collections::HashMap<_, _> = entities
                        .iter()
                        .map(|e| (e.id.clone(), e.name.clone()))
                        .collect();

                    // Add extracted entities
                    for entity in entities {
                        graph.add_entity(entity)?;
                    }

                    // Add extracted relationships with optional triple reflection validation
                    if let Some(ref validator) = rel_extractor {
                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            "Triple reflection enabled: validating {} relationships",
                            relationships.len()
                        );

                        let mut validated_count = 0;
                        let mut filtered_count = 0;

                        for relationship in relationships {
                            // Get entity names for validation
                            let source_name = entity_map
                                .get(&relationship.source)
                                .or_else(|| {
                                    graph
                                        .entities()
                                        .find(|e| e.id == relationship.source)
                                        .map(|e| &e.name)
                                })
                                .map(|s| s.as_str())
                                .unwrap_or(relationship.source.0.as_str());
                            let target_name = entity_map
                                .get(&relationship.target)
                                .or_else(|| {
                                    graph
                                        .entities()
                                        .find(|e| e.id == relationship.target)
                                        .map(|e| &e.name)
                                })
                                .map(|s| s.as_str())
                                .unwrap_or(relationship.target.0.as_str());

                            // Validate triple with LLM
                            match validator
                                .validate_triple(
                                    source_name,
                                    &relationship.relation_type,
                                    target_name,
                                    &chunk.content,
                                )
                                .await
                            {
                                Ok(validation) => {
                                    if validation.is_valid
                                        && validation.confidence
                                            >= self.config.entities.validation_min_confidence
                                    {
                                        // Valid relationship, add to graph
                                        if let Err(e) = graph.add_relationship(relationship) {
                                            #[cfg(feature = "tracing")]
                                            tracing::debug!(
                                                "Failed to add validated relationship: {}",
                                                e
                                            );
                                        } else {
                                            validated_count += 1;
                                        }
                                    } else {
                                        // Invalid or low-confidence, filter out
                                        filtered_count += 1;
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!(
                                            "Filtered relationship {} --[{}]--> {} (valid={}, conf={:.2}): {}",
                                            source_name, relationship.relation_type, target_name,
                                            validation.is_valid, validation.confidence, validation.reason
                                        );
                                    }
                                },
                                Err(e) => {
                                    // Validation failed, add anyway with warning
                                    #[cfg(feature = "tracing")]
                                    tracing::warn!(
                                        "Validation error, adding relationship anyway: {}",
                                        e
                                    );
                                    let _ = graph.add_relationship(relationship);
                                },
                            }
                        }

                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            "Triple reflection complete: {} validated, {} filtered",
                            validated_count,
                            filtered_count
                        );
                    } else {
                        // No validation, add all relationships
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

                    pb.inc(1);
                }

                pb.finish_with_message("Entity extraction complete");

                // Phase 1.3: ATOM Atomic Fact Extraction (if enabled)
                if self.config.entities.use_atomic_facts {
                    use crate::entity::AtomicFactExtractor;

                    #[cfg(feature = "tracing")]
                    tracing::info!("Starting atomic fact extraction (ATOM methodology)");

                    let atomic_extractor = AtomicFactExtractor::new(client.clone())
                        .with_max_tokens(self.config.entities.max_fact_tokens);

                    let pb_atomic = make_pb(total_chunks as u64,
                        ProgressStyle::default_bar()
                            .template("   [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} atomic facts ({eta})")
                            .expect("Invalid progress bar template")
                            .progress_chars("=>-")
                    );
                    pb_atomic.set_message("Extracting atomic facts");

                    let mut total_facts = 0;
                    let mut total_atomic_entities = 0;
                    let mut total_atomic_relationships = 0;

                    for (idx, chunk) in chunks.iter().enumerate() {
                        pb_atomic.set_message(format!(
                            "Chunk {}/{} (extracting atomic facts)",
                            idx + 1,
                            total_chunks
                        ));

                        #[cfg(feature = "tracing")]
                        tracing::info!("Processing chunk {}/{} (Atomic)", idx + 1, total_chunks);

                        match atomic_extractor.extract_atomic_facts(chunk).await {
                            Ok(facts) => {
                                total_facts += facts.len();

                                // Convert atomic facts to graph elements
                                let (atomic_entities, atomic_relationships) =
                                    atomic_extractor.atomics_to_graph_elements(facts, &chunk.id);

                                total_atomic_entities += atomic_entities.len();
                                total_atomic_relationships += atomic_relationships.len();

                                // Add atomic entities to graph
                                for entity in atomic_entities {
                                    if let Err(e) = graph.add_entity(entity) {
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!("Failed to add atomic entity: {}", e);
                                    }
                                }

                                // Add atomic relationships to graph
                                for relationship in atomic_relationships {
                                    if let Err(e) = graph.add_relationship(relationship) {
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!("Failed to add atomic relationship: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    chunk_id = %chunk.id,
                                    error = %e,
                                    "Atomic fact extraction failed for chunk"
                                );
                            },
                        }

                        pb_atomic.inc(1);
                    }

                    pb_atomic.finish_with_message(format!(
                        "Atomic extraction complete: {} facts → {} entities, {} relationships",
                        total_facts, total_atomic_entities, total_atomic_relationships
                    ));

                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        facts_extracted = total_facts,
                        atomic_entities = total_atomic_entities,
                        atomic_relationships = total_atomic_relationships,
                        "ATOM atomic fact extraction complete"
                    );
                }
            }
        } else if self.config.chat_enabled() {
            // LLM single-pass extraction (chat backend enabled, gleaning disabled)
            //
            // Uses LLMEntityExtractor directly for one extraction round per chunk.
            // num_ctx is calculated dynamically from the built prompt + 20% margin,
            // and keep_alive is forwarded so Ollama preserves the KV cache between chunks.
            #[cfg(feature = "async")]
            {
                use crate::entity::llm_extractor::LLMEntityExtractor;
                use crate::chat::ChatClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM single-pass entity extraction (no gleaning, keep_alive={:?})",
                    self.config.ollama.keep_alive,
                );

                let client = match ChatClient::from_config(&self.config.ollama, &self.config.openai) {
                    Some(c) => c,
                    None => {
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            "Skipping LLM extraction: neither config.ollama.enabled nor config.openai.enabled"
                        );
                        return Ok(());
                    },
                };
                let entity_types = if self.config.entities.entity_types.is_empty() {
                    vec![
                        "PERSON".to_string(),
                        "ORGANIZATION".to_string(),
                        "LOCATION".to_string(),
                    ]
                } else {
                    self.config.entities.entity_types.clone()
                };

                // Read the cap from whichever backend is the active chat
                // route — historically this only consulted ollama.* even
                // when openai was enabled, which silently capped openai
                // extraction at the ollama default. `None` is honored end
                // to end and means "no cap" (model stops at EOS).
                let extraction_max_tokens: Option<usize> = if self.config.openai.enabled {
                    self.config.openai.max_tokens.map(|n| n as usize)
                } else {
                    self.config.ollama.max_tokens.map(|n| n as usize)
                };
                let extraction_temperature = if self.config.openai.enabled {
                    self.config.openai.temperature.unwrap_or(0.1)
                } else {
                    self.config.ollama.temperature.unwrap_or(0.1)
                };

                let extractor = LLMEntityExtractor::new(client, entity_types)
                    .with_temperature(extraction_temperature)
                    .with_max_tokens_opt(extraction_max_tokens)
                    .with_keep_alive(self.config.ollama.keep_alive.clone());

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-"),
                );
                pb.set_message("Extracting entities with LLM (single-pass)");

                for (idx, chunk) in chunks.iter().enumerate() {
                    pb.set_message(format!(
                        "Chunk {}/{} (LLM single-pass)",
                        idx + 1,
                        total_chunks
                    ));

                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        "Processing chunk {}/{} (LLM single-pass)",
                        idx + 1,
                        total_chunks
                    );

                    match extractor.extract_from_chunk(chunk).await {
                        Ok((entities, relationships)) => {
                            for entity in entities {
                                if let Err(e) = graph.add_entity(entity) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("Failed to add entity: {}", e);
                                }
                            }
                            for relationship in relationships {
                                if let Err(e) = graph.add_relationship(relationship) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("Failed to add relationship: {}", e);
                                }
                            }
                        },
                        Err(e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk.id,
                                error = %e,
                                "LLM extraction failed for chunk, skipping"
                            );
                        },
                    }

                    pb.inc(1);
                }

                pb.finish_with_message("LLM single-pass extraction complete");
            }
        } else if self.config.gliner.enabled {
            // GLiNER-Relex joint NER + RE extraction
            //
            // gline-rs is synchronous (ONNX Runtime blocks the calling thread),
            // so we wrap each chunk in `spawn_blocking` to avoid stalling the
            // Tokio runtime.  A new `GLiNERExtractor` (with lazy model loading)
            // is created once outside the loop; the `Arc` inside it makes it
            // cheaply cloneable across blocking tasks.
            #[cfg(feature = "gliner")]
            {
                use crate::entity::GLiNERExtractor;
                use std::sync::Arc;

                let extractor = Arc::new(
                    GLiNERExtractor::new(self.config.gliner.clone()).map_err(|e| {
                        crate::core::error::GraphRAGError::EntityExtraction {
                            message: format!("GLiNER init failed: {e}"),
                        }
                    })?,
                );

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template(
                            "   [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} chunks ({eta})",
                        )
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-"),
                );
                pb.set_message("Extracting entities with GLiNER-Relex");

                for (idx, chunk) in chunks.iter().enumerate() {
                    pb.set_message(format!("Chunk {}/{} (GLiNER-Relex)", idx + 1, total_chunks));

                    let ext = Arc::clone(&extractor);
                    let ch = chunk.clone();
                    let result = tokio::task::spawn_blocking(move || ext.extract_from_chunk(&ch))
                        .await
                        .map_err(|e| crate::core::error::GraphRAGError::EntityExtraction {
                            message: format!("spawn_blocking join error: {e}"),
                        })?;

                    match result {
                        Ok((entities, relationships)) => {
                            for entity in entities {
                                if let Err(e) = graph.add_entity(entity) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("GLiNER: failed to add entity: {}", e);
                                }
                            }
                            for rel in relationships {
                                if let Err(e) = graph.add_relationship(rel) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("GLiNER: failed to add relationship: {}", e);
                                }
                            }
                        },
                        Err(e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk.id,
                                error = %e,
                                "GLiNER extraction failed for chunk, skipping"
                            );
                        },
                    }

                    pb.inc(1);
                }

                pb.finish_with_message("GLiNER-Relex extraction complete");
            }
            #[cfg(not(feature = "gliner"))]
            return Err(crate::core::error::GraphRAGError::Config {
                message: "GLiNER enabled in config but crate compiled without --features gliner"
                    .into(),
            });
        } else {
            // Pattern-based extraction (regex + capitalization)
            use crate::entity::EntityExtractor;

            #[cfg(feature = "tracing")]
            tracing::info!("Using pattern-based entity extraction");

            let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

            // Create progress bar for pattern-based extraction
            let pb = make_pb(
                total_chunks as u64,
                ProgressStyle::default_bar()
                    .template(
                        "   [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} chunks ({eta})",
                    )
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-"),
            );
            pb.set_message("Extracting entities (pattern-based)");

            for (idx, chunk) in chunks.iter().enumerate() {
                pb.set_message(format!(
                    "Chunk {}/{} (pattern-based)",
                    idx + 1,
                    total_chunks
                ));

                #[cfg(feature = "tracing")]
                tracing::info!("Processing chunk {}/{} (Pattern)", idx + 1, total_chunks);

                let entities = extractor.extract_from_chunk(chunk)?;
                for entity in entities {
                    graph.add_entity(entity)?;
                }

                pb.inc(1);
            }

            pb.finish_with_message("Entity extraction complete");

            // PHASE 2: Extract and add relationships between entities (for pattern-based only)
            // Gleaning extractor already extracts relationships in Phase 1
            // Only proceed if graph construction config enables relationship extraction
            if self.config.graph.extract_relationships {
                let all_entities: Vec<_> = graph.entities().cloned().collect();

                // Create progress bar for relationship extraction
                let rel_pb = make_pb(total_chunks as u64,
                ProgressStyle::default_bar()
                    .template("   [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} chunks ({eta})")
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-")
            );
                rel_pb.set_message("Extracting relationships");

                for (idx, chunk) in chunks.iter().enumerate() {
                    rel_pb.set_message(format!(
                        "Chunk {}/{} (relationships)",
                        idx + 1,
                        total_chunks
                    ));
                    // Get entities that appear in this chunk
                    let chunk_entities: Vec<_> = all_entities
                        .iter()
                        .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                        .cloned()
                        .collect();

                    if chunk_entities.len() < 2 {
                        rel_pb.inc(1);
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
                            embedding: None,
                            temporal_type: None,
                            temporal_range: None,
                            causal_strength: None,
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

                    rel_pb.inc(1);
                }

                rel_pb.finish_with_message("Relationship extraction complete");
            } // End of extract_relationships check
        } // End of pattern-based extraction

        // Persist to workspace if storage is configured
        self.save_to_workspace()?;

        Ok(())
    }

    /// Build the knowledge graph from added documents (synchronous fallback)
    ///
    /// This is a synchronous version for when the async feature is not enabled.
    /// Only supports pattern-based entity extraction.
    #[cfg(not(feature = "async"))]
    pub fn build_graph(&mut self) -> Result<()> {
        use crate::entity::EntityExtractor;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let chunks: Vec<_> = graph.chunks().cloned().collect();

        #[cfg(feature = "tracing")]
        tracing::info!("Using pattern-based entity extraction (sync mode)");

        let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

        for chunk in &chunks {
            let entities = extractor.extract_from_chunk(chunk)?;
            for entity in entities {
                graph.add_entity(entity)?;
            }
        }

        // Extract relationships if enabled
        if self.config.graph.extract_relationships {
            let all_entities: Vec<_> = graph.entities().cloned().collect();

            for chunk in &chunks {
                let chunk_entities: Vec<_> = all_entities
                    .iter()
                    .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                    .cloned()
                    .collect();

                if chunk_entities.len() < 2 {
                    continue;
                }

                let relationships = extractor.extract_relationships(&chunk_entities, chunk)?;

                for (source_id, target_id, relation_type) in relationships {
                    let relationship = Relationship {
                        source: source_id.clone(),
                        target: target_id.clone(),
                        relation_type: relation_type.clone(),
                        confidence: self.config.graph.relationship_confidence_threshold,
                        context: vec![chunk.id.clone()],
                    };

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
        }

        Ok(())
    }

    /// Query the system associated with reasoning (Query Decomposition)
    /// This splits the query into sub-queries, gathers context for all of them, and synthesizes an answer.
    #[cfg(feature = "async")]
    pub async fn ask_with_reasoning(&mut self, query: &str) -> Result<String> {
        // If planner is not available, fallback to standard ask
        if self.query_planner.is_none() {
            return self.ask(query).await;
        }

        self.ensure_initialized()?;
        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        let planner = self.query_planner.as_ref().unwrap();
        tracing::info!("Decomposing query: {}", query);

        // Decompose query
        let sub_queries = match planner.decompose(query).await {
            Ok(sq) => sq,
            Err(e) => {
                tracing::warn!(
                    "Query decomposition failed, falling back to standard query: {}",
                    e
                );
                vec![query.to_string()]
            },
        };

        tracing::info!("Sub-queries: {:?}", sub_queries);

        // Gather results for all sub-queries
        let mut all_results = Vec::new();
        for sub_query in sub_queries {
            match self.query_internal_with_results(&sub_query).await {
                Ok(results) => all_results.extend(results),
                Err(e) => tracing::warn!("Failed to execute sub-query '{}': {}", sub_query, e),
            }
        }

        if all_results.is_empty() {
            return Ok("No relevant information found for the decomposed queries.".to_string());
        }

        // Deduplicate results by ID
        // (Simple optimization to avoid duplicate context)
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut unique_results = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for result in all_results {
            if !seen_ids.contains(&result.id) {
                seen_ids.insert(result.id.clone());
                unique_results.push(result);
            }
        }

        if self.config.chat_enabled() {
            // Initial synthesis
            let mut current_answer = self
                .generate_semantic_answer_from_results(query, &unique_results)
                .await?;

            // Critic refinement loop
            if let Some(critic) = &self.critic {
                let mut attempts = 0;
                let max_retries = 3;

                while attempts < max_retries {
                    let context_strings: Vec<String> =
                        unique_results.iter().map(|r| r.content.clone()).collect();

                    let evaluation = match critic
                        .evaluate(query, &context_strings, &current_answer)
                        .await
                    {
                        Ok(eval) => eval,
                        Err(e) => {
                            tracing::warn!("Critic evaluation failed: {}", e);
                            break;
                        },
                    };

                    tracing::info!(
                        "Critic Evaluation (Attempt {}): Score={:.2}, Grounded={}, Feedback='{}'",
                        attempts + 1,
                        evaluation.score,
                        evaluation.grounded,
                        evaluation.feedback
                    );

                    if evaluation.score >= 0.7 && evaluation.grounded {
                        tracing::info!("Answer accepted by critic.");
                        break;
                    }

                    tracing::warn!("Answer rejected by critic. Refining...");

                    // Refine the answer using the feedback
                    current_answer = critic
                        .refine(query, &current_answer, &evaluation.feedback)
                        .await?;
                    attempts += 1;
                }
            }

            return Ok(current_answer);
        }

        // Fallback formatting
        let formatted: Vec<String> = unique_results
            .into_iter()
            .take(10)
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();
        Ok(formatted.join("\n"))
    }

    /// Query the system for relevant information
    #[cfg(feature = "async")]
    pub async fn ask(&mut self, query: &str) -> Result<String> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        // Get full search results with metadata
        let search_results = self.query_internal_with_results(query).await?;

        // If a chat backend is enabled, generate semantic answer using LLM
        if self.config.chat_enabled() {
            return self
                .generate_semantic_answer_from_results(query, &search_results)
                .await;
        }

        // Fallback: return formatted search results
        let formatted: Vec<String> = search_results
            .into_iter()
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();
        Ok(formatted.join("\n"))
    }

    /// Query the system for relevant information (synchronous version)
    #[cfg(not(feature = "async"))]
    pub fn ask(&mut self, query: &str) -> Result<String> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        let results = self.query_internal(query)?;
        Ok(results.join("\n"))
    }

    /// Query the system and return an explained answer with reasoning trace
    ///
    /// Unlike `ask()`, this method returns detailed information about:
    /// - Confidence score
    /// - Source references
    /// - Step-by-step reasoning
    /// - Key entities used
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::prelude::*;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let mut graphrag = GraphRAG::quick_start("Your document text").await?;
    /// let explained = graphrag.ask_explained("What is the main topic?").await?;
    ///
    /// println!("Answer: {}", explained.answer);
    /// println!("Confidence: {:.0}%", explained.confidence * 100.0);
    ///
    /// for step in &explained.reasoning_steps {
    ///     println!("Step {}: {}", step.step_number, step.description);
    /// }
    ///
    /// for source in &explained.sources {
    ///     println!("Source: {} (relevance: {:.0}%)",
    ///         source.id, source.relevance_score * 100.0);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn ask_explained(&mut self, query: &str) -> Result<retrieval::ExplainedAnswer> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        // Get search results
        let search_results = self.query_internal_with_results(query).await?;

        // Generate the answer
        let answer = if self.config.chat_enabled() {
            self.generate_semantic_answer_from_results(query, &search_results)
                .await?
        } else {
            // Fallback: concatenate top results
            search_results
                .iter()
                .take(3)
                .map(|r| r.content.clone())
                .collect::<Vec<_>>()
                .join(" ")
        };

        // Build the explained answer
        let explained = retrieval::ExplainedAnswer::from_results(answer, &search_results, query);

        Ok(explained)
    }

    /// Internal query method (public for CLI access to raw results)
    pub async fn query_internal(&mut self, query: &str) -> Result<Vec<String>> {
        let retrieval = self
            .retrieval_system
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Add embeddings to graph if not already present
        retrieval.add_embeddings_to_graph(graph).await?;

        // Use hybrid query for real semantic search
        let search_results = retrieval.hybrid_query(query, graph).await?;

        // Convert search results to strings
        let result_strings: Vec<String> = search_results
            .into_iter()
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();

        Ok(result_strings)
    }

    /// Internal query method that returns full SearchResult objects
    async fn query_internal_with_results(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::SearchResult>> {
        let retrieval = self
            .retrieval_system
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Add embeddings to graph if not already present
        retrieval.add_embeddings_to_graph(graph).await?;

        // Use hybrid query for real semantic search
        retrieval.hybrid_query(query, graph).await
    }

    /// Generate semantic answer from SearchResult objects
    #[cfg(feature = "async")]
    async fn generate_semantic_answer_from_results(
        &self,
        query: &str,
        search_results: &[retrieval::SearchResult],
    ) -> Result<String> {
        use crate::chat::ChatClient;

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Build context from search results by fetching actual chunk content.
        // We track chunk IDs to avoid duplicating the same chunk from multiple entity results.
        let mut context_parts = Vec::new();
        let mut seen_chunk_ids = std::collections::HashSet::new();

        for result in search_results.iter() {
            // For entity results, fetch the chunks where the entity appears
            if result.result_type == retrieval::ResultType::Entity
                && !result.source_chunks.is_empty()
            {
                let entity_label = result
                    .content
                    .split(" (score:")
                    .next()
                    .unwrap_or(&result.content);
                for chunk_id_str in &result.source_chunks {
                    if seen_chunk_ids.contains(chunk_id_str) {
                        continue;
                    }
                    let chunk_id = ChunkId::new(chunk_id_str.clone());
                    if let Some(chunk) = graph.chunks().find(|c| c.id == chunk_id) {
                        seen_chunk_ids.insert(chunk_id_str.clone());
                        context_parts.push((
                            result.score,
                            format!(
                                "[Entity: {} | Relevance: {:.2}]\n{}",
                                entity_label, result.score, chunk.content
                            ),
                        ));
                    }
                }
            }
            // For chunk results, use the full content directly
            else if result.result_type == retrieval::ResultType::Chunk {
                if !seen_chunk_ids.contains(&result.id) {
                    seen_chunk_ids.insert(result.id.clone());
                    context_parts.push((
                        result.score,
                        format!(
                            "[Chunk | Relevance: {:.2}]\n{}",
                            result.score, result.content
                        ),
                    ));
                }
            }
            // For other result types, use content as-is
            else {
                context_parts.push((
                    result.score,
                    format!(
                        "[{:?} | Relevance: {:.2}]\n{}",
                        result.result_type, result.score, result.content
                    ),
                ));
            }
        }

        // Sort by relevance descending, then join
        context_parts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let context = context_parts
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        if context.trim().is_empty() {
            return Ok("No relevant information found in the knowledge graph.".to_string());
        }

        // Pick chat backend (ollama or openai). Bail early if neither is enabled.
        let client = ChatClient::from_config(&self.config.ollama, &self.config.openai)
            .ok_or_else(|| GraphRAGError::Generation {
                message: "no chat backend enabled (config.ollama.enabled / config.openai.enabled both false)".to_string(),
            })?;

        // Build prompt for semantic answer generation with RAG best practices (2025)
        let prompt = format!(
            "You are a knowledgeable assistant specialized in answering questions based on a knowledge graph.\n\n\
            IMPORTANT INSTRUCTIONS:\n\
            - Answer ONLY using information from the provided context below\n\
            - Synthesize information from ALL context sections to give a comprehensive answer\n\
            - Provide direct, conversational, and natural responses\n\
            - Do NOT show your reasoning process or use <think> tags\n\
            - If the context lacks sufficient information, clearly state: \"I don't have enough information to answer this question.\"\n\
            - Aim for a complete answer (3-6 sentences) that covers different aspects found across the context\n\
            - Use a natural, helpful tone as if speaking to a person\n\n\
            CONTEXT:\n\
            {}\n\n\
            QUESTION: {}\n\n\
            ANSWER (direct response only, no reasoning):",
            context, query
        );

        // Dynamic num_ctx: prompt tokens + generous output budget + 20% margin
        let max_answer_tokens: u32 = 800;
        let prompt_tokens = (prompt.len() / 4) as u32;
        let total = prompt_tokens + max_answer_tokens;
        let with_margin = (total as f32 * 1.20) as u32;
        let num_ctx = (((with_margin + 1023) / 1024) * 1024)
            .max(4096)
            .min(131_072);

        let params = crate::ollama::OllamaGenerationParams {
            num_predict: Some(max_answer_tokens),
            temperature: self.config.ollama.temperature,
            num_ctx: Some(num_ctx),
            keep_alive: self.config.ollama.keep_alive.clone(),
            ..Default::default()
        };

        // Generate answer using LLM with dynamic context window
        match client.generate_with_params(&prompt, params).await {
            Ok(answer) => {
                // Post-processing: Remove <think> tags if present (Qwen3)
                let cleaned_answer = Self::remove_thinking_tags(&answer);
                Ok(cleaned_answer.trim().to_string())
            },
            Err(e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    "LLM generation failed: {}. Falling back to search results.",
                    e
                );

                // Fallback: return formatted search results
                Ok(format!(
                    "Relevant information from knowledge graph:\n\n{}",
                    context
                ))
            },
        }
    }

    /// Remove thinking tags from LLM output (for Qwen3 and similar models)
    ///
    /// Qwen3 often outputs <think>...</think> tags showing internal reasoning.
    /// This function removes all such tags and their content.
    #[cfg(feature = "async")]
    fn remove_thinking_tags(text: &str) -> String {
        // Remove all <think>...</think> blocks (including nested ones)
        // Use a simple approach: repeatedly remove until no more found
        let mut result = text.to_string();

        while let Some(start) = result.find("<think>") {
            // Find corresponding closing tag
            if let Some(end) = result[start..].find("</think>") {
                // Remove the entire block
                let end_pos = start + end + "</think>".len();
                result.replace_range(start..end_pos, "");
            } else {
                // No closing tag found, just remove opening tag
                result.replace_range(start..start + "<think>".len(), "");
                break;
            }
        }

        result.trim().to_string()
    }

    /// Get a reference to the current configuration
    pub fn config(&self) -> &Config {
        &self.config
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
            graph
                .relationships()
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
    #[cfg(all(feature = "pagerank", feature = "async"))]
    pub async fn ask_with_pagerank(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::pagerank_retrieval::ScoredResult>> {
        use crate::retrieval::pagerank_retrieval::PageRankRetrievalSystem;

        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let pagerank_system = PageRankRetrievalSystem::new(10);
        pagerank_system.search_with_pagerank(query, graph, Some(5))
    }

    /// Query using PageRank-based retrieval (when pagerank feature is enabled, sync version)
    #[cfg(all(feature = "pagerank", not(feature = "async")))]
    pub fn ask_with_pagerank(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::pagerank_retrieval::ScoredResult>> {
        use crate::retrieval::pagerank_retrieval::PageRankRetrievalSystem;

        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        let graph = self
            .knowledge_graph
            .as_ref()
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

    // ================================
    // CONVENIENCE CONSTRUCTORS
    // ================================

    /// Create GraphRAG from a JSON5 config file
    ///
    /// This is a convenience method that loads a JSON5 config file and creates a GraphRAG instance.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "json5-support")]
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// let graphrag = GraphRAG::from_json5_file("config/templates/symposium_zero_cost.graphrag.json5")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "json5-support")]
    pub fn from_json5_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use crate::config::json5_loader::load_json5_config;
        use crate::config::setconfig::SetConfig;

        let set_config = load_json5_config::<SetConfig, _>(path)?;
        let config = set_config.to_graphrag_config();
        Self::new(config)
    }

    /// Create GraphRAG from a config file (auto-detect format: TOML, JSON5, YAML, JSON)
    ///
    /// This method automatically detects the config file format based on the file extension
    /// and loads it appropriately.
    ///
    /// Supported formats:
    /// - `.toml` - TOML format
    /// - `.json5` - JSON5 format (requires `json5-support` feature)
    /// - `.yaml`, `.yml` - YAML format
    /// - `.json` - JSON format
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// // Auto-detect format from extension
    /// let graphrag = GraphRAG::from_config_file("config/templates/symposium_zero_cost.graphrag.json5")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use crate::config::setconfig::SetConfig;

        let set_config = SetConfig::from_file(path)?;
        let config = set_config.to_graphrag_config();
        Self::new(config)
    }

    /// Complete workflow: load config + process document + build graph
    ///
    /// This is the most convenient method for getting started with GraphRAG. It:
    /// 1. Loads the config file (auto-detecting the format)
    /// 2. Initializes the GraphRAG system
    /// 3. Loads and processes the document
    /// 4. Builds the knowledge graph
    ///
    /// After this method completes, the GraphRAG instance is ready to answer queries.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// // Complete workflow in one call
    /// let mut graphrag = GraphRAG::from_config_and_document(
    ///     "config/templates/symposium_zero_cost.graphrag.json5",
    ///     "docs-example/Symposium.txt"
    /// ).await?;
    ///
    /// // Ready to query
    /// let answer = graphrag.ask("What is Socrates' view on love?").await?;
    /// println!("Answer: {}", answer);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn from_config_and_document<P1, P2>(
        config_path: P1,
        document_path: P2,
    ) -> Result<Self>
    where
        P1: AsRef<std::path::Path>,
        P2: AsRef<std::path::Path>,
    {
        // Load config
        let mut graphrag = Self::from_config_file(config_path)?;

        // Initialize
        graphrag.initialize()?;

        // Load document
        let content = std::fs::read_to_string(document_path).map_err(GraphRAGError::Io)?;

        graphrag.add_document_from_text(&content)?;

        // Build graph
        graphrag.build_graph().await?;

        Ok(graphrag)
    }

    /// Quick start: Create a ready-to-query GraphRAG instance from text in one call
    ///
    /// This is the simplest way to get started with GraphRAG. It:
    /// 1. Creates a new instance with default or hierarchical configuration
    /// 2. Initializes all components
    /// 3. Processes your text document
    /// 4. Builds the knowledge graph
    ///
    /// After this call, you can immediately use `ask()` to query the system.
    ///
    /// # Example: Hello World in 5 lines
    /// ```rust,no_run
    /// use graphrag_core::prelude::*;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let mut graphrag = GraphRAG::quick_start("Your document text here").await?;
    /// let answer = graphrag.ask("What is this document about?").await?;
    /// println!("{}", answer);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Configuration
    /// - With `hierarchical-config` feature: Uses layered config (defaults → user → project → env)
    /// - Without: Uses sensible defaults optimized for local Ollama setup
    #[cfg(feature = "async")]
    pub async fn quick_start(text: &str) -> Result<Self> {
        // Load config (hierarchical if available, otherwise defaults)
        let config = Config::load()?;

        let mut graphrag = Self::new(config)?;
        graphrag.initialize()?;
        graphrag.add_document_from_text(text)?;
        graphrag.build_graph().await?;

        Ok(graphrag)
    }

    /// Quick start with custom configuration
    ///
    /// Like `quick_start()`, but allows you to customize the configuration
    /// using the builder pattern before processing the document.
    ///
    /// # Example
    /// ```rust,no_run
    /// use graphrag_core::prelude::*;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let mut graphrag = GraphRAG::quick_start_with_config(
    ///     "Your document text",
    ///     |builder| builder
    ///         .with_chunk_size(256)
    ///         .with_ollama_enabled(true)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn quick_start_with_config<F>(text: &str, configure: F) -> Result<Self>
    where
        F: FnOnce(crate::builder::GraphRAGBuilder) -> crate::builder::GraphRAGBuilder,
    {
        let builder = configure(Self::builder());
        let mut graphrag = builder.build()?;
        graphrag.initialize()?;
        graphrag.add_document_from_text(text)?;
        graphrag.build_graph().await?;

        Ok(graphrag)
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

    #[test]
    fn test_builder_pattern() {
        let graphrag = GraphRAG::builder()
            .with_output_dir("./test_output")
            .with_chunk_size(512)
            .with_top_k(10)
            .build();
        assert!(graphrag.is_ok());
    }
}
