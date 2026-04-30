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
    /// Chunks whose entity/relationship extraction has already been
    /// merged into `knowledge_graph`. Set by `build_graph` (every
    /// chunk processed) and `extend_graph` (only the delta). The
    /// only consumer is `extend_graph`'s "skip already-extracted
    /// chunks" filter; everything else still iterates over
    /// `knowledge_graph.chunks()` directly.
    processed_chunks: std::collections::HashSet<core::ChunkId>,
    #[cfg(feature = "parallel-processing")]
    #[allow(dead_code)]
    parallel_processor: Option<parallel::ParallelProcessor>,
}

/// Internal accumulator for `extend_graph`'s per-chunk pass. Tracks
/// what changed so the public `ExtendSummary` can report deltas
/// rather than raw graph totals.
#[derive(Default)]
struct ExtractMetrics {
    new_entities: usize,
    new_relationships: usize,
    mentions_merged: usize,
}

/// LightRAG-style dual-level query keywords.
///
/// Reference: LightRAG paper (arXiv:2410.05779). At query time, one
/// LLM call extracts both keyword sets from the user query; each
/// drives a separate retrieval stream:
///   - `low_level` keywords are embedded and used to vector-search
///     the *entity* store for top-K seed entities (concrete things).
///   - `high_level` keywords are embedded and used to vector-search
///     the *relationship* store for top-K seed relations (themes,
///     abstract connections).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct QueryKeywords {
    /// Specific named entities, attributes, or details mentioned in
    /// the query (e.g. "PostgreSQL", "ACID compliance").
    pub low_level: Vec<String>,
    /// Overarching themes, concepts, or topics the query is about
    /// (e.g. "database transactions", "concurrency control").
    pub high_level: Vec<String>,
}

/// Wire envelope for parsing the dual-keyword JSON LLM output. Field
/// names match LightRAG's published prompt; not exposed publicly.
#[derive(Debug, serde::Deserialize)]
struct QueryKeywordsWire {
    high_level_keywords: Option<Vec<String>>,
    low_level_keywords: Option<Vec<String>>,
}

/// Caller-supplied seed populations for [`GraphRAG::ask_with_dual_seeds`].
/// Each LightRAG retrieval mode maps to a different non-empty subset:
/// local → entities; global → relations; hybrid → entities + relations;
/// mix → all three (entities, relations, chunks).
#[derive(Debug, Clone, Default)]
pub struct DualSeeds {
    /// Entity ids — typically the top-K result of vector-searching
    /// an entity-description embedding store with the embedded
    /// `low_level` keywords.
    pub entities: Vec<EntityId>,
    /// Relation triples `(source_entity_id, target_entity_id,
    /// relation_type)` — typically from vector-searching a
    /// relationship-description embedding store with the embedded
    /// `high_level` keywords.
    pub relations: Vec<(EntityId, EntityId, String)>,
    /// Direct chunk ids — for LightRAG's `mix` mode, where chunk-level
    /// vector search results are merged with the entity/relation
    /// expansion. Empty for local/global/hybrid.
    pub chunks: Vec<ChunkId>,
}

/// Summary returned by [`GraphRAG::extend_graph`] — what changed in
/// this incremental pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtendSummary {
    /// Number of chunks that were extracted in this call. Zero on
    /// the no-op fast path (live chunk count hasn't grown since the
    /// last build/extend).
    pub chunks_processed: usize,
    /// New entities added to the graph (after dedup-by-id; existing
    /// entities re-mentioned in new chunks have their `mentions`
    /// extended in place rather than counted here).
    pub new_entities: usize,
    /// New relationships added to the graph in this call.
    pub new_relationships: usize,
    /// Existing entities whose `mentions` were extended because they
    /// were re-mentioned in a new chunk. Useful for assessing whether
    /// downstream community/PageRank recompute is warranted.
    pub mentions_merged: usize,
    /// Graph totals after the extend pass.
    pub total_entities: usize,
    pub total_relationships: usize,
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
            processed_chunks: std::collections::HashSet::new(),
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

        if self.config.ollama.enabled {
            let client = ollama::OllamaClient::new(self.config.ollama.clone());
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
        } else if self.config.ollama.enabled {
            // LLM single-pass extraction (Ollama enabled, gleaning disabled)
            //
            // Uses LLMEntityExtractor directly for one extraction round per chunk.
            // num_ctx is calculated dynamically from the built prompt + 20% margin,
            // and keep_alive is forwarded so Ollama preserves the KV cache between chunks.
            #[cfg(feature = "async")]
            {
                use crate::entity::llm_extractor::LLMEntityExtractor;
                use crate::ollama::OllamaClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM single-pass entity extraction (no gleaning, keep_alive={:?})",
                    self.config.ollama.keep_alive,
                );

                let client = OllamaClient::new(self.config.ollama.clone());
                let entity_types = if self.config.entities.entity_types.is_empty() {
                    vec![
                        "PERSON".to_string(),
                        "ORGANIZATION".to_string(),
                        "LOCATION".to_string(),
                    ]
                } else {
                    self.config.entities.entity_types.clone()
                };

                let extractor = LLMEntityExtractor::new(client, entity_types)
                    .with_temperature(self.config.ollama.temperature.unwrap_or(0.1))
                    .with_max_tokens(self.config.ollama.max_tokens.unwrap_or(1500) as usize)
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

        // Mark every chunk currently in the graph as having been
        // processed by entity extraction. `extend_graph` consults this
        // set to decide which chunks are "new" since the last build.
        // Doing it here (rather than per-chunk inside the loops above)
        // keeps the four extraction branches identical to upstream.
        if let Some(g) = self.knowledge_graph.as_ref() {
            self.processed_chunks
                .extend(g.chunks().map(|c| c.id.clone()));
        }

        // Persist to workspace if storage is configured
        self.save_to_workspace()?;

        Ok(())
    }

    /// Extract entities and relationships from chunks ingested **since
    /// the last `build_graph` or `extend_graph` call**, merging the
    /// results into the existing knowledge graph without re-walking
    /// chunks that have already been processed.
    ///
    /// Mirrors Microsoft GraphRAG's `graphrag append` semantics: a
    /// cheap call to fire after a batch of `add_document` so newly-
    /// ingested content shows up in queries, without paying for a
    /// wholesale re-extraction over the corpus.
    ///
    /// # Behaviour
    /// - Filters `knowledge_graph.chunks()` against
    ///   `self.processed_chunks`; only un-processed chunks reach the
    ///   extractor. The filtered set is the "delta" since last build.
    /// - Runs the same extractor that `build_graph` would pick
    ///   (gleaning / LLM single-pass / pattern-based) over the delta
    ///   chunks. GLiNER is not yet wired here — call `build_graph`
    ///   for that path.
    /// - **Dedupes by entity ID** before adding. If a delta chunk
    ///   re-mentions an entity that already exists, the existing
    ///   entity's `mentions` are extended in place (no new node).
    ///   Mirrors Microsoft's stable-id pattern (v0.5.0+).
    /// - Skips relationships whose source or target entity isn't in
    ///   the graph (existing `add_relationship` behaviour).
    /// - Updates `processed_chunks` with the delta on success.
    ///
    /// # No-op fast path
    /// Returns immediately with `chunks_processed = 0` when the live
    /// chunk count equals `processed_chunks.len()`. Cron-driven
    /// callers can fire this on a tight cadence without paying any
    /// LLM cost when nothing has been ingested.
    ///
    /// # Errors
    /// Bubbles up extractor errors. If the configured approach is
    /// `gliner` (or `chat_enabled() == false` and no pattern fallback
    /// is configured), returns `GraphRAGError::Config` rather than
    /// silently doing nothing.
    #[cfg(feature = "async")]
    pub async fn extend_graph(&mut self) -> Result<ExtendSummary> {
        use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

        // 1. Snapshot delta — copy the chunk list to drop the borrow
        //    on knowledge_graph before any mutating extraction calls.
        let (delta_chunks, total_entities_before, total_relationships_before) = {
            let graph = self
                .knowledge_graph
                .as_ref()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Knowledge graph not initialized".to_string(),
                })?;
            let delta: Vec<_> = graph
                .chunks()
                .filter(|c| !self.processed_chunks.contains(&c.id))
                .cloned()
                .collect();
            let entities = graph.entities().count();
            let relationships = graph.relationships().count();
            (delta, entities, relationships)
        };

        let total_delta = delta_chunks.len();

        // 2. Fast no-op path. Cron-callable without burning anything.
        if total_delta == 0 {
            return Ok(ExtendSummary {
                chunks_processed: 0,
                new_entities: 0,
                new_relationships: 0,
                mentions_merged: 0,
                total_entities: total_entities_before,
                total_relationships: total_relationships_before,
            });
        }

        let suppress = self.config.suppress_progress_bars;
        let make_pb = move |total: u64, style: ProgressStyle| -> ProgressBar {
            let pb = ProgressBar::new(total).with_style(style);
            if suppress {
                pb.set_draw_target(ProgressDrawTarget::hidden());
            }
            pb
        };

        let mut metrics = ExtractMetrics::default();

        #[cfg(feature = "tracing")]
        tracing::info!(
            "extend_graph: processing {} delta chunks (approach='{}', use_gleaning={}, ollama.enabled={})",
            total_delta,
            self.config.approach,
            self.config.entities.use_gleaning,
            self.config.ollama.enabled,
        );

        // 3. Pick the same extraction path build_graph would pick. The
        //    only difference vs build_graph is per-chunk dedup on add.
        if self.config.entities.use_gleaning && self.config.ollama.enabled {
            self.extend_with_gleaning(&delta_chunks, &mut metrics, &make_pb)
                .await?;
        } else if self.config.ollama.enabled {
            self.extend_with_llm_single_pass(&delta_chunks, &mut metrics, &make_pb)
                .await?;
        } else if self.config.gliner.enabled {
            #[cfg(feature = "gliner")]
            {
                self.extend_with_gliner(&delta_chunks, &mut metrics, &make_pb)
                    .await?;
            }
            #[cfg(not(feature = "gliner"))]
            return Err(GraphRAGError::Config {
                message: "extend_graph: gliner.enabled but crate compiled \
                          without --features gliner"
                    .to_string(),
            });
        } else {
            // Pattern-based — works without an LLM; useful for tests
            // and for setups that don't have a chat backend.
            self.extend_with_pattern_extraction(&delta_chunks, &mut metrics, &make_pb)?;
        }

        // 4. Mark processed.
        for chunk in &delta_chunks {
            self.processed_chunks.insert(chunk.id.clone());
        }

        // 5. Final totals (re-read; extraction may have added entities).
        let (total_entities, total_relationships) = {
            let graph = self
                .knowledge_graph
                .as_ref()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Knowledge graph went away mid-extend (impossible)".to_string(),
                })?;
            (graph.entities().count(), graph.relationships().count())
        };

        // 6. Persist.
        self.save_to_workspace()?;

        Ok(ExtendSummary {
            chunks_processed: total_delta,
            new_entities: metrics.new_entities,
            new_relationships: metrics.new_relationships,
            mentions_merged: metrics.mentions_merged,
            total_entities,
            total_relationships,
        })
    }

    /// Reset `processed_chunks` so the next `extend_graph` re-extracts
    /// every chunk currently in the graph. Useful after a config
    /// change (entity_types, prompts) where you want to re-extract
    /// everything but don't want to wipe the graph first.
    pub fn clear_processed_chunks(&mut self) {
        self.processed_chunks.clear();
    }

    /// Number of chunks that have been processed by entity extraction
    /// (via `build_graph` or `extend_graph`). Surfaced for callers
    /// that want to expose freshness telemetry alongside graph stats.
    pub fn processed_chunk_count(&self) -> usize {
        self.processed_chunks.len()
    }

    /// Seed `processed_chunks` with chunk ids that have *already* been
    /// extracted in a prior session.
    ///
    /// Intended for hydration paths: a host that persists per-document
    /// state in an external store (e.g. graphrag-server's Qdrant) can
    /// rebuild the in-memory chunk index by re-chunking the source text
    /// through `add_document_from_text` and then call this to mark
    /// those chunks as already-extracted, so the next `extend_graph`
    /// only operates on truly-new chunks rather than re-extracting the
    /// whole corpus on every restart.
    ///
    /// Idempotent: passing chunk ids already in the set has no effect.
    /// Chunk ids that aren't present in the in-memory graph are still
    /// inserted (delta computation in `extend_graph` ignores them
    /// naturally since it iterates the in-memory chunk set).
    pub fn seed_processed_chunks<I>(&mut self, chunk_ids: I)
    where
        I: IntoIterator<Item = ChunkId>,
    {
        self.processed_chunks.extend(chunk_ids);
    }

    /// LLM single-pass extension path — mirrors the LLM single-pass
    /// branch in `build_graph` but operates on the supplied delta
    /// slice instead of every chunk in the graph, and dedupes
    /// entities by id on insert.
    #[cfg(feature = "async")]
    async fn extend_with_llm_single_pass(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::entity::llm_extractor::LLMEntityExtractor;
        use crate::ollama::OllamaClient;
        use indicatif::ProgressStyle;

        let client = OllamaClient::new(self.config.ollama.clone());

        let entity_types = if self.config.entities.entity_types.is_empty() {
            vec![
                "PERSON".to_string(),
                "ORGANIZATION".to_string(),
                "LOCATION".to_string(),
            ]
        } else {
            self.config.entities.entity_types.clone()
        };

        let extractor = LLMEntityExtractor::new(client, entity_types)
            .with_temperature(self.config.ollama.temperature.unwrap_or(0.1))
            .with_max_tokens(self.config.ollama.max_tokens.unwrap_or(1500) as usize)
            .with_keep_alive(self.config.ollama.keep_alive.clone());

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template(
                    "   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} delta chunks ({eta})",
                )
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (LLM single-pass)");

        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (LLM single-pass)",
                idx + 1,
                delta_chunks.len()
            ));

            #[cfg(feature = "tracing")]
            tracing::info!(
                "extend_graph: processing delta chunk {}/{} (LLM single-pass)",
                idx + 1,
                delta_chunks.len()
            );

            match extractor.extract_from_chunk(chunk).await {
                Ok((entities, relationships)) => {
                    let graph =
                        self.knowledge_graph
                            .as_mut()
                            .ok_or_else(|| GraphRAGError::Config {
                                message: "Knowledge graph went away mid-extract".to_string(),
                            })?;
                    for entity in entities {
                        Self::merge_entity(graph, entity, metrics)?;
                    }
                    for relationship in relationships {
                        Self::merge_relationship(graph, relationship, metrics);
                    }
                },
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        chunk_id = %chunk.id,
                        error = %e,
                        "extend_graph: LLM extraction failed for delta chunk; skipping"
                    );
                    let _ = e;
                },
            }

            pb.inc(1);
        }

        pb.finish_with_message("LLM single-pass extension complete");
        Ok(())
    }

    /// Gleaning extension path — same shape as the gleaning branch in
    /// `build_graph`, restricted to the delta slice with dedup-on-add.
    /// Triple-reflection validation is honored.
    #[cfg(feature = "async")]
    async fn extend_with_gleaning(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::entity::GleaningEntityExtractor;
        use crate::ollama::OllamaClient;
        use indicatif::ProgressStyle;

        let client = OllamaClient::new(self.config.ollama.clone());

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
        let extractor = GleaningEntityExtractor::new(client.clone(), gleaning_config);

        let rel_extractor = if self.config.entities.enable_triple_reflection {
            Some(crate::entity::LLMRelationshipExtractor::new(Some(
                &self.config.ollama,
            ))?)
        } else {
            None
        };

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template(
                    "   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} delta chunks ({eta})",
                )
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (gleaning)");

        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (gleaning, {} rounds)",
                idx + 1,
                delta_chunks.len(),
                self.config.entities.max_gleaning_rounds
            ));

            #[cfg(feature = "tracing")]
            tracing::info!(
                "extend_graph: processing delta chunk {}/{} (gleaning)",
                idx + 1,
                delta_chunks.len()
            );

            let (entities, relationships) = extractor.extract_with_gleaning(chunk).await?;

            let entity_map: std::collections::HashMap<_, _> = entities
                .iter()
                .map(|e| (e.id.clone(), e.name.clone()))
                .collect();

            // Add entities first so relationship merge can reference them.
            {
                let graph = self
                    .knowledge_graph
                    .as_mut()
                    .ok_or_else(|| GraphRAGError::Config {
                        message: "Knowledge graph went away mid-extract".to_string(),
                    })?;
                for entity in entities {
                    Self::merge_entity(graph, entity, metrics)?;
                }
            }

            // Relationships, optionally validated.
            if let Some(ref validator) = rel_extractor {
                for relationship in relationships {
                    let (source_name, target_name) = {
                        let graph = self.knowledge_graph.as_ref().unwrap();
                        let source_name = entity_map
                            .get(&relationship.source)
                            .cloned()
                            .or_else(|| {
                                graph
                                    .entities()
                                    .find(|e| e.id == relationship.source)
                                    .map(|e| e.name.clone())
                            })
                            .unwrap_or_else(|| relationship.source.0.clone());
                        let target_name = entity_map
                            .get(&relationship.target)
                            .cloned()
                            .or_else(|| {
                                graph
                                    .entities()
                                    .find(|e| e.id == relationship.target)
                                    .map(|e| e.name.clone())
                            })
                            .unwrap_or_else(|| relationship.target.0.clone());
                        (source_name, target_name)
                    };
                    match validator
                        .validate_triple(
                            &source_name,
                            &relationship.relation_type,
                            &target_name,
                            &chunk.content,
                        )
                        .await
                    {
                        Ok(validation) => {
                            if validation.is_valid
                                && validation.confidence
                                    >= self.config.entities.validation_min_confidence
                            {
                                let graph = self.knowledge_graph.as_mut().unwrap();
                                Self::merge_relationship(graph, relationship, metrics);
                            } else {
                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    "extend_graph: filtered relationship (valid={}, conf={:.2})",
                                    validation.is_valid,
                                    validation.confidence
                                );
                            }
                        },
                        Err(_e) => {
                            // Validation errored → add anyway (matches build_graph's behaviour).
                            let graph = self.knowledge_graph.as_mut().unwrap();
                            Self::merge_relationship(graph, relationship, metrics);
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                "extend_graph: validation error, adding relationship anyway: {}",
                                _e
                            );
                        },
                    }
                }
            } else {
                let graph = self.knowledge_graph.as_mut().unwrap();
                for relationship in relationships {
                    Self::merge_relationship(graph, relationship, metrics);
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("Gleaning extension complete");
        Ok(())
    }

    /// GLiNER joint NER + RE extension path — mirrors the GLiNER
    /// branch in `build_graph`, restricted to the delta slice with
    /// dedup-on-add via `merge_entity` / `merge_relationship`.
    ///
    /// `gline-rs` is synchronous (ONNX Runtime blocks the calling
    /// thread), so each per-chunk call runs inside
    /// `tokio::task::spawn_blocking`. The `Arc`-cloned extractor is
    /// cheaply shareable across blocking tasks.
    ///
    /// **Untested**: matches build_graph's GLiNER branch — upstream
    /// has no test coverage for that path either, since GLiNER needs
    /// a downloaded ONNX model at runtime and produces non-
    /// deterministic output. The four pattern-based `extend_graph_*`
    /// tests prove the dispatcher routes correctly; this path's
    /// correctness rests on visual parity with build_graph's GLiNER
    /// branch and on the shared `merge_entity` / `merge_relationship`
    /// dedup helpers (which are exercised by the pattern-based tests).
    #[cfg(feature = "gliner")]
    async fn extend_with_gliner(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::entity::GLiNERExtractor;
        use indicatif::ProgressStyle;
        use std::sync::Arc;

        // Lazy model load happens inside GLiNERExtractor::new; failures
        // surface here rather than on first chunk so the caller sees
        // them immediately.
        let extractor = Arc::new(GLiNERExtractor::new(self.config.gliner.clone()).map_err(
            |e| crate::core::error::GraphRAGError::EntityExtraction {
                message: format!("GLiNER init failed: {e}"),
            },
        )?);

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template("   [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} delta chunks ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (GLiNER-Relex)");

        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (GLiNER-Relex)",
                idx + 1,
                delta_chunks.len()
            ));

            #[cfg(feature = "tracing")]
            tracing::info!(
                "extend_graph: processing delta chunk {}/{} (GLiNER)",
                idx + 1,
                delta_chunks.len()
            );

            let ext = Arc::clone(&extractor);
            let ch = chunk.clone();
            let result = tokio::task::spawn_blocking(move || ext.extract_from_chunk(&ch))
                .await
                .map_err(|e| crate::core::error::GraphRAGError::EntityExtraction {
                    message: format!("spawn_blocking join error: {e}"),
                })?;

            match result {
                Ok((entities, relationships)) => {
                    let graph =
                        self.knowledge_graph
                            .as_mut()
                            .ok_or_else(|| GraphRAGError::Config {
                                message: "Knowledge graph went away mid-extract".to_string(),
                            })?;
                    for entity in entities {
                        Self::merge_entity(graph, entity, metrics)?;
                    }
                    for rel in relationships {
                        Self::merge_relationship(graph, rel, metrics);
                    }
                },
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        chunk_id = %chunk.id,
                        error = %_e,
                        "extend_graph: GLiNER extraction failed for delta chunk; skipping"
                    );
                },
            }

            pb.inc(1);
        }

        pb.finish_with_message("GLiNER-Relex extension complete");
        Ok(())
    }

    /// Pattern-based extension path — regex/capitalization extraction
    /// over the delta slice. No LLM dependency, useful for setups that
    /// don't have a chat backend wired and for testing.
    fn extend_with_pattern_extraction(
        &mut self,
        delta_chunks: &[crate::core::TextChunk],
        metrics: &mut ExtractMetrics,
        make_pb: &(impl Fn(u64, indicatif::ProgressStyle) -> indicatif::ProgressBar + Send + Sync),
    ) -> Result<()> {
        use crate::entity::EntityExtractor;
        use indicatif::ProgressStyle;

        let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

        let pb = make_pb(
            delta_chunks.len() as u64,
            ProgressStyle::default_bar()
                .template(
                    "   [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} delta chunks ({eta})",
                )
                .expect("Invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message("Extending graph (pattern-based)");

        // Phase 1: entities
        for (idx, chunk) in delta_chunks.iter().enumerate() {
            pb.set_message(format!(
                "Delta chunk {}/{} (pattern entities)",
                idx + 1,
                delta_chunks.len()
            ));

            let entities = extractor.extract_from_chunk(chunk)?;
            let graph = self
                .knowledge_graph
                .as_mut()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Knowledge graph went away mid-extract".to_string(),
                })?;
            for entity in entities {
                Self::merge_entity(graph, entity, metrics)?;
            }
            pb.inc(1);
        }
        pb.finish_with_message("Pattern entity extension complete");

        // Phase 2: relationships, only if config requests them.
        if self.config.graph.extract_relationships {
            let all_entities: Vec<_> = {
                let graph = self.knowledge_graph.as_ref().unwrap();
                graph.entities().cloned().collect()
            };

            let rel_pb = make_pb(
                delta_chunks.len() as u64,
                ProgressStyle::default_bar()
                    .template("   [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} delta chunks rels ({eta})")
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-"),
            );
            rel_pb.set_message("Extending relationships (pattern-based)");

            for (idx, chunk) in delta_chunks.iter().enumerate() {
                rel_pb.set_message(format!(
                    "Delta chunk {}/{} (pattern relationships)",
                    idx + 1,
                    delta_chunks.len()
                ));

                let chunk_entities: Vec<_> = all_entities
                    .iter()
                    .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                    .cloned()
                    .collect();
                if chunk_entities.len() < 2 {
                    rel_pb.inc(1);
                    continue;
                }

                let relationships = extractor.extract_relationships(&chunk_entities, chunk)?;
                let graph = self.knowledge_graph.as_mut().unwrap();
                for (source_id, target_id, relation_type) in relationships {
                    let relationship = Relationship {
                        source: source_id,
                        target: target_id,
                        relation_type,
                        confidence: self.config.graph.relationship_confidence_threshold,
                        context: vec![chunk.id.clone()],
                        embedding: None,
                        temporal_type: None,
                        temporal_range: None,
                        causal_strength: None,
                    };
                    Self::merge_relationship(graph, relationship, metrics);
                }
                rel_pb.inc(1);
            }
            rel_pb.finish_with_message("Pattern relationship extension complete");
        }

        Ok(())
    }

    /// Add `new_entity` to `graph`, merging into the existing entity
    /// if the id already exists. Used by the extend path; build_graph
    /// keeps its no-dedup behaviour for backward compatibility.
    ///
    /// Merge semantics: existing entity's `mentions` are extended with
    /// any not-already-present mentions from `new_entity` (compared by
    /// `(chunk_id, start_offset)`); `confidence` is bumped to the max
    /// of the two values.
    /// Thin metrics-tracking wrapper over `KnowledgeGraph::add_entity`.
    /// Since `add_entity` itself dedupes by id and merges mentions in
    /// place, this only counts "was this a new node vs a merge into
    /// an existing one" so callers (extend_graph) can report
    /// `new_entities` vs `mentions_merged` accurately.
    fn merge_entity(
        graph: &mut KnowledgeGraph,
        new_entity: Entity,
        metrics: &mut ExtractMetrics,
    ) -> Result<()> {
        let id = new_entity.id.clone();
        let was_existing = graph.get_entity(&id).is_some();
        let mentions_to_merge = if was_existing {
            // Count the mentions that ARE new vs the existing ones,
            // before add_entity does the merge. We only care about
            // the count here; the merge happens inside add_entity.
            let existing_mentions: Vec<_> = graph
                .get_entity(&id)
                .map(|e| {
                    e.mentions
                        .iter()
                        .map(|m| (m.chunk_id.clone(), m.start_offset))
                        .collect()
                })
                .unwrap_or_default();
            new_entity
                .mentions
                .iter()
                .filter(|m| {
                    !existing_mentions
                        .iter()
                        .any(|(c, off)| c == &m.chunk_id && *off == m.start_offset)
                })
                .count()
        } else {
            0
        };
        graph.add_entity(new_entity)?;
        if was_existing {
            metrics.mentions_merged += mentions_to_merge;
        } else {
            metrics.new_entities += 1;
        }
        Ok(())
    }

    /// Thin metrics-tracking wrapper over `KnowledgeGraph::add_relationship`.
    /// `add_relationship` itself dedupes by `(source, target,
    /// relation_type)` and silently ignores missing-endpoint errors
    /// (matching build_graph's existing behavior — the relationship's
    /// target may have been extracted from a chunk that hasn't been
    /// processed yet). This wrapper just counts how many were genuinely
    /// new for telemetry by scanning before the add.
    fn merge_relationship(
        graph: &mut KnowledgeGraph,
        relationship: Relationship,
        metrics: &mut ExtractMetrics,
    ) {
        let was_existing = graph.relationships().any(|r| {
            r.source == relationship.source
                && r.target == relationship.target
                && r.relation_type == relationship.relation_type
        });
        if graph.add_relationship(relationship).is_ok() && !was_existing {
            metrics.new_relationships += 1;
        }
        // add_relationship error (missing endpoint) is intentionally
        // ignored — see method docs.
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

        if self.config.ollama.enabled {
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

        // If Ollama is enabled, generate semantic answer using LLM
        if self.config.ollama.enabled {
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
        let answer = if self.config.ollama.enabled {
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

    // =====================================================================
    // LightRAG-style dual-level retrieval
    //
    // Reference: "LightRAG: Simple and Fast Retrieval-Augmented Generation"
    //   (Guo, Wang, Lin, Hu, Bei, Chen, Liao, Lu, Zhang, Yan, Lu;
    //   arXiv:2410.05779, 2024)
    //
    // The paper's core idea: instead of MS GraphRAG's expensive
    // community-detection-and-LLM-summary index step, LightRAG keeps
    // index-time work to entity + relation extraction (with descriptions),
    // embeds both, and shifts intelligence to query time. Each query is
    // decomposed into TWO keyword sets via one LLM call:
    //
    //   - low_level_keywords: specific named entities, attributes,
    //     properties mentioned in the question
    //   - high_level_keywords: themes, concepts, abstract topics
    //
    // The two sets feed two retrieval streams:
    //   - low_level → vector-search the entity store → seed entities →
    //     1-hop expand → mentioning chunks
    //   - high_level → vector-search the relation store → seed
    //     relationships → resolve endpoints → mentioning chunks
    //
    // The four LightRAG modes are characterized by which streams run:
    //   - naive   : neither (chunk-vector RAG only — same as `ask`)
    //   - local   : low-level only (entities)  (== `ask_with_seed_entities`)
    //   - global  : high-level only (relations)
    //   - hybrid  : both streams, merged
    //   - mix     : hybrid + chunk-vector merged
    //
    // graphrag-core exposes one unified API for all of these via
    // `ask_with_dual_seeds` — the caller (e.g. graphrag-server) is
    // responsible for the vector-search step (it owns the embedding
    // service and the Qdrant sidecars); graphrag-core does the
    // graph expansion, context assembly, and LLM call.
    // =====================================================================

    /// LightRAG dual-level keyword extraction. Returns the two keyword
    /// sets the paper specifies, packaged for easy use as embedding
    /// inputs by the caller.
    #[cfg(feature = "async")]
    pub async fn extract_query_keywords(&self, query: &str) -> Result<QueryKeywords> {
        use crate::ollama::OllamaClient;

        let client = OllamaClient::new(self.config.ollama.clone());

        let prompt = format!(
            "---Goal---\n\
             Given a user query, extract two sets of keywords:\n\
             - low_level_keywords: specific named entities, attributes, or details mentioned in the query (people, products, places, concrete things)\n\
             - high_level_keywords: overarching themes, concepts, or topics the query is about (abstractions, categories, processes)\n\n\
             ---Format---\n\
             Output ONLY a single valid JSON object, nothing else (no prose, no markdown, no explanation):\n\
             {{\"high_level_keywords\": [\"...\", \"...\"], \"low_level_keywords\": [\"...\", \"...\"]}}\n\n\
             Each list should have 1-5 keywords. Empty lists are allowed when nothing matches.\n\n\
             ---Query---\n\
             {}\n\n\
             ---Output---",
            query
        );

        let max_answer_tokens: u32 = 256;
        let prompt_tokens = (prompt.len() / 4) as u32;
        let total = prompt_tokens + max_answer_tokens;
        let with_margin = (total as f32 * 1.20) as u32;
        let num_ctx = (((with_margin + 1023) / 1024) * 1024).max(2048).min(16384);
        let params = crate::ollama::OllamaGenerationParams {
            num_predict: Some(max_answer_tokens),
            temperature: Some(0.1), // low temperature; we want deterministic JSON
            num_ctx: Some(num_ctx),
            keep_alive: self.config.ollama.keep_alive.clone(),
            ..Default::default()
        };

        let raw = client
            .generate_with_params(&prompt, params)
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("dual-keyword extraction LLM call failed: {}", e),
            })?;

        let stripped = Self::remove_thinking_tags(&raw);
        // Accept either a bare JSON object or one wrapped in ```json fences.
        let cleaned = stripped
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        // Find the first { and last } and parse just that — robust to
        // models that prepend explanatory prose despite our "ONLY JSON"
        // instruction.
        let json_slice = match (cleaned.find('{'), cleaned.rfind('}')) {
            (Some(a), Some(b)) if b > a => &cleaned[a..=b],
            _ => cleaned,
        };

        match serde_json::from_str::<QueryKeywordsWire>(json_slice) {
            Ok(wire) => Ok(QueryKeywords {
                low_level: wire.low_level_keywords.unwrap_or_default(),
                high_level: wire.high_level_keywords.unwrap_or_default(),
            }),
            Err(e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    error = %e,
                    raw_response = %raw.chars().take(200).collect::<String>(),
                    "dual-keyword JSON parse failed; returning empty keyword sets so caller can fall back"
                );
                let _ = e;
                Ok(QueryKeywords::default())
            },
        }
    }

    /// LightRAG-style retrieval over an arbitrary mix of seed populations
    /// — entities, relationships, and/or text chunks — combined into one
    /// LLM-answered context.
    ///
    /// The four LightRAG retrieval modes are all expressible by varying
    /// which fields of `seeds` are non-empty:
    ///
    /// | LightRAG mode | seeds.entities | seeds.relations | seeds.chunks |
    /// |---|---|---|---|
    /// | local   | non-empty | empty | empty |
    /// | global  | empty | non-empty | empty |
    /// | hybrid  | non-empty | non-empty | empty |
    /// | mix     | non-empty | non-empty | non-empty |
    ///
    /// All four expand entity seeds to 1-hop neighbors (capped by
    /// `max_neighbors_per_seed`), resolve relation seeds to their
    /// source/target entities (and pull those endpoints' neighbors
    /// too), gather every mentioning chunk for every entity touched,
    /// merge the chunk seeds in, and feed the assembled
    /// ENTITIES / RELATIONSHIPS / SOURCE TEXT block to the chat
    /// backend — the same prompt skeleton `ask_with_seed_entities`
    /// uses, so output style stays consistent across modes.
    ///
    /// graphrag-core does not own the seeding step — the caller is
    /// responsible for vector-searching the entity / relationship /
    /// chunk stores and producing the seed sets. graphrag-server's
    /// Qdrant sidecars (`{coll}-entities`, `{coll}-relationships`,
    /// `{coll}` for chunks) are the canonical seeding surface.
    #[cfg(feature = "async")]
    pub async fn ask_with_dual_seeds(
        &self,
        query: &str,
        seeds: &DualSeeds,
        max_neighbors_per_seed: usize,
    ) -> Result<retrieval::ExplainedAnswer> {
        use crate::ollama::OllamaClient;
        use std::collections::{HashMap, HashSet};

        let kg = self.knowledge_graph.as_ref().ok_or_else(|| {
            GraphRAGError::Config { message: "Knowledge graph not initialized".to_string() }
        })?;

        let mut entity_set: HashMap<EntityId, Entity> = HashMap::new();
        let mut chunk_ids: HashSet<ChunkId> = HashSet::new();
        let mut bridge_rels: Vec<(String, String, String)> = Vec::new();

        // ---- Stream 1: entity seeds (low-level / local) ---------------
        for seed_id in &seeds.entities {
            if let Some(seed) = kg.get_entity(seed_id) {
                let seed_name = seed.name.clone();
                entity_set.insert(seed_id.clone(), seed.clone());
                for m in &seed.mentions {
                    chunk_ids.insert(m.chunk_id.clone());
                }
                let neighbors = kg.get_neighbors(seed_id);
                for (neighbor, rel) in neighbors.into_iter().take(max_neighbors_per_seed) {
                    bridge_rels.push((
                        seed_name.clone(),
                        rel.relation_type.clone(),
                        neighbor.name.clone(),
                    ));
                    if !entity_set.contains_key(&neighbor.id) {
                        entity_set.insert(neighbor.id.clone(), neighbor.clone());
                        for m in &neighbor.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                    }
                }
            }
        }

        // ---- Stream 2: relation seeds (high-level / global) -----------
        for (src_id, tgt_id, relation_type) in &seeds.relations {
            // Record the bridge triple even if endpoints are missing,
            // so the LLM sees the relation. Use stored ids as fallback names.
            let src_name = kg
                .get_entity(src_id)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| src_id.0.clone());
            let tgt_name = kg
                .get_entity(tgt_id)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| tgt_id.0.clone());
            bridge_rels.push((src_name, relation_type.clone(), tgt_name));

            for endpoint_id in [src_id, tgt_id] {
                if let Some(endpoint) = kg.get_entity(endpoint_id) {
                    if !entity_set.contains_key(&endpoint.id) {
                        entity_set.insert(endpoint.id.clone(), endpoint.clone());
                        for m in &endpoint.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                        // Also expand 1-hop from each relation endpoint.
                        let neighbors = kg.get_neighbors(endpoint_id);
                        for (neighbor, rel) in
                            neighbors.into_iter().take(max_neighbors_per_seed)
                        {
                            bridge_rels.push((
                                endpoint.name.clone(),
                                rel.relation_type.clone(),
                                neighbor.name.clone(),
                            ));
                            if !entity_set.contains_key(&neighbor.id) {
                                entity_set.insert(neighbor.id.clone(), neighbor.clone());
                                for m in &neighbor.mentions {
                                    chunk_ids.insert(m.chunk_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- Stream 3: direct chunk seeds (mix) -----------------------
        for cid in &seeds.chunks {
            chunk_ids.insert(cid.clone());
        }

        // ---- Build context block -------------------------------------
        let entities_block = if entity_set.is_empty() {
            "(no entities resolved in graph)".to_string()
        } else {
            entity_set
                .values()
                .map(|e| {
                    format!(
                        "- {} (type={}, mentioned_in={} chunks, confidence={:.2})",
                        e.name,
                        e.entity_type,
                        e.mentions.len(),
                        e.confidence
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Dedupe bridge_rels in case a triple appears via both streams.
        let mut seen_rels: HashSet<(String, String, String)> = HashSet::new();
        bridge_rels.retain(|t| seen_rels.insert(t.clone()));

        let relationships_block = if bridge_rels.is_empty() {
            "(no relationships gathered)".to_string()
        } else {
            bridge_rels
                .iter()
                .map(|(s, r, t)| format!("- {} --[{}]--> {}", s, r, t))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let chunks_block = chunk_ids
            .iter()
            .filter_map(|cid| kg.chunks().find(|c| c.id == *cid))
            .map(|c| format!("- {}", c.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let context = format!(
            "ENTITIES:\n{}\n\nRELATIONSHIPS:\n{}\n\nSOURCE TEXT:\n{}",
            entities_block, relationships_block, chunks_block,
        );

        // ---- LLM call -------------------------------------------------
        let client = OllamaClient::new(self.config.ollama.clone());

        let prompt = format!(
            "You are a knowledgeable assistant answering questions grounded in a knowledge graph.\n\n\
             IMPORTANT INSTRUCTIONS:\n\
             - Answer ONLY using the provided entities, relationships, and source text below\n\
             - Synthesize across all three sections; relationships in particular often supply the connective tissue\n\
             - Provide direct, conversational, natural responses\n\
             - Do NOT show your reasoning process or use <think> tags\n\
             - If the context lacks sufficient information, clearly state: \"I don't have enough information to answer this question.\"\n\
             - Aim for a complete answer (3-6 sentences)\n\n\
             CONTEXT:\n\
             {}\n\n\
             QUESTION: {}\n\n\
             ANSWER (direct response only, no reasoning):",
            context, query
        );

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

        let raw_answer = client
            .generate_with_params(&prompt, params)
            .await
            .map_err(|e| GraphRAGError::Generation {
                message: format!("LLM generation failed: {}", e),
            })?;
        let answer = Self::remove_thinking_tags(&raw_answer).trim().to_string();

        // ---- Pack ExplainedAnswer ------------------------------------
        let total_seeds =
            seeds.entities.len() + seeds.relations.len() + seeds.chunks.len();
        let confidence = if total_seeds == 0 {
            0.0
        } else {
            let chunk_ratio = (chunk_ids.len() as f32) / (total_seeds as f32 * 2.0);
            (chunk_ratio.min(1.0) * 0.7 + 0.3).min(1.0)
        };

        let mut sources: Vec<retrieval::SourceReference> = Vec::new();
        for cid in chunk_ids.iter().take(12) {
            if let Some(chunk) = kg.chunks().find(|c| c.id == *cid) {
                sources.push(retrieval::SourceReference {
                    id: cid.0.clone(),
                    source_type: retrieval::SourceType::TextChunk,
                    excerpt: chunk.content.chars().take(160).collect(),
                    relevance_score: confidence,
                });
            }
        }
        for (s, r, t) in bridge_rels.iter().take(10) {
            sources.push(retrieval::SourceReference {
                id: format!("{} --[{}]--> {}", s, r, t),
                source_type: retrieval::SourceType::Relationship,
                excerpt: format!("{} {} {}", s, r, t),
                relevance_score: 0.5,
            });
        }

        let key_entities: Vec<String> =
            entity_set.values().map(|e| e.name.clone()).collect();

        let mode_label = match (
            !seeds.entities.is_empty(),
            !seeds.relations.is_empty(),
            !seeds.chunks.is_empty(),
        ) {
            (true, false, false) => "local",
            (false, true, false) => "global",
            (true, true, false) => "hybrid",
            (true, true, true) | (false, _, true) | (true, false, true) => "mix",
            (false, false, false) => "empty",
        };

        let reasoning_steps = vec![
            retrieval::ReasoningStep {
                step_number: 1,
                description: format!(
                    "LightRAG {} mode: caller supplied {} entity seeds, {} relation seeds, {} chunk seeds",
                    mode_label,
                    seeds.entities.len(),
                    seeds.relations.len(),
                    seeds.chunks.len()
                ),
                entities_used: seeds.entities.iter().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 1.0,
            },
            retrieval::ReasoningStep {
                step_number: 2,
                description: format!(
                    "Expanded each seed to 1-hop neighbors (max {} per seed); gathered {} entities total, {} relationship bridges",
                    max_neighbors_per_seed,
                    entity_set.len(),
                    bridge_rels.len()
                ),
                entities_used: entity_set.keys().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 0.9,
            },
            retrieval::ReasoningStep {
                step_number: 3,
                description: format!(
                    "Collected {} mentioning chunks across all gathered entities and direct chunk seeds",
                    chunk_ids.len()
                ),
                entities_used: vec![],
                evidence_snippet: None,
                confidence: 0.85,
            },
            retrieval::ReasoningStep {
                step_number: 4,
                description: "Sent assembled entities + relationships + source text to chat backend".to_string(),
                entities_used: vec![],
                evidence_snippet: None,
                confidence,
            },
        ];

        Ok(retrieval::ExplainedAnswer {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        })
    }

    /// Microsoft GraphRAG `local_search`-style query: answer the
    /// question by seeding retrieval from a caller-supplied set of
    /// entity ids (typically obtained by vector-searching an
    /// entity-description embedding store), expanding to their
    /// 1-hop neighbors via the relationship graph, gathering the
    /// chunks that mention any of those entities, and feeding the
    /// assembled context to the chat backend.
    ///
    /// Mirrors the seed-traverse-summarize shape of MS GraphRAG's
    /// `local_search` — but the seeding step is the caller's
    /// responsibility (graphrag-core does not own the entity
    /// vector store; graphrag-server's Qdrant sidecar is one such
    /// store).
    ///
    /// `max_neighbors_per_seed` caps fanout so a high-degree entity
    /// doesn't explode the context. Typical value: 3-5.
    ///
    /// Returns an `ExplainedAnswer` with seeds + neighbors as
    /// key entities, mentioning chunks + relationship triples as
    /// sources, and a confidence heuristic over the count of
    /// grounded chunks vs. seed count.
    #[cfg(feature = "async")]
    pub async fn ask_with_seed_entities(
        &self,
        query: &str,
        seed_entity_ids: &[EntityId],
        max_neighbors_per_seed: usize,
    ) -> Result<retrieval::ExplainedAnswer> {
        use crate::ollama::OllamaClient;
        use std::collections::{HashMap, HashSet};

        let kg = self.knowledge_graph.as_ref().ok_or_else(|| {
            GraphRAGError::Config { message: "Knowledge graph not initialized".to_string() }
        })?;

        // 1. Seed expansion: collect seed entities + 1-hop neighbors,
        //    plus the relationship triples that bridge them.
        let mut entity_set: HashMap<EntityId, Entity> = HashMap::new();
        let mut chunk_ids: HashSet<ChunkId> = HashSet::new();
        let mut bridge_rels: Vec<(String, String, String)> = Vec::new();

        for seed_id in seed_entity_ids {
            if let Some(seed) = kg.get_entity(seed_id) {
                let seed_name = seed.name.clone();
                entity_set.insert(seed_id.clone(), seed.clone());
                for m in &seed.mentions {
                    chunk_ids.insert(m.chunk_id.clone());
                }

                let neighbors = kg.get_neighbors(seed_id);
                for (neighbor, rel) in neighbors.into_iter().take(max_neighbors_per_seed) {
                    bridge_rels.push((
                        seed_name.clone(),
                        rel.relation_type.clone(),
                        neighbor.name.clone(),
                    ));
                    if !entity_set.contains_key(&neighbor.id) {
                        entity_set.insert(neighbor.id.clone(), neighbor.clone());
                        for m in &neighbor.mentions {
                            chunk_ids.insert(m.chunk_id.clone());
                        }
                    }
                }
            }
        }

        // 2. Build the MS-style context block.
        let entities_block = if entity_set.is_empty() {
            "(no seed entities resolved in graph)".to_string()
        } else {
            entity_set
                .values()
                .map(|e| {
                    format!(
                        "- {} (type={}, mentioned_in={} chunks, confidence={:.2})",
                        e.name, e.entity_type, e.mentions.len(), e.confidence
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let relationships_block = if bridge_rels.is_empty() {
            "(no relationships gathered)".to_string()
        } else {
            bridge_rels
                .iter()
                .map(|(s, r, t)| format!("- {} --[{}]--> {}", s, r, t))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let chunks_block = chunk_ids
            .iter()
            .filter_map(|cid| kg.chunks().find(|c| c.id == *cid))
            .map(|c| format!("- {}", c.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let context = format!(
            "ENTITIES:\n{}\n\nRELATIONSHIPS:\n{}\n\nSOURCE TEXT:\n{}",
            entities_block, relationships_block, chunks_block,
        );

        // 3. LLM call. Same prompt skeleton + thinking-tag hygiene as
        //    `generate_semantic_answer_from_results`. Ollama-only
        //    primitives match the rest of this branch (PR C);
        //    the `ChatClient`-via-PR-B variant lands when this
        //    method graduates to the openai-compat fork.
        let client = OllamaClient::new(self.config.ollama.clone());

        let prompt = format!(
            "You are a knowledgeable assistant answering questions grounded in a knowledge graph.\n\n\
             IMPORTANT INSTRUCTIONS:\n\
             - Answer ONLY using the provided entities, relationships, and source text below\n\
             - Synthesize across all three sections; relationships in particular often supply the connective tissue\n\
             - Provide direct, conversational, natural responses\n\
             - Do NOT show your reasoning process or use <think> tags\n\
             - If the context lacks sufficient information, clearly state: \"I don't have enough information to answer this question.\"\n\
             - Aim for a complete answer (3-6 sentences)\n\n\
             CONTEXT:\n\
             {}\n\n\
             QUESTION: {}\n\n\
             ANSWER (direct response only, no reasoning):",
            context, query
        );

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

        let raw_answer = client.generate_with_params(&prompt, params).await.map_err(|e| {
            GraphRAGError::Generation { message: format!("LLM generation failed: {}", e) }
        })?;
        let answer = Self::remove_thinking_tags(&raw_answer).trim().to_string();

        // 4. Pack ExplainedAnswer.
        let confidence = if seed_entity_ids.is_empty() {
            0.0
        } else {
            let chunk_ratio = (chunk_ids.len() as f32) / (seed_entity_ids.len() as f32 * 3.0);
            (chunk_ratio.min(1.0) * 0.7 + 0.3).min(1.0)
        };

        let mut sources: Vec<retrieval::SourceReference> = Vec::new();
        for cid in chunk_ids.iter().take(10) {
            if let Some(chunk) = kg.chunks().find(|c| c.id == *cid) {
                sources.push(retrieval::SourceReference {
                    id: cid.0.clone(),
                    source_type: retrieval::SourceType::TextChunk,
                    excerpt: chunk.content.chars().take(160).collect(),
                    relevance_score: confidence,
                });
            }
        }
        for (s, r, t) in bridge_rels.iter().take(8) {
            sources.push(retrieval::SourceReference {
                id: format!("{} --[{}]--> {}", s, r, t),
                source_type: retrieval::SourceType::Relationship,
                excerpt: format!("{} {} {}", s, r, t),
                relevance_score: 0.5,
            });
        }

        let key_entities: Vec<String> = entity_set.values().map(|e| e.name.clone()).collect();

        let reasoning_steps = vec![
            retrieval::ReasoningStep {
                step_number: 1,
                description: format!(
                    "Seeded local_search with {} entity ids supplied by caller (vector-search top-K)",
                    seed_entity_ids.len()
                ),
                entities_used: seed_entity_ids.iter().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 1.0,
            },
            retrieval::ReasoningStep {
                step_number: 2,
                description: format!(
                    "Expanded to 1-hop neighbors (max {} per seed); gathered {} entities total, {} relationship bridges",
                    max_neighbors_per_seed, entity_set.len(), bridge_rels.len()
                ),
                entities_used: entity_set.keys().map(|e| e.0.clone()).collect(),
                evidence_snippet: None,
                confidence: 0.9,
            },
            retrieval::ReasoningStep {
                step_number: 3,
                description: format!(
                    "Collected {} mentioning chunks across all gathered entities",
                    chunk_ids.len()
                ),
                entities_used: vec![],
                evidence_snippet: None,
                confidence: 0.85,
            },
            retrieval::ReasoningStep {
                step_number: 4,
                description: "Sent assembled entities + relationships + source text to chat backend".to_string(),
                entities_used: vec![],
                evidence_snippet: None,
                confidence,
            },
        ];

        Ok(retrieval::ExplainedAnswer {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        })
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
        use crate::ollama::OllamaClient;

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

        // Create Ollama client
        let client = OllamaClient::new(self.config.ollama.clone());

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

    // ─────────────────────────────────────────────────────────────────
    // extend_graph tests — pattern-based extraction so we don't need a
    // live LLM. The pattern path is deterministic (regex + capitalized-
    // word rules), runs synchronously inside the async fn, and is the
    // path that exercises every dedup-on-merge code path in
    // `merge_entity` / `merge_relationship`.
    //
    // Strategy: feed N documents → build_graph → snapshot graph → add
    // M more documents → extend_graph → assert that:
    //   1. extend_graph processed exactly M chunks (the delta).
    //   2. Re-extending without new content is a true no-op.
    //   3. Entities re-mentioned in delta chunks have their `mentions`
    //      extended in place — no duplicate node, mentions_merged is
    //      counted correctly.
    //   4. Total entity count after extend matches what a clean
    //      `build_graph` over the same corpus would produce.
    // ─────────────────────────────────────────────────────────────────

    /// Configure pattern-based (no-LLM) GraphRAG with deterministic
    /// chunking. `chunk_size = 1000` keeps each test doc as one chunk.
    fn pattern_config() -> Config {
        let mut cfg = Config::default();
        cfg.suppress_progress_bars = true;
        // Ollama disabled and gliner disabled → falls through to the
        // pattern-based extractor in build_graph and extend_graph.
        // No LLM round-trips; tests run deterministically.
        cfg.ollama.enabled = false;
        cfg.entities.use_gleaning = false;
        cfg.gliner.enabled = false;
        cfg.text.chunk_size = 1000;
        cfg.text.chunk_overlap = 0;
        cfg.graph.extract_relationships = true;
        cfg
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn extend_graph_no_new_chunks_is_a_fast_noop() {
        let cfg = pattern_config();
        let mut g = GraphRAG::new(cfg).expect("GraphRAG::new");
        g.initialize().expect("initialize");
        g.add_document_from_text(
            "Alice Smith works at Acme Corp in New York. Bob Jones manages the team.",
        )
        .expect("add_document_from_text");
        g.build_graph().await.expect("build_graph");

        let entities_before = g.knowledge_graph().unwrap().entities().count();
        let relationships_before = g.knowledge_graph().unwrap().relationships().count();
        let processed_before = g.processed_chunk_count();

        let summary = g.extend_graph().await.expect("extend_graph (no-op)");

        assert_eq!(summary.chunks_processed, 0, "no-op should report 0 chunks");
        assert_eq!(summary.new_entities, 0);
        assert_eq!(summary.new_relationships, 0);
        assert_eq!(summary.mentions_merged, 0);
        assert_eq!(summary.total_entities, entities_before);
        assert_eq!(summary.total_relationships, relationships_before);
        assert_eq!(g.processed_chunk_count(), processed_before);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn extend_graph_processes_only_delta_chunks() {
        let cfg = pattern_config();
        let mut g = GraphRAG::new(cfg).expect("GraphRAG::new");
        g.initialize().expect("initialize");

        // Initial corpus: 1 doc → 1 chunk.
        g.add_document_from_text("Alice Smith works at Acme Corp. Bob Jones works at Acme Corp.")
            .expect("doc 1");
        g.build_graph().await.expect("build_graph");
        let processed_after_build = g.processed_chunk_count();
        assert_eq!(processed_after_build, 1, "1 chunk after first build");

        // Add a second doc → second chunk.
        g.add_document_from_text(
            "Charlie Brown lives in Chicago. Charlie Brown works at Globex Corp.",
        )
        .expect("doc 2");
        // Extend should only walk the new chunk, not the first.
        let summary = g.extend_graph().await.expect("extend_graph");

        assert_eq!(
            summary.chunks_processed, 1,
            "extend should walk exactly the 1 delta chunk"
        );
        assert_eq!(g.processed_chunk_count(), 2);
        // The pattern extractor pulls capitalized multi-word names; we
        // assert at least the delta-doc-introduced entities (Charlie
        // Brown, Chicago, Globex Corp) registered. Exact counts depend
        // on regex specifics — assert "more than zero" and "summary
        // adds up to graph totals".
        assert!(
            summary.new_entities + summary.mentions_merged > 0,
            "extend should add or merge at least one entity from the delta chunk"
        );
        assert_eq!(
            summary.total_entities,
            g.knowledge_graph().unwrap().entities().count()
        );
        assert_eq!(
            summary.total_relationships,
            g.knowledge_graph().unwrap().relationships().count()
        );
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn extend_graph_dedupes_entities_by_id() {
        // The first doc mentions "Acme Corp"; a second doc re-mentions
        // it. extend_graph should merge the new mention into the
        // existing entity rather than creating a duplicate node.
        let cfg = pattern_config();
        let mut g = GraphRAG::new(cfg).expect("GraphRAG::new");
        g.initialize().expect("initialize");

        g.add_document_from_text(
            "Alice Smith works at Acme Corp in New York. Bob Jones manages Acme Corp.",
        )
        .expect("doc 1");
        g.build_graph().await.expect("build_graph");

        let entities_after_build = g.knowledge_graph().unwrap().entities().count();
        // Find the "Acme Corp"-shaped entity (pattern extractor lowercases
        // names into the id; we just look up by name).
        let acme_node_count_before = g
            .knowledge_graph()
            .unwrap()
            .entities()
            .filter(|e| e.name.to_lowercase().contains("acme"))
            .count();

        // Delta: another doc that re-mentions "Acme Corp".
        g.add_document_from_text("Diana Wilson recently joined Acme Corp.")
            .expect("doc 2");
        let summary = g.extend_graph().await.expect("extend_graph");

        // After extend: Acme should still have exactly the same number
        // of node entries as before (no duplicate). The mention from
        // the delta chunk should have been merged in place.
        let acme_node_count_after = g
            .knowledge_graph()
            .unwrap()
            .entities()
            .filter(|e| e.name.to_lowercase().contains("acme"))
            .count();

        assert_eq!(
            acme_node_count_after, acme_node_count_before,
            "extend_graph must not duplicate Acme Corp; expected {} got {}",
            acme_node_count_before, acme_node_count_after
        );
        // Total entity count grew by exactly summary.new_entities
        // (Diana Wilson is the new one; Acme is merged not added).
        assert_eq!(
            g.knowledge_graph().unwrap().entities().count(),
            entities_after_build + summary.new_entities
        );
        // And the merge is reflected in the summary counter.
        assert!(
            summary.mentions_merged > 0 || summary.new_entities > 0,
            "delta chunk introduced something; summary must reflect it: {:?}",
            summary,
        );
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn extend_graph_after_clear_processed_re_extracts_everything() {
        // clear_processed_chunks() resets the tracking set so the next
        // extend call walks every chunk again. Useful after a config
        // change (entity_types, prompts) where the user wants to
        // re-extract without wiping the graph first.
        let cfg = pattern_config();
        let mut g = GraphRAG::new(cfg).expect("GraphRAG::new");
        g.initialize().expect("initialize");
        g.add_document_from_text("Alice Smith works at Acme Corp.")
            .expect("doc 1");
        g.build_graph().await.expect("build_graph");
        assert_eq!(g.processed_chunk_count(), 1);

        // After clear, the same chunk is "new" again.
        g.clear_processed_chunks();
        assert_eq!(g.processed_chunk_count(), 0);

        let summary = g.extend_graph().await.expect("re-extend after clear");
        assert_eq!(
            summary.chunks_processed, 1,
            "after clear_processed_chunks, extend re-walks the chunk"
        );
        assert_eq!(g.processed_chunk_count(), 1);
    }
}
