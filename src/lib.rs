//! # GraphRAG-RS
//!
//! A high-performance, modular Rust implementation of GraphRAG (Graph-based Retrieval Augmented Generation)
//! for building knowledge graphs from documents and performing intelligent semantic search.
//!
//! ## Features
//!
//! - **Trait-first architecture** for maximum modularity and testability
//! - **Feature-gated dependencies** for minimal compilation footprint
//! - **Parallel processing** support for high-performance document processing
//! - **Multiple LLM providers** (Ollama, OpenAI coming soon)
//! - **Flexible storage backends** (memory, persistent)
//! - **Advanced vector similarity search** with HNSW indexing
//! - **Hierarchical document summarization**
//! - **Function calling capabilities** for tool usage
//! - **Comprehensive monitoring and metrics**
//!
//! ## Quick Start
//!
//! ### New Simplified API (2-3 lines)
//!
//! ```rust
//! use graphrag_rs::GraphRAG;
//!
//! # async fn example() -> graphrag_rs::Result<()> {
//! // Auto-detect LLM provider and initialize with optimal defaults
//! let mut graphrag = GraphRAG::builder()
//!     .auto_detect_llm()
//!     .build()?;
//!
//! // Query the system
//! let answer = graphrag.query("What is...")?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Builder Usage
//!
//! ```rust
//! use graphrag_rs::{GraphRAG, ConfigPreset, LLMProvider};
//!
//! # fn example() -> graphrag_rs::Result<()> {
//! // Production setup with custom configuration
//! let mut graphrag = GraphRAG::builder()
//!     .with_preset(ConfigPreset::Production)
//!     .with_llm_provider(LLMProvider::Mock)
//!     .with_parallel_processing(true, Some(8))
//!     .with_text_config(1000, 200)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Legacy API (still supported)
//!
//! ```rust
//! use graphrag_rs::{GraphRAG, Config};
//!
//! # fn example() -> graphrag_rs::Result<()> {
//! let config = Config::default();
//! let mut graphrag = GraphRAG::new(config)?;
//! graphrag.initialize()?;
//! # Ok(())
//! # }
//! ```

// ================================
// MODULE DECLARATIONS
// ================================

pub mod api;
pub mod automatic_entity_linking;
pub mod builder;
pub mod config;
pub mod core;
pub mod corpus;
pub mod entity;
pub mod generation;
pub mod graph;
pub mod inference;
pub mod lightrag;
pub mod monitoring;
pub mod parallel;

#[cfg(feature = "neural-embeddings")]
pub mod embeddings;
pub mod phase_saver;
pub mod query;
pub mod reranking;
pub mod retrieval;
pub mod summarization;
pub mod text;
pub mod vector;

// Feature-gated modules
#[cfg(feature = "function-calling")]
pub mod function_calling;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "ollama")]
pub mod async_processing;

#[cfg(feature = "caching")]
pub mod caching;

#[cfg(feature = "caching")]
pub mod caching_test;

#[cfg(feature = "rograg")]
pub mod rograg;

// Storage module (conditionally compiled)
#[cfg(any(feature = "memory-storage", feature = "persistent-storage"))]
pub mod storage;

// Test utilities (available in test builds and when testing)
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

// ================================
// PRELUDE AND SIMPLIFIED EXPORTS
// ================================

/// Prelude module containing the most commonly used types
pub mod prelude {
    // Only the essentials for basic usage
    pub use crate::{GraphRAG, Result};
    pub use crate::builder::{GraphRAGBuilder, ConfigPreset, LLMProvider};
    pub use crate::core::{Document, DocumentId};

    // Simplified API
    pub use crate::api::simple::*;
    pub use crate::api::easy::SimpleGraphRAG;
}

/// Simple API for beginners (one-function interface)
pub mod simple {
    pub use crate::api::simple::*;
}

/// Easy API for basic stateful usage
pub mod easy {
    pub use crate::api::easy::*;
}

/// Builder API for intermediate users
pub mod build {
    pub use crate::builder::*;
}

/// Advanced API with full trait system for extensibility
pub mod advanced {
    pub use crate::core::traits::*;

    #[cfg(any(feature = "memory-storage", feature = "persistent-storage"))]
    pub use crate::storage::*;

    pub use crate::retrieval::{RetrievalSystem};
    pub use crate::generation::{AnswerGenerator, GeneratedAnswer, GenerationConfig, MockLLM};
}

// ================================
// FLAT PUBLIC API EXPORTS (for backward compatibility)
// ================================

// Core types and errors (for backward compatibility)
#[deprecated(note = "Import from crate::prelude instead for common types")]
pub use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, ErrorContext, ErrorSeverity,
    GraphRAGError, KnowledgeGraph, RegistryBuilder, Relationship, Result, ServiceConfig,
    ServiceContext, ServiceRegistry, TextChunk,
};

// Core traits for extensibility
pub use crate::core::traits::{
    ConfigProvider, Embedder, EntityExtractor, FunctionRegistry, GenerationParams, GraphStats,
    GraphStore, LanguageModel, MetricsCollector, ModelInfo, Retriever, SearchResult, Serializer,
    Storage, Timer, VectorStore,
};

// Async traits (when async-traits feature is enabled)
#[cfg(feature = "async-traits")]
pub use crate::core::traits::{
    AsyncConfigProvider, AsyncEmbedder, AsyncEntityExtractor, AsyncFunctionRegistry,
    AsyncGraphStore, AsyncLanguageModel, AsyncMetricsCollector, AsyncRetriever, AsyncSerializer,
    AsyncStorage, AsyncTimer, AsyncVectorStore, BoxedAsyncEmbedder, BoxedAsyncLanguageModel,
    BoxedAsyncRetriever, BoxedAsyncVectorStore, FunctionInfo, MetricRecord, ModelUsageStats,
    ParameterInfo, RetrievalStats,
};

// Async utilities and adapters
#[cfg(feature = "async-traits")]
pub use crate::core::traits::{async_utils, sync_to_async};

// Configuration
pub use crate::config::Config;

// Builder pattern for simplified API
pub use crate::builder::{ConfigPreset, GraphRAGBuilder, LLMProvider};

// Entity extraction
pub use crate::entity::EntityExtractor as BasicEntityExtractor;

// Graph operations - already exported from core in line 79

// Query and retrieval
pub use crate::query::QueryExpander as QueryBuilder; // Fix: use existing type
pub use crate::retrieval::RetrievalSystem;

// Generation
pub use crate::generation::{
    AnswerGenerator, GeneratedAnswer, GenerationConfig, GeneratorStatistics, MockLLM,
};

// Async generation components
#[cfg(feature = "async-traits")]
pub use crate::generation::async_mock_llm::AsyncMockLLM;

// Async main system
#[cfg(feature = "async-traits")]
pub mod async_graphrag;

#[cfg(feature = "async-traits")]
pub use crate::async_graphrag::{
    AsyncGraphRAG, AsyncGraphRAGBuilder, AsyncHealthStatus, AsyncPerformanceStats,
};

// Summarization and hierarchical processing
pub use crate::summarization::{
    DocumentTree, HierarchicalConfig, QueryResult as SummarizationQueryResult,
};

// Monitoring and metrics
pub use crate::monitoring::{benchmark::BenchmarkConfig, metrics::PerformanceMetrics};

// Parallel processing
pub use crate::config::ParallelConfig;
pub use crate::parallel::{ParallelProcessor, ParallelStatistics};

// Text processing
pub use crate::text::TextProcessor;

// Vector operations
pub use crate::vector::VectorIndex;

// Feature-gated exports
#[cfg(feature = "function-calling")]
pub use crate::function_calling::{
    enhanced_registry::EnhancedToolRegistry as EnhancedFunctionRegistry, CallableFunction,
    FunctionCall, FunctionCallStatistics, FunctionCaller, FunctionContext, FunctionDefinition,
    FunctionResult,
};

#[cfg(feature = "ollama")]
pub use crate::ollama::{OllamaClient, OllamaConfig, OllamaEmbeddings as OllamaEmbedder};

// Async Ollama components
#[cfg(all(feature = "ollama", feature = "async-traits"))]
pub use crate::ollama::{AsyncOllamaGenerator, AsyncOllamaGeneratorBuilder};

// Async processing exports
#[cfg(feature = "ollama")]
pub use crate::async_processing::{AsyncConfig, AsyncGraphRAGCore, ProcessingResult as AsyncProcessingResult};


#[cfg(feature = "memory-storage")]
pub use crate::storage::MemoryStorage;

#[cfg(feature = "caching")]
pub use crate::caching::{
    CachedLLMClient, CacheConfig, CacheStatistics, CacheHealth, CacheMetrics,
    CacheKey, CacheKeyGenerator, EvictionPolicy, WarmingStrategy, WarmingConfig, CacheWarmer,
};

#[cfg(feature = "caching")]
pub use crate::caching::cache_key::KeyStrategy;

// PageRank feature exports
#[cfg(feature = "pagerank")]
pub use crate::graph::pagerank::{PageRankConfig, PersonalizedPageRank, ScoreWeights};
#[cfg(feature = "pagerank")]
pub use crate::retrieval::{PageRankRetrievalSystem, ScoredResult};

// Incremental updates feature exports
#[cfg(feature = "incremental")]
pub use crate::graph::incremental::{
    IncrementalGraphStore, IncrementalGraphManager, IncrementalConfig, IncrementalStatistics,
    ProductionGraphStore, ChangeRecord, ChangeType, Operation, ChangeData, GraphDelta,
    DeltaStatus, ConflictResolver, ConflictStrategy, Conflict, ConflictType, ConflictResolution,
    UpdateId, TransactionId, ChangeEvent, ChangeEventType, GraphStatistics, ConsistencyReport,
    InvalidationStrategy, CacheRegion, SelectiveInvalidation, UpdateMonitor, BatchProcessor,
    IncrementalPageRank, BatchMetrics,
};

// ROGRAG feature exports
#[cfg(feature = "rograg")]
pub use crate::rograg::{
    RogragProcessor, RogragProcessorBuilder, RogragConfig, RogragResponse,
    QueryDecomposer, HybridQueryDecomposer, SemanticQueryDecomposer, SyntacticQueryDecomposer,
    DecompositionResult, Subquery, SubqueryType, QueryDependency,
    FuzzyMatcher, FuzzyMatchResult, FuzzyMatch, MatchType, MatchStrategy,
    IntentClassifier, IntentResult, QueryIntent, IntentClassificationConfig,
    LogicFormRetriever, LogicFormQuery, LogicFormResult, Predicate, ArgumentType,
    StreamingResponseBuilder, StreamingConfig, SynthesisStrategy, ResponseChunk,
    QueryValidator, ValidationResult, ValidationIssue, IssueType, IssueSeverity,
    QualityMetricsConfig, QualityMetricsOptions, QueryMetrics, ResponseQuality, PerformanceStatistics,
    ComparativeAnalysis, HealthCheckResult, HealthStatus as RogragHealthStatus, SystemStatistics,
    SubqueryResult, SubqueryResultType, ProcessingStats,
};

// LightRAG exports
pub use crate::lightrag::{
    LightRAG, LightRAGConfig, ProcessingResult, RetrievalResult, RetrievalLevel,
    DualRetriever, RetrievalMode, FusionWeights,
    GraphIndexer, EntityReference, ConceptSummary,
    IncrementalGraphUpdater, UpdateStats,
};
pub use crate::lightrag::incremental_updater::ConflictStrategy as LightRAGConflictStrategy;

// LightRAG KV Store exports
pub use crate::lightrag::kv_store::{
    KVStore, KVData, EntityData, RelationData, ConceptData, MemoryKVStore,
};

// Redis KV Store (feature-gated)
#[cfg(feature = "redis-storage")]
pub use crate::lightrag::kv_store::RedisKVStore;

// Neural Embeddings exports (feature-gated)
#[cfg(feature = "neural-embeddings")]
pub use crate::embeddings::{
    EmbeddingManager, EmbeddingConfig, EmbeddingProviderType, Device, PoolingStrategy,
    EmbeddingDimensions, EmbeddingStats, HybridEmbedder, EmbeddingProvider,
};

#[cfg(feature = "neural-embeddings")]
pub use crate::embeddings::neural::{
    NeuralEmbedder, SentenceTransformer, ModelManager, PretrainedModel,
    EmbeddingCache, CacheEntry,
};

#[cfg(feature = "neural-embeddings")]
pub use crate::embeddings::neural::models::ModelInfo as NeuralModelInfo;

// Corpus Processing exports (Multi-Document Processing)
pub use crate::corpus::{
    CorpusProcessor, CorpusProcessingResult,
    DocumentManager, DocumentMetadata, DocumentCollection,
    CrossDocumentEntityLinker, EntityCluster, LinkingStrategy,
    CorpusKnowledgeGraph, GlobalEntity, GlobalRelation,
    CollectionProcessor, ProcessingPipeline, CorpusStats,
};

pub use crate::corpus::document_manager::DocumentType;
pub use crate::corpus::entity_linker::CrossDocumentEntity;

// PersistentStorage will be added when needed

// Test utilities (for integration tests)
#[cfg(any(test, feature = "test-utils"))]
pub use crate::test_utils::*;

// Remove this for now since HnswVectorStore doesn't exist yet

// ================================
// MAIN GRAPHRAG SYSTEM
// ================================

// These are handled by the public re-exports above

/// Main GraphRAG system
pub struct GraphRAG {
    config: Config,
    knowledge_graph: Option<KnowledgeGraph>,
    retrieval_system: Option<RetrievalSystem>,
    document_trees: std::collections::HashMap<core::DocumentId, DocumentTree>,
    hierarchical_config: HierarchicalConfig,
    answer_generator: Option<AnswerGenerator>,
    parallel_processor: Option<ParallelProcessor>,
}

impl GraphRAG {
    /// Create GraphRAG from a text string with auto-configuration
    pub fn from_text(text: &str) -> Result<Self> {
        let mut graph = Self::builder()
            .auto_detect_llm()
            .with_preset(ConfigPreset::Basic)
            .build()?;

        graph.add_document_from_text(text)?;
        Ok(graph)
    }

    /// Create GraphRAG from a file path with auto-configuration
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_text(&text)
    }

    /// One-shot question answering for simple use cases
    pub fn quick_answer(text: &str, question: &str) -> Result<String> {
        let mut graph = Self::from_text(text)?;
        graph.ask(question)
    }

    /// Create a new GraphRAG instance with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        let parallel_processor = if config.parallel.enabled {
            let mut processor = ParallelProcessor::new(config.parallel.clone());
            processor.initialize()?;
            Some(processor)
        } else {
            None
        };

        Ok(Self {
            config,
            knowledge_graph: None,
            retrieval_system: None,
            document_trees: std::collections::HashMap::new(),
            hierarchical_config: HierarchicalConfig::default(),
            answer_generator: None,
            parallel_processor,
        })
    }

    /// Create a new GraphRAG instance with custom hierarchical configuration
    pub fn with_hierarchical_config(
        config: Config,
        hierarchical_config: HierarchicalConfig,
    ) -> Result<Self> {
        let parallel_processor = if config.parallel.enabled {
            let mut processor = ParallelProcessor::new(config.parallel.clone());
            processor.initialize()?;
            Some(processor)
        } else {
            None
        };

        Ok(Self {
            config,
            knowledge_graph: None,
            retrieval_system: None,
            document_trees: std::collections::HashMap::new(),
            hierarchical_config,
            answer_generator: None,
            parallel_processor,
        })
    }

    /// Initialize the GraphRAG system
    pub fn initialize(&mut self) -> Result<()> {
        println!("Initializing GraphRAG system...");

        // Initialize components based on configuration
        self.knowledge_graph = Some(KnowledgeGraph::new());

        // Initialize retrieval system with or without parallel processing
        self.retrieval_system = if let Some(ref processor) = self.parallel_processor {
            Some(RetrievalSystem::with_parallel_processing(
                &self.config,
                processor.clone(),
            )?)
        } else {
            Some(RetrievalSystem::new(&self.config)?)
        };

        // Initialize answer generator with mock LLM
        let llm = Box::new(MockLLM::new()?);
        let generation_config = GenerationConfig::default();
        self.answer_generator = Some(AnswerGenerator::new(llm, generation_config)?);

        // Print parallel processing status
        if let Some(ref processor) = self.parallel_processor {
            let stats = processor.get_statistics();
            stats.print();
        }

        println!("GraphRAG system initialized successfully");
        Ok(())
    }

    /// Build the knowledge graph from documents
    pub fn build_graph(&mut self) -> Result<()> {
        use crate::automatic_entity_linking::{AutomaticEntityLinker, EntityLinkingConfig};
        use crate::entity::EntityExtractor;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        println!("Building knowledge graph...");

        // Create entity extractor with config
        let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

        // Extract entities from all chunks in the graph
        let chunks: Vec<_> = graph.chunks().cloned().collect();
        let mut total_entities = 0;

        // Phase 1: Extract raw entities using traditional approach
        println!("🔍 Phase 1: Extracting raw entities...");
        for chunk in &chunks {
            // Extract entities from this chunk
            let entities = extractor.extract_from_chunk(chunk)?;

            // Add entities to the graph and update chunk entity references
            let mut chunk_entity_ids = Vec::new();
            for entity in entities {
                chunk_entity_ids.push(entity.id.clone());
                graph.add_entity(entity)?;
                total_entities += 1;
            }

            // Update chunk with entity references
            if let Some(existing_chunk) = graph.get_chunk_mut(&chunk.id) {
                existing_chunk.entities = chunk_entity_ids;
            }
        }

        // Phase 2: Apply automatic entity linking for improvement
        println!("🤖 Phase 2: Applying automatic entity linking...");
        let mut auto_linker = AutomaticEntityLinker::new(EntityLinkingConfig::default());

        // Process chunks to build automatic mappings
        match auto_linker.process_chunks(&chunks) {
            Ok(mappings) => {
                println!(
                    "   Found {} entity mappings for deduplication",
                    mappings.len()
                );

                if !mappings.is_empty() {
                    // Apply mappings to improve entity quality
                    // Note: This is a simplified integration - in a full implementation,
                    // we would update the actual entities in the graph based on mappings
                    auto_linker.print_statistics();
                }
            }
            Err(e) => {
                println!("   ⚠️ Automatic entity linking failed: {e}");
                println!("   Continuing with original entities...");
            }
        }

        println!("✅ Knowledge graph built: {total_entities} entities extracted");

        Ok(())
    }

    /// Query the system for relevant information with auto-initialization
    pub fn ask(&mut self, query: &str) -> Result<String> {
        // Auto-initialize if not already done
        self.ensure_initialized()?;

        // Auto-build graph if documents added but graph not built
        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        // Now perform the actual query and return a simple string
        let results = self.query_internal(query)?;
        Ok(results.join("\n"))
    }

    /// Query the system for relevant information (legacy method)
    pub fn query(&self, query: &str) -> Result<Vec<String>> {
        let retrieval = self
            .retrieval_system
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized. Use .ask() for auto-initialization or call .initialize() first".to_string(),
            })?;

        retrieval.query(query)
    }

    /// Internal query method that assumes initialization is complete
    fn query_internal(&self, query: &str) -> Result<Vec<String>> {
        let retrieval = self
            .retrieval_system
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        retrieval.query(query)
    }

    /// Check if the GraphRAG system is initialized
    pub fn is_initialized(&self) -> bool {
        self.knowledge_graph.is_some() && self.retrieval_system.is_some() && self.answer_generator.is_some()
    }

    /// Check if documents have been added
    pub fn has_documents(&self) -> bool {
        if let Some(ref graph) = self.knowledge_graph {
            graph.chunks().count() > 0
        } else {
            false
        }
    }

    /// Check if the graph has been built (has entities)
    pub fn has_graph(&self) -> bool {
        if let Some(ref graph) = self.knowledge_graph {
            graph.entities().count() > 0
        } else {
            false
        }
    }

    /// Ensure the system is initialized, initializing if needed
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.is_initialized() {
            self.initialize()
        } else {
            Ok(())
        }
    }

    /// Add a document from text content with auto-generated ID
    pub fn add_document_from_text(&mut self, text: &str) -> Result<()> {
        use crate::text::TextProcessor;
        use indexmap::IndexMap;

        let doc_id = DocumentId::new(format!("doc_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));

        let document = Document {
            id: doc_id,
            title: "Document".to_string(),
            content: text.to_string(),
            metadata: IndexMap::new(),
            chunks: Vec::new(),
        };

        // Process text into chunks
        let text_processor = TextProcessor::new(self.config.text.chunk_size, self.config.text.chunk_overlap)?;
        let chunks = text_processor.chunk_text(&document)?;

        let document_with_chunks = Document {
            chunks,
            ..document
        };

        self.add_document(document_with_chunks)
    }

    /// Add a document to the system
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        // Build hierarchical tree for the document first
        self.build_document_tree(&document)?;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        graph.add_document(document)
    }

    /// Build hierarchical tree for a document
    pub fn build_document_tree(&mut self, document: &Document) -> Result<()> {
        if document.chunks.is_empty() {
            return Ok(());
        }

        println!("Building hierarchical tree for document: {}", document.id);

        let mut tree = if let Some(ref processor) = self.parallel_processor {
            DocumentTree::with_parallel_processing(
                document.id.clone(),
                self.hierarchical_config.clone(),
                processor.clone(),
            )?
        } else {
            DocumentTree::new(document.id.clone(), self.hierarchical_config.clone())?
        };

        tree.build_from_chunks(document.chunks.clone())?;

        let stats = tree.get_statistics();
        stats.print();

        self.document_trees.insert(document.id.clone(), tree);

        Ok(())
    }

    /// Query using hierarchical summarization
    pub fn hierarchical_query(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<summarization::QueryResult>> {
        let mut all_results = Vec::new();

        // Query all document trees
        for tree in self.document_trees.values() {
            let tree_results = tree.query(query, max_results)?;
            all_results.extend(tree_results);
        }

        // Sort by score and limit results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_results.truncate(max_results);

        Ok(all_results)
    }

    /// Get a document tree by document ID
    pub fn get_document_tree(&self, doc_id: &core::DocumentId) -> Option<&DocumentTree> {
        self.document_trees.get(doc_id)
    }

    /// Save all document trees to JSON files
    pub fn save_trees_to_json(&self, output_dir: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;

        for (doc_id, tree) in &self.document_trees {
            let filename = format!("{doc_id}_tree.json");
            let filepath = Path::new(output_dir).join(filename);

            let json_content = tree.to_json()?;
            fs::write(&filepath, json_content)?;

            println!(
                "Saved tree for document {} to {}",
                doc_id,
                filepath.display()
            );
        }

        Ok(())
    }

    /// Load document trees from JSON files
    pub fn load_trees_from_json(&mut self, input_dir: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let dir_path = Path::new(input_dir);
        if !dir_path.exists() {
            return Err(GraphRAGError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Directory not found: {input_dir}"),
            )));
        }

        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    if filename.ends_with("_tree.json") {
                        let json_content = fs::read_to_string(&path)?;
                        let tree = DocumentTree::from_json(&json_content)?;
                        let doc_id = tree.document_id().clone();

                        self.document_trees.insert(doc_id.clone(), tree);
                        println!(
                            "Loaded tree for document {} from {}",
                            doc_id,
                            path.display()
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate an answer to a question using the complete GraphRAG pipeline
    pub fn answer_question(&mut self, question: &str) -> Result<GeneratedAnswer> {
        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let retrieval =
            self.retrieval_system
                .as_mut()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "Retrieval system not initialized".to_string(),
                })?;

        let generator =
            self.answer_generator
                .as_ref()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "Answer generator not initialized".to_string(),
                })?;

        // Perform hybrid retrieval
        let search_results =
            retrieval.hybrid_query_with_trees(question, graph, &self.document_trees)?;

        // Get hierarchical results
        let hierarchical_results = self.hierarchical_query(question, 5)?;

        // Generate answer
        generator.generate_answer(question, search_results, hierarchical_results)
    }

    /// Generate an answer with custom generation configuration
    pub fn answer_question_with_config(
        &mut self,
        question: &str,
        generation_config: GenerationConfig,
    ) -> Result<GeneratedAnswer> {
        // Temporarily update config
        let old_config = if let Some(ref mut generator) = self.answer_generator {
            let stats = generator.get_statistics();
            let old_config = stats.config.clone();
            generator.update_config(generation_config);
            Some(old_config)
        } else {
            return Err(GraphRAGError::Generation {
                message: "Answer generator not initialized".to_string(),
            });
        };

        // Generate answer
        let result = self.answer_question(question);

        // Restore old config
        if let (Some(ref mut generator), Some(config)) =
            (self.answer_generator.as_mut(), old_config)
        {
            generator.update_config(config);
        }

        result
    }

    /// Set custom answer generator
    pub fn set_answer_generator(&mut self, generator: AnswerGenerator) {
        self.answer_generator = Some(generator);
    }

    /// Get answer generator statistics
    pub fn get_generation_statistics(&self) -> Option<generation::GeneratorStatistics> {
        self.answer_generator.as_ref().map(|g| g.get_statistics())
    }

    /// Initialize embeddings and vector index for the knowledge graph
    pub fn initialize_embeddings(&mut self) -> Result<()> {
        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let retrieval =
            self.retrieval_system
                .as_mut()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "Retrieval system not initialized".to_string(),
                })?;

        // Add embeddings to entities and chunks
        retrieval.add_embeddings_to_graph(graph)?;
        println!("Generated embeddings and built vector index");

        Ok(())
    }

    /// Get parallel processing statistics
    pub fn get_parallel_statistics(&self) -> Option<parallel::ParallelStatistics> {
        self.parallel_processor.as_ref().map(|p| p.get_statistics())
    }

    /// Check if parallel processing is enabled
    pub fn is_parallel_enabled(&self) -> bool {
        self.parallel_processor.is_some()
    }

    /// Get mutable access to the knowledge graph
    pub fn get_knowledge_graph_mut(&mut self) -> Option<&mut KnowledgeGraph> {
        self.knowledge_graph.as_mut()
    }

    /// Save complete pipeline state to directory
    pub fn save_pipeline_state(&self, output_dir: &str) -> Result<()> {
        use std::fs;

        // Create output directory structure
        fs::create_dir_all(output_dir)?;

        // Save knowledge graph
        self.save_knowledge_graph_to_json(&format!("{output_dir}/phase4_knowledge_graph.json"))?;

        // Save document trees (already implemented)
        self.save_trees_to_json(&format!("{output_dir}/phase6_hierarchical_trees"))?;

        // Save retrieval system state
        self.save_retrieval_state_to_json(&format!("{output_dir}/phase7_retrieval_state.json"))?;

        println!("Complete pipeline state saved to {output_dir}");
        Ok(())
    }

    /// Save knowledge graph to JSON file
    pub fn save_knowledge_graph_to_json(&self, file_path: &str) -> Result<()> {
        if let Some(ref graph) = self.knowledge_graph {
            graph.save_to_json(file_path)
        } else {
            Err(GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })
        }
    }

    /// Save retrieval system state to JSON
    pub fn save_retrieval_state_to_json(&self, file_path: &str) -> Result<()> {
        if let Some(ref retrieval) = self.retrieval_system {
            retrieval.save_state_to_json(file_path)
        } else {
            Err(GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })
        }
    }

    /// Get performance insights across all components
    pub fn get_performance_insights(&self) -> PerformanceInsights {
        let mut insights = PerformanceInsights::new();

        if let Some(ref processor) = self.parallel_processor {
            insights.parallel_stats = Some(processor.get_statistics());
        }

        // TODO: Collect performance stats from other components
        // This would require extending each component to track performance metrics

        insights
    }
}

/// Performance insights across all GraphRAG components
#[derive(Debug)]
pub struct PerformanceInsights {
    pub parallel_stats: Option<parallel::ParallelStatistics>,
    pub text_processing_stats: Option<(usize, std::time::Duration)>,
    pub vector_processing_stats: Option<(usize, std::time::Duration)>,
    pub graph_construction_stats: Option<(usize, std::time::Duration)>,
    pub retrieval_stats: Option<(usize, std::time::Duration)>,
}

impl PerformanceInsights {
    pub fn new() -> Self {
        Self {
            parallel_stats: None,
            text_processing_stats: None,
            vector_processing_stats: None,
            graph_construction_stats: None,
            retrieval_stats: None,
        }
    }

    pub fn print(&self) {
        println!("🔍 GraphRAG Performance Insights");
        println!("{}", "=".repeat(40));

        if let Some(ref parallel_stats) = self.parallel_stats {
            parallel_stats.print();
        } else {
            println!("⚠️  Parallel processing disabled");
        }

        if let Some((ops, duration)) = self.text_processing_stats {
            println!("📝 Text Processing: {ops} operations in {duration:?}");
        }

        if let Some((ops, duration)) = self.vector_processing_stats {
            println!("🧮 Vector Processing: {ops} operations in {duration:?}");
        }

        if let Some((ops, duration)) = self.graph_construction_stats {
            println!("🕸️  Graph Construction: {ops} operations in {duration:?}");
        }

        if let Some((ops, duration)) = self.retrieval_stats {
            println!("🔍 Retrieval: {ops} operations in {duration:?}");
        }
    }
}

impl Default for PerformanceInsights {
    fn default() -> Self {
        Self::new()
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
}
