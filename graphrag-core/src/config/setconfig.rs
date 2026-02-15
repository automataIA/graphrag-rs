//! TOML Configuration System for GraphRAG
//! Complete configuration management with extensive TOML support

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Complete GraphRAG configuration loaded from TOML
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SetConfig {
    /// Pipeline mode/approach configuration
    #[serde(default)]
    pub mode: ModeConfig,

    /// Semantic/Neural pipeline configuration
    #[serde(default)]
    pub semantic: Option<SemanticPipelineConfig>,

    /// Algorithmic/Classic NLP pipeline configuration
    #[serde(default)]
    pub algorithmic: Option<AlgorithmicPipelineConfig>,

    /// Hybrid pipeline configuration
    #[serde(default)]
    pub hybrid: Option<HybridPipelineConfig>,

    /// General system settings
    #[serde(default)]
    pub general: GeneralConfig,

    /// Pipeline configuration
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// Storage configuration
    #[serde(default)]
    pub storage: StorageConfig,

    /// Model configuration
    #[serde(default)]
    pub models: ModelsConfig,

    /// Performance tuning
    #[serde(default)]
    pub performance: PerformanceConfig,

    /// Ollama-specific configuration
    #[serde(default)]
    pub ollama: OllamaSetConfig,

    /// Experimental features
    #[serde(default)]
    pub experimental: ExperimentalConfig,

    /// Top-level entity extraction configuration (for gleaning)
    #[serde(default)]
    pub entity_extraction: EntityExtractionTopLevelConfig,

    /// Auto-save configuration for workspace persistence
    #[serde(default)]
    pub auto_save: AutoSaveSetConfig,
}

/// Auto-save configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoSaveSetConfig {
    /// Enable auto-save functionality
    #[serde(default)]
    pub enabled: bool,

    /// Auto-save interval in seconds (0 = save after every graph build)
    #[serde(default = "default_auto_save_interval")]
    pub interval_seconds: u64,

    /// Workspace name for auto-saves (if None, uses "autosave")
    #[serde(default)]
    pub workspace_name: Option<String>,

    /// Maximum number of auto-save versions to keep (0 = unlimited)
    #[serde(default = "default_max_auto_save_versions")]
    pub max_versions: usize,
}

impl Default for AutoSaveSetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_seconds: default_auto_save_interval(),
            workspace_name: None,
            max_versions: default_max_auto_save_versions(),
        }
    }
}

/// General system configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Logging level (error, warn, info, debug, trace)
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Output directory for results
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Path to the input document to process
    #[serde(default)]
    pub input_document_path: Option<String>,

    /// Maximum threads (0 = auto-detect)
    #[serde(default)]
    pub max_threads: Option<usize>,

    /// Enable performance profiling
    #[serde(default)]
    pub enable_profiling: bool,
}

/// Pipeline execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Workflows to execute in sequence
    #[serde(default = "default_workflows")]
    pub workflows: Vec<String>,

    /// Enable parallel execution
    #[serde(default = "default_true")]
    pub parallel_execution: bool,

    /// Text extraction configuration
    #[serde(default)]
    pub text_extraction: TextExtractionConfig,

    /// Entity extraction configuration
    #[serde(default)]
    pub entity_extraction: EntityExtractionConfig,

    /// Graph building configuration
    #[serde(default)]
    pub graph_building: GraphBuildingConfig,

    /// Community detection configuration
    #[serde(default)]
    pub community_detection: CommunityDetectionConfig,
}

/// Text extraction and chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextExtractionConfig {
    /// Chunk size for text splitting
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Overlap between chunks
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,

    /// Clean control characters
    #[serde(default = "default_true")]
    pub clean_control_chars: bool,

    /// Minimum chunk size to keep
    #[serde(default = "default_min_chunk_size")]
    pub min_chunk_size: usize,

    /// Text cleaning options
    #[serde(default)]
    pub cleaning: Option<CleaningConfig>,
}

/// Text cleaning options configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningConfig {
    /// Remove URLs from text
    #[serde(default)]
    pub remove_urls: bool,

    /// Remove email addresses
    #[serde(default)]
    pub remove_emails: bool,

    /// Normalize whitespace
    #[serde(default = "default_true")]
    pub normalize_whitespace: bool,

    /// Remove special characters
    #[serde(default)]
    pub remove_special_chars: bool,
}

/// Entity extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionConfig {
    /// Model name for NER
    #[serde(default = "default_ner_model")]
    pub model_name: String,

    /// Temperature for LLM generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens for extraction
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Entity types to extract (dynamic configuration)
    pub entity_types: Option<Vec<String>>,

    /// Confidence threshold for entity extraction (top-level)
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,

    /// Custom extraction prompt
    pub custom_prompt: Option<String>,

    /// Entity filtering options
    #[serde(default)]
    pub filters: Option<EntityFiltersConfig>,
}

/// Entity filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFiltersConfig {
    /// Minimum entity length
    #[serde(default = "default_min_entity_length")]
    pub min_entity_length: usize,

    /// Maximum entity length
    #[serde(default = "default_max_entity_length")]
    pub max_entity_length: usize,

    /// Allowed entity types
    pub allowed_entity_types: Option<Vec<String>>,

    /// Confidence threshold
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,

    /// Allowed regex patterns for entity matching
    pub allowed_patterns: Option<Vec<String>>,

    /// Excluded regex patterns for entity filtering
    pub excluded_patterns: Option<Vec<String>>,

    /// Enable fuzzy matching for entity resolution
    #[serde(default)]
    pub enable_fuzzy_matching: bool,
}

/// Graph building configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBuildingConfig {
    /// Relation scoring algorithm
    #[serde(default = "default_relation_scorer")]
    pub relation_scorer: String,

    /// Minimum relation score threshold
    #[serde(default = "default_min_relation_score")]
    pub min_relation_score: f32,

    /// Maximum connections per node
    #[serde(default = "default_max_connections")]
    pub max_connections_per_node: usize,

    /// Use bidirectional relationships
    #[serde(default = "default_true")]
    pub bidirectional_relations: bool,
}

/// Community detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionConfig {
    /// Algorithm for community detection
    #[serde(default = "default_community_algorithm")]
    pub algorithm: String,

    /// Resolution parameter
    #[serde(default = "default_resolution")]
    pub resolution: f32,

    /// Minimum community size
    #[serde(default = "default_min_community_size")]
    pub min_community_size: usize,

    /// Maximum community size (0 = unlimited)
    #[serde(default)]
    pub max_community_size: usize,
}

/// Storage backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database type
    #[serde(default = "default_database_type")]
    pub database_type: String,

    /// Database path for SQLite
    #[serde(default = "default_database_path")]
    pub database_path: String,

    /// Enable WAL for SQLite
    #[serde(default = "default_true")]
    pub enable_wal: bool,

    /// PostgreSQL configuration
    pub postgresql: Option<PostgreSQLConfig>,

    /// Neo4j configuration
    pub neo4j: Option<Neo4jConfig>,
}

/// PostgreSQL database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgreSQLConfig {
    /// PostgreSQL server host
    pub host: String,
    /// PostgreSQL server port
    pub port: u16,
    /// Database name
    pub database: String,
    /// Username for authentication
    pub username: String,
    /// Password for authentication
    pub password: String,
    /// Connection pool size
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
}

/// Neo4j graph database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    /// Neo4j server URI
    pub uri: String,
    /// Username for authentication
    pub username: String,
    /// Password for authentication
    pub password: String,
    /// Enable encrypted connections
    #[serde(default)]
    pub encrypted: bool,
}

/// Model configuration for LLM and embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Primary LLM for generation
    #[serde(default = "default_primary_llm")]
    pub primary_llm: String,

    /// Embedding model
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// Maximum context length
    #[serde(default = "default_max_context")]
    pub max_context_length: usize,

    /// LLM parameters
    #[serde(default)]
    pub llm_params: Option<LLMParamsConfig>,

    /// Local model configuration
    #[serde(default)]
    pub local: Option<LocalModelsConfig>,
}

/// LLM generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMParamsConfig {
    /// Sampling temperature (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Nucleus sampling parameter (0.0-1.0)
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Frequency penalty (-2.0-2.0)
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Presence penalty (-2.0-2.0)
    #[serde(default)]
    pub presence_penalty: f32,

    /// Sequences where the model will stop generating
    pub stop_sequences: Option<Vec<String>>,
}

/// Local model configuration (Ollama)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelsConfig {
    /// Ollama API base URL
    #[serde(default = "default_ollama_url")]
    pub ollama_base_url: String,

    /// Local model name for generation
    #[serde(default = "default_ollama_model")]
    pub model_name: String,

    /// Local embedding model name
    #[serde(default = "default_ollama_embedding")]
    pub embedding_model: String,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable batch processing
    #[serde(default = "default_true")]
    pub batch_processing: bool,

    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Worker threads
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,

    /// Memory limit per worker (MB)
    #[serde(default = "default_memory_limit")]
    pub memory_limit_mb: usize,
}

/// Ollama-specific configuration for local LLM and embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaSetConfig {
    /// Enable Ollama integration
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Ollama host
    #[serde(default = "default_ollama_host")]
    pub host: String,

    /// Ollama port
    #[serde(default = "default_ollama_port")]
    pub port: u16,

    /// Chat model name
    #[serde(default = "default_chat_model")]
    pub chat_model: String,

    /// Embedding model name
    #[serde(default = "default_embedding_model_ollama")]
    pub embedding_model: String,

    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Maximum retries
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Fallback to hash-based embeddings
    #[serde(default)]
    pub fallback_to_hash: bool,

    /// Maximum tokens
    pub max_tokens: Option<u32>,

    /// Temperature
    pub temperature: Option<f32>,
}

/// Experimental features configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperimentalConfig {
    /// Enable neural reranking
    #[serde(default)]
    pub neural_reranking: bool,

    /// Enable federated learning
    #[serde(default)]
    pub federated_learning: bool,

    /// Enable real-time updates
    #[serde(default)]
    pub real_time_updates: bool,

    /// Enable distributed processing
    #[serde(default)]
    pub distributed_processing: bool,

    /// Enable LazyGraphRAG mode (no prior summarization, 0.1% indexing cost)
    #[serde(default)]
    pub lazy_graphrag: bool,

    /// Enable E2GraphRAG mode (efficient entity extraction without LLM)
    #[serde(default)]
    pub e2_graphrag: bool,

    /// LazyGraphRAG configuration
    #[serde(default)]
    pub lazy_graphrag_config: Option<LazyGraphRAGConfig>,

    /// E2GraphRAG configuration
    #[serde(default)]
    pub e2_graphrag_config: Option<E2GraphRAGConfig>,
}

/// LazyGraphRAG configuration
/// Concept-based retrieval without prior summarization (Microsoft Research, 2025)
/// Achieves 0.1% of full GraphRAG indexing cost and 700x cheaper query costs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyGraphRAGConfig {
    /// Enable concept extraction (noun phrases without LLM)
    #[serde(default = "default_true")]
    pub use_concept_extraction: bool,

    /// Minimum concept length in characters
    #[serde(default = "default_min_concept_length")]
    pub min_concept_length: usize,

    /// Maximum concept length in words
    #[serde(default = "default_max_concept_words")]
    pub max_concept_words: usize,

    /// Co-occurrence threshold (minimum shared chunks for relationship)
    #[serde(default = "default_co_occurrence_threshold")]
    pub co_occurrence_threshold: usize,

    /// Enable query refinement with iterative deepening
    #[serde(default = "default_true")]
    pub use_query_refinement: bool,

    /// Maximum refinement iterations
    #[serde(default = "default_max_refinement_iterations")]
    pub max_refinement_iterations: usize,

    /// Enable bidirectional entity-chunk indexing for fast lookups
    #[serde(default = "default_true")]
    pub use_bidirectional_index: bool,
}

impl Default for LazyGraphRAGConfig {
    fn default() -> Self {
        Self {
            use_concept_extraction: true,
            min_concept_length: 3,
            max_concept_words: 5,
            co_occurrence_threshold: 1,
            use_query_refinement: true,
            max_refinement_iterations: 3,
            use_bidirectional_index: true,
        }
    }
}

/// E2GraphRAG configuration
/// Efficient entity extraction using SpaCy-like approach without LLM
/// Achieves 10x faster indexing and 100x faster retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2GraphRAGConfig {
    /// Enable lightweight NER (no LLM required)
    #[serde(default = "default_true")]
    pub use_lightweight_ner: bool,

    /// Entity types to extract (using pattern matching)
    #[serde(default = "default_e2_entity_types")]
    pub entity_types: Vec<String>,

    /// Minimum entity confidence for pattern-based extraction
    #[serde(default = "default_e2_min_confidence")]
    pub min_confidence: f32,

    /// Enable capitalization-based named entity detection
    #[serde(default = "default_true")]
    pub use_capitalization_detection: bool,

    /// Enable noun phrase extraction
    #[serde(default = "default_true")]
    pub use_noun_phrase_extraction: bool,

    /// Minimum entity frequency (entities must appear at least N times)
    #[serde(default = "default_min_entity_frequency")]
    pub min_entity_frequency: usize,

    /// Use fast co-occurrence for relationships (no LLM)
    #[serde(default = "default_true")]
    pub use_fast_cooccurrence: bool,

    /// Enable bidirectional entity-chunk indexing
    #[serde(default = "default_true")]
    pub use_bidirectional_index: bool,
}

impl Default for E2GraphRAGConfig {
    fn default() -> Self {
        Self {
            use_lightweight_ner: true,
            entity_types: default_e2_entity_types(),
            min_confidence: 0.6,
            use_capitalization_detection: true,
            use_noun_phrase_extraction: true,
            min_entity_frequency: 1,
            use_fast_cooccurrence: true,
            use_bidirectional_index: true,
        }
    }
}

// =============================================================================
// PIPELINE APPROACH CONFIGURATION (Semantic vs Algorithmic vs Hybrid)
// =============================================================================

/// Pipeline mode/approach configuration
/// Determines which pipeline implementation to use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeConfig {
    /// Pipeline approach: "semantic", "algorithmic", or "hybrid"
    /// - semantic: Neural embeddings + LLM extraction + vector search
    /// - algorithmic: Pattern matching + TF-IDF + BM25 keyword search
    /// - hybrid: Combines both with weighted fusion
    #[serde(default = "default_approach")]
    pub approach: String,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            approach: default_approach(),
        }
    }
}

/// Semantic/Neural pipeline configuration
/// Uses deep learning models for embeddings, entity extraction, and retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPipelineConfig {
    /// Enable semantic pipeline
    #[serde(default)]
    pub enabled: bool,

    /// Embeddings configuration for semantic approach
    pub embeddings: SemanticEmbeddingsConfig,

    /// Entity extraction configuration for semantic approach
    pub entity_extraction: SemanticEntityConfig,

    /// Retrieval configuration for semantic approach
    pub retrieval: SemanticRetrievalConfig,

    /// Graph construction configuration for semantic approach
    pub graph_construction: SemanticGraphConfig,
}

/// Semantic embeddings configuration (neural models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEmbeddingsConfig {
    /// Backend: "huggingface", "openai", "voyage", "cohere", "jina", "mistral", "together", "ollama"
    #[serde(default = "default_semantic_embedding_backend")]
    pub backend: String,

    /// Model identifier (provider-specific)
    #[serde(default = "default_semantic_embedding_model")]
    pub model: String,

    /// Embedding dimension
    #[serde(default = "default_semantic_embedding_dim")]
    pub dimension: usize,

    /// Use GPU acceleration if available
    #[serde(default = "default_true")]
    pub use_gpu: bool,

    /// Similarity metric (cosine, euclidean, dot_product)
    #[serde(default = "default_similarity_metric")]
    pub similarity_metric: String,

    /// Batch size for embeddings generation
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

/// Semantic entity extraction configuration (LLM-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntityConfig {
    /// Extraction method (always "llm" for semantic)
    #[serde(default = "default_semantic_entity_method")]
    pub method: String,

    /// Enable gleaning (iterative refinement)
    #[serde(default = "default_true")]
    pub use_gleaning: bool,

    /// Maximum gleaning rounds
    #[serde(default = "default_max_gleaning_rounds")]
    pub max_gleaning_rounds: usize,

    /// LLM model for extraction
    #[serde(default = "default_chat_model")]
    pub model: String,

    /// Temperature for LLM
    #[serde(default = "default_semantic_temperature")]
    pub temperature: f32,

    /// Confidence threshold
    #[serde(default = "default_semantic_confidence")]
    pub confidence_threshold: f32,
}

/// Semantic retrieval configuration (vector search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRetrievalConfig {
    /// Retrieval strategy (always "vector" for semantic)
    #[serde(default = "default_semantic_retrieval_strategy")]
    pub strategy: String,

    /// Use HNSW index for fast approximate search
    #[serde(default = "default_true")]
    pub use_hnsw: bool,

    /// HNSW ef_construction parameter
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: usize,

    /// HNSW M parameter (connections per node)
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: usize,

    /// Top-k results
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Similarity threshold
    #[serde(default = "default_semantic_similarity_threshold")]
    pub similarity_threshold: f32,
}

/// Semantic graph construction configuration (embedding-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGraphConfig {
    /// Relation scorer (always "embedding_similarity" for semantic)
    #[serde(default = "default_semantic_relation_scorer")]
    pub relation_scorer: String,

    /// Use transformer embeddings for relationships
    #[serde(default = "default_true")]
    pub use_transformer_embeddings: bool,

    /// Minimum relation score
    #[serde(default = "default_min_relation_score")]
    pub min_relation_score: f32,
}

/// Algorithmic/Classic NLP pipeline configuration
/// Uses pattern matching, TF-IDF, and keyword-based methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct AlgorithmicPipelineConfig {
    /// Enable algorithmic pipeline
    #[serde(default)]
    pub enabled: bool,

    /// Embeddings configuration for algorithmic approach
    pub embeddings: AlgorithmicEmbeddingsConfig,

    /// Entity extraction configuration for algorithmic approach
    pub entity_extraction: AlgorithmicEntityConfig,

    /// Retrieval configuration for algorithmic approach
    pub retrieval: AlgorithmicRetrievalConfig,

    /// Graph construction configuration for algorithmic approach
    pub graph_construction: AlgorithmicGraphConfig,
}

/// Algorithmic embeddings configuration (hash-based, TF-IDF)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicEmbeddingsConfig {
    /// Backend (always "hash" for algorithmic)
    #[serde(default = "default_algorithmic_embedding_backend")]
    pub backend: String,

    /// Embedding dimension
    #[serde(default = "default_algorithmic_embedding_dim")]
    pub dimension: usize,

    /// Use TF-IDF weighting
    #[serde(default = "default_true")]
    pub use_tfidf: bool,

    /// Vocabulary size
    #[serde(default = "default_vocabulary_size")]
    pub vocabulary_size: usize,

    /// Minimum term frequency
    #[serde(default = "default_min_term_frequency")]
    pub min_term_frequency: usize,

    /// Maximum document frequency (0.0-1.0)
    #[serde(default = "default_max_document_frequency")]
    pub max_document_frequency: f32,
}

/// Algorithmic entity extraction configuration (pattern-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicEntityConfig {
    /// Extraction method (always "pattern" for algorithmic)
    #[serde(default = "default_algorithmic_entity_method")]
    pub method: String,

    /// Use NER rules
    #[serde(default = "default_true")]
    pub use_ner_rules: bool,

    /// Use POS tagging
    #[serde(default)]
    pub use_pos_tagging: bool,

    /// Minimum entity length
    #[serde(default = "default_min_entity_length")]
    pub min_entity_length: usize,

    /// Confidence threshold
    #[serde(default = "default_algorithmic_confidence")]
    pub confidence_threshold: f32,

    /// Regex patterns for entity matching
    pub patterns: Option<Vec<String>>,
}

/// Algorithmic retrieval configuration (BM25 keyword search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicRetrievalConfig {
    /// Retrieval strategy (always "bm25" for algorithmic)
    #[serde(default = "default_algorithmic_retrieval_strategy")]
    pub strategy: String,

    /// BM25 k1 parameter (term frequency saturation)
    #[serde(default = "default_bm25_k1")]
    pub k1: f32,

    /// BM25 b parameter (length normalization)
    #[serde(default = "default_bm25_b")]
    pub b: f32,

    /// Use stemming
    #[serde(default = "default_true")]
    pub use_stemming: bool,

    /// Language for stemming
    #[serde(default = "default_language")]
    pub language: String,

    /// Top-k results
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// Algorithmic graph construction configuration (token overlap)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicGraphConfig {
    /// Relation scorer (jaccard, cosine on token vectors)
    #[serde(default = "default_algorithmic_relation_scorer")]
    pub relation_scorer: String,

    /// Use co-occurrence for relationship detection
    #[serde(default = "default_true")]
    pub use_cooccurrence: bool,

    /// Co-occurrence window size
    #[serde(default = "default_cooccurrence_window")]
    pub window_size: usize,

    /// Minimum relation score
    #[serde(default = "default_algorithmic_min_relation_score")]
    pub min_relation_score: f32,
}

/// Hybrid pipeline configuration
/// Combines semantic and algorithmic approaches with weighted fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridPipelineConfig {
    /// Enable hybrid pipeline
    #[serde(default)]
    pub enabled: bool,

    /// Weight configuration for combining approaches
    pub weights: HybridWeightsConfig,

    /// Embeddings configuration for hybrid
    pub embeddings: HybridEmbeddingsConfig,

    /// Entity extraction configuration for hybrid
    pub entity_extraction: HybridEntityConfig,

    /// Retrieval configuration for hybrid
    pub retrieval: HybridRetrievalConfig,

    /// Graph construction configuration for hybrid
    pub graph_construction: HybridGraphConfig,

    /// Fallback strategy when primary fails
    #[serde(default = "default_hybrid_fallback_strategy")]
    pub fallback_strategy: String,

    /// Enable cross-validation between approaches
    #[serde(default = "default_true")]
    pub cross_validation: bool,
}

/// Hybrid weight configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridWeightsConfig {
    /// Weight for semantic approach (0.0-1.0)
    #[serde(default = "default_hybrid_semantic_weight")]
    pub semantic_weight: f32,

    /// Weight for algorithmic approach (0.0-1.0)
    #[serde(default = "default_hybrid_algorithmic_weight")]
    pub algorithmic_weight: f32,
}

/// Hybrid embeddings configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEmbeddingsConfig {
    /// Primary backend (neural)
    #[serde(default = "default_semantic_embedding_backend")]
    pub primary: String,

    /// Fallback backend (hash-based)
    #[serde(default = "default_algorithmic_embedding_backend")]
    pub fallback: String,

    /// Combine scores from both
    #[serde(default = "default_true")]
    pub combine_scores: bool,

    /// Auto-fallback when primary unavailable
    #[serde(default = "default_true")]
    pub auto_fallback: bool,
}

/// Hybrid entity extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEntityConfig {
    /// Use both LLM and pattern extraction
    #[serde(default = "default_true")]
    pub use_both: bool,

    /// Weight for LLM extraction (0.0-1.0)
    #[serde(default = "default_hybrid_llm_weight")]
    pub llm_weight: f32,

    /// Weight for pattern extraction (0.0-1.0)
    #[serde(default = "default_hybrid_pattern_weight")]
    pub pattern_weight: f32,

    /// Cross-validate LLM results with patterns
    #[serde(default = "default_true")]
    pub cross_validate: bool,

    /// Confidence boost when both agree
    #[serde(default = "default_hybrid_confidence_boost")]
    pub confidence_boost: f32,
}

/// Hybrid retrieval configuration (RRF fusion)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRetrievalConfig {
    /// Retrieval strategy (always "fusion" for hybrid)
    #[serde(default = "default_hybrid_retrieval_strategy")]
    pub strategy: String,

    /// Combine vector and BM25
    #[serde(default = "default_true")]
    pub combine_vector_bm25: bool,

    /// Weight for vector search
    #[serde(default = "default_hybrid_vector_weight")]
    pub vector_weight: f32,

    /// Weight for BM25 search
    #[serde(default = "default_hybrid_bm25_weight")]
    pub bm25_weight: f32,

    /// RRF constant (typically 60)
    #[serde(default = "default_rrf_constant")]
    pub rrf_constant: usize,
}

/// Hybrid graph construction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridGraphConfig {
    /// Primary relation scorer (embedding-based)
    #[serde(default = "default_semantic_relation_scorer")]
    pub primary_scorer: String,

    /// Fallback relation scorer (token-based)
    #[serde(default = "default_algorithmic_relation_scorer")]
    pub fallback_scorer: String,

    /// Combine scores from both scorers
    #[serde(default = "default_true")]
    pub combine_scores: bool,
}

/// Top-level entity extraction configuration (gleaning settings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionTopLevelConfig {
    /// Enable entity extraction
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Minimum confidence threshold
    #[serde(default = "default_confidence_threshold")]
    pub min_confidence: f32,

    /// Use LLM-based gleaning
    #[serde(default)]
    pub use_gleaning: bool,

    /// Maximum gleaning rounds
    #[serde(default = "default_gleaning_rounds")]
    pub max_gleaning_rounds: usize,

    /// Gleaning improvement threshold
    #[serde(default = "default_gleaning_improvement")]
    pub gleaning_improvement_threshold: f32,

    /// Enable semantic merging
    #[serde(default)]
    pub semantic_merging: bool,

    /// Merge similarity threshold
    #[serde(default = "default_merge_threshold")]
    pub merge_similarity_threshold: f32,

    /// Enable automatic linking
    #[serde(default)]
    pub automatic_linking: bool,

    /// Linking confidence threshold
    #[serde(default = "default_confidence_threshold")]
    pub linking_confidence_threshold: f32,
}

impl Default for EntityExtractionTopLevelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: default_confidence_threshold(),
            use_gleaning: false,
            max_gleaning_rounds: default_gleaning_rounds(),
            gleaning_improvement_threshold: default_gleaning_improvement(),
            semantic_merging: false,
            merge_similarity_threshold: default_merge_threshold(),
            automatic_linking: false,
            linking_confidence_threshold: default_confidence_threshold(),
        }
    }
}

// Default value functions
fn default_log_level() -> String {
    "info".to_string()
}
fn default_output_dir() -> String {
    "./output".to_string()
}
fn default_true() -> bool {
    true
}
fn default_workflows() -> Vec<String> {
    vec![
        "extract_text".to_string(),
        "extract_entities".to_string(),
        "build_graph".to_string(),
        "detect_communities".to_string(),
    ]
}
fn default_chunk_size() -> usize {
    512
}
fn default_chunk_overlap() -> usize {
    64
}
fn default_min_chunk_size() -> usize {
    50
}
fn default_ner_model() -> String {
    "microsoft/DialoGPT-medium".to_string()
}
fn default_temperature() -> f32 {
    0.1
}
fn default_max_tokens() -> usize {
    2048
}
fn default_min_entity_length() -> usize {
    3
}
fn default_max_entity_length() -> usize {
    100
}
fn default_confidence_threshold() -> f32 {
    0.8
}
fn default_relation_scorer() -> String {
    "cosine_similarity".to_string()
}
fn default_min_relation_score() -> f32 {
    0.7
}
fn default_max_connections() -> usize {
    10
}
fn default_community_algorithm() -> String {
    "leiden".to_string()
}
fn default_resolution() -> f32 {
    1.0
}
fn default_min_community_size() -> usize {
    3
}
fn default_database_type() -> String {
    "sqlite".to_string()
}
fn default_database_path() -> String {
    "./graphrag.db".to_string()
}
fn default_pool_size() -> usize {
    10
}
fn default_primary_llm() -> String {
    "gpt-4".to_string()
}
fn default_embedding_model() -> String {
    "text-embedding-ada-002".to_string()
}
fn default_max_context() -> usize {
    4096
}
fn default_top_p() -> f32 {
    0.9
}
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_ollama_model() -> String {
    "llama2:7b".to_string()
}
fn default_ollama_embedding() -> String {
    "nomic-embed-text".to_string()
}
fn default_batch_size() -> usize {
    100
}
fn default_worker_threads() -> usize {
    4
}
fn default_memory_limit() -> usize {
    1024
}
fn default_ollama_host() -> String {
    "http://localhost".to_string()
}
fn default_ollama_port() -> u16 {
    11434
}
fn default_chat_model() -> String {
    "llama3.1:8b".to_string()
}
fn default_embedding_model_ollama() -> String {
    "nomic-embed-text".to_string()
}
fn default_timeout() -> u64 {
    60
}
fn default_max_retries() -> u32 {
    3
}
fn default_gleaning_rounds() -> usize {
    3
}
fn default_gleaning_improvement() -> f32 {
    0.1
}
fn default_merge_threshold() -> f32 {
    0.85
}

// =============================================================================
// Default functions for Pipeline Approach Configuration
// =============================================================================

// Mode defaults
fn default_approach() -> String {
    "semantic".to_string() // Default to semantic pipeline
}

// Semantic pipeline defaults
fn default_semantic_embedding_backend() -> String {
    "huggingface".to_string()
}
fn default_semantic_embedding_model() -> String {
    "sentence-transformers/all-MiniLM-L6-v2".to_string()
}
fn default_semantic_embedding_dim() -> usize {
    384 // MiniLM-L6-v2 dimension
}
fn default_similarity_metric() -> String {
    "cosine".to_string()
}
fn default_semantic_entity_method() -> String {
    "llm".to_string()
}
fn default_max_gleaning_rounds() -> usize {
    3
}
fn default_semantic_temperature() -> f32 {
    0.1
}
fn default_semantic_confidence() -> f32 {
    0.7
}
fn default_semantic_retrieval_strategy() -> String {
    "vector".to_string()
}
fn default_hnsw_ef_construction() -> usize {
    200
}
fn default_hnsw_m() -> usize {
    16
}
fn default_top_k() -> usize {
    10
}
fn default_semantic_similarity_threshold() -> f32 {
    0.7
}
fn default_semantic_relation_scorer() -> String {
    "embedding_similarity".to_string()
}

// Algorithmic pipeline defaults
fn default_algorithmic_embedding_backend() -> String {
    "hash".to_string()
}
fn default_algorithmic_embedding_dim() -> usize {
    128
}
fn default_vocabulary_size() -> usize {
    10000
}
fn default_min_term_frequency() -> usize {
    2
}
fn default_max_document_frequency() -> f32 {
    0.8
}
fn default_algorithmic_entity_method() -> String {
    "pattern".to_string()
}
fn default_algorithmic_confidence() -> f32 {
    0.75
}
fn default_algorithmic_retrieval_strategy() -> String {
    "bm25".to_string()
}
fn default_bm25_k1() -> f32 {
    1.5
}
fn default_bm25_b() -> f32 {
    0.75
}
fn default_language() -> String {
    "english".to_string()
}
fn default_algorithmic_relation_scorer() -> String {
    "jaccard".to_string()
}
fn default_cooccurrence_window() -> usize {
    10
}
fn default_algorithmic_min_relation_score() -> f32 {
    0.6
}

// Hybrid pipeline defaults
fn default_hybrid_semantic_weight() -> f32 {
    0.6
}
fn default_hybrid_algorithmic_weight() -> f32 {
    0.4
}
fn default_hybrid_llm_weight() -> f32 {
    0.7
}
fn default_hybrid_pattern_weight() -> f32 {
    0.3
}
fn default_hybrid_confidence_boost() -> f32 {
    0.15
}
fn default_hybrid_retrieval_strategy() -> String {
    "fusion".to_string()
}
fn default_hybrid_vector_weight() -> f32 {
    0.6
}
fn default_hybrid_bm25_weight() -> f32 {
    0.4
}
fn default_rrf_constant() -> usize {
    60
}
fn default_hybrid_fallback_strategy() -> String {
    "semantic_first".to_string()
}
fn default_auto_save_interval() -> u64 {
    300  // 5 minutes
}
fn default_max_auto_save_versions() -> usize {
    5  // Keep 5 versions by default
}

// LazyGraphRAG default functions
fn default_min_concept_length() -> usize {
    3  // Minimum 3 characters for concepts
}
fn default_max_concept_words() -> usize {
    5  // Maximum 5 words per concept
}
fn default_co_occurrence_threshold() -> usize {
    1  // Minimum 1 shared chunk for relationship
}
fn default_max_refinement_iterations() -> usize {
    3  // Up to 3 query refinement iterations
}

// E2GraphRAG default functions
fn default_e2_entity_types() -> Vec<String> {
    vec![
        "PERSON".to_string(),
        "ORGANIZATION".to_string(),
        "LOCATION".to_string(),
        "CONCEPT".to_string(),
    ]
}
fn default_e2_min_confidence() -> f32 {
    0.6  // 60% minimum confidence for pattern-based extraction
}
fn default_min_entity_frequency() -> usize {
    1  // Entities must appear at least once
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            output_dir: default_output_dir(),
            input_document_path: None,
            max_threads: None,
            enable_profiling: false,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            workflows: default_workflows(),
            parallel_execution: default_true(),
            text_extraction: TextExtractionConfig::default(),
            entity_extraction: EntityExtractionConfig::default(),
            graph_building: GraphBuildingConfig::default(),
            community_detection: CommunityDetectionConfig::default(),
        }
    }
}

impl Default for TextExtractionConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            clean_control_chars: default_true(),
            min_chunk_size: default_min_chunk_size(),
            cleaning: None,
        }
    }
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            model_name: default_ner_model(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            entity_types: None,
            confidence_threshold: default_confidence_threshold(),
            custom_prompt: None,
            filters: None,
        }
    }
}

impl Default for GraphBuildingConfig {
    fn default() -> Self {
        Self {
            relation_scorer: default_relation_scorer(),
            min_relation_score: default_min_relation_score(),
            max_connections_per_node: default_max_connections(),
            bidirectional_relations: default_true(),
        }
    }
}

impl Default for CommunityDetectionConfig {
    fn default() -> Self {
        Self {
            algorithm: default_community_algorithm(),
            resolution: default_resolution(),
            min_community_size: default_min_community_size(),
            max_community_size: 0,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_type: default_database_type(),
            database_path: default_database_path(),
            enable_wal: default_true(),
            postgresql: None,
            neo4j: None,
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            primary_llm: default_primary_llm(),
            embedding_model: default_embedding_model(),
            max_context_length: default_max_context(),
            llm_params: None,
            local: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            batch_processing: default_true(),
            batch_size: default_batch_size(),
            worker_threads: default_worker_threads(),
            memory_limit_mb: default_memory_limit(),
        }
    }
}

impl Default for OllamaSetConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            host: default_ollama_host(),
            port: default_ollama_port(),
            chat_model: default_chat_model(),
            embedding_model: default_embedding_model_ollama(),
            timeout_seconds: default_timeout(),
            max_retries: default_max_retries(),
            fallback_to_hash: false,
            max_tokens: Some(800),
            temperature: Some(0.3),
        }
    }
}

// =============================================================================
// Default implementations for Pipeline Approach Configuration
// =============================================================================

impl Default for SemanticPipelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            embeddings: SemanticEmbeddingsConfig::default(),
            entity_extraction: SemanticEntityConfig::default(),
            retrieval: SemanticRetrievalConfig::default(),
            graph_construction: SemanticGraphConfig::default(),
        }
    }
}

impl Default for SemanticEmbeddingsConfig {
    fn default() -> Self {
        Self {
            backend: default_semantic_embedding_backend(),
            model: default_semantic_embedding_model(),
            dimension: default_semantic_embedding_dim(),
            use_gpu: default_true(),
            similarity_metric: default_similarity_metric(),
            batch_size: default_batch_size(),
        }
    }
}

impl Default for SemanticEntityConfig {
    fn default() -> Self {
        Self {
            method: default_semantic_entity_method(),
            use_gleaning: default_true(),
            max_gleaning_rounds: default_max_gleaning_rounds(),
            model: default_chat_model(),
            temperature: default_semantic_temperature(),
            confidence_threshold: default_semantic_confidence(),
        }
    }
}

impl Default for SemanticRetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: default_semantic_retrieval_strategy(),
            use_hnsw: default_true(),
            hnsw_ef_construction: default_hnsw_ef_construction(),
            hnsw_m: default_hnsw_m(),
            top_k: default_top_k(),
            similarity_threshold: default_semantic_similarity_threshold(),
        }
    }
}

impl Default for SemanticGraphConfig {
    fn default() -> Self {
        Self {
            relation_scorer: default_semantic_relation_scorer(),
            use_transformer_embeddings: default_true(),
            min_relation_score: default_min_relation_score(),
        }
    }
}


impl Default for AlgorithmicEmbeddingsConfig {
    fn default() -> Self {
        Self {
            backend: default_algorithmic_embedding_backend(),
            dimension: default_algorithmic_embedding_dim(),
            use_tfidf: default_true(),
            vocabulary_size: default_vocabulary_size(),
            min_term_frequency: default_min_term_frequency(),
            max_document_frequency: default_max_document_frequency(),
        }
    }
}

impl Default for AlgorithmicEntityConfig {
    fn default() -> Self {
        Self {
            method: default_algorithmic_entity_method(),
            use_ner_rules: default_true(),
            use_pos_tagging: false,
            min_entity_length: default_min_entity_length(),
            confidence_threshold: default_algorithmic_confidence(),
            patterns: None,
        }
    }
}

impl Default for AlgorithmicRetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: default_algorithmic_retrieval_strategy(),
            k1: default_bm25_k1(),
            b: default_bm25_b(),
            use_stemming: default_true(),
            language: default_language(),
            top_k: default_top_k(),
        }
    }
}

impl Default for AlgorithmicGraphConfig {
    fn default() -> Self {
        Self {
            relation_scorer: default_algorithmic_relation_scorer(),
            use_cooccurrence: default_true(),
            window_size: default_cooccurrence_window(),
            min_relation_score: default_algorithmic_min_relation_score(),
        }
    }
}

impl Default for HybridPipelineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            weights: HybridWeightsConfig::default(),
            embeddings: HybridEmbeddingsConfig::default(),
            entity_extraction: HybridEntityConfig::default(),
            retrieval: HybridRetrievalConfig::default(),
            graph_construction: HybridGraphConfig::default(),
            fallback_strategy: default_hybrid_fallback_strategy(),
            cross_validation: default_true(),
        }
    }
}

impl Default for HybridWeightsConfig {
    fn default() -> Self {
        Self {
            semantic_weight: default_hybrid_semantic_weight(),
            algorithmic_weight: default_hybrid_algorithmic_weight(),
        }
    }
}

impl Default for HybridEmbeddingsConfig {
    fn default() -> Self {
        Self {
            primary: default_semantic_embedding_backend(),
            fallback: default_algorithmic_embedding_backend(),
            combine_scores: default_true(),
            auto_fallback: default_true(),
        }
    }
}

impl Default for HybridEntityConfig {
    fn default() -> Self {
        Self {
            use_both: default_true(),
            llm_weight: default_hybrid_llm_weight(),
            pattern_weight: default_hybrid_pattern_weight(),
            cross_validate: default_true(),
            confidence_boost: default_hybrid_confidence_boost(),
        }
    }
}

impl Default for HybridRetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: default_hybrid_retrieval_strategy(),
            combine_vector_bm25: default_true(),
            vector_weight: default_hybrid_vector_weight(),
            bm25_weight: default_hybrid_bm25_weight(),
            rrf_constant: default_rrf_constant(),
        }
    }
}

impl Default for HybridGraphConfig {
    fn default() -> Self {
        Self {
            primary_scorer: default_semantic_relation_scorer(),
            fallback_scorer: default_algorithmic_relation_scorer(),
            combine_scores: default_true(),
        }
    }
}

impl SetConfig {
    /// Load configuration from TOML or JSON5 file (auto-detects format by extension)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref)?;

        // Detect format by file extension
        let extension = path_ref
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let config: SetConfig = match extension {
            #[cfg(feature = "json5-support")]
            "json5" | "json" => {
                json5::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                    message: format!("JSON5 parse error: {e}"),
                })?
            }
            #[cfg(not(feature = "json5-support"))]
            "json5" | "json" => {
                return Err(crate::core::GraphRAGError::Config {
                    message: "JSON5 support not enabled. Rebuild with --features json5-support".to_string(),
                });
            }
            _ => {
                toml::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                    message: format!("TOML parse error: {e}"),
                })?
            }
        };

        Ok(config)
    }

    /// Save configuration to TOML file with comments
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let toml_string =
            toml::to_string_pretty(&self).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("TOML serialize error: {e}"),
            })?;

        // Add header comment
        let commented_toml = format!(
            "# =============================================================================\n\
             # GraphRAG Configuration File\n\
             # Complete configuration with extensive parameters for easy customization\n\
             # =============================================================================\n\n{toml_string}"
        );

        fs::write(path, commented_toml)?;
        Ok(())
    }

    /// Convert to the existing Config format for compatibility
    pub fn to_graphrag_config(&self) -> crate::Config {
        let mut config = crate::Config {
            approach: self.mode.approach.clone(),
            ..Default::default()
        };

        // Map text processing
        config.text.chunk_size = self.pipeline.text_extraction.chunk_size;
        config.text.chunk_overlap = self.pipeline.text_extraction.chunk_overlap;

        // Map entity extraction based on approach
        config.entities.min_confidence = self
            .entity_extraction
            .min_confidence;

        // Map entity types from pipeline.entity_extraction
        if let Some(ref types) = self.pipeline.entity_extraction.entity_types {
            config.entities.entity_types = types.clone();
        }

        // Configure gleaning based on approach:
        // - semantic: use LLM-based gleaning
        // - algorithmic: use pattern-based extraction
        // - hybrid: use both (for compatibility, map to gleaning)
        match self.mode.approach.as_str() {
            "semantic" => {
                if let Some(ref semantic) = self.semantic {
                    config.entities.use_gleaning = semantic.entity_extraction.use_gleaning;
                    config.entities.max_gleaning_rounds = semantic.entity_extraction.max_gleaning_rounds;
                    config.entities.min_confidence = semantic.entity_extraction.confidence_threshold;
                } else {
                    // Fallback for semantic approach: ALWAYS enable gleaning when mode.approach = "semantic"
                    // This ensures JSON5 configs with mode.approach="semantic" use LLM-based extraction
                    config.entities.use_gleaning = true;
                    config.entities.max_gleaning_rounds = if self.entity_extraction.use_gleaning {
                        self.entity_extraction.max_gleaning_rounds
                    } else {
                        default_max_gleaning_rounds() // Use default if not explicitly set
                    };
                    // Use top-level min_confidence if available
                    config.entities.min_confidence = self.entity_extraction.min_confidence;
                }
            }
            "algorithmic" => {
                // Algorithmic uses pattern-based extraction, no gleaning
                config.entities.use_gleaning = false;
                if let Some(ref algorithmic) = self.algorithmic {
                    config.entities.min_confidence = algorithmic.entity_extraction.confidence_threshold;
                }
            }
            "hybrid" => {
                // Hybrid can use both, enable gleaning for LLM component
                config.entities.use_gleaning = true;
                if self.hybrid.is_some() {
                    // Use hybrid configuration if available
                    config.entities.max_gleaning_rounds = 2; // Reduced for hybrid efficiency
                }
            }
            _ => {
                // Unknown approach, use top-level config as fallback
                config.entities.use_gleaning = self.entity_extraction.use_gleaning;
                config.entities.max_gleaning_rounds = self.entity_extraction.max_gleaning_rounds;
            }
        }

        // Map graph building
        config.graph.similarity_threshold = self.pipeline.graph_building.min_relation_score;
        config.graph.max_connections = self.pipeline.graph_building.max_connections_per_node;
        config.graph.extract_relationships = true; // Enable by default for TOML configs
        config.graph.relationship_confidence_threshold = 0.5; // Default threshold

        // Map retrieval
        config.retrieval.top_k = 10; // Default

        // Map embeddings
        config.embeddings.dimension = 768; // Default for nomic-embed-text
        config.embeddings.backend = "ollama".to_string();
        config.embeddings.fallback_to_hash = self.ollama.fallback_to_hash;

        // Map parallel processing
        config.parallel.enabled = self.pipeline.parallel_execution;
        config.parallel.num_threads = self.performance.worker_threads;

        // Map Ollama configuration
        config.ollama = crate::ollama::OllamaConfig {
            enabled: self.ollama.enabled,
            host: self.ollama.host.clone(),
            port: self.ollama.port,
            chat_model: self.ollama.chat_model.clone(),
            embedding_model: self.ollama.embedding_model.clone(),
            timeout_seconds: self.ollama.timeout_seconds,
            max_retries: self.ollama.max_retries,
            fallback_to_hash: self.ollama.fallback_to_hash,
            max_tokens: self.ollama.max_tokens,
            temperature: self.ollama.temperature,
            enable_caching: true, // Default to enabled
        };

        // Map auto-save configuration
        config.auto_save = crate::config::AutoSaveConfig {
            enabled: self.auto_save.enabled,
            interval_seconds: self.auto_save.interval_seconds,
            workspace_name: self.auto_save.workspace_name.clone(),
            max_versions: self.auto_save.max_versions,
        };

        config
    }
}
