//! TOML Configuration System for GraphRAG
//! Complete configuration management with extensive TOML support

use crate::config::validation::Validatable;
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

    /// LLM configuration (OpenAI-compatible: llama-server, Cerebras, Groq)
    #[serde(default)]
    pub llm: LlmSetConfig,

    /// Removed `ollama` section. Present only to fail loudly on legacy configs.
    #[serde(
        default,
        skip_serializing,
        rename = "ollama",
        deserialize_with = "reject_legacy_ollama"
    )]
    pub legacy_ollama: Option<()>,

    /// GLiNER-Relex extractor configuration
    #[serde(default)]
    pub gliner: GlinerSetConfig,

    /// Cross-encoder reranker configuration (opt-in)
    #[serde(default)]
    pub reranker: RerankerSetConfig,

    /// Per-chapter summaries as retrievable chunks (opt-in)
    #[serde(default)]
    pub chapter_summaries: ChapterSummariesSetConfig,

    /// One LLM context line per chunk, prepended at index time (opt-in)
    #[serde(default)]
    pub chunk_context: ChunkContextSetConfig,

    /// Retrieval options shared by all approaches (currently only HyDE)
    #[serde(default)]
    pub retrieval: RetrievalSetConfig,

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

/// Auto-save / storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoSaveSetConfig {
    /// Enable persistent storage.
    /// `false` (default) → graph lives in memory only.
    /// `true` → graph is saved to disk after `build_graph()` and reloaded on the next run.
    #[serde(default)]
    pub enabled: bool,

    /// Base directory for workspace storage. Required when `enabled = true`.
    /// Example: `"./output"` or `"/data/graphrag"`.
    /// The workspace folder is created at `<base_dir>/<workspace_name>/`.
    #[serde(default)]
    pub base_dir: Option<String>,

    /// Auto-save interval in seconds (0 = save after every graph build)
    #[serde(default = "default_auto_save_interval")]
    pub interval_seconds: u64,

    /// Workspace name — sub-folder inside `base_dir` (default: "default").
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
            base_dir: None,
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

    /// Temperature for entity extraction (0.0 = fully deterministic JSON output)
    #[serde(default = "default_extraction_temperature")]
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

/// Local model configuration (llama-server)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelsConfig {
    /// OpenAI-compatible base URL of the local llama-server
    #[serde(default = "default_llm_base_url")]
    pub base_url: String,

    /// Local model name for generation
    #[serde(default = "default_chat_model")]
    pub model_name: String,

    /// Local embedding model name
    #[serde(default = "default_llm_embedding_model")]
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

/// LLM configuration: any OpenAI-compatible endpoint (llama-server, Cerebras, Groq).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmSetConfig {
    /// Enable LLM-backed stages
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Chat API base URL including `/v1`
    #[serde(default = "default_llm_base_url")]
    pub base_url: String,

    /// Embedding API base URL including `/v1` (llama-server `--embeddings`)
    #[serde(default = "default_llm_embedding_base_url")]
    pub embedding_base_url: String,

    /// Name of the environment variable holding the API key (never the key itself)
    #[serde(default)]
    pub api_key_env: Option<String>,

    /// Chat model name
    #[serde(default = "default_chat_model")]
    pub chat_model: String,

    /// Embedding model name
    #[serde(default = "default_llm_embedding_model")]
    pub embedding_model: String,

    /// Timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Maximum attempts per request
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Fallback to hash-based embeddings
    #[serde(default)]
    pub fallback_to_hash: bool,

    /// Maximum tokens
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Temperature
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Cache identical requests in memory
    #[serde(default = "default_true")]
    pub enable_caching: bool,

    /// Maximum estimated tokens of retrieved context in an answer prompt
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,

    /// Maximum LLM requests in flight during entity extraction (match llama-server `-np`)
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    /// Cache raw extraction responses on disk in the auto_save workspace
    #[serde(default = "default_true")]
    pub extraction_cache: bool,

    /// Send `chat_template_kwargs.enable_thinking=false` with each chat request
    #[serde(default = "default_true")]
    pub disable_thinking: bool,
}

/// Runtime retrieval options at the top level of a SetConfig document (`retrieval: { hyde, graph_facts }`).
/// Lenient on purpose: older templates carry other keys under a root `retrieval` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalSetConfig {
    /// HyDE query expansion in the ask path (one extra LLM call per question)
    #[serde(default)]
    pub hyde: bool,
    /// Add the relationships of the seed entities to the answer context
    #[serde(default)]
    pub graph_facts: bool,
    /// Map-reduce at query time: one short LLM call per candidate chunk (N extra calls per question)
    #[serde(default)]
    pub map_reduce: bool,
    /// Max chunks mapped when `map_reduce` is on
    #[serde(default = "default_map_reduce_chunks")]
    pub map_reduce_chunks: usize,
}

fn default_map_reduce_chunks() -> usize {
    60
}

impl Default for RetrievalSetConfig {
    fn default() -> Self {
        Self {
            hyde: false,
            graph_facts: false,
            map_reduce: false,
            map_reduce_chunks: default_map_reduce_chunks(),
        }
    }
}

/// Cross-encoder reranker configuration (llama.cpp `llama-server --reranking`).
/// Kept in sync with `RerankerConfig::default()` (drift-guard test below).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankerSetConfig {
    /// Enable reranking in the ask path
    #[serde(default)]
    pub enabled: bool,
    /// Reranker server base URL
    #[serde(default = "default_reranker_endpoint")]
    pub api_endpoint: String,
    /// Reranker model alias
    #[serde(default = "default_reranker_model")]
    pub model: String,
    /// Max distinct candidate chunks sent to the reranker
    #[serde(default = "default_reranker_candidates")]
    pub candidates: usize,
    /// Number of reranked chunks placed first in the context
    #[serde(default = "default_reranker_top_k")]
    pub top_k: usize,
    /// Name of the environment variable holding the API key (never the key itself)
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Request timeout in seconds
    #[serde(default = "default_reranker_timeout")]
    pub timeout_seconds: u64,
}

fn default_reranker_endpoint() -> String {
    "http://localhost:8082".to_string()
}
fn default_reranker_model() -> String {
    "bge-reranker-v2-m3".to_string()
}
fn default_reranker_candidates() -> usize {
    30
}
fn default_reranker_top_k() -> usize {
    8
}
fn default_reranker_timeout() -> u64 {
    30
}

impl Default for RerankerSetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_endpoint: default_reranker_endpoint(),
            model: default_reranker_model(),
            candidates: default_reranker_candidates(),
            top_k: default_reranker_top_k(),
            api_key_env: None,
            timeout_seconds: default_reranker_timeout(),
        }
    }
}

/// Per-chapter summaries written at index time.
/// Kept in sync with `ChapterSummariesConfig::default()` (drift-guard test below).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChapterSummariesSetConfig {
    /// Write one LLM summary per chapter as an extra chunk (needs `llm.enabled`)
    #[serde(default)]
    pub enabled: bool,
    /// Output token budget of one summary
    #[serde(default = "default_chapter_summary_max_tokens")]
    pub max_tokens: u32,
    /// Max summary chunks added to the answer context, on top of the original chunks
    #[serde(default = "default_chapter_summary_max_in_context")]
    pub max_in_context: usize,
}

fn default_chapter_summary_max_tokens() -> u32 {
    400
}

fn default_chapter_summary_max_in_context() -> usize {
    2
}

impl Default for ChapterSummariesSetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_tokens: default_chapter_summary_max_tokens(),
            max_in_context: default_chapter_summary_max_in_context(),
        }
    }
}

/// One context sentence per chunk written at index time.
/// Kept in sync with `ChunkContextConfig::default()` (drift-guard test below).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkContextSetConfig {
    /// Prepend one LLM-written context sentence to every chunk (needs `llm.enabled`)
    #[serde(default)]
    pub enabled: bool,
    /// Preceding chunks of the same document shown to the model
    #[serde(default = "default_chunk_context_window")]
    pub window: usize,
    /// Output token budget of one context line
    #[serde(default = "default_chunk_context_max_tokens")]
    pub max_tokens: u32,
}

fn default_chunk_context_window() -> usize {
    4
}

fn default_chunk_context_max_tokens() -> u32 {
    120
}

impl Default for ChunkContextSetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window: default_chunk_context_window(),
            max_tokens: default_chunk_context_max_tokens(),
        }
    }
}

/// GLiNER-Relex extractor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlinerSetConfig {
    /// Enable GLiNER-Relex extraction
    #[serde(default)]
    pub enabled: bool,
    /// Path to the ONNX model file
    #[serde(default)]
    pub model_path: String,
    /// Path to tokenizer.json (defaults to same dir as model_path)
    #[serde(default)]
    pub tokenizer_path: String,
    /// "span" (default) or "token"
    #[serde(default = "default_gliner_mode")]
    pub mode: String,
    /// Entity labels to extract
    #[serde(default = "default_gliner_entity_labels")]
    pub entity_labels: Vec<String>,
    /// Relation labels to extract
    #[serde(default = "default_gliner_relation_labels")]
    pub relation_labels: Vec<String>,
    /// Minimum entity confidence threshold
    #[serde(default = "default_entity_threshold")]
    pub entity_threshold: f32,
    /// Minimum relation confidence threshold
    #[serde(default = "default_relation_threshold")]
    pub relation_threshold: f32,
    /// Use GPU (CUDA) for inference
    #[serde(default)]
    pub use_gpu: bool,
    /// Max concurrent chunk inferences (None → 4)
    #[serde(default)]
    pub max_concurrent_chunks: Option<usize>,
    /// GLiNER finds entities, the LLM extracts only the relationships among them (async only)
    #[serde(default)]
    pub llm_relations: bool,
    /// Extraction backend: "gline" (default) or "gliner2" (feature `gliner2`)
    #[serde(default = "default_gliner_backend")]
    pub backend: String,
    /// Jev classification of entity types and relations (mode "g5")
    #[serde(default)]
    pub jev: crate::config::JevConfig,
}

fn default_gliner_backend() -> String {
    "gline".to_string()
}
fn default_gliner_mode() -> String {
    "span".to_string()
}
// Kept in sync with `GlinerConfig::default()` in config/mod.rs (canonical runtime
// defaults). The drift-guard test `gliner_setconfig_default_matches_runtime` enforces it.
fn default_gliner_entity_labels() -> Vec<String> {
    vec![
        "person".into(),
        "organization".into(),
        "location".into(),
        "concept".into(),
    ]
}
fn default_gliner_relation_labels() -> Vec<String> {
    vec!["related to".into(), "part of".into(), "causes".into()]
}
fn default_entity_threshold() -> f32 {
    0.4
}
fn default_relation_threshold() -> f32 {
    0.5
}

impl Default for GlinerSetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: String::new(),
            tokenizer_path: String::new(),
            mode: default_gliner_mode(),
            entity_labels: default_gliner_entity_labels(),
            relation_labels: default_gliner_relation_labels(),
            entity_threshold: default_entity_threshold(),
            relation_threshold: default_relation_threshold(),
            use_gpu: false,
            max_concurrent_chunks: None,
            llm_relations: false,
            backend: default_gliner_backend(),
            jev: crate::config::JevConfig::default(),
        }
    }
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
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Embeddings configuration for semantic approach
    #[serde(default)]
    pub embeddings: SemanticEmbeddingsConfig,

    /// Entity extraction configuration for semantic approach
    #[serde(default)]
    pub entity_extraction: SemanticEntityConfig,

    /// Retrieval configuration for semantic approach
    #[serde(default)]
    pub retrieval: SemanticRetrievalConfig,

    /// Graph construction configuration for semantic approach
    #[serde(default)]
    pub graph_construction: SemanticGraphConfig,
}

/// Semantic embeddings configuration (neural models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEmbeddingsConfig {
    /// Backend: "huggingface", "openai", "voyage", "cohere", "jina", "mistral", "together", "llama_cpp"
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlgorithmicPipelineConfig {
    /// Enable algorithmic pipeline
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Embeddings configuration for algorithmic approach
    #[serde(default)]
    pub embeddings: AlgorithmicEmbeddingsConfig,

    /// Entity extraction configuration for algorithmic approach
    #[serde(default)]
    pub entity_extraction: AlgorithmicEntityConfig,

    /// Retrieval configuration for algorithmic approach
    #[serde(default)]
    pub retrieval: AlgorithmicRetrievalConfig,

    /// Graph construction configuration for algorithmic approach
    #[serde(default)]
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
    #[serde(default, deserialize_with = "deserialize_patterns")]
    pub patterns: Option<Vec<String>>,
}

/// Accept both the current list form and the map form used by older
/// algorithmic templates (`{ person: "...", concept: "..." }`). The public
/// representation stays a list because the extractor consumes regex strings.
fn deserialize_patterns<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| serde::de::Error::custom("patterns must contain strings"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Some),
        serde_json::Value::Object(values) => values
            .into_values()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| serde::de::Error::custom("pattern values must be strings"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Some),
        _ => Err(serde::de::Error::custom("patterns must be a list or map")),
    }
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
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Weight configuration for combining approaches
    #[serde(default)]
    pub weights: HybridWeightsConfig,

    /// Embeddings configuration for hybrid
    #[serde(default)]
    pub embeddings: HybridEmbeddingsConfig,

    /// Entity extraction configuration for hybrid
    #[serde(default)]
    pub entity_extraction: HybridEntityConfig,

    /// Retrieval configuration for hybrid
    #[serde(default)]
    pub retrieval: HybridRetrievalConfig,

    /// Graph construction configuration for hybrid
    #[serde(default)]
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

    /// Ask the extraction LLM for `status` (narrated | said) and `said_by` on each
    /// relationship; `said` ones get a `[detto da X]` marker in the description.
    /// Needs a re-index. Default false.
    #[serde(default)]
    pub epistemic_status: bool,

    /// Second, relation-only LLM call for the chunks the single-pass extraction left
    /// under-extracted (at least 3 entities, at most 1 relationship). Applies only to LLM
    /// single-pass extraction (not gleaning, GLiNER, cascade or pattern). One extra call only
    /// for the selected chunks; needs a re-index to take effect. Default false.
    #[serde(default)]
    pub selective_gleaning: bool,

    /// Selective gleaning: a chunk qualifies when the first pass found at least this many
    /// entities. No effect when `selective_gleaning` is false. Default 3.
    #[serde(default = "default_selective_min_entities")]
    pub selective_min_entities: usize,

    /// Selective gleaning: ...and at most this many relationships. No effect when
    /// `selective_gleaning` is false. Default 1.
    #[serde(default = "default_selective_max_relationships")]
    pub selective_max_relationships: usize,

    /// Output token cap for LLM extraction calls. Absent = use `llm.max_tokens`.
    #[serde(default)]
    pub extraction_max_tokens: Option<usize>,

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

    /// Path to a TOML/JSON ontology file; `None` = built-in generic ontology
    #[serde(default)]
    pub ontology_path: Option<String>,
}

impl Default for EntityExtractionTopLevelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: default_confidence_threshold(),
            use_gleaning: false,
            epistemic_status: false,
            selective_gleaning: false,
            selective_min_entities: default_selective_min_entities(),
            selective_max_relationships: default_selective_max_relationships(),
            extraction_max_tokens: None,
            max_gleaning_rounds: default_gleaning_rounds(),
            gleaning_improvement_threshold: default_gleaning_improvement(),
            semantic_merging: false,
            merge_similarity_threshold: default_merge_threshold(),
            automatic_linking: false,
            linking_confidence_threshold: default_confidence_threshold(),
            ontology_path: None,
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
fn reject_legacy_ollama<'de, D>(_deserializer: D) -> std::result::Result<Option<()>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Err(serde::de::Error::custom(
        "the `ollama` section was removed: rename it to `llm` and use `base_url` \
         (e.g. \"http://127.0.0.1:8080/v1\") instead of `host`/`port`",
    ))
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
fn default_extraction_temperature() -> f32 {
    0.0
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
fn default_batch_size() -> usize {
    100
}
fn default_worker_threads() -> usize {
    4
}
fn default_memory_limit() -> usize {
    1024
}
fn default_llm_base_url() -> String {
    crate::llm::DEFAULT_LLM_BASE_URL.to_string()
}
fn default_llm_embedding_base_url() -> String {
    crate::llm::DEFAULT_EMBEDDING_BASE_URL.to_string()
}
fn default_chat_model() -> String {
    "qwen3.5-9b".to_string()
}
fn default_llm_embedding_model() -> String {
    "multilingual-e5-small".to_string()
}
fn default_timeout() -> u64 {
    120
}
fn default_max_retries() -> u32 {
    3
}
fn default_max_context_tokens() -> u32 {
    crate::llm::DEFAULT_MAX_CONTEXT_TOKENS
}
fn default_max_concurrent_requests() -> usize {
    crate::llm::DEFAULT_MAX_CONCURRENT_REQUESTS
}
fn default_gleaning_rounds() -> usize {
    3
}
fn default_selective_min_entities() -> usize {
    3
}
fn default_selective_max_relationships() -> usize {
    1
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
    300 // 5 minutes
}
fn default_max_auto_save_versions() -> usize {
    5 // Keep 5 versions by default
}

// LazyGraphRAG default functions
fn default_min_concept_length() -> usize {
    3 // Minimum 3 characters for concepts
}
fn default_max_concept_words() -> usize {
    5 // Maximum 5 words per concept
}
fn default_co_occurrence_threshold() -> usize {
    1 // Minimum 1 shared chunk for relationship
}
fn default_max_refinement_iterations() -> usize {
    3 // Up to 3 query refinement iterations
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
    0.6 // 60% minimum confidence for pattern-based extraction
}
fn default_min_entity_frequency() -> usize {
    1 // Entities must appear at least once
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

impl Default for LlmSetConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            base_url: default_llm_base_url(),
            embedding_base_url: default_llm_embedding_base_url(),
            api_key_env: None,
            chat_model: default_chat_model(),
            embedding_model: default_llm_embedding_model(),
            timeout_seconds: default_timeout(),
            max_retries: default_max_retries(),
            fallback_to_hash: false,
            max_tokens: Some(800),
            temperature: Some(0.3),
            enable_caching: default_true(),
            max_context_tokens: default_max_context_tokens(),
            max_concurrent_requests: default_max_concurrent_requests(),
            extraction_cache: default_true(),
            disable_thinking: default_true(),
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
        let extension = path_ref.extension().and_then(|e| e.to_str()).unwrap_or("");

        let config: SetConfig = match extension {
            #[cfg(feature = "json5-support")]
            "json5" => {
                json5::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                    message: format!("JSON5 parse error: {e}"),
                })?
            },
            "json" => {
                serde_json::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                    message: format!("JSON parse error: {e}"),
                })?
            },
            #[cfg(not(feature = "json5-support"))]
            "json5" => {
                return Err(crate::core::GraphRAGError::Config {
                    message: "JSON5 support not enabled. Rebuild with --features json5-support"
                        .to_string(),
                });
            },
            "toml" | "tml" => {
                toml::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                    message: format!("TOML parse error: {e}"),
                })?
            },
            _ => {
                return Err(crate::core::GraphRAGError::Config {
                    message: format!("Unsupported configuration format: .{extension}"),
                })
            },
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

    /// Convert to the canonical runtime Config after semantic validation.
    pub fn to_graphrag_config(&self) -> Result<crate::Config> {
        let validation = self.validate();
        if !validation.is_valid {
            return Err(crate::core::GraphRAGError::Config {
                message: validation.errors.join("; "),
            });
        }
        Ok(self.to_graphrag_config_unchecked())
    }

    /// Convert to the canonical runtime Config for validation internals.
    pub(crate) fn to_graphrag_config_unchecked(&self) -> crate::Config {
        let mut config = crate::Config {
            approach: self.mode.approach.clone(),
            ..Default::default()
        };

        config.output_dir = self.general.output_dir.clone();

        // Map text processing
        config.text.chunk_size = self.pipeline.text_extraction.chunk_size;
        config.text.chunk_overlap = self.pipeline.text_extraction.chunk_overlap;

        // Map entity extraction based on approach
        config.entities.min_confidence = self.entity_extraction.min_confidence;

        // Map entity types from pipeline.entity_extraction
        if let Some(ref types) = self.pipeline.entity_extraction.entity_types {
            config.entities.entity_types = types.clone();
        }
        config.entities.ontology_path = self.entity_extraction.ontology_path.clone();
        config.entities.epistemic_status = self.entity_extraction.epistemic_status;
        config.entities.selective_gleaning = self.entity_extraction.selective_gleaning;
        config.entities.selective_min_entities = self.entity_extraction.selective_min_entities;
        config.entities.selective_max_relationships =
            self.entity_extraction.selective_max_relationships;
        config.entities.extraction_max_tokens = self.entity_extraction.extraction_max_tokens;

        // Configure gleaning based on approach:
        // - semantic: use LLM-based gleaning
        // - algorithmic: use pattern-based extraction
        // - hybrid: use both (for compatibility, map to gleaning)
        match self.mode.approach.as_str() {
            "semantic" => {
                if let Some(ref semantic) = self.semantic {
                    config.entities.use_gleaning = semantic.entity_extraction.use_gleaning;
                    config.entities.max_gleaning_rounds =
                        semantic.entity_extraction.max_gleaning_rounds;
                    config.entities.min_confidence =
                        semantic.entity_extraction.confidence_threshold;
                } else {
                    // No semantic sub-section: use top-level entity_extraction settings directly
                    config.entities.use_gleaning = self.entity_extraction.use_gleaning;
                    config.entities.max_gleaning_rounds =
                        self.entity_extraction.max_gleaning_rounds;
                    config.entities.min_confidence = self.entity_extraction.min_confidence;
                }
            },
            "algorithmic" => {
                // Algorithmic uses pattern-based extraction, no gleaning
                config.entities.use_gleaning = false;
                if let Some(ref algorithmic) = self.algorithmic {
                    config.entities.min_confidence =
                        algorithmic.entity_extraction.confidence_threshold;
                }
            },
            "hybrid" => {
                // Hybrid can use both, enable gleaning for LLM component
                config.entities.use_gleaning = true;
                if self.hybrid.is_some() {
                    // Use hybrid configuration if available
                    config.entities.max_gleaning_rounds = 2; // Reduced for hybrid efficiency
                }
            },
            _ => {
                // Unknown approach, use top-level config as fallback
                config.entities.use_gleaning = self.entity_extraction.use_gleaning;
                config.entities.max_gleaning_rounds = self.entity_extraction.max_gleaning_rounds;
            },
        }

        // Map graph building
        config.graph.similarity_threshold = self.pipeline.graph_building.min_relation_score;
        config.graph.max_connections = self.pipeline.graph_building.max_connections_per_node;
        config.graph.extract_relationships = true; // Enable by default for TOML configs
        config.graph.relationship_confidence_threshold = 0.5; // Default threshold

        // Map retrieval from the selected pipeline.  The runtime has one
        // canonical top_k value, while SetConfig keeps per-approach adapters.
        config.retrieval.top_k = match self.mode.approach.as_str() {
            "semantic" => self
                .semantic
                .as_ref()
                .map(|pipeline| pipeline.retrieval.top_k)
                .unwrap_or(config.retrieval.top_k),
            "algorithmic" => self
                .algorithmic
                .as_ref()
                .map(|pipeline| pipeline.retrieval.top_k)
                .unwrap_or(config.retrieval.top_k),
            "hybrid" => config.retrieval.top_k,
            _ => config.retrieval.top_k,
        };

        // Map embeddings: the second llama-server serves multilingual-e5-small.
        if self.llm.enabled {
            config.embeddings.dimension = 384;
            config.embeddings.backend = "llama_cpp".to_string();
            config.embeddings.model = Some(self.llm.embedding_model.clone());
            config.embeddings.api_endpoint = Some(format!(
                "{}/embeddings",
                self.llm.embedding_base_url.trim_end_matches('/')
            ));
        }
        config.embeddings.fallback_to_hash = self.llm.fallback_to_hash;

        // Map parallel processing
        config.parallel.enabled = self.pipeline.parallel_execution;
        config.parallel.num_threads = self.performance.worker_threads;

        // Map LLM configuration
        config.llm = crate::llm::LlmConfig {
            enabled: self.llm.enabled,
            base_url: self.llm.base_url.clone(),
            embedding_base_url: self.llm.embedding_base_url.clone(),
            api_key_env: self.llm.api_key_env.clone(),
            chat_model: self.llm.chat_model.clone(),
            embedding_model: self.llm.embedding_model.clone(),
            timeout_seconds: self.llm.timeout_seconds,
            max_retries: self.llm.max_retries,
            fallback_to_hash: self.llm.fallback_to_hash,
            max_tokens: self.llm.max_tokens,
            temperature: self.llm.temperature,
            enable_caching: self.llm.enable_caching,
            max_context_tokens: self.llm.max_context_tokens,
            max_concurrent_requests: self.llm.max_concurrent_requests,
            extraction_cache: self.llm.extraction_cache,
            disable_thinking: self.llm.disable_thinking,
        };

        // Map GLiNER configuration
        config.gliner = crate::config::GlinerConfig {
            enabled: self.gliner.enabled,
            model_path: self.gliner.model_path.clone(),
            tokenizer_path: self.gliner.tokenizer_path.clone(),
            mode: self.gliner.mode.clone(),
            entity_labels: self.gliner.entity_labels.clone(),
            relation_labels: self.gliner.relation_labels.clone(),
            entity_threshold: self.gliner.entity_threshold,
            relation_threshold: self.gliner.relation_threshold,
            use_gpu: self.gliner.use_gpu,
            max_concurrent_chunks: self.gliner.max_concurrent_chunks,
            llm_relations: self.gliner.llm_relations,
            backend: self.gliner.backend.clone(),
            jev: self.gliner.jev.clone(),
        };

        config.retrieval.hyde = self.retrieval.hyde;
        config.retrieval.graph_facts = self.retrieval.graph_facts;
        config.retrieval.map_reduce = self.retrieval.map_reduce;
        config.retrieval.map_reduce_chunks = self.retrieval.map_reduce_chunks;

        // Map reranker configuration
        config.reranker = crate::config::RerankerConfig {
            enabled: self.reranker.enabled,
            api_endpoint: self.reranker.api_endpoint.clone(),
            model: self.reranker.model.clone(),
            candidates: self.reranker.candidates,
            top_k: self.reranker.top_k,
            api_key_env: self.reranker.api_key_env.clone(),
            timeout_seconds: self.reranker.timeout_seconds,
        };

        config.chapter_summaries = crate::config::ChapterSummariesConfig {
            enabled: self.chapter_summaries.enabled,
            max_tokens: self.chapter_summaries.max_tokens,
            max_in_context: self.chapter_summaries.max_in_context,
        };

        config.chunk_context = crate::config::ChunkContextConfig {
            enabled: self.chunk_context.enabled,
            window: self.chunk_context.window,
            max_tokens: self.chunk_context.max_tokens,
        };

        // Map auto-save configuration
        config.auto_save = crate::config::AutoSaveConfig {
            enabled: self.auto_save.enabled,
            base_dir: self.auto_save.base_dir.clone(),
            interval_seconds: self.auto_save.interval_seconds,
            workspace_name: self.auto_save.workspace_name.clone(),
            max_versions: self.auto_save.max_versions,
        };

        config
    }
}

#[cfg(test)]
mod drift_guard_tests {
    //! Guards against silent drift between the serde-facing `*SetConfig` leaf
    //! structs and their canonical runtime counterparts in `config/mod.rs` /
    //! `llm/mod.rs`. These structs are mechanical mirrors kept in sync by hand
    //! (the "5-point-sync" convention); these tests fail loudly when
    //! a field's default diverges so the drift is caught at build time.
    //!
    //! NOTE: `LlmConfig` is *intentionally* excluded — its runtime default is
    //! offline-first (`enabled = false`, `fallback_to_hash = true`) while
    //! `LlmSetConfig` is the user-facing "I want an LLM" schema
    //! (`enabled = true`). That divergence is by design, not drift.

    use super::*;
    use crate::config::{AutoSaveConfig, GlinerConfig};
    #[test]
    fn reranker_setconfig_default_matches_runtime() {
        let set = RerankerSetConfig::default();
        let runtime = crate::config::RerankerConfig::default();
        assert_eq!(set.enabled, runtime.enabled);
        assert_eq!(set.api_endpoint, runtime.api_endpoint);
        assert_eq!(set.model, runtime.model);
        assert_eq!(set.candidates, runtime.candidates);
        assert_eq!(set.top_k, runtime.top_k);
        assert_eq!(set.api_key_env, runtime.api_key_env);
        assert_eq!(set.timeout_seconds, runtime.timeout_seconds);
        assert!(!runtime.enabled, "reranker must stay opt-in");
    }

    #[test]
    fn gliner_setconfig_default_matches_runtime() {
        let set = GlinerSetConfig::default();
        let runtime = GlinerConfig::default();
        assert_eq!(set.mode, runtime.mode, "gliner.mode drifted");
        assert_eq!(
            set.entity_labels, runtime.entity_labels,
            "gliner.entity_labels drifted"
        );
        assert_eq!(
            set.relation_labels, runtime.relation_labels,
            "gliner.relation_labels drifted"
        );
        assert_eq!(
            set.entity_threshold, runtime.entity_threshold,
            "gliner.entity_threshold drifted"
        );
        assert_eq!(
            set.relation_threshold, runtime.relation_threshold,
            "gliner.relation_threshold drifted"
        );
        assert_eq!(set.use_gpu, runtime.use_gpu, "gliner.use_gpu drifted");
        assert_eq!(
            set.llm_relations, runtime.llm_relations,
            "gliner.llm_relations drifted"
        );
        assert_eq!(set.backend, runtime.backend, "gliner.backend drifted");
    }

    #[test]
    fn autosave_setconfig_default_matches_runtime() {
        let set = AutoSaveSetConfig::default();
        let runtime = AutoSaveConfig::default();
        assert_eq!(set.enabled, runtime.enabled, "auto_save.enabled drifted");
        assert_eq!(
            set.interval_seconds, runtime.interval_seconds,
            "auto_save.interval_seconds drifted"
        );
        assert_eq!(
            set.max_versions, runtime.max_versions,
            "auto_save.max_versions drifted"
        );
    }
}

#[cfg(test)]
mod llm_concurrency_config_tests {
    use super::*;

    #[test]
    fn llm_concurrency_and_cache_defaults_and_override() {
        let default = LlmSetConfig::default();
        assert_eq!(default.max_concurrent_requests, 4);
        assert!(default.extraction_cache);
        let runtime = crate::llm::LlmConfig::default();
        assert_eq!(runtime.max_concurrent_requests, 4);
        assert!(runtime.extraction_cache);

        let set: SetConfig =
            toml::from_str("[llm]\nmax_concurrent_requests = 2\nextraction_cache = false\n")
                .unwrap();
        let config = set.to_graphrag_config().unwrap();
        assert_eq!(config.llm.max_concurrent_requests, 2);
        assert!(!config.llm.extraction_cache);
    }
}

#[cfg(test)]
mod llm_disable_thinking_config_tests {
    use super::*;

    #[test]
    fn disable_thinking_defaults_true_and_overrides() {
        assert!(LlmSetConfig::default().disable_thinking);
        assert!(crate::llm::LlmConfig::default().disable_thinking);
        let set: SetConfig = toml::from_str("[llm]\ndisable_thinking = false\n").unwrap();
        assert!(!set.to_graphrag_config().unwrap().llm.disable_thinking);
        let set: SetConfig = toml::from_str("[llm]\nenabled = true\n").unwrap();
        assert!(set.to_graphrag_config().unwrap().llm.disable_thinking);
    }
}

#[cfg(test)]
mod reranker_config_tests {
    use super::*;

    #[test]
    fn reranker_is_disabled_by_default_and_maps_overrides() {
        let config = SetConfig::default().to_graphrag_config().unwrap();
        assert!(!config.reranker.enabled);
        assert_eq!(config.reranker, crate::config::RerankerConfig::default());

        let set: SetConfig = toml::from_str(
            "[reranker]\nenabled = true\ncandidates = 12\ntop_k = 3\napi_key_env = \"RERANK_KEY\"\n",
        )
        .unwrap();
        let r = set.to_graphrag_config().unwrap().reranker;
        assert!(r.enabled);
        assert_eq!((r.candidates, r.top_k), (12, 3));
        assert_eq!(r.api_key_env.as_deref(), Some("RERANK_KEY"));
        assert_eq!(r.api_endpoint, "http://localhost:8082");

        assert!(toml::from_str::<SetConfig>("[reranker]\nbogus = 1\n").is_err());
    }
}

#[cfg(test)]
mod gliner_llm_relations_tests {
    use super::*;

    #[test]
    fn llm_relations_defaults_false_everywhere() {
        assert!(!GlinerSetConfig::default().llm_relations);
        assert!(!crate::config::GlinerConfig::default().llm_relations);
        let set: SetConfig = toml::from_str("[gliner]\nenabled = true\n").unwrap();
        assert!(!set.to_graphrag_config().unwrap().gliner.llm_relations);
    }

    #[test]
    fn llm_relations_maps_from_setconfig_toml() {
        let set: SetConfig =
            toml::from_str("[gliner]\nenabled = true\nllm_relations = true\n").unwrap();
        assert!(set.to_graphrag_config().unwrap().gliner.llm_relations);
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn llm_relations_parses_from_json5_set_and_runtime_configs() {
        // SetConfig document (has `mode`)
        let set: SetConfig = json5::from_str(
            r#"{ mode: { approach: "semantic" }, gliner: { llm_relations: true } }"#,
        )
        .unwrap();
        assert!(set.to_graphrag_config().unwrap().gliner.llm_relations);
        // runtime Config document
        let config =
            crate::config::Config::from_json5_str(r#"{ gliner: { llm_relations: true } }"#)
                .unwrap();
        assert!(config.gliner.llm_relations);
        let config =
            crate::config::Config::from_json5_str(r#"{ gliner: { enabled: true } }"#).unwrap();
        assert!(!config.gliner.llm_relations);
    }
}

#[cfg(test)]
mod hyde_config_tests {
    use super::*;

    #[test]
    fn hyde_is_off_by_default_everywhere() {
        assert!(!crate::config::Config::default().retrieval.hyde);
        assert!(
            !SetConfig::default()
                .to_graphrag_config()
                .unwrap()
                .retrieval
                .hyde
        );
    }

    #[test]
    fn hyde_maps_from_toml_set_config() {
        let set: SetConfig = toml::from_str("[retrieval]\nhyde = true\n").unwrap();
        assert!(set.to_graphrag_config().unwrap().retrieval.hyde);
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn hyde_loads_from_json5_via_cli_loader() {
        use crate::config::loader::{load_config_from_str, ConfigFormat};
        // SetConfig document (has `mode`): the shape of the bench configs
        let set = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, retrieval: { hyde: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(set.retrieval.hyde);
        // runtime Config document
        let runtime =
            load_config_from_str(r#"{ retrieval: { hyde: true } }"#, ConfigFormat::Json5).unwrap();
        assert!(runtime.retrieval.hyde);
        // absent key keeps the default
        let off =
            load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                .unwrap();
        assert!(!off.retrieval.hyde);
    }
}

#[cfg(test)]
mod graph_facts_config_tests {
    use super::*;

    #[test]
    fn graph_facts_is_off_by_default_everywhere() {
        assert!(!crate::config::Config::default().retrieval.graph_facts);
        assert!(
            !SetConfig::default()
                .to_graphrag_config()
                .unwrap()
                .retrieval
                .graph_facts
        );
    }

    #[test]
    fn graph_facts_maps_from_toml_set_config() {
        let set: SetConfig = toml::from_str("[retrieval]\ngraph_facts = true\n").unwrap();
        assert!(set.to_graphrag_config().unwrap().retrieval.graph_facts);
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn graph_facts_loads_from_json5_via_cli_loader() {
        use crate::config::loader::{load_config_from_str, ConfigFormat};
        let set = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, retrieval: { graph_facts: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(set.retrieval.graph_facts);
        let runtime = load_config_from_str(
            r#"{ retrieval: { graph_facts: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(runtime.retrieval.graph_facts);
        let off =
            load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                .unwrap();
        assert!(!off.retrieval.graph_facts);
    }
}
#[cfg(test)]
mod map_reduce_config_tests {
    use super::*;

    #[test]
    fn map_reduce_is_off_by_default_everywhere() {
        let runtime = crate::config::Config::default().retrieval;
        assert!(!runtime.map_reduce);
        assert_eq!(runtime.map_reduce_chunks, 60);
        let set = RetrievalSetConfig::default();
        assert!(!set.map_reduce);
        assert_eq!(set.map_reduce_chunks, 60);
        let mapped = SetConfig::default().to_graphrag_config().unwrap().retrieval;
        assert!(!mapped.map_reduce);
        assert_eq!(mapped.map_reduce_chunks, 60);
    }

    #[test]
    fn map_reduce_maps_from_toml_set_config() {
        let set: SetConfig =
            toml::from_str("[retrieval]\nmap_reduce = true\nmap_reduce_chunks = 12\n").unwrap();
        let r = set.to_graphrag_config().unwrap().retrieval;
        assert!(r.map_reduce);
        assert_eq!(r.map_reduce_chunks, 12);
        let only_flag: SetConfig = toml::from_str("[retrieval]\nmap_reduce = true\n").unwrap();
        assert_eq!(only_flag.retrieval.map_reduce_chunks, 60);
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn map_reduce_loads_from_json5_via_cli_loader() {
        use crate::config::loader::{load_config_from_str, ConfigFormat};
        let set = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, retrieval: { map_reduce: true, map_reduce_chunks: 20 } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(set.retrieval.map_reduce);
        assert_eq!(set.retrieval.map_reduce_chunks, 20);
        let runtime = load_config_from_str(
            r#"{ retrieval: { map_reduce: true, map_reduce_chunks: 7 } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(runtime.retrieval.map_reduce);
        assert_eq!(runtime.retrieval.map_reduce_chunks, 7);
        let off =
            load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                .unwrap();
        assert!(!off.retrieval.map_reduce);
        assert_eq!(off.retrieval.map_reduce_chunks, 60);
    }
}

#[cfg(test)]
mod chapter_summaries_config_tests {
    use super::*;

    #[test]
    fn chapter_summaries_are_off_by_default_everywhere() {
        let runtime = crate::config::Config::default().chapter_summaries;
        assert!(!runtime.enabled);
        assert_eq!(runtime.max_tokens, 400);
        assert_eq!(runtime.max_in_context, 2);
        let set = ChapterSummariesSetConfig::default();
        assert!(!set.enabled);
        assert_eq!(set.max_tokens, 400);
        assert_eq!(set.max_in_context, 2);
        let mapped = SetConfig::default()
            .to_graphrag_config()
            .unwrap()
            .chapter_summaries;
        assert_eq!(mapped, runtime);
    }

    #[test]
    fn chapter_summaries_map_from_toml_set_config() {
        let set: SetConfig = toml::from_str(
            "[chapter_summaries]\nenabled = true\nmax_tokens = 250\nmax_in_context = 5\n",
        )
        .unwrap();
        let c = set.to_graphrag_config().unwrap().chapter_summaries;
        assert!(c.enabled);
        assert_eq!(c.max_tokens, 250);
        assert_eq!(c.max_in_context, 5);
        let only_flag: SetConfig = toml::from_str("[chapter_summaries]\nenabled = true\n").unwrap();
        assert_eq!(only_flag.chapter_summaries.max_tokens, 400);
        assert_eq!(only_flag.chapter_summaries.max_in_context, 2);
        assert_eq!(
            only_flag
                .to_graphrag_config()
                .unwrap()
                .chapter_summaries
                .max_in_context,
            2
        );
        assert!(toml::from_str::<SetConfig>("[chapter_summaries]\nbogus = 1\n").is_err());
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn chapter_summaries_load_from_json5_via_cli_loader() {
        use crate::config::loader::{load_config_from_str, ConfigFormat};
        let set = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, chapter_summaries: { enabled: true, max_tokens: 300, max_in_context: 3 } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(set.chapter_summaries.enabled);
        assert_eq!(set.chapter_summaries.max_tokens, 300);
        assert_eq!(set.chapter_summaries.max_in_context, 3);
        let runtime = load_config_from_str(
            r#"{ chapter_summaries: { enabled: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(runtime.chapter_summaries.enabled);
        assert_eq!(runtime.chapter_summaries.max_tokens, 400);
        assert_eq!(runtime.chapter_summaries.max_in_context, 2);
        let off =
            load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                .unwrap();
        assert!(!off.chapter_summaries.enabled);
    }
}

#[cfg(test)]
mod chunk_context_config_tests {
    use super::*;

    #[test]
    fn chunk_context_is_off_by_default_everywhere() {
        let runtime = crate::config::Config::default().chunk_context;
        assert!(!runtime.enabled);
        assert_eq!(runtime.window, 4);
        assert_eq!(runtime.max_tokens, 120);
        let set = ChunkContextSetConfig::default();
        assert!(!set.enabled);
        assert_eq!(set.window, 4);
        assert_eq!(set.max_tokens, 120);
        let mapped = SetConfig::default()
            .to_graphrag_config()
            .unwrap()
            .chunk_context;
        assert_eq!(mapped, runtime);
        let from_toml: SetConfig = toml::from_str("").unwrap();
        assert!(!from_toml.chunk_context.enabled);
        assert!(
            !from_toml
                .to_graphrag_config()
                .unwrap()
                .chunk_context
                .enabled
        );
    }

    #[test]
    fn chunk_context_maps_from_toml_set_config() {
        let set: SetConfig =
            toml::from_str("[chunk_context]\nenabled = true\nwindow = 2\nmax_tokens = 80\n")
                .unwrap();
        let c = set.to_graphrag_config().unwrap().chunk_context;
        assert!(c.enabled);
        assert_eq!(c.window, 2);
        assert_eq!(c.max_tokens, 80);
        let only_flag: SetConfig = toml::from_str("[chunk_context]\nenabled = true\n").unwrap();
        let c = only_flag.to_graphrag_config().unwrap().chunk_context;
        assert_eq!((c.window, c.max_tokens), (4, 120));
        assert!(toml::from_str::<SetConfig>("[chunk_context]\nbogus = 1\n").is_err());
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn chunk_context_loads_from_json5_via_cli_loader() {
        use crate::config::loader::{load_config_from_str, ConfigFormat};
        let set = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, chunk_context: { enabled: true, window: 2, max_tokens: 80 } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(set.chunk_context.enabled);
        assert_eq!(set.chunk_context.window, 2);
        assert_eq!(set.chunk_context.max_tokens, 80);
        let only_flag = load_config_from_str(
            r#"{ chunk_context: { enabled: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert_eq!(only_flag.chunk_context.window, 4);
        assert_eq!(only_flag.chunk_context.max_tokens, 120);
        let off =
            load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                .unwrap();
        assert!(!off.chunk_context.enabled);
    }
}

#[cfg(test)]
mod gliner_jev_tests {
    use super::*;
    #[cfg(feature = "json5-support")]
    use crate::config::loader::{load_config_from_str, ConfigFormat};

    #[test]
    fn jev_defaults_and_overrides() {
        let d = crate::config::JevConfig::default();
        assert!(!d.enabled);
        assert_eq!(d.api_endpoint, "https://api.typesafe.ai/v1/systemone");
        assert_eq!(d.api_key_env, "TYPESAFE_API_KEY");
        assert_eq!(
            (d.max_entities_per_chunk, d.max_concurrent, d.timeout_secs),
            (8, 4, 120)
        );
        assert_eq!(
            (
                d.candidate_floor,
                d.min_type_prob,
                d.min_link,
                d.min_evidence
            ),
            (0.2, 0.0, 0.5, 0.5)
        );
        assert!(d.classify_entity_types && d.raw_log_path.is_none());
        assert!(d.filter_entities && d.min_entity == 0.5);

        let set: SetConfig = toml::from_str("[gliner]\nenabled = true\n").unwrap();
        assert!(!set.to_graphrag_config().unwrap().gliner.jev.enabled);
        let set: SetConfig =
            toml::from_str("[gliner.jev]\nenabled = true\nmin_link = 0.7\n").unwrap();
        let j = set.to_graphrag_config().unwrap().gliner.jev;
        assert!(j.enabled && j.min_link == 0.7 && j.min_evidence == 0.5);
        #[cfg(feature = "json5-support")]
        {
            let set = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, gliner: { jev: { enabled: true, raw_log_path: "/x/y.jsonl" } } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
            let j = set.gliner.jev;
            assert!(j.enabled && j.raw_log_path.as_deref() == Some("/x/y.jsonl"));
        }
    }

    #[cfg(test)]
    mod epistemic_status_config_tests {
        use super::*;

        #[test]
        fn epistemic_status_is_off_by_default_everywhere() {
            assert!(!crate::config::Config::default().entities.epistemic_status);
            assert!(!EntityExtractionTopLevelConfig::default().epistemic_status);
            let mapped = SetConfig::default().to_graphrag_config().unwrap();
            assert!(!mapped.entities.epistemic_status);
        }

        #[test]
        fn epistemic_status_maps_from_toml_in_every_approach() {
            for approach in ["semantic", "algorithmic", "hybrid"] {
                let set: SetConfig = toml::from_str(&format!(
                "[mode]\napproach = \"{approach}\"\n[entity_extraction]\nepistemic_status = true\n"
            ))
                .unwrap();
                assert!(
                    set.to_graphrag_config().unwrap().entities.epistemic_status,
                    "approach {approach}"
                );
            }
            // semantic without a `semantic` sub-section, flag not set
            let set: SetConfig = toml::from_str("[mode]\napproach = \"semantic\"\n").unwrap();
            assert!(set.semantic.is_none());
            assert!(!set.to_graphrag_config().unwrap().entities.epistemic_status);
        }

        #[test]
        fn epistemic_status_defaults_in_runtime_config_serde() {
            let json = serde_json::to_string(&crate::config::Config::default()).unwrap();
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value["entities"]["epistemic_status"] = true.into();
            let cfg: crate::config::Config = serde_json::from_value(value).unwrap();
            assert!(cfg.entities.epistemic_status);
            // absent field deserializes to false (old runtime configs)
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value["entities"]
                .as_object_mut()
                .unwrap()
                .remove("epistemic_status");
            let cfg: crate::config::Config = serde_json::from_value(value).unwrap();
            assert!(!cfg.entities.epistemic_status);
        }

        #[cfg(feature = "json5-support")]
        #[test]
        fn epistemic_status_loads_from_json5() {
            use crate::config::loader::{load_config_from_str, ConfigFormat};
            let on = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, entity_extraction: { epistemic_status: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
            assert!(on.entities.epistemic_status);
            let off =
                load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                    .unwrap();
            assert!(!off.entities.epistemic_status);
        }
    }
}

#[cfg(test)]
mod selective_gleaning_config_tests {
    use super::*;

    #[test]
    fn selective_gleaning_is_off_by_default_everywhere() {
        assert!(!crate::config::Config::default().entities.selective_gleaning);
        assert!(!EntityExtractionTopLevelConfig::default().selective_gleaning);
        let mapped = SetConfig::default().to_graphrag_config().unwrap();
        assert!(!mapped.entities.selective_gleaning);
        assert_eq!(
            crate::config::Config::default()
                .entities
                .selective_min_entities,
            3
        );
        assert_eq!(
            crate::config::Config::default()
                .entities
                .selective_max_relationships,
            1
        );
        assert_eq!(
            EntityExtractionTopLevelConfig::default().selective_min_entities,
            3
        );
        assert_eq!(
            EntityExtractionTopLevelConfig::default().selective_max_relationships,
            1
        );
        assert_eq!(mapped.entities.selective_min_entities, 3);
        assert_eq!(mapped.entities.selective_max_relationships, 1);
    }

    #[test]
    fn extraction_max_tokens_is_unset_by_default_and_maps_when_set() {
        assert_eq!(
            crate::config::Config::default()
                .entities
                .extraction_max_tokens,
            None
        );
        assert_eq!(
            EntityExtractionTopLevelConfig::default().extraction_max_tokens,
            None
        );
        let unset = SetConfig::default().to_graphrag_config().unwrap();
        assert_eq!(unset.entities.extraction_max_tokens, None);
        let mut set = SetConfig::default();
        set.entity_extraction.extraction_max_tokens = Some(4096);
        let mapped = set.to_graphrag_config().unwrap();
        assert_eq!(mapped.entities.extraction_max_tokens, Some(4096));
    }

    #[test]
    fn selective_gleaning_maps_from_toml_in_every_approach() {
        for approach in ["semantic", "algorithmic", "hybrid"] {
            let set: SetConfig = toml::from_str(&format!(
                "[mode]\napproach = \"{approach}\"\n[entity_extraction]\nselective_gleaning = true\n"
            ))
            .unwrap();
            assert!(
                set.to_graphrag_config()
                    .unwrap()
                    .entities
                    .selective_gleaning,
                "approach {approach}"
            );
        }
        let set: SetConfig = toml::from_str("[mode]\napproach = \"semantic\"\n").unwrap();
        assert!(set.semantic.is_none());
        assert!(
            !set.to_graphrag_config()
                .unwrap()
                .entities
                .selective_gleaning
        );
    }

    #[test]
    fn selective_thresholds_map_from_toml() {
        let set: SetConfig = toml::from_str(
            "[entity_extraction]\nselective_min_entities = 2\nselective_max_relationships = 4\n",
        )
        .unwrap();
        let cfg = set.to_graphrag_config().unwrap();
        assert_eq!(cfg.entities.selective_min_entities, 2);
        assert_eq!(cfg.entities.selective_max_relationships, 4);
    }

    #[test]
    fn selective_gleaning_defaults_in_runtime_config_serde() {
        let json = serde_json::to_string(&crate::config::Config::default()).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["entities"]["selective_gleaning"] = true.into();
        let cfg: crate::config::Config = serde_json::from_value(value).unwrap();
        assert!(cfg.entities.selective_gleaning);
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["entities"]
            .as_object_mut()
            .unwrap()
            .remove("selective_gleaning");
        let cfg: crate::config::Config = serde_json::from_value(value).unwrap();
        assert!(!cfg.entities.selective_gleaning);
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let entities = value["entities"].as_object_mut().unwrap();
        entities.remove("selective_min_entities");
        entities.remove("selective_max_relationships");
        let cfg: crate::config::Config = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.entities.selective_min_entities, 3);
        assert_eq!(cfg.entities.selective_max_relationships, 1);
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn selective_gleaning_loads_from_json5() {
        use crate::config::loader::{load_config_from_str, ConfigFormat};
        let on = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, entity_extraction: { selective_gleaning: true } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert!(on.entities.selective_gleaning);
        let off =
            load_config_from_str(r#"{ mode: { approach: "semantic" } }"#, ConfigFormat::Json5)
                .unwrap();
        assert!(!off.entities.selective_gleaning);
        let custom = load_config_from_str(
            r#"{ mode: { approach: "semantic" }, entity_extraction: { selective_min_entities: 2, selective_max_relationships: 4 } }"#,
            ConfigFormat::Json5,
        )
        .unwrap();
        assert_eq!(custom.entities.selective_min_entities, 2);
        assert_eq!(custom.entities.selective_max_relationships, 4);
    }
}
