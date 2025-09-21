//! TOML Configuration System for GraphRAG
//! Complete configuration management with extensive TOML support

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Complete GraphRAG configuration loaded from TOML
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TomlConfig {
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
    pub ollama: OllamaTomlConfig,

    /// Experimental features
    #[serde(default)]
    pub experimental: ExperimentalConfig,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgreSQLConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    #[serde(default)]
    pub encrypted: bool,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMParamsConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_top_p")]
    pub top_p: f32,

    #[serde(default)]
    pub frequency_penalty: f32,

    #[serde(default)]
    pub presence_penalty: f32,

    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelsConfig {
    #[serde(default = "default_ollama_url")]
    pub ollama_base_url: String,

    #[serde(default = "default_ollama_model")]
    pub model_name: String,

    #[serde(default = "default_ollama_embedding")]
    pub embedding_model: String,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTomlConfig {
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

impl Default for OllamaTomlConfig {
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

impl TomlConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: TomlConfig =
            toml::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("TOML parse error: {e}"),
            })?;
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
        let mut config = crate::Config::default();

        // Map text processing
        config.text.chunk_size = self.pipeline.text_extraction.chunk_size;
        config.text.chunk_overlap = self.pipeline.text_extraction.chunk_overlap;

        // Map entity extraction
        config.entities.min_confidence = self
            .pipeline
            .entity_extraction
            .filters
            .as_ref()
            .map(|f| f.confidence_threshold)
            .unwrap_or(0.7);

        // Map graph building
        config.graph.similarity_threshold = self.pipeline.graph_building.min_relation_score;
        config.graph.max_connections = self.pipeline.graph_building.max_connections_per_node;

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
        };

        config
    }
}
