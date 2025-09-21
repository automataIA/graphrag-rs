use crate::Result;
use std::fs;

pub mod enhancements;
pub mod loader;
pub mod toml_config;

pub use toml_config::TomlConfig;

/// Configuration for the GraphRAG system
#[derive(Debug, Clone)]
pub struct Config {
    /// Vector embedding configuration
    pub embeddings: EmbeddingConfig,

    /// Graph construction parameters
    pub graph: GraphConfig,

    /// Text processing settings
    pub text: TextConfig,

    /// Entity extraction settings
    pub entities: EntityConfig,

    /// Retrieval system configuration
    pub retrieval: RetrievalConfig,

    /// Parallel processing configuration
    pub parallel: ParallelConfig,

    /// Ollama integration configuration
    pub ollama: crate::ollama::OllamaConfig,

    /// Latest enhancements configuration
    pub enhancements: enhancements::EnhancementsConfig,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Dimension of the embedding vectors
    pub dimension: usize,

    /// Embedding backend: "hash", "ollama", or "candle"
    pub backend: String,

    /// Whether to fallback to hash-based embeddings if primary backend fails
    pub fallback_to_hash: bool,

    /// API endpoint for embeddings (if using external service)
    pub api_endpoint: Option<String>,

    /// API key for external embedding service
    pub api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of connections per node
    pub max_connections: usize,

    /// Similarity threshold for creating edges
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct TextConfig {
    /// Maximum chunk size for text processing
    pub chunk_size: usize,

    /// Overlap between chunks
    pub chunk_overlap: usize,

    /// Languages to support for text processing
    pub languages: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EntityConfig {
    /// Minimum confidence score for entity extraction
    pub min_confidence: f32,

    /// Types of entities to extract
    pub entity_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Number of top results to return
    pub top_k: usize,

    /// Search algorithm to use
    pub search_algorithm: String,
}

#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use for parallel processing (0 = auto-detect)
    pub num_threads: usize,

    /// Enable parallel processing
    pub enabled: bool,

    /// Minimum batch size for parallel processing
    pub min_batch_size: usize,

    /// Chunk size for parallel text processing
    pub chunk_batch_size: usize,

    /// Parallel processing for embeddings
    pub parallel_embeddings: bool,

    /// Parallel graph construction
    pub parallel_graph_ops: bool,

    /// Parallel vector operations
    pub parallel_vector_ops: bool,
}

// Default value functions
fn default_embedding_dim() -> usize {
    384
}
fn default_embedding_backend() -> String {
    "hash".to_string()
}
fn default_max_connections() -> usize {
    10
}
fn default_similarity_threshold() -> f32 {
    0.8
}
fn default_chunk_size() -> usize {
    1000
}
fn default_chunk_overlap() -> usize {
    200
}
fn default_languages() -> Vec<String> {
    vec!["en".to_string()]
}
fn default_min_confidence() -> f32 {
    0.7
}
fn default_entity_types() -> Vec<String> {
    vec![
        "PERSON".to_string(),
        "ORG".to_string(),
        "LOCATION".to_string(),
    ]
}
fn default_top_k() -> usize {
    10
}
fn default_search_algorithm() -> String {
    "cosine".to_string()
}
fn default_num_threads() -> usize {
    0
} // Auto-detect
fn default_min_batch_size() -> usize {
    10
}
fn default_chunk_batch_size() -> usize {
    100
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embeddings: EmbeddingConfig {
                dimension: default_embedding_dim(),
                backend: default_embedding_backend(),
                fallback_to_hash: true,
                api_endpoint: None,
                api_key: None,
            },
            graph: GraphConfig {
                max_connections: default_max_connections(),
                similarity_threshold: default_similarity_threshold(),
            },
            text: TextConfig {
                chunk_size: default_chunk_size(),
                chunk_overlap: default_chunk_overlap(),
                languages: default_languages(),
            },
            entities: EntityConfig {
                min_confidence: default_min_confidence(),
                entity_types: default_entity_types(),
            },
            retrieval: RetrievalConfig {
                top_k: default_top_k(),
                search_algorithm: default_search_algorithm(),
            },
            parallel: ParallelConfig {
                num_threads: default_num_threads(),
                enabled: true,
                min_batch_size: default_min_batch_size(),
                chunk_batch_size: default_chunk_batch_size(),
                parallel_embeddings: true,
                parallel_graph_ops: true,
                parallel_vector_ops: true,
            },
            ollama: crate::ollama::OllamaConfig::default(),
            enhancements: enhancements::EnhancementsConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a JSON file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let parsed = json::parse(&content)?;

        let config = Config {
            embeddings: EmbeddingConfig {
                dimension: parsed["embeddings"]["dimension"]
                    .as_usize()
                    .unwrap_or(default_embedding_dim()),
                backend: parsed["embeddings"]["backend"]
                    .as_str()
                    .unwrap_or(&default_embedding_backend())
                    .to_string(),
                fallback_to_hash: parsed["embeddings"]["fallback_to_hash"]
                    .as_bool()
                    .unwrap_or(true),
                api_endpoint: parsed["embeddings"]["api_endpoint"]
                    .as_str()
                    .map(|s| s.to_string()),
                api_key: parsed["embeddings"]["api_key"]
                    .as_str()
                    .map(|s| s.to_string()),
            },
            graph: GraphConfig {
                max_connections: parsed["graph"]["max_connections"]
                    .as_usize()
                    .unwrap_or(default_max_connections()),
                similarity_threshold: parsed["graph"]["similarity_threshold"]
                    .as_f32()
                    .unwrap_or(default_similarity_threshold()),
            },
            text: TextConfig {
                chunk_size: parsed["text"]["chunk_size"]
                    .as_usize()
                    .unwrap_or(default_chunk_size()),
                chunk_overlap: parsed["text"]["chunk_overlap"]
                    .as_usize()
                    .unwrap_or(default_chunk_overlap()),
                languages: if parsed["text"]["languages"].is_array() {
                    parsed["text"]["languages"]
                        .members()
                        .map(|v| v.as_str().unwrap_or("en").to_string())
                        .collect()
                } else {
                    default_languages()
                },
            },
            entities: EntityConfig {
                min_confidence: parsed["entities"]["min_confidence"]
                    .as_f32()
                    .unwrap_or(default_min_confidence()),
                entity_types: if parsed["entities"]["entity_types"].is_array() {
                    parsed["entities"]["entity_types"]
                        .members()
                        .map(|v| v.as_str().unwrap_or("PERSON").to_string())
                        .collect()
                } else {
                    default_entity_types()
                },
            },
            retrieval: RetrievalConfig {
                top_k: parsed["retrieval"]["top_k"]
                    .as_usize()
                    .unwrap_or(default_top_k()),
                search_algorithm: parsed["retrieval"]["search_algorithm"]
                    .as_str()
                    .unwrap_or(&default_search_algorithm())
                    .to_string(),
            },
            parallel: ParallelConfig {
                num_threads: parsed["parallel"]["num_threads"]
                    .as_usize()
                    .unwrap_or(default_num_threads()),
                enabled: parsed["parallel"]["enabled"].as_bool().unwrap_or(true),
                min_batch_size: parsed["parallel"]["min_batch_size"]
                    .as_usize()
                    .unwrap_or(default_min_batch_size()),
                chunk_batch_size: parsed["parallel"]["chunk_batch_size"]
                    .as_usize()
                    .unwrap_or(default_chunk_batch_size()),
                parallel_embeddings: parsed["parallel"]["parallel_embeddings"]
                    .as_bool()
                    .unwrap_or(true),
                parallel_graph_ops: parsed["parallel"]["parallel_graph_ops"]
                    .as_bool()
                    .unwrap_or(true),
                parallel_vector_ops: parsed["parallel"]["parallel_vector_ops"]
                    .as_bool()
                    .unwrap_or(true),
            },
            ollama: crate::ollama::OllamaConfig {
                enabled: parsed["ollama"]["enabled"].as_bool().unwrap_or(false),
                host: parsed["ollama"]["host"]
                    .as_str()
                    .unwrap_or("http://localhost")
                    .to_string(),
                port: parsed["ollama"]["port"].as_u16().unwrap_or(11434),
                embedding_model: parsed["ollama"]["embedding_model"]
                    .as_str()
                    .unwrap_or("nomic-embed-text")
                    .to_string(),
                chat_model: parsed["ollama"]["chat_model"]
                    .as_str()
                    .unwrap_or("llama3.2:3b")
                    .to_string(),
                timeout_seconds: parsed["ollama"]["timeout_seconds"].as_u64().unwrap_or(30),
                max_retries: parsed["ollama"]["max_retries"].as_u32().unwrap_or(3),
                fallback_to_hash: parsed["ollama"]["fallback_to_hash"]
                    .as_bool()
                    .unwrap_or(true),
                max_tokens: parsed["ollama"]["max_tokens"].as_u32(),
                temperature: parsed["ollama"]["temperature"].as_f32(),
            },
            enhancements: enhancements::EnhancementsConfig {
                enabled: parsed["enhancements"]["enabled"].as_bool().unwrap_or(true),
                query_analysis: enhancements::QueryAnalysisConfig {
                    enabled: parsed["enhancements"]["query_analysis"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    min_confidence: parsed["enhancements"]["query_analysis"]["min_confidence"]
                        .as_f32()
                        .unwrap_or(0.6),
                    enable_strategy_suggestion: parsed["enhancements"]["query_analysis"]
                        ["enable_strategy_suggestion"]
                        .as_bool()
                        .unwrap_or(true),
                    enable_keyword_analysis: parsed["enhancements"]["query_analysis"]
                        ["enable_keyword_analysis"]
                        .as_bool()
                        .unwrap_or(true),
                    enable_complexity_scoring: parsed["enhancements"]["query_analysis"]
                        ["enable_complexity_scoring"]
                        .as_bool()
                        .unwrap_or(true),
                },
                adaptive_retrieval: enhancements::AdaptiveRetrievalConfig {
                    enabled: parsed["enhancements"]["adaptive_retrieval"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    use_query_analysis: parsed["enhancements"]["adaptive_retrieval"]
                        ["use_query_analysis"]
                        .as_bool()
                        .unwrap_or(true),
                    enable_cross_strategy_fusion: parsed["enhancements"]["adaptive_retrieval"]
                        ["enable_cross_strategy_fusion"]
                        .as_bool()
                        .unwrap_or(true),
                    diversity_threshold: parsed["enhancements"]["adaptive_retrieval"]
                        ["diversity_threshold"]
                        .as_f32()
                        .unwrap_or(0.8),
                    enable_diversity_selection: parsed["enhancements"]["adaptive_retrieval"]
                        ["enable_diversity_selection"]
                        .as_bool()
                        .unwrap_or(true),
                    enable_confidence_weighting: parsed["enhancements"]["adaptive_retrieval"]
                        ["enable_confidence_weighting"]
                        .as_bool()
                        .unwrap_or(true),
                },
                performance_benchmarking: enhancements::BenchmarkingConfig {
                    enabled: parsed["enhancements"]["performance_benchmarking"]["enabled"]
                        .as_bool()
                        .unwrap_or(false),
                    auto_recommendations: parsed["enhancements"]["performance_benchmarking"]
                        ["auto_recommendations"]
                        .as_bool()
                        .unwrap_or(true),
                    comprehensive_testing: parsed["enhancements"]["performance_benchmarking"]
                        ["comprehensive_testing"]
                        .as_bool()
                        .unwrap_or(false),
                    iterations: parsed["enhancements"]["performance_benchmarking"]["iterations"]
                        .as_usize()
                        .unwrap_or(3),
                    include_parallel: parsed["enhancements"]["performance_benchmarking"]
                        ["include_parallel"]
                        .as_bool()
                        .unwrap_or(true),
                    enable_memory_profiling: parsed["enhancements"]["performance_benchmarking"]
                        ["enable_memory_profiling"]
                        .as_bool()
                        .unwrap_or(false),
                },
                enhanced_function_registry: enhancements::FunctionRegistryConfig {
                    enabled: parsed["enhancements"]["enhanced_function_registry"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    categorization: parsed["enhancements"]["enhanced_function_registry"]
                        ["categorization"]
                        .as_bool()
                        .unwrap_or(true),
                    usage_statistics: parsed["enhancements"]["enhanced_function_registry"]
                        ["usage_statistics"]
                        .as_bool()
                        .unwrap_or(true),
                    dynamic_registration: parsed["enhancements"]["enhanced_function_registry"]
                        ["dynamic_registration"]
                        .as_bool()
                        .unwrap_or(true),
                    performance_monitoring: parsed["enhancements"]["enhanced_function_registry"]
                        ["performance_monitoring"]
                        .as_bool()
                        .unwrap_or(false),
                    recommendation_system: parsed["enhancements"]["enhanced_function_registry"]
                        ["recommendation_system"]
                        .as_bool()
                        .unwrap_or(true),
                },
            },
        };

        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn to_file(&self, path: &str) -> Result<()> {
        let mut config_json = json::JsonValue::new_object();

        // Embeddings
        let mut embeddings = json::JsonValue::new_object();
        embeddings["dimension"] = json::JsonValue::from(self.embeddings.dimension);
        if let Some(ref endpoint) = self.embeddings.api_endpoint {
            embeddings["api_endpoint"] = json::JsonValue::from(endpoint.as_str());
        }
        if let Some(ref key) = self.embeddings.api_key {
            embeddings["api_key"] = json::JsonValue::from(key.as_str());
        }
        config_json["embeddings"] = embeddings;

        // Graph
        let mut graph = json::JsonValue::new_object();
        graph["max_connections"] = json::JsonValue::from(self.graph.max_connections);
        graph["similarity_threshold"] = json::JsonValue::from(self.graph.similarity_threshold);
        config_json["graph"] = graph;

        // Text
        let mut text = json::JsonValue::new_object();
        text["chunk_size"] = json::JsonValue::from(self.text.chunk_size);
        text["chunk_overlap"] = json::JsonValue::from(self.text.chunk_overlap);
        let languages_array: Vec<json::JsonValue> = self
            .text
            .languages
            .iter()
            .map(|s| json::JsonValue::from(s.as_str()))
            .collect();
        text["languages"] = json::JsonValue::from(languages_array);
        config_json["text"] = text;

        // Entities
        let mut entities = json::JsonValue::new_object();
        entities["min_confidence"] = json::JsonValue::from(self.entities.min_confidence);
        let entity_types_array: Vec<json::JsonValue> = self
            .entities
            .entity_types
            .iter()
            .map(|s| json::JsonValue::from(s.as_str()))
            .collect();
        entities["entity_types"] = json::JsonValue::from(entity_types_array);
        config_json["entities"] = entities;

        // Retrieval
        let mut retrieval = json::JsonValue::new_object();
        retrieval["top_k"] = json::JsonValue::from(self.retrieval.top_k);
        retrieval["search_algorithm"] =
            json::JsonValue::from(self.retrieval.search_algorithm.as_str());
        config_json["retrieval"] = retrieval;

        // Parallel
        let mut parallel = json::JsonValue::new_object();
        parallel["num_threads"] = json::JsonValue::from(self.parallel.num_threads);
        parallel["enabled"] = json::JsonValue::from(self.parallel.enabled);
        parallel["min_batch_size"] = json::JsonValue::from(self.parallel.min_batch_size);
        parallel["chunk_batch_size"] = json::JsonValue::from(self.parallel.chunk_batch_size);
        parallel["parallel_embeddings"] = json::JsonValue::from(self.parallel.parallel_embeddings);
        parallel["parallel_graph_ops"] = json::JsonValue::from(self.parallel.parallel_graph_ops);
        parallel["parallel_vector_ops"] = json::JsonValue::from(self.parallel.parallel_vector_ops);
        config_json["parallel"] = parallel;

        // Enhancements
        let mut enhancements = json::JsonValue::new_object();
        enhancements["enabled"] = json::JsonValue::from(self.enhancements.enabled);

        let mut query_analysis = json::JsonValue::new_object();
        query_analysis["enabled"] = json::JsonValue::from(self.enhancements.query_analysis.enabled);
        query_analysis["min_confidence"] =
            json::JsonValue::from(self.enhancements.query_analysis.min_confidence);
        query_analysis["enable_strategy_suggestion"] =
            json::JsonValue::from(self.enhancements.query_analysis.enable_strategy_suggestion);
        query_analysis["enable_keyword_analysis"] =
            json::JsonValue::from(self.enhancements.query_analysis.enable_keyword_analysis);
        query_analysis["enable_complexity_scoring"] =
            json::JsonValue::from(self.enhancements.query_analysis.enable_complexity_scoring);
        enhancements["query_analysis"] = query_analysis;

        let mut adaptive_retrieval = json::JsonValue::new_object();
        adaptive_retrieval["enabled"] =
            json::JsonValue::from(self.enhancements.adaptive_retrieval.enabled);
        adaptive_retrieval["use_query_analysis"] =
            json::JsonValue::from(self.enhancements.adaptive_retrieval.use_query_analysis);
        adaptive_retrieval["enable_cross_strategy_fusion"] = json::JsonValue::from(
            self.enhancements
                .adaptive_retrieval
                .enable_cross_strategy_fusion,
        );
        adaptive_retrieval["diversity_threshold"] =
            json::JsonValue::from(self.enhancements.adaptive_retrieval.diversity_threshold);
        adaptive_retrieval["enable_diversity_selection"] = json::JsonValue::from(
            self.enhancements
                .adaptive_retrieval
                .enable_diversity_selection,
        );
        adaptive_retrieval["enable_confidence_weighting"] = json::JsonValue::from(
            self.enhancements
                .adaptive_retrieval
                .enable_confidence_weighting,
        );
        enhancements["adaptive_retrieval"] = adaptive_retrieval;

        let mut performance_benchmarking = json::JsonValue::new_object();
        performance_benchmarking["enabled"] =
            json::JsonValue::from(self.enhancements.performance_benchmarking.enabled);
        performance_benchmarking["auto_recommendations"] = json::JsonValue::from(
            self.enhancements
                .performance_benchmarking
                .auto_recommendations,
        );
        performance_benchmarking["comprehensive_testing"] = json::JsonValue::from(
            self.enhancements
                .performance_benchmarking
                .comprehensive_testing,
        );
        performance_benchmarking["iterations"] =
            json::JsonValue::from(self.enhancements.performance_benchmarking.iterations);
        performance_benchmarking["include_parallel"] =
            json::JsonValue::from(self.enhancements.performance_benchmarking.include_parallel);
        performance_benchmarking["enable_memory_profiling"] = json::JsonValue::from(
            self.enhancements
                .performance_benchmarking
                .enable_memory_profiling,
        );
        enhancements["performance_benchmarking"] = performance_benchmarking;

        let mut enhanced_function_registry = json::JsonValue::new_object();
        enhanced_function_registry["enabled"] =
            json::JsonValue::from(self.enhancements.enhanced_function_registry.enabled);
        enhanced_function_registry["categorization"] =
            json::JsonValue::from(self.enhancements.enhanced_function_registry.categorization);
        enhanced_function_registry["usage_statistics"] = json::JsonValue::from(
            self.enhancements
                .enhanced_function_registry
                .usage_statistics,
        );
        enhanced_function_registry["dynamic_registration"] = json::JsonValue::from(
            self.enhancements
                .enhanced_function_registry
                .dynamic_registration,
        );
        enhanced_function_registry["performance_monitoring"] = json::JsonValue::from(
            self.enhancements
                .enhanced_function_registry
                .performance_monitoring,
        );
        enhanced_function_registry["recommendation_system"] = json::JsonValue::from(
            self.enhancements
                .enhanced_function_registry
                .recommendation_system,
        );
        enhancements["enhanced_function_registry"] = enhanced_function_registry;

        config_json["enhancements"] = enhancements;

        let content = json::stringify_pretty(config_json, 2);
        fs::write(path, content)?;
        Ok(())
    }
}
