// Allow dead code for configuration structures used for TOML parsing
#![allow(dead_code)]

use crate::config::Config;
use crate::core::{GraphRAGError, Result};
use std::fs;
use std::path::Path;

use toml;

use serde_json;

/// Configuration file format
#[derive(Debug, Clone)]
pub enum ConfigFormat {
    /// TOML configuration format
    Toml,
    /// JSON configuration format
    Json,
    /// JSON5 configuration format
    Json5,
    /// YAML configuration format
    Yaml,
}

impl ConfigFormat {
    /// Determine configuration format from file extension
    pub fn from_extension(path: &str) -> Self {
        let path = Path::new(path);
        match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => ConfigFormat::Toml,
            Some("json") => ConfigFormat::Json,
            Some("json5") => ConfigFormat::Json5,
            Some("yaml" | "yml") => ConfigFormat::Yaml,
            _ => ConfigFormat::Toml, // Default
        }
    }
}

/// Load configuration from file
pub fn load_config(path: &str) -> Result<Config> {
    let format = ConfigFormat::from_extension(path);

    if !Path::new(path).exists() {
        return Err(GraphRAGError::Config {
            message: format!("Configuration file not found: {path}"),
        });
    }

    let content = fs::read_to_string(path)?;

    load_config_from_str(&content, format)
}

/// Parse and validate configuration content through the same dispatcher used
/// for files. This is used by in-memory ingress points such as the server API.
pub fn load_config_from_str(content: &str, format: ConfigFormat) -> Result<Config> {
    match format {
        ConfigFormat::Toml => load_toml_config(content),
        ConfigFormat::Json => load_json_config(content),
        ConfigFormat::Json5 => load_json5_config(content),
        ConfigFormat::Yaml => load_yaml_config(content),
    }
}

fn load_toml_config(content: &str) -> Result<Config> {
    let value: toml::Value = toml::from_str(content).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to parse TOML config: {e}"),
    })?;

    let json_value = serde_json::to_value(&value).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to convert TOML config: {e}"),
    })?;
    match classify_config(&json_value)? {
        ConfigKind::Runtime => return checked_runtime_config(Config::from_toml_str(content)?),
        ConfigKind::Set => {
            let set_config: crate::config::SetConfig =
                value.try_into().map_err(|e| GraphRAGError::Config {
                    message: format!("Failed to parse SetConfig: {e}"),
                })?;
            return checked_set_config(set_config);
        },
        ConfigKind::Legacy => {},
    }

    let raw_config: RawConfig = value.try_into().map_err(|e| GraphRAGError::Config {
        message: format!("Failed to parse legacy TOML config: {e}"),
    })?;

    convert_raw_config(raw_config)
}

fn load_json_config(content: &str) -> Result<Config> {
    let value: serde_json::Value =
        serde_json::from_str(content).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse JSON config: {e}"),
        })?;

    match classify_config(&value)? {
        ConfigKind::Runtime => checked_runtime_config(Config::from_json_str(content)?),
        ConfigKind::Set => {
            let set_config: crate::config::SetConfig =
                serde_json::from_value(value).map_err(|e| GraphRAGError::Config {
                    message: format!("Failed to parse SetConfig: {e}"),
                })?;
            checked_set_config(set_config)
        },
        ConfigKind::Legacy => {
            let raw_config: RawConfig =
                serde_json::from_value(value).map_err(|e| GraphRAGError::Config {
                    message: format!("Failed to parse legacy JSON config: {e}"),
                })?;
            convert_raw_config(raw_config)
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigKind {
    Runtime,
    Set,
    Legacy,
}

fn classify_config(value: &serde_json::Value) -> Result<ConfigKind> {
    let object = value.as_object().ok_or_else(|| GraphRAGError::Config {
        message: "Configuration root must be an object".to_string(),
    })?;
    // Runtime and legacy schemas ignore unknown sections; fail loudly instead
    // of silently dropping a pre-llama.cpp `ollama` block.
    if object.contains_key("ollama") {
        return Err(GraphRAGError::Config {
            message: "the `ollama` section was removed: rename it to `llm` and use `base_url` \
                      (e.g. \"http://127.0.0.1:8080/v1\") instead of `host`/`port`"
                .to_string(),
        });
    }
    // These keys exist only on the runtime Config surface.  Several other
    // fields (output_dir, llm, storage, experimental, and entity_extraction)
    // are intentionally shared with SetConfig and must not make a valid
    // template look like an ambiguous mixture.
    let runtime_keys = [
        "max_entities_per_chunk",
        "similarity_threshold",
        "approach",
        "embeddings",
        "graph",
        "text",
        "entities",
        "retrieval",
        "parallel",
        "enhancements",
        "summarization",
        "zero_cost_approach",
        "advanced_features",
        "suppress_progress_bars",
        "chunk_size",
        "chunk_overlap",
        "top_k_results",
    ];
    let set_keys = [
        "mode",
        "semantic",
        "algorithmic",
        "hybrid",
        "general",
        "pipeline",
        "models",
        "performance",
    ];
    let legacy_keys = [
        "system",
        "features",
        "text_processing",
        "graph_construction",
        "vector_processing",
        "query_processing",
        "adaptive_retrieval",
        "ranking_policies",
        "reranking",
        "generation",
        "async_processing",
        "function_calling",
        "monitoring",
        "parallel_processing",
        "logging",
    ];
    let has_runtime = runtime_keys.iter().any(|key| object.contains_key(*key));
    let has_set = set_keys.iter().any(|key| object.contains_key(*key));
    let has_legacy = legacy_keys.iter().any(|key| object.contains_key(*key));
    // A number of older SetConfig templates put extension sections such as
    // `embeddings`, `retrieval`, or `summarization` at the root. Those names
    // overlap with Config, but their shape is different. Only reject a mixed
    // document when the overlapping section actually has canonical runtime
    // fields; otherwise the explicit SetConfig discriminator wins.
    let canonical_runtime_conflict = ["text", "graph", "entities", "parallel"]
        .iter()
        .any(|key| object.contains_key(*key))
        || object
            .get("retrieval")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|section| {
                section.contains_key("top_k") || section.contains_key("search_algorithm")
            })
        || object.contains_key("chunk_size")
        || object.contains_key("chunk_overlap")
        || object.contains_key("top_k_results");
    match (has_runtime, has_set, has_legacy) {
        (true, true, _) if canonical_runtime_conflict => Err(GraphRAGError::Config {
            message: "Configuration mixes runtime and SetConfig sections".to_string(),
        }),
        (true, true, _) => Ok(ConfigKind::Set),
        (true, false, true) => Err(GraphRAGError::Config {
            message: "Configuration mixes runtime and legacy sections".to_string(),
        }),
        (true, false, false) => Ok(ConfigKind::Runtime),
        (false, true, _) => Ok(ConfigKind::Set),
        (false, false, true) => Ok(ConfigKind::Legacy),
        // Shared fields (for example output_dir or llm) have a natural
        // runtime meaning and are therefore canonical runtime documents when
        // no SetConfig discriminator is present. SetConfig documents must
        // carry an explicit discriminator such as `mode` or `general`.
        (false, false, false) => Ok(ConfigKind::Runtime),
    }
}

fn checked_set_config(set_config: crate::config::SetConfig) -> Result<Config> {
    let validation = crate::config::Validatable::validate(&set_config);
    if !validation.is_valid {
        return Err(GraphRAGError::Config {
            message: validation.errors.join("; "),
        });
    }
    set_config.to_graphrag_config()
}

fn checked_runtime_config(config: Config) -> Result<Config> {
    let validation = crate::config::Validatable::validate(&config);
    if validation.is_valid {
        Ok(config)
    } else {
        Err(GraphRAGError::Config {
            message: validation.errors.join("; "),
        })
    }
}

#[cfg(feature = "json5-support")]
fn load_json5_config(content: &str) -> Result<Config> {
    let value: serde_json::Value = json5::from_str(content).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to parse JSON5 config: {e}"),
    })?;
    match classify_config(&value)? {
        ConfigKind::Runtime => checked_runtime_config(Config::from_json5_str(content)?),
        ConfigKind::Set => {
            let set_config: crate::config::SetConfig =
                json5::from_str(content).map_err(|e| GraphRAGError::Config {
                    message: format!("Failed to parse SetConfig: {e}"),
                })?;
            checked_set_config(set_config)
        },
        ConfigKind::Legacy => Err(GraphRAGError::Config {
            message: "Legacy JSON5 configuration is not supported".to_string(),
        }),
    }
}

#[cfg(not(feature = "json5-support"))]
fn load_json5_config(_content: &str) -> Result<Config> {
    Err(GraphRAGError::Config {
        message: "JSON5 support not enabled".to_string(),
    })
}

#[cfg(feature = "yaml-support")]
fn load_yaml_config(content: &str) -> Result<Config> {
    let value: serde_json::Value =
        serde_yaml::from_str(content).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse YAML config: {e}"),
        })?;

    match classify_config(&value)? {
        ConfigKind::Runtime => {
            checked_runtime_config(serde_yaml::from_str(content).map_err(|e| {
                GraphRAGError::Config {
                    message: format!("Failed to parse runtime YAML config: {e}"),
                }
            })?)
        },
        ConfigKind::Set => {
            let set_config: crate::config::SetConfig =
                serde_json::from_value(value).map_err(|e| GraphRAGError::Config {
                    message: format!("Failed to parse SetConfig: {e}"),
                })?;
            checked_set_config(set_config)
        },
        ConfigKind::Legacy => {
            let raw_config: RawConfig =
                serde_json::from_value(value).map_err(|e| GraphRAGError::Config {
                    message: format!("Failed to parse legacy YAML config: {e}"),
                })?;
            convert_raw_config(raw_config)
        },
    }
}

#[cfg(not(feature = "yaml-support"))]
fn load_yaml_config(_content: &str) -> Result<Config> {
    Err(GraphRAGError::Config {
        message: "YAML support not enabled. Enable 'yaml-support' feature.".to_string(),
    })
}

/// Raw configuration structure that matches the TOML file
#[derive(Debug, serde::Deserialize, Default)]
#[allow(dead_code)]
struct RawConfig {
    #[serde(default)]
    system: SystemConfig,
    #[serde(default)]
    features: FeaturesConfig,
    #[serde(default)]
    text_processing: RawTextProcessingConfig,
    #[serde(default)]
    entity_extraction: RawEntityExtractionConfig,
    #[serde(default)]
    graph_construction: RawGraphConstructionConfig,
    #[serde(default)]
    vector_processing: RawVectorProcessingConfig,
    #[serde(default)]
    query_processing: RawQueryProcessingConfig,
    #[serde(default)]
    adaptive_retrieval: RawAdaptiveRetrievalConfig,
    #[serde(default)]
    ranking_policies: RawRankingPoliciesConfig,
    #[serde(default)]
    reranking: RawRerankingConfig,
    #[serde(default)]
    generation: RawGenerationConfig,
    #[serde(default)]
    llm: RawLlmConfig,
    #[serde(default)]
    async_processing: RawAsyncProcessingConfig,
    #[serde(default)]
    function_calling: RawFunctionCallingConfig,
    #[serde(default)]
    monitoring: RawMonitoringConfig,
    #[serde(default)]
    storage: RawStorageConfig,
    #[serde(default)]
    parallel_processing: RawParallelProcessingConfig,
    #[serde(default)]
    logging: RawLoggingConfig,
    #[serde(default)]
    experimental: RawExperimentalConfig,
}

#[derive(Debug, serde::Deserialize, Default)]
#[allow(dead_code)]
struct SystemConfig {
    log_level: Option<String>,
    max_memory_mb: Option<u64>,
    temp_dir: Option<String>,
    output_dir: Option<String>,
}

#[derive(Debug, serde::Deserialize, Default)]
#[allow(dead_code)]
struct FeaturesConfig {
    text_processing: Option<bool>,
    entity_extraction: Option<bool>,
    graph_construction: Option<bool>,
    vector_processing: Option<bool>,
    async_processing: Option<bool>,
    function_calling: Option<bool>,
    monitoring: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
#[allow(dead_code)]
struct RawTextProcessingConfig {
    enabled: Option<bool>,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    min_chunk_size: Option<usize>,
    max_chunk_size: Option<usize>,
    normalize_whitespace: Option<bool>,
    remove_artifacts: Option<bool>,
    extract_keywords: Option<bool>,
    keyword_min_score: Option<f64>,
    #[serde(default)]
    enrichment: Option<RawEnrichmentConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
#[allow(dead_code)]
struct RawEnrichmentConfig {
    enabled: Option<bool>,
    auto_detect_format: Option<bool>,
    parser_type: Option<String>,
    extract_keywords: Option<bool>,
    max_keywords_per_chunk: Option<usize>,
    use_tfidf: Option<bool>,
    generate_summaries: Option<bool>,
    min_chunk_length_for_summary: Option<usize>,
    max_summary_length: Option<usize>,
    extract_chapter: Option<bool>,
    extract_section: Option<bool>,
    extract_position: Option<bool>,
    calculate_confidence: Option<bool>,
    detect_headings: Option<bool>,
    detect_numbering: Option<bool>,
    detect_underlines: Option<bool>,
    detect_all_caps: Option<bool>,
    detect_roman_numerals: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
#[allow(dead_code)]
struct RawEntityExtractionConfig {
    enabled: Option<bool>,
    min_confidence: Option<f32>,
    use_gleaning: Option<bool>,
    max_gleaning_rounds: Option<usize>,
    gleaning_improvement_threshold: Option<f64>,
    semantic_merging: Option<bool>,
    merge_similarity_threshold: Option<f64>,
    automatic_linking: Option<bool>,
    linking_confidence_threshold: Option<f64>,
    gleaning: Option<RawGleaningConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawGleaningConfig {
    focus_areas: Option<Vec<String>>,
    context_window: Option<usize>,
    llm_temperature: Option<f64>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawGraphConstructionConfig {
    enabled: Option<bool>,
    incremental_updates: Option<bool>,
    use_pagerank: Option<bool>,
    pagerank_damping: Option<f64>,
    pagerank_iterations: Option<usize>,
    pagerank_convergence: Option<f64>,
    extract_relationships: Option<bool>,
    relationship_confidence_threshold: Option<f64>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawVectorProcessingConfig {
    enabled: Option<bool>,
    embedding_model: Option<String>,
    embedding_dimensions: Option<usize>,
    use_hnsw_index: Option<bool>,
    hnsw_ef_construction: Option<usize>,
    hnsw_m: Option<usize>,
    similarity_threshold: Option<f64>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawQueryProcessingConfig {
    enabled: Option<bool>,
    use_advanced_pipeline: Option<bool>,
    use_intent_classification: Option<bool>,
    use_concept_extraction: Option<bool>,
    use_temporal_parsing: Option<bool>,
    confidence_threshold: Option<f64>,
    intent_classification: Option<RawIntentClassificationConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawIntentClassificationConfig {
    factual_patterns: Option<Vec<String>>,
    relational_patterns: Option<Vec<String>>,
    temporal_patterns: Option<Vec<String>>,
    causal_patterns: Option<Vec<String>>,
    comparative_patterns: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawAdaptiveRetrievalConfig {
    enabled: Option<bool>,
    default_strategies: Option<Vec<String>>,
    strategy_weights: Option<std::collections::HashMap<String, f64>>,
    dynamic_weighting: Option<bool>,
    diversity_factor: Option<f64>,
    max_results_per_strategy: Option<usize>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawRankingPoliciesConfig {
    enabled: Option<bool>,
    use_elbow_detection: Option<bool>,
    use_top_k_diversity: Option<bool>,
    use_threshold_filtering: Option<bool>,
    use_intent_aware_ranking: Option<bool>,
    use_confidence_filtering: Option<bool>,
    elbow_detection: Option<RawElbowDetectionConfig>,
    top_k: Option<RawTopKConfig>,
    threshold: Option<RawThresholdConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawElbowDetectionConfig {
    min_results: Option<usize>,
    max_results: Option<usize>,
    smoothing_factor: Option<f64>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawTopKConfig {
    k: Option<usize>,
    diversity_threshold: Option<f64>,
    entity_type_balance: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawThresholdConfig {
    min_score: Option<f64>,
    confidence_weight: Option<f64>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawRerankingConfig {
    enabled: Option<bool>,
    use_confidence_filtering: Option<bool>,
    use_cross_encoder: Option<bool>,
    use_diversity_selection: Option<bool>,
    final_result_limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawGenerationConfig {
    enabled: Option<bool>,
    use_context_assembly: Option<bool>,
    max_context_length: Option<usize>,
    use_prompt_templates: Option<bool>,
    include_citations: Option<bool>,
    include_confidence_scores: Option<bool>,
    templates: Option<RawTemplatesConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawTemplatesConfig {
    factual: Option<String>,
    relational: Option<String>,
    temporal: Option<String>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawLlmConfig {
    enabled: Option<bool>,
    base_url: Option<String>,
    model_name: Option<String>,
    embedding_model: Option<String>,
    timeout_seconds: Option<u64>,
    max_retries: Option<u32>,
    generation: Option<RawLlmGenerationConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawLlmGenerationConfig {
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawAsyncProcessingConfig {
    enabled: Option<bool>,
    max_concurrent_llm_calls: Option<usize>,
    max_concurrent_embeddings: Option<usize>,
    max_concurrent_documents: Option<usize>,
    llm_rate_limit_per_second: Option<f64>,
    embedding_rate_limit_per_second: Option<f64>,
    batching: Option<RawBatchingConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawBatchingConfig {
    batch_size: Option<usize>,
    batch_timeout_seconds: Option<u64>,
    max_batch_memory_mb: Option<usize>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawFunctionCallingConfig {
    enabled: Option<bool>,
    max_function_calls: Option<usize>,
    timeout_per_call_seconds: Option<u64>,
    allow_nested_calls: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawMonitoringConfig {
    enabled: Option<bool>,
    collect_performance_metrics: Option<bool>,
    collect_usage_statistics: Option<bool>,
    health_check_interval_seconds: Option<u64>,
    log_slow_operations: Option<bool>,
    slow_operation_threshold_ms: Option<u64>,
    benchmarking: Option<RawBenchmarkingConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawBenchmarkingConfig {
    enabled: Option<bool>,
    run_periodic_benchmarks: Option<bool>,
    benchmark_interval_hours: Option<u64>,
    auto_recommendations: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawStorageConfig {
    r#type: Option<String>,
    workspace_isolation: Option<bool>,
    max_workspaces: Option<usize>,
    backup_enabled: Option<bool>,
    backup_interval_hours: Option<u64>,
    persistent: Option<RawPersistentConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawPersistentConfig {
    database_path: Option<String>,
    enable_wal: Option<bool>,
    cache_size_mb: Option<usize>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawParallelProcessingConfig {
    enabled: Option<bool>,
    max_threads: Option<usize>,
    thread_pool_size: Option<usize>,
    load_balancing: Option<bool>,
    work_stealing: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawLoggingConfig {
    level: Option<String>,
    format: Option<String>,
    include_timestamps: Option<bool>,
    include_module_path: Option<bool>,
    log_to_file: Option<bool>,
    log_file: Option<String>,
    max_log_file_mb: Option<usize>,
    rotate_logs: Option<bool>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawExperimentalConfig {
    neural_reranking: Option<bool>,
    federated_learning: Option<bool>,
    real_time_updates: Option<bool>,
    distributed_processing: Option<bool>,
}

/// Convert raw configuration to the main Config struct
fn convert_raw_config(raw: RawConfig) -> Result<Config> {
    let mut config = Config::default();

    // Apply feature toggles
    if let Some(_text_enabled) = raw.features.text_processing {
        // Apply to appropriate config sections
    }

    if let Some(output_dir) = raw.system.output_dir {
        config.output_dir = output_dir;
    }

    // Apply text processing configuration
    if let Some(chunk_size) = raw.text_processing.chunk_size {
        config.text.chunk_size = chunk_size;
    }
    if let Some(chunk_overlap) = raw.text_processing.chunk_overlap {
        config.text.chunk_overlap = chunk_overlap;
    }

    // Apply entity extraction configuration
    if let Some(min_confidence) = raw.entity_extraction.min_confidence {
        config.entities.min_confidence = min_confidence;
    }

    // Apply graph construction configuration
    if let Some(extract_rels) = raw.graph_construction.extract_relationships {
        config.graph.extract_relationships = extract_rels;
    }
    if let Some(threshold) = raw.graph_construction.relationship_confidence_threshold {
        config.graph.relationship_confidence_threshold = threshold as f32;
    }

    // Apply parallel processing configuration
    if let Some(enabled) = raw.parallel_processing.enabled {
        config.parallel.enabled = enabled;
    }
    if let Some(max_threads) = raw.parallel_processing.max_threads {
        config.parallel.num_threads = if max_threads == 0 {
            #[cfg(feature = "parallel-processing")]
            {
                num_cpus::get()
            }
            #[cfg(not(feature = "parallel-processing"))]
            {
                1
            }
        } else {
            max_threads
        };
    }

    checked_runtime_config(config)
}

/// Save configuration to file
pub fn save_config(config: &Config, path: &str) -> Result<()> {
    let format = ConfigFormat::from_extension(path);

    match format {
        ConfigFormat::Toml => save_toml_config(config, path),
        ConfigFormat::Json => save_json_config(config, path),
        ConfigFormat::Json5 => save_json_config(config, path),
        ConfigFormat::Yaml => save_yaml_config(config, path),
    }
}

fn save_toml_config(config: &Config, path: &str) -> Result<()> {
    let content = toml::to_string_pretty(config).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to serialize TOML config: {e}"),
    })?;
    fs::write(path, content)?;
    Ok(())
}

fn save_json_config(config: &Config, path: &str) -> Result<()> {
    let content = serde_json::to_string_pretty(config).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to serialize JSON config: {e}"),
    })?;
    fs::write(path, content)?;
    Ok(())
}

#[cfg(feature = "yaml-support")]
fn save_yaml_config(config: &Config, path: &str) -> Result<()> {
    let content = serde_yaml::to_string(config).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to serialize YAML config: {e}"),
    })?;
    fs::write(path, content)?;
    Ok(())
}

#[cfg(not(feature = "yaml-support"))]
fn save_yaml_config(_config: &Config, _path: &str) -> Result<()> {
    Err(GraphRAGError::Config {
        message: "YAML support not enabled.".to_string(),
    })
}

// Removed convert_config_to_raw function as we now use static templates

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_format_detection() {
        assert!(matches!(
            ConfigFormat::from_extension("config.toml"),
            ConfigFormat::Toml
        ));
        assert!(matches!(
            ConfigFormat::from_extension("config.json"),
            ConfigFormat::Json
        ));
        assert!(matches!(
            ConfigFormat::from_extension("config.yaml"),
            ConfigFormat::Yaml
        ));
        assert!(matches!(
            ConfigFormat::from_extension("config.yml"),
            ConfigFormat::Yaml
        ));
        assert!(matches!(
            ConfigFormat::from_extension("config"),
            ConfigFormat::Toml
        ));
    }

    #[test]
    fn legacy_ollama_section_is_rejected_for_every_schema() {
        let runtime = "chunk_size = 500\n[embeddings]\ndimension = 384\n[ollama]\nenabled = true\n";
        let legacy = "[system]\n[ollama]\nenabled = true\n";
        for content in [runtime, legacy] {
            let err = load_config_from_str(content, ConfigFormat::Toml).unwrap_err();
            assert!(
                err.to_string().contains("`ollama` section was removed"),
                "{err}"
            );
        }
        let err = load_config_from_str(r#"{"system": {}, "ollama": {}}"#, ConfigFormat::Json)
            .unwrap_err();
        assert!(
            err.to_string().contains("`ollama` section was removed"),
            "{err}"
        );
    }

    #[cfg(feature = "yaml-support")]
    #[test]
    fn yaml_config_round_trips_runtime_config() {
        let mut config = Config::default();
        config.output_dir = "./yaml-round-trip-output".to_string();
        config.text.chunk_size = 731;

        let path = tempfile::NamedTempFile::new().unwrap();
        let path = path.path().with_extension("yaml");
        save_config(&config, path.to_str().unwrap()).unwrap();

        let loaded = load_config(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.output_dir, config.output_dir);
        assert_eq!(loaded.text.chunk_size, config.text.chunk_size);

        fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "yaml-support")]
    #[test]
    fn yaml_legacy_configuration_is_still_supported() {
        let content =
            "system:\n  output_dir: ./legacy-output\ntext_processing:\n  chunk_size: 640\n";
        let config = load_config_from_str(content, ConfigFormat::Yaml).unwrap();

        assert_eq!(config.output_dir, "./legacy-output");
        assert_eq!(config.text.chunk_size, 640);
    }
}
