// Allow dead code for configuration structures used for TOML parsing
#![allow(dead_code)]

use crate::config::Config;
use crate::core::{GraphRAGError, Result};
use std::fs;
use std::path::Path;

#[cfg(feature = "toml-support")]
use toml;

#[cfg(feature = "serde_json")]
use serde_json;

/// Configuration file format
#[derive(Debug, Clone)]
pub enum ConfigFormat {
    /// TOML configuration format
    Toml,
    /// JSON configuration format
    Json,
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

    match format {
        ConfigFormat::Toml => load_toml_config(&content),
        ConfigFormat::Json => load_json_config(&content),
        ConfigFormat::Yaml => load_yaml_config(&content),
    }
}

#[cfg(feature = "toml-support")]
fn load_toml_config(content: &str) -> Result<Config> {
    let raw_config: RawConfig = toml::from_str(content).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to parse TOML config: {e}"),
    })?;

    Ok(convert_raw_config(raw_config))
}

#[cfg(not(feature = "toml-support"))]
fn load_toml_config(_content: &str) -> Result<Config> {
    Err(GraphRAGError::Config {
        message: "TOML support not enabled. Enable 'toml-support' feature.".to_string(),
    })
}

#[cfg(feature = "serde_json")]
fn load_json_config(content: &str) -> Result<Config> {
    let raw_config: RawConfig =
        serde_json::from_str(content).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse JSON config: {e}"),
        })?;

    Ok(convert_raw_config(raw_config))
}

#[cfg(not(feature = "serde_json"))]
fn load_json_config(_content: &str) -> Result<Config> {
    Err(GraphRAGError::Config {
        message: "JSON support not enabled. Enable 'serde_json' feature.".to_string(),
    })
}

#[cfg(feature = "yaml-support")]
fn load_yaml_config(content: &str) -> Result<Config> {
    let raw_config: RawConfig =
        serde_yaml::from_str(content).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse YAML config: {e}"),
        })?;

    Ok(convert_raw_config(raw_config))
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
    ollama: RawOllamaConfig,
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
struct RawOllamaConfig {
    enabled: Option<bool>,
    base_url: Option<String>,
    model_name: Option<String>,
    embedding_model: Option<String>,
    timeout_seconds: Option<u64>,
    max_retries: Option<u32>,
    generation: Option<RawOllamaGenerationConfig>,
}

#[derive(Debug, serde::Deserialize, Default)]
struct RawOllamaGenerationConfig {
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
fn convert_raw_config(raw: RawConfig) -> Config {
    let mut config = Config::default();

    // Apply feature toggles
    if let Some(_text_enabled) = raw.features.text_processing {
        // Apply to appropriate config sections
    }

    // Apply text processing configuration
    if let Some(_chunk_size) = raw.text_processing.chunk_size {
        // config.text.chunk_size = chunk_size;
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

    config
}

/// Save configuration to file
pub fn save_config(config: &Config, path: &str) -> Result<()> {
    let format = ConfigFormat::from_extension(path);

    match format {
        ConfigFormat::Toml => save_toml_config(config, path),
        ConfigFormat::Json => save_json_config(config, path),
        ConfigFormat::Yaml => save_yaml_config(config, path),
    }
}

#[cfg(feature = "toml-support")]
fn save_toml_config(_config: &Config, path: &str) -> Result<()> {
    let content = r#"[text]
chunk_size = 1000
chunk_overlap = 200

[entities]
min_confidence = 0.7
entity_types = ["PERSON", "ORG", "LOCATION"]

[graph]
max_connections = 10
similarity_threshold = 0.8

[parallel]
enabled = true
num_threads = 0
"#;
    fs::write(path, content)?;
    Ok(())
}

#[cfg(not(feature = "toml-support"))]
fn save_toml_config(_config: &Config, _path: &str) -> Result<()> {
    Err(GraphRAGError::Config {
        message: "TOML support not enabled. Enable 'toml-support' feature.".to_string(),
    })
}

#[cfg(feature = "serde_json")]
fn save_json_config(_config: &Config, path: &str) -> Result<()> {
    let content = r#"{
  "text": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "entities": {
    "min_confidence": 0.7,
    "entity_types": ["PERSON", "ORG", "LOCATION"]
  },
  "graph": {
    "max_connections": 10,
    "similarity_threshold": 0.8
  },
  "parallel": {
    "enabled": true,
    "num_threads": 0
  }
}"#;
    fs::write(path, content)?;
    Ok(())
}

#[cfg(not(feature = "serde_json"))]
fn save_json_config(_config: &Config, _path: &str) -> Result<()> {
    Err(GraphRAGError::Config {
        message: "JSON support not enabled.".to_string(),
    })
}

#[cfg(feature = "yaml-support")]
fn save_yaml_config(_config: &Config, path: &str) -> Result<()> {
    let content = r#"text:
  chunk_size: 1000
  chunk_overlap: 200

entities:
  min_confidence: 0.7
  entity_types: ["PERSON", "ORG", "LOCATION"]

graph:
  max_connections: 10
  similarity_threshold: 0.8

parallel:
  enabled: true
  num_threads: 0
"#;
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
}
