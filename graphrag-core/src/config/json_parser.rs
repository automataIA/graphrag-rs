//! Hand-rolled JSON loader/writer for `Config`.
//!
//! Extracted from `config/mod.rs` (Phase 4 file split). Provides
//! `Config::from_file` (read) and `Config::to_file` (write) via the lightweight
//! `json` crate. Distinct from `config::json5_loader` (serde-based typed JSON5)
//! and `config::loader` (multi-format dispatcher).

#![allow(clippy::needless_borrow, clippy::manual_clamp)]

use super::*;
use crate::Result;
use std::fs;

impl Config {
    /// Load configuration from a JSON file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let parsed = json::parse(&content)?;

        let config = Config {
            output_dir: parsed["output_dir"]
                .as_str()
                .unwrap_or("./output")
                .to_string(),
            suppress_progress_bars: parsed["suppress_progress_bars"].as_bool().unwrap_or(false),
            chunk_size: parsed["chunk_size"]
                .as_usize()
                .unwrap_or(default_chunk_size()),
            chunk_overlap: parsed["chunk_overlap"]
                .as_usize()
                .unwrap_or(default_chunk_overlap()),
            max_entities_per_chunk: parsed["max_entities_per_chunk"].as_usize(),
            top_k_results: parsed["top_k_results"].as_usize(),
            similarity_threshold: parsed["similarity_threshold"].as_f32(),
            approach: parsed["approach"]
                .as_str()
                .unwrap_or(&default_approach())
                .to_string(),
            embeddings: EmbeddingConfig {
                dimension: parsed["embeddings"]["dimension"]
                    .as_usize()
                    .unwrap_or(default_embedding_dim()),
                backend: parsed["embeddings"]["backend"]
                    .as_str()
                    .unwrap_or(&default_embedding_backend())
                    .to_string(),
                model: parsed["embeddings"]["model"]
                    .as_str()
                    .map(|s| s.to_string()),
                fallback_to_hash: parsed["embeddings"]["fallback_to_hash"]
                    .as_bool()
                    .unwrap_or(true),
                api_endpoint: parsed["embeddings"]["api_endpoint"]
                    .as_str()
                    .map(|s| s.to_string()),
                api_key: parsed["embeddings"]["api_key"]
                    .as_str()
                    .map(|s| s.to_string()),
                cache_dir: parsed["embeddings"]["cache_dir"]
                    .as_str()
                    .map(|s| s.to_string()),
                batch_size: parsed["embeddings"]["batch_size"]
                    .as_usize()
                    .unwrap_or(default_batch_size()),
            },
            graph: GraphConfig {
                max_connections: parsed["graph"]["max_connections"]
                    .as_usize()
                    .unwrap_or(default_max_connections()),
                similarity_threshold: parsed["graph"]["similarity_threshold"]
                    .as_f32()
                    .unwrap_or(default_similarity_threshold()),
                extract_relationships: parsed["graph"]["extract_relationships"]
                    .as_bool()
                    .unwrap_or(default_true()),
                relationship_confidence_threshold: parsed["graph"]
                    ["relationship_confidence_threshold"]
                    .as_f32()
                    .unwrap_or(default_relationship_confidence()),
                traversal: TraversalConfigParams {
                    max_depth: parsed["graph"]["traversal"]["max_depth"]
                        .as_usize()
                        .unwrap_or(default_max_traversal_depth()),
                    max_paths: parsed["graph"]["traversal"]["max_paths"]
                        .as_usize()
                        .unwrap_or(default_max_paths()),
                    use_edge_weights: parsed["graph"]["traversal"]["use_edge_weights"]
                        .as_bool()
                        .unwrap_or(default_true()),
                    min_relationship_strength: parsed["graph"]["traversal"]
                        ["min_relationship_strength"]
                        .as_f32()
                        .unwrap_or(default_min_relationship_strength()),
                },
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
                use_gleaning: parsed["entities"]["use_gleaning"]
                    .as_bool()
                    .unwrap_or(false),
                max_gleaning_rounds: parsed["entities"]["max_gleaning_rounds"]
                    .as_usize()
                    .unwrap_or(default_max_gleaning_rounds()),
                enable_triple_reflection: parsed["entities"]["enable_triple_reflection"]
                    .as_bool()
                    .unwrap_or(false),
                validation_min_confidence: parsed["entities"]["validation_min_confidence"]
                    .as_f32()
                    .unwrap_or(default_validation_confidence()),
                use_atomic_facts: parsed["entities"]["use_atomic_facts"]
                    .as_bool()
                    .unwrap_or(false),
                max_fact_tokens: parsed["entities"]["max_fact_tokens"]
                    .as_usize()
                    .unwrap_or(default_max_fact_tokens()),
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
                enable_caching: parsed["ollama"]["enable_caching"].as_bool().unwrap_or(true),
                keep_alive: parsed["ollama"]["keep_alive"]
                    .as_str()
                    .map(|s| s.to_string()),
                num_ctx: parsed["ollama"]["num_ctx"].as_u32(),
            },
            gliner: GlinerConfig {
                enabled: parsed["gliner"]["enabled"].as_bool().unwrap_or(false),
                model_path: parsed["gliner"]["model_path"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                tokenizer_path: parsed["gliner"]["tokenizer_path"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                mode: parsed["gliner"]["mode"]
                    .as_str()
                    .unwrap_or("span")
                    .to_string(),
                entity_labels: if parsed["gliner"]["entity_labels"].is_array() {
                    parsed["gliner"]["entity_labels"]
                        .members()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                } else {
                    vec!["person".into(), "organization".into(), "location".into()]
                },
                relation_labels: if parsed["gliner"]["relation_labels"].is_array() {
                    parsed["gliner"]["relation_labels"]
                        .members()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                } else {
                    vec!["related to".into(), "part of".into()]
                },
                entity_threshold: parsed["gliner"]["entity_threshold"].as_f32().unwrap_or(0.4),
                relation_threshold: parsed["gliner"]["relation_threshold"]
                    .as_f32()
                    .unwrap_or(0.5),
                use_gpu: parsed["gliner"]["use_gpu"].as_bool().unwrap_or(false),
                max_concurrent_chunks: parsed["gliner"]["max_concurrent_chunks"].as_usize(),
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
                #[cfg(feature = "lightrag")]
                lightrag: enhancements::LightRAGConfig {
                    enabled: parsed["enhancements"]["lightrag"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    max_keywords: parsed["enhancements"]["lightrag"]["max_keywords"]
                        .as_usize()
                        .unwrap_or(20),
                    high_level_weight: parsed["enhancements"]["lightrag"]["high_level_weight"]
                        .as_f32()
                        .unwrap_or(0.6),
                    low_level_weight: parsed["enhancements"]["lightrag"]["low_level_weight"]
                        .as_f32()
                        .unwrap_or(0.4),
                    merge_strategy: parsed["enhancements"]["lightrag"]["merge_strategy"]
                        .as_str()
                        .unwrap_or("weighted")
                        .to_string(),
                    language: parsed["enhancements"]["lightrag"]["language"]
                        .as_str()
                        .unwrap_or("English")
                        .to_string(),
                    enable_cache: parsed["enhancements"]["lightrag"]["enable_cache"]
                        .as_bool()
                        .unwrap_or(true),
                },
                #[cfg(feature = "leiden")]
                leiden: enhancements::LeidenCommunitiesConfig {
                    enabled: parsed["enhancements"]["leiden"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    max_cluster_size: parsed["enhancements"]["leiden"]["max_cluster_size"]
                        .as_usize()
                        .unwrap_or(10),
                    use_lcc: parsed["enhancements"]["leiden"]["use_lcc"]
                        .as_bool()
                        .unwrap_or(true),
                    seed: parsed["enhancements"]["leiden"]["seed"].as_u64(),
                    resolution: parsed["enhancements"]["leiden"]["resolution"]
                        .as_f32()
                        .unwrap_or(1.0),
                    max_levels: parsed["enhancements"]["leiden"]["max_levels"]
                        .as_usize()
                        .unwrap_or(5),
                    min_improvement: parsed["enhancements"]["leiden"]["min_improvement"]
                        .as_f32()
                        .unwrap_or(0.001),
                    enable_hierarchical: parsed["enhancements"]["leiden"]["enable_hierarchical"]
                        .as_bool()
                        .unwrap_or(true),
                    generate_summaries: parsed["enhancements"]["leiden"]["generate_summaries"]
                        .as_bool()
                        .unwrap_or(true),
                    max_summary_length: parsed["enhancements"]["leiden"]["max_summary_length"]
                        .as_usize()
                        .unwrap_or(5),
                    use_extractive_summary: parsed["enhancements"]["leiden"]
                        ["use_extractive_summary"]
                        .as_bool()
                        .unwrap_or(true),
                    adaptive_routing: enhancements::AdaptiveRoutingConfig {
                        enabled: parsed["enhancements"]["leiden"]["adaptive_routing"]["enabled"]
                            .as_bool()
                            .unwrap_or(true),
                        default_level: parsed["enhancements"]["leiden"]["adaptive_routing"]
                            ["default_level"]
                            .as_usize()
                            .unwrap_or(1),
                        keyword_weight: parsed["enhancements"]["leiden"]["adaptive_routing"]
                            ["keyword_weight"]
                            .as_f32()
                            .unwrap_or(0.5),
                        length_weight: parsed["enhancements"]["leiden"]["adaptive_routing"]
                            ["length_weight"]
                            .as_f32()
                            .unwrap_or(0.3),
                        entity_weight: parsed["enhancements"]["leiden"]["adaptive_routing"]
                            ["entity_weight"]
                            .as_f32()
                            .unwrap_or(0.2),
                    },
                },
                #[cfg(feature = "cross-encoder")]
                cross_encoder: enhancements::CrossEncoderConfig {
                    enabled: parsed["enhancements"]["cross_encoder"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    model_name: parsed["enhancements"]["cross_encoder"]["model_name"]
                        .as_str()
                        .unwrap_or("cross-encoder/ms-marco-MiniLM-L-6-v2")
                        .to_string(),
                    max_length: parsed["enhancements"]["cross_encoder"]["max_length"]
                        .as_usize()
                        .unwrap_or(512),
                    batch_size: parsed["enhancements"]["cross_encoder"]["batch_size"]
                        .as_usize()
                        .unwrap_or(32),
                    top_k: parsed["enhancements"]["cross_encoder"]["top_k"]
                        .as_usize()
                        .unwrap_or(10),
                    min_confidence: parsed["enhancements"]["cross_encoder"]["min_confidence"]
                        .as_f32()
                        .unwrap_or(0.0),
                    normalize_scores: parsed["enhancements"]["cross_encoder"]["normalize_scores"]
                        .as_bool()
                        .unwrap_or(true),
                },
                #[cfg(feature = "lazygraphrag")]
                concept_selection: enhancements::ConceptSelectionConfig {
                    enabled: parsed["enhancements"]["concept_selection"]["enabled"]
                        .as_bool()
                        .unwrap_or(true),
                    top_k: parsed["enhancements"]["concept_selection"]["top_k"]
                        .as_usize()
                        .unwrap_or(20),
                    min_score: parsed["enhancements"]["concept_selection"]["min_score"]
                        .as_f32()
                        .unwrap_or(0.1),
                    degree_weight: parsed["enhancements"]["concept_selection"]["degree_weight"]
                        .as_f32()
                        .unwrap_or(0.4),
                    pagerank_weight: parsed["enhancements"]["concept_selection"]["pagerank_weight"]
                        .as_f32()
                        .unwrap_or(0.4),
                    idf_weight: parsed["enhancements"]["concept_selection"]["idf_weight"]
                        .as_f32()
                        .unwrap_or(0.2),
                    use_semantic_matching: parsed["enhancements"]["concept_selection"]
                        ["use_semantic_matching"]
                        .as_bool()
                        .unwrap_or(true),
                    max_query_concepts: parsed["enhancements"]["concept_selection"]
                        ["max_query_concepts"]
                        .as_usize()
                        .unwrap_or(10),
                },
            },
            auto_save: AutoSaveConfig {
                enabled: parsed["auto_save"]["enabled"].as_bool().unwrap_or(false),
                base_dir: parsed["auto_save"]["base_dir"]
                    .as_str()
                    .map(|s| s.to_string()),
                interval_seconds: parsed["auto_save"]["interval_seconds"]
                    .as_u64()
                    .unwrap_or(default_auto_save_interval()),
                workspace_name: parsed["auto_save"]["workspace_name"]
                    .as_str()
                    .map(|s| s.to_string()),
                max_versions: parsed["auto_save"]["max_versions"]
                    .as_usize()
                    .unwrap_or(default_max_versions()),
            },
            summarization: if parsed["summarization"].is_object() {
                crate::summarization::HierarchicalConfig {
                    merge_size: parsed["summarization"]["merge_size"]
                        .as_usize()
                        .unwrap_or(3),
                    max_summary_length: parsed["summarization"]["max_summary_length"]
                        .as_usize()
                        .unwrap_or(250),
                    min_node_size: parsed["summarization"]["min_node_size"]
                        .as_usize()
                        .unwrap_or(50),
                    overlap_sentences: parsed["summarization"]["overlap_sentences"]
                        .as_usize()
                        .unwrap_or(2),
                    llm_config: if parsed["summarization"]["llm_config"].is_object() {
                        crate::summarization::LLMConfig {
                            enabled: parsed["summarization"]["llm_config"]["enabled"]
                                .as_bool()
                                .unwrap_or(false),
                            model_name: parsed["summarization"]["llm_config"]["model_name"]
                                .as_str()
                                .unwrap_or("llama3.1:8b")
                                .to_string(),
                            temperature: parsed["summarization"]["llm_config"]["temperature"]
                                .as_f32()
                                .unwrap_or(0.3),
                            max_tokens: parsed["summarization"]["llm_config"]["max_tokens"]
                                .as_usize()
                                .unwrap_or(180),
                            strategy: match parsed["summarization"]["llm_config"]["strategy"]
                                .as_str()
                                .unwrap_or("progressive")
                            {
                                "uniform" => crate::summarization::LLMStrategy::Uniform,
                                "adaptive" => crate::summarization::LLMStrategy::Adaptive,
                                "progressive" => crate::summarization::LLMStrategy::Progressive,
                                _ => crate::summarization::LLMStrategy::Progressive,
                            },
                            level_configs: std::collections::HashMap::new(), // Would need more complex parsing
                        }
                    } else {
                        crate::summarization::LLMConfig::default()
                    },
                }
            } else {
                crate::summarization::HierarchicalConfig::default()
            },
            zero_cost_approach: if parsed["zero_cost_approach"].is_object() {
                ZeroCostApproachConfig {
                    approach: parsed["zero_cost_approach"]["approach"]
                        .as_str()
                        .unwrap_or("pure_algorithmic")
                        .to_string(),
                    lazy_graphrag: if parsed["zero_cost_approach"]["lazy_graphrag"].is_object() {
                        LazyGraphRAGConfig {
                            enabled: parsed["zero_cost_approach"]["lazy_graphrag"]["enabled"]
                                .as_bool()
                                .unwrap_or(false),
                            concept_extraction: ConceptExtractionConfig::default(),
                            co_occurrence: CoOccurrenceConfig::default(),
                            indexing: LazyIndexingConfig::default(),
                            query_expansion: LazyQueryExpansionConfig::default(),
                            relevance_scoring: LazyRelevanceScoringConfig::default(),
                        }
                    } else {
                        LazyGraphRAGConfig::default()
                    },
                    e2_graphrag: E2GraphRAGConfig::default(),
                    pure_algorithmic: PureAlgorithmicConfig::default(),
                    hybrid_strategy: HybridStrategyConfig::default(),
                }
            } else {
                ZeroCostApproachConfig::default()
            },
            advanced_features: AdvancedFeaturesConfig::default(),
        };

        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn to_file(&self, path: &str) -> Result<()> {
        let mut config_json = json::JsonValue::new_object();

        // Embeddings
        let mut embeddings = json::JsonValue::new_object();
        embeddings["dimension"] = json::JsonValue::from(self.embeddings.dimension);
        if let Some(endpoint) = &self.embeddings.api_endpoint {
            embeddings["api_endpoint"] = json::JsonValue::from(endpoint.as_str());
        }
        if let Some(key) = &self.embeddings.api_key {
            embeddings["api_key"] = json::JsonValue::from(key.as_str());
        }
        config_json["embeddings"] = embeddings;

        // Graph
        let mut graph = json::JsonValue::new_object();
        graph["max_connections"] = json::JsonValue::from(self.graph.max_connections);
        graph["similarity_threshold"] = json::JsonValue::from(self.graph.similarity_threshold);
        graph["extract_relationships"] = json::JsonValue::from(self.graph.extract_relationships);
        graph["relationship_confidence_threshold"] =
            json::JsonValue::from(self.graph.relationship_confidence_threshold);

        let mut traversal = json::JsonValue::new_object();
        traversal["max_depth"] = json::JsonValue::from(self.graph.traversal.max_depth);
        traversal["max_paths"] = json::JsonValue::from(self.graph.traversal.max_paths);
        traversal["use_edge_weights"] =
            json::JsonValue::from(self.graph.traversal.use_edge_weights);
        traversal["min_relationship_strength"] =
            json::JsonValue::from(self.graph.traversal.min_relationship_strength);
        graph["traversal"] = traversal;

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
        entities["use_gleaning"] = json::JsonValue::from(self.entities.use_gleaning);
        entities["max_gleaning_rounds"] = json::JsonValue::from(self.entities.max_gleaning_rounds);
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

        // Summarization
        let mut summarization = json::JsonValue::new_object();
        summarization["merge_size"] = json::JsonValue::from(self.summarization.merge_size);
        summarization["max_summary_length"] =
            json::JsonValue::from(self.summarization.max_summary_length);
        summarization["min_node_size"] = json::JsonValue::from(self.summarization.min_node_size);
        summarization["overlap_sentences"] =
            json::JsonValue::from(self.summarization.overlap_sentences);

        let mut llm_config = json::JsonValue::new_object();
        llm_config["enabled"] = json::JsonValue::from(self.summarization.llm_config.enabled);
        llm_config["model_name"] =
            json::JsonValue::from(self.summarization.llm_config.model_name.as_str());
        llm_config["temperature"] =
            json::JsonValue::from(self.summarization.llm_config.temperature);
        llm_config["max_tokens"] = json::JsonValue::from(self.summarization.llm_config.max_tokens);
        let strategy_str = match self.summarization.llm_config.strategy {
            crate::summarization::LLMStrategy::Uniform => "uniform",
            crate::summarization::LLMStrategy::Adaptive => "adaptive",
            crate::summarization::LLMStrategy::Progressive => "progressive",
        };
        llm_config["strategy"] = json::JsonValue::from(strategy_str);

        summarization["llm_config"] = llm_config;
        config_json["summarization"] = summarization;

        let content = json::stringify_pretty(config_json, 2);
        fs::write(path, content)?;
        Ok(())
    }
}
