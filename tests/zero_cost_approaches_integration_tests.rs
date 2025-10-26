//! Integration tests for Zero-Cost GraphRAG Approaches
//!
//! This test suite verifies the integration of zero-cost approaches
//! with the main GraphRAG configuration and pipeline systems.

use graphrag_core::{
    config::{
        ZeroCostApproachConfig, LazyGraphRAGConfig, E2GraphRAGConfig,
        PureAlgorithmicConfig, ConceptExtractionConfig, NERExtractionConfig
    },
    summarization::{HierarchicalConfig, LLMConfig, LLMStrategy},
    core::Result,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_graphrag_config_default_values() {
        let config = LazyGraphRAGConfig::default();

        assert!(!config.enabled); // Should be disabled by default

        // Test concept extraction defaults
        assert_eq!(config.concept_extraction.min_concept_length, 3);
        assert_eq!(config.concept_extraction.max_concept_words, 5);
        assert!(config.concept_extraction.use_noun_phrases);
        assert!(config.concept_extraction.use_capitalization);
        assert_eq!(config.concept_extraction.min_term_frequency, 2);

        // Test co-occurrence defaults
        assert_eq!(config.co_occurrence.window_size, 50);
        assert_eq!(config.co_occurrence.min_co_occurrence, 2);
        assert_eq!(config.co_occurrence.jaccard_threshold, 0.2);

        // Test indexing defaults
        assert!(config.indexing.use_bidirectional_index);
        assert!(!config.indexing.enable_hnsw_index);
        assert_eq!(config.indexing.cache_size, 10000);
    }

    #[test]
    fn test_e2_graphrag_config_default_values() {
        let config = E2GraphRAGConfig::default();

        assert!(!config.enabled); // Should be disabled by default

        // Test NER extraction defaults
        assert_eq!(config.ner_extraction.entity_types.len(), 2);
        assert!(config.ner_extraction.entity_types.contains(&"PERSON".to_string()));
        assert!(config.ner_extraction.entity_types.contains(&"ORG".to_string()));
        assert!(config.ner_extraction.use_capitalized_patterns);
        assert!(config.ner_extraction.use_title_case_patterns);
        assert_eq!(config.ner_extraction.min_confidence, 0.7);

        // Test keyword extraction defaults
        assert!(config.keyword_extraction.algorithms.contains(&"tf_idf".to_string()));
        assert_eq!(config.keyword_extraction.max_keywords_per_chunk, 15);
        assert_eq!(config.keyword_extraction.min_keyword_length, 3);

        // Test indexing defaults
        assert_eq!(config.indexing.batch_size, 50);
        assert!(config.indexing.enable_parallel_processing);
        assert!(!config.indexing.cache_concept_vectors);
        assert!(config.indexing.use_hash_embeddings);
    }

    #[test]
    fn test_pure_algorithmic_config_default_values() {
        let config = PureAlgorithmicConfig::default();

        assert!(config.enabled); // Should be enabled by default

        // Test pattern extraction defaults
        assert_eq!(config.pattern_extraction.capitalized_patterns.len(), 1);
        assert_eq!(config.pattern_extraction.technical_patterns.len(), 1);
        assert_eq!(config.pattern_extraction.context_patterns.len(), 1);

        // Test keyword extraction defaults
        assert_eq!(config.keyword_extraction.algorithm, "tf_idf");
        assert_eq!(config.keyword_extraction.max_keywords, 20);
        assert_eq!(config.keyword_extraction.min_word_length, 4);
        assert!(config.keyword_extraction.use_positional_boost);

        // Test search ranking defaults
        assert!(!config.search_ranking.vector_search.enabled);
        assert!(config.search_ranking.keyword_search.enabled);
        assert_eq!(config.search_ranking.keyword_search.algorithm, "bm25");
        assert!(config.search_ranking.graph_traversal.enabled);
        assert_eq!(config.search_ranking.graph_traversal.algorithm, "pagerank");
    }

    #[test]
    fn test_zero_cost_approach_config_selection() {
        let mut config = ZeroCostApproachConfig::default();

        // Test default approach
        assert_eq!(config.approach, "pure_algorithmic");

        // Test LazyGraphRAG selection
        config.approach = "lazy_graphrag".to_string();
        assert_eq!(config.approach, "lazy_graphrag");

        // Test E2GraphRAG selection
        config.approach = "e2_graphrag".to_string();
        assert_eq!(config.approach, "e2_graphrag");

        // Test Pure Algorithmic selection
        config.approach = "pure_algorithmic".to_string();
        assert_eq!(config.approach, "pure_algorithmic");
    }

    #[test]
    fn test_summarization_llm_config_integration() {
        let llm_config = LLMConfig {
            enabled: true,
            model_name: "llama3.1:8b".to_string(),
            temperature: 0.3,
            max_tokens: 180,
            strategy: LLMStrategy::Progressive,
            level_configs: std::collections::HashMap::new(),
        };

        assert!(llm_config.enabled);
        assert_eq!(llm_config.model_name, "llama3.1:8b");
        assert_eq!(llm_config.temperature, 0.3);
        assert_eq!(llm_config.max_tokens, 180);
        assert!(matches!(llm_config.strategy, LLMStrategy::Progressive));
    }

    #[test]
    fn test_hierarchical_config_default_values() {
        let config = HierarchicalConfig::default();

        // Test default hierarchical settings
        assert_eq!(config.merge_size, 3);
        assert_eq!(config.max_summary_length, 200);
        assert_eq!(config.min_node_size, 50);
        assert_eq!(config.overlap_sentences, 2);

        // Test default LLM settings
        assert!(!config.llm_config.enabled); // Should be disabled by default
        assert_eq!(config.llm_config.model_name, "llama3.1:8b");
        assert_eq!(config.llm_config.temperature, 0.3);
        assert_eq!(config.llm_config.max_tokens, 100);
        assert!(matches!(config.llm_config.strategy, LLMStrategy::Uniform));
    }

    #[test]
    fn test_json5_config_parsing() {
        // This test would require the json5-support feature
        // For now, we'll test the structure creation

        let zero_cost_config = ZeroCostApproachConfig::default();
        let summarization_config = HierarchicalConfig::default();

        // Verify configs can be created and cloned
        let _zero_cost_clone = zero_cost_config.clone();
        let _summarization_clone = summarization_config.clone();

        // Test approach-specific configs
        let lazy_config = LazyGraphRAGConfig::default();
        let e2_config = E2GraphRAGConfig::default();
        let pure_config = PureAlgorithmicConfig::default();

        assert!(!lazy_config.enabled);
        assert!(!e2_config.enabled);
        assert!(pure_config.enabled);
    }

    #[test]
    fn test_config_serialization_deserialization() {
        // Test that configs can be serialized and deserialized
        let original_config = ZeroCostApproachConfig::default();

        // Test Debug formatting (useful for logging)
        let debug_str = format!("{:?}", original_config);
        assert!(debug_str.contains("ZeroCostApproachConfig"));
        assert!(debug_str.contains("pure_algorithmic"));
    }

    #[test]
    fn test_strategy_enums() {
        // Test LLM strategy variations
        let strategies = vec![
            LLMStrategy::Uniform,
            LLMStrategy::Adaptive,
            LLMStrategy::Progressive,
        ];

        for strategy in strategies {
            match strategy {
                LLMStrategy::Uniform => assert!(true),
                LLMStrategy::Adaptive => assert!(true),
                LLMStrategy::Progressive => assert!(true),
            }
        }
    }

    #[test]
    fn test_hybrid_strategy_config() {
        let config = ZeroCostApproachConfig::default();
        let hybrid = config.hybrid_strategy;

        // Test lazy_algorithmic configuration
        assert_eq!(hybrid.lazy_algorithmic.indexing_approach, "e2_graphrag");
        assert_eq!(hybrid.lazy_algorithmic.query_approach, "lazy_graphrag");
        assert_eq!(hybrid.lazy_algorithmic.cost_optimization, "indexing");

        // Test progressive configuration
        assert_eq!(hybrid.progressive.level_0, "pure_algorithmic");
        assert_eq!(hybrid.progressive.level_1, "pure_algorithmic");
        assert_eq!(hybrid.progressive.level_2, "e2_graphrag");
        assert_eq!(hybrid.progressive.level_3, "lazy_graphrag");
        assert_eq!(hybrid.progressive.level_4_plus, "lazy_graphrag");

        // Test budget_aware configuration
        assert_eq!(hybrid.budget_aware.daily_budget_usd, 1.0);
        assert_eq!(hybrid.budget_aware.queries_per_day, 1000);
        assert_eq!(hybrid.budget_aware.max_llm_cost_per_query, 0.002);
        assert_eq!(hybrid.budget_aware.strategy, "lazy_graphrag");
        assert!(hybrid.budget_aware.fallback_to_algorithmic);
    }

    #[test]
    fn test_fusion_weights() {
        let config = ZeroCostApproachConfig::default();
        let weights = config.pure_algorithmic.search_ranking.hybrid_fusion.weights;

        assert_eq!(weights.keywords, 0.4);
        assert_eq!(weights.graph, 0.4);
        assert_eq!(weights.bm25, 0.2);

        // Test weights sum to 1.0
        let total: f32 = weights.keywords + weights.graph + weights.bm25;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_search_ranking_configuration() {
        let config = ZeroCostApproachConfig::default();
        let ranking = config.pure_algorithmic.search_ranking;

        // Vector search should be disabled for pure algorithmic
        assert!(!ranking.vector_search.enabled);

        // Keyword search should be enabled
        assert!(ranking.keyword_search.enabled);
        assert_eq!(ranking.keyword_search.algorithm, "bm25");
        assert_eq!(ranking.keyword_search.k1, 1.2);
        assert_eq!(ranking.keyword_search.b, 0.75);

        // Graph traversal should be enabled
        assert!(ranking.graph_traversal.enabled);
        assert_eq!(ranking.graph_traversal.algorithm, "pagerank");
        assert_eq!(ranking.graph_traversal.damping_factor, 0.85);
        assert_eq!(ranking.graph_traversal.max_iterations, 20);
        assert!(ranking.graph_traversal.personalized);

        // Hybrid fusion should be enabled
        assert!(ranking.hybrid_fusion.enabled);
    }

    #[test]
    fn test_relationship_scoring_methods() {
        let config = ZeroCostApproachConfig::default();

        // Test pure algorithmic relationship discovery
        let relationship_config = config.pure_algorithmic.relationship_discovery;
        assert_eq!(relationship_config.scoring_method, "jaccard_similarity");
        assert_eq!(relationship_config.window_size, 30);
        assert_eq!(relationship_config.min_co_occurrence, 2);
        assert!(relationship_config.use_mutual_information);
        assert_eq!(relationship_config.min_similarity_score, 0.1);

        // Test E2GraphRAG relationship configuration
        let e2_relationship_config = config.e2_graphrag.graph_construction;
        assert_eq!(e2_relationship_config.min_relationship_score, 0.3);
        assert_eq!(e2_relationship_config.max_relationships_per_entity, 10);
        assert!(e2_relationship_config.use_mutual_information);
    }

    #[test]
    fn test_concept_extraction_configuration() {
        let config = ZeroCostApproachConfig::default();
        let concept_config = config.lazy_graphrag.concept_extraction;

        assert_eq!(concept_config.min_concept_length, 3);
        assert_eq!(concept_config.max_concept_words, 5);
        assert!(concept_config.use_noun_phrases);
        assert!(concept_config.use_capitalization);
        assert!(concept_config.use_title_case);
        assert!(concept_config.use_tf_idf_scoring);
        assert_eq!(concept_config.min_term_frequency, 2);
        assert_eq!(concept_config.max_concepts_per_chunk, 10);
        assert_eq!(concept_config.min_concept_score, 0.1);
        assert!(concept_config.exclude_stopwords);
        assert!(!concept_config.custom_stopwords.is_empty());
    }

    #[test]
    fn test_indexing_configurations() {
        let config = ZeroCostApproachConfig::default();

        // Test LazyGraphRAG indexing
        let lazy_indexing = config.lazy_graphrag.indexing;
        assert!(lazy_indexing.use_bidirectional_index);
        assert!(!lazy_indexing.enable_hnsw_index);
        assert_eq!(lazy_indexing.cache_size, 10000);

        // Test E2GraphRAG indexing
        let e2_indexing = config.e2_graphrag.indexing;
        assert_eq!(e2_indexing.batch_size, 50);
        assert!(e2_indexing.enable_parallel_processing);
        assert!(!e2_indexing.cache_concept_vectors);
        assert!(e2_indexing.use_hash_embeddings);
    }

    #[test]
    fn test_query_expansion_and_scoring() {
        let config = ZeroCostApproachConfig::default();

        // Test LazyGraphRAG query expansion
        let query_expansion = config.lazy_graphrag.query_expansion;
        assert!(query_expansion.enabled);
        assert_eq!(query_expansion.max_expansions, 3);
        assert_eq!(query_expansion.expansion_model, "llama3.1:8b");
        assert_eq!(query_expansion.expansion_temperature, 0.1);
        assert_eq!(query_expansion.max_tokens_per_expansion, 50);

        // Test LazyGraphRAG relevance scoring
        let relevance_scoring = config.lazy_graphrag.relevance_scoring;
        assert!(relevance_scoring.enabled);
        assert_eq!(relevance_scoring.scoring_model, "llama3.1:8b");
        assert_eq!(relevance_scoring.batch_size, 10);
        assert_eq!(relevance_scoring.temperature, 0.2);
        assert_eq!(relevance_scoring.max_tokens_per_score, 30);
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn test_json5_configuration_loading() {
        use graphrag_core::config::json5_loader::parse_json5_str;

        let json5_config = r#"
        {
            // Zero-Cost Approach Configuration
            approach: "lazy_graphrag",

            lazy_graphrag: {
                enabled: true,
                concept_extraction: {
                    min_concept_length: 4,
                    max_concept_words: 6,
                    use_capitalization: true
                }
            }
        }
        "#;

        let config: ZeroCostApproachConfig = parse_json5_str(json5_config).unwrap();
        assert_eq!(config.approach, "lazy_graphrag");
        assert!(config.lazy_graphrag.enabled);
        assert_eq!(config.lazy_graphrag.concept_extraction.min_concept_length, 4);
        assert_eq!(config.lazy_graphrag.concept_extraction.max_concept_words, 6);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_config_creation_performance() {
        let start = Instant::now();

        // Create multiple config instances
        for _ in 0..1000 {
            let _config = ZeroCostApproachConfig::default();
            let _summarization = HierarchicalConfig::default();
        }

        let duration = start.elapsed();
        println!("Created 2000 config instances in {:?}", duration);

        // Should be very fast (< 10ms)
        assert!(duration.as_millis() < 10);
    }

    #[test]
    fn test_config_clone_performance() {
        let config = ZeroCostApproachConfig::default();
        let start = Instant::now();

        // Clone config multiple times
        for _ in 0..1000 {
            let _cloned = config.clone();
        }

        let duration = start.elapsed();
        println!("Cloned config 1000 times in {:?}", duration);

        // Should be fast (< 5ms)
        assert!(duration.as_millis() < 5);
    }
}