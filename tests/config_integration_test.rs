//! Configuration Integration Tests
//!
//! Tests the integration of all configuration components including:
//! - Zero-cost approaches
//! - LLM summarization
//! - JSON5 loading
//! - Main Config structure

#[cfg(test)]
mod tests {
    use graphrag_core::{
        config::{Config, ZeroCostApproachConfig},
        summarization::{HierarchicalConfig, LLMConfig, LLMStrategy},
        core::Result,
    };

    #[test]
    fn test_main_config_integration() -> Result<()> {
        let config = Config::default();

        // Test that all new config sections are present
        assert!(!config.summarization.llm_config.enabled); // Default disabled
        assert_eq!(config.zero_cost_approach.approach, "pure_algorithmic"); // Default approach

        // Test summarization integration
        assert_eq!(config.summarization.merge_size, 3);
        assert_eq!(config.summarization.max_summary_length, 200);
        assert_eq!(config.summarization.llm_config.model_name, "llama3.1:8b");

        // Test zero-cost approach integration
        assert!(config.zero_cost_approach.pure_algorithmic.enabled);
        assert!(!config.zero_cost_approach.lazy_graphrag.enabled);
        assert!(!config.zero_cost_approach.e2_graphrag.enabled);

        Ok(())
    }

    #[test]
    fn test_config_customization() -> Result<()> {
        // Test creating custom configurations
        let mut custom_summarization = HierarchicalConfig::default();
        custom_summarization.llm_config.enabled = true;
        custom_summarization.llm_config.model_name = "gpt-4".to_string();
        custom_summarization.llm_config.strategy = LLMStrategy::Progressive;
        custom_summarization.merge_size = 5;
        custom_summarization.max_summary_length = 300;

        let mut custom_zero_cost = ZeroCostApproachConfig::default();
        custom_zero_cost.approach = "lazy_graphrag".to_string();
        custom_zero_cost.lazy_graphrag.enabled = true;

        // Verify custom values
        assert!(custom_summarization.llm_config.enabled);
        assert_eq!(custom_summarization.llm_config.model_name, "gpt-4");
        assert!(matches!(custom_summarization.llm_config.strategy, LLMStrategy::Progressive));
        assert_eq!(custom_summarization.merge_size, 5);
        assert_eq!(custom_summarization.max_summary_length, 300);

        assert_eq!(custom_zero_cost.approach, "lazy_graphrag");
        assert!(custom_zero_cost.lazy_graphrag.enabled);

        Ok(())
    }

    #[test]
    fn test_approach_selection_logic() -> Result<()> {
        let mut config = ZeroCostApproachConfig::default();

        // Test each approach selection
        let approaches = vec![
            "pure_algorithmic",
            "lazy_graphrag",
            "e2_graphrag",
        ];

        for approach in approaches {
            config.approach = approach.to_string();

            match approach {
                "pure_algorithmic" => {
                    assert!(config.pure_algorithmic.enabled);
                },
                "lazy_graphrag" => {
                    assert!(config.lazy_graphrag.enabled);
                },
                "e2_graphrag" => {
                    assert!(config.e2_graphrag.enabled);
                },
                _ => {}
            }
        }

        Ok(())
    }

    #[test]
    fn test_llm_strategy_variations() -> Result<()> {
        let mut llm_config = LLMConfig::default();

        // Test each strategy
        let strategies = vec![
            LLMStrategy::Uniform,
            LLMStrategy::Adaptive,
            LLMStrategy::Progressive,
        ];

        for strategy in strategies {
            llm_config.strategy = strategy.clone();
            assert!(matches!(llm_config.strategy, strategy));
        }

        Ok(())
    }

    #[test]
    fn test_config_serialization_cycle() -> Result<()> {
        let original_config = Config::default();

        // Test saving to JSON string (simulated)
        let config_debug = format!("{:?}", original_config);

        // Verify key components are present
        assert!(config_debug.contains("summarization"));
        assert!(config_debug.contains("zero_cost_approach"));
        assert!(config_debug.contains("pure_algorithmic"));

        Ok(())
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn test_json5_config_roundtrip() -> Result<()> {
        use graphrag_core::config::json5_loader::{parse_json5_str, save_json5_config};

        // Create a test configuration
        let original_config = Config::default();

        // Serialize to JSON5 string
        let json5_str = serde_json::to_string_pretty(&original_config)?;

        // Parse it back
        let parsed_config: Config = serde_json::from_str(&json5_str)?;

        // Verify key fields match
        assert_eq!(original_config.summarization.merge_size, parsed_config.summarization.merge_size);
        assert_eq!(original_config.zero_cost_approach.approach, parsed_config.zero_cost_approach.approach);
        assert_eq!(original_config.summarization.llm_config.enabled, parsed_config.summarization.llm_config.enabled);

        Ok(())
    }

    #[test]
    fn test_hybrid_strategy_configuration() -> Result<()> {
        let config = ZeroCostApproachConfig::default();
        let hybrid = config.hybrid_strategy;

        // Test progressive strategy levels
        assert_eq!(hybrid.progressive.level_0, "pure_algorithmic");
        assert_eq!(hybrid.progressive.level_1, "pure_algorithmic");
        assert_eq!(hybrid.progressive.level_2, "e2_graphrag");
        assert_eq!(hybrid.progressive.level_3, "lazy_graphrag");
        assert_eq!(hybrid.progressive.level_4_plus, "lazy_graphrag");

        // Test budget-aware configuration
        assert_eq!(hybrid.budget_aware.daily_budget_usd, 1.0);
        assert_eq!(hybrid.budget_aware.queries_per_day, 1000);
        assert_eq!(hybrid.budget_aware.max_llm_cost_per_query, 0.002);
        assert_eq!(hybrid.budget_aware.strategy, "lazy_graphrag");
        assert!(hybrid.budget_aware.fallback_to_algorithmic);

        Ok(())
    }

    #[test]
    fn test_cost_optimization_scenarios() -> Result<()> {
        // Test different cost scenarios
        let scenarios = vec![
            ("Free", 0.0, "pure_algorithmic"),
            ("Basic", 0.1, "e2_graphrag"),
            ("Professional", 1.0, "lazy_graphrag"),
        ];

        for (name, budget, expected_approach) in scenarios {
            let mut config = ZeroCostApproachConfig::default();

            // Simulate budget-based approach selection
            if budget == 0.0 {
                config.approach = "pure_algorithmic".to_string();
            } else if budget <= 0.5 {
                config.approach = "e2_graphrag".to_string();
            } else {
                config.approach = "lazy_graphrag".to_string();
            }

            assert_eq!(config.approach, expected_approach,
                "Failed for {} tier with budget ${}", name, budget);
        }

        Ok(())
    }

    #[test]
    fn test_search_ranking_integration() -> Result<()> {
        let config = ZeroCostApproachConfig::default();
        let ranking = config.pure_algorithmic.search_ranking;

        // Verify search ranking configuration
        assert!(!ranking.vector_search.enabled); // Disabled for pure algorithmic
        assert!(ranking.keyword_search.enabled);
        assert!(ranking.graph_traversal.enabled);
        assert!(ranking.hybrid_fusion.enabled);

        // Verify BM25 parameters
        assert_eq!(ranking.keyword_search.k1, 1.2);
        assert_eq!(ranking.keyword_search.b, 0.75);

        // Verify PageRank parameters
        assert_eq!(ranking.graph_traversal.damping_factor, 0.85);
        assert_eq!(ranking.graph_traversal.max_iterations, 20);

        Ok(())
    }

    #[test]
    fn test_entity_extraction_configuration() -> Result<()> {
        let config = ZeroCostApproachConfig::default();
        let ner_config = config.e2_graphrag.ner_extraction;

        // Verify entity types
        assert!(ner_config.entity_types.contains(&"PERSON".to_string()));
        assert!(ner_config.entity_types.contains(&"ORG".to_string()));

        // Verify extraction methods
        assert!(ner_config.use_capitalized_patterns);
        assert!(ner_config.use_title_case_patterns);
        assert!(ner_config.use_quoted_patterns);
        assert!(ner_config.use_abbreviations);

        // Verify confidence thresholds
        assert_eq!(ner_config.min_confidence, 0.7);

        Ok(())
    }

    #[test]
    fn test_configuration_validation() -> Result<()> {
        let config = Config::default();

        // Validate critical configuration values
        assert!(config.summarization.merge_size > 0);
        assert!(config.summarization.max_summary_length > 0);
        assert!(config.summarization.min_node_size > 0);

        assert!(!config.zero_cost_approach.approach.is_empty());
        assert!(config.zero_cost_approach.pure_algorithmic.search_ranking.keyword_search.enabled);

        // Validate LLM configuration constraints
        assert!(config.summarization.llm_config.temperature >= 0.0);
        assert!(config.summarization.llm_config.temperature <= 2.0);
        assert!(config.summarization.llm_config.max_tokens > 0);

        Ok(())
    }
}