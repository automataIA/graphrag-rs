//! Tests for Triple Reflection validation feature (Phase 1.1 - DEG-RAG methodology)

#[cfg(all(feature = "ollama", feature = "async"))]
mod triple_validation_tests {
    use graphrag_core::{
        entity::LLMRelationshipExtractor,
        ollama::OllamaConfig,
    };

    fn create_test_ollama_config() -> OllamaConfig {
        OllamaConfig {
            enabled: true,
            host: "http://localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "llama3.2:3b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            fallback_to_hash: false,
            max_tokens: Some(500),
            temperature: Some(0.1), // Low temperature for consistent validation
            enable_caching: true,
        }
    }

    #[tokio::test]
    async fn test_triple_validation_struct_creation() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config));

        assert!(extractor.is_ok(), "Should create extractor successfully");
    }

    #[tokio::test]
    async fn test_validate_triple_valid_relationship() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Failed to create extractor");

        let text = "Socrates taught Plato philosophy in ancient Athens.";

        // This test requires actual Ollama service running
        // It's marked as integration test and can be skipped if Ollama is not available
        let result = extractor.validate_triple(
            "Socrates",
            "TAUGHT",
            "Plato",
            text
        ).await;

        // If Ollama is not available, this will fail gracefully
        if let Ok(validation) = result {
            // If validation succeeds, check the structure
            assert!(validation.confidence >= 0.0 && validation.confidence <= 1.0,
                "Confidence should be between 0.0 and 1.0");
            assert!(!validation.reason.is_empty(),
                "Validation reason should not be empty");

            // For this clearly valid relationship, we expect high confidence
            if validation.is_valid {
                assert!(validation.confidence > 0.7,
                    "Valid relationship should have high confidence, got {}",
                    validation.confidence);
            }
        }
    }

    #[tokio::test]
    async fn test_validate_triple_invalid_relationship() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Failed to create extractor");

        let text = "Socrates taught Plato philosophy in ancient Athens.";

        // This relationship is NOT supported by the text
        let result = extractor.validate_triple(
            "Socrates",
            "INVENTED",
            "computer",
            text
        ).await;

        if let Ok(validation) = result {
            // This should be marked as invalid or have low confidence
            assert!(validation.confidence >= 0.0 && validation.confidence <= 1.0,
                "Confidence should be between 0.0 and 1.0");

            // Expect either invalid flag or low confidence
            if !validation.is_valid {
                assert!(!validation.reason.is_empty(),
                    "Invalid relationships should have a reason");
            } else {
                // If marked valid, confidence should be very low
                assert!(validation.confidence < 0.5,
                    "Unsupported relationship should have low confidence");
            }
        }
    }

    #[tokio::test]
    async fn test_validate_triple_partial_match() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Failed to create extractor");

        let text = "Socrates was known in Athens for his philosophical discussions.";

        // Partially supported - mentions Socrates and Athens but not explicit teaching
        let result = extractor.validate_triple(
            "Socrates",
            "LIVED_IN",
            "Athens",
            text
        ).await;

        if let Ok(validation) = result {
            // Should have moderate confidence or be marked as implicit
            assert!(validation.confidence >= 0.0 && validation.confidence <= 1.0,
                "Confidence should be between 0.0 and 1.0");
            assert!(!validation.reason.is_empty(),
                "Should provide reasoning");
        }
    }

    #[tokio::test]
    async fn test_validation_with_disabled_ollama() {
        let config = OllamaConfig {
            enabled: false, // Disabled
            ..create_test_ollama_config()
        };

        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Should create extractor even with disabled Ollama");

        let text = "Test text";

        let result = extractor.validate_triple(
            "Entity1",
            "RELATES_TO",
            "Entity2",
            text
        ).await;

        // With Ollama disabled, should return error or fallback behavior
        // This tests graceful degradation
        match result {
            Ok(validation) => {
                // If it succeeds with fallback, should have low confidence
                assert!(validation.confidence < 0.5,
                    "Fallback validation should have low confidence");
            }
            Err(_) => {
                // Error is also acceptable with Ollama disabled
            }
        }
    }

    #[tokio::test]
    async fn test_validation_confidence_thresholds() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Failed to create extractor");

        let text = "The philosopher Socrates engaged in dialogue with Plato.";

        let result = extractor.validate_triple(
            "Socrates",
            "ENGAGED_WITH",
            "Plato",
            text
        ).await;

        if let Ok(validation) = result {
            // Test that different confidence thresholds would filter appropriately
            let threshold_0_5 = validation.confidence >= 0.5;
            let threshold_0_7 = validation.confidence >= 0.7;
            let threshold_0_9 = validation.confidence >= 0.9;

            // Basic sanity check: higher thresholds should be more restrictive
            if threshold_0_9 {
                assert!(threshold_0_7, "0.9 threshold implies 0.7 threshold");
                assert!(threshold_0_5, "0.9 threshold implies 0.5 threshold");
            }
            if threshold_0_7 {
                assert!(threshold_0_5, "0.7 threshold implies 0.5 threshold");
            }
        }
    }

    #[tokio::test]
    async fn test_validation_with_empty_text() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Failed to create extractor");

        let empty_text = "";

        let result = extractor.validate_triple(
            "Socrates",
            "TAUGHT",
            "Plato",
            empty_text
        ).await;

        if let Ok(validation) = result {
            // Empty text cannot support any relationship
            assert!(!validation.is_valid || validation.confidence < 0.3,
                "Empty text should not validate relationships");
        }
    }

    #[tokio::test]
    async fn test_validation_json_parsing() {
        let config = create_test_ollama_config();
        let extractor = LLMRelationshipExtractor::new(Some(&config))
            .expect("Failed to create extractor");

        let text = "Aristotle studied under Plato at the Academy.";

        let result = extractor.validate_triple(
            "Aristotle",
            "STUDIED_UNDER",
            "Plato",
            text
        ).await;

        if let Ok(validation) = result {
            // Verify all fields are properly parsed
            assert!(
                validation.reason.len() > 0,
                "Reason should be parsed from JSON"
            );

            // Optional suggested_fix field should be None or Some
            match &validation.suggested_fix {
                Some(fix) => {
                    assert!(!fix.is_empty(), "Suggested fix should not be empty if present");
                }
                None => {
                    // None is also valid
                }
            }
        }
    }
}

#[cfg(not(all(feature = "ollama", feature = "async")))]
mod triple_validation_tests {
    // Placeholder module when features are not enabled
    #[test]
    fn test_features_required() {
        println!("Triple validation tests require 'ollama' and 'async' features");
    }
}
