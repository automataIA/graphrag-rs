//! Tests for Ollama enhancements: streaming, custom parameters, caching, metrics

#[cfg(all(feature = "ollama", feature = "async"))]
mod ollama_tests {
    use graphrag_core::ollama::{OllamaClient, OllamaConfig, OllamaGenerationParams};

    #[tokio::test]
    async fn test_ollama_config_with_caching() {
        let config = OllamaConfig {
            enabled: true,
            host: "http://localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "llama3.2:3b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            fallback_to_hash: true,
            max_tokens: Some(100),
            temperature: Some(0.7),
            enable_caching: true,
        };

        assert!(config.enable_caching);
        assert_eq!(config.max_tokens, Some(100));
    }

    #[tokio::test]
    async fn test_generation_params_custom() {
        let params = OllamaGenerationParams {
            num_predict: Some(500),
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(50),
            stop: Some(vec!["END".to_string(), "STOP".to_string()]),
            repeat_penalty: Some(1.2),
        };

        assert_eq!(params.num_predict, Some(500));
        assert_eq!(params.temperature, Some(0.8));
        assert_eq!(params.top_p, Some(0.95));
        assert_eq!(params.top_k, Some(50));
        assert_eq!(params.stop, Some(vec!["END".to_string(), "STOP".to_string()]));
        assert_eq!(params.repeat_penalty, Some(1.2));
    }

    #[tokio::test]
    async fn test_ollama_client_stats() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);

        let stats = client.get_stats();
        assert_eq!(stats.get_total_requests(), 0);
        assert_eq!(stats.get_successful_requests(), 0);
        assert_eq!(stats.get_failed_requests(), 0);
        assert_eq!(stats.get_total_tokens(), 0);
        assert_eq!(stats.get_success_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_ollama_stats_recording() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);
        let stats = client.get_stats();

        // Record some successes
        stats.record_success(100);
        stats.record_success(150);
        stats.record_success(200);

        assert_eq!(stats.get_total_requests(), 3);
        assert_eq!(stats.get_successful_requests(), 3);
        assert_eq!(stats.get_failed_requests(), 0);
        assert_eq!(stats.get_total_tokens(), 450);
        assert_eq!(stats.get_success_rate(), 1.0);

        // Record a failure
        stats.record_failure();

        assert_eq!(stats.get_total_requests(), 4);
        assert_eq!(stats.get_successful_requests(), 3);
        assert_eq!(stats.get_failed_requests(), 1);
        assert_eq!(stats.get_success_rate(), 0.75);
    }

    #[tokio::test]
    #[cfg(feature = "dashmap")]
    async fn test_ollama_client_cache() {
        let config = OllamaConfig {
            enable_caching: true,
            ..Default::default()
        };
        let client = OllamaClient::new(config);

        // Initially empty
        assert_eq!(client.cache_size(), 0);

        // Note: Cache is populated by actual API calls, not directly accessible
        // This test just verifies the cache API exists
        client.clear_cache();
        assert_eq!(client.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_generation_params_default() {
        let params = OllamaGenerationParams::default();

        assert_eq!(params.num_predict, Some(2000));
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.top_k, Some(40));
        assert_eq!(params.repeat_penalty, Some(1.1));
        assert_eq!(params.stop, None);
    }

    // Integration test with adapter
    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_ollama_adapter_with_params() {
        use graphrag_core::core::ollama_adapters::OllamaLanguageModelAdapter;
        use graphrag_core::core::traits::{AsyncLanguageModel, GenerationParams};

        let config = OllamaConfig::default();
        let adapter = OllamaLanguageModelAdapter::new(config);

        // Test model info
        let info = adapter.model_info().await;
        assert_eq!(info.name, "llama3.2:3b");
        assert_eq!(info.max_context_length, Some(4096));
        assert!(info.supports_streaming);

        // Test usage stats (should be zero initially)
        let stats = adapter.get_usage_stats().await.unwrap();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.error_rate, 0.0);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_ollama_embedder_adapter() {
        use graphrag_core::core::ollama_adapters::OllamaEmbedderAdapter;
        use graphrag_core::core::traits::AsyncEmbedder;

        let adapter = OllamaEmbedderAdapter::new("nomic-embed-text:latest", 768);

        // Test dimension
        assert_eq!(adapter.dimension(), 768);

        // Test availability (returns true by default)
        assert!(adapter.is_ready().await);
    }
}
