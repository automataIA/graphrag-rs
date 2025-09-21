//! Comprehensive integration tests for the caching system

#[cfg(feature = "caching")]
mod caching_tests {
    use graphrag_rs::{
        CachedLLMClient, CacheConfig, CacheKeyGenerator, EvictionPolicy, KeyStrategy,
        WarmingConfig, WarmingStrategy, MockLLM, LanguageModel, GenerationParams, Result,
    };
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn test_basic_cache_functionality() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        let prompt = "What is artificial intelligence?";

        // First call should be a cache miss
        let start = Instant::now();
        let response1 = client.complete_async(prompt).await?;
        let first_call_duration = start.elapsed();

        let stats_after_first = client.cache_statistics();
        assert_eq!(stats_after_first.cache_misses, 1);
        assert_eq!(stats_after_first.cache_hits, 0);
        assert_eq!(stats_after_first.current_size, 1);

        // Second call should be a cache hit (and much faster)
        let start = Instant::now();
        let response2 = client.complete_async(prompt).await?;
        let second_call_duration = start.elapsed();

        let stats_after_second = client.cache_statistics();
        assert_eq!(stats_after_second.cache_misses, 1);
        assert_eq!(stats_after_second.cache_hits, 1);
        assert_eq!(stats_after_second.current_size, 1);

        // Responses should be identical
        assert_eq!(response1, response2);

        // Second call should be significantly faster
        assert!(second_call_duration < first_call_duration / 2);

        // Hit rate should be 50%
        assert!((stats_after_second.hit_rate - 0.5).abs() < 0.01);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_with_different_parameters() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        let prompt = "What is machine learning?";
        let params1 = GenerationParams {
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        };
        let params2 = GenerationParams {
            temperature: Some(0.9),
            max_tokens: Some(100),
            ..Default::default()
        };

        // Calls with different parameters should create separate cache entries
        let response1 = client.complete_with_params_async(prompt, params1.clone()).await?;
        let _response2 = client.complete_with_params_async(prompt, params2).await?;

        let stats = client.cache_statistics();
        assert_eq!(stats.cache_misses, 2); // Both should be misses
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.current_size, 2); // Two separate entries

        // Repeat first call - should be a hit
        let response3 = client.complete_with_params_async(prompt, params1).await?;
        assert_eq!(response1, response3);

        let stats_after_repeat = client.cache_statistics();
        assert_eq!(stats_after_repeat.cache_hits, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_eviction() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::builder()
            .max_capacity(3) // Small cache to trigger eviction
            .ttl_seconds(300)
            .eviction_policy(EvictionPolicy::LRU)
            .build();

        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        // Fill cache beyond capacity
        for i in 0..5 {
            let prompt = format!("Query number {i}");
            let _ = client.complete_async(&prompt).await?;
        }

        let stats = client.cache_statistics();
        // Should have evicted some entries
        assert!(stats.current_size <= 3);
        assert!(stats.evictions > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_invalidation() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        let prompt = "What is deep learning?";

        // Cache the response
        let _ = client.complete_async(prompt).await?;
        assert!(client.is_cached(prompt, None).await);

        // Invalidate the specific entry
        let was_invalidated = client.invalidate(prompt, None).await?;
        assert!(was_invalidated);
        assert!(!client.is_cached(prompt, None).await);

        // Clear entire cache
        let _ = client.complete_async(prompt).await?; // Re-cache
        assert!(client.is_cached(prompt, None).await);

        client.clear_cache().await;
        assert!(!client.is_cached(prompt, None).await);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_warming() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let warming_config = WarmingConfig::builder()
            .strategy(WarmingStrategy::PredefinedQueries)
            .max_queries(5)
            .delay_between_requests(Duration::from_millis(10))
            .build();

        let client = CachedLLMClient::with_warming(mock_llm, cache_config, warming_config).await?;

        // Cache should be empty initially
        let stats_before = client.cache_statistics();
        assert_eq!(stats_before.current_size, 0);

        // Warm the cache
        client.warm_cache().await?;

        // Cache should now have entries
        let stats_after = client.cache_statistics();
        assert!(stats_after.current_size > 0);
        assert!(stats_after.insertions > 0);

        // Test that warmed queries are now cache hits
        let warmed_query = "What is artificial intelligence?";
        if client.is_cached(warmed_query, None).await {
            let start = Instant::now();
            let _ = client.complete_async(warmed_query).await?;
            let duration = start.elapsed();

            // Should be very fast (cache hit)
            assert!(duration < Duration::from_millis(10));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_different_key_strategies() -> Result<()> {
        let _mock_llm = MockLLM::new()?;

        // Test semantic normalization
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(MockLLM::new()?, cache_config).await?;

        let semantic_generator = CacheKeyGenerator::with_strategy(KeyStrategy::Semantic)
            .normalize_whitespace(true)
            .ignore_case(true);
        client.update_key_strategy(semantic_generator).await;

        // These should generate the same cache key due to normalization
        let response1 = client.complete_async("What is AI?").await?;
        let response2 = client.complete_async("what   is    ai  ?").await?; // Different case/spacing

        let stats = client.cache_statistics();
        assert_eq!(stats.cache_hits, 1); // Second should be a hit
        assert_eq!(response1, response2);

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_cache_access() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::high_performance();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        // Spawn multiple concurrent tasks accessing the cache
        let mut handles = Vec::new();
        for i in 0..10 {
            let client_clone = client.clone();
            let query = format!("Concurrent query {}", i % 3); // Some overlap

            let handle = tokio::spawn(async move {
                client_clone.complete_async(&query).await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap()?;
            results.push(result);
        }

        // Verify no panics or errors occurred
        assert_eq!(results.len(), 10);

        let stats = client.cache_statistics();
        // Should have some cache hits due to overlapping queries
        assert!(stats.cache_hits > 0);
        assert!(stats.total_requests >= 10);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_health_monitoring() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::builder()
            .max_capacity(10)
            .ttl_seconds(300)
            .enable_statistics(true)
            .build();

        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        // Generate some cache activity
        for i in 0..15 {
            let query = format!("Health test query {i}");
            let _ = client.complete_async(&query).await?;
        }

        let health = client.cache_health();

        // Should report high utilization due to small cache size
        assert!(!health.alerts.is_empty() || health.metrics.current_size > 8);

        // Efficiency score should be reasonable
        let efficiency = health.metrics.efficiency_score();
        assert!((0.0..=1.0).contains(&efficiency));

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_statistics_accuracy() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        let queries = vec![
            "Query 1",
            "Query 2",
            "Query 1", // Repeat
            "Query 3",
            "Query 2", // Repeat
            "Query 1", // Repeat
        ];

        for query in &queries {
            let _ = client.complete_async(query).await?;
        }

        let stats = client.cache_statistics();

        // 6 total requests, 3 unique queries, 3 repeats (hits)
        assert_eq!(stats.total_requests, 6);
        assert_eq!(stats.cache_misses, 3); // Unique queries
        assert_eq!(stats.cache_hits, 3);   // Repeated queries
        assert_eq!(stats.current_size, 3); // Unique entries

        // Hit rate should be 50%
        assert!((stats.hit_rate - 0.5).abs() < 0.01);

        // Time saved should be recorded for hits
        assert!(stats.total_time_saved > Duration::ZERO);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        let initial_stats = client.cache_statistics();
        let initial_memory = initial_stats.memory_usage_bytes;

        // Add some entries
        for i in 0..5 {
            let query = format!("Memory test query {i}");
            let _ = client.complete_async(&query).await?;
        }

        let final_stats = client.cache_statistics();
        let final_memory = final_stats.memory_usage_bytes;

        // Memory usage should have increased
        assert!(final_memory > initial_memory);

        // Should have a reasonable memory usage report
        let memory_human = final_stats.memory_usage_human;
        assert!(memory_human.contains("B") || memory_human.contains("KB"));

        Ok(())
    }

    #[tokio::test]
    async fn test_cost_savings_calculation() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        // Simulate repeated queries (common in real usage)
        let queries = vec!["Popular query"; 10]; // Same query 10 times

        for query in &queries {
            let _ = client.complete_async(query).await?;
        }

        let stats = client.cache_statistics();

        // Should be 1 miss + 9 hits
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 9);

        // Calculate cost savings
        let cost_per_call = 0.002; // $0.002 per LLM call
        let savings = stats.cost_savings(cost_per_call);

        // Savings should be 9 * $0.002 = $0.018
        assert!((savings - 0.018).abs() < 0.001);

        Ok(())
    }

    #[tokio::test]
    async fn test_language_model_trait_compatibility() -> Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await?;

        // Test sync LanguageModel trait methods
        assert!(client.is_available());

        let model_info = client.model_info();
        assert!(model_info.name.contains("Cached"));

        // Test complete method
        let response = client.complete("Test prompt")?;
        assert!(!response.is_empty());

        // Test complete_with_params method
        let params = GenerationParams::default();
        let response_with_params = client.complete_with_params("Test prompt", params)?;
        assert!(!response_with_params.is_empty());

        // Since we used the same prompt, should have cache hit
        let stats = client.cache_statistics();
        assert!(stats.cache_hits > 0);

        Ok(())
    }
}

#[cfg(not(feature = "caching"))]
mod disabled_caching_tests {
    #[test]
    fn caching_feature_disabled() {
        // This test ensures the module compiles even when caching is disabled
        println!("Caching feature is disabled");
    }
}