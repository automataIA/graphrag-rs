//! Simple test for caching functionality

#[cfg(feature = "caching")]
mod test_caching {
    #[allow(unused_imports)]
    use crate::{MockLLM, CacheConfig, CachedLLMClient};

    #[tokio::test]
    async fn test_basic_caching() -> crate::core::Result<()> {
        let mock_llm = MockLLM::new()?;
        let cache_config = CacheConfig::development();
        let client = CachedLLMClient::new(mock_llm, cache_config).await
            .map_err(|e| crate::core::GraphRAGError::Generation {
                message: format!("Cache error: {e:?}"),
            })?;

        let prompt = "What is AI?";

        // First call - cache miss
        let response1 = client.complete_async(prompt).await?;
        let stats1 = client.cache_statistics();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);

        // Second call - cache hit
        let response2 = client.complete_async(prompt).await?;
        let stats2 = client.cache_statistics();
        assert_eq!(stats2.cache_misses, 1);
        assert_eq!(stats2.cache_hits, 1);

        assert_eq!(response1, response2);
        Ok(())
    }
}

#[cfg(not(feature = "caching"))]
mod test_no_caching {
    #[test]
    fn caching_disabled() {
        println!("Caching feature is disabled");
    }
}