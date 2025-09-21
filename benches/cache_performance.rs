//! Benchmark comparing cached vs non-cached LLM performance

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use graphrag_rs::{CacheConfig, CachedLLMClient, EvictionPolicy, LanguageModel, MockLLM};
use tokio::runtime::Runtime;

/// Test queries for benchmarking
const TEST_QUERIES: &[&str] = &[
    "What is artificial intelligence?",
    "Explain machine learning algorithms",
    "How do neural networks work?",
    "What is deep learning?",
    "Define natural language processing",
    "What is computer vision?",
    "Explain reinforcement learning",
    "What is data science?",
    "Define big data analytics",
    "What is cloud computing?",
];

fn benchmark_uncached_llm(c: &mut Criterion) {
    let _rt = Runtime::new().unwrap();
    let mock_llm = MockLLM::new().unwrap();

    let mut group = c.benchmark_group("uncached_llm");
    group.throughput(Throughput::Elements(TEST_QUERIES.len() as u64));

    group.bench_function("sequential_queries", |b| {
        b.iter(|| {
            for &query in TEST_QUERIES {
                let _ = mock_llm.complete(query).unwrap();
            }
        })
    });

    group.bench_function("repeated_queries", |b| {
        b.iter(|| {
            // Repeat the same query multiple times (simulates real-world repetition)
            for _ in 0..10 {
                let _ = mock_llm.complete(TEST_QUERIES[0]).unwrap();
            }
        })
    });

    group.finish();
}

fn benchmark_cached_llm(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("cached_llm");
    group.throughput(Throughput::Elements(TEST_QUERIES.len() as u64));

    // Test with different cache configurations
    let configs = vec![
        ("development", CacheConfig::development()),
        ("production", CacheConfig::production()),
        ("high_performance", CacheConfig::high_performance()),
    ];

    for (config_name, cache_config) in configs {
        group.bench_with_input(
            BenchmarkId::new("sequential_queries", config_name),
            &cache_config,
            |b, config| {
                b.iter(|| {
                    rt.block_on(async {
                        let mock_llm = MockLLM::new().unwrap();
                        let cached_client = CachedLLMClient::new(mock_llm, config.clone())
                            .await
                            .unwrap();

                        for &query in TEST_QUERIES {
                            let _ = cached_client.complete_async(query).await.unwrap();
                        }
                    })
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("repeated_queries", config_name),
            &cache_config,
            |b, config| {
                b.iter(|| {
                    rt.block_on(async {
                        let mock_llm = MockLLM::new().unwrap();
                        let cached_client = CachedLLMClient::new(mock_llm, config.clone())
                            .await
                            .unwrap();

                        // Repeat the same query multiple times
                        for _ in 0..10 {
                            let _ = cached_client.complete_async(TEST_QUERIES[0]).await.unwrap();
                        }
                    })
                })
            },
        );
    }

    group.finish();
}

fn benchmark_cache_hit_rate(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("cache_hit_rate");

    // Pre-warm cache for pure hit rate testing
    group.bench_function("pure_cache_hits", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mock_llm = MockLLM::new().unwrap();
                let cache_config = CacheConfig::high_performance();
                let cached_client = CachedLLMClient::new(mock_llm, cache_config).await.unwrap();

                // Warm the cache first
                for &query in TEST_QUERIES {
                    let _ = cached_client.complete_async(query).await.unwrap();
                }

                // Now measure pure cache hits
                for &query in TEST_QUERIES {
                    let _ = cached_client.complete_async(query).await.unwrap();
                }
            })
        })
    });

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_usage");

    // Test memory efficiency with different cache sizes
    let cache_sizes = vec![100, 1000, 10000, 100000];

    for cache_size in cache_sizes {
        group.bench_with_input(
            BenchmarkId::new("memory_per_entry", cache_size),
            &cache_size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let mock_llm = MockLLM::new().unwrap();
                        let cache_config = CacheConfig::builder()
                            .max_capacity(size)
                            .ttl_seconds(3600)
                            .eviction_policy(EvictionPolicy::LRU)
                            .build();

                        let cached_client =
                            CachedLLMClient::new(mock_llm, cache_config).await.unwrap();

                        // Fill cache with varied queries
                        for i in 0..size.min(1000) {
                            let query = format!("Test query number {i}");
                            let _ = cached_client.complete_async(&query).await.unwrap();
                        }

                        let stats = cached_client.cache_statistics();
                        stats.memory_usage_bytes
                    })
                })
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_access");

    group.bench_function("concurrent_cache_access", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mock_llm = MockLLM::new().unwrap();
                let cache_config = CacheConfig::high_performance();
                let cached_client = CachedLLMClient::new(mock_llm, cache_config).await.unwrap();

                // Spawn multiple concurrent tasks
                let mut handles = Vec::new();
                for i in 0..10 {
                    let client = cached_client.clone();
                    let query = TEST_QUERIES[i % TEST_QUERIES.len()].to_string();

                    let handle = tokio::spawn(async move { client.complete_async(&query).await });
                    handles.push(handle);
                }

                // Wait for all tasks to complete
                for handle in handles {
                    let _ = handle.await.unwrap().unwrap();
                }
            })
        })
    });

    group.finish();
}

fn benchmark_eviction_policies(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("eviction_policies");

    let policies = vec![
        ("LRU", EvictionPolicy::LRU),
        ("LFU", EvictionPolicy::LFU),
        ("TTL", EvictionPolicy::TTL),
        ("Adaptive", EvictionPolicy::Adaptive),
    ];

    for (policy_name, policy) in policies {
        group.bench_with_input(
            BenchmarkId::new("policy_performance", policy_name),
            &policy,
            |b, &pol| {
                b.iter(|| {
                    rt.block_on(async {
                        let mock_llm = MockLLM::new().unwrap();
                        let cache_config = CacheConfig::builder()
                            .max_capacity(100) // Small cache to trigger evictions
                            .ttl_seconds(300)
                            .eviction_policy(pol)
                            .build();

                        let cached_client =
                            CachedLLMClient::new(mock_llm, cache_config).await.unwrap();

                        // Generate enough queries to trigger evictions
                        for i in 0..200 {
                            let query = format!("Query {i}");
                            let _ = cached_client.complete_async(&query).await.unwrap();
                        }
                    })
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_uncached_llm,
    benchmark_cached_llm,
    benchmark_cache_hit_rate,
    benchmark_memory_usage,
    benchmark_concurrent_access,
    benchmark_eviction_policies
);
criterion_main!(benches);
