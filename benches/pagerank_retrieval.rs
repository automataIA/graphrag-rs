use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graphrag_rs::{
    core::{ChunkId, DocumentId, Entity, EntityId, KnowledgeGraph, TextChunk},
    Config, PageRankConfig, PageRankRetrievalSystem,
};
use std::collections::HashMap;

/// Create a test knowledge graph with varying sizes for benchmarking
fn create_test_graph(num_entities: usize, num_connections: usize) -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::new();

    // Add entities
    for i in 0..num_entities {
        let entity = Entity::new(
            EntityId::new(format!("entity_{i}")),
            format!("Entity {i}"),
            if i % 3 == 0 {
                "PERSON"
            } else if i % 3 == 1 {
                "ORGANIZATION"
            } else {
                "LOCATION"
            }
            .to_string(),
            0.8 + (i as f32 * 0.01) % 0.2, // Vary confidence between 0.8-1.0
        );
        graph.add_entity(entity).unwrap();
    }

    // Add relationships between entities
    for i in 0..num_connections.min(num_entities * (num_entities - 1) / 2) {
        let source_idx = i % num_entities;
        let target_idx = (i + 1 + source_idx) % num_entities;

        if source_idx != target_idx {
            let relationship = graphrag_rs::core::Relationship {
                source: EntityId::new(format!("entity_{source_idx}")),
                target: EntityId::new(format!("entity_{target_idx}")),
                relation_type: "RELATED_TO".to_string(),
                confidence: 0.7 + (i as f32 * 0.01) % 0.3,
                context: vec![],
            };
            graph.add_relationship(relationship).unwrap();
        }
    }

    // Add chunks that reference the entities
    for i in 0..(num_entities / 3) {
        let chunk = TextChunk::new(
            ChunkId::new(format!("chunk_{i}")),
            DocumentId::new(format!("doc_{}", i / 10)),
            format!(
                "This is a test chunk {} that mentions Entity {} and Entity {}.",
                i,
                i * 3 % num_entities,
                (i * 3 + 1) % num_entities
            ),
            i * 100,
            (i + 1) * 100,
        )
        .with_entities(vec![
            EntityId::new(format!("entity_{}", i * 3 % num_entities)),
            EntityId::new(format!("entity_{}", (i * 3 + 1) % num_entities)),
        ]);

        graph.add_chunk(chunk).unwrap();
    }

    graph
}

/// Benchmark different PageRank configurations
fn bench_pagerank_configs(c: &mut Criterion) {
    let graph = create_test_graph(1000, 2000);

    let configs = vec![
        (
            "Sequential",
            PageRankConfig {
                parallel_enabled: false,
                cache_size: 100,
                sparse_threshold: 10000, // Force sparse
                ..PageRankConfig::default()
            },
        ),
        (
            "Parallel",
            PageRankConfig {
                parallel_enabled: true,
                cache_size: 100,
                sparse_threshold: 10000, // Force sparse
                ..PageRankConfig::default()
            },
        ),
        (
            "Dense_Optimized",
            PageRankConfig {
                parallel_enabled: false,
                cache_size: 1000,
                sparse_threshold: 500, // Force dense for smaller graphs
                ..PageRankConfig::default()
            },
        ),
        (
            "Cached_Parallel",
            PageRankConfig {
                parallel_enabled: true,
                cache_size: 2000,
                sparse_threshold: 500,
                ..PageRankConfig::default()
            },
        ),
    ];

    let mut group = c.benchmark_group("pagerank_configurations");

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("pagerank_calculation", name),
            &config,
            |b, _config| {
                let pagerank_calc = graph.build_pagerank_calculator().unwrap();
                let reset_probs = HashMap::new();

                b.iter(|| black_box(pagerank_calc.calculate_scores(&reset_probs).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark retrieval system performance with different graph sizes
fn bench_retrieval_scalability(c: &mut Criterion) {
    let graph_sizes = vec![100, 500, 1000, 2500];

    let mut group = c.benchmark_group("retrieval_scalability");

    for &size in &graph_sizes {
        let graph = create_test_graph(size, size * 2);

        group.bench_with_input(
            BenchmarkId::new("pagerank_retrieval", size),
            &size,
            |b, _| {
                let mut retrieval_system =
                    PageRankRetrievalSystem::new(10).with_pagerank_config(PageRankConfig {
                        parallel_enabled: true,
                        cache_size: 1000,
                        max_iterations: 30, // Reduced for benchmarking
                        tolerance: 1e-4,    // Slightly relaxed
                        ..PageRankConfig::default()
                    });

                retrieval_system.initialize_vector_index(&graph).unwrap();

                b.iter(|| {
                    black_box(
                        retrieval_system
                            .search_with_pagerank("Entity 42", &graph, Some(10))
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch query performance
fn bench_batch_queries(c: &mut Criterion) {
    let graph = create_test_graph(1000, 2000);
    let mut retrieval_system =
        PageRankRetrievalSystem::new(10).with_pagerank_config(PageRankConfig {
            parallel_enabled: true,
            cache_size: 2000,
            ..PageRankConfig::default()
        });

    retrieval_system.initialize_vector_index(&graph).unwrap();

    let queries = ["Entity 100",
        "Entity 200",
        "Entity 300",
        "Entity 400",
        "Entity 500",
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "related entities",
        "test query"];

    let batch_sizes = vec![1, 5, 10, 20];

    let mut group = c.benchmark_group("batch_queries");

    for &batch_size in &batch_sizes {
        let batch_queries: Vec<&str> = queries.iter().cycle().take(batch_size).cloned().collect();

        group.bench_with_input(
            BenchmarkId::new("batch_search", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        retrieval_system
                            .batch_search(&batch_queries, &graph, Some(5))
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark comparison: Traditional vs PageRank retrieval
fn bench_retrieval_comparison(c: &mut Criterion) {
    let graph = create_test_graph(1000, 1500);

    // Traditional retrieval setup
    let config = Config::default();
    let mut traditional_retrieval = graphrag_rs::retrieval::RetrievalSystem::new(&config).unwrap();
    traditional_retrieval.index_graph(&graph).unwrap();

    // PageRank retrieval setup
    let mut pagerank_retrieval =
        PageRankRetrievalSystem::new(10).with_pagerank_config(PageRankConfig {
            parallel_enabled: true,
            cache_size: 2000,
            max_iterations: 25,
            ..PageRankConfig::default()
        });
    pagerank_retrieval.initialize_vector_index(&graph).unwrap();

    let test_queries = vec![
        "Entity 50",
        "ORGANIZATION related",
        "Entity 100 Entity 200",
        "test query",
    ];

    let mut group = c.benchmark_group("retrieval_comparison");

    for query in &test_queries {
        // Benchmark traditional retrieval
        group.bench_with_input(
            BenchmarkId::new("traditional", query),
            query,
            |b, &query| {
                b.iter(|| black_box(traditional_retrieval.hybrid_query(query, &graph).unwrap()));
            },
        );

        // Benchmark PageRank retrieval
        group.bench_with_input(BenchmarkId::new("pagerank", query), query, |b, &query| {
            b.iter(|| {
                black_box(
                    pagerank_retrieval
                        .search_with_pagerank(query, &graph, Some(10))
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage and cache effectiveness
fn bench_cache_performance(c: &mut Criterion) {
    let graph = create_test_graph(500, 1000);

    let cache_sizes = vec![100, 500, 1000, 2000];

    let mut group = c.benchmark_group("cache_performance");

    for &cache_size in &cache_sizes {
        group.bench_with_input(
            BenchmarkId::new("cached_retrieval", cache_size),
            &cache_size,
            |b, &cache_size| {
                let mut retrieval_system =
                    PageRankRetrievalSystem::new(10).with_pagerank_config(PageRankConfig {
                        cache_size,
                        parallel_enabled: true,
                        ..PageRankConfig::default()
                    });

                retrieval_system.initialize_vector_index(&graph).unwrap();

                b.iter(|| {
                    // Query the same entity multiple times to test cache hits
                    for i in 0..5 {
                        black_box(
                            retrieval_system
                                .search_with_pagerank(
                                    &format!("Entity {}", i * 10),
                                    &graph,
                                    Some(5),
                                )
                                .unwrap(),
                        );
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pagerank_configs,
    bench_retrieval_scalability,
    bench_batch_queries,
    bench_retrieval_comparison,
    bench_cache_performance
);

criterion_main!(benches);
