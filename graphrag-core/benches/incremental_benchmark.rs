//! Benchmark: Incremental Updates vs Full Rebuild
//!
//! Compares performance between:
//! 1. Incremental approach with lazy propagation, delta computation, async batching
//! 2. Full rebuild approach (traditional method)
//!
//! Expected results:
//! - Incremental: 10x faster for updates
//! - 80-90% reduction in operations
//! - 1000+ docs/sec throughput

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graphrag_core::incremental::{
    IncrementalGraphManager, IncrementalConfig, GraphNode, NodeType,
    ConflictResolution,
};
use std::collections::HashMap;

/// Generate a test document with realistic content
fn generate_test_node(id: usize) -> GraphNode {
    GraphNode {
        id: format!("node_{}", id),
        label: format!("Test Node {}", id),
        node_type: NodeType::Entity,
        attributes: HashMap::from([
            ("type".to_string(), "test_entity".to_string()),
            ("index".to_string(), id.to_string()),
            ("description".to_string(), format!("This is test node number {}", id)),
            ("category".to_string(), (id % 10).to_string()),
        ]),
        embeddings: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        version: 1,
    }
}

/// Benchmark: Incremental approach with all optimizations
fn bench_incremental_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_updates");

    for num_docs in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("incremental", num_docs),
            num_docs,
            |b, &num_docs| {
                b.iter(|| {
                    // Config with all optimizations enabled
                    let config = IncrementalConfig {
                        auto_detect_changes: true,
                        min_entity_confidence: 0.7,
                        max_batch_size: 1000,
                        parallel_updates: true,
                        conflict_resolution: ConflictResolution::LatestWins,
                        enable_lazy_propagation: true,
                        lazy_propagation_threshold: 100,
                        enable_delta_computation: true,
                        delta_use_bloom_filter: true,
                    };

                    let mut manager = IncrementalGraphManager::new(config);

                    // Add initial batch (50% of total)
                    let initial_batch = num_docs / 2;
                    for i in 0..initial_batch {
                        let node = generate_test_node(i);
                        let _ = manager.add_node(node);
                    }

                    // Create snapshot for delta computation
                    manager.update_snapshot();

                    // Add remaining nodes incrementally (this is what we measure)
                    for i in initial_batch..num_docs {
                        let node = generate_test_node(i);
                        let _ = black_box(manager.add_node(node));
                    }

                    // Force propagate to ensure all updates are applied
                    let _ = manager.force_propagate_updates();

                    manager
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Full rebuild approach (no optimizations)
fn bench_full_rebuild(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_rebuild");

    for num_docs in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("rebuild", num_docs),
            num_docs,
            |b, &num_docs| {
                b.iter(|| {
                    // Config with all optimizations disabled
                    let config = IncrementalConfig {
                        auto_detect_changes: false, // No change detection
                        min_entity_confidence: 0.7,
                        max_batch_size: 1000,
                        parallel_updates: false, // Sequential
                        conflict_resolution: ConflictResolution::LatestWins,
                        enable_lazy_propagation: false, // No lazy propagation
                        lazy_propagation_threshold: 0,
                        enable_delta_computation: false, // No delta computation
                        delta_use_bloom_filter: false,
                    };

                    let mut manager = IncrementalGraphManager::new(config);

                    // Simulate full rebuild: add all nodes sequentially
                    for i in 0..num_docs {
                        let node = generate_test_node(i);
                        let _ = black_box(manager.add_node(node));
                    }

                    manager
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Lazy propagation effectiveness
fn bench_lazy_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_propagation");

    // Test with lazy propagation enabled
    group.bench_function("with_lazy_propagation", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                enable_lazy_propagation: true,
                lazy_propagation_threshold: 50,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            // Add 100 nodes - propagation should be deferred
            for i in 0..100 {
                let node = generate_test_node(i);
                let _ = manager.add_node(node);
            }

            // Force propagate once at the end
            let _ = black_box(manager.force_propagate_updates());
        });
    });

    // Test without lazy propagation (immediate propagation)
    group.bench_function("without_lazy_propagation", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                enable_lazy_propagation: false,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            // Add 100 nodes - each triggers immediate propagation
            for i in 0..100 {
                let node = generate_test_node(i);
                let _ = black_box(manager.add_node(node));
            }
        });
    });

    group.finish();
}

/// Benchmark: Delta computation effectiveness
fn bench_delta_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_computation");

    // Test with delta computation and bloom filters
    group.bench_function("with_delta_bloom", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                enable_delta_computation: true,
                delta_use_bloom_filter: true,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            // Add initial batch
            for i in 0..50 {
                let node = generate_test_node(i);
                let _ = manager.add_node(node);
            }

            // Create snapshot
            manager.update_snapshot();

            // Add more nodes
            for i in 50..100 {
                let node = generate_test_node(i);
                let _ = manager.add_node(node);
            }

            // Compute delta (only changed nodes)
            let _ = black_box(manager.compute_delta_since_last_snapshot());
        });
    });

    // Test without delta computation
    group.bench_function("without_delta", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                enable_delta_computation: false,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            // Add all nodes
            for i in 0..100 {
                let node = generate_test_node(i);
                let _ = black_box(manager.add_node(node));
            }

            // Without delta, we'd need to recompute everything
            let _ = manager.create_snapshot();
        });
    });

    group.finish();
}

/// Benchmark: Parallel vs Sequential updates
fn bench_parallel_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_updates");

    // Parallel updates
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                parallel_updates: true,
                max_batch_size: 1000,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            for i in 0..100 {
                let node = generate_test_node(i);
                let _ = black_box(manager.add_node(node));
            }
        });
    });

    // Sequential updates
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                parallel_updates: false,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            for i in 0..100 {
                let node = generate_test_node(i);
                let _ = black_box(manager.add_node(node));
            }
        });
    });

    group.finish();
}

/// Benchmark: Snapshot creation performance
fn bench_snapshot_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_creation");

    for num_nodes in [50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("snapshot", num_nodes),
            num_nodes,
            |b, &num_nodes| {
                // Setup: create manager with nodes
                let config = IncrementalConfig::default();
                let mut manager = IncrementalGraphManager::new(config);

                for i in 0..num_nodes {
                    let node = generate_test_node(i);
                    let _ = manager.add_node(node);
                }

                // Benchmark snapshot creation
                b.iter(|| {
                    let snapshot = black_box(manager.create_snapshot());
                    snapshot
                });
            },
        );
    }

    group.finish();
}

/// Summary benchmark: Complete workflow comparison
fn bench_complete_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_workflow");

    // Fully optimized workflow
    group.bench_function("optimized", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                auto_detect_changes: true,
                enable_lazy_propagation: true,
                lazy_propagation_threshold: 50,
                enable_delta_computation: true,
                delta_use_bloom_filter: true,
                parallel_updates: true,
                max_batch_size: 1000,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            // Initial batch
            for i in 0..50 {
                let _ = manager.add_node(generate_test_node(i));
            }

            manager.update_snapshot();

            // Incremental updates
            for i in 50..100 {
                let _ = manager.add_node(generate_test_node(i));
            }

            let _ = manager.force_propagate_updates();
            let _ = black_box(manager.compute_delta_since_last_snapshot());
        });
    });

    // Unoptimized workflow
    group.bench_function("unoptimized", |b| {
        b.iter(|| {
            let config = IncrementalConfig {
                auto_detect_changes: false,
                enable_lazy_propagation: false,
                enable_delta_computation: false,
                parallel_updates: false,
                ..Default::default()
            };

            let mut manager = IncrementalGraphManager::new(config);

            // All updates sequentially
            for i in 0..100 {
                let _ = black_box(manager.add_node(generate_test_node(i)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_incremental_updates,
    bench_full_rebuild,
    bench_lazy_propagation,
    bench_delta_computation,
    bench_parallel_updates,
    bench_snapshot_creation,
    bench_complete_workflow,
);

criterion_main!(benches);
