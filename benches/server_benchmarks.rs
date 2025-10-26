//! Performance benchmarks for GraphRAG server operations
//!
//! These benchmarks measure the performance of core server operations:
//! - Document upload and processing
//! - Knowledge graph building
//! - Query execution
//! - Entity retrieval

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use graphrag_rs::{GraphRAG, Config};

/// Benchmark document upload
fn bench_document_upload(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_upload");

    let documents = [
        "This is a test document about artificial intelligence and machine learning.",
        "GraphRAG is a system that combines graph databases with retrieval augmented generation.",
        "The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
    ];

    for (i, doc) in documents.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::from_parameter(i),
            doc,
            |b, &doc| {
                b.iter(|| {
                    let config = Config::default();
                    let mut graphrag = GraphRAG::new(config).unwrap();
                    graphrag.initialize().unwrap();
                    graphrag.add_document_from_text(doc).unwrap();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark graph building
fn bench_graph_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build");

    let test_cases = vec![
        ("small", 1, "Short test document."),
        ("medium", 3, "This is a longer document with more content about various topics."),
        ("large", 5, "This is a comprehensive document discussing multiple entities and relationships in detail."),
    ];

    for (name, doc_count, content) in test_cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(doc_count, content),
            |b, &(count, text)| {
                b.iter(|| {
                    let config = Config::default();
                    let mut graphrag = GraphRAG::new(config).unwrap();
                    graphrag.initialize().unwrap();

                    // Add documents
                    for _ in 0..count {
                        graphrag.add_document_from_text(text).unwrap();
                    }

                    // Build graph
                    graphrag.build_graph().unwrap();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark query execution
fn bench_query_execution(c: &mut Criterion) {
    // Set up GraphRAG with some data
    let config = Config::default();
    let mut graphrag = GraphRAG::new(config).unwrap();
    graphrag.initialize().unwrap();

    graphrag.add_document_from_text(
        "GraphRAG is a powerful system for knowledge graph construction and semantic search."
    ).unwrap();
    graphrag.build_graph().unwrap();

    let mut group = c.benchmark_group("query_execution");

    let queries = [
        "What is GraphRAG?",
        "How does semantic search work?",
        "Explain knowledge graphs",
    ];

    for (i, query) in queries.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::from_parameter(i),
            query,
            |b, &q| {
                b.iter(|| {
                    black_box(graphrag.query(q).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark entity retrieval
fn bench_entity_retrieval(c: &mut Criterion) {
    // Set up GraphRAG with some data
    let config = Config::default();
    let mut graphrag = GraphRAG::new(config).unwrap();
    graphrag.initialize().unwrap();

    graphrag.add_document_from_text(
        "Alice works at TechCorp. Bob is the CEO of DataSystems. Charlie collaborates with Alice on AI projects."
    ).unwrap();
    graphrag.build_graph().unwrap();

    c.bench_function("entity_retrieval", |b| {
        b.iter(|| {
            if let Some(graph) = graphrag.get_knowledge_graph() {
                black_box(graph.entity_count());
                black_box(graph.relationship_count());
            }
        });
    });
}

/// Benchmark concurrent queries
fn bench_concurrent_queries(c: &mut Criterion) {
    use std::sync::Arc;
    use parking_lot::RwLock;

    // Set up GraphRAG with some data
    let config = Config::default();
    let mut graphrag = GraphRAG::new(config).unwrap();
    graphrag.initialize().unwrap();

    graphrag.add_document_from_text(
        "Concurrent processing enables multiple queries to be executed simultaneously."
    ).unwrap();
    graphrag.build_graph().unwrap();

    let graphrag = Arc::new(RwLock::new(graphrag));

    let mut group = c.benchmark_group("concurrent_queries");

    for thread_count in [1, 2, 4].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(thread_count),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    use std::thread;

                    let handles: Vec<_> = (0..threads)
                        .map(|i| {
                            let graphrag = Arc::clone(&graphrag);
                            thread::spawn(move || {
                                let g = graphrag.read();
                                black_box(g.query(&format!("Query {}", i)).unwrap());
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_document_upload,
    bench_graph_build,
    bench_query_execution,
    bench_entity_retrieval,
    bench_concurrent_queries
);

criterion_main!(benches);
