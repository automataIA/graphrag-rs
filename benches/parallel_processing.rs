use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use graphrag_rs::{
    config::ParallelConfig,
    core::{ChunkId, TextChunk, DocumentId},
    parallel::{ParallelProcessor, ItemComplexity, ProcessingStrategy},
    vector::{EmbeddingGenerator, VectorIndex},
    summarization::{DocumentTree, HierarchicalConfig},
};

#[allow(dead_code)]
fn create_test_chunks(count: usize) -> Vec<TextChunk> {
    (0..count)
        .map(|i| TextChunk {
            id: ChunkId::new(format!("chunk_{i}")),
            document_id: DocumentId::new("test_doc".to_string()),
            content: format!(
                "This is test content for chunk {i}. It contains multiple sentences to test summarization. \
                The content is designed to be realistic and provide meaningful test data for parallel processing benchmarks."
            ),
            start_offset: i * 100,
            end_offset: (i + 1) * 100,
            entities: Vec::new(),
            embedding: None,
        })
        .collect()
}

fn create_test_texts(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!(
            "Test document {i} with meaningful content for embedding generation and processing"
        ))
        .collect()
}

fn bench_parallel_embedding_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_embedding_generation");

    let parallel_config = ParallelConfig {
        enabled: true,
        num_threads: 0, // Auto-detect
        min_batch_size: 10,
        chunk_batch_size: 50,
        parallel_embeddings: true,
        parallel_graph_ops: true,
        parallel_vector_ops: true,
    };

    let mut processor = ParallelProcessor::new(parallel_config);
    processor.initialize().expect("Failed to initialize parallel processor");

    for size in [50, 100, 500, 1000].iter() {
        let texts = create_test_texts(*size);
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Sequential benchmark
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &text_refs,
            |b, texts| {
                let mut generator = EmbeddingGenerator::new(128);
                b.iter(|| {
                    for text in black_box(texts.iter()) {
                        black_box(generator.generate_embedding(text));
                    }
                });
            },
        );

        // Parallel benchmark
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &text_refs,
            |b, texts| {
                let mut generator = EmbeddingGenerator::new(128);
                b.iter(|| {
                    black_box(generator.batch_generate(black_box(texts)));
                });
            },
        );

        // Chunked parallel benchmark
        group.bench_with_input(
            BenchmarkId::new("chunked_parallel", size),
            &text_refs,
            |b, texts| {
                let mut generator = EmbeddingGenerator::new(128);
                b.iter(|| {
                    black_box(generator.batch_generate_chunked(black_box(texts), 50));
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vector_operations");

    let parallel_config = ParallelConfig {
        enabled: true,
        num_threads: 0,
        min_batch_size: 10,
        chunk_batch_size: 100,
        parallel_embeddings: true,
        parallel_graph_ops: true,
        parallel_vector_ops: true,
    };

    let processor = ParallelProcessor::new(parallel_config);

    for size in [100, 500, 1000, 2000].iter() {
        let vectors: Vec<(String, Vec<f32>)> = (0..*size)
            .map(|i| {
                let vec: Vec<f32> = (0..128).map(|j| (i + j) as f32 / 100.0).collect();
                (format!("vec_{i}"), vec)
            })
            .collect();

        // Sequential batch add
        group.bench_with_input(
            BenchmarkId::new("batch_add_sequential", size),
            &vectors,
            |b, vecs| {
                b.iter(|| {
                    let mut index = VectorIndex::new();
                    for (id, embedding) in black_box(vecs.iter().cloned()) {
                        index.add_vector(id, embedding).unwrap();
                        black_box(());
                    }
                });
            },
        );

        // Parallel batch add
        group.bench_with_input(
            BenchmarkId::new("batch_add_parallel", size),
            &vectors,
            |b, vecs| {
                b.iter(|| {
                    let mut index = VectorIndex::with_parallel_processing(processor.clone());
                    index.batch_add_vectors(black_box(vecs.clone())).unwrap();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_similarity_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_similarity_computation");

    let parallel_config = ParallelConfig {
        enabled: true,
        num_threads: 0,
        min_batch_size: 10,
        chunk_batch_size: 100,
        parallel_embeddings: true,
        parallel_graph_ops: true,
        parallel_vector_ops: true,
    };

    let processor = ParallelProcessor::new(parallel_config);

    for size in [50, 100, 200].iter() {
        let mut index_sequential = VectorIndex::new();
        let mut index_parallel = VectorIndex::with_parallel_processing(processor.clone());

        // Setup test data
        for i in 0..*size {
            let vec: Vec<f32> = (0..128).map(|j| (i + j) as f32 / 100.0).collect();
            let id = format!("vec_{i}");
            index_sequential.add_vector(id.clone(), vec.clone()).unwrap();
            index_parallel.add_vector(id, vec).unwrap();
        }

        // Sequential similarity computation
        group.bench_with_input(
            BenchmarkId::new("similarity_sequential", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(index_sequential.compute_all_similarities());
                });
            },
        );

        // Parallel similarity computation
        group.bench_with_input(
            BenchmarkId::new("similarity_parallel", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(index_parallel.compute_all_similarities());
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_parallel_node_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_node_creation");

    let config = HierarchicalConfig::default();
    let doc_id = DocumentId::new("test_doc".to_string());

    for size in [20, 50, 100, 200].iter() {
        let chunks = create_test_chunks(*size);

        // Sequential node creation
        group.bench_with_input(
            BenchmarkId::new("node_creation_sequential", size),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    let mut tree = DocumentTree::new(doc_id.clone(), config.clone()).unwrap();
                    tree.build_from_chunks(black_box(chunks.clone())).unwrap();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn bench_processing_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("processing_strategies");

    let parallel_config = ParallelConfig {
        enabled: true,
        num_threads: 0,
        min_batch_size: 10,
        chunk_batch_size: 50,
        parallel_embeddings: true,
        parallel_graph_ops: true,
        parallel_vector_ops: true,
    };

    let mut processor = ParallelProcessor::new(parallel_config);
    processor.initialize().unwrap();

    let items: Vec<i32> = (0..1000).collect();
    let operation = |x: i32| -> i32 { x * x + x + 1 };

    for strategy in [
        ProcessingStrategy::Sequential,
        ProcessingStrategy::ParallelMap,
        ProcessingStrategy::ChunkedParallel,
        ProcessingStrategy::WorkStealing,
    ].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{strategy:?}")),
            strategy,
            |b, strategy| {
                b.iter(|| {
                    black_box(processor.execute_with_strategy(
                        black_box(items.clone()),
                        black_box(operation),
                        black_box(*strategy),
                    ));
                });
            },
        );
    }

    // Benchmark adaptive strategy selection
    for complexity in [ItemComplexity::Low, ItemComplexity::Medium, ItemComplexity::High].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("adaptive_{complexity:?}")),
            complexity,
            |b, complexity| {
                b.iter(|| {
                    black_box(processor.execute_adaptive(
                        black_box(items.clone()),
                        black_box(operation),
                        black_box(*complexity),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_embedding_generation,
    bench_parallel_vector_operations,
    bench_parallel_similarity_computation,
    bench_processing_strategies
);

criterion_main!(benches);