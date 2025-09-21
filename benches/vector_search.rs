use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graphrag_rs::vector::{VectorIndex, VectorUtils};

fn benchmark_vector_index_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_index");

    // Different index sizes
    let sizes = vec![100, 1000, 10000];
    let dimension = 384; // Common embedding dimension

    for size in sizes {
        // Benchmark index building
        group.bench_with_input(
            BenchmarkId::new("build_index", format!("{size}_vectors")),
            &size,
            |b, &size| {
                b.iter_batched(
                    || create_test_vectors(size, dimension),
                    |mut index| {
                        index.build_index().unwrap();
                        black_box(())
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        // Benchmark search operations
        let mut index = create_test_vectors(size, dimension);
        index.build_index().unwrap();
        let query = VectorUtils::random_vector(dimension);

        group.bench_with_input(
            BenchmarkId::new("search", format!("{size}_vectors")),
            &(&index, &query),
            |b, (index, query)| b.iter(|| black_box(index.search(query, 10).unwrap())),
        );

        // Benchmark similarity threshold search
        group.bench_with_input(
            BenchmarkId::new("find_similar", format!("{size}_vectors")),
            &(&index, &query),
            |b, (index, query)| b.iter(|| black_box(index.find_similar(query, 0.8).unwrap())),
        );
    }

    group.finish();
}

fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    let dimensions = vec![128, 384, 768, 1536]; // Common embedding dimensions

    for dim in dimensions {
        let vec_a = VectorUtils::random_vector(dim);
        let vec_b = VectorUtils::random_vector(dim);

        // Benchmark cosine similarity
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", format!("dim_{dim}")),
            &(&vec_a, &vec_b),
            |b, (vec_a, vec_b)| b.iter(|| black_box(VectorUtils::cosine_similarity(vec_a, vec_b))),
        );

        // Benchmark Euclidean distance
        group.bench_with_input(
            BenchmarkId::new("euclidean_distance", format!("dim_{dim}")),
            &(&vec_a, &vec_b),
            |b, (vec_a, vec_b)| b.iter(|| black_box(VectorUtils::euclidean_distance(vec_a, vec_b))),
        );

        // Benchmark normalization
        group.bench_with_input(
            BenchmarkId::new("normalize", format!("dim_{dim}")),
            &vec_a,
            |b, vec| {
                b.iter_batched(
                    || vec.clone(),
                    |mut v| {
                        VectorUtils::normalize(&mut v);
                        black_box(v)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let index_size = 1000;
    let dimension = 384;
    let mut index = create_test_vectors(index_size, dimension);
    index.build_index().unwrap();

    let batch_sizes = vec![1, 10, 50, 100];

    for batch_size in batch_sizes {
        let queries: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| VectorUtils::random_vector(dimension))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_search", format!("{batch_size}_queries")),
            &(&index, &queries),
            |b, (index, queries)| b.iter(|| black_box(index.batch_search(queries, 10).unwrap())),
        );
    }

    // Benchmark centroid calculation
    let vector_counts = vec![10, 100, 1000];
    for count in vector_counts {
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|_| VectorUtils::random_vector(dimension))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("centroid", format!("{count}_vectors")),
            &vectors,
            |b, vectors| b.iter(|| black_box(VectorUtils::centroid(vectors))),
        );
    }

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let sizes = vec![1000, 5000, 10000];
    let dimension = 384;

    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("index_creation", format!("{size}_vectors")),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut index = VectorIndex::new();
                    for i in 0..size {
                        let vector = VectorUtils::random_vector(dimension);
                        index.add_vector(format!("vec_{i}"), vector).unwrap();
                    }
                    index.build_index().unwrap();
                    black_box(index)
                })
            },
        );
    }

    group.finish();
}

// Helper function to create test vectors
fn create_test_vectors(count: usize, dimension: usize) -> VectorIndex {
    let mut index = VectorIndex::new();

    for i in 0..count {
        let vector = VectorUtils::random_vector(dimension);
        index.add_vector(format!("vector_{i}"), vector).unwrap();
    }

    index
}

criterion_group!(
    benches,
    benchmark_vector_index_operations,
    benchmark_vector_operations,
    benchmark_batch_operations,
    benchmark_memory_usage
);
criterion_main!(benches);
