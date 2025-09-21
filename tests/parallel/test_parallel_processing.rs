use graphrag_rs::{
    config::{Config, ParallelConfig},
    core::{ChunkId, DocumentId, EntityId, KnowledgeGraph, TextChunk},
    parallel::{ParallelProcessor, ItemComplexity, ProcessingStrategy},
    retrieval::RetrievalSystem,
    vector::{EmbeddingGenerator, VectorIndex},
    summarization::{DocumentTree, HierarchicalConfig},
};
use std::time::Instant;

fn create_test_parallel_config() -> ParallelConfig {
    ParallelConfig {
        enabled: true,
        num_threads: 2, // Use small number for testing
        min_batch_size: 5,
        chunk_batch_size: 20,
        parallel_embeddings: true,
        parallel_graph_ops: true,
        parallel_vector_ops: true,
    }
}

fn create_test_chunks(count: usize) -> Vec<TextChunk> {
    (0..count)
        .map(|i| TextChunk {
            id: ChunkId::new(format!("chunk_{}", i)),
            content: format!(
                "This is test content for chunk {}. It contains information about topic {}.",
                i, i % 3
            ),
            start_offset: i * 50,
            end_offset: (i + 1) * 50,
            entities: Vec::new(),
            embedding: None,
        })
        .collect()
}

#[test]
fn test_parallel_processor_initialization() {
    let config = create_test_parallel_config();
    let mut processor = ParallelProcessor::new(config);

    assert!(processor.initialize().is_ok());
    assert!(processor.should_use_parallel(10));
    assert!(!processor.should_use_parallel(2));
}

#[test]
fn test_parallel_embedding_generation() {
    let config = create_test_parallel_config();
    let mut processor = ParallelProcessor::new(config);
    processor.initialize().unwrap();

    let mut generator = EmbeddingGenerator::new(64);

    // Test batch generation
    let texts = vec!["hello world", "test document", "parallel processing", "embedding generation"];
    let embeddings = generator.batch_generate(&texts);

    assert_eq!(embeddings.len(), texts.len());
    assert!(embeddings.iter().all(|emb| emb.len() == 64));

    // Test chunked generation
    let large_texts: Vec<&str> = (0..50)
        .map(|i| match i % 3 {
            0 => "test document one",
            1 => "test document two",
            _ => "test document three",
        })
        .collect();

    let chunked_embeddings = generator.batch_generate_chunked(&large_texts, 10);
    assert_eq!(chunked_embeddings.len(), large_texts.len());
    assert!(chunked_embeddings.iter().all(|emb| emb.len() == 64));
}

#[test]
fn test_parallel_vector_operations() {
    let config = create_test_parallel_config();
    let processor = ParallelProcessor::new(config);

    let mut index = VectorIndex::with_parallel_processing(processor);

    // Test parallel batch adding
    let vectors: Vec<(String, Vec<f32>)> = (0..20)
        .map(|i| {
            let vec: Vec<f32> = (0..32).map(|j| (i + j) as f32 / 10.0).collect();
            (format!("vec_{}", i), vec)
        })
        .collect();

    let start = Instant::now();
    index.batch_add_vectors(vectors.clone()).unwrap();
    let parallel_duration = start.elapsed();

    assert_eq!(index.len(), 20);
    assert!(index.contains("vec_0"));
    assert!(index.contains("vec_19"));

    // Test parallel similarity computation
    let similarities = index.compute_all_similarities();
    assert!(!similarities.is_empty());

    println!("Parallel vector operations completed in {:?}", parallel_duration);
}

#[test]
fn test_parallel_node_creation() {
    let chunks = create_test_chunks(15); // More than threshold for parallel processing
    let config = HierarchicalConfig::default();
    let doc_id = DocumentId::new("test_parallel_doc".to_string());

    let parallel_config = create_test_parallel_config();
    let processor = ParallelProcessor::new(parallel_config);

    let start = Instant::now();
    let mut tree = DocumentTree::with_parallel_processing(doc_id, config, processor).unwrap();
    tree.build_from_chunks(chunks).unwrap();
    let parallel_duration = start.elapsed();

    let stats = tree.get_statistics();
    assert!(stats.total_nodes > 0);
    assert!(stats.max_level >= 0);

    println!("Parallel node creation completed in {:?}", parallel_duration);
}

#[test]
fn test_processing_strategy_selection() {
    let config = create_test_parallel_config();
    let mut processor = ParallelProcessor::new(config);
    processor.initialize().unwrap();

    let items: Vec<i32> = (0..100).collect();
    let operation = |x: i32| -> i32 { x * 2 };

    // Test different strategies
    let strategies = [
        ProcessingStrategy::Sequential,
        ProcessingStrategy::ParallelMap,
        ProcessingStrategy::ChunkedParallel,
        ProcessingStrategy::WorkStealing,
    ];

    for strategy in &strategies {
        let start = Instant::now();
        let results = processor.execute_with_strategy(
            items.clone(),
            operation,
            *strategy,
        );
        let duration = start.elapsed();

        assert_eq!(results.len(), items.len());
        assert_eq!(results[0], 0);
        assert_eq!(results[99], 198);

        println!("Strategy {:?} completed in {:?}", strategy, duration);
    }
}

#[test]
fn test_adaptive_strategy_selection() {
    let config = create_test_parallel_config();
    let mut processor = ParallelProcessor::new(config);
    processor.initialize().unwrap();

    let small_items: Vec<i32> = (0..5).collect();
    let large_items: Vec<i32> = (0..100).collect();
    let operation = |x: i32| -> i32 { x + 1 };

    // Test adaptive selection with different complexities
    for complexity in [ItemComplexity::Low, ItemComplexity::Medium, ItemComplexity::High] {
        // Small workload
        let small_results = processor.execute_adaptive(
            small_items.clone(),
            operation,
            complexity,
        );
        assert_eq!(small_results.len(), 5);

        // Large workload
        let large_results = processor.execute_adaptive(
            large_items.clone(),
            operation,
            complexity,
        );
        assert_eq!(large_results.len(), 100);
        assert_eq!(large_results[0], 1);
        assert_eq!(large_results[99], 100);
    }
}

#[test]
fn test_parallel_error_handling() {
    let config = create_test_parallel_config();
    let processor = ParallelProcessor::new(config);

    let mut index = VectorIndex::with_parallel_processing(processor);

    // Test error handling with invalid vectors
    let invalid_vectors = vec![
        ("valid".to_string(), vec![1.0, 2.0, 3.0]),
        ("empty".to_string(), vec![]), // This should cause an error
        ("valid2".to_string(), vec![4.0, 5.0, 6.0]),
    ];

    // The implementation should handle errors gracefully
    let result = index.batch_add_vectors(invalid_vectors);
    assert!(result.is_ok()); // Should fall back to sequential processing
}

#[test]
fn test_parallel_performance_monitoring() {
    let config = create_test_parallel_config();
    let mut processor = ParallelProcessor::new(config);
    processor.initialize().unwrap();

    let stats = processor.get_statistics();
    assert!(stats.enabled);
    assert!(stats.initialized);
    assert!(stats.num_threads > 0);

    println!("Parallel processor statistics:");
    stats.print();
}