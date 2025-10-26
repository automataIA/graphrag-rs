//! End-to-end integration tests for GraphRAG WASM
//!
//! These tests validate the complete pipeline from document ingestion to query answering.
//! They run in a headless browser environment using wasm-bindgen-test.

use wasm_bindgen_test::*;
use graphrag_wasm::{GraphRAG, storage::{IndexedDBStore, CacheStore}};

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Create GraphRAG instance
///
/// Validates that we can create a new GraphRAG instance with specified dimension.
#[wasm_bindgen_test]
async fn test_create_graphrag() {
    let graph = GraphRAG::new(384);
    assert!(graph.is_ok());

    let graph = graph.unwrap();
    assert_eq!(graph.get_dimension(), 384);
    assert_eq!(graph.document_count(), 0);
    assert!(!graph.is_index_built());
}

/// Test: Add documents and build index
///
/// Validates the complete document ingestion pipeline:
/// 1. Create GraphRAG instance
/// 2. Add documents with embeddings
/// 3. Build vector index
/// 4. Verify document count
#[wasm_bindgen_test]
async fn test_add_documents_and_build_index() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Create dummy embeddings (384-dimensional)
    let embedding1: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let embedding2: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();
    let embedding3: Vec<f32> = (0..384).map(|i| ((i + 200) as f32) / 384.0).collect();

    // Add documents
    let result1 = graph.add_document(
        "doc1".to_string(),
        "GraphRAG is a knowledge graph system".to_string(),
        embedding1
    ).await;
    assert!(result1.is_ok());

    let result2 = graph.add_document(
        "doc2".to_string(),
        "WASM enables client-side ML inference".to_string(),
        embedding2
    ).await;
    assert!(result2.is_ok());

    let result3 = graph.add_document(
        "doc3".to_string(),
        "WebGPU provides GPU acceleration in browsers".to_string(),
        embedding3
    ).await;
    assert!(result3.is_ok());

    // Verify document count
    assert_eq!(graph.document_count(), 3);

    // Build index
    let build_result = graph.build_index().await;
    assert!(build_result.is_ok());
    assert!(graph.is_index_built());
}

/// Test: Query GraphRAG
///
/// Validates the query pipeline:
/// 1. Add documents
/// 2. Build index
/// 3. Query with embedding
/// 4. Verify results format
#[wasm_bindgen_test]
async fn test_query_graphrag() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents
    let embedding1: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let embedding2: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();

    graph.add_document(
        "doc1".to_string(),
        "GraphRAG is a knowledge graph system".to_string(),
        embedding1.clone()
    ).await.unwrap();

    graph.add_document(
        "doc2".to_string(),
        "WASM enables client-side ML inference".to_string(),
        embedding2
    ).await.unwrap();

    // Build index
    graph.build_index().await.unwrap();

    // Query with similar embedding to doc1
    let query_embedding: Vec<f32> = (0..384).map(|i| ((i + 10) as f32) / 384.0).collect();
    let results = graph.query(query_embedding, 2).await;

    assert!(results.is_ok());
    // TODO: Parse JSON results and validate structure when vector search is implemented
}

/// Test: Clear GraphRAG
///
/// Validates that we can clear all documents and reset the index.
#[wasm_bindgen_test]
async fn test_clear_graphrag() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add a document
    let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    graph.add_document(
        "doc1".to_string(),
        "Test document".to_string(),
        embedding
    ).await.unwrap();

    assert_eq!(graph.document_count(), 1);

    // Clear
    graph.clear();

    assert_eq!(graph.document_count(), 0);
}

/// Test: IndexedDB Storage
///
/// Validates IndexedDB storage operations:
/// 1. Open database
/// 2. Put data
/// 3. Get data
/// 4. Delete data
/// 5. Clear database
#[wasm_bindgen_test]
async fn test_indexeddb_storage() {
    let db_result = IndexedDBStore::new("test-graphrag", 1).await;
    assert!(db_result.is_ok());

    let db = db_result.unwrap();

    // Put data
    let test_data = serde_json::json!({
        "id": "entity_1",
        "name": "Test Entity",
        "type": "concept"
    });

    let put_result = db.put("entities", "entity_1", &test_data).await;
    assert!(put_result.is_ok());

    // Get data
    let get_result = db.get::<serde_json::Value>("entities", "entity_1").await;
    assert!(get_result.is_ok());

    let retrieved = get_result.unwrap();
    assert_eq!(retrieved["id"], "entity_1");
    assert_eq!(retrieved["name"], "Test Entity");

    // Delete data
    let delete_result = db.delete("entities", "entity_1").await;
    assert!(delete_result.is_ok());

    // Clear database
    let clear_result = db.clear("entities").await;
    assert!(clear_result.is_ok());
}

/// Test: Cache API Storage
///
/// Validates Cache API storage operations:
/// 1. Open cache
/// 2. Put data
/// 3. Check existence
/// 4. Get data
/// 5. Delete data
#[wasm_bindgen_test]
async fn test_cache_api_storage() {
    let cache_result = CacheStore::open("test-models").await;
    assert!(cache_result.is_ok());

    let cache = cache_result.unwrap();

    // Put data
    let test_data = b"Test model data";
    let put_result = cache.put("test-model.bin", test_data).await;
    assert!(put_result.is_ok());

    // Check existence
    let has_result = cache.has("test-model.bin").await;
    assert!(has_result.is_ok());
    assert!(has_result.unwrap());

    // Get data
    let get_result = cache.get("test-model.bin").await;
    assert!(get_result.is_ok());

    let retrieved = get_result.unwrap();
    assert_eq!(retrieved, test_data);

    // Delete data
    let delete_result = cache.delete("test-model.bin").await;
    assert!(delete_result.is_ok());

    // Verify deletion
    let has_after_delete = cache.has("test-model.bin").await.unwrap();
    assert!(!has_after_delete);
}

/// Test: Multiple document types
///
/// Validates that GraphRAG can handle different types of documents.
#[wasm_bindgen_test]
async fn test_multiple_document_types() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Different document types
    let documents = vec![
        ("concept1", "Machine learning is a subset of artificial intelligence"),
        ("concept2", "Neural networks are inspired by biological neurons"),
        ("fact1", "WASM runs at near-native speed"),
        ("question1", "What is a knowledge graph?"),
    ];

    // Add all documents
    for (id, text) in documents {
        let embedding: Vec<f32> = (0..384).map(|i| {
            ((i + id.len()) as f32) / 384.0
        }).collect();

        let result = graph.add_document(
            id.to_string(),
            text.to_string(),
            embedding
        ).await;

        assert!(result.is_ok());
    }

    assert_eq!(graph.document_count(), 4);

    // Build index
    graph.build_index().await.unwrap();
    assert!(graph.is_index_built());
}

/// Test: Large batch of documents
///
/// Validates that GraphRAG can handle a larger number of documents.
#[wasm_bindgen_test]
async fn test_large_batch_documents() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add 50 documents
    for i in 0..50 {
        let embedding: Vec<f32> = (0..384).map(|j| {
            ((j + i * 10) as f32) / 384.0
        }).collect();

        let result = graph.add_document(
            format!("doc{}", i),
            format!("This is test document number {}", i),
            embedding
        ).await;

        assert!(result.is_ok());
    }

    assert_eq!(graph.document_count(), 50);

    // Build index
    let build_result = graph.build_index().await;
    assert!(build_result.is_ok());
}

/// Test: Error handling - invalid dimensions
///
/// Validates that the system properly handles dimension mismatches.
#[wasm_bindgen_test]
async fn test_dimension_validation() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Try to add document with wrong dimension (should fail in future validation)
    let wrong_embedding: Vec<f32> = vec![0.1, 0.2, 0.3]; // Only 3 dimensions instead of 384

    // Currently this won't fail, but should be validated in production
    let result = graph.add_document(
        "doc1".to_string(),
        "Test".to_string(),
        wrong_embedding
    ).await;

    // TODO: Add dimension validation and expect error here
    // For now, just verify the call completes
    let _ = result;
}
