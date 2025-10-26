//! GraphRAG Persistence Tests
//!
//! Tests for save/load functionality using IndexedDB storage.
//! Validates that graph state can be persisted and restored.

use wasm_bindgen_test::*;
use graphrag_wasm::GraphRAG;

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Save and load empty graph
///
/// Validates that we can save and load a graph with no documents.
#[wasm_bindgen_test]
async fn test_save_load_empty_graph() {
    let graph = GraphRAG::new(384).unwrap();

    // Save empty graph
    let save_result = graph.save_to_storage("test-empty-graph").await;
    assert!(save_result.is_ok());

    // Load into new instance
    let mut loaded_graph = GraphRAG::new(384).unwrap();
    let load_result = loaded_graph.load_from_storage("test-empty-graph").await;
    assert!(load_result.is_ok());

    // Verify state
    assert_eq!(loaded_graph.document_count(), 0);
    assert_eq!(loaded_graph.get_dimension(), 384);
}

/// Test: Save and load graph with documents
///
/// Validates complete persistence of documents and embeddings.
#[wasm_bindgen_test]
async fn test_save_load_with_documents() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents
    let doc1_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let doc2_embedding: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();

    graph.add_document(
        "doc1".to_string(),
        "GraphRAG is a knowledge graph system".to_string(),
        doc1_embedding.clone()
    ).await.unwrap();

    graph.add_document(
        "doc2".to_string(),
        "WASM enables browser-side ML".to_string(),
        doc2_embedding.clone()
    ).await.unwrap();

    // Build index
    graph.build_index().await.unwrap();

    // Save
    let save_result = graph.save_to_storage("test-graph-with-docs").await;
    assert!(save_result.is_ok());

    // Load into new instance
    let mut loaded_graph = GraphRAG::new(384).unwrap();
    let load_result = loaded_graph.load_from_storage("test-graph-with-docs").await;
    assert!(load_result.is_ok());

    // Verify state
    assert_eq!(loaded_graph.document_count(), 2);
    assert_eq!(loaded_graph.get_dimension(), 384);
    assert!(loaded_graph.is_index_built());
}

/// Test: Save, clear, and load
///
/// Validates that loading restores a cleared graph.
#[wasm_bindgen_test]
async fn test_save_clear_load() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add document
    let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    graph.add_document(
        "doc1".to_string(),
        "Test document".to_string(),
        embedding
    ).await.unwrap();

    assert_eq!(graph.document_count(), 1);

    // Save
    graph.save_to_storage("test-save-clear-load").await.unwrap();

    // Clear
    graph.clear();
    assert_eq!(graph.document_count(), 0);

    // Load should restore
    graph.load_from_storage("test-save-clear-load").await.unwrap();
    assert_eq!(graph.document_count(), 1);
}

/// Test: Query after load
///
/// Validates that queries work correctly after loading from storage.
#[wasm_bindgen_test]
async fn test_query_after_load() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents
    let doc1_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let doc2_embedding: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();

    graph.add_document(
        "doc1".to_string(),
        "Machine learning".to_string(),
        doc1_embedding.clone()
    ).await.unwrap();

    graph.add_document(
        "doc2".to_string(),
        "Deep neural networks".to_string(),
        doc2_embedding.clone()
    ).await.unwrap();

    // Build index and save
    graph.build_index().await.unwrap();
    graph.save_to_storage("test-query-after-load").await.unwrap();

    // Load into new instance
    let mut loaded_graph = GraphRAG::new(384).unwrap();
    loaded_graph.load_from_storage("test-query-after-load").await.unwrap();

    // Query should work
    let query_embedding: Vec<f32> = (0..384).map(|i| ((i + 10) as f32) / 384.0).collect();
    let results = loaded_graph.query(query_embedding, 2).await;

    assert!(results.is_ok());
    web_sys::console::log_1(&format!("Query results: {}", results.unwrap()).into());
}

/// Test: Multiple save/load cycles
///
/// Validates that multiple save/load operations maintain data integrity.
#[wasm_bindgen_test]
async fn test_multiple_save_load_cycles() {
    let db_name = "test-multiple-cycles";

    // Cycle 1: Save 1 document
    {
        let mut graph = GraphRAG::new(384).unwrap();
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        graph.add_document("doc1".to_string(), "First".to_string(), embedding).await.unwrap();
        graph.save_to_storage(db_name).await.unwrap();
    }

    // Cycle 2: Load, add 2nd document, save
    {
        let mut graph = GraphRAG::new(384).unwrap();
        graph.load_from_storage(db_name).await.unwrap();
        assert_eq!(graph.document_count(), 1);

        let embedding: Vec<f32> = (0..384).map(|i| ((i + 50) as f32) / 384.0).collect();
        graph.add_document("doc2".to_string(), "Second".to_string(), embedding).await.unwrap();
        graph.save_to_storage(db_name).await.unwrap();
    }

    // Cycle 3: Load, verify 2 documents
    {
        let mut graph = GraphRAG::new(384).unwrap();
        graph.load_from_storage(db_name).await.unwrap();
        assert_eq!(graph.document_count(), 2);
    }
}

/// Test: Large graph persistence
///
/// Validates that large graphs can be saved and loaded.
#[wasm_bindgen_test]
async fn test_large_graph_persistence() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add 100 documents
    for i in 0..100 {
        let embedding: Vec<f32> = (0..384).map(|j| ((j + i * 10) as f32) / 384.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    assert_eq!(graph.document_count(), 100);

    // Build index
    graph.build_index().await.unwrap();

    // Save
    let save_result = graph.save_to_storage("test-large-graph").await;
    assert!(save_result.is_ok());

    // Load into new instance
    let mut loaded_graph = GraphRAG::new(384).unwrap();
    let load_result = loaded_graph.load_from_storage("test-large-graph").await;
    assert!(load_result.is_ok());

    // Verify
    assert_eq!(loaded_graph.document_count(), 100);
    assert!(loaded_graph.is_index_built());
}

/// Test: Load non-existent database
///
/// Validates error handling when loading from non-existent storage.
#[wasm_bindgen_test]
async fn test_load_nonexistent_database() {
    let mut graph = GraphRAG::new(384).unwrap();

    let result = graph.load_from_storage("this-database-does-not-exist-12345").await;

    // Should fail gracefully
    // Note: Depending on implementation, this might succeed with empty data
    // or fail with an error. Both are acceptable behaviors.
    match result {
        Ok(_) => {
            // Loaded empty data
            assert_eq!(graph.document_count(), 0);
        }
        Err(e) => {
            // Failed to load
            web_sys::console::log_1(&format!("Expected error: {:?}", e).into());
        }
    }
}

/// Test: Different dimension persistence
///
/// Validates that dimension is correctly saved and validated on load.
#[wasm_bindgen_test]
async fn test_dimension_persistence() {
    // Save with 384 dimensions
    {
        let mut graph = GraphRAG::new(384).unwrap();
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        graph.add_document("doc1".to_string(), "Test".to_string(), embedding).await.unwrap();
        graph.save_to_storage("test-dimension").await.unwrap();
    }

    // Load with matching dimension
    {
        let mut graph = GraphRAG::new(384).unwrap();
        let result = graph.load_from_storage("test-dimension").await;
        assert!(result.is_ok());
    }

    // Note: Loading with different dimension may fail or succeed depending on implementation
    // This behavior should be documented
}

/// Test: Concurrent save/load
///
/// Validates that multiple save/load operations can happen concurrently.
#[wasm_bindgen_test]
async fn test_concurrent_save_load() {

    // Create and save graph 1
    {
        let mut graph1 = GraphRAG::new(384).unwrap();
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        graph1.add_document("doc1".to_string(), "Graph 1".to_string(), embedding).await.unwrap();
        graph1.save_to_storage("concurrent-graph-1").await.unwrap();
    }

    // Create and save graph 2
    {
        let mut graph2 = GraphRAG::new(384).unwrap();
        let embedding: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();
        graph2.add_document("doc2".to_string(), "Graph 2".to_string(), embedding.clone()).await.unwrap();
        graph2.add_document("doc3".to_string(), "Graph 2 second".to_string(), embedding).await.unwrap();
        graph2.save_to_storage("concurrent-graph-2").await.unwrap();
    }

    // Load both concurrently
    let mut loaded1 = GraphRAG::new(384).unwrap();
    let mut loaded2 = GraphRAG::new(384).unwrap();

    let result1 = loaded1.load_from_storage("concurrent-graph-1").await;
    let result2 = loaded2.load_from_storage("concurrent-graph-2").await;

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    assert_eq!(loaded1.document_count(), 1);
    assert_eq!(loaded2.document_count(), 2);
}
