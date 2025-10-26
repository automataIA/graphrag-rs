//! Voy Vector Search Tests
//!
//! Tests for Voy k-d tree integration and vector similarity search.
//! These tests validate that Voy provides accurate nearest neighbor search.

use wasm_bindgen_test::*;
use graphrag_wasm::GraphRAG;

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Basic Voy index build
///
/// Validates that we can build a Voy index with embeddings.
#[wasm_bindgen_test]
async fn test_voy_index_build() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents
    for i in 0..10 {
        let embedding: Vec<f32> = (0..384).map(|j| ((j + i * 10) as f32) / 384.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    // Build Voy index
    let result = graph.build_index().await;
    assert!(result.is_ok());
    assert!(graph.is_index_built());

    // Check index info
    let info = graph.index_info();
    web_sys::console::log_1(&format!("Index info: {}", info).into());
}

/// Test: Voy search accuracy
///
/// Validates that Voy returns correct nearest neighbors.
#[wasm_bindgen_test]
async fn test_voy_search_accuracy() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Create 3 documents with known embeddings
    // Doc 0: all 0.0
    let embedding0: Vec<f32> = vec![0.0; 384];

    // Doc 1: all 0.5
    let embedding1: Vec<f32> = vec![0.5; 384];

    // Doc 2: all 1.0
    let embedding2: Vec<f32> = vec![1.0; 384];

    graph.add_document("doc0".to_string(), "Zero".to_string(), embedding0).await.unwrap();
    graph.add_document("doc1".to_string(), "Half".to_string(), embedding1).await.unwrap();
    graph.add_document("doc2".to_string(), "One".to_string(), embedding2).await.unwrap();

    // Build index
    graph.build_index().await.unwrap();

    // Query with embedding close to doc1 (all 0.6)
    let query: Vec<f32> = vec![0.6; 384];
    let results = graph.query(query, 1).await.unwrap();

    web_sys::console::log_1(&format!("Search results: {}", results).into());

    // Parse results (should be JSON)
    // Expected: doc1 should be the closest
    assert!(results.contains("doc1") || results.contains("\"id\":1"));
}

/// Test: k-NN parameter
///
/// Validates that we can retrieve different numbers of neighbors.
#[wasm_bindgen_test]
async fn test_knn_parameter() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add 10 documents
    for i in 0..10 {
        let embedding: Vec<f32> = (0..384).map(|j| ((j + i) as f32) / 384.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    graph.build_index().await.unwrap();

    let query: Vec<f32> = (0..384).map(|j| (j as f32) / 384.0).collect();

    // Get top 1
    let results_1 = graph.query(query.clone(), 1).await.unwrap();
    web_sys::console::log_1(&format!("Top 1: {}", results_1).into());

    // Get top 3
    let results_3 = graph.query(query.clone(), 3).await.unwrap();
    web_sys::console::log_1(&format!("Top 3: {}", results_3).into());

    // Get top 5
    let results_5 = graph.query(query.clone(), 5).await.unwrap();
    web_sys::console::log_1(&format!("Top 5: {}", results_5).into());

    // Each should return valid JSON
    // Top 5 should have more content than top 1
    assert!(results_5.len() >= results_1.len());
}

/// Test: Voy with identical embeddings
///
/// Validates behavior when multiple documents have the same embedding.
#[wasm_bindgen_test]
async fn test_identical_embeddings() {
    let mut graph = GraphRAG::new(384).unwrap();

    let identical_embedding: Vec<f32> = vec![0.5; 384];

    // Add 3 documents with identical embeddings
    for i in 0..3 {
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            identical_embedding.clone()
        ).await.unwrap();
    }

    graph.build_index().await.unwrap();

    // Query with same embedding
    let results = graph.query(identical_embedding.clone(), 3).await;
    assert!(results.is_ok());

    web_sys::console::log_1(&format!("Identical embedding results: {}", results.unwrap()).into());
}

/// Test: Voy index rebuild
///
/// Validates that we can rebuild the index after adding more documents.
#[wasm_bindgen_test]
async fn test_index_rebuild() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add 5 documents and build index
    for i in 0..5 {
        let embedding: Vec<f32> = (0..384).map(|j| ((j + i * 10) as f32) / 384.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    graph.build_index().await.unwrap();
    assert!(graph.is_index_built());

    // Add 5 more documents
    for i in 5..10 {
        let embedding: Vec<f32> = (0..384).map(|j| ((j + i * 10) as f32) / 384.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    // Rebuild index
    let rebuild_result = graph.build_index().await;
    assert!(rebuild_result.is_ok());
    assert!(graph.is_index_built());

    // Query should work with all 10 documents
    let query: Vec<f32> = (0..384).map(|j| (j as f32) / 384.0).collect();
    let results = graph.query(query, 10).await;
    assert!(results.is_ok());
}

/// Test: Voy search performance
///
/// Validates that Voy provides fast search performance.
#[wasm_bindgen_test]
async fn test_search_performance() {
    use web_sys::window;

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

    // Build index
    let build_start = window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    graph.build_index().await.unwrap();

    let build_end = window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let build_time = build_end - build_start;
    web_sys::console::log_1(&format!("Index build time: {:.2}ms", build_time).into());

    // Query
    let query_start = window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let query: Vec<f32> = (0..384).map(|j| (j as f32) / 384.0).collect();
    let _results = graph.query(query, 10).await.unwrap();

    let query_end = window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let query_time = query_end - query_start;
    web_sys::console::log_1(&format!("Query time: {:.2}ms", query_time).into());

    // Query should be fast (< 50ms for 100 docs)
    assert!(query_time < 50.0);
}

/// Test: Voy fallback to brute-force
///
/// Validates graceful fallback when Voy is not available.
#[wasm_bindgen_test]
async fn test_brute_force_fallback() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents
    for i in 0..5 {
        let embedding: Vec<f32> = (0..384).map(|j| ((j + i * 10) as f32) / 384.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    // Even if Voy fails to build, brute-force should work
    graph.build_index().await.ok(); // Don't assert - may fail if Voy not loaded

    // Query should still work (falls back to brute-force)
    let query: Vec<f32> = (0..384).map(|j| (j as f32) / 384.0).collect();
    let results = graph.query(query, 3).await;

    assert!(results.is_ok());
    web_sys::console::log_1(&format!("Fallback results: {}", results.unwrap()).into());
}

/// Test: Empty index query
///
/// Validates error handling when querying without building index.
#[wasm_bindgen_test]
async fn test_empty_index_query() {
    let graph = GraphRAG::new(384).unwrap();

    // Query without any documents
    let query: Vec<f32> = (0..384).map(|j| (j as f32) / 384.0).collect();
    let results = graph.query(query, 5).await;

    // Should either return empty results or error gracefully
    match results {
        Ok(json) => {
            web_sys::console::log_1(&format!("Empty query result: {}", json).into());
            // Should be empty array or similar
            assert!(json.contains("[]") || json.contains("\"results\":[]"));
        }
        Err(e) => {
            web_sys::console::log_1(&format!("Empty query error (expected): {:?}", e).into());
        }
    }
}

/// Test: High-dimensional embeddings
///
/// Validates that Voy works with different embedding dimensions.
#[wasm_bindgen_test]
async fn test_high_dimensional_embeddings() {
    // Test with 768 dimensions (BERT size)
    let mut graph = GraphRAG::new(768).unwrap();

    for i in 0..5 {
        let embedding: Vec<f32> = (0..768).map(|j| ((j + i * 10) as f32) / 768.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    let result = graph.build_index().await;
    assert!(result.is_ok());

    let query: Vec<f32> = (0..768).map(|j| (j as f32) / 768.0).collect();
    let results = graph.query(query, 3).await;
    assert!(results.is_ok());
}

/// Test: Low-dimensional embeddings
///
/// Validates that Voy works with smaller embedding dimensions.
#[wasm_bindgen_test]
async fn test_low_dimensional_embeddings() {
    // Test with 128 dimensions (smaller models)
    let mut graph = GraphRAG::new(128).unwrap();

    for i in 0..5 {
        let embedding: Vec<f32> = (0..128).map(|j| ((j + i * 10) as f32) / 128.0).collect();
        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    let result = graph.build_index().await;
    assert!(result.is_ok());

    let query: Vec<f32> = (0..128).map(|j| (j as f32) / 128.0).collect();
    let results = graph.query(query, 3).await;
    assert!(results.is_ok());
}
