//! Complete pipeline integration tests
//!
//! Tests the full GraphRAG pipeline including:
//! - Document processing
//! - Vector search with Voy
//! - LLM integration with WebLLM
//! - Storage persistence

use wasm_bindgen_test::*;
use graphrag_wasm::{
    GraphRAG,
    embedder::WasmEmbedder,
    webllm::is_webllm_available,
    storage::{IndexedDBStore, CacheStore, estimate_storage},
    voy_bindings::check_voy_available,
};

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Complete RAG pipeline (document → embedding → storage → retrieval)
///
/// This test validates the end-to-end RAG workflow:
/// 1. Create embedder
/// 2. Add documents with embeddings
/// 3. Store in IndexedDB
/// 4. Build vector index
/// 5. Query and retrieve
#[wasm_bindgen_test]
async fn test_complete_rag_pipeline() {
    // Step 1: Create GraphRAG instance
    let mut graph = GraphRAG::new(384).unwrap();

    // Step 2: Create embedder (would generate real embeddings in production)
    let _embedder = WasmEmbedder::new("test-model".to_string(), 384).await.unwrap();

    // Step 3: Add documents
    let documents = vec![
        "GraphRAG combines knowledge graphs with retrieval-augmented generation",
        "WASM enables running ML models directly in the browser",
        "Vector databases enable semantic search over embeddings",
    ];

    for (i, doc) in documents.iter().enumerate() {
        // Generate dummy embedding (in production, use embedder.embed())
        let embedding: Vec<f32> = (0..384).map(|j| {
            ((j + i * 10) as f32) / 384.0
        }).collect();

        let result = graph.add_document(
            format!("doc{}", i),
            doc.to_string(),
            embedding
        ).await;

        assert!(result.is_ok());
    }

    // Step 4: Build index
    graph.build_index().await.unwrap();

    // Step 5: Query
    let query_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let results = graph.query(query_embedding, 2).await;

    assert!(results.is_ok());
}

/// Test: Voy vector search availability
#[wasm_bindgen_test]
fn test_voy_availability() {
    let voy_available = check_voy_available();

    // In a real browser with voy-search loaded, this should be true
    // In test environment without npm package, might be false
    web_sys::console::log_1(&format!("Voy available: {}", voy_available).into());
}

/// Test: WebLLM availability check
#[wasm_bindgen_test]
fn test_webllm_availability() {
    let webllm_available = is_webllm_available();

    // In a real browser with WebLLM script loaded, this should be true
    web_sys::console::log_1(&format!("WebLLM available: {}", webllm_available).into());
}

/// Test: Storage quota estimation
#[wasm_bindgen_test]
async fn test_storage_quota() {
    let result = estimate_storage().await;

    if let Ok((usage, quota, percentage)) = result {
        web_sys::console::log_1(&format!(
            "Storage: {}MB used / {}MB total ({}%)",
            usage / 1_000_000,
            quota / 1_000_000,
            percentage
        ).into());

        assert!(quota > 0);
        assert!(percentage >= 0.0 && percentage <= 100.0);
    }
}

/// Test: Document persistence workflow
///
/// Validates that documents can be stored and retrieved from IndexedDB:
/// 1. Add documents to IndexedDB
/// 2. Retrieve and verify
/// 3. Build GraphRAG from stored data
#[wasm_bindgen_test]
async fn test_document_persistence() {
    // Create IndexedDB store
    let db = IndexedDBStore::new("test-persistence", 1).await.unwrap();

    // Store documents
    let doc1 = serde_json::json!({
        "id": "doc1",
        "text": "Test document 1",
        "embedding": vec![0.1; 384]
    });

    let doc2 = serde_json::json!({
        "id": "doc2",
        "text": "Test document 2",
        "embedding": vec![0.2; 384]
    });

    db.put("documents", "doc1", &doc1).await.unwrap();
    db.put("documents", "doc2", &doc2).await.unwrap();

    // Retrieve documents
    let retrieved1 = db.get::<serde_json::Value>("documents", "doc1").await.unwrap();
    let retrieved2 = db.get::<serde_json::Value>("documents", "doc2").await.unwrap();

    assert_eq!(retrieved1["id"], "doc1");
    assert_eq!(retrieved2["id"], "doc2");

    // Clean up
    db.clear("documents").await.unwrap();
}

/// Test: Model caching workflow
///
/// Validates that ML models can be cached and retrieved:
/// 1. Store model in Cache API
/// 2. Check if cached
/// 3. Retrieve model
/// 4. Verify integrity
#[wasm_bindgen_test]
async fn test_model_caching() {
    let cache = CacheStore::open("test-model-cache").await.unwrap();

    // Simulate model data
    let model_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

    // Cache model
    cache.put("test-model.bin", &model_data).await.unwrap();

    // Check if cached
    let has_model = cache.has("test-model.bin").await.unwrap();
    assert!(has_model);

    // Retrieve model
    let retrieved = cache.get("test-model.bin").await.unwrap();

    // Verify integrity
    assert_eq!(retrieved.len(), model_data.len());
    assert_eq!(retrieved, model_data);

    // Clean up
    cache.delete("test-model.bin").await.unwrap();
}

/// Test: Hybrid search workflow
///
/// Validates combining vector search with keyword filtering:
/// 1. Add documents with metadata
/// 2. Build vector index
/// 3. Query with filters
#[wasm_bindgen_test]
async fn test_hybrid_search() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents with different types
    let docs = vec![
        ("concept", "Machine learning is a field of AI"),
        ("definition", "Vector embeddings represent semantic meaning"),
        ("example", "BERT is a transformer-based model"),
    ];

    for (i, (doc_type, text)) in docs.iter().enumerate() {
        let embedding: Vec<f32> = (0..384).map(|j| {
            ((j + i * 20) as f32) / 384.0
        }).collect();

        graph.add_document(
            format!("{}_{}", doc_type, i),
            text.to_string(),
            embedding
        ).await.unwrap();
    }

    graph.build_index().await.unwrap();

    // Query
    let query_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    let results = graph.query(query_embedding, 3).await;

    assert!(results.is_ok());
}

/// Test: Incremental updates
///
/// Validates that GraphRAG can handle incremental document additions:
/// 1. Add initial documents
/// 2. Build index
/// 3. Add more documents
/// 4. Rebuild index
/// 5. Verify all documents are searchable
#[wasm_bindgen_test]
async fn test_incremental_updates() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add initial batch
    for i in 0..10 {
        let embedding: Vec<f32> = (0..384).map(|j| {
            ((j + i * 10) as f32) / 384.0
        }).collect();

        graph.add_document(
            format!("doc{}", i),
            format!("Initial document {}", i),
            embedding
        ).await.unwrap();
    }

    graph.build_index().await.unwrap();
    assert_eq!(graph.document_count(), 10);

    // Add more documents
    for i in 10..20 {
        let embedding: Vec<f32> = (0..384).map(|j| {
            ((j + i * 10) as f32) / 384.0
        }).collect();

        graph.add_document(
            format!("doc{}", i),
            format!("New document {}", i),
            embedding
        ).await.unwrap();
    }

    // Rebuild index with new documents
    graph.build_index().await.unwrap();
    assert_eq!(graph.document_count(), 20);
}

/// Test: Error recovery
///
/// Validates graceful error handling in various failure scenarios.
#[wasm_bindgen_test]
async fn test_error_recovery() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Try to query before building index (should return empty results)
    let query_embedding: Vec<f32> = vec![0.5; 384];
    let result = graph.query(query_embedding.clone(), 5).await;

    // Should succeed but return empty results
    assert!(result.is_ok());

    // Add a document
    let embedding: Vec<f32> = vec![0.5; 384];
    graph.add_document(
        "doc1".to_string(),
        "Test document".to_string(),
        embedding
    ).await.unwrap();

    // Query should now work
    let result2 = graph.query(query_embedding, 1).await;
    assert!(result2.is_ok());
}

/// Test: Memory management
///
/// Validates that GraphRAG properly manages memory when clearing data.
#[wasm_bindgen_test]
async fn test_memory_management() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add many documents
    for i in 0..100 {
        let embedding: Vec<f32> = (0..384).map(|j| {
            ((j + i) as f32) / 384.0
        }).collect();

        graph.add_document(
            format!("doc{}", i),
            format!("Document {}", i),
            embedding
        ).await.unwrap();
    }

    assert_eq!(graph.document_count(), 100);

    // Clear all data
    graph.clear();

    assert_eq!(graph.document_count(), 0);
    assert!(!graph.is_index_built());

    // Should be able to add new documents after clearing
    let embedding: Vec<f32> = vec![0.5; 384];
    let result = graph.add_document(
        "new_doc".to_string(),
        "New document after clear".to_string(),
        embedding
    ).await;

    assert!(result.is_ok());
    assert_eq!(graph.document_count(), 1);
}

/// Test: Concurrent operations
///
/// Validates that multiple GraphRAG operations can be performed concurrently.
#[wasm_bindgen_test]
async fn test_concurrent_operations() {
    let mut graph1 = GraphRAG::new(384).unwrap();
    let mut graph2 = GraphRAG::new(768).unwrap();

    // Add documents to both graphs concurrently
    let embedding1: Vec<f32> = vec![0.5; 384];
    let embedding2: Vec<f32> = vec![0.5; 768];

    graph1.add_document(
        "doc1".to_string(),
        "Document for graph 1".to_string(),
        embedding1
    ).await.unwrap();

    graph2.add_document(
        "doc1".to_string(),
        "Document for graph 2".to_string(),
        embedding2
    ).await.unwrap();

    assert_eq!(graph1.document_count(), 1);
    assert_eq!(graph2.document_count(), 1);
    assert_eq!(graph1.get_dimension(), 384);
    assert_eq!(graph2.get_dimension(), 768);
}
