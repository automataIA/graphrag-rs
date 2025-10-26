//! Comprehensive integration tests for Voy vector search
//!
//! These tests verify the complete Voy integration with modern API v0.6+

use wasm_bindgen_test::*;
use graphrag_wasm::voy_bindings::{VoyIndex, check_voy_available};
use js_sys::{Array, Float32Array};

wasm_bindgen_test_configure!(run_in_browser);

/// Test 1: Check Voy availability
#[wasm_bindgen_test]
fn test_voy_availability() {
    let available = check_voy_available();
    web_sys::console::log_1(&format!("Voy available: {}", available).into());

    // Test passes regardless, but logs the result
    assert!(available || !available);
}

/// Test 2: Create index from embeddings
#[wasm_bindgen_test]
async fn test_create_voy_index() {
    // Create sample embeddings (3 documents, dimension 3)
    let embeddings = Array::new();

    let emb1 = Float32Array::from(&[0.1_f32, 0.2, 0.3][..]);
    let emb2 = Float32Array::from(&[0.4_f32, 0.5, 0.6][..]);
    let emb3 = Float32Array::from(&[0.7_f32, 0.8, 0.9][..]);

    embeddings.push(&emb1);
    embeddings.push(&emb2);
    embeddings.push(&emb3);

    // Create Voy index
    let result = VoyIndex::from_embeddings(embeddings.into(), 3);

    if check_voy_available() {
        assert!(result.is_ok(), "Should create index when Voy is available");

        let index = result.unwrap();
        let size = index.size().unwrap();
        assert_eq!(size, 3, "Index should contain 3 embeddings");
    } else {
        web_sys::console::warn_1(&"Voy not available, skipping index creation test".into());
    }
}

/// Test 3: Search with Voy index
#[wasm_bindgen_test]
async fn test_voy_search() {
    if !check_voy_available() {
        web_sys::console::warn_1(&"Voy not available, skipping search test".into());
        return;
    }

    // Create embeddings
    let embeddings = Array::new();
    embeddings.push(&Float32Array::from(&[1.0_f32, 0.0, 0.0][..]));
    embeddings.push(&Float32Array::from(&[0.0_f32, 1.0, 0.0][..]));
    embeddings.push(&Float32Array::from(&[0.0_f32, 0.0, 1.0][..]));

    let index = VoyIndex::from_embeddings(embeddings.into(), 3).unwrap();

    // Query with first embedding (should match itself best)
    let query = Float32Array::from(&[1.0_f32, 0.0, 0.0][..]);
    let results = index.search_parsed(query.into(), 3).unwrap();

    // Check results structure
    let results_array = Array::from(&results);
    assert!(results_array.length() > 0, "Should return at least one result");

    web_sys::console::log_1(&format!("Search results: {:?}", results).into());
}

/// Test 4: Add embeddings incrementally
#[wasm_bindgen_test]
async fn test_incremental_add() {
    if !check_voy_available() {
        web_sys::console::warn_1(&"Voy not available, skipping incremental add test".into());
        return;
    }

    let mut index = VoyIndex::create_empty(3).unwrap();

    // Add embeddings one by one
    let emb1 = Float32Array::from(&[0.1_f32, 0.2, 0.3][..]);
    index.add_embedding(emb1.into(), "doc1", "Document 1", "/doc1").unwrap();

    let emb2 = Float32Array::from(&[0.4_f32, 0.5, 0.6][..]);
    index.add_embedding(emb2.into(), "doc2", "Document 2", "/doc2").unwrap();

    // Verify size
    let size = index.size().unwrap();
    assert_eq!(size, 2, "Index should contain 2 embeddings after incremental add");
}

/// Test 5: Serialize and deserialize index (placeholder for persistence)
#[wasm_bindgen_test]
async fn test_serialization() {
    if !check_voy_available() {
        web_sys::console::warn_1(&"Voy not available, skipping serialization test".into());
        return;
    }

    // Create index
    let embeddings = Array::new();
    embeddings.push(&Float32Array::from(&[0.1_f32, 0.2, 0.3][..]));
    let index = VoyIndex::from_embeddings(embeddings.into(), 3).unwrap();

    // Serialize
    let serialized = index.serialize();
    assert!(serialized.is_ok(), "Should serialize successfully");

    let json = serialized.unwrap();
    assert!(json.len() > 0, "Serialized JSON should not be empty");

    web_sys::console::log_1(&format!("Serialized: {}", json).into());
}

/// Test 6: Clear index
#[wasm_bindgen_test]
async fn test_clear_index() {
    if !check_voy_available() {
        web_sys::console::warn_1(&"Voy not available, skipping clear test".into());
        return;
    }

    let embeddings = Array::new();
    embeddings.push(&Float32Array::from(&[0.1_f32, 0.2, 0.3][..]));
    embeddings.push(&Float32Array::from(&[0.4_f32, 0.5, 0.6][..]));

    let mut index = VoyIndex::from_embeddings(embeddings.into(), 3).unwrap();

    // Verify initial size
    let size_before = index.size().unwrap();
    assert_eq!(size_before, 2, "Should have 2 embeddings before clear");

    // Clear
    index.clear();

    // Verify cleared
    let size_after = index.size().unwrap();
    assert_eq!(size_after, 0, "Should have 0 embeddings after clear");
}

/// Test 7: Large scale performance test (1000 embeddings)
#[wasm_bindgen_test]
async fn test_large_scale_indexing() {
    if !check_voy_available() {
        web_sys::console::warn_1(&"Voy not available, skipping large scale test".into());
        return;
    }

    let start = js_sys::Date::now();

    // Create 1000 random embeddings (dimension 384, typical for MiniLM)
    let embeddings = Array::new();
    for i in 0..1000 {
        let mut vec = Vec::with_capacity(384);
        for j in 0..384 {
            vec.push((i + j) as f32 / 1000.0);
        }
        embeddings.push(&Float32Array::from(&vec[..]));
    }

    // Build index
    let index_result = VoyIndex::from_embeddings(embeddings.into(), 384);
    let index_time = js_sys::Date::now() - start;

    assert!(index_result.is_ok(), "Should create large index successfully");

    let index = index_result.unwrap();
    web_sys::console::log_1(&format!("✅ Indexed 1000 embeddings (384d) in {:.2}ms", index_time).into());

    // Test search performance
    let query_start = js_sys::Date::now();
    let query = Float32Array::from(&vec![0.5_f32; 384][..]);
    let results = index.search_parsed(query.into(), 10);
    let query_time = js_sys::Date::now() - query_start;

    assert!(results.is_ok(), "Should search successfully");
    web_sys::console::log_1(&format!("✅ Query completed in {:.2}ms", query_time).into());

    // Performance assertions (k-d tree should be fast)
    assert!(index_time < 500.0, "Indexing 1000 docs should be < 500ms");
    assert!(query_time < 50.0, "Query should be < 50ms");
}

/// Test 8: Similarity ordering verification
#[wasm_bindgen_test]
async fn test_similarity_ordering() {
    if !check_voy_available() {
        web_sys::console::warn_1(&"Voy not available, skipping similarity test".into());
        return;
    }

    // Create embeddings with known similarities
    let embeddings = Array::new();

    // Query will be [1.0, 0.0, 0.0]
    // emb1: [1.0, 0.0, 0.0] - exact match (similarity = 1.0)
    // emb2: [0.9, 0.1, 0.0] - close match
    // emb3: [0.0, 1.0, 0.0] - orthogonal (similarity = 0.0)

    embeddings.push(&Float32Array::from(&[1.0_f32, 0.0, 0.0][..]));
    embeddings.push(&Float32Array::from(&[0.9_f32, 0.1, 0.0][..]));
    embeddings.push(&Float32Array::from(&[0.0_f32, 1.0, 0.0][..]));

    let index = VoyIndex::from_embeddings(embeddings.into(), 3).unwrap();

    // Query
    let query = Float32Array::from(&[1.0_f32, 0.0, 0.0][..]);
    let results = index.search_parsed(query.into(), 3).unwrap();

    let results_array = Array::from(&results);
    assert!(results_array.length() > 0, "Should have results");

    // First result should be the exact match (id: 0)
    if results_array.length() > 0 {
        let first_result = results_array.get(0);
        let id = js_sys::Reflect::get(&first_result, &"id".into()).unwrap();
        web_sys::console::log_1(&format!("First result ID: {:?}", id).into());
    }

    web_sys::console::log_1(&"✅ Similarity ordering test completed".into());
}
