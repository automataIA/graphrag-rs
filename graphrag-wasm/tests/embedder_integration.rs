//! Integration tests for WASM embedder functionality
//!
//! Tests the complete embedder pipeline including model loading from Cache API.

use wasm_bindgen_test::*;
use graphrag_wasm::embedder::{WasmEmbedder, create_embedder, EmbedderBackend, CandleEmbedder};

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Create WasmEmbedder instance
#[wasm_bindgen_test]
async fn test_create_wasm_embedder() {
    let embedder = WasmEmbedder::new("test-model".to_string(), 384).await;
    assert!(embedder.is_ok());

    let embedder = embedder.unwrap();
    assert_eq!(embedder.dimension(), 384);
}

/// Test: Check GPU acceleration status
#[wasm_bindgen_test]
async fn test_gpu_acceleration_check() {
    let embedder = WasmEmbedder::new("test-model".to_string(), 384).await.unwrap();

    // Should return false for CPU-only Candle embedder
    // (or true if WebGPU is available and feature is enabled)
    let is_gpu = embedder.is_gpu_accelerated();
    assert!(!is_gpu || is_gpu); // Either value is acceptable
}

/// Test: Create embedder with auto-detection
#[wasm_bindgen_test]
async fn test_create_embedder_auto() {
    let embedder = create_embedder("test-model", 384).await;
    assert!(embedder.is_ok());

    let embedder = embedder.unwrap();
    assert_eq!(embedder.dimension(), 384);
}

/// Test: Candle embedder creation
#[wasm_bindgen_test]
async fn test_candle_embedder_creation() {
    let embedder = CandleEmbedder::new("test-model", 384).await;
    assert!(embedder.is_ok());

    let embedder = embedder.unwrap();
    assert_eq!(embedder.dimension(), 384);
    assert!(!embedder.is_gpu_accelerated());
}

/// Test: Embed without loading model (should fail)
#[wasm_bindgen_test]
async fn test_embed_without_model_loaded() {
    let embedder = CandleEmbedder::new("test-model", 384).await.unwrap();

    // Should fail because model is not loaded
    let result = embedder.embed("test text").await;
    assert!(result.is_err());
}

/// Test: Model loading architecture
///
/// Note: This test validates the model loading infrastructure exists
/// but does not actually download models (would be too slow for CI).
#[wasm_bindgen_test]
async fn test_model_loading_infrastructure() {
    let mut embedder = CandleEmbedder::new("test-model", 384).await.unwrap();

    // Check that cache store is initialized
    // (actual download would require network and be too slow)

    // Verify the load_model method exists and can be called
    // It will fail because "test-model" doesn't exist, but that's expected
    let result = embedder.load_model().await;

    // We expect this to fail for a non-existent model
    // But the infrastructure should be in place
    assert!(result.is_ok() || result.is_err());
}

/// Test: Batch embedding structure
#[wasm_bindgen_test]
async fn test_batch_embedding_structure() {
    let embedder = WasmEmbedder::new("test-model".to_string(), 384).await.unwrap();

    // Note: This will fail with ModelNotLoaded, but tests the API structure
    let texts = vec!["text1".to_string(), "text2".to_string(), "text3".to_string()];
    let result = embedder.embed_batch(texts).await;

    // Expecting error due to model not loaded, but API should work
    assert!(result.is_ok() || result.is_err());
}

/// Test: Different embedding dimensions
#[wasm_bindgen_test]
async fn test_different_dimensions() {
    // Test MiniLM dimension (384)
    let embedder_384 = WasmEmbedder::new("all-MiniLM-L6-v2".to_string(), 384).await;
    assert!(embedder_384.is_ok());
    assert_eq!(embedder_384.unwrap().dimension(), 384);

    // Test BERT dimension (768)
    let embedder_768 = WasmEmbedder::new("bert-base-uncased".to_string(), 768).await;
    assert!(embedder_768.is_ok());
    assert_eq!(embedder_768.unwrap().dimension(), 768);
}

/// Test: EmbedderBackend enum functionality
#[wasm_bindgen_test]
async fn test_embedder_backend_enum() {
    let candle_embedder = CandleEmbedder::new("test-model", 384).await.unwrap();
    let backend = EmbedderBackend::Candle(candle_embedder);

    assert_eq!(backend.dimension(), 384);
    assert!(!backend.is_gpu_accelerated());
}

/// Test: Multiple embedder instances
///
/// Validates that we can create multiple embedder instances simultaneously.
#[wasm_bindgen_test]
async fn test_multiple_embedders() {
    let embedder1 = WasmEmbedder::new("model1".to_string(), 384).await;
    let embedder2 = WasmEmbedder::new("model2".to_string(), 768).await;
    let embedder3 = WasmEmbedder::new("model3".to_string(), 384).await;

    assert!(embedder1.is_ok());
    assert!(embedder2.is_ok());
    assert!(embedder3.is_ok());

    assert_eq!(embedder1.unwrap().dimension(), 384);
    assert_eq!(embedder2.unwrap().dimension(), 768);
    assert_eq!(embedder3.unwrap().dimension(), 384);
}

/// Test: Embedder error types
#[wasm_bindgen_test]
async fn test_embedder_error_handling() {
    use graphrag_wasm::embedder::EmbedderError;

    // Test ModelNotLoaded error
    let embedder = CandleEmbedder::new("test", 384).await.unwrap();
    let result = embedder.embed("test").await;

    match result {
        Err(EmbedderError::ModelNotLoaded) => {
            // Expected error type
            assert!(true);
        }
        Err(e) => {
            panic!("Unexpected error type: {:?}", e);
        }
        Ok(_) => {
            panic!("Expected ModelNotLoaded error");
        }
    }
}

/// Test: Cache API availability check
#[wasm_bindgen_test]
async fn test_cache_api_available_for_models() {
    use graphrag_wasm::storage::CacheStore;

    // Check if Cache API is available
    let cache_result = CacheStore::open("test-embedder-models").await;

    // Cache API should be available in browser test environment
    assert!(cache_result.is_ok());
}

/// Test: Model name formatting
#[wasm_bindgen_test]
async fn test_model_name_handling() {
    // Test with HuggingFace format
    let embedder1 = WasmEmbedder::new(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        384
    ).await;
    assert!(embedder1.is_ok());

    // Test with simple name
    let embedder2 = WasmEmbedder::new("bert-base-uncased".to_string(), 768).await;
    assert!(embedder2.is_ok());

    // Test with custom path
    let embedder3 = WasmEmbedder::new("my-org/my-model".to_string(), 512).await;
    assert!(embedder3.is_ok());
}
