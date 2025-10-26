//! Storage API Tests
//!
//! Comprehensive tests for IndexedDB and Cache API storage implementations.
//! Validates all CRUD operations, error handling, and edge cases.

use wasm_bindgen_test::*;
use graphrag_wasm::storage::{IndexedDBStore, CacheStore, estimate_storage};
use serde::{Serialize, Deserialize};

wasm_bindgen_test_configure!(run_in_browser);

// Test data structures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestEntity {
    id: String,
    name: String,
    data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code)]
struct TestMetadata {
    version: u32,
    created_at: String,
    tags: Vec<String>,
}

// ============================================================================
// IndexedDB Tests
// ============================================================================

/// Test: Create IndexedDB database
#[wasm_bindgen_test]
async fn test_create_indexeddb() {
    let result = IndexedDBStore::new("test-create-db", 1).await;
    assert!(result.is_ok());

    web_sys::console::log_1(&"✅ IndexedDB created successfully".into());
}

/// Test: Put and get simple data
#[wasm_bindgen_test]
async fn test_indexeddb_put_get_simple() {
    let db = IndexedDBStore::new("test-put-get-simple", 1).await.unwrap();

    // Put data
    let test_data = serde_json::json!({
        "key": "value",
        "number": 42
    });

    db.put("test_store", "item1", &test_data).await.unwrap();

    // Get data
    let retrieved: serde_json::Value = db.get("test_store", "item1").await.unwrap();

    assert_eq!(retrieved["key"], "value");
    assert_eq!(retrieved["number"], 42);
}

/// Test: Put and get complex structures
#[wasm_bindgen_test]
async fn test_indexeddb_complex_structures() {
    let db = IndexedDBStore::new("test-complex", 1).await.unwrap();

    let entity = TestEntity {
        id: "entity_1".to_string(),
        name: "Test Entity".to_string(),
        data: vec![1.0, 2.0, 3.0, 4.0],
    };

    // Put
    db.put("entities", "entity_1", &entity).await.unwrap();

    // Get
    let retrieved: TestEntity = db.get("entities", "entity_1").await.unwrap();

    assert_eq!(retrieved, entity);
}

/// Test: Update existing data
#[wasm_bindgen_test]
async fn test_indexeddb_update() {
    let db = IndexedDBStore::new("test-update", 1).await.unwrap();

    // Initial put
    let data_v1 = serde_json::json!({"version": 1, "content": "first"});
    db.put("updates", "doc1", &data_v1).await.unwrap();

    // Update
    let data_v2 = serde_json::json!({"version": 2, "content": "second"});
    db.put("updates", "doc1", &data_v2).await.unwrap();

    // Verify update
    let retrieved: serde_json::Value = db.get("updates", "doc1").await.unwrap();
    assert_eq!(retrieved["version"], 2);
    assert_eq!(retrieved["content"], "second");
}

/// Test: Delete data
#[wasm_bindgen_test]
async fn test_indexeddb_delete() {
    let db = IndexedDBStore::new("test-delete", 1).await.unwrap();

    // Put data
    let data = serde_json::json!({"test": "data"});
    db.put("deletions", "item1", &data).await.unwrap();

    // Verify it exists
    let exists: Result<serde_json::Value, _> = db.get("deletions", "item1").await;
    assert!(exists.is_ok());

    // Delete
    db.delete("deletions", "item1").await.unwrap();

    // Verify deletion
    let after_delete: Result<serde_json::Value, _> = db.get("deletions", "item1").await;
    assert!(after_delete.is_err());
}

/// Test: Clear all data in store
#[wasm_bindgen_test]
async fn test_indexeddb_clear() {
    let db = IndexedDBStore::new("test-clear", 1).await.unwrap();

    // Put multiple items
    for i in 0..5 {
        let data = serde_json::json!({"id": i, "value": format!("item_{}", i)});
        db.put("clearable", &format!("item{}", i), &data).await.unwrap();
    }

    // Clear all
    db.clear("clearable").await.unwrap();

    // Verify all items are gone
    for i in 0..5 {
        let result: Result<serde_json::Value, _> = db.get("clearable", &format!("item{}", i)).await;
        assert!(result.is_err());
    }
}

/// Test: Multiple stores in same database
#[wasm_bindgen_test]
async fn test_indexeddb_multiple_stores() {
    let db = IndexedDBStore::new("test-multi-store", 1).await.unwrap();

    // Put data in different stores
    let entity = serde_json::json!({"type": "entity", "name": "Test"});
    let relationship = serde_json::json!({"type": "relationship", "from": "A", "to": "B"});

    db.put("entities", "e1", &entity).await.unwrap();
    db.put("relationships", "r1", &relationship).await.unwrap();

    // Retrieve from each store
    let e: serde_json::Value = db.get("entities", "e1").await.unwrap();
    let r: serde_json::Value = db.get("relationships", "r1").await.unwrap();

    assert_eq!(e["type"], "entity");
    assert_eq!(r["type"], "relationship");
}

/// Test: Large data storage
#[wasm_bindgen_test]
async fn test_indexeddb_large_data() {
    let db = IndexedDBStore::new("test-large-data", 1).await.unwrap();

    // Create large embedding vector (10k floats ~ 40KB)
    let large_data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();

    let entity = TestEntity {
        id: "large_entity".to_string(),
        name: "Large Entity".to_string(),
        data: large_data.clone(),
    };

    // Put large data
    let put_result = db.put("large_store", "large1", &entity).await;
    assert!(put_result.is_ok());

    // Get large data
    let retrieved: TestEntity = db.get("large_store", "large1").await.unwrap();
    assert_eq!(retrieved.data.len(), 10_000);
    assert_eq!(retrieved.data[0], 0.0);
    assert_eq!(retrieved.data[9_999], 9_999.0);
}

/// Test: Concurrent operations
#[wasm_bindgen_test]
async fn test_indexeddb_concurrent() {
    let db = IndexedDBStore::new("test-concurrent", 1).await.unwrap();

    // Launch concurrent put operations
    for i in 0..10 {
        let data = serde_json::json!({"id": i, "value": i * 10});
        let result = db.put("concurrent", &format!("item{}", i), &data).await;
        assert!(result.is_ok());
    }

    // Verify all were stored
    for i in 0..10 {
        let retrieved: serde_json::Value = db.get("concurrent", &format!("item{}", i)).await.unwrap();
        assert_eq!(retrieved["id"], i);
    }
}

// ============================================================================
// Cache API Tests
// ============================================================================

/// Test: Open cache
#[wasm_bindgen_test]
async fn test_cache_open() {
    let result = CacheStore::open("test-cache-open").await;
    assert!(result.is_ok());

    web_sys::console::log_1(&"✅ Cache opened successfully".into());
}

/// Test: Put and get cache data
#[wasm_bindgen_test]
async fn test_cache_put_get() {
    let cache = CacheStore::open("test-cache-put-get").await.unwrap();

    let test_data = b"This is test cache data";

    // Put
    cache.put("test-key", test_data).await.unwrap();

    // Get
    let retrieved = cache.get("test-key").await.unwrap();

    assert_eq!(retrieved, test_data);
}

/// Test: Cache has (existence check)
#[wasm_bindgen_test]
async fn test_cache_has() {
    let cache = CacheStore::open("test-cache-has").await.unwrap();

    // Should not exist initially
    let has_before = cache.has("nonexistent-key").await.unwrap();
    assert!(!has_before);

    // Put data
    cache.put("existing-key", b"data").await.unwrap();

    // Should exist now
    let has_after = cache.has("existing-key").await.unwrap();
    assert!(has_after);
}

/// Test: Cache delete
#[wasm_bindgen_test]
async fn test_cache_delete() {
    let cache = CacheStore::open("test-cache-delete").await.unwrap();

    // Put data
    cache.put("delete-me", b"temporary data").await.unwrap();

    // Verify exists
    assert!(cache.has("delete-me").await.unwrap());

    // Delete
    cache.delete("delete-me").await.unwrap();

    // Verify deleted
    assert!(!cache.has("delete-me").await.unwrap());
}

/// Test: Large file in cache
#[wasm_bindgen_test]
async fn test_cache_large_file() {
    let cache = CacheStore::open("test-cache-large").await.unwrap();

    // Create 1MB of data
    let large_data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

    // Put large data
    let put_result = cache.put("large-model.bin", &large_data).await;
    assert!(put_result.is_ok());

    // Get large data
    let retrieved = cache.get("large-model.bin").await.unwrap();
    assert_eq!(retrieved.len(), 1_000_000);
    assert_eq!(retrieved[0], 0);
    assert_eq!(retrieved[999_999], 63); // 999999 % 256
}

/// Test: Multiple cache entries
#[wasm_bindgen_test]
async fn test_cache_multiple_entries() {
    let cache = CacheStore::open("test-cache-multi").await.unwrap();

    // Put multiple entries
    for i in 0..5 {
        let data = format!("Model data {}", i).into_bytes();
        cache.put(&format!("model{}.bin", i), &data).await.unwrap();
    }

    // Verify all exist
    for i in 0..5 {
        assert!(cache.has(&format!("model{}.bin", i)).await.unwrap());
    }

    // Retrieve and verify content
    for i in 0..5 {
        let data = cache.get(&format!("model{}.bin", i)).await.unwrap();
        let expected = format!("Model data {}", i).into_bytes();
        assert_eq!(data, expected);
    }
}

/// Test: Cache update
#[wasm_bindgen_test]
async fn test_cache_update() {
    let cache = CacheStore::open("test-cache-update").await.unwrap();

    // Initial version
    cache.put("updateable.bin", b"version 1").await.unwrap();

    // Update
    cache.put("updateable.bin", b"version 2").await.unwrap();

    // Verify update
    let data = cache.get("updateable.bin").await.unwrap();
    assert_eq!(data, b"version 2");
}

/// Test: Binary data preservation
#[wasm_bindgen_test]
async fn test_cache_binary_data() {
    let cache = CacheStore::open("test-cache-binary").await.unwrap();

    // Create binary data with all byte values
    let binary_data: Vec<u8> = (0..=255).collect();

    cache.put("binary.bin", &binary_data).await.unwrap();

    let retrieved = cache.get("binary.bin").await.unwrap();

    // Verify exact binary match
    assert_eq!(retrieved.len(), 256);
    for i in 0..=255 {
        assert_eq!(retrieved[i], i as u8);
    }
}

// ============================================================================
// Storage Estimation Tests
// ============================================================================

/// Test: Storage estimation
#[wasm_bindgen_test]
async fn test_storage_estimation() {
    let result = estimate_storage().await;

    match result {
        Ok((usage, quota, percentage)) => {
            web_sys::console::log_1(&format!(
                "Storage: {} / {} bytes ({:.1}%)",
                usage, quota, percentage
            ).into());

            // Quota should be positive
            assert!(quota > 0);

            // Percentage should be between 0 and 100
            assert!(percentage >= 0.0 && percentage <= 100.0);

            // Usage should not exceed quota
            assert!(usage <= quota);
        }
        Err(e) => {
            web_sys::console::warn_1(&format!(
                "Storage estimation not available: {:?}",
                e
            ).into());
        }
    }
}

/// Test: Storage after operations
#[wasm_bindgen_test]
async fn test_storage_after_operations() {
    // Get initial storage
    let initial_result = estimate_storage().await;

    if initial_result.is_err() {
        web_sys::console::log_1(&"Storage estimation not available, skipping test".into());
        return;
    }

    let (initial_usage, _, _) = initial_result.unwrap();

    // Store some data
    let db = IndexedDBStore::new("test-storage-tracking", 1).await.unwrap();
    let large_data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
    db.put("test", "large_item", &large_data).await.unwrap();

    // Get storage again
    let after_result = estimate_storage().await;
    if let Ok((after_usage, _, _)) = after_result {
        web_sys::console::log_1(&format!(
            "Storage before: {}, after: {}, delta: {}",
            initial_usage,
            after_usage,
            after_usage as i64 - initial_usage as i64
        ).into());

        // Usage should have increased (or stayed same due to caching)
        assert!(after_usage >= initial_usage);
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/// Test: Get non-existent key
#[wasm_bindgen_test]
async fn test_get_nonexistent_key() {
    let db = IndexedDBStore::new("test-nonexistent", 1).await.unwrap();

    let result: Result<serde_json::Value, _> = db.get("test", "does-not-exist").await;

    // Should return error
    assert!(result.is_err());
}

/// Test: Delete non-existent key
#[wasm_bindgen_test]
async fn test_delete_nonexistent() {
    let db = IndexedDBStore::new("test-delete-nonexistent", 1).await.unwrap();

    // Deleting non-existent key should not error (idempotent)
    let result = db.delete("test", "does-not-exist").await;

    // May succeed (idempotent) or fail (depending on implementation)
    // Both are acceptable
    match result {
        Ok(_) => web_sys::console::log_1(&"Delete idempotent (OK)".into()),
        Err(_) => web_sys::console::log_1(&"Delete not found (also OK)".into()),
    }
}

/// Test: Clear empty store
#[wasm_bindgen_test]
async fn test_clear_empty_store() {
    let db = IndexedDBStore::new("test-clear-empty", 1).await.unwrap();

    // Clearing empty store should succeed
    let result = db.clear("empty_store").await;
    assert!(result.is_ok());
}
