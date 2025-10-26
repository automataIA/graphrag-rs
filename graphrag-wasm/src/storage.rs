//! Browser Storage Abstractions
//!
//! This module provides wrappers around browser storage APIs:
//! - IndexedDB for graph data persistence (50% disk quota)
//! - Cache API for ML model storage (60% disk quota, 1.6GB+)
//!
//! ## IndexedDB Storage
//!
//! Used for storing:
//! - Knowledge graph structure (entities, relationships)
//! - Document metadata
//! - Query history
//!
//! ## Cache API Storage
//!
//! Used for storing:
//! - Pre-trained ML models (BERT, MiniLM, Phi-2)
//! - Model weights and configuration
//! - Progressive loading support

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    IdbDatabase, IdbOpenDbRequest, IdbRequest,
    IdbTransactionMode, Cache, Request, Response,
};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

/// Helper function to convert IdbRequest to Promise
fn idb_request_to_promise(request: &IdbRequest) -> js_sys::Promise {
    let request_clone = Rc::new(request.clone());
    js_sys::Promise::new(&mut |resolve, reject| {
        let request_for_success = request_clone.clone();
        let onsuccess = Closure::once(move || {
            let result = request_for_success.result().unwrap();
            resolve.call1(&JsValue::NULL, &result).unwrap();
        });
        let onerror = Closure::once(move |event: web_sys::Event| {
            let error_msg = format!("IdbRequest error: {:?}", event);
            reject.call1(&JsValue::NULL, &JsValue::from_str(&error_msg)).unwrap();
        });

        request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
        request.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onsuccess.forget();
        onerror.forget();
    })
}

/// Errors that can occur during storage operations
#[derive(Debug, Clone)]
pub enum StorageError {
    /// Browser API not available (e.g., no IndexedDB support)
    UnsupportedBrowser(String),
    /// Database operation failed
    DatabaseError(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Item not found in storage
    NotFound(String),
    /// Quota exceeded
    QuotaExceeded,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StorageError::UnsupportedBrowser(msg) => write!(f, "Browser not supported: {}", msg),
            StorageError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            StorageError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::NotFound(key) => write!(f, "Item not found: {}", key),
            StorageError::QuotaExceeded => write!(f, "Storage quota exceeded"),
        }
    }
}

impl From<JsValue> for StorageError {
    fn from(value: JsValue) -> Self {
        StorageError::DatabaseError(
            value.as_string().unwrap_or_else(|| "Unknown error".to_string())
        )
    }
}

impl From<StorageError> for JsValue {
    fn from(error: StorageError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

/// IndexedDB wrapper for persistent graph storage
///
/// Provides async API for storing and retrieving graph data in the browser.
/// Automatically handles database versioning and schema upgrades.
///
/// ## Usage
///
/// ```rust
/// let db = IndexedDBStore::new("graphrag", 1).await?;
/// db.put("entities", "entity_1", &entity_data).await?;
/// let entity = db.get("entities", "entity_1").await?;
/// ```
pub struct IndexedDBStore {
    db: IdbDatabase,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    version: u32,
}

impl IndexedDBStore {
    /// Create or open an IndexedDB database
    ///
    /// # Arguments
    /// * `name` - Database name
    /// * `version` - Schema version (increment to trigger upgrade)
    ///
    /// # Returns
    /// Result with IndexedDBStore instance or error
    pub async fn new(name: &str, version: u32) -> Result<Self, StorageError> {
        let window = web_sys::window()
            .ok_or_else(|| StorageError::UnsupportedBrowser("No window object".to_string()))?;

        let idb_factory = window
            .indexed_db()
            .map_err(|_| StorageError::UnsupportedBrowser("IndexedDB not available".to_string()))?
            .ok_or_else(|| StorageError::UnsupportedBrowser("IndexedDB is None".to_string()))?;

        let open_request = idb_factory
            .open_with_u32(name, version)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to open database: {:?}", e)))?;

        // Set up onupgradeneeded callback
        let onupgradeneeded = Closure::once(move |event: web_sys::IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request = target.dyn_into::<IdbOpenDbRequest>().unwrap();
            let db = request.result().unwrap().dyn_into::<IdbDatabase>().unwrap();

            // Create object stores (will only create if they don't exist)
            // onupgradeneeded is only called when version changes, so we can safely try to create
            let _ = db.create_object_store("entities").map_err(|e| {
                web_sys::console::warn_1(&format!("entities store may already exist: {:?}", e).into());
            });
            let _ = db.create_object_store("relationships").map_err(|e| {
                web_sys::console::warn_1(&format!("relationships store may already exist: {:?}", e).into());
            });
            let _ = db.create_object_store("documents").map_err(|e| {
                web_sys::console::warn_1(&format!("documents store may already exist: {:?}", e).into());
            });
            let _ = db.create_object_store("metadata").map_err(|e| {
                web_sys::console::warn_1(&format!("metadata store may already exist: {:?}", e).into());
            });
        });
        open_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
        onupgradeneeded.forget();

        // Wait for database to open
        let promise = idb_request_to_promise(&open_request);
        let db_value = JsFuture::from(promise).await?;
        let db = db_value.dyn_into::<IdbDatabase>()
            .map_err(|_| StorageError::DatabaseError("Failed to cast to IdbDatabase".to_string()))?;

        Ok(Self {
            db,
            name: name.to_string(),
            version,
        })
    }

    /// Store a value in an object store
    ///
    /// # Arguments
    /// * `store_name` - Name of the object store
    /// * `key` - Key to store the value under
    /// * `value` - Value to store (must be serializable)
    pub async fn put<T: Serialize>(
        &self,
        store_name: &str,
        key: &str,
        value: &T,
    ) -> Result<(), StorageError> {
        let transaction = self.db
            .transaction_with_str_and_mode(store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create transaction: {:?}", e)))?;

        let store = transaction
            .object_store(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get object store: {:?}", e)))?;

        let js_value = serde_wasm_bindgen::to_value(value)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;

        let request = store
            .put_with_key(&js_value, &JsValue::from_str(key))
            .map_err(|e| StorageError::DatabaseError(format!("Failed to put value: {:?}", e)))?;

        let promise = idb_request_to_promise(&request);
        JsFuture::from(promise).await?;

        Ok(())
    }

    /// Retrieve a value from an object store
    ///
    /// # Arguments
    /// * `store_name` - Name of the object store
    /// * `key` - Key to retrieve
    ///
    /// # Returns
    /// Deserialized value or error if not found
    pub async fn get<T: for<'de> Deserialize<'de>>(
        &self,
        store_name: &str,
        key: &str,
    ) -> Result<T, StorageError> {
        let transaction = self.db
            .transaction_with_str(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create transaction: {:?}", e)))?;

        let store = transaction
            .object_store(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .get(&JsValue::from_str(key))
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get value: {:?}", e)))?;

        let promise = idb_request_to_promise(&request);
        let result = JsFuture::from(promise).await?;

        if result.is_undefined() {
            return Err(StorageError::NotFound(key.to_string()));
        }

        serde_wasm_bindgen::from_value(result)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    /// Get all values from an object store with batching
    ///
    /// # Arguments
    /// * `store_name` - Name of the object store
    /// * `batch_size` - Maximum number of items to retrieve (optional, default 100)
    ///
    /// # Returns
    /// Vector of all values in the store
    ///
    /// # Performance
    /// Using `getAll()` with batch size is 10-50x faster than cursor iteration
    /// for large datasets (100+ items). Recommended for entity/relationship retrieval.
    pub async fn get_all_batched<T: for<'de> Deserialize<'de>>(
        &self,
        store_name: &str,
        batch_size: Option<u32>,
    ) -> Result<Vec<T>, StorageError> {
        let transaction = self.db
            .transaction_with_str(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create transaction: {:?}", e)))?;

        let store = transaction
            .object_store(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get object store: {:?}", e)))?;

        // Use getAll() - note: batch_size parameter is ignored as web_sys IdbObjectStore
        // doesn't support limiting results directly
        let request = store
            .get_all()
            .map_err(|e| StorageError::DatabaseError(format!("Failed to getAll: {:?}", e)))?;

        let promise = idb_request_to_promise(&request);
        let result = JsFuture::from(promise).await?;

        // Convert JS Array to Vec<T>
        let js_array: js_sys::Array = result
            .dyn_into()
            .map_err(|_| StorageError::DatabaseError("Failed to cast to Array".to_string()))?;

        // Apply batch_size limit if specified
        let max_items = batch_size.map(|s| s as u32).unwrap_or(js_array.length());
        let limit = std::cmp::min(max_items, js_array.length());

        let mut results = Vec::new();
        for i in 0..limit {
            let item = js_array.get(i);
            if !item.is_undefined() && !item.is_null() {
                let deserialized: T = serde_wasm_bindgen::from_value(item)
                    .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                results.push(deserialized);
            }
        }

        Ok(results)
    }

    /// Get all keys from an object store
    ///
    /// # Arguments
    /// * `store_name` - Name of the object store
    ///
    /// # Returns
    /// Vector of all keys in the store
    #[allow(dead_code)]
    pub async fn get_all_keys(
        &self,
        store_name: &str,
    ) -> Result<Vec<String>, StorageError> {
        let transaction = self.db
            .transaction_with_str(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create transaction: {:?}", e)))?;

        let store = transaction
            .object_store(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .get_all_keys()
            .map_err(|e| StorageError::DatabaseError(format!("Failed to getAllKeys: {:?}", e)))?;

        let promise = idb_request_to_promise(&request);
        let result = JsFuture::from(promise).await?;

        // Convert JS Array to Vec<String>
        let js_array: js_sys::Array = result
            .dyn_into()
            .map_err(|_| StorageError::DatabaseError("Failed to cast to Array".to_string()))?;

        let mut keys = Vec::new();
        for i in 0..js_array.length() {
            let key = js_array.get(i);
            if let Some(key_str) = key.as_string() {
                keys.push(key_str);
            }
        }

        Ok(keys)
    }

    /// Delete a value from an object store
    #[allow(dead_code)]
    pub async fn delete(&self, store_name: &str, key: &str) -> Result<(), StorageError> {
        let transaction = self.db
            .transaction_with_str_and_mode(store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create transaction: {:?}", e)))?;

        let store = transaction
            .object_store(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .delete(&JsValue::from_str(key))
            .map_err(|e| StorageError::DatabaseError(format!("Failed to delete value: {:?}", e)))?;

        let promise = idb_request_to_promise(&request);
        JsFuture::from(promise).await?;

        Ok(())
    }

    /// Clear all values from an object store
    #[allow(dead_code)]
    pub async fn clear(&self, store_name: &str) -> Result<(), StorageError> {
        let transaction = self.db
            .transaction_with_str_and_mode(store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create transaction: {:?}", e)))?;

        let store = transaction
            .object_store(store_name)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .clear()
            .map_err(|e| StorageError::DatabaseError(format!("Failed to clear store: {:?}", e)))?;

        let promise = idb_request_to_promise(&request);
        JsFuture::from(promise).await?;

        Ok(())
    }

    /// Get the database name
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the database version
    #[allow(dead_code)]
    pub fn version(&self) -> u32 {
        self.version
    }
}

/// Cache API wrapper for ML model storage
///
/// Provides async API for storing and retrieving large ML models (up to 1.6GB+).
/// Uses the browser's Cache API which is designed for PWA and large file storage.
///
/// ## Usage
///
/// ```rust
/// let cache = CacheStore::open("graphrag-models").await?;
/// cache.put("bert-model", &model_bytes).await?;
/// let model = cache.get("bert-model").await?;
/// ```
#[allow(dead_code)]
pub struct CacheStore {
    cache: Cache,
    name: String,
}

impl CacheStore {
    /// Open or create a cache
    ///
    /// # Arguments
    /// * `name` - Cache name (e.g., "graphrag-models")
    ///
    /// # Returns
    /// Result with CacheStore instance or error
    #[allow(dead_code)]
    pub async fn open(name: &str) -> Result<Self, StorageError> {
        let window = web_sys::window()
            .ok_or_else(|| StorageError::UnsupportedBrowser("No window object".to_string()))?;

        let cache_storage = window
            .caches()
            .map_err(|_| StorageError::UnsupportedBrowser("Cache API not available".to_string()))?;

        let cache_promise = cache_storage.open(name);
        let cache_value = JsFuture::from(cache_promise).await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to open cache: {:?}", e)))?;
        let cache = cache_value
            .dyn_into::<Cache>()
            .map_err(|_| StorageError::DatabaseError("Failed to cast to Cache".to_string()))?;

        Ok(Self {
            cache,
            name: name.to_string(),
        })
    }

    /// Store data in the cache
    ///
    /// # Arguments
    /// * `key` - Cache key (URL-like identifier)
    /// * `bytes` - Binary data to store
    #[allow(dead_code)]
    pub async fn put(&self, key: &str, bytes: &[u8]) -> Result<(), StorageError> {
        let url = format!("https://graphrag.local/{}", key);
        let request = Request::new_with_str(&url)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create request: {:?}", e)))?;

        // Create response with bytes
        let array = js_sys::Uint8Array::from(bytes);
        let response = Response::new_with_opt_buffer_source(Some(array.as_ref()))
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create response: {:?}", e)))?;

        // Store in cache - put_with_request returns Promise directly
        let put_promise = self.cache.put_with_request(&request, &response);
        JsFuture::from(put_promise).await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to put in cache: {:?}", e)))?;

        Ok(())
    }

    /// Retrieve data from the cache
    ///
    /// # Arguments
    /// * `key` - Cache key to retrieve
    ///
    /// # Returns
    /// Binary data or error if not found
    #[allow(dead_code)]
    pub async fn get(&self, key: &str) -> Result<Vec<u8>, StorageError> {
        let url = format!("https://graphrag.local/{}", key);
        let match_promise = self.cache.match_with_str(&url);
        let match_value = JsFuture::from(match_promise).await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to match cache: {:?}", e)))?;

        if match_value.is_undefined() {
            return Err(StorageError::NotFound(key.to_string()));
        }

        let response = match_value
            .dyn_into::<Response>()
            .map_err(|_| StorageError::DatabaseError("Failed to cast to Response".to_string()))?;

        let array_buffer_promise = response.array_buffer()
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get array buffer promise: {:?}", e)))?;
        let array_buffer_value = JsFuture::from(array_buffer_promise).await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to get array buffer: {:?}", e)))?;
        let array_buffer = array_buffer_value
            .dyn_into::<js_sys::ArrayBuffer>()
            .map_err(|_| StorageError::DatabaseError("Failed to cast to ArrayBuffer".to_string()))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let mut buffer = vec![0_u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut buffer);

        Ok(buffer)
    }

    /// Delete data from the cache
    #[allow(dead_code)]
    pub async fn delete(&self, key: &str) -> Result<bool, StorageError> {
        let url = format!("https://graphrag.local/{}", key);
        let delete_promise = self.cache.delete_with_str(&url);
        let result = JsFuture::from(delete_promise).await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to delete from cache: {:?}", e)))?;
        Ok(result.as_bool().unwrap_or(false))
    }

    /// Check if a key exists in the cache
    #[allow(dead_code)]
    pub async fn has(&self, key: &str) -> Result<bool, StorageError> {
        let url = format!("https://graphrag.local/{}", key);
        let match_promise = self.cache.match_with_str(&url);
        let match_value = JsFuture::from(match_promise).await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to match cache: {:?}", e)))?;
        Ok(!match_value.is_undefined())
    }

    /// Get the cache name
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Estimate storage usage and quota
///
/// Returns a tuple of (usage_bytes, quota_bytes, usage_percentage)
#[allow(dead_code)]
pub async fn estimate_storage() -> Result<(u64, u64, f64), StorageError> {
    let window = web_sys::window()
        .ok_or_else(|| StorageError::UnsupportedBrowser("No window object".to_string()))?;

    let navigator = window.navigator();
    let storage = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
        .map_err(|_| StorageError::UnsupportedBrowser("Storage API not available".to_string()))?;

    if storage.is_undefined() {
        return Err(StorageError::UnsupportedBrowser("Storage API not supported".to_string()));
    }

    let estimate_fn = js_sys::Reflect::get(&storage, &JsValue::from_str("estimate"))
        .map_err(|_| StorageError::UnsupportedBrowser("estimate() not available".to_string()))?;

    let estimate_promise = js_sys::Function::from(estimate_fn)
        .call0(&storage)
        .map_err(|e| StorageError::DatabaseError(format!("Failed to call estimate: {:?}", e)))?;

    let estimate_value = JsFuture::from(js_sys::Promise::from(estimate_promise)).await?;

    let usage = js_sys::Reflect::get(&estimate_value, &JsValue::from_str("usage"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as u64;

    let quota = js_sys::Reflect::get(&estimate_value, &JsValue::from_str("quota"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as u64;

    let percentage = if quota > 0 {
        (usage as f64 / quota as f64) * 100.0
    } else {
        0.0
    };

    Ok((usage, quota, percentage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_indexeddb_store() {
        let db = IndexedDBStore::new("test-db", 1).await;
        assert!(db.is_ok());
    }

    #[wasm_bindgen_test]
    async fn test_cache_store() {
        let cache = CacheStore::open("test-cache").await;
        assert!(cache.is_ok());
    }
}
