//! JavaScript bindings for Voy vector search engine
//!
//! Voy is a WebAssembly semantic search engine (npm: voy-search v0.6+)
//! This module provides Rust bindings to the modern Voy JavaScript API.
//!
//! ## Usage
//!
//! First, include Voy in your HTML:
//! ```html
//! <script type="module">
//!   import { Voy } from "https://cdn.jsdelivr.net/npm/voy-search@0.6.3/dist/voy.js";
//!   window.Voy = Voy;
//! </script>
//! ```
//!
//! Then use from Rust/WASM:
//! ```rust
//! let index = VoyIndex::from_embeddings(embeddings, 384)?;
//! let results = index.search_parsed(query, 5)?;
//! ```

use wasm_bindgen::prelude::*;
use web_sys::console;

/// Voy vector search index
///
/// This wraps the JavaScript Voy class which uses k-d trees
/// for efficient nearest neighbor search in the browser.
///
/// Modern API (v0.6+):
/// - 75KB bundle size
/// - k-d tree algorithm
/// - Cosine similarity search
/// - Serialization support
#[wasm_bindgen]
pub struct VoyIndex {
    inner: VoyClass,
}

// WASM runs in a single-threaded environment, so Send + Sync are safe
// These are required for Leptos 0.8 signals
unsafe impl Send for VoyIndex {}
unsafe impl Sync for VoyIndex {}

/// Configuration for creating a Voy index
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct VoyConfig {
    /// Embedding dimension
    pub dimension: usize,
}

#[wasm_bindgen]
impl VoyConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

/// Result from a Voy search query
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct VoySearchResult {
    /// Index of the matching item
    pub id: usize,
    /// Distance to the query vector
    pub distance: f32,
}

/// JavaScript Voy module interface (Modern API v0.6+)
///
/// Uses the Voy class constructor pattern:
/// ```javascript
/// import { Voy } from "voy-search";
/// const index = new Voy(resource);
/// index.search(query, k);
/// ```
#[wasm_bindgen]
extern "C" {
    /// Voy class from voy-search npm package
    #[wasm_bindgen(js_namespace = window, js_name = Voy)]
    pub type VoyClass;

    /// Create new Voy index with resource
    #[wasm_bindgen(constructor, js_namespace = window, js_class = Voy)]
    fn new(resource: JsValue) -> VoyClass;

    /// Search for k nearest neighbors
    #[wasm_bindgen(method, js_namespace = window, js_class = Voy)]
    fn search(this: &VoyClass, query: JsValue, k: usize) -> JsValue;

    /// Add resource to index
    #[wasm_bindgen(method, js_namespace = window, js_class = Voy)]
    fn add(this: &VoyClass, resource: JsValue) -> JsValue;

    /// Remove resource from index
    #[wasm_bindgen(method, js_namespace = window, js_class = Voy)]
    fn remove(this: &VoyClass, resource: JsValue) -> JsValue;

    /// Clear all resources
    #[wasm_bindgen(method, js_namespace = window, js_class = Voy)]
    fn clear(this: &VoyClass);

    /// Serialize index to JSON
    #[wasm_bindgen(method, js_namespace = window, js_class = Voy)]
    fn serialize(this: &VoyClass) -> JsValue;
}

#[wasm_bindgen]
impl VoyIndex {
    /// Create a new Voy index from embeddings
    ///
    /// # Arguments
    /// * `embeddings` - Array of embeddings (Float32Array or nested arrays)
    /// * `dimension` - Embedding dimension
    ///
    /// # Example
    /// ```javascript
    /// const embeddings = [
    ///   { id: "0", title: "doc1", url: "/0", embeddings: [0.1, 0.2, 0.3] },
    ///   { id: "1", title: "doc2", url: "/1", embeddings: [0.4, 0.5, 0.6] }
    /// ];
    /// const index = VoyIndex.from_embeddings(embeddings, 3);
    /// ```
    #[wasm_bindgen(js_name = "fromEmbeddings")]
    pub fn from_embeddings(embeddings: JsValue, dimension: usize) -> Result<VoyIndex, JsValue> {
        // Call the static fromEmbeddings method from our SimpleVectorSearch class
        let window = web_sys::window().ok_or("No window object")?;
        let voy_class = js_sys::Reflect::get(&window, &"Voy".into())?;

        // Call static method: Voy.fromEmbeddings(embeddings, dimension)
        let from_embeddings_fn = js_sys::Reflect::get(&voy_class, &"fromEmbeddings".into())?;
        let from_embeddings_fn: &js_sys::Function = from_embeddings_fn.unchecked_ref();
        let inner = js_sys::Reflect::apply(
            from_embeddings_fn,
            &JsValue::NULL,
            &js_sys::Array::of2(&embeddings, &JsValue::from_f64(dimension as f64))
        )?;

        // Cast to VoyClass
        let inner = inner.unchecked_into::<VoyClass>();

        console::log_1(&format!("✅ Created vector index with dimension {}", dimension).into());

        Ok(VoyIndex { inner })
    }

    /// Create an empty Voy index
    #[wasm_bindgen(js_name = "createEmpty")]
    pub fn create_empty(_dimension: usize) -> Result<VoyIndex, JsValue> {
        // Create empty resource
        let resource = js_sys::Object::new();
        js_sys::Reflect::set(&resource, &"embeddings".into(), &js_sys::Array::new())?;

        let inner = VoyClass::new(resource.into());

        console::log_1(&"Created empty Voy index".into());

        Ok(VoyIndex { inner })
    }

    /// Add a single embedding to the index
    ///
    /// # Arguments
    /// * `embedding` - Vector embedding as Float32Array
    /// * `id` - Document ID
    /// * `title` - Document title
    /// * `url` - Document URL
    pub fn add_embedding(
        &mut self,
        embedding: JsValue,
        id: &str,
        title: &str,
        url: &str
    ) -> Result<(), JsValue> {
        // Create resource object
        let resource_obj = js_sys::Object::new();
        js_sys::Reflect::set(&resource_obj, &"id".into(), &JsValue::from_str(id))?;
        js_sys::Reflect::set(&resource_obj, &"title".into(), &JsValue::from_str(title))?;
        js_sys::Reflect::set(&resource_obj, &"url".into(), &JsValue::from_str(url))?;
        js_sys::Reflect::set(&resource_obj, &"embeddings".into(), &embedding)?;

        // Wrap in resource
        let resource_array = js_sys::Array::new();
        resource_array.push(&resource_obj);

        let resource = js_sys::Object::new();
        js_sys::Reflect::set(&resource, &"embeddings".into(), &resource_array)?;

        self.inner.add(resource.into());
        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query embedding as Float32Array
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// Array of search results with indices and distances
    pub fn search_neighbors(&self, query: JsValue, k: usize) -> Result<JsValue, JsValue> {
        let results = self.inner.search(query, k);
        Ok(results)
    }

    /// Search and return parsed results with structured format
    ///
    /// # Arguments
    /// * `query` - Query embedding as Float32Array
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Array of {id, distance, title, url} objects
    #[wasm_bindgen(js_name = "searchParsed")]
    pub fn search_parsed(&self, query: JsValue, k: usize) -> Result<JsValue, JsValue> {
        let results = self.inner.search(query, k);

        // Voy returns: Array<{id, title, url, embeddings, neighbors: Array<{id, similarity}>}>
        // We want: Array<{id, distance, title, url}>
        if let Ok(arr) = js_sys::Array::from(&results).dyn_into::<js_sys::Array>() {
            let result_array = js_sys::Array::new();

            for i in 0..arr.length() {
                let item = arr.get(i);

                // Each result has: {id, title, url, neighbors: [...]}
                // The neighbors array contains the actual search results
                let neighbors = js_sys::Reflect::get(&item, &"neighbors".into()).ok();

                if let Some(neighbors_val) = neighbors {
                    if let Ok(neighbors_arr) = js_sys::Array::from(&neighbors_val).dyn_into::<js_sys::Array>() {
                        for j in 0..neighbors_arr.length() {
                            let neighbor = neighbors_arr.get(j);

                            // Get id and similarity from neighbor
                            let id = js_sys::Reflect::get(&neighbor, &"id".into())
                                .ok()
                                .and_then(|v| v.as_string());

                            let similarity = js_sys::Reflect::get(&neighbor, &"similarity".into())
                                .ok()
                                .and_then(|v| v.as_f64());

                            // Create result object
                            if let (Some(id_str), Some(sim)) = (id, similarity) {
                                let obj = js_sys::Object::new();
                                js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(&id_str))?;
                                js_sys::Reflect::set(&obj, &"distance".into(), &JsValue::from_f64(1.0 - sim))?; // Convert similarity to distance
                                js_sys::Reflect::set(&obj, &"similarity".into(), &JsValue::from_f64(sim))?;

                                result_array.push(&obj);
                            }
                        }
                    }
                }
            }

            Ok(result_array.into())
        } else {
            Err(JsValue::from_str("Failed to parse search results"))
        }
    }

    /// Serialize the index for storage (returns JSON)
    pub fn serialize(&self) -> Result<String, JsValue> {
        let serialized = self.inner.serialize();
        let json_str = js_sys::JSON::stringify(&serialized)?;
        Ok(json_str.as_string().unwrap_or_default())
    }

    /// Clear all embeddings from index
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get the number of indexed embeddings (if available)
    pub fn size(&self) -> Result<usize, JsValue> {
        // Try to get size from serialized data
        let serialized = self.inner.serialize();
        let embeddings = js_sys::Reflect::get(&serialized, &"embeddings".into())?;

        if let Ok(arr) = js_sys::Array::from(&embeddings).dyn_into::<js_sys::Array>() {
            Ok(arr.length() as usize)
        } else {
            Ok(0)
        }
    }
}

/// Check if Voy is available in the browser
///
/// Voy must be loaded via:
/// ```html
/// <script type="module">
///   import { Voy } from "https://cdn.jsdelivr.net/npm/voy-search@0.6.3/dist/voy.js";
///   window.Voy = Voy;
/// </script>
/// ```
#[wasm_bindgen(js_name = "checkVoyAvailable")]
pub fn check_voy_available() -> bool {
    let window = match web_sys::window() {
        Some(w) => w,
        None => {
            console::error_1(&"No window object found".into());
            return false;
        }
    };

    // Check for window.Voy (modern API)
    match js_sys::Reflect::get(&window, &"Voy".into()) {
        Ok(val) if !val.is_undefined() && !val.is_null() => {
            console::log_1(&"✅ Voy v0.6+ is available (modern API)".into());
            true
        }
        _ => {
            // Check if it's still loading
            let voy_ready = js_sys::Reflect::get(&window, &"voyReady".into())
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if voy_ready {
                console::log_1(&"✅ Voy marked as ready".into());
                return true;
            }

            console::warn_1(&"⚠️  Voy not found. It may still be loading. Please wait or refresh the page.".into());
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_voy_config() {
        let config = VoyConfig::new(384);
        assert_eq!(config.dimension, 384);
    }

    #[wasm_bindgen_test]
    fn test_check_voy_available() {
        // This will log a warning if Voy is not loaded
        let _available = check_voy_available();
    }
}
