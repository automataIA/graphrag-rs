//! # GraphRAG WASM
//!
//! WASM bindings for GraphRAG - enables 100% client-side knowledge graphs in the browser.
//!
//! ## Features
//!
//! - **Pure Rust Vector Search**: Native WASM cosine similarity search
//! - **ONNX Runtime Web**: GPU-accelerated embeddings via WebGPU
//! - **IndexedDB Storage**: Persistent graph data in browser
//! - **Leptos 0.8 UI**: Reactive web components
//!
//! ## Quick Start
//!
//! ```javascript
//! import init, { GraphRAG } from './graphrag_wasm.js';
//!
//! async function main() {
//!   await init();
//!   const graph = new GraphRAG(384);
//!   await graph.add_document("doc1", "Your text here", embedding);
//!   await graph.build_index();
//!   const results = await graph.query(query_embedding, 5);
//!   console.log(results);
//! }
//! ```

use wasm_bindgen::prelude::*;
use std::collections::HashMap;

// WASM backend modules
pub mod storage;
pub mod embedder;
pub mod webllm;
pub mod ollama_http; // Ollama HTTP client (alternative to WebLLM)
pub mod llm_provider; // Unified LLM provider abstraction
pub mod vector_search; // Pure Rust vector search (replaces JavaScript Voy)
pub mod voy_bindings; // Voy vector search bindings
pub mod webgpu_check;

#[cfg(feature = "webgpu")]
pub mod gpu_embedder;

pub mod onnx_embedder;
pub mod entity_extractor;

// Leptos UI components (merged from graphrag-leptos)
pub mod components;

// Re-export ONNX types for easy access
pub use onnx_embedder::{WasmOnnxEmbedder, check_onnx_runtime};

// Re-export Ollama HTTP client
pub use ollama_http::{OllamaHttpClient, OllamaHttpConfig};

// Re-export unified LLM provider
pub use llm_provider::{UnifiedLlmClient, LlmProviderConfig, LlmProviderType};

// Re-export WebLLM client
pub use webllm::WebLLMClient;

// Re-export Leptos components for convenience
pub use components::{
    ChatWindow, QueryInterface, GraphStats, DocumentManager, GraphVisualization,
    ChatMessage, MessageRole, GraphNode, GraphEdge,
};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Initialize WASM module with panic hook
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
    log("GraphRAG WASM initialized");
}

/// Check if WebGPU is available in the browser
#[wasm_bindgen]
pub async fn check_webgpu_support() -> Result<bool, JsValue> {
    use web_sys::window;
    use js_sys::Reflect;

    let window = window().ok_or_else(|| JsValue::from_str("No window found"))?;
    let navigator = window.navigator();
    let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu"))?;

    Ok(!gpu.is_undefined())
}

/// GraphRAG instance for WASM
///
/// This provides a complete client-side knowledge graph implementation
/// using pure Rust vector search and IndexedDB for persistence.
#[wasm_bindgen]
pub struct GraphRAG {
    vector_index: Option<vector_search::VectorIndex>,
    documents: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    dimension: usize,
    index_built: bool,
    // Knowledge graph data
    #[wasm_bindgen(skip)]
    pub entities: Vec<entity_extractor::Entity>,
    #[wasm_bindgen(skip)]
    pub relationships: Vec<entity_extractor::Relationship>,
    // Hierarchical community structure
    #[wasm_bindgen(skip)]
    pub hierarchical_communities: Option<graphrag_core::graph::leiden::HierarchicalCommunities>,
}

// WASM runs in a single-threaded environment, so Send + Sync are safe
// These are required for Leptos 0.8 signals
unsafe impl Send for GraphRAG {}
unsafe impl Sync for GraphRAG {}

// Manual Clone implementation since VoyIndex (JavaScript object) can't be cloned
impl Clone for GraphRAG {
    fn clone(&self) -> Self {
        // Clone all fields except vector_index (which will be None in the clone)
        // The index can be rebuilt if needed via build_index()
        GraphRAG {
            vector_index: None, // JavaScript objects can't be cloned
            documents: self.documents.clone(),
            embeddings: self.embeddings.clone(),
            dimension: self.dimension,
            index_built: false, // Reset since index wasn't cloned
            entities: self.entities.clone(),
            relationships: self.relationships.clone(),
            hierarchical_communities: self.hierarchical_communities.clone(),
        }
    }
}

#[wasm_bindgen]
impl GraphRAG {
    /// Create a new GraphRAG instance
    ///
    /// # Arguments
    /// * `dimension` - Embedding dimension (384 for MiniLM, 768 for BERT)
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Result<GraphRAG, JsValue> {
        log(&format!("Creating GraphRAG with pure Rust vector search (dimension: {})", dimension));

        Ok(GraphRAG {
            vector_index: None,
            documents: Vec::new(),
            embeddings: Vec::new(),
            dimension,
            index_built: false,
            entities: Vec::new(),
            relationships: Vec::new(),
            hierarchical_communities: None,
        })
    }

    /// Add a document to the knowledge graph
    ///
    /// # Arguments
    /// * `id` - Unique document identifier
    /// * `text` - Document text content
    /// * `embedding` - Pre-computed embedding vector
    pub async fn add_document(&mut self, id: String, text: String, embedding: Vec<f32>) -> Result<(), JsValue> {
        log(&format!("Adding document '{}': {} chars", id, text.len()));

        // Store document and embedding (vector search will be added later)
        self.documents.push(text);
        self.embeddings.push(embedding);

        log(&format!("Document '{}' added successfully", id));
        Ok(())
    }

    /// Build the vector index
    ///
    /// Must be called after adding documents and before querying.
    pub async fn build_index(&mut self) -> Result<(), JsValue> {
        log("Building pure Rust vector index...");

        if self.embeddings.is_empty() {
            return Err(JsValue::from_str("No embeddings to index"));
        }

        // Create vector index using pure Rust
        let index = vector_search::VectorIndex::from_embeddings(self.embeddings.clone());
        self.vector_index = Some(index);
        self.index_built = true;

        log(&format!("✅ Pure Rust vector index built: {} documents", self.documents.len()));
        Ok(())
    }

    /// Query the knowledge graph
    ///
    /// # Arguments
    /// * `query_embedding` - Pre-computed query embedding
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    /// JSON string with array of {id, similarity, text} objects
    pub async fn query(&self, query_embedding: Vec<f32>, top_k: usize) -> Result<String, JsValue> {
        log(&format!("Querying with pure Rust vector search, top_k={}", top_k));

        if self.embeddings.is_empty() {
            let json_results: Vec<serde_json::Value> = vec![];
            return serde_json::to_string(&json_results)
                .map_err(|e| JsValue::from_str(&e.to_string()));
        }

        // Use pure Rust vector search
        if let Some(index) = &self.vector_index {
            log("Using pure Rust cosine similarity search ⚡");

            let search_results = index.search(&query_embedding, top_k);

            let results: Vec<serde_json::Value> = search_results
                .iter()
                .map(|result| {
                    let id = result.id.parse::<usize>().unwrap_or(0);
                    serde_json::json!({
                        "id": id,
                        "similarity": result.similarity,
                        "distance": result.distance,
                        "text": self.documents.get(id).unwrap_or(&String::new())
                    })
                })
                .collect();

            return serde_json::to_string(&results)
                .map_err(|e| JsValue::from_str(&e.to_string()));
        }

        // Fallback: No index built
        Err(JsValue::from_str("Index not built. Call build_index() first."))
    }

    /// Get the number of documents in the graph
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Get the embedding dimension
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    /// Check if the index has been built
    pub fn is_index_built(&self) -> bool {
        self.index_built
    }

    /// Get information about the vector index
    pub fn index_info(&self) -> String {
        if self.vector_index.is_some() {
            format!("Pure Rust cosine similarity index with {} vectors (dimension: {})",
                    self.embeddings.len(), self.dimension)
        } else {
            "Index not built yet".to_string()
        }
    }

    /// Clear all documents and reset the index
    pub fn clear(&mut self) {
        log("Clearing all documents and knowledge graph");

        self.documents.clear();
        self.embeddings.clear();
        self.entities.clear();
        self.relationships.clear();
        self.hierarchical_communities = None;
        self.vector_index = None;
        self.index_built = false;
    }

    /// Save the index to IndexedDB for persistence
    pub async fn save_to_storage(&self, db_name: &str) -> Result<(), JsValue> {
        use crate::storage::IndexedDBStore;

        log(&format!("💾 Saving knowledge graph to IndexedDB: {}", db_name));

        let db = IndexedDBStore::new(db_name, 1).await?;

        // Save documents
        db.put("documents", "all_docs", &self.documents).await?;
        log(&format!("  ✓ Saved {} documents", self.documents.len()));

        // Save embeddings and metadata
        db.put("metadata", "embeddings", &self.embeddings).await?;
        db.put("metadata", "dimension", &self.dimension).await?;
        log(&format!("  ✓ Saved {} embeddings (dim: {})", self.embeddings.len(), self.dimension));

        // Save entities
        db.put("entities", "all_entities", &self.entities).await?;
        log(&format!("  ✓ Saved {} entities", self.entities.len()));

        // Save relationships
        db.put("relationships", "all_relationships", &self.relationships).await?;
        log(&format!("  ✓ Saved {} relationships", self.relationships.len()));

        // Save hierarchical communities if they exist
        if let Some(ref communities) = self.hierarchical_communities {
            db.put("communities", "hierarchical", communities).await?;
            let max_level = communities.levels.keys().max().copied().unwrap_or(0);
            log(&format!("  ✓ Saved hierarchical communities ({} levels)", max_level + 1));
        }

        log(&format!("✅ Complete knowledge graph saved: {} docs, {} entities, {} relationships",
            self.documents.len(), self.entities.len(), self.relationships.len()));
        Ok(())
    }

    /// Load the index from IndexedDB
    pub async fn load_from_storage(&mut self, db_name: &str) -> Result<(), JsValue> {
        use crate::storage::IndexedDBStore;

        log(&format!("📥 Loading knowledge graph from IndexedDB: {}", db_name));

        let db = IndexedDBStore::new(db_name, 1).await?;

        // Load documents
        self.documents = db.get("documents", "all_docs").await?;
        log(&format!("  ✓ Loaded {} documents", self.documents.len()));

        // Load embeddings and metadata
        self.embeddings = db.get("metadata", "embeddings").await?;
        self.dimension = db.get("metadata", "dimension").await?;
        log(&format!("  ✓ Loaded {} embeddings (dim: {})", self.embeddings.len(), self.dimension));

        // Load entities (use default empty vec if not found - backward compatibility)
        self.entities = db.get("entities", "all_entities").await
            .unwrap_or_else(|_| {
                log("  ⚠️  No entities found in storage (legacy format)");
                Vec::new()
            });
        log(&format!("  ✓ Loaded {} entities", self.entities.len()));

        // Load relationships (use default empty vec if not found - backward compatibility)
        self.relationships = db.get("relationships", "all_relationships").await
            .unwrap_or_else(|_| {
                log("  ⚠️  No relationships found in storage (legacy format)");
                Vec::new()
            });
        log(&format!("  ✓ Loaded {} relationships", self.relationships.len()));

        // Load hierarchical communities (optional - backward compatibility)
        self.hierarchical_communities = db.get("communities", "hierarchical").await.ok();
        if let Some(ref communities) = self.hierarchical_communities {
            let max_level = communities.levels.keys().max().copied().unwrap_or(0);
            log(&format!("  ✓ Loaded hierarchical communities ({} levels)", max_level + 1));
        } else {
            log("  ⚠️  No hierarchical communities found in storage");
        }

        // Rebuild vector index
        self.build_index().await?;

        log(&format!("✅ Complete knowledge graph loaded: {} docs, {} entities, {} relationships",
            self.documents.len(), self.entities.len(), self.relationships.len()));
        Ok(())
    }

    /// Get the number of entities in the knowledge graph
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get the number of relationships in the knowledge graph
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    /// Get complete statistics as JSON string
    pub fn get_stats(&self) -> String {
        serde_json::json!({
            "documents": self.documents.len(),
            "embeddings": self.embeddings.len(),
            "entities": self.entities.len(),
            "relationships": self.relationships.len(),
            "dimension": self.dimension,
            "index_built": self.index_built,
        }).to_string()
    }

    /// Get all entities as JSON string
    pub fn get_entities_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.entities)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get all relationships as JSON string
    pub fn get_relationships_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.relationships)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ========== Hierarchical Communities Methods ==========

    /// Detect hierarchical communities using Leiden algorithm
    ///
    /// # Arguments
    /// * `config_json` - JSON configuration for Leiden algorithm (optional, use "{}" for defaults)
    ///
    /// # Example config:
    /// ```json
    /// {
    ///   "max_cluster_size": 10,
    ///   "use_lcc": true,
    ///   "resolution": 1.0,
    ///   "max_levels": 5,
    ///   "min_improvement": 0.001
    /// }
    /// ```
    pub async fn detect_communities(&mut self, config_json: &str) -> Result<(), JsValue> {
        use graphrag_core::graph::leiden::{LeidenCommunityDetector, LeidenConfig};
        use petgraph::graph::{Graph, NodeIndex};

        log("🔍 Detecting hierarchical communities with Leiden algorithm...");

        // Parse config from JSON or use defaults
        let config: LeidenConfig = if config_json.trim().is_empty() || config_json == "{}" {
            log("  Using default Leiden configuration");
            LeidenConfig::default()
        } else {
            serde_json::from_str(config_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid config JSON: {}", e)))?
        };

        // Build petgraph from entities and relationships
        // Note: Edge weight is f32 (1.0 for all relationships for now)
        let mut graph: Graph<String, f32, petgraph::Undirected> = Graph::new_undirected();
        let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

        // Add nodes for all entities
        for entity in &self.entities {
            let idx = graph.add_node(entity.name.clone());
            node_indices.insert(entity.name.clone(), idx);
        }

        log(&format!("  ✓ Added {} nodes to graph", graph.node_count()));

        // Add edges for all relationships (with weight 1.0)
        for rel in &self.relationships {
            if let (Some(&from_idx), Some(&to_idx)) =
                (node_indices.get(&rel.from), node_indices.get(&rel.to)) {
                graph.add_edge(from_idx, to_idx, 1.0);
            }
        }

        log(&format!("  ✓ Added {} edges to graph", graph.edge_count()));

        // Detect communities
        let detector = LeidenCommunityDetector::new(config);
        let communities = detector.detect_communities(&graph)
            .map_err(|e| JsValue::from_str(&format!("Leiden detection failed: {}", e)))?;

        let max_level = communities.levels.keys().max().copied().unwrap_or(0);
        log(&format!("✅ Detected {} hierarchical levels", max_level + 1));

        // Store the communities
        self.hierarchical_communities = Some(communities);

        Ok(())
    }

    /// Get the number of hierarchical levels (0 if no communities detected)
    pub fn get_max_level(&self) -> usize {
        self.hierarchical_communities
            .as_ref()
            .and_then(|c| c.levels.keys().max().copied())
            .map(|max| max + 1) // Convert 0-indexed to count
            .unwrap_or(0)
    }

    /// Get communities at a specific level as JSON
    ///
    /// Returns JSON array of communities with their entities
    pub fn get_communities_at_level(&self, level: usize) -> Result<String, JsValue> {
        let communities = self.hierarchical_communities.as_ref()
            .ok_or_else(|| JsValue::from_str("No communities detected. Call detect_communities() first."))?;

        let level_communities = communities.levels.get(&level)
            .ok_or_else(|| JsValue::from_str(&format!("Level {} does not exist", level)))?;

        // Group entities by community ID
        let mut community_groups: HashMap<usize, Vec<String>> = HashMap::new();

        for (node_idx, community_id) in level_communities {
            // Get entity name from node index (we need to map back)
            // This is a simplified version - in production we'd store the mapping
            community_groups.entry(*community_id)
                .or_insert_with(Vec::new)
                .push(format!("node_{}", node_idx.index()));
        }

        serde_json::to_string(&community_groups)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))
    }

    /// Get summary for a specific community
    pub fn get_community_summary(&self, community_id: usize) -> Result<String, JsValue> {
        let communities = self.hierarchical_communities.as_ref()
            .ok_or_else(|| JsValue::from_str("No communities detected"))?;

        communities.summaries.get(&community_id)
            .cloned()
            .ok_or_else(|| JsValue::from_str(&format!("No summary for community {}", community_id)))
    }

    /// Get all community summaries as JSON
    pub fn get_all_summaries(&self) -> Result<String, JsValue> {
        let communities = self.hierarchical_communities.as_ref()
            .ok_or_else(|| JsValue::from_str("No communities detected"))?;

        serde_json::to_string(&communities.summaries)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))
    }

    /// Query using adaptive routing (automatically selects best hierarchical level)
    ///
    /// # Arguments
    /// * `query` - User query string
    /// * `config_json` - Adaptive routing configuration (optional, use "{}" for defaults)
    ///
    /// # Example config:
    /// ```json
    /// {
    ///   "enabled": true,
    ///   "default_level": 1,
    ///   "keyword_weight": 0.5,
    ///   "length_weight": 0.3,
    ///   "entity_weight": 0.2
    /// }
    /// ```
    ///
    /// # Returns
    /// JSON with query analysis and results:
    /// ```json
    /// {
    ///   "analysis": {
    ///     "suggested_level": 1,
    ///     "keyword_score": 0.5,
    ///     "length_score": 0.3,
    ///     "entity_score": 0.2
    ///   },
    ///   "results": [
    ///     {"level": 1, "community_id": 5, "summary": "..."}
    ///   ]
    /// }
    /// ```
    pub async fn query_adaptive(&self, query: &str, config_json: &str) -> Result<String, JsValue> {
        use graphrag_core::query::AdaptiveRoutingConfig;
        use petgraph::graph::Graph;

        let communities = self.hierarchical_communities.as_ref()
            .ok_or_else(|| JsValue::from_str("No communities detected. Call detect_communities() first."))?;

        // Parse adaptive routing config
        let config: AdaptiveRoutingConfig = if config_json.trim().is_empty() || config_json == "{}" {
            AdaptiveRoutingConfig::default()
        } else {
            serde_json::from_str(config_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid config JSON: {}", e)))?
        };

        // Build graph from entities and relationships (same as detect_communities)
        let mut graph: Graph<String, f32, petgraph::Undirected> = Graph::new_undirected();
        let mut node_indices = HashMap::new();

        for entity in &self.entities {
            let idx = graph.add_node(entity.name.clone());
            node_indices.insert(entity.name.clone(), idx);
        }

        for rel in &self.relationships {
            if let (Some(&from_idx), Some(&to_idx)) =
                (node_indices.get(&rel.from), node_indices.get(&rel.to)) {
                graph.add_edge(from_idx, to_idx, 1.0);
            }
        }

        // Perform adaptive retrieval
        let (analysis, results) = communities.adaptive_retrieve_detailed(query, &graph, config);

        // Format response
        let response = serde_json::json!({
            "analysis": {
                "suggested_level": analysis.suggested_level,
                "keyword_score": analysis.keyword_score,
                "length_score": analysis.length_score,
                "entity_score": analysis.entity_score,
            },
            "results": results.iter().map(|(level, comm_id, summary)| {
                serde_json::json!({
                    "level": level,
                    "community_id": comm_id,
                    "summary": summary,
                })
            }).collect::<Vec<_>>(),
        });

        serde_json::to_string(&response)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))
    }

    /// Query at a specific hierarchical level
    ///
    /// # Arguments
    /// * `query` - User query string
    /// * `level` - Hierarchical level to search (0 = finest granularity)
    ///
    /// # Returns
    /// JSON array of matching communities:
    /// ```json
    /// [
    ///   {"level": 1, "community_id": 5, "summary": "..."},
    ///   {"level": 1, "community_id": 8, "summary": "..."}
    /// ]
    /// ```
    pub async fn query_at_level(&self, query: &str, level: usize) -> Result<String, JsValue> {
        use petgraph::graph::Graph;

        let communities = self.hierarchical_communities.as_ref()
            .ok_or_else(|| JsValue::from_str("No communities detected. Call detect_communities() first."))?;

        // Build graph from entities and relationships
        let mut graph: Graph<String, f32, petgraph::Undirected> = Graph::new_undirected();
        let mut node_indices = HashMap::new();

        for entity in &self.entities {
            let idx = graph.add_node(entity.name.clone());
            node_indices.insert(entity.name.clone(), idx);
        }

        for rel in &self.relationships {
            if let (Some(&from_idx), Some(&to_idx)) =
                (node_indices.get(&rel.from), node_indices.get(&rel.to)) {
                graph.add_edge(from_idx, to_idx, 1.0);
            }
        }

        // Perform retrieval at specific level
        let results = communities.retrieve_at_level(query, &graph, level);

        // Format response
        let response: Vec<_> = results.iter().map(|(level, comm_id, summary)| {
            serde_json::json!({
                "level": level,
                "community_id": comm_id,
                "summary": summary,
            })
        }).collect();

        serde_json::to_string(&response)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {}", e)))
    }
}

/// Non-WASM methods for internal use (accessible from Rust code)
impl GraphRAG {
    /// Add entities to the knowledge graph
    pub fn add_entities(&mut self, entities: Vec<entity_extractor::Entity>) {
        self.entities.extend(entities);
    }

    /// Add relationships to the knowledge graph
    pub fn add_relationships(&mut self, relationships: Vec<entity_extractor::Relationship>) {
        self.relationships.extend(relationships);
    }

    /// Get reference to entities (for internal use)
    pub fn entities(&self) -> &[entity_extractor::Entity] {
        &self.entities
    }

    /// Get reference to relationships (for internal use)
    pub fn relationships(&self) -> &[entity_extractor::Relationship] {
        &self.relationships
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_graphrag_creation() {
        let graph = GraphRAG::new(384);
        assert!(graph.is_ok());
    }
}
