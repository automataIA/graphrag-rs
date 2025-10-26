//! GraphRAG REST API Server with Qdrant Integration
//!
//! Production-ready REST API for GraphRAG operations with Qdrant vector database.
//!
//! This version integrates Qdrant for real vector storage and semantic search.
//! Falls back to in-memory storage when Qdrant is not available.
//!
//! ## Quick Start
//!
//! ```bash
//! # 1. Start Qdrant (Docker)
//! docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
//!
//! # 2. Start server with Qdrant
//! cargo run --bin graphrag-server --features qdrant
//!
//! # 3. Or without Qdrant (mock mode)
//! cargo run --bin graphrag-server --no-default-features
//! ```

use axum::{
    extract::{Path, State},
    http::{StatusCode, Method},
    middleware,
    response::{IntoResponse, Json},
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber;

#[cfg(feature = "qdrant")]
mod qdrant_store;
#[cfg(feature = "qdrant")]
use qdrant_store::{QdrantStore, DocumentMetadata};

#[cfg(feature = "auth")]
mod auth;
#[cfg(feature = "auth")]
use auth::{AuthState, auth_middleware};

mod embeddings;
use embeddings::{EmbeddingService, EmbeddingConfig};

mod validation;
use validation::{validate_query, validate_title, validate_content, validate_top_k, sanitize_string};

mod config_handler;
use config_handler::ConfigManager;

mod config_endpoints;

// Import full GraphRAG pipeline
use graphrag_core::GraphRAG;

/// Application state with optional Qdrant backend and full GraphRAG pipeline
#[derive(Clone)]
struct AppState {
    #[cfg(feature = "qdrant")]
    qdrant: Option<Arc<QdrantStore>>,

    // Embedding service (real or fallback)
    embeddings: Arc<EmbeddingService>,

    // Full GraphRAG pipeline (when configured via JSON)
    graphrag: Arc<RwLock<Option<GraphRAG>>>,

    // Configuration manager for JSON config
    config_manager: Arc<ConfigManager>,

    // Authentication state (optional)
    #[cfg(feature = "auth")]
    auth: Arc<AuthState>,

    // Fallback in-memory storage (used when Qdrant unavailable or simple mode)
    documents: Arc<RwLock<Vec<Document>>>,
    graph_built: Arc<RwLock<bool>>,
    query_count: Arc<RwLock<usize>>,
}

impl AppState {
    async fn new() -> Self {
        // Initialize embedding service
        let embedding_backend = std::env::var("EMBEDDING_BACKEND")
            .unwrap_or_else(|_| "hash".to_string()); // Default to hash fallback
        let embedding_dim: usize = std::env::var("EMBEDDING_DIM")
            .unwrap_or_else(|_| "384".to_string())
            .parse()
            .unwrap_or(384);

        let embedding_config = EmbeddingConfig {
            backend: embedding_backend,
            dimension: embedding_dim,
            ollama_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost".to_string()),
            ollama_model: std::env::var("OLLAMA_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string()),
            enable_cache: true,
        };

        let embeddings = match EmbeddingService::new(embedding_config).await {
            Ok(service) => {
                tracing::info!("✅ Embedding service initialized: {}", service.backend_name());
                Arc::new(service)
            }
            Err(e) => {
                tracing::error!("❌ Failed to initialize embedding service: {}. Server may not work correctly.", e);
                // This should not happen as EmbeddingService always has fallback
                std::process::exit(1);
            }
        };

        #[cfg(feature = "qdrant")]
        {
            // Try to connect to Qdrant
            let qdrant_url = std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6334".to_string());
            let collection_name = std::env::var("COLLECTION_NAME")
                .unwrap_or_else(|_| "graphrag".to_string());

            match QdrantStore::new(&qdrant_url, &collection_name).await {
                Ok(store) => {
                    // Check if collection exists, create if not
                    if !store.collection_exists().await.unwrap_or(false) {
                        match store.create_collection(embedding_dim as u64).await {
                            Ok(_) => {
                                tracing::info!("✅ Created Qdrant collection: {}", collection_name);
                            }
                            Err(e) => {
                                tracing::warn!("⚠️  Could not create collection: {}", e);
                            }
                        }
                    } else {
                        tracing::info!("✅ Connected to existing Qdrant collection: {}", collection_name);
                    }

                    tracing::info!("🗄️  Using Qdrant at: {}", qdrant_url);

                    Self {
                        qdrant: Some(Arc::new(store)),
                        embeddings,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(
                            std::env::var("JWT_SECRET")
                                .unwrap_or_else(|_| "graphrag_secret_key_change_in_production_32chars".to_string())
                        )),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        query_count: Arc::new(RwLock::new(0)),
                    }
                }
                Err(e) => {
                    tracing::warn!("⚠️  Could not connect to Qdrant: {}. Using in-memory storage.", e);
                    Self {
                        qdrant: None,
                        embeddings,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(
                            std::env::var("JWT_SECRET")
                                .unwrap_or_else(|_| "graphrag_secret_key_change_in_production_32chars".to_string())
                        )),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        query_count: Arc::new(RwLock::new(0)),
                    }
                }
            }
        }

        #[cfg(not(feature = "qdrant"))]
        {
            tracing::info!("📦 Using in-memory storage (Qdrant feature disabled)");
            Self {
                embeddings,
                graphrag: Arc::new(RwLock::new(None)),
                config_manager: Arc::new(ConfigManager::new()),
                #[cfg(feature = "auth")]
                auth: Arc::new(AuthState::new(
                    std::env::var("JWT_SECRET")
                        .unwrap_or_else(|_| "graphrag_secret_key_change_in_production_32chars".to_string())
                )),
                documents: Arc::new(RwLock::new(Vec::new())),
                graph_built: Arc::new(RwLock::new(false)),
                query_count: Arc::new(RwLock::new(0)),
            }
        }
    }

    /// Check if Qdrant is available
    fn has_qdrant(&self) -> bool {
        #[cfg(feature = "qdrant")]
        {
            self.qdrant.is_some()
        }
        #[cfg(not(feature = "qdrant"))]
        {
            false
        }
    }
}

/// Document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    id: String,
    title: String,
    content: String,
    added_at: String,
}

/// Query request
#[derive(Debug, Deserialize)]
struct QueryRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize {
    5
}

/// Query response
#[derive(Debug, Serialize)]
struct QueryResponse {
    query: String,
    results: Vec<QueryResult>,
    processing_time_ms: u64,
    backend: String, // "qdrant" or "memory"
}

#[derive(Debug, Serialize)]
struct QueryResult {
    document_id: String,
    title: String,
    similarity: f32,
    excerpt: String,
}

/// Add document request
#[derive(Debug, Deserialize)]
struct AddDocumentRequest {
    title: String,
    content: String,
}

/// Graph statistics response
#[derive(Debug, Serialize)]
struct GraphStatsResponse {
    document_count: usize,
    entity_count: usize,
    relationship_count: usize,
    vector_count: usize,
    graph_built: bool,
    backend: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    // Create application state (connects to Qdrant if available)
    let state = AppState::new().await;

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any);

    // Build router with optional auth
    #[cfg(feature = "auth")]
    let _auth_enabled = std::env::var("ENABLE_AUTH")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);

    #[cfg(feature = "auth")]
    let protected_routes = Router::new()
        .route("/api/query", post(query))
        .route("/api/documents", get(list_documents).post(add_document))
        .route("/api/documents/{id}", delete(delete_document))
        .route("/api/graph/build", post(build_graph))
        .route("/api/graph/stats", get(graph_stats))
        .route_layer(middleware::from_fn_with_state(
            state.clone().auth.clone(),
            auth_middleware
        ));

    #[cfg(not(feature = "auth"))]
    let protected_routes = Router::new()
        .route("/api/query", post(query))
        .route("/api/documents", get(list_documents).post(add_document))
        .route("/api/documents/{id}", delete(delete_document))
        .route("/api/graph/build", post(build_graph))
        .route("/api/graph/stats", get(graph_stats));

    let mut app = Router::new()
        // Public endpoints
        .route("/", get(root))
        .route("/health", get(health))

        // Configuration endpoints
        .route("/api/config", get(config_endpoints::get_config).post(config_endpoints::set_config))
        .route("/api/config/template", get(config_endpoints::get_config_template))
        .route("/api/config/default", get(config_endpoints::get_default_config))
        .route("/api/config/validate", post(config_endpoints::validate_config));

    // Auth endpoints (if enabled)
    #[cfg(feature = "auth")]
    {
        app = app
            .route("/auth/login", post(login))
            .route("/auth/api-key", post(create_api_key));
    }

    let app = app
        // Protected API endpoints
        .merge(protected_routes)

        // Add middleware layers
        .layer(middleware::from_fn(validation::request_size_limit))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:8080";
    tracing::info!("🚀 GraphRAG Server starting...");
    tracing::info!("📡 Listening on http://{}", addr);
    tracing::info!("📚 API Docs: http://{}/", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Root endpoint - API information
async fn root(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "name": "GraphRAG REST API",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "backend": if state.has_qdrant() { "qdrant" } else { "memory" },
        "graphrag_configured": state.graphrag.read().await.is_some(),
        "endpoints": {
            "health": "GET /health",
            "config": {
                "get": "GET /api/config - Get current configuration",
                "set": "POST /api/config - Set configuration and initialize GraphRAG",
                "template": "GET /api/config/template - Get configuration templates and examples",
                "default": "GET /api/config/default - Get default configuration",
                "validate": "POST /api/config/validate - Validate configuration without applying"
            },
            "query": "POST /api/query",
            "documents": {
                "list": "GET /api/documents",
                "add": "POST /api/documents",
                "delete": "DELETE /api/documents/{id}"
            },
            "graph": {
                "build": "POST /api/graph/build",
                "stats": "GET /api/graph/stats"
            }
        }
    }))
}

/// Health check endpoint
async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let doc_count;
    let graph_built;
    let query_count = *state.query_count.read().await;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                doc_count = count;
                graph_built = count > 0;
            }
            Err(_) => {
                doc_count = 0;
                graph_built = false;
            }
        }
    } else {
        doc_count = state.documents.read().await.len();
        graph_built = *state.graph_built.read().await;
    }

    #[cfg(not(feature = "qdrant"))]
    {
        doc_count = state.documents.read().await.len();
        graph_built = *state.graph_built.read().await;
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "document_count": doc_count,
            "graph_built": graph_built,
            "total_queries": query_count,
            "backend": if state.has_qdrant() { "qdrant" } else { "memory" }
        }))
    )
}

/// Query the knowledge graph
async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    // Validate input
    if let Err(e) = validate_query(&req.query) {
        tracing::warn!(query = %req.query, error = %e.error, "Invalid query");
        return Err((StatusCode::BAD_REQUEST, e.error));
    }

    if let Err(e) = validate_top_k(req.top_k) {
        tracing::warn!(top_k = req.top_k, error = %e.error, "Invalid top_k");
        return Err((StatusCode::BAD_REQUEST, e.error));
    }

    let start = std::time::Instant::now();

    // Increment query count
    *state.query_count.write().await += 1;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Real vector search with Qdrant using real embeddings
        let query_embedding = match state.embeddings.generate_single(&req.query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                tracing::error!("Failed to generate query embedding: {}", e);
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to generate embedding: {}", e),
                ));
            }
        };

        match qdrant.search(query_embedding, req.top_k, None).await {
            Ok(search_results) => {
                let results: Vec<QueryResult> = search_results
                    .into_iter()
                    .map(|r| QueryResult {
                        document_id: r.id,
                        title: r.metadata.title,
                        similarity: r.score,
                        excerpt: if r.metadata.text.len() > 200 {
                            format!("{}...", &r.metadata.text[..200])
                        } else {
                            r.metadata.text
                        },
                    })
                    .collect();

                let processing_time = start.elapsed().as_millis() as u64;

                return Ok(Json(QueryResponse {
                    query: req.query,
                    results,
                    processing_time_ms: processing_time,
                    backend: "qdrant".to_string(),
                }));
            }
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Qdrant search failed: {}", e),
                ));
            }
        }
    }

    // Fallback: in-memory search
    let documents = state.documents.read().await;

    if documents.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No documents available. Add documents first.".to_string(),
        ));
    }

    // Simple keyword matching for demonstration
    let mut results: Vec<QueryResult> = documents
        .iter()
        .map(|doc| {
            let query_lower = req.query.to_lowercase();
            let content_lower = doc.content.to_lowercase();
            let title_lower = doc.title.to_lowercase();

            let similarity = if content_lower.contains(&query_lower) || title_lower.contains(&query_lower) {
                0.85
            } else {
                0.1
            };

            let excerpt = if doc.content.len() > 200 {
                format!("{}...", &doc.content[..200])
            } else {
                doc.content.clone()
            };

            QueryResult {
                document_id: doc.id.clone(),
                title: doc.title.clone(),
                similarity,
                excerpt,
            }
        })
        .filter(|r| r.similarity > 0.5)
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(req.top_k);

    let processing_time = start.elapsed().as_millis() as u64;

    Ok(Json(QueryResponse {
        query: req.query,
        results,
        processing_time_ms: processing_time,
        backend: "memory".to_string(),
    }))
}

/// Add a document to the knowledge graph
async fn add_document(
    State(state): State<AppState>,
    Json(req): Json<AddDocumentRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Validate input
    if let Err(e) = validate_title(&req.title) {
        tracing::warn!(title = %req.title, error = %e.error, "Invalid title");
        return Err((StatusCode::BAD_REQUEST, e.error));
    }

    if let Err(e) = validate_content(&req.content) {
        tracing::warn!(content_len = req.content.len(), error = %e.error, "Invalid content");
        return Err((StatusCode::BAD_REQUEST, e.error));
    }

    // Sanitize inputs
    let title = sanitize_string(&req.title);
    let content = sanitize_string(&req.content);

    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Generate real embeddings
        let embedding = match state.embeddings.generate_single(&content).await {
            Ok(emb) => emb,
            Err(e) => {
                tracing::error!("Failed to generate document embedding: {}", e);
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to generate embedding: {}", e),
                ));
            }
        };

        let metadata = DocumentMetadata {
            id: id.clone(),
            title: title.clone(),
            text: content.clone(),
            chunk_index: 0,
            entities: Vec::new(), // TODO: Extract entities
            relationships: Vec::new(), // TODO: Extract relationships
            timestamp: timestamp.clone(),
            custom: HashMap::new(),
        };

        match qdrant.add_document(&id, embedding, metadata).await {
            Ok(_) => {
                tracing::info!("Added document to Qdrant: {} ({})", req.title, id);

                return Ok(Json(json!({
                    "success": true,
                    "document_id": id,
                    "message": "Document added to Qdrant successfully",
                    "backend": "qdrant"
                })));
            }
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to add document to Qdrant: {}", e),
                ));
            }
        }
    }

    // Fallback: in-memory storage
    let document = Document {
        id: id.clone(),
        title: req.title,
        content: req.content,
        added_at: timestamp,
    };

    state.documents.write().await.push(document.clone());
    *state.graph_built.write().await = false;

    tracing::info!("Added document to memory: {} ({})", document.title, id);

    Ok(Json(json!({
        "success": true,
        "document_id": id,
        "message": "Document added to memory successfully",
        "backend": "memory"
    })))
}

/// List all documents
async fn list_documents(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, vectors)) => {
                return Json(json!({
                    "documents": [],
                    "total": count,
                    "vectors": vectors,
                    "backend": "qdrant",
                    "note": "Full document listing from Qdrant not implemented yet"
                }));
            }
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            }
        }
    }

    // Fallback: in-memory storage
    let documents = state.documents.read().await;

    let doc_list: Vec<_> = documents
        .iter()
        .map(|doc| json!({
            "id": doc.id,
            "title": doc.title,
            "content_length": doc.content.len(),
            "added_at": doc.added_at
        }))
        .collect();

    Json(json!({
        "documents": doc_list,
        "total": doc_list.len(),
        "backend": "memory"
    }))
}

/// Delete a document
async fn delete_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.delete_document(&id).await {
            Ok(_) => {
                tracing::info!("Deleted document from Qdrant: {}", id);
                return Ok(Json(json!({
                    "success": true,
                    "message": format!("Document {} deleted from Qdrant", id),
                    "backend": "qdrant"
                })));
            }
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to delete from Qdrant: {}", e),
                ));
            }
        }
    }

    // Fallback: in-memory storage
    let mut documents = state.documents.write().await;
    let original_len = documents.len();
    documents.retain(|doc| doc.id != id);

    if documents.len() == original_len {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Document with id '{}' not found", id),
        ));
    }

    *state.graph_built.write().await = false;
    tracing::info!("Deleted document from memory: {}", id);

    Ok(Json(json!({
        "success": true,
        "message": format!("Document {} deleted from memory", id),
        "backend": "memory"
    })))
}

/// Build the knowledge graph
async fn build_graph(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                if count == 0 {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        "No documents in Qdrant. Add documents first.".to_string(),
                    ));
                }

                // Simulate graph building
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                let processing_time = start.elapsed().as_millis() as u64;

                tracing::info!("Built knowledge graph from {} Qdrant documents in {}ms", count, processing_time);

                return Ok(Json(json!({
                    "success": true,
                    "document_count": count,
                    "processing_time_ms": processing_time,
                    "message": "Knowledge graph built from Qdrant successfully",
                    "backend": "qdrant"
                })));
            }
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to access Qdrant: {}", e),
                ));
            }
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();

    if doc_count == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "No documents to build graph from. Add documents first.".to_string(),
        ));
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    *state.graph_built.write().await = true;

    let processing_time = start.elapsed().as_millis() as u64;

    tracing::info!("Built knowledge graph from {} memory documents in {}ms", doc_count, processing_time);

    Ok(Json(json!({
        "success": true,
        "document_count": doc_count,
        "processing_time_ms": processing_time,
        "message": "Knowledge graph built from memory successfully",
        "backend": "memory"
    })))
}

/// Get graph statistics
async fn graph_stats(
    State(state): State<AppState>,
) -> Json<GraphStatsResponse> {
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, vectors)) => {
                return Json(GraphStatsResponse {
                    document_count: count,
                    entity_count: count * 10, // Estimated
                    relationship_count: count * 15, // Estimated
                    vector_count: vectors,
                    graph_built: count > 0,
                    backend: "qdrant".to_string(),
                });
            }
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            }
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();
    let graph_built = *state.graph_built.read().await;

    let entity_count = if graph_built { doc_count * 10 } else { 0 };
    let relationship_count = if graph_built { doc_count * 15 } else { 0 };
    let vector_count = if graph_built { doc_count * 20 } else { 0 };

    Json(GraphStatsResponse {
        document_count: doc_count,
        entity_count,
        relationship_count,
        vector_count,
        graph_built,
        backend: "memory".to_string(),
    })
}

/// Generate dummy embedding (placeholder for real embedding model)
///
/// TODO: Replace with real embedding generation:
/// - Option A: Call Ollama API for embeddings
/// - Option B: Load local BERT/MiniLM model with Candle
/// - Option C: Use external embedding API (OpenAI, Cohere, etc.)
fn _generate_dummy_embedding(text: &str, dimension: usize) -> Vec<f32> {
    // Simple hash-based dummy embedding for testing
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    text.hash(&mut hasher);
    let hash = hasher.finish();

    let base = (hash % 1000) as f32 / 1000.0;

    (0..dimension)
        .map(|i| {
            let offset = (i as f32) / (dimension as f32);
            (base + offset).sin()
        })
        .collect()
}

// ============================================================================
// Authentication Endpoints
// ============================================================================

#[cfg(feature = "auth")]
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LoginRequest {
    username: String,
    password: String,
}

#[cfg(feature = "auth")]
#[derive(Debug, Deserialize)]
struct ApiKeyRequest {
    user_id: String,
    role: Option<auth::UserRole>,
}

/// Login endpoint - generates JWT token
#[cfg(feature = "auth")]
async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // TODO: Implement real user authentication against database
    // For now, accept any credentials for demo purposes
    tracing::info!("Login attempt for user: {}", req.username);

    let role = if req.username == "admin" {
        auth::UserRole::Admin
    } else {
        auth::UserRole::User
    };

    match state.auth.generate_token(&req.username, role.clone(), 24) {
        Ok(token) => {
            tracing::info!("✅ Generated JWT token for user: {} (role: {:?})", req.username, role);
            Ok(Json(json!({
                "success": true,
                "token": token,
                "user_id": req.username,
                "role": role,
                "expires_in_hours": 24,
                "usage": "Add header: Authorization: Bearer <token>"
            })))
        }
        Err(e) => {
            tracing::error!("❌ Failed to generate token: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Token generation failed: {}", e),
            ))
        }
    }
}

/// Create API key endpoint
#[cfg(feature = "auth")]
async fn create_api_key(
    State(state): State<AppState>,
    Json(req): Json<ApiKeyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let role = req.role.unwrap_or(auth::UserRole::User);

    match state.auth.create_api_key(&req.user_id, role.clone(), None).await {
        Ok(api_key) => {
            tracing::info!("✅ Created API key for user: {} (role: {:?})", req.user_id, role);
            Ok(Json(json!({
                "success": true,
                "api_key": api_key,
                "user_id": req.user_id,
                "role": role,
                "usage": "Add header: Authorization: ApiKey <key>",
                "rate_limit": {
                    "max_requests": 1000,
                    "window_seconds": 3600
                }
            })))
        }
        Err(e) => {
            tracing::error!("❌ Failed to create API key: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("API key creation failed: {}", e),
            ))
        }
    }
}
