//! Local GraphRAG REST API Server with Actix-web and Apistos OpenAPI.
//!
//! The server is intended for a trusted local process. It has no enabled
//! multi-user authentication and binds to loopback unless explicitly changed.
//!
//! ## Features
//! - Automatic OpenAPI 3.0.3 documentation via Apistos
//! - Interactive Swagger UI at /swagger
//! - Qdrant vector database integration (optional)
//! - Request validation and OpenAPI documentation
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
//! # 3. Or without Qdrant (in-memory document retrieval)
//! cargo run --bin graphrag-server --no-default-features
//!
//! # 4. View Swagger UI
//! # Browser: http://localhost:8080/swagger
//! ```

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use actix_cors::Cors;
use actix_web::{
    http::header,
    web::{self, Data, Json, Path as WebPath},
    App, HttpServer, Responder,
};
use apistos::{
    api_operation,
    app::{BuildConfig, OpenApiWrapper},
    info::Info,
    spec::Spec,
    web::{delete, get, post, resource, scope, ServiceConfig as OpenApiServiceConfig},
    SwaggerUIConfig,
};
use serde_json::json;
#[cfg(feature = "qdrant")]
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
mod models;
use models::*;

#[cfg(feature = "qdrant")]
mod qdrant_store;
#[cfg(feature = "qdrant")]
use qdrant_store::{DocumentMetadata, QdrantStore};

#[cfg(feature = "auth")]
mod auth;
#[cfg(feature = "auth")]
use auth::AuthState;

mod embeddings;
use embeddings::{EmbeddingConfig, EmbeddingService};

mod validation;
use validation::{
    sanitize_string, validate_content, validate_query, validate_title, validate_top_k,
};

mod config_handler;

mod config_endpoints;

// Import full GraphRAG pipeline
use graphrag_core::GraphRAG;

const RESPONSE_EXCERPT_BYTES: usize = 200;

fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn response_excerpt(text: &str) -> String {
    if text.len() > RESPONSE_EXCERPT_BYTES {
        format!("{}...", truncate_utf8(text, RESPONSE_EXCERPT_BYTES))
    } else {
        text.to_owned()
    }
}

/// Collapse pipeline search results into one entry per source document, best chunk first.
/// Scores are normalized by the top score so `similarity` stays in 0..=1.
fn pipeline_query_results(
    graphrag: &GraphRAG,
    mut search_results: Vec<graphrag_core::retrieval::SearchResult>,
    top_k: usize,
    fallback_title: impl Fn(&str) -> Option<String>,
) -> Vec<QueryResult> {
    search_results.sort_by(|a, b| b.score.total_cmp(&a.score));
    let top_score = search_results.first().map_or(0.0, |r| r.score);
    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();
    for result in &search_results {
        let chunk = result
            .source_chunks
            .iter()
            .chain(std::iter::once(&result.id))
            .find_map(|id| graphrag.get_chunk(id));
        let Some(chunk) = chunk else { continue };
        let doc_id = chunk.document_id.0.as_str();
        if !seen.insert(doc_id.to_string()) {
            continue;
        }
        let title = graphrag
            .knowledge_graph()
            .and_then(|g| g.get_document(&chunk.document_id))
            .map(|d| d.title.clone())
            .or_else(|| fallback_title(doc_id))
            .unwrap_or_default();
        let similarity = if top_score > 0.0 {
            (result.score / top_score).clamp(0.0, 1.0)
        } else {
            0.0
        };
        results.push(QueryResult {
            document_id: doc_id.to_string(),
            title,
            similarity,
            excerpt: response_excerpt(&chunk.content),
        });
        if results.len() == top_k {
            break;
        }
    }
    results
}

/// JSON extractor config: body size limit, and malformed bodies reported in the
/// same JSON error shape as every other `ApiError` (actix defaults to plain text).
fn json_config() -> web::JsonConfig {
    web::JsonConfig::default()
        .limit(validation::MAX_BODY_SIZE)
        .error_handler(|err, _req| ApiError::BadRequest(err.to_string()).into())
}

fn parse_bind_addr(value: Option<&str>) -> color_eyre::Result<SocketAddr> {
    let value = value.unwrap_or("127.0.0.1:8080");
    value.parse::<SocketAddr>().map_err(|error| {
        color_eyre::eyre::eyre!(
            "invalid GRAPHRAG_BIND_ADDR '{}': expected an IP address and port ({})",
            value,
            error
        )
    })
}

fn parse_cors_origins(value: Option<&str>) -> color_eyre::Result<Option<Vec<String>>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let origins: Vec<&str> = value.split(',').map(str::trim).collect();
    if origins.is_empty() || origins.iter().any(|origin| origin.is_empty()) {
        return Err(color_eyre::eyre::eyre!(
            "GRAPHRAG_CORS_ORIGINS must contain one or more non-empty origins"
        ));
    }
    if origins.iter().any(|origin| !is_valid_origin(origin)) {
        return Err(color_eyre::eyre::eyre!(
            "GRAPHRAG_CORS_ORIGINS accepts explicit http:// or https:// origins without paths, credentials, or wildcards"
        ));
    }
    Ok(Some(origins.into_iter().map(str::to_owned).collect()))
}

fn is_valid_origin(origin: &str) -> bool {
    if origin == "*" || origin.contains(char::is_whitespace) {
        return false;
    }
    let authority = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"));
    let Some(authority) = authority else {
        return false;
    };
    if authority.is_empty() {
        return false;
    }
    if authority
        .chars()
        .any(|character| matches!(character, '/' | '?' | '#' | '@'))
    {
        return false;
    }

    // An origin may contain an optional numeric port. Bracketed IPv6 is
    // accepted; unbracketed additional colons are rejected as ambiguous.
    let host = if let Some(rest) = authority.strip_prefix('[') {
        let Some(end) = rest.find(']') else {
            return false;
        };
        let host = &rest[..end];
        if host.is_empty() {
            return false;
        }
        let suffix = &rest[end + 1..];
        if let Some(port) = suffix.strip_prefix(':') {
            if port.is_empty() || port.parse::<u16>().is_err() {
                return false;
            }
        } else if !suffix.is_empty() {
            return false;
        }
        if host.parse::<std::net::Ipv6Addr>().is_err() {
            return false;
        }
        return true;
    } else {
        let mut pieces = authority.split(':');
        let host = pieces.next().unwrap_or_default();
        if host.is_empty() || pieces.clone().count() > 1 {
            return false;
        }
        if let Some(port) = pieces.next() {
            if port.is_empty() || port.parse::<u16>().is_err() {
                return false;
            }
        }
        host
    };

    host.chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '.' | '-'))
}

fn build_cors(origins: Option<&[String]>) -> Cors {
    let Some(origins) = origins else {
        // Cors::default() does not grant cross-origin access. Same-origin
        // requests remain available to the local trusted service.
        return Cors::default().max_age(3600);
    };

    let cors = origins
        .iter()
        .fold(Cors::default(), |cors, origin| cors.allowed_origin(origin));
    cors.allowed_methods(["GET", "POST", "DELETE", "OPTIONS"])
        .allowed_headers([header::ACCEPT, header::CONTENT_TYPE, header::AUTHORIZATION])
        .max_age(3600)
}

macro_rules! register_api_routes {
    ($cfg:expr) => {
        $cfg.service(
            scope("/api/config")
                .service(
                    resource("")
                        .route(get().to(config_endpoints::get_config))
                        .route(post().to(config_endpoints::set_config)),
                )
                .service(
                    resource("/template").route(get().to(config_endpoints::get_config_template)),
                )
                .service(resource("/default").route(get().to(config_endpoints::get_default_config)))
                .service(resource("/validate").route(post().to(config_endpoints::validate_config))),
        )
        .service(
            scope("/api")
                .service(
                    scope("/documents")
                        .service(
                            resource("")
                                .route(get().to(list_documents))
                                .route(post().to(add_document)),
                        )
                        .service(resource("/{id}").route(delete().to(delete_document))),
                )
                .service(scope("/query").service(resource("").route(post().to(query))))
                .service(
                    scope("/graph")
                        .service(resource("/build").route(post().to(build_graph)))
                        .service(resource("/stats").route(get().to(graph_stats))),
                ),
        );
    };
}

/// Register production routes with Apistos so the OpenAPI operations remain
/// attached to the same handlers as the running server.
fn configure_routes(cfg: &mut OpenApiServiceConfig<'_>) {
    register_api_routes!(cfg);
}

/// Register the same route set for plain Actix in-process contract tests.
#[cfg(test)]
fn configure_actix_routes(cfg: &mut web::ServiceConfig) {
    register_api_routes!(cfg);
}

/// A configuration and its initialized pipeline are swapped together.
#[derive(Default)]
struct ActiveState {
    graphrag: Option<GraphRAG>,
    graph_built: bool,
}

/// Application state with optional Qdrant backend and full GraphRAG pipeline
#[derive(Clone)]
struct AppState {
    #[cfg(feature = "qdrant")]
    qdrant: Option<Arc<QdrantStore>>,

    // Embedding service (real or fallback)
    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    embeddings: Arc<EmbeddingService>,

    // The active pipeline and its metadata are one atomic snapshot.
    active: Arc<RwLock<ActiveState>>,

    // Authentication state (optional)
    #[cfg(feature = "auth")]
    auth: Arc<AuthState>,

    // Fallback in-memory storage (used when Qdrant unavailable or simple mode)
    documents: Arc<RwLock<Vec<Document>>>,
    query_count: Arc<RwLock<usize>>,
}

/// Connect to Qdrant and ensure the collection exists.
///
/// `QdrantStore::new` connects lazily, so the collection probe doubles as the
/// reachability check: an unreachable server returns `Err` and the caller falls
/// back to in-memory storage instead of failing every later request.
#[cfg(feature = "qdrant")]
async fn connect_qdrant(
    url: &str,
    collection: &str,
    dimension: u64,
) -> Result<QdrantStore, String> {
    let store = QdrantStore::new(url, collection)
        .await
        .map_err(|e| e.to_string())?;
    if store.collection_exists().await.map_err(|e| e.to_string())? {
        tracing::info!("✅ Connected to existing Qdrant collection: {}", collection);
    } else {
        match store.create_collection(dimension).await {
            Ok(_) => tracing::info!("✅ Created Qdrant collection: {}", collection),
            Err(e) => tracing::warn!("⚠️  Could not create collection: {}", e),
        }
    }
    Ok(store)
}

impl AppState {
    async fn new() -> Self {
        // Initialize embedding service
        let embedding_backend =
            std::env::var("EMBEDDING_BACKEND").unwrap_or_else(|_| "hash".to_string()); // Default to hash fallback
        let embedding_dim: usize = std::env::var("EMBEDDING_DIM")
            .unwrap_or_else(|_| "384".to_string())
            .parse()
            .unwrap_or(384);

        let embedding_config = EmbeddingConfig {
            backend: embedding_backend,
            dimension: embedding_dim,
            endpoint: std::env::var("EMBEDDING_ENDPOINT")
                .unwrap_or_else(|_| "http://127.0.0.1:8081/v1/embeddings".to_string()),
            model: std::env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "multilingual-e5-small".to_string()),
            enable_cache: true,
        };

        let embeddings = match EmbeddingService::new(embedding_config).await {
            Ok(service) => {
                tracing::info!(
                    "✅ Embedding service initialized: {}",
                    service.backend_name()
                );
                Arc::new(service)
            },
            Err(e) => {
                tracing::error!(
                    "❌ Failed to initialize embedding service: {}. Server may not work correctly.",
                    e
                );
                std::process::exit(1);
            },
        };

        #[cfg(feature = "qdrant")]
        {
            // Try to connect to Qdrant
            let qdrant_url =
                std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
            let collection_name =
                std::env::var("COLLECTION_NAME").unwrap_or_else(|_| "graphrag".to_string());

            match connect_qdrant(&qdrant_url, &collection_name, embedding_dim as u64).await {
                Ok(store) => {
                    tracing::info!("🗄️  Using Qdrant at: {}", qdrant_url);

                    Self {
                        qdrant: Some(Arc::new(store)),
                        embeddings,
                        active: Arc::new(RwLock::new(ActiveState::default())),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                            |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                        ))),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        query_count: Arc::new(RwLock::new(0)),
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        "⚠️  Could not connect to Qdrant: {}. Using in-memory storage.",
                        e
                    );
                    Self {
                        qdrant: None,
                        embeddings,
                        active: Arc::new(RwLock::new(ActiveState::default())),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                            |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                        ))),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        query_count: Arc::new(RwLock::new(0)),
                    }
                },
            }
        }

        #[cfg(not(feature = "qdrant"))]
        {
            tracing::info!("📦 Using in-memory storage (Qdrant feature disabled)");
            Self {
                embeddings,
                active: Arc::new(RwLock::new(ActiveState::default())),
                #[cfg(feature = "auth")]
                auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                    |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                ))),
                documents: Arc::new(RwLock::new(Vec::new())),
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

// ============================================================================
// API Handlers
// ============================================================================

/// Root endpoint - API information
#[api_operation(
    tag = "info",
    summary = "Get API information",
    description = "Returns basic information about the GraphRAG API, including version, status, and available endpoints"
)]
async fn root(state: Data<AppState>) -> impl Responder {
    Json(json!({
        "name": "GraphRAG REST API",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "backend": if state.has_qdrant() { "qdrant" } else { "memory" },
        "graphrag_configured": state.active.read().await.graphrag.is_some(),
        "documentation": "/swagger",
        "openapi_spec": "/openapi.json",
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
#[api_operation(
    tag = "health",
    summary = "Health check",
    description = "Returns the current health status of the service, including document count, graph status, and total queries processed"
)]
async fn health(state: Data<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    let doc_count;
    let active = state.active.read().await;
    let pipeline_configured = active.graphrag.is_some();
    let active_graph_built = active.graph_built;
    drop(active);
    let graph_built;
    let query_count = *state.query_count.read().await;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                doc_count = count;
                graph_built = pipeline_configured && active_graph_built;
            },
            Err(_) => {
                doc_count = 0;
                graph_built = false;
            },
        }
    } else {
        doc_count = state.documents.read().await.len();
        graph_built = pipeline_configured && active_graph_built;
    }

    #[cfg(not(feature = "qdrant"))]
    {
        doc_count = state.documents.read().await.len();
        graph_built = pipeline_configured && active_graph_built;
    }

    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        document_count: doc_count,
        graph_built,
        total_queries: query_count,
        backend: if state.has_qdrant() {
            "qdrant".to_string()
        } else {
            "memory".to_string()
        },
    }))
}

/// Query the knowledge graph
#[api_operation(
    tag = "query",
    summary = "Query the knowledge graph",
    description = "Search documents using semantic similarity. Returns ranked results with similarity scores.",
    error_code = 400,
    error_code = 500
)]
async fn query(
    state: Data<AppState>,
    body: Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    // Validate input
    if let Err(e) = validate_query(&body.query) {
        tracing::warn!(query = %body.query, error = %e.error, "Invalid query");
        return Err(ApiError::BadRequest(e.error));
    }

    if let Err(e) = validate_top_k(body.top_k) {
        tracing::warn!(top_k = body.top_k, error = %e.error, "Invalid top_k");
        return Err(ApiError::BadRequest(e.error));
    }

    let start = std::time::Instant::now();

    // Increment query count
    *state.query_count.write().await += 1;

    // A configured, built pipeline answers with its own retrieval (graph + vectors).
    {
        let mut active = state.active.write().await;
        let graph_built = active.graph_built;
        if let (Some(graphrag), true) = (active.graphrag.as_mut(), graph_built) {
            let search_results = graphrag
                .search(&body.query)
                .await
                .map_err(|e| ApiError::InternalError(format!("GraphRAG search failed: {e}")))?;
            let titles = state.documents.read().await;
            let results = pipeline_query_results(graphrag, search_results, body.top_k, |id| {
                titles.iter().find(|d| d.id == id).map(|d| d.title.clone())
            });
            return Ok(Json(QueryResponse {
                query: body.query.clone(),
                results,
                processing_time_ms: start.elapsed().as_millis() as u64,
                backend: "graphrag-pipeline".to_string(),
            }));
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Real vector search with Qdrant using real embeddings
        let query_embedding = match state.embeddings.generate_query(&body.query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                tracing::error!("Failed to generate query embedding: {}", e);
                return Err(ApiError::InternalError(format!(
                    "Failed to generate embedding: {e}"
                )));
            },
        };

        match qdrant.search(query_embedding, body.top_k, None).await {
            Ok(search_results) => {
                let results: Vec<QueryResult> = search_results
                    .into_iter()
                    .map(|r| QueryResult {
                        document_id: r.id,
                        title: r.metadata.title,
                        similarity: r.score,
                        excerpt: response_excerpt(&r.metadata.text),
                    })
                    .collect();

                let processing_time = start.elapsed().as_millis() as u64;

                return Ok(Json(QueryResponse {
                    query: body.query.clone(),
                    results,
                    processing_time_ms: processing_time,
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Qdrant search failed: {e}"
                )));
            },
        }
    }

    // Fallback: in-memory search
    let documents = state.documents.read().await;

    if documents.is_empty() {
        return Err(ApiError::BadRequest(
            "No documents available. Add documents first.".to_string(),
        ));
    }

    // Simple keyword matching for demonstration
    let mut results: Vec<QueryResult> = documents
        .iter()
        .map(|doc| {
            let query_lower = body.query.to_lowercase();
            let content_lower = doc.content.to_lowercase();
            let title_lower = doc.title.to_lowercase();

            let similarity =
                if content_lower.contains(&query_lower) || title_lower.contains(&query_lower) {
                    0.85
                } else {
                    0.1
                };

            let excerpt = response_excerpt(&doc.content);

            QueryResult {
                document_id: doc.id.clone(),
                title: doc.title.clone(),
                similarity,
                excerpt,
            }
        })
        .filter(|r| r.similarity > 0.5)
        .collect();

    results.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(body.top_k);

    let processing_time = start.elapsed().as_millis() as u64;

    Ok(Json(QueryResponse {
        query: body.query.clone(),
        results,
        processing_time_ms: processing_time,
        backend: "memory".to_string(),
    }))
}

/// Add a document to the knowledge graph
#[api_operation(
    tag = "documents",
    summary = "Add a new document",
    description = "Add a new document to the knowledge graph. The document will be embedded and indexed for search.",
    error_code = 400,
    error_code = 500
)]
async fn add_document(
    state: Data<AppState>,
    body: Json<AddDocumentRequest>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    // Validate input
    if let Err(e) = validate_title(&body.title) {
        tracing::warn!(title = %body.title, error = %e.error, "Invalid title");
        return Err(ApiError::BadRequest(e.error));
    }

    if let Err(e) = validate_content(&body.content) {
        tracing::warn!(content_len = body.content.len(), error = %e.error, "Invalid content");
        return Err(ApiError::BadRequest(e.error));
    }

    // Sanitize inputs
    let title = sanitize_string(&body.title);
    let content = sanitize_string(&body.content);

    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Generate real embeddings
        let embedding = match state.embeddings.generate_single(&content).await {
            Ok(emb) => emb,
            Err(e) => {
                tracing::error!("Failed to generate document embedding: {}", e);
                return Err(ApiError::InternalError(format!(
                    "Failed to generate embedding: {e}"
                )));
            },
        };

        let metadata = DocumentMetadata {
            id: id.clone(),
            title: title.clone(),
            text: content.clone(),
            chunk_index: 0,
            entities: Vec::new(),
            relationships: Vec::new(),
            timestamp: timestamp.clone(),
            custom: HashMap::new(),
        };

        match qdrant.add_document(&id, embedding, metadata).await {
            Ok(_) => {
                add_to_configured_graphrag(&state, &id, &title, &content).await?;
                tracing::info!("Added document to Qdrant: {} ({})", title, id);

                return Ok(Json(DocumentOperationResponse {
                    success: true,
                    document_id: Some(id),
                    message: "Document added to Qdrant successfully".to_string(),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to add document to Qdrant: {e}"
                )));
            },
        }
    }

    // Fallback: in-memory storage
    let document = Document {
        id: id.clone(),
        title,
        content,
        added_at: timestamp,
    };

    state.documents.write().await.push(document.clone());
    add_to_configured_graphrag(&state, &id, &document.title, &document.content).await?;
    tracing::info!("Added document to memory: {} ({})", document.title, id);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(id),
        message: "Document added to memory successfully".to_string(),
        backend: "memory".to_string(),
    }))
}

async fn add_to_configured_graphrag(
    state: &Data<AppState>,
    id: &str,
    title: &str,
    content: &str,
) -> Result<(), ApiError> {
    let mut active = state.active.write().await;
    if let Some(graphrag) = active.graphrag.as_mut() {
        let pipeline_error = |e| {
            ApiError::InternalError(format!("Failed to add document to GraphRAG pipeline: {e}"))
        };
        let mut document = graphrag_core::core::Document::new(
            graphrag_core::core::DocumentId::new(id.to_string()),
            title.to_string(),
            content.to_string(),
        );
        // build_graph() only processes chunks: an unchunked document yields an empty graph.
        let text = &graphrag.config().text;
        document.chunks =
            graphrag_core::text::TextProcessor::new(text.chunk_size, text.chunk_overlap)
                .and_then(|processor| processor.chunk_text(&document))
                .map_err(pipeline_error)?;
        graphrag.add_document(document).map_err(pipeline_error)?;
        active.graph_built = false;
    }
    Ok(())
}

/// List all documents
#[api_operation(
    tag = "documents",
    summary = "List all documents",
    description = "Retrieve a list of all documents in the knowledge graph"
)]
async fn list_documents(state: Data<AppState>) -> Json<ListDocumentsResponse> {
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _vectors)) => {
                return Json(ListDocumentsResponse {
                    documents: Vec::new(),
                    total: count,
                    backend: "qdrant".to_string(),
                    note: Some("Full document listing from Qdrant not implemented yet".to_string()),
                });
            },
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let documents = state.documents.read().await;

    let doc_list: Vec<DocumentSummary> = documents
        .iter()
        .map(|doc| DocumentSummary {
            id: doc.id.clone(),
            title: doc.title.clone(),
            content_length: doc.content.len(),
            added_at: doc.added_at.clone(),
        })
        .collect();

    Json(ListDocumentsResponse {
        documents: doc_list.clone(),
        total: doc_list.len(),
        backend: "memory".to_string(),
        note: None,
    })
}

/// Delete a document
#[api_operation(
    tag = "documents",
    summary = "Delete a document",
    description = "Remove a document from storage-only mode by ID; unavailable while a GraphRAG pipeline is configured",
    error_code = 404,
    error_code = 500
)]
async fn delete_document(
    state: Data<AppState>,
    id: WebPath<String>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    let doc_id = id.into_inner();

    // The active pipeline is a single config/graph snapshot, but the core
    // graph currently has no atomic document-removal operation. Refuse before
    // touching Qdrant or memory so a successful response cannot leave the
    // configured graph stale.
    if state.active.read().await.graphrag.is_some() {
        return Err(ApiError::BadRequest(
            "Document deletion is unavailable while a GraphRAG pipeline is configured; rebuild the pipeline from its workspace instead"
                .to_string(),
        ));
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.delete_document(&doc_id).await {
            Ok(_) => {
                tracing::info!("Deleted document from Qdrant: {}", doc_id);
                return Ok(Json(DocumentOperationResponse {
                    success: true,
                    document_id: Some(doc_id.clone()),
                    message: format!("Document {doc_id} deleted from Qdrant"),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to delete from Qdrant: {e}"
                )));
            },
        }
    }

    // Fallback: in-memory storage
    let mut documents = state.documents.write().await;
    let original_len = documents.len();
    documents.retain(|doc| doc.id != doc_id);

    if documents.len() == original_len {
        return Err(ApiError::NotFound(format!(
            "Document with id '{doc_id}' not found"
        )));
    }

    tracing::info!("Deleted document from memory: {}", doc_id);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(doc_id.clone()),
        message: format!("Document {doc_id} deleted from memory"),
        backend: "memory".to_string(),
    }))
}

/// Build the knowledge graph
#[api_operation(
    tag = "graph",
    summary = "Build the knowledge graph",
    description = "Process all documents and build the knowledge graph structure",
    error_code = 400,
    error_code = 500
)]
async fn build_graph(state: Data<AppState>) -> Result<Json<BuildGraphResponse>, ApiError> {
    let start = std::time::Instant::now();

    // A configured GraphRAG pipeline is the only backend that can build a
    // knowledge graph.  Keep the state lock while building so readers cannot
    // observe a config/metadata pair from different generations.
    {
        let mut active = state.active.write().await;
        if let Some(graphrag) = active.graphrag.as_mut() {
            graphrag.build_graph().await.map_err(|e| {
                ApiError::InternalError(format!("GraphRAG pipeline build failed: {e}"))
            })?;
            let processing_time = start.elapsed().as_millis() as u64;
            let (entities, relationships, document_count) = graphrag
                .knowledge_graph()
                .map(|kg| {
                    (
                        kg.entities().count(),
                        kg.relationships().count(),
                        kg.documents().count(),
                    )
                })
                .unwrap_or((0, 0, 0));
            active.graph_built = true;

            tracing::info!(
                "Built knowledge graph via pipeline in {}ms ({} entities, {} relationships)",
                processing_time,
                entities,
                relationships
            );

            return Ok(Json(BuildGraphResponse {
                success: true,
                document_count,
                processing_time_ms: processing_time,
                message: format!(
                    "Knowledge graph built: {entities} entities, {relationships} relationships"
                ),
                backend: "graphrag-pipeline".to_string(),
            }));
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                if count == 0 {
                    return Err(ApiError::BadRequest(
                        "No documents in Qdrant. Add documents first.".to_string(),
                    ));
                }

                return Err(ApiError::BadRequest(format!(
                    "Knowledge graph construction is unavailable for the Qdrant document store ({count} documents); configure a GraphRAG pipeline first"
                )));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to access Qdrant: {e}"
                )));
            },
        }
    }

    // The memory backend indexes documents for local retrieval, but does not
    // claim to have built a knowledge graph.
    let doc_count = state.documents.read().await.len();

    if doc_count == 0 {
        return Err(ApiError::BadRequest(
            "No documents to build graph from. Add documents first.".to_string(),
        ));
    }

    Err(ApiError::BadRequest(format!(
        "Knowledge graph construction is unavailable for the in-memory document store ({doc_count} documents); configure a GraphRAG pipeline first"
    )))
}

/// Get graph statistics
#[api_operation(
    tag = "graph",
    summary = "Get graph statistics",
    description = "Retrieve statistics about the knowledge graph, including document count, entity count, and relationship count"
)]
async fn graph_stats(state: Data<AppState>) -> Json<GraphStatsResponse> {
    // Try real GraphRAG pipeline stats first
    {
        let active = state.active.read().await;
        if let Some(graphrag) = active.graphrag.as_ref() {
            if let Some(kg) = graphrag.knowledge_graph() {
                let entity_count = kg.entities().count();
                let relationship_count = kg.relationships().count();
                let doc_count = kg.documents().count();
                let chunk_count = kg.chunks().count();

                return Json(GraphStatsResponse {
                    document_count: doc_count,
                    entity_count,
                    relationship_count,
                    vector_count: chunk_count,
                    graph_built: active.graph_built,
                    backend: "graphrag-pipeline".to_string(),
                });
            }
        }
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, vectors)) => {
                return Json(GraphStatsResponse {
                    document_count: count,
                    entity_count: 0,
                    relationship_count: 0,
                    vector_count: vectors,
                    graph_built: count > 0,
                    backend: "qdrant".to_string(),
                });
            },
            Err(e) => {
                tracing::error!("Failed to get Qdrant stats: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();
    Json(GraphStatsResponse {
        document_count: doc_count,
        entity_count: 0,
        relationship_count: 0,
        vector_count: 0,
        graph_built: false,
        backend: "memory".to_string(),
    })
}

// ============================================================================
// Authentication Endpoints (feature-gated)
// ============================================================================

#[cfg(feature = "auth")]
#[api_operation(
    tag = "auth",
    summary = "User login",
    description = "Authenticate user and receive JWT token",
    error_code = 401,
    error_code = 500
)]
async fn login(
    state: Data<AppState>,
    body: Json<LoginRequest>,
) -> Result<Json<LoginResponse>, ApiError> {
    // TODO: Implement real user authentication against database
    // For now, accept any credentials for demo purposes
    tracing::info!("Login attempt for user: {}", body.username);

    let role = if body.username == "admin" {
        auth::UserRole::Admin
    } else {
        auth::UserRole::User
    };

    match state.auth.generate_token(&body.username, role.clone(), 24) {
        Ok(token) => {
            tracing::info!(
                "✅ Generated JWT token for user: {} (role: {:?})",
                body.username,
                role
            );
            Ok(Json(LoginResponse {
                success: true,
                token,
                user_id: body.username.clone(),
                role: format!("{:?}", role),
                expires_in_hours: 24,
                usage: "Add header: Authorization: Bearer <token>".to_string(),
            }))
        },
        Err(e) => {
            tracing::error!("❌ Failed to generate token: {}", e);
            Err(ApiError::InternalError(format!(
                "Token generation failed: {}",
                e
            )))
        },
    }
}

#[cfg(feature = "auth")]
#[api_operation(
    tag = "auth",
    summary = "Create API key",
    description = "Generate an API key for programmatic access",
    error_code = 500
)]
async fn create_api_key(
    state: Data<AppState>,
    body: Json<ApiKeyRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let role = body
        .role
        .as_deref()
        .and_then(|r| match r {
            "Admin" => Some(auth::UserRole::Admin),
            _ => Some(auth::UserRole::User),
        })
        .unwrap_or(auth::UserRole::User);

    match state
        .auth
        .create_api_key(&body.user_id, role.clone(), None)
        .await
    {
        Ok(api_key) => {
            tracing::info!(
                "✅ Created API key for user: {} (role: {:?})",
                body.user_id,
                role
            );
            Ok(Json(json!({
                "success": true,
                "api_key": api_key,
                "user_id": body.user_id,
                "role": format!("{:?}", role),
                "usage": "Add header: Authorization: ApiKey <key>",
                "rate_limit": {
                    "max_requests": 1000,
                    "window_seconds": 3600
                }
            })))
        },
        Err(e) => {
            tracing::error!("❌ Failed to create API key: {}", e);
            Err(ApiError::InternalError(format!(
                "API key creation failed: {}",
                e
            )))
        },
    }
}

// ============================================================================
// Main Server Configuration
// ============================================================================

#[actix_web::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let bind_addr = parse_bind_addr(std::env::var("GRAPHRAG_BIND_ADDR").ok().as_deref())?;
    let cors_origins = parse_cors_origins(std::env::var("GRAPHRAG_CORS_ORIGINS").ok().as_deref())?;

    // Create application state (connects to Qdrant if available)
    let state = AppState::new().await;
    let state_data = Data::new(state.clone());

    // Configure OpenAPI specification
    let spec = Spec {
        info: Info {
            title: "GraphRAG REST API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: Some(concat!(
                "Local REST API for GraphRAG document retrieval and configured graph pipelines.\n\n",
                "## Features\n",
                "- Semantic search over documents\n",
                "- Knowledge graph construction through an initialized GraphRAG pipeline\n",
                "- Local lexical or vector retrieval\n",
                "- Qdrant document storage (optional)\n",
                "- No multi-user authentication; keep the listener on a trusted network\n\n",
                "## Getting Started\n",
                "1. Configure a graph pipeline via `POST /api/config` when graph construction is needed\n",
                "2. Add documents via `POST /api/documents`\n",
                "3. Build the configured graph via `POST /api/graph/build`\n",
                "4. Query via `POST /api/query`\n"
            ).to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    // JWT secret warning (item 3.4)
    #[cfg(feature = "auth")]
    if std::env::var("JWT_SECRET").is_err() {
        tracing::warn!(
            "⚠️  JWT_SECRET not set! Using insecure default. Set JWT_SECRET env var in production."
        );
    }

    tracing::info!("🚀 GraphRAG Server starting...");
    tracing::info!("📡 Listening on http://{}", bind_addr);
    tracing::info!("📚 Swagger UI: http://{}/swagger", bind_addr);
    tracing::info!("📄 OpenAPI spec: http://{}/openapi.json", bind_addr);
    tracing::info!(
        "🗄️  Backend: {}",
        if state.has_qdrant() {
            "Qdrant"
        } else {
            "In-memory"
        }
    );

    HttpServer::new(move || {
        let cors = build_cors(cors_origins.as_deref());

        App::new()
            // OpenAPI documentation
            .document(spec.clone())

            // Global middleware
            .wrap(cors)
            .wrap(actix_web::middleware::Logger::default())

            // Application state
            .app_data(state_data.clone())

            // Request body size limits (10MB for general payload, 10MB for JSON)
            .app_data(web::PayloadConfig::new(validation::MAX_BODY_SIZE))
            .app_data(json_config())

            // `/swagger/` and other trailing-slash variants resolve to their canonical route.
            .wrap(actix_web::middleware::NormalizePath::trim())

            // Public routes
            .service(resource("/").route(get().to(root)))
            .service(resource("/health").route(get().to(health)))

            // Register the same route configuration used by in-process tests.
            .configure(configure_routes)

            // Auth routes (temporarily disabled - feature "auth" is disabled)
            // #[cfg(feature = "auth")]
            // .service(
            //     scope("/auth")
            //         .service(resource("/login").route(post().to(login)))
            //         .service(resource("/api-key").route(post().to(create_api_key)))
            // )

            // Build OpenAPI spec endpoint + interactive Swagger UI at /swagger
            .build_with(
                "/openapi.json",
                BuildConfig::default().with(SwaggerUIConfig::new(&"/swagger")),
            )
    })
    .bind(bind_addr)?
    .run()
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn memory_state() -> Data<AppState> {
        let embeddings = EmbeddingService::new(EmbeddingConfig {
            backend: "hash".to_string(),
            dimension: 384,
            endpoint: "http://127.0.0.1:9/v1/embeddings".to_string(),
            model: "unused".to_string(),
            enable_cache: false,
        })
        .await
        .expect("hash embedding service should initialize without network");

        Data::new(AppState {
            #[cfg(feature = "qdrant")]
            qdrant: None,
            embeddings: Arc::new(embeddings),
            active: Arc::new(RwLock::new(ActiveState::default())),
            #[cfg(feature = "auth")]
            auth: Arc::new(AuthState::new(
                "test_secret_key_32_characters_long".to_string(),
            )),
            documents: Arc::new(RwLock::new(Vec::new())),
            query_count: Arc::new(RwLock::new(0)),
        })
    }

    #[tokio::test]
    async fn configured_pipeline_chunks_added_documents_before_build() {
        let state = memory_state().await;
        let mut graphrag = GraphRAG::new(graphrag_core::Config::default()).unwrap();
        graphrag.initialize().unwrap();
        state.active.write().await.graphrag = Some(graphrag);

        add_to_configured_graphrag(
            &state,
            "doc-1",
            "Promessi",
            "Renzo Tramaglino vive a Lecco e vuole sposare Lucia Mondella. \
             Don Rodrigo minaccia Don Abbondio. Fra Cristoforo aiuta Renzo e Lucia.",
        )
        .await
        .unwrap();

        let mut active = state.active.write().await;
        let graphrag = active.graphrag.as_mut().unwrap();
        let graph = graphrag.knowledge_graph().unwrap();
        assert!(graph.chunks().count() > 0, "added document must be chunked");

        graphrag.build_graph().await.unwrap();
        assert!(graphrag.knowledge_graph().unwrap().entities().count() > 0);
    }

    #[cfg(feature = "qdrant")]
    #[tokio::test]
    async fn unreachable_qdrant_is_not_selected_as_backend() {
        assert!(connect_qdrant("http://127.0.0.1:9", "graphrag_test", 384)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn query_uses_configured_pipeline_after_build() {
        let state = memory_state().await;
        let mut graphrag = GraphRAG::new(graphrag_core::Config::default()).unwrap();
        graphrag.initialize().unwrap();
        state.active.write().await.graphrag = Some(graphrag);
        add_to_configured_graphrag(
            &state,
            "doc-1",
            "Prisma",
            "Maria Rossi directs the Prisma project at Aurora Laboratory in Turin.",
        )
        .await
        .unwrap();
        build_graph(state.clone()).await.unwrap();

        let response = query(
            state.clone(),
            Json(QueryRequest {
                query: "Who directs the Prisma project?".to_string(),
                top_k: 3,
            }),
        )
        .await
        .unwrap();

        assert_eq!(response.backend, "graphrag-pipeline");
        assert_eq!(
            response.results.len(),
            1,
            "results are grouped per document"
        );
        assert_eq!(response.results[0].document_id, "doc-1");
        assert_eq!(response.results[0].title, "Prisma");
        assert!(response.results[0].excerpt.contains("Maria Rossi"));
        assert!((0.0..=1.0).contains(&response.results[0].similarity));
    }

    #[actix_web::test]
    async fn malformed_json_body_returns_json_error() {
        let state = memory_state().await;
        let app = actix_web::test::init_service(
            App::new()
                .app_data(state)
                .app_data(json_config())
                .configure(configure_actix_routes),
        )
        .await;
        let response = actix_web::test::call_service(
            &app,
            actix_web::test::TestRequest::post()
                .uri("/api/query")
                .insert_header(("content-type", "application/json"))
                .set_payload("{not json")
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), actix_web::http::StatusCode::BAD_REQUEST);
        let body: serde_json::Value = actix_web::test::read_body_json(response).await;
        assert_eq!(body["error"], "Bad Request");
        assert!(body["message"].as_str().unwrap().contains("Json"));
    }

    #[test]
    fn bind_defaults_to_loopback_and_rejects_malformed_values() {
        assert_eq!(
            parse_bind_addr(None).unwrap(),
            "127.0.0.1:8080".parse::<SocketAddr>().unwrap()
        );
        assert_eq!(
            parse_bind_addr(Some("0.0.0.0:18080")).unwrap(),
            "0.0.0.0:18080".parse::<SocketAddr>().unwrap()
        );
        assert!(parse_bind_addr(Some("localhost:8080")).is_err());
    }

    #[test]
    fn cors_requires_explicit_http_origins() {
        assert!(parse_cors_origins(None).unwrap().is_none());
        assert_eq!(
            parse_cors_origins(Some(" http://localhost:3000,https://app.example.test "))
                .unwrap()
                .unwrap(),
            vec![
                "http://localhost:3000".to_string(),
                "https://app.example.test".to_string()
            ]
        );
        assert!(parse_cors_origins(Some("*")).is_err());
        assert!(parse_cors_origins(Some("localhost:3000")).is_err());
        assert!(parse_cors_origins(Some(" ")).is_err());
        assert!(parse_cors_origins(Some("http://localhost:3000,")).is_err());
        assert!(parse_cors_origins(Some("http://localhost:3000/path")).is_err());
        assert!(parse_cors_origins(Some("http://user:pass@localhost:3000")).is_err());
        assert!(parse_cors_origins(Some("http://[::1]:3000")).is_ok());
    }

    #[test]
    fn excerpts_never_split_utf8() {
        let text = format!("{} end", "文".repeat(100));
        let excerpt = response_excerpt(&text);
        assert!(excerpt.is_char_boundary(excerpt.len()));
        assert!(excerpt.starts_with(&"文".repeat(66)));
        assert!(excerpt.ends_with("..."));
    }

    #[actix_web::test]
    async fn invalid_config_does_not_replace_active_snapshot() {
        let state = memory_state().await;
        config_endpoints::set_config(state.clone(), Json(json!({"retrieval": {"top_k": 3}})))
            .await
            .expect("valid candidate should initialize");
        let before = config_endpoints::get_config(state.clone())
            .await
            .expect("active config")
            .0;

        let error =
            config_endpoints::set_config(state.clone(), Json(json!({"retrieval": {"top_k": 0}})))
                .await
                .expect_err("invalid candidate must be rejected");
        assert!(error.to_string().contains("Top-k"));
        let after = config_endpoints::get_config(state)
            .await
            .expect("active config remains available")
            .0;
        assert_eq!(before, after);
    }

    #[actix_web::test]
    async fn failed_workspace_initialization_does_not_replace_active_snapshot() {
        let state = memory_state().await;
        config_endpoints::set_config(state.clone(), Json(json!({"retrieval": {"top_k": 3}})))
            .await
            .expect("valid candidate should initialize");

        add_document(
            state.clone(),
            Json(AddDocumentRequest {
                title: "Rollback document".to_string(),
                content: "The active graph survives a failed replacement.".to_string(),
            }),
        )
        .await
        .expect("document should be added to the active pipeline");
        build_graph(state.clone())
            .await
            .expect("active pipeline should build before replacement");
        let before = config_endpoints::get_config(state.clone())
            .await
            .expect("active config")
            .0;
        let before_stats = graph_stats(state.clone()).await.0;

        let base_dir =
            std::env::temp_dir().join(format!("graphrag-server-corrupt-{}", uuid::Uuid::new_v4()));
        let workspace = base_dir.join("active");
        std::fs::create_dir_all(&workspace).expect("create corrupt workspace");
        std::fs::write(workspace.join("graph.json"), b"{ definitely not graph json")
            .expect("write corrupt graph");

        let candidate = json!({
            "retrieval": {"top_k": 3},
            "auto_save": {
                "enabled": true,
                "base_dir": base_dir,
                "workspace_name": "active"
            }
        });
        let error = config_endpoints::set_config(state.clone(), Json(candidate))
            .await
            .expect_err("corrupt workspace must reject candidate");
        assert!(error.to_string().contains("initialization"));

        let after = config_endpoints::get_config(state.clone())
            .await
            .expect("active config remains available")
            .0;
        assert_eq!(before, after);
        assert_eq!(graph_stats(state.clone()).await.0, before_stats);
        let query = query(
            state,
            Json(QueryRequest {
                query: "survives".to_string(),
                top_k: 3,
            }),
        )
        .await
        .expect("previous graph remains queryable")
        .0;
        assert_eq!(query.results.len(), 1);
        std::fs::remove_dir_all(base_dir).expect("remove corrupt workspace");
    }

    #[actix_web::test]
    async fn production_routes_cover_config_documents_query_and_cors() {
        let state = memory_state().await;
        let origins = vec!["http://localhost:3000".to_string()];
        let app = actix_web::test::init_service(
            App::new()
                .wrap(build_cors(Some(&origins)))
                .app_data(state)
                .configure(configure_actix_routes),
        )
        .await;

        let response = actix_web::test::call_service(
            &app,
            actix_web::test::TestRequest::post()
                .uri("/api/config")
                .set_json(json!({"retrieval": {"top_k": 3}}))
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);

        let response = actix_web::test::call_service(
            &app,
            actix_web::test::TestRequest::get()
                .uri("/api/config")
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);

        let response = actix_web::test::call_service(
            &app,
            actix_web::test::TestRequest::post()
                .uri("/api/documents")
                .set_json(json!({
                    "title": "日本語 document",
                    "content": "日本語の内容と UTF-8 を検索できます。"
                }))
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);

        let response = actix_web::test::call_service(
            &app,
            actix_web::test::TestRequest::get()
                .uri("/api/documents")
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);

        let query_response: QueryResponse = actix_web::test::call_and_read_body_json(
            &app,
            actix_web::test::TestRequest::post()
                .uri("/api/query")
                .set_json(json!({"query": "日本語", "topK": 3}))
                .to_request(),
        )
        .await;
        assert_eq!(query_response.results.len(), 1);

        let response = actix_web::test::call_service(
            &app,
            actix_web::test::TestRequest::default()
                .method(actix_web::http::Method::OPTIONS)
                .uri("/api/query")
                .insert_header((header::ORIGIN, "http://localhost:3000"))
                .insert_header((header::ACCESS_CONTROL_REQUEST_METHOD, "POST"))
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .unwrap(),
            "http://localhost:3000"
        );
    }
}
