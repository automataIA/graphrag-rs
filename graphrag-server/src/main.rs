//! GraphRAG REST API Server with Actix-web and Apistos OpenAPI
//!
//! Production-ready REST API for GraphRAG operations with automatic OpenAPI documentation.
//!
//! ## Features
//! - Automatic OpenAPI 3.0.3 documentation via Apistos
//! - Interactive Swagger UI at /swagger
//! - Qdrant vector database integration (optional)
//! - JWT and API key authentication (optional)
//! - Request validation and rate limiting
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
//!
//! # 4. View Swagger UI
//! # Browser: http://localhost:8080/swagger
//! ```

use actix_cors::Cors;
use actix_web::{
    web::{self, Data, Json, Path as WebPath},
    App, HttpServer, Responder,
};
use apistos::{
    api_operation,
    app::OpenApiWrapper,
    info::Info,
    spec::Spec,
    web::{delete, get, post, resource, scope},
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber;

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
use config_handler::ConfigManager;

mod config_endpoints;

#[cfg(feature = "qdrant")]
mod graph_persistence;

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
    /// RFC 3339 timestamp of the last successful /api/graph/build (None
    /// before the first build). Surfaced via /api/graph/stats so agents
    /// can decide whether the graph is fresh enough to query.
    last_built_at: Arc<RwLock<Option<String>>>,
    query_count: Arc<RwLock<usize>>,
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
            ollama_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost".to_string()),
            ollama_model: std::env::var("OLLAMA_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string()),
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

            match QdrantStore::new(&qdrant_url, &collection_name).await {
                Ok(store) => {
                    // Check if collection exists, create if not
                    if !store.collection_exists().await.unwrap_or(false) {
                        match store.create_collection(embedding_dim as u64).await {
                            Ok(_) => {
                                tracing::info!("✅ Created Qdrant collection: {}", collection_name);
                            },
                            Err(e) => {
                                tracing::warn!("⚠️  Could not create collection: {}", e);
                            },
                        }
                    } else {
                        tracing::info!(
                            "✅ Connected to existing Qdrant collection: {}",
                            collection_name
                        );
                    }

                    tracing::info!("🗄️  Using Qdrant at: {}", qdrant_url);

                    Self {
                        qdrant: Some(Arc::new(store)),
                        embeddings,
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                            |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                        ))),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        last_built_at: Arc::new(RwLock::new(None)),
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
                        graphrag: Arc::new(RwLock::new(None)),
                        config_manager: Arc::new(ConfigManager::new()),
                        #[cfg(feature = "auth")]
                        auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                            |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                        ))),
                        documents: Arc::new(RwLock::new(Vec::new())),
                        graph_built: Arc::new(RwLock::new(false)),
                        last_built_at: Arc::new(RwLock::new(None)),
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
                graphrag: Arc::new(RwLock::new(None)),
                config_manager: Arc::new(ConfigManager::new()),
                #[cfg(feature = "auth")]
                auth: Arc::new(AuthState::new(std::env::var("JWT_SECRET").unwrap_or_else(
                    |_| "graphrag_secret_key_change_in_production_32chars".to_string(),
                ))),
                documents: Arc::new(RwLock::new(Vec::new())),
                graph_built: Arc::new(RwLock::new(false)),
                last_built_at: Arc::new(RwLock::new(None)),
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
        "graphrag_configured": state.graphrag.read().await.is_some(),
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
            "query": {
                "endpoint": "POST /api/query",
                "modes": {
                    "search": "vector similarity over Qdrant (default; fast; no LLM)",
                    "ask": "graph-aware retrieval + LLM-composed answer",
                    "explain": "ask + confidence + source attribution + reasoning trace",
                    "reason": "query decomposition for multi-hop questions"
                }
            },
            "documents": {
                "list": "GET /api/documents",
                "add": "POST /api/documents",
                "delete": "DELETE /api/documents/{id}"
            },
            "graph": {
                "build": "POST /api/graph/build",
                "append": "POST /api/graph/append",
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
    let graph_built;
    let query_count = *state.query_count.read().await;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        match qdrant.stats().await {
            Ok((count, _)) => {
                doc_count = count;
                graph_built = count > 0;
            },
            Err(_) => {
                doc_count = 0;
                graph_built = false;
            },
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
///
/// Routes by `mode`:
/// - `search` (default): Qdrant vector search; returns ranked excerpts.
///   ~350ms, no LLM call.
/// - `ask`: graph-aware retrieval + LLM-generated answer. Slower
///   (LLM round-trip) but produces a synthesized response, not just
///   excerpts.
/// - `explain`: same as `ask` plus confidence, source attribution
///   (chunks + entities + relationships), reasoning steps, and
///   key entities the answer relied on.
/// - `reason`: query decomposition for multi-hop questions; sub-queries
///   are answered and composed. Slowest but best for compound questions.
///
/// `ask`/`explain`/`reason` require a configured chat backend (POST /config
/// with `openai.enabled = true` or `ollama.enabled = true`). Without one
/// they return 400.
#[api_operation(
    tag = "query",
    summary = "Query the knowledge graph",
    description = "Search documents (mode=search, default) or ask the graph-aware engine for an LLM-composed answer (mode=ask|explain|reason).",
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

    let mode = body.mode.unwrap_or_default();

    // Graph-aware modes: dispatch to graphrag-core. We always also attach
    // the vector-search hits as `results` so the caller still gets source
    // excerpts even when reading the LLM `answer`.
    if !matches!(mode, QueryMode::Search) {
        return graph_aware_query(&state, &body, mode, start).await;
    }

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Real vector search with Qdrant using real embeddings
        let query_embedding = match state.embeddings.generate_single(&body.query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                tracing::error!("Failed to generate query embedding: {}", e);
                return Err(ApiError::InternalError(format!(
                    "Failed to generate embedding: {}",
                    e
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
                        excerpt: if r.metadata.text.len() > 200 {
                            format!("{}...", &r.metadata.text[..200])
                        } else {
                            r.metadata.text
                        },
                    })
                    .collect();

                let processing_time = start.elapsed().as_millis() as u64;

                return Ok(Json(QueryResponse {
                    query: body.query.clone(),
                    mode: mode.as_str().to_string(),
                    results,
                    answer: None,
                    confidence: None,
                    key_entities: None,
                    reasoning_steps: None,
                    sources: None,
                    processing_time_ms: processing_time,
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Qdrant search failed: {}",
                    e
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
    results.truncate(body.top_k);

    let processing_time = start.elapsed().as_millis() as u64;

    Ok(Json(QueryResponse {
        query: body.query.clone(),
        mode: mode.as_str().to_string(),
        results,
        answer: None,
        confidence: None,
        key_entities: None,
        reasoning_steps: None,
        sources: None,
        processing_time_ms: processing_time,
        backend: "memory".to_string(),
    }))
}

/// Graph-aware query path. Dispatches to `GraphRAG::ask`, `ask_explained`,
/// or `ask_with_reasoning` depending on `mode`. Always also runs a vector
/// search in parallel so the caller gets `results` (source excerpts) even
/// when the LLM call drives the `answer`.
async fn graph_aware_query(
    state: &AppState,
    body: &QueryRequest,
    mode: QueryMode,
    start: std::time::Instant,
) -> Result<Json<QueryResponse>, ApiError> {
    // Pre-compute vector hits (best-effort; failures don't block the
    // graph path because `answer` is the primary signal here).
    let vector_results: Vec<QueryResult> = {
        #[cfg(feature = "qdrant")]
        if let Some(qdrant) = &state.qdrant {
            match state.embeddings.generate_single(&body.query).await {
                Ok(embedding) => match qdrant.search(embedding, body.top_k, None).await {
                    Ok(results) => results
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
                        .collect(),
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        }
        #[cfg(not(feature = "qdrant"))]
        {
            Vec::new()
        }
    };

    let mut graphrag_guard = state.graphrag.write().await;
    let graphrag = graphrag_guard.as_mut().ok_or_else(|| {
        ApiError::BadRequest(
            "Mode requires a configured chat backend. POST /config with \
             openai.enabled=true or ollama.enabled=true first."
                .to_string(),
        )
    })?;

    match mode {
        QueryMode::Ask => {
            let answer = graphrag.ask(&body.query).await.map_err(|e| {
                tracing::error!(error = %e, "ask() failed");
                ApiError::InternalError(format!("ask() failed: {}", e))
            })?;
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(answer),
                confidence: None,
                key_entities: None,
                reasoning_steps: None,
                sources: None,
                processing_time_ms: processing_time,
                backend: "graphrag".to_string(),
            }))
        },
        QueryMode::Explain => {
            let explained = graphrag.ask_explained(&body.query).await.map_err(|e| {
                tracing::error!(error = %e, "ask_explained() failed");
                ApiError::InternalError(format!("ask_explained() failed: {}", e))
            })?;
            let sources: Vec<SourceReferenceDto> = explained
                .sources
                .iter()
                .map(|s| SourceReferenceDto {
                    id: s.id.clone(),
                    kind: match s.source_type {
                        graphrag_core::retrieval::SourceType::TextChunk => SourceKind::TextChunk,
                        graphrag_core::retrieval::SourceType::Entity => SourceKind::Entity,
                        graphrag_core::retrieval::SourceType::Relationship => {
                            SourceKind::Relationship
                        },
                        graphrag_core::retrieval::SourceType::Summary => SourceKind::Summary,
                    },
                    excerpt: s.excerpt.clone(),
                    relevance: s.relevance_score,
                })
                .collect();
            let reasoning_steps: Vec<ReasoningStepDto> = explained
                .reasoning_steps
                .iter()
                .map(|s| ReasoningStepDto {
                    step: s.step_number,
                    description: s.description.clone(),
                    entities_used: s.entities_used.clone(),
                    evidence: s.evidence_snippet.clone(),
                    confidence: s.confidence,
                })
                .collect();
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(explained.answer.clone()),
                confidence: Some(explained.confidence),
                key_entities: Some(explained.key_entities.clone()),
                reasoning_steps: Some(reasoning_steps),
                sources: Some(sources),
                processing_time_ms: processing_time,
                backend: "graphrag".to_string(),
            }))
        },
        QueryMode::Reason => {
            let answer = graphrag
                .ask_with_reasoning(&body.query)
                .await
                .map_err(|e| {
                    tracing::error!(error = %e, "ask_with_reasoning() failed");
                    ApiError::InternalError(format!("ask_with_reasoning() failed: {}", e))
                })?;
            let processing_time = start.elapsed().as_millis() as u64;
            Ok(Json(QueryResponse {
                query: body.query.clone(),
                mode: mode.as_str().to_string(),
                results: vector_results,
                answer: Some(answer),
                confidence: None,
                key_entities: None,
                reasoning_steps: None,
                sources: None,
                processing_time_ms: processing_time,
                backend: "graphrag".to_string(),
            }))
        },
        QueryMode::Search => unreachable!("search dispatched outside graph_aware_query"),
    }
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
    let user_id = body.id.clone();

    let id = uuid::Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();
    // SHA-256 of the sanitized content; drives ingest-time dedup so the
    // same source ingested twice doesn't end up as two Qdrant points
    // (which is what was producing the duplicate query results).
    let content_hash = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(content.as_bytes());
        format!("{:x}", h.finalize())
    };

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Dedup check first — if a point with the same content_hash
        // already exists, return its id without re-embedding. Save NPU
        // time and avoid duplicate vectors in the index.
        if let Ok(Some((existing_id, existing_md))) =
            qdrant.find_by_content_hash(&content_hash).await
        {
            tracing::info!(
                "Skipping ingest: content_hash matches existing doc '{}' ({})",
                existing_md.title,
                existing_id
            );
            return Ok(Json(DocumentOperationResponse {
                success: true,
                document_id: Some(existing_id),
                message: "Document already indexed (content_hash match)".to_string(),
                backend: "qdrant".to_string(),
            }));
        }

        // Generate real embeddings
        let embedding = match state.embeddings.generate_single(&content).await {
            Ok(emb) => emb,
            Err(e) => {
                tracing::error!("Failed to generate document embedding: {}", e);
                return Err(ApiError::InternalError(format!(
                    "Failed to generate embedding: {}",
                    e
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
            content_hash: Some(content_hash.clone()),
            user_id: user_id.clone(),
            custom: HashMap::new(),
        };

        match qdrant.add_document(&id, embedding, metadata).await {
            Ok(_) => {
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
                    "Failed to add document to Qdrant: {}",
                    e
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
    *state.graph_built.write().await = false;

    tracing::info!("Added document to memory: {} ({})", document.title, id);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(id),
        message: "Document added to memory successfully".to_string(),
        backend: "memory".to_string(),
    }))
}

/// List all documents
#[api_operation(
    tag = "documents",
    summary = "List all documents",
    description = "Retrieve a list of all documents in the knowledge graph"
)]
async fn list_documents(state: Data<AppState>) -> Json<ListDocumentsResponse> {
    // Hard cap on the page size: ingesters can drive the corpus to
    // many thousands of points. 256 is plenty for an agent inspecting
    // what's indexed; deeper enumeration should use search.
    const LIST_LIMIT: u32 = 256;

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Get total count separately — list_documents pages through
        // the collection but a separate /count is one cheap call.
        let total = match qdrant.stats().await {
            Ok((c, _)) => c,
            Err(e) => {
                tracing::warn!("Qdrant stats failed: {}", e);
                0
            },
        };
        match qdrant.list_documents(LIST_LIMIT).await {
            Ok(rows) => {
                let documents: Vec<DocumentSummary> = rows
                    .into_iter()
                    .map(|r| DocumentSummary {
                        id: r.id,
                        user_id: r.user_id,
                        title: r.title,
                        content_length: None,
                        excerpt: Some(r.excerpt),
                        added_at: r.timestamp,
                    })
                    .collect();
                let truncated = (documents.len() as u32) >= LIST_LIMIT && total > documents.len();
                return Json(ListDocumentsResponse {
                    documents,
                    total,
                    backend: "qdrant".to_string(),
                    note: if truncated {
                        Some(format!(
                            "Showing first {} of {} documents — use search to drill in",
                            LIST_LIMIT, total
                        ))
                    } else {
                        None
                    },
                });
            },
            Err(e) => {
                tracing::error!("Qdrant list_documents failed: {}", e);
            },
        }
    }

    // Fallback: in-memory storage
    let documents = state.documents.read().await;

    let doc_list: Vec<DocumentSummary> = documents
        .iter()
        .map(|doc| DocumentSummary {
            id: doc.id.clone(),
            user_id: None,
            title: doc.title.clone(),
            content_length: Some(doc.content.len()),
            excerpt: None,
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
    description = "Remove a document from the knowledge graph by ID",
    error_code = 404,
    error_code = 500
)]
async fn delete_document(
    state: Data<AppState>,
    id: WebPath<String>,
) -> Result<Json<DocumentOperationResponse>, ApiError> {
    let supplied = id.into_inner();

    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // Two-step lookup: try the supplied id as a user-supplied id
        // first (the kind callers actually remember), fall back to
        // treating it as the Qdrant point UUID. This is the fix for
        // "delete by user-id returns 500" — the Qdrant point id is a
        // UUID assigned at ingest, but callers think in terms of the
        // id they handed us.
        let resolved = match qdrant.find_id_by_user_id(&supplied).await {
            Ok(Some(uuid)) => uuid,
            _ => supplied.clone(),
        };

        match qdrant.delete_document(&resolved).await {
            Ok(_) => {
                tracing::info!(
                    "Deleted document from Qdrant: supplied={} resolved={}",
                    supplied,
                    resolved
                );
                return Ok(Json(DocumentOperationResponse {
                    success: true,
                    document_id: Some(resolved.clone()),
                    message: format!("Document {} deleted from Qdrant", resolved),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to delete from Qdrant: {}",
                    e
                )));
            },
        }
    }

    // Fallback: in-memory storage
    let mut documents = state.documents.write().await;
    let original_len = documents.len();
    documents.retain(|doc| doc.id != supplied);

    if documents.len() == original_len {
        return Err(ApiError::NotFound(format!(
            "Document with id '{}' not found",
            supplied
        )));
    }

    *state.graph_built.write().await = false;
    tracing::info!("Deleted document from memory: {}", supplied);

    Ok(Json(DocumentOperationResponse {
        success: true,
        document_id: Some(supplied.clone()),
        message: format!("Document {} deleted from memory", supplied),
        backend: "memory".to_string(),
    }))
}

/// Build the knowledge graph (full re-extraction; deprecated for routine use)
///
/// **Deprecated for routine use.** The server persists the entity graph to
/// Qdrant on every successful build/append and rehydrates it on startup,
/// so a full rebuild is no longer needed across restarts. The 30-minute
/// `/api/graph/append` cron handles new ingests. Reserve this endpoint
/// for explicit user requests or recovery after a config change
/// (entity_types, prompts, chat model swap). The endpoint stays mounted
/// for those cases — it isn't going away — but agents should prefer
/// `/api/graph/append` for everything routine.
#[api_operation(
    tag = "graph",
    summary = "Build the knowledge graph (DEPRECATED for routine use — prefer /api/graph/append)",
    description = "Full LLM re-extraction over the entire corpus. DEPRECATED for routine use — the entity graph now persists to Qdrant and rehydrates on startup, so manual rebuilds are not needed in normal operation. Use /api/graph/append (which the cron timer also calls) for incremental updates. Reserve this endpoint for explicit user-requested rebuilds or recovery after a config change.",
    deprecated = true,
    error_code = 400,
    error_code = 500
)]
async fn build_graph(state: Data<AppState>) -> Result<Json<BuildGraphResponse>, ApiError> {
    let start = std::time::Instant::now();

    // Try the real GraphRAG pipeline first
    {
        let mut graphrag_guard = state.graphrag.write().await;
        if let Some(ref mut graphrag) = *graphrag_guard {
            // Use actual pipeline to build graph
            match graphrag.build_graph().await {
                Ok(_) => {
                    let (entities, relationships) = graphrag
                        .knowledge_graph()
                        .map(|kg| (kg.entities().count(), kg.relationships().count()))
                        .unwrap_or((0, 0));

                    // Persist the freshly-built graph to Qdrant so it
                    // survives a server restart. Best-effort: a Qdrant
                    // failure logs and continues — the in-memory graph
                    // is still usable for the rest of the session.
                    #[cfg(feature = "qdrant")]
                    if let Some(qdrant) = state.qdrant.as_ref() {
                        match graph_persistence::persist_in_memory_graph(graphrag, qdrant, state.embeddings.as_ref()).await
                        {
                            Ok((e, r)) => tracing::info!(
                                "💾 Persisted graph to Qdrant: {} entities, {} relationships",
                                e, r
                            ),
                            Err(err) => tracing::warn!(
                                error = %err,
                                "graph persistence failed; in-memory build is still good but won't survive restart"
                            ),
                        }
                    }

                    let processing_time = start.elapsed().as_millis() as u64;

                    *state.graph_built.write().await = true;
                    *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());

                    tracing::info!(
                        "Built knowledge graph via pipeline in {}ms ({} entities, {} relationships)",
                        processing_time, entities, relationships
                    );

                    return Ok(Json(BuildGraphResponse {
                        success: true,
                        document_count: state.documents.read().await.len(),
                        processing_time_ms: processing_time,
                        message: format!(
                            "Knowledge graph built: {} entities, {} relationships",
                            entities, relationships
                        ),
                        backend: "graphrag-pipeline".to_string(),
                    }));
                },
                Err(e) => {
                    tracing::warn!("GraphRAG pipeline build failed, trying fallback: {}", e);
                    // Fall through to lower-priority backends
                },
            }
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

                let processing_time = start.elapsed().as_millis() as u64;

                tracing::info!(
                    "Built knowledge graph from {} Qdrant documents in {}ms",
                    count,
                    processing_time
                );

                *state.graph_built.write().await = true;
                *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());

                return Ok(Json(BuildGraphResponse {
                    success: true,
                    document_count: count,
                    processing_time_ms: processing_time,
                    message: "Knowledge graph built from Qdrant successfully".to_string(),
                    backend: "qdrant".to_string(),
                }));
            },
            Err(e) => {
                return Err(ApiError::InternalError(format!(
                    "Failed to access Qdrant: {}",
                    e
                )));
            },
        }
    }

    // Fallback: in-memory storage
    let doc_count = state.documents.read().await.len();

    if doc_count == 0 {
        return Err(ApiError::BadRequest(
            "No documents to build graph from. Add documents first.".to_string(),
        ));
    }

    *state.graph_built.write().await = true;
    *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());
    let processing_time = start.elapsed().as_millis() as u64;

    tracing::info!(
        "Built knowledge graph from {} memory documents in {}ms",
        doc_count,
        processing_time
    );

    Ok(Json(BuildGraphResponse {
        success: true,
        document_count: doc_count,
        processing_time_ms: processing_time,
        message: "Knowledge graph built from memory successfully".to_string(),
        backend: "memory".to_string(),
    }))
}

/// Append-extract entities for chunks ingested since the last build.
///
/// Semantic-equivalent of Microsoft GraphRAG's `graphrag append`: run
/// after a batch of `/api/documents` calls so newly-ingested content
/// shows up in queries, without paying for a wholesale re-extraction
/// of everything that was already indexed.
///
/// Internally calls `GraphRAG::extend_graph` — a real incremental
/// pass that only walks the chunks ingested since the last build /
/// extend, dedupes entities by id (mentions of an existing entity
/// extend its `mentions` in place rather than creating a duplicate
/// node), and merges relationships keyed by (source, target,
/// relation_type). Cost scales with the size of the delta, not with
/// the total corpus.
///
/// Fast-paths: returns `{success: true, document_count: 0,
/// message: "no new chunks since last build"}` immediately when the
/// chunk count hasn't grown since the previous build/append. Cheap
/// for cron-driven callers that fire periodically regardless of
/// whether anything new was ingested.
#[api_operation(
    tag = "graph",
    summary = "Append new chunks to the knowledge graph",
    description = "Run entity extraction on chunks ingested since the last build. Walks only the delta (no full rebuild), dedupes entities by id, merges relationships. Cheap no-op when nothing new. Use after a batch of /api/documents calls; do NOT call once per document.",
    error_code = 500
)]
async fn append_graph(state: Data<AppState>) -> Result<Json<BuildGraphResponse>, ApiError> {
    let start = std::time::Instant::now();

    let mut graphrag_guard = state.graphrag.write().await;
    let Some(graphrag) = graphrag_guard.as_mut() else {
        return Err(ApiError::BadRequest(
            "GraphRAG not initialized. Call POST /config first.".to_string(),
        ));
    };

    match graphrag.extend_graph().await {
        Ok(summary) => {
            // No-op fast path: nothing was ingested since last build.
            // Cron-callers fire this regardless of whether anything's
            // changed; surface that as a clear message rather than a
            // misleading "appended 0 chunks". Skip persistence — the
            // graph is unchanged.
            if summary.chunks_processed == 0 {
                let processing_time = start.elapsed().as_millis() as u64;
                return Ok(Json(BuildGraphResponse {
                    success: true,
                    document_count: 0,
                    processing_time_ms: processing_time,
                    message: format!(
                        "No new chunks since last build ({} processed). Nothing to append.",
                        graphrag.processed_chunk_count()
                    ),
                    backend: "graphrag-pipeline".to_string(),
                }));
            }

            // Persist the extended graph to Qdrant. Best-effort.
            #[cfg(feature = "qdrant")]
            if let Some(qdrant) = state.qdrant.as_ref() {
                match graph_persistence::persist_in_memory_graph(graphrag, qdrant, state.embeddings.as_ref()).await {
                    Ok((e, r)) => tracing::info!(
                        "💾 Persisted graph to Qdrant: {} entities, {} relationships",
                        e, r
                    ),
                    Err(err) => tracing::warn!(
                        error = %err,
                        "graph persistence failed; in-memory append is still good but won't survive restart"
                    ),
                }
            }

            let processing_time = start.elapsed().as_millis() as u64;

            *state.graph_built.write().await = true;
            *state.last_built_at.write().await = Some(chrono::Utc::now().to_rfc3339());

            tracing::info!(
                "extend_graph: {} delta chunks, +{} entities, +{} rels, {} mentions merged ({}ms; graph: {} entities, {} rels)",
                summary.chunks_processed,
                summary.new_entities,
                summary.new_relationships,
                summary.mentions_merged,
                processing_time,
                summary.total_entities,
                summary.total_relationships,
            );

            Ok(Json(BuildGraphResponse {
                success: true,
                document_count: summary.chunks_processed,
                processing_time_ms: processing_time,
                message: format!(
                    "Appended {} new chunks: +{} entities, +{} relationships, {} mentions merged ({} entities, {} relationships total)",
                    summary.chunks_processed,
                    summary.new_entities,
                    summary.new_relationships,
                    summary.mentions_merged,
                    summary.total_entities,
                    summary.total_relationships,
                ),
                backend: "graphrag-pipeline".to_string(),
            }))
        },
        Err(e) => Err(ApiError::InternalError(format!("Append failed: {}", e))),
    }
}

/// Get graph statistics
#[api_operation(
    tag = "graph",
    summary = "Get graph statistics",
    description = "Retrieve statistics about the knowledge graph, including document count, entity count, and relationship count"
)]
async fn graph_stats(state: Data<AppState>) -> Json<GraphStatsResponse> {
    // Read last_built_at once; same value for every branch below.
    let last_built_at = state.last_built_at.read().await.clone();

    // Try real GraphRAG pipeline stats first
    {
        let graphrag_guard = state.graphrag.read().await;
        if let Some(ref graphrag) = *graphrag_guard {
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
                    graph_built: true,
                    last_built_at,
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
                    last_built_at,
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
    let graph_built = *state.graph_built.read().await;

    Json(GraphStatsResponse {
        document_count: doc_count,
        entity_count: 0,
        relationship_count: 0,
        vector_count: 0,
        graph_built,
        last_built_at,
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
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    // Create application state (connects to Qdrant if available)
    let state = AppState::new().await;
    let state_data = Data::new(state.clone());

    // Configure OpenAPI specification
    let spec = Spec {
        info: Info {
            title: "GraphRAG REST API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: Some(concat!(
                "Production-ready REST API for GraphRAG operations with Qdrant vector database.\n\n",
                "## Features\n",
                "- Semantic search over documents\n",
                "- Knowledge graph construction\n",
                "- Real-time vector embeddings\n",
                "- Qdrant integration (optional)\n",
                "- JWT authentication (optional)\n\n",
                "## Getting Started\n",
                "1. Add documents via `POST /api/documents`\n",
                "2. Build graph via `POST /api/graph/build`\n",
                "3. Query via `POST /api/query`\n"
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
    tracing::info!("📡 Listening on http://0.0.0.0:8080");
    tracing::info!("📚 Swagger UI: http://0.0.0.0:8080/swagger");
    tracing::info!("📄 OpenAPI spec: http://0.0.0.0:8080/openapi.json");
    tracing::info!(
        "🗄️  Backend: {}",
        if state.has_qdrant() {
            "Qdrant"
        } else {
            "In-memory"
        }
    );

    HttpServer::new(move || {
        // Configure CORS for each app instance
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

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
            .app_data(web::JsonConfig::default().limit(validation::MAX_BODY_SIZE))

            // Public routes
            .service(resource("/").route(get().to(root)))
            .service(resource("/health").route(get().to(health)))

            // API routes
            .service(
                scope("/api")
                    // Documents endpoints
                    .service(
                        scope("/documents")
                            .service(resource("").route(get().to(list_documents)))
                            .service(resource("").route(post().to(add_document)))
                            .service(resource("/{id}").route(delete().to(delete_document)))
                    )
                    // Query endpoints
                    .service(
                        scope("/query")
                            .service(resource("").route(post().to(query)))
                    )
                    // Graph endpoints
                    .service(
                        scope("/graph")
                            .service(resource("/build").route(post().to(build_graph)))
                            .service(resource("/append").route(post().to(append_graph)))
                            .service(resource("/stats").route(get().to(graph_stats)))
                    )
            )

            // Auth routes (temporarily disabled - feature "auth" is disabled)
            // #[cfg(feature = "auth")]
            // .service(
            //     scope("/auth")
            //         .service(resource("/login").route(post().to(login)))
            //         .service(resource("/api-key").route(post().to(create_api_key)))
            // )

            // Config endpoints (item 3.3): registered via plain Actix-web routes.
            // NOTE: To include them in OpenAPI spec, add #[api_operation] macros to
            //       each handler in config_endpoints.rs, then register via Apistos scope/resource.

            // Build OpenAPI spec endpoint
            .build("/openapi.json")

            // Config endpoints (plain Actix-web routing — no #[api_operation] yet)
            .service(
                web::scope("/api/config")
                    .route("", web::get().to(config_endpoints::get_config))
                    .route("", web::post().to(config_endpoints::set_config))
                    .route("/template", web::get().to(config_endpoints::get_config_template))
                    .route("/default", web::get().to(config_endpoints::get_default_config))
                    .route("/validate", web::post().to(config_endpoints::validate_config))
            )
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
