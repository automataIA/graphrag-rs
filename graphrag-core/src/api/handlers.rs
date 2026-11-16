//! REST API handlers for GraphRAG server
//!
//! Extracted from bin/graphrag_server for testability

use crate::{GraphRAG, GraphRAGError};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Default GraphRAG instance
    pub graphrag: Arc<RwLock<GraphRAG>>,
    /// Per-session GraphRAG instances keyed by session id
    pub sessions: Arc<RwLock<HashMap<String, Arc<RwLock<GraphRAG>>>>>,
}

/// Health check endpoint
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now()
    }))
}

/// Body of a query request
#[derive(Deserialize)]
pub struct QueryRequest {
    /// Natural-language question
    pub query: String,
    /// Query tuning options
    #[serde(default)]
    pub options: QueryOptions,
}

/// Options controlling query execution
#[derive(Deserialize, Default)]
pub struct QueryOptions {
    /// Maximum number of answers (default 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Reuse cached answers when available
    #[serde(default)]
    pub use_cache: bool,
    /// Include source references in the response
    #[serde(default)]
    pub include_sources: bool,
    /// Include a confidence score in the response
    #[serde(default)]
    pub include_confidence: bool,
}

fn default_limit() -> usize {
    10
}

/// Response returned by the query endpoint
#[derive(Serialize)]
pub struct QueryResponse {
    /// Answer fragments
    pub answer: Vec<String>,
    /// Source references, when requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<String>>,
    /// Confidence score, when requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Timing and token usage
    pub metadata: ResponseMetadata,
}

/// Metadata attached to a query response
#[derive(Serialize)]
pub struct ResponseMetadata {
    /// Query wall time in milliseconds
    pub query_time_ms: u64,
    /// Tokens consumed by the query
    pub tokens_used: usize,
}

/// Answer a query against the shared GraphRAG instance
pub async fn handle_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, AppError> {
    let start = std::time::Instant::now();

    let mut graphrag = state.graphrag.write().await;
    let results = graphrag
        .query_internal(&req.query)
        .await
        .map_err(AppError::GraphRAG)?;

    let query_time = start.elapsed().as_millis() as u64;

    let response = QueryResponse {
        answer: results,
        sources: if req.options.include_sources {
            Some(vec!["doc1".to_string()])
        } else {
            None
        },
        confidence: if req.options.include_confidence {
            Some(0.85)
        } else {
            None
        },
        metadata: ResponseMetadata {
            query_time_ms: query_time,
            tokens_used: 100,
        },
    };

    Ok(Json(response))
}

/// Body of an add-document request
#[derive(Deserialize)]
pub struct DocumentRequest {
    /// Document identifier
    pub id: String,
    /// Document text
    pub content: String,
    /// Arbitrary key/value metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Add a document to the shared GraphRAG instance
pub async fn add_document(
    State(state): State<AppState>,
    Json(req): Json<DocumentRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let start = std::time::Instant::now();

    let mut graphrag = state.graphrag.write().await;
    graphrag
        .add_document_from_text(&req.content)
        .map_err(|e| AppError::Internal(format!("Failed to add document: {}", e)))?;

    let processing_time = start.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "status": "success",
        "document_id": req.id,
        "message": "Document added and processed successfully",
        "processing_time_ms": processing_time,
        "metadata": {
            "content_length": req.content.len(),
            "has_metadata": !req.metadata.is_empty()
        }
    })))
}

/// Fetch a document by id
pub async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.knowledge_graph() {
        let doc_id = crate::core::DocumentId::new(id.clone());
        if let Some(doc) = graph.get_document(&doc_id) {
            return Ok(Json(serde_json::json!({
                "id": doc.id.to_string(),
                "content": doc.content,
                "metadata": doc.metadata
            })));
        }
    }

    Err(AppError::NotFound(format!("Document not found: {}", id)))
}

/// Return entity, relationship and chunk counts
pub async fn graph_stats(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.knowledge_graph() {
        Ok(Json(serde_json::json!({
            "entities": graph.entity_count(),
            "relationships": graph.relationship_count(),
            "documents": graph.document_count(),
            "nodes": graph.entity_count(),
            "edges": graph.relationship_count()
        })))
    } else {
        Ok(Json(serde_json::json!({
            "entities": 0,
            "relationships": 0,
            "documents": 0,
            "nodes": 0,
            "edges": 0,
            "message": "Knowledge graph not initialized"
        })))
    }
}

/// Export the knowledge graph as nodes and edges
pub async fn export_graph(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.knowledge_graph() {
        let nodes: Vec<serde_json::Value> = graph
            .entities()
            .map(|entity| {
                serde_json::json!({
                    "id": entity.id.to_string(),
                    "name": entity.name,
                    "type": entity.entity_type,
                    "confidence": entity.confidence
                })
            })
            .collect();

        let edges: Vec<serde_json::Value> = graph
            .relationships()
            .map(|rel| {
                serde_json::json!({
                    "source": rel.source.to_string(),
                    "target": rel.target.to_string(),
                    "type": rel.relation_type,
                    "confidence": rel.confidence
                })
            })
            .collect();

        Ok(Json(serde_json::json!({
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": nodes.len(),
                "edge_count": edges.len()
            }
        })))
    } else {
        Ok(Json(serde_json::json!({
            "nodes": [],
            "edges": [],
            "message": "Knowledge graph not initialized"
        })))
    }
}

/// Query parameters for listing entities
#[derive(Deserialize)]
pub struct ListEntitiesQuery {
    /// Page number (1-based)
    #[serde(default = "default_page")]
    pub page: usize,
    /// Entities per page
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    /// Only return entities of this type
    #[serde(default)]
    pub entity_type: Option<String>,
}

fn default_page() -> usize {
    1
}
fn default_page_size() -> usize {
    20
}

/// List entities with pagination and optional type filter
pub async fn list_entities(
    State(state): State<AppState>,
    Query(params): Query<ListEntitiesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.knowledge_graph() {
        let mut entities: Vec<serde_json::Value> = graph
            .entities()
            .filter(|entity| {
                params
                    .entity_type
                    .as_ref()
                    .is_none_or(|t| entity.entity_type == *t)
            })
            .map(|entity| {
                serde_json::json!({
                    "id": entity.id.to_string(),
                    "name": entity.name,
                    "type": entity.entity_type,
                    "confidence": entity.confidence
                })
            })
            .collect();

        let total = entities.len();
        let start = (params.page - 1) * params.page_size;
        entities = entities
            .into_iter()
            .skip(start)
            .take(params.page_size)
            .collect();

        Ok(Json(serde_json::json!({
            "entities": entities,
            "page": params.page,
            "page_size": params.page_size,
            "total": total,
            "total_pages": total.div_ceil(params.page_size)
        })))
    } else {
        Ok(Json(serde_json::json!({
            "entities": [],
            "page": params.page,
            "page_size": params.page_size,
            "total": 0,
            "message": "Knowledge graph not initialized"
        })))
    }
}

/// Return session and graph metrics
pub async fn get_metrics(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    let mut metrics = serde_json::json!({
        "sessions": {
            "active": state.sessions.read().await.len(),
            "total_created": state.sessions.read().await.len()
        }
    });

    if let Some(graph) = graphrag.knowledge_graph() {
        metrics["graph"] = serde_json::json!({
            "entities": graph.entity_count(),
            "relationships": graph.relationship_count(),
            "documents": graph.document_count()
        });
    }

    Ok(Json(metrics))
}

// === Error Handling ===

/// Error type returned by the API handlers
#[derive(Debug)]
pub enum AppError {
    /// Error from the GraphRAG core
    GraphRAG(GraphRAGError),
    /// Requested resource does not exist
    NotFound(String),
    /// Request is malformed
    BadRequest(String),
    /// Unexpected server-side failure
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_message) = match self {
            AppError::GraphRAG(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": error_message
        }));

        (status, body).into_response()
    }
}
