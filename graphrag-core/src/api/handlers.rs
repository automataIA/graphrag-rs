//! REST API handlers for GraphRAG server
//!
//! Extracted from bin/graphrag_server for testability

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use crate::{GraphRAG, GraphRAGError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub graphrag: Arc<RwLock<GraphRAG>>,
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

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default)]
    pub options: QueryOptions,
}

#[derive(Deserialize, Default)]
pub struct QueryOptions {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default)]
    pub include_sources: bool,
    #[serde(default)]
    pub include_confidence: bool,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub answer: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub metadata: ResponseMetadata,
}

#[derive(Serialize)]
pub struct ResponseMetadata {
    pub query_time_ms: u64,
    pub tokens_used: usize,
}

pub async fn handle_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, AppError> {
    let start = std::time::Instant::now();

    let graphrag = state.graphrag.read().await;
    let results = graphrag.query(&req.query)
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

#[derive(Deserialize)]
pub struct DocumentRequest {
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

pub async fn add_document(
    State(state): State<AppState>,
    Json(req): Json<DocumentRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let start = std::time::Instant::now();

    let mut graphrag = state.graphrag.write().await;
    graphrag.add_document_from_text(&req.content)
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

pub async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.get_knowledge_graph() {
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

pub async fn graph_stats(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.get_knowledge_graph() {
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

pub async fn export_graph(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.get_knowledge_graph() {
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

#[derive(Deserialize)]
pub struct ListEntitiesQuery {
    #[serde(default = "default_page")]
    pub page: usize,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default)]
    pub entity_type: Option<String>,
}

fn default_page() -> usize { 1 }
fn default_page_size() -> usize { 20 }

pub async fn list_entities(
    State(state): State<AppState>,
    Query(params): Query<ListEntitiesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graphrag = state.graphrag.read().await;

    if let Some(graph) = graphrag.get_knowledge_graph() {
        let mut entities: Vec<serde_json::Value> = graph
            .entities()
            .filter(|entity| {
                params.entity_type.as_ref()
                    .map_or(true, |t| entity.entity_type == *t)
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
        entities = entities.into_iter().skip(start).take(params.page_size).collect();

        Ok(Json(serde_json::json!({
            "entities": entities,
            "page": params.page,
            "page_size": params.page_size,
            "total": total,
            "total_pages": (total + params.page_size - 1) / params.page_size
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

    if let Some(graph) = graphrag.get_knowledge_graph() {
        metrics["graph"] = serde_json::json!({
            "entities": graph.entity_count(),
            "relationships": graph.relationship_count(),
            "documents": graph.document_count()
        });
    }

    Ok(Json(metrics))
}

// === Error Handling ===

#[derive(Debug)]
pub enum AppError {
    GraphRAG(GraphRAGError),
    NotFound(String),
    BadRequest(String),
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
