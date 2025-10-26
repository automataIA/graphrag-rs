//! HTTP endpoint tests for GraphRAG REST API
//!
//! Tests actual HTTP requests/responses using axum test utilities

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
    Router,
    routing::{get, post},
};
use graphrag_rs::{GraphRAG, Config};
use graphrag_rs::api::handlers::{
    AppState, health_check, handle_query, add_document, get_document,
    graph_stats, export_graph, list_entities, get_metrics,
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

/// Helper to create test app with handlers
fn create_test_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/query", post(handle_query))
        .route("/api/v1/documents", post(add_document))
        .route("/api/v1/documents/{id}", get(get_document))
        .route("/api/v1/graph/stats", get(graph_stats))
        .route("/api/v1/graph/export", get(export_graph))
        .route("/api/v1/entities", get(list_entities))
        .route("/api/v1/admin/metrics", get(get_metrics))
        .with_state(state)
}

/// Helper to create initialized AppState
async fn create_test_state() -> AppState {
    let config = Config::default();
    let mut graphrag = GraphRAG::new(config).expect("Failed to create GraphRAG");
    graphrag.initialize().expect("Failed to initialize");

    AppState {
        graphrag: Arc::new(RwLock::new(graphrag)),
        sessions: Arc::new(RwLock::new(HashMap::new())),
    }
}

#[tokio::test]
async fn test_health_endpoint() {
    let state = create_test_state().await;
    let app = create_test_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "healthy");
    assert!(json["version"].is_string());
    assert!(json["timestamp"].is_string());
}

#[tokio::test]
async fn test_document_upload() {
    let state = create_test_state().await;
    let app = create_test_app(state);

    let payload = json!({
        "id": "test-doc-1",
        "content": "Alice works at OpenAI.",
        "metadata": {}
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/documents")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "success");
    assert_eq!(json["document_id"], "test-doc-1");
    assert!(json["processing_time_ms"].is_number());
}

#[tokio::test]
async fn test_query_endpoint() {
    let state = create_test_state().await;

    // Add a document first
    {
        let mut graphrag = state.graphrag.write().await;
        graphrag.add_document_from_text("Alice is a researcher at MIT.").unwrap();
        graphrag.build_graph().unwrap();
    }

    let app = create_test_app(state);

    let payload = json!({
        "query": "Who is Alice?",
        "options": {}
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/query")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["answer"].is_array());
    assert!(json["metadata"]["query_time_ms"].is_number());
}

#[tokio::test]
async fn test_graph_stats_endpoint() {
    let state = create_test_state().await;

    // Add test data
    {
        let mut graphrag = state.graphrag.write().await;
        graphrag.add_document_from_text("Bob knows Alice.").unwrap();
        graphrag.build_graph().unwrap();
    }

    let app = create_test_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/graph/stats")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["entities"].is_number());
    assert!(json["relationships"].is_number());
    assert!(json["documents"].is_number());
}

#[tokio::test]
async fn test_graph_export_endpoint() {
    let state = create_test_state().await;

    // Add test data
    {
        let mut graphrag = state.graphrag.write().await;
        graphrag.add_document_from_text("Alice works with Bob.").unwrap();
        graphrag.build_graph().unwrap();
    }

    let app = create_test_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/graph/export")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["nodes"].is_array());
    assert!(json["edges"].is_array());
    assert!(json["metadata"].is_object());
}

#[tokio::test]
async fn test_list_entities_endpoint() {
    let state = create_test_state().await;

    // Add test data
    {
        let mut graphrag = state.graphrag.write().await;
        graphrag.add_document_from_text("Alice and Bob work at OpenAI.").unwrap();
        graphrag.build_graph().unwrap();
    }

    let app = create_test_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/entities?page=1&page_size=10")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["entities"].is_array());
    assert_eq!(json["page"], 1);
    assert_eq!(json["page_size"], 10);
    assert!(json["total"].is_number());
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let state = create_test_state().await;

    let app = create_test_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/admin/metrics")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["sessions"].is_object());
}

#[tokio::test]
async fn test_document_not_found() {
    let state = create_test_state().await;
    let app = create_test_app(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/documents/non-existent-doc")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["error"].is_string());
}

#[tokio::test]
async fn test_pagination() {
    let state = create_test_state().await;

    // Add multiple documents
    {
        let mut graphrag = state.graphrag.write().await;
        for i in 0..5 {
            graphrag.add_document_from_text(&format!("Person{} works at Company{}.", i, i)).unwrap();
        }
        graphrag.build_graph().unwrap();
    }

    let app = create_test_app(state);

    // Page 1
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/entities?page=1&page_size=2")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let page1: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(page1["page"], 1);
    assert!(page1["entities"].as_array().unwrap().len() <= 2);

    // Page 2
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/entities?page=2&page_size=2")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let page2: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(page2["page"], 2);
}

#[tokio::test]
async fn test_query_with_options() {
    let state = create_test_state().await;

    {
        let mut graphrag = state.graphrag.write().await;
        graphrag.add_document_from_text("GraphRAG is a knowledge graph system.").unwrap();
        graphrag.build_graph().unwrap();
    }

    let app = create_test_app(state);

    let payload = json!({
        "query": "What is GraphRAG?",
        "options": {
            "include_sources": true,
            "include_confidence": true,
            "limit": 5
        }
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/query")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["sources"].is_array(), "Expected sources array when include_sources=true");
    assert!(json["confidence"].is_number(), "Expected confidence when include_confidence=true");
}
