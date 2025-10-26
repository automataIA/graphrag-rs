//! REST API Module for GraphRAG
//!
//! Provides a simple way to add REST API capabilities to any GraphRAG instance

#[cfg(feature = "web-api")]
pub mod server {
    use crate::{GraphRAG, Result, GraphRAGError};
    use axum::{
        extract::{Path, Query, State},
        http::StatusCode,
        response::{IntoResponse, Json},
        routing::{get, post},
        Router,
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// API server configuration
    #[derive(Debug, Clone)]
    pub struct ApiConfig {
        pub host: String,
        pub port: u16,
        pub enable_cors: bool,
        pub enable_metrics: bool,
    }

    impl Default for ApiConfig {
        fn default() -> Self {
            Self {
                host: "0.0.0.0".to_string(),
                port: 8080,
                enable_cors: true,
                enable_metrics: true,
            }
        }
    }

    /// Start the REST API server
    pub async fn start_server(
        graphrag: GraphRAG,
        config: ApiConfig,
    ) -> Result<()> {
        let state = ApiState {
            graphrag: Arc::new(RwLock::new(graphrag)),
        };

        let app = create_router(state, &config);

        let addr = format!("{}:{}", config.host, config.port);
        tracing::info!(addr = %addr, "GraphRAG API listening");

        let listener = tokio::net::TcpListener::bind(&addr).await
            .map_err(|e| GraphRAGError::Network {
                message: format!("Failed to bind to {}: {}", addr, e),
            })?;

        axum::serve(listener, app).await
            .map_err(|e| GraphRAGError::Network {
                message: format!("Server error: {}", e),
            })?;

        Ok(())
    }

    #[derive(Clone)]
    struct ApiState {
        graphrag: Arc<RwLock<GraphRAG>>,
    }

    fn create_router(state: ApiState, config: &ApiConfig) -> Router {
        let mut router = Router::new()
            .route("/health", get(health))
            .route("/query", post(query))
            .route("/stats", get(stats))
            .with_state(state);

        if config.enable_cors {
            use tower_http::cors::CorsLayer;
            router = router.layer(CorsLayer::permissive());
        }

        if config.enable_metrics {
            use tower_http::trace::TraceLayer;
            router = router.layer(TraceLayer::new_for_http());
        }

        router
    }

    async fn health() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now()
        }))
    }

    #[derive(Deserialize)]
    struct QueryRequest {
        query: String,
    }

    #[derive(Serialize)]
    struct QueryResponse {
        results: Vec<String>,
        time_ms: u64,
    }

    async fn query(
        State(state): State<ApiState>,
        Json(req): Json<QueryRequest>,
    ) -> Result<Json<QueryResponse>, StatusCode> {
        let start = std::time::Instant::now();

        let graphrag = state.graphrag.read().await;
        let results = graphrag.query(&req.query)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        Ok(Json(QueryResponse {
            results,
            time_ms: start.elapsed().as_millis() as u64,
        }))
    }

    async fn stats(
        State(_state): State<ApiState>,
    ) -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "nodes": 0,
            "edges": 0,
            "documents": 0,
        }))
    }
}

/// Client for interacting with GraphRAG REST API
#[cfg(feature = "web-api")]
pub mod client {
    use crate::{Result, GraphRAGError};
    use serde::{Deserialize, Serialize};

    pub struct ApiClient {
        base_url: String,
        client: ureq::Agent,
    }

    impl ApiClient {
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into(),
                client: ureq::Agent::new(),
            }
        }

        pub fn query(&self, query: &str) -> Result<Vec<String>> {
            #[derive(Serialize)]
            struct Request {
                query: String,
            }

            #[derive(Deserialize)]
            struct Response {
                results: Vec<String>,
            }

            let response: Response = self.client
                .post(&format!("{}/query", self.base_url))
                .send_json(Request {
                    query: query.to_string(),
                })
                .map_err(|e| GraphRAGError::Network {
                    message: format!("API request failed: {}", e),
                })?
                .into_json()
                .map_err(|e| GraphRAGError::Json(e.into()))?;

            Ok(response.results)
        }

        pub fn health_check(&self) -> Result<bool> {
            let response = self.client
                .get(&format!("{}/health", self.base_url))
                .call()
                .map_err(|e| GraphRAGError::Network {
                    message: format!("Health check failed: {}", e),
                })?;

            Ok(response.status() == 200)
        }
    }
}