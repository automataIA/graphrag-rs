//! Configuration endpoints for GraphRAG Server
//!
//! These endpoints allow dynamic configuration of the GraphRAG pipeline via JSON REST API

use super::{config_handler, AppState};
use crate::models::ApiError;
use actix_web::web::{Data, Json};
use apistos::api_operation;
use serde_json::json;

/// GET /api/config - Get current configuration
#[api_operation(skip)]
pub async fn get_config(state: Data<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let active = state.active.read().await;
    let Some(graphrag) = active.graphrag.as_ref() else {
        return Err(ApiError::NotFound(
            "No configuration set. Use POST /api/config to initialize.".to_string(),
        ));
    };
    let config = serde_json::to_value(graphrag.config())
        .map_err(|e| ApiError::InternalError(format!("Failed to serialize config: {e}")))?;

    Ok(Json(json!({
        "success": true,
        "config": config,
        "graphrag_initialized": true,
        "graph_built": active.graph_built
    })))
}

/// POST /api/config - Set configuration and initialize GraphRAG
#[api_operation(skip)]
pub async fn set_config(
    state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!("Received configuration request");

    // Parse the configuration from JSON
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {e}")))?;

    // Parse and validate without changing the active state.  Initialization is
    // also performed before taking the write lock, so a failed candidate cannot
    // partially replace the currently serving pipeline.
    let config = graphrag_core::config::loader::load_config_from_str(
        &config_json,
        graphrag_core::config::loader::ConfigFormat::Json,
    )
    .map_err(|error| ApiError::BadRequest(error.to_string()))?;

    // Initialize GraphRAG with the config
    tracing::info!("Initializing GraphRAG with custom configuration...");

    let mut graphrag = graphrag_core::GraphRAG::new(config)
        .map_err(|e| ApiError::InternalError(format!("GraphRAG init failed: {e}")))?;

    graphrag
        .initialize()
        .map_err(|e| ApiError::InternalError(format!("GraphRAG initialization failed: {e}")))?;

    let graph_built = graphrag.has_graph();

    // Config and graph metadata are one atomic snapshot for readers.
    *state.active.write().await = super::ActiveState {
        graphrag: Some(graphrag),
        graph_built,
    };

    tracing::info!("✅ GraphRAG initialized successfully with custom configuration");

    Ok(Json(json!({
        "success": true,
        "message": "GraphRAG initialized with custom configuration",
        "configured": true,
        "mode": "full_pipeline",
        "graph_built": graph_built
    })))
}

/// GET /api/config/template - Get configuration template
#[api_operation(skip)]
pub async fn get_config_template() -> Json<config_handler::ConfigTemplateResponse> {
    Json(config_handler::get_config_templates())
}

/// GET /api/config/default - Get default configuration
#[api_operation(skip)]
pub async fn get_default_config() -> Json<serde_json::Value> {
    let default_json = config_handler::ConfigManager::default_config_json();
    let config: serde_json::Value = serde_json::from_str(&default_json).unwrap_or(json!({}));

    Json(json!({
        "config": config,
        "description": "Default GraphRAG configuration with sensible defaults"
    }))
}

/// POST /api/config/validate - Validate configuration without applying
#[api_operation(skip)]
pub async fn validate_config(
    _state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {e}")))?;

    // Parse and run the same dispatcher and semantic checks used by the
    // configuration manager and CLI.
    match graphrag_core::config::loader::load_config_from_str(
        &config_json,
        graphrag_core::config::loader::ConfigFormat::Json,
    ) {
        Ok(config) => {
            let validation = graphrag_core::config::Validatable::validate(&config);
            if validation.is_valid {
                Ok(Json(json!({
                    "valid": true,
                    "message": "Configuration is valid"
                })))
            } else {
                Ok(Json(json!({ "valid": false, "errors": validation.errors })))
            }
        },
        Err(e) => Ok(Json(json!({
            "valid": false,
            "errors": [format!("Parse error: {}", e)]
        }))),
    }
}
