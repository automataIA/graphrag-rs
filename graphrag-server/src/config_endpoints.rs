//! Configuration endpoints for GraphRAG Server
//!
//! These endpoints allow dynamic configuration of the GraphRAG pipeline via JSON REST API

use super::{config_handler, AppState};
use crate::embeddings::EmbeddingService;
use crate::models::ApiError;
use actix_web::web::{Data, Json};
use serde_json::json;
use std::sync::Arc;

/// GET /api/config - Get current configuration
pub async fn get_config(state: Data<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    if !state.config_manager.is_configured().await {
        return Err(ApiError::NotFound(
            "No configuration set. Use POST /api/config to initialize.".to_string(),
        ));
    }

    match state.config_manager.to_json().await {
        Ok(config_json) => {
            let config: serde_json::Value = serde_json::from_str(&config_json)
                .map_err(|e| ApiError::InternalError(e.to_string()))?;

            Ok(Json(json!({
                "success": true,
                "config": config,
                "graphrag_initialized": state.graphrag.read().await.is_some()
            })))
        },
        Err(e) => Err(ApiError::InternalError(e)),
    }
}

/// POST /api/config - Set configuration and initialize GraphRAG.
///
/// Atomically rebuilds the embedding subsystem to match the new config:
///
/// 1. Merge the posted patch into the active config (deep merge).
/// 2. Build a fresh `EmbeddingService` from the merged
///    `config.embeddings`. Probe-embed a known string and reject the
///    POST with HTTP 400 if the returned vector length doesn't equal
///    `config.embeddings.dimension`. This is the single chokepoint that
///    prevents silent Qdrant corruption — without it, a config that
///    claims dim=1024 but talks to a 768-D upstream only fails at
///    insert time, after we've already accepted documents.
/// 3. Hot-swap `state.embeddings` (lock-free via `ArcSwap`) and update
///    `state.config` so `/health`, `/config`, and `/embeddings/stats`
///    immediately reflect the new struct.
/// 4. Build a fresh `GraphRAG` instance with the new config and inject
///    the new embedder before `initialize()`.
pub async fn set_config(
    state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!("Received configuration request");

    // Parse the configuration from JSON
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {}", e)))?;

    // Set configuration via ConfigManager (handles deep-merge + validation)
    state
        .config_manager
        .set_from_json(&config_json)
        .await
        .map_err(ApiError::BadRequest)?;

    // Get the validated config
    let config = state
        .config_manager
        .get_config()
        .await
        .ok_or(ApiError::InternalError("Failed to get config".to_string()))?;

    // Build the new EmbeddingService from the merged embeddings block.
    // Failure here is a 400 — the caller's config is bad (unreachable
    // upstream, missing fields), not an internal server error.
    let new_embeddings = EmbeddingService::from_config(&config.embeddings)
        .await
        .map_err(|e| {
            ApiError::BadRequest(format!(
                "Embedding service rebuild failed for backend={}: {}",
                config.embeddings.backend, e
            ))
        })?;

    // Probe-embed to validate dimension end-to-end. Fails HERE rather
    // than at the next /api/documents POST, which would silently insert
    // wrong-dim vectors into a Qdrant collection sized to the configured
    // dimension and corrupt the index. Skip when backend=hash because
    // the hash generator is sized to `config.dimension` by construction.
    //
    // Also catches the upstream-misconfigured case: the EmbeddingService
    // silently falls through to hash when the real backend returns a
    // wrong-dim vector (so `generate_single` would return a hash vector
    // of the correct size and the dim check alone would pass). We
    // detect that by also asserting `backend_live()` — if the configured
    // backend isn't live but the user asked for it, that's also a 400.
    if config.embeddings.backend != "hash" {
        if !new_embeddings.backend_live() {
            return Err(ApiError::BadRequest(format!(
                "Embedding backend '{}' is not reachable at endpoint '{}'. \
                 Check the upstream server, or set embeddings.backend = \"hash\".",
                config.embeddings.backend,
                config.embeddings.api_endpoint.as_deref().unwrap_or("(none)")
            )));
        }

        let stats_before = new_embeddings.get_stats().await;
        let probe = new_embeddings.generate_single("graphrag dimension probe").await;
        let stats_after = new_embeddings.get_stats().await;

        match probe {
            Ok(v) if v.len() != config.embeddings.dimension => {
                return Err(ApiError::BadRequest(format!(
                    "Embedding dimension mismatch: config.embeddings.dimension={} but \
                     backend={} returned {}-D vectors. Update config.embeddings.dimension \
                     or change the model.",
                    config.embeddings.dimension,
                    config.embeddings.backend,
                    v.len()
                )));
            },
            Ok(_) if stats_after.fallback_used > stats_before.fallback_used => {
                // Real backend errored mid-call (e.g. dim-mismatch caught
                // inside generate_with_openai), fallback covered it. The
                // returned vector is hash-sized = config.dimension, so
                // the length check passed silently. Reject the POST.
                return Err(ApiError::BadRequest(format!(
                    "Embedding backend '{}' probe failed (fell through to hash fallback). \
                     Check server logs for the upstream error, or set embeddings.backend = \"hash\".",
                    config.embeddings.backend
                )));
            },
            Ok(_) => {},
            Err(e) => {
                return Err(ApiError::BadRequest(format!(
                    "Embedding probe failed during /config validation: {e}"
                )));
            },
        }
    }

    // Atomically swap the live embedder. Existing in-flight requests
    // that already grabbed an `Arc<EmbeddingService>` snapshot will
    // continue using the old one; new calls see the new service.
    let new_embeddings = Arc::new(new_embeddings);
    state.embeddings.store(new_embeddings.clone());

    // Update the live config snapshot. AFTER the embedder swap so any
    // reader that wins the race sees old-config + old-embedder or
    // new-config + new-embedder, never new-config + old-embedder.
    *state.config.write().await = config.clone();

    // Re-print the unified backend log line so users see the swap.
    crate::log_unified_embedding_line(&config.embeddings, new_embeddings.backend_live());

    // Initialize GraphRAG with the config
    tracing::info!("Initializing GraphRAG with custom configuration...");

    let mut graphrag = graphrag_core::GraphRAG::new(config)
        .map_err(|e| ApiError::InternalError(format!("GraphRAG init failed: {}", e)))?;

    // Inject the freshly-built embedding service into graphrag-core
    // BEFORE initialize() runs. Single source of truth: graphrag-core's
    // retrieval system, semantic chunker, and entity-vector path all
    // route through this same service.
    graphrag.set_embedding_provider(new_embeddings.clone());

    graphrag
        .initialize()
        .map_err(|e| ApiError::InternalError(format!("GraphRAG initialization failed: {}", e)))?;

    // Hydrate from Qdrant: every document already in the persistent store
    // gets re-chunked and pushed into graphrag-core's in-memory
    // KnowledgeGraph, then their chunk ids are seeded into
    // `processed_chunks`. Without this, after a server restart the
    // in-memory chunk index is empty, /api/graph/build only sees chunks
    // added since restart (a tiny fraction of the corpus), and
    // /api/graph/append's no-op fast-path lies about how much has
    // actually been processed. With it: graph_stats matches Qdrant
    // truth, build_graph covers the full corpus, append_graph only
    // re-extracts genuinely-new chunks.
    //
    // We also restore the previously-extracted entity + relationship
    // graph from the entities/relationships sidecar collections (Phase H).
    // Without that, every restart wipes the LLM-extracted graph and
    // forces re-extraction; with it, build_graph/extend_graph state
    // genuinely survives restarts.
    let mut hydration_summary = json!({
        "documents": 0,
        "chunks": 0,
        "skipped": 0,
        "entities": 0,
        "relationships": 0,
        "relationships_skipped_orphan": 0,
    });
    #[cfg(feature = "qdrant")]
    if let Some(qdrant) = &state.qdrant {
        // 1_000_000 is an arbitrary "drain everything" cap — Qdrant's
        // scroll naturally short-circuits when next_page_offset is None.
        match qdrant.list_full_documents(1_000_000).await {
            Ok(docs) => {
                let mut hydrated_docs = 0usize;
                let mut skipped = 0usize;
                let chunks_before = graphrag
                    .knowledge_graph()
                    .map(|kg| kg.chunks().count())
                    .unwrap_or(0);
                for (_id, md) in &docs {
                    if md.text.is_empty() {
                        skipped += 1;
                        continue;
                    }
                    if let Err(e) = graphrag.add_document_from_text(&md.text) {
                        tracing::warn!(
                            error = %e,
                            title = %md.title,
                            "hydration: add_document_from_text failed; skipping"
                        );
                        skipped += 1;
                        continue;
                    }
                    hydrated_docs += 1;
                }
                let chunks_after = graphrag
                    .knowledge_graph()
                    .map(|kg| kg.chunks().count())
                    .unwrap_or(chunks_before);
                let hydrated_chunks = chunks_after.saturating_sub(chunks_before);

                // Mark every chunk we just rebuilt as "already extracted"
                // so /api/graph/append won't re-run LLM extraction over
                // the entire restored corpus on its first tick. New
                // chunks added via /api/documents after this seeding
                // are NOT in the set yet, so they remain in the
                // append-graph delta as expected.
                let chunk_ids: Vec<_> = graphrag
                    .knowledge_graph()
                    .map(|kg| kg.chunks().map(|c| c.id.clone()).collect())
                    .unwrap_or_default();
                graphrag.seed_processed_chunks(chunk_ids);

                tracing::info!(
                    "🔄 Hydrated KnowledgeGraph from Qdrant: {} documents, {} chunks ({} skipped)",
                    hydrated_docs,
                    hydrated_chunks,
                    skipped
                );

                hydration_summary = json!({
                    "documents": hydrated_docs,
                    "chunks": hydrated_chunks,
                    "skipped": skipped,
                    "entities": 0,
                    "relationships": 0,
                    "relationships_skipped_orphan": 0,
                });
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "hydration: list_full_documents failed; starting with empty in-memory graph"
                );
            },
        }

        // Phase H: restore the LLM-extracted entity + relationship graph
        // from its sidecar collections. Runs after chunk hydration so
        // the KnowledgeGraph already exists and entities have a parent
        // to attach to. Best-effort: a load failure (e.g. missing
        // sidecar collection on a fresh deploy) is normal — we just
        // start with an empty entity graph and the next build_graph
        // populates it.
        match crate::graph_persistence::hydrate_in_memory_graph(&mut graphrag, qdrant).await {
            Ok((entities_restored, rels_restored, rels_skipped)) => {
                if entities_restored + rels_restored > 0 {
                    tracing::info!(
                        "🔄 Restored entity graph from Qdrant: {} entities, {} relationships ({} orphan rels skipped)",
                        entities_restored,
                        rels_restored,
                        rels_skipped,
                    );
                }
                if let Some(obj) = hydration_summary.as_object_mut() {
                    obj.insert("entities".into(), json!(entities_restored));
                    obj.insert("relationships".into(), json!(rels_restored));
                    obj.insert("relationships_skipped_orphan".into(), json!(rels_skipped));
                }
            },
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "graph restore failed; starting with no entities (next build_graph will repopulate)"
                );
            },
        }
    }

    // Store the initialized GraphRAG
    *state.graphrag.write().await = Some(graphrag);

    tracing::info!("✅ GraphRAG initialized successfully with custom configuration");

    Ok(Json(json!({
        "success": true,
        "message": "GraphRAG initialized with custom configuration",
        "configured": true,
        "mode": "full_pipeline",
        "hydrated": hydration_summary,
    })))
}

/// GET /api/config/template - Get configuration template
pub async fn get_config_template() -> Json<config_handler::ConfigTemplateResponse> {
    Json(config_handler::get_config_templates())
}

/// GET /api/config/default - Get default configuration
pub async fn get_default_config() -> Json<serde_json::Value> {
    let default_json = config_handler::ConfigManager::default_config_json();
    let config: serde_json::Value = serde_json::from_str(&default_json).unwrap_or(json!({}));

    Json(json!({
        "config": config,
        "description": "Default GraphRAG configuration with sensible defaults"
    }))
}

/// POST /api/config/validate - Validate configuration without applying
pub async fn validate_config(
    _state: Data<AppState>,
    payload: Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config_json = serde_json::to_string(&payload)
        .map_err(|e| ApiError::BadRequest(format!("Invalid JSON: {}", e)))?;

    // Try to parse as Config
    match serde_json::from_str::<graphrag_core::Config>(&config_json) {
        Ok(_) => Ok(Json(json!({
            "valid": true,
            "message": "Configuration is valid"
        }))),
        Err(e) => Ok(Json(json!({
            "valid": false,
            "errors": [format!("Parse error: {}", e)]
        }))),
    }
}
