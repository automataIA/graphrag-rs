//! Embeddings module for GraphRAG Server
//!
//! Single concrete embedder that the whole server uses for both the
//! document and query paths. Driven entirely by
//! [`graphrag_core::config::EmbeddingConfig`] — there is no
//! server-local config struct anymore. The same struct lives in the
//! persisted server config, in `/api/embeddings/stats`, in `/health`,
//! and in graphrag-core's retrieval system, so the four answers can't
//! drift.
//!
//! Backends:
//! - `openai` — any OpenAI-compatible HTTP server (vLLM, OVMS, llama.cpp, etc.)
//! - `ollama` — native Ollama API (parsed out of `api_endpoint`)
//! - `hash`   — deterministic hash-based fallback (no I/O)

use graphrag_core::config::EmbeddingConfig;
use graphrag_core::vector::EmbeddingGenerator;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

#[cfg(feature = "ollama")]
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};

/// OpenAI-compatible embedding HTTP client. Holds the configured URL,
/// model name, and API key; used by `generate_with_openai`.
#[cfg(feature = "openai")]
struct OpenAIClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

/// Embedding service. Constructed once at boot from
/// `Config.embeddings`, then re-constructed atomically by `POST /config`.
pub struct EmbeddingService {
    config: EmbeddingConfig,
    #[cfg(feature = "ollama")]
    ollama_client: Option<Arc<Ollama>>,
    #[cfg(feature = "openai")]
    openai_client: Option<Arc<OpenAIClient>>,
    fallback_generator: Arc<RwLock<EmbeddingGenerator>>,
    stats: Arc<RwLock<EmbeddingStats>>,
}

/// Embedding statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct EmbeddingStats {
    pub total_requests: usize,
    pub backend_success: usize,
    pub backend_failures: usize,
    pub fallback_used: usize,
    pub cache_hits: usize,
}

/// Embedding error type
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Ollama error: {0}")]
    #[allow(dead_code)]
    OllamaError(String),

    #[error("OpenAI-compat error: {0}")]
    #[allow(dead_code)]
    OpenAIError(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Invalid dimension: expected {expected}, got {actual}")]
    #[allow(dead_code)] // Only constructed inside the openai / ollama
    // feature-gated branches.
    DimensionMismatch { expected: usize, actual: usize },
}

impl From<reqwest::Error> for EmbeddingError {
    fn from(e: reqwest::Error) -> Self {
        EmbeddingError::OpenAIError(e.to_string())
    }
}

#[cfg(feature = "ollama")]
impl From<ollama_rs::error::OllamaError> for EmbeddingError {
    fn from(e: ollama_rs::error::OllamaError) -> Self {
        EmbeddingError::OllamaError(e.to_string())
    }
}

impl EmbeddingService {
    /// Build a service from the core `EmbeddingConfig`. This is the
    /// only constructor — boot and `POST /config` both go through here.
    /// On success the caller should log the unified backend line so
    /// users see exactly which path was wired.
    pub async fn from_config(cfg: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        // OpenAI-compat backend (vLLM, OVMS, llama-server, OpenAI itself).
        // Probes /models to confirm reachability; doesn't fail-hard if the
        // server doesn't list our model name (some servers expose synthetic
        // names via Mediapipe graphs / single-model mode).
        #[cfg(feature = "openai")]
        let openai_client = if cfg.backend == "openai" {
            let endpoint = cfg.api_endpoint.as_deref().unwrap_or("");
            let api_key = cfg.api_key.clone().unwrap_or_default();
            let model = cfg.model.clone().unwrap_or_default();

            if endpoint.is_empty() {
                return Err(EmbeddingError::OpenAIError(
                    "embeddings.api_endpoint is required for backend=openai".to_string(),
                ));
            }

            let http = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(|e| EmbeddingError::OpenAIError(format!("client build failed: {e}")))?;

            let probe_url = format!("{}/models", endpoint.trim_end_matches('/'));
            let mut req = http.get(&probe_url);
            if !api_key.is_empty() {
                req = req.bearer_auth(&api_key);
            }
            match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    Some(Arc::new(OpenAIClient {
                        http,
                        base_url: endpoint.to_string(),
                        model,
                        api_key,
                    }))
                },
                Ok(resp) => {
                    warn!(
                        "OpenAI-compat /models returned {} at {}; falling through to hash fallback",
                        resp.status(),
                        endpoint
                    );
                    None
                },
                Err(e) => {
                    warn!(
                        "OpenAI-compat probe failed at {}: {}; falling through to hash fallback",
                        endpoint, e
                    );
                    None
                },
            }
        } else {
            None
        };

        #[cfg(not(feature = "openai"))]
        if cfg.backend == "openai" {
            warn!("backend=openai but openai feature not compiled in — falling through to hash");
        }

        // Ollama backend. `api_endpoint` may be either "host:port",
        // "http://host:port", or unset (defaults to localhost:11434) —
        // `parse_ollama_endpoint` handles all three.
        #[cfg(feature = "ollama")]
        let ollama_client = if cfg.backend == "ollama" {
            let model = cfg.model.clone().unwrap_or_else(|| "nomic-embed-text".to_string());
            let (host, port) = parse_ollama_endpoint(cfg.api_endpoint.as_deref());
            let ollama = Ollama::new(host.clone(), port);

            match ollama.list_local_models().await {
                Ok(models) => {
                    if models.iter().any(|m| m.name == model) {
                        Some(Arc::new(ollama))
                    } else {
                        warn!(
                            "Ollama embedding model '{}' not present at {}:{}; falling through to hash. \
                             Run: ollama pull {}",
                            model, host, port, model
                        );
                        None
                    }
                },
                Err(e) => {
                    warn!(
                        "Ollama unreachable at {}:{}: {}; falling through to hash",
                        host, port, e
                    );
                    None
                },
            }
        } else {
            None
        };

        #[cfg(not(feature = "ollama"))]
        if cfg.backend == "ollama" {
            warn!("backend=ollama but ollama feature not compiled in — falling through to hash");
        }

        // Always available: hash-based fallback. Sized to the configured
        // dimension so the fallback path produces vectors the rest of the
        // pipeline accepts (Qdrant collection dim, retrieval cosine, etc.).
        let fallback_generator = Arc::new(RwLock::new(EmbeddingGenerator::new(cfg.dimension)));

        Ok(Self {
            config: cfg.clone(),
            #[cfg(feature = "ollama")]
            ollama_client,
            #[cfg(feature = "openai")]
            openai_client,
            fallback_generator,
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        })
    }

    /// Generate embeddings for a batch of texts. Tries the configured
    /// backend; falls through to the hash generator if the backend errors.
    pub async fn generate(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut stats = self.stats.write().await;
        stats.total_requests += texts.len();
        drop(stats);

        // OpenAI-compat backend (vLLM, OVMS, llama-server, ...).
        #[cfg(feature = "openai")]
        if let Some(client) = &self.openai_client {
            match self.generate_with_openai(client, texts).await {
                Ok(embeddings) => {
                    let mut stats = self.stats.write().await;
                    stats.backend_success += texts.len();
                    return Ok(embeddings);
                },
                Err(e) => {
                    warn!("OpenAI-compat embedding failed: {}. Using fallback.", e);
                    let mut stats = self.stats.write().await;
                    stats.backend_failures += texts.len();
                },
            }
        }

        // Try Ollama next.
        #[cfg(feature = "ollama")]
        if let Some(ollama) = &self.ollama_client {
            match self.generate_with_ollama(ollama, texts).await {
                Ok(embeddings) => {
                    let mut stats = self.stats.write().await;
                    stats.backend_success += texts.len();
                    return Ok(embeddings);
                },
                Err(e) => {
                    warn!("Ollama embedding failed: {}. Using fallback.", e);
                    let mut stats = self.stats.write().await;
                    stats.backend_failures += texts.len();
                },
            }
        }

        // Fallback to hash-based embeddings
        let mut stats = self.stats.write().await;
        stats.fallback_used += texts.len();
        drop(stats);

        self.generate_with_fallback(texts).await
    }

    /// Generate embeddings using an OpenAI-compatible server (vLLM, OVMS,
    /// llama-server, OpenAI itself, …). One POST per text — keeps the
    /// dimension-validation path simple.
    #[cfg(feature = "openai")]
    async fn generate_with_openai(
        &self,
        client: &OpenAIClient,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!("{}/embeddings", client.base_url.trim_end_matches('/'));
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let body = serde_json::json!({
                "model": client.model,
                "input": text,
            });
            let mut req = client.http.post(&url).json(&body);
            if !client.api_key.is_empty() {
                req = req.bearer_auth(&client.api_key);
            }

            let resp = req.send().await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(EmbeddingError::OpenAIError(format!(
                    "HTTP {status} from {url}: {body}"
                )));
            }

            let parsed: serde_json::Value = resp.json().await?;
            let embedding: Vec<f32> = parsed
                .get("data")
                .and_then(|d| d.get(0))
                .and_then(|d0| d0.get("embedding"))
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    EmbeddingError::GenerationFailed(format!(
                        "OpenAI response missing data[0].embedding: {parsed}"
                    ))
                })?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            if embedding.len() != self.config.dimension {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: embedding.len(),
                });
            }
            results.push(embedding);
        }

        Ok(results)
    }

    /// Generate single embedding
    pub async fn generate_single(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.generate(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::GenerationFailed("No embedding generated".to_string()))
    }

    /// Generate embeddings using Ollama
    #[cfg(feature = "ollama")]
    async fn generate_with_ollama(
        &self,
        ollama: &Ollama,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let model = self
            .config
            .model
            .clone()
            .unwrap_or_else(|| "nomic-embed-text".to_string());
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let request =
                GenerateEmbeddingsRequest::new(model.clone(), text.to_string().into());

            let response = ollama.generate_embeddings(request).await?;

            let embedding = response.embeddings.into_iter().next().ok_or_else(|| {
                EmbeddingError::GenerationFailed("No embedding in response".to_string())
            })?;

            // Validate dimension
            if embedding.len() != self.config.dimension {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: embedding.len(),
                });
            }

            results.push(embedding);
        }

        Ok(results)
    }

    /// Generate embeddings using hash-based fallback
    async fn generate_with_fallback(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut generator = self.fallback_generator.write().await;
        let results = generator.batch_generate(texts);
        Ok(results)
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> EmbeddingStats {
        self.stats.read().await.clone()
    }

    /// Snapshot the config this service was built from. Used by
    /// `/health`, `/config`, and `/embeddings/stats` so they all read
    /// the same struct that `from_config` actually consumed.
    #[allow(dead_code)] // Used by integration tests / future endpoints.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Whether the configured backend is actually live (probe succeeded
    /// at construction). Useful for `/health` to distinguish "running on
    /// real backend" vs "fell through to hash because the upstream was down".
    pub fn backend_live(&self) -> bool {
        #[cfg(feature = "openai")]
        if self.openai_client.is_some() {
            return true;
        }
        #[cfg(feature = "ollama")]
        if self.ollama_client.is_some() {
            return true;
        }
        // Hash backend is "live" by definition — it has no upstream to probe.
        self.config.backend == "hash"
    }
}

/// Parse an Ollama endpoint string into `(host_url, port)`. Accepts:
/// - `None` → defaults to `("http://localhost", 11434)`
/// - `"localhost:11434"` (no scheme) → adds `http://`
/// - `"http://host:11434"` (full URL) → splits port off
/// - `"http://host"` (no explicit port) → defaults port to 11434
///
/// Lives here (not in graphrag-core) because ollama-rs takes host and
/// port as separate args; the core `EmbeddingConfig` carries a single
/// `api_endpoint` field to stay backend-agnostic.
#[cfg(feature = "ollama")]
fn parse_ollama_endpoint(endpoint: Option<&str>) -> (String, u16) {
    let raw = endpoint.unwrap_or("http://localhost:11434");
    let with_scheme = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };

    // Strip scheme for splitting host:port
    let without_scheme = with_scheme.split("://").nth(1).unwrap_or(&with_scheme);
    let (host_part, port) = match without_scheme.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(11434)),
        None => (without_scheme.to_string(), 11434),
    };
    let scheme = with_scheme.split("://").next().unwrap_or("http");
    (format!("{scheme}://{host_part}"), port)
}

// Bridge `EmbeddingService` (the server's real, multi-backend embedder)
// into graphrag-core's `AsyncEmbedder` trait. With this impl, the server
// can hand its `Arc<EmbeddingService>` to `GraphRAG::set_embedding_provider`
// and every internal embedding call inside graphrag-core (query embedding
// in `hybrid_query`, sentence embedding in `SemanticChunker`, etc.) routes
// through this real service instead of the hash-based dummy generator.
#[async_trait::async_trait]
impl graphrag_core::core::traits::AsyncEmbedder for EmbeddingService {
    type Error = graphrag_core::core::GraphRAGError;

    async fn embed(&self, text: &str) -> graphrag_core::Result<Vec<f32>> {
        self.generate_single(text)
            .await
            .map_err(|e| graphrag_core::core::GraphRAGError::Embedding {
                message: format!("EmbeddingService::generate_single: {}", e),
            })
    }

    async fn embed_batch(&self, texts: &[&str]) -> graphrag_core::Result<Vec<Vec<f32>>> {
        self.generate(texts)
            .await
            .map_err(|e| graphrag_core::core::GraphRAGError::Embedding {
                message: format!("EmbeddingService::generate: {}", e),
            })
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    async fn is_ready(&self) -> bool {
        // The service self-tests its backends in `from_config()` (probe
        // + fallback); by the time we hand it to core it's always ready
        // in the sense that embed() will succeed (real backend or hash
        // fallback).
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_cfg(dim: usize) -> EmbeddingConfig {
        EmbeddingConfig {
            dimension: dim,
            backend: "hash".to_string(),
            model: None,
            fallback_to_hash: true,
            api_endpoint: None,
            api_key: None,
            cache_dir: None,
            batch_size: 32,
        }
    }

    #[tokio::test]
    async fn test_fallback_embeddings() {
        let service = EmbeddingService::from_config(&hash_cfg(384)).await.unwrap();
        let embeddings = service.generate(&["test", "hello"]).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }
}
