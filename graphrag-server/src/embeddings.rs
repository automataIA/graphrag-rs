//! Embeddings module for GraphRAG Server
//!
//! Provides a unified interface for generating embeddings using various backends:
//! - Ollama (local LLM service)
//! - Hash-based fallback (deterministic, no external dependencies)
//!
//! ## Usage
//!
//! ```rust
//! let embedder = EmbeddingService::new(EmbeddingConfig::default()).await?;
//! let embedding = embedder.generate(&["Hello world"]).await?;
//! ```

use graphrag_core::vector::EmbeddingGenerator;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

#[cfg(feature = "ollama")]
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding backend: "ollama", "openai", or "hash"
    pub backend: String,
    /// Embedding dimension (384 for MiniLM, 768 for BERT, 1024 for mxbai)
    pub dimension: usize,
    /// Ollama base URL (if using Ollama)
    pub ollama_url: String,
    /// Ollama embedding model name
    pub ollama_model: String,
    /// OpenAI-compatible base URL (e.g. "http://localhost:8000/v1" for
    /// vLLM, "http://localhost:9000/v3" for OpenVINO Model Server,
    /// "http://localhost:17171/v1" for llama-server, or
    /// "https://api.openai.com/v1" for the real thing).
    pub openai_url: String,
    /// OpenAI-compat model name (sent in the JSON body's `model` field).
    pub openai_model: String,
    /// API key. Empty string is fine for self-hosted servers (vLLM, OVMS,
    /// llama-server) that don't authenticate.
    pub openai_api_key: String,
    /// Enable caching
    pub enable_cache: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: "ollama".to_string(),
            dimension: 384,
            ollama_url: "http://localhost:11434".to_string(),
            ollama_model: "nomic-embed-text".to_string(),
            openai_url: "http://localhost:8000/v1".to_string(),
            openai_model: "BAAI/bge-m3".to_string(),
            openai_api_key: String::new(),
            enable_cache: true,
        }
    }
}

/// OpenAI-compatible embedding HTTP client. Holds the configured URL,
/// model name, and API key; used by `generate_with_openai`.
#[cfg(feature = "openai")]
struct OpenAIClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

/// Embedding service with automatic fallback
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
    pub ollama_success: usize,
    pub ollama_failures: usize,
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
    #[allow(dead_code)]
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
    /// Create a new embedding service
    pub async fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        info!(
            "Initializing embedding service with backend: {}",
            config.backend
        );

        // Try to initialize Ollama if requested
        #[cfg(feature = "ollama")]
        let ollama_client = if config.backend == "ollama" {
            let ollama = Ollama::new(config.ollama_url.clone(), 11434);

            // Check if Ollama is available
            match ollama.list_local_models().await {
                Ok(models) => {
                    info!("✓ Ollama connection established");

                    // Check if embedding model exists
                    let model_exists = models.iter().any(|m| m.name == config.ollama_model);

                    if model_exists {
                        info!("✓ Embedding model '{}' is available", config.ollama_model);
                        Some(Arc::new(ollama))
                    } else {
                        warn!(
                            "⚠ Embedding model '{}' not found. Using fallback. Run: ollama pull {}",
                            config.ollama_model, config.ollama_model
                        );
                        None
                    }
                },
                Err(e) => {
                    warn!(
                        "⚠ Ollama service not available: {}. Using fallback embeddings.",
                        e
                    );
                    None
                },
            }
        } else {
            info!("Using hash-based fallback embeddings (no Ollama)");
            None
        };

        #[cfg(not(feature = "ollama"))]
        if config.backend == "ollama" {
            warn!("⚠ Ollama support not compiled in. Using fallback embeddings. Rebuild with --features ollama");
        }

        // OpenAI-compat backend (vLLM, OVMS, llama-server, etc.). Gated
        // behind feature = "openai" — when off, the user's
        // `EMBEDDING_BACKEND=openai` falls through to hash with a clear
        // log line, mirroring the ollama-feature-off behavior.
        #[cfg(feature = "openai")]
        let openai_client = if config.backend == "openai" {
            let http = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(|e| EmbeddingError::OpenAIError(format!("client build failed: {e}")))?;

            // Probe the /models endpoint to confirm the server is up. We
            // don't fail-hard if the model isn't listed — some servers
            // (vLLM single-model mode, OVMS Mediapipe graphs) report
            // synthetic names that don't match config.openai_model.
            let probe_url = format!("{}/models", config.openai_url.trim_end_matches('/'));
            let mut req = http.get(&probe_url);
            if !config.openai_api_key.is_empty() {
                req = req.bearer_auth(&config.openai_api_key);
            }
            match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("✓ OpenAI-compat server reachable at {}", config.openai_url);
                    Some(Arc::new(OpenAIClient {
                        http,
                        base_url: config.openai_url.clone(),
                        model: config.openai_model.clone(),
                        api_key: config.openai_api_key.clone(),
                    }))
                },
                Ok(resp) => {
                    warn!(
                        "⚠ OpenAI-compat /models returned {}. Using fallback embeddings. URL: {}",
                        resp.status(),
                        config.openai_url
                    );
                    None
                },
                Err(e) => {
                    warn!(
                        "⚠ OpenAI-compat probe failed: {}. Using fallback embeddings. URL: {}",
                        e, config.openai_url
                    );
                    None
                },
            }
        } else {
            None
        };

        #[cfg(not(feature = "openai"))]
        if config.backend == "openai" {
            warn!("⚠ OpenAI-compat embeddings not compiled in. Using fallback. Rebuild with --features openai");
        }

        // Always create fallback generator
        let fallback_generator = Arc::new(RwLock::new(EmbeddingGenerator::new(config.dimension)));

        Ok(Self {
            config,
            #[cfg(feature = "ollama")]
            ollama_client,
            #[cfg(feature = "openai")]
            openai_client,
            fallback_generator,
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        })
    }

    /// Generate embeddings for a batch of texts. Tries the configured
    /// backend (ollama or openai), falls through to the hash generator if
    /// the backend errors.
    pub async fn generate(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut stats = self.stats.write().await;
        stats.total_requests += texts.len();
        drop(stats); // Release lock

        // OpenAI-compat backend (vLLM, OVMS, llama-server, ...).
        #[cfg(feature = "openai")]
        if let Some(client) = &self.openai_client {
            match self.generate_with_openai(client, texts).await {
                Ok(embeddings) => {
                    let mut stats = self.stats.write().await;
                    stats.ollama_success += texts.len();
                    return Ok(embeddings);
                },
                Err(e) => {
                    warn!("OpenAI-compat embedding failed: {}. Using fallback.", e);
                    let mut stats = self.stats.write().await;
                    stats.ollama_failures += texts.len();
                },
            }
        }

        // Try Ollama next.
        #[cfg(feature = "ollama")]
        if let Some(ollama) = &self.ollama_client {
            match self.generate_with_ollama(ollama, texts).await {
                Ok(embeddings) => {
                    let mut stats = self.stats.write().await;
                    stats.ollama_success += texts.len();
                    return Ok(embeddings);
                },
                Err(e) => {
                    warn!("Ollama embedding failed: {}. Using fallback.", e);
                    let mut stats = self.stats.write().await;
                    stats.ollama_failures += texts.len();
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
    /// llama-server, OpenAI itself, …). Sends one POST per text — most
    /// servers also accept a batch (input as an array) but using
    /// per-text requests keeps the dimension-validation path simple.
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
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let request = GenerateEmbeddingsRequest::new(
                self.config.ollama_model.clone(),
                text.to_string().into(),
            );

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
    #[allow(dead_code)]
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get current statistics
    #[allow(dead_code)]
    pub async fn get_stats(&self) -> EmbeddingStats {
        self.stats.read().await.clone()
    }

    /// Check if Ollama is available
    #[allow(dead_code)]
    pub fn is_ollama_available(&self) -> bool {
        #[cfg(feature = "ollama")]
        {
            self.ollama_client.is_some()
        }
        #[cfg(not(feature = "ollama"))]
        {
            false
        }
    }

    /// Get backend name
    pub fn backend_name(&self) -> &str {
        #[cfg(feature = "openai")]
        {
            if self.openai_client.is_some() {
                return "openai";
            }
        }
        #[cfg(feature = "ollama")]
        {
            if self.ollama_client.is_some() {
                return "ollama";
            }
        }
        "hash-fallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_embeddings() {
        let config = EmbeddingConfig {
            backend: "hash".to_string(),
            dimension: 384,
            ..Default::default()
        };

        let service = EmbeddingService::new(config).await.unwrap();
        let embeddings = service.generate(&["test", "hello"]).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }

    #[tokio::test]
    async fn test_ollama_embeddings() {
        let config = EmbeddingConfig::default();

        if let Ok(service) = EmbeddingService::new(config).await {
            if service.is_ollama_available() {
                let embeddings = service.generate(&["test"]).await.unwrap();
                assert_eq!(embeddings.len(), 1);
                println!("Ollama embedding dimension: {}", embeddings[0].len());
            } else {
                println!("Ollama not available, using fallback");
            }
        }
    }
}
