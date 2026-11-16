//! Embeddings module for GraphRAG Server
//!
//! Provides a unified interface for generating embeddings using various backends:
//! - llama.cpp (`llama-server --embeddings --pooling mean`, multilingual-e5-small)
//! - Hash-based fallback (deterministic, no external dependencies)
//!
//! ## Usage
//!
//! ```rust
//! use graphrag_server::embeddings::{EmbeddingConfig, EmbeddingService};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let embedder = EmbeddingService::new(EmbeddingConfig::default()).await?;
//!     let _embedding = embedder.generate(&["Hello world"]).await?;
//!     Ok(())
//! }
//! ```

use graphrag_core::vector::EmbeddingGenerator;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

#[cfg(feature = "llm")]
use graphrag_core::llama_cpp::{E5Intent, LlamaCppEmbeddingProvider};

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding backend: "llama_cpp" or "hash"
    pub backend: String,
    /// Embedding dimension (384 for multilingual-e5-small)
    pub dimension: usize,
    /// llama.cpp embeddings endpoint (e.g. `http://127.0.0.1:8081/v1/embeddings`)
    pub endpoint: String,
    /// Embedding model name (llama-server `--alias`)
    pub model: String,
    /// Enable caching
    pub enable_cache: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: "llama_cpp".to_string(),
            dimension: 384,
            endpoint: "http://127.0.0.1:8081/v1/embeddings".to_string(),
            model: "multilingual-e5-small".to_string(),
            enable_cache: true,
        }
    }
}

/// Embedding service with automatic fallback
pub struct EmbeddingService {
    config: EmbeddingConfig,
    #[cfg(feature = "llm")]
    provider: Option<Arc<LlamaCppEmbeddingProvider>>,
    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    fallback_generator: Arc<RwLock<EmbeddingGenerator>>,
    stats: Arc<RwLock<EmbeddingStats>>,
}

/// Embedding statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct EmbeddingStats {
    pub total_requests: usize,
    pub llm_success: usize,
    pub llm_failures: usize,
    pub fallback_used: usize,
    pub cache_hits: usize,
}

/// Embedding error type
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Generation failed: {0}")]
    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    GenerationFailed(String),

    #[error("Invalid dimension: expected {expected}, got {actual}")]
    #[allow(dead_code)]
    DimensionMismatch { expected: usize, actual: usize },
}

impl EmbeddingService {
    /// Create a new embedding service.
    ///
    /// With the `llama_cpp` backend the server is probed once at startup. If it
    /// is unreachable the whole service uses hash embeddings; once the llama.cpp
    /// provider is active, later failures are returned as errors instead of
    /// silently mixing hash vectors into an e5 index.
    pub async fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        info!(
            "Initializing embedding service with backend: {}",
            config.backend
        );

        let wants_llama = matches!(config.backend.as_str(), "llama_cpp" | "llama.cpp");

        #[cfg(feature = "llm")]
        let provider = if wants_llama {
            Self::probe_llama_cpp(&config).await
        } else {
            info!("Using hash-based fallback embeddings (no llama.cpp)");
            None
        };

        #[cfg(not(feature = "llm"))]
        if wants_llama {
            warn!("⚠ llama.cpp support not compiled in. Using fallback embeddings. Rebuild with --features llm");
        }

        // Always create fallback generator
        let fallback_generator = Arc::new(RwLock::new(EmbeddingGenerator::new(config.dimension)));

        Ok(Self {
            config,
            #[cfg(feature = "llm")]
            provider,
            fallback_generator,
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        })
    }

    #[cfg(feature = "llm")]
    async fn probe_llama_cpp(config: &EmbeddingConfig) -> Option<Arc<LlamaCppEmbeddingProvider>> {
        let core_config = graphrag_core::config::EmbeddingConfig {
            backend: "llama_cpp".to_string(),
            dimension: config.dimension,
            model: Some(config.model.clone()),
            api_endpoint: Some(config.endpoint.clone()),
            fallback_to_hash: false,
            api_key: None,
            cache_dir: None,
            batch_size: 32,
        };
        let provider = match LlamaCppEmbeddingProvider::from_config(&core_config, 30) {
            Ok(provider) => Arc::new(provider),
            Err(e) => {
                warn!(
                    "⚠ Invalid llama.cpp embedding config: {}. Using fallback embeddings.",
                    e
                );
                return None;
            },
        };
        let probe = Arc::clone(&provider);
        match tokio::task::spawn_blocking(move || probe.embed_document("probe")).await {
            Ok(Ok(_)) => {
                info!(
                    "✓ llama.cpp embedding server ready ({}, model '{}')",
                    config.endpoint, config.model
                );
                Some(provider)
            },
            Ok(Err(e)) => {
                warn!(
                    "⚠ llama.cpp embedding server not available: {}. Using fallback embeddings.",
                    e
                );
                None
            },
            Err(e) => {
                warn!(
                    "⚠ llama.cpp probe task failed: {}. Using fallback embeddings.",
                    e
                );
                None
            },
        }
    }

    /// Generate document (passage) embeddings for a batch of texts
    #[allow(dead_code)]
    pub async fn generate(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.generate_with_intent(texts, false).await
    }

    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    async fn generate_with_intent(
        &self,
        texts: &[&str],
        is_query: bool,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut stats = self.stats.write().await;
        stats.total_requests += texts.len();
        drop(stats);

        #[cfg(feature = "llm")]
        if let Some(provider) = &self.provider {
            let provider = Arc::clone(provider);
            let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
            let intent = if is_query {
                E5Intent::Query
            } else {
                E5Intent::Document
            };
            let result = tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
                provider.embed_batch(&refs, intent)
            })
            .await
            .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))?;
            let mut stats = self.stats.write().await;
            return match result {
                Ok(embeddings) => {
                    stats.llm_success += texts.len();
                    Ok(embeddings)
                },
                Err(e) => {
                    stats.llm_failures += texts.len();
                    Err(EmbeddingError::GenerationFailed(e.to_string()))
                },
            };
        }
        #[cfg(not(feature = "llm"))]
        let _ = is_query;

        // Hash-based embeddings
        let mut stats = self.stats.write().await;
        stats.fallback_used += texts.len();
        drop(stats);

        self.generate_with_fallback(texts).await
    }

    /// Generate a single document (passage) embedding
    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    pub async fn generate_single(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Self::first(self.generate_with_intent(&[text], false).await?)
    }

    /// Generate a single query embedding (E5 `query:` prefix)
    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    pub async fn generate_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Self::first(self.generate_with_intent(&[text], true).await?)
    }

    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
    fn first(results: Vec<Vec<f32>>) -> Result<Vec<f32>, EmbeddingError> {
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::GenerationFailed("No embedding generated".to_string()))
    }

    /// Generate embeddings using hash-based fallback
    #[cfg_attr(not(feature = "qdrant"), allow(dead_code))]
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

    /// Check if the llama.cpp embedding server is in use
    #[allow(dead_code)]
    pub fn is_llm_available(&self) -> bool {
        #[cfg(feature = "llm")]
        {
            self.provider.is_some()
        }
        #[cfg(not(feature = "llm"))]
        {
            false
        }
    }

    /// Get backend name
    pub fn backend_name(&self) -> &str {
        #[cfg(feature = "llm")]
        {
            if self.provider.is_some() {
                return "llama_cpp";
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
    async fn test_unreachable_llama_cpp_falls_back_to_hash() {
        let config = EmbeddingConfig {
            endpoint: "http://127.0.0.1:9/v1/embeddings".to_string(),
            ..Default::default()
        };

        let service = EmbeddingService::new(config).await.unwrap();
        assert!(!service.is_llm_available());
        assert_eq!(service.backend_name(), "hash-fallback");
        assert_eq!(service.generate_query("ciao").await.unwrap().len(), 384);
    }

    /// Requires `llama-server --embeddings` on 127.0.0.1:8081.
    #[cfg(feature = "llm")]
    #[tokio::test]
    #[ignore]
    async fn live_llama_cpp_embeddings() {
        let service = EmbeddingService::new(EmbeddingConfig::default())
            .await
            .unwrap();
        assert_eq!(service.backend_name(), "llama_cpp");
        let query = service.generate_query("Chi dirige Prisma?").await.unwrap();
        let docs = service
            .generate(&["Maria Rossi dirige Prisma.", "Il gatto dorme."])
            .await
            .unwrap();
        assert_eq!(query.len(), 384);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&query, &docs[0]) > dot(&query, &docs[1]));
    }
}
