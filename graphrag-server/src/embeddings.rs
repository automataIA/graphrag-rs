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
use ollama_rs::{
    Ollama,
    generation::embeddings::request::GenerateEmbeddingsRequest,
};

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding backend: "ollama" or "hash"
    pub backend: String,
    /// Embedding dimension (384 for MiniLM, 768 for BERT)
    pub dimension: usize,
    /// Ollama base URL (if using Ollama)
    pub ollama_url: String,
    /// Ollama embedding model name
    pub ollama_model: String,
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
            enable_cache: true,
        }
    }
}

/// Embedding service with automatic fallback
pub struct EmbeddingService {
    config: EmbeddingConfig,
    #[cfg(feature = "ollama")]
    ollama_client: Option<Arc<Ollama>>,
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

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Invalid dimension: expected {expected}, got {actual}")]
    #[allow(dead_code)]
    DimensionMismatch { expected: usize, actual: usize },
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
                    // Model names from Ollama include tags (e.g., "nomic-embed-text:latest")
                    // so we check if the model name starts with the config model name
                    let model_exists = models.iter().any(|m| {
                        m.name == config.ollama_model
                            || m.name.starts_with(&format!("{}:", config.ollama_model))
                    });

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
                }
                Err(e) => {
                    warn!("⚠ Ollama service not available: {}. Using fallback embeddings.", e);
                    None
                }
            }
        } else {
            info!("Using hash-based fallback embeddings (no Ollama)");
            None
        };

        #[cfg(not(feature = "ollama"))]
        if config.backend == "ollama" {
            warn!("⚠ Ollama support not compiled in. Using fallback embeddings. Rebuild with --features ollama");
        }

        // Always create fallback generator
        let fallback_generator = Arc::new(RwLock::new(EmbeddingGenerator::new(config.dimension)));

        Ok(Self {
            config,
            #[cfg(feature = "ollama")]
            ollama_client,
            fallback_generator,
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        })
    }

    /// Generate embeddings for a batch of texts
    pub async fn generate(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut stats = self.stats.write().await;
        stats.total_requests += texts.len();
        drop(stats); // Release lock

        // Try Ollama first if available
        #[cfg(feature = "ollama")]
        if let Some(ollama) = &self.ollama_client {
            match self.generate_with_ollama(ollama, texts).await {
                Ok(embeddings) => {
                    let mut stats = self.stats.write().await;
                    stats.ollama_success += texts.len();
                    return Ok(embeddings);
                }
                Err(e) => {
                    warn!("Ollama embedding failed: {}. Using fallback.", e);
                    let mut stats = self.stats.write().await;
                    stats.ollama_failures += texts.len();
                }
            }
        }

        // Fallback to hash-based embeddings
        let mut stats = self.stats.write().await;
        stats.fallback_used += texts.len();
        drop(stats);

        self.generate_with_fallback(texts).await
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
    async fn generate_with_ollama(&self, ollama: &Ollama, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let request = GenerateEmbeddingsRequest::new(
                self.config.ollama_model.clone(),
                text.to_string().into(),
            );

            let response = ollama.generate_embeddings(request).await?;

            let embedding = response
                .embeddings
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::GenerationFailed("No embedding in response".to_string()))?;

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
    async fn generate_with_fallback(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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
