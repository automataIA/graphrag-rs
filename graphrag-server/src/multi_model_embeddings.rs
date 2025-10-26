//! Multi-Model Embeddings Support
//!
//! This module provides a unified interface for multiple embedding providers:
//! - OpenAI (Ada-2, text-embedding-3-small, text-embedding-3-large)
//! - Cohere (embed-english-v3.0, embed-multilingual-v3.0)
//! - Local models (Sentence Transformers via Ollama/Candle)
//! - HuggingFace Inference API
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │     EmbeddingRouter                 │
//! │   (Auto-selection & fallback)       │
//! └──────────────┬──────────────────────┘
//!                │
//!     ┌──────────┼──────────┬──────────┐
//!     ▼          ▼          ▼          ▼
//! ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
//! │ OpenAI │ │ Cohere │ │  Local │ │  HF    │
//! └────────┘ └────────┘ └────────┘ └────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use async_trait::async_trait;

/// Embedding model provider
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingProvider {
    /// OpenAI embeddings
    OpenAI,
    /// Cohere embeddings
    Cohere,
    /// Local models (Ollama/Candle)
    Local,
    /// HuggingFace Inference API
    HuggingFace,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider
    pub provider: EmbeddingProvider,
    /// Model name (e.g., "text-embedding-3-small", "embed-english-v3.0")
    pub model_name: String,
    /// Embedding dimension
    pub dimension: usize,
    /// API key (if required)
    pub api_key: Option<String>,
    /// API endpoint (for custom deployments)
    pub endpoint: Option<String>,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Local,
            model_name: "nomic-embed-text".to_string(),
            dimension: 384,
            api_key: None,
            endpoint: None,
            batch_size: 32,
        }
    }
}

/// Embedding result with metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Model used
    pub model: String,
    /// Provider used
    pub provider: EmbeddingProvider,
    /// Token count (if available)
    pub token_count: Option<usize>,
}

/// Embedding provider trait
#[async_trait]
pub trait EmbeddingProviderTrait: Send + Sync {
    /// Generate embedding for text
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for batch of texts
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Get model dimension
    fn dimension(&self) -> usize;

    /// Get model name
    fn model_name(&self) -> &str;
}

/// Embedding errors
#[derive(Debug, Clone)]
pub enum EmbeddingError {
    /// API error
    ApiError(String),
    /// Network error
    NetworkError(String),
    /// Invalid input
    InvalidInput(String),
    /// Rate limit exceeded
    RateLimited,
    /// Provider unavailable
    Unavailable,
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            EmbeddingError::ApiError(msg) => write!(f, "API error: {}", msg),
            EmbeddingError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            EmbeddingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            EmbeddingError::RateLimited => write!(f, "Rate limit exceeded"),
            EmbeddingError::Unavailable => write!(f, "Provider unavailable"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// OpenAI embeddings provider
pub struct OpenAIProvider {
    _api_key: String,
    model_name: String,
    dimension: usize,
    _endpoint: String,
}

impl OpenAIProvider {
    /// Create new OpenAI provider
    pub fn new(api_key: String, model_name: String, dimension: usize) -> Self {
        Self {
            _api_key: api_key,
            model_name,
            dimension,
            _endpoint: "https://api.openai.com/v1/embeddings".to_string(),
        }
    }
}

#[async_trait]
impl EmbeddingProviderTrait for OpenAIProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // TODO: Implement actual OpenAI API call
        // For now, return placeholder
        tracing::debug!("OpenAI embedding for: {} chars", text.len());
        Ok(vec![0.0; self.dimension])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Cohere embeddings provider
pub struct CohereProvider {
    _api_key: String,
    model_name: String,
    dimension: usize,
}

impl CohereProvider {
    /// Create new Cohere provider
    pub fn new(api_key: String, model_name: String, dimension: usize) -> Self {
        Self {
            _api_key: api_key,
            model_name,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingProviderTrait for CohereProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // TODO: Implement actual Cohere API call
        tracing::debug!("Cohere embedding for: {} chars", text.len());
        Ok(vec![0.0; self.dimension])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Embedding router with auto-fallback
pub struct EmbeddingRouter {
    /// Primary provider
    primary: Arc<dyn EmbeddingProviderTrait>,
    /// Fallback providers
    fallbacks: Vec<Arc<dyn EmbeddingProviderTrait>>,
    /// Router statistics
    stats: Arc<parking_lot::RwLock<RouterStats>>,
}

/// Router statistics
#[derive(Debug, Default, Clone)]
pub struct RouterStats {
    /// Primary provider requests
    pub primary_requests: u64,
    /// Primary provider successes
    pub primary_successes: u64,
    /// Fallback requests per provider
    pub fallback_requests: Vec<u64>,
    /// Fallback successes per provider
    pub fallback_successes: Vec<u64>,
}

impl EmbeddingRouter {
    /// Create new router
    pub fn new(primary: Arc<dyn EmbeddingProviderTrait>) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
            stats: Arc::new(parking_lot::RwLock::new(RouterStats::default())),
        }
    }

    /// Add fallback provider
    pub fn with_fallback(mut self, provider: Arc<dyn EmbeddingProviderTrait>) -> Self {
        self.fallbacks.push(provider);
        self
    }

    /// Generate embedding with automatic fallback
    pub async fn embed(&self, text: &str) -> Result<EmbeddingResult, EmbeddingError> {
        self.stats.write().primary_requests += 1;

        // Try primary
        match self.primary.embed(text).await {
            Ok(embedding) => {
                self.stats.write().primary_successes += 1;
                return Ok(EmbeddingResult {
                    embedding,
                    model: self.primary.model_name().to_string(),
                    provider: EmbeddingProvider::Local, // TODO: Track actual provider
                    token_count: None,
                });
            }
            Err(e) => {
                tracing::warn!("Primary provider failed: {}", e);
            }
        }

        // Try fallbacks
        for (i, fallback) in self.fallbacks.iter().enumerate() {
            match fallback.embed(text).await {
                Ok(embedding) => {
                    // Update stats
                    let mut stats = self.stats.write();
                    while stats.fallback_successes.len() <= i {
                        stats.fallback_requests.push(0);
                        stats.fallback_successes.push(0);
                    }
                    stats.fallback_requests[i] += 1;
                    stats.fallback_successes[i] += 1;

                    return Ok(EmbeddingResult {
                        embedding,
                        model: fallback.model_name().to_string(),
                        provider: EmbeddingProvider::Local, // TODO: Track actual provider
                        token_count: None,
                    });
                }
                Err(e) => {
                    tracing::warn!("Fallback {} failed: {}", i, e);
                }
            }
        }

        Err(EmbeddingError::Unavailable)
    }

    /// Generate batch embeddings
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Get router statistics
    pub fn stats(&self) -> RouterStats {
        self.stats.read().clone()
    }

    /// Get primary success rate
    pub fn primary_success_rate(&self) -> f64 {
        let stats = self.stats.read();
        if stats.primary_requests == 0 {
            0.0
        } else {
            (stats.primary_successes as f64) / (stats.primary_requests as f64)
        }
    }
}

/// Model registry for managing multiple models
pub struct ModelRegistry {
    /// Available models
    models: HashMap<String, Arc<dyn EmbeddingProviderTrait>>,
    /// Default model
    default: String,
}

impl ModelRegistry {
    /// Create new registry
    pub fn new(default: String) -> Self {
        Self {
            models: HashMap::new(),
            default,
        }
    }

    /// Register a model
    pub fn register(&mut self, name: String, provider: Arc<dyn EmbeddingProviderTrait>) {
        self.models.insert(name, provider);
    }

    /// Get model by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn EmbeddingProviderTrait>> {
        self.models.get(name).cloned()
    }

    /// Get default model
    pub fn get_default(&self) -> Option<Arc<dyn EmbeddingProviderTrait>> {
        self.models.get(&self.default).cloned()
    }

    /// List all models
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider {
        dimension: usize,
        model_name: String,
    }

    #[async_trait]
    impl EmbeddingProviderTrait for MockProvider {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            Ok(vec![1.0; self.dimension])
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(vec![vec![1.0; self.dimension]; texts.len()])
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn model_name(&self) -> &str {
            &self.model_name
        }
    }

    #[tokio::test]
    async fn test_router() {
        let primary = Arc::new(MockProvider {
            dimension: 384,
            model_name: "primary".to_string(),
        });

        let fallback = Arc::new(MockProvider {
            dimension: 384,
            model_name: "fallback".to_string(),
        });

        let router = EmbeddingRouter::new(primary).with_fallback(fallback);

        let result = router.embed("test").await.unwrap();
        assert_eq!(result.embedding.len(), 384);
        assert_eq!(router.primary_success_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_registry() {
        let mut registry = ModelRegistry::new("default".to_string());

        registry.register(
            "default".to_string(),
            Arc::new(MockProvider {
                dimension: 384,
                model_name: "test".to_string(),
            }),
        );

        let provider = registry.get_default().unwrap();
        let embedding = provider.embed("test").await.unwrap();
        assert_eq!(embedding.len(), 384);
    }
}
