//! Embedding generation for GraphRAG
//!
//! This module provides embedding generation capabilities using various backends:
//! - Hugging Face Hub models (via hf-hub crate)
//! - Local models (ONNX, Candle)
//! - API providers (OpenAI, Voyage AI, Cohere, etc.)

use crate::core::error::Result;

/// Hugging Face Hub integration for downloading and using embedding models
#[cfg(feature = "huggingface-hub")]
pub mod huggingface;

/// Neural embedding models (local inference)
#[cfg(feature = "neural-embeddings")]
pub mod neural;

/// API-based embedding providers (OpenAI, Voyage AI, Cohere, etc.)
#[cfg(feature = "ureq")]
pub mod api_providers;

/// TOML configuration for embedding providers
pub mod config;

/// Trait for embedding providers
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Initialize the embedding provider
    async fn initialize(&mut self) -> Result<()>;

    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch processing)
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension
    fn dimensions(&self) -> usize;

    /// Check if the provider is available and ready
    fn is_available(&self) -> bool;

    /// Get the provider name
    fn provider_name(&self) -> &str;
}

/// Configuration for embedding providers
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Provider type (huggingface, openai, voyage, cohere, etc.)
    pub provider: EmbeddingProviderType,

    /// Model name/identifier
    pub model: String,

    /// API key (if required)
    pub api_key: Option<String>,

    /// Cache directory for downloaded models
    pub cache_dir: Option<String>,

    /// Batch size for processing multiple texts
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProviderType::HuggingFace,
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            api_key: None,
            cache_dir: None,
            batch_size: 32,
        }
    }
}

/// Supported embedding provider types
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingProviderType {
    /// Hugging Face Hub models (free, downloadable)
    HuggingFace,

    /// OpenAI embeddings API
    OpenAI,

    /// Voyage AI embeddings API (recommended by Anthropic)
    VoyageAI,

    /// Cohere embeddings API
    Cohere,

    /// Jina AI embeddings API
    JinaAI,

    /// Mistral AI embeddings API
    Mistral,

    /// Together AI embeddings API
    TogetherAI,

    /// Local ONNX model
    Onnx,

    /// Local Candle model
    Candle,

    /// Custom provider
    Custom(String),
}

impl std::fmt::Display for EmbeddingProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace => write!(f, "HuggingFace"),
            Self::OpenAI => write!(f, "OpenAI"),
            Self::VoyageAI => write!(f, "VoyageAI"),
            Self::Cohere => write!(f, "Cohere"),
            Self::JinaAI => write!(f, "JinaAI"),
            Self::Mistral => write!(f, "Mistral"),
            Self::TogetherAI => write!(f, "TogetherAI"),
            Self::Onnx => write!(f, "ONNX"),
            Self::Candle => write!(f, "Candle"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.provider, EmbeddingProviderType::HuggingFace);
        assert_eq!(config.model, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(
            EmbeddingProviderType::HuggingFace.to_string(),
            "HuggingFace"
        );
        assert_eq!(EmbeddingProviderType::OpenAI.to_string(), "OpenAI");
        assert_eq!(EmbeddingProviderType::VoyageAI.to_string(), "VoyageAI");
    }
}
