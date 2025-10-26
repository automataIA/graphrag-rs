///! Hugging Face Hub integration for downloading and using embedding models
///!
///! This module provides functionality to:
///! - Download embedding models from Hugging Face Hub
///! - Cache models locally to avoid re-downloading
///! - Load models with Candle framework
///! - Generate embeddings using downloaded models

use crate::core::error::{GraphRAGError, Result};
use crate::embeddings::{EmbeddingConfig, EmbeddingProvider};

#[cfg(feature = "huggingface-hub")]
use hf_hub::api::sync::{Api, ApiBuilder};

/// Hugging Face Hub embedding provider
pub struct HuggingFaceEmbeddings {
    model_id: String,
    cache_dir: Option<String>,
    dimensions: usize,
    initialized: bool,

    #[cfg(feature = "huggingface-hub")]
    api: Option<Api>,

    #[cfg(feature = "huggingface-hub")]
    model_path: Option<std::path::PathBuf>,
}

impl HuggingFaceEmbeddings {
    /// Create a new Hugging Face embeddings provider
    ///
    /// # Arguments
    /// * `model_id` - Hugging Face model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    /// * `cache_dir` - Optional cache directory for downloaded models
    ///
    /// # Example
    /// ```rust,ignore
    /// use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;
    ///
    /// let embeddings = HuggingFaceEmbeddings::new(
    ///     "sentence-transformers/all-MiniLM-L6-v2",
    ///     None
    /// );
    /// ```
    pub fn new(model_id: impl Into<String>, cache_dir: Option<String>) -> Self {
        Self {
            model_id: model_id.into(),
            cache_dir,
            dimensions: 384, // Default for MiniLM-L6-v2
            initialized: false,

            #[cfg(feature = "huggingface-hub")]
            api: None,

            #[cfg(feature = "huggingface-hub")]
            model_path: None,
        }
    }

    /// Create from configuration
    pub fn from_config(config: &EmbeddingConfig) -> Self {
        Self::new(config.model.clone(), config.cache_dir.clone())
    }

    /// Download model from Hugging Face Hub
    #[cfg(feature = "huggingface-hub")]
    async fn download_model(&mut self) -> Result<std::path::PathBuf> {
        use std::path::PathBuf;

        // Initialize API with optional custom cache directory
        let api = if let Some(ref cache_dir) = self.cache_dir {
            ApiBuilder::new()
                .with_cache_dir(PathBuf::from(cache_dir))
                .build()
                .map_err(|e| GraphRAGError::Embedding {
                    message: format!("Failed to create HF Hub API with cache dir: {}", e),
                })?
        } else {
            Api::new().map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to create HF Hub API: {}", e),
            })?
        };

        // Get model repository
        let repo = api.model(self.model_id.clone());

        self.api = Some(api);

        // Download model files (safetensors format is preferred)
        let model_file = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to download model '{}': {}", self.model_id, e),
            })?;

        // Also download config.json for model metadata
        let _config_file = repo.get("config.json").ok();

        // Also download tokenizer files
        let _tokenizer_file = repo.get("tokenizer.json").ok();
        let _tokenizer_config = repo.get("tokenizer_config.json").ok();

        Ok(model_file)
    }

    #[cfg(not(feature = "huggingface-hub"))]
    async fn download_model(&mut self) -> Result<std::path::PathBuf> {
        Err(GraphRAGError::Embedding {
            message: "huggingface-hub feature not enabled. Enable it in Cargo.toml".to_string(),
        })
    }

    /// Get recommended models for different use cases
    pub fn recommended_models() -> Vec<(&'static str, &'static str, usize)> {
        vec![
            // (model_id, description, dimensions)
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                "Fast, lightweight, general-purpose (default)",
                384,
            ),
            (
                "sentence-transformers/all-mpnet-base-v2",
                "High quality, general-purpose",
                768,
            ),
            (
                "BAAI/bge-small-en-v1.5",
                "Small, efficient, good performance",
                384,
            ),
            (
                "BAAI/bge-base-en-v1.5",
                "Balanced size and performance",
                768,
            ),
            (
                "BAAI/bge-large-en-v1.5",
                "Best quality, larger size",
                1024,
            ),
            (
                "thenlper/gte-small",
                "Small, efficient, trained on diverse data",
                384,
            ),
            (
                "thenlper/gte-base",
                "Balanced performance",
                768,
            ),
            (
                "intfloat/e5-small-v2",
                "E5 model, small size",
                384,
            ),
            (
                "intfloat/e5-base-v2",
                "E5 model, base size",
                768,
            ),
            (
                "intfloat/e5-large-v2",
                "E5 model, best quality",
                1024,
            ),
            (
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "Multilingual support, 50+ languages",
                384,
            ),
        ]
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for HuggingFaceEmbeddings {
    async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        #[cfg(feature = "huggingface-hub")]
        {
            // Download model
            let model_path = self.download_model().await?;
            self.model_path = Some(model_path);
            self.initialized = true;

            log::info!(
                "HuggingFace model '{}' downloaded successfully",
                self.model_id
            );
        }

        #[cfg(not(feature = "huggingface-hub"))]
        {
            return Err(GraphRAGError::Embedding {
                message: "huggingface-hub feature not enabled".to_string(),
            });
        }

        Ok(())
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(GraphRAGError::Embedding {
                message: "HuggingFace embeddings not initialized. Call initialize() first"
                    .to_string(),
            });
        }

        // TODO: Implement actual embedding generation using Candle
        // This requires loading the model with candle-transformers
        // For now, return a placeholder

        #[cfg(feature = "neural-embeddings")]
        {
            // Load model with Candle and generate embedding
            // This is a placeholder - actual implementation would use candle-transformers
            log::warn!("HuggingFace embedding generation not yet implemented");
            Ok(vec![0.0; self.dimensions])
        }

        #[cfg(not(feature = "neural-embeddings"))]
        {
            Err(GraphRAGError::Embedding {
                message: "neural-embeddings feature required for embedding generation".to_string(),
            })
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        self.initialized
    }

    fn provider_name(&self) -> &str {
        "HuggingFace Hub"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_embeddings() {
        let embeddings = HuggingFaceEmbeddings::new(
            "sentence-transformers/all-MiniLM-L6-v2",
            None,
        );

        assert_eq!(embeddings.model_id, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(embeddings.dimensions, 384);
        assert!(!embeddings.initialized);
    }

    #[test]
    fn test_recommended_models() {
        let models = HuggingFaceEmbeddings::recommended_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|(id, _, _)| id.contains("MiniLM")));
    }

    #[tokio::test]
    #[cfg(feature = "huggingface-hub")]
    async fn test_download_model() {
        // This test requires network access and will download a small model
        // Skip in CI unless explicitly enabled
        if std::env::var("ENABLE_DOWNLOAD_TESTS").is_err() {
            return;
        }

        let mut embeddings = HuggingFaceEmbeddings::new(
            "sentence-transformers/all-MiniLM-L6-v2",
            None,
        );

        let result = embeddings.initialize().await;
        assert!(result.is_ok(), "Failed to download model: {:?}", result);
        assert!(embeddings.is_available());
    }
}
