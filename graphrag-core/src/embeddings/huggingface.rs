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
#[cfg(feature = "neural-embeddings")]
use candle_core::{Device, Tensor};
#[cfg(feature = "neural-embeddings")]
use candle_nn::VarBuilder;
#[cfg(feature = "neural-embeddings")]
use candle_transformers::models::bert::{BertModel, Config, Dtype};
#[cfg(feature = "neural-embeddings")]
use tokenizers::Tokenizer;

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

    #[cfg(feature = "neural-embeddings")]
    model: Option<BertModel>,

    #[cfg(feature = "neural-embeddings")]
    tokenizer: Option<Tokenizer>,

    #[cfg(feature = "neural-embeddings")]
    device: Option<Device>,
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

            #[cfg(feature = "neural-embeddings")]
            model: None,

            #[cfg(feature = "neural-embeddings")]
            tokenizer: None,

            #[cfg(feature = "neural-embeddings")]
            device: None,
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

        // Also download config.json for model metadata (and cache it)
        let _ = repo
            .get("config.json")
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to download config for '{}': {}", self.model_id, e),
            })?;

        // Also download tokenizer files (and cache them)
        let _ = repo
            .get("tokenizer.json")
            .map_err(|e| GraphRAGError::Embedding {
                message: format!(
                    "Failed to download tokenizer for '{}': {}",
                    self.model_id, e
                ),
            })?;
        let _ = repo.get("tokenizer_config.json").ok();

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
                "BAAI/bge-m3",
                "State-of-the-art multilingual, dense + sparse + colbert",
                1024,
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
            self.model_path = Some(model_path.clone());

            #[cfg(feature = "neural-embeddings")]
            {
                // Initialize Candle model and tokenizer
                let api = self.api.as_ref().ok_or_else(|| GraphRAGError::Embedding {
                    message: "HF API not initialized".to_string(),
                })?;
                let repo = api.model(self.model_id.clone());

                let tokenizer_filename =
                    repo.get("tokenizer.json")
                        .map_err(|e| GraphRAGError::Embedding {
                            message: format!("Failed to get tokenizer: {}", e),
                        })?;

                let config_filename =
                    repo.get("config.json")
                        .map_err(|e| GraphRAGError::Embedding {
                            message: format!("Failed to get config: {}", e),
                        })?;

                let device = Device::Cpu; // Default to CPU for max compatibility
                let config: Config =
                    serde_json::from_str(&std::fs::read_to_string(config_filename).map_err(
                        |e| GraphRAGError::Embedding {
                            message: format!("Failed to read config file: {}", e),
                        },
                    )?)
                    .map_err(|e| GraphRAGError::Embedding {
                        message: format!("Failed to parse config: {}", e),
                    })?;

                let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| {
                    GraphRAGError::Embedding {
                        message: format!("Failed to load tokenizer: {}", e),
                    }
                })?;

                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[model_path], Dtype::F32, &device)
                        .map_err(|e| GraphRAGError::Embedding {
                            message: format!("Failed to load model weights: {}", e),
                        })?
                };

                let model = BertModel::load(vb, &config).map_err(|e| GraphRAGError::Embedding {
                    message: format!("Failed to load BERT model: {}", e),
                })?;

                self.model = Some(model);
                self.tokenizer = Some(tokenizer);
                self.device = Some(device);

                // Update dimensions based on loaded model config
                self.dimensions = config.hidden_size;
            }

            self.initialized = true;

            log::info!(
                "HuggingFace model '{}' initialized successfully (dims: {})",
                self.model_id,
                self.dimensions
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

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(GraphRAGError::Embedding {
                message: "HuggingFace embeddings not initialized. Call initialize() first"
                    .to_string(),
            });
        }

        #[cfg(feature = "neural-embeddings")]
        {
            let model = self.model.as_ref().unwrap();
            let tokenizer = self.tokenizer.as_ref().unwrap();
            let device = self.device.as_ref().unwrap();

            let tokens = tokenizer
                .encode(text, true)
                .map_err(|e| GraphRAGError::Embedding {
                    message: format!("Tokenization failed: {}", e),
                })?;

            let token_ids = Tensor::new(tokens.get_ids(), device)
                .map_err(|e| GraphRAGError::Embedding {
                    message: format!("Failed to create tensor: {}", e),
                })?
                .unsqueeze(0)
                .map_err(|_| GraphRAGError::Embedding {
                    message: "Failed to unsqueeze".to_string(),
                })?;

            let token_type_ids = token_ids
                .zeros_like()
                .map_err(|_| GraphRAGError::Embedding {
                    message: "Failed to create zero tensor".to_string(),
                })?;

            let embeddings = model.forward(&token_ids, &token_type_ids).map_err(|e| {
                GraphRAGError::Embedding {
                    message: format!("Model forward pass failed: {}", e),
                }
            })?;

            // Mean pooling
            let (_n_sentence, n_tokens, _hidden_size) =
                embeddings.dims3().map_err(|e| GraphRAGError::Embedding {
                    message: format!("Failed to get dimensions: {}", e),
                })?;

            let embeddings = (embeddings.sum(1).map_err(|e| GraphRAGError::Embedding {
                message: format!("Sum failed: {}", e),
            })? / (n_tokens as f64))
                .map_err(|e| GraphRAGError::Embedding {
                    message: format!("Division failed: {}", e),
                })?;

            let embeddings_vec = embeddings
                .squeeze(0)
                .map_err(|_| GraphRAGError::Embedding {
                    message: "Squeeze failed".to_string(),
                })?
                .to_vec1::<f32>()
                .map_err(|e| GraphRAGError::Embedding {
                    message: format!("Failed to convert to vec: {}", e),
                })?;

            Ok(embeddings_vec)
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
        let embeddings = HuggingFaceEmbeddings::new("sentence-transformers/all-MiniLM-L6-v2", None);

        assert_eq!(
            embeddings.model_id,
            "sentence-transformers/all-MiniLM-L6-v2"
        );
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

        let mut embeddings =
            HuggingFaceEmbeddings::new("sentence-transformers/all-MiniLM-L6-v2", None);

        let result = embeddings.initialize().await;
        assert!(result.is_ok(), "Failed to download model: {:?}", result);
        assert!(embeddings.is_available());
    }
}
