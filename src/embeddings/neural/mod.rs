//! Neural embedding implementations using sentence transformers

pub mod transformer;
pub mod models;
pub mod tokenizer;
pub mod cache;

pub use transformer::{SentenceTransformer, PoolingStrategy as TransformerPoolingStrategy};
pub use models::{ModelManager, PretrainedModel, ModelInfo};
pub use tokenizer::{MultiThreadTokenizer, TokenizedInput};
pub use cache::{EmbeddingCache, CacheEntry};

use crate::core::{Result, GraphRAGError};
use crate::embeddings::{Device, PoolingStrategy};

/// Neural embedding provider implementing sentence transformers
pub struct NeuralEmbedder {
    transformer: Option<SentenceTransformer>,
    model_manager: ModelManager,
    cache: EmbeddingCache,
    config: NeuralConfig,
}

#[derive(Debug, Clone)]
pub struct NeuralConfig {
    pub model_name: String,
    pub device: Device,
    pub max_length: usize,
    pub pooling: PoolingStrategy,
    pub cache_size: usize,
    pub batch_size: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            device: Device::Auto,
            max_length: 512,
            pooling: PoolingStrategy::Mean,
            cache_size: 10000,
            batch_size: 32,
        }
    }
}

impl NeuralEmbedder {
    pub fn new(config: NeuralConfig) -> Result<Self> {
        let model_manager = ModelManager::new()?;
        let cache = EmbeddingCache::new(config.cache_size);

        Ok(Self {
            transformer: None,
            model_manager,
            cache,
            config,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        // Try to load the model
        match self.load_model().await {
            Ok(transformer) => {
                self.transformer = Some(transformer);
                println!("✅ Neural embedder initialized with model: {}", self.config.model_name);
                Ok(())
            }
            Err(e) => {
                println!("⚠️  Failed to initialize neural embedder: {e}");
                Err(e)
            }
        }
    }

    async fn load_model(&mut self) -> Result<SentenceTransformer> {
        // Check if model exists locally
        if !self.model_manager.model_exists(&self.config.model_name)? {
            println!("📥 Downloading model: {}", self.config.model_name);
            self.model_manager.download_model(&self.config.model_name).await?;
        }

        // Load the transformer
        let transformer = SentenceTransformer::load(
            &self.config.model_name,
            &self.config.device,
            self.config.max_length,
            &self.config.pooling,
        )?;

        Ok(transformer)
    }

    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.embedding);
        }

        // Ensure transformer is loaded
        if self.transformer.is_none() {
            self.initialize().await?;
        }

        // Generate embedding
        let embedding = if let Some(ref transformer) = self.transformer {
            transformer.encode(&[text])?
                .into_iter()
                .next()
                .ok_or_else(|| GraphRAGError::VectorSearch {
                    message: "Failed to generate embedding".to_string(),
                })?
        } else {
            return Err(GraphRAGError::VectorSearch {
                message: "Neural transformer not initialized".to_string(),
            });
        };

        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }

    pub async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            if let Some(cached) = self.cache.get(text) {
                results.push(Some(cached.embedding));
            } else {
                results.push(None);
                uncached_texts.push(*text);
                uncached_indices.push(i);
            }
        }

        // Process uncached texts
        if !uncached_texts.is_empty() {
            // Ensure transformer is loaded
            if self.transformer.is_none() {
                self.initialize().await?;
            }

            if let Some(ref transformer) = self.transformer {
                let embeddings = transformer.encode(&uncached_texts)?;

                // Insert into cache and results
                for (text, embedding) in uncached_texts.iter().zip(embeddings.iter()) {
                    self.cache.insert(text.to_string(), embedding.clone());
                }

                // Fill in the results
                for (local_idx, global_idx) in uncached_indices.iter().enumerate() {
                    results[*global_idx] = Some(embeddings[local_idx].clone());
                }
            } else {
                return Err(GraphRAGError::VectorSearch {
                    message: "Neural transformer not initialized".to_string(),
                });
            }
        }

        // Convert Option<Vec<f32>> to Vec<f32>
        results
            .into_iter()
            .map(|opt| opt.ok_or_else(|| GraphRAGError::VectorSearch {
                message: "Failed to generate embedding".to_string(),
            }))
            .collect()
    }

    pub fn get_cache_stats(&self) -> (usize, usize) {
        self.cache.stats()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn is_initialized(&self) -> bool {
        self.transformer.is_some()
    }

    pub fn get_model_info(&self) -> Option<ModelInfo> {
        self.model_manager.get_model_info(&self.config.model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_neural_embedder_creation() {
        let config = NeuralConfig::default();
        let embedder = NeuralEmbedder::new(config);
        assert!(embedder.is_ok());
    }

    #[tokio::test]
    async fn test_neural_embedder_fallback() {
        // Test that embedder gracefully handles missing models
        let config = NeuralConfig {
            model_name: "non-existent-model".to_string(),
            ..Default::default()
        };

        let mut embedder = NeuralEmbedder::new(config).unwrap();
        let result = embedder.embed("test text").await;

        // Should fail gracefully
        assert!(result.is_err());
    }
}