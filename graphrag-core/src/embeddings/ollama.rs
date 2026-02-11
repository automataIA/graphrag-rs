//! Ollama embedding provider
//!
//! This module provides embedding generation using local Ollama models.

use crate::core::error::{GraphRAGError, Result};
use crate::embeddings::EmbeddingProvider;
use ollama_rs::Ollama;

/// Ollama embedding provider
pub struct OllamaEmbeddings {
    model: String,
    client: Ollama,
    dimensions: usize,
}

impl OllamaEmbeddings {
    /// Create a new Ollama embeddings provider
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            client: Ollama::default(),
            dimensions: 1024, // Default assume widely used models like nomic-embed-text
        }
    }

    /// Set the expected dimensions for this model
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OllamaEmbeddings {
    async fn initialize(&mut self) -> Result<()> {
        // Ping Ollama to check availability
        // Since ollama-rs doesn't have a direct ping, we can try listing models
        match self.client.list_local_models().await {
            Ok(_) => Ok(()),
            Err(e) => Err(GraphRAGError::Embedding {
                message: format!("Failed to connect to Ollama: {}", e),
            }),
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        use ollama_rs::generation::embeddings::request::EmbeddingsInput;

        let embeddings = self
            .client
            .generate_embeddings(
                ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest::new(
                    self.model.clone(),
                    EmbeddingsInput::Single(text.to_string()),
                ),
            )
            .await
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Ollama embedding generation failed: {}", e),
            })?;

        // The embeddings field is a Vec<Vec<f64>>, we need to get the first one
        let embedding: Vec<f32> = embeddings
            .embeddings
            .first()
            .ok_or_else(|| GraphRAGError::Embedding {
                message: "No embeddings returned from Ollama".to_string(),
            })?
            .iter()
            .map(|&x| x as f32)
            .collect();
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        // Ollama currently processes one by one in the API wrapper usually,
        // but we can loop here.
        for text in texts {
            results.push(self.embed(text).await?);
        }

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        // We can't easily check sync availability without IO, assume yes if init passed
        true
    }

    fn provider_name(&self) -> &str {
        "Ollama"
    }
}
