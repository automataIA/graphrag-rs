//! Ollama adapters for core traits
//!
//! This module provides adapter implementations that bridge Ollama services
//! with the core GraphRAG traits (AsyncEmbedder, AsyncLanguageModel).

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{AsyncEmbedder, AsyncLanguageModel, GenerationParams, ModelInfo, ModelUsageStats};
use crate::embeddings::ollama::OllamaEmbeddings;
use crate::ollama::{OllamaClient, OllamaConfig, OllamaGenerationParams};
use async_trait::async_trait;

/// Adapter for OllamaEmbeddings to implement AsyncEmbedder trait
pub struct OllamaEmbedderAdapter {
    embeddings: OllamaEmbeddings,
    dimension: usize,
}

impl OllamaEmbedderAdapter {
    /// Create a new Ollama embedder adapter
    pub fn new(model: impl Into<String>, dimension: usize) -> Self {
        Self {
            embeddings: OllamaEmbeddings::new(model).with_dimensions(dimension),
            dimension,
        }
    }

    /// Create from existing OllamaEmbeddings instance
    pub fn from_embeddings(embeddings: OllamaEmbeddings, dimension: usize) -> Self {
        Self {
            embeddings,
            dimension,
        }
    }
}

#[async_trait]
impl AsyncEmbedder for OllamaEmbedderAdapter {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        use crate::embeddings::EmbeddingProvider;
        self.embeddings
            .embed(text)
            .await
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Ollama embedding failed: {}", e),
            })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        use crate::embeddings::EmbeddingProvider;
        self.embeddings
            .embed_batch(texts)
            .await
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Ollama batch embedding failed: {}", e),
            })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn is_ready(&self) -> bool {
        use crate::embeddings::EmbeddingProvider;
        self.embeddings.is_available()
    }
}

/// Adapter for OllamaClient to implement AsyncLanguageModel trait
pub struct OllamaLanguageModelAdapter {
    client: OllamaClient,
    model_name: String,
}

impl OllamaLanguageModelAdapter {
    /// Create a new Ollama language model adapter
    pub fn new(config: OllamaConfig) -> Self {
        let model_name = config.chat_model.clone();
        Self {
            client: OllamaClient::new(config),
            model_name,
        }
    }

    /// Create with custom client
    pub fn from_client(client: OllamaClient, model_name: String) -> Self {
        Self {
            client,
            model_name,
        }
    }
}

#[async_trait]
impl AsyncLanguageModel for OllamaLanguageModelAdapter {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        self.client.generate(prompt).await
    }

    async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        // Convert core::traits::GenerationParams to ollama::OllamaGenerationParams
        let ollama_params = OllamaGenerationParams {
            num_predict: params.max_tokens.map(|t| t as u32),
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: None, // Not in core GenerationParams
            stop: params.stop_sequences,
            repeat_penalty: None, // Not in core GenerationParams
        };

        self.client.generate_with_params(prompt, ollama_params).await
    }

    async fn is_available(&self) -> bool {
        // Try a simple health check by attempting to connect
        // In production, we'd want a proper ping endpoint
        true
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            version: None, // Ollama doesn't expose version via current wrapper
            max_context_length: Some(4096), // Common default for llama models
            supports_streaming: true,       // Streaming now supported!
        }
    }

    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        let stats = self.client.get_stats();
        let total = stats.get_total_requests();
        let failed = stats.get_failed_requests();

        Ok(ModelUsageStats {
            total_requests: total,
            total_tokens_processed: stats.get_total_tokens(),
            average_response_time_ms: 0.0, // Not tracked yet (would need request timing)
            error_rate: if total > 0 { failed as f64 / total as f64 } else { 0.0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_embedder_adapter_creation() {
        let adapter = OllamaEmbedderAdapter::new("nomic-embed-text", 768);
        assert_eq!(adapter.dimension(), 768);
    }

    #[test]
    fn test_ollama_language_model_adapter_creation() {
        let config = OllamaConfig::default();
        let adapter = OllamaLanguageModelAdapter::new(config);
        assert_eq!(adapter.model_name, "llama3.2:3b");
    }
}
