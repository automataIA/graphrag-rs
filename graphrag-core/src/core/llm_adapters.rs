//! LLM adapters for core traits
//!
//! Bridges the OpenAI-compatible [`LlmClient`] and the llama.cpp e5
//! embedding provider with the core GraphRAG traits
//! ([`AsyncEmbedder`], [`AsyncLanguageModel`]).

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{
    AsyncEmbedder, AsyncLanguageModel, GenerationParams, ModelInfo, ModelUsageStats,
};
use crate::llama_cpp::{E5Intent, LlamaCppEmbeddingProvider};
use crate::llm::{LlmClient, LlmConfig, LlmGenerationParams};
use async_trait::async_trait;
use std::sync::Arc;

/// Adapter exposing the llama.cpp multilingual-e5 provider as [`AsyncEmbedder`].
///
/// Texts are embedded with the `passage:` intent (indexing side).
pub struct LlamaCppEmbedderAdapter {
    provider: Arc<LlamaCppEmbeddingProvider>,
}

impl LlamaCppEmbedderAdapter {
    /// Wrap an existing provider.
    pub fn new(provider: LlamaCppEmbeddingProvider) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }

    async fn run<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&LlamaCppEmbeddingProvider) -> Result<T> + Send + 'static,
    {
        let provider = Arc::clone(&self.provider);
        tokio::task::spawn_blocking(move || f(&provider))
            .await
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("embedding worker failed: {e}"),
            })?
    }
}

#[async_trait]
impl AsyncEmbedder for LlamaCppEmbedderAdapter {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.to_string();
        self.run(move |p| p.embed(&text, E5Intent::Document)).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        self.run(move |p| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            p.embed_batch(&refs, E5Intent::Document)
        })
        .await
    }

    fn dimension(&self) -> usize {
        self.provider.dimensions()
    }

    async fn is_ready(&self) -> bool {
        true
    }
}

/// Adapter for [`LlmClient`] implementing [`AsyncLanguageModel`].
pub struct LlmLanguageModelAdapter {
    client: LlmClient,
    model_name: String,
}

impl LlmLanguageModelAdapter {
    /// Create a new adapter from configuration
    pub fn new(config: LlmConfig) -> Self {
        let model_name = config.chat_model.clone();
        Self {
            client: LlmClient::new(config),
            model_name,
        }
    }

    /// Create with custom client
    pub fn from_client(client: LlmClient, model_name: String) -> Self {
        Self { client, model_name }
    }
}

#[async_trait]
impl AsyncLanguageModel for LlmLanguageModelAdapter {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        self.client.generate(prompt).await
    }

    async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        let llm_params = LlmGenerationParams {
            max_tokens: params.max_tokens.map(|t| t as u32),
            temperature: params.temperature,
            top_p: params.top_p,
            stop: params.stop_sequences,
            ..Default::default()
        };
        self.client.generate_with_params(prompt, llm_params).await
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            version: None,
            max_context_length: None,
            supports_streaming: true,
        }
    }

    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        let stats = self.client.get_stats();
        let total = stats.get_total_requests();
        let failed = stats.get_failed_requests();

        Ok(ModelUsageStats {
            total_requests: total,
            total_tokens_processed: stats.get_total_tokens(),
            average_response_time_ms: 0.0,
            error_rate: if total > 0 {
                failed as f64 / total as f64
            } else {
                0.0
            },
        })
    }
}
