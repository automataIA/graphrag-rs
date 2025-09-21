//! Async Ollama generation implementation demonstrating AsyncLanguageModel integration
//!
//! This module provides an async implementation of Ollama that integrates with the AsyncLanguageModel trait,
//! showing how to adapt existing async APIs to the new trait system.

use crate::core::traits::{AsyncLanguageModel, GenerationParams, ModelInfo, ModelUsageStats};
use crate::core::{GraphRAGError, Result};
use crate::ollama::{OllamaClient, OllamaConfig, OllamaError};
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::Stream;
use ollama_rs::generation::completion::request::GenerationRequest;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Async Ollama language model that implements AsyncLanguageModel
#[derive(Debug)]
pub struct AsyncOllamaGenerator {
    client: Arc<OllamaClient>,
    config: OllamaConfig,
    stats: Arc<AsyncOllamaStats>,
}

/// Statistics tracking for async Ollama operations
#[derive(Debug, Default)]
struct AsyncOllamaStats {
    total_requests: AtomicU64,
    total_tokens_processed: AtomicU64,
    total_response_time: Arc<RwLock<Duration>>,
    error_count: AtomicU64,
    successful_requests: AtomicU64,
}

impl AsyncOllamaGenerator {
    /// Create a new async Ollama generator
    pub async fn new(config: OllamaConfig) -> Result<Self> {
        let client =
            Arc::new(
                OllamaClient::new(config.clone()).map_err(|e| GraphRAGError::Generation {
                    message: format!("Failed to create Ollama client: {e}"),
                })?,
            );

        // Validate that the client can connect and required models are available
        if config.enabled {
            client
                .validate_models()
                .await
                .map_err(|e| GraphRAGError::Generation {
                    message: format!("Ollama model validation failed: {e}"),
                })?;
        }

        Ok(Self {
            client,
            config,
            stats: Arc::new(AsyncOllamaStats::default()),
        })
    }

    /// Create without model validation (useful for testing)
    pub fn new_unchecked(config: OllamaConfig) -> Result<Self> {
        let client =
            Arc::new(
                OllamaClient::new(config.clone()).map_err(|e| GraphRAGError::Generation {
                    message: format!("Failed to create Ollama client: {e}"),
                })?,
            );

        Ok(Self {
            client,
            config,
            stats: Arc::new(AsyncOllamaStats::default()),
        })
    }

    /// Get the underlying Ollama client
    pub fn client(&self) -> &OllamaClient {
        &self.client
    }

    /// Get current configuration
    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }

    /// Update statistics after a request
    async fn update_stats(&self, tokens: usize, response_time: Duration, is_error: bool) {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        if is_error {
            self.stats.error_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats
                .successful_requests
                .fetch_add(1, Ordering::Relaxed);
            self.stats
                .total_tokens_processed
                .fetch_add(tokens as u64, Ordering::Relaxed);
        }

        let mut total_time = self.stats.total_response_time.write().await;
        *total_time += response_time;
    }

    /// Convert OllamaError to GraphRAGError
    fn convert_error(error: OllamaError) -> GraphRAGError {
        match error {
            OllamaError::ConnectionFailed(msg) => GraphRAGError::Generation {
                message: format!("Ollama connection failed: {msg}"),
            },
            OllamaError::ModelNotFound(model) => GraphRAGError::Generation {
                message: format!("Ollama model not found: {model}"),
            },
            OllamaError::Timeout | OllamaError::RequestTimeout => GraphRAGError::Generation {
                message: "Ollama request timed out".to_string(),
            },
            OllamaError::ApiError(msg) => GraphRAGError::Generation {
                message: format!("Ollama API error: {msg}"),
            },
            OllamaError::JsonError(msg) => GraphRAGError::Generation {
                message: format!("Ollama JSON error: {msg}"),
            },
            OllamaError::ConfigError(msg) => GraphRAGError::Generation {
                message: format!("Ollama configuration error: {msg}"),
            },
            OllamaError::GenerationError(msg) => GraphRAGError::Generation {
                message: format!("Ollama generation error: {msg}"),
            },
            OllamaError::DimensionMismatch { expected, got } => GraphRAGError::Generation {
                message: format!("Embedding dimension mismatch: expected {expected}, got {got}"),
            },
        }
    }

    /// Generate completion with enhanced error handling and metrics
    async fn generate_with_metrics(&self, prompt: &str, model: Option<&str>) -> Result<String> {
        let start_time = Instant::now();
        let model_name = model.unwrap_or(&self.config.chat_model);

        let request = GenerationRequest::new(model_name.to_string(), prompt.to_string());

        let result = self
            .client
            .with_timeout(self.client.inner().generate(request))
            .await
            .map_err(Self::convert_error);

        let response_time = start_time.elapsed();
        let tokens = self.estimate_tokens_sync(prompt);

        self.update_stats(tokens, response_time, result.is_err())
            .await;

        match result {
            Ok(response) => Ok(response.response),
            Err(e) => {
                eprintln!("Ollama generation error: {e}");
                Err(e)
            }
        }
    }

    /// Estimate tokens synchronously for metrics
    fn estimate_tokens_sync(&self, text: &str) -> usize {
        // Simple estimation: ~4 characters per token
        text.len() / 4
    }

    /// Process multiple prompts with concurrency control
    async fn process_batch_internal(
        &self,
        prompts: &[&str],
        max_concurrent: usize,
    ) -> Result<Vec<String>> {
        use futures::stream::{FuturesUnordered, StreamExt};

        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        if max_concurrent <= 1 {
            // Sequential processing
            let mut results = Vec::with_capacity(prompts.len());
            for prompt in prompts {
                let result = self.generate_with_metrics(prompt, None).await?;
                results.push(result);
            }
            return Ok(results);
        }

        // Concurrent processing with rate limiting
        let mut futures: FuturesUnordered<BoxFuture<Result<String>>> = FuturesUnordered::new();
        let mut results = Vec::with_capacity(prompts.len());
        let mut pending = 0;
        let mut prompt_iter = prompts.iter();

        // Start initial batch
        while pending < max_concurrent {
            if let Some(prompt) = prompt_iter.next() {
                let self_clone = self.clone();
                let prompt_owned = prompt.to_string();
                futures.push(Box::pin(async move {
                    self_clone.generate_with_metrics(&prompt_owned, None).await
                }));
                pending += 1;
            } else {
                break;
            }
        }

        // Process remaining prompts as previous ones complete
        while let Some(result) = futures.next().await {
            results.push(result?);
            pending -= 1;

            // Start next prompt if available
            if let Some(prompt) = prompt_iter.next() {
                let self_clone = self.clone();
                let prompt_owned = prompt.to_string();
                let future: BoxFuture<Result<String>> =
                    Box::pin(
                        async move { self_clone.generate_with_metrics(&prompt_owned, None).await },
                    );
                futures.push(future);
                pending += 1;
            }
        }

        Ok(results)
    }
}

#[async_trait]
impl AsyncLanguageModel for AsyncOllamaGenerator {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        if !self.config.enabled {
            return Err(GraphRAGError::Generation {
                message: "Ollama is disabled in configuration".to_string(),
            });
        }

        self.generate_with_metrics(prompt, None).await
    }

    async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        if !self.config.enabled {
            return Err(GraphRAGError::Generation {
                message: "Ollama is disabled in configuration".to_string(),
            });
        }

        // For now, we'll use basic completion and ignore parameters
        // In a full implementation, you would map GenerationParams to Ollama's request parameters
        let _ = params; // Suppress unused warning
        self.generate_with_metrics(prompt, None).await
    }

    async fn complete_batch(&self, prompts: &[&str]) -> Result<Vec<String>> {
        if !self.config.enabled {
            return Err(GraphRAGError::Generation {
                message: "Ollama is disabled in configuration".to_string(),
            });
        }

        self.process_batch_internal(prompts, 3).await // Default concurrency of 3
    }

    async fn complete_batch_concurrent(
        &self,
        prompts: &[&str],
        max_concurrent: usize,
    ) -> Result<Vec<String>> {
        if !self.config.enabled {
            return Err(GraphRAGError::Generation {
                message: "Ollama is disabled in configuration".to_string(),
            });
        }

        self.process_batch_internal(prompts, max_concurrent).await
    }

    async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        if !self.config.enabled {
            return Err(GraphRAGError::Generation {
                message: "Ollama is disabled in configuration".to_string(),
            });
        }

        // For now, convert regular completion to stream
        // In a full implementation, you would use Ollama's streaming API
        let result = self.complete(prompt).await?;
        let stream = futures::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }

    async fn is_available(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        self.client.health_check().await.unwrap_or_default()
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: format!("Ollama-{}", self.config.chat_model),
            version: None, // Ollama doesn't provide version info easily
            max_context_length: Some(4096), // Default, should be configurable
            supports_streaming: true,
        }
    }

    async fn health_check(&self) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        self.client
            .health_check()
            .await
            .map_err(Self::convert_error)
    }

    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        let total_requests = self.stats.total_requests.load(Ordering::Relaxed);
        let total_tokens = self.stats.total_tokens_processed.load(Ordering::Relaxed);
        let error_count = self.stats.error_count.load(Ordering::Relaxed);
        let total_time = *self.stats.total_response_time.read().await;

        let average_response_time_ms = if total_requests > 0 {
            total_time.as_millis() as f64 / total_requests as f64
        } else {
            0.0
        };

        let error_rate = if total_requests > 0 {
            error_count as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(ModelUsageStats {
            total_requests,
            total_tokens_processed: total_tokens,
            average_response_time_ms,
            error_rate,
        })
    }

    async fn estimate_tokens(&self, prompt: &str) -> Result<usize> {
        Ok(self.estimate_tokens_sync(prompt))
    }
}

impl Clone for AsyncOllamaGenerator {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
        }
    }
}

/// Builder for AsyncOllamaGenerator with validation and fallback options
pub struct AsyncOllamaGeneratorBuilder {
    config: OllamaConfig,
    validate_on_build: bool,
    fallback_to_mock: bool,
}

impl AsyncOllamaGeneratorBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: OllamaConfig::default(),
            validate_on_build: true,
            fallback_to_mock: false,
        }
    }

    /// Create builder with custom configuration
    pub fn with_config(config: OllamaConfig) -> Self {
        Self {
            config,
            validate_on_build: true,
            fallback_to_mock: false,
        }
    }

    /// Set whether to validate models on build
    pub fn validate_on_build(mut self, validate: bool) -> Self {
        self.validate_on_build = validate;
        self
    }

    /// Set whether to fallback to mock LLM if Ollama is unavailable
    pub fn fallback_to_mock(mut self, fallback: bool) -> Self {
        self.fallback_to_mock = fallback;
        self
    }

    /// Set the chat model
    pub fn chat_model(mut self, model: String) -> Self {
        self.config.chat_model = model;
        self
    }

    /// Set the host
    pub fn host(mut self, host: String) -> Self {
        self.config.host = host;
        self
    }

    /// Set the port
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Build the AsyncOllamaGenerator
    pub async fn build(
        self,
    ) -> Result<Box<dyn AsyncLanguageModel<Error = GraphRAGError> + Send + Sync>> {
        if self.validate_on_build {
            match AsyncOllamaGenerator::new(self.config.clone()).await {
                Ok(generator) => Ok(Box::new(generator)),
                Err(e) => {
                    if self.fallback_to_mock {
                        eprintln!("Ollama unavailable, falling back to mock LLM: {e}");
                        #[cfg(feature = "async-traits")]
                        {
                            let mock_llm = crate::generation::async_mock_llm::AsyncMockLLM::new()
                                .await
                                .map_err(|e| GraphRAGError::Generation {
                                    message: format!("Failed to create fallback mock LLM: {e}"),
                                })?;
                            Ok(Box::new(mock_llm))
                        }
                        #[cfg(not(feature = "async-traits"))]
                        Err(e)
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            let generator = AsyncOllamaGenerator::new_unchecked(self.config)?;
            Ok(Box::new(generator))
        }
    }
}

impl Default for AsyncOllamaGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_ollama_generator_builder() {
        let builder = AsyncOllamaGeneratorBuilder::new()
            .validate_on_build(false)
            .fallback_to_mock(true);

        // This should succeed even if Ollama is not available
        let result = builder.build().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_ollama_generator_unchecked() {
        let config = OllamaConfig {
            enabled: false, // Disabled so we don't need actual Ollama
            ..Default::default()
        };

        let result = AsyncOllamaGenerator::new_unchecked(config);
        assert!(result.is_ok());

        let generator = result.unwrap();
        let available = generator.is_available().await;
        assert!(!available); // Should be false since it's disabled
    }

    #[tokio::test]
    async fn test_disabled_ollama_operations() {
        let config = OllamaConfig {
            enabled: false,
            ..Default::default()
        };

        let generator = AsyncOllamaGenerator::new_unchecked(config).unwrap();

        // All operations should fail with disabled error
        let complete_result = generator.complete("test").await;
        assert!(complete_result.is_err());

        let batch_result = generator.complete_batch(&["test"]).await;
        assert!(batch_result.is_err());

        let available = generator.is_available().await;
        assert!(!available);
    }

    #[tokio::test]
    async fn test_model_info() {
        let config = OllamaConfig {
            enabled: false,
            chat_model: "test-model".to_string(),
            ..Default::default()
        };

        let generator = AsyncOllamaGenerator::new_unchecked(config).unwrap();
        let info = generator.model_info().await;

        assert!(info.name.contains("test-model"));
        assert!(info.supports_streaming);
    }

    #[tokio::test]
    async fn test_usage_stats() {
        let config = OllamaConfig {
            enabled: false,
            ..Default::default()
        };

        let generator = AsyncOllamaGenerator::new_unchecked(config).unwrap();
        let stats = generator.get_usage_stats().await.unwrap();

        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_tokens_processed, 0);
        assert_eq!(stats.error_rate, 0.0);
    }

    #[tokio::test]
    async fn test_token_estimation() {
        let config = OllamaConfig {
            enabled: false,
            ..Default::default()
        };

        let generator = AsyncOllamaGenerator::new_unchecked(config).unwrap();
        let tokens = generator
            .estimate_tokens("This is a test prompt")
            .await
            .unwrap();
        assert!(tokens > 0);
    }
}
