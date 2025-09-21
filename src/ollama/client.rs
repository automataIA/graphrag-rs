//! Ollama client wrapper for GraphRAG

use super::{OllamaConfig, OllamaError, OllamaResult};
use ollama_rs::Ollama;
use std::time::Duration;

/// Wrapper around ollama-rs client with GraphRAG-specific functionality
#[derive(Debug)]
pub struct OllamaClient {
    client: Ollama,
    config: OllamaConfig,
}

impl OllamaClient {
    /// Create new Ollama client
    pub fn new(config: OllamaConfig) -> OllamaResult<Self> {
        config.validate().map_err(OllamaError::ConfigError)?;

        let client = Ollama::new(config.host.clone(), config.port);

        Ok(Self { client, config })
    }

    /// Check if Ollama service is reachable
    pub async fn health_check(&self) -> OllamaResult<bool> {
        match self.client.list_local_models().await {
            Ok(_) => Ok(true),
            Err(e) => {
                println!("Ollama health check failed: {e}");
                Ok(false)
            }
        }
    }

    /// List available models
    pub async fn list_models(&self) -> OllamaResult<Vec<String>> {
        let models = self.client.list_local_models().await?;
        Ok(models.into_iter().map(|m| m.name).collect())
    }

    /// Check if a specific model is available
    pub async fn model_exists(&self, model_name: &str) -> OllamaResult<bool> {
        let models = self.list_models().await?;
        Ok(models.iter().any(|m| m.contains(model_name)))
    }

    /// Get embedding model name
    pub fn embedding_model(&self) -> &str {
        &self.config.embedding_model
    }

    /// Get chat model name
    pub fn chat_model(&self) -> &str {
        &self.config.chat_model
    }

    /// Get the underlying ollama-rs client
    pub fn inner(&self) -> &Ollama {
        &self.client
    }

    /// Get configuration
    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }

    /// Validate that required models are available
    pub async fn validate_models(&self) -> OllamaResult<()> {
        if !self.health_check().await? {
            return Err(OllamaError::ConnectionFailed(
                "Cannot reach Ollama service".to_string(),
            ));
        }

        // Check embedding model
        if !self.model_exists(&self.config.embedding_model).await? {
            return Err(OllamaError::ModelNotFound(
                self.config.embedding_model.clone(),
            ));
        }

        // Check chat model
        if !self.model_exists(&self.config.chat_model).await? {
            return Err(OllamaError::ModelNotFound(self.config.chat_model.clone()));
        }

        Ok(())
    }

    /// Create a timeout-wrapped future for Ollama operations
    pub async fn with_timeout<F, T>(&self, future: F) -> OllamaResult<T>
    where
        F: std::future::Future<Output = Result<T, ollama_rs::error::OllamaError>>,
    {
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);

        match tokio::time::timeout(timeout_duration, future).await {
            Ok(result) => result.map_err(OllamaError::from),
            Err(_) => Err(OllamaError::Timeout),
        }
    }

    /// Generate text response using the chat model
    pub async fn generate_response(&self, prompt: &str) -> OllamaResult<String> {
        use ollama_rs::generation::completion::request::GenerationRequest;

        let request = GenerationRequest::new(
            self.config.chat_model.clone(),
            prompt.to_string(),
        );

        let response = self.with_timeout(
            self.client.generate(request)
        ).await?;

        Ok(response.response)
    }
}

impl Clone for OllamaClient {
    fn clone(&self) -> Self {
        Self {
            client: Ollama::new(self.config.host.clone(), self.config.port),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = OllamaConfig::default();

        // Should fail when enabled but not validated
        let mut enabled_config = config.clone();
        enabled_config.enabled = true;

        let client = OllamaClient::new(enabled_config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = OllamaConfig {
            enabled: true,
            ..Default::default()
        };
        config.host.clear(); // Invalid empty host

        let result = OllamaClient::new(config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config).unwrap();

        // This will only succeed if Ollama is actually running
        let health = client.health_check().await;
        println!("Health check result: {health:?}");
        // Don't assert since Ollama might not be running in tests
    }
}
