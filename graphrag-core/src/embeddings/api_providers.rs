//! API-based embedding providers (OpenAI, Voyage AI, Cohere, Jina AI, Mistral, etc.)
//!
//! This module provides embedding generation using external API services.
//! All providers implement the `EmbeddingProvider` trait for consistency.

use crate::core::error::{GraphRAGError, Result};
use crate::embeddings::{EmbeddingConfig, EmbeddingProvider, EmbeddingProviderType};

#[cfg(feature = "ureq")]
use ureq;

/// Generic HTTP-based embedding provider
pub struct HttpEmbeddingProvider {
    provider_type: EmbeddingProviderType,
    api_key: String,
    model: String,
    endpoint: String,
    dimensions: usize,

    #[cfg(feature = "ureq")]
    client: ureq::Agent,
}

impl HttpEmbeddingProvider {
    /// Create OpenAI embeddings provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = HttpEmbeddingProvider::openai(
    ///     "sk-...".to_string(),
    ///     "text-embedding-3-small".to_string()
    /// );
    /// ```
    pub fn openai(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-3-small" => 1536,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        };

        Self {
            provider_type: EmbeddingProviderType::OpenAI,
            api_key,
            model,
            endpoint: "https://api.openai.com/v1/embeddings".to_string(),
            dimensions,
            #[cfg(feature = "ureq")]
            client: ureq::Agent::new(),
        }
    }

    /// Create Voyage AI embeddings provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = HttpEmbeddingProvider::voyage_ai(
    ///     "pa-...".to_string(),
    ///     "voyage-3-large".to_string()
    /// );
    /// ```
    pub fn voyage_ai(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "voyage-3-large" => 1024,
            "voyage-3.5" => 1024,
            "voyage-3.5-lite" => 1024,
            "voyage-code-3" => 1024,
            "voyage-finance-2" => 1024,
            "voyage-law-2" => 1024,
            _ => 1024,
        };

        Self {
            provider_type: EmbeddingProviderType::VoyageAI,
            api_key,
            model,
            endpoint: "https://api.voyageai.com/v1/embeddings".to_string(),
            dimensions,
            #[cfg(feature = "ureq")]
            client: ureq::Agent::new(),
        }
    }

    /// Create Cohere embeddings provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = HttpEmbeddingProvider::cohere(
    ///     "...".to_string(),
    ///     "embed-english-v3.0".to_string()
    /// );
    /// ```
    pub fn cohere(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "embed-v4" | "embed-english-v3.0" | "embed-multilingual-v3.0" => 1024,
            "embed-english-light-v3.0" => 384,
            _ => 1024,
        };

        Self {
            provider_type: EmbeddingProviderType::Cohere,
            api_key,
            model,
            endpoint: "https://api.cohere.ai/v1/embed".to_string(),
            dimensions,
            #[cfg(feature = "ureq")]
            client: ureq::Agent::new(),
        }
    }

    /// Create Jina AI embeddings provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = HttpEmbeddingProvider::jina_ai(
    ///     "jina_...".to_string(),
    ///     "jina-embeddings-v3".to_string()
    /// );
    /// ```
    pub fn jina_ai(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "jina-embeddings-v4" => 1024,
            "jina-clip-v2" => 768,
            "jina-embeddings-v3" => 1024,
            _ => 1024,
        };

        Self {
            provider_type: EmbeddingProviderType::JinaAI,
            api_key,
            model,
            endpoint: "https://api.jina.ai/v1/embeddings".to_string(),
            dimensions,
            #[cfg(feature = "ureq")]
            client: ureq::Agent::new(),
        }
    }

    /// Create Mistral AI embeddings provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = HttpEmbeddingProvider::mistral(
    ///     "...".to_string(),
    ///     "mistral-embed".to_string()
    /// );
    /// ```
    pub fn mistral(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "mistral-embed" | "codestral-embed" => 1024,
            _ => 1024,
        };

        Self {
            provider_type: EmbeddingProviderType::Mistral,
            api_key,
            model,
            endpoint: "https://api.mistral.ai/v1/embeddings".to_string(),
            dimensions,
            #[cfg(feature = "ureq")]
            client: ureq::Agent::new(),
        }
    }

    /// Create Together AI embeddings provider
    ///
    /// # Example
    /// ```rust,ignore
    /// let provider = HttpEmbeddingProvider::together_ai(
    ///     "...".to_string(),
    ///     "BAAI/bge-large-en-v1.5".to_string()
    /// );
    /// ```
    pub fn together_ai(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "BAAI/bge-large-en-v1.5" | "WhereIsAI/UAE-Large-V1" => 1024,
            "BAAI/bge-base-en-v1.5" => 768,
            _ => 768,
        };

        Self {
            provider_type: EmbeddingProviderType::TogetherAI,
            api_key,
            model,
            endpoint: "https://api.together.xyz/v1/embeddings".to_string(),
            dimensions,
            #[cfg(feature = "ureq")]
            client: ureq::Agent::new(),
        }
    }

    /// Create provider from configuration
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self> {
        let api_key = config.api_key.clone().ok_or_else(|| {
            GraphRAGError::Embedding {
                message: format!("API key required for {} provider", config.provider),
            }
        })?;

        let provider = match config.provider {
            EmbeddingProviderType::OpenAI => Self::openai(api_key, config.model.clone()),
            EmbeddingProviderType::VoyageAI => Self::voyage_ai(api_key, config.model.clone()),
            EmbeddingProviderType::Cohere => Self::cohere(api_key, config.model.clone()),
            EmbeddingProviderType::JinaAI => Self::jina_ai(api_key, config.model.clone()),
            EmbeddingProviderType::Mistral => Self::mistral(api_key, config.model.clone()),
            EmbeddingProviderType::TogetherAI => Self::together_ai(api_key, config.model.clone()),
            _ => {
                return Err(GraphRAGError::Embedding {
                    message: format!("Unsupported API provider: {}", config.provider),
                })
            }
        };

        Ok(provider)
    }

    #[cfg(feature = "ureq")]
    fn make_request(&self, input: &str) -> Result<Vec<f32>> {
        // Build request body based on provider
        let request_body = match self.provider_type {
            EmbeddingProviderType::OpenAI => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "input": input,
                })
            }
            EmbeddingProviderType::VoyageAI => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "input": input,
                    "input_type": "document",
                })
            }
            EmbeddingProviderType::Cohere => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "texts": vec![input],
                    "input_type": "search_document",
                    "embedding_types": vec!["float"],
                })
            }
            EmbeddingProviderType::JinaAI | EmbeddingProviderType::Mistral | EmbeddingProviderType::TogetherAI => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "input": input,
                })
            }
            _ => {
                return Err(GraphRAGError::Embedding {
                    message: "Unsupported provider type".to_string(),
                })
            }
        };

        // Make HTTP request
        let response = self
            .client
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(request_body)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("HTTP request failed: {}", e),
            })?;

        // Parse response
        let json_response: serde_json::Value =
            response.into_json().map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to parse JSON response: {}", e),
            })?;

        // Extract embedding based on provider response format
        let embedding = match self.provider_type {
            EmbeddingProviderType::OpenAI
            | EmbeddingProviderType::VoyageAI
            | EmbeddingProviderType::JinaAI
            | EmbeddingProviderType::Mistral
            | EmbeddingProviderType::TogetherAI => {
                // OpenAI-compatible format: { "data": [{ "embedding": [...] }] }
                json_response["data"][0]["embedding"]
                    .as_array()
                    .ok_or_else(|| GraphRAGError::Embedding {
                        message: "Invalid response format: expected array".to_string(),
                    })?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            }
            EmbeddingProviderType::Cohere => {
                // Cohere format: { "embeddings": [[...]] }
                json_response["embeddings"][0]
                    .as_array()
                    .ok_or_else(|| GraphRAGError::Embedding {
                        message: "Invalid response format: expected array".to_string(),
                    })?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            }
            _ => vec![],
        };

        if embedding.is_empty() {
            return Err(GraphRAGError::Embedding {
                message: "No embedding returned from API".to_string(),
            });
        }

        Ok(embedding)
    }

    #[cfg(not(feature = "ureq"))]
    fn make_request(&self, _input: &str) -> Result<Vec<f32>> {
        Err(GraphRAGError::Embedding {
            message: "ureq feature required for HTTP-based embeddings".to_string(),
        })
    }

    /// Make batch embedding request for multiple texts
    #[cfg(feature = "ureq")]
    fn make_batch_request(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Build request body based on provider
        let request_body = match self.provider_type {
            EmbeddingProviderType::OpenAI => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "input": inputs,
                })
            }
            EmbeddingProviderType::VoyageAI => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "input": inputs,
                    "input_type": "document",
                })
            }
            EmbeddingProviderType::Cohere => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "texts": inputs,
                    "input_type": "search_document",
                    "embedding_types": vec!["float"],
                })
            }
            EmbeddingProviderType::JinaAI | EmbeddingProviderType::Mistral | EmbeddingProviderType::TogetherAI => {
                serde_json::json!({
                    "model": self.model.clone(),
                    "input": inputs,
                })
            }
            _ => {
                return Err(GraphRAGError::Embedding {
                    message: "Unsupported provider type for batch".to_string(),
                })
            }
        };

        // Make HTTP request
        let response = self
            .client
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(request_body)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Batch HTTP request failed: {}", e),
            })?;

        // Parse response
        let json_response: serde_json::Value =
            response.into_json().map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to parse batch JSON response: {}", e),
            })?;

        // Extract embeddings based on provider response format
        let embeddings = match self.provider_type {
            EmbeddingProviderType::OpenAI
            | EmbeddingProviderType::VoyageAI
            | EmbeddingProviderType::JinaAI
            | EmbeddingProviderType::Mistral
            | EmbeddingProviderType::TogetherAI => {
                // OpenAI-compatible format: { "data": [{ "embedding": [...] }, ...] }
                let data_array = json_response["data"]
                    .as_array()
                    .ok_or_else(|| GraphRAGError::Embedding {
                        message: "Invalid batch response format: expected data array".to_string(),
                    })?;

                data_array
                    .iter()
                    .map(|item| {
                        item["embedding"]
                            .as_array()
                            .ok_or_else(|| GraphRAGError::Embedding {
                                message: "Invalid embedding format in batch".to_string(),
                            })
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                                    .collect()
                            })
                    })
                    .collect::<Result<Vec<Vec<f32>>>>()?
            }
            EmbeddingProviderType::Cohere => {
                // Cohere format: { "embeddings": [[...], [...], ...] }
                let embeddings_array = json_response["embeddings"]
                    .as_array()
                    .ok_or_else(|| GraphRAGError::Embedding {
                        message: "Invalid Cohere batch response format".to_string(),
                    })?;

                embeddings_array
                    .iter()
                    .map(|emb| {
                        emb.as_array()
                            .ok_or_else(|| GraphRAGError::Embedding {
                                message: "Invalid embedding array in Cohere batch".to_string(),
                            })
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                                    .collect()
                            })
                    })
                    .collect::<Result<Vec<Vec<f32>>>>()?
            }
            _ => vec![],
        };

        if embeddings.is_empty() || embeddings.len() != inputs.len() {
            return Err(GraphRAGError::Embedding {
                message: format!(
                    "Batch embedding count mismatch: expected {}, got {}",
                    inputs.len(),
                    embeddings.len()
                ),
            });
        }

        Ok(embeddings)
    }

    #[cfg(not(feature = "ureq"))]
    fn make_batch_request(&self, _inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err(GraphRAGError::Embedding {
            message: "ureq feature required for batch embeddings".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for HttpEmbeddingProvider {
    async fn initialize(&mut self) -> Result<()> {
        // API providers don't need initialization
        Ok(())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.make_request(text)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Use batch API for providers that support it
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // For single text, use regular embed
        if texts.len() == 1 {
            return Ok(vec![self.embed(texts[0]).await?]);
        }

        #[cfg(feature = "ureq")]
        {
            // Try batch request for supported providers
            match self.make_batch_request(texts) {
                Ok(embeddings) => return Ok(embeddings),
                Err(_) => {
                    // Fallback to sequential requests if batch fails
                }
            }
        }

        // Fallback: sequential requests
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
        #[cfg(feature = "ureq")]
        {
            !self.api_key.is_empty()
        }

        #[cfg(not(feature = "ureq"))]
        {
            false
        }
    }

    fn provider_name(&self) -> &str {
        match self.provider_type {
            EmbeddingProviderType::OpenAI => "OpenAI",
            EmbeddingProviderType::VoyageAI => "Voyage AI",
            EmbeddingProviderType::Cohere => "Cohere",
            EmbeddingProviderType::JinaAI => "Jina AI",
            EmbeddingProviderType::Mistral => "Mistral AI",
            EmbeddingProviderType::TogetherAI => "Together AI",
            _ => "Unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider_creation() {
        let provider = HttpEmbeddingProvider::openai(
            "sk-test".to_string(),
            "text-embedding-3-small".to_string(),
        );

        assert_eq!(provider.provider_name(), "OpenAI");
        assert_eq!(provider.dimensions(), 1536);
        assert_eq!(provider.endpoint, "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn test_voyage_provider_creation() {
        let provider = HttpEmbeddingProvider::voyage_ai(
            "pa-test".to_string(),
            "voyage-3-large".to_string(),
        );

        assert_eq!(provider.provider_name(), "Voyage AI");
        assert_eq!(provider.dimensions(), 1024);
    }

    #[test]
    fn test_provider_from_config() {
        let config = EmbeddingConfig {
            provider: EmbeddingProviderType::OpenAI,
            model: "text-embedding-3-small".to_string(),
            api_key: Some("sk-test".to_string()),
            cache_dir: None,
            batch_size: 32,
        };

        let provider = HttpEmbeddingProvider::from_config(&config);
        assert!(provider.is_ok());

        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "OpenAI");
    }

    #[test]
    fn test_config_without_api_key_fails() {
        let config = EmbeddingConfig {
            provider: EmbeddingProviderType::OpenAI,
            model: "text-embedding-3-small".to_string(),
            api_key: None,
            cache_dir: None,
            batch_size: 32,
        };

        let result = HttpEmbeddingProvider::from_config(&config);
        assert!(result.is_err());
    }
}
