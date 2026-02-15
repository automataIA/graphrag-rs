//! Ollama LLM integration
//!
//! This module provides integration with Ollama for local LLM inference.

use crate::core::{GraphRAGError, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Generation parameters for Ollama requests
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaGenerationParams {
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    /// Temperature for sampling (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p nucleus sampling threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Repeat penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
}

impl Default for OllamaGenerationParams {
    fn default() -> Self {
        Self {
            num_predict: Some(2000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            stop: None,
            repeat_penalty: Some(1.1),
        }
    }
}

/// Usage statistics for Ollama client
#[derive(Debug, Clone, Default)]
pub struct OllamaUsageStats {
    /// Total number of requests
    pub total_requests: Arc<AtomicU64>,
    /// Total number of successful requests
    pub successful_requests: Arc<AtomicU64>,
    /// Total number of failed requests
    pub failed_requests: Arc<AtomicU64>,
    /// Total tokens generated (approximate)
    pub total_tokens: Arc<AtomicU64>,
}

impl OllamaUsageStats {
    /// Create new usage statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request
    pub fn record_success(&self, tokens: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total requests
    pub fn get_total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get successful requests
    pub fn get_successful_requests(&self) -> u64 {
        self.successful_requests.load(Ordering::Relaxed)
    }

    /// Get failed requests
    pub fn get_failed_requests(&self) -> u64 {
        self.failed_requests.load(Ordering::Relaxed)
    }

    /// Get total tokens
    pub fn get_total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Get success rate (0.0 - 1.0)
    pub fn get_success_rate(&self) -> f64 {
        let total = self.get_total_requests();
        if total == 0 {
            return 0.0;
        }
        self.get_successful_requests() as f64 / total as f64
    }
}

/// Ollama configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaConfig {
    /// Enable Ollama integration
    pub enabled: bool,
    /// Ollama host URL
    pub host: String,
    /// Ollama port
    pub port: u16,
    /// Model for embeddings
    pub embedding_model: String,
    /// Model for chat/generation
    pub chat_model: String,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Fallback to hash-based IDs on error
    pub fallback_to_hash: bool,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for generation (0.0 - 1.0)
    pub temperature: Option<f32>,
    /// Enable model caching
    pub enable_caching: bool,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "http://localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "llama3.2:3b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            fallback_to_hash: true,
            max_tokens: Some(2000),
            temperature: Some(0.7),
            enable_caching: true,
        }
    }
}

/// Ollama client for LLM inference
#[derive(Clone)]
pub struct OllamaClient {
    config: OllamaConfig,
    #[cfg(feature = "ureq")]
    client: ureq::Agent,
    /// Usage statistics
    stats: OllamaUsageStats,
    /// Response cache (prompt -> response)
    #[cfg(feature = "dashmap")]
    cache: Arc<dashmap::DashMap<String, String>>,
}

impl std::fmt::Debug for OllamaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaClient")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config: config.clone(),
            #[cfg(feature = "ureq")]
            client: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(config.timeout_seconds))
                .build(),
            stats: OllamaUsageStats::new(),
            #[cfg(feature = "dashmap")]
            cache: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Get usage statistics
    pub fn get_stats(&self) -> &OllamaUsageStats {
        &self.stats
    }

    /// Clear the cache
    #[cfg(feature = "dashmap")]
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get cache size
    #[cfg(feature = "dashmap")]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Generate text completion using Ollama API
    #[cfg(feature = "ureq")]
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        // Check cache first if enabled
        #[cfg(feature = "dashmap")]
        {
            if self.config.enable_caching {
                if let Some(cached_response) = self.cache.get(prompt) {
                    #[cfg(feature = "tracing")]
                    tracing::debug!("Cache hit for prompt (length: {})", prompt.len());
                    return Ok(cached_response.clone());
                }
            }
        }

        // Use default parameters
        let params = OllamaGenerationParams {
            num_predict: self.config.max_tokens,
            temperature: self.config.temperature,
            ..Default::default()
        };

        self.generate_with_params(prompt, params).await
    }

    /// Generate text completion with custom parameters
    #[cfg(feature = "ureq")]
    pub async fn generate_with_params(&self, prompt: &str, params: OllamaGenerationParams) -> Result<String> {
        let endpoint = format!("{}:{}/api/generate", self.config.host, self.config.port);

        let mut request_body = serde_json::json!({
            "model": self.config.chat_model,
            "prompt": prompt,
            "stream": false,
        });

        // Add custom parameters as options
        let options = serde_json::to_value(&params).map_err(|e| GraphRAGError::Generation {
            message: format!("Failed to serialize generation params: {}", e),
        })?;

        if !options.as_object().unwrap().is_empty() {
            request_body["options"] = options;
        }

        // Make HTTP request with retry logic
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match self.client
                .post(&endpoint)
                .set("Content-Type", "application/json")
                .send_json(&request_body)
            {
                Ok(response) => {
                    let json_response: serde_json::Value = response
                        .into_json()
                        .map_err(|e| GraphRAGError::Generation {
                            message: format!("Failed to parse JSON response: {}", e),
                        })?;

                    // Extract response text
                    if let Some(response_text) = json_response["response"].as_str() {
                        let response_string = response_text.to_string();

                        // Estimate tokens (rough: ~4 chars per token)
                        let estimated_tokens = (prompt.len() + response_string.len()) / 4;
                        self.stats.record_success(estimated_tokens as u64);

                        // Cache the response if enabled
                        #[cfg(feature = "dashmap")]
                        {
                            if self.config.enable_caching {
                                self.cache.insert(prompt.to_string(), response_string.clone());

                                #[cfg(feature = "tracing")]
                                tracing::debug!("Cached response for prompt (length: {})", prompt.len());
                            }
                        }

                        return Ok(response_string);
                    } else {
                        self.stats.record_failure();
                        return Err(GraphRAGError::Generation {
                            message: format!("Invalid response format: {:?}", json_response),
                        });
                    }
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("Ollama API request failed (attempt {}): {}", attempt, e);
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        // Wait before retry (exponential backoff)
                        tokio::time::sleep(std::time::Duration::from_millis(100 * attempt as u64)).await;
                    }
                }
            }
        }

        self.stats.record_failure();
        Err(GraphRAGError::Generation {
            message: format!("Ollama API failed after {} retries: {:?}",
                           self.config.max_retries, last_error),
        })
    }

    /// Generate streaming completion
    ///
    /// Returns a channel receiver that yields tokens as they are generated.
    /// This enables real-time display of generation progress.
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::ollama::{OllamaClient, OllamaConfig};
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let client = OllamaClient::new(OllamaConfig::default());
    /// let mut rx = client.generate_streaming("Write a story").await?;
    ///
    /// while let Some(token) = rx.recv().await {
    ///     print!("{}", token);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(feature = "ureq", feature = "tokio"))]
    pub async fn generate_streaming(&self, prompt: &str) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let endpoint = format!("{}:{}/api/generate", self.config.host, self.config.port);

        let params = OllamaGenerationParams {
            num_predict: self.config.max_tokens,
            temperature: self.config.temperature,
            ..Default::default()
        };

        let mut request_body = serde_json::json!({
            "model": self.config.chat_model,
            "prompt": prompt,
            "stream": true,  // Enable streaming
        });

        // Add custom parameters
        let options = serde_json::to_value(&params).map_err(|e| GraphRAGError::Generation {
            message: format!("Failed to serialize generation params: {}", e),
        })?;

        if !options.as_object().unwrap().is_empty() {
            request_body["options"] = options;
        }

        // Create channel for streaming tokens
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Clone data needed for async task
        let client = self.client.clone();
        let stats = self.stats.clone();
        let prompt_len = prompt.len();

        // Spawn background task to read streaming response
        tokio::spawn(async move {
            match client
                .post(&endpoint)
                .set("Content-Type", "application/json")
                .send_json(&request_body)
            {
                Ok(response) => {
                    let reader = std::io::BufReader::new(response.into_reader());
                    use std::io::BufRead;

                    let mut total_response = String::new();

                    for line in reader.lines() {
                        match line {
                            Ok(line_str) => {
                                if line_str.is_empty() {
                                    continue;
                                }

                                // Parse JSON response for this chunk
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line_str) {
                                    if let Some(token) = json["response"].as_str() {
                                        total_response.push_str(token);

                                        // Send token through channel
                                        if tx.send(token.to_string()).await.is_err() {
                                            // Receiver dropped, stop streaming
                                            break;
                                        }
                                    }

                                    // Check if done
                                    if json["done"].as_bool() == Some(true) {
                                        // Record success
                                        let estimated_tokens = (prompt_len + total_response.len()) / 4;
                                        stats.record_success(estimated_tokens as u64);
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                #[cfg(feature = "tracing")]
                                tracing::error!("Error reading streaming response: {}", e);
                                stats.record_failure();
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!("Failed to initiate streaming request: {}", e);
                    stats.record_failure();
                }
            }
        });

        Ok(rx)
    }

    /// Generate text completion (sync fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub async fn generate(&self, _prompt: &str) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for Ollama integration".to_string(),
        })
    }

    /// Generate with custom parameters (fallback)
    #[cfg(not(feature = "ureq"))]
    pub async fn generate_with_params(&self, _prompt: &str, _params: OllamaGenerationParams) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for Ollama integration".to_string(),
        })
    }
}
