//! Ollama LLM integration
//!
//! This module provides integration with Ollama for local LLM inference.

use crate::core::{GraphRAGError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
    /// Context window size in tokens.
    ///
    /// **Critical for long documents**: Ollama silently truncates prompts that exceed
    /// the default context size (often 2k-8k tokens). Set this to accommodate the
    /// full document + chunk + instructions when using Contextual Enrichment.
    ///
    /// For KV Cache efficiency, calculate as:
    /// `tokens(instructions) + tokens(document) + tokens(max_chunk) + output_tokens + 5% margin`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    /// How long to keep the model loaded in memory after the request (e.g. "1h", "30m", "0").
    ///
    /// **Critical for KV Cache**: Without this, Ollama may unload the model between
    /// consecutive requests, destroying the KV cache and forcing a full re-evaluation
    /// of the static document prefix for every chunk. Set to "1h" when processing
    /// multiple chunks from the same document.
    ///
    /// This is a top-level Ollama API field, not an option — serialized separately.
    #[serde(skip)]
    pub keep_alive: Option<String>,

    /// KV cache context from a previous `/api/generate` response.
    ///
    /// When set, the model **continues from this token state** instead of re-evaluating
    /// the entire prompt. Use this for the two-step KV cache pattern:
    ///
    /// 1. **Prime**: send the full document, get `context` back (loads doc into KV cache)
    /// 2. **Per chunk**: send only the chunk text with the priming `context`
    ///    → Ollama skips document re-evaluation, only evaluates ~128 chunk tokens
    ///
    /// This is a top-level Ollama API field — serialized separately.
    #[serde(skip)]
    pub context: Option<Vec<i64>>,
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
            num_ctx: None,
            keep_alive: None,
            context: None,
        }
    }
}

/// Full response from `/api/generate`, including KV cache context and token stats.
///
/// Used by [`OllamaClient::generate_with_full_response`] to support the two-step
/// KV cache pattern (prime with document, then enrich each chunk cheaply).
#[derive(Debug, Clone)]
pub struct OllamaGenerateResponse {
    /// The generated text
    pub text: String,
    /// KV cache token state — pass back as `OllamaGenerationParams::context` on the
    /// next request to continue from this exact point without re-evaluating prior tokens.
    pub context: Vec<i64>,
    /// Tokens actually evaluated in the prompt (vs reused from KV cache).
    /// With KV cache working: ~= chunk_tokens.  Without: ~= full_prompt_tokens.
    pub prompt_eval_count: u64,
    /// Tokens generated in the response.
    pub eval_count: u64,
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
    /// How long to keep the model loaded in memory between requests (e.g. "1h", "30m", "0").
    ///
    /// Without this, Ollama may unload the model between requests, destroying the KV cache
    /// and forcing full re-evaluation of long document contexts on every request.
    /// Set to "1h" when processing multiple chunks from the same document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Default context window size for generation requests.
    ///
    /// Ollama silently truncates prompts exceeding this value (default is often 2048-8192).
    /// For long-document processing, set this to at least:
    /// `tokens(document) + tokens(max_chunk) + tokens(instructions) + 150 output tokens`
    /// Use `None` to let Ollama use its model default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
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
            keep_alive: None,
            num_ctx: None,
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

    /// Access the underlying Ollama configuration
    pub fn config(&self) -> &OllamaConfig {
        &self.config
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
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<String> {
        let endpoint = format!("{}:{}/api/generate", self.config.host, self.config.port);

        // Extract keep_alive before serializing params (it's a top-level field, not an option)
        let keep_alive = params
            .keep_alive
            .clone()
            .or_else(|| self.config.keep_alive.clone());

        let mut request_body = serde_json::json!({
            "model": self.config.chat_model,
            "prompt": prompt,
            "stream": false,
        });

        // keep_alive is a top-level field (controls model unloading between requests)
        if let Some(ref ka) = keep_alive {
            request_body["keep_alive"] = serde_json::Value::String(ka.clone());
        }

        // context is a top-level field: KV cache token state from a previous response.
        // When set, the model continues from this state, skipping re-evaluation of prior tokens.
        if let Some(ref ctx) = params.context {
            request_body["context"] = serde_json::Value::Array(
                ctx.iter()
                    .map(|&t| serde_json::Value::Number(t.into()))
                    .collect(),
            );
        }

        // Build options object: serialized params + num_ctx
        let mut options = serde_json::to_value(&params).map_err(|e| GraphRAGError::Generation {
            message: format!("Failed to serialize generation params: {}", e),
        })?;

        // Add num_ctx to options (overrides config default if set in params)
        let effective_num_ctx = params.num_ctx.or(self.config.num_ctx);
        if let Some(num_ctx) = effective_num_ctx {
            if let Some(obj) = options.as_object_mut() {
                obj.insert(
                    "num_ctx".to_string(),
                    serde_json::Value::Number(num_ctx.into()),
                );
            }
        }

        if !options.as_object().map_or(true, |o| o.is_empty()) {
            request_body["options"] = options;
        }

        // Make HTTP request with retry logic
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match self
                .client
                .post(&endpoint)
                .set("Content-Type", "application/json")
                .send_json(&request_body)
            {
                Ok(response) => {
                    let json_response: serde_json::Value =
                        response
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
                                self.cache
                                    .insert(prompt.to_string(), response_string.clone());

                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    "Cached response for prompt (length: {})",
                                    prompt.len()
                                );
                            }
                        }

                        return Ok(response_string);
                    } else {
                        self.stats.record_failure();
                        return Err(GraphRAGError::Generation {
                            message: format!("Invalid response format: {:?}", json_response),
                        });
                    }
                },
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("Ollama API request failed (attempt {}): {}", attempt, e);
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        // Wait before retry (exponential backoff)
                        tokio::time::sleep(std::time::Duration::from_millis(100 * attempt as u64))
                            .await;
                    }
                },
            }
        }

        self.stats.record_failure();
        Err(GraphRAGError::Generation {
            message: format!(
                "Ollama API failed after {} retries: {:?}",
                self.config.max_retries, last_error
            ),
        })
    }

    /// Generate text and return the full response including KV cache context and token stats.
    ///
    /// Use this for the two-step contextual enrichment pattern:
    ///
    /// ```no_run
    /// # use graphrag_core::ollama::{OllamaClient, OllamaConfig, OllamaGenerationParams};
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let client = OllamaClient::new(OllamaConfig::default());
    ///
    /// // Step 1: Prime — load the document into Ollama's KV cache
    /// let prime_params = OllamaGenerationParams {
    ///     num_predict: Some(1), // generate minimal output; we just want the context
    ///     keep_alive: Some("1h".to_string()),
    ///     num_ctx: Some(32768),
    ///     ..Default::default()
    /// };
    /// let prime = client.generate_with_full_response("<document>..full doc..</document>", prime_params).await?;
    /// println!("Prompt tokens evaluated: {}", prime.prompt_eval_count); // ~doc_tokens
    ///
    /// // Step 2: Per chunk — only the chunk tokens are evaluated
    /// for chunk in chunks {
    ///     let params = OllamaGenerationParams {
    ///         num_predict: Some(80),
    ///         context: Some(prime.context.clone()),  // ← KV cache reuse!
    ///         keep_alive: Some("1h".to_string()),
    ///         ..Default::default()
    ///     };
    ///     let resp = client.generate_with_full_response(&chunk, params).await?;
    ///     println!("Chunk tokens evaluated: {}", resp.prompt_eval_count); // ~chunk_tokens, not doc_tokens!
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "ureq")]
    pub async fn generate_with_full_response(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<OllamaGenerateResponse> {
        let endpoint = format!("{}:{}/api/generate", self.config.host, self.config.port);

        let keep_alive = params
            .keep_alive
            .clone()
            .or_else(|| self.config.keep_alive.clone());

        let mut request_body = serde_json::json!({
            "model": self.config.chat_model,
            "prompt": prompt,
            "stream": false,
        });

        if let Some(ref ka) = keep_alive {
            request_body["keep_alive"] = serde_json::Value::String(ka.clone());
        }

        if let Some(ref ctx) = params.context {
            request_body["context"] = serde_json::Value::Array(
                ctx.iter()
                    .map(|&t| serde_json::Value::Number(t.into()))
                    .collect(),
            );
        }

        let mut options = serde_json::to_value(&params).map_err(|e| GraphRAGError::Generation {
            message: format!("Failed to serialize generation params: {}", e),
        })?;

        let effective_num_ctx = params.num_ctx.or(self.config.num_ctx);
        if let Some(num_ctx) = effective_num_ctx {
            if let Some(obj) = options.as_object_mut() {
                obj.insert(
                    "num_ctx".to_string(),
                    serde_json::Value::Number(num_ctx.into()),
                );
            }
        }

        if !options.as_object().map_or(true, |o| o.is_empty()) {
            request_body["options"] = options;
        }

        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match self
                .client
                .post(&endpoint)
                .set("Content-Type", "application/json")
                .send_json(&request_body)
            {
                Ok(response) => {
                    let json_response: serde_json::Value =
                        response
                            .into_json()
                            .map_err(|e| GraphRAGError::Generation {
                                message: format!("Failed to parse JSON response: {}", e),
                            })?;

                    let text = json_response["response"]
                        .as_str()
                        .ok_or_else(|| GraphRAGError::Generation {
                            message: format!("Invalid response format: {:?}", json_response),
                        })?
                        .to_string();

                    let context: Vec<i64> = json_response["context"]
                        .as_array()
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                        .unwrap_or_default();

                    let prompt_eval_count =
                        json_response["prompt_eval_count"].as_u64().unwrap_or(0);
                    let eval_count = json_response["eval_count"].as_u64().unwrap_or(0);

                    let estimated_tokens = (prompt.len() + text.len()) / 4;
                    self.stats.record_success(estimated_tokens as u64);

                    return Ok(OllamaGenerateResponse {
                        text,
                        context,
                        prompt_eval_count,
                        eval_count,
                    });
                },
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries {
                        tokio::time::sleep(std::time::Duration::from_millis(100 * attempt as u64))
                            .await;
                    }
                },
            }
        }

        self.stats.record_failure();
        Err(GraphRAGError::Generation {
            message: format!(
                "Ollama API failed after {} retries: {:?}",
                self.config.max_retries, last_error
            ),
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
    pub async fn generate_streaming(
        &self,
        prompt: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
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
                                if let Ok(json) =
                                    serde_json::from_str::<serde_json::Value>(&line_str)
                                {
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
                                        let estimated_tokens =
                                            (prompt_len + total_response.len()) / 4;
                                        stats.record_success(estimated_tokens as u64);
                                        break;
                                    }
                                }
                            },
                            Err(e) => {
                                #[cfg(feature = "tracing")]
                                tracing::error!("Error reading streaming response: {}", e);
                                stats.record_failure();
                                break;
                            },
                        }
                    }
                },
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!("Failed to initiate streaming request: {}", e);
                    stats.record_failure();
                },
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
    pub async fn generate_with_params(
        &self,
        _prompt: &str,
        _params: OllamaGenerationParams,
    ) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for Ollama integration".to_string(),
        })
    }
}
