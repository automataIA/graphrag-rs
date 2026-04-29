//! OpenAI-compatible chat backend.
//!
//! Mirrors the surface of `crate::ollama::OllamaClient` so consumers can
//! talk to any OpenAI-spec server (vLLM, llama-server, real OpenAI,
//! OpenRouter, ...) without changing call sites. Pair with the
//! `ChatClient` enum dispatcher in `crate::chat` to route all chat-LLM
//! traffic — entity extraction, query planning, gleaning — between an
//! Ollama-protocol server and an OpenAI-compat one based on `Config`.

use crate::core::error::{GraphRAGError, Result};
use crate::ollama::{OllamaGenerationParams, OllamaUsageStats};
use serde::{Deserialize, Serialize};

/// OpenAI-compatible chat-backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// Whether to use the OpenAI-compat backend for chat. When true,
    /// takes precedence over `Config.ollama.enabled` for chat LLM use.
    #[serde(default)]
    pub enabled: bool,

    /// Full base URL including the version path. Examples:
    /// - `http://localhost:8000/v1` (vLLM)
    /// - `http://localhost:17171/v1` (llama-server)
    /// - `https://api.openai.com/v1` (real OpenAI)
    /// - `https://openrouter.ai/api/v1` (OpenRouter)
    #[serde(default = "default_base_url")]
    pub base_url: String,

    /// Chat model identifier. For self-hosted servers this is whatever
    /// the server reports under GET /models; for real OpenAI it's
    /// "gpt-4o-mini" / "gpt-4o" / "o1-preview" / etc.
    #[serde(default = "default_chat_model")]
    pub chat_model: String,

    /// Bearer token. Empty string disables the Authorization header
    /// (fine for self-hosted servers without auth).
    #[serde(default)]
    pub api_key: String,

    /// HTTP request timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Maximum retry attempts on transient errors.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Maximum tokens to generate per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Whether to cache responses by prompt hash. Stub (no cache yet).
    #[serde(default = "default_enable_caching")]
    pub enable_caching: bool,

    /// Extra top-level fields merged into every chat-completions request
    /// body. Used to pass non-standard knobs that specific OpenAI-compat
    /// servers accept — e.g. llama.cpp / vLLM honor
    /// `chat_template_kwargs: {"enable_thinking": false}` to suppress
    /// Qwen3-style reasoning without changing the server's CLI flags.
    /// Must be a JSON object; non-object values are ignored on merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Value>,
}

fn default_base_url() -> String { "http://localhost:8000/v1".to_string() }
fn default_chat_model() -> String { "gpt-4o-mini".to_string() }
fn default_timeout() -> u64 { 60 }
fn default_max_retries() -> u32 { 3 }
fn default_enable_caching() -> bool { true }

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: default_base_url(),
            chat_model: default_chat_model(),
            api_key: String::new(),
            timeout_seconds: default_timeout(),
            max_retries: default_max_retries(),
            max_tokens: Some(2000),
            temperature: Some(0.7),
            enable_caching: default_enable_caching(),
            extra_body: None,
        }
    }
}

/// OpenAI-compatible chat client.
///
/// Uses synchronous `ureq` (matching `OllamaClient`'s choice) wrapped in
/// `tokio::task::spawn_blocking` for async callers. Keeps stats and a
/// (currently bypassed) cache hook for parity with `OllamaClient`.
#[derive(Clone)]
pub struct OpenAIClient {
    config: OpenAIConfig,
    #[cfg(feature = "ureq")]
    agent: ureq::Agent,
    stats: OllamaUsageStats,
}

impl std::fmt::Debug for OpenAIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIClient")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl OpenAIClient {
    /// Build a new client. Network is not touched until the first
    /// `generate*` call.
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "ureq")]
            agent: ureq::AgentBuilder::new().build(),
            stats: OllamaUsageStats::new(),
        }
    }

    /// Read access to the configured chat model (used by adapters / logs).
    pub fn config(&self) -> &OpenAIConfig {
        &self.config
    }

    /// Stats handle (parity with `OllamaClient`).
    pub fn get_stats(&self) -> &OllamaUsageStats {
        &self.stats
    }

    /// Single-shot chat completion with default params.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let params = OllamaGenerationParams {
            num_predict: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: None,
            top_k: None,
            stop: None,
            repeat_penalty: None,
            num_ctx: None,
            keep_alive: None,
            context: None,
        };
        self.generate_with_params(prompt, params).await
    }

    /// Chat completion with caller-supplied params. The
    /// `OllamaGenerationParams` shape is kept as the canonical params
    /// type so consumers don't have to branch on backend; only the
    /// fields that map to OpenAI (max_tokens, temperature, top_p, stop)
    /// are honored. `top_k` / `repeat_penalty` / `keep_alive` /
    /// `num_ctx` / `context` are Ollama-specific and silently ignored.
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<String> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        // Build request JSON. Single-turn chat (one user message) — same
        // shape consumers send to OllamaClient.generate{,_with_params}.
        let mut body = serde_json::json!({
            "model": self.config.chat_model,
            "messages": [
                { "role": "user", "content": prompt }
            ],
            "stream": false,
        });
        if let Some(t) = params.temperature.or(self.config.temperature) {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(n) = params.num_predict.or(self.config.max_tokens) {
            body["max_tokens"] = serde_json::json!(n);
        }
        if let Some(p) = params.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if let Some(stops) = params.stop {
            if !stops.is_empty() {
                body["stop"] = serde_json::json!(stops);
            }
        }
        // Merge config.extra_body keys at the top level. Lets callers pass
        // server-specific knobs (e.g. `chat_template_kwargs` on llama.cpp
        // to disable Qwen3 thinking) without expanding this struct for
        // every backend quirk. Existing keys win — set fields on this
        // struct take precedence over extra_body collisions.
        if let (Some(serde_json::Value::Object(extras)), Some(obj)) =
            (self.config.extra_body.as_ref(), body.as_object_mut())
        {
            for (k, v) in extras {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }

        let api_key = self.config.api_key.clone();
        let timeout = std::time::Duration::from_secs(self.config.timeout_seconds);
        let stats = self.stats.clone();

        // ureq is sync — push the network call onto the blocking pool so
        // we don't block the tokio reactor. Mirrors what OllamaClient does.
        let response = tokio::task::spawn_blocking(move || -> Result<String> {
            #[cfg(feature = "ureq")]
            {
                let mut req = ureq::AgentBuilder::new()
                    .timeout(timeout)
                    .build()
                    .post(&url)
                    .set("Content-Type", "application/json");
                if !api_key.is_empty() {
                    req = req.set("Authorization", &format!("Bearer {}", api_key));
                }

                let resp = match req.send_json(body) {
                    Ok(r) => r,
                    Err(ureq::Error::Status(code, r)) => {
                        let body = r.into_string().unwrap_or_default();
                        stats.record_failure();
                        return Err(GraphRAGError::Generation {
                            message: format!("OpenAI HTTP {code}: {body}"),
                        });
                    },
                    Err(e) => {
                        stats.record_failure();
                        return Err(GraphRAGError::Generation {
                            message: format!("OpenAI transport error: {e}"),
                        });
                    },
                };

                let parsed: serde_json::Value = match resp.into_json() {
                    Ok(v) => v,
                    Err(e) => {
                        stats.record_failure();
                        return Err(GraphRAGError::Generation {
                            message: format!("OpenAI response parse: {e}"),
                        });
                    },
                };

                // OpenAI/llama-server/vLLM all return:
                //   { choices: [{ message: { role: "assistant", content: "..." } }],
                //     usage: { prompt_tokens, completion_tokens, total_tokens } }
                let content = parsed
                    .pointer("/choices/0/message/content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let total_tokens = parsed
                    .pointer("/usage/total_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                stats.record_success(total_tokens);

                Ok(content)
            }
            #[cfg(not(feature = "ureq"))]
            {
                let _ = (url, body, api_key, timeout);
                stats.record_failure();
                Err(GraphRAGError::Generation {
                    message: "OpenAI client requires the `ureq` feature".to_string(),
                })
            }
        })
        .await
        .map_err(|e| GraphRAGError::Generation {
            message: format!("OpenAI join error: {e}"),
        })??;

        Ok(response)
    }

    /// Stub for parity with OllamaClient.
    pub fn clear_cache(&self) {}

    /// Stub for parity.
    pub fn cache_size(&self) -> usize { 0 }
}
