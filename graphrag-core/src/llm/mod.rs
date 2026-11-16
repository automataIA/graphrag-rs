//! OpenAI-compatible LLM integration.
//!
//! One client for every chat backend that speaks the OpenAI
//! `/v1/chat/completions` protocol: a local `llama-server` (llama.cpp),
//! Cerebras, Groq, or any other compatible endpoint.
//!
//! The API key is never stored in configuration: [`LlmConfig::api_key_env`]
//! names the environment variable that holds it, and the client reads it at
//! request time.

use crate::core::{GraphRAGError, Result};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// Sampling and output parameters for one chat completion request.
///
/// Fields left as `None` are omitted from the request so the server default
/// applies. `top_k` and `repeat_penalty` are llama.cpp extensions; leave them
/// unset for remote providers that reject unknown fields.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LlmGenerationParams {
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling (llama.cpp extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Repeat penalty (llama.cpp extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// OpenAI `response_format` object, e.g.
    /// `{"type":"json_schema","json_schema":{"name":"x","schema":{...}}}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
}

impl Default for LlmGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: Some(2000),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop: None,
            repeat_penalty: None,
            response_format: None,
        }
    }
}

impl LlmGenerationParams {
    /// Constrain the output to a JSON schema (OpenAI `json_schema` response format).
    pub fn with_json_schema(mut self, name: &str, schema: serde_json::Value) -> Self {
        self.response_format = Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": name, "strict": true, "schema": schema},
        }));
        self
    }
}

/// Server-reported token usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LlmUsage {
    /// Input tokens.
    pub prompt_tokens: u64,
    /// Generated tokens.
    pub completion_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
}

/// Result of one chat completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmGenerateResponse {
    /// Text of `choices[0].message.content`.
    pub text: String,
    /// Usage object, when the server returns a complete one.
    pub usage: Option<LlmUsage>,
    /// Prompt tokens reused from the server KV cache (llama.cpp `timings.cache_n`).
    pub cached_prompt_tokens: Option<u64>,
}

/// Process-wide token totals (see [`global_llm_usage`]).
///
/// Counters hold only server-reported figures: a response without a `usage`
/// object adds to `responses_without_usage` and nothing else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LlmUsageTotals {
    /// Successful responses received from a server (extraction-cache hits never get here).
    pub requests: u64,
    /// Server-reported prompt tokens.
    pub prompt_tokens: u64,
    /// Server-reported completion tokens.
    pub completion_tokens: u64,
    /// Prompt tokens the server reported as served from its cache.
    pub cached_prompt_tokens: u64,
    /// Successful responses that carried no `usage` object (streaming included).
    pub responses_without_usage: u64,
    /// Chat responses with content that were cut at the token limit (`finish_reason` "length").
    pub truncated_responses: u64,
}

impl LlmUsageTotals {
    /// Difference `self - earlier` (field-wise, saturating at 0).
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(earlier.requests),
            prompt_tokens: self.prompt_tokens.saturating_sub(earlier.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_sub(earlier.completion_tokens),
            cached_prompt_tokens: self
                .cached_prompt_tokens
                .saturating_sub(earlier.cached_prompt_tokens),
            responses_without_usage: self
                .responses_without_usage
                .saturating_sub(earlier.responses_without_usage),
            truncated_responses: self
                .truncated_responses
                .saturating_sub(earlier.truncated_responses),
        }
    }
}

static GLOBAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_COMPLETION_TOKENS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_CACHED_PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_WITHOUT_USAGE: AtomicU64 = AtomicU64::new(0);
static GLOBAL_TRUNCATED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the process-wide usage accumulated by every [`LlmClient`].
pub fn global_llm_usage() -> LlmUsageTotals {
    LlmUsageTotals {
        requests: GLOBAL_REQUESTS.load(Ordering::Relaxed),
        prompt_tokens: GLOBAL_PROMPT_TOKENS.load(Ordering::Relaxed),
        completion_tokens: GLOBAL_COMPLETION_TOKENS.load(Ordering::Relaxed),
        cached_prompt_tokens: GLOBAL_CACHED_PROMPT_TOKENS.load(Ordering::Relaxed),
        responses_without_usage: GLOBAL_WITHOUT_USAGE.load(Ordering::Relaxed),
        truncated_responses: GLOBAL_TRUNCATED.load(Ordering::Relaxed),
    }
}

/// Add one successful response to the process-wide totals.
pub fn record_global_usage(response: &LlmGenerateResponse) {
    GLOBAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    match response.usage {
        Some(usage) => {
            GLOBAL_PROMPT_TOKENS.fetch_add(usage.prompt_tokens, Ordering::Relaxed);
            GLOBAL_COMPLETION_TOKENS.fetch_add(usage.completion_tokens, Ordering::Relaxed);
        },
        None => {
            GLOBAL_WITHOUT_USAGE.fetch_add(1, Ordering::Relaxed);
        },
    }
    if let Some(cached) = response.cached_prompt_tokens {
        GLOBAL_CACHED_PROMPT_TOKENS.fetch_add(cached, Ordering::Relaxed);
    }
}

/// Usage statistics for an [`LlmClient`].
#[derive(Debug, Clone, Default)]
pub struct LlmUsageStats {
    /// Total number of requests
    pub total_requests: Arc<AtomicU64>,
    /// Total number of successful requests
    pub successful_requests: Arc<AtomicU64>,
    /// Total number of failed requests
    pub failed_requests: Arc<AtomicU64>,
    /// Total tokens (server-reported when available, otherwise estimated)
    pub total_tokens: Arc<AtomicU64>,
}

impl LlmUsageStats {
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

/// LLM configuration (OpenAI-compatible endpoints).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Enable LLM-backed stages.
    pub enabled: bool,
    /// Chat API base URL, including `/v1` (e.g. `http://127.0.0.1:8080/v1`).
    pub base_url: String,
    /// Embedding API base URL, including `/v1` (second llama-server with `--embeddings`).
    pub embedding_base_url: String,
    /// Name of the environment variable that holds the API key (never the key itself).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Model name for embeddings
    pub embedding_model: String,
    /// Model name for chat/generation
    pub chat_model: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum attempts per request (transport errors, HTTP 429 and 5xx are retried)
    pub max_retries: u32,
    /// Fall back to hash embeddings when the embedding server is unavailable
    pub fallback_to_hash: bool,
    /// Default maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Default sampling temperature
    pub temperature: Option<f32>,
    /// Cache identical requests in memory
    pub enable_caching: bool,
    /// Maximum estimated tokens of retrieved context placed in an answer prompt
    /// (estimate: chars / 3). Keep below the server per-slot context minus the answer.
    pub max_context_tokens: u32,
    /// Maximum LLM requests in flight during entity extraction. Match the server
    /// slots (`llama-server -np N`); 1 = sequential.
    pub max_concurrent_requests: usize,
    /// Cache raw extraction responses on disk under `<workspace>/extraction_cache/`
    /// (only active when `auto_save` provides a workspace directory).
    pub extraction_cache: bool,
    /// Send `chat_template_kwargs: {"enable_thinking": false}` with every chat request so
    /// reasoning models (Qwen3.x) answer directly instead of spending `max_tokens` on
    /// `reasoning_content`. Hosted APIs that reject the field are detected automatically
    /// (retried with `reasoning_effort: "none"`, then without either; remembered per client).
    pub disable_thinking: bool,
}

/// Default budget for retrieved context in answer prompts.
pub const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 6000;

/// Default number of concurrent LLM extraction requests (matches `llama-server -np 4`).
pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 4;

/// Default chat base URL (local llama-server).
pub const DEFAULT_LLM_BASE_URL: &str = "http://127.0.0.1:8080/v1";
/// Default embedding base URL (local llama-server started with `--embeddings`).
pub const DEFAULT_EMBEDDING_BASE_URL: &str = "http://127.0.0.1:8081/v1";

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: DEFAULT_LLM_BASE_URL.to_string(),
            embedding_base_url: DEFAULT_EMBEDDING_BASE_URL.to_string(),
            api_key_env: None,
            embedding_model: "multilingual-e5-small".to_string(),
            chat_model: "qwen3.5-9b".to_string(),
            timeout_seconds: 120,
            max_retries: 3,
            fallback_to_hash: true,
            max_tokens: Some(2000),
            temperature: Some(0.7),
            enable_caching: true,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            extraction_cache: true,
            disable_thinking: true,
        }
    }
}

impl LlmConfig {
    /// Full chat completions URL derived from [`Self::base_url`].
    pub fn chat_completions_url(&self) -> String {
        join_url(&self.base_url, "/chat/completions")
    }

    /// Full embeddings URL derived from [`Self::embedding_base_url`].
    pub fn embeddings_url(&self) -> String {
        join_url(&self.embedding_base_url, "/embeddings")
    }

    /// Read the API key from the environment variable named by [`Self::api_key_env`].
    ///
    /// Returns `Ok(None)` when no variable is configured (local llama-server),
    /// and an error when a variable is configured but missing or empty.
    pub fn api_key(&self) -> Result<Option<String>> {
        let Some(var) = self.api_key_env.as_deref().filter(|v| !v.trim().is_empty()) else {
            return Ok(None);
        };
        match std::env::var(var) {
            Ok(key) if !key.trim().is_empty() => Ok(Some(key)),
            _ => Err(GraphRAGError::Config {
                message: format!(
                    "llm.api_key_env names {var}, but that variable is unset or empty"
                ),
            }),
        }
    }
}

fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with(path) {
        base.to_string()
    } else {
        format!("{base}{path}")
    }
}

fn generation_error(message: impl Into<String>) -> GraphRAGError {
    GraphRAGError::Generation {
        message: message.into(),
    }
}

/// Parse an OpenAI-compatible chat completion response body.
pub fn parse_chat_response(value: &serde_json::Value) -> Result<LlmGenerateResponse> {
    let text = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| {
            let choice = &value["choices"][0];
            let message = &choice["message"];
            let non_empty = |field: &str| message[field].as_str().is_some_and(|r| !r.trim().is_empty());
            let reasoning = non_empty("reasoning_content") || non_empty("reasoning");
            let truncated = choice["finish_reason"].as_str() == Some("length");
            if reasoning {
                generation_error(
                    "chat response has empty content: the model spent its output on reasoning/thinking \
                     (message.reasoning_content or message.reasoning is set, finish_reason may be \
                     \"length\"). Raise llm.max_tokens or set llm.disable_thinking = true (llama-server: \
                     --chat-template-kwargs '{\"enable_thinking\":false}')",
                )
            } else if truncated {
                generation_error(
                    "chat response has empty content and finish_reason is \"length\": the output hit \
                     the token limit before any content was produced. Raise llm.max_tokens or set \
                     llm.disable_thinking = true",
                )
            } else {
                generation_error("chat response has no non-empty choices[0].message.content")
            }
        })?
        .to_string();
    let usage = match value.get("usage") {
        None | Some(serde_json::Value::Null) => None,
        Some(usage) => {
            let field = |name: &str| {
                usage
                    .get(name)
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| generation_error(format!("usage.{name} is missing or invalid")))
            };
            Some(LlmUsage {
                prompt_tokens: field("prompt_tokens")?,
                completion_tokens: field("completion_tokens")?,
                total_tokens: field("total_tokens")?,
            })
        },
    };
    let cached_prompt_tokens = value
        .get("timings")
        .and_then(|t| t.get("cache_n"))
        .and_then(serde_json::Value::as_u64)
        // OpenAI-style servers (Cerebras, OpenAI) report it inside `usage`
        .or_else(|| {
            value
                .get("usage")
                .and_then(|u| u.get("prompt_tokens_details"))
                .and_then(|d| d.get("cached_tokens"))
                .and_then(serde_json::Value::as_u64)
        });
    if value["choices"][0]["finish_reason"].as_str() == Some("length") {
        // The warning is emitted by the caller, which knows the `max_tokens` sent.
        GLOBAL_TRUNCATED.fetch_add(1, Ordering::Relaxed);
    }
    Ok(LlmGenerateResponse {
        text,
        usage,
        cached_prompt_tokens,
    })
}

/// Warning text for a response cut at the token limit, naming the `max_tokens` sent with
/// the request; `None` when the response was not truncated.
#[cfg(feature = "llm")]
fn truncation_warning(
    body: &serde_json::Value,
    value: &serde_json::Value,
    response: &LlmGenerateResponse,
) -> Option<String> {
    if value["choices"][0]["finish_reason"].as_str() != Some("length") {
        return None;
    }
    let sent = body["max_tokens"]
        .as_u64()
        .map_or("not set".to_string(), |n| n.to_string());
    let completion = response
        .usage
        .map_or("unknown".to_string(), |u| u.completion_tokens.to_string());
    Some(format!(
        "LLM response cut at the token limit (finish_reason \"length\", max_tokens sent: {sent}, completion tokens: {completion}); the output may be incomplete, raise the limit that applies to this call (llm.max_tokens for answers, entity_extraction.extraction_max_tokens for extraction)"
    ))
}

/// Heuristic: did the server reject the body field `field`? True for a 4xx error naming
/// it, or a generic HTTP 400 saying a field is unknown/unrecognized/extra.
#[cfg(feature = "llm")]
fn rejects_field(error: &GraphRAGError, field: &str) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("llm http 4")
        && (message.contains(field)
            || (message.contains("llm http 400")
                && [
                    "unknown",
                    "unrecognized",
                    "extra",
                    "unexpected",
                    "not permitted",
                ]
                .iter()
                .any(|w| message.contains(w))))
}

/// How reasoning is switched off, remembered per client (`LlmClient::reasoning_mode`).
/// llama.cpp understands `chat_template_kwargs`; hosted APIs (Cerebras, Groq) reject it
/// and take `reasoning_effort: "none"`; if that is rejected too, nothing is sent.
const REASONING_KWARGS: u8 = 0;
const REASONING_EFFORT: u8 = 1;
#[cfg(feature = "llm")]
const REASONING_NONE: u8 = 2;

/// Longest single wait between retries.
#[cfg(feature = "llm")]
const MAX_RETRY_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// Rate-limited (HTTP 429) requests get at least this many attempts, whatever
/// `max_retries` says: free tiers answer 429 with a short `retry-after`.
#[cfg(feature = "llm")]
const RATE_LIMIT_MIN_ATTEMPTS: u32 = 8;

/// Server-suggested wait: the `retry-after` header (seconds, integer or float), else
/// `try again in <n>s` / `<n>ms` / `<m>m<s>s` in the error body.
#[cfg(feature = "llm")]
fn retry_wait_hint(retry_after: Option<&str>, body: &str) -> Option<std::time::Duration> {
    let secs = |v: f64| (v.is_finite() && v >= 0.0).then(|| std::time::Duration::from_secs_f64(v));
    if let Some(d) = retry_after.and_then(|v| v.trim().parse::<f64>().ok().and_then(secs)) {
        return Some(d);
    }
    let lower = body.to_ascii_lowercase();
    let mut rest = &lower[lower.find("try again in ")? + "try again in ".len()..];
    let mut total = 0.0;
    let mut parsed = false;
    loop {
        let digits = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        let Ok(value) = rest[..digits].parse::<f64>() else {
            break;
        };
        rest = &rest[digits..];
        let unit = rest
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(rest.len());
        total += match &rest[..unit] {
            "ms" => value / 1000.0,
            "s" => value,
            "m" => value * 60.0,
            _ => break,
        };
        parsed = true;
        rest = &rest[unit..];
    }
    parsed.then(|| secs(total)).flatten()
}

/// Wait before retry number `attempt` (1-based). 429: the server hint, else exponential
/// backoff (1s, 2s, 4s, ...). 503 with a hint: the hint. Anything else: 200ms * attempt.
/// A single wait never exceeds [`MAX_RETRY_WAIT`].
#[cfg(feature = "llm")]
fn retry_delay(
    status: Option<u16>,
    hint: Option<std::time::Duration>,
    attempt: u32,
) -> std::time::Duration {
    use std::time::Duration;
    match (status, hint) {
        (Some(429), hint) => hint
            .unwrap_or_else(|| Duration::from_secs(1u64 << (attempt.saturating_sub(1)).min(6)))
            .min(MAX_RETRY_WAIT),
        (Some(503), Some(hint)) => hint.min(MAX_RETRY_WAIT),
        _ => Duration::from_millis(200 * u64::from(attempt)),
    }
}

/// Outcome of a single HTTP attempt: retryable or final.
#[cfg(feature = "llm")]
enum Attempt {
    /// `status` is the HTTP status (None for transport errors); `hint` the server's
    /// suggested wait.
    Retry {
        error: GraphRAGError,
        status: Option<u16>,
        hint: Option<std::time::Duration>,
    },
    Fail(GraphRAGError),
}

/// OpenAI-compatible chat client.
#[derive(Clone)]
pub struct LlmClient {
    config: LlmConfig,
    #[cfg(feature = "llm")]
    agent: ureq::Agent,
    stats: LlmUsageStats,
    cache: Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,
    /// One of `REASONING_*`; advanced when the server rejects a reasoning-off field.
    reasoning_mode: Arc<AtomicU8>,
}

impl std::fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmClient")
            .field("base_url", &self.config.base_url)
            .field("chat_model", &self.config.chat_model)
            .field("api_key_env", &self.config.api_key_env)
            .field("stats", &self.stats)
            .finish()
    }
}

impl LlmClient {
    /// Create a new client. No network I/O happens here.
    pub fn new(config: LlmConfig) -> Self {
        Self {
            #[cfg(feature = "llm")]
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(
                    config.timeout_seconds.max(1),
                ))
                .build(),
            config,
            stats: LlmUsageStats::new(),
            cache: Arc::default(),
            reasoning_mode: Arc::new(AtomicU8::new(REASONING_KWARGS)),
        }
    }

    /// Get usage statistics
    pub fn get_stats(&self) -> &LlmUsageStats {
        &self.stats
    }

    /// Access the underlying configuration
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Clear the response cache
    pub fn clear_cache(&self) {
        self.cache_guard().clear();
    }

    /// Number of cached responses
    pub fn cache_size(&self) -> usize {
        self.cache_guard().len()
    }

    fn cache_guard(&self) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, String>> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Default parameters derived from the configuration.
    pub fn default_params(&self) -> LlmGenerationParams {
        LlmGenerationParams {
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            ..Default::default()
        }
    }

    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    fn request_body(
        &self,
        prompt: &str,
        params: &LlmGenerationParams,
        stream: bool,
    ) -> Result<serde_json::Value> {
        let mut body = serde_json::to_value(params)
            .map_err(|e| generation_error(format!("failed to serialize generation params: {e}")))?;
        let object = body
            .as_object_mut()
            .ok_or_else(|| generation_error("generation params did not serialize to an object"))?;
        object.insert("model".into(), self.config.chat_model.clone().into());
        object.insert(
            "messages".into(),
            serde_json::json!([{"role": "user", "content": prompt}]),
        );
        object.insert("stream".into(), stream.into());
        if self.config.disable_thinking {
            match self.reasoning_mode.load(Ordering::Relaxed) {
                REASONING_KWARGS => {
                    object.insert(
                        "chat_template_kwargs".into(),
                        serde_json::json!({"enable_thinking": false}),
                    );
                },
                REASONING_EFFORT => {
                    object.insert("reasoning_effort".into(), "none".into());
                },
                _ => {},
            }
        }
        Ok(body)
    }

    /// One blocking HTTP request, no retries and no cache.
    #[cfg(feature = "llm")]
    fn attempt(
        &self,
        body: &serde_json::Value,
    ) -> std::result::Result<LlmGenerateResponse, Attempt> {
        let url = self.config.chat_completions_url();
        let key = self.config.api_key().map_err(Attempt::Fail)?;
        let mut request = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = key {
            request = request.set("Authorization", &format!("Bearer {key}"));
        }
        match request.send_json(body) {
            Ok(response) => {
                let value: serde_json::Value = response.into_json().map_err(|e| {
                    Attempt::Fail(generation_error(format!(
                        "chat response from {url} was not valid JSON: {e}"
                    )))
                })?;
                let parsed = parse_chat_response(&value).map_err(Attempt::Fail)?;
                if let Some(message) = truncation_warning(body, &value, &parsed) {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("{message}");
                    #[cfg(not(feature = "tracing"))]
                    drop(message);
                }
                Ok(parsed)
            },
            Err(ureq::Error::Status(status, response)) => {
                let retry_after = response.header("retry-after").map(str::to_owned);
                let detail = crate::llama_cpp::bounded_response_body(response);
                let error = generation_error(format!("LLM HTTP {status} from {url}: {detail}"));
                if status == 429 || status >= 500 {
                    Err(Attempt::Retry {
                        error,
                        status: Some(status),
                        hint: retry_wait_hint(retry_after.as_deref(), &detail),
                    })
                } else {
                    Err(Attempt::Fail(error))
                }
            },
            Err(ureq::Error::Transport(transport)) => Err(Attempt::Retry {
                error: generation_error(format!(
                    "LLM request to {url} failed (timeout {}s): {transport}",
                    self.config.timeout_seconds
                )),
                status: None,
                hint: None,
            }),
        }
    }

    /// Blocking chat completion with retries (no cache). Usable outside async code.
    #[cfg(feature = "llm")]
    pub fn generate_blocking(
        &self,
        prompt: &str,
        params: &LlmGenerationParams,
    ) -> Result<LlmGenerateResponse> {
        let mut body = self.request_body(prompt, params, false)?;
        let mut attempts = self.config.max_retries.max(1);
        let mut attempt = 0u32;
        let last_error = loop {
            attempt += 1;
            match self.attempt(&body) {
                Ok(response) => {
                    let tokens = response
                        .usage
                        .map(|u| u.total_tokens)
                        .unwrap_or(((prompt.len() + response.text.len()) / 4) as u64);
                    self.stats.record_success(tokens);
                    record_global_usage(&response);
                    return Ok(response);
                },
                Err(Attempt::Fail(error)) => {
                    if self.downgrade_reasoning(&mut body, &error) {
                        // Hosted API refused a reasoning-off field: the body now uses the
                        // next option (remembered); resend without spending an attempt.
                        attempt -= 1;
                        continue;
                    }
                    self.stats.record_failure();
                    return Err(error);
                },
                Err(Attempt::Retry {
                    error,
                    status,
                    hint,
                }) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("LLM request failed (attempt {attempt}/{attempts}): {error}");
                    if status == Some(429) {
                        attempts = attempts.max(RATE_LIMIT_MIN_ATTEMPTS);
                    }
                    if attempt >= attempts {
                        break error;
                    }
                    std::thread::sleep(retry_delay(status, hint, attempt));
                },
            }
        };
        self.stats.record_failure();
        Err(last_error)
    }

    /// If `error` is the server rejecting the reasoning-off field present in `body`,
    /// switch `body` (and this client, for later requests) to the next option:
    /// `chat_template_kwargs` -> `reasoning_effort: "none"` -> nothing. Returns whether
    /// `body` changed.
    #[cfg(feature = "llm")]
    fn downgrade_reasoning(&self, body: &mut serde_json::Value, error: &GraphRAGError) -> bool {
        let Some(object) = body.as_object_mut() else {
            return false;
        };
        if object.contains_key("chat_template_kwargs")
            && rejects_field(error, "chat_template_kwargs")
        {
            object.remove("chat_template_kwargs");
            object.insert("reasoning_effort".into(), "none".into());
            self.reasoning_mode
                .fetch_max(REASONING_EFFORT, Ordering::Relaxed);
            true
        } else if object.contains_key("reasoning_effort")
            && rejects_field(error, "reasoning_effort")
        {
            object.remove("reasoning_effort");
            self.reasoning_mode
                .fetch_max(REASONING_NONE, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    #[cfg(feature = "llm")]
    fn cache_key(&self, prompt: &str, params: &LlmGenerationParams) -> String {
        format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}",
            self.config.base_url,
            self.config.chat_model,
            serde_json::to_string(params).unwrap_or_default(),
            prompt
        )
    }

    /// Chat completion returning text plus usage and KV-cache reuse stats.
    ///
    /// Runs the blocking HTTP call on tokio's blocking pool, so it never stalls
    /// the async executor. Identical requests are served from the in-memory
    /// cache when `enable_caching` is on (text only; usage is then `None`).
    #[cfg(feature = "llm")]
    pub async fn generate_with_full_response(
        &self,
        prompt: &str,
        params: LlmGenerationParams,
    ) -> Result<LlmGenerateResponse> {
        let key = self.cache_key(prompt, &params);
        if self.config.enable_caching {
            if let Some(text) = self.cache_guard().get(&key).cloned() {
                return Ok(LlmGenerateResponse {
                    text,
                    usage: None,
                    cached_prompt_tokens: None,
                });
            }
        }
        let client = self.clone();
        let prompt_owned = prompt.to_string();
        let response =
            tokio::task::spawn_blocking(move || client.generate_blocking(&prompt_owned, &params))
                .await
                .map_err(|e| generation_error(format!("LLM worker failed: {e}")))??;
        if self.config.enable_caching {
            self.cache_guard().insert(key, response.text.clone());
        }
        Ok(response)
    }

    /// Chat completion with explicit parameters.
    #[cfg(feature = "llm")]
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        params: LlmGenerationParams,
    ) -> Result<String> {
        Ok(self.generate_with_full_response(prompt, params).await?.text)
    }

    /// Chat completion with the configured default parameters.
    #[cfg(feature = "llm")]
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        self.generate_with_params(prompt, self.default_params())
            .await
    }

    /// Streaming chat completion (server-sent events). Yields content deltas.
    #[cfg(feature = "llm")]
    pub async fn generate_streaming(
        &self,
        prompt: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let body = self.request_body(prompt, &self.default_params(), true)?;
        let url = self.config.chat_completions_url();
        let key = self.config.api_key()?;
        let agent = self.agent.clone();
        let stats = self.stats.clone();
        let prompt_len = prompt.len();
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_blocking(move || {
            use std::io::BufRead;
            let mut request = agent.post(&url).set("Content-Type", "application/json");
            if let Some(key) = key {
                request = request.set("Authorization", &format!("Bearer {key}"));
            }
            let response = match request.send_json(&body) {
                Ok(response) => response,
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!("failed to start LLM stream: {_e}");
                    stats.record_failure();
                    return;
                },
            };
            let reader = std::io::BufReader::new(response.into_reader());
            let mut total = 0usize;
            for line in reader.lines() {
                let Ok(line) = line else {
                    stats.record_failure();
                    return;
                };
                let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if payload == "[DONE]" {
                    break;
                }
                let Ok(json) = serde_json::from_str::<serde_json::Value>(payload) else {
                    continue;
                };
                if let Some(token) = json["choices"][0]["delta"]["content"].as_str() {
                    total += token.len();
                    if tx.blocking_send(token.to_string()).is_err() {
                        break;
                    }
                }
            }
            stats.record_success(((prompt_len + total) / 4) as u64);
            // streamed responses carry no usage object
            record_global_usage(&LlmGenerateResponse {
                text: String::new(),
                usage: None,
                cached_prompt_tokens: None,
            });
        });
        Ok(rx)
    }

    /// Fallback when the `llm` feature is disabled.
    #[cfg(not(feature = "llm"))]
    pub async fn generate(&self, _prompt: &str) -> Result<String> {
        Err(generation_error(
            "the `llm` feature is required for LLM generation",
        ))
    }

    /// Fallback when the `llm` feature is disabled.
    #[cfg(not(feature = "llm"))]
    pub async fn generate_with_full_response(
        &self,
        _prompt: &str,
        _params: LlmGenerationParams,
    ) -> Result<LlmGenerateResponse> {
        Err(generation_error(
            "the `llm` feature is required for LLM generation",
        ))
    }

    /// Fallback when the `llm` feature is disabled.
    #[cfg(not(feature = "llm"))]
    pub async fn generate_with_params(
        &self,
        _prompt: &str,
        _params: LlmGenerationParams,
    ) -> Result<String> {
        Err(generation_error(
            "the `llm` feature is required for LLM generation",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_joined_once() {
        let mut config = LlmConfig::default();
        assert_eq!(
            config.chat_completions_url(),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
        assert_eq!(
            config.embeddings_url(),
            "http://127.0.0.1:8081/v1/embeddings"
        );
        config.base_url = "https://api.cerebras.ai/v1/".into();
        assert_eq!(
            config.chat_completions_url(),
            "https://api.cerebras.ai/v1/chat/completions"
        );
        config.base_url = "http://x/v1/chat/completions".into();
        assert_eq!(
            config.chat_completions_url(),
            "http://x/v1/chat/completions"
        );
    }

    #[test]
    fn api_key_env_missing_is_an_error_and_absent_is_none() {
        let mut config = LlmConfig::default();
        assert_eq!(config.api_key().unwrap(), None);
        config.api_key_env = Some("GRAPHRAG_TEST_SURELY_UNSET_KEY_VAR".into());
        assert!(config.api_key().is_err());
    }

    #[test]
    fn parse_reads_openai_cached_tokens() {
        let value = serde_json::json!({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 64}
            }
        });
        assert_eq!(
            parse_chat_response(&value).unwrap().cached_prompt_tokens,
            Some(64)
        );
    }

    #[test]
    fn recording_a_response_accumulates_global_usage() {
        // The accumulator is process-wide and other tests may add to it concurrently,
        // so assert lower bounds on the delta, never exact values.
        let before = global_llm_usage();
        let value = serde_json::json!({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 64}
            }
        });
        record_global_usage(&parse_chat_response(&value).unwrap());
        let no_usage = serde_json::json!({"choices": [{"message": {"content": "ok"}}]});
        record_global_usage(&parse_chat_response(&no_usage).unwrap());
        let delta = global_llm_usage().since(&before);
        assert!(delta.requests >= 2);
        assert!(delta.prompt_tokens >= 100);
        assert!(delta.completion_tokens >= 5);
        assert!(delta.cached_prompt_tokens >= 64);
        assert!(delta.responses_without_usage >= 1);
    }

    #[cfg(feature = "llm")]
    #[test]
    fn truncation_warning_reports_the_max_tokens_sent() {
        let cut = serde_json::json!({
            "choices": [{"message": {"content": "{"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 300, "total_tokens": 310}
        });
        let parsed = parse_chat_response(&cut).unwrap();
        let msg =
            truncation_warning(&serde_json::json!({"max_tokens": 300}), &cut, &parsed).unwrap();
        assert!(msg.contains("max_tokens sent: 300"), "{msg}");
        assert!(msg.contains("completion tokens: 300"), "{msg}");
        let msg = truncation_warning(&serde_json::json!({}), &cut, &parsed).unwrap();
        assert!(msg.contains("max_tokens sent: not set"), "{msg}");
        let done = serde_json::json!({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        });
        let parsed = parse_chat_response(&done).unwrap();
        assert!(truncation_warning(&serde_json::json!({}), &done, &parsed).is_none());
    }

    #[test]
    fn length_finish_reason_with_content_counts_a_truncation() {
        let before = global_llm_usage();
        let cut = serde_json::json!({
            "choices": [{"message": {"content": "{\"entities\": [{"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}
        });
        assert!(parse_chat_response(&cut).is_ok());
        assert!(global_llm_usage().since(&before).truncated_responses >= 1);
        let done = serde_json::json!({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        });
        assert!(parse_chat_response(&done).is_ok());
    }

    #[test]
    fn usage_totals_since_subtracts_and_serializes() {
        let earlier = LlmUsageTotals {
            requests: 1,
            prompt_tokens: 10,
            completion_tokens: 2,
            cached_prompt_tokens: 4,
            responses_without_usage: 0,
            truncated_responses: 0,
        };
        let later = LlmUsageTotals {
            requests: 4,
            prompt_tokens: 110,
            completion_tokens: 12,
            cached_prompt_tokens: 9,
            responses_without_usage: 1,
            truncated_responses: 2,
        };
        let delta = later.since(&earlier);
        assert_eq!(
            delta,
            LlmUsageTotals {
                requests: 3,
                prompt_tokens: 100,
                completion_tokens: 10,
                cached_prompt_tokens: 5,
                responses_without_usage: 1,
                truncated_responses: 2,
            }
        );
        // never underflows if the order is swapped
        assert_eq!(earlier.since(&later), LlmUsageTotals::default());
        let json = serde_json::to_value(delta).unwrap();
        assert_eq!(json["prompt_tokens"], 100);
        assert_eq!(json["responses_without_usage"], 1);
        assert_eq!(json["truncated_responses"], 2);
    }

    #[test]
    fn request_body_omits_unset_params_and_carries_schema() {
        let client = LlmClient::new(LlmConfig::default());
        let params = LlmGenerationParams {
            max_tokens: Some(10),
            temperature: Some(0.0),
            ..Default::default()
        }
        .with_json_schema("x", serde_json::json!({"type": "object"}));
        let body = client.request_body("hi", &params, false).unwrap();
        assert_eq!(body["model"], "qwen3.5-9b");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert!(body.get("top_k").is_none());
        assert!(body.get("repeat_penalty").is_none());
    }

    #[test]
    fn parse_reads_usage_and_cache_hits() {
        let value = serde_json::json!({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            "timings": {"cache_n": 236}
        });
        let response = parse_chat_response(&value).unwrap();
        assert_eq!(response.text, "ok");
        assert_eq!(response.usage.unwrap().total_tokens, 7);
        assert_eq!(response.cached_prompt_tokens, Some(236));
        let partial = serde_json::json!({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4}
        });
        assert!(parse_chat_response(&partial).is_err());
        assert!(parse_chat_response(&serde_json::json!({"choices": []})).is_err());
    }

    #[test]
    fn empty_content_with_reasoning_field_explains_reasoning() {
        let value = serde_json::json!({
            "choices": [{"finish_reason": "length",
                         "message": {"role": "assistant", "reasoning": "thinking..."}}]
        });
        let msg = parse_chat_response(&value).unwrap_err().to_string();
        assert!(msg.contains("reasoning"), "{msg}");
        assert!(msg.contains("max_tokens"), "{msg}");
        assert!(msg.contains("disable_thinking"), "{msg}");
    }

    #[test]
    fn empty_content_finish_length_without_reasoning_mentions_max_tokens() {
        let value = serde_json::json!({
            "choices": [{"finish_reason": "length", "message": {"role": "assistant"}}]
        });
        let msg = parse_chat_response(&value).unwrap_err().to_string();
        assert!(msg.contains("finish_reason"), "{msg}");
        assert!(msg.contains("max_tokens"), "{msg}");
    }

    #[cfg(feature = "llm")]
    #[test]
    fn retry_wait_hint_prefers_header_then_body() {
        use std::time::Duration;
        assert_eq!(retry_wait_hint(Some("8"), ""), Some(Duration::from_secs(8)));
        assert_eq!(
            retry_wait_hint(Some("0.5"), "try again in 9s"),
            Some(Duration::from_millis(500))
        );
        assert_eq!(
            retry_wait_hint(None, "Rate limit. Please try again in 8.34s. Visit"),
            Some(Duration::from_secs_f64(8.34))
        );
        assert_eq!(
            retry_wait_hint(None, "Please try again in 250ms."),
            Some(Duration::from_millis(250))
        );
        assert_eq!(
            retry_wait_hint(None, "Please try again in 1m26.4s."),
            Some(Duration::from_secs_f64(86.4))
        );
        assert_eq!(retry_wait_hint(Some("Wed, 21 Oct"), "nothing"), None);
        assert_eq!(retry_wait_hint(Some("-3"), ""), None);
    }

    #[cfg(feature = "llm")]
    #[test]
    fn retry_delay_uses_hint_backs_off_and_caps() {
        use std::time::Duration;
        let hint = Some(Duration::from_secs(8));
        assert_eq!(retry_delay(Some(429), hint, 1), Duration::from_secs(8));
        assert_eq!(retry_delay(Some(429), None, 1), Duration::from_secs(1));
        assert_eq!(retry_delay(Some(429), None, 3), Duration::from_secs(4));
        assert_eq!(retry_delay(Some(429), None, 30), Duration::from_secs(60));
        assert_eq!(
            retry_delay(Some(429), Some(Duration::from_secs(900)), 1),
            Duration::from_secs(60)
        );
        assert_eq!(retry_delay(Some(503), hint, 1), Duration::from_secs(8));
        // Other retryable errors keep the short linear backoff.
        assert_eq!(retry_delay(Some(503), None, 2), Duration::from_millis(400));
        assert_eq!(retry_delay(Some(500), hint, 2), Duration::from_millis(400));
        assert_eq!(retry_delay(None, None, 1), Duration::from_millis(200));
    }
}
