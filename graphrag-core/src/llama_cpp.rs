//! llama.cpp multilingual-e5 embedding provider.
//!
//! Talks to a `llama-server` started with `--embeddings --pooling mean`.
//! Chat generation lives in [`crate::llm::LlmClient`].

use crate::config::EmbeddingConfig;
use crate::core::{GraphRAGError, Result};

fn provider_error(message: impl Into<String>) -> GraphRAGError {
    GraphRAGError::Embedding {
        message: message.into(),
    }
}

/// Keep provider diagnostics useful without allowing an upstream error body to
/// become an unbounded GraphRAG error or break UTF-8 boundaries.
pub(crate) fn bounded_response_body(response: ureq::Response) -> String {
    const MAX_BYTES: usize = 4096;
    let body = response
        .into_string()
        .unwrap_or_else(|_| "<unreadable response body>".to_string());
    if body.len() <= MAX_BYTES {
        return body;
    }
    let end = body
        .char_indices()
        .take_while(|(index, _)| *index < MAX_BYTES)
        .map(|(index, _)| index)
        .last()
        .unwrap_or(0);
    format!("{}…", &body[..end])
}

fn provider_http_error(endpoint: &str, error: ureq::Error, timeout_seconds: u64) -> GraphRAGError {
    match error {
        ureq::Error::Status(status, response) => {
            let body = bounded_response_body(response);
            provider_error(format!(
                "llama.cpp HTTP {status} from {endpoint} (timeout {timeout_seconds}s): {body}"
            ))
        },
        ureq::Error::Transport(transport) => provider_error(format!(
            "llama.cpp request to {endpoint} failed (timeout {timeout_seconds}s): {transport}"
        )),
    }
}

fn validate_endpoint(endpoint: &str) -> Result<()> {
    let Some(authority) = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
    else {
        return Err(provider_error(format!(
            "llama.cpp endpoint must use http or https and include a host: {endpoint}"
        )));
    };
    let host = authority.split('/').next().unwrap_or_default();
    if host.is_empty() || host.contains(char::is_whitespace) || host.contains('@') {
        return Err(provider_error(format!(
            "llama.cpp endpoint must include a valid host: {endpoint}"
        )));
    }
    Ok(())
}

fn endpoint_with_path(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with(path.trim_start_matches('/')) {
        base.to_string()
    } else {
        format!("{base}{path}")
    }
}

fn base_endpoint(endpoint: &str) -> String {
    endpoint
        .trim_end_matches('/')
        .strip_suffix("/v1/embeddings")
        .or_else(|| {
            endpoint
                .trim_end_matches('/')
                .strip_suffix("/v1/chat/completions")
        })
        .unwrap_or(endpoint.trim_end_matches('/'))
        .to_string()
}

/// E5 embedding client for the llama.cpp embedding server.
///
/// llama.cpp exposes tokenizer IDs separately, so we explicitly apply the E5
/// intent prefix and enforce the model's 512-token limit before requesting the
/// embedding.  The server performs masked mean pooling; this client verifies and
/// L2-normalizes the returned vector.
#[derive(Clone)]
pub struct LlamaCppEmbeddingProvider {
    endpoint: String,
    tokenize_endpoint: String,
    model: String,
    timeout_seconds: u64,
    dimensions: usize,
    client: ureq::Agent,
}

impl std::fmt::Debug for LlamaCppEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppEmbeddingProvider")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("timeout_seconds", &self.timeout_seconds)
            .finish()
    }
}

impl LlamaCppEmbeddingProvider {
    /// Construct an E5 provider from runtime embedding settings.
    pub fn from_config(config: &EmbeddingConfig, timeout_seconds: u64) -> Result<Self> {
        let endpoint = config.api_endpoint.clone().ok_or_else(|| {
            provider_error("llama.cpp embeddings require embeddings.api_endpoint")
        })?;
        validate_endpoint(&endpoint)?;
        if config.dimension != 384 {
            return Err(provider_error(format!(
                "multilingual-e5-small requires dimension 384, got {}",
                config.dimension
            )));
        }
        if config
            .model
            .as_deref()
            .unwrap_or("multilingual-e5-small")
            .trim()
            .is_empty()
        {
            return Err(provider_error(
                "llama.cpp embedding model must not be empty",
            ));
        }
        if timeout_seconds == 0 {
            return Err(provider_error(
                "llama.cpp timeout_seconds must be greater than 0",
            ));
        }
        let model = config
            .model
            .clone()
            .unwrap_or_else(|| "multilingual-e5-small".to_string());
        let endpoint = endpoint_with_path(&endpoint, "/v1/embeddings");
        let tokenize_endpoint = endpoint_with_path(&base_endpoint(&endpoint), "/tokenize");
        Ok(Self {
            endpoint,
            tokenize_endpoint,
            model,
            timeout_seconds,
            dimensions: 384,
            client: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(timeout_seconds))
                .build(),
        })
    }

    /// Return the fixed multilingual-e5-small output dimension.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn prefixed(intent: E5Intent, text: &str) -> String {
        format!("{}{}", intent.prefix(), text.trim())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let response = self
            .client
            .post(&self.tokenize_endpoint)
            .set("Content-Type", "application/json")
            .send_json(serde_json::json!({"content": text, "add_special": true}))
            .map_err(|e| provider_http_error(&self.tokenize_endpoint, e, self.timeout_seconds))?;
        let value: serde_json::Value = response.into_json().map_err(|e| {
            provider_error(format!(
                "llama.cpp tokenize response was not valid JSON: {e}"
            ))
        })?;
        let mut tokens: Vec<u32> = value
            .get("tokens")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| provider_error("llama.cpp tokenize response is missing tokens[]"))?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        provider_error("llama.cpp tokenize response contains a non-integer token")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if tokens.is_empty() {
            return Err(provider_error("llama.cpp tokenizer returned no tokens"));
        }
        if tokens.len() > 512 {
            let eos = *tokens
                .last()
                .ok_or_else(|| provider_error("tokenizer returned no EOS token"))?;
            tokens.truncate(511);
            tokens.push(eos);
        }
        Ok(tokens)
    }

    /// Embed text after applying the E5 query or passage intent prefix.
    pub fn embed(&self, text: &str, intent: E5Intent) -> Result<Vec<f32>> {
        let prefixed = Self::prefixed(intent, text);
        let tokens = self.tokenize(&prefixed)?;
        let response = self
            .client
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .send_json(serde_json::json!({"model": self.model, "input": tokens}))
            .map_err(|e| provider_http_error(&self.endpoint, e, self.timeout_seconds))?;
        let value: serde_json::Value = response.into_json().map_err(|e| {
            provider_error(format!(
                "llama.cpp embedding response was not valid JSON: {e}"
            ))
        })?;
        let items = value
            .get("data")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| provider_error("llama.cpp embedding response is missing data[]"))?;
        let vector = items
            .first()
            .and_then(|item| item.get("embedding"))
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                provider_error("llama.cpp embedding response is missing data[0].embedding[]")
            })?;
        if vector.len() != self.dimensions {
            return Err(provider_error(format!(
                "llama.cpp embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        let mut result = Vec::with_capacity(vector.len());
        for value in vector {
            let value = value
                .as_f64()
                .ok_or_else(|| provider_error("llama.cpp embedding contains a non-number"))?
                as f32;
            if !value.is_finite() {
                return Err(provider_error(
                    "llama.cpp embedding contains NaN or infinity",
                ));
            }
            result.push(value);
        }
        let norm = result
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(provider_error(
                "llama.cpp embedding has zero or invalid norm",
            ));
        }
        for value in &mut result {
            *value = (*value as f64 / norm) as f32;
        }
        Ok(result)
    }

    /// Embed a document passage using the `passage:` intent.
    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text, E5Intent::Document)
    }
    /// Embed a query using the `query:` intent.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text, E5Intent::Query)
    }
    /// Embed a batch in input order.
    pub fn embed_batch(&self, texts: &[&str], intent: E5Intent) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let token_batches = texts
            .iter()
            .map(|text| self.tokenize(&Self::prefixed(intent, text)))
            .collect::<Result<Vec<_>>>()?;
        let response = self
            .client
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .send_json(serde_json::json!({
                "model": self.model,
                "input": token_batches,
            }))
            .map_err(|e| provider_http_error(&self.endpoint, e, self.timeout_seconds))?;
        let value: serde_json::Value = response.into_json().map_err(|e| {
            provider_error(format!(
                "llama.cpp embedding response was not valid JSON: {e}"
            ))
        })?;
        let items = value
            .get("data")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| provider_error("llama.cpp embedding response is missing data[]"))?;
        if items.len() != texts.len() {
            return Err(provider_error(format!(
                "llama.cpp embedding batch returned {} vectors for {} inputs",
                items.len(),
                texts.len()
            )));
        }
        let mut ordered: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for (position, item) in items.iter().enumerate() {
            let index = item
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| {
                    provider_error(format!(
                        "llama.cpp embedding batch item {position} is missing integer index"
                    ))
                })?;
            if index >= texts.len() || ordered[index].is_some() {
                return Err(provider_error(format!(
                    "llama.cpp embedding batch returned invalid or duplicate index {index}"
                )));
            }
            let vector = item
                .get("embedding")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    provider_error("llama.cpp embedding batch item is missing embedding[]")
                })?;
            if vector.len() != self.dimensions {
                return Err(provider_error(format!(
                    "llama.cpp embedding dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    vector.len()
                )));
            }
            let mut normalized = Vec::with_capacity(vector.len());
            for value in vector {
                let value = value
                    .as_f64()
                    .ok_or_else(|| provider_error("llama.cpp embedding contains a non-number"))?
                    as f32;
                if !value.is_finite() {
                    return Err(provider_error(
                        "llama.cpp embedding contains NaN or infinity",
                    ));
                }
                normalized.push(value);
            }
            let norm = normalized
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() || norm == 0.0 {
                return Err(provider_error(
                    "llama.cpp embedding has zero or invalid norm",
                ));
            }
            for value in &mut normalized {
                *value = (*value as f64 / norm) as f32;
            }
            ordered[index] = Some(normalized);
        }
        ordered
            .into_iter()
            .map(|vector| {
                vector
                    .ok_or_else(|| provider_error("llama.cpp embedding batch has a missing index"))
            })
            .collect()
    }
}

/// E5 intent prefix applied before tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum E5Intent {
    /// Prefix with `query: `.
    Query,
    /// Prefix with `passage: `.
    Document,
}

impl E5Intent {
    fn prefix(self) -> &'static str {
        match self {
            Self::Query => "query: ",
            Self::Document => "passage: ",
        }
    }
}
