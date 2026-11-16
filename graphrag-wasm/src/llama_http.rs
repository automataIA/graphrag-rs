//! llama.cpp (OpenAI-compatible) HTTP client for WASM
//!
//! This module provides HTTP-based integration with a `llama-server` running on
//! localhost. It's an alternative to WebLLM for users who want to use their
//! local GPU via llama.cpp.
//!
//! Architecture:
//! Browser (WASM) → HTTP Request → llama-server (127.0.0.1:8080/v1) → Response
//!
//! No API key is sent: secrets must never be embedded in browser code. Use a
//! local llama-server (CORS is enabled by default) or a server-side proxy.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

/// Default llama-server OpenAI-compatible base URL.
pub const DEFAULT_LLAMA_BASE_URL: &str = "http://127.0.0.1:8080/v1";

/// Configuration for the llama.cpp HTTP client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppHttpConfig {
    /// OpenAI-compatible base URL (default: http://127.0.0.1:8080/v1)
    pub endpoint: String,

    /// Model to use (llama-server `--alias`, e.g. "qwen3.5-9b")
    pub model: String,

    /// Temperature for generation (0.0 - 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// System prompt (optional)
    pub system_prompt: Option<String>,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    2000
}

impl Default for LlamaCppHttpConfig {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_LLAMA_BASE_URL.to_string(),
            model: "qwen3.5-9b".to_string(),
            temperature: 0.7,
            max_tokens: 2000,
            system_prompt: None,
        }
    }
}

/// Request body for the OpenAI-compatible /chat/completions endpoint
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Response from /chat/completions
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    #[serde(default)]
    content: Option<String>,
}

/// llama.cpp HTTP client for WASM
#[wasm_bindgen]
pub struct LlamaCppHttpClient {
    config: LlamaCppHttpConfig,
}

#[wasm_bindgen]
impl LlamaCppHttpClient {
    /// Create a new llama.cpp HTTP client with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: LlamaCppHttpConfig::default(),
        }
    }

    /// Create client with custom base URL and model
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(endpoint: String, model: String) -> Self {
        Self {
            config: LlamaCppHttpConfig {
                endpoint,
                model,
                ..Default::default()
            },
        }
    }

    /// Set the model to use
    #[wasm_bindgen(js_name = setModel)]
    pub fn set_model(&mut self, model: String) {
        self.config.model = model;
    }

    /// Set the temperature (0.0 - 1.0)
    #[wasm_bindgen(js_name = setTemperature)]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature.clamp(0.0, 1.0);
    }

    /// Set system prompt
    #[wasm_bindgen(js_name = setSystemPrompt)]
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.config.system_prompt = Some(prompt);
    }

    /// Generate a completion for a single prompt (one user turn)
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&self, prompt: String) -> Result<String, JsValue> {
        self.chat(prompt).await
    }

    /// Generate chat-style completion using /chat/completions
    #[wasm_bindgen(js_name = chat)]
    pub async fn chat(&self, user_message: String) -> Result<String, JsValue> {
        let request_body = self.chat_request(user_message);
        let url = self.url("/chat/completions");
        let response = self.make_request(&url, &request_body).await?;

        let response_text = Self::get_response_text(response).await?;
        parse_chat_content(&response_text).map_err(|e| JsValue::from_str(&e))
    }

    /// Check if the llama.cpp server is available
    #[wasm_bindgen(js_name = checkAvailability)]
    pub async fn check_availability(&self) -> Result<bool, JsValue> {
        let url = self.url("/models");

        match self.make_get_request(&url).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get list of available models (raw `/models` JSON)
    #[wasm_bindgen(js_name = listModels)]
    pub async fn list_models(&self) -> Result<JsValue, JsValue> {
        let url = self.url("/models");
        let response = self.make_get_request(&url).await?;
        let response_text = Self::get_response_text(response).await?;

        // Return as JsValue to be parsed in JavaScript
        Ok(JsValue::from_str(&response_text))
    }
}

fn parse_chat_content(body: &str) -> Result<String, String> {
    let parsed: ChatCompletionResponse =
        serde_json::from_str(body).map_err(|e| format!("Failed to parse response: {}", e))?;
    parsed
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| "Response has no choices[0].message.content".to_string())
}

impl LlamaCppHttpClient {
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.config.endpoint.trim_end_matches('/'), path)
    }

    fn chat_request(&self, user_message: String) -> ChatCompletionRequest {
        let mut messages = Vec::new();
        if let Some(ref system) = self.config.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_message,
        });
        ChatCompletionRequest {
            model: self.config.model.clone(),
            messages,
            stream: false,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        }
    }
}

// Private helper methods
impl LlamaCppHttpClient {
    /// Make HTTP POST request
    async fn make_request<T: Serialize>(&self, url: &str, body: &T) -> Result<Response, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        // Serialize request body
        let body_str = serde_json::to_string(body)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize request: {}", e)))?;

        // Create request
        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body_str));

        let request = Request::new_with_str_and_init(url, &opts)?;
        request.headers().set("Content-Type", "application/json")?;

        // Send request
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let response: Response = resp_value.dyn_into()?;

        if !response.ok() {
            return Err(JsValue::from_str(&format!(
                "HTTP error: {} {}",
                response.status(),
                response.status_text()
            )));
        }

        Ok(response)
    }

    /// Make HTTP GET request
    async fn make_get_request(&self, url: &str) -> Result<Response, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts)?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let response: Response = resp_value.dyn_into()?;

        if !response.ok() {
            return Err(JsValue::from_str(&format!(
                "HTTP error: {} {}",
                response.status(),
                response.status_text()
            )));
        }

        Ok(response)
    }

    /// Get response text from Response object
    async fn get_response_text(response: Response) -> Result<String, JsValue> {
        let text_promise = response.text()?;
        let text_value = JsFuture::from(text_promise).await?;
        text_value
            .as_string()
            .ok_or_else(|| JsValue::from_str("Response is not a string"))
    }
}

impl Default for LlamaCppHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlamaCppHttpConfig::default();
        assert_eq!(config.endpoint, "http://127.0.0.1:8080/v1");
        assert_eq!(config.model, "qwen3.5-9b");
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 2000);
    }

    #[test]
    fn test_custom_config_and_urls() {
        let client = LlamaCppHttpClient::with_config(
            "http://127.0.0.1:8080/v1/".to_string(),
            "qwen3.5-9b".to_string(),
        );
        assert_eq!(client.config.model, "qwen3.5-9b");
        assert_eq!(
            client.url("/chat/completions"),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
    }

    #[test]
    fn test_chat_request_and_response() {
        let mut client = LlamaCppHttpClient::new();
        client.set_system_prompt("sys".to_string());
        let body = serde_json::to_value(client.chat_request("ciao".to_string())).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "ciao");
        assert_eq!(body["stream"], false);

        let reply = r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;
        assert_eq!(parse_chat_content(reply).unwrap(), "ok");
        assert!(parse_chat_content(r#"{"choices":[]}"#).is_err());
    }
}
