//! Ollama HTTP client for WASM
//!
//! This module provides HTTP-based integration with Ollama server running on localhost.
//! It's an alternative to WebLLM for users who want to use their local GPU via Ollama.
//!
//! Architecture:
//! Browser (WASM) → HTTP Request → Ollama Server (localhost:11434) → Response

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

/// Configuration for Ollama HTTP client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaHttpConfig {
    /// Ollama server endpoint (default: http://localhost:11434)
    pub endpoint: String,

    /// Model to use (e.g., "llama3.1:8b", "qwen2.5:7b")
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

impl Default for OllamaHttpConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".to_string(),
            model: "llama3.1:8b".to_string(),
            temperature: 0.7,
            max_tokens: 2000,
            system_prompt: None,
        }
    }
}

/// Request body for Ollama /api/generate endpoint
#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

/// Request body for Ollama /api/chat endpoint
#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Ollama generation options
#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

/// Response from Ollama /api/generate endpoint
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaGenerateResponse {
    response: String,
    #[allow(dead_code)]
    done: bool,
    #[serde(default)]
    #[allow(dead_code)]
    total_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    load_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    eval_count: Option<u32>,
}

/// Response from Ollama /api/chat endpoint
#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: ChatMessageResponse,
    #[allow(dead_code)]
    done: bool,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    #[allow(dead_code)]
    role: String,
    content: String,
}

/// Ollama HTTP client for WASM
#[wasm_bindgen]
pub struct OllamaHttpClient {
    config: OllamaHttpConfig,
}

#[wasm_bindgen]
impl OllamaHttpClient {
    /// Create a new Ollama HTTP client with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: OllamaHttpConfig::default(),
        }
    }

    /// Create client with custom endpoint and model
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(endpoint: String, model: String) -> Self {
        Self {
            config: OllamaHttpConfig {
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

    /// Generate text completion using /api/generate
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&self, prompt: String) -> Result<String, JsValue> {
        let request_body = OllamaGenerateRequest {
            model: self.config.model.clone(),
            prompt,
            system: self.config.system_prompt.clone(),
            stream: false, // Non-streaming for simplicity
            options: Some(OllamaOptions {
                temperature: Some(self.config.temperature),
                num_predict: Some(self.config.max_tokens),
            }),
        };

        let url = format!("{}/api/generate", self.config.endpoint);
        let response = self.make_request(&url, &request_body).await?;

        let response_text = Self::get_response_text(response).await?;
        let ollama_response: OllamaGenerateResponse = serde_json::from_str(&response_text)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse response: {}", e)))?;

        Ok(ollama_response.response)
    }

    /// Generate chat-style completion using /api/chat
    #[wasm_bindgen(js_name = chat)]
    pub async fn chat(&self, user_message: String) -> Result<String, JsValue> {
        let mut messages = Vec::new();

        // Add system prompt if configured
        if let Some(ref system) = self.config.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Add user message
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_message,
        });

        let request_body = OllamaChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: false,
            options: Some(OllamaOptions {
                temperature: Some(self.config.temperature),
                num_predict: Some(self.config.max_tokens),
            }),
        };

        let url = format!("{}/api/chat", self.config.endpoint);
        let response = self.make_request(&url, &request_body).await?;

        let response_text = Self::get_response_text(response).await?;
        let ollama_response: OllamaChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse response: {}", e)))?;

        Ok(ollama_response.message.content)
    }

    /// Check if Ollama server is available
    #[wasm_bindgen(js_name = checkAvailability)]
    pub async fn check_availability(&self) -> Result<bool, JsValue> {
        let url = format!("{}/api/tags", self.config.endpoint);

        match self.make_get_request(&url).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get list of available models
    #[wasm_bindgen(js_name = listModels)]
    pub async fn list_models(&self) -> Result<JsValue, JsValue> {
        let url = format!("{}/api/tags", self.config.endpoint);
        let response = self.make_get_request(&url).await?;
        let response_text = Self::get_response_text(response).await?;

        // Return as JsValue to be parsed in JavaScript
        Ok(JsValue::from_str(&response_text))
    }
}

// Private helper methods
impl OllamaHttpClient {
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

impl Default for OllamaHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OllamaHttpConfig::default();
        assert_eq!(config.endpoint, "http://localhost:11434");
        assert_eq!(config.model, "llama3.1:8b");
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 2000);
    }

    #[test]
    fn test_client_creation() {
        let client = OllamaHttpClient::new();
        assert_eq!(client.config.model, "llama3.1:8b");
    }

    #[test]
    fn test_custom_config() {
        let client = OllamaHttpClient::with_config(
            "http://localhost:11434".to_string(),
            "qwen2.5:7b".to_string(),
        );
        assert_eq!(client.config.model, "qwen2.5:7b");
    }
}
