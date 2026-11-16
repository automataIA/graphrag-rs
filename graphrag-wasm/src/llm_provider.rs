//! LLM Provider abstraction for WASM
//!
//! This module provides a unified interface for different LLM backends in WASM:
//! - WebLLM (100% in-browser, GPU via WebGPU)
//! - llama.cpp HTTP (local llama-server, OpenAI-compatible API)
//!
//! This allows users to choose between privacy (WebLLM) and performance (llama.cpp).

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// LLM provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[wasm_bindgen]
pub enum LlmProviderType {
    /// WebLLM - 100% in-browser inference with WebGPU
    WebLLM,
    /// llama.cpp HTTP - Local llama-server via OpenAI-compatible API
    LlamaCppHttp,
}

/// Configuration for LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    /// Provider type
    pub provider: LlmProviderType,

    /// Model name (provider-specific)
    pub model: String,

    /// Temperature (0.0 - 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// System prompt (optional)
    pub system_prompt: Option<String>,

    /// llama-server base URL (for LlamaCppHttp provider)
    #[serde(default = "default_llm_endpoint")]
    pub llm_endpoint: String,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    2000
}

fn default_llm_endpoint() -> String {
    crate::llama_http::DEFAULT_LLAMA_BASE_URL.to_string()
}

impl Default for LlmProviderConfig {
    fn default() -> Self {
        Self {
            provider: LlmProviderType::WebLLM,
            model: "Phi-3-mini-4k-instruct-q4f16_1-MLC".to_string(),
            temperature: 0.7,
            max_tokens: 2000,
            system_prompt: None,
            llm_endpoint: default_llm_endpoint(),
        }
    }
}

/// Async trait for LLM providers
///
/// Note: We can't use async_trait in WASM, so we use manual async methods
pub trait LlmProvider {
    /// Generate a response from a prompt
    ///
    /// Returns a Promise that resolves to the generated text
    fn generate(&self, prompt: String) -> js_sys::Promise;

    /// Check if the provider is available
    fn is_available(&self) -> js_sys::Promise;

    /// Get the provider name
    fn provider_name(&self) -> String;

    /// Get the current model
    fn model_name(&self) -> String;
}

/// Unified LLM client that wraps WebLLM or llama.cpp HTTP
#[wasm_bindgen]
pub struct UnifiedLlmClient {
    config: LlmProviderConfig,
    webllm_client: Option<crate::webllm::WebLLMClient>,
    llama_client: Option<crate::llama_http::LlamaCppHttpClient>,
}

#[wasm_bindgen]
impl UnifiedLlmClient {
    /// Create a new unified LLM client with WebLLM (default)
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: LlmProviderConfig::default(),
            webllm_client: Some(crate::webllm::WebLLMClient::new()),
            llama_client: None,
        }
    }

    /// Create client configured for WebLLM
    #[wasm_bindgen(js_name = withWebLLM)]
    pub fn with_webllm(model: String) -> Self {
        let config = LlmProviderConfig {
            provider: LlmProviderType::WebLLM,
            model: model.clone(),
            ..Default::default()
        };

        let mut webllm_client = crate::webllm::WebLLMClient::new();
        webllm_client.set_model(model);

        Self {
            config,
            webllm_client: Some(webllm_client),
            llama_client: None,
        }
    }

    /// Create client configured for llama.cpp HTTP (llama-server)
    #[wasm_bindgen(js_name = withLlamaCpp)]
    pub fn with_llama_cpp(endpoint: String, model: String) -> Self {
        let config = LlmProviderConfig {
            provider: LlmProviderType::LlamaCppHttp,
            model: model.clone(),
            llm_endpoint: endpoint.clone(),
            ..Default::default()
        };

        let llama_client = crate::llama_http::LlamaCppHttpClient::with_config(endpoint, model);

        Self {
            config,
            webllm_client: None,
            llama_client: Some(llama_client),
        }
    }

    /// Set temperature for generation
    #[wasm_bindgen(js_name = setTemperature)]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature.clamp(0.0, 1.0);

        if let Some(ref mut webllm) = self.webllm_client {
            webllm.set_temperature(temperature);
        }
        if let Some(ref mut llama) = self.llama_client {
            llama.set_temperature(temperature);
        }
    }

    /// Set system prompt
    #[wasm_bindgen(js_name = setSystemPrompt)]
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.config.system_prompt = Some(prompt.clone());

        if let Some(ref mut webllm) = self.webllm_client {
            webllm.set_system_prompt(prompt.clone());
        }
        if let Some(ref mut llama) = self.llama_client {
            llama.set_system_prompt(prompt);
        }
    }

    /// Get current provider type
    #[wasm_bindgen(js_name = getProvider)]
    pub fn get_provider(&self) -> String {
        match self.config.provider {
            LlmProviderType::WebLLM => "WebLLM".to_string(),
            LlmProviderType::LlamaCppHttp => "llama.cpp HTTP".to_string(),
        }
    }

    /// Get current model name
    #[wasm_bindgen(js_name = getModel)]
    pub fn get_model(&self) -> String {
        self.config.model.clone()
    }

    /// Generate text response
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&mut self, prompt: String) -> Result<String, JsValue> {
        match self.config.provider {
            LlmProviderType::WebLLM => {
                if let Some(ref mut webllm) = self.webllm_client {
                    webllm.generate(prompt).await
                } else {
                    Err(JsValue::from_str("WebLLM client not initialized"))
                }
            },
            LlmProviderType::LlamaCppHttp => {
                if let Some(ref llama) = self.llama_client {
                    llama.generate(prompt).await
                } else {
                    Err(JsValue::from_str("llama.cpp client not initialized"))
                }
            },
        }
    }

    /// Generate chat-style response
    #[wasm_bindgen(js_name = chat)]
    pub async fn chat(&mut self, message: String) -> Result<String, JsValue> {
        match self.config.provider {
            LlmProviderType::WebLLM => {
                if let Some(ref mut webllm) = self.webllm_client {
                    webllm.chat(message).await
                } else {
                    Err(JsValue::from_str("WebLLM client not initialized"))
                }
            },
            LlmProviderType::LlamaCppHttp => {
                if let Some(ref llama) = self.llama_client {
                    llama.chat(message).await
                } else {
                    Err(JsValue::from_str("llama.cpp client not initialized"))
                }
            },
        }
    }

    /// Check if the provider is available and ready
    #[wasm_bindgen(js_name = checkAvailability)]
    pub async fn check_availability(&self) -> Result<bool, JsValue> {
        match self.config.provider {
            LlmProviderType::WebLLM => {
                // WebLLM is always "available" but may need initialization
                Ok(true)
            },
            LlmProviderType::LlamaCppHttp => {
                if let Some(ref llama) = self.llama_client {
                    llama.check_availability().await
                } else {
                    Ok(false)
                }
            },
        }
    }
}

impl Default for UnifiedLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmProviderConfig::default();
        assert_eq!(config.provider, LlmProviderType::WebLLM);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 2000);
    }

    #[test]
    fn test_llama_cpp_config() {
        let config = LlmProviderConfig {
            provider: LlmProviderType::LlamaCppHttp,
            model: "qwen3.5-9b".to_string(),
            ..Default::default()
        };
        assert_eq!(config.provider, LlmProviderType::LlamaCppHttp);
        assert_eq!(config.llm_endpoint, "http://127.0.0.1:8080/v1");
        assert_eq!(config.model, "qwen3.5-9b");
    }

    #[test]
    fn test_client_creation() {
        let client = UnifiedLlmClient::new();
        assert_eq!(client.get_provider(), "WebLLM");
    }
}
