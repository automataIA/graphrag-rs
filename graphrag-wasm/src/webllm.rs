//! WebLLM Bindings for GPU-Accelerated LLM
//!
//! Provides Rust bindings to WebLLM (https://github.com/mlc-ai/web-llm),
//! a production-ready WebGPU-accelerated LLM runtime for browsers.
//!
//! ## Performance
//! - 40-62 tokens/second with WebGPU
//! - Supports Llama 3, Phi-3, Gemma, Qwen models
//! - Progressive model loading with cache
//!
//! ## Setup
//!
//! Add to your `index.html`:
//!
//! ```html
//! <script type="module">
//!   import * as webllm from "https://esm.run/@mlc-ai/web-llm";
//!   window.webllm = webllm;
//! </script>
//! ```
//!
//! ## Usage
//!
//! ```rust
//! let llm = WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await?;
//! let response = llm.chat("Hello!", |progress, text| {
//!     console_log!("Loading: {}% - {}", progress * 100.0, text);
//! }).await?;
//! ```

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use serde::{Deserialize, Serialize};

/// Message for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Create an assistant message
    #[allow(dead_code)]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

/// WebLLM error types
#[derive(Debug, Clone)]
pub enum WebLLMError {
    /// WebLLM not loaded in window
    NotLoaded,
    /// Model initialization failed
    InitializationFailed(String),
    /// Inference failed
    InferenceFailed(String),
    /// WebGPU not available
    #[allow(dead_code)]
    WebGPUNotAvailable,
}

impl std::fmt::Display for WebLLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WebLLMError::NotLoaded => write!(f, "WebLLM not loaded. Add <script> tag to index.html"),
            WebLLMError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            WebLLMError::InferenceFailed(msg) => write!(f, "Inference failed: {}", msg),
            WebLLMError::WebGPUNotAvailable => write!(f, "WebGPU not available in this browser"),
        }
    }
}

impl From<JsValue> for WebLLMError {
    fn from(value: JsValue) -> Self {
        WebLLMError::InferenceFailed(
            value.as_string().unwrap_or_else(|| "Unknown error".to_string())
        )
    }
}

/// WebLLM engine wrapper
///
/// Provides async interface to WebLLM for GPU-accelerated LLM inference.
pub struct WebLLM {
    engine: JsValue,
    model_id: String,
}

// SAFETY: In WASM, there are no threads, so Send and Sync are safe
// These are needed for Leptos signals which require Send + Sync
unsafe impl Send for WebLLM {}
unsafe impl Sync for WebLLM {}

impl WebLLM {
    /// Initialize WebLLM with a model
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "Phi-3-mini-4k-instruct-q4f16_1-MLC")
    ///
    /// # Available Models
    /// - "Llama-3.2-1B-Instruct-q4f16_1-MLC" (fast, 62 tok/s)
    /// - "Phi-3-mini-4k-instruct-q4f16_1-MLC" (balanced, 40 tok/s)
    /// - "Qwen2-1.5B-Instruct-q4f16_1-MLC" (compact, 50 tok/s)
    ///
    /// # Returns
    /// WebLLM instance or error
    pub async fn new(model_id: &str) -> Result<Self, WebLLMError> {
        // Check if WebLLM is loaded
        let window = web_sys::window()
            .ok_or(WebLLMError::NotLoaded)?;

        let webllm = js_sys::Reflect::get(&window, &JsValue::from_str("webllm"))
            .map_err(|_| WebLLMError::NotLoaded)?;

        if webllm.is_undefined() {
            return Err(WebLLMError::NotLoaded);
        }

        // Get CreateMLCEngine function
        let create_engine_fn = js_sys::Reflect::get(&webllm, &JsValue::from_str("CreateMLCEngine"))
            .map_err(|_| WebLLMError::NotLoaded)?;

        // Call CreateMLCEngine(model_id)
        let engine_promise = js_sys::Function::from(create_engine_fn)
            .call1(&webllm, &JsValue::from_str(model_id))
            .map_err(|e| WebLLMError::InitializationFailed(format!("{:?}", e)))?;

        // Wait for engine to initialize
        let engine = JsFuture::from(js_sys::Promise::from(engine_promise))
            .await
            .map_err(|e| WebLLMError::InitializationFailed(format!("{:?}", e)))?;

        Ok(Self {
            engine,
            model_id: model_id.to_string(),
        })
    }

    /// Initialize with progress callback
    ///
    /// # Arguments
    /// * `model_id` - Model identifier
    /// * `on_progress` - Callback for progress updates (progress: f64, text: String)
    pub async fn new_with_progress<F>(
        model_id: &str,
        on_progress: F,
    ) -> Result<Self, WebLLMError>
    where
        F: Fn(f64, String) + 'static,
    {
        let window = web_sys::window()
            .ok_or(WebLLMError::NotLoaded)?;

        let webllm = js_sys::Reflect::get(&window, &JsValue::from_str("webllm"))
            .map_err(|_| WebLLMError::NotLoaded)?;

        if webllm.is_undefined() {
            return Err(WebLLMError::NotLoaded);
        }

        // Create progress callback
        let callback = Closure::wrap(Box::new(move |report: JsValue| {
            if let Ok(obj) = report.dyn_into::<js_sys::Object>() {
                let text = js_sys::Reflect::get(&obj, &JsValue::from_str("text"))
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_else(|| "Loading...".to_string());

                let progress = js_sys::Reflect::get(&obj, &JsValue::from_str("progress"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                on_progress(progress, text);
            }
        }) as Box<dyn Fn(JsValue)>);

        // Create config with progress callback
        let config = js_sys::Object::new();
        js_sys::Reflect::set(
            &config,
            &JsValue::from_str("initProgressCallback"),
            callback.as_ref().unchecked_ref(),
        ).map_err(|_| WebLLMError::InitializationFailed("Failed to set callback".to_string()))?;

        // Get CreateMLCEngine function
        let create_engine_fn = js_sys::Reflect::get(&webllm, &JsValue::from_str("CreateMLCEngine"))
            .map_err(|_| WebLLMError::NotLoaded)?;

        // Call CreateMLCEngine(model_id, config)
        let engine_promise = js_sys::Function::from(create_engine_fn)
            .call2(&webllm, &JsValue::from_str(model_id), &config.into())
            .map_err(|e| WebLLMError::InitializationFailed(format!("{:?}", e)))?;

        let engine = JsFuture::from(js_sys::Promise::from(engine_promise))
            .await
            .map_err(|e| WebLLMError::InitializationFailed(format!("{:?}", e)))?;

        callback.forget(); // Keep callback alive

        Ok(Self {
            engine,
            model_id: model_id.to_string(),
        })
    }

    /// Send a chat message and get response
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages
    /// * `temperature` - Sampling temperature (0.0-2.0, default 0.7)
    /// * `max_tokens` - Maximum tokens to generate (default 512)
    ///
    /// # Returns
    /// Assistant's response text
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<String, WebLLMError> {
        // Create messages array
        let messages_array = js_sys::Array::new();
        for msg in messages {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &JsValue::from_str("role"), &JsValue::from_str(&msg.role))
                .map_err(|_| WebLLMError::InferenceFailed("Failed to set role".to_string()))?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("content"), &JsValue::from_str(&msg.content))
                .map_err(|_| WebLLMError::InferenceFailed("Failed to set content".to_string()))?;
            messages_array.push(&obj);
        }

        // Get engine.chat.completions.create
        let chat = js_sys::Reflect::get(&self.engine, &JsValue::from_str("chat"))?;
        let completions = js_sys::Reflect::get(&chat, &JsValue::from_str("completions"))?;
        let create_fn = js_sys::Reflect::get(&completions, &JsValue::from_str("create"))?;

        // Build request object
        let request = js_sys::Object::new();
        js_sys::Reflect::set(&request, &JsValue::from_str("messages"), &messages_array)?;
        js_sys::Reflect::set(&request, &JsValue::from_str("stream"), &JsValue::from_bool(false))?;
        js_sys::Reflect::set(
            &request,
            &JsValue::from_str("temperature"),
            &JsValue::from_f64(temperature.unwrap_or(0.7)),
        )?;
        if let Some(max_tokens) = max_tokens {
            js_sys::Reflect::set(
                &request,
                &JsValue::from_str("max_tokens"),
                &JsValue::from_f64(max_tokens as f64),
            )?;
        }

        // Call create()
        let promise = js_sys::Reflect::apply(
            &create_fn.into(),
            &completions,
            &js_sys::Array::of1(&request),
        )?;
        let result = JsFuture::from(js_sys::Promise::from(promise)).await?;

        // Extract response
        let choices = js_sys::Reflect::get(&result, &JsValue::from_str("choices"))?;
        let first = js_sys::Reflect::get(&choices, &0_u32.into())?;
        let message = js_sys::Reflect::get(&first, &JsValue::from_str("message"))?;
        let content = js_sys::Reflect::get(&message, &JsValue::from_str("content"))?;

        content.as_string().ok_or(WebLLMError::InferenceFailed("No content in response".to_string()))
    }

    /// Simple chat with a single user message
    ///
    /// # Arguments
    /// * `user_message` - User's message
    ///
    /// # Returns
    /// Assistant's response
    pub async fn ask(&self, user_message: &str) -> Result<String, WebLLMError> {
        self.chat(vec![ChatMessage::user(user_message)], None, None).await
    }

    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Stream chat response with real-time token generation
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages
    /// * `on_chunk` - Callback for each generated token chunk
    /// * `temperature` - Sampling temperature (0.0-2.0, default 0.7)
    /// * `max_tokens` - Maximum tokens to generate (default 512)
    ///
    /// # Returns
    /// Complete response text
    pub async fn chat_stream<F>(
        &self,
        messages: Vec<ChatMessage>,
        mut on_chunk: F,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<String, WebLLMError>
    where
        F: FnMut(String) + 'static,
    {
        // Create messages array
        let messages_array = js_sys::Array::new();
        for msg in messages {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &JsValue::from_str("role"), &JsValue::from_str(&msg.role))
                .map_err(|_| WebLLMError::InferenceFailed("Failed to set role".to_string()))?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("content"), &JsValue::from_str(&msg.content))
                .map_err(|_| WebLLMError::InferenceFailed("Failed to set content".to_string()))?;
            messages_array.push(&obj);
        }

        // Get engine.chat.completions.create
        let chat = js_sys::Reflect::get(&self.engine, &JsValue::from_str("chat"))?;
        let completions = js_sys::Reflect::get(&chat, &JsValue::from_str("completions"))?;
        let create_fn = js_sys::Reflect::get(&completions, &JsValue::from_str("create"))?;

        // Build request object with streaming enabled
        let request = js_sys::Object::new();
        js_sys::Reflect::set(&request, &JsValue::from_str("messages"), &messages_array)?;
        js_sys::Reflect::set(&request, &JsValue::from_str("stream"), &JsValue::from_bool(true))?;
        js_sys::Reflect::set(
            &request,
            &JsValue::from_str("temperature"),
            &JsValue::from_f64(temperature.unwrap_or(0.7)),
        )?;
        if let Some(max_tokens) = max_tokens {
            js_sys::Reflect::set(
                &request,
                &JsValue::from_str("max_tokens"),
                &JsValue::from_f64(max_tokens as f64),
            )?;
        }

        // Call create() - returns AsyncIterator
        let promise = js_sys::Reflect::apply(
            &create_fn.into(),
            &completions,
            &js_sys::Array::of1(&request),
        )?;
        let async_iter = JsFuture::from(js_sys::Promise::from(promise)).await?;

        // Iterate over chunks
        let mut full_response = String::new();
        let iter = js_sys::try_iter(&async_iter)
            .map_err(|_| WebLLMError::InferenceFailed("Failed to get iterator".to_string()))?
            .ok_or(WebLLMError::InferenceFailed("No iterator returned".to_string()))?;

        for chunk_result in iter {
            let chunk = chunk_result.map_err(|_| WebLLMError::InferenceFailed("Iteration error".to_string()))?;

            // Extract delta content from chunk
            if let Ok(choices) = js_sys::Reflect::get(&chunk, &JsValue::from_str("choices")) {
                if let Ok(first) = js_sys::Reflect::get(&choices, &0_u32.into()) {
                    if let Ok(delta) = js_sys::Reflect::get(&first, &JsValue::from_str("delta")) {
                        if let Ok(content) = js_sys::Reflect::get(&delta, &JsValue::from_str("content")) {
                            if let Some(text) = content.as_string() {
                                full_response.push_str(&text);
                                on_chunk(text);
                            }
                        }
                    }
                }
            }
        }

        Ok(full_response)
    }
}

/// WASM bindings for WebLLM
#[wasm_bindgen]
pub struct WasmWebLLM {
    inner: Option<WebLLM>,
}

#[wasm_bindgen]
impl WasmWebLLM {
    /// Initialize WebLLM
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "Phi-3-mini-4k-instruct-q4f16_1-MLC")
    pub async fn new(model_id: String) -> Result<WasmWebLLM, JsValue> {
        let llm = WebLLM::new(&model_id).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmWebLLM {
            inner: Some(llm),
        })
    }

    /// Initialize with progress callback
    ///
    /// # Arguments
    /// * `model_id` - Model identifier
    /// * `on_progress` - JavaScript callback function(progress: number, text: string)
    pub async fn new_with_progress(model_id: String, on_progress: js_sys::Function) -> Result<WasmWebLLM, JsValue> {
        let llm = WebLLM::new_with_progress(&model_id, move |progress, text| {
            let _ = on_progress.call2(
                &JsValue::NULL,
                &JsValue::from_f64(progress),
                &JsValue::from_str(&text),
            );
        }).await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmWebLLM {
            inner: Some(llm),
        })
    }

    /// Send a simple message and get response
    ///
    /// # Arguments
    /// * `message` - User message
    ///
    /// # Returns
    /// Assistant's response
    pub async fn ask(&self, message: String) -> Result<String, JsValue> {
        if let Some(llm) = &self.inner {
            llm.ask(&message).await
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Err(JsValue::from_str("LLM not initialized"))
        }
    }

    /// Send chat messages and get response
    ///
    /// # Arguments
    /// * `messages` - JSON array of {role: string, content: string}
    /// * `temperature` - Sampling temperature (optional)
    /// * `max_tokens` - Maximum tokens (optional)
    pub async fn chat(&self, messages: JsValue, temperature: Option<f64>, max_tokens: Option<u32>) -> Result<String, JsValue> {
        if let Some(llm) = &self.inner {
            let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages)
                .map_err(|e| JsValue::from_str(&format!("Failed to parse messages: {}", e)))?;

            llm.chat(msgs, temperature, max_tokens).await
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Err(JsValue::from_str("LLM not initialized"))
        }
    }

    /// Get the model ID
    pub fn model_id(&self) -> String {
        self.inner.as_ref().map(|llm| llm.model_id().to_string()).unwrap_or_default()
    }

    /// Stream chat response with real-time token generation
    ///
    /// # Arguments
    /// * `messages` - JSON array of {role: string, content: string}
    /// * `on_chunk` - JavaScript callback function(chunk: string)
    /// * `temperature` - Sampling temperature (optional)
    /// * `max_tokens` - Maximum tokens (optional)
    ///
    /// # Returns
    /// Complete response text
    pub async fn chat_stream(
        &self,
        messages: JsValue,
        on_chunk: js_sys::Function,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<String, JsValue> {
        if let Some(llm) = &self.inner {
            let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages)
                .map_err(|e| JsValue::from_str(&format!("Failed to parse messages: {}", e)))?;

            llm.chat_stream(
                msgs,
                move |chunk| {
                    let _ = on_chunk.call1(&JsValue::NULL, &JsValue::from_str(&chunk));
                },
                temperature,
                max_tokens,
            )
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Err(JsValue::from_str("LLM not initialized"))
        }
    }
}

/// List of recommended WebLLM models
#[wasm_bindgen]
pub fn get_recommended_models() -> JsValue {
    let models = vec![
        serde_json::json!({
            "id": "Llama-3.2-1B-Instruct-q4f16_1-MLC",
            "name": "Llama 3.2 1B",
            "size": "1.2GB",
            "speed": "62 tok/s",
            "description": "Fast and efficient, great for most tasks"
        }),
        serde_json::json!({
            "id": "Phi-3-mini-4k-instruct-q4f16_1-MLC",
            "name": "Phi-3 Mini",
            "size": "2.4GB",
            "speed": "40 tok/s",
            "description": "Balanced performance and quality"
        }),
        serde_json::json!({
            "id": "Qwen2-1.5B-Instruct-q4f16_1-MLC",
            "name": "Qwen2 1.5B",
            "size": "1.6GB",
            "speed": "50 tok/s",
            "description": "Compact and fast Chinese/English model"
        }),
        serde_json::json!({
            "id": "gemma-2b-it-q4f16_1-MLC",
            "name": "Gemma 2B",
            "size": "2.0GB",
            "speed": "45 tok/s",
            "description": "Google's efficient instruction-tuned model"
        }),
    ];

    serde_wasm_bindgen::to_value(&models).unwrap_or(JsValue::NULL)
}

/// Check if WebLLM is available in the browser
#[wasm_bindgen]
pub fn is_webllm_available() -> bool {
    if let Some(window) = web_sys::window() {
        if let Ok(webllm) = js_sys::Reflect::get(&window, &JsValue::from_str("webllm")) {
            return !webllm.is_undefined();
        }
    }
    false
}

/// WebLLM Client wrapper for compatibility with LLM provider abstraction
///
/// This provides a similar API to OllamaHttpClient for seamless integration
#[wasm_bindgen]
pub struct WebLLMClient {
    model: String,
    temperature: f32,
    system_prompt: Option<String>,
    engine: Option<WebLLM>,
}

#[wasm_bindgen]
impl WebLLMClient {
    /// Create a new WebLLM client with default model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: "Phi-3-mini-4k-instruct-q4f16_1-MLC".to_string(),
            temperature: 0.7,
            system_prompt: None,
            engine: None,
        }
    }

    /// Set the model to use
    #[wasm_bindgen(js_name = setModel)]
    pub fn set_model(&mut self, model: String) {
        self.model = model;
        self.engine = None; // Reset engine when model changes
    }

    /// Set the temperature (0.0 - 1.0)
    #[wasm_bindgen(js_name = setTemperature)]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.clamp(0.0, 1.0);
    }

    /// Set system prompt
    #[wasm_bindgen(js_name = setSystemPrompt)]
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
    }

    /// Ensure engine is initialized
    async fn ensure_initialized(&mut self) -> Result<(), JsValue> {
        if self.engine.is_none() {
            let engine = WebLLM::new(&self.model).await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            self.engine = Some(engine);
        }
        Ok(())
    }

    /// Generate text completion
    #[wasm_bindgen(js_name = generate)]
    pub async fn generate(&mut self, prompt: String) -> Result<String, JsValue> {
        self.ensure_initialized().await?;

        let engine = self.engine.as_ref()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        let mut messages = Vec::new();

        // Add system prompt if configured
        if let Some(ref system) = self.system_prompt {
            messages.push(ChatMessage::system(system.clone()));
        }

        // Add user prompt
        messages.push(ChatMessage::user(prompt));

        engine.chat(messages, Some(self.temperature as f64), None).await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Generate chat-style response
    #[wasm_bindgen(js_name = chat)]
    pub async fn chat(&mut self, message: String) -> Result<String, JsValue> {
        self.generate(message).await
    }
}

impl Default for WebLLMClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_chat_message() {
        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }
}
