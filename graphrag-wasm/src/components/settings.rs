//! Settings component for configuring embeddings and LLM providers
//!
//! Allows users to:
//! - Choose embedding provider (ONNX, OpenAI, Voyage, Cohere, etc.)
//! - Select embedding model for that provider
//! - Choose LLM provider (WebLLM, Ollama)
//! - Select LLM model for that provider
//! - Cache settings in IndexedDB

use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use web_sys::{Event, HtmlInputElement, HtmlSelectElement};

// Helper functions for event handling
fn event_target_value(ev: &Event) -> String {
    ev.target()
        .and_then(|t| {
            // Try as input first
            if let Ok(input) = t.clone().dyn_into::<HtmlInputElement>() {
                Some(input.value())
            } else if let Ok(select) = t.dyn_into::<HtmlSelectElement>() {
                // Try as select
                Some(select.value())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn event_target_checked(ev: &Event) -> bool {
    ev.target()
        .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
        .map(|input| input.checked())
        .unwrap_or_default()
}

/// User settings for providers and models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserSettings {
    // Embedding settings
    pub embedding_provider: EmbeddingProviderType,
    pub embedding_model: String,
    pub embedding_api_key: Option<String>,

    // LLM settings
    pub llm_provider: LlmProviderType,
    pub llm_model: String,
    pub llm_endpoint: Option<String>, // For Ollama
    pub llm_temperature: f32,

    // UI preferences
    pub cache_models: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingProviderType {
    ONNX,           // Local ONNX Runtime Web
    OpenAI,         // OpenAI API
    VoyageAI,       // Voyage AI API
    Cohere,         // Cohere API
    JinaAI,         // Jina AI API
    Mistral,        // Mistral API
    TogetherAI,     // Together AI API
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LlmProviderType {
    WebLLM,         // In-browser WebLLM
    OllamaHTTP,     // Local Ollama server
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            embedding_provider: EmbeddingProviderType::ONNX,
            // ✅ Changed to MPNet-base-v2 (768d) for better semantic understanding
            // Recommended for philosophical texts like Plato's Symposium
            embedding_model: "all-mpnet-base-v2".to_string(),
            embedding_api_key: None,
            llm_provider: LlmProviderType::WebLLM,
            llm_model: "Phi-3-mini-4k-instruct-q4f16_1-MLC".to_string(),
            llm_endpoint: Some("http://localhost:11434".to_string()),
            // ✅ Reduced from 0.7 to 0.2 for factual accuracy
            // Lower temperature = more precise, less creative
            llm_temperature: 0.2,
            cache_models: true,
        }
    }
}

impl EmbeddingProviderType {
    /// Get available models for this provider
    pub fn available_models(&self) -> Vec<(&'static str, &'static str)> {
        match self {
            EmbeddingProviderType::ONNX => vec![
                ("all-MiniLM-L6-v2", "MiniLM-L6 (384 dim, fast)"),
                ("all-mpnet-base-v2", "MPNet base (768 dim, quality)"),
            ],
            EmbeddingProviderType::OpenAI => vec![
                ("text-embedding-3-small", "Small (1536 dim, $0.02/1M tokens)"),
                ("text-embedding-3-large", "Large (3072 dim, $0.13/1M tokens)"),
                ("text-embedding-ada-002", "Ada-002 (1536 dim, legacy)"),
            ],
            EmbeddingProviderType::VoyageAI => vec![
                ("voyage-3", "Voyage 3 (1024 dim, best quality)"),
                ("voyage-3-lite", "Voyage 3 Lite (512 dim, fast)"),
                ("voyage-code-2", "Voyage Code 2 (1536 dim, for code)"),
            ],
            EmbeddingProviderType::Cohere => vec![
                ("embed-english-v3.0", "English v3 (1024 dim)"),
                ("embed-multilingual-v3.0", "Multilingual v3 (1024 dim)"),
                ("embed-english-light-v3.0", "English Light (384 dim)"),
            ],
            EmbeddingProviderType::JinaAI => vec![
                ("jina-embeddings-v2-base-en", "Base EN (768 dim)"),
                ("jina-embeddings-v2-small-en", "Small EN (512 dim)"),
            ],
            EmbeddingProviderType::Mistral => vec![
                ("mistral-embed", "Mistral Embed (1024 dim)"),
            ],
            EmbeddingProviderType::TogetherAI => vec![
                ("togethercomputer/m2-bert-80M-8k-retrieval", "M2-BERT (768 dim)"),
                ("BAAI/bge-large-en-v1.5", "BGE Large (1024 dim)"),
            ],
        }
    }

    pub fn requires_api_key(&self) -> bool {
        !matches!(self, EmbeddingProviderType::ONNX)
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            EmbeddingProviderType::ONNX => "ONNX Runtime Web (Local)",
            EmbeddingProviderType::OpenAI => "OpenAI",
            EmbeddingProviderType::VoyageAI => "Voyage AI",
            EmbeddingProviderType::Cohere => "Cohere",
            EmbeddingProviderType::JinaAI => "Jina AI",
            EmbeddingProviderType::Mistral => "Mistral",
            EmbeddingProviderType::TogetherAI => "Together AI",
        }
    }
}

impl LlmProviderType {
    /// Get available models for this provider
    pub fn available_models(&self) -> Vec<(&'static str, &'static str)> {
        match self {
            LlmProviderType::WebLLM => vec![
                ("Llama-3.2-1B-Instruct-q4f16_1-MLC", "Llama 3.2 1B (1.2GB, 62 tok/s)"),
                ("Phi-3-mini-4k-instruct-q4f16_1-MLC", "Phi-3 Mini (2.4GB, 40 tok/s)"),
                ("Qwen2-1.5B-Instruct-q4f16_1-MLC", "Qwen2 1.5B (1.6GB, 50 tok/s)"),
                ("gemma-2b-it-q4f16_1-MLC", "Gemma 2B (2.0GB, 45 tok/s)"),
            ],
            LlmProviderType::OllamaHTTP => vec![
                ("llama3.1:8b", "Llama 3.1 8B (Best balance)"),
                ("qwen2.5:7b", "Qwen 2.5 7B (Reasoning)"),
                ("mistral:7b", "Mistral 7B (Fast)"),
                ("llama3.1:70b", "Llama 3.1 70B (Highest quality)"),
                ("qwen2.5:32b", "Qwen 2.5 32B (Advanced)"),
            ],
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            LlmProviderType::WebLLM => "WebLLM (In-Browser)",
            LlmProviderType::OllamaHTTP => "Ollama (Local Server)",
        }
    }
}

/// Settings component
#[component]
pub fn SettingsPanel() -> impl IntoView {
    let (settings, set_settings) = signal(UserSettings::default());
    let (saving, set_saving) = signal(false);
    let (save_status, set_save_status) = signal(String::new());

    // Load settings from IndexedDB on mount
    Effect::new(move |_| {
        spawn_local(async move {
            if let Ok(loaded_settings) = load_settings_from_storage().await {
                set_settings.set(loaded_settings);
            }
        });
    });

    // Save settings to IndexedDB
    let save_settings = move |_| {
        set_saving.set(true);
        let current_settings = settings.get();

        spawn_local(async move {
            match save_settings_to_storage(&current_settings).await {
                Ok(_) => {
                    set_save_status.set("✅ Settings saved successfully!".to_string());
                    set_saving.set(false);

                    // Clear success message after 3 seconds
                    let status_setter = set_save_status.clone();
                    spawn_local(async move {
                        gloo_timers::future::TimeoutFuture::new(3000).await;
                        status_setter.set(String::new());
                    });
                }
                Err(e) => {
                    set_save_status.set(format!("❌ Error: {}", e));
                    set_saving.set(false);
                }
            }
        });
    };

    view! {
        <div class="w-full max-w-4xl mx-auto p-6 space-y-6">
            <div class="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 shadow-lg">
                <h2 class="text-2xl font-bold text-purple-900 mb-6">
                    "⚙️ GraphRAG Settings"
                </h2>

                // Embedding Provider Section
                <div class="bg-white rounded-lg p-6 mb-6 shadow-sm">
                    <h3 class="text-xl font-semibold text-purple-800 mb-4">
                        "📊 Embedding Provider"
                    </h3>

                    // Provider selection
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            "Provider"
                        </label>
                        <select
                            class="select select-bordered w-full"
                            on:change=move |ev| {
                                let value = event_target_value(&ev);
                                let provider = match value.as_str() {
                                    "ONNX" => EmbeddingProviderType::ONNX,
                                    "OpenAI" => EmbeddingProviderType::OpenAI,
                                    "VoyageAI" => EmbeddingProviderType::VoyageAI,
                                    "Cohere" => EmbeddingProviderType::Cohere,
                                    "JinaAI" => EmbeddingProviderType::JinaAI,
                                    "Mistral" => EmbeddingProviderType::Mistral,
                                    "TogetherAI" => EmbeddingProviderType::TogetherAI,
                                    _ => EmbeddingProviderType::ONNX,
                                };
                                set_settings.update(|s| {
                                    s.embedding_provider = provider;
                                    // Reset to first available model
                                    s.embedding_model = provider.available_models()[0].0.to_string();
                                });
                            }
                        >
                            {move || {
                                let current = settings.get().embedding_provider;
                                [
                                    EmbeddingProviderType::ONNX,
                                    EmbeddingProviderType::OpenAI,
                                    EmbeddingProviderType::VoyageAI,
                                    EmbeddingProviderType::Cohere,
                                    EmbeddingProviderType::JinaAI,
                                    EmbeddingProviderType::Mistral,
                                    EmbeddingProviderType::TogetherAI,
                                ]
                                .into_iter()
                                .map(|provider| {
                                    view! {
                                        <option
                                            value=format!("{:?}", provider)
                                            selected=provider == current
                                        >
                                            {provider.display_name()}
                                        </option>
                                    }
                                })
                                .collect_view()
                            }}
                        </select>
                    </div>

                    // Model selection
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            "Model"
                        </label>
                        <select
                            class="select select-bordered w-full"
                            on:change=move |ev| {
                                let model = event_target_value(&ev);
                                set_settings.update(|s| s.embedding_model = model);
                            }
                        >
                            {move || {
                                let current_settings = settings.get();
                                current_settings.embedding_provider
                                    .available_models()
                                    .into_iter()
                                    .map(|(id, desc)| {
                                        view! {
                                            <option
                                                value=id
                                                selected=current_settings.embedding_model == id
                                            >
                                                {desc}
                                            </option>
                                        }
                                    })
                                    .collect_view()
                            }}
                        </select>
                    </div>

                    // API Key (if required)
                    {move || {
                        if settings.get().embedding_provider.requires_api_key() {
                            view! {
                                <div class="mb-4">
                                    <label class="block text-sm font-medium text-gray-700 mb-2">
                                        "API Key"
                                    </label>
                                    <input
                                        type="password"
                                        class="input input-bordered w-full"
                                        placeholder="Enter API key"
                                        value=move || settings.get().embedding_api_key.unwrap_or_default()
                                        on:input=move |ev| {
                                            let key = event_target_value(&ev);
                                            set_settings.update(|s| s.embedding_api_key = Some(key));
                                        }
                                    />
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="alert alert-info">
                                    <span>"🔒 ONNX runs locally in your browser - no API key needed"</span>
                                </div>
                            }.into_any()
                        }
                    }}
                </div>

                // LLM Provider Section
                <div class="bg-white rounded-lg p-6 mb-6 shadow-sm">
                    <h3 class="text-xl font-semibold text-purple-800 mb-4">
                        "🤖 LLM Provider"
                    </h3>

                    // Provider selection
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            "Provider"
                        </label>
                        <select
                            class="select select-bordered w-full"
                            on:change=move |ev| {
                                let value = event_target_value(&ev);
                                let provider = match value.as_str() {
                                    "WebLLM" => LlmProviderType::WebLLM,
                                    "OllamaHTTP" => LlmProviderType::OllamaHTTP,
                                    _ => LlmProviderType::WebLLM,
                                };
                                set_settings.update(|s| {
                                    s.llm_provider = provider;
                                    // Reset to first available model
                                    s.llm_model = provider.available_models()[0].0.to_string();
                                });
                            }
                        >
                            {move || {
                                let current = settings.get().llm_provider;
                                [LlmProviderType::WebLLM, LlmProviderType::OllamaHTTP]
                                    .into_iter()
                                    .map(|provider| {
                                        view! {
                                            <option
                                                value=format!("{:?}", provider)
                                                selected=provider == current
                                            >
                                                {provider.display_name()}
                                            </option>
                                        }
                                    })
                                    .collect_view()
                            }}
                        </select>
                    </div>

                    // Model selection
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            "Model"
                        </label>
                        <select
                            class="select select-bordered w-full"
                            on:change=move |ev| {
                                let model = event_target_value(&ev);
                                set_settings.update(|s| s.llm_model = model);
                            }
                        >
                            {move || {
                                let current_settings = settings.get();
                                current_settings.llm_provider
                                    .available_models()
                                    .into_iter()
                                    .map(|(id, desc)| {
                                        view! {
                                            <option
                                                value=id
                                                selected=current_settings.llm_model == id
                                            >
                                                {desc}
                                            </option>
                                        }
                                    })
                                    .collect_view()
                            }}
                        </select>
                    </div>

                    // Ollama endpoint (if Ollama selected)
                    {move || {
                        if settings.get().llm_provider == LlmProviderType::OllamaHTTP {
                            view! {
                                <div class="mb-4">
                                    <label class="block text-sm font-medium text-gray-700 mb-2">
                                        "Ollama Endpoint"
                                    </label>
                                    <input
                                        type="text"
                                        class="input input-bordered w-full"
                                        placeholder="http://localhost:11434"
                                        value=move || settings.get().llm_endpoint.unwrap_or_default()
                                        on:input=move |ev| {
                                            let endpoint = event_target_value(&ev);
                                            set_settings.update(|s| s.llm_endpoint = Some(endpoint));
                                        }
                                    />
                                    <p class="text-sm text-gray-500 mt-1">
                                        "Make sure Ollama is running with CORS enabled"
                                    </p>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="alert alert-info">
                                    <span>"🌐 WebLLM runs entirely in your browser - no server needed"</span>
                                </div>
                            }.into_any()
                        }
                    }}

                    // Temperature
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            {move || format!("Temperature: {:.2}", settings.get().llm_temperature)}
                        </label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            class="range range-primary"
                            value=move || (settings.get().llm_temperature * 100.0) as i32
                            on:input=move |ev| {
                                let value = event_target_value(&ev).parse::<f32>().unwrap_or(70.0);
                                set_settings.update(|s| s.llm_temperature = value / 100.0);
                            }
                        />
                        <div class="flex justify-between text-xs text-gray-500 mt-1">
                            <span>"Precise"</span>
                            <span>"Balanced"</span>
                            <span>"Creative"</span>
                        </div>
                    </div>
                </div>

                // Cache Settings
                <div class="bg-white rounded-lg p-6 mb-6 shadow-sm">
                    <h3 class="text-xl font-semibold text-purple-800 mb-4">
                        "💾 Cache Settings"
                    </h3>

                    <div class="form-control">
                        <label class="label cursor-pointer">
                            <span class="label-text">
                                "Cache models in browser for future use"
                            </span>
                            <input
                                type="checkbox"
                                class="toggle toggle-primary"
                                checked=move || settings.get().cache_models
                                on:change=move |ev| {
                                    let checked = event_target_checked(&ev);
                                    set_settings.update(|s| s.cache_models = checked);
                                }
                            />
                        </label>
                    </div>
                </div>

                // Save Button
                <div class="flex justify-between items-center">
                    <button
                        class="btn btn-primary btn-lg"
                        disabled=move || saving.get()
                        on:click=save_settings
                    >
                        {move || if saving.get() {
                            "Saving..."
                        } else {
                            "💾 Save Settings"
                        }}
                    </button>

                    {move || {
                        let status = save_status.get();
                        if !status.is_empty() {
                            view! {
                                <div class="text-lg font-semibold">
                                    {status}
                                </div>
                            }.into_any()
                        } else {
                            view! { <div></div> }.into_any()
                        }
                    }}
                </div>
            </div>
        </div>
    }
}

/// Save settings to IndexedDB
async fn save_settings_to_storage(settings: &UserSettings) -> Result<(), String> {
    use crate::storage::IndexedDBStore;

    let db = IndexedDBStore::new("graphrag_settings", 1)
        .await
        .map_err(|e| format!("Failed to open IndexedDB: {:?}", e))?;

    db.put("settings", "user_settings", settings)
        .await
        .map_err(|e| format!("Failed to save settings: {:?}", e))?;

    Ok(())
}

/// Load settings from IndexedDB
async fn load_settings_from_storage() -> Result<UserSettings, String> {
    use crate::storage::IndexedDBStore;

    let db = IndexedDBStore::new("graphrag_settings", 1)
        .await
        .map_err(|e| format!("Failed to open IndexedDB: {:?}", e))?;

    db.get("settings", "user_settings")
        .await
        .map_err(|e| format!("Failed to load settings: {:?}", e))
}
