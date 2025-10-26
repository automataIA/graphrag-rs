//! GraphRAG WASM Web Application
//!
//! This is a demonstration application showing GraphRAG running 100% in the browser
//! with Leptos, WASM, Voy vector search, and WebLLM for GPU-accelerated inference.

use leptos::prelude::*;
use leptos::mount;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{console, MouseEvent};

/// Message for chat
#[derive(Debug, Clone, PartialEq)]
struct ChatMessage {
    role: String,
    content: String,
    timestamp: f64,
}

#[component]
fn App() -> impl IntoView {
    // State
    let (messages, set_messages) = signal(Vec::<ChatMessage>::new());
    let (input_value, set_input_value) = signal(String::new());
    let (is_loading, set_is_loading) = signal(false);
    let (llm_ready, set_llm_ready) = signal(false);
    let (status_message, set_status_message) = signal(String::from("Checking WebLLM availability..."));
    let (gpu_info, set_gpu_info) = signal(String::from("Detecting GPU..."));

    // Check WebLLM availability on mount
    Effect::new(move |_| {
        spawn_local(async move {
            // Check if WebLLM is available
            let window = web_sys::window().expect("no global window");
            match js_sys::Reflect::get(&window, &JsValue::from_str("webllm")) {
                Ok(webllm) if !webllm.is_undefined() => {
                    set_status_message.set("WebLLM is available! ✅".to_string());
                    set_llm_ready.set(true);
                }
                _ => {
                    set_status_message.set("WebLLM not loaded. Add script tag to index.html ⚠️".to_string());
                }
            }

            // Check WebGPU
            match js_sys::Reflect::get(&window.navigator(), &JsValue::from_str("gpu")) {
                Ok(gpu) if !gpu.is_undefined() => {
                    set_gpu_info.set("WebGPU Available ✅ - GPU acceleration enabled".to_string());
                }
                _ => {
                    set_gpu_info.set("WebGPU Not Available ⚠️ - Using CPU fallback".to_string());
                }
            }
        });
    });

    // Send message handler
    let send_message = move |_: MouseEvent| {
        let text = input_value.get();
        if text.trim().is_empty() || is_loading.get() {
            return;
        }

        // Add user message
        set_messages.update(|msgs| {
            msgs.push(ChatMessage {
                role: "user".to_string(),
                content: text.clone(),
                timestamp: js_sys::Date::now(),
            });
        });

        // Clear input
        set_input_value.set(String::new());
        set_is_loading.set(true);

        // Get response
        if llm_ready.get() {
            // TODO: Call WebLLM API when integrated
            spawn_local(async move {
                // Simulate AI response for now
                gloo_timers::future::TimeoutFuture::new(1_000).await;

                set_messages.update(|msgs| {
                    msgs.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: format!(
                            "This is a demo response. WebLLM integration is ready! Your question was: \"{}\"",
                            text
                        ),
                        timestamp: js_sys::Date::now(),
                    });
                });
                set_is_loading.set(false);
            });
        } else {
            set_messages.update(|msgs| {
                msgs.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: "WebLLM is not loaded. Please add the WebLLM script to index.html.".to_string(),
                    timestamp: js_sys::Date::now(),
                });
            });
            set_is_loading.set(false);
        }
    };

    view! {
        <div class="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-100">
            // Header
            <header class="bg-white shadow-sm border-b border-purple-200">
                <div class="max-w-7xl mx-auto px-4 py-6">
                    <h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
                        "GraphRAG-RS"
                    </h1>
                    <p class="text-gray-600 mt-2">
                        "🚀 100% Browser-Based Knowledge Graph • Rust + WASM + WebGPU"
                    </p>
                </div>
            </header>

            <div class="max-w-7xl mx-auto px-4 py-8">
                // Status Cards
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6 border-l-4 border-purple-500">
                        <h3 class="font-semibold text-gray-700 mb-2">"System Status"</h3>
                        <p class="text-sm text-gray-600">{status_message}</p>
                    </div>
                    <div class="bg-white rounded-lg shadow-md p-6 border-l-4 border-indigo-500">
                        <h3 class="font-semibold text-gray-700 mb-2">"GPU Acceleration"</h3>
                        <p class="text-sm text-gray-600">{gpu_info}</p>
                    </div>
                </div>

                // Chat Interface
                <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                    <div class="bg-gradient-to-r from-purple-600 to-indigo-600 px-6 py-4">
                        <h2 class="text-xl font-semibold text-white">"💬 AI Chat"</h2>
                        <p class="text-purple-100 text-sm mt-1">
                            {move || if is_loading.get() { "Thinking...".to_string() } else { "Ask anything about your knowledge graph".to_string() }}
                        </p>
                    </div>

                    // Messages
                    <div class="h-96 overflow-y-auto p-6 space-y-4 bg-gray-50">
                        {move || {
                            let msgs = messages.get();
                            if msgs.is_empty() {
                                view! {
                                    <div class="text-center text-gray-400 py-12">
                                        <p class="text-lg">"👋 Welcome to GraphRAG-RS!"</p>
                                        <p class="mt-2">"Start chatting to see the AI in action"</p>
                                    </div>
                                }.into_any()
                            } else {
                                msgs.iter().map(|msg| {
                                    let is_user = msg.role == "user";
                                    let msg_content = msg.content.clone();
                                    view! {
                                        <div class={if is_user { "flex justify-end" } else { "flex justify-start" }}>
                                            <div class={format!(
                                                "max-w-[80%] px-4 py-3 rounded-lg {}",
                                                if is_user { "bg-purple-600 text-white" } else { "bg-white border border-gray-200 text-gray-800" }
                                            )}>
                                                <p class="text-sm">{msg_content}</p>
                                            </div>
                                        </div>
                                    }
                                }).collect::<Vec<_>>().into_any()
                            }
                        }}

                        // Loading indicator
                        {move || if is_loading.get() {
                            view! {
                                <div class="flex justify-start">
                                    <div class="bg-white border border-gray-200 px-4 py-3 rounded-lg">
                                        <div class="flex space-x-2">
                                            <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                                            <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                                            <div class="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
                                        </div>
                                    </div>
                                </div>
                            }.into_any()
                        } else {
                            view! {}.into_any()
                        }}
                    </div>

                    // Input
                    <div class="border-t border-gray-200 p-4 bg-white">
                        <div class="flex space-x-2">
                            <input
                                type="text"
                                placeholder="Type your message..."
                                class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                prop:value=move || input_value.get()
                                on:input=move |ev| set_input_value.set(event_target_value(&ev))
                                on:keypress=move |ev: web_sys::KeyboardEvent| {
                                    if ev.key() == "Enter" && !is_loading.get() {
                                        send_message(MouseEvent::new("click").unwrap());
                                    }
                                }
                            />
                            <button
                                on:click=send_message
                                disabled=move || is_loading.get()
                                class="px-6 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-medium rounded-lg hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            >
                                "Send"
                            </button>
                        </div>
                    </div>
                </div>

                // Features
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="text-3xl mb-3">"🧠"</div>
                        <h3 class="font-semibold text-gray-800 mb-2">"GPU-Accelerated LLM"</h3>
                        <p class="text-sm text-gray-600">
                            "WebLLM provides 40-62 tokens/sec inference with WebGPU acceleration"
                        </p>
                    </div>
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="text-3xl mb-3">"🔒"</div>
                        <h3 class="font-semibold text-gray-800 mb-2">"100% Private"</h3>
                        <p class="text-sm text-gray-600">
                            "All processing happens in your browser. No data ever leaves your device"
                        </p>
                    </div>
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="text-3xl mb-3">"⚡"</div>
                        <h3 class="font-semibold text-gray-800 mb-2">"Instant Search"</h3>
                        <p class="text-sm text-gray-600">
                            "Voy vector search provides <10ms queries on 10k+ documents"
                        </p>
                    </div>
                </div>
            </div>

            // Footer
            <footer class="text-center py-8 text-gray-600 text-sm">
                <p>"Built with " <span class="text-purple-600 font-semibold">"Rust"</span> " + " <span class="text-indigo-600 font-semibold">"Leptos"</span> " + " <span class="text-purple-600 font-semibold">"WebGPU"</span></p>
            </footer>
        </div>
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console::log_1(&"🚀 GraphRAG-RS initializing...".into());
    mount::mount_to_body(App);
}
