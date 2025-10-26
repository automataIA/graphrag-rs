//! Chat component for GraphRAG
//!
//! Provides a chat interface for interacting with the knowledge graph using LLM.

use leptos::*;

/// Chat message
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: f64,
}

/// Message role (user or assistant)
#[derive(Debug, Clone, PartialEq)]
pub enum MessageRole {
    User,
    Assistant,
}

/// Chat window component
///
/// Displays a chat interface with message history and input field.
/// Integrates with WebLLM for GPU-accelerated responses.
///
/// # Props
/// * `model` - Optional LLM model ID (default: "Phi-3-mini-4k-instruct-q4f16_1-MLC")
/// * `on_message` - Callback when user sends a message
#[component]
pub fn ChatWindow(
    #[prop(optional, into)] model: Option<String>,
    #[prop(optional)] on_message: Option<Box<dyn Fn(String)>>,
) -> impl IntoView {
    let (messages, set_messages) = create_signal(Vec::<ChatMessage>::new());
    let (input_value, set_input_value) = create_signal(String::new());
    let (is_loading, set_is_loading) = create_signal(false);

    let send_message = move |_| {
        let text = input_value.get();
        if text.trim().is_empty() {
            return;
        }

        // Add user message
        set_messages.update(|msgs| {
            msgs.push(ChatMessage {
                role: MessageRole::User,
                content: text.clone(),
                timestamp: js_sys::Date::now(),
            });
        });

        // Clear input
        set_input_value.set(String::new());

        // Call callback if provided
        if let Some(callback) = &on_message {
            callback(text.clone());
        }

        // TODO: Call WebLLM for response
        set_is_loading.set(true);

        // Simulate response (replace with actual WebLLM call)
        set_timeout(
            move || {
                set_messages.update(|msgs| {
                    msgs.push(ChatMessage {
                        role: MessageRole::Assistant,
                        content: "This is a placeholder response. WebLLM integration coming soon!".to_string(),
                        timestamp: js_sys::Date::now(),
                    });
                });
                set_is_loading.set(false);
            },
            std::time::Duration::from_secs(1),
        );
    };

    view! {
        <div class="chat-window flex flex-col h-full bg-white rounded-lg shadow-lg">
            // Header
            <div class="chat-header bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-4 rounded-t-lg">
                <h3 class="text-lg font-semibold">"💬 GraphRAG Chat"</h3>
                <p class="text-sm opacity-90">
                    {move || if is_loading.get() { "Thinking..." } else { "Ask questions about your knowledge graph" }}
                </p>
            </div>

            // Messages
            <div class="chat-messages flex-1 overflow-y-auto p-4 space-y-4 min-h-[400px] max-h-[600px]">
                {move || messages.get().iter().map(|msg| {
                    let is_user = matches!(msg.role, MessageRole::User);
                    view! {
                        <div class={format!("message flex {}", if is_user { "justify-end" } else { "justify-start" })}>
                            <div class={format!(
                                "max-w-[80%] p-3 rounded-lg {}",
                                if is_user { "bg-purple-600 text-white" } else { "bg-gray-100 text-gray-800" }
                            )}>
                                <p class="text-sm">{&msg.content}</p>
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>()}

                {move || if is_loading.get() {
                    view! {
                        <div class="message flex justify-start">
                            <div class="bg-gray-100 p-3 rounded-lg">
                                <div class="flex space-x-2">
                                    <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                    <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                                    <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
                                </div>
                            </div>
                        </div>
                    }.into_view()
                } else {
                    view! {}.into_view()
                }}
            </div>

            // Input
            <div class="chat-input border-t p-4">
                <div class="flex space-x-2">
                    <input
                        type="text"
                        placeholder="Type your question..."
                        class="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                        prop:value=move || input_value.get()
                        on:input=move |ev| set_input_value.set(event_target_value(&ev))
                        on:keypress=move |ev| {
                            if ev.key() == "Enter" {
                                send_message(ev);
                            }
                        }
                    />
                    <button
                        on:click=send_message
                        disabled=move || is_loading.get()
                        class="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    >
                        "Send"
                    </button>
                </div>
            </div>
        </div>
    }
}
