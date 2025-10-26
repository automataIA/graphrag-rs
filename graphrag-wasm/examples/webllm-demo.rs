//! WebLLM Integration Example
//!
//! Demonstrates GPU-accelerated LLM inference in the browser using WebLLM.
//! This example shows:
//! - Model initialization with progress tracking
//! - Chat completions with temperature control
//! - Streaming responses for real-time feedback
//! - Integration with GraphRAG for RAG workflows
//!
//! ## Setup
//!
//! Add to your `index.html`:
//! ```html
//! <script type="module">
//!   import * as webllm from "https://esm.run/@mlc-ai/web-llm";
//!   window.webllm = webllm;
//! </script>
//! ```
//!
//! ## Build
//!
//! ```bash
//! wasm-pack build --target web --out-dir www/pkg
//! ```

use wasm_bindgen::prelude::*;
use graphrag_wasm::webllm::{WebLLM, ChatMessage};

/// Example 1: Basic chat completion
#[wasm_bindgen]
pub async fn example_basic_chat() -> Result<String, JsValue> {
    // Initialize WebLLM with Phi-3 Mini model
    let llm = WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Send a simple message
    let response = llm.ask("What is GraphRAG?").await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(response)
}

/// Example 2: Chat with progress tracking
#[wasm_bindgen]
pub async fn example_with_progress() -> Result<String, JsValue> {
    // Initialize with progress callback
    let llm = WebLLM::new_with_progress(
        "Llama-3.2-1B-Instruct-q4f16_1-MLC",
        |progress, text| {
            web_sys::console::log_1(&format!(
                "Loading: {:.1}% - {}",
                progress * 100.0,
                text
            ).into());
        },
    )
    .await
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Chat after model is loaded
    let response = llm.ask("Explain knowledge graphs in one sentence.").await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(response)
}

/// Example 3: Multi-turn conversation
#[wasm_bindgen]
pub async fn example_conversation() -> Result<String, JsValue> {
    let llm = WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create conversation with system message
    let messages = vec![
        ChatMessage::system("You are a helpful assistant that explains technical concepts clearly."),
        ChatMessage::user("What is RAG?"),
    ];

    let response = llm.chat(messages, Some(0.7), Some(256)).await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(response)
}

/// Example 4: Streaming responses
#[wasm_bindgen]
pub async fn example_streaming() -> Result<String, JsValue> {
    let llm = WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let messages = vec![
        ChatMessage::user("Write a haiku about Rust programming."),
    ];

    // Stream response token by token
    let full_response = llm.chat_stream(
        messages,
        |chunk| {
            web_sys::console::log_1(&format!("Token: {}", chunk).into());
        },
        Some(0.8),
        Some(100),
    )
    .await
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(full_response)
}

/// Example 5: RAG workflow (GraphRAG + WebLLM)
#[wasm_bindgen]
pub async fn example_rag_workflow() -> Result<String, JsValue> {
    use graphrag_wasm::GraphRAG;

    // Step 1: Set up GraphRAG vector store
    let mut graph = GraphRAG::new(384)?;

    // Add knowledge base documents (in real app, use actual embeddings)
    let doc1_embedding = vec![0.1; 384]; // Replace with real embeddings
    graph.add_document(
        "doc1".to_string(),
        "GraphRAG is a knowledge graph-based retrieval system.".to_string(),
        doc1_embedding,
    ).await?;

    let doc2_embedding = vec![0.2; 384];
    graph.add_document(
        "doc2".to_string(),
        "WebLLM enables GPU-accelerated LLM inference in browsers.".to_string(),
        doc2_embedding,
    ).await?;

    graph.build_index().await?;

    // Step 2: Retrieve relevant documents
    let query_embedding = vec![0.15; 384]; // Replace with real query embedding
    let search_results = graph.query(query_embedding, 3).await?;

    web_sys::console::log_1(&format!("Retrieved docs: {}", search_results).into());

    // Step 3: Use WebLLM to generate answer
    let llm = WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let messages = vec![
        ChatMessage::system("You are a helpful assistant. Answer based on the provided context."),
        ChatMessage::user(&format!(
            "Context:\n{}\n\nQuestion: What technologies are being used?\n\nAnswer:",
            search_results
        )),
    ];

    let response = llm.chat(messages, Some(0.7), Some(512)).await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(response)
}

/// Example 6: Model selection helper
#[wasm_bindgen]
pub fn get_model_recommendations() -> JsValue {
    graphrag_wasm::webllm::get_recommended_models()
}

/// Example 7: Check WebLLM availability
#[wasm_bindgen]
pub fn check_webllm() -> bool {
    graphrag_wasm::webllm::is_webllm_available()
}

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());

    web_sys::console::log_1(&"WebLLM examples initialized".into());

    // Check if WebLLM is available
    if check_webllm() {
        web_sys::console::log_1(&"✅ WebLLM is available".into());
    } else {
        web_sys::console::warn_1(&"⚠️ WebLLM not found. Add <script> tag to index.html".into());
    }
}
