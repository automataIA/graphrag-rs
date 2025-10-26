//! WebLLM Integration Tests
//!
//! Tests for WebLLM bindings, model initialization, chat completions, and streaming.
//! These tests require WebLLM to be available in the browser environment.

use wasm_bindgen_test::*;
use graphrag_wasm::webllm::{WebLLM, ChatMessage, is_webllm_available, get_recommended_models};

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Check WebLLM availability
///
/// Validates that we can detect if WebLLM is available in the environment.
#[wasm_bindgen_test]
fn test_webllm_availability() {
    let available = is_webllm_available();
    // This test passes regardless of availability - just checks the function works
    web_sys::console::log_1(&format!("WebLLM available: {}", available).into());
}

/// Test: Get recommended models
///
/// Validates that we can retrieve the list of recommended models.
#[wasm_bindgen_test]
fn test_get_recommended_models() {
    let models = get_recommended_models();
    // Should return a JsValue (array or object)
    assert!(!models.is_undefined());
    assert!(!models.is_null());
}

/// Test: ChatMessage creation
///
/// Validates the ChatMessage helper functions.
#[wasm_bindgen_test]
fn test_chat_message_creation() {
    // User message
    let user_msg = ChatMessage::user("Hello");
    assert_eq!(user_msg.role, "user");
    assert_eq!(user_msg.content, "Hello");

    // Assistant message
    let assistant_msg = ChatMessage::assistant("Hi there!");
    assert_eq!(assistant_msg.role, "assistant");
    assert_eq!(assistant_msg.content, "Hi there!");

    // System message
    let system_msg = ChatMessage::system("You are a helpful assistant");
    assert_eq!(system_msg.role, "system");
    assert_eq!(system_msg.content, "You are a helpful assistant");
}

/// Test: WebLLM initialization (skipped if WebLLM not available)
///
/// Note: This test is disabled by default as it downloads a 2GB+ model.
/// To run: Add `#[wasm_bindgen_test]` attribute and have WebLLM loaded.
#[allow(dead_code)]
async fn test_webllm_initialization() {
    if !is_webllm_available() {
        web_sys::console::warn_1(&"WebLLM not available, skipping test".into());
        return;
    }

    // Initialize with smallest model
    let result = WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await;

    match result {
        Ok(llm) => {
            web_sys::console::log_1(&"✅ WebLLM initialized successfully".into());
            // Model is ready to use
            drop(llm);
        }
        Err(e) => {
            web_sys::console::error_1(&format!("❌ WebLLM initialization failed: {}", e).into());
        }
    }
}

/// Test: WebLLM progress tracking
///
/// Note: This test is disabled by default as it downloads a model.
#[allow(dead_code)]
async fn test_webllm_progress_tracking() {
    if !is_webllm_available() {
        return;
    }

    use std::rc::Rc;
    use std::cell::RefCell;

    let progress_calls = Rc::new(RefCell::new(0));
    let progress_calls_clone = progress_calls.clone();

    let result = WebLLM::new_with_progress(
        "Llama-3.2-1B-Instruct-q4f16_1-MLC",
        move |progress, text| {
            *progress_calls_clone.borrow_mut() += 1;
            web_sys::console::log_1(&format!(
                "Progress: {:.1}% - {}",
                progress * 100.0,
                text
            ).into());
        },
    ).await;

    if let Ok(llm) = result {
        // Should have received multiple progress callbacks
        assert!(*progress_calls.borrow() > 0);
        drop(llm);
    }
}

/// Test: Simple chat completion
///
/// Note: This test is disabled by default as it requires model download.
#[allow(dead_code)]
async fn test_simple_chat() {
    if !is_webllm_available() {
        return;
    }

    let llm = match WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await {
        Ok(llm) => llm,
        Err(_) => return,
    };

    let result = llm.ask("What is 2+2?").await;

    match result {
        Ok(response) => {
            web_sys::console::log_1(&format!("Response: {}", response).into());
            assert!(!response.is_empty());
        }
        Err(e) => {
            web_sys::console::error_1(&format!("Chat failed: {}", e).into());
        }
    }
}

/// Test: Multi-turn conversation
///
/// Note: This test is disabled by default as it requires model download.
#[allow(dead_code)]
async fn test_multi_turn_conversation() {
    if !is_webllm_available() {
        return;
    }

    let llm = match WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await {
        Ok(llm) => llm,
        Err(_) => return,
    };

    // First turn
    let messages1 = vec![
        ChatMessage::system("You are a math tutor."),
        ChatMessage::user("What is 5 + 3?"),
    ];

    let response1 = llm.chat(messages1.clone(), Some(0.7), Some(100)).await;
    assert!(response1.is_ok());

    // Second turn (context maintained)
    let mut messages2 = messages1.clone();
    messages2.push(ChatMessage::assistant(response1.unwrap()));
    messages2.push(ChatMessage::user("Now multiply that by 2."));

    let response2 = llm.chat(messages2, Some(0.7), Some(100)).await;
    assert!(response2.is_ok());
}

/// Test: Streaming responses
///
/// Note: This test is disabled by default as it requires model download.
#[allow(dead_code)]
async fn test_streaming_chat() {
    if !is_webllm_available() {
        return;
    }

    let llm = match WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await {
        Ok(llm) => llm,
        Err(_) => return,
    };

    let messages = vec![
        ChatMessage::user("Count from 1 to 5."),
    ];

    use std::rc::Rc;
    use std::cell::RefCell;

    let chunks = Rc::new(RefCell::new(Vec::new()));
    let chunks_clone = chunks.clone();

    let full_response = llm.chat_stream(
        messages,
        move |chunk| {
            web_sys::console::log_1(&format!("Chunk: {}", chunk).into());
            chunks_clone.borrow_mut().push(chunk);
        },
        Some(0.8),
        Some(50),
    ).await;

    match full_response {
        Ok(response) => {
            assert!(!response.is_empty());
            assert!(!chunks.borrow().is_empty());
            web_sys::console::log_1(&format!("Full response: {}", response).into());
        }
        Err(e) => {
            web_sys::console::error_1(&format!("Streaming failed: {}", e).into());
        }
    }
}

/// Test: Temperature parameter
///
/// Note: This test is disabled by default as it requires model download.
#[allow(dead_code)]
async fn test_temperature_control() {
    if !is_webllm_available() {
        return;
    }

    let llm = match WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await {
        Ok(llm) => llm,
        Err(_) => return,
    };

    let messages = vec![
        ChatMessage::user("Write a creative sentence."),
    ];

    // Low temperature (more deterministic)
    let response_low = llm.chat(messages.clone(), Some(0.1), Some(50)).await;
    assert!(response_low.is_ok());

    // High temperature (more creative)
    let response_high = llm.chat(messages, Some(1.0), Some(50)).await;
    assert!(response_high.is_ok());

    // Both should return valid responses
    web_sys::console::log_1(&format!("Low temp: {}", response_low.unwrap()).into());
    web_sys::console::log_1(&format!("High temp: {}", response_high.unwrap()).into());
}

/// Test: Max tokens limiting
///
/// Note: This test is disabled by default as it requires model download.
#[allow(dead_code)]
async fn test_max_tokens_limit() {
    if !is_webllm_available() {
        return;
    }

    let llm = match WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC").await {
        Ok(llm) => llm,
        Err(_) => return,
    };

    let messages = vec![
        ChatMessage::user("Tell me a long story."),
    ];

    // Very short response (10 tokens)
    let response_short = llm.chat(messages.clone(), Some(0.7), Some(10)).await;
    assert!(response_short.is_ok());
    let short_text = response_short.unwrap();

    // Longer response (100 tokens)
    let response_long = llm.chat(messages, Some(0.7), Some(100)).await;
    assert!(response_long.is_ok());
    let long_text = response_long.unwrap();

    // Long response should be longer than short response
    assert!(long_text.len() > short_text.len());
}

/// Test: Error handling - Invalid model
///
/// Validates that initialization fails gracefully with invalid model ID.
#[wasm_bindgen_test]
async fn test_invalid_model_error() {
    if !is_webllm_available() {
        return;
    }

    let result = WebLLM::new("invalid-model-id-that-does-not-exist").await;

    // Should fail with error
    assert!(result.is_err());

    if let Err(e) = result {
        web_sys::console::log_1(&format!("Expected error: {}", e).into());
    }
}
