#![cfg(feature = "llm")]

use graphrag_core::{
    config::EmbeddingConfig,
    llama_cpp::{E5Intent, LlamaCppEmbeddingProvider},
    llm::{LlmClient, LlmConfig, LlmGenerationParams},
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

/// Reads one HTTP request, returning `(raw header block, body)`.
fn read_request(mut stream: &TcpStream) -> (String, String) {
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        header.push(byte[0]);
    }
    let header_text = String::from_utf8(header).unwrap();
    let length = header_text
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then_some(value.trim())
        })
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).unwrap();
    (header_text, String::from_utf8(body).unwrap())
}

fn response(stream: &mut TcpStream, status: &str, body: &str) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}

fn server<F>(requests: usize, responder: F) -> (String, thread::JoinHandle<Vec<String>>)
where
    F: Fn(usize, &str) -> (&'static str, String) + Send + 'static,
{
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let mut bodies = Vec::new();
        for index in 0..requests {
            let (mut stream, _) = listener.accept().unwrap();
            let (header, body) = read_request(&stream);
            let (status, response_body) = responder(index, &body);
            response(&mut stream, status, &response_body);
            bodies.push(format!("{header}{body}"));
        }
        bodies
    });
    (address, handle)
}

fn embedding_config(endpoint: String) -> EmbeddingConfig {
    let mut config = EmbeddingConfig {
        dimension: 384,
        backend: "llama_cpp".to_string(),
        model: Some("multilingual-e5-small".to_string()),
        fallback_to_hash: false,
        api_endpoint: Some(format!("{endpoint}/v1/embeddings")),
        api_key: None,
        cache_dir: None,
        batch_size: 32,
    };
    config.api_endpoint = Some(format!("{endpoint}/v1/embeddings"));
    config
}

fn vector(axis: usize) -> Vec<f32> {
    let mut values = vec![0.0; 384];
    values[axis] = 1.0;
    values
}

#[test]
fn batch_uses_one_embedding_request_and_restores_input_order() {
    let (base, handle) = server(3, |index, body| {
        if index < 2 {
            assert!(body.contains("query: ") || body.contains("passage: "));
            return ("200 OK", r#"{"tokens":[0,1,2]}"#.to_string());
        }
        let request: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(request["input"].as_array().unwrap().len(), 2);
        let second = serde_json::json!({"index": 1, "embedding": vector(1)});
        // Deliberately return data out of order. The provider must restore input order.
        (
            "200 OK",
            serde_json::json!({"data": [second, {"index": 0, "embedding": vector(0)}]}).to_string(),
        )
    });
    let provider = LlamaCppEmbeddingProvider::from_config(&embedding_config(base), 5).unwrap();
    let vectors = provider
        .embed_batch(&["uno", "日本語"], E5Intent::Document)
        .unwrap();
    handle.join().unwrap();
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0][0], 1.0);
    assert_eq!(vectors[1][1], 1.0);
}

#[test]
fn tokenizer_limit_keeps_first_511_tokens_and_original_eos() {
    let (base, handle) = server(2, |index, body| {
        if index == 0 {
            return (
                "200 OK",
                serde_json::json!({"tokens": (0..600).collect::<Vec<_>>()}).to_string(),
            );
        }
        let request: serde_json::Value = serde_json::from_str(body).unwrap();
        let input = request["input"].as_array().unwrap();
        assert_eq!(input.len(), 512);
        assert_eq!(input.first().and_then(serde_json::Value::as_u64), Some(0));
        assert_eq!(input[510].as_u64(), Some(510));
        assert_eq!(input[511].as_u64(), Some(599));
        (
            "200 OK",
            serde_json::json!({"data": [{"index": 0, "embedding": vector(0)}]}).to_string(),
        )
    });
    let provider = LlamaCppEmbeddingProvider::from_config(&embedding_config(base), 5).unwrap();
    provider.embed_query("long input").unwrap();
    handle.join().unwrap();
}

#[test]
fn batch_rejects_duplicate_or_missing_indices() {
    for data in [
        serde_json::json!([{"index": 0, "embedding": vector(0)}, {"index": 0, "embedding": vector(1)}]),
        serde_json::json!([{"index": 0, "embedding": vector(0)}, {"index": 2, "embedding": vector(1)}]),
        serde_json::json!([{"index": 0, "embedding": vector(0)}, {"embedding": vector(1)}]),
        serde_json::json!([{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": vector(1)}]),
    ] {
        let (base, handle) = server(3, move |index, _| {
            if index < 2 {
                ("200 OK", r#"{"tokens":[0,1]}"#.to_string())
            } else {
                ("200 OK", serde_json::json!({"data": data}).to_string())
            }
        });
        let provider = LlamaCppEmbeddingProvider::from_config(&embedding_config(base), 5).unwrap();
        assert!(provider.embed_batch(&["a", "b"], E5Intent::Query).is_err());
        handle.join().unwrap();
    }
}

fn llm_client(base: &str, max_retries: u32, api_key_env: Option<&str>) -> LlmClient {
    LlmClient::new(LlmConfig {
        enabled: true,
        base_url: format!("{base}/v1"),
        max_retries,
        api_key_env: api_key_env.map(str::to_string),
        enable_caching: false,
        ..Default::default()
    })
}

#[test]
fn generation_preserves_optional_usage_and_rejects_partial_usage() {
    let params = LlmGenerationParams::default();

    let (base, handle) = server(1, |_, _| {
        (
            "200 OK",
            r#"{"choices":[{"message":{"content":"ok"}}]}"#.to_string(),
        )
    });
    let response = llm_client(&base, 1, None)
        .generate_blocking("hello", &params)
        .unwrap();
    assert_eq!(response.text, "ok");
    assert!(response.usage.is_none());
    let requests = handle.join().unwrap();
    assert!(requests[0].starts_with("POST /v1/chat/completions "));
    assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));

    let (base, handle) = server(1, |_, _| {
        (
            "200 OK",
            r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1}}"#.to_string(),
        )
    });
    assert!(llm_client(&base, 1, None)
        .generate_blocking("hello", &params)
        .is_err());
    handle.join().unwrap();

    let (base, handle) = server(1, |_, _| {
        (
            "200 OK",
            r#"{"choices":[{"message":{"content":"measured"}}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7},"timings":{"cache_n":2}}"#.to_string(),
        )
    });
    let response = llm_client(&base, 1, None)
        .generate_blocking("hello", &params)
        .unwrap();
    assert_eq!(response.usage.map(|usage| usage.total_tokens), Some(7));
    assert_eq!(response.cached_prompt_tokens, Some(2));
    handle.join().unwrap();

    for body in [
        "not json".to_string(),
        r#"{"choices":[]}"#.to_string(),
        r#"{"choices":"malformed"}"#.to_string(),
        r#"{"choices":[{"message":{"content":""}}]}"#.to_string(),
    ] {
        let (base, handle) = server(1, move |_, _| ("200 OK", body.clone()));
        // Malformed 200 responses are not retried: one request only.
        assert!(llm_client(&base, 3, None)
            .generate_blocking("hello", &params)
            .is_err());
        handle.join().unwrap();
    }
}

#[test]
fn generation_retries_server_errors_and_sends_bearer_key() {
    const KEY_ENV: &str = "GRAPHRAG_TEST_LLM_KEY_RETRY";
    std::env::set_var(KEY_ENV, "test-secret");
    let (base, handle) = server(2, |index, _| {
        if index == 0 {
            ("503 Service Unavailable", "{}".to_string())
        } else {
            (
                "200 OK",
                r#"{"choices":[{"message":{"content":"second"}}]}"#.to_string(),
            )
        }
    });
    let response = llm_client(&base, 2, Some(KEY_ENV))
        .generate_blocking("hello", &LlmGenerationParams::default())
        .unwrap();
    assert_eq!(response.text, "second");
    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert!(request
            .to_ascii_lowercase()
            .contains("authorization: bearer test-secret"));
    }
    std::env::remove_var(KEY_ENV);
}

#[test]
fn generation_fails_fast_when_named_key_is_missing() {
    let client = llm_client("http://127.0.0.1:9", 3, Some("GRAPHRAG_TEST_LLM_KEY_UNSET"));
    let error = client
        .generate_blocking("hello", &LlmGenerationParams::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("GRAPHRAG_TEST_LLM_KEY_UNSET"));
    assert!(!error.contains("test-secret"));
}

#[test]
fn provider_http_errors_include_status_and_bounded_utf8_body() {
    let (base, handle) = server(1, |_, _| ("503 Service Unavailable", "é".repeat(10_000)));
    let provider = LlamaCppEmbeddingProvider::from_config(&embedding_config(base), 5).unwrap();
    let error = provider.embed_query("test").unwrap_err().to_string();
    handle.join().unwrap();
    assert!(error.contains("HTTP 503"));
    assert!(error.contains("é"));
    assert!(error.len() < 5_000);
}

fn request_json(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw.split_once("\r\n\r\n").unwrap().1).unwrap()
}

const OK_BODY: &str = r#"{"choices":[{"message":{"content":"ok"}}]}"#;

#[test]
fn thinking_is_disabled_per_request_by_default() {
    let (base, handle) = server(1, |_, _| ("200 OK", OK_BODY.to_string()));
    llm_client(&base, 1, None)
        .generate_blocking("hello", &LlmGenerationParams::default())
        .unwrap();
    let requests = handle.join().unwrap();
    assert_eq!(
        request_json(&requests[0])["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
}

#[test]
fn thinking_kwargs_absent_when_disable_thinking_is_false() {
    let (base, handle) = server(1, |_, _| ("200 OK", OK_BODY.to_string()));
    let client = LlmClient::new(LlmConfig {
        enabled: true,
        base_url: format!("{base}/v1"),
        max_retries: 1,
        enable_caching: false,
        disable_thinking: false,
        ..Default::default()
    });
    client
        .generate_blocking("hello", &LlmGenerationParams::default())
        .unwrap();
    let requests = handle.join().unwrap();
    assert!(request_json(&requests[0])
        .get("chat_template_kwargs")
        .is_none());
}

#[test]
fn rejected_chat_template_kwargs_is_retried_once_and_remembered() {
    let (base, handle) = server(3, |_, body| {
        if body.contains("chat_template_kwargs") {
            (
                "400 Bad Request",
                r#"{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs"}}"#
                    .to_string(),
            )
        } else {
            ("200 OK", OK_BODY.to_string())
        }
    });
    let client = llm_client(&base, 1, None);
    let params = LlmGenerationParams::default();
    assert_eq!(client.generate_blocking("a", &params).unwrap().text, "ok");
    assert_eq!(client.generate_blocking("b", &params).unwrap().text, "ok");
    let requests = handle.join().unwrap();
    assert!(request_json(&requests[0])
        .get("chat_template_kwargs")
        .is_some());
    assert!(request_json(&requests[1])
        .get("chat_template_kwargs")
        .is_none());
    assert!(request_json(&requests[2])
        .get("chat_template_kwargs")
        .is_none());
}

#[test]
fn unrelated_400_is_not_retried_without_kwargs() {
    let (base, handle) = server(1, |_, _| {
        ("400 Bad Request", r#"{"error":"bad model"}"#.to_string())
    });
    assert!(llm_client(&base, 1, None)
        .generate_blocking("a", &LlmGenerationParams::default())
        .is_err());
    handle.join().unwrap();
}

#[test]
fn reasoning_only_response_explains_thinking() {
    let (base, handle) = server(1, |_, _| {
        (
            "200 OK",
            r#"{"choices":[{"finish_reason":"length","message":{"content":"","reasoning_content":"Let me think..."}}]}"#
                .to_string(),
        )
    });
    let error = llm_client(&base, 1, None)
        .generate_blocking("a", &LlmGenerationParams::default())
        .unwrap_err()
        .to_string();
    handle.join().unwrap();
    assert!(error.contains("reasoning"), "{error}");
    assert!(error.contains("disable_thinking"), "{error}");
    assert!(error.contains("enable_thinking"), "{error}");
}
