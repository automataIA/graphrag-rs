//! Hosted OpenAI-compatible APIs (Groq, Cerebras) against a fake chat server.
#![cfg(feature = "llm")]

use graphrag_core::llm::{LlmClient, LlmConfig};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

fn read_request(mut stream: &TcpStream) -> String {
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        header.push(byte[0]);
    }
    let header = String::from_utf8(header).unwrap();
    let length: usize = header
        .lines()
        .find_map(|l| {
            let (n, v) = l.split_once(':')?;
            n.eq_ignore_ascii_case("content-length").then_some(v.trim())
        })
        .unwrap()
        .parse()
        .unwrap();
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).unwrap();
    String::from_utf8(body).unwrap()
}

/// Serves `replies` (status line, extra headers, body) in order, one per connection,
/// and returns every request body seen.
fn fake_server(
    replies: Vec<(&'static str, &'static str, &'static str)>,
) -> (String, thread::JoinHandle<Vec<serde_json::Value>>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let mut seen = Vec::new();
        for (status, headers, body) in replies {
            let (mut stream, _) = listener.accept().unwrap();
            seen.push(serde_json::from_str(&read_request(&stream)).unwrap());
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
        seen
    });
    (url, handle)
}

const OK: &str = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
const KWARGS_400: &str =
    r#"{"message":"chat_template_kwargs: property 'chat_template_kwargs' is unsupported"}"#;
const EFFORT_400: &str =
    r#"{"error":"reasoning_effort: property 'reasoning_effort' is unsupported"}"#;

fn client(url: String) -> LlmClient {
    LlmClient::new(LlmConfig {
        base_url: url,
        disable_thinking: true,
        max_retries: 3,
        ..LlmConfig::default()
    })
}

#[test]
fn rejected_kwargs_fall_back_to_reasoning_effort_and_are_remembered() {
    let (url, server) = fake_server(vec![
        ("400 Bad Request", "", KWARGS_400),
        ("200 OK", "", OK),
        ("200 OK", "", OK),
    ]);
    let client = client(url);
    let params = client.default_params();
    assert_eq!(client.generate_blocking("a", &params).unwrap().text, "ok");
    assert_eq!(client.generate_blocking("b", &params).unwrap().text, "ok");
    let seen = server.join().unwrap();
    assert_eq!(seen[0]["chat_template_kwargs"]["enable_thinking"], false);
    assert!(seen[0].get("reasoning_effort").is_none());
    assert_eq!(seen[1]["reasoning_effort"], "none");
    assert!(seen[1].get("chat_template_kwargs").is_none());
    assert_eq!(seen[2]["reasoning_effort"], "none");
    assert!(seen[2].get("chat_template_kwargs").is_none());
}

#[test]
fn rejected_reasoning_effort_is_dropped_too() {
    let (url, server) = fake_server(vec![
        ("400 Bad Request", "", KWARGS_400),
        ("400 Bad Request", "", EFFORT_400),
        ("200 OK", "", OK),
        ("200 OK", "", OK),
    ]);
    let client = client(url);
    let params = client.default_params();
    assert_eq!(client.generate_blocking("a", &params).unwrap().text, "ok");
    assert_eq!(client.generate_blocking("b", &params).unwrap().text, "ok");
    let seen = server.join().unwrap();
    assert_eq!(seen[1]["reasoning_effort"], "none");
    for later in &seen[2..] {
        assert!(later.get("reasoning_effort").is_none());
        assert!(later.get("chat_template_kwargs").is_none());
    }
}

#[test]
fn rate_limit_with_retry_after_is_retried() {
    let (url, server) = fake_server(vec![
        (
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"Rate limit reached. Please try again in 0s."}}"#,
        ),
        ("200 OK", "", OK),
    ]);
    let mut config = LlmConfig {
        base_url: url,
        max_retries: 1,
        ..LlmConfig::default()
    };
    config.disable_thinking = false;
    let client = LlmClient::new(config);
    // max_retries = 1 would give up after one attempt for ordinary errors,
    // but rate-limit responses get at least 8 attempts.
    assert_eq!(
        client
            .generate_blocking("a", &client.default_params())
            .unwrap()
            .text,
        "ok"
    );
    assert_eq!(server.join().unwrap().len(), 2);
}
