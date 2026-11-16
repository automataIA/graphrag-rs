//! llama.cpp reranker client against a fake `/v1/rerank` server.
#![cfg(feature = "llm")]

use graphrag_core::{config::RerankerConfig, reranker::LlamaCppReranker};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

fn read_request(mut stream: &TcpStream) -> (String, String) {
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
    (header, String::from_utf8(body).unwrap())
}

fn one_shot(
    status: &'static str,
    body: &'static str,
) -> (String, thread::JoinHandle<(String, String)>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_request(&stream);
        write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        request
    });
    (url, handle)
}

fn config(url: String) -> RerankerConfig {
    RerankerConfig {
        enabled: true,
        api_endpoint: url,
        ..RerankerConfig::default()
    }
}

#[tokio::test]
async fn maps_out_of_order_scores_back_to_input_indices() {
    let (url, server) = one_shot(
        "200 OK",
        r#"{"results":[{"index":2,"relevance_score":-8.3},{"index":0,"relevance_score":3.04}]}"#,
    );
    let client = LlamaCppReranker::from_config(&config(url));
    let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let ranked = client.rerank("domanda?", docs, Some(2)).await.unwrap();
    assert_eq!(ranked, vec![(0, 3.04), (2, -8.3)]);

    let (header, body) = server.join().unwrap();
    assert!(header.starts_with("POST /v1/rerank "), "{header}");
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["model"], "bge-reranker-v2-m3");
    assert_eq!(body["query"], "domanda?");
    assert_eq!(body["documents"], serde_json::json!(["a", "b", "c"]));
    assert_eq!(body["top_n"], 2);
}

#[tokio::test]
async fn http_error_is_an_error_not_a_panic() {
    let (url, server) = one_shot("500 Internal Server Error", r#"{"error":"boom"}"#);
    let client = LlamaCppReranker::from_config(&config(url));
    let err = client
        .rerank("q", vec!["a".to_string()], None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("500"), "{err}");
    server.join().unwrap();
}

#[tokio::test]
async fn unreachable_server_is_an_error() {
    let client = LlamaCppReranker::from_config(&config("http://127.0.0.1:1".to_string()));
    assert!(client
        .rerank("q", vec!["a".to_string()], None)
        .await
        .is_err());
}

#[tokio::test]
async fn missing_api_key_env_is_a_config_error() {
    let mut cfg = config("http://127.0.0.1:1".to_string());
    cfg.api_key_env = Some("GRAPHRAG_TEST_SURELY_UNSET_RERANK_KEY".to_string());
    let err = LlamaCppReranker::from_config(&cfg)
        .rerank("q", vec!["a".to_string()], None)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("GRAPHRAG_TEST_SURELY_UNSET"),
        "{err}"
    );
}
