//! LLM extraction constrained by the shared ontology.
//!
//! Mock-server tests run offline. The `#[ignore]` test needs a local
//! `llama-server --alias qwen3.5-9b --port 8080`:
//! `cargo test -p graphrag-core --features llm --test ontology_extraction -- --ignored`
#![cfg(feature = "llm")]

use graphrag_core::{
    core::{ChunkId, DocumentId, TextChunk},
    entity::LLMEntityExtractor,
    llm::{LlmClient, LlmConfig},
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

fn read_body(mut stream: &TcpStream) -> String {
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while !header.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        header.push(byte[0]);
    }
    let header = String::from_utf8(header).unwrap();
    let length: usize = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().unwrap())
        })
        .unwrap();
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).unwrap();
    String::from_utf8(body).unwrap()
}

/// Serves `replies` in order and returns the request bodies it received.
fn server(replies: Vec<(&'static str, String)>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        replies
            .into_iter()
            .map(|(status, body)| {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_body(&stream);
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
                request
            })
            .collect()
    });
    (base, handle)
}

fn chat_reply(content: &serde_json::Value) -> String {
    serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": content.to_string()}}]
    })
    .to_string()
}

fn chunk(text: &str) -> TextChunk {
    TextChunk::new(
        ChunkId::new("c1".to_string()),
        DocumentId::new("d1".to_string()),
        text.to_string(),
        0,
        text.len(),
    )
}

fn extractor(base_url: String) -> LLMEntityExtractor {
    let config = LlmConfig {
        enabled: true,
        base_url,
        max_retries: 1,
        ..LlmConfig::default()
    };
    LLMEntityExtractor::new(
        LlmClient::new(config),
        vec!["PERSON".to_string(), "ORGANIZATION".to_string()],
    )
}

const TEXT: &str = "Anna Rossi works for Acme. Anna admires Bruno.";

fn model_output() -> serde_json::Value {
    serde_json::json!({
        "entities": [
            {"name": "Anna Rossi", "type": "PERSON", "description": "employee"},
            {"name": "Acme", "type": "company", "description": "a firm"},
            {"name": "Bruno", "type": "PERSON", "description": ""}
        ],
        "relationships": [
            {"source": "Anna Rossi", "target": "Acme", "description": "Anna works for Acme", "type": "WORKS_FOR", "strength": 0.9},
            {"source": "Anna Rossi", "target": "Bruno", "description": "Anna envies Bruno", "type": "envies", "strength": 0.6}
        ]
    })
}

#[tokio::test]
async fn request_is_schema_constrained_and_output_normalized() {
    let (base, handle) = server(vec![("200 OK", chat_reply(&model_output()))]);
    let (entities, relationships) = extractor(base)
        .extract_from_chunk(&chunk(TEXT))
        .await
        .unwrap();
    let bodies = handle.join().unwrap();

    let request: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    let format = &request["response_format"];
    assert_eq!(format["type"], "json_schema");
    let rel_props =
        &format["json_schema"]["schema"]["properties"]["relationships"]["items"]["properties"];
    let enums: Vec<_> = rel_props["type"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .collect();
    assert!(enums.contains(&&serde_json::json!("WORKS_FOR")));
    assert!(enums.contains(&&serde_json::json!("OTHER")));
    let prompt = request["messages"][0]["content"].as_str().unwrap();
    assert!(
        prompt.contains("- WORKS_FOR:"),
        "prompt lists relation definitions"
    );

    let acme = entities.iter().find(|e| e.name == "Acme").unwrap();
    assert_eq!(acme.entity_type, "ORGANIZATION");
    assert_eq!(relationships[0].relation_type, "WORKS_FOR");
    assert_eq!(relationships[1].relation_type, "OTHER");
    assert_eq!(
        relationships[1].description.as_deref(),
        Some("envies: Anna envies Bruno")
    );
}

#[tokio::test]
async fn schema_rejection_retries_without_schema() {
    let (base, handle) = server(vec![
        (
            "400 Bad Request",
            r#"{"error":{"message":"response_format json_schema not supported"}}"#.to_string(),
        ),
        ("200 OK", chat_reply(&model_output())),
    ]);
    let (_, relationships) = extractor(base)
        .extract_from_chunk(&chunk(TEXT))
        .await
        .unwrap();
    let bodies = handle.join().unwrap();
    assert!(bodies[0].contains("response_format"));
    assert!(!bodies[1].contains("response_format"));
    assert_eq!(relationships.len(), 2);
}

/// Live check against a local llama-server: relation types must come from the
/// ontology and the obvious relations must be classified correctly.
#[tokio::test]
#[ignore = "needs llama-server on 127.0.0.1:8080"]
async fn live_llama_server_uses_closed_relation_list() {
    let text = "Marco Polo was born in Venice. He wrote Il Milione together with \
                Rustichello da Pisa, who was his cellmate in Genoa. Kublai Khan \
                ruled the Mongol Empire and welcomed Marco Polo at his court.";
    let extractor = LLMEntityExtractor::new(
        LlmClient::new(LlmConfig {
            enabled: true,
            ..LlmConfig::default()
        }),
        Vec::new(),
    );
    let names = extractor.ontology().relation_type_names();
    let (entities, relationships) = extractor.extract_from_chunk(&chunk(text)).await.unwrap();
    for r in &relationships {
        eprintln!(
            "{} -[{}]-> {} | {:?}",
            r.source, r.relation_type, r.target, r.description
        );
        assert!(
            r.relation_type == "OTHER" || names.contains(&r.relation_type),
            "{} not in ontology",
            r.relation_type
        );
    }
    for e in &entities {
        eprintln!("{} : {}", e.name, e.entity_type);
    }
    let has = |ty: &str| relationships.iter().any(|r| r.relation_type == ty);
    assert!(has("LOCATED_IN"), "born in Venice → LOCATED_IN");
    assert!(has("CREATED"), "wrote Il Milione → CREATED");
}
