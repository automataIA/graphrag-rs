//! LLM relation stage of the GLiNER -> LLM cascade, against a fake chat server.
//!
//! The entities are supplied directly (as GLiNER would produce them), so no ONNX
//! model is needed.
#![cfg(feature = "llm")]

use graphrag_core::{
    core::{ChunkId, DocumentId, Entity, EntityId, TextChunk},
    entity::{ExtractionCache, LLMEntityExtractor},
    llm::{LlmClient, LlmConfig},
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
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

struct Fake {
    base_url: String,
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
}

/// Answers every request with `status` and, when 200, `content` as the assistant message.
fn fake_server(status: &'static str, content: String) -> Fake {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let (seen, content) = (seen.clone(), content.clone());
            thread::spawn(move || {
                let body = read_body(&stream);
                seen.lock()
                    .unwrap()
                    .push(serde_json::from_str(&body).unwrap());
                let reply = serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": content}}]
                })
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            });
        }
    });
    Fake { base_url, requests }
}

fn chunk(text: &str) -> TextChunk {
    TextChunk::new(
        ChunkId::new("c1".to_string()),
        DocumentId::new("d".to_string()),
        text.to_string(),
        0,
        text.len(),
    )
}

fn entity(name: &str, ty: &str) -> Entity {
    Entity::new(
        EntityId::from_type_and_name(ty, name),
        name.to_string(),
        ty.to_string(),
        0.9,
    )
}

fn extractor(base_url: &str) -> LLMEntityExtractor {
    let config = LlmConfig {
        enabled: true,
        base_url: base_url.to_string(),
        max_retries: 1,
        enable_caching: false,
        ..LlmConfig::default()
    };
    LLMEntityExtractor::new(LlmClient::new(config), vec!["PERSON".to_string()])
}

const TEXT: &str = "Geppetto costruì Pinocchio nella bottega di Geppetto.";

fn entities() -> Vec<Entity> {
    vec![
        entity("Geppetto", "PERSON"),
        entity("Pinocchio", "PERSON"),
        entity("Bottega", "LOCATION"),
    ]
}

#[tokio::test]
async fn relations_are_extracted_validated_and_normalized() {
    // good pair, unknown target, self loop, and a free-form type label
    let content = serde_json::json!({"relationships": [
        {"source": "Geppetto", "target": "Pinocchio", "description": "lo ha costruito", "type": "CREATED", "strength": 0.9},
        {"source": "Geppetto", "target": "Mangiafuoco", "description": "inventato", "type": "FRIEND_OF", "strength": 0.9},
        {"source": "Pinocchio", "target": "Pinocchio", "description": "se stesso", "type": "OTHER", "strength": 0.9},
        {"source": "Geppetto", "target": "Bottega", "description": "vive nella bottega", "type": "lives in", "strength": 2.0}
    ]})
    .to_string();
    let fake = fake_server("200 OK", content);
    let ents = entities();

    let rels = extractor(&fake.base_url)
        .extract_relationships_for_entities(&chunk(TEXT), &ents)
        .await
        .unwrap();

    assert_eq!(
        rels.len(),
        2,
        "unknown entity and self-loop dropped: {rels:?}"
    );
    assert_eq!(rels[0].source, ents[0].id);
    assert_eq!(rels[0].target, ents[1].id);
    assert_eq!(rels[0].relation_type, "CREATED");
    assert_eq!(rels[0].context, vec![ChunkId::new("c1".to_string())]);
    assert_eq!(
        rels[1].relation_type, "LOCATED_IN",
        "type goes through the ontology"
    );
    assert_eq!(rels[1].confidence, 1.0);

    // the request: relation-only prompt + schema constrained to the chunk entities
    let requests = fake.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let prompt = requests[0]["messages"][0]["content"].as_str().unwrap();
    assert!(prompt.contains("- Pinocchio (PERSON)") && prompt.contains(TEXT));
    let schema = &requests[0]["response_format"]["json_schema"]["schema"];
    let props = &schema["properties"]["relationships"]["items"]["properties"];
    assert_eq!(
        props["source"]["enum"],
        serde_json::json!(["Geppetto", "Pinocchio", "Bottega"])
    );
    assert_eq!(props["source"]["enum"], props["target"]["enum"]);
}

#[tokio::test]
async fn fewer_than_two_distinct_entities_make_no_llm_call() {
    let fake = fake_server("200 OK", r#"{"relationships":[]}"#.to_string());
    let ex = extractor(&fake.base_url);
    let c = chunk(TEXT);

    assert!(ex
        .extract_relationships_for_entities(&c, &[])
        .await
        .unwrap()
        .is_empty());
    let one = vec![entity("Geppetto", "PERSON")];
    assert!(ex
        .extract_relationships_for_entities(&c, &one)
        .await
        .unwrap()
        .is_empty());
    // same surface form twice (e.g. two labels) is still one distinct entity
    let dup = vec![entity("Geppetto", "PERSON"), entity("Geppetto", "OBJECT")];
    assert!(ex
        .extract_relationships_for_entities(&c, &dup)
        .await
        .unwrap()
        .is_empty());
    assert!(fake.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn llm_failure_is_an_error_not_an_empty_result() {
    let fake = fake_server("500 Internal Server Error", String::new());
    let err = extractor(&fake.base_url)
        .extract_relationships_for_entities(&chunk(TEXT), &entities())
        .await;
    assert!(err.is_err(), "a failed LLM call must surface: {err:?}");
}

#[tokio::test]
async fn relation_responses_use_the_extraction_cache() {
    let content = serde_json::json!({"relationships": [
        {"source": "Geppetto", "target": "Pinocchio", "description": "d", "type": "CREATED", "strength": 0.8}
    ]})
    .to_string();
    let fake = fake_server("200 OK", content);
    let dir = tempfile::tempdir().unwrap();
    let cached =
        || extractor(&fake.base_url).with_cache(ExtractionCache::new(dir.path().to_path_buf()));
    let ents = entities();

    let first = cached()
        .extract_relationships_for_entities(&chunk(TEXT), &ents)
        .await
        .unwrap();
    let second = cached()
        .extract_relationships_for_entities(&chunk(TEXT), &ents)
        .await
        .unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_eq!(
        fake.requests.lock().unwrap().len(),
        1,
        "second call is a cache hit"
    );

    // a different entity list is a different prompt -> miss
    cached()
        .extract_relationships_for_entities(&chunk(TEXT), &ents[..2])
        .await
        .unwrap();
    assert_eq!(fake.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn concurrent_relation_stage_keeps_chunk_order() {
    let content = serde_json::json!({"relationships": [
        {"source": "Geppetto", "target": "Pinocchio", "description": "d", "type": "CREATED", "strength": 0.8}
    ]})
    .to_string();
    let fake = fake_server("200 OK", content);
    let ex = extractor(&fake.base_url);
    let chunks: Vec<TextChunk> = (0..5).map(|i| chunk(&format!("Testo {i}"))).collect();
    let few = [entity("Geppetto", "PERSON")];
    let many = entities();
    // chunk 2 has a single entity -> no call, empty result, still in position
    let jobs: Vec<(&TextChunk, &[Entity])> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (c, if i == 2 { &few[..] } else { &many[..] }))
        .collect();

    let mut delivered = Vec::new();
    let results = ex
        .extract_relationships_concurrent(&jobs, 3, |i| delivered.push(i))
        .await;

    assert_eq!(results.len(), 5);
    assert_eq!(delivered, vec![0, 1, 2, 3, 4]);
    let counts: Vec<usize> = results.iter().map(|r| r.as_ref().unwrap().len()).collect();
    assert_eq!(counts, vec![1, 1, 0, 1, 1]);
    assert_eq!(fake.requests.lock().unwrap().len(), 4);
}
