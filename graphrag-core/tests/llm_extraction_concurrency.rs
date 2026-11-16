//! Concurrent LLM extraction and on-disk extraction cache, against a threaded fake server.
#![cfg(feature = "llm")]

use graphrag_core::{
    core::{ChunkId, DocumentId, TextChunk},
    entity::{ExtractionCache, LLMEntityExtractor},
    llm::{LlmClient, LlmConfig},
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

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
    requests: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    models: Arc<Mutex<Vec<String>>>,
}

/// Threaded server: each connection waits `delay`, then answers with one entity
/// named after the "ITEM<n>" marker found in the prompt.
fn fake_server(delay: Duration) -> Fake {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let requests = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let models = Arc::new(Mutex::new(Vec::new()));
    let (r, f, p, m) = (
        requests.clone(),
        in_flight.clone(),
        peak.clone(),
        models.clone(),
    );
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let (r, f, p, m) = (r.clone(), f.clone(), p.clone(), m.clone());
            thread::spawn(move || {
                let body = read_body(&stream);
                r.fetch_add(1, Ordering::SeqCst);
                let now = f.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(now, Ordering::SeqCst);
                let request: serde_json::Value = serde_json::from_str(&body).unwrap();
                m.lock()
                    .unwrap()
                    .push(request["model"].as_str().unwrap_or_default().to_string());
                let prompt = request["messages"][0]["content"].as_str().unwrap();
                let n = prompt
                    .split("ITEM")
                    .nth(1)
                    .map(|s| {
                        s.chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect::<String>()
                    })
                    .unwrap_or_default();
                thread::sleep(delay);
                f.fetch_sub(1, Ordering::SeqCst);
                let content = serde_json::json!({
                    "entities": [{"name": format!("Entity{n}"), "type": "PERSON", "description": "d"}],
                    "relationships": []
                })
                .to_string();
                let reply = serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": content}}]
                })
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            });
        }
    });
    Fake {
        base_url,
        requests,
        peak,
        models,
    }
}

fn chunk(i: usize) -> TextChunk {
    let text = format!("Frammento ITEM{i} della storia.");
    TextChunk::new(
        ChunkId::new(format!("c{i}")),
        DocumentId::new("d".to_string()),
        text.clone(),
        0,
        text.len(),
    )
}

fn extractor(base_url: &str, model: &str) -> LLMEntityExtractor {
    let config = LlmConfig {
        enabled: true,
        base_url: base_url.to_string(),
        chat_model: model.to_string(),
        max_retries: 1,
        enable_caching: false,
        ..LlmConfig::default()
    };
    LLMEntityExtractor::new(LlmClient::new(config), vec!["PERSON".to_string()])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chunks_run_concurrently_and_keep_chunk_order() {
    let delay = Duration::from_millis(400);
    let fake = fake_server(delay);
    let ex = extractor(&fake.base_url, "m");
    let chunks: Vec<_> = (0..8).map(chunk).collect();

    let started = Instant::now();
    let results = ex.extract_chunks_concurrent(&chunks, 4, |_| {}).await;
    let elapsed = started.elapsed();

    assert_eq!(results.len(), 8);
    for (i, r) in results.iter().enumerate() {
        let (entities, _) = r.as_ref().unwrap();
        assert_eq!(entities[0].name, format!("Entity{i}"), "chunk order kept");
    }
    // sequential would be 8 * 400ms = 3200ms; ideal with 4 slots is 800ms
    assert!(elapsed < Duration::from_millis(2000), "took {elapsed:?}");
    assert!(fake.peak.load(Ordering::SeqCst) >= 3, "requests overlapped");
    assert!(fake.peak.load(Ordering::SeqCst) <= 4, "limit respected");
}

#[tokio::test]
async fn limit_one_is_sequential() {
    let fake = fake_server(Duration::from_millis(50));
    let ex = extractor(&fake.base_url, "m");
    let chunks: Vec<_> = (0..3).map(chunk).collect();
    let results = ex.extract_chunks_concurrent(&chunks, 1, |_| {}).await;
    assert!(results.iter().all(|r| r.is_ok()));
    assert_eq!(fake.peak.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cache_hit_skips_llm_and_key_parts_cause_misses() {
    let fake = fake_server(Duration::from_millis(5));
    let dir = tempfile::tempdir().unwrap();
    let cached = |model: &str| {
        extractor(&fake.base_url, model).with_cache(ExtractionCache::new(dir.path().to_path_buf()))
    };

    let c = chunk(1);
    let first = cached("m").extract_from_chunk(&c).await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 1);

    // same key -> hit, identical result, no new request
    let second = cached("m").extract_from_chunk(&c).await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 1);
    assert_eq!(first.0[0].name, second.0[0].name);

    // different model -> miss
    cached("other").extract_from_chunk(&c).await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 2);

    // different temperature -> miss
    cached("m")
        .with_temperature(0.5)
        .extract_from_chunk(&c)
        .await
        .unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 3);

    // different chunk text -> miss
    cached("m").extract_from_chunk(&chunk(2)).await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 4);

    // different ontology -> miss
    let mut onto = graphrag_core::ontology::Ontology::generic();
    onto.version = "9.9.9".to_string();
    cached("m")
        .with_ontology(onto)
        .extract_from_chunk(&c)
        .await
        .unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 5);
    assert!(fake.models.lock().unwrap().contains(&"other".to_string()));
}

#[tokio::test]
async fn corrupt_cache_entry_is_a_miss() {
    let fake = fake_server(Duration::from_millis(5));
    let dir = tempfile::tempdir().unwrap();
    let ex = || {
        extractor(&fake.base_url, "m").with_cache(ExtractionCache::new(dir.path().to_path_buf()))
    };
    let c = chunk(1);
    ex().extract_from_chunk(&c).await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 1);

    for entry in std::fs::read_dir(dir.path()).unwrap() {
        std::fs::write(entry.unwrap().path(), "{ not json").unwrap();
    }
    let again = ex().extract_from_chunk(&c).await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 2, "corrupt = miss");
    assert_eq!(again.0[0].name, "Entity1");
}

#[tokio::test]
async fn unwritable_cache_dir_is_not_an_error() {
    let fake = fake_server(Duration::from_millis(5));
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("a_file");
    std::fs::write(&file, "x").unwrap();
    // cache dir path is under a regular file: cannot be created
    let ex = extractor(&fake.base_url, "m").with_cache(ExtractionCache::new(file.join("sub")));
    assert!(ex.extract_from_chunk(&chunk(1)).await.is_ok());
}

#[test]
fn workspace_with_only_extraction_cache_still_initializes() {
    use graphrag_core::{Config, GraphRAG};
    let base = tempfile::tempdir().unwrap();
    // an interrupted first index leaves only the cache behind
    std::fs::create_dir_all(base.path().join("ws").join("extraction_cache")).unwrap();
    let mut config = Config::default();
    config.auto_save.enabled = true;
    config.auto_save.base_dir = Some(base.path().to_string_lossy().into_owned());
    config.auto_save.workspace_name = Some("ws".to_string());
    let mut graphrag = GraphRAG::new(config).unwrap();
    graphrag
        .initialize()
        .expect("no graph saved yet: not an error");
}

/// End-to-end through `GraphRAG::build_graph`: concurrent single-pass extraction,
/// deterministic graph, and a warm cache on the second indexing run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn build_graph_is_concurrent_and_second_run_hits_cache() {
    use graphrag_core::core::Document;
    use graphrag_core::{Config, GraphRAG};

    let fake = fake_server(Duration::from_millis(300));
    let base = tempfile::tempdir().unwrap();
    let ws = base.path().join("ws");

    let make = || {
        let mut config = Config::quick(&ws);
        config.llm.enabled = true;
        config.llm.base_url = fake.base_url.clone();
        config.llm.max_retries = 1;
        config.llm.enable_caching = false;
        config.llm.max_concurrent_requests = 4;
        config.entities.use_gleaning = false;
        config.suppress_progress_bars = true;
        GraphRAG::new(config).unwrap()
    };
    let document = || {
        let id = DocumentId::new("d".to_string());
        let chunks = (0..8)
            .map(|i| {
                let text = format!("Frammento ITEM{i} della storia.");
                TextChunk::new(
                    ChunkId::new(format!("c{i}")),
                    id.clone(),
                    text.clone(),
                    0,
                    text.len(),
                )
            })
            .collect();
        Document {
            id,
            title: "t".to_string(),
            content: String::new(),
            metadata: Default::default(),
            chunks,
        }
    };

    let mut first = make();
    first.initialize().unwrap();
    first.add_document(document()).unwrap();
    let started = Instant::now();
    first.build_graph().await.unwrap();
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_millis(2000), "took {elapsed:?}");
    assert_eq!(fake.requests.load(Ordering::SeqCst), 8);
    let names: Vec<_> = first
        .knowledge_graph()
        .unwrap()
        .entities()
        .map(|e| e.name.clone())
        .collect();
    assert_eq!(
        names,
        (0..8).map(|i| format!("Entity{i}")).collect::<Vec<_>>(),
        "graph content follows chunk order"
    );
    assert!(ws.join("extraction_cache").is_dir());

    // second run: drop the saved graph, keep the cache -> zero LLM calls
    for entry in std::fs::read_dir(&ws).unwrap() {
        let path = entry.unwrap().path();
        if path.file_name().unwrap() != "extraction_cache" {
            if path.is_dir() {
                std::fs::remove_dir_all(path).unwrap();
            } else {
                std::fs::remove_file(path).unwrap();
            }
        }
    }
    let mut second = make();
    second.initialize().unwrap();
    second.add_document(document()).unwrap();
    second.build_graph().await.unwrap();
    assert_eq!(fake.requests.load(Ordering::SeqCst), 8, "all cache hits");
    assert_eq!(second.knowledge_graph().unwrap().entities().count(), 8);
}
