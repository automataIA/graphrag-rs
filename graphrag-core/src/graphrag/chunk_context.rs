//! One context line per chunk written at index time (`chunk_context.enabled`).
//!
//! One sequential LLM call per chunk (each call needs the previous chunk's sentence). The
//! stored chunk text becomes `[Contesto: <sentence>]\n\n<original text>`, so embeddings,
//! lexical index, reranker input and answer context all carry the sentence. Extraction and
//! chapter summaries always read the original text through [`with_original_content`].

use crate::core::TextChunk;

#[cfg(feature = "async")]
use super::GraphRAG;

/// Opening of the context prefix; also the idempotency marker.
const MARKER: &str = "[Contesto: ";

/// End of the context prefix. The stored sentence has every whitespace run collapsed to a
/// single space, so `]\n\n` cannot occur inside it: the FIRST occurrence ends the prefix.
const PREFIX_END: &str = "]\n\n";

#[cfg(feature = "async")]
const SYSTEM_PROMPT: &str = "Sei un assistente che prepara brani di un libro per un motore di ricerca. Scrivi UNA sola frase in italiano (massimo 35 parole) che colloca il BRANO nella storia: chi sono i personaggi presenti, chiamati col loro nome proprio anche se nel brano sono indicati solo con soprannomi o pronomi, dove si trovano e cosa sta succedendo. Usa solo informazioni presenti nel testo fornito. Rispondi solo con la frase.";

/// Original chunk text: the content without a leading `[Contesto: …]` prefix (unchanged
/// when there is none).
pub fn strip_chunk_context(content: &str) -> &str {
    if !content.starts_with(MARKER) {
        return content;
    }
    content
        .split_once(PREFIX_END)
        .map_or(content, |(_, original)| original)
}

/// Copy of `chunk` whose content is the original text (see [`strip_chunk_context`]).
pub(super) fn with_original_content(chunk: &TextChunk) -> TextChunk {
    let mut chunk = chunk.clone();
    chunk.content = strip_chunk_context(&chunk.content).to_string();
    chunk
}

/// Single user prompt: system text, blank line, then the situation/passages/chunk layout.
#[cfg(feature = "async")]
fn build_prompt(window: &[&str], previous: Option<&str>, chunk: &str) -> String {
    let mut prompt = format!("{SYSTEM_PROMPT}\n\n");
    if let Some(sentence) = previous {
        prompt.push_str(&format!("SITUAZIONE PRECEDENTE: {sentence}\n\n"));
    }
    if !window.is_empty() {
        prompt.push_str(&format!("BRANI PRECEDENTI:\n{}\n\n", window.join("\n\n")));
    }
    prompt.push_str(&format!("BRANO:\n{chunk}"));
    prompt
}

#[cfg(feature = "async")]
impl GraphRAG {
    /// Prepend the context sentence to the stored chunks of the pending documents (no-op
    /// unless `chunk_context.enabled` and `llm.enabled`). A failed or empty answer leaves
    /// the chunk unchanged and the next chunk simply has the last successful sentence of
    /// the same document as previous situation; this never fails the build.
    pub(super) async fn write_chunk_contexts(&mut self) {
        let settings = self.config.chunk_context.clone();
        if !settings.enabled {
            return;
        }
        if !self.config.llm.enabled {
            tracing::warn!("chunk_context.enabled is set but llm.enabled is false: no context");
            return;
        }
        let Some(graph) = self.knowledge_graph.as_ref() else {
            return;
        };

        let llm = &self.config.llm;
        let client = crate::llm::LlmClient::new(llm.clone());
        let params = crate::llm::LlmGenerationParams {
            // Reasoning models need room for the thinking block before the sentence.
            max_tokens: Some(if llm.disable_thinking {
                settings.max_tokens
            } else {
                settings.max_tokens.max(4096)
            }),
            temperature: Some(0.0),
            ..Default::default()
        };

        let (mut enriched, mut failed, mut tokens) = (0usize, 0usize, 0u64);
        let mut updates: Vec<(crate::core::ChunkId, String)> = Vec::new();
        for doc_id in &self.pending_documents {
            let Some(doc) = graph.get_document(doc_id) else {
                continue;
            };
            let chunks: Vec<TextChunk> = doc
                .chunks
                .iter()
                .filter(|c| !super::summaries::is_summary_chunk(c))
                .map(|c| with_original_content(graph.get_chunk(&c.id).unwrap_or(c)))
                .collect();
            // stored content, to tell which chunks already carry the prefix
            let stored: Vec<&str> = doc
                .chunks
                .iter()
                .filter(|c| !super::summaries::is_summary_chunk(c))
                .map(|c| graph.get_chunk(&c.id).unwrap_or(c).content.as_str())
                .collect();

            let mut previous: Option<String> = None;
            for (i, chunk) in chunks.iter().enumerate() {
                if stored[i].starts_with(MARKER) {
                    continue;
                }
                let from = i.saturating_sub(settings.window);
                let window: Vec<&str> =
                    chunks[from..i].iter().map(|c| c.content.as_str()).collect();
                let prompt = build_prompt(&window, previous.as_deref(), &chunk.content);
                match client
                    .generate_with_full_response(&prompt, params.clone())
                    .await
                {
                    Ok(response) => {
                        tokens += response.usage.map_or(0, |u| u.total_tokens);
                        // Reasoning models may put a <think> block before the sentence.
                        let text = Self::remove_thinking_tags(&response.text);
                        let sentence = text.split_whitespace().collect::<Vec<_>>().join(" ");
                        if sentence.is_empty() {
                            failed += 1;
                            tracing::warn!(chunk_id = %chunk.id, "empty chunk context skipped");
                            continue;
                        }
                        updates.push((
                            chunk.id.clone(),
                            format!("{MARKER}{sentence}{PREFIX_END}{}", chunk.content),
                        ));
                        previous = Some(sentence);
                        enriched += 1;
                    },
                    Err(e) => {
                        failed += 1;
                        tracing::warn!(chunk_id = %chunk.id, error = %e, "chunk context failed, skipped");
                    },
                }
            }
        }
        tracing::info!(enriched, failed, tokens, "chunk contexts done");

        if let Some(graph) = self.knowledge_graph.as_mut() {
            for (id, content) in updates {
                if let Some(chunk) = graph.get_chunk_mut(&id) {
                    chunk.content = content;
                }
            }
        }
    }
}

#[cfg(test)]
mod strip_tests {
    use super::*;

    #[test]
    fn strips_only_a_leading_prefix() {
        assert_eq!(
            strip_chunk_context("[Contesto: Una frase.]\n\nTesto."),
            "Testo."
        );
        assert_eq!(strip_chunk_context("Testo."), "Testo.");
        let later = "Prima.\n\n[Contesto: x]\n\nDopo.";
        assert_eq!(strip_chunk_context(later), later);
        // a `]` inside the sentence does not end the prefix, only `]` + blank line does
        assert_eq!(
            strip_chunk_context("[Contesto: Pinocchio [il burattino] corre.]\n\nTesto\n\nbis."),
            "Testo\n\nbis."
        );
        // marker without a blank-line terminator: nothing to strip
        assert_eq!(strip_chunk_context("[Contesto: monco"), "[Contesto: monco");
    }

    #[test]
    fn with_original_content_leaves_other_fields_alone() {
        use crate::core::{ChunkId, DocumentId};
        let c = TextChunk::new(
            ChunkId::new("c".into()),
            DocumentId::new("d".into()),
            "[Contesto: S.]\n\nTesto".to_string(),
            3,
            9,
        );
        let o = with_original_content(&c);
        assert_eq!(
            (o.content.as_str(), o.start_offset, o.end_offset),
            ("Testo", 3, 9)
        );
        assert_eq!(with_original_content(&o).content, "Testo");
    }
}

#[cfg(all(test, feature = "llm", feature = "async"))]
mod build_tests {
    use super::*;
    use crate::config::Config;
    use crate::core::{ChunkId, Document, DocumentId};
    use std::sync::{Arc, Mutex};

    /// Fake chat server recording every request body. Context prompts get
    /// `  Frase <n>\n  di contesto <think>x</think>` (n = 1, 2, ... in arrival order);
    /// HTTP 500 when the BRANO contains `BOOM`, an empty answer when it contains `VUOTO`.
    /// Extraction prompts get an empty result.
    fn chat_stub() -> (String, Arc<Mutex<Vec<String>>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::new()));
        let shared = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let log = shared.clone();
                std::thread::spawn(move || {
                    let mut raw = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        let n = stream.read(&mut buf).unwrap_or(0);
                        raw.extend_from_slice(&buf[..n]);
                        let text = String::from_utf8_lossy(&raw).to_string();
                        if let Some(pos) = text.find("\r\n\r\n") {
                            let len = text[..pos]
                                .to_ascii_lowercase()
                                .split("content-length:")
                                .nth(1)
                                .and_then(|r| r.lines().next())
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            if raw.len() >= pos + 4 + len || n == 0 {
                                break;
                            }
                        } else if n == 0 {
                            break;
                        }
                    }
                    let text = String::from_utf8_lossy(&raw).to_string();
                    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                    let prompt = prompt_of(&body);
                    let is_context = prompt.starts_with(SYSTEM_PROMPT);
                    let n = {
                        let mut log = log.lock().unwrap();
                        log.push(body.clone());
                        log.iter()
                            .filter(|b| prompt_of(b).starts_with(SYSTEM_PROMPT))
                            .count()
                    };
                    let brano = prompt.rsplit("BRANO:\n").next().unwrap_or("");
                    if is_context && brano.contains("BOOM") {
                        let _ = stream.write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom",
                        );
                        return;
                    }
                    let content = if !is_context {
                        r#"{"entities":[],"relationships":[]}"#.to_string()
                    } else if brano.contains("VUOTO") {
                        " ".to_string()
                    } else {
                        format!("  Frase {n}\n  di contesto <think>x</think>")
                    };
                    let json = serde_json::json!({
                        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                    })
                    .to_string();
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                            json.len()
                        )
                        .as_bytes(),
                    );
                });
            }
        });
        (url, log)
    }

    fn prompt_of(body: &str) -> String {
        serde_json::from_str::<serde_json::Value>(body).unwrap()["messages"][0]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    /// Prompts of the context requests, in arrival order.
    fn context_prompts(log: &Mutex<Vec<String>>) -> Vec<String> {
        let all: Vec<String> = log.lock().unwrap().iter().map(|b| prompt_of(b)).collect();
        all.into_iter()
            .filter(|p| p.starts_with(SYSTEM_PROMPT))
            .collect()
    }

    fn extraction_prompts(log: &Mutex<Vec<String>>) -> Vec<String> {
        let all: Vec<String> = log.lock().unwrap().iter().map(|b| prompt_of(b)).collect();
        all.into_iter()
            .filter(|p| !p.starts_with(SYSTEM_PROMPT))
            .collect()
    }

    fn doc(id: &str, texts: &[&str]) -> Document {
        let chunks = texts
            .iter()
            .enumerate()
            .map(|(n, t)| {
                TextChunk::new(
                    ChunkId::new(format!("{id}_{n}")),
                    DocumentId::new(id.to_string()),
                    t.to_string(),
                    n * 100,
                    n * 100 + 50,
                )
            })
            .collect();
        Document {
            id: DocumentId::new(id.to_string()),
            title: id.to_string(),
            content: String::new(),
            metadata: indexmap::IndexMap::new(),
            chunks,
        }
    }

    fn config(url: &str, enabled: bool) -> Config {
        let mut config = Config::default().with_llm_base_url(url);
        config.llm.max_retries = 1;
        config.llm.enable_caching = false;
        config.entities.use_gleaning = false;
        config.chunk_context.enabled = enabled;
        config.chunk_context.window = 2;
        config
    }

    fn content(g: &GraphRAG, id: &str) -> String {
        g.knowledge_graph()
            .unwrap()
            .get_chunk(&ChunkId::new(id.to_string()))
            .unwrap()
            .content
            .clone()
    }

    async fn built(cfg: Config, doc: Document) -> GraphRAG {
        let mut g = GraphRAG::new(cfg).unwrap();
        g.initialize().unwrap();
        g.add_document(doc).unwrap();
        g.build_graph()
            .await
            .expect("the pass never fails the build");
        g
    }

    const TEXTS: [&str; 4] = ["alfa uno", "beta due", "gamma tre", "delta quattro"];

    #[tokio::test]
    async fn flag_off_makes_no_request_and_keeps_contents() {
        let (url, log) = chat_stub();
        let g = built(config(&url, false), doc("d", &TEXTS)).await;
        assert!(context_prompts(&log).is_empty());
        for (n, t) in TEXTS.iter().enumerate() {
            assert_eq!(content(&g, &format!("d_{n}")), *t);
        }
    }

    #[tokio::test]
    async fn enabled_without_llm_does_nothing() {
        let (url, log) = chat_stub();
        let mut cfg = config(&url, true);
        cfg.llm.enabled = false;
        let g = built(cfg, doc("d", &TEXTS)).await;
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(content(&g, "d_0"), "alfa uno");
    }

    #[tokio::test]
    async fn flag_on_prefixes_every_chunk_and_feeds_the_window() {
        let (url, log) = chat_stub();
        let g = built(config(&url, true), doc("d", &TEXTS)).await;
        // answers are whitespace-collapsed and stripped of <think> blocks
        assert_eq!(
            content(&g, "d_0"),
            "[Contesto: Frase 1 di contesto]\n\nalfa uno"
        );
        assert_eq!(
            content(&g, "d_3"),
            "[Contesto: Frase 4 di contesto]\n\ndelta quattro"
        );

        let prompts = context_prompts(&log);
        assert_eq!(prompts.len(), 4);
        // first chunk: no previous situation, no previous passages
        assert!(prompts[0].starts_with(&format!("{SYSTEM_PROMPT}\n\nBRANO:\nalfa uno")));
        assert!(!prompts[0].contains("SITUAZIONE PRECEDENTE"));
        assert!(!prompts[0].contains("BRANI PRECEDENTI"));
        // chunk 3 (window 2): chunks 1 and 2 in original form, previous sentence, no prefix
        assert_eq!(
            prompts[3],
            format!(
                "{SYSTEM_PROMPT}\n\nSITUAZIONE PRECEDENTE: Frase 3 di contesto\n\n\
                 BRANI PRECEDENTI:\nbeta due\n\ngamma tre\n\nBRANO:\ndelta quattro"
            )
        );
        assert!(prompts.iter().all(|p| !p.contains("[Contesto:")));
        // extraction saw only original text
        let extraction = extraction_prompts(&log);
        assert!(!extraction.is_empty());
        assert!(extraction.iter().all(|p| !p.contains("[Contesto:")));

        // second build: nothing pending, no request; contents unchanged
        let before = log.lock().unwrap().len();
        let mut g = g;
        g.build_graph().await.unwrap();
        assert_eq!(log.lock().unwrap().len(), before);
    }

    #[tokio::test]
    async fn already_prefixed_chunks_make_no_request() {
        let (url, log) = chat_stub();
        let prefixed = ["[Contesto: Già.]\n\nalfa", "[Contesto: Già.]\n\nbeta"];
        let g = built(config(&url, true), doc("d", &prefixed)).await;
        assert!(context_prompts(&log).is_empty());
        assert_eq!(content(&g, "d_0"), prefixed[0]);
    }

    #[tokio::test]
    async fn failure_and_empty_answer_keep_the_chunk_and_the_build_ok() {
        let (url, log) = chat_stub();
        let texts = ["alfa", "BOOM beta", "VUOTO gamma", "delta"];
        let g = built(config(&url, true), doc("d", &texts)).await;
        assert_eq!(
            content(&g, "d_0"),
            "[Contesto: Frase 1 di contesto]\n\nalfa"
        );
        assert_eq!(content(&g, "d_1"), "BOOM beta");
        assert_eq!(content(&g, "d_2"), "VUOTO gamma");
        // the next chunk reuses the last successful sentence of the document
        assert!(content(&g, "d_3").starts_with("[Contesto: Frase "));
        let prompts = context_prompts(&log);
        assert!(prompts[3].contains("SITUAZIONE PRECEDENTE: Frase 1 di contesto"));
    }

    #[tokio::test]
    async fn summary_chunks_are_skipped() {
        let (url, log) = chat_stub();
        let mut d = doc("d", &["alfa", "beta"]);
        d.chunks.insert(
            1,
            TextChunk::new(
                super::super::summaries::summary_chunk_id(&DocumentId::new("d".into()), 0),
                DocumentId::new("d".into()),
                "[Riassunto del capitolo: I.]\nx".to_string(),
                0,
                1,
            ),
        );
        let g = built(config(&url, true), d).await;
        // two real chunks -> two requests (the summary chunk is in the document list but skipped)
        assert_eq!(context_prompts(&log).len(), 2);
        let prompts = context_prompts(&log);
        assert!(!prompts[1].contains("Riassunto"));
        assert_eq!(
            content(&g, "d_summary_0"),
            "[Riassunto del capitolo: I.]\nx"
        );
    }

    #[tokio::test]
    async fn rebuild_after_clear_graph_extracts_original_text_and_adds_no_second_prefix() {
        let (url, log) = chat_stub();
        let mut g = built(config(&url, true), doc("d", &TEXTS)).await;
        let contexts = context_prompts(&log).len();
        assert_eq!(contexts, 4);
        let extraction_before = extraction_prompts(&log).len();

        g.clear_graph().unwrap();
        g.build_graph().await.unwrap();

        let extraction = extraction_prompts(&log);
        assert!(
            extraction.len() > extraction_before,
            "second build extracts again"
        );
        assert!(extraction.iter().all(|p| !p.contains("[Contesto:")));
        assert_eq!(context_prompts(&log).len(), contexts, "already prefixed");
        for n in 0..4 {
            let c = content(&g, &format!("d_{n}"));
            assert_eq!(c.matches("[Contesto: ").count(), 1, "{c}");
        }
    }

    #[test]
    fn chunks_to_extract_hands_out_original_text() {
        let mut d = doc("d", &["alfa"]);
        d.chunks[0].content = "[Contesto: S.]\n\nalfa".into();
        let mut graph = crate::core::KnowledgeGraph::new();
        graph.add_document(d).unwrap();
        let pending: indexmap::IndexSet<DocumentId> =
            [DocumentId::new("d".into())].into_iter().collect();
        let chunks = GraphRAG::chunks_to_extract(&graph, &pending);
        assert_eq!(chunks[0].content, "alfa");
        // the store keeps the prefix
        assert!(graph
            .get_chunk(&ChunkId::new("d_0".into()))
            .unwrap()
            .content
            .starts_with("[Contesto: "));
    }

    #[tokio::test]
    async fn prefix_is_persisted_reloaded_and_retrievable() {
        let (url, _log) = chat_stub();
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        let mut cfg = config(&url, true);
        cfg.auto_save = Config::quick(&workspace).auto_save;
        let mut g = built(cfg.clone(), doc("d", &TEXTS)).await;

        let results = g.query_internal("Frase di contesto").await.unwrap();
        assert!(
            results.iter().any(|r| r.contains("[Contesto: Frase")),
            "retrieval must see the prefixed text: {results:?}"
        );

        let mut reloaded = GraphRAG::new(cfg).unwrap();
        reloaded.initialize().unwrap();
        assert_eq!(
            content(&reloaded, "d_2"),
            "[Contesto: Frase 3 di contesto]\n\ngamma tre"
        );
        // `Document::chunks` is not persisted with the chunk bodies (empty after reload):
        // the chunk store is the only copy retrieval and persistence read.
        let graph = reloaded.knowledge_graph().unwrap();
        let doc = graph.get_document(&DocumentId::new("d".into())).unwrap();
        assert!(doc.chunks.is_empty());
    }
}
