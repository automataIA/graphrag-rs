//! Per-chapter summaries written at index time (`chapter_summaries.enabled`).
//!
//! One LLM call per chapter (or per group of consecutive chunks when the document has no
//! detected chapters). Each summary is stored as an extra chunk of the knowledge graph, so
//! it gets its embedding, enters the lexical index and is persisted exactly like any other
//! chunk. Summary chunks are not part of `Document::chunks`, are never sent to entity
//! extraction and are never summarised again (`is_summary_chunk`).

use crate::core::{ChunkId, DocumentId, TextChunk};

use super::GraphRAG;

/// Chunks per group when a document carries no chapter metadata at all.
pub(super) const FALLBACK_GROUP_SIZE: usize = 12;

/// Cap of the chapter text sent in one call (characters, cut at a chunk boundary).
pub(super) const MAX_INPUT_CHARS: usize = 24_000;

const ID_INFIX: &str = "_summary_";

/// Id of the summary chunk of group `index` of document `doc`.
pub(super) fn summary_chunk_id(doc: &DocumentId, index: usize) -> ChunkId {
    ChunkId::new(format!("{doc}{ID_INFIX}{index}"))
}

/// True for chunks written by the chapter-summary pass (`<docid>_summary_<n>`).
pub fn is_summary_chunk(chunk: &TextChunk) -> bool {
    is_summary_chunk_id(&chunk.id.0)
}

/// True for the id of a chapter-summary chunk (`<docid>_summary_<n>`).
pub fn is_summary_chunk_id(id: &str) -> bool {
    id.rsplit_once(ID_INFIX)
        .is_some_and(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Consecutive chunks of one document summarised together.
#[derive(Debug, Clone)]
pub(super) struct SummaryGroup {
    /// Position among the groups of the document (0-based, part of the chunk id)
    pub index: usize,
    /// Chapter shared by the chunks (`None` in the fallback grouping)
    pub chapter: Option<String>,
    /// First and last chunk position (1-based, document order, summaries excluded)
    pub first: usize,
    pub last: usize,
    pub chunks: Vec<TextChunk>,
}

impl SummaryGroup {
    /// Fixed marker line that opens the content of the summary chunk.
    pub fn marker(&self) -> String {
        match &self.chapter {
            Some(chapter) => format!("[Riassunto del capitolo: {chapter}]"),
            None => format!("[Riassunto dei passaggi {}–{}]", self.first, self.last),
        }
    }
}

/// Group the chunks of one document (in document order, summary chunks ignored): runs of
/// consecutive chunks with the same `metadata.chapter`, or groups of
/// [`FALLBACK_GROUP_SIZE`] when no chunk carries a chapter. Chunks without chapter inside a
/// document that has some form their own runs (labelled by chunk range).
pub(super) fn group_chunks(chunks: &[TextChunk]) -> Vec<SummaryGroup> {
    let chunks: Vec<&TextChunk> = chunks.iter().filter(|c| !is_summary_chunk(c)).collect();
    let any_chapter = chunks.iter().any(|c| c.metadata.chapter.is_some());
    let mut groups: Vec<SummaryGroup> = Vec::new();
    for (pos, chunk) in chunks.iter().enumerate() {
        let chapter = chunk.metadata.chapter.clone();
        let starts_new = match groups.last() {
            None => true,
            Some(g) if any_chapter => g.chapter != chapter,
            Some(g) => g.chunks.len() >= FALLBACK_GROUP_SIZE,
        };
        if starts_new {
            groups.push(SummaryGroup {
                index: groups.len(),
                chapter,
                first: pos + 1,
                last: pos + 1,
                chunks: Vec::new(),
            });
        }
        let group = groups.last_mut().expect("group just pushed");
        group.last = pos + 1;
        group.chunks.push((*chunk).clone());
    }
    groups
}

/// Prompt of one summary call: static instructions first, the chapter text last (the
/// server prompt cache reuses the shared prefix).
pub(super) fn build_summary_prompt(text: &str) -> String {
    format!(
        "Summarise what happens in the part of the text given below (a chapter or a run of \
consecutive passages).\n\
Rules:\n\
- Write 5 to 8 sentences, in the language of the text.\n\
- Name every character and place involved, using the exact names used in the text.\n\
- Follow the chronological order of the events.\n\
- State only what the text says; add nothing from outside it.\n\
- Output only the summary, no title and no preamble.\n\n\
TEXT:\n{text}"
    )
}

/// Chapter text for one group: chunk texts joined in order, cut at a chunk boundary once
/// [`MAX_INPUT_CHARS`] would be exceeded. The flag says whether chunks were left out.
pub(super) fn group_input(group: &SummaryGroup) -> (String, bool) {
    let mut text = String::new();
    for (i, chunk) in group.chunks.iter().enumerate() {
        let extra = chunk.content.len() + if text.is_empty() { 0 } else { 2 };
        if text.len() + extra > MAX_INPUT_CHARS {
            if text.is_empty() {
                // a single oversized chunk: cut at a char boundary
                let mut end = MAX_INPUT_CHARS.min(chunk.content.len());
                while !chunk.content.is_char_boundary(end) {
                    end -= 1;
                }
                return (chunk.content[..end].to_string(), true);
            }
            return (text, i < group.chunks.len());
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&chunk.content);
    }
    (text, false)
}

#[cfg(feature = "async")]
struct SummaryJob {
    id: ChunkId,
    document_id: DocumentId,
    chapter: Option<String>,
    marker: String,
    start_offset: usize,
    end_offset: usize,
    prompt: String,
}

#[cfg(feature = "async")]
impl GraphRAG {
    /// Write the chapter summaries of the pending documents (no-op unless
    /// `chapter_summaries.enabled`). Failed calls are skipped with a warning; this never
    /// fails the build.
    pub(super) async fn write_chapter_summaries(&mut self) {
        if !self.config.chapter_summaries.enabled {
            return;
        }
        if !self.config.llm.enabled {
            tracing::warn!(
                "chapter_summaries.enabled is set but llm.enabled is false: no summaries"
            );
            return;
        }
        let Some(graph) = self.knowledge_graph.as_ref() else {
            return;
        };

        let mut jobs: Vec<SummaryJob> = Vec::new();
        for doc_id in &self.pending_documents {
            let Some(doc) = graph.get_document(doc_id) else {
                continue;
            };
            let chunks: Vec<TextChunk> = doc
                .chunks
                .iter()
                .map(|c| {
                    super::chunk_context::with_original_content(graph.get_chunk(&c.id).unwrap_or(c))
                })
                .collect();
            for group in group_chunks(&chunks) {
                let id = summary_chunk_id(doc_id, group.index);
                // already written (a rebuild after `clear_graph` or a failed extraction)
                if graph.get_chunk(&id).is_some() {
                    continue;
                }
                let (text, truncated) = group_input(&group);
                if truncated {
                    tracing::info!(
                        chunk_id = %id,
                        "chapter text cut to about {MAX_INPUT_CHARS} characters for its summary"
                    );
                }
                jobs.push(SummaryJob {
                    marker: group.marker(),
                    start_offset: group.chunks.first().map_or(0, |c| c.start_offset),
                    end_offset: group.chunks.last().map_or(0, |c| c.end_offset),
                    chapter: group.chapter.clone(),
                    prompt: build_summary_prompt(&text),
                    document_id: doc_id.clone(),
                    id,
                });
            }
        }
        if jobs.is_empty() {
            return;
        }

        let llm = &self.config.llm;
        let client = crate::llm::LlmClient::new(llm.clone());
        let params = crate::llm::LlmGenerationParams {
            // Reasoning models need room for the thinking block before the summary.
            max_tokens: Some(if llm.disable_thinking {
                self.config.chapter_summaries.max_tokens
            } else {
                self.config.chapter_summaries.max_tokens.max(4096)
            }),
            temperature: Some(0.0),
            ..Default::default()
        };
        let results = crate::entity::llm_extractor::run_ordered_concurrent(
            &jobs,
            llm.max_concurrent_requests.max(1),
            |job| {
                let client = &client;
                let params = params.clone();
                async move {
                    client
                        .generate_with_full_response(&job.prompt, params)
                        .await
                }
            },
            |_| {},
        )
        .await;

        let (mut written, mut failed, mut tokens) = (0usize, 0usize, 0u64);
        let mut summaries = Vec::new();
        for (job, result) in jobs.into_iter().zip(results) {
            match result {
                Ok(response) => {
                    tokens += response.usage.map_or(0, |u| u.total_tokens);
                    // Reasoning models may put a <think> block before the summary.
                    let text = Self::remove_thinking_tags(&response.text);
                    let text = text.trim();
                    if text.is_empty() {
                        failed += 1;
                        tracing::warn!(chunk_id = %job.id, "empty chapter summary skipped");
                        continue;
                    }
                    let mut chunk = TextChunk::new(
                        job.id,
                        job.document_id,
                        format!("{}\n{text}", job.marker),
                        job.start_offset,
                        job.end_offset,
                    );
                    chunk.metadata.chapter = job.chapter;
                    summaries.push(chunk);
                    written += 1;
                },
                Err(e) => {
                    failed += 1;
                    tracing::warn!(chunk_id = %job.id, error = %e, "chapter summary failed, skipped");
                },
            }
        }
        tracing::info!(
            groups = written + failed,
            written,
            failed,
            tokens,
            "chapter summaries done"
        );
        if let Some(graph) = self.knowledge_graph.as_mut() {
            for chunk in summaries {
                let _ = graph.add_chunk(chunk);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(doc: &str, n: usize, chapter: Option<&str>) -> TextChunk {
        let mut c = TextChunk::new(
            ChunkId::new(format!("{doc}_{n}")),
            DocumentId::new(doc.to_string()),
            format!("testo {n}"),
            n * 10,
            n * 10 + 9,
        );
        c.metadata.chapter = chapter.map(str::to_string);
        c
    }

    pub(super) fn summary(doc: &str, n: usize) -> TextChunk {
        TextChunk::new(
            summary_chunk_id(&DocumentId::new(doc.to_string()), n),
            DocumentId::new(doc.to_string()),
            "[Riassunto del capitolo: I.]\nx".to_string(),
            0,
            1,
        )
    }

    #[test]
    fn summary_chunk_ids_are_recognised() {
        assert!(is_summary_chunk(&summary("doc_ab", 0)));
        assert!(is_summary_chunk(&summary("doc_ab", 12)));
        assert!(!is_summary_chunk(&chunk("doc_ab", 3, None)));
        assert!(!is_summary_chunk(&chunk("doc_summary_x", 3, None)));
        assert_eq!(
            summary_chunk_id(&DocumentId::new("d".into()), 7).0,
            "d_summary_7"
        );
    }

    #[test]
    fn groups_follow_chapters_in_document_order() {
        let chunks = vec![
            chunk("d", 0, Some("I.")),
            chunk("d", 1, Some("I.")),
            chunk("d", 2, Some("II.")),
            chunk("d", 3, Some("III.")),
            chunk("d", 4, Some("III.")),
        ];
        let groups = group_chunks(&chunks);
        let shape: Vec<_> = groups
            .iter()
            .map(|g| (g.index, g.chapter.clone().unwrap(), g.chunks.len()))
            .collect();
        assert_eq!(
            shape,
            [
                (0, "I.".into(), 2),
                (1, "II.".into(), 1),
                (2, "III.".into(), 2)
            ]
        );
        assert_eq!(groups[2].marker(), "[Riassunto del capitolo: III.]");
        assert_eq!((groups[2].first, groups[2].last), (4, 5));
    }

    #[test]
    fn without_chapters_groups_of_twelve() {
        let chunks: Vec<_> = (0..30).map(|n| chunk("d", n, None)).collect();
        let groups = group_chunks(&chunks);
        let sizes: Vec<_> = groups.iter().map(|g| g.chunks.len()).collect();
        assert_eq!(sizes, [12, 12, 6]);
        assert_eq!(groups[1].marker(), "[Riassunto dei passaggi 13–24]");
        assert!(groups.iter().all(|g| g.chapter.is_none()));
        assert_eq!(groups[2].index, 2);
    }

    #[test]
    fn chunks_without_chapter_amid_chapters_form_their_own_run() {
        let chunks = vec![
            chunk("d", 0, None),
            chunk("d", 1, Some("I.")),
            chunk("d", 2, Some("I.")),
        ];
        let groups = group_chunks(&chunks);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].marker(), "[Riassunto dei passaggi 1–1]");
        assert_eq!(groups[1].chapter.as_deref(), Some("I."));
    }

    #[test]
    fn summary_chunks_are_excluded_from_grouping() {
        let chunks = vec![
            chunk("d", 0, Some("I.")),
            summary("d", 0),
            chunk("d", 1, Some("I.")),
        ];
        let groups = group_chunks(&chunks);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].chunks.len(), 2);
        assert!(groups[0].chunks.iter().all(|c| !is_summary_chunk(c)));
        let only: Vec<_> = (0..3).map(|n| summary("d", n)).collect();
        assert!(group_chunks(&only).is_empty());
    }

    #[test]
    fn prompt_ends_with_the_chapter_text() {
        let a = build_summary_prompt("Capitolo uno.");
        let b = build_summary_prompt("Altro testo.");
        assert!(a.ends_with("Capitolo uno."));
        assert!(a.contains("5 to 8 sentences") && a.contains("language of the text"));
        let split = a.find("TEXT:\n").unwrap();
        assert_eq!(
            a[..split],
            b[..split],
            "static prefix must not depend on text"
        );
    }

    #[test]
    fn long_chapter_input_is_cut_at_a_chunk_boundary() {
        let mut chunks = Vec::new();
        for n in 0..5 {
            let mut c = chunk("d", n, Some("I."));
            c.content = format!("{n}{}", "x".repeat(9_999));
            chunks.push(c);
        }
        let group = group_chunks(&chunks).remove(0);
        let (text, truncated) = group_input(&group);
        assert!(truncated);
        assert!(text.len() <= MAX_INPUT_CHARS);
        assert!(text.contains(&format!("1{}", "x".repeat(9_999))));
        assert!(!text.contains('2'), "whole chunks only");

        let small = group_chunks(&[chunk("d", 0, Some("I.")), chunk("d", 1, Some("I."))]).remove(0);
        assert_eq!(
            group_input(&small),
            ("testo 0\n\ntesto 1".to_string(), false)
        );
    }

    /// Indexes pinocchio.txt through the build path; returns the document chunks.
    #[cfg(feature = "json5-support")]
    fn pinocchio_chunks(summaries: bool) -> Vec<TextChunk> {
        let root = env!("CARGO_MANIFEST_DIR");
        let mut config = crate::config::loader::load_config(&format!(
            "{root}/../tests/e2e/configs/it_pinocchio_cerebras.json5"
        ))
        .unwrap();
        // do not reload a real workspace from disk
        config.auto_save.enabled = false;
        config.chapter_summaries.enabled = summaries;
        let text =
            std::fs::read_to_string(format!("{root}/../tests/e2e/data/it/pinocchio.txt")).unwrap();
        let mut graphrag = GraphRAG::new(config).unwrap();
        graphrag.initialize().unwrap();
        graphrag.add_document_from_text(&text).unwrap();
        let graph = graphrag.knowledge_graph().unwrap();
        let chunks = graph.documents().next().unwrap().chunks.clone();
        chunks
    }

    /// Flag off: the build path sets no chapter and summaries use the fallback grouping.
    /// Flag on: the 36 roman-numeral headings of the book become the chapters, and chunk
    /// ids and contents stay identical to the flag-off run.
    #[cfg(feature = "json5-support")]
    #[test]
    fn pinocchio_build_chunks_get_real_chapters_only_when_summaries_are_on() {
        let off = pinocchio_chunks(false);
        assert!(off.iter().all(|c| c.metadata.chapter.is_none()));
        assert_eq!(
            group_chunks(&off).len(),
            off.len().div_ceil(FALLBACK_GROUP_SIZE)
        );

        let on = pinocchio_chunks(true);
        let ids = |v: &[TextChunk]| v.iter().map(|c| c.id.to_string()).collect::<Vec<_>>();
        let contents = |v: &[TextChunk]| v.iter().map(|c| c.content.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&on), ids(&off));
        assert_eq!(contents(&on), contents(&off));
        // only the chapter differs: nothing else is touched
        for (a, b) in on.iter().zip(&off) {
            let mut stripped = a.metadata.clone();
            stripped.chapter = None;
            assert_eq!(format!("{stripped:?}"), format!("{:?}", b.metadata));
        }

        let without = on.iter().filter(|c| c.metadata.chapter.is_none()).count();
        let groups = group_chunks(&on);
        eprintln!(
            "pinocchio: {} chunks, {} without chapter, {} groups",
            on.len(),
            without,
            groups.len()
        );
        assert_eq!(without, 0);
        let numerals = [
            "I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.",
            "XIII.", "XIV.", "XV.", "XVI.", "XVII.", "XVIII.", "XIX.", "XX.", "XXI.", "XXII.",
            "XXIII.", "XXIV.", "XXV.", "XXVI.", "XXVII.", "XXVIII.", "XXIX.", "XXX.", "XXXI.",
            "XXXII.", "XXXIII.", "XXXIV.", "XXXV.", "XXXVI.",
        ];
        let found: Vec<_> = groups.iter().map(|g| g.chapter.clone().unwrap()).collect();
        assert_eq!(found, numerals);
    }
}

#[cfg(all(test, feature = "llm", feature = "async"))]
mod build_tests {
    use super::tests::summary;
    use super::*;
    use crate::config::Config;
    use crate::core::{Document, KnowledgeGraph};
    use std::sync::{Arc, Mutex};

    /// Fake chat server recording every request body. Summary prompts get `SUMMARY <text>`
    /// (HTTP 500 when the chapter text contains `BOOM`), extraction prompts an empty result.
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
                    log.lock().unwrap().push(body.clone());
                    let is_summary = body.contains("Summarise what happens");
                    if is_summary && body.contains("BOOM") {
                        let _ = stream.write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom",
                        );
                        return;
                    }
                    let content = if is_summary {
                        "SUMMARY text".to_string()
                    } else {
                        r#"{"entities":[],"relationships":[]}"#.to_string()
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

    fn summary_calls(log: &Mutex<Vec<String>>) -> usize {
        log.lock()
            .unwrap()
            .iter()
            .filter(|b| b.contains("Summarise what happens"))
            .count()
    }

    /// Document of three chapters: I (2 chunks), II (1 chunk, `BOOM` when `boom`), III (1).
    fn chaptered_doc(id: &str, boom: bool) -> Document {
        let layout = [
            ("I.", "uno"),
            ("I.", "due"),
            ("II.", "tre"),
            ("III.", "quattro"),
        ];
        let chunks = layout
            .iter()
            .enumerate()
            .map(|(n, (chapter, word))| {
                let text = if boom && *chapter == "II." {
                    format!("BOOM {word}")
                } else {
                    format!("capitolo {word} {id}")
                };
                let mut c = TextChunk::new(
                    ChunkId::new(format!("{id}_{n}")),
                    DocumentId::new(id.to_string()),
                    text,
                    n * 100,
                    n * 100 + 50,
                );
                c.metadata.chapter = Some(chapter.to_string());
                c
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

    fn config(url: &str, summaries: bool) -> Config {
        let mut config = Config::default().with_llm_base_url(url);
        config.llm.max_retries = 1;
        config.llm.enable_caching = false;
        config.entities.use_gleaning = false;
        config.chapter_summaries.enabled = summaries;
        config
    }

    fn summary_ids(g: &GraphRAG) -> Vec<String> {
        let mut ids: Vec<String> = g
            .knowledge_graph()
            .unwrap()
            .chunks()
            .filter(|c| is_summary_chunk(c))
            .map(|c| c.id.to_string())
            .collect();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn flag_on_writes_one_summary_chunk_per_chapter_and_skips_a_failure() {
        let (url, log) = chat_stub();
        let mut g = GraphRAG::new(config(&url, true)).unwrap();
        g.initialize().unwrap();
        g.add_document(chaptered_doc("docA", true)).unwrap();
        g.build_graph()
            .await
            .expect("a failed summary never fails the build");

        // three groups, one call each; chapter II failed and is skipped
        assert_eq!(summary_calls(&log), 3);
        assert_eq!(summary_ids(&g), ["docA_summary_0", "docA_summary_2"]);
        let graph = g.knowledge_graph().unwrap();
        let first = graph
            .get_chunk(&ChunkId::new("docA_summary_0".into()))
            .unwrap();
        assert_eq!(first.content, "[Riassunto del capitolo: I.]\nSUMMARY text");
        assert_eq!(first.metadata.chapter.as_deref(), Some("I."));
        assert_eq!(first.document_id, DocumentId::new("docA".into()));
        // the prompt carries both chunks of chapter I, in order
        let bodies = log.lock().unwrap().clone();
        let prompt = bodies
            .iter()
            .find(|b| b.contains("Summarise") && b.contains("capitolo uno"))
            .unwrap();
        let (uno, due) = (
            prompt.find("capitolo uno").unwrap(),
            prompt.find("capitolo due").unwrap(),
        );
        assert!(uno < due);
        // the document keeps only its own chunks and extraction never saw a summary
        assert_eq!(
            graph
                .get_document(&DocumentId::new("docA".into()))
                .unwrap()
                .chunks
                .len(),
            4
        );
        assert!(!bodies
            .iter()
            .any(|b| !b.contains("Summarise") && b.contains("Riassunto del capitolo")));

        // nothing pending: a second build makes no call at all
        let before = log.lock().unwrap().len();
        g.build_graph().await.unwrap();
        assert_eq!(log.lock().unwrap().len(), before);

        // a new document gets its own summaries only
        g.add_document(chaptered_doc("docB", false)).unwrap();
        g.build_graph().await.unwrap();
        assert_eq!(summary_calls(&log), 3 + 3);
        assert_eq!(
            summary_ids(&g),
            [
                "docA_summary_0",
                "docA_summary_2",
                "docB_summary_0",
                "docB_summary_1",
                "docB_summary_2"
            ]
        );
    }

    #[tokio::test]
    async fn clear_graph_rebuild_neither_extracts_nor_regenerates_summaries() {
        let (url, log) = chat_stub();
        let mut g = GraphRAG::new(config(&url, true)).unwrap();
        g.initialize().unwrap();
        g.add_document(chaptered_doc("docA", false)).unwrap();
        g.build_graph().await.unwrap();
        assert_eq!(summary_calls(&log), 3);

        g.clear_graph().unwrap();
        let graph = g.knowledge_graph().unwrap();
        let extracted = GraphRAG::chunks_to_extract(graph, &g.pending_documents);
        assert_eq!(extracted.len(), 4);
        assert!(extracted.iter().all(|c| !is_summary_chunk(c)));
        g.build_graph().await.unwrap();
        assert_eq!(summary_calls(&log), 3, "existing summaries are kept");
    }

    #[test]
    fn chunks_to_extract_skips_summary_chunks_even_inside_a_document() {
        let mut doc = chaptered_doc("docA", false);
        doc.chunks.push(summary("docA", 0));
        let mut graph = KnowledgeGraph::new();
        graph.add_document(doc).unwrap();
        let pending: indexmap::IndexSet<DocumentId> =
            [DocumentId::new("docA".into())].into_iter().collect();
        let ids: Vec<String> = GraphRAG::chunks_to_extract(&graph, &pending)
            .into_iter()
            .map(|c| c.id.to_string())
            .collect();
        assert_eq!(ids, ["docA_0", "docA_1", "docA_2", "docA_3"]);
    }

    #[tokio::test]
    async fn flag_off_makes_no_summary_call_and_adds_no_chunk() {
        let (url, log) = chat_stub();
        let mut g = GraphRAG::new(config(&url, false)).unwrap();
        g.initialize().unwrap();
        g.add_document(chaptered_doc("docA", false)).unwrap();
        g.build_graph().await.unwrap();
        assert_eq!(summary_calls(&log), 0);
        assert_eq!(g.knowledge_graph().unwrap().chunks().count(), 4);
    }

    #[tokio::test]
    async fn enabled_without_llm_does_nothing() {
        let (url, log) = chat_stub();
        let mut cfg = config(&url, true);
        cfg.llm.enabled = false;
        let mut g = GraphRAG::new(cfg).unwrap();
        g.initialize().unwrap();
        g.add_document(chaptered_doc("docA", false)).unwrap();
        g.build_graph().await.unwrap();
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(g.knowledge_graph().unwrap().chunks().count(), 4);
    }

    #[tokio::test]
    async fn summary_chunks_are_embedded_retrievable_and_persisted() {
        let (url, _log) = chat_stub();
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        let mut cfg = config(&url, true);
        cfg.auto_save = Config::quick(&workspace).auto_save;
        let mut g = GraphRAG::new(cfg.clone()).unwrap();
        g.initialize().unwrap();
        g.add_document(chaptered_doc("docA", false)).unwrap();
        g.build_graph().await.unwrap();

        // embeddings are added to every graph chunk at query time (hash fallback is offline)
        let results = g.query_internal("Riassunto del capitolo").await.unwrap();
        let graph = g.knowledge_graph().unwrap();
        let id = ChunkId::new("docA_summary_1".into());
        assert!(graph.get_chunk(&id).unwrap().embedding.is_some());
        assert!(
            results
                .iter()
                .any(|r| r.contains("[Riassunto del capitolo: II.]")),
            "summary chunk must be retrievable: {results:?}"
        );

        // the workspace saved by the build holds the summaries and reloads with them
        let mut reloaded = GraphRAG::new(cfg).unwrap();
        reloaded.initialize().unwrap();
        assert_eq!(
            summary_ids(&reloaded),
            ["docA_summary_0", "docA_summary_1", "docA_summary_2"]
        );
        let chunk = reloaded.knowledge_graph().unwrap().get_chunk(&id).unwrap();
        assert!(chunk.content.starts_with("[Riassunto del capitolo: II.]"));
    }

    #[tokio::test]
    async fn documents_without_chapters_are_summarised_in_groups_of_twelve() {
        let (url, log) = chat_stub();
        let mut cfg = config(&url, true);
        cfg.text.chunk_size = 100;
        cfg.text.chunk_overlap = 0;
        let mut g = GraphRAG::new(cfg).unwrap();
        g.initialize().unwrap();
        g.add_document_from_text(&"parola ".repeat(400)).unwrap();
        let chunks = g.knowledge_graph().unwrap().chunks().count();
        assert!(chunks > FALLBACK_GROUP_SIZE);
        g.build_graph().await.unwrap();
        let groups = chunks.div_ceil(FALLBACK_GROUP_SIZE);
        assert_eq!(summary_calls(&log), groups);
        let graph = g.knowledge_graph().unwrap();
        let first = graph
            .chunks()
            .find(|c| c.id.0.ends_with("_summary_0"))
            .unwrap();
        assert!(first.content.starts_with("[Riassunto dei passaggi 1–12]"));
        assert!(first.metadata.chapter.is_none());
    }
}
