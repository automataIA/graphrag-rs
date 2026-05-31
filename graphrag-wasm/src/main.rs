#![recursion_limit = "512"]

//! GraphRAG WASM — chat-shell layout (Nordic-Minimal) backed by real pipeline:
//! ONNX embeddings, WebLLM (Qwen / Phi-3), pure-Rust vector search, IndexedDB
//! persistence, graphrag-core entity extraction.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{Event, HtmlInputElement};

use graphrag_core::{core::GraphRAGError, Config, GraphRAG};

mod components;
mod entity_extractor;
mod onnx_embedder;
mod persist;
mod storage;
mod vector_search;
mod webllm;

use components::chat_shell::{
    build_ref_cards, build_subgraph, parse_answer_with_cites, AnswerSegment, ChatTurn, NodeKind,
    RefCard, SubgraphData, SubgraphNode,
};
use entity_extractor::{extract_entities, extract_entities_simple, Entity, Relationship};
use onnx_embedder::OnnxEmbedder;
use vector_search::VectorIndex;
use webllm::WebLLM;

#[allow(dead_code)]
type GraphRAGResult<T> = Result<T, GraphRAGError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub name: String,
    pub content: String,
    pub size_bytes: usize,
    pub added_at: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BuildStatus {
    Idle,
    Building(BuildStage),
    Ready,
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum BuildStage {
    Chunking { progress: f32 },
    Extracting { progress: f32 },
    Embedding { progress: f32 },
    Indexing { progress: f32 },
}

#[derive(Clone, Debug, Default)]
pub struct GraphStats {
    pub documents: usize,
    pub chunks: usize,
    pub entities: usize,
    pub relationships: usize,
    pub embeddings: usize,
}

const EMBED_MODEL_LABEL: &str = "all-MiniLM-L6";
const LLM_MODEL_ID: &str = "Phi-3-mini-4k-instruct-q4f16_1-MLC";
const LLM_LABEL: &str = "Phi-3-mini";

// ─── App root ───────────────────────────────────────────────────────────────

#[component]
fn App() -> impl IntoView {
    let (documents, set_documents) = signal(Vec::<Document>::new());
    let (build_status, set_build_status) = signal(BuildStatus::Idle);
    let (graph_stats, set_graph_stats) = signal(GraphStats::default());
    let graphrag_instance: StoredValue<Option<GraphRAG>> = StoredValue::new(None);
    let vector_index: StoredValue<Option<VectorIndex>> = StoredValue::new(None);

    let (turns, set_turns) = signal(Vec::<ChatTurn>::new());
    let (subgraph, set_subgraph) = signal(SubgraphData::default());
    let (active_ref, set_active_ref) = signal::<Option<u32>>(None);
    let (query_input, set_query_input) = signal(String::new());
    let (loading, set_loading) = signal(false);
    let (view_mode_hier, set_view_mode_hier) = signal(false);

    // Restore documents from IndexedDB
    spawn_local(async move {
        if let Some(store) = persist::open_store().await {
            let saved = persist::load_all_documents(&store).await;
            if !saved.is_empty() {
                web_sys::console::log_1(
                    &format!("📦 Restored {} document(s) from IndexedDB", saved.len()).into(),
                );
                set_documents.set(saved);
            }
        }
    });

    // Document mutation helpers
    let push_document = move |doc: Document| {
        let mut docs = documents.get_untracked();
        docs.push(doc.clone());
        set_documents.set(docs);
        spawn_local(async move {
            if let Some(store) = persist::open_store().await {
                persist::save_document(&store, &doc).await;
            }
        });
    };

    let remove_document = move |id: String| {
        let docs = documents.get_untracked();
        let filtered: Vec<_> = docs.into_iter().filter(|d| d.id != id).collect();
        set_documents.set(filtered);
        spawn_local(async move {
            if let Some(store) = persist::open_store().await {
                persist::delete_document(&store, &id).await;
            }
        });
    };

    let on_file_upload = move |ev: Event| {
        let input = match ev.target().and_then(|t| t.dyn_into::<HtmlInputElement>().ok()) {
            Some(i) => i,
            None => return,
        };
        if let Some(files) = input.files() {
            for i in 0..files.length() {
                if let Some(file) = files.get(i) {
                    let file_name = file.name();
                    spawn_local(async move {
                        let reader = match web_sys::FileReader::new() {
                            Ok(r) => r,
                            Err(_) => return,
                        };
                        let reader_clone = reader.clone();
                        let push = push_document;
                        let onload = Closure::wrap(Box::new(move |_: Event| {
                            if let Ok(result) = reader_clone.result() {
                                if let Some(text) = result.as_string() {
                                    push(Document {
                                        id: format!("doc-{}", js_sys::Date::now()),
                                        name: file_name.clone(),
                                        content: text.clone(),
                                        size_bytes: text.len(),
                                        added_at: js_sys::Date::now(),
                                    });
                                }
                            }
                        }) as Box<dyn Fn(Event)>);
                        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                        onload.forget();
                        let _ = reader.read_as_text(&file);
                    });
                }
            }
        }
    };

    let load_symposium = move |_| {
        spawn_local(async move {
            web_sys::console::log_1(&"📖 Loading Symposium demo...".into());
            match gloo_net::http::Request::get("./Symposium.txt").send().await {
                Ok(response) => match response.text().await {
                    Ok(text) => {
                        push_document(Document {
                            id: format!("doc-{}", js_sys::Date::now()),
                            name: "Plato's Symposium".to_string(),
                            content: text.clone(),
                            size_bytes: text.len(),
                            added_at: js_sys::Date::now(),
                        });
                        web_sys::console::log_1(&"✅ Symposium loaded".into());
                    },
                    Err(e) => {
                        web_sys::console::error_1(&format!("Read failed: {:?}", e).into());
                    },
                },
                Err(e) => {
                    web_sys::console::error_1(&format!("Fetch failed: {:?}", e).into());
                },
            }
        });
    };

    // ── Build pipeline ──────────────────────────────────────────────────────
    let build_graph = move |_| {
        let docs = documents.get_untracked();
        if docs.is_empty() {
            return;
        }
        set_build_status.set(BuildStatus::Building(BuildStage::Chunking { progress: 0.0 }));

        spawn_local(async move {
            web_sys::console::log_1(&"🚀 Build start".into());

            let config = Config::default();
            let mut graphrag = match GraphRAG::new(config) {
                Ok(g) => g,
                Err(e) => {
                    set_build_status.set(BuildStatus::Error(format!("new(): {}", e)));
                    return;
                },
            };
            if let Err(e) = graphrag.initialize() {
                set_build_status.set(BuildStatus::Error(format!("init(): {}", e)));
                return;
            }

            let total_docs = docs.len();
            for (i, doc) in docs.iter().enumerate() {
                set_build_status.set(BuildStatus::Building(BuildStage::Chunking {
                    progress: ((i + 1) as f32 / total_docs as f32) * 100.0,
                }));
                if let Err(e) = graphrag.add_document_from_text(&doc.content) {
                    web_sys::console::warn_1(
                        &format!("add_document {}: {}", doc.name, e).into(),
                    );
                }
                gloo_timers::future::TimeoutFuture::new(30).await;
            }
            let mut all_chunks = 0;
            if let Some(kg) = graphrag.knowledge_graph() {
                all_chunks = kg.chunks().count();
            }

            // Entity extraction
            set_build_status.set(BuildStatus::Building(BuildStage::Extracting {
                progress: 0.0,
            }));
            let llm_result = WebLLM::new("Qwen2-1.5B-Instruct-q4f16_1-MLC").await;
            let mut all_entities: Vec<Entity> = Vec::new();
            let mut all_relationships: Vec<Relationship> = Vec::new();
            if let Ok(llm) = llm_result {
                for (idx, doc) in docs.iter().enumerate() {
                    set_build_status.set(BuildStatus::Building(BuildStage::Extracting {
                        progress: ((idx + 1) as f32 / docs.len() as f32) * 100.0,
                    }));
                    match extract_entities(&llm, &doc.content).await {
                        Ok(result) => {
                            all_entities.extend(result.entities);
                            all_relationships.extend(result.relationships);
                        },
                        Err(e) => web_sys::console::warn_1(
                            &format!("extract_entities: {}", e).into(),
                        ),
                    }
                    gloo_timers::future::TimeoutFuture::new(100).await;
                }
            } else {
                web_sys::console::warn_1(&"WebLLM offline — rule-based extraction".into());
                for doc in docs.iter() {
                    let result = extract_entities_simple(&doc.content);
                    all_entities.extend(result.entities);
                    all_relationships.extend(result.relationships);
                }
            }
            all_entities.sort_by(|a, b| a.name.cmp(&b.name));
            all_entities.dedup_by(|a, b| a.name == b.name);
            let entity_count = all_entities.len();
            let relationship_count = all_relationships.len();

            // Embeddings
            set_build_status.set(BuildStatus::Building(BuildStage::Embedding {
                progress: 0.0,
            }));
            use gloo_net::http::Request;
            let tokenizer_result = Request::get("./tokenizer.json").send().await;
            let embedder_result = if let Ok(response) = tokenizer_result {
                if let Ok(tokenizer_json) = response.text().await {
                    OnnxEmbedder::from_tokenizer_json(384, &tokenizer_json)
                } else {
                    Err(onnx_embedder::OnnxEmbedderError::InvalidInput(
                        "tokenizer.json read failed".to_string(),
                    ))
                }
            } else {
                Err(onnx_embedder::OnnxEmbedderError::InvalidInput(
                    "tokenizer.json fetch failed".to_string(),
                ))
            };
            let mut embedder = match embedder_result {
                Ok(e) => Some(e),
                Err(e) => {
                    web_sys::console::warn_1(&format!("embedder: {}", e).into());
                    None
                },
            };
            if let Some(ref mut emb) = embedder {
                if let Err(e) = emb.load_model("./models/minilm-l6.onnx", true).await {
                    web_sys::console::warn_1(&format!("load_model: {}", e).into());
                    embedder = None;
                }
            }
            let mut chunk_data: Vec<(String, String)> = Vec::new();
            if let Some(kg) = graphrag.knowledge_graph() {
                for chunk in kg.chunks() {
                    chunk_data.push((chunk.id.0.clone(), chunk.content.clone()));
                }
            }
            let embedding_dim = 384usize;
            let mut embeddings: Vec<Vec<f32>> = Vec::new();
            if let Some(ref emb) = embedder {
                for (i, (_id, content)) in chunk_data.iter().enumerate() {
                    set_build_status.set(BuildStatus::Building(BuildStage::Embedding {
                        progress: ((i + 1) as f32 / chunk_data.len() as f32) * 100.0,
                    }));
                    if let Ok(embedding) = emb.embed(content).await {
                        embeddings.push(embedding);
                    }
                    if i % 10 == 0 {
                        gloo_timers::future::TimeoutFuture::new(10).await;
                    }
                }
            } else {
                for (i, (_id, content)) in chunk_data.iter().enumerate() {
                    set_build_status.set(BuildStatus::Building(BuildStage::Embedding {
                        progress: ((i + 1) as f32 / chunk_data.len() as f32) * 100.0,
                    }));
                    let mut embedding = vec![0.0f32; embedding_dim];
                    let bytes = content.as_bytes();
                    for (idx, chunk) in bytes.chunks(4).enumerate() {
                        let hash = chunk
                            .iter()
                            .fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
                        let normalized = (hash as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                        embedding[idx % embedding_dim] += normalized;
                    }
                    let norm: f32 =
                        embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut embedding {
                            *x /= norm;
                        }
                    }
                    embeddings.push(embedding);
                    if i % 10 == 0 {
                        gloo_timers::future::TimeoutFuture::new(10).await;
                    }
                }
            }

            // Index
            set_build_status
                .set(BuildStatus::Building(BuildStage::Indexing { progress: 50.0 }));
            let mut index = VectorIndex::new();
            for ((chunk_id, _content), embedding) in chunk_data.iter().zip(embeddings.iter()) {
                index.add(embedding.clone(), chunk_id.clone(), chunk_id.clone());
            }
            set_build_status
                .set(BuildStatus::Building(BuildStage::Indexing { progress: 100.0 }));
            gloo_timers::future::TimeoutFuture::new(80).await;

            vector_index.set_value(Some(index));
            graphrag_instance.set_value(Some(graphrag));
            set_build_status.set(BuildStatus::Ready);
            set_graph_stats.set(GraphStats {
                documents: total_docs,
                chunks: all_chunks,
                entities: entity_count,
                relationships: relationship_count,
                embeddings: embeddings.len(),
            });
            web_sys::console::log_1(
                &format!(
                    "✅ Build complete: {} docs, {} chunks, {} entities",
                    total_docs, all_chunks, entity_count
                )
                .into(),
            );
        });
    };

    // ── Query pipeline ──────────────────────────────────────────────────────
    let run_query = move |query_text: String| {
        if query_text.trim().is_empty() {
            return;
        }
        if !matches!(build_status.get_untracked(), BuildStatus::Ready) {
            web_sys::console::warn_1(&"Graph not ready — build first".into());
            return;
        }
        set_loading.set(true);
        let docs_snapshot = documents.get_untracked();
        let doc_lookup: std::collections::HashMap<String, String> = docs_snapshot
            .iter()
            .enumerate()
            .map(|(idx, d)| (format!("doc_{}", idx), d.name.clone()))
            .collect();

        spawn_local(async move {
            let t_start = js_sys::Date::now();

            use gloo_net::http::Request;
            let tokenizer_result = Request::get("./tokenizer.json").send().await;
            let embedder_result = if let Ok(response) = tokenizer_result {
                if let Ok(tokenizer_json) = response.text().await {
                    OnnxEmbedder::from_tokenizer_json(384, &tokenizer_json)
                } else {
                    Err(onnx_embedder::OnnxEmbedderError::InvalidInput("read".into()))
                }
            } else {
                Err(onnx_embedder::OnnxEmbedderError::InvalidInput("fetch".into()))
            };

            let top: Vec<(String, f64)> = match embedder_result {
                Ok(mut embedder) => {
                    if embedder
                        .load_model("./models/minilm-l6.onnx", true)
                        .await
                        .is_ok()
                    {
                        if let Ok(q_emb) = embedder.embed(&query_text).await {
                            vector_index.with_value(|idx_opt| {
                                idx_opt
                                    .as_ref()
                                    .map(|idx| {
                                        idx.search(&q_emb, 5)
                                            .iter()
                                            .map(|r| (r.id.clone(), r.similarity))
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                },
                Err(_) => Vec::new(),
            };

            // Build refs + subgraph
            let (refs, subg) = graphrag_instance.with_value(|gr_opt| {
                if let Some(gr) = gr_opt {
                    let refs = build_ref_cards(gr, &top, &doc_lookup);
                    let top_ids: Vec<String> =
                        top.iter().map(|(id, _)| id.clone()).collect();
                    let sg = build_subgraph(gr, &top_ids);
                    (refs, sg)
                } else {
                    (Vec::new(), SubgraphData::default())
                }
            });
            set_subgraph.set(subg);

            // Build context for LLM
            let mut context_for_llm = String::new();
            graphrag_instance.with_value(|gr_opt| {
                if let Some(gr) = gr_opt {
                    for (i, (chunk_id, _)) in top.iter().enumerate() {
                        if let Some(chunk) = gr.get_chunk(chunk_id) {
                            context_for_llm.push_str(&format!(
                                "[Source {}]\n{}\n\n",
                                i + 1,
                                chunk.content
                            ));
                        }
                    }
                }
            });

            // LLM synthesis
            let answer_text: String = if context_for_llm.is_empty() {
                "No relevant sources retrieved.".to_string()
            } else {
                match WebLLM::new(LLM_MODEL_ID).await {
                    Ok(llm) => {
                        let messages = vec![
                            webllm::ChatMessage::system(
                                "You answer questions strictly from the provided sources. \
                                Cite each claim immediately with [N] where N is the 1-indexed \
                                source number you used. Keep answers concise (2–4 sentences).",
                            ),
                            webllm::ChatMessage::user(format!(
                                "Question: {}\n\nSources:\n{}",
                                query_text, context_for_llm
                            )),
                        ];
                        match llm.chat(messages, Some(0.4), Some(420)).await {
                            Ok(a) => a,
                            Err(e) => {
                                web_sys::console::error_1(&format!("LLM: {}", e).into());
                                format!("(LLM unavailable — {} retrieved sources)", refs.len())
                            },
                        }
                    },
                    Err(e) => {
                        web_sys::console::error_1(&format!("WebLLM init: {}", e).into());
                        format!("(LLM unavailable — {} retrieved sources)", refs.len())
                    },
                }
            };

            let segments: Vec<AnswerSegment> =
                parse_answer_with_cites(&answer_text, refs.len());
            let dt_ms = (js_sys::Date::now() - t_start) as u32;
            let eyebrow = format!(
                "hybrid retrieval · k={} · {}ms",
                top.len().max(1),
                dt_ms
            );
            let turn = ChatTurn {
                user_q: query_text,
                eyebrow,
                answer: segments,
                refs,
                timestamp: format_now(),
            };
            set_turns.update(|t| t.push(turn));
            set_loading.set(false);
        });
    };

    let on_submit_query = move |_| {
        let q = query_input.get_untracked();
        if q.trim().is_empty() {
            return;
        }
        set_query_input.set(String::new());
        run_query(q);
    };

    // Derived: can_build / can_query
    let can_build = move || {
        !documents.get().is_empty() && !matches!(build_status.get(), BuildStatus::Building(_))
    };
    let is_ready = move || matches!(build_status.get(), BuildStatus::Ready);

    view! {
        <div class="bg-fx"></div>
        <div class="app">
            <LeftRail
                documents=documents
                build_status=build_status
                graph_stats=graph_stats
                view_mode_hier=view_mode_hier
                set_view_mode_hier=set_view_mode_hier
                on_file_upload=on_file_upload
                on_load_demo=load_symposium
                on_build=build_graph
                on_remove=remove_document
                can_build=can_build
            />

            <main class="stage" data-screen-label="Chat thread">
                <StageHead documents=documents build_status=build_status />
                <div class="stage-scroll">
                    {move || {
                        let turn_list = turns.get();
                        if turn_list.is_empty() {
                            view! {
                                <div class="empty-stage">
                                    <p>
                                        <strong>"Add a source"</strong>
                                        " (left rail), press "
                                        <strong>"Rebuild graph"</strong>
                                        ", then ask a question below."
                                    </p>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="thread">
                                    {turn_list.into_iter().enumerate().map(|(idx, t)| view! {
                                        <TurnView
                                            turn=t
                                            turn_idx=idx
                                            active_ref=active_ref
                                            set_active_ref=set_active_ref
                                        />
                                    }).collect_view()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
                <Composer
                    query_input=query_input
                    set_query_input=set_query_input
                    on_submit=on_submit_query
                    is_ready=is_ready
                    loading=loading
                />
            </main>

            <RightRail
                subgraph=subgraph
                build_status=build_status
                graph_stats=graph_stats
                turns=turns
                active_ref=active_ref
                set_active_ref=set_active_ref
            />
        </div>
    }
}

// ─── Left rail ──────────────────────────────────────────────────────────────

#[component]
fn LeftRail<FU, FD, FB, FR, FC>(
    documents: ReadSignal<Vec<Document>>,
    build_status: ReadSignal<BuildStatus>,
    graph_stats: ReadSignal<GraphStats>,
    view_mode_hier: ReadSignal<bool>,
    set_view_mode_hier: WriteSignal<bool>,
    on_file_upload: FU,
    on_load_demo: FD,
    on_build: FB,
    on_remove: FR,
    can_build: FC,
) -> impl IntoView
where
    FU: Fn(Event) + Copy + 'static + Send + Sync,
    FD: Fn(web_sys::MouseEvent) + Copy + 'static + Send + Sync,
    FB: Fn(web_sys::MouseEvent) + Copy + 'static + Send + Sync,
    FR: Fn(String) + Copy + 'static + Send + Sync,
    FC: Fn() -> bool + Copy + 'static + Send + Sync,
{
    let indexed_chip = move || match build_status.get() {
        BuildStatus::Ready => view! {
            <span class="chip chip-mono chip-on"><span class="dot dot-on"></span> "indexed"</span>
        }
        .into_any(),
        BuildStatus::Building(_) => view! {
            <span class="chip chip-mono chip-warn"><span class="dot dot-pulse"></span> "building"</span>
        }
        .into_any(),
        BuildStatus::Error(_) => view! {
            <span class="chip chip-mono"><span class="dot"></span> "error"</span>
        }
        .into_any(),
        BuildStatus::Idle => view! {
            <span class="chip chip-mono"><span class="dot"></span> "idle"</span>
        }
        .into_any(),
    };

    view! {
        <aside class="rail rail-left" data-screen-label="Library">
            <div class="rail-section">
                <div class="rail-h">
                    <span class="rail-h-eyebrow">"Workspace"</span>
                    <label class="rail-cmd" title="Upload document">
                        "+"
                        <input
                            type="file"
                            multiple
                            accept=".txt,.md"
                            on:change=on_file_upload
                        />
                    </label>
                </div>
                <h1 class="brand">
                    <span class="brand-mark">"◐"</span>
                    <span class="brand-name">"graphrag"</span>
                    <span class="brand-sub">".rs"</span>
                </h1>
                <div class="status-row">
                    {indexed_chip}
                    <span class="chip chip-mono">"wasm · webgpu"</span>
                </div>
            </div>

            <div class="rail-section">
                <div class="rail-h">
                    <span class="rail-h-eyebrow">
                        {move || format!("Sources · {}", documents.get().len())}
                    </span>
                    <button class="rail-cmd" title="Load demo" on:click=on_load_demo>
                        "S"
                    </button>
                </div>
                {move || {
                    let docs = documents.get();
                    if docs.is_empty() {
                        view! {
                            <div class="doc-empty">"No sources yet. Press “S” to load Plato’s Symposium."</div>
                        }.into_any()
                    } else {
                        view! {
                            <ul class="doc-list">
                                {docs.into_iter().enumerate().map(|(idx, d)| {
                                    let glyph_letter = d.name
                                        .chars()
                                        .find(|c| c.is_alphanumeric())
                                        .map(|c| c.to_uppercase().to_string())
                                        .unwrap_or_else(|| "D".to_string());
                                    let tok_estimate = (d.size_bytes / 5).max(1);
                                    let tok_str = if tok_estimate >= 1000 {
                                        format!("{}k tok", tok_estimate / 1000)
                                    } else {
                                        format!("{} tok", tok_estimate)
                                    };
                                    let is_active = idx == 0;
                                    let class_str = if is_active {
                                        "doc-item is-active"
                                    } else {
                                        "doc-item"
                                    };
                                    let id = d.id.clone();
                                    let glyph_class = if idx == 0 { "hi" } else if idx == 1 { "accent" } else { "" };
                                    view! {
                                        <li class=class_str>
                                            <div class="doc-glyph" data-c=glyph_class>{glyph_letter}</div>
                                            <div class="doc-meta">
                                                <div class="doc-title">{d.name.clone()}</div>
                                                <div class="doc-sub">
                                                    <span>{infer_author(&d.name)}</span>
                                                    <span class="dotsep">"·"</span>
                                                    <span>{tok_str}</span>
                                                </div>
                                            </div>
                                            {is_active.then(|| view! { <span class="doc-active-bar"></span> })}
                                            <button
                                                class="ghost-btn"
                                                style="margin-left:auto"
                                                title="Remove"
                                                on:click=move |_| on_remove(id.clone())
                                            >
                                                "×"
                                            </button>
                                        </li>
                                    }
                                }).collect_view()}
                            </ul>
                        }.into_any()
                    }
                }}
            </div>

            <div class="rail-section">
                <div class="rail-h"><span class="rail-h-eyebrow">"View"</span></div>
                <div class="seg">
                    <button
                        class=move || if !view_mode_hier.get() { "seg-btn is-on" } else { "seg-btn" }
                        on:click=move |_| set_view_mode_hier.set(false)
                    >
                        "Flat"
                    </button>
                    <button
                        class=move || if view_mode_hier.get() { "seg-btn is-on" } else { "seg-btn" }
                        on:click=move |_| set_view_mode_hier.set(true)
                    >
                        "Hierarchy"
                    </button>
                </div>
            </div>

            <div class="rail-section rail-foot">
                <button
                    class="build-btn"
                    on:click=on_build
                    prop:disabled=move || !can_build()
                >
                    {move || match build_status.get() {
                        BuildStatus::Building(_) => "Building…".to_string(),
                        BuildStatus::Ready => "Rebuild graph".to_string(),
                        _ => "Build graph".to_string(),
                    }}
                    <span class="kbd">"⏎"</span>
                </button>
                <div class="foot-meta">
                    <span class="chip chip-mono">{EMBED_MODEL_LABEL}</span>
                    <span class="chip chip-mono">{LLM_LABEL}</span>
                    <span class="chip chip-mono">
                        {move || format!("{} ent", graph_stats.get().entities)}
                    </span>
                </div>
            </div>
        </aside>
    }
}

fn infer_author(filename: &str) -> String {
    let lower = filename.to_lowercase();
    if lower.contains("symposium") || lower.contains("plato") {
        "Plato".to_string()
    } else if lower.contains("aristotle") {
        "Aristotle".to_string()
    } else {
        "—".to_string()
    }
}

// ─── Stage head ─────────────────────────────────────────────────────────────

#[component]
fn StageHead(
    documents: ReadSignal<Vec<Document>>,
    build_status: ReadSignal<BuildStatus>,
) -> impl IntoView {
    view! {
        <header class="stage-head">
            <div class="stage-head-l">
                <span class="rail-h-eyebrow">"Source"</span>
                <h2 class="stage-title">
                    {move || documents.get().first().map(|d| d.name.clone()).unwrap_or_else(|| "no source".to_string())}
                    <span class="stage-title-sub">
                        {move || {
                            let docs = documents.get();
                            if docs.is_empty() {
                                "—".to_string()
                            } else {
                                format!("{} doc{}", docs.len(), if docs.len() == 1 { "" } else { "s" })
                            }
                        }}
                    </span>
                </h2>
            </div>
            <div class="stage-head-r">
                <span class="chip chip-mono">
                    <span class="dot dot-on"></span> "client-side"
                </span>
                <span class="chip chip-mono">"IndexedDB"</span>
                {move || match build_status.get() {
                    BuildStatus::Ready => view! {
                        <span class="chip chip-mono chip-on">"graph ready"</span>
                    }.into_any(),
                    BuildStatus::Building(_) => view! {
                        <span class="chip chip-mono chip-warn">"building"</span>
                    }.into_any(),
                    BuildStatus::Error(_) => view! {
                        <span class="chip chip-mono">"error"</span>
                    }.into_any(),
                    BuildStatus::Idle => view! {
                        <span class="chip chip-mono">"idle"</span>
                    }.into_any(),
                }}
            </div>
        </header>
    }
}

// ─── Turn view ──────────────────────────────────────────────────────────────

#[component]
fn TurnView(
    turn: ChatTurn,
    turn_idx: usize,
    active_ref: ReadSignal<Option<u32>>,
    set_active_ref: WriteSignal<Option<u32>>,
) -> impl IntoView {
    let segments = turn.answer.clone();
    let eyebrow = turn.eyebrow.clone();
    let timestamp = turn.timestamp.clone();
    let user_q = turn.user_q.clone();
    let ref_count = turn.refs.len();

    view! {
        <article class="turn" data-turn=turn_idx.to_string()>
            <div class="role">
                <span class="role-mark">"You"</span>
                "You · "{timestamp}
            </div>
            <div class="bubble-q"><p>{user_q}</p></div>

            <div class="role">
                <span class="role-mark is-ai">"AI"</span>
                {format!("Synthesis · {} · grounded in {} sources", LLM_LABEL, ref_count)}
            </div>
            <div class="answer">
                <span class="answer-eyebrow">{eyebrow}</span>
                <p>
                    {segments.into_iter().map(|seg| match seg {
                        AnswerSegment::Text(t) => view! { <span>{t}</span> }.into_any(),
                        AnswerSegment::Cite(n) => {
                            let class_fn = move || {
                                if active_ref.get() == Some(n) { "cite is-active" } else { "cite" }
                            };
                            view! {
                                <button
                                    class=class_fn
                                    data-ref=n.to_string()
                                    on:mouseenter=move |_| set_active_ref.set(Some(n))
                                    on:mouseleave=move |_| set_active_ref.set(None)
                                    on:click=move |_| set_active_ref.set(Some(n))
                                >
                                    {n.to_string()}
                                </button>
                            }.into_any()
                        }
                    }).collect_view()}
                </p>
                <div class="answer-foot">
                    <button class="ghost-btn">"copy"</button>
                    <button class="ghost-btn">"view subgraph"</button>
                </div>
            </div>
        </article>
    }
}

// ─── Composer ───────────────────────────────────────────────────────────────

#[component]
fn Composer<FS, FR>(
    query_input: ReadSignal<String>,
    set_query_input: WriteSignal<String>,
    on_submit: FS,
    is_ready: FR,
    loading: ReadSignal<bool>,
) -> impl IntoView
where
    FS: Fn(web_sys::SubmitEvent) + Copy + 'static + Send + Sync,
    FR: Fn() -> bool + Copy + 'static + Send + Sync,
{
    view! {
        <div class="composer-wrap">
            <form
                class="composer"
                on:submit=move |ev: web_sys::SubmitEvent| {
                    ev.prevent_default();
                    on_submit(ev);
                }
            >
                <span class="prompt" aria-hidden="true">"›"</span>
                <input
                    type="text"
                    placeholder="Ask another question about the indexed sources…"
                    on:input=move |ev| set_query_input.set(event_target_value(&ev))
                    prop:value=move || query_input.get()
                    prop:disabled=move || !is_ready() || loading.get()
                />
                <button
                    class="go"
                    type="submit"
                    prop:disabled=move || !is_ready() || loading.get()
                >
                    {move || if loading.get() { "thinking" } else { "retrieve" }}
                    <span class="kbd">"⏎"</span>
                </button>
            </form>
            <div class="composer-hint">
                <span>"retrieval: vector · k=5"</span>
                <span>"⌘K for commands"</span>
            </div>
        </div>
    }
}

// ─── Right rail ─────────────────────────────────────────────────────────────

#[component]
fn RightRail(
    subgraph: ReadSignal<SubgraphData>,
    build_status: ReadSignal<BuildStatus>,
    graph_stats: ReadSignal<GraphStats>,
    turns: ReadSignal<Vec<ChatTurn>>,
    active_ref: ReadSignal<Option<u32>>,
    set_active_ref: WriteSignal<Option<u32>>,
) -> impl IntoView {
    view! {
        <aside class="rail rail-right" data-screen-label="Graph, pipeline & references">
            <div class="rail-section">
                <div class="rail-h">
                    <span class="rail-h-eyebrow">"Graph · subgraph for this thread"</span>
                    <span class="chip chip-mono">
                        {move || {
                            let s = subgraph.get();
                            format!("{} · {}", s.nodes.len(), s.edges.len())
                        }}
                    </span>
                </div>
                <GraphPanel subgraph=subgraph />
            </div>

            <div class="rail-section">
                <div class="rail-h">
                    <span class="rail-h-eyebrow">"Pipeline"</span>
                    {move || match build_status.get() {
                        BuildStatus::Ready => view! {
                            <span class="chip chip-mono chip-on">"ready"</span>
                        }.into_any(),
                        BuildStatus::Building(_) => view! {
                            <span class="chip chip-mono chip-warn">"running"</span>
                        }.into_any(),
                        BuildStatus::Error(ref e) => {
                            let msg = e.clone();
                            view! { <span class="chip chip-mono" title=msg>"error"</span> }.into_any()
                        },
                        BuildStatus::Idle => view! {
                            <span class="chip chip-mono">"idle"</span>
                        }.into_any(),
                    }}
                </div>
                <PipelineList build_status=build_status />
                <div class="ministat-row">
                    <div class="ministat">
                        <div class="ministat-v">{move || graph_stats.get().entities.to_string()}</div>
                        <div class="ministat-k">"entities"</div>
                    </div>
                    <div class="ministat">
                        <div class="ministat-v">{move || graph_stats.get().relationships.to_string()}</div>
                        <div class="ministat-k">"edges"</div>
                    </div>
                    <div class="ministat">
                        <div class="ministat-v">{move || graph_stats.get().chunks.to_string()}</div>
                        <div class="ministat-k">"chunks"</div>
                    </div>
                </div>
            </div>

            <div class="rail-section">
                <div class="rail-h">
                    <span class="rail-h-eyebrow">"References · this thread"</span>
                    <span class="chip chip-mono">
                        {move || turns.get().last().map(|t| t.refs.len()).unwrap_or(0).to_string()}
                    </span>
                </div>
                <RefList
                    turns=turns
                    active_ref=active_ref
                    set_active_ref=set_active_ref
                />
            </div>
        </aside>
    }
}

// ─── Graph panel ────────────────────────────────────────────────────────────

#[component]
fn GraphPanel(subgraph: ReadSignal<SubgraphData>) -> impl IntoView {
    view! {
        <div class="graph-panel">
            <div class="graph-frame">
                {move || {
                    let sg = subgraph.get();
                    if sg.nodes.is_empty() {
                        view! {
                            <div class="graph-empty">"Subgraph appears after the first query."</div>
                        }.into_any()
                    } else {
                        view! {
                            <svg viewBox="0 0 320 240" preserveAspectRatio="xMidYMid meet" aria-label="Knowledge subgraph">
                                <defs>
                                    <pattern id="paper-grid" width="24" height="24" patternUnits="userSpaceOnUse">
                                        <path d="M 24 0 L 0 0 0 24" fill="none" stroke="var(--borderSoft)" stroke-width="0.5" opacity="0.5"/>
                                    </pattern>
                                </defs>
                                <rect x="0" y="0" width="320" height="240" fill="url(#paper-grid)"/>
                                <g class="g-edges">
                                    {sg.edges.iter().map(|(x1, y1, x2, y2, primary)| {
                                        let stroke = if *primary { "var(--accent)" } else { "var(--border)" };
                                        let width = if *primary { 1.4 } else { 0.7 };
                                        let opacity = if *primary { 0.9 } else { 0.22 };
                                        view! {
                                            <line
                                                x1=format!("{:.2}", x1)
                                                y1=format!("{:.2}", y1)
                                                x2=format!("{:.2}", x2)
                                                y2=format!("{:.2}", y2)
                                                stroke=stroke
                                                stroke-width=width.to_string()
                                                opacity=opacity.to_string()
                                            />
                                        }
                                    }).collect_view()}
                                </g>
                                <g class="g-nodes">
                                    {sg.nodes.iter().map(render_node).collect_view()}
                                </g>
                            </svg>
                        }.into_any()
                    }
                }}
            </div>
            <div class="graph-legend">
                <span><i class="lg-concept"></i>"concept"</span>
                <span><i class="lg-person"></i>"person"</span>
                <span><i class="lg-location"></i>"location"</span>
                <span><i class="lg-deity"></i>"deity"</span>
                <span><i class="lg-event"></i>"event"</span>
            </div>
        </div>
    }
}

fn render_node(n: &SubgraphNode) -> impl IntoView {
    let opacity = if n.primary { 1.0 } else { 0.45 };
    let inner_r = (n.radius - 4.0).max(3.5);
    let ring_color = NodeKind::ring_color(&n.kind);
    let fill = NodeKind::fill_color(&n.kind);
    let label = n.label.chars().take(14).collect::<String>();
    let label_y = n.y - n.radius - 2.0;
    view! {
        <g opacity=opacity.to_string()>
            {n.primary.then(|| view! {
                <circle
                    cx=format!("{:.2}", n.x)
                    cy=format!("{:.2}", n.y)
                    r=format!("{:.2}", n.radius)
                    fill=ring_color
                />
            })}
            <circle
                cx=format!("{:.2}", n.x)
                cy=format!("{:.2}", n.y)
                r=format!("{:.2}", inner_r)
                fill=fill
                stroke="var(--surface)"
                stroke-width="1.5"
            />
            <text
                x=format!("{:.2}", n.x)
                y=format!("{:.2}", label_y)
                text-anchor="middle"
                style="font:600 10px/1 'Geist', system-ui, sans-serif; fill: var(--ink);"
            >
                {label}
            </text>
        </g>
    }
}

// ─── Pipeline list ──────────────────────────────────────────────────────────

#[component]
fn PipelineList(build_status: ReadSignal<BuildStatus>) -> impl IntoView {
    view! {
        <div class="pipeline">
        <ul class="stages">
            {move || {
                let st = build_status.get();
                let stages = pipeline_progress(&st);
                stages.into_iter().enumerate().map(|(idx, s)| {
                    let class = if s.done { "pls is-done" }
                        else if s.active { "pls is-active" }
                        else { "pls" };
                    view! {
                        <li class=class>
                            <span class="pls-no">{format!("{:02}", idx + 1)}</span>
                            <div class="pls-body">
                                <div class="pls-row">
                                    <span class="pls-label">{s.label.to_string()}</span>
                                    <span class="pls-pct">{format!("{}%", s.pct as u32)}</span>
                                </div>
                                <div class="pls-bar">
                                    <div class="pls-fill" style=format!("width:{}%", s.pct as u32)></div>
                                </div>
                                <div class="pls-hint">{s.hint.to_string()}</div>
                            </div>
                        </li>
                    }
                }).collect_view()
            }}
        </ul>
        </div>
    }
}

struct PipelineRow {
    label: &'static str,
    hint: &'static str,
    pct: f32,
    done: bool,
    active: bool,
}

fn pipeline_progress(status: &BuildStatus) -> Vec<PipelineRow> {
    let (chunk, extract, embed, index) = match status {
        BuildStatus::Idle => (0.0, 0.0, 0.0, 0.0),
        BuildStatus::Ready => (100.0, 100.0, 100.0, 100.0),
        BuildStatus::Error(_) => (0.0, 0.0, 0.0, 0.0),
        BuildStatus::Building(BuildStage::Chunking { progress }) => (*progress, 0.0, 0.0, 0.0),
        BuildStatus::Building(BuildStage::Extracting { progress }) => (100.0, *progress, 0.0, 0.0),
        BuildStatus::Building(BuildStage::Embedding { progress }) => {
            (100.0, 100.0, *progress, 0.0)
        },
        BuildStatus::Building(BuildStage::Indexing { progress }) => {
            (100.0, 100.0, 100.0, *progress)
        },
    };
    let mark = |pct: f32| (pct >= 100.0, pct > 0.0 && pct < 100.0);
    let (c_d, c_a) = mark(chunk);
    let (x_d, x_a) = mark(extract);
    let (e_d, e_a) = mark(embed);
    let (i_d, i_a) = mark(index);
    vec![
        PipelineRow {
            label: "Chunk",
            hint: "split into ~512-tok chunks",
            pct: chunk,
            done: c_d,
            active: c_a,
        },
        PipelineRow {
            label: "Extract entities",
            hint: "WebLLM (Qwen) NER+RE",
            pct: extract,
            done: x_d,
            active: x_a,
        },
        PipelineRow {
            label: "Embed",
            hint: "all-MiniLM-L6 · 384d",
            pct: embed,
            done: e_d,
            active: e_a,
        },
        PipelineRow {
            label: "Link & rank",
            hint: "vector index · cosine",
            pct: index,
            done: i_d,
            active: i_a,
        },
    ]
}

// ─── Ref list ───────────────────────────────────────────────────────────────

#[component]
fn RefList(
    turns: ReadSignal<Vec<ChatTurn>>,
    active_ref: ReadSignal<Option<u32>>,
    set_active_ref: WriteSignal<Option<u32>>,
) -> impl IntoView {
    view! {
        <div class="ref-block" id="refList">
            {move || {
                let last_refs: Vec<RefCard> = turns.get()
                    .last()
                    .map(|t| t.refs.clone())
                    .unwrap_or_default();
                if last_refs.is_empty() {
                    view! {
                        <div class="doc-empty">"References will populate after your first query."</div>
                    }.into_any()
                } else {
                    view! {
                        <>
                            {last_refs.into_iter().map(|r| {
                                let n = r.num;
                                let class_fn = move || {
                                    if active_ref.get() == Some(n) {
                                        "ref-card is-hovered"
                                    } else {
                                        "ref-card"
                                    }
                                };
                                view! {
                                    <div
                                        class=class_fn
                                        data-num=r.num.to_string()
                                        on:mouseenter=move |_| set_active_ref.set(Some(n))
                                        on:mouseleave=move |_| set_active_ref.set(None)
                                    >
                                        <div class="ref-head">
                                            <span class="ref-num">{r.num.to_string()}</span>
                                            <span class="ref-doc">{r.doc.clone()}</span>
                                            <span class="ref-loc">{r.loc.clone()}</span>
                                        </div>
                                        <div class="ref-heading">{r.heading.clone()}</div>
                                        <p class="ref-text">{r.text.clone()}</p>
                                    </div>
                                }
                            }).collect_view()}
                        </>
                    }.into_any()
                }
            }}
        </div>
    }
}

// ─── Utilities ──────────────────────────────────────────────────────────────

fn format_now() -> String {
    let date = js_sys::Date::new_0();
    let h = date.get_hours();
    let m = date.get_minutes();
    let am = h < 12;
    let h12 = if h == 0 { 12 } else if h > 12 { h - 12 } else { h };
    format!("{}:{:02} {}", h12, m, if am { "am" } else { "pm" })
}

fn main() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
    web_sys::console::log_1(&"🚀 GraphRAG WASM — chat shell".into());
    leptos::mount::mount_to_body(App);
}
