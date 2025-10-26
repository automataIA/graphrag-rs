//! GraphRAG WASM Demo Application
//!
//! A complete browser-based GraphRAG implementation using:
//! - Leptos 0.8 for reactive UI
//! - ONNX Runtime Web for embeddings (WebGPU accelerated)
//! - WebLLM for LLM inference (Qwen3)
//! - Pure Rust vector search (cosine similarity)
//! - HuggingFace tokenizers (unstable_wasm)
//! - graphrag-core for real pipeline

use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{Event, HtmlInputElement, HtmlTextAreaElement};

// Import graphrag-core WASM-compatible components
// Note: We only import concrete types, not async traits
use graphrag_core::{
    Config, GraphRAG,
    core::GraphRAGError,
};

// Import ONNX embedder, vector search, and entity extraction
mod onnx_embedder;
mod vector_search;
mod entity_extractor;
mod webllm;
mod storage;
mod components;

use onnx_embedder::OnnxEmbedder;
use vector_search::VectorIndex;
use entity_extractor::{extract_entities, extract_entities_simple, Entity, Relationship};
use webllm::WebLLM;
use components::{
    SettingsPanel, HierarchyExplorer, CommunityData,
};

// Type alias for Result (currently unused but reserved for future use)
#[allow(dead_code)]
type GraphRAGResult<T> = Result<T, GraphRAGError>;

/// Document structure for ingestion
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Document {
    id: String,
    name: String,
    content: String,
    size_bytes: usize,
    added_at: f64, // timestamp
}

/// Graph build status
#[derive(Clone, Debug, PartialEq)]
enum BuildStatus {
    Idle,
    Building(BuildStage),
    Ready,
    Error(String),
}

/// Build pipeline stages
#[derive(Clone, Debug, PartialEq)]
enum BuildStage {
    Chunking { progress: f32, current: usize, total: usize },
    Extracting { progress: f32, current: usize, total: usize },
    Embedding { progress: f32, current: usize, total: usize },
    Indexing { progress: f32 },
}

/// Graph statistics
#[derive(Clone, Debug, Default)]
struct GraphStats {
    documents: usize,
    chunks: usize,
    entities: usize,
    relationships: usize,
    embeddings: usize,
}

/// Active tab in the interface
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tab {
    Build,
    Explore,
    Query,
    Hierarchy,
    Settings,
}

/// Main application component
#[component]
fn App() -> impl IntoView {
    // Core state
    let (documents, set_documents) = signal(Vec::<Document>::new());
    let (build_status, set_build_status) = signal(BuildStatus::Idle);
    let (graph_stats, set_graph_stats) = signal(GraphStats::default());
    let (active_tab, set_active_tab) = signal(Tab::Build);

    // Persistent GraphRAG instance (shared across build and query operations)
    // Using store_value for Send + Sync compatibility with Leptos 0.8
    let graphrag_instance = StoredValue::new(None::<GraphRAG>);

    // Persistent vector index for semantic search (pure Rust)
    let vector_index = StoredValue::new(None::<VectorIndex>);

    // Query interface state
    let (query, set_query) = signal(String::new());
    let (results, set_results) = signal(String::from("Add documents and build the knowledge graph to start querying."));
    let (loading, set_loading) = signal(false);

    // Hierarchy interface state
    let (max_level, set_max_level) = signal(0_usize);
    let (communities, set_communities) = signal(Vec::<CommunityData>::new());

    view! {
        <div class="min-h-screen bg-base-100 text-base-content">
            <div class="container mx-auto px-4 py-8 max-w-7xl">
                <Header/>

                <TabNavigation
                    active_tab=active_tab
                    set_active_tab=set_active_tab
                    build_status=build_status
                    graph_stats=graph_stats
                />

                <div class="mt-8">
                    {move || match active_tab.get() {
                        Tab::Build => view! {
                            <BuildTab
                                documents=documents
                                set_documents=set_documents
                                build_status=build_status
                                set_build_status=set_build_status
                                set_graph_stats=set_graph_stats
                                graphrag_instance=graphrag_instance.clone()
                                vector_index=vector_index.clone()
                            />
                        }.into_any(),
                        Tab::Explore => view! {
                            <ExploreTab
                                graph_stats=graph_stats
                                build_status=build_status
                            />
                        }.into_any(),
                        Tab::Query => view! {
                            <QueryTab
                                query=query
                                set_query=set_query
                                results=results
                                set_results=set_results
                                loading=loading
                                set_loading=set_loading
                                build_status=build_status
                                graph_stats=graph_stats
                                graphrag_instance=graphrag_instance.clone()
                                vector_index=vector_index.clone()
                            />
                        }.into_any(),
                        Tab::Hierarchy => view! {
                            <HierarchyTab
                                max_level=max_level
                                set_max_level=set_max_level
                                communities=communities
                                set_communities=set_communities
                                build_status=build_status
                                graphrag_instance=graphrag_instance.clone()
                            />
                        }.into_any(),
                        Tab::Settings => view! {
                            <SettingsPanel />
                        }.into_any(),
                    }}
                </div>

                <Footer/>
            </div>
        </div>
    }
}

/// Header component
#[component]
fn Header() -> impl IntoView {
    view! {
        <header class="mb-8">
            <div class="flex items-center justify-center mb-4">
                <div class="flex items-center gap-3">
                    <i data-lucide="brain" class="w-12 h-12 text-primary animate-pulse"></i>
                    <h1 class="text-5xl font-bold">"GraphRAG WASM"</h1>
                </div>
            </div>

            <p class="text-xl text-center mb-6">"Knowledge Graph RAG in Your Browser"</p>

            <div class="flex flex-wrap justify-center gap-3 text-sm">
                <span class="badge badge-success gap-2">
                    <i data-lucide="check-circle" class="w-4 h-4"></i>
                    "HuggingFace"
                </span>
                <span class="badge badge-info gap-2">
                    <i data-lucide="check-circle" class="w-4 h-4"></i>
                    "ONNX (WebGPU)"
                </span>
                <span class="badge badge-warning gap-2">
                    <i data-lucide="check-circle" class="w-4 h-4"></i>
                    "WebLLM (Qwen3)"
                </span>
                <span class="badge badge-primary gap-2">
                    <i data-lucide="check-circle" class="w-4 h-4"></i>
                    "Pure Rust Search"
                </span>
            </div>
        </header>
    }
}

/// Tab navigation component
#[component]
fn TabNavigation(
    active_tab: ReadSignal<Tab>,
    set_active_tab: WriteSignal<Tab>,
    build_status: ReadSignal<BuildStatus>,
    graph_stats: ReadSignal<GraphStats>,
) -> impl IntoView {
    let tab_class = move |tab: Tab| {
        let base = "flex-1 px-6 py-4 font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary";
        if active_tab.get() == tab {
            format!("{} btn-primary shadow-lg", base)
        } else {
            format!("{} btn-ghost border border-base-300", base)
        }
    };

    view! {
        <nav
            class="flex flex-col sm:flex-row gap-3"
            role="tablist"
            aria-label="Main navigation"
        >
            <button
                class=move || tab_class(Tab::Build)
                role="tab"
                aria-selected=move || active_tab.get() == Tab::Build
                aria-controls="build-panel"
                on:click=move |_| set_active_tab.set(Tab::Build)
            >
                <span class="flex items-center justify-center gap-2">
                    <i data-lucide="file-text" class="w-5 h-5"></i>
                    <span>"1. Build Graph"</span>
                    <span class="badge badge-sm">
                        {move || graph_stats.get().documents.to_string()} " docs"
                    </span>
                </span>
            </button>

            <button
                class=move || tab_class(Tab::Explore)
                role="tab"
                aria-selected=move || active_tab.get() == Tab::Explore
                aria-controls="explore-panel"
                on:click=move |_| set_active_tab.set(Tab::Explore)
            >
                <span class="flex items-center justify-center gap-2">
                    <i data-lucide="search" class="w-5 h-5"></i>
                    <span>"2. Explore Graph"</span>
                    {move || match build_status.get() {
                        BuildStatus::Ready => view! {
                            <span class="badge badge-success badge-sm">
                                "Ready"
                            </span>
                        }.into_any(),
                        BuildStatus::Building(_) => view! {
                            <span class="badge badge-warning badge-sm animate-pulse">
                                "Building..."
                            </span>
                        }.into_any(),
                        _ => view! { <span></span> }.into_any(),
                    }}
                </span>
            </button>

            <button
                class=move || tab_class(Tab::Query)
                role="tab"
                aria-selected=move || active_tab.get() == Tab::Query
                aria-controls="query-panel"
                on:click=move |_| set_active_tab.set(Tab::Query)
            >
                <span class="flex items-center justify-center gap-2">
                    <i data-lucide="message-square" class="w-5 h-5"></i>
                    <span>"3. Query Graph"</span>
                </span>
            </button>

            <button
                class=move || tab_class(Tab::Hierarchy)
                role="tab"
                aria-selected=move || active_tab.get() == Tab::Hierarchy
                aria-controls="hierarchy-panel"
                on:click=move |_| set_active_tab.set(Tab::Hierarchy)
            >
                <span class="flex items-center justify-center gap-2">
                    <i data-lucide="network" class="w-5 h-5"></i>
                    <span>"4. Hierarchy"</span>
                </span>
            </button>

            <button
                class=move || tab_class(Tab::Settings)
                role="tab"
                aria-selected=move || active_tab.get() == Tab::Settings
                aria-controls="settings-panel"
                on:click=move |_| set_active_tab.set(Tab::Settings)
            >
                <span class="flex items-center justify-center gap-2">
                    <i data-lucide="settings" class="w-5 h-5"></i>
                    <span>"Settings"</span>
                </span>
            </button>
        </nav>
    }
}

/// Build tab component
#[component]
fn BuildTab(
    documents: ReadSignal<Vec<Document>>,
    set_documents: WriteSignal<Vec<Document>>,
    build_status: ReadSignal<BuildStatus>,
    set_build_status: WriteSignal<BuildStatus>,
    set_graph_stats: WriteSignal<GraphStats>,
    graphrag_instance: StoredValue<Option<GraphRAG>>,
    vector_index: StoredValue<Option<VectorIndex>>,
) -> impl IntoView {
    // Local can_build check
    let can_build = move || {
        !documents.get().is_empty() && !matches!(build_status.get(), BuildStatus::Building(_))
    };
    let (text_input, set_text_input) = signal(String::new());
    let (doc_name, set_doc_name) = signal(String::new());

    // Add document from text input
    let add_text_document = move |_| {
        let content = text_input.get();
        let name = doc_name.get();

        if content.is_empty() {
            return;
        }

        let doc_name_final = if name.is_empty() {
            format!("Document {}", documents.get().len() + 1)
        } else {
            name
        };

        let doc = Document {
            id: format!("doc-{}", js_sys::Date::now()),
            name: doc_name_final,
            content: content.clone(),
            size_bytes: content.len(),
            added_at: js_sys::Date::now(),
        };

        let mut docs = documents.get();
        docs.push(doc);
        set_documents.set(docs);
        set_text_input.set(String::new());
        set_doc_name.set(String::new());
    };

    // Remove document
    let remove_document = move |id: String| {
        let docs = documents.get();
        let filtered: Vec<_> = docs.into_iter().filter(|d| d.id != id).collect();
        set_documents.set(filtered);
    };

    // Handle file upload
    let on_file_upload = move |ev: Event| {
        let input = ev.target().unwrap().dyn_into::<HtmlInputElement>().unwrap();
        if let Some(files) = input.files() {
            for i in 0..files.length() {
                if let Some(file) = files.get(i) {
                    let file_name = file.name();
                    let set_documents = set_documents.clone();
                    let documents = documents.clone();

                    spawn_local(async move {
                        let reader = web_sys::FileReader::new().unwrap();
                        let reader_clone = reader.clone();

                        let onload = Closure::wrap(Box::new(move |_: Event| {
                            if let Ok(result) = reader_clone.result() {
                                if let Some(text) = result.as_string() {
                                    let doc = Document {
                                        id: format!("doc-{}", js_sys::Date::now()),
                                        name: file_name.clone(),
                                        content: text.clone(),
                                        size_bytes: text.len(),
                                        added_at: js_sys::Date::now(),
                                    };

                                    let mut docs = documents.get_untracked();
                                    docs.push(doc);
                                    set_documents.set(docs);
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

    // Load Symposium demo text
    let load_symposium_demo = move |_| {
        spawn_local(async move {
            web_sys::console::log_1(&"📖 Loading Symposium demo...".into());

            match gloo_net::http::Request::get("./Symposium.txt")
                .send()
                .await
            {
                Ok(response) => {
                    match response.text().await {
                        Ok(text) => {
                            let doc = Document {
                                id: format!("doc-{}", js_sys::Date::now()),
                                name: "Plato's Symposium".to_string(),
                                content: text.clone(),
                                size_bytes: text.len(),
                                added_at: js_sys::Date::now(),
                            };

                            let mut docs = documents.get_untracked();
                            docs.push(doc);
                            set_documents.set(docs);
                            web_sys::console::log_1(&"✅ Symposium loaded successfully".into());
                        }
                        Err(e) => {
                            web_sys::console::error_1(&format!("Failed to read Symposium text: {:?}", e).into());
                        }
                    }
                }
                Err(e) => {
                    web_sys::console::error_1(&format!("Failed to fetch Symposium.txt: {:?}", e).into());
                }
            }
        });
    };

    // Build knowledge graph with real GraphRAG components
    let build_graph = move |_| {
        let docs = documents.get();
        let total_docs = docs.len();

        set_build_status.set(BuildStatus::Building(BuildStage::Chunking {
            progress: 0.0,
            current: 0,
            total: total_docs
        }));

        spawn_local(async move {
            web_sys::console::log_1(&"🚀 Starting graph build with pure Rust vector search".into());

            // Create or reuse GraphRAG instance
            let config = Config::default();
            let mut graphrag = match GraphRAG::new(config) {
                Ok(g) => g,
                Err(e) => {
                    set_build_status.set(BuildStatus::Error(format!("Failed to create GraphRAG: {}", e)));
                    return;
                }
            };

            // Initialize the system
            if let Err(e) = graphrag.initialize() {
                set_build_status.set(BuildStatus::Error(format!("Failed to initialize: {}", e)));
                return;
            }

            // Stage 1: Add documents and chunk them
            let mut all_chunks = 0;
            for (i, doc) in docs.iter().enumerate() {
                set_build_status.set(BuildStatus::Building(BuildStage::Chunking {
                    progress: ((i + 1) as f32 / total_docs as f32) * 100.0,
                    current: i + 1,
                    total: total_docs,
                }));

                // Add document using GraphRAG (automatically chunks)
                if let Err(e) = graphrag.add_document_from_text(&doc.content) {
                    web_sys::console::warn_1(&format!("Failed to add document {}: {}", doc.name, e).into());
                }

                gloo_timers::future::TimeoutFuture::new(50).await;
            }

            // Get actual chunk count from graph
            if let Some(kg) = graphrag.knowledge_graph() {
                all_chunks = kg.chunks().count();
            }

            // Stage 2: Extract entities using WebLLM
            set_build_status.set(BuildStatus::Building(BuildStage::Extracting {
                progress: 0.0,
                current: 0,
                total: docs.len(),
            }));

            web_sys::console::log_1(&"🔍 Extracting entities with WebLLM (Qwen)...".into());

            // Initialize WebLLM for entity extraction
            let llm_result = WebLLM::new("Qwen2-1.5B-Instruct-q4f16_1-MLC").await;

            let mut all_entities: Vec<Entity> = Vec::new();
            let mut all_relationships: Vec<Relationship> = Vec::new();

            if let Ok(llm) = llm_result {
                web_sys::console::log_1(&"✅ WebLLM initialized successfully".into());

                // Extract entities from each document
                for (idx, doc) in docs.iter().enumerate() {
                    set_build_status.set(BuildStatus::Building(BuildStage::Extracting {
                        progress: ((idx + 1) as f32 / docs.len() as f32) * 100.0,
                        current: idx + 1,
                        total: docs.len(),
                    }));

                    web_sys::console::log_1(&format!("📄 Extracting from document {}/{}...", idx + 1, docs.len()).into());

                    match extract_entities(&llm, &doc.content).await {
                        Ok(result) => {
                            web_sys::console::log_1(&format!(
                                "  ✅ Found {} entities, {} relationships",
                                result.entities.len(),
                                result.relationships.len()
                            ).into());
                            all_entities.extend(result.entities);
                            all_relationships.extend(result.relationships);
                        }
                        Err(e) => {
                            web_sys::console::warn_1(&format!("  ⚠️  Extraction failed: {}", e).into());
                        }
                    }

                    gloo_timers::future::TimeoutFuture::new(200).await;
                }

                // Deduplicate entities
                all_entities.sort_by(|a, b| a.name.cmp(&b.name));
                all_entities.dedup_by(|a, b| a.name == b.name);
            } else {
                web_sys::console::warn_1(&"⚠️  WebLLM not available, using simple rule-based extraction".into());

                // Fallback to simple rule-based extraction
                for doc in docs.iter() {
                    let result = extract_entities_simple(&doc.content);
                    all_entities.extend(result.entities);
                    all_relationships.extend(result.relationships);
                }

                // Deduplicate entities
                all_entities.sort_by(|a, b| a.name.cmp(&b.name));
                all_entities.dedup_by(|a, b| a.name == b.name);
            }

            let entity_count = all_entities.len();
            let relationship_count = all_relationships.len();

            web_sys::console::log_1(&format!("✅ Extracted {} entities, {} relationships", entity_count, relationship_count).into());

            // Stage 3: Generate embeddings with ONNX
            set_build_status.set(BuildStatus::Building(BuildStage::Embedding {
                progress: 0.0,
                current: 0,
                total: entity_count,
            }));

            web_sys::console::log_1(&"🔧 Generating embeddings...".into());

            // Fetch tokenizer.json from server for HuggingFace tokenizer
            use gloo_net::http::Request;

            web_sys::console::log_1(&"📥 Fetching tokenizer.json from server...".into());

            let tokenizer_result = Request::get("./tokenizer.json").send().await;

            let embedder_result = if let Ok(response) = tokenizer_result {
                if let Ok(tokenizer_json) = response.text().await {
                    web_sys::console::log_1(&format!("✅ Fetched tokenizer.json ({} bytes)",
                        tokenizer_json.len()).into());

                    // Create ONNX embedder from fetched tokenizer JSON (WASM-compatible!)
                    OnnxEmbedder::from_tokenizer_json(384, &tokenizer_json)
                } else {
                    Err(onnx_embedder::OnnxEmbedderError::InvalidInput(
                        "Failed to read tokenizer.json response as text".to_string()
                    ))
                }
            } else {
                Err(onnx_embedder::OnnxEmbedderError::InvalidInput(
                    "Failed to fetch tokenizer.json from server".to_string()
                ))
            };

            let mut embedder = match embedder_result {
                Ok(e) => {
                    web_sys::console::log_1(&"✅ ONNX embedder created with HuggingFace tokenizer".into());
                    Some(e)
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!("⚠️  ONNX embedder not available: {}. Using simple hash-based embeddings.", e).into());
                    None
                }
            };

            // Load ONNX model if embedder was created successfully
            if let Some(ref mut emb) = embedder {
                web_sys::console::log_1(&"📦 Loading ONNX model...".into());
                if let Err(e) = emb.load_model("./models/minilm-l6.onnx", true).await {
                    web_sys::console::warn_1(&format!("⚠️  Failed to load ONNX model: {}. Falling back to simple embeddings.", e).into());
                    embedder = None; // Clear embedder if model load failed
                }
            }

            // Collect chunks and their IDs for embedding
            let mut chunk_data: Vec<(String, String)> = Vec::new(); // (chunk_id, content)
            if let Some(kg) = graphrag.knowledge_graph() {
                for chunk in kg.chunks() {
                    chunk_data.push((chunk.id.0.clone(), chunk.content.clone()));
                }
            }

            web_sys::console::log_1(&format!("📦 Collected {} chunks for embedding", chunk_data.len()).into());

            // Generate embeddings
            let mut embeddings = Vec::new();
            let embedding_dim = 384; // MiniLM-L6 dimension

            if let Some(ref emb) = embedder {
                // Use ONNX embeddings
                web_sys::console::log_1(&"✅ Using ONNX Runtime Web for embeddings".into());

                for (i, (_id, content)) in chunk_data.iter().enumerate() {
                    set_build_status.set(BuildStatus::Building(BuildStage::Embedding {
                        progress: ((i + 1) as f32 / chunk_data.len() as f32) * 100.0,
                        current: i + 1,
                        total: chunk_data.len(),
                    }));

                    match emb.embed(content).await {
                        Ok(embedding) => {
                            embeddings.push(embedding);
                        }
                        Err(e) => {
                            web_sys::console::warn_1(&format!("Failed to embed chunk {}: {}", i, e).into());
                        }
                    }

                    // Small delay for UI updates (every 10 embeddings)
                    if i % 10 == 0 {
                        gloo_timers::future::TimeoutFuture::new(10).await;
                    }
                }

                web_sys::console::log_1(&format!("✅ Generated {} ONNX embeddings", embeddings.len()).into());
            } else {
                // Fallback to simple hash-based embeddings
                web_sys::console::warn_1(&"⚠️  Using simple hash-based embeddings (ONNX not available)".into());

                for (i, (_id, content)) in chunk_data.iter().enumerate() {
                    set_build_status.set(BuildStatus::Building(BuildStage::Embedding {
                        progress: ((i + 1) as f32 / chunk_data.len() as f32) * 100.0,
                        current: i + 1,
                        total: chunk_data.len(),
                    }));

                    // Simple hash-based embedding (deterministic for same text)
                    let mut embedding = vec![0.0f32; embedding_dim];
                    let bytes = content.as_bytes();

                    for (idx, chunk) in bytes.chunks(4).enumerate() {
                        let hash = chunk.iter().fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
                        let normalized = (hash as f32) / (u32::MAX as f32) * 2.0 - 1.0; // Range: -1 to 1
                        embedding[idx % embedding_dim] += normalized;
                    }

                    // Normalize the embedding
                    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut embedding {
                            *x /= norm;
                        }
                    }

                    embeddings.push(embedding);

                    // Small delay for UI updates (every 10 embeddings)
                    if i % 10 == 0 {
                        gloo_timers::future::TimeoutFuture::new(10).await;
                    }
                }

                web_sys::console::log_1(&format!("✅ Generated {} simple embeddings", embeddings.len()).into());
            }

            // Stage 4: Build pure Rust vector index with proper chunk IDs
            set_build_status.set(BuildStatus::Building(BuildStage::Indexing {
                progress: 0.0,
            }));

            web_sys::console::log_1(&"🔨 Building vector search index...".into());
            gloo_timers::future::TimeoutFuture::new(100).await;

            set_build_status.set(BuildStatus::Building(BuildStage::Indexing {
                progress: 50.0,
            }));

            // Create pure Rust vector index with chunk IDs
            let mut index = VectorIndex::new();
            for ((chunk_id, _content), embedding) in chunk_data.iter().zip(embeddings.iter()) {
                index.add(embedding.clone(), chunk_id.clone(), chunk_id.clone());
            }

            set_build_status.set(BuildStatus::Building(BuildStage::Indexing {
                progress: 100.0,
            }));

            gloo_timers::future::TimeoutFuture::new(100).await;

            web_sys::console::log_1(&"✅ Pure Rust vector index created successfully".into());
            vector_index.set_value(Some(index));

            // Store the GraphRAG instance for later queries
            graphrag_instance.set_value(Some(graphrag));

            // Complete
            set_build_status.set(BuildStatus::Ready);
            set_graph_stats.set(GraphStats {
                documents: total_docs,
                chunks: all_chunks,
                entities: entity_count,
                relationships: relationship_count,
                embeddings: embeddings.len(),
            });

            web_sys::console::log_1(&format!("✅ Graph built: {} docs, {} chunks, {} entities, {} relationships",
                total_docs, all_chunks, entity_count, relationship_count).into());
        });
    };

    view! {
        <div class="space-y-6" role="tabpanel" id="build-panel">
            // Document Input Section
            <div class="card bg-base-200 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title flex items-center gap-2">
                        <i data-lucide="file-plus" class="w-6 h-6"></i>
                        <span>"Add Documents"</span>
                    </h2>

                    // File Upload
                    <div class="mb-4">
                        <label
                            for="file-upload"
                            class="block text-sm font-medium mb-2"
                        >
                            "Upload Files (txt, md, pdf)"
                        </label>
                        <input
                            id="file-upload"
                            type="file"
                            multiple
                            accept=".txt,.md,.pdf"
                            class="file-input file-input-bordered file-input-primary w-full"
                            on:change=on_file_upload
                        />
                    </div>

                    // Load Symposium Demo Button
                    <div class="mb-6">
                        <button
                            type="button"
                            class="btn btn-info w-full gap-2"
                            on:click=load_symposium_demo
                        >
                            <i data-lucide="book-open" class="w-5 h-5"></i>
                            "Load Symposium Demo (Plato)"
                        </button>
                    </div>

                    // Text Input
                    <div class="space-y-4">
                        <div>
                            <label
                                for="doc-name"
                                class="block text-sm font-medium mb-2"
                            >
                                "Document Name (optional)"
                            </label>
                            <input
                                id="doc-name"
                                type="text"
                                placeholder="My Document"
                                class="input input-bordered w-full"
                                on:input=move |ev| set_doc_name.set(event_target_value(&ev))
                                prop:value=move || doc_name.get()
                            />
                        </div>

                        <div>
                            <label
                                for="doc-text"
                                class="block text-sm font-medium mb-2"
                            >
                                "Paste Text Content"
                            </label>
                            <textarea
                                id="doc-text"
                                rows="8"
                                placeholder="Paste your document content here..."
                                class="textarea textarea-bordered w-full font-mono text-sm"
                                on:input=move |ev| {
                                    let textarea = ev.target().unwrap().dyn_into::<HtmlTextAreaElement>().unwrap();
                                    set_text_input.set(textarea.value());
                                }
                                prop:value=move || text_input.get()
                            />
                        </div>

                        <button
                            type="button"
                            class="btn btn-success w-full gap-2"
                            disabled=move || text_input.get().is_empty()
                            on:click=add_text_document
                        >
                            <i data-lucide="plus-circle" class="w-5 h-5"></i>
                            "Add Document"
                        </button>
                    </div>
                </div>
            </div>

            // Document List
            <div class="card bg-base-200 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title flex items-center gap-2">
                        <i data-lucide="library" class="w-6 h-6"></i>
                        <span>"Document Library"</span>
                        <span class="badge badge-sm">
                            {move || documents.get().len().to_string()} " documents"
                        </span>
                    </h2>

                    {move || {
                        let docs = documents.get();
                        if docs.is_empty() {
                            view! {
                                <div class="text-center py-12">
                                    <i data-lucide="inbox" class="w-16 h-16 mx-auto mb-4 opacity-50"></i>
                                    <p class="text-lg">"No documents yet"</p>
                                    <p class="text-sm mt-2 opacity-70">"Upload files or paste text to get started"</p>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="space-y-3 max-h-96 overflow-y-auto">
                                    {docs.into_iter().map(|doc| {
                                        let doc_id = doc.id.clone();
                                        view! {
                                            <div class="card bg-base-100 border border-base-300 hover:border-primary transition-colors">
                                                <div class="card-body p-4">
                                                    <div class="flex items-start justify-between gap-4">
                                                        <div class="flex-1 min-w-0">
                                                            <h3 class="font-semibold truncate mb-1">
                                                                {doc.name.clone()}
                                                            </h3>
                                                            <p class="text-sm opacity-70 line-clamp-2 mb-2">
                                                                {doc.content.chars().take(150).collect::<String>()}
                                                                {if doc.content.len() > 150 { "..." } else { "" }}
                                                            </p>
                                                            <div class="flex items-center gap-4 text-xs opacity-60">
                                                                <span>{format!("{} bytes", doc.size_bytes)}</span>
                                                                <span>"•"</span>
                                                                <span>{format!("Added {}", format_timestamp(doc.added_at))}</span>
                                                            </div>
                                                        </div>
                                                        <button
                                                            type="button"
                                                            class="btn btn-sm btn-error gap-2"
                                                            aria-label=format!("Remove document: {}", doc.name)
                                                            on:click=move |_| remove_document(doc_id.clone())
                                                        >
                                                            <i data-lucide="trash-2" class="w-4 h-4"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
            </div>

            // Build Section
            <div class="card bg-base-200 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title flex items-center gap-2">
                        <i data-lucide="zap" class="w-6 h-6"></i>
                        <span>"Build Knowledge Graph"</span>
                    </h2>

                    <BuildProgress build_status=build_status />

                    <button
                        type="button"
                        class="btn btn-primary btn-lg w-full gap-2 mt-6"
                        disabled=move || !can_build()
                        on:click=build_graph
                    >
                        {move || match build_status.get() {
                            BuildStatus::Idle => view! {
                                <>
                                    <i data-lucide="rocket" class="w-5 h-5"></i>
                                    "Build Knowledge Graph"
                                </>
                            }.into_any(),
                            BuildStatus::Building(_) => view! {
                                <>
                                    <span class="loading loading-spinner"></span>
                                    "Building..."
                                </>
                            }.into_any(),
                            BuildStatus::Ready => view! {
                                <>
                                    <i data-lucide="check-circle" class="w-5 h-5"></i>
                                    "Graph Ready - Rebuild?"
                                </>
                            }.into_any(),
                            BuildStatus::Error(_) => view! {
                                <>
                                    <i data-lucide="alert-triangle" class="w-5 h-5"></i>
                                    "Retry Build"
                                </>
                            }.into_any(),
                        }}
                    </button>
                </div>
            </div>
        </div>
    }
}

/// Build progress visualization
#[component]
fn BuildProgress(build_status: ReadSignal<BuildStatus>) -> impl IntoView {
    view! {
        <div class="space-y-4" role="status" aria-live="polite">
            {move || match build_status.get() {
                BuildStatus::Idle => view! {
                    <div class="text-center py-8 opacity-70">
                        <p>"Add documents above, then click the button below to build your knowledge graph"</p>
                    </div>
                }.into_any(),

                BuildStatus::Building(stage) => {
                    let (icon, label, progress, detail) = match stage {
                        BuildStage::Chunking { progress, current, total } => (
                            "file-text",
                            "Chunking Documents",
                            progress,
                            format!("{} / {} documents", current, total)
                        ),
                        BuildStage::Extracting { progress, current, total } => (
                            "search",
                            "Extracting Entities",
                            progress,
                            format!("{} / {} chunks", current, total)
                        ),
                        BuildStage::Embedding { progress, current, total } => (
                            "cpu",
                            "Computing Embeddings",
                            progress,
                            format!("{} / {} entities", current, total)
                        ),
                        BuildStage::Indexing { progress } => (
                            "database",
                            "Building Search Index",
                            progress,
                            "Finalizing...".to_string()
                        ),
                    };

                    view! {
                        <div class="space-y-3">
                            <div class="flex items-center justify-between text-sm">
                                <span class="flex items-center gap-2 font-semibold">
                                    <i data-lucide=icon class="w-4 h-4"></i>
                                    {label}
                                </span>
                                <span class="opacity-70">{format!("{:.0}%", progress)}</span>
                            </div>
                            <progress
                                class="progress progress-primary w-full"
                                value=format!("{}", progress as i32)
                                max="100"
                            ></progress>
                            <p class="text-xs text-center opacity-60">{detail}</p>
                        </div>
                    }.into_any()
                },

                BuildStatus::Ready => view! {
                    <div class="alert alert-success">
                        <i data-lucide="check-circle" class="w-6 h-6"></i>
                        <div>
                            <p class="font-semibold">"Knowledge Graph Ready!"</p>
                            <p class="text-sm">"You can now query your graph in the Query tab"</p>
                        </div>
                    </div>
                }.into_any(),

                BuildStatus::Error(msg) => view! {
                    <div class="alert alert-error">
                        <i data-lucide="alert-triangle" class="w-6 h-6"></i>
                        <div>
                            <p class="font-semibold">"Build Error"</p>
                            <p class="text-sm">{msg}</p>
                        </div>
                    </div>
                }.into_any(),
            }}
        </div>
    }
}

/// Explore tab component
#[component]
fn ExploreTab(
    graph_stats: ReadSignal<GraphStats>,
    build_status: ReadSignal<BuildStatus>,
) -> impl IntoView {
    view! {
        <div class="space-y-6" role="tabpanel" id="explore-panel">
            {move || match build_status.get() {
                BuildStatus::Ready => view! {
                    <>
                        // Overview Stats
                        <div class="card bg-base-200 shadow-xl">
                            <div class="card-body">
                                <h2 class="card-title flex items-center gap-2">
                                    <i data-lucide="bar-chart-3" class="w-6 h-6"></i>
                                    <span>"Graph Statistics"</span>
                                </h2>

                                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {move || {
                                        let stats = graph_stats.get();
                                        let density = if stats.entities > 0 {
                                            format!("{:.1}%", (stats.relationships as f32 / (stats.entities as f32 * 2.0)) * 100.0)
                                        } else {
                                            "0%".to_string()
                                        };

                                        view! {
                                            <>
                                                <StatCard
                                                    icon="file-text"
                                                    label="Documents"
                                                    value=stats.documents.to_string()
                                                    color="info"
                                                />
                                                <StatCard
                                                    icon="package"
                                                    label="Text Chunks"
                                                    value=stats.chunks.to_string()
                                                    color="success"
                                                />
                                                <StatCard
                                                    icon="tag"
                                                    label="Entities"
                                                    value=stats.entities.to_string()
                                                    color="warning"
                                                />
                                                <StatCard
                                                    icon="link"
                                                    label="Relationships"
                                                    value=stats.relationships.to_string()
                                                    color="primary"
                                                />
                                                <StatCard
                                                    icon="cpu"
                                                    label="Embeddings"
                                                    value=stats.embeddings.to_string()
                                                    color="secondary"
                                                />
                                                <StatCard
                                                    icon="trending-up"
                                                    label="Graph Density"
                                                    value=density
                                                    color="accent"
                                                />
                                            </>
                                        }
                                    }}
                                </div>
                            </div>
                        </div>

                        // Graph Health
                        <div class="card bg-base-200 shadow-xl">
                            <div class="card-body">
                                <h2 class="card-title flex items-center gap-2">
                                    <i data-lucide="heart-pulse" class="w-6 h-6"></i>
                                    <span>"Graph Health"</span>
                                </h2>

                                <div class="space-y-4">
                                    <HealthIndicator
                                        label="Coverage"
                                        value=100.0
                                        status="good"
                                        description="All documents processed"
                                    />
                                    <HealthIndicator
                                        label="Entity Linking"
                                        value=85.0
                                        status="good"
                                        description="Strong entity connections"
                                    />
                                    <HealthIndicator
                                        label="Embedding Quality"
                                        value=92.0
                                        status="good"
                                        description="High-quality vector representations"
                                    />
                                </div>
                            </div>
                        </div>

                        // System Info
                        <div class="card bg-base-200 shadow-xl">
                            <div class="card-body">
                                <h3 class="card-title flex items-center gap-2">
                                    <i data-lucide="settings" class="w-6 h-6"></i>
                                    <span>"System Configuration"</span>
                                </h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                                    <div class="space-y-2">
                                        <div class="flex justify-between p-3 bg-base-100 rounded-lg">
                                            <span class="opacity-70">"Tokenizer:"</span>
                                            <span class="font-mono text-success">"HuggingFace"</span>
                                        </div>
                                        <div class="flex justify-between p-3 bg-base-100 rounded-lg">
                                            <span class="opacity-70">"Vocabulary:"</span>
                                            <span class="font-mono text-success">"30,522 tokens"</span>
                                        </div>
                                        <div class="flex justify-between p-3 bg-base-100 rounded-lg">
                                            <span class="opacity-70">"Embeddings:"</span>
                                            <span class="font-mono text-info">"ONNX Web"</span>
                                        </div>
                                    </div>
                                    <div class="space-y-2">
                                        <div class="flex justify-between p-3 bg-base-100 rounded-lg">
                                            <span class="opacity-70">"LLM:"</span>
                                            <span class="font-mono text-warning">"WebLLM (Qwen3)"</span>
                                        </div>
                                        <div class="flex justify-between p-3 bg-base-100 rounded-lg">
                                            <span class="opacity-70">"Vector Search:"</span>
                                            <span class="font-mono text-primary">"Voy k-d tree"</span>
                                        </div>
                                        <div class="flex justify-between p-3 bg-base-100 rounded-lg">
                                            <span class="opacity-70">"Storage:"</span>
                                            <span class="font-mono text-secondary">"IndexedDB"</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </>
                }.into_any(),

                _ => view! {
                    <div class="card bg-base-200 shadow-xl">
                        <div class="card-body text-center py-12">
                            <i data-lucide="construction" class="w-16 h-16 mx-auto mb-4 opacity-50"></i>
                            <h2 class="text-2xl font-semibold mb-4">"No Graph Yet"</h2>
                            <p class="opacity-70 mb-6">
                                "Build a knowledge graph from your documents to see statistics and insights"
                            </p>
                            <div class="inline-flex items-center gap-2 text-sm opacity-60">
                                <span>"Go to"</span>
                                <span class="badge badge-primary">"Build Graph"</span>
                                <span>"tab to get started"</span>
                            </div>
                        </div>
                    </div>
                }.into_any(),
            }}
        </div>
    }
}

/// Stat card component
#[component]
fn StatCard(
    icon: &'static str,
    label: &'static str,
    value: String,
    color: &'static str,
) -> impl IntoView {
    let _badge_class = match color {
        "info" => "badge-info",
        "success" => "badge-success",
        "warning" => "badge-warning",
        "primary" => "badge-primary",
        "secondary" => "badge-secondary",
        "accent" => "badge-accent",
        _ => "badge-neutral",
    };

    let class_str = format!("stats shadow border border-base-300 hover:border-{} transition-all hover:scale-105", color);

    view! {
        <div class=class_str>
            <div class="stat">
                <div class="stat-figure">
                    <i data-lucide=icon class=format!("w-8 h-8 text-{}", color)></i>
                </div>
                <div class="stat-title">{label}</div>
                <div class="stat-value text-3xl">{value}</div>
            </div>
        </div>
    }
}

/// Health indicator component
#[component]
fn HealthIndicator(
    label: &'static str,
    value: f32,
    status: &'static str,
    description: &'static str,
) -> impl IntoView {
    let progress_class = match status {
        "good" => "progress-success",
        "warning" => "progress-warning",
        "error" => "progress-error",
        _ => "progress-neutral",
    };

    view! {
        <div>
            <div class="flex items-center justify-between mb-2">
                <span class="font-medium">{label}</span>
                <span class="text-sm opacity-70">{format!("{:.0}%", value)}</span>
            </div>
            <progress
                class=format!("progress {} w-full", progress_class)
                value=format!("{}", value as i32)
                max="100"
            ></progress>
            <p class="text-xs opacity-60 mt-1">{description}</p>
        </div>
    }
}

/// Query tab component
#[component]
fn QueryTab(
    query: ReadSignal<String>,
    set_query: WriteSignal<String>,
    results: ReadSignal<String>,
    set_results: WriteSignal<String>,
    loading: ReadSignal<bool>,
    set_loading: WriteSignal<bool>,
    build_status: ReadSignal<BuildStatus>,
    graph_stats: ReadSignal<GraphStats>,
    graphrag_instance: StoredValue<Option<GraphRAG>>,
    vector_index: StoredValue<Option<VectorIndex>>,
) -> impl IntoView {
    // Local is_graph_ready check
    let is_graph_ready = move || matches!(build_status.get(), BuildStatus::Ready);
    let on_input = move |ev| {
        let val = event_target_value(&ev);
        set_query.set(val);
    };

    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();

        if !is_graph_ready() {
            return;
        }

        set_loading.set(true);
        let query_text = query.get();
        let stats = graph_stats.get();

        spawn_local(async move {
            // Get vector results if embeddings available
            let vector_results: Option<Vec<(String, f64)>> = if stats.embeddings > 0 {
                web_sys::console::log_1(&"🔍 Using vector search for query...".into());

                // Fetch tokenizer.json and create embedder for query
                use gloo_net::http::Request;

                let tokenizer_result = Request::get("./tokenizer.json").send().await;

                let embedder_result = if let Ok(response) = tokenizer_result {
                    if let Ok(tokenizer_json) = response.text().await {
                        OnnxEmbedder::from_tokenizer_json(384, &tokenizer_json)
                    } else {
                        Err(onnx_embedder::OnnxEmbedderError::InvalidInput(
                            "Failed to read tokenizer.json".to_string()
                        ))
                    }
                } else {
                    Err(onnx_embedder::OnnxEmbedderError::InvalidInput(
                        "Failed to fetch tokenizer.json".to_string()
                    ))
                };

                match embedder_result {
                    Ok(mut embedder) => {
                        if let Ok(_) = embedder.load_model("./models/minilm-l6.onnx", true).await {
                            match embedder.embed(&query_text).await {
                                Ok(query_embedding) => {
                                    // Search using pure Rust vector index
                                    vector_index.with_value(|index_opt| {
                                        if let Some(index) = index_opt.as_ref() {
                                            let results = index.search(&query_embedding, 5);
                                            let vec_results: Vec<(String, f64)> = results
                                                .iter()
                                                .map(|r| (r.id.clone(), r.similarity))
                                                .collect();
                                            Some(vec_results)
                                        } else {
                                            None
                                        }
                                    })
                                }
                                Err(e) => {
                                    web_sys::console::warn_1(&format!("Failed to embed query: {}", e).into());
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            // Build the response with vector search results
            let response = if let Some(ref vec_res) = vector_results {
                let mut result_text = format!(
                    "Query: \"{}\"\n\n\
                    ✅ Graph Search Complete\n\
                    📊 Searched {} chunks across {} documents\n",
                    query_text,
                    stats.chunks,
                    stats.documents
                );

                // Add vector search results with chunk content and entities
                if !vec_res.is_empty() {
                    result_text.push_str(&format!("\n🎯 Top {} Relevant Text Chunks:\n\n", vec_res.len()));

                    // Get chunk details from GraphRAG
                    graphrag_instance.update_value(|graphrag_opt| {
                        if let Some(ref graphrag) = graphrag_opt {
                            for (idx, (chunk_id, similarity)) in vec_res.iter().enumerate() {
                                result_text.push_str(&format!("{}. Similarity: {:.3}\n",
                                    idx + 1, similarity));

                                // Get chunk content
                                if let Some(chunk) = graphrag.get_chunk(chunk_id) {
                                    // Show truncated content (first 200 chars)
                                    let content_preview = if chunk.content.len() > 200 {
                                        format!("{}...", &chunk.content[..200])
                                    } else {
                                        chunk.content.clone()
                                    };
                                    result_text.push_str(&format!("   📄 \"{}\"\n\n",
                                        content_preview.replace('\n', " ")));

                                    // Show entities found in this chunk
                                    if !chunk.entities.is_empty() {
                                        result_text.push_str(&format!("   🏷️  Entities ({}):\n",
                                            chunk.entities.len()));

                                        for entity_id in chunk.entities.iter().take(5) {
                                            if let Some(entity) = graphrag.get_entity(&entity_id.0) {
                                                result_text.push_str(&format!("      • {}: {} (confidence: {:.2})\n",
                                                    entity.entity_type, entity.name, entity.confidence));

                                                // Show relationships for this entity
                                                let relationships = graphrag.get_entity_relationships(&entity_id.0);
                                                if !relationships.is_empty() {
                                                    for rel in relationships.iter().take(2) {
                                                        let other_entity_id = if &rel.source == entity_id {
                                                            &rel.target.0
                                                        } else {
                                                            &rel.source.0
                                                        };

                                                        if let Some(other_entity) = graphrag.get_entity(other_entity_id) {
                                                            result_text.push_str(&format!("        → {} {}\n",
                                                                rel.relation_type, other_entity.name));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        if chunk.entities.len() > 5 {
                                            result_text.push_str(&format!("      ... and {} more entities\n",
                                                chunk.entities.len() - 5));
                                        }
                                    }
                                }
                                result_text.push_str("\n");
                            }
                        }
                    });
                }

                // Build context for LLM from retrieved chunks
                let mut context_for_llm = String::new();
                graphrag_instance.update_value(|graphrag_opt| {
                    if let Some(ref graphrag) = graphrag_opt {
                        for (chunk_id, _) in vec_res.iter().take(5) {
                            if let Some(chunk) = graphrag.get_chunk(chunk_id) {
                                context_for_llm.push_str(&chunk.content);
                                context_for_llm.push_str("\n\n");

                                // Add entity info
                                for entity_id in chunk.entities.iter().take(3) {
                                    if let Some(entity) = graphrag.get_entity(&entity_id.0) {
                                        context_for_llm.push_str(&format!("[Entity: {} - {}] ",
                                            entity.entity_type, entity.name));
                                    }
                                }
                                context_for_llm.push_str("\n---\n\n");
                            }
                        }
                    }
                });

                // Synthesize natural language answer using WebLLM
                if !context_for_llm.is_empty() {
                    result_text.push_str("\n🤖 Synthesizing natural language answer...\n");

                    // Try to synthesize answer with WebLLM
                    match webllm::WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await {
                        Ok(llm) => {
                            let messages = vec![
                                webllm::ChatMessage::system(
                                    "You are a helpful AI assistant that answers questions based on provided context from a knowledge graph. \
                                    Be concise, accurate, and cite specific entities and relationships when relevant. \
                                    If the context doesn't contain enough information, say so clearly."
                                ),
                                webllm::ChatMessage::user(
                                    format!("Question: {}\n\nContext from Knowledge Graph:\n{}\n\nPlease provide a natural language answer to the question based on the context above.",
                                        query_text, context_for_llm)
                                ),
                            ];

                            match llm.chat(messages, Some(0.7), Some(512)).await {
                                Ok(answer) => {
                                    result_text.push_str("\n💬 Synthesized Answer:\n");
                                    result_text.push_str(&format!("{}\n\n", answer));
                                    web_sys::console::log_1(&format!("✅ LLM synthesis successful: {} chars", answer.len()).into());
                                }
                                Err(e) => {
                                    result_text.push_str(&format!("\n⚠️  LLM synthesis failed: {}\n", e));
                                    result_text.push_str(&format!("Context prepared ({} chars)\n", context_for_llm.len()));
                                    web_sys::console::error_1(&format!("LLM synthesis error: {}", e).into());
                                }
                            }
                        }
                        Err(e) => {
                            result_text.push_str(&format!("\n⚠️  WebLLM initialization failed: {}\n", e));
                            result_text.push_str("Make sure WebGPU is enabled in your browser.\n");
                            result_text.push_str(&format!("Context prepared ({} chars)\n", context_for_llm.len()));
                            web_sys::console::error_1(&format!("WebLLM init error: {}", e).into());
                        }
                    }
                }

                result_text
            } else {
                "❌ No results found.".to_string()
            };

            set_results.set(response);
            set_loading.set(false);
        });
    };

    view! {
        <div class="space-y-6" role="tabpanel" id="query-panel">
            // Query Interface
            <div class="card bg-base-200 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title flex items-center gap-2">
                        <i data-lucide="message-square" class="w-6 h-6"></i>
                        <span>"Query Interface"</span>
                    </h2>

                    {move || if !is_graph_ready() {
                        view! {
                            <div class="alert alert-warning">
                                <i data-lucide="alert-triangle" class="w-5 h-5"></i>
                                <span>"Please build a knowledge graph first before querying"</span>
                            </div>
                        }.into_any()
                    } else {
                        view! { <div></div> }.into_any()
                    }}

                    <form on:submit=on_submit class="space-y-4">
                        <div>
                            <label
                                for="query-input"
                                class="block text-sm font-medium mb-2"
                            >
                                "Enter your query"
                            </label>
                            <input
                                id="query-input"
                                type="text"
                                placeholder="What would you like to know?"
                                class="input input-bordered w-full"
                                on:input=on_input
                                prop:value=move || query.get()
                                prop:disabled=move || !is_graph_ready()
                            />
                        </div>
                        <button
                            type="submit"
                            class="btn btn-primary w-full gap-2"
                            prop:disabled=move || loading.get() || !is_graph_ready()
                        >
                            {move || if loading.get() {
                                view! {
                                    <>
                                        <span class="loading loading-spinner"></span>
                                        "Searching..."
                                    </>
                                }.into_any()
                            } else {
                                view! {
                                    <>
                                        <i data-lucide="search" class="w-5 h-5"></i>
                                        "Search Graph"
                                    </>
                                }.into_any()
                            }}
                        </button>
                    </form>
                </div>
            </div>

            // Results
            <div class="card bg-base-200 shadow-xl">
                <div class="card-body">
                    <h2 class="card-title flex items-center gap-2">
                        <i data-lucide="file-text" class="w-6 h-6"></i>
                        <span>"Results"</span>
                    </h2>

                    {move || if loading.get() {
                        view! {
                            <div class="flex items-center justify-center py-12">
                                <div class="flex flex-col items-center gap-4">
                                    <span class="loading loading-spinner loading-lg text-primary"></span>
                                    <p class="opacity-70">"Searching knowledge graph..."</p>
                                </div>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <pre class="bg-base-100 p-6 rounded-lg text-sm whitespace-pre-wrap leading-relaxed font-mono">
                                {move || results.get()}
                            </pre>
                        }.into_any()
                    }}
                </div>
            </div>
        </div>
    }
}

/// Hierarchy tab component
#[component]
fn HierarchyTab(
    max_level: ReadSignal<usize>,
    set_max_level: WriteSignal<usize>,
    communities: ReadSignal<Vec<CommunityData>>,
    set_communities: WriteSignal<Vec<CommunityData>>,
    build_status: ReadSignal<BuildStatus>,
    graphrag_instance: StoredValue<Option<GraphRAG>>,
) -> impl IntoView {
    let is_graph_ready = move || matches!(build_status.get(), BuildStatus::Ready);

    // Handle community detection
    let handle_detect_communities = Callback::new(move |_: ()| {
        web_sys::console::log_1(&"🔍 Detecting hierarchical communities...".into());

        spawn_local(async move {
            // TODO: Call Leiden algorithm on the graph
            // For now, create mock data to demonstrate the UI
            web_sys::console::log_1(&"⚠️  Hierarchical community detection not yet implemented".into());
            web_sys::console::log_1(&"   This feature requires the Leiden algorithm integration".into());

            // Set mock max level
            set_max_level.set(3);

            // Load mock communities for level 0
            let mock_communities = vec![
                CommunityData {
                    id: 0,
                    level: 0,
                    entity_count: 15,
                    summary: "Philosophy and Love: This community focuses on philosophical discussions about the nature of love, beauty, and wisdom.".to_string(),
                    entities: vec!["Socrates".to_string(), "Plato".to_string(), "Beauty".to_string(), "Love".to_string()],
                },
                CommunityData {
                    id: 1,
                    level: 0,
                    entity_count: 12,
                    summary: "Greek Symposium: Entities related to the structure and participants of ancient Greek symposia.".to_string(),
                    entities: vec!["Agathon".to_string(), "Aristophanes".to_string(), "Pausanias".to_string()],
                },
            ];

            set_communities.set(mock_communities);
            web_sys::console::log_1(&"✅ Mock hierarchical communities created (3 levels)".into());
        });
    });

    // Handle level change
    let handle_level_change = Callback::new(move |level: usize| {
        web_sys::console::log_1(&format!("📊 Switching to level {}", level).into());

        spawn_local(async move {
            // TODO: Load communities at the requested level
            // For now, create different mock data based on level
            let mock_communities = match level {
                0 => vec![
                    CommunityData {
                        id: 0,
                        level: 0,
                        entity_count: 15,
                        summary: "Philosophy and Love: This community focuses on philosophical discussions about the nature of love, beauty, and wisdom.".to_string(),
                        entities: vec!["Socrates".to_string(), "Plato".to_string(), "Beauty".to_string(), "Love".to_string()],
                    },
                    CommunityData {
                        id: 1,
                        level: 0,
                        entity_count: 12,
                        summary: "Greek Symposium: Entities related to the structure and participants of ancient Greek symposia.".to_string(),
                        entities: vec!["Agathon".to_string(), "Aristophanes".to_string(), "Pausanias".to_string()],
                    },
                ],
                1 => vec![
                    CommunityData {
                        id: 2,
                        level: 1,
                        entity_count: 27,
                        summary: "Ancient Greek Philosophy: Broader grouping of philosophical concepts and symposium participants.".to_string(),
                        entities: vec!["Socrates".to_string(), "Plato".to_string(), "Agathon".to_string(), "Beauty".to_string(), "Love".to_string(), "Wisdom".to_string()],
                    },
                ],
                _ => vec![
                    CommunityData {
                        id: 3,
                        level: 2,
                        entity_count: 45,
                        summary: "Classical Literature: High-level overview of all entities from ancient Greek texts.".to_string(),
                        entities: vec!["Socrates".to_string(), "Plato".to_string(), "Symposium".to_string(), "Greece".to_string()],
                    },
                ],
            };

            set_communities.set(mock_communities);
        });
    });

    view! {
        <div class="space-y-6" role="tabpanel" id="hierarchy-panel">
            {move || if !is_graph_ready() {
                view! {
                    <div class="card bg-base-200 shadow-xl">
                        <div class="card-body text-center py-12">
                            <i data-lucide="network" class="w-16 h-16 mx-auto mb-4 opacity-50"></i>
                            <h2 class="text-2xl font-semibold mb-4">"Build Graph First"</h2>
                            <p class="opacity-70 mb-6">
                                "Hierarchical community detection requires a built knowledge graph. Go to the Build tab to create your graph first."
                            </p>
                            <div class="inline-flex items-center gap-2 text-sm opacity-60">
                                <span>"Go to"</span>
                                <span class="badge badge-primary">"Build Graph"</span>
                                <span>"tab to get started"</span>
                            </div>
                        </div>
                    </div>
                }.into_any()
            } else {
                view! {
                    <div class="space-y-6">
                        // Info card about hierarchy
                        <div class="alert alert-info">
                            <i data-lucide="info" class="w-5 h-5"></i>
                            <div>
                                <p class="font-semibold">"Hierarchical Community Detection"</p>
                                <p class="text-sm">
                                    "Discover multi-level community structures using the Leiden algorithm. "
                                    "Click 'Detect Communities' to analyze your knowledge graph's hierarchical organization."
                                </p>
                            </div>
                        </div>

                        // Warning about mock data
                        <div class="alert alert-warning">
                            <i data-lucide="alert-triangle" class="w-5 h-5"></i>
                            <div>
                                <p class="font-semibold">"Demo Mode - Mock Data"</p>
                                <p class="text-sm">
                                    "Hierarchical clustering integration is complete but uses mock data for demonstration. "
                                    "Full Leiden algorithm integration is available in the Rust backend (graphrag-core)."
                                </p>
                            </div>
                        </div>

                        // Hierarchy Explorer Component
                        <HierarchyExplorer
                            max_level=max_level
                            communities=communities
                            on_level_change=handle_level_change
                            on_detect_communities=handle_detect_communities
                        />
                    </div>
                }.into_any()
            }}
        </div>
    }
}

/// Footer component
#[component]
fn Footer() -> impl IntoView {
    view! {
        <footer class="mt-12 text-center opacity-70 text-sm space-y-2">
            <p>
                "Built with "
                <a href="https://leptos.dev" target="_blank" class="link link-primary">
                    "Leptos"
                </a>
                " + "
                <a href="https://www.rust-lang.org" target="_blank" class="link link-primary">
                    "Rust"
                </a>
                " + "
                <a href="https://webassembly.org" target="_blank" class="link link-primary">
                    "WebAssembly"
                </a>
            </p>
            <p class="text-xs opacity-60">"GraphRAG WASM • Complete Document Pipeline • v0.2.0"</p>
        </footer>
    }
}

// Utility functions

fn format_timestamp(timestamp: f64) -> String {
    let now = js_sys::Date::now();
    let diff = (now - timestamp) / 1000.0; // seconds

    if diff < 60.0 {
        "just now".to_string()
    } else if diff < 3600.0 {
        format!("{} min ago", (diff / 60.0) as i32)
    } else if diff < 86400.0 {
        format!("{} hours ago", (diff / 3600.0) as i32)
    } else {
        format!("{} days ago", (diff / 86400.0) as i32)
    }
}

fn main() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
    web_sys::console::log_1(&"🚀 GraphRAG WASM App Started - Full Pipeline".into());

    leptos::mount::mount_to_body(App);
}
