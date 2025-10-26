//! GraphRAG Leptos Demo - ONNX Embeddings Integration
//!
//! This demo shows how to integrate the Leptos UI components with ONNX Runtime Web
//! for GPU-accelerated embeddings in the browser.
//!
//! ## Features
//! - Interactive chat interface with GraphRAG
//! - ONNX Runtime Web + WebGPU acceleration (20-40x faster than CPU)
//! - Real-time graph visualization
//! - Document upload and indexing
//! - Semantic search with embeddings

use leptos::prelude::*;
use leptos::prelude::{signal, Effect};
use leptos::task::spawn_local;
use leptos_meta::*;
use graphrag_leptos::*;
use graphrag_wasm::GraphRAG;
use graphrag_wasm::onnx_embedder::{WasmOnnxEmbedder, check_onnx_runtime};

/// Main application component
#[component]
fn App() -> impl IntoView {
    provide_meta_context();

    // State management
    let (graph_instance, set_graph_instance) = signal(None::<GraphRAG>);
    let (embedder, set_embedder) = signal(None::<WasmOnnxEmbedder>);
    let (_is_loading, set_is_loading) = signal(false);
    let (status_message, set_status_message) = signal(String::from("Initializing..."));
    let (error_message, set_error_message) = signal(None::<String>);

    // Graph stats
    let (entity_count, _set_entity_count) = signal(0_usize);
    let (relationship_count, _set_relationship_count) = signal(0_usize);
    let (document_count, set_document_count) = signal(0_usize);
    let (vector_count, _set_vector_count) = signal(0_usize);

    // Graph visualization
    let (graph_nodes, _set_graph_nodes) = signal(Vec::<GraphNode>::new());
    let (graph_edges, _set_graph_edges) = signal(Vec::<GraphEdge>::new());

    // Initialize ONNX embedder and GraphRAG on mount
    Effect::new(move |_| {
        spawn_local(async move {
            if !check_onnx_runtime() {
                set_error_message.set(Some(
                    "ONNX Runtime not found! Please add the script tag to index.html".to_string()
                ));
                set_status_message.set("ONNX Runtime missing".to_string());
                return;
            }

            set_status_message.set("Loading ONNX model...".to_string());

            // Create ONNX embedder
            match WasmOnnxEmbedder::new(384) {
                Ok(mut emb) => {
                    // Load model with WebGPU
                    match emb.load_model("./models/all-MiniLM-L6-v2.onnx", Some(true)).await {
                        Ok(_) => {
                            web_sys::console::log_1(&"✅ ONNX model loaded with WebGPU".into());
                            set_embedder.set(Some(emb));

                            // Create GraphRAG instance
                            match GraphRAG::new(384) {
                                Ok(graph) => {
                                    set_graph_instance.set(Some(graph));
                                    set_status_message.set("Ready! Add documents or ask questions.".to_string());
                                    set_error_message.set(None);
                                }
                                Err(e) => {
                                    let err_msg = format!("Failed to create GraphRAG: {:?}", e);
                                    web_sys::console::error_1(&err_msg.clone().into());
                                    set_error_message.set(Some(err_msg));
                                }
                            }
                        }
                        Err(e) => {
                            let err_msg = format!("Failed to load ONNX model: {:?}", e);
                            web_sys::console::error_1(&err_msg.clone().into());
                            set_error_message.set(Some(err_msg));
                            set_status_message.set("Model loading failed".to_string());
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Failed to create ONNX embedder: {:?}", e);
                    web_sys::console::error_1(&err_msg.clone().into());
                    set_error_message.set(Some(err_msg));
                }
            }
        });
    });

    // Handle query from ChatWindow
    let handle_query = Callback::new(move |query: String| {
        spawn_local(async move {
            set_is_loading.set(true);
            set_status_message.set(format!("Processing: {}", query));

            if let (Some(emb), Some(graph)) = (embedder.get(), graph_instance.get()) {
                // Generate query embedding
                match emb.embed(&query).await {
                    Ok(query_emb) => {
                        // Convert to Vec<f32>
                        let mut query_vec = vec![0.0f32; query_emb.length() as usize];
                        query_emb.copy_to(&mut query_vec);

                        // Query GraphRAG
                        match graph.query(query_vec, 3).await {
                            Ok(results) => {
                                web_sys::console::log_1(&format!("✅ Query results: {}", results).into());
                                set_status_message.set("Query completed".to_string());
                                // TODO: Update UI with results
                            }
                            Err(e) => {
                                let err = format!("Query failed: {:?}", e);
                                web_sys::console::error_1(&err.clone().into());
                                set_error_message.set(Some(err));
                            }
                        }
                    }
                    Err(e) => {
                        let err = format!("Embedding generation failed: {:?}", e);
                        web_sys::console::error_1(&err.clone().into());
                        set_error_message.set(Some(err));
                    }
                }
            } else {
                set_error_message.set(Some("System not initialized".to_string()));
            }

            set_is_loading.set(false);
        });
    });

    // Handle document upload
    let handle_upload = Callback::new(move |files: Vec<String>| {
        spawn_local(async move {
            set_is_loading.set(true);
            set_status_message.set(format!("Processing {} documents...", files.len()));

            if let (Some(emb), Some(mut graph)) = (embedder.get(), graph_instance.get()) {
                for (idx, file_name) in files.iter().enumerate() {
                    // For demo, use file name as content
                    let content = format!("Document about {}", file_name);

                    // Generate embedding
                    match emb.embed(&content).await {
                        Ok(embedding_js) => {
                            let mut embedding_vec = vec![0.0f32; embedding_js.length() as usize];
                            embedding_js.copy_to(&mut embedding_vec);

                            match graph.add_document(
                                format!("doc_{}", idx),
                                content.clone(),
                                embedding_vec
                            ).await {
                                Ok(_) => {
                                    web_sys::console::log_1(&format!("✅ Added: {}", file_name).into());
                                    set_document_count.update(|c| *c += 1);
                                }
                                Err(e) => {
                                    web_sys::console::error_1(&format!("Failed to add {}: {:?}", file_name, e).into());
                                }
                            }
                        }
                        Err(e) => {
                            web_sys::console::error_1(&format!("Embedding failed for {}: {:?}", file_name, e).into());
                        }
                    }
                }

                // Build index after adding documents
                match graph.build_index().await {
                    Ok(_) => {
                        set_graph_instance.set(Some(graph));
                        set_status_message.set(format!("Indexed {} documents", files.len()));
                    }
                    Err(e) => {
                        set_error_message.set(Some(format!("Index build failed: {:?}", e)));
                    }
                }
            }

            set_is_loading.set(false);
        });
    });

    // Handle document removal
    let handle_remove = Callback::new(move |_doc_id: String| {
        // TODO: Implement document removal
        set_document_count.update(|c| *c = c.saturating_sub(1));
    });

    view! {
        <Html attr:lang="en" />
        <Title text="GraphRAG Leptos Demo - ONNX Embeddings" />
        <Meta name="description" content="GraphRAG with ONNX Runtime Web and Leptos" />
        <Meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <Link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" />
        <Script src="https://cdn.tailwindcss.com" />

        <main class="min-h-screen bg-base-200 p-4">
            <div class="container mx-auto max-w-7xl">
                // Header
                <div class="hero bg-base-100 rounded-lg shadow-xl mb-4">
                    <div class="hero-content text-center">
                        <div class="max-w-md">
                            <h1 class="text-5xl font-bold">"GraphRAG Demo"</h1>
                            <p class="py-6">
                                "GPU-accelerated embeddings with ONNX Runtime Web + WebGPU"
                            </p>

                            // Status indicator
                            <div class="alert" class:alert-success={move || error_message.get().is_none()}
                                 class:alert-error={move || error_message.get().is_some()}>
                                <span>{move || status_message.get()}</span>
                            </div>

                            // Error display
                            {move || error_message.get().map(|err| view! {
                                <div class="alert alert-error mt-2">
                                    <span>{err}</span>
                                </div>
                            })}
                        </div>
                    </div>
                </div>

                // Main content grid
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    // Left column: Chat and Query
                    <div class="lg:col-span-2 space-y-4">
                        <ChatWindow on_query=handle_query />
                    </div>

                    // Right column: Stats and Controls
                    <div class="space-y-4">
                        <GraphStats
                            entity_count=entity_count
                            relationship_count=relationship_count
                            document_count=document_count
                            vector_count=vector_count
                        />

                        <DocumentManager
                            on_upload=handle_upload
                            on_remove=handle_remove
                        />
                    </div>
                </div>

                // Graph visualization
                <div class="mt-4">
                    <GraphVisualization
                        nodes=graph_nodes
                        edges=graph_edges
                    />
                </div>

                // Footer
                <footer class="footer footer-center p-4 mt-8 bg-base-100 text-base-content rounded-lg">
                    <div>
                        <p>"GraphRAG Leptos Demo - Built with Rust + WASM + ONNX Runtime Web"</p>
                        <p class="text-sm opacity-60">
                            "Using all-MiniLM-L6-v2 (384-dim) with WebGPU acceleration"
                        </p>
                    </div>
                </footer>
            </div>
        </main>
    }
}

/// Entry point
fn main() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());

    web_sys::console::log_1(&"GraphRAG Leptos Demo starting...".into());

    mount_to_body(|| view! { <App /> })
}
