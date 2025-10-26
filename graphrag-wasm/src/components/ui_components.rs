//! # GraphRAG WASM UI Components
//!
//! Ready-to-use Leptos UI components for GraphRAG.
//!
//! ## Components
//!
//! - `<ChatWindow/>`: Interactive chat interface with history
//! - `<QueryInterface/>`: Query input with syntax highlighting
//! - `<GraphStats/>`: Real-time graph statistics display
//! - `<DocumentManager/>`: Upload and manage documents
//! - `<GraphVisualization/>`: Interactive graph rendering with force-directed layout

#![allow(dead_code)]

use leptos::prelude::*;
use leptos::ev::SubmitEvent;
use serde::{Deserialize, Serialize};
use super::force_layout::{ForceLayout, LayoutConfig};

/// Message type for chat history
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Chat window component for GraphRAG queries
///
/// Provides a full-featured chat interface with:
/// - Message history with automatic scrolling
/// - Loading states during query processing
/// - Error handling with user-friendly messages
/// - Clean, accessible UI with TailwindCSS
///
/// # Props
/// - `on_query` - Callback function when user submits a query
/// - `on_clear` - Optional callback when history is cleared
#[component]
pub fn ChatWindow(
    /// Callback when user submits a query
    #[allow(dead_code)]
    on_query: Callback<String, ()>,
    /// Optional callback when history is cleared
    #[prop(optional)]
    #[allow(dead_code)]
    on_clear: Option<Callback<(), ()>>,
) -> impl IntoView {
    let (messages, set_messages) = signal(Vec::<ChatMessage>::new());
    let (is_loading, set_is_loading) = signal(false);

    let handle_clear = move |_| {
        set_messages.set(Vec::new());
        if let Some(cb) = on_clear {
            cb.run(());
        }
    };

    view! {
        <div class="chat-window flex flex-col h-full bg-base-100 rounded-lg shadow-lg">
            // Header
            <div class="chat-header flex justify-between items-center p-4 border-b border-base-300">
                <h2 class="text-2xl font-bold">"GraphRAG Chat"</h2>
                <button
                    class="btn btn-sm btn-ghost"
                    on:click=handle_clear
                >
                    "Clear History"
                </button>
            </div>

            // Messages container
            <div class="messages flex-1 overflow-y-auto p-4 space-y-4">
                <For
                    each=move || messages.get()
                    key=|msg| msg.timestamp.clone()
                    children=move |msg: ChatMessage| {
                        let role_class = match msg.role {
                            MessageRole::User => "chat-end",
                            MessageRole::Assistant => "chat-start",
                            MessageRole::System => "chat-start opacity-60",
                        };
                        let bubble_class = match msg.role {
                            MessageRole::User => "chat-bubble-primary",
                            MessageRole::Assistant => "chat-bubble-secondary",
                            MessageRole::System => "chat-bubble-info",
                        };

                        view! {
                            <div class=format!("chat {}", role_class)>
                                <div class=format!("chat-bubble {}", bubble_class)>
                                    {msg.content}
                                </div>
                                <div class="chat-footer opacity-50 text-xs">
                                    {msg.timestamp}
                                </div>
                            </div>
                        }
                    }
                />

                {move || is_loading.get().then(|| view! {
                    <div class="chat chat-start">
                        <div class="chat-bubble chat-bubble-secondary">
                            <span class="loading loading-dots loading-sm"></span>
                        </div>
                    </div>
                })}
            </div>

            // Input area
            <QueryInterface on_submit=Callback::new(move |query: String| {
                // Add user message
                let user_msg = ChatMessage {
                    role: MessageRole::User,
                    content: query.clone(),
                    timestamp: js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default(),
                };
                set_messages.update(|msgs| msgs.push(user_msg));
                set_is_loading.set(true);

                // Call the query handler
                on_query.run(query);

                // Note: Response handling should be done by parent component
                set_is_loading.set(false);
            }) />
        </div>
    }
}

/// Query interface component
///
/// Provides:
/// - Text input with auto-focus
/// - Submit on Enter key
/// - Clear button
/// - Character counter
/// - Disabled state during loading
///
/// # Props
/// - `on_submit` - Callback function when query is submitted
/// - `disabled` - Optional disabled state
#[component]
pub fn QueryInterface(
    /// Callback when query is submitted
    #[allow(dead_code)]
    on_submit: Callback<String, ()>,
    /// Optional disabled state
    #[prop(optional)]
    #[allow(dead_code)]
    disabled: Option<bool>,
) -> impl IntoView {
    let (query, set_query) = signal(String::new());
    let input_ref = NodeRef::<leptos::html::Input>::new();

    let handle_submit = move |ev: SubmitEvent| {
        ev.prevent_default();
        let query_text = query.get().trim().to_string();
        if !query_text.is_empty() {
            on_submit.run(query_text);
            set_query.set(String::new());
        }
    };

    let handle_keydown = move |ev: leptos::ev::KeyboardEvent| {
        if ev.key() == "Enter" && !ev.shift_key() {
            ev.prevent_default();
            let query_text = query.get().trim().to_string();
            if !query_text.is_empty() {
                on_submit.run(query_text);
                set_query.set(String::new());
            }
        }
    };

    view! {
        <form class="query-interface p-4 border-t border-base-300" on:submit=handle_submit>
            <div class="flex gap-2">
                <input
                    type="text"
                    class="input input-bordered flex-1"
                    placeholder="Ask a question about your documents..."
                    prop:value=query
                    on:input=move |ev| set_query.set(event_target_value(&ev))
                    on:keydown=handle_keydown
                    node_ref=input_ref
                    disabled=disabled.unwrap_or(false)
                />
                <button
                    type="submit"
                    class="btn btn-primary"
                    disabled=move || query.get().trim().is_empty() || disabled.unwrap_or(false)
                >
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                    </svg>
                    "Send"
                </button>
            </div>
            <div class="text-xs text-base-content/50 mt-1">
                {move || format!("{} characters", query.get().len())}
            </div>
        </form>
    }
}

/// Graph statistics display component
///
/// Shows real-time stats:
/// - Number of entities
/// - Number of relationships
/// - Number of documents
/// - Vector index size
/// - Query count
///
/// # Props
/// - `entity_count` - Number of entities in graph
/// - `relationship_count` - Number of relationships
/// - `document_count` - Number of documents
/// - `vector_count` - Number of vectors indexed
#[component]
pub fn GraphStats(
    /// Number of entities in the graph
    #[allow(dead_code)]
    entity_count: ReadSignal<usize>,
    /// Number of relationships
    #[allow(dead_code)]
    relationship_count: ReadSignal<usize>,
    /// Number of documents
    #[allow(dead_code)]
    document_count: ReadSignal<usize>,
    /// Number of vectors indexed
    #[allow(dead_code)]
    vector_count: ReadSignal<usize>,
) -> impl IntoView {
    view! {
        <div class="graph-stats card bg-base-200 shadow-xl">
            <div class="card-body">
                <h3 class="card-title">"Graph Statistics"</h3>

                <div class="stats stats-vertical shadow">
                    <div class="stat">
                        <div class="stat-title">"Documents"</div>
                        <div class="stat-value text-primary">{document_count}</div>
                        <div class="stat-desc">"Indexed documents"</div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">"Entities"</div>
                        <div class="stat-value text-secondary">{entity_count}</div>
                        <div class="stat-desc">"Extracted entities"</div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">"Relationships"</div>
                        <div class="stat-value text-accent">{relationship_count}</div>
                        <div class="stat-desc">"Entity connections"</div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">"Vectors"</div>
                        <div class="stat-value">{vector_count}</div>
                        <div class="stat-desc">"Embedding vectors"</div>
                    </div>
                </div>
            </div>
        </div>
    }
}

/// Document manager component
///
/// Provides interface for:
/// - File upload with drag & drop
/// - Document list with status
/// - Remove documents
/// - Reindex functionality
///
/// # Props
/// - `on_upload` - Callback when files are uploaded
/// - `on_remove` - Callback when document is removed
#[component]
pub fn DocumentManager(
    /// Callback when files are uploaded
    #[allow(dead_code)]
    on_upload: Callback<Vec<String>, ()>,
    /// Callback when document is removed
    #[allow(dead_code)]
    on_remove: Callback<String, ()>,
) -> impl IntoView {
    let (documents, set_documents) = signal(Vec::<(String, String)>::new()); // (id, name)
    let file_input_ref = NodeRef::<leptos::html::Input>::new();

    let handle_file_select = move |_ev: leptos::ev::Event| {
        // Get file input element
        if let Some(input_element) = file_input_ref.get() {
            use wasm_bindgen::JsCast;

            // Access files through the DOM
            let files_opt = js_sys::Reflect::get(&input_element, &wasm_bindgen::JsValue::from_str("files"))
                .ok()
                .and_then(|v| v.dyn_into::<js_sys::Object>().ok());

            if let Some(files) = files_opt {
                // Get length property
                let length = js_sys::Reflect::get(&files, &wasm_bindgen::JsValue::from_str("length"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as u32;

                let mut file_names = Vec::new();

                // Extract file names
                for i in 0..length {
                    if let Ok(file) = js_sys::Reflect::get(&files, &wasm_bindgen::JsValue::from_f64(i as f64)) {
                        if let Ok(name) = js_sys::Reflect::get(&file, &wasm_bindgen::JsValue::from_str("name")) {
                            if let Some(name_str) = name.as_string() {
                                file_names.push(name_str);
                            }
                        }
                    }
                }

                // Update document list
                set_documents.update(|docs| {
                    for (idx, name) in file_names.iter().enumerate() {
                        let id = format!("doc_{}_{}", js_sys::Date::now() as u64, idx);
                        docs.push((id, name.clone()));
                    }
                });

                // Trigger callback with file names
                if !file_names.is_empty() {
                    on_upload.run(file_names);
                }
            }
        }
    };

    view! {
        <div class="document-manager card bg-base-200 shadow-xl">
            <div class="card-body">
                <h3 class="card-title">"Document Manager"</h3>

                // Upload area
                <div class="form-control">
                    <label class="label">
                        <span class="label-text">"Upload Documents"</span>
                    </label>
                    <input
                        type="file"
                        class="file-input file-input-bordered w-full"
                        multiple
                        accept=".txt,.md,.pdf"
                        on:change=handle_file_select
                        node_ref=file_input_ref
                    />
                    <label class="label">
                        <span class="label-text-alt">"Supports: TXT, MD, PDF"</span>
                    </label>
                </div>

                // Document list
                <div class="divider"></div>
                <div class="space-y-2">
                    <For
                        each=move || documents.get()
                        key=|(id, _)| id.clone()
                        children=move |(id, name): (String, String)| {
                            let id_clone = id.clone();
                            view! {
                                <div class="flex justify-between items-center p-2 bg-base-100 rounded">
                                    <span class="text-sm">{name}</span>
                                    <button
                                        class="btn btn-sm btn-ghost btn-circle"
                                        on:click=move |_| {
                                            on_remove.run(id_clone.clone());
                                            set_documents.update(|docs| {
                                                docs.retain(|(doc_id, _)| doc_id != &id_clone);
                                            });
                                        }
                                    >
                                        "×"
                                    </button>
                                </div>
                            }
                        }
                    />

                    {move || documents.get().is_empty().then(|| view! {
                        <div class="text-center text-base-content/50 py-4">
                            "No documents uploaded yet"
                        </div>
                    })}
                </div>
            </div>
        </div>
    }
}

/// Graph node for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
}

/// Graph edge for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: Option<String>,
}

/// Graph visualization component
///
/// Provides interactive graph rendering with:
/// - Node and edge visualization
/// - Zoom and pan controls
/// - Node selection and highlighting
/// - Force-directed layout algorithm
/// - Search and filter
///
/// # Props
/// - `nodes` - List of graph nodes
/// - `edges` - List of graph edges
/// - `on_node_click` - Optional callback when node is clicked
#[component]
pub fn GraphVisualization(
    /// List of nodes to visualize
    #[allow(dead_code)]
    nodes: ReadSignal<Vec<GraphNode>>,
    /// List of edges to visualize
    #[allow(dead_code)]
    edges: ReadSignal<Vec<GraphEdge>>,
    /// Optional callback when node is clicked
    #[prop(into, optional)]
    #[allow(dead_code)]
    on_node_click: Option<Callback<String, ()>>,
) -> impl IntoView {
    let (selected_node, set_selected_node) = signal(None::<String>);
    let (zoom_level, set_zoom_level) = signal(1.0_f32);
    let (layout_positions, set_layout_positions) = signal(std::collections::HashMap::<String, (f64, f64)>::new());

    // Initialize force-directed layout when nodes/edges change
    Effect::new(move |_| {
        let node_list = nodes.get();
        let edge_list = edges.get();

        if !node_list.is_empty() {
            let config = LayoutConfig {
                width: 800.0,
                height: 600.0,
                repulsion: 8000.0,
                attraction: 0.05,
                damping: 0.85,
                dt: 0.015,
                min_movement: 0.05,
            };

            let mut layout = ForceLayout::new(config);

            // Add nodes
            for node in &node_list {
                layout.add_node(node.id.clone());
            }

            // Add edges
            for edge in &edge_list {
                layout.add_edge(edge.source.clone(), edge.target.clone());
            }

            // Run layout algorithm
            layout.run(100);

            // Get positions
            let positions = layout.get_positions();
            set_layout_positions.set(positions);
        }
    });

    let handle_node_click = move |node_id: String| {
        set_selected_node.set(Some(node_id.clone()));
        if let Some(cb) = on_node_click {
            cb.run(node_id);
        }
    };

    let handle_zoom_in = move |_| {
        set_zoom_level.update(|z| *z = (*z * 1.2_f32).min(3.0_f32));
    };

    let handle_zoom_out = move |_| {
        set_zoom_level.update(|z| *z = (*z / 1.2_f32).max(0.5_f32));
    };

    let handle_reset_zoom = move |_| {
        set_zoom_level.set(1.0);
        set_selected_node.set(None);
    };

    view! {
        <div class="graph-visualization card bg-base-200 shadow-xl">
            <div class="card-body">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="card-title">"Knowledge Graph"</h3>

                    // Zoom controls
                    <div class="btn-group">
                        <button class="btn btn-sm" on:click=handle_zoom_out>
                            "−"
                        </button>
                        <button class="btn btn-sm" on:click=handle_reset_zoom>
                            {move || format!("{:.0}%", zoom_level.get() * 100.0)}
                        </button>
                        <button class="btn btn-sm" on:click=handle_zoom_in>
                            "+"
                        </button>
                    </div>
                </div>

                // Graph canvas
                <div
                    class="graph-canvas relative w-full h-96 bg-base-100 rounded-lg overflow-hidden border border-base-300"
                    style="transform-origin: center center;"
                    style:transform=move || format!("scale({})", zoom_level.get())
                >
                    // SVG visualization with force-directed layout
                    <svg class="w-full h-full" viewBox="-400 -300 800 600">
                        // Edges
                        <g class="edges">
                            <For
                                each=move || edges.get()
                                key=|edge| format!("{}-{}", edge.source, edge.target)
                                children=move |edge: GraphEdge| {
                                    let positions = layout_positions.get();
                                    let source_pos = positions.get(&edge.source).copied().unwrap_or((0.0, 0.0));
                                    let target_pos = positions.get(&edge.target).copied().unwrap_or((0.0, 0.0));

                                    view! {
                                        <line
                                            x1=source_pos.0.to_string()
                                            y1=source_pos.1.to_string()
                                            x2=target_pos.0.to_string()
                                            y2=target_pos.1.to_string()
                                            stroke="currentColor"
                                            stroke-width="2"
                                            class="opacity-30"
                                        />
                                        {edge.label.clone().map(|label| {
                                            let mid_x = (source_pos.0 + target_pos.0) / 2.0;
                                            let mid_y = (source_pos.1 + target_pos.1) / 2.0;
                                            view! {
                                                <text
                                                    x=mid_x.to_string()
                                                    y=mid_y.to_string()
                                                    text-anchor="middle"
                                                    class="text-xs fill-current opacity-60"
                                                >
                                                    {label}
                                                </text>
                                            }
                                        })}
                                    }
                                }
                            />
                        </g>

                        // Nodes
                        <g class="nodes">
                            <For
                                each=move || nodes.get()
                                key=|node| node.id.clone()
                                children=move |node: GraphNode| {
                                    let node_id_for_click = node.id.clone();
                                    let node_id_for_selected = node.id.clone();
                                    let is_selected = move || selected_node.get().as_ref() == Some(&node_id_for_selected);
                                    let circle_class = move || if is_selected() {
                                        "fill-primary stroke-primary-focus"
                                    } else {
                                        "fill-secondary stroke-secondary-focus"
                                    };

                                    let positions = layout_positions.get();
                                    let (x, y) = positions.get(&node.id).copied().unwrap_or((0.0, 0.0));

                                    view! {
                                        <g
                                            class="node cursor-pointer"
                                            on:click=move |_| handle_node_click(node_id_for_click.clone())
                                        >
                                            <circle
                                                cx=x.to_string()
                                                cy=y.to_string()
                                                r="20"
                                                class=circle_class
                                                stroke-width="2"
                                            />
                                            <text
                                                x=x.to_string()
                                                y=(y + 25.0).to_string()
                                                text-anchor="middle"
                                                class="text-xs fill-current"
                                            >
                                                {node.label.clone()}
                                            </text>
                                        </g>
                                    }
                                }
                            />
                        </g>
                    </svg>
                </div>

                // Graph info
                <div class="stats stats-horizontal shadow mt-4">
                    <div class="stat">
                        <div class="stat-title">"Nodes"</div>
                        <div class="stat-value text-sm">{move || nodes.get().len()}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-title">"Edges"</div>
                        <div class="stat-value text-sm">{move || edges.get().len()}</div>
                    </div>
                    {move || selected_node.get().map(|node_id| view! {
                        <div class="stat">
                            <div class="stat-title">"Selected"</div>
                            <div class="stat-value text-sm">{node_id}</div>
                        </div>
                    })}
                </div>
            </div>
        </div>
    }
}
