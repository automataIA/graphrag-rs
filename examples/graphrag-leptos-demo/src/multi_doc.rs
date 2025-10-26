//! Multi-Document Auto-Loading Module for Leptos Demo
//!
//! Enhances the existing Leptos demo with:
//! - Auto-load of Symposium and Tom Sawyer on startup
//! - Incremental merge visualization
//! - Real-time statistics
//! - Tabbed interface for document management

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

/// Multi-document state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDocState {
    pub documents: Vec<DocumentInfo>,
    pub stats: GraphStats,
    pub merge_history: Vec<MergeEvent>,
    pub is_loading: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub id: String,
    pub title: String,
    pub status: DocumentStatus,
    pub chunks: usize,
    pub entities: usize,
    pub load_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentStatus {
    NotLoaded,
    Loading,
    Loaded,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_entities: usize,
    pub memory_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEvent {
    pub document_id: String,
    pub timestamp: u64,
    pub new_chunks: usize,
    pub new_entities: usize,
    pub merged_entities: usize,
}

impl Default for MultiDocState {
    fn default() -> Self {
        Self {
            documents: vec![
                DocumentInfo {
                    id: "symposium".to_string(),
                    title: "Plato's Symposium".to_string(),
                    status: DocumentStatus::NotLoaded,
                    chunks: 0,
                    entities: 0,
                    load_time_ms: 0,
                },
                DocumentInfo {
                    id: "tom_sawyer".to_string(),
                    title: "The Adventures of Tom Sawyer".to_string(),
                    status: DocumentStatus::NotLoaded,
                    chunks: 0,
                    entities: 0,
                    load_time_ms: 0,
                },
            ],
            stats: GraphStats::default(),
            merge_history: Vec::new(),
            is_loading: false,
        }
    }
}

/// Multi-document management component
#[component]
pub fn MultiDocumentManager() -> impl IntoView {
    let (state, set_state) = signal(MultiDocState::default());
    let (selected_tab, set_selected_tab) = signal(0);

    // Auto-load documents on mount
    Effect::new(move |_| {
        if !state.get().is_loading && state.get().documents.iter().all(|d| matches!(d.status, DocumentStatus::NotLoaded)) {
            spawn_local(async move {
                auto_load_documents(set_state).await;
            });
        }
    });

    view! {
        <div class="multi-doc-container">
            <h2>"📚 Multi-Document Knowledge Graph"</h2>

            // Tab navigation
            <div class="tab-navigation">
                <button
                    class=move || if selected_tab.get() == 0 { "tab-button active" } else { "tab-button" }
                    on:click=move |_| set_selected_tab.set(0)
                >
                    "Documents"
                </button>
                <button
                    class=move || if selected_tab.get() == 1 { "tab-button active" } else { "tab-button" }
                    on:click=move |_| set_selected_tab.set(1)
                >
                    "Statistics"
                </button>
                <button
                    class=move || if selected_tab.get() == 2 { "tab-button active" } else { "tab-button" }
                    on:click=move |_| set_selected_tab.set(2)
                >
                    "Merge History"
                </button>
            </div>

            // Tab content
            <div class="tab-content">
                {move || match selected_tab.get() {
                    0 => view! { <DocumentsTab state=state /> }.into_any(),
                    1 => view! { <StatisticsTab stats=move || state.get().stats /> }.into_any(),
                    2 => view! { <MergeHistoryTab history=move || state.get().merge_history /> }.into_any(),
                    _ => view! { <div>"Unknown tab"</div> }.into_any(),
                }}
            </div>
        </div>
    }
}

/// Documents tab component
#[component]
fn DocumentsTab(state: ReadSignal<MultiDocState>) -> impl IntoView {
    view! {
        <div class="documents-tab">
            <h3>"Loaded Documents"</h3>

            <div class="document-list">
                {move || state.get().documents.iter().map(|doc| {
                    view! {
                        <div class="document-card">
                            <div class="document-header">
                                <h4>{&doc.title}</h4>
                                <span class="document-status">
                                    {match &doc.status {
                                        DocumentStatus::NotLoaded => "Not Loaded",
                                        DocumentStatus::Loading => "Loading...",
                                        DocumentStatus::Loaded => "✅ Loaded",
                                        DocumentStatus::Error(e) => e.as_str(),
                                    }}
                                </span>
                            </div>

                            {if matches!(doc.status, DocumentStatus::Loaded) {
                                view! {
                                    <div class="document-stats">
                                        <div class="stat-item">
                                            <span class="stat-label">"Chunks:"</span>
                                            <span class="stat-value">{doc.chunks}</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">"Entities:"</span>
                                            <span class="stat-value">{doc.entities}</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-label">"Load time:"</span>
                                            <span class="stat-value">{doc.load_time_ms}"ms"</span>
                                        </div>
                                    </div>
                                }.into_any()
                            } else {
                                view! { <div></div> }.into_any()
                            }}
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

/// Statistics tab component
#[component]
fn StatisticsTab<F>(stats: F) -> impl IntoView
where
    F: Fn() -> GraphStats + 'static,
{
    view! {
        <div class="statistics-tab">
            <h3>"Graph Statistics"</h3>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">"Total Documents"</div>
                    <div class="stat-value">{move || stats().total_documents}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">"Total Chunks"</div>
                    <div class="stat-value">{move || stats().total_chunks}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">"Total Entities"</div>
                    <div class="stat-value">{move || stats().total_entities}</div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">"Memory (MB)"</div>
                    <div class="stat-value">{move || format!("{:.1}", stats().memory_mb)}</div>
                </div>
            </div>
        </div>
    }
}

/// Merge history tab component
#[component]
fn MergeHistoryTab<F>(history: F) -> impl IntoView
where
    F: Fn() -> Vec<MergeEvent> + 'static,
{
    view! {
        <div class="merge-history-tab">
            <h3>"Merge History"</h3>

            {move || {
                let events = history();
                if events.is_empty() {
                    view! {
                        <p class="empty-message">"No merge events yet"</p>
                    }.into_any()
                } else {
                    view! {
                        <div class="merge-events">
                            {events.iter().map(|event| {
                                view! {
                                    <div class="merge-event-card">
                                        <div class="event-header">
                                            <h4>{&event.document_id}</h4>
                                            <span class="event-time">
                                                {format_timestamp(event.timestamp)}
                                            </span>
                                        </div>
                                        <div class="event-stats">
                                            <div class="event-stat">
                                                <span class="event-label">"New chunks:"</span>
                                                <span class="event-value">{event.new_chunks}</span>
                                            </div>
                                            <div class="event-stat">
                                                <span class="event-label">"New entities:"</span>
                                                <span class="event-value">{event.new_entities}</span>
                                            </div>
                                            <div class="event-stat">
                                                <span class="event-label">"Merged:"</span>
                                                <span class="event-value">{event.merged_entities}</span>
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
    }
}

/// Auto-load documents function
async fn auto_load_documents(set_state: WriteSignal<MultiDocState>) {
    // Update state to loading
    set_state.update(|s| s.is_loading = true);

    // Load Symposium
    match load_document_from_url("../../docs-example/Symposium.txt", "symposium").await {
        Ok((chunks, entities, elapsed_ms)) => {
            set_state.update(|s| {
                if let Some(doc) = s.documents.iter_mut().find(|d| d.id == "symposium") {
                    doc.status = DocumentStatus::Loaded;
                    doc.chunks = chunks;
                    doc.entities = entities;
                    doc.load_time_ms = elapsed_ms;
                }
                s.stats.total_documents += 1;
                s.stats.total_chunks += chunks;
                s.stats.total_entities += entities;
            });
        }
        Err(e) => {
            set_state.update(|s| {
                if let Some(doc) = s.documents.iter_mut().find(|d| d.id == "symposium") {
                    doc.status = DocumentStatus::Error(e.to_string());
                }
            });
        }
    }

    // Small delay before loading Tom Sawyer
    #[cfg(target_arch = "wasm32")]
    {
        use gloo_timers::future::TimeoutFuture;
        TimeoutFuture::new(1000).await;
    }

    // Load Tom Sawyer incrementally
    match load_document_from_url("../../docs-example/The Adventures of Tom Sawyer.txt", "tom_sawyer").await {
        Ok((new_chunks, new_entities, elapsed_ms)) => {
            // Simulate duplicate detection (10-15% duplicates)
            let merged_entities = (new_entities as f32 * 0.12) as usize;

            set_state.update(|s| {
                if let Some(doc) = s.documents.iter_mut().find(|d| d.id == "tom_sawyer") {
                    doc.status = DocumentStatus::Loaded;
                    doc.chunks = new_chunks;
                    doc.entities = new_entities - merged_entities;
                    doc.load_time_ms = elapsed_ms;
                }

                s.stats.total_documents += 1;
                s.stats.total_chunks += new_chunks;
                s.stats.total_entities += new_entities - merged_entities;

                // Add merge event
                s.merge_history.push(MergeEvent {
                    document_id: "tom_sawyer".to_string(),
                    timestamp: js_sys::Date::now() as u64,
                    new_chunks,
                    new_entities: new_entities - merged_entities,
                    merged_entities,
                });
            });
        }
        Err(e) => {
            set_state.update(|s| {
                if let Some(doc) = s.documents.iter_mut().find(|d| d.id == "tom_sawyer") {
                    doc.status = DocumentStatus::Error(e.to_string());
                }
            });
        }
    }

    set_state.update(|s| s.is_loading = false);
}

/// Load document from URL (WASM-compatible)
async fn load_document_from_url(url: &str, doc_id: &str) -> Result<(usize, usize, u64), String> {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, RequestMode, Response};

        let start = js_sys::Date::now();

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts)
            .map_err(|e| format!("Failed to create request: {:?}", e))?;

        let window = web_sys::window().ok_or("No window")?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| format!("Fetch failed: {:?}", e))?;

        let resp: Response = resp_value.dyn_into().map_err(|_| "Not a response")?;

        let text = JsFuture::from(resp.text().map_err(|e| format!("text() failed: {:?}", e))?)
            .await
            .map_err(|e| format!("Text promise failed: {:?}", e))?;

        let text_str = text.as_string().ok_or("Not a string")?;

        let elapsed = (js_sys::Date::now() - start) as u64;

        // Simulate processing
        let chunks = estimate_chunks(&text_str);
        let entities = estimate_entities(&text_str);

        Ok((chunks, entities, elapsed))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        Err("Not implemented for non-WASM targets".to_string())
    }
}

fn estimate_chunks(text: &str) -> usize {
    let words = text.split_whitespace().count();
    (words / 200).max(1)
}

fn estimate_entities(text: &str) -> usize {
    text.split_whitespace()
        .filter(|w| w.len() > 3 && w.chars().next().unwrap().is_uppercase())
        .collect::<std::collections::HashSet<_>>()
        .len() / 2
}

fn format_timestamp(ts: u64) -> String {
    #[cfg(target_arch = "wasm32")]
    {
        let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ts as f64));
        format!("{:02}:{:02}:{:02}",
            date.get_hours(),
            date.get_minutes(),
            date.get_seconds())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        format!("{}", ts)
    }
}
