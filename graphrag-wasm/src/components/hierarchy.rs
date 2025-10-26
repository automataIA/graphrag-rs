//! # Hierarchical GraphRAG UI Components
//!
//! Components for exploring hierarchical community structures detected by the Leiden algorithm.
//!
//! ## Components
//!
//! - `<HierarchyExplorer/>`: Main hierarchy navigation interface
//! - `<CommunityCard/>`: Display individual community information
//! - `<LevelSelector/>`: Dropdown for selecting hierarchical levels
//! - `<AdaptiveQueryPanel/>`: Query interface with adaptive routing

#![allow(dead_code)]

use leptos::prelude::*;
use leptos::ev::SubmitEvent;
use serde::{Deserialize, Serialize};

/// Community data structure for UI display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityData {
    pub id: usize,
    pub level: usize,
    pub entity_count: usize,
    pub summary: String,
    pub entities: Vec<String>,
}

/// Query analysis result from adaptive routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysisResult {
    pub suggested_level: usize,
    pub keyword_score: f32,
    pub length_score: f32,
    pub entity_score: f32,
}

/// Query result with community information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub level: usize,
    pub community_id: usize,
    pub summary: String,
}

/// Level Selector Component
///
/// Provides a dropdown to select hierarchical levels with statistics.
///
/// # Props
/// - `max_level` - Maximum level available (e.g., 3 means levels 0, 1, 2)
/// - `current_level` - Currently selected level
/// - `on_level_change` - Callback when level is changed
#[component]
pub fn LevelSelector(
    /// Maximum hierarchical level available
    max_level: ReadSignal<usize>,
    /// Currently selected level
    current_level: ReadSignal<usize>,
    /// Callback when level changes
    on_level_change: Callback<usize, ()>,
) -> impl IntoView {
    view! {
        <div class="level-selector flex items-center gap-3">
            <label class="label">
                <span class="label-text font-semibold">"Hierarchical Level:"</span>
            </label>

            <div class="btn-group">
                <For
                    each=move || {
                        let max = max_level.get();
                        (0..max).collect::<Vec<_>>()
                    }
                    key=|level| *level
                    children=move |level: usize| {
                        let is_selected = move || current_level.get() == level;
                        let btn_class = move || if is_selected() {
                            "btn btn-sm btn-primary"
                        } else {
                            "btn btn-sm btn-ghost"
                        };

                        view! {
                            <button
                                class=btn_class
                                on:click=move |_| on_level_change.run(level)
                            >
                                {format!("L{}", level)}
                            </button>
                        }
                    }
                />
            </div>

            <div class="badge badge-info badge-outline">
                {move || {
                    let level = current_level.get();
                    match level {
                        0 => "Finest detail",
                        1 => "Medium detail",
                        _ => "High-level overview",
                    }
                }}
            </div>
        </div>
    }
}

/// Community Card Component
///
/// Displays information about a single community with expandable details.
///
/// # Props
/// - `community` - Community data to display
/// - `on_expand` - Optional callback when card is expanded
#[component]
pub fn CommunityCard(
    /// Community data
    community: ReadSignal<CommunityData>,
    /// Optional callback when expanded
    #[prop(optional)]
    on_expand: Option<Callback<usize, ()>>,
) -> impl IntoView {
    let (is_expanded, set_is_expanded) = signal(false);

    let toggle_expand = move |_| {
        let new_state = !is_expanded.get();
        set_is_expanded.set(new_state);
        if new_state {
            if let Some(cb) = on_expand {
                cb.run(community.get().id);
            }
        }
    };

    view! {
        <div class="community-card card bg-base-100 shadow-md hover:shadow-lg transition-shadow">
            <div class="card-body p-4">
                // Header
                <div class="flex justify-between items-start">
                    <div>
                        <h4 class="card-title text-sm">
                            {move || format!("Community {}", community.get().id)}
                        </h4>
                        <div class="badge badge-sm badge-secondary mt-1">
                            {move || format!("Level {}", community.get().level)}
                        </div>
                    </div>

                    <div class="stat-value text-sm text-primary">
                        {move || community.get().entity_count}
                    </div>
                </div>

                // Summary preview
                <div class="text-xs text-base-content/70 mt-2">
                    {move || {
                        let summary = community.get().summary;
                        if summary.len() > 100 && !is_expanded.get() {
                            format!("{}...", &summary[..100])
                        } else {
                            summary
                        }
                    }}
                </div>

                // Entities list (when expanded)
                <Show when=move || is_expanded.get()>
                    <div class="divider my-2"></div>
                    <div class="space-y-1">
                        <p class="text-xs font-semibold text-base-content/60">"Entities:"</p>
                        <div class="flex flex-wrap gap-1">
                            <For
                                each=move || community.get().entities
                                key=|entity| entity.clone()
                                children=move |entity: String| {
                                    view! {
                                        <span class="badge badge-sm badge-outline">
                                            {entity}
                                        </span>
                                    }
                                }
                            />
                        </div>
                    </div>
                </Show>

                // Expand button
                <div class="card-actions justify-end mt-2">
                    <button
                        class="btn btn-xs btn-ghost"
                        on:click=toggle_expand
                    >
                        {move || if is_expanded.get() { "Collapse ▲" } else { "Expand ▼" }}
                    </button>
                </div>
            </div>
        </div>
    }
}

/// Hierarchy Explorer Component
///
/// Main interface for exploring hierarchical community structures.
///
/// # Props
/// - `max_level` - Maximum hierarchical level
/// - `on_level_change` - Callback when level is changed
/// - `on_detect_communities` - Callback to trigger community detection
#[component]
pub fn HierarchyExplorer(
    /// Maximum hierarchical level available
    max_level: ReadSignal<usize>,
    /// Communities at current level
    communities: ReadSignal<Vec<CommunityData>>,
    /// Callback when level changes
    on_level_change: Callback<usize, ()>,
    /// Callback to detect communities
    on_detect_communities: Callback<(), ()>,
) -> impl IntoView {
    let (current_level, set_current_level) = signal(0_usize);
    let (is_detecting, set_is_detecting) = signal(false);

    let handle_level_change = move |level: usize| {
        set_current_level.set(level);
        on_level_change.run(level);
    };

    let handle_detect = move |_| {
        set_is_detecting.set(true);
        on_detect_communities.run(());
        set_is_detecting.set(false);
    };

    view! {
        <div class="hierarchy-explorer space-y-4">
            // Header
            <div class="flex justify-between items-center">
                <h2 class="text-2xl font-bold">"Hierarchical Communities"</h2>

                <button
                    class="btn btn-primary btn-sm"
                    on:click=handle_detect
                    disabled=move || is_detecting.get()
                >
                    <Show
                        when=move || is_detecting.get()
                        fallback=|| view! { "Detect Communities" }
                    >
                        <span class="loading loading-spinner loading-sm"></span>
                    </Show>
                </button>
            </div>

            // Level selector
            <Show when=move || { max_level.get() > 0 }>
                <div class="card bg-base-200 shadow">
                    <div class="card-body p-4">
                        <LevelSelector
                            max_level=max_level
                            current_level=current_level
                            on_level_change=Callback::new(handle_level_change)
                        />

                        <div class="text-xs text-base-content/60 mt-2">
                            {move || {
                                let level = current_level.get();
                                match level {
                                    0 => "Level 0: Finest granularity - Most detailed communities with specific entity relationships",
                                    1 => "Level 1: Medium granularity - Balanced view of community structures",
                                    2 => "Level 2: High-level overview - Broad themes and major entity groups",
                                    _ => "Level 3+: Very broad overview - Highest-level abstractions",
                                }
                            }}
                        </div>
                    </div>
                </div>
            </Show>

            // Communities grid
            <Show
                when=move || !communities.get().is_empty()
                fallback=move || view! {
                    <div class="card bg-base-200 shadow">
                        <div class="card-body text-center">
                            <div class="text-base-content/50">
                                {move || if max_level.get() == 0 {
                                    "No communities detected yet. Click 'Detect Communities' to analyze your knowledge graph."
                                } else {
                                    "No communities at this level."
                                }}
                            </div>
                        </div>
                    </div>
                }
            >
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <For
                        each=move || communities.get()
                        key=|comm| comm.id
                        children=move |comm: CommunityData| {
                            let comm_signal = RwSignal::new(comm);
                            view! {
                                <CommunityCard community=comm_signal.read_only() />
                            }
                        }
                    />
                </div>

                <div class="stats shadow mt-4">
                    <div class="stat">
                        <div class="stat-title">"Communities at Level " {current_level}</div>
                        <div class="stat-value text-2xl">{move || communities.get().len()}</div>
                        <div class="stat-desc">
                            {move || format!("{} total entities",
                                communities.get().iter().map(|c| c.entity_count).sum::<usize>()
                            )}
                        </div>
                    </div>
                </div>
            </Show>
        </div>
    }
}

/// Adaptive Query Panel Component
///
/// Enhanced query interface with adaptive routing that shows query complexity analysis.
///
/// # Props
/// - `on_query` - Callback when query is submitted
/// - `on_manual_level` - Callback when user overrides automatic level selection
#[component]
pub fn AdaptiveQueryPanel(
    /// Callback when query is submitted (adaptive routing)
    on_query: Callback<String, ()>,
    /// Optional callback when user manually selects level
    #[prop(optional)]
    on_manual_level: Option<Callback<(String, usize), ()>>,
) -> impl IntoView {
    let (query, set_query) = signal(String::new());
    let (use_adaptive, set_use_adaptive) = signal(true);
    let (manual_level, set_manual_level) = signal(0_usize);
    let (analysis, _set_analysis) = signal(None::<QueryAnalysisResult>);
    let (is_querying, set_is_querying) = signal(false);

    let handle_submit = move |ev: SubmitEvent| {
        ev.prevent_default();
        let query_text = query.get().trim().to_string();

        if !query_text.is_empty() {
            set_is_querying.set(true);

            if use_adaptive.get() {
                on_query.run(query_text.clone());
            } else {
                if let Some(cb) = on_manual_level {
                    cb.run((query_text.clone(), manual_level.get()));
                }
            }

            set_query.set(String::new());
            set_is_querying.set(false);
        }
    };

    view! {
        <div class="adaptive-query-panel card bg-base-200 shadow-xl">
            <div class="card-body">
                <h3 class="card-title">"Adaptive Query"</h3>

                <form on:submit=handle_submit class="space-y-4">
                    // Query input
                    <div class="form-control">
                        <label class="label">
                            <span class="label-text">"Ask a question"</span>
                        </label>
                        <textarea
                            class="textarea textarea-bordered h-24"
                            placeholder="e.g., 'Overview of AI technologies' or 'Relationship between GPT and Transformers'"
                            prop:value=query
                            on:input=move |ev| set_query.set(event_target_value(&ev))
                            disabled=move || is_querying.get()
                        />
                        <label class="label">
                            <span class="label-text-alt">
                                {move || format!("{} characters", query.get().len())}
                            </span>
                        </label>
                    </div>

                    // Adaptive routing toggle
                    <div class="form-control">
                        <label class="label cursor-pointer justify-start gap-3">
                            <input
                                type="checkbox"
                                class="checkbox checkbox-primary"
                                prop:checked=use_adaptive
                                on:change=move |ev| set_use_adaptive.set(event_target_checked(&ev))
                            />
                            <span class="label-text">"Use Adaptive Routing (automatic level selection)"</span>
                        </label>
                    </div>

                    // Manual level selection (when adaptive is off)
                    <Show when=move || !use_adaptive.get()>
                        <div class="form-control">
                            <label class="label">
                                <span class="label-text">"Select Level Manually"</span>
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="3"
                                prop:value=move || manual_level.get()
                                on:input=move |ev| {
                                    if let Ok(val) = event_target_value(&ev).parse::<usize>() {
                                        set_manual_level.set(val);
                                    }
                                }
                                class="range range-primary"
                            />
                            <div class="w-full flex justify-between text-xs px-2 mt-1">
                                <span>"L0 (Detailed)"</span>
                                <span>"L1"</span>
                                <span>"L2"</span>
                                <span>"L3 (Broad)"</span>
                            </div>
                        </div>
                    </Show>

                    // Query analysis display (when available)
                    <Show when=move || analysis.get().is_some()>
                        {move || analysis.get().map(|a| view! {
                            <div class="alert alert-info">
                                <div class="flex-1">
                                    <div class="font-semibold">
                                        "Query Analysis: Suggested Level " {a.suggested_level}
                                    </div>
                                    <div class="text-xs mt-1 space-y-1">
                                        <div>"Keyword score: " {format!("{:.2}", a.keyword_score)}</div>
                                        <div>"Length score: " {format!("{:.2}", a.length_score)}</div>
                                        <div>"Entity score: " {format!("{:.2}", a.entity_score)}</div>
                                    </div>
                                </div>
                            </div>
                        })}
                    </Show>

                    // Submit button
                    <div class="form-control">
                        <button
                            type="submit"
                            class="btn btn-primary"
                            disabled=move || query.get().trim().is_empty() || is_querying.get()
                        >
                            <Show
                                when=move || is_querying.get()
                                fallback=|| view! { "Submit Query" }
                            >
                                <span class="loading loading-spinner loading-sm"></span>
                                " Querying..."
                            </Show>
                        </button>
                    </div>
                </form>

                // Help text
                <div class="divider"></div>
                <div class="text-xs text-base-content/60">
                    <p class="font-semibold mb-1">"Adaptive Routing Tips:"</p>
                    <ul class="list-disc list-inside space-y-1">
                        <li>"Broad queries (overview, summary) → Higher levels"</li>
                        <li>"Specific queries (relationships, details) → Lower levels"</li>
                        <li>"Short queries (1-3 words) tend to be broad"</li>
                        <li>"Long queries with multiple entities tend to be specific"</li>
                    </ul>
                </div>
            </div>
        </div>
    }
}
