//! Leptos UI Components for GraphRAG WASM
//!
//! This module provides ready-to-use reactive Leptos components for building
//! GraphRAG user interfaces in the browser.

pub mod force_layout;
pub mod hierarchy;
pub mod settings;
pub mod ui_components;

// Re-export settings components
pub use settings::SettingsPanel;

// Re-export UI components
#[allow(unused_imports)]
pub use ui_components::{
    ChatMessage, ChatWindow, DocumentManager, GraphEdge, GraphNode, GraphStats, GraphVisualization,
    MessageRole, QueryInterface,
};

// Re-export hierarchy components
#[allow(unused_imports)]
pub use hierarchy::{
    AdaptiveQueryPanel, CommunityCard, CommunityData, HierarchyExplorer, LevelSelector,
    QueryAnalysisResult, QueryResult,
};
