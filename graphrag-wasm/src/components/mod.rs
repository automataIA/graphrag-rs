//! Leptos UI Components for GraphRAG WASM
//!
//! This module provides ready-to-use reactive Leptos components for building
//! GraphRAG user interfaces in the browser.

pub mod force_layout;
pub mod ui_components;
pub mod settings;
pub mod hierarchy;

// Re-export settings components
pub use settings::SettingsPanel;

// Re-export UI components
#[allow(unused_imports)]
pub use ui_components::{
    ChatWindow, QueryInterface, GraphStats, DocumentManager, GraphVisualization,
    ChatMessage, MessageRole, GraphNode, GraphEdge,
};

// Re-export hierarchy components
#[allow(unused_imports)]
pub use hierarchy::{
    HierarchyExplorer, CommunityCard, LevelSelector, AdaptiveQueryPanel,
    CommunityData, QueryAnalysisResult, QueryResult,
};
