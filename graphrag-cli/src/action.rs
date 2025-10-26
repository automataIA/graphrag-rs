//! Action types for event-driven architecture
//!
//! Actions represent all possible events and state changes in the application.
//! They are used to communicate between components and drive the application's
//! event loop.

use std::path::PathBuf;

/// Main action enum for application events
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum Action {
    // ========= Application Lifecycle =========
    /// Periodic tick for animations/updates
    Tick,
    /// Trigger a render
    Render,
    /// Terminal was resized
    Resize(u16, u16),
    /// Quit the application
    Quit,

    // ========= Input Handling =========
    /// Insert a character at cursor
    InputChar(char),
    /// Delete character before cursor
    DeleteChar,
    /// Submit input (Enter key)
    SubmitInput,
    /// Clear all input
    ClearInput,

    // ========= Focus & Navigation =========
    /// Focus the query input widget
    FocusQueryInput,
    /// Focus the results viewer widget
    FocusResultsViewer,
    /// Focus the raw results viewer widget
    FocusRawResultsViewer,
    /// Focus the info panel widget
    FocusInfoPanel,
    /// Move focus to next pane
    NextPane,
    /// Move focus to previous pane
    PreviousPane,

    // ========= Scrolling (Vim-Style) =========
    /// Scroll up one line (k)
    ScrollUp,
    /// Scroll down one line (j)
    ScrollDown,
    /// Scroll up one page (Ctrl+U)
    ScrollPageUp,
    /// Scroll down one page (Ctrl+D)
    ScrollPageDown,
    /// Scroll to top (Home)
    ScrollToTop,
    /// Scroll to bottom (End)
    ScrollToBottom,

    // ========= GraphRAG Operations =========
    /// Load a configuration file (JSON5, JSON, TOML)
    LoadConfig(PathBuf),
    /// Load a document into the knowledge graph
    LoadDocument(PathBuf),
    /// Execute a natural language query
    ExecuteQuery(String),
    /// Execute a slash command
    ExecuteSlashCommand(String),

    // ========= Status Updates =========
    /// Set status message with type
    SetStatus(StatusType, String),
    /// Clear current status
    ClearStatus,
    /// Show progress indicator
    StartProgress(String),
    /// Stop progress indicator
    StopProgress,

    // ========= Help System =========
    /// Toggle help overlay
    ToggleHelp,

    // ========= Async Operation Results =========
    /// Query completed successfully
    QuerySuccess(String),
    /// Query failed with error
    QueryError(String),
    /// Document loaded successfully
    DocumentLoaded(String),
    /// Document load failed
    DocumentLoadError(String),
    /// Configuration loaded successfully
    ConfigLoaded(String),
    /// Configuration load failed
    ConfigLoadError(String),

    // ========= Workspace Operations =========
    /// Switch to a different workspace
    SwitchWorkspace(String),
    /// Refresh workspace statistics
    RefreshStats,

    // ========= No Operation =========
    /// No action to perform
    Noop,
}

/// Status indicator types with associated colors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusType {
    /// Information (blue ℹ)
    Info,
    /// Success (green ✓)
    Success,
    /// Warning (yellow ⚠)
    Warning,
    /// Error (red ✗)
    Error,
    /// Progress (cyan ⟳)
    Progress,
}

impl StatusType {
    /// Get the icon symbol for this status type
    pub fn icon(&self) -> &str {
        match self {
            StatusType::Info => "ℹ",
            StatusType::Success => "✓",
            StatusType::Warning => "⚠",
            StatusType::Error => "✗",
            StatusType::Progress => "⟳",
        }
    }

    /// Get the color for this status type
    pub fn color(&self) -> ratatui::style::Color {
        use ratatui::style::Color;
        match self {
            StatusType::Info => Color::Blue,
            StatusType::Success => Color::Green,
            StatusType::Warning => Color::Yellow,
            StatusType::Error => Color::Red,
            StatusType::Progress => Color::Cyan,
        }
    }
}
