//! Main application logic and event loop

use crate::{
    action::{Action, QueryExplainedPayload, QueryMode, SourceRef, StatusType},
    commands::SlashCommand,
    handlers::{FileOperations, GraphRAGHandler},
    query_history::{QueryEntry, QueryHistory},
    theme::Theme,
    tui::{Event, Tui},
    ui::{HelpOverlay, InfoPanel, QueryInput, RawResultsViewer, ResultsViewer, StatusBar},
    workspace::{WorkspaceManager, WorkspaceMetadata},
};
use chrono::Utc;
use color_eyre::eyre::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout};
use std::{path::PathBuf, time::Instant};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

/// Main application state
pub struct App {
    /// Should the app quit?
    should_quit: bool,
    /// TUI instance
    tui: Tui,
    /// GraphRAG handler
    graphrag: GraphRAGHandler,
    /// Action sender
    action_tx: UnboundedSender<Action>,
    /// Action receiver
    action_rx: UnboundedReceiver<Action>,
    /// Query input component
    query_input: QueryInput,
    /// Results viewer component (LLM-processed answer)
    results_viewer: ResultsViewer,
    /// Raw results viewer component (search results before LLM)
    raw_results_viewer: RawResultsViewer,
    /// Info panel component
    info_panel: InfoPanel,
    /// Status bar component
    status_bar: StatusBar,
    /// Help overlay component
    help_overlay: HelpOverlay,
    /// Query history
    query_history: QueryHistory,
    /// Workspace manager
    #[allow(dead_code)]
    workspace_manager: WorkspaceManager,
    /// Current workspace metadata
    #[allow(dead_code)]
    workspace_metadata: Option<WorkspaceMetadata>,
    /// Configuration file path
    config_path: Option<PathBuf>,
    /// Active query mode (Ask / Explain / Reason)
    query_mode: QueryMode,
    /// Currently focused pane: 0=input, 1=results, 2=raw, 3=info
    focused_pane: u8,
    /// Theme
    #[allow(dead_code)]
    theme: Theme,
}

impl App {
    /// Create a new application
    pub fn new(config_path: Option<PathBuf>, _workspace: Option<String>) -> Result<Self> {
        let (action_tx, action_rx) = mpsc::unbounded_channel();

        let workspace_manager = WorkspaceManager::new()?;

        Ok(Self {
            should_quit: false,
            tui: Tui::new()?,
            graphrag: GraphRAGHandler::new(),
            action_tx,
            action_rx,
            query_input: QueryInput::new(),
            results_viewer: ResultsViewer::new(),
            raw_results_viewer: RawResultsViewer::new(),
            info_panel: InfoPanel::new(),
            status_bar: StatusBar::new(),
            help_overlay: HelpOverlay::new(),
            query_history: QueryHistory::new(),
            workspace_manager,
            workspace_metadata: None,
            config_path,
            query_mode: QueryMode::default(),
            focused_pane: 0,
            theme: Theme::default(),
        })
    }

    /// Set focused pane and update all component focus states.
    /// pane: 0=input, 1=results, 2=raw_results, 3=info_panel
    fn set_focus(&mut self, pane: u8) {
        self.focused_pane = pane;
        self.query_input.set_focused(pane == 0);
        self.results_viewer.set_focused(pane == 1);
        self.raw_results_viewer.set_focused(pane == 2);
        self.info_panel.set_focused(pane == 3);
    }

    /// Run the application
    pub async fn run(&mut self) -> Result<()> {
        // Enter TUI mode
        self.tui.enter()?;

        // Load config if provided
        if let Some(ref config_path) = self.config_path.clone() {
            self.action_tx
                .send(Action::LoadConfig(config_path.clone()))?;
        } else {
            self.action_tx.send(Action::SetStatus(
                StatusType::Warning,
                "No config loaded. Use /config <file> to load configuration".to_string(),
            ))?;
        }

        // Main event loop
        while !self.should_quit {
            // Get next event
            if let Some(event) = self.tui.next().await {
                self.handle_event(event).await?;
            }

            // Process all pending actions
            while let Ok(action) = self.action_rx.try_recv() {
                self.update(action).await?;

                if self.should_quit {
                    break;
                }
            }

            // Render
            let query_input = &mut self.query_input;
            let results_viewer = &mut self.results_viewer;
            let raw_results_viewer = &mut self.raw_results_viewer;
            let info_panel = &mut self.info_panel;
            let status_bar = &mut self.status_bar;
            let help_overlay = &mut self.help_overlay;

            self.tui.terminal.draw(|f| {
                use crate::ui::components::Component;

                // Main vertical layout: Input + Content + Status
                let main_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3), // Query input
                        Constraint::Min(0),    // Content area (results + info)
                        Constraint::Length(3), // Status bar
                    ])
                    .split(f.area());

                // Horizontal split: Left side (70%) and Info Panel (30%)
                let content_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(70), // Left side (results + raw results)
                        Constraint::Percentage(30), // Info panel
                    ])
                    .split(main_chunks[1]);

                // Vertical split for left side: Results Viewer (top) and Raw Results (bottom)
                let left_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(60), // Results viewer (LLM answer)
                        Constraint::Percentage(40), // Raw results viewer (search results)
                    ])
                    .split(content_chunks[0]);

                // Render components
                query_input.render(f, main_chunks[0]);
                results_viewer.render(f, left_chunks[0]);
                raw_results_viewer.render(f, left_chunks[1]);
                info_panel.render(f, content_chunks[1]);
                status_bar.render(f, main_chunks[2]);

                // Help overlay (on top if visible)
                if help_overlay.is_visible() {
                    help_overlay.render(f, f.area());
                }
            })?;
        }

        // Exit TUI mode
        self.tui.exit()?;

        Ok(())
    }

    /// Handle terminal events
    async fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Crossterm(crossterm_event) => {
                if let crossterm::event::Event::Key(key) = crossterm_event {
                    self.handle_key_event(key)?;
                }
            },
            Event::Tick => {
                // Periodic update
            },
            Event::Render => {
                // Render is handled in main loop
            },
            Event::Resize(w, h) => {
                self.action_tx.send(Action::Resize(w, h))?;
            },
        }

        Ok(())
    }

    /// Handle keyboard input
    fn handle_key_event(&mut self, key: KeyEvent) -> Result<()> {
        // Help overlay has priority
        if self.help_overlay.is_visible() {
            if matches!(key.code, KeyCode::Char('?') | KeyCode::Esc) {
                self.action_tx.send(Action::ToggleHelp)?;
            }
            return Ok(());
        }

        // Global shortcuts (check these first, before passing to input)
        match (key.code, key.modifiers) {
            // Quit
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.action_tx.send(Action::Quit)?;
                return Ok(());
            },
            // Help — works regardless of SHIFT state (? is Shift+/ on US keyboards;
            // some terminals send SHIFT, others send NONE). Also map Ctrl+H for safety.
            (KeyCode::Char('?'), _) | (KeyCode::Char('h'), KeyModifiers::CONTROL) => {
                self.action_tx.send(Action::ToggleHelp)?;
                return Ok(());
            },
            // Ctrl+N: cycle focus forward through panels
            // Input(0) → Results(1) → Raw(2) → Info(3) → Input(0)
            // When InfoPanel is focused, Ctrl+N instead cycles its internal tabs
            (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                if self.focused_pane == 3 {
                    // InfoPanel focused: cycle internal tabs
                    self.action_tx.send(Action::NextTab)?;
                } else {
                    self.action_tx.send(Action::NextPane)?;
                }
                return Ok(());
            },
            // Ctrl+P: cycle focus backward
            (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                self.action_tx.send(Action::PreviousPane)?;
                return Ok(());
            },
            // Ctrl+1/2/3/4: direct focus jump
            (KeyCode::Char('1'), KeyModifiers::CONTROL) => {
                self.set_focus(0);
                self.action_tx.send(Action::FocusQueryInput)?;
                return Ok(());
            },
            (KeyCode::Char('2'), KeyModifiers::CONTROL) => {
                self.set_focus(1);
                self.action_tx.send(Action::FocusResultsViewer)?;
                return Ok(());
            },
            (KeyCode::Char('3'), KeyModifiers::CONTROL) => {
                self.set_focus(2);
                self.action_tx.send(Action::FocusRawResultsViewer)?;
                return Ok(());
            },
            (KeyCode::Char('4'), KeyModifiers::CONTROL) => {
                self.set_focus(3);
                self.action_tx.send(Action::FocusInfoPanel)?;
                return Ok(());
            },
            // Esc: return focus to input
            (KeyCode::Esc, KeyModifiers::NONE) => {
                self.set_focus(0);
                self.action_tx.send(Action::FocusQueryInput)?;
                return Ok(());
            },
            // Arrow keys for scrolling when a viewer panel is focused (not input)
            (KeyCode::Down, KeyModifiers::NONE) if self.focused_pane != 0 => {
                self.action_tx.send(Action::ScrollDown)?;
                return Ok(());
            },
            (KeyCode::Up, KeyModifiers::NONE) if self.focused_pane != 0 => {
                self.action_tx.send(Action::ScrollUp)?;
                return Ok(());
            },
            // Alt+Up/Down for scrolling anywhere (even in input)
            (KeyCode::Down, KeyModifiers::ALT) => {
                self.action_tx.send(Action::ScrollDown)?;
                return Ok(());
            },
            (KeyCode::Up, KeyModifiers::ALT) => {
                self.action_tx.send(Action::ScrollUp)?;
                return Ok(());
            },
            _ => {},
        }

        // Pass input to query input component first (it has priority for typing)
        if let Some(action) = self.query_input.handle_key(key) {
            self.action_tx.send(action)?;
            return Ok(());
        }

        // Scrolling keybindings (only activated if input didn't consume the key)
        match (key.code, key.modifiers) {
            // Scrolling with j/k (vim-style) - only when not typing in input
            (KeyCode::Char('j'), KeyModifiers::NONE) => {
                self.action_tx.send(Action::ScrollDown)?;
            },
            (KeyCode::Char('k'), KeyModifiers::NONE) => {
                self.action_tx.send(Action::ScrollUp)?;
            },
            // Scrolling with PageDown / PageUp
            (KeyCode::PageDown, KeyModifiers::NONE)
            | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                self.action_tx.send(Action::ScrollPageDown)?;
            },
            (KeyCode::PageUp, KeyModifiers::NONE) | (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.action_tx.send(Action::ScrollPageUp)?;
            },
            // Scrolling with Home/End
            (KeyCode::Home, KeyModifiers::NONE) => {
                self.action_tx.send(Action::ScrollToTop)?;
            },
            (KeyCode::End, KeyModifiers::NONE) => {
                self.action_tx.send(Action::ScrollToBottom)?;
            },
            _ => {},
        }

        Ok(())
    }

    /// Update application state based on action
    async fn update(&mut self, action: Action) -> Result<()> {
        use crate::ui::components::Component;

        // Update components
        self.query_input.handle_action(&action);
        self.results_viewer.handle_action(&action);
        self.raw_results_viewer.handle_action(&action);
        self.info_panel.handle_action(&action);
        self.status_bar.handle_action(&action);
        self.help_overlay.handle_action(&action);

        // Handle actions
        match action {
            Action::Quit => {
                self.should_quit = true;
            },
            Action::LoadConfig(path) => {
                self.handle_load_config(path).await?;
            },
            Action::ExecuteQuery(query) => {
                // Route by current mode
                match self.query_mode {
                    QueryMode::Ask => self.handle_execute_query(query).await?,
                    QueryMode::Explain => self.handle_execute_explained_query(query).await?,
                    QueryMode::Reason => self.handle_execute_reason_query(query).await?,
                }
            },
            Action::ExecuteExplainedQuery(query) => {
                self.handle_execute_explained_query(query).await?;
            },
            Action::ExecuteReasonQuery(query) => {
                self.handle_execute_reason_query(query).await?;
            },
            Action::SetQueryMode(mode) => {
                self.query_mode = mode;
                // status_bar already updated via handle_action above
            },
            Action::NextPane => {
                let next = (self.focused_pane + 1) % 4;
                self.set_focus(next);
            },
            Action::PreviousPane => {
                let prev = (self.focused_pane + 3) % 4;
                self.set_focus(prev);
            },
            Action::ExecuteSlashCommand(cmd) => {
                self.handle_slash_command(cmd).await?;
            },
            // QueryExplainedSuccess is handled by components (results_viewer, info_panel)
            _ => {},
        }

        Ok(())
    }

    /// Handle loading configuration
    async fn handle_load_config(&mut self, path: PathBuf) -> Result<()> {
        self.action_tx.send(Action::StartProgress(
            "Loading configuration...".to_string(),
        ))?;

        match crate::config::load_config(&path).await {
            Ok(config) => {
                self.graphrag.initialize(config).await?;
                self.config_path = Some(path.clone());

                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::ConfigLoaded(format!(
                    "Configuration loaded from {}",
                    path.display()
                )))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    "Configuration loaded successfully".to_string(),
                ))?;

                // Update stats
                self.update_stats().await;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::ConfigLoadError(format!(
                    "Failed to load config: {}",
                    e
                )))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Config load failed: {}", e),
                ))?;
            },
        }

        self.set_focus(0);
        Ok(())
    }

    /// Handle executing a query
    async fn handle_execute_query(&mut self, query: String) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first with /config".to_string(),
            ))?;
            return Ok(());
        }

        // Show "Generating answer..." message in Results Viewer
        self.results_viewer.set_content(vec![
            "🤖 Generating Answer...".to_string(),
            "━".repeat(50),
            String::new(),
            format!("Query: {}", query),
            String::new(),
            "⟳ Searching knowledge graph...".to_string(),
            "⟳ Processing results with LLM...".to_string(),
            String::new(),
            "Please wait...".to_string(),
        ]);

        self.action_tx
            .send(Action::StartProgress(format!("Executing query: {}", query)))?;

        let start = Instant::now();

        match self.graphrag.query_with_raw(&query).await {
            Ok((answer, raw_results)) => {
                let duration = start.elapsed();

                // Add to query history
                let entry = QueryEntry {
                    query: query.clone(),
                    timestamp: Utc::now(),
                    duration_ms: duration.as_millis(),
                    results_count: raw_results.len(),
                    results_preview: vec![answer[..answer.len().min(200)].to_string()],
                };

                self.query_history.add_entry(entry.clone());

                // Update info panel with query history
                self.info_panel
                    .add_query(entry.query, entry.duration_ms, entry.results_count);

                // Update raw results viewer with search results
                let mut raw_display = vec![
                    format!("🔍 Raw Search Results ({})", raw_results.len()),
                    "━".repeat(50),
                    String::new(),
                ];
                raw_display.extend(raw_results);
                self.raw_results_viewer.set_content(raw_display);

                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::QuerySuccess(answer))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    format!("Query completed in {}ms", duration.as_millis()),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::QueryError(format!("Query failed: {}", e)))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Query error: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle executing a query in Explain mode (returns confidence + sources)
    async fn handle_execute_explained_query(&mut self, query: String) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first with /config".to_string(),
            ))?;
            return Ok(());
        }

        self.results_viewer.set_content(vec![
            "## Generating Answer (EXPLAIN mode)…".to_string(),
            String::new(),
            format!("**Query:** {}", query),
            String::new(),
            "- Searching knowledge graph…".to_string(),
            "- Computing confidence and source references…".to_string(),
        ]);

        self.action_tx
            .send(Action::StartProgress(format!("EXPLAIN query: {}", query)))?;

        let start = Instant::now();

        match self.graphrag.query_explained(&query).await {
            Ok(result) => {
                let duration = start.elapsed();

                let entry = QueryEntry {
                    query: query.clone(),
                    timestamp: Utc::now(),
                    duration_ms: duration.as_millis(),
                    results_count: result.sources.len(),
                    results_preview: vec![result.answer[..result.answer.len().min(200)].to_string()],
                };
                self.query_history.add_entry(entry.clone());
                self.info_panel
                    .add_query(entry.query, entry.duration_ms, entry.results_count);

                // Populate raw results with source list
                let mut raw_display = vec![
                    format!(
                        "Sources ({}) | Confidence: {:.0}%",
                        result.sources.len(),
                        result.confidence * 100.0
                    ),
                    "━".repeat(50),
                    String::new(),
                ];
                for (i, src) in result.sources.iter().enumerate() {
                    raw_display.push(format!(
                        "{}. [score: {:.2}] {}",
                        i + 1,
                        src.relevance_score,
                        src.id
                    ));
                    let excerpt = &src.excerpt[..src.excerpt.len().min(120)];
                    raw_display.push(format!("   {}", excerpt));
                    raw_display.push(String::new());
                }
                self.raw_results_viewer.set_content(raw_display);

                // Build payload for components
                let payload = QueryExplainedPayload {
                    answer: result.answer.clone(),
                    confidence: result.confidence,
                    sources: result
                        .sources
                        .iter()
                        .map(|s| SourceRef {
                            id: s.id.clone(),
                            excerpt: s.excerpt.clone(),
                            relevance_score: s.relevance_score,
                        })
                        .collect(),
                };

                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::QueryExplainedSuccess(Box::new(payload)))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    format!(
                        "EXPLAIN query done in {}ms | confidence: {:.0}%",
                        duration.as_millis(),
                        result.confidence * 100.0
                    ),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::QueryError(format!("Query failed: {}", e)))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Query error: {}", e),
                ))?;
            },
        }

        self.set_focus(0);
        Ok(())
    }

    /// Handle executing a query in Reason mode (query decomposition)
    async fn handle_execute_reason_query(&mut self, query: String) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first with /config".to_string(),
            ))?;
            return Ok(());
        }

        self.results_viewer.set_content(vec![
            "## Generating Answer (REASON mode)…".to_string(),
            String::new(),
            format!("**Query:** {}", query),
            String::new(),
            "- Decomposing query into sub-questions…".to_string(),
            "- Gathering context for each sub-question…".to_string(),
            "- Synthesizing comprehensive answer…".to_string(),
        ]);

        self.action_tx
            .send(Action::StartProgress(format!("REASON query: {}", query)))?;

        let start = Instant::now();

        match self.graphrag.query_with_reasoning(&query).await {
            Ok(answer) => {
                let duration = start.elapsed();

                let entry = QueryEntry {
                    query: query.clone(),
                    timestamp: Utc::now(),
                    duration_ms: duration.as_millis(),
                    results_count: 0,
                    results_preview: vec![answer[..answer.len().min(200)].to_string()],
                };
                self.query_history.add_entry(entry.clone());
                self.info_panel
                    .add_query(entry.query, entry.duration_ms, entry.results_count);

                self.raw_results_viewer.set_content(vec![
                    "REASON mode — query decomposition active".to_string(),
                    "━".repeat(50),
                    String::new(),
                    "Sub-queries were generated and answered individually.".to_string(),
                    "The result above synthesizes all sub-answers.".to_string(),
                ]);

                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::QuerySuccess(answer))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    format!("REASON query done in {}ms", duration.as_millis()),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::QueryError(format!("Query failed: {}", e)))?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Query error: {}", e),
                ))?;
            },
        }

        self.set_focus(0);
        Ok(())
    }

    /// Handle slash command
    async fn handle_slash_command(&mut self, cmd_str: String) -> Result<()> {
        match SlashCommand::parse(&cmd_str) {
            Ok(cmd) => match cmd {
                SlashCommand::Config(path) => {
                    let expanded = FileOperations::expand_tilde(&path);
                    self.action_tx.send(Action::LoadConfig(expanded))?;
                },
                SlashCommand::Load(path, rebuild) => {
                    self.handle_load_document_with_rebuild(path, rebuild)
                        .await?;
                },
                SlashCommand::Clear => {
                    self.handle_clear_graph().await?;
                },
                SlashCommand::Rebuild => {
                    self.handle_rebuild_graph().await?;
                },
                SlashCommand::Stats => {
                    self.handle_show_stats().await?;
                },
                SlashCommand::Entities(filter) => {
                    self.handle_list_entities(filter).await?;
                },
                SlashCommand::Workspace(name) => {
                    self.handle_load_workspace(name).await?;
                },
                SlashCommand::WorkspaceList => {
                    self.handle_list_workspaces().await?;
                },
                SlashCommand::WorkspaceSave(name) => {
                    self.handle_save_workspace(name).await?;
                },
                SlashCommand::WorkspaceDelete(name) => {
                    self.handle_delete_workspace(name).await?;
                },
                SlashCommand::Reason(query) => {
                    self.handle_execute_reason_query(query).await?;
                },
                SlashCommand::Mode(mode_str) => {
                    let mode = match mode_str.as_str() {
                        "ask" => QueryMode::Ask,
                        "explain" => QueryMode::Explain,
                        "reason" => QueryMode::Reason,
                        other => {
                            self.action_tx.send(Action::SetStatus(
                                StatusType::Error,
                                format!("Unknown mode '{}'. Use: ask | explain | reason", other),
                            ))?;
                            return Ok(());
                        },
                    };
                    self.query_mode = mode;
                    self.action_tx.send(Action::SetQueryMode(mode))?;
                    self.action_tx.send(Action::SetStatus(
                        StatusType::Success,
                        format!("Query mode set to: {}", mode.label()),
                    ))?;
                },
                SlashCommand::ConfigShow => {
                    self.handle_config_show().await?;
                },
                SlashCommand::Export(path) => {
                    self.handle_export(path).await?;
                },
                SlashCommand::Help => {
                    let help_text = SlashCommand::help_text();
                    self.results_viewer
                        .set_content(help_text.lines().map(|s| s.to_string()).collect());
                },
            },
            Err(e) => {
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Command error: {}", e),
                ))?;
            },
        }

        // Always return focus to input after any slash command completes
        self.set_focus(0);

        Ok(())
    }

    /// Handle loading a document with rebuild option
    async fn handle_load_document_with_rebuild(
        &mut self,
        path: PathBuf,
        rebuild: bool,
    ) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first".to_string(),
            ))?;
            return Ok(());
        }

        let expanded = FileOperations::expand_tilde(&path);

        let progress_msg = if rebuild {
            format!("Loading document (rebuild): {}", expanded.display())
        } else {
            format!("Loading document: {}", expanded.display())
        };

        self.action_tx.send(Action::StartProgress(progress_msg))?;

        match self
            .graphrag
            .load_document_with_options(&expanded, rebuild)
            .await
        {
            Ok(message) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::DocumentLoaded(message.clone()))?;
                self.action_tx
                    .send(Action::SetStatus(StatusType::Success, message))?;

                self.update_stats().await;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to load document: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle clearing the knowledge graph
    async fn handle_clear_graph(&mut self) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first".to_string(),
            ))?;
            return Ok(());
        }

        self.action_tx.send(Action::StartProgress(
            "Clearing knowledge graph...".to_string(),
        ))?;

        match self.graphrag.clear_graph().await {
            Ok(message) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::SetStatus(StatusType::Success, message.clone()))?;

                // Display confirmation in results viewer
                self.results_viewer.set_content(vec![
                    "✓ Knowledge Graph Cleared".to_string(),
                    "━".repeat(50),
                    String::new(),
                    message,
                    String::new(),
                    "The knowledge graph has been cleared.".to_string(),
                    "All entities and relationships have been removed.".to_string(),
                    "Documents and chunks are preserved.".to_string(),
                    String::new(),
                    "Use /rebuild to rebuild from loaded documents.".to_string(),
                    "Or use /load <file> to load a new document.".to_string(),
                ]);

                self.update_stats().await;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to clear graph: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle rebuilding the knowledge graph
    async fn handle_rebuild_graph(&mut self) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first".to_string(),
            ))?;
            return Ok(());
        }

        self.action_tx.send(Action::StartProgress(
            "Rebuilding knowledge graph...".to_string(),
        ))?;

        match self.graphrag.rebuild_graph().await {
            Ok(message) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx
                    .send(Action::SetStatus(StatusType::Success, message.clone()))?;

                // Display confirmation in results viewer
                self.results_viewer.set_content(vec![
                    "✓ Knowledge Graph Rebuilt".to_string(),
                    "━".repeat(50),
                    String::new(),
                    message,
                    String::new(),
                    "The knowledge graph has been rebuilt from loaded documents.".to_string(),
                    "All entities and relationships have been re-extracted.".to_string(),
                    String::new(),
                    "You can now query the updated graph.".to_string(),
                ]);

                self.update_stats().await;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to rebuild graph: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle showing statistics
    async fn handle_show_stats(&mut self) -> Result<()> {
        if let Some(stats) = self.graphrag.get_stats().await {
            let lines = vec![
                "📊 Knowledge Graph Statistics".to_string(),
                "━".repeat(50),
                format!("Entities:      {}", stats.entities),
                format!("Relationships: {}", stats.relationships),
                format!("Documents:     {}", stats.documents),
                format!("Chunks:        {}", stats.chunks),
                String::new(),
                format!("Total Queries: {}", self.query_history.total_queries()),
            ];

            self.results_viewer.set_content(lines);
            self.action_tx.send(Action::SetStatus(
                StatusType::Info,
                "Stats displayed".to_string(),
            ))?;
        } else {
            self.action_tx.send(Action::SetStatus(
                StatusType::Warning,
                "No graph loaded yet".to_string(),
            ))?;
        }

        Ok(())
    }

    /// Handle listing entities
    async fn handle_list_entities(&mut self, filter: Option<String>) -> Result<()> {
        match self.graphrag.get_entities(filter.as_deref()).await {
            Ok(entities) => {
                let mut lines = vec![
                    format!(
                        "🔍 Entities{}",
                        if filter.is_some() {
                            format!(" (filtered by '{}')", filter.unwrap())
                        } else {
                            String::new()
                        }
                    ),
                    "━".repeat(50),
                ];

                if entities.is_empty() {
                    lines.push("No entities found.".to_string());
                } else {
                    for (i, entity) in entities.iter().take(50).enumerate() {
                        lines.push(format!(
                            "{}. {} ({})",
                            i + 1,
                            entity.name,
                            entity.entity_type
                        ));
                    }

                    if entities.len() > 50 {
                        lines.push(String::new());
                        lines.push(format!("... and {} more entities", entities.len() - 50));
                    }
                }

                self.results_viewer.set_content(lines);
                self.action_tx.send(Action::SetStatus(
                    StatusType::Info,
                    format!("Found {} entities", entities.len()),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to list entities: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle loading workspace
    async fn handle_load_workspace(&mut self, name: String) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first".to_string(),
            ))?;
            return Ok(());
        }

        self.action_tx.send(Action::StartProgress(format!(
            "Loading workspace '{}'...",
            name
        )))?;

        // Get workspace directory from workspace_manager (default: ~/.graphrag/workspaces)
        let workspace_dir = dirs::data_dir()
            .map(|p| p.join("graphrag").join("workspaces"))
            .unwrap_or_else(|| std::path::PathBuf::from("./workspaces"));

        match self
            .graphrag
            .load_workspace(workspace_dir.to_str().unwrap(), &name)
            .await
        {
            Ok(message) => {
                self.action_tx.send(Action::StopProgress)?;

                // Display success in results viewer
                self.results_viewer.set_content(vec![
                    "✓ Workspace Loaded".to_string(),
                    "━".repeat(50),
                    String::new(),
                    message,
                    String::new(),
                    "The workspace has been loaded successfully.".to_string(),
                    "You can now query the loaded graph.".to_string(),
                ]);

                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    format!("Workspace '{}' loaded", name),
                ))?;

                self.update_stats().await;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to load workspace: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle listing workspaces
    async fn handle_list_workspaces(&mut self) -> Result<()> {
        let workspace_dir = dirs::data_dir()
            .map(|p| p.join("graphrag").join("workspaces"))
            .unwrap_or_else(|| std::path::PathBuf::from("./workspaces"));

        match self
            .graphrag
            .list_workspaces(workspace_dir.to_str().unwrap())
            .await
        {
            Ok(list_output) => {
                self.results_viewer
                    .set_content(list_output.lines().map(|s| s.to_string()).collect());
                self.action_tx.send(Action::SetStatus(
                    StatusType::Info,
                    "Workspace list displayed".to_string(),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to list workspaces: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle saving workspace
    async fn handle_save_workspace(&mut self, name: String) -> Result<()> {
        if !self.graphrag.is_initialized().await {
            self.action_tx.send(Action::SetStatus(
                StatusType::Error,
                "GraphRAG not initialized. Load a config first".to_string(),
            ))?;
            return Ok(());
        }

        self.action_tx.send(Action::StartProgress(format!(
            "Saving workspace '{}'...",
            name
        )))?;

        let workspace_dir = dirs::data_dir()
            .map(|p| p.join("graphrag").join("workspaces"))
            .unwrap_or_else(|| std::path::PathBuf::from("./workspaces"));

        match self
            .graphrag
            .save_workspace(workspace_dir.to_str().unwrap(), &name)
            .await
        {
            Ok(message) => {
                self.action_tx.send(Action::StopProgress)?;

                // Display success in results viewer
                self.results_viewer.set_content(vec![
                    "✓ Workspace Saved".to_string(),
                    "━".repeat(50),
                    String::new(),
                    message,
                    String::new(),
                    format!("Workspace location: {}", workspace_dir.display()),
                    String::new(),
                    "You can load this workspace later with:".to_string(),
                    format!("  /workspace {}", name),
                ]);

                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    format!("Workspace '{}' saved", name),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to save workspace: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle deleting workspace
    async fn handle_delete_workspace(&mut self, name: String) -> Result<()> {
        self.action_tx.send(Action::StartProgress(format!(
            "Deleting workspace '{}'...",
            name
        )))?;

        let workspace_dir = dirs::data_dir()
            .map(|p| p.join("graphrag").join("workspaces"))
            .unwrap_or_else(|| std::path::PathBuf::from("./workspaces"));

        match self
            .graphrag
            .delete_workspace(workspace_dir.to_str().unwrap(), &name)
            .await
        {
            Ok(message) => {
                self.action_tx.send(Action::StopProgress)?;

                // Display success in results viewer
                self.results_viewer.set_content(vec![
                    "✓ Workspace Deleted".to_string(),
                    "━".repeat(50),
                    String::new(),
                    message,
                    String::new(),
                    "The workspace has been permanently deleted.".to_string(),
                ]);

                self.action_tx.send(Action::SetStatus(
                    StatusType::Success,
                    format!("Workspace '{}' deleted", name),
                ))?;
            },
            Err(e) => {
                self.action_tx.send(Action::StopProgress)?;
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Failed to delete workspace: {}", e),
                ))?;
            },
        }

        Ok(())
    }

    /// Handle /config show — display current config file in results viewer
    async fn handle_config_show(&mut self) -> Result<()> {
        if let Some(ref path) = self.config_path.clone() {
            match tokio::fs::read_to_string(path).await {
                Ok(content) => {
                    let mut lines = vec![
                        format!("# Config: {}", path.display()),
                        String::new(),
                        "```".to_string(),
                    ];
                    lines.extend(content.lines().map(|l| l.to_string()));
                    lines.push("```".to_string());
                    self.results_viewer.set_content(lines);
                    self.action_tx.send(Action::SetStatus(
                        StatusType::Info,
                        format!("Showing config: {}", path.display()),
                    ))?;
                },
                Err(e) => {
                    self.action_tx.send(Action::SetStatus(
                        StatusType::Error,
                        format!("Cannot read config file: {}", e),
                    ))?;
                },
            }
        } else {
            self.action_tx.send(Action::SetStatus(
                StatusType::Warning,
                "No config loaded. Use /config <file> first.".to_string(),
            ))?;
        }
        Ok(())
    }

    /// Handle /export <file> — write query history to a Markdown file
    async fn handle_export(&mut self, path: PathBuf) -> Result<()> {
        let entries = self.query_history.last_n(1000);
        if entries.is_empty() {
            self.action_tx.send(Action::SetStatus(
                StatusType::Warning,
                "No query history to export.".to_string(),
            ))?;
            return Ok(());
        }

        let mut md = String::from("# GraphRAG Query History\n\n");
        for (i, entry) in entries.iter().enumerate() {
            md.push_str(&format!("## Query {}\n\n", i + 1));
            md.push_str(&format!("**Q:** {}\n\n", entry.query));
            if !entry.results_preview.is_empty() {
                md.push_str(&format!("**A:** {}\n\n", entry.results_preview[0]));
            }
            md.push_str(&format!(
                "*{}ms · {} sources*\n\n---\n\n",
                entry.duration_ms, entry.results_count
            ));
        }

        let expanded = FileOperations::expand_tilde(&path);
        match tokio::fs::write(&expanded, md.as_bytes()).await {
            Ok(()) => {
                let msg = format!(
                    "Exported {} queries to {}",
                    entries.len(),
                    expanded.display()
                );
                self.results_viewer.set_content(vec![
                    "## Export Complete".to_string(),
                    String::new(),
                    msg.clone(),
                ]);
                self.action_tx
                    .send(Action::SetStatus(StatusType::Success, msg))?;
            },
            Err(e) => {
                self.action_tx.send(Action::SetStatus(
                    StatusType::Error,
                    format!("Export failed: {}", e),
                ))?;
            },
        }
        Ok(())
    }

    /// Update graph statistics in info panel
    async fn update_stats(&mut self) {
        if let Some(stats) = self.graphrag.get_stats().await {
            self.info_panel.set_stats(stats);
        }
    }
}
