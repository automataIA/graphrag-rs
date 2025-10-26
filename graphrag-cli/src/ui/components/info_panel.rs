//! Info panel component showing stats and query history

use crate::{action::Action, handlers::graphrag::GraphStats, theme::Theme};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// Query history entry
#[derive(Debug, Clone)]
pub struct QueryHistoryEntry {
    pub query: String,
    pub duration_ms: u128,
    pub results_count: usize,
}

/// Info panel showing GraphRAG stats and query history
pub struct InfoPanel {
    /// Current graph statistics
    stats: Option<GraphStats>,
    /// Workspace name
    workspace: Option<String>,
    /// Query history (limited to last 5)
    history: Vec<QueryHistoryEntry>,
    /// Total queries executed
    total_queries: usize,
    /// Is this widget focused?
    focused: bool,
    /// Theme
    theme: Theme,
}

impl InfoPanel {
    pub fn new() -> Self {
        Self {
            stats: None,
            workspace: None,
            history: Vec::new(),
            total_queries: 0,
            focused: false,
            theme: Theme::default(),
        }
    }

    /// Set focused state
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    /// Update statistics
    pub fn set_stats(&mut self, stats: GraphStats) {
        self.stats = Some(stats);
    }

    /// Set workspace name
    #[allow(dead_code)]
    pub fn set_workspace(&mut self, name: String) {
        self.workspace = Some(name);
    }

    /// Add query to history
    pub fn add_query(&mut self, query: String, duration_ms: u128, results_count: usize) {
        self.history.insert(
            0,
            QueryHistoryEntry {
                query,
                duration_ms,
                results_count,
            },
        );

        // Keep only last 5
        if self.history.len() > 5 {
            self.history.truncate(5);
        }

        self.total_queries += 1;
    }
}

impl super::Component for InfoPanel {
    fn handle_action(&mut self, action: &Action) -> Option<Action> {
        match action {
            Action::RefreshStats => {
                // Stats will be updated by app
                None
            }
            Action::FocusInfoPanel => {
                self.set_focused(true);
                None
            }
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Split into two sections: stats (top) and history (bottom)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Render stats section
        self.render_stats(f, chunks[0]);

        // Render history section
        self.render_history(f, chunks[1]);
    }
}

impl InfoPanel {
    fn render_stats(&self, f: &mut Frame, area: Rect) {
        let border_style = if self.focused {
            self.theme.border_focused()
        } else {
            self.theme.border()
        };

        let title = if self.focused {
            " 📊 GraphRAG Info (focused) "
        } else {
            " 📊 GraphRAG Info "
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(border_style);

        let content = if let Some(ref stats) = self.stats {
            vec![
                Line::from(vec![
                    Span::styled("Entities: ", self.theme.dimmed()),
                    Span::styled(stats.entities.to_string(), self.theme.highlight()),
                ]),
                Line::from(vec![
                    Span::styled("Relations: ", self.theme.dimmed()),
                    Span::styled(stats.relationships.to_string(), self.theme.highlight()),
                ]),
                Line::from(vec![
                    Span::styled("Documents: ", self.theme.dimmed()),
                    Span::styled(stats.documents.to_string(), self.theme.highlight()),
                ]),
                Line::from(vec![
                    Span::styled("Chunks: ", self.theme.dimmed()),
                    Span::styled(stats.chunks.to_string(), self.theme.highlight()),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Workspace: ", self.theme.dimmed()),
                    Span::styled(
                        self.workspace
                            .as_deref()
                            .unwrap_or("default")
                            .to_string(),
                        self.theme.info(),
                    ),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(
                "No GraphRAG loaded",
                self.theme.dimmed(),
            ))]
        };

        let paragraph = Paragraph::new(content).block(block);
        f.render_widget(paragraph, area);
    }

    fn render_history(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" 📜 Query History ")
            .borders(Borders::ALL)
            .border_style(self.theme.border());

        if self.history.is_empty() {
            let paragraph = Paragraph::new(Span::styled(
                "No queries yet",
                self.theme.dimmed(),
            ))
            .block(block);
            f.render_widget(paragraph, area);
            return;
        }

        let items: Vec<ListItem> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                // Truncate query if too long
                let query_display = if entry.query.len() > 30 {
                    format!("{}...", &entry.query[..27])
                } else {
                    entry.query.clone()
                };

                let content = vec![
                    Line::from(vec![
                        Span::styled(format!("{}. ", i + 1), self.theme.dimmed()),
                        Span::styled(query_display, self.theme.text()),
                    ]),
                    Line::from(vec![
                        Span::styled("   ", Style::default()),
                        Span::styled(
                            format!("{}ms • {} results", entry.duration_ms, entry.results_count),
                            self.theme.dimmed(),
                        ),
                    ]),
                ];

                ListItem::new(content)
            })
            .collect();

        let list = List::new(items)
            .block(block)
            .style(self.theme.text());

        f.render_widget(list, area);
    }
}

impl Default for InfoPanel {
    fn default() -> Self {
        Self::new()
    }
}
