//! Info panel component with tabbed view: Stats | Sources | History

use crate::{
    action::{Action, QueryExplainedPayload, SourceRef},
    handlers::graphrag::GraphStats,
    theme::Theme,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs},
    Frame,
};

/// Query history entry
#[derive(Debug, Clone)]
pub struct QueryHistoryEntry {
    pub query: String,
    pub duration_ms: u128,
    pub results_count: usize,
}

/// Info panel with three tabs: Stats, Sources, History
pub struct InfoPanel {
    /// Current graph statistics
    stats: Option<GraphStats>,
    /// Workspace name
    workspace: Option<String>,
    /// Query history (limited to last 10)
    history: Vec<QueryHistoryEntry>,
    /// Total queries executed
    total_queries: usize,
    /// Active tab: 0 = Stats, 1 = Sources, 2 = History
    active_tab: usize,
    /// Sources from the last explained query
    sources: Vec<SourceRef>,
    /// Confidence from the last explained query
    confidence: Option<f32>,
    /// Scroll offset within Sources or History tab
    scroll_offset: usize,
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
            active_tab: 0,
            sources: Vec::new(),
            confidence: None,
            scroll_offset: 0,
            focused: false,
            theme: Theme::default(),
        }
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    #[allow(dead_code)]
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    pub fn set_stats(&mut self, stats: GraphStats) {
        self.stats = Some(stats);
    }

    #[allow(dead_code)]
    pub fn set_workspace(&mut self, name: String) {
        self.workspace = Some(name);
    }

    pub fn add_query(&mut self, query: String, duration_ms: u128, results_count: usize) {
        self.history.insert(
            0,
            QueryHistoryEntry {
                query,
                duration_ms,
                results_count,
            },
        );
        if self.history.len() > 10 {
            self.history.truncate(10);
        }
        self.total_queries += 1;
    }

    /// Update sources from an explained query result and auto-switch to Sources tab
    pub fn set_sources(&mut self, payload: &QueryExplainedPayload) {
        self.sources = payload.sources.clone();
        self.confidence = Some(payload.confidence);
        self.active_tab = 1; // auto-switch to Sources tab
        self.scroll_offset = 0;
    }

    /// Cycle to next tab (wraps around)
    pub fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % 3;
        self.scroll_offset = 0;
    }

    fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    fn scroll_down(&mut self, max: usize) {
        if self.scroll_offset + 1 < max {
            self.scroll_offset += 1;
        }
    }
}

impl super::Component for InfoPanel {
    fn handle_action(&mut self, action: &Action) -> Option<Action> {
        match action {
            Action::RefreshStats => None,
            Action::FocusInfoPanel => {
                self.set_focused(true);
                None
            },
            Action::QueryExplainedSuccess(payload) => {
                self.set_sources(payload);
                None
            },
            Action::NextTab => {
                if self.focused {
                    self.next_tab();
                }
                None
            },
            Action::ScrollUp => {
                if self.focused && self.active_tab != 0 {
                    self.scroll_up();
                }
                None
            },
            Action::ScrollDown => {
                if self.focused && self.active_tab != 0 {
                    let max = match self.active_tab {
                        1 => self.sources.len(),
                        2 => self.history.len(),
                        _ => 0,
                    };
                    self.scroll_down(max);
                }
                None
            },
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect) {
        // Split: tab bar (3 rows) + content (rest)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        self.render_tab_bar(f, chunks[0]);

        match self.active_tab {
            0 => self.render_stats(f, chunks[1]),
            1 => self.render_sources(f, chunks[1]),
            2 => self.render_history(f, chunks[1]),
            _ => {},
        }
    }
}

impl InfoPanel {
    fn render_tab_bar(&self, f: &mut Frame, area: Rect) {
        let border_style = if self.focused {
            self.theme.border_focused()
        } else {
            self.theme.border()
        };

        let titles = vec!["Stats", "Sources", "History"];
        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(border_style)
                    .title(if self.focused {
                        " Info Panel [ACTIVE] (Ctrl+N cycles tabs | Ctrl+P back) "
                    } else {
                        " Info Panel (Ctrl+4 or Ctrl+N to focus) "
                    }),
            )
            .select(self.active_tab)
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            )
            .style(self.theme.dimmed());

        f.render_widget(tabs, area);
    }

    fn render_stats(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_style(if self.focused {
                self.theme.border_focused()
            } else {
                self.theme.border()
            });

        let content = if let Some(ref stats) = self.stats {
            vec![
                Line::from(""),
                Line::from(vec![
                    Span::styled("  Entities:  ", self.theme.dimmed()),
                    Span::styled(stats.entities.to_string(), self.theme.highlight()),
                ]),
                Line::from(vec![
                    Span::styled("  Relations: ", self.theme.dimmed()),
                    Span::styled(stats.relationships.to_string(), self.theme.highlight()),
                ]),
                Line::from(vec![
                    Span::styled("  Documents: ", self.theme.dimmed()),
                    Span::styled(stats.documents.to_string(), self.theme.highlight()),
                ]),
                Line::from(vec![
                    Span::styled("  Chunks:    ", self.theme.dimmed()),
                    Span::styled(stats.chunks.to_string(), self.theme.highlight()),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  Queries:   ", self.theme.dimmed()),
                    Span::styled(self.total_queries.to_string(), self.theme.info()),
                ]),
                Line::from(vec![
                    Span::styled("  Workspace: ", self.theme.dimmed()),
                    Span::styled(
                        self.workspace.as_deref().unwrap_or("default").to_string(),
                        self.theme.info(),
                    ),
                ]),
            ]
        } else {
            vec![
                Line::from(""),
                Line::from(Span::styled("  No GraphRAG loaded.", self.theme.dimmed())),
                Line::from(""),
                Line::from(Span::styled("  Use /config <file>", self.theme.dimmed())),
                Line::from(Span::styled("  to get started.", self.theme.dimmed())),
            ]
        };

        let paragraph = Paragraph::new(content).block(block);
        f.render_widget(paragraph, area);
    }

    fn render_sources(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_style(if self.focused {
                self.theme.border_focused()
            } else {
                self.theme.border()
            });

        if self.sources.is_empty() {
            let paragraph = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled("  No sources yet.", self.theme.dimmed())),
                Line::from(""),
                Line::from(Span::styled(
                    "  Use /mode explain then",
                    self.theme.dimmed(),
                )),
                Line::from(Span::styled("  ask a question.", self.theme.dimmed())),
            ])
            .block(block);
            f.render_widget(paragraph, area);
            return;
        }

        // Confidence bar at the top
        let conf = self.confidence.unwrap_or(0.0);
        let conf_color = if conf < 0.3 {
            Color::Red
        } else if conf < 0.7 {
            Color::Yellow
        } else {
            Color::Green
        };
        let bar = confidence_bar(conf, 8);

        let mut items: Vec<ListItem> = vec![
            ListItem::new(Line::from(vec![
                Span::styled("  Confidence: ", self.theme.dimmed()),
                Span::styled(
                    format!("{:.0}% {}", conf * 100.0, bar),
                    Style::default().fg(conf_color),
                ),
            ])),
            ListItem::new(Line::from(Span::styled(
                format!("  Sources: {}", self.sources.len()),
                self.theme.dimmed(),
            ))),
            ListItem::new(Line::from("")),
        ];

        for (i, src) in self.sources.iter().skip(self.scroll_offset).enumerate() {
            let excerpt = if src.excerpt.len() > 60 {
                format!("{}…", &src.excerpt[..57])
            } else {
                src.excerpt.clone()
            };

            items.push(ListItem::new(vec![
                Line::from(vec![
                    Span::styled(
                        format!("  {}. ", i + 1 + self.scroll_offset),
                        self.theme.dimmed(),
                    ),
                    Span::styled(
                        format!("[{:.2}] ", src.relevance_score),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::styled(
                        src.id[..src.id.len().min(20)].to_string(),
                        self.theme.highlight(),
                    ),
                ]),
                Line::from(vec![
                    Span::raw("     ".to_owned()),
                    Span::styled(excerpt, self.theme.dimmed()),
                ]),
                Line::from(""),
            ]));
        }

        let list = List::new(items).block(block);
        f.render_widget(list, area);
    }

    fn render_history(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_style(if self.focused {
                self.theme.border_focused()
            } else {
                self.theme.border()
            });

        if self.history.is_empty() {
            let paragraph = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled("  No queries yet.", self.theme.dimmed())),
            ])
            .block(block);
            f.render_widget(paragraph, area);
            return;
        }

        let items: Vec<ListItem> = self
            .history
            .iter()
            .skip(self.scroll_offset)
            .enumerate()
            .map(|(i, entry)| {
                let query_display = if entry.query.len() > 28 {
                    format!("{}…", &entry.query[..25])
                } else {
                    entry.query.clone()
                };

                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("  {}. ", i + 1 + self.scroll_offset),
                            self.theme.dimmed(),
                        ),
                        Span::styled(query_display, self.theme.text()),
                    ]),
                    Line::from(vec![
                        Span::raw("     ".to_owned()),
                        Span::styled(
                            format!("{}ms · {} src", entry.duration_ms, entry.results_count),
                            self.theme.dimmed(),
                        ),
                    ]),
                    Line::from(""),
                ])
            })
            .collect();

        let list = List::new(items).block(block).style(self.theme.text());
        f.render_widget(list, area);
    }
}

impl Default for InfoPanel {
    fn default() -> Self {
        Self::new()
    }
}

fn confidence_bar(score: f32, width: usize) -> String {
    let filled = (score * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
