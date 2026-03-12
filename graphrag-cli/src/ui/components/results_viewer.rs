//! Results viewer component with scrolling support and markdown rendering

use crate::{action::Action, theme::Theme};
use ratatui::{
    layout::{Margin, Rect},
    text::{Line, Text},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
    Frame,
};

/// Results viewer with vim-style scrolling and markdown rendering
pub struct ResultsViewer {
    /// Rendered content lines (markdown-parsed)
    content_lines: Vec<Line<'static>>,
    /// Vertical scroll position
    scroll_offset: usize,
    /// Scrollbar state
    scrollbar_state: ScrollbarState,
    /// Is this widget focused?
    focused: bool,
    /// Theme
    theme: Theme,
}

impl ResultsViewer {
    pub fn new() -> Self {
        let welcome = crate::ui::markdown::parse_markdown(
            "# Welcome to GraphRAG CLI\n\
             \n\
             To get started:\n\
             \n\
             - Load a config: `/config path/to/config.json5`\n\
             - Load a document: `/load path/to/document.txt`\n\
             - Ask questions in natural language!\n\
             \n\
             Press **?** for help  |  Use `/mode explain` for richer output",
        );
        let mut rv = Self {
            content_lines: welcome,
            scroll_offset: 0,
            scrollbar_state: ScrollbarState::default(),
            focused: false,
            theme: Theme::default(),
        };
        rv.update_scrollbar();
        rv
    }

    /// Set content from plain string lines (runs through markdown parser)
    pub fn set_content(&mut self, lines: Vec<String>) {
        let combined = lines.join("\n");
        self.content_lines = crate::ui::markdown::parse_markdown(&combined);
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    /// Set content from pre-built `Line` values (bypasses markdown parser)
    pub fn set_lines(&mut self, lines: Vec<Line<'static>>) {
        self.content_lines = lines;
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    /// Append plain lines to existing content
    #[allow(dead_code)]
    pub fn append_content(&mut self, lines: Vec<String>) {
        let combined = lines.join("\n");
        let mut new_lines = crate::ui::markdown::parse_markdown(&combined);
        self.content_lines.append(&mut new_lines);
        self.update_scrollbar();
    }

    /// Clear content
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.content_lines.clear();
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
        self.update_scrollbar();
    }

    pub fn scroll_down(&mut self) {
        if self.scroll_offset < self.content_lines.len().saturating_sub(1) {
            self.scroll_offset += 1;
        }
        self.update_scrollbar();
    }

    pub fn scroll_page_up(&mut self, page_size: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(page_size);
        self.update_scrollbar();
    }

    pub fn scroll_page_down(&mut self, page_size: usize) {
        let max_scroll = self.content_lines.len().saturating_sub(1);
        self.scroll_offset = (self.scroll_offset + page_size).min(max_scroll);
        self.update_scrollbar();
    }

    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = self.content_lines.len().saturating_sub(1);
        self.update_scrollbar();
    }

    fn update_scrollbar(&mut self) {
        self.scrollbar_state = self
            .scrollbar_state
            .content_length(self.content_lines.len())
            .position(self.scroll_offset);
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }
}

impl super::Component for ResultsViewer {
    fn handle_action(&mut self, action: &Action) -> Option<Action> {
        match action {
            Action::ScrollUp => {
                if self.focused {
                    self.scroll_up();
                }
                None
            },
            Action::ScrollDown => {
                if self.focused {
                    self.scroll_down();
                }
                None
            },
            Action::ScrollPageUp => {
                if self.focused {
                    self.scroll_page_up(10);
                }
                None
            },
            Action::ScrollPageDown => {
                if self.focused {
                    self.scroll_page_down(10);
                }
                None
            },
            Action::ScrollToTop => {
                if self.focused {
                    self.scroll_to_top();
                }
                None
            },
            Action::ScrollToBottom => {
                if self.focused {
                    self.scroll_to_bottom();
                }
                None
            },
            Action::FocusResultsViewer => {
                self.set_focused(true);
                None
            },
            Action::QuerySuccess(result) => {
                self.set_content(vec![
                    "## Query Result".to_string(),
                    String::new(),
                    result.clone(),
                ]);
                None
            },
            Action::QueryExplainedSuccess(payload) => {
                use ratatui::{
                    style::{Color, Style},
                    text::Span,
                };
                let conf_color = confidence_color(payload.confidence);
                let conf_bar = confidence_bar(payload.confidence, 10);
                let header_line = Line::from(vec![
                    Span::styled("Query Result  ".to_owned(), self.theme.title()),
                    Span::styled(
                        format!(
                            "[EXPLAIN | {:.0}% {}]",
                            payload.confidence * 100.0,
                            conf_bar
                        ),
                        Style::default().fg(conf_color),
                    ),
                ]);
                let mut lines: Vec<Line<'static>> = vec![
                    header_line,
                    Line::from(Span::styled(
                        "━".repeat(50),
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(""),
                ];
                lines.extend(crate::ui::markdown::parse_markdown(&payload.answer));
                self.set_lines(lines);
                None
            },
            Action::QueryError(error) => {
                self.set_content(vec![
                    "## Query Error".to_string(),
                    String::new(),
                    format!("> {}", error),
                ]);
                None
            },
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect) {
        let border_style = if self.focused {
            self.theme.border_focused()
        } else {
            self.theme.border()
        };

        let title = if self.focused {
            " Results Viewer [ACTIVE] (j/k or ↑↓ to scroll | Ctrl+N next panel) "
        } else {
            " Results Viewer (Ctrl+2 or Ctrl+N to focus) "
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(border_style);

        let visible: Vec<Line> = self
            .content_lines
            .iter()
            .skip(self.scroll_offset)
            .cloned()
            .collect();

        let paragraph = Paragraph::new(Text::from(visible))
            .block(block)
            .wrap(Wrap { trim: false })
            .style(self.theme.text());

        f.render_widget(paragraph, area);

        if self.content_lines.len() > area.height as usize {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));

            let scrollbar_area = area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            });

            f.render_stateful_widget(scrollbar, scrollbar_area, &mut self.scrollbar_state);
        }
    }
}

impl Default for ResultsViewer {
    fn default() -> Self {
        Self::new()
    }
}

fn confidence_color(score: f32) -> ratatui::style::Color {
    use ratatui::style::Color;
    if score < 0.3 {
        Color::Red
    } else if score < 0.7 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn confidence_bar(score: f32, width: usize) -> String {
    let filled = (score * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
