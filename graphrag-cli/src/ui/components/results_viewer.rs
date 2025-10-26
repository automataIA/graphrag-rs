//! Results viewer component with scrolling support

use crate::{action::Action, theme::Theme};
use ratatui::{
    layout::{Margin, Rect},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
    Frame,
};

/// Results viewer with vim-style scrolling
pub struct ResultsViewer {
    /// Display content (lines)
    content: Vec<String>,
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
        Self {
            content: vec![
                "Welcome to GraphRAG CLI!".to_string(),
                "".to_string(),
                "To get started:".to_string(),
                "1. Switch to Command Mode with Shift+Tab".to_string(),
                "2. Load a config: /config path/to/config.json5".to_string(),
                "3. Load documents: /load path/to/document.txt".to_string(),
                "4. Switch to Query Mode and ask questions!".to_string(),
                "".to_string(),
                "Press ? for help".to_string(),
            ],
            scroll_offset: 0,
            scrollbar_state: ScrollbarState::default(),
            focused: false,
            theme: Theme::default(),
        }
    }

    /// Set content
    pub fn set_content(&mut self, lines: Vec<String>) {
        self.content = lines;
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    /// Append lines to content
    #[allow(dead_code)]
    pub fn append_content(&mut self, lines: Vec<String>) {
        self.content.extend(lines);
        self.update_scrollbar();
    }

    /// Clear content
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.content.clear();
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    /// Scroll up one line
    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
        self.update_scrollbar();
    }

    /// Scroll down one line
    pub fn scroll_down(&mut self) {
        if self.scroll_offset < self.content.len().saturating_sub(1) {
            self.scroll_offset += 1;
        }
        self.update_scrollbar();
    }

    /// Scroll up one page
    pub fn scroll_page_up(&mut self, page_size: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(page_size);
        self.update_scrollbar();
    }

    /// Scroll down one page
    pub fn scroll_page_down(&mut self, page_size: usize) {
        let max_scroll = self.content.len().saturating_sub(1);
        self.scroll_offset = (self.scroll_offset + page_size).min(max_scroll);
        self.update_scrollbar();
    }

    /// Scroll to top
    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = 0;
        self.update_scrollbar();
    }

    /// Scroll to bottom
    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = self.content.len().saturating_sub(1);
        self.update_scrollbar();
    }

    /// Update scrollbar state
    fn update_scrollbar(&mut self) {
        self.scrollbar_state = self
            .scrollbar_state
            .content_length(self.content.len())
            .position(self.scroll_offset);
    }

    /// Set focused state
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
            }
            Action::ScrollDown => {
                if self.focused {
                    self.scroll_down();
                }
                None
            }
            Action::ScrollPageUp => {
                if self.focused {
                    self.scroll_page_up(10);
                }
                None
            }
            Action::ScrollPageDown => {
                if self.focused {
                    self.scroll_page_down(10);
                }
                None
            }
            Action::ScrollToTop => {
                if self.focused {
                    self.scroll_to_top();
                }
                None
            }
            Action::ScrollToBottom => {
                if self.focused {
                    self.scroll_to_bottom();
                }
                None
            }
            Action::FocusResultsViewer => {
                self.set_focused(true);
                None
            }
            Action::QuerySuccess(result) => {
                self.set_content(vec![
                    "Query Result:".to_string(),
                    "━".repeat(50),
                    result.clone(),
                ]);
                None
            }
            Action::QueryError(error) => {
                self.set_content(vec![
                    "Query Error:".to_string(),
                    "━".repeat(50),
                    error.clone(),
                ]);
                None
            }
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
            " Results Viewer (j/k to scroll) "
        } else {
            " Results Viewer "
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(border_style);

        let text = self.content.join("\n");

        let paragraph = Paragraph::new(text)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((self.scroll_offset as u16, 0))
            .style(self.theme.text());

        f.render_widget(paragraph, area);

        // Render scrollbar if content is larger than area
        if self.content.len() > area.height as usize {
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
