//! Raw results viewer component - shows search results before LLM processing

use crate::{action::Action, theme::Theme};
use ratatui::{
    layout::{Margin, Rect},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
    Frame,
};

/// Raw results viewer with scrolling support
pub struct RawResultsViewer {
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

impl RawResultsViewer {
    pub fn new() -> Self {
        Self {
            content: vec![
                "Raw search results will appear here.".to_string(),
                "".to_string(),
                "These are the entities and relationships retrieved from".to_string(),
                "the knowledge graph before LLM processing.".to_string(),
            ],
            scroll_offset: 0,
            scrollbar_state: ScrollbarState::default(),
            focused: false,
            theme: Theme::default(),
        }
    }

    /// Set content from search results
    pub fn set_content(&mut self, lines: Vec<String>) {
        self.content = lines;
        self.scroll_offset = 0;
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

    /// Set focused state
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    /// Update scrollbar state
    fn update_scrollbar(&mut self) {
        self.scrollbar_state = self
            .scrollbar_state
            .content_length(self.content.len())
            .position(self.scroll_offset);
    }
}

impl super::Component for RawResultsViewer {
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
            Action::FocusRawResultsViewer => {
                self.set_focused(true);
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
            " Raw Search Results (j/k to scroll) "
        } else {
            " Raw Search Results "
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
            .style(self.theme.text_dim());  // Use dimmed text for raw results

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

impl Default for RawResultsViewer {
    fn default() -> Self {
        Self::new()
    }
}
