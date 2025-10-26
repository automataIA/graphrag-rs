//! Query input component
//!
//! Single input box that automatically detects slash commands

use crate::{action::Action, theme::Theme};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Block, Borders},
    Frame,
};
use tui_textarea::TextArea;

/// Query input widget - handles both queries and slash commands
pub struct QueryInput {
    /// Text area for input
    textarea: TextArea<'static>,
    /// Is this widget focused?
    focused: bool,
    /// Theme
    theme: Theme,
}

impl QueryInput {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_cursor_line_style(Style::default());
        textarea.set_placeholder_text("Enter query or /command... (e.g., \"What are the main entities?\" or \"/config file.json5\")");

        Self {
            textarea,
            focused: true,
            theme: Theme::default(),
        }
    }

    /// Handle keyboard input directly
    /// Returns Some(Action) if an action should be triggered, None if key was consumed for input
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<Action> {
        // Only handle keys when input is focused
        if !self.focused {
            return None;
        }

        // Handle special keys that should trigger actions
        match (key.code, key.modifiers) {
            // Submit on Enter
            (KeyCode::Enter, KeyModifiers::NONE) => {
                let content = self.textarea.lines().join("\n");

                if content.trim().is_empty() {
                    return Some(Action::SetStatus(
                        crate::action::StatusType::Warning,
                        "Cannot submit empty input".to_string(),
                    ));
                }

                // Clear textarea
                self.textarea = TextArea::default();
                self.textarea.set_cursor_line_style(Style::default());
                self.textarea.set_placeholder_text("Enter query or /command... (e.g., \"What are the main entities?\" or \"/config file.json5\")");

                // Auto-detect: slash command vs query
                if crate::mode::is_slash_command(&content) {
                    Some(Action::ExecuteSlashCommand(content))
                } else {
                    Some(Action::ExecuteQuery(content))
                }
            }
            // Clear input (consume the key, don't pass to scrolling)
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                self.textarea = TextArea::default();
                self.textarea.set_cursor_line_style(Style::default());
                self.textarea.set_placeholder_text("Enter query or /command... (e.g., \"What are the main entities?\" or \"/config file.json5\")");
                Some(Action::Noop)  // Return Noop to indicate key was consumed
            }
            // Let textarea handle everything else - return Noop to indicate consumption
            _ => {
                self.textarea.input(key);
                Some(Action::Noop)  // Key was consumed by input
            }
        }
    }

    /// Set focused state
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }
}

impl super::Component for QueryInput {
    fn handle_action(&mut self, action: &Action) -> Option<Action> {
        match action {
            Action::FocusQueryInput => {
                self.set_focused(true);
                None
            }
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect) {
        let border_color = if self.focused {
            ratatui::style::Color::Green
        } else {
            self.theme.border
        };

        let title = if self.focused {
            "💬 Input (Enter to submit | Ctrl+D to clear)"
        } else {
            "💬 Input (Inactive)"
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color).add_modifier(
                if self.focused {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                },
            ));

        self.textarea.set_block(block);
        self.textarea.set_cursor_style(if self.focused {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        });

        f.render_widget(&self.textarea, area);
    }
}

impl Default for QueryInput {
    fn default() -> Self {
        Self::new()
    }
}
