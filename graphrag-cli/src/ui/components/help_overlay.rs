//! Help overlay component showing keybindings

use crate::theme::Theme;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, BorderType, Clear, Paragraph},
    Frame,
};

/// Help overlay with keybindings
pub struct HelpOverlay {
    /// Is help visible?
    visible: bool,
    /// Theme
    theme: Theme,
}

impl HelpOverlay {
    pub fn new() -> Self {
        Self {
            visible: false,
            theme: Theme::default(),
        }
    }

    /// Toggle visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Show help
    #[allow(dead_code)]
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hide help
    #[allow(dead_code)]
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Check if visible
    pub fn is_visible(&self) -> bool {
        self.visible
    }
}

impl super::Component for HelpOverlay {
    fn handle_action(&mut self, action: &crate::action::Action) -> Option<crate::action::Action> {
        match action {
            crate::action::Action::ToggleHelp => {
                self.toggle();
                None
            }
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, _area: Rect) {
        if !self.visible {
            return;
        }

        // Center the popup (60% x 60%)
        let area = centered_rect(60, 70, f.area());

        // Clear background
        f.render_widget(Clear, area);

        let block = Block::default()
            .title(" GraphRAG CLI - Help ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(self.theme.border_focused());

        let help_text = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("Global Shortcuts", self.theme.title()),
            ]),
            Line::from("━".repeat(55)),
            keybinding_line("?", "Toggle this help overlay", &self.theme),
            keybinding_line("q / Ctrl+C", "Quit application", &self.theme),
            keybinding_line("Tab", "Switch focus between panes", &self.theme),
            Line::from(""),
            Line::from(vec![
                Span::styled("Input Box", self.theme.title()),
            ]),
            Line::from("━".repeat(55)),
            keybinding_line("Enter", "Submit query or /command", &self.theme),
            keybinding_line("Ctrl+D", "Clear input", &self.theme),
            keybinding_line("Backspace", "Delete character", &self.theme),
            Line::from(vec![
                Span::styled("  Tip: ", self.theme.dimmed()),
                Span::styled("Type queries directly or use /commands", self.theme.text()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Results Viewer Navigation", self.theme.title()),
            ]),
            Line::from("━".repeat(55)),
            keybinding_line("j / ↓", "Scroll down one line", &self.theme),
            keybinding_line("k / ↑", "Scroll up one line", &self.theme),
            keybinding_line("Ctrl+D", "Scroll down one page", &self.theme),
            keybinding_line("Ctrl+U", "Scroll up one page", &self.theme),
            keybinding_line("Home", "Jump to top", &self.theme),
            keybinding_line("End", "Jump to bottom", &self.theme),
            Line::from(""),
            Line::from(vec![
                Span::styled("Slash Commands", self.theme.title()),
            ]),
            Line::from("━".repeat(55)),
            keybinding_line("/config <file>", "Load configuration", &self.theme),
            keybinding_line("/load <file>", "Load document", &self.theme),
            keybinding_line("/stats", "Show graph statistics", &self.theme),
            keybinding_line("/entities [filter]", "List entities", &self.theme),
            keybinding_line("/workspace <name>", "Switch workspace", &self.theme),
            keybinding_line("/help", "Show command help", &self.theme),
            Line::from(""),
            Line::from(vec![
                Span::styled("Status Indicators", self.theme.title()),
            ]),
            Line::from("━".repeat(55)),
            Line::from(vec![
                Span::styled("ℹ  ", self.theme.info()),
                Span::styled("Info  ", self.theme.text()),
                Span::styled("✓  ", self.theme.success()),
                Span::styled("Success  ", self.theme.text()),
                Span::styled("⚠  ", self.theme.warning()),
                Span::styled("Warning", self.theme.text()),
            ]),
            Line::from(vec![
                Span::styled("✗  ", self.theme.error()),
                Span::styled("Error  ", self.theme.text()),
                Span::styled("⟳  ", self.theme.progress()),
                Span::styled("Progress", self.theme.text()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Press ? to close this help", self.theme.dimmed()),
            ]),
        ];

        let paragraph = Paragraph::new(help_text)
            .block(block)
            .alignment(Alignment::Left);

        f.render_widget(paragraph, area);
    }
}

/// Helper to create a keybinding line
fn keybinding_line<'a>(key: &'a str, description: &'a str, theme: &'a Theme) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("  {:15} ", key), theme.highlight()),
        Span::styled(description.to_string(), theme.text()),
    ])
}

/// Helper to center a rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

impl Default for HelpOverlay {
    fn default() -> Self {
        Self::new()
    }
}
