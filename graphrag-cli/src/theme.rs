//! Color themes and styling for the TUI
//!
//! Provides consistent color schemes and style utilities across the application.

use ratatui::style::{Color, Modifier, Style};

/// Application color theme
#[derive(Debug, Clone)]
pub struct Theme {
    /// Primary accent color (used for highlights)
    pub primary: Color,
    /// Secondary accent color
    #[allow(dead_code)]
    pub secondary: Color,
    /// Success color (green)
    pub success: Color,
    /// Error color (red)
    pub error: Color,
    /// Warning color (yellow)
    pub warning: Color,
    /// Info color (blue)
    pub info: Color,
    /// Progress color (cyan)
    pub progress: Color,
    /// Background color
    #[allow(dead_code)]
    pub background: Color,
    /// Foreground/text color
    pub foreground: Color,
    /// Border color (normal state)
    pub border: Color,
    /// Border color (focused state)
    pub border_focused: Color,
    /// Selection/highlight color
    #[allow(dead_code)]
    pub selection: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Blue,
            success: Color::Green,
            error: Color::Red,
            warning: Color::Yellow,
            info: Color::Blue,
            progress: Color::Cyan,
            background: Color::Black,
            foreground: Color::White,
            border: Color::DarkGray,
            border_focused: Color::Cyan,
            selection: Color::DarkGray,
        }
    }
}

impl Theme {
    /// Create a light theme variant
    #[allow(dead_code)]
    pub fn light() -> Self {
        Self {
            primary: Color::Blue,
            secondary: Color::Cyan,
            success: Color::Green,
            error: Color::Red,
            warning: Color::Yellow,
            info: Color::Blue,
            progress: Color::Cyan,
            background: Color::White,
            foreground: Color::Black,
            border: Color::Gray,
            border_focused: Color::Blue,
            selection: Color::LightBlue,
        }
    }

    /// Get style for normal text
    pub fn text(&self) -> Style {
        Style::default().fg(self.foreground)
    }

    /// Get style for dimmed text
    pub fn text_dim(&self) -> Style {
        Style::default().fg(Color::DarkGray)
    }

    /// Get style for title text
    pub fn title(&self) -> Style {
        Style::default()
            .fg(self.primary)
            .add_modifier(Modifier::BOLD)
    }

    /// Get style for success messages
    pub fn success(&self) -> Style {
        Style::default().fg(self.success)
    }

    /// Get style for error messages
    pub fn error(&self) -> Style {
        Style::default()
            .fg(self.error)
            .add_modifier(Modifier::BOLD)
    }

    /// Get style for warning messages
    pub fn warning(&self) -> Style {
        Style::default().fg(self.warning)
    }

    /// Get style for info messages
    pub fn info(&self) -> Style {
        Style::default().fg(self.info)
    }

    /// Get style for progress indicators
    pub fn progress(&self) -> Style {
        Style::default().fg(self.progress)
    }

    /// Get style for borders (normal)
    pub fn border(&self) -> Style {
        Style::default().fg(self.border)
    }

    /// Get style for borders (focused)
    pub fn border_focused(&self) -> Style {
        Style::default()
            .fg(self.border_focused)
            .add_modifier(Modifier::BOLD)
    }

    /// Get style for selected items
    #[allow(dead_code)]
    pub fn selected(&self) -> Style {
        Style::default()
            .bg(self.selection)
            .add_modifier(Modifier::BOLD)
    }

    /// Get style for highlighted text
    pub fn highlight(&self) -> Style {
        Style::default()
            .fg(self.primary)
            .add_modifier(Modifier::BOLD)
    }

    /// Get style for dimmed/secondary text
    pub fn dimmed(&self) -> Style {
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM)
    }
}
