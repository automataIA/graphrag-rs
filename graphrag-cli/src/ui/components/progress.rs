//! Progress indicator component

use crate::theme::Theme;
use ratatui::{
    layout::Rect,
    text::Span,
    widgets::{Block, Borders, Gauge},
    Frame,
};

/// Progress indicator widget
#[allow(dead_code)]
pub struct ProgressIndicator {
    /// Progress value (0-100)
    progress: u16,
    /// Progress message
    message: String,
    /// Is progress active?
    active: bool,
    /// Theme
    theme: Theme,
}

impl ProgressIndicator {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            progress: 0,
            message: String::new(),
            active: false,
            theme: Theme::default(),
        }
    }

    /// Start progress
    #[allow(dead_code)]
    pub fn start(&mut self, message: String) {
        self.active = true;
        self.message = message;
        self.progress = 0;
    }

    /// Update progress
    #[allow(dead_code)]
    pub fn set_progress(&mut self, progress: u16) {
        self.progress = progress.min(100);
    }

    /// Stop progress
    #[allow(dead_code)]
    pub fn stop(&mut self) {
        self.active = false;
        self.progress = 0;
        self.message.clear();
    }

    /// Check if active
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl super::Component for ProgressIndicator {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        if !self.active {
            return;
        }

        let block = Block::default()
            .title(" Processing... ")
            .borders(Borders::ALL)
            .border_style(self.theme.progress());

        let label = format!("{}% - {}", self.progress, self.message);

        let gauge = Gauge::default()
            .block(block)
            .gauge_style(self.theme.progress())
            .percent(self.progress)
            .label(Span::from(label));

        f.render_widget(gauge, area);
    }
}

impl Default for ProgressIndicator {
    fn default() -> Self {
        Self::new()
    }
}
