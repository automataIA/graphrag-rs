//! Status bar component with color-coded indicators

use crate::{action::{Action, StatusType}, theme::Theme, ui::Spinner};
use ratatui::{
    layout::{Alignment, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Status bar with indicator
pub struct StatusBar {
    /// Current status message
    message: String,
    /// Status type (determines icon and color)
    status_type: StatusType,
    /// Is progress indicator active?
    progress_active: bool,
    /// Progress message
    progress_message: String,
    /// Animated spinner
    spinner: Spinner,
    /// Theme
    theme: Theme,
}

impl StatusBar {
    pub fn new() -> Self {
        Self {
            message: "Ready".to_string(),
            status_type: StatusType::Info,
            progress_active: false,
            progress_message: String::new(),
            spinner: Spinner::new(),
            theme: Theme::default(),
        }
    }

    /// Set status message
    pub fn set_status(&mut self, status_type: StatusType, message: String) {
        self.status_type = status_type;
        self.message = message;
        self.progress_active = false;
    }

    /// Clear status (reset to default)
    pub fn clear(&mut self) {
        self.message = "Ready".to_string();
        self.status_type = StatusType::Info;
        self.progress_active = false;
    }

    /// Start progress indicator
    pub fn start_progress(&mut self, message: String) {
        self.progress_active = true;
        self.progress_message = message;
        self.status_type = StatusType::Progress;
    }

    /// Stop progress indicator
    pub fn stop_progress(&mut self) {
        self.progress_active = false;
        self.progress_message.clear();
    }
}

impl super::Component for StatusBar {
    fn handle_action(&mut self, action: &Action) -> Option<Action> {
        match action {
            Action::SetStatus(status_type, message) => {
                self.set_status(*status_type, message.clone());
                None
            }
            Action::ClearStatus => {
                self.clear();
                None
            }
            Action::StartProgress(message) => {
                self.start_progress(message.clone());
                None
            }
            Action::StopProgress => {
                self.stop_progress();
                None
            }
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.theme.border());

        let display_message = if self.progress_active {
            // Update spinner animation and show with progress message
            let spinner_frame = self.spinner.tick();
            format!("{} {} {}", spinner_frame, self.status_type.icon(), self.progress_message)
        } else {
            format!("{} {}", self.status_type.icon(), self.message)
        };

        let style = Style::default()
            .fg(self.status_type.color())
            .add_modifier(if matches!(self.status_type, StatusType::Error | StatusType::Warning) {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });

        let help_hint = Span::styled(
            " | Press ? for help | Esc to focus input | Ctrl+C to quit",
            self.theme.dimmed(),
        );

        let line = Line::from(vec![
            Span::styled(display_message, style),
            help_hint,
        ]);

        let paragraph = Paragraph::new(line)
            .block(block)
            .alignment(Alignment::Left);

        f.render_widget(paragraph, area);
    }
}

impl Default for StatusBar {
    fn default() -> Self {
        Self::new()
    }
}
