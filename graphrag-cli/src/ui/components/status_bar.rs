//! Status bar component with color-coded indicators and query mode badge

use crate::{
    action::{Action, QueryMode, StatusType},
    theme::Theme,
    ui::Spinner,
};
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Status bar with indicator and query mode badge
pub struct StatusBar {
    message: String,
    status_type: StatusType,
    progress_active: bool,
    progress_message: String,
    spinner: Spinner,
    /// Current query mode (displayed as right-aligned badge)
    query_mode: QueryMode,
    theme: Theme,
}

impl StatusBar {
    pub fn new() -> Self {
        Self {
            message: "Ready — use /config to load a configuration".to_string(),
            status_type: StatusType::Info,
            progress_active: false,
            progress_message: String::new(),
            spinner: Spinner::new(),
            query_mode: QueryMode::default(),
            theme: Theme::default(),
        }
    }

    pub fn set_status(&mut self, status_type: StatusType, message: String) {
        self.status_type = status_type;
        self.message = message;
        self.progress_active = false;
    }

    pub fn clear(&mut self) {
        self.message = "Ready".to_string();
        self.status_type = StatusType::Info;
        self.progress_active = false;
    }

    pub fn start_progress(&mut self, message: String) {
        self.progress_active = true;
        self.progress_message = message;
        self.status_type = StatusType::Progress;
    }

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
            },
            Action::ClearStatus => {
                self.clear();
                None
            },
            Action::StartProgress(message) => {
                self.start_progress(message.clone());
                None
            },
            Action::StopProgress => {
                self.stop_progress();
                None
            },
            Action::SetQueryMode(mode) => {
                self.query_mode = *mode;
                None
            },
            _ => None,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.theme.border());

        let display_message = if self.progress_active {
            let spinner_frame = self.spinner.tick();
            format!(
                "{} {} {}",
                spinner_frame,
                self.status_type.icon(),
                self.progress_message
            )
        } else {
            format!("{} {}", self.status_type.icon(), self.message)
        };

        let msg_style = Style::default().fg(self.status_type.color()).add_modifier(
            if matches!(self.status_type, StatusType::Error | StatusType::Warning) {
                Modifier::BOLD
            } else {
                Modifier::empty()
            },
        );

        // Query mode badge color
        let (mode_color, mode_label) = match self.query_mode {
            QueryMode::Ask => (Color::DarkGray, " [ASK] "),
            QueryMode::Explain => (Color::Cyan, " [EXPLAIN] "),
            QueryMode::Reason => (Color::Magenta, " [REASON] "),
        };

        let mode_badge = Span::styled(
            mode_label.to_owned(),
            Style::default()
                .fg(Color::Black)
                .bg(mode_color)
                .add_modifier(Modifier::BOLD),
        );

        let hint = Span::styled(
            " | Ctrl+N next | ↑↓ scroll | ? help | Esc input | Ctrl+C quit".to_owned(),
            self.theme.dimmed(),
        );

        let line = Line::from(vec![
            Span::styled(display_message, msg_style),
            hint,
            mode_badge,
        ]);

        let paragraph = Paragraph::new(line).block(block).alignment(Alignment::Left);

        f.render_widget(paragraph, area);
    }
}

impl Default for StatusBar {
    fn default() -> Self {
        Self::new()
    }
}
