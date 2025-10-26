//! Terminal User Interface management
//!
//! Handles terminal initialization, cleanup, and event streaming.

use color_eyre::eyre::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, EventStream},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use futures::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Stdout};
use tokio::{
    sync::mpsc::{self, UnboundedReceiver, UnboundedSender},
    task::JoinHandle,
    time::{self, Duration},
};
use tokio_util::sync::CancellationToken;

/// Event types from the terminal
#[derive(Debug, Clone)]
pub enum Event {
    /// Keyboard or mouse event from crossterm
    Crossterm(CrosstermEvent),
    /// Periodic tick for animations/updates
    Tick,
    /// Render frame
    Render,
    /// Terminal was resized
    Resize(u16, u16),
}

/// Terminal User Interface
pub struct Tui {
    /// The terminal instance
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
    /// Background task handle
    task: JoinHandle<()>,
    /// Cancellation token for cleanup
    cancellation_token: CancellationToken,
    /// Event receiver
    event_rx: UnboundedReceiver<Event>,
    /// Event sender (for external use if needed)
    _event_tx: UnboundedSender<Event>,
    /// Frame rate (FPS)
    #[allow(dead_code)]
    frame_rate: f64,
    /// Tick rate (events per second)
    #[allow(dead_code)]
    tick_rate: f64,
}

impl Tui {
    /// Create a new TUI instance
    pub fn new() -> Result<Self> {
        let frame_rate = 60.0; // 60 FPS
        let tick_rate = 4.0; // 4 ticks per second

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let cancellation_token = CancellationToken::new();

        // Spawn event handler task
        let task = {
            let event_tx = event_tx.clone();
            let cancellation_token = cancellation_token.clone();
            let tick_duration = Duration::from_secs_f64(1.0 / tick_rate);
            let render_duration = Duration::from_secs_f64(1.0 / frame_rate);

            tokio::spawn(async move {
                let mut reader = EventStream::new();
                let mut tick_interval = time::interval(tick_duration);
                let mut render_interval = time::interval(render_duration);

                loop {
                    tokio::select! {
                        biased;

                        _ = cancellation_token.cancelled() => {
                            break;
                        }
                        maybe_event = reader.next() => {
                            match maybe_event {
                                Some(Ok(evt)) => {
                                    // Handle resize events specially
                                    if let CrosstermEvent::Resize(w, h) = evt {
                                        let _ = event_tx.send(Event::Resize(w, h));
                                    }
                                    let _ = event_tx.send(Event::Crossterm(evt));
                                }
                                Some(Err(_)) => {}
                                None => break,
                            }
                        }
                        _ = tick_interval.tick() => {
                            let _ = event_tx.send(Event::Tick);
                        }
                        _ = render_interval.tick() => {
                            let _ = event_tx.send(Event::Render);
                        }
                    }
                }
            })
        };

        // Initialize terminal
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

        Ok(Self {
            terminal,
            task,
            cancellation_token,
            event_rx,
            _event_tx: event_tx,
            frame_rate,
            tick_rate,
        })
    }

    /// Enter the alternate screen and enable raw mode
    pub fn enter(&mut self) -> Result<()> {
        enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        io::stdout().execute(EnableMouseCapture)?;
        self.terminal.hide_cursor()?;
        self.terminal.clear()?;
        Ok(())
    }

    /// Leave the alternate screen and disable raw mode
    pub fn exit(&mut self) -> Result<()> {
        self.terminal.show_cursor()?;
        io::stdout().execute(DisableMouseCapture)?;
        io::stdout().execute(LeaveAlternateScreen)?;
        disable_raw_mode()?;
        Ok(())
    }

    /// Cancel the background task
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Get the next event
    pub async fn next(&mut self) -> Option<Event> {
        self.event_rx.recv().await
    }

    /// Get frame rate
    #[allow(dead_code)]
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    /// Get tick rate
    #[allow(dead_code)]
    pub fn tick_rate(&self) -> f64 {
        self.tick_rate
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        self.cancel();
        let _ = self.exit();
        self.task.abort();
    }
}
