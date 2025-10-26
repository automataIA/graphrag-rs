//! Animated spinner for progress indication

use std::time::{Duration, Instant};

/// Animated spinner with various styles
pub struct Spinner {
    /// Current frame index
    frame_index: usize,
    /// Frames for animation
    frames: Vec<&'static str>,
    /// Last update time
    last_update: Instant,
    /// Animation speed (milliseconds per frame)
    speed_ms: u64,
}

impl Spinner {
    /// Create a new spinner with default animation
    pub fn new() -> Self {
        Self {
            frame_index: 0,
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            last_update: Instant::now(),
            speed_ms: 80,
        }
    }

    /// Create a dots spinner
    #[allow(dead_code)]
    pub fn dots() -> Self {
        Self {
            frame_index: 0,
            frames: vec!["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
            last_update: Instant::now(),
            speed_ms: 80,
        }
    }

    /// Create a simple spinner
    #[allow(dead_code)]
    pub fn simple() -> Self {
        Self {
            frame_index: 0,
            frames: vec!["|", "/", "-", "\\"],
            last_update: Instant::now(),
            speed_ms: 100,
        }
    }

    /// Update animation and get current frame
    pub fn tick(&mut self) -> &'static str {
        let now = Instant::now();
        if now.duration_since(self.last_update) >= Duration::from_millis(self.speed_ms) {
            self.frame_index = (self.frame_index + 1) % self.frames.len();
            self.last_update = now;
        }
        self.frames[self.frame_index]
    }

    /// Get current frame without updating
    pub fn current_frame(&self) -> &'static str {
        self.frames[self.frame_index]
    }

    /// Reset to first frame
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.frame_index = 0;
        self.last_update = Instant::now();
    }
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}
