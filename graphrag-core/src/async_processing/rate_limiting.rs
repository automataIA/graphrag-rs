//! Rate limiting for API call throttling and concurrency control.
//!
//! This module provides the [`RateLimiter`] for controlling both the concurrency and
//! frequency of API calls to prevent throttling and respect service limits.
//!
//! # Main Types
//!
//! - [`RateLimiter`]: Dual-level rate limiter with semaphore-based concurrency control
//!   and time-based request throttling
//!
//! # Features
//!
//! - Separate rate limiting for LLM and embedding API calls
//! - Semaphore-based concurrency control (max N simultaneous calls)
//! - Time-based rate limiting (max N calls per second)
//! - Automatic waiting when limits are reached
//! - RAII-style permit handling with automatic release
//! - Health checking for congestion detection
//! - Per-second rate window with automatic reset
//!
//! # Rate Limiting Strategy
//!
//! The rate limiter implements a two-tier approach:
//!
//! 1. **Concurrency Control**: Uses semaphores to limit how many API calls can run
//!    simultaneously. This prevents overwhelming the system with too many parallel requests.
//!
//! 2. **Time-Based Rate Limiting**: Tracks requests per second and automatically waits
//!    when the limit is reached. The counter resets every second.
//!
//! # Basic Usage
//!
//! ```rust,ignore
//! use graphrag_core::async_processing::{RateLimiter, AsyncConfig};
//!
//! let config = AsyncConfig {
//!     max_concurrent_llm_calls: 3,
//!     llm_rate_limit_per_second: 2.0,
//!     max_concurrent_embeddings: 5,
//!     embedding_rate_limit_per_second: 10.0,
//!     ..Default::default()
//! };
//!
//! let rate_limiter = RateLimiter::new(&config);
//!
//! // Acquire permit for LLM call (blocks if needed)
//! let permit = rate_limiter.acquire_llm_permit().await?;
//! // ... make LLM API call ...
//! // Permit is automatically released when dropped
//!
//! // Check available capacity
//! let available = rate_limiter.get_available_llm_permits();
//! println!("Available LLM permits: {}", available);
//!
//! // Health check
//! let status = rate_limiter.health_check();
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::time;

use super::{AsyncConfig, ComponentStatus};
use crate::core::GraphRAGError;

/// Rate limiter for controlling API call frequency and concurrency
///
/// Provides dual-level throttling: semaphore-based concurrency control and
/// time-based rate limiting for both LLM and embedding API calls.
#[derive(Debug)]
pub struct RateLimiter {
    /// Semaphore limiting concurrent LLM API calls
    llm_semaphore: Arc<Semaphore>,
    /// Semaphore limiting concurrent embedding API calls
    embedding_semaphore: Arc<Semaphore>,
    /// Tracker for LLM API call rate limiting
    llm_rate_tracker: Arc<tokio::sync::Mutex<RateTracker>>,
    /// Tracker for embedding API call rate limiting
    embedding_rate_tracker: Arc<tokio::sync::Mutex<RateTracker>>,
    /// Configuration settings
    config: AsyncConfig,
}

/// Internal tracker for time-based rate limiting
#[derive(Debug)]
struct RateTracker {
    /// Timestamp of the last request
    last_request: Option<Instant>,
    /// Number of requests made in the current second
    requests_this_second: u32,
    /// Maximum requests allowed per second
    rate_limit: f64,
}

impl RateTracker {
    /// Creates a new rate tracker with specified rate limit
    ///
    /// # Parameters
    /// - `rate_limit`: Maximum requests allowed per second
    fn new(rate_limit: f64) -> Self {
        Self {
            last_request: None,
            requests_this_second: 0,
            rate_limit,
        }
    }

    /// Checks rate limit and waits if necessary before allowing request
    ///
    /// Automatically resets the counter when entering a new second. If the
    /// rate limit is reached, waits until the next second before proceeding.
    ///
    /// # Returns
    /// Ok if request can proceed, or an error if rate limiting fails
    async fn wait_if_needed(&mut self) -> Result<(), GraphRAGError> {
        let now = Instant::now();

        if let Some(last_request) = self.last_request {
            let time_since_last = now.duration_since(last_request);

            // Reset counter if we're in a new second
            if time_since_last >= Duration::from_secs(1) {
                self.requests_this_second = 0;
            }

            // Check if we need to wait
            if self.requests_this_second as f64 >= self.rate_limit {
                let wait_time = Duration::from_secs(1) - time_since_last;
                if wait_time > Duration::ZERO {
                    time::sleep(wait_time).await;
                }
                self.requests_this_second = 0;
            }
        }

        self.last_request = Some(now);
        self.requests_this_second += 1;

        Ok(())
    }
}

impl RateLimiter {
    /// Creates a new rate limiter from configuration
    ///
    /// Initializes semaphores and rate trackers for both LLM and embedding API calls.
    ///
    /// # Parameters
    /// - `config`: Configuration specifying concurrency and rate limits
    pub fn new(config: &AsyncConfig) -> Self {
        Self {
            llm_semaphore: Arc::new(Semaphore::new(config.max_concurrent_llm_calls)),
            embedding_semaphore: Arc::new(Semaphore::new(config.max_concurrent_embeddings)),
            llm_rate_tracker: Arc::new(tokio::sync::Mutex::new(RateTracker::new(
                config.llm_rate_limit_per_second,
            ))),
            embedding_rate_tracker: Arc::new(tokio::sync::Mutex::new(RateTracker::new(
                config.embedding_rate_limit_per_second,
            ))),
            config: config.clone(),
        }
    }

    /// Acquires a permit for making an LLM API call
    ///
    /// Blocks until both concurrency and rate limits allow the call to proceed.
    /// The permit must be held for the duration of the API call and will be
    /// released when dropped.
    ///
    /// # Returns
    /// Semaphore permit on success, or an error if acquisition fails
    pub async fn acquire_llm_permit(&self) -> Result<SemaphorePermit<'_>, GraphRAGError> {
        // First acquire the semaphore permit for concurrency control
        let permit = self
            .llm_semaphore
            .acquire()
            .await
            .map_err(|e| GraphRAGError::RateLimit {
                message: format!("Failed to acquire LLM permit: {e}"),
            })?;

        // Then check rate limiting
        {
            let mut rate_tracker = self.llm_rate_tracker.lock().await;
            rate_tracker.wait_if_needed().await?;
        }

        Ok(permit)
    }

    /// Acquires a permit for making an embedding API call
    ///
    /// Blocks until both concurrency and rate limits allow the call to proceed.
    /// The permit must be held for the duration of the API call and will be
    /// released when dropped.
    ///
    /// # Returns
    /// Semaphore permit on success, or an error if acquisition fails
    pub async fn acquire_embedding_permit(&self) -> Result<SemaphorePermit<'_>, GraphRAGError> {
        // First acquire the semaphore permit for concurrency control
        let permit =
            self.embedding_semaphore
                .acquire()
                .await
                .map_err(|e| GraphRAGError::RateLimit {
                    message: format!("Failed to acquire embedding permit: {e}"),
                })?;

        // Then check rate limiting
        {
            let mut rate_tracker = self.embedding_rate_tracker.lock().await;
            rate_tracker.wait_if_needed().await?;
        }

        Ok(permit)
    }

    /// Returns the number of available LLM permits
    ///
    /// # Returns
    /// Number of LLM API calls that can be made immediately without waiting
    pub fn get_available_llm_permits(&self) -> usize {
        self.llm_semaphore.available_permits()
    }

    /// Returns the number of available embedding permits
    ///
    /// # Returns
    /// Number of embedding API calls that can be made immediately without waiting
    pub fn get_available_embedding_permits(&self) -> usize {
        self.embedding_semaphore.available_permits()
    }

    /// Performs a health check on the rate limiter
    ///
    /// Checks permit availability to determine if the system is healthy or
    /// experiencing congestion.
    ///
    /// # Returns
    /// Component status indicating health (Healthy, Warning, or Error)
    pub fn health_check(&self) -> ComponentStatus {
        let llm_available = self.get_available_llm_permits();
        let embedding_available = self.get_available_embedding_permits();

        if llm_available == 0 && embedding_available == 0 {
            ComponentStatus::Warning("No permits available".to_string())
        } else if llm_available == 0 {
            ComponentStatus::Warning("No LLM permits available".to_string())
        } else if embedding_available == 0 {
            ComponentStatus::Warning("No embedding permits available".to_string())
        } else {
            ComponentStatus::Healthy
        }
    }

    /// Returns the current configuration
    ///
    /// # Returns
    /// Reference to the async processing configuration
    pub fn get_config(&self) -> &AsyncConfig {
        &self.config
    }
}
