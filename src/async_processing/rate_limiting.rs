use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::time;

use super::{AsyncConfig, ComponentStatus};
use crate::core::GraphRAGError;

#[derive(Debug)]
pub struct RateLimiter {
    llm_semaphore: Arc<Semaphore>,
    embedding_semaphore: Arc<Semaphore>,
    llm_rate_tracker: Arc<tokio::sync::Mutex<RateTracker>>,
    embedding_rate_tracker: Arc<tokio::sync::Mutex<RateTracker>>,
    config: AsyncConfig,
}

#[derive(Debug)]
struct RateTracker {
    last_request: Option<Instant>,
    requests_this_second: u32,
    rate_limit: f64,
}

impl RateTracker {
    fn new(rate_limit: f64) -> Self {
        Self {
            last_request: None,
            requests_this_second: 0,
            rate_limit,
        }
    }

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

    pub fn get_available_llm_permits(&self) -> usize {
        self.llm_semaphore.available_permits()
    }

    pub fn get_available_embedding_permits(&self) -> usize {
        self.embedding_semaphore.available_permits()
    }

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

    pub fn get_config(&self) -> &AsyncConfig {
        &self.config
    }
}
