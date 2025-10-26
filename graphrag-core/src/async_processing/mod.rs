//! Async processing utilities for GraphRAG operations
//!
//! This module provides async processing capabilities including:
//! - Concurrent document processing
//! - Rate limiting for API calls
//! - Performance monitoring and metrics
//! - Thread pool management
//! - Task scheduling and coordination

use indexmap::IndexMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::core::{Document, GraphRAGError, KnowledgeGraph};

pub mod concurrent_pipeline;
pub mod monitoring;
pub mod rate_limiting;

pub use concurrent_pipeline::ConcurrentProcessor;
pub use monitoring::ProcessingMetrics;
pub use rate_limiting::RateLimiter;

/// Result of processing a single document
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Unique identifier of the processed document
    pub document_id: crate::core::DocumentId,
    /// Number of entities successfully extracted from the document
    pub entities_extracted: usize,
    /// Number of text chunks processed
    pub chunks_processed: usize,
    /// Total time taken to process the document
    pub processing_time: Duration,
    /// Whether the processing completed successfully
    pub success: bool,
}

/// Configuration for async processing operations
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Maximum number of concurrent LLM API calls allowed
    pub max_concurrent_llm_calls: usize,
    /// Maximum number of concurrent embedding API calls allowed
    pub max_concurrent_embeddings: usize,
    /// Maximum number of documents to process concurrently
    pub max_concurrent_documents: usize,
    /// Rate limit for LLM API calls (requests per second)
    pub llm_rate_limit_per_second: f64,
    /// Rate limit for embedding API calls (requests per second)
    pub embedding_rate_limit_per_second: f64,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            max_concurrent_llm_calls: 3,
            max_concurrent_embeddings: 5,
            max_concurrent_documents: 10,
            llm_rate_limit_per_second: 2.0,
            embedding_rate_limit_per_second: 10.0,
        }
    }
}

/// Core async GraphRAG processor with concurrency control and monitoring
#[derive(Debug)]
pub struct AsyncGraphRAGCore {
    /// Shared knowledge graph for storing entities and relationships
    graph: Arc<RwLock<KnowledgeGraph>>,
    /// Rate limiter for throttling API calls
    rate_limiter: Arc<RateLimiter>,
    /// Concurrent processor for batch document processing
    concurrent_processor: Arc<ConcurrentProcessor>,
    /// Metrics collector for tracking processing statistics
    metrics: Arc<ProcessingMetrics>,
    /// Configuration settings for async operations
    config: AsyncConfig,
}

impl AsyncGraphRAGCore {
    /// Creates a new async GraphRAG core instance with configuration
    ///
    /// Initializes the knowledge graph, rate limiter, concurrent processor,
    /// and metrics tracking system.
    ///
    /// # Parameters
    /// - `graph`: Knowledge graph for entity storage
    /// - `config`: Configuration for async processing behavior
    ///
    /// # Returns
    /// Configured async GraphRAG instance, or an error if initialization fails
    pub async fn new(graph: KnowledgeGraph, config: AsyncConfig) -> Result<Self, GraphRAGError> {
        let rate_limiter = Arc::new(RateLimiter::new(&config));
        let concurrent_processor = Arc::new(ConcurrentProcessor::new(config.max_concurrent_documents));
        let metrics = Arc::new(ProcessingMetrics::new());

        Ok(Self {
            graph: Arc::new(RwLock::new(graph)),
            rate_limiter,
            concurrent_processor,
            metrics,
            config,
        })
    }

    /// Processes multiple documents concurrently with rate limiting
    ///
    /// Distributes documents across concurrent workers while respecting
    /// rate limits and collecting metrics.
    ///
    /// # Parameters
    /// - `documents`: Collection of documents to process
    ///
    /// # Returns
    /// Vector of processing results for all documents, or an error
    pub async fn process_documents_async(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<ProcessingResult>, GraphRAGError> {
        let start_time = Instant::now();
        self.metrics.increment_batch_processing_started();

        tracing::info!(document_count = documents.len(), "Processing documents concurrently");

        let results = self
            .concurrent_processor
            .process_batch(
                documents,
                Arc::clone(&self.graph),
                Arc::clone(&self.rate_limiter),
                Arc::clone(&self.metrics),
            )
            .await?;

        let duration = start_time.elapsed();
        self.metrics.record_batch_processing_duration(duration);

        tracing::info!(
            duration_ms = duration.as_millis(),
            successes = results.len(),
            "Batch processing completed"
        );

        Ok(results)
    }

    /// Processes a single document asynchronously with rate limiting
    ///
    /// Applies entity extraction and updates the knowledge graph for one document.
    /// Automatically handles rate limiting and metrics collection.
    ///
    /// # Parameters
    /// - `document`: Document to process
    ///
    /// # Returns
    /// Processing result containing extraction statistics, or an error
    pub async fn process_single_document_async(
        &self,
        document: Document,
    ) -> Result<ProcessingResult, GraphRAGError> {
        let start_time = Instant::now();
        self.metrics.increment_document_processing_started();

        // Acquire rate limiting permits
        let _llm_permit = self.rate_limiter.acquire_llm_permit().await?;

        let result = {
            let _graph = self.graph.read().await;
            // For now, create a simple processing result
            // In a full implementation, this would use proper entity extraction
            ProcessingResult {
                document_id: document.id.clone(),
                entities_extracted: 0,
                chunks_processed: document.chunks.len(),
                processing_time: start_time.elapsed(),
                success: true,
            }
        };

        let duration = start_time.elapsed();

        if result.success {
            self.metrics.increment_document_processing_success();
            self.metrics.record_document_processing_duration(duration);
        } else {
            self.metrics.increment_document_processing_error();
            tracing::warn!(document_id = %result.document_id, "Document processing failed");
        }

        Ok(result)
    }

    /// Executes a query against the knowledge graph asynchronously
    ///
    /// Processes a user query by searching the knowledge graph and generating
    /// a response using retrieved entities and relationships.
    ///
    /// # Parameters
    /// - `query`: User's query string
    ///
    /// # Returns
    /// Generated response string, or an error if processing fails
    pub async fn query_async(
        &self,
        query: &str,
    ) -> Result<String, GraphRAGError> {
        let start_time = Instant::now();
        self.metrics.increment_query_started();

        // Acquire rate limiting permits
        let _llm_permit = self.rate_limiter.acquire_llm_permit().await?;

        // Basic implementation - in production this would use proper query processing
        let result = {
            let graph = self.graph.read().await;
            let entity_count = graph.entities().count();

            if entity_count == 0 {
                Err(GraphRAGError::Unsupported {
                    operation: "query processing".to_string(),
                    reason: "No entities in knowledge graph".to_string(),
                })
            } else {
                // Simple placeholder response
                Ok(format!(
                    "Query processed: '{query}'. Found {entity_count} entities in graph. This is a basic implementation."
                ))
            }
        };

        let duration = start_time.elapsed();

        match &result {
            Ok(_) => {
                self.metrics.increment_query_success();
                self.metrics.record_query_duration(duration);
                tracing::info!(duration_ms = duration.as_millis(), "Query completed successfully");
            }
            Err(e) => {
                self.metrics.increment_query_error();
                tracing::error!(error = %e, "Query processing error");
            }
        }

        result
    }

    /// Retrieves current processing metrics
    ///
    /// # Returns
    /// Reference to the metrics collector
    pub fn get_metrics(&self) -> &ProcessingMetrics {
        &self.metrics
    }

    /// Retrieves current configuration
    ///
    /// # Returns
    /// Reference to the async processing configuration
    pub fn get_config(&self) -> &AsyncConfig {
        &self.config
    }

    /// Performs health check on all system components
    ///
    /// Checks the status of the knowledge graph and rate limiter to determine
    /// overall system health.
    ///
    /// # Returns
    /// Health status summary for all components
    pub async fn health_check(&self) -> HealthStatus {
        let graph_status = {
            let graph = self.graph.read().await;
            if graph.entities().count() > 0 {
                ComponentStatus::Healthy
            } else {
                ComponentStatus::Warning("No entities in graph".to_string())
            }
        };

        let rate_limiter_status = self.rate_limiter.health_check();

        HealthStatus {
            overall: if matches!(graph_status, ComponentStatus::Healthy)
                && matches!(rate_limiter_status, ComponentStatus::Healthy)
            {
                ComponentStatus::Healthy
            } else {
                ComponentStatus::Warning("Some components have issues".to_string())
            },
            components: indexmap::indexmap! {
                "graph".to_string() => graph_status,
                "rate_limiter".to_string() => rate_limiter_status,
            },
        }
    }

    /// Shuts down the async processor gracefully
    ///
    /// Allows current operations to complete before shutting down. In a full
    /// implementation, this cancels pending tasks and cleans up resources.
    ///
    /// # Returns
    /// Ok on successful shutdown, or an error if cleanup fails
    pub async fn shutdown(&self) -> Result<(), GraphRAGError> {
        tracing::info!("Shutting down async GraphRAG processor");

        // In a full implementation, this would:
        // - Cancel running tasks
        // - Wait for current operations to complete
        // - Clean up resources

        tracing::info!("Async processor shutdown complete");
        Ok(())
    }
}

/// Status of individual system components
#[derive(Debug, Clone)]
pub enum ComponentStatus {
    /// Component is functioning normally
    Healthy,
    /// Component is operational but has issues (includes warning message)
    Warning(String),
    /// Component has failed (includes error message)
    Error(String),
}

/// Overall health status including all system components
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Aggregate health status across all components
    pub overall: ComponentStatus,
    /// Individual status for each named component
    pub components: IndexMap<String, ComponentStatus>,
}

/// Simple task scheduler for managing async operations
#[derive(Debug)]
pub struct TaskScheduler {
    /// Maximum number of tasks allowed to run concurrently
    max_concurrent_tasks: usize,
}

impl TaskScheduler {
    /// Creates a new task scheduler with specified concurrency limit
    ///
    /// # Parameters
    /// - `max_concurrent_tasks`: Maximum number of concurrent tasks
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            max_concurrent_tasks,
        }
    }

    /// Schedules and executes an async task
    ///
    /// In production, this would implement proper task queuing and scheduling.
    /// Currently executes tasks immediately.
    ///
    /// # Parameters
    /// - `task`: Future to execute
    ///
    /// # Returns
    /// Result of the task execution
    pub async fn schedule_task<F, T>(&self, task: F) -> Result<T, GraphRAGError>
    where
        F: std::future::Future<Output = Result<T, GraphRAGError>>,
    {
        // Basic implementation - in production this would use proper task scheduling
        task.await
    }

    /// Returns the maximum number of concurrent tasks allowed
    pub fn max_concurrent_tasks(&self) -> usize {
        self.max_concurrent_tasks
    }
}

/// Performance tracker for monitoring async operation timing
#[derive(Debug, Default)]
pub struct PerformanceTracker {
    /// Total number of operations recorded
    total_operations: std::sync::atomic::AtomicU64,
    /// Cumulative duration of all operations
    total_duration: std::sync::Mutex<Duration>,
}

impl PerformanceTracker {
    /// Creates a new performance tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the duration of a completed operation
    ///
    /// # Parameters
    /// - `duration`: Time taken for the operation
    pub fn record_operation(&self, duration: Duration) {
        self.total_operations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut total_duration = self.total_duration.lock().unwrap();
        *total_duration += duration;
    }

    /// Calculates the average duration per operation
    ///
    /// # Returns
    /// Average duration, or zero if no operations have been recorded
    pub fn get_average_duration(&self) -> Duration {
        let total_ops = self.total_operations.load(std::sync::atomic::Ordering::Relaxed);
        if total_ops == 0 {
            return Duration::from_secs(0);
        }

        let total_duration = *self.total_duration.lock().unwrap();
        total_duration / total_ops as u32
    }

    /// Returns the total number of operations recorded
    pub fn get_total_operations(&self) -> u64 {
        self.total_operations.load(std::sync::atomic::Ordering::Relaxed)
    }
}
