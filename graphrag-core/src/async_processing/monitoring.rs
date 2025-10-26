//! Performance monitoring and metrics collection for GraphRAG operations.
//!
//! This module provides the [`ProcessingMetrics`] collector for tracking comprehensive
//! statistics about query execution, document processing, batch operations, rate limiting,
//! and system resource usage.
//!
//! # Main Types
//!
//! - [`ProcessingMetrics`]: Thread-safe metrics collector using atomic operations
//! - [`MetricsSummary`]: Comprehensive snapshot of all collected metrics
//! - [`QueryMetrics`]: Query-specific statistics
//! - [`DocumentMetrics`]: Document processing statistics
//! - [`SystemMetrics`]: System-level performance metrics
//!
//! # Features
//!
//! - Thread-safe atomic counters for concurrent access
//! - Duration tracking with automatic sliding window (last 1000 entries)
//! - Success rate calculations for queries and document processing
//! - Peak memory usage tracking
//! - Uptime monitoring
//! - Statistical aggregations (averages, rates)
//! - Formatted summary reporting
//!
//! # Basic Usage
//!
//! ```rust,ignore
//! use graphrag_core::async_processing::ProcessingMetrics;
//! use std::time::Instant;
//!
//! let metrics = ProcessingMetrics::new();
//!
//! // Track a query
//! metrics.increment_query_started();
//! let start = Instant::now();
//! // ... perform query ...
//! metrics.record_query_duration(start.elapsed());
//! metrics.increment_query_success();
//!
//! // Track document processing
//! metrics.increment_document_processing_started();
//! // ... process document ...
//! metrics.increment_document_processing_success();
//!
//! // Get summary statistics
//! let summary = metrics.get_summary();
//! println!("Query success rate: {:.1}%",
//!     summary.queries.success_rate * 100.0
//! );
//! println!("Average document duration: {:?}",
//!     summary.documents.average_duration
//! );
//!
//! // Print full report
//! metrics.print_summary();
//! ```

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Metrics collector for tracking processing performance and statistics
///
/// Thread-safe metrics tracking for queries, document processing, batches,
/// rate limiting, and system resource usage. Uses atomic operations for
/// counters and locks for duration collections.
#[derive(Debug)]
pub struct ProcessingMetrics {
    // Query metrics
    /// Number of queries started
    queries_started: AtomicUsize,
    /// Number of queries that completed successfully
    queries_succeeded: AtomicUsize,
    /// Number of queries that failed
    queries_failed: AtomicUsize,
    /// Collection of query execution durations (capped at 1000 entries)
    query_durations: Arc<RwLock<Vec<Duration>>>,

    // Document processing metrics
    /// Number of document processing operations started
    documents_started: AtomicUsize,
    /// Number of documents processed successfully
    documents_succeeded: AtomicUsize,
    /// Number of document processing operations that failed
    documents_failed: AtomicUsize,
    /// Collection of document processing durations (capped at 1000 entries)
    document_durations: Arc<RwLock<Vec<Duration>>>,

    // Batch processing metrics
    /// Number of batch processing operations started
    batches_started: AtomicUsize,
    /// Collection of batch processing durations (capped at 100 entries)
    batch_durations: Arc<RwLock<Vec<Duration>>>,

    // Rate limiting metrics
    /// Number of rate limiting errors encountered
    rate_limit_errors: AtomicUsize,

    // System metrics
    /// Peak memory usage observed in bytes
    peak_memory_usage: AtomicU64,
    /// Timestamp when metrics tracking started
    creation_time: Instant,
}

impl ProcessingMetrics {
    /// Creates a new metrics collector with all counters initialized to zero
    pub fn new() -> Self {
        Self {
            queries_started: AtomicUsize::new(0),
            queries_succeeded: AtomicUsize::new(0),
            queries_failed: AtomicUsize::new(0),
            query_durations: Arc::new(RwLock::new(Vec::new())),

            documents_started: AtomicUsize::new(0),
            documents_succeeded: AtomicUsize::new(0),
            documents_failed: AtomicUsize::new(0),
            document_durations: Arc::new(RwLock::new(Vec::new())),

            batches_started: AtomicUsize::new(0),
            batch_durations: Arc::new(RwLock::new(Vec::new())),

            rate_limit_errors: AtomicUsize::new(0),

            peak_memory_usage: AtomicU64::new(0),
            creation_time: Instant::now(),
        }
    }

    // Query metrics
    /// Increments the counter for queries started
    pub fn increment_query_started(&self) {
        self.queries_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the counter for successfully completed queries
    pub fn increment_query_success(&self) {
        self.queries_succeeded.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the counter for failed queries
    pub fn increment_query_error(&self) {
        self.queries_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the execution duration of a query
    ///
    /// Keeps only the most recent 1000 measurements to prevent unbounded memory growth.
    ///
    /// # Parameters
    /// - `duration`: Time taken to execute the query
    pub fn record_query_duration(&self, duration: Duration) {
        let mut durations = self.query_durations.write().expect("Lock poisoned");
        durations.push(duration);
        // Keep only last 1000 measurements to prevent memory growth
        if durations.len() > 1000 {
            durations.remove(0);
        }
    }

    // Document processing metrics
    /// Increments the counter for document processing operations started
    pub fn increment_document_processing_started(&self) {
        self.documents_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the counter for successfully processed documents
    pub fn increment_document_processing_success(&self) {
        self.documents_succeeded.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the counter for failed document processing operations
    pub fn increment_document_processing_error(&self) {
        self.documents_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the processing duration of a document
    ///
    /// Keeps only the most recent 1000 measurements to prevent unbounded memory growth.
    ///
    /// # Parameters
    /// - `duration`: Time taken to process the document
    pub fn record_document_processing_duration(&self, duration: Duration) {
        let mut durations = self.document_durations.write().expect("Lock poisoned");
        durations.push(duration);
        // Keep only last 1000 measurements to prevent memory growth
        if durations.len() > 1000 {
            durations.remove(0);
        }
    }

    // Batch processing metrics
    /// Increments the counter for batch processing operations started
    pub fn increment_batch_processing_started(&self) {
        self.batches_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the processing duration of a batch
    ///
    /// Keeps only the most recent 100 measurements to prevent unbounded memory growth.
    ///
    /// # Parameters
    /// - `duration`: Time taken to process the batch
    pub fn record_batch_processing_duration(&self, duration: Duration) {
        let mut durations = self.batch_durations.write().expect("Lock poisoned");
        durations.push(duration);
        // Keep only last 100 measurements to prevent memory growth
        if durations.len() > 100 {
            durations.remove(0);
        }
    }

    // Rate limiting metrics
    /// Increments the counter for rate limiting errors
    pub fn increment_rate_limit_errors(&self) {
        self.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
    }

    // System metrics
    /// Updates peak memory usage if the new value exceeds the current peak
    ///
    /// # Parameters
    /// - `memory_bytes`: Current memory usage in bytes
    pub fn update_peak_memory_usage(&self, memory_bytes: u64) {
        let current = self.peak_memory_usage.load(Ordering::Relaxed);
        if memory_bytes > current {
            self.peak_memory_usage
                .store(memory_bytes, Ordering::Relaxed);
        }
    }

    // Getters for current values
    /// Returns the number of queries started
    pub fn get_queries_started(&self) -> usize {
        self.queries_started.load(Ordering::Relaxed)
    }

    /// Returns the number of queries that succeeded
    pub fn get_queries_succeeded(&self) -> usize {
        self.queries_succeeded.load(Ordering::Relaxed)
    }

    /// Returns the number of queries that failed
    pub fn get_queries_failed(&self) -> usize {
        self.queries_failed.load(Ordering::Relaxed)
    }

    /// Returns the number of document processing operations started
    pub fn get_documents_started(&self) -> usize {
        self.documents_started.load(Ordering::Relaxed)
    }

    /// Returns the number of documents processed successfully
    pub fn get_documents_succeeded(&self) -> usize {
        self.documents_succeeded.load(Ordering::Relaxed)
    }

    /// Returns the number of document processing operations that failed
    pub fn get_documents_failed(&self) -> usize {
        self.documents_failed.load(Ordering::Relaxed)
    }

    /// Returns the number of batch processing operations started
    pub fn get_batches_started(&self) -> usize {
        self.batches_started.load(Ordering::Relaxed)
    }

    /// Returns the number of rate limiting errors encountered
    pub fn get_rate_limit_errors(&self) -> usize {
        self.rate_limit_errors.load(Ordering::Relaxed)
    }

    /// Returns the peak memory usage observed in bytes
    pub fn get_peak_memory_usage(&self) -> u64 {
        self.peak_memory_usage.load(Ordering::Relaxed)
    }

    /// Returns the time elapsed since metrics tracking started
    pub fn get_uptime(&self) -> Duration {
        self.creation_time.elapsed()
    }

    // Statistical methods
    /// Calculates the average query execution duration
    ///
    /// # Returns
    /// Average duration if queries have been recorded, None otherwise
    pub fn get_average_query_duration(&self) -> Option<Duration> {
        let durations = self.query_durations.read().expect("Lock poisoned");
        if durations.is_empty() {
            None
        } else {
            let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
            Some(Duration::from_nanos(total_nanos / durations.len() as u64))
        }
    }

    /// Calculates the average document processing duration
    ///
    /// # Returns
    /// Average duration if documents have been processed, None otherwise
    pub fn get_average_document_duration(&self) -> Option<Duration> {
        let durations = self.document_durations.read().expect("Lock poisoned");
        if durations.is_empty() {
            None
        } else {
            let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
            Some(Duration::from_nanos(total_nanos / durations.len() as u64))
        }
    }

    /// Calculates the query success rate
    ///
    /// # Returns
    /// Ratio of successful queries to total queries (0.0-1.0), or 1.0 if no queries
    pub fn get_query_success_rate(&self) -> f64 {
        let total = self.get_queries_started();
        if total == 0 {
            1.0
        } else {
            self.get_queries_succeeded() as f64 / total as f64
        }
    }

    /// Calculates the document processing success rate
    ///
    /// # Returns
    /// Ratio of successful documents to total documents (0.0-1.0), or 1.0 if no documents
    pub fn get_document_success_rate(&self) -> f64 {
        let total = self.get_documents_started();
        if total == 0 {
            1.0
        } else {
            self.get_documents_succeeded() as f64 / total as f64
        }
    }

    // Summary report
    /// Generates a comprehensive summary of all metrics
    ///
    /// # Returns
    /// Structured summary containing query, document, and system metrics
    pub fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            queries: QueryMetrics {
                started: self.get_queries_started(),
                succeeded: self.get_queries_succeeded(),
                failed: self.get_queries_failed(),
                success_rate: self.get_query_success_rate(),
                average_duration: self.get_average_query_duration(),
            },
            documents: DocumentMetrics {
                started: self.get_documents_started(),
                succeeded: self.get_documents_succeeded(),
                failed: self.get_documents_failed(),
                success_rate: self.get_document_success_rate(),
                average_duration: self.get_average_document_duration(),
            },
            system: SystemMetrics {
                batches_processed: self.get_batches_started(),
                rate_limit_errors: self.get_rate_limit_errors(),
                peak_memory_usage: self.get_peak_memory_usage(),
                uptime: self.get_uptime(),
            },
        }
    }

    /// Prints a formatted summary of all metrics to the log
    pub fn print_summary(&self) {
        let summary = self.get_summary();
        tracing::info!("Processing Metrics Summary");

        tracing::info!(
            started = summary.queries.started,
            succeeded = summary.queries.succeeded,
            failed = summary.queries.failed,
            success_rate = format!("{:.1}%", summary.queries.success_rate * 100.0),
            average_duration = ?summary.queries.average_duration,
            "Query metrics"
        );

        tracing::info!(
            started = summary.documents.started,
            succeeded = summary.documents.succeeded,
            failed = summary.documents.failed,
            success_rate = format!("{:.1}%", summary.documents.success_rate * 100.0),
            average_duration = ?summary.documents.average_duration,
            "Document metrics"
        );

        let peak_memory_mb = if summary.system.peak_memory_usage > 0 {
            Some(summary.system.peak_memory_usage / 1024 / 1024)
        } else {
            None
        };

        tracing::info!(
            batches_processed = summary.system.batches_processed,
            rate_limit_errors = summary.system.rate_limit_errors,
            peak_memory_mb = ?peak_memory_mb,
            uptime = ?summary.system.uptime,
            "System metrics"
        );
    }
}

/// Comprehensive metrics summary containing all tracked statistics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Query-related metrics
    pub queries: QueryMetrics,
    /// Document processing metrics
    pub documents: DocumentMetrics,
    /// System-level metrics
    pub system: SystemMetrics,
}

/// Statistics for query operations
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Number of queries started
    pub started: usize,
    /// Number of queries that succeeded
    pub succeeded: usize,
    /// Number of queries that failed
    pub failed: usize,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Average query execution duration
    pub average_duration: Option<Duration>,
}

/// Statistics for document processing operations
#[derive(Debug, Clone)]
pub struct DocumentMetrics {
    /// Number of document processing operations started
    pub started: usize,
    /// Number of documents processed successfully
    pub succeeded: usize,
    /// Number of document processing operations that failed
    pub failed: usize,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Average document processing duration
    pub average_duration: Option<Duration>,
}

/// System-level performance metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Number of batch processing operations completed
    pub batches_processed: usize,
    /// Number of rate limiting errors encountered
    pub rate_limit_errors: usize,
    /// Peak memory usage observed in bytes
    pub peak_memory_usage: u64,
    /// Time elapsed since metrics tracking started
    pub uptime: Duration,
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self::new()
    }
}
