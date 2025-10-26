//! Concurrent document processing pipeline for parallel GraphRAG operations.
//!
//! This module provides the [`ConcurrentProcessor`] for processing multiple documents
//! in parallel while respecting concurrency limits and coordinating with rate limiters
//! and metrics tracking.
//!
//! # Main Types
//!
//! - [`ConcurrentProcessor`]: Manages concurrent document processing with configurable
//!   parallelism limits
//!
//! # Features
//!
//! - Concurrent processing of document batches with configurable limits
//! - Automatic chunking to respect concurrency constraints
//! - Integration with rate limiting to prevent API throttling
//! - Comprehensive metrics tracking for all operations
//! - Error isolation: failures in one document don't affect others
//! - Task spawning with tokio for true parallel execution
//!
//! # Basic Usage
//!
//! ```rust,ignore
//! use graphrag_core::async_processing::ConcurrentProcessor;
//! use std::sync::Arc;
//!
//! let processor = ConcurrentProcessor::new(10); // Max 10 concurrent documents
//!
//! let results = processor.process_batch(
//!     documents,
//!     graph,
//!     rate_limiter,
//!     metrics
//! ).await?;
//!
//! for result in results {
//!     println!("Processed document {} in {:?}",
//!         result.document_id,
//!         result.processing_time
//!     );
//! }
//! ```

use futures::future::join_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use super::ProcessingResult;
use super::{ProcessingMetrics, RateLimiter};
use crate::core::{Document, GraphRAGError, KnowledgeGraph};

/// Concurrent document processor for parallel GraphRAG operations
///
/// Manages concurrent processing of multiple documents while respecting
/// concurrency limits and coordinating with rate limiters and metrics tracking.
#[derive(Debug)]
pub struct ConcurrentProcessor {
    /// Maximum number of documents to process concurrently
    max_concurrent_documents: usize,
}

impl ConcurrentProcessor {
    /// Creates a new concurrent processor with specified concurrency limit
    ///
    /// # Parameters
    /// - `max_concurrent_documents`: Maximum number of documents to process in parallel
    pub fn new(max_concurrent_documents: usize) -> Self {
        Self {
            max_concurrent_documents,
        }
    }

    /// Processes a batch of documents concurrently with rate limiting and metrics tracking
    ///
    /// Documents are processed in chunks according to the concurrency limit. Each chunk
    /// is fully processed before moving to the next, with a small delay between chunks
    /// to prevent system overload.
    ///
    /// # Parameters
    /// - `documents`: Collection of documents to process
    /// - `graph`: Shared knowledge graph for storing extracted entities
    /// - `rate_limiter`: Rate limiter for API call throttling
    /// - `metrics`: Metrics collector for tracking processing statistics
    ///
    /// # Returns
    /// Vector of processing results for successfully processed documents, or an error
    pub async fn process_batch(
        &self,
        documents: Vec<Document>,
        graph: Arc<RwLock<KnowledgeGraph>>,
        rate_limiter: Arc<RateLimiter>,
        metrics: Arc<ProcessingMetrics>,
    ) -> Result<Vec<ProcessingResult>, GraphRAGError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        tracing::info!(
            document_count = documents.len(),
            max_concurrency = self.max_concurrent_documents,
            "Processing documents batch"
        );

        // Process documents in chunks to respect concurrency limits
        let chunk_size = self.max_concurrent_documents;
        let mut all_results = Vec::new();
        let mut total_errors = 0;

        for (chunk_idx, chunk) in documents.chunks(chunk_size).enumerate() {
            tracing::debug!(
                chunk_number = chunk_idx + 1,
                chunk_size = chunk.len(),
                "Processing chunk"
            );

            let chunk_start = Instant::now();

            // Create tasks for this chunk
            let tasks: Vec<_> = chunk
                .iter()
                .cloned()
                .map(|document| {
                    let graph = Arc::clone(&graph);
                    let rate_limiter = Arc::clone(&rate_limiter);
                    let metrics = Arc::clone(&metrics);
                    let doc_id = document.id.clone();

                    tokio::spawn(async move {
                        let doc_start = Instant::now();

                        // Acquire rate limiting permit
                        let _permit = match rate_limiter.acquire_llm_permit().await {
                            Ok(permit) => permit,
                            Err(e) => {
                                metrics.increment_rate_limit_errors();
                                return Err(e);
                            }
                        };

                        // Process the document
                        let result =
                            Self::process_single_document_internal(&graph, document, &metrics)
                                .await;

                        let duration = doc_start.elapsed();

                        match &result {
                            Ok(_) => {
                                tracing::debug!(document_id = %doc_id, duration_ms = duration.as_millis(), "Document completed");
                                metrics.record_document_processing_duration(duration);
                            }
                            Err(e) => {
                                tracing::warn!(document_id = %doc_id, duration_ms = duration.as_millis(), error = %e, "Document failed");
                                metrics.increment_document_processing_error();
                            }
                        }

                        result
                    })
                })
                .collect();

            // Wait for all tasks in this chunk to complete
            let chunk_results = join_all(tasks).await;

            // Collect results and handle errors
            for (task_idx, task_result) in chunk_results.into_iter().enumerate() {
                match task_result {
                    Ok(Ok(processing_result)) => {
                        all_results.push(processing_result);
                        metrics.increment_document_processing_success();
                    }
                    Ok(Err(processing_error)) => {
                        total_errors += 1;
                        tracing::error!(
                            chunk_number = chunk_idx + 1,
                            task_number = task_idx + 1,
                            error = %processing_error,
                            "Processing error"
                        );
                    }
                    Err(join_error) => {
                        total_errors += 1;
                        tracing::error!(
                            chunk_number = chunk_idx + 1,
                            task_number = task_idx + 1,
                            error = %join_error,
                            "Task join error"
                        );
                    }
                }
            }

            let chunk_duration = chunk_start.elapsed();
            tracing::debug!(
                chunk_number = chunk_idx + 1,
                duration_ms = chunk_duration.as_millis(),
                successes = chunk.len() - total_errors.min(chunk.len()),
                errors = total_errors.min(chunk.len()),
                "Chunk completed"
            );

            // Small delay between chunks to prevent overwhelming the system
            if chunk_idx + 1 < documents.chunks(chunk_size).len() {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }

        if total_errors > 0 {
            tracing::warn!(
                total_errors = total_errors,
                total_documents = documents.len(),
                "Batch processing completed with errors"
            );
        }

        Ok(all_results)
    }

    /// Processes a single document internally with graph access
    ///
    /// This is the core document processing logic executed within concurrent tasks.
    /// Currently implements a basic placeholder that will be replaced with full
    /// entity extraction in production.
    ///
    /// # Parameters
    /// - `graph`: Shared knowledge graph for entity storage
    /// - `document`: Document to process
    /// - `_metrics`: Metrics collector (currently unused in implementation)
    ///
    /// # Returns
    /// Processing result containing extraction statistics and timing
    async fn process_single_document_internal(
        graph: &Arc<RwLock<KnowledgeGraph>>,
        document: Document,
        _metrics: &ProcessingMetrics,
    ) -> Result<ProcessingResult, GraphRAGError> {
        let start_time = Instant::now();

        // For now, create a simple processing result
        // In a full async implementation, this would use async entity extraction
        let result = {
            let _graph_read = graph.read().await;
            ProcessingResult {
                document_id: document.id.clone(),
                entities_extracted: 0, // Would be actual count from extraction
                chunks_processed: document.chunks.len(),
                processing_time: start_time.elapsed(),
                success: true,
            }
        };

        Ok(result)
    }
}
