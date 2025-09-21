use futures::future::join_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use super::ProcessingResult;
use super::{ProcessingMetrics, RateLimiter};
use crate::core::{Document, GraphRAGError, KnowledgeGraph};

#[derive(Debug)]
pub struct ConcurrentProcessor {
    max_concurrent_documents: usize,
}

impl ConcurrentProcessor {
    pub fn new(max_concurrent_documents: usize) -> Self {
        Self {
            max_concurrent_documents,
        }
    }

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

        println!(
            "📋 Processing {} documents with max concurrency: {}",
            documents.len(),
            self.max_concurrent_documents
        );

        // Process documents in chunks to respect concurrency limits
        let chunk_size = self.max_concurrent_documents;
        let mut all_results = Vec::new();
        let mut total_errors = 0;

        for (chunk_idx, chunk) in documents.chunks(chunk_size).enumerate() {
            println!(
                "📦 Processing chunk {} with {} documents",
                chunk_idx + 1,
                chunk.len()
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
                                println!("  ✅ Document {doc_id} completed in {duration:?}");
                                metrics.record_document_processing_duration(duration);
                            }
                            Err(e) => {
                                println!("  ❌ Document {doc_id} failed in {duration:?}: {e}");
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
                        eprintln!(
                            "Processing error in chunk {} task {}: {}",
                            chunk_idx + 1,
                            task_idx + 1,
                            processing_error
                        );
                    }
                    Err(join_error) => {
                        total_errors += 1;
                        eprintln!(
                            "Task join error in chunk {} task {}: {}",
                            chunk_idx + 1,
                            task_idx + 1,
                            join_error
                        );
                    }
                }
            }

            let chunk_duration = chunk_start.elapsed();
            println!(
                "📦 Chunk {} completed in {:?}: {} successes, {} errors",
                chunk_idx + 1,
                chunk_duration,
                chunk.len() - total_errors.min(chunk.len()),
                total_errors.min(chunk.len())
            );

            // Small delay between chunks to prevent overwhelming the system
            if chunk_idx + 1 < documents.chunks(chunk_size).len() {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }

        if total_errors > 0 {
            println!(
                "⚠️ Batch processing completed with {} errors out of {} documents",
                total_errors,
                documents.len()
            );
        }

        Ok(all_results)
    }

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
