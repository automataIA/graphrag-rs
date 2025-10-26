//! Parallel processing utilities for GraphRAG
//!
//! This module provides parallel processing capabilities using Rayon
//! for improved performance on multi-core systems.

use crate::core::Result;

/// Parallel processor for batch operations
#[derive(Debug, Clone)]
pub struct ParallelProcessor {
    num_threads: usize,
}

/// Statistics about parallel processing performance
#[derive(Debug, Clone, Default)]
pub struct ParallelStatistics {
    /// Number of tasks processed
    pub tasks_processed: usize,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Average time per task
    pub avg_time_per_task_ms: f64,
}

/// Performance monitor for parallel operations
#[derive(Debug, Clone, Default)]
pub struct PerformanceMonitor {
    stats: ParallelStatistics,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self::default()
    }

    /// Time an operation and record statistics
    pub fn time_operation<F, T>(&mut self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = std::time::Instant::now();
        let result = operation()?;
        let elapsed = start.elapsed();

        self.stats.tasks_processed += 1;
        self.stats.total_time_ms += elapsed.as_millis() as u64;
        self.stats.avg_time_per_task_ms =
            self.stats.total_time_ms as f64 / self.stats.tasks_processed as f64;

        Ok(result)
    }

    /// Get current statistics
    pub fn stats(&self) -> &ParallelStatistics {
        &self.stats
    }

    /// Get statistics (alternative method name)
    pub fn get_stats(&self) -> &ParallelStatistics {
        &self.stats
    }

    /// Get average operation duration
    pub fn average_duration(&self) -> f64 {
        self.stats.avg_time_per_task_ms
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.stats = ParallelStatistics::default();
    }
}

impl Default for ParallelProcessor {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }
}

impl ParallelProcessor {
    /// Create a new parallel processor with the specified number of threads
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Get processor configuration
    pub fn config(&self) -> ParallelConfig {
        ParallelConfig {
            num_threads: self.num_threads,
            batch_size: 100,
            chunk_batch_size: 50,
        }
    }

    /// Execute work in parallel
    pub fn execute_parallel<T, F>(&self, items: Vec<T>, f: F) -> Vec<T>
    where
        T: Send + Sync,
        F: Fn(&T) -> T + Send + Sync,
    {
        #[cfg(feature = "parallel-processing")]
        {
            use rayon::prelude::*;
            items.par_iter().map(f).collect()
        }
        #[cfg(not(feature = "parallel-processing"))]
        {
            items.iter().map(f).collect()
        }
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> ParallelStatistics {
        ParallelStatistics::default()
    }

    /// Determine if parallel processing should be used
    pub fn should_use_parallel(&self, item_count: usize) -> bool {
        item_count > 10 && self.num_threads > 1
    }
}

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use
    pub num_threads: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Chunk batch size for parallel processing
    pub chunk_batch_size: usize,
}

/// Configure Rayon thread pool
#[cfg(feature = "parallel-processing")]
pub fn configure_thread_pool(num_threads: usize) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| crate::core::GraphRAGError::Config {
            message: format!("Failed to configure thread pool: {}", e)
        })
}

/// No-op when parallel processing is disabled
#[cfg(not(feature = "parallel-processing"))]
pub fn configure_thread_pool(_num_threads: usize) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_processor() {
        let processor = ParallelProcessor::new(4);
        assert_eq!(processor.num_threads(), 4);

        let items = vec![1, 2, 3, 4, 5];
        let results = processor.execute_parallel(items, |x| x * 2);
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_processor_config() {
        let processor = ParallelProcessor::default();
        let items = vec![1, 2, 3];
        let results = processor.execute_parallel(items, |x| x * 2);
        assert_eq!(results, vec![2, 4, 6]);
    }
}
