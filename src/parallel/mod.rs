use crate::config::ParallelConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel-processing")]
use rayon::prelude::*;

/// Parallel processing utilities and thread pool management
#[derive(Debug, Clone)]
pub struct ParallelProcessor {
    config: ParallelConfig,
    initialized: bool,
}

impl ParallelProcessor {
    /// Create a new parallel processor with the given configuration
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }

    /// Initialize the thread pool based on configuration
    pub fn initialize(&mut self) -> crate::Result<()> {
        if !self.config.enabled {
            println!("Parallel processing disabled");
            return Ok(());
        }

        let num_threads = if self.config.num_threads == 0 {
            #[cfg(feature = "parallel-processing")]
            {
                num_cpus::get()
            }
            #[cfg(not(feature = "parallel-processing"))]
            {
                1
            }
        } else {
            self.config.num_threads
        };

        #[cfg(feature = "parallel-processing")]
        {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
            {
                Ok(_) => {
                    println!("Initialized parallel processing with {num_threads} threads");
                }
                Err(e) if e.to_string().contains("already been initialized") => {
                    println!("Parallel processing using existing global thread pool");
                }
                Err(e) => {
                    return Err(crate::GraphRAGError::Config {
                        message: format!("Failed to initialize thread pool: {e}"),
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            println!("Parallel processing feature not enabled - using single thread");
        }

        self.initialized = true;
        Ok(())
    }

    /// Execute a parallel operation with timing and memory monitoring
    #[cfg(feature = "parallel-processing")]
    pub fn execute_parallel<T, F, I>(&self, items: I, operation: F) -> Vec<T>
    where
        T: Send,
        F: Fn(I::Item) -> T + Sync + Send,
        I: IntoParallelIterator,
        I::Item: Send,
    {
        if !self.config.enabled || !self.initialized {
            // Fall back to sequential processing
            return items.into_par_iter().map(operation).collect();
        }

        let start_time = Instant::now();
        let result = items.into_par_iter().map(operation).collect();
        let duration = start_time.elapsed();

        if duration > Duration::from_millis(100) {
            println!("Parallel operation completed in {duration:?}");
        }

        result
    }

    /// Execute a sequential operation (when parallel feature is disabled)
    #[cfg(not(feature = "parallel-processing"))]
    pub fn execute_parallel<T, F, I>(&self, items: I, operation: F) -> Vec<T>
    where
        T: Send,
        F: Fn(I::Item) -> T + Sync + Send,
        I: IntoIterator,
        I::Item: Send,
    {
        let start_time = Instant::now();
        let result: Vec<T> = items.into_iter().map(operation).collect();
        let duration = start_time.elapsed();

        if duration > Duration::from_millis(100) {
            println!("Sequential operation completed in {duration:?}");
        }

        result
    }

    /// Execute parallel operation with dynamic batching
    #[cfg(feature = "parallel-processing")]
    pub fn execute_batched_parallel<T, F, I>(&self, items: Vec<I>, operation: F) -> Vec<T>
    where
        T: Send,
        F: Fn(&[I]) -> Vec<T> + Sync + Send,
        I: Send + Sync,
    {
        if !self.config.enabled || !self.initialized || items.len() < self.config.min_batch_size {
            // Process as single batch
            return operation(&items);
        }

        let batch_size = self.calculate_optimal_batch_size(items.len());
        let start_time = Instant::now();

        let result: Vec<T> = items
            .chunks(batch_size)
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(operation)
            .collect();

        let duration = start_time.elapsed();
        if duration > Duration::from_millis(100) {
            println!(
                "Batched parallel operation completed in {duration:?} with batch size {batch_size}"
            );
        }

        result
    }

    /// Execute work-stealing parallel operation with load balancing
    #[cfg(feature = "parallel-processing")]
    pub fn execute_work_stealing<T, F, I>(&self, items: Vec<I>, operation: F) -> Vec<T>
    where
        T: Send,
        F: Fn(I) -> T + Sync + Send,
        I: Send,
    {
        if !self.config.enabled || !self.initialized {
            return items.into_iter().map(operation).collect();
        }

        use rayon::prelude::*;

        let start_time = Instant::now();

        // Use rayon's work-stealing algorithm for optimal load balancing
        let result: Vec<T> = items.into_par_iter().map(operation).collect();

        let duration = start_time.elapsed();
        if duration > Duration::from_millis(100) {
            println!("Work-stealing parallel operation completed in {duration:?}");
        }

        result
    }

    /// Execute parallel operation with configurable strategy selection
    #[cfg(feature = "parallel-processing")]
    pub fn execute_with_strategy<T, F, I>(
        &self,
        items: Vec<I>,
        operation: F,
        strategy: ProcessingStrategy,
    ) -> Vec<T>
    where
        T: Send,
        F: Fn(I) -> T + Sync + Send + Clone,
        I: Send + Sync + Clone,
    {
        match strategy {
            ProcessingStrategy::Sequential => items.into_iter().map(operation).collect(),
            ProcessingStrategy::ParallelMap => self.execute_parallel(items, operation),
            ProcessingStrategy::ChunkedParallel => {
                let batch_operation = |chunk: &[I]| -> Vec<T> {
                    chunk.iter().cloned().map(&operation).collect()
                };
                self.execute_batched_parallel(items, batch_operation)
            }
            ProcessingStrategy::WorkStealing => self.execute_work_stealing(items, operation),
        }
    }

    /// Execute with automatic strategy selection based on workload
    #[cfg(feature = "parallel-processing")]
    pub fn execute_adaptive<T, F, I>(
        &self,
        items: Vec<I>,
        operation: F,
        complexity: ItemComplexity,
    ) -> Vec<T>
    where
        T: Send,
        F: Fn(I) -> T + Sync + Send + Clone,
        I: Send + Sync + Clone,
    {
        let strategy = ProcessingStrategy::determine_strategy(items.len(), complexity, &self.config);
        self.execute_with_strategy(items, operation, strategy)
    }

    /// Execute sequential operation with batching (when parallel feature is disabled)
    #[cfg(not(feature = "parallel-processing"))]
    pub fn execute_batched_parallel<T, F, I>(&self, items: Vec<I>, operation: F) -> Vec<T>
    where
        T: Send,
        F: Fn(&[I]) -> Vec<T> + Sync + Send,
        I: Send + Sync,
    {
        let start_time = Instant::now();
        let result = operation(&items);
        let duration = start_time.elapsed();

        if duration > Duration::from_millis(100) {
            println!("Sequential batched operation completed in {duration:?}");
        }

        result
    }

    /// Calculate optimal batch size based on workload and configuration
    #[cfg(feature = "parallel-processing")]
    fn calculate_optimal_batch_size(&self, total_items: usize) -> usize {
        let num_threads = rayon::current_num_threads();
        let base_batch_size = (total_items / num_threads).max(self.config.min_batch_size);

        // Adjust based on workload characteristics
        if total_items > 10000 {
            base_batch_size.min(self.config.chunk_batch_size * 2)
        } else {
            base_batch_size.min(self.config.chunk_batch_size)
        }
    }

    /// Calculate optimal batch size (sequential fallback)
    #[cfg(not(feature = "parallel-processing"))]
    fn calculate_optimal_batch_size(&self, total_items: usize) -> usize {
        total_items.min(self.config.chunk_batch_size)
    }

    /// Check if parallel processing should be used for the given workload
    pub fn should_use_parallel(&self, item_count: usize) -> bool {
        self.config.enabled && self.initialized && item_count >= self.config.min_batch_size
    }

    /// Get parallel processing statistics
    pub fn get_statistics(&self) -> ParallelStatistics {
        ParallelStatistics {
            enabled: self.config.enabled,
            initialized: self.initialized,
            num_threads: if self.initialized {
                rayon::current_num_threads()
            } else {
                0
            },
            config: self.config.clone(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }
}

/// Statistics about parallel processing
#[derive(Debug)]
pub struct ParallelStatistics {
    pub enabled: bool,
    pub initialized: bool,
    pub num_threads: usize,
    pub config: ParallelConfig,
}

impl ParallelStatistics {
    /// Print parallel processing statistics
    pub fn print(&self) {
        println!("Parallel Processing Statistics:");
        println!("  Enabled: {}", self.enabled);
        println!("  Initialized: {}", self.initialized);
        if self.initialized {
            println!("  Active threads: {}", self.num_threads);
        }
        println!("  Configuration:");
        println!(
            "    Target threads: {}",
            if self.config.num_threads == 0 {
                "auto".to_string()
            } else {
                self.config.num_threads.to_string()
            }
        );
        println!("    Min batch size: {}", self.config.min_batch_size);
        println!("    Chunk batch size: {}", self.config.chunk_batch_size);
        println!(
            "    Parallel embeddings: {}",
            self.config.parallel_embeddings
        );
        println!("    Parallel graph ops: {}", self.config.parallel_graph_ops);
        println!(
            "    Parallel vector ops: {}",
            self.config.parallel_vector_ops
        );
    }
}

/// Performance monitoring utilities
#[derive(Debug)]
pub struct PerformanceMonitor {
    operation_count: Arc<AtomicUsize>,
    total_duration: Arc<std::sync::Mutex<Duration>>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            operation_count: Arc::new(AtomicUsize::new(0)),
            total_duration: Arc::new(std::sync::Mutex::new(Duration::ZERO)),
        }
    }

    pub fn time_operation<T, F>(&self, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        self.operation_count.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut total) = self.total_duration.lock() {
            *total += duration;
        }

        result
    }

    pub fn get_stats(&self) -> (usize, Duration) {
        let count = self.operation_count.load(Ordering::Relaxed);
        let total = self
            .total_duration
            .lock()
            .map(|d| *d)
            .unwrap_or(Duration::ZERO);
        (count, total)
    }

    pub fn average_duration(&self) -> Duration {
        let (count, total) = self.get_stats();
        if count > 0 {
            total / count as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn reset(&self) {
        self.operation_count.store(0, Ordering::Relaxed);
        if let Ok(mut total) = self.total_duration.lock() {
            *total = Duration::ZERO;
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel processing strategies for different workload types
#[derive(Debug, Clone, Copy)]
pub enum ProcessingStrategy {
    /// Sequential processing (no parallelism)
    Sequential,
    /// Simple parallel mapping
    ParallelMap,
    /// Chunked parallel processing with optimal batch sizes
    ChunkedParallel,
    /// Work-stealing parallel processing
    WorkStealing,
}

impl ProcessingStrategy {
    /// Determine optimal strategy based on workload characteristics
    pub fn determine_strategy(
        item_count: usize,
        item_complexity: ItemComplexity,
        config: &ParallelConfig,
    ) -> Self {
        if !config.enabled || item_count < config.min_batch_size {
            return Self::Sequential;
        }

        match item_complexity {
            ItemComplexity::Low if item_count < 1000 => Self::ParallelMap,
            ItemComplexity::Low => Self::ChunkedParallel,
            ItemComplexity::Medium | ItemComplexity::High => Self::WorkStealing,
        }
    }
}

/// Complexity classification for workload items
#[derive(Debug, Clone, Copy)]
pub enum ItemComplexity {
    /// Simple operations (< 1ms per item)
    Low,
    /// Medium operations (1-10ms per item)
    Medium,
    /// Complex operations (> 10ms per item)
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_processor_creation() {
        let config = ParallelConfig {
            num_threads: 2,
            enabled: true,
            min_batch_size: 5,
            chunk_batch_size: 50,
            parallel_embeddings: true,
            parallel_graph_ops: true,
            parallel_vector_ops: true,
        };

        let processor = ParallelProcessor::new(config);
        assert!(processor.config.enabled);
        assert_eq!(processor.config.num_threads, 2);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();

        let result = monitor.time_operation(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        let (count, total) = monitor.get_stats();
        assert_eq!(count, 1);
        assert!(total >= Duration::from_millis(10));
    }

    #[test]
    fn test_strategy_determination() {
        let config = ParallelConfig {
            num_threads: 4,
            enabled: true,
            min_batch_size: 10,
            chunk_batch_size: 100,
            parallel_embeddings: true,
            parallel_graph_ops: true,
            parallel_vector_ops: true,
        };

        // Small workload should be sequential
        let strategy = ProcessingStrategy::determine_strategy(5, ItemComplexity::Low, &config);
        matches!(strategy, ProcessingStrategy::Sequential);

        // Large, simple workload should use chunked parallel
        let strategy = ProcessingStrategy::determine_strategy(5000, ItemComplexity::Low, &config);
        matches!(strategy, ProcessingStrategy::ChunkedParallel);

        // Complex workload should use work-stealing
        let strategy = ProcessingStrategy::determine_strategy(100, ItemComplexity::High, &config);
        matches!(strategy, ProcessingStrategy::WorkStealing);
    }
}
