//! Performance benchmarking utilities for GraphRAG components

use crate::{
    config::ParallelConfig, parallel::ParallelProcessor, retrieval::RetrievalSystem,
    text::TextProcessor, vector::EmbeddingGenerator, Result,
};
use std::time::{Duration, Instant};

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub sequential_time: Duration,
    pub parallel_time: Option<Duration>,
    pub speedup_factor: Option<f64>,
    pub throughput_items_per_second: f64,
    pub memory_usage_estimate: Option<usize>,
}

/// Configuration for performance benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of iterations to average over
    pub iterations: usize,
    /// Whether to include parallel benchmarks
    pub include_parallel: bool,
    /// Warmup iterations before measuring
    pub warmup_iterations: usize,
    /// Memory profiling enabled
    pub profile_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 5,
            include_parallel: true,
            warmup_iterations: 2,
            profile_memory: false,
        }
    }
}

/// Performance benchmarking system
pub struct PerformanceBenchmark {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl PerformanceBenchmark {
    /// Create a new performance benchmark
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Benchmark text processing operations
    pub fn benchmark_text_processing(&mut self, test_data: &[String]) -> Result<BenchmarkResult> {
        let text_processor = TextProcessor::new(1000, 100)?;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for text in test_data.iter().take(5) {
                let _ = text_processor.extract_keywords(text, 10);
            }
        }

        // Sequential benchmark
        let start = Instant::now();
        for _ in 0..self.config.iterations {
            for text in test_data {
                let _ = text_processor.extract_keywords(text, 10);
            }
        }
        let sequential_time = start.elapsed() / self.config.iterations as u32;

        // Parallel benchmark (if configured)
        let parallel_time = if self.config.include_parallel {
            let parallel_config = ParallelConfig {
                num_threads: 0, // Auto-detect
                enabled: true,
                min_batch_size: 100,
                chunk_batch_size: 1000,
                parallel_embeddings: true,
                parallel_graph_ops: true,
                parallel_vector_ops: true,
            };
            let parallel_processor = ParallelProcessor::new(parallel_config);
            let text_processor_parallel =
                TextProcessor::with_parallel_processing(1000, 100, parallel_processor)?;

            let start = Instant::now();
            for _ in 0..self.config.iterations {
                let text_refs: Vec<&str> = test_data.iter().map(|s| s.as_str()).collect();
                let _ = text_processor_parallel.batch_extract_keywords(&text_refs, 10);
            }
            Some(start.elapsed() / self.config.iterations as u32)
        } else {
            None
        };

        let speedup_factor =
            parallel_time.map(|pt| sequential_time.as_secs_f64() / pt.as_secs_f64());
        let throughput = test_data.len() as f64 / sequential_time.as_secs_f64();

        let result = BenchmarkResult {
            operation_name: "Text Processing (Keyword Extraction)".to_string(),
            sequential_time,
            parallel_time,
            speedup_factor,
            throughput_items_per_second: throughput,
            memory_usage_estimate: None,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Benchmark embedding generation
    pub fn benchmark_embeddings(&mut self, test_texts: &[String]) -> Result<BenchmarkResult> {
        let mut embedding_generator = EmbeddingGenerator::new(128); // 128-dimensional embeddings

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for text in test_texts.iter().take(3) {
                let _ = embedding_generator.generate_embedding(text);
            }
        }

        // Sequential benchmark
        let start = Instant::now();
        for _ in 0..self.config.iterations {
            for text in test_texts {
                let _ = embedding_generator.generate_embedding(text);
            }
        }
        let sequential_time = start.elapsed() / self.config.iterations as u32;

        // Parallel benchmark (simulated with sequential calls for now)
        let parallel_time = if self.config.include_parallel {
            let start = Instant::now();
            for _ in 0..self.config.iterations {
                for text in test_texts {
                    let _ = embedding_generator.generate_embedding(text);
                }
            }
            Some(start.elapsed() / self.config.iterations as u32)
        } else {
            None
        };

        let speedup_factor =
            parallel_time.map(|pt| sequential_time.as_secs_f64() / pt.as_secs_f64());
        let throughput = test_texts.len() as f64 / sequential_time.as_secs_f64();

        let result = BenchmarkResult {
            operation_name: "Embedding Generation".to_string(),
            sequential_time,
            parallel_time,
            speedup_factor,
            throughput_items_per_second: throughput,
            memory_usage_estimate: Some(test_texts.len() * 384 * 4), // Estimate for 384-dim float embeddings
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Benchmark retrieval operations
    pub fn benchmark_retrieval(
        &mut self,
        retrieval_system: &mut RetrievalSystem,
        test_queries: &[String],
    ) -> Result<BenchmarkResult> {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for query in test_queries.iter().take(3) {
                let _ = retrieval_system.vector_search(query, 10);
            }
        }

        // Vector search benchmark
        let start = Instant::now();
        for _ in 0..self.config.iterations {
            for query in test_queries {
                let _ = retrieval_system.vector_search(query, 10);
            }
        }
        let vector_time = start.elapsed() / self.config.iterations as u32;

        // Hybrid search benchmark
        let start = Instant::now();
        for _ in 0..self.config.iterations {
            for query in test_queries {
                let _ = retrieval_system.query(query);
            }
        }
        let hybrid_time = start.elapsed() / self.config.iterations as u32;

        let throughput = test_queries.len() as f64 / vector_time.as_secs_f64();
        let hybrid_speedup = vector_time.as_secs_f64() / hybrid_time.as_secs_f64();

        let result = BenchmarkResult {
            operation_name: "Retrieval Operations".to_string(),
            sequential_time: vector_time,
            parallel_time: Some(hybrid_time),
            speedup_factor: Some(hybrid_speedup),
            throughput_items_per_second: throughput,
            memory_usage_estimate: None,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Run comprehensive benchmark suite
    pub fn run_comprehensive_benchmark(
        &mut self,
        mut test_data: BenchmarkData,
    ) -> Result<BenchmarkSummary> {
        println!("🚀 Starting Comprehensive Performance Benchmark");
        println!("{}", "=".repeat(60));

        let mut benchmark_results = Vec::new();

        // Text processing benchmark
        if !test_data.text_samples.is_empty() {
            println!("📝 Benchmarking text processing...");
            let result = self.benchmark_text_processing(&test_data.text_samples)?;
            benchmark_results.push(result);
        }

        // Embedding benchmark
        if !test_data.embedding_texts.is_empty() {
            println!("🧮 Benchmarking embedding generation...");
            let result = self.benchmark_embeddings(&test_data.embedding_texts)?;
            benchmark_results.push(result);
        }

        // Retrieval benchmark
        if let Some(ref mut retrieval_system) = test_data.retrieval_system {
            if !test_data.query_samples.is_empty() {
                println!("🔍 Benchmarking retrieval operations...");
                let result =
                    self.benchmark_retrieval(retrieval_system, &test_data.query_samples)?;
                benchmark_results.push(result);
            }
        }

        let summary = BenchmarkSummary::new(benchmark_results);
        summary.print_summary();

        Ok(summary)
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Clear benchmark results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }
}

impl Default for PerformanceBenchmark {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

/// Test data for benchmarking
pub struct BenchmarkData<'a> {
    pub text_samples: Vec<String>,
    pub embedding_texts: Vec<String>,
    pub query_samples: Vec<String>,
    pub retrieval_system: Option<&'a mut RetrievalSystem>,
}

impl<'a> Default for BenchmarkData<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> BenchmarkData<'a> {
    pub fn new() -> Self {
        Self {
            text_samples: Vec::new(),
            embedding_texts: Vec::new(),
            query_samples: Vec::new(),
            retrieval_system: None,
        }
    }

    /// Add text samples for processing benchmarks
    pub fn with_text_samples(mut self, samples: Vec<String>) -> Self {
        self.text_samples = samples;
        self
    }

    /// Add embedding texts for embedding benchmarks
    pub fn with_embedding_texts(mut self, texts: Vec<String>) -> Self {
        self.embedding_texts = texts;
        self
    }

    /// Add query samples for retrieval benchmarks
    pub fn with_query_samples(mut self, queries: Vec<String>) -> Self {
        self.query_samples = queries;
        self
    }

    /// Add retrieval system for retrieval benchmarks
    pub fn with_retrieval_system(mut self, system: &'a mut RetrievalSystem) -> Self {
        self.retrieval_system = Some(system);
        self
    }
}

/// Summary of benchmark results
#[derive(Debug)]
pub struct BenchmarkSummary {
    pub results: Vec<BenchmarkResult>,
    pub total_operations: usize,
    pub average_sequential_time: Duration,
    pub average_parallel_speedup: Option<f64>,
    pub performance_recommendations: Vec<String>,
}

impl BenchmarkSummary {
    pub fn new(results: Vec<BenchmarkResult>) -> Self {
        let total_operations = results.len();

        let total_sequential_millis: u64 = results
            .iter()
            .map(|r| r.sequential_time.as_millis() as u64)
            .sum();

        let average_sequential_time = Duration::from_millis(if total_operations > 0 {
            total_sequential_millis / total_operations as u64
        } else {
            0
        });

        let speedups: Vec<f64> = results.iter().filter_map(|r| r.speedup_factor).collect();

        let average_parallel_speedup = if !speedups.is_empty() {
            Some(speedups.iter().sum::<f64>() / speedups.len() as f64)
        } else {
            None
        };

        let performance_recommendations = Self::generate_recommendations(&results);

        Self {
            results,
            total_operations,
            average_sequential_time,
            average_parallel_speedup,
            performance_recommendations,
        }
    }

    fn generate_recommendations(results: &[BenchmarkResult]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for result in results {
            if let Some(speedup) = result.speedup_factor {
                if speedup < 1.5 {
                    recommendations.push(format!(
                        "⚠️  {} shows low parallel speedup ({:.2}x). Consider optimizing parallel strategy.",
                        result.operation_name, speedup
                    ));
                } else if speedup > 3.0 {
                    recommendations.push(format!(
                        "✅ {} shows excellent parallel performance ({:.2}x speedup).",
                        result.operation_name, speedup
                    ));
                }
            }

            if result.throughput_items_per_second < 10.0 {
                recommendations.push(format!(
                    "📈 {} has low throughput ({:.1} items/sec). Consider batch processing.",
                    result.operation_name, result.throughput_items_per_second
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations
                .push("🎯 All operations show good performance characteristics.".to_string());
        }

        recommendations
    }

    pub fn print_summary(&self) {
        println!("\n📊 Benchmark Summary");
        println!("{}", "=".repeat(50));
        println!("Total operations benchmarked: {}", self.total_operations);
        println!(
            "Average sequential time: {:.2}ms",
            self.average_sequential_time.as_millis()
        );

        if let Some(speedup) = self.average_parallel_speedup {
            println!("Average parallel speedup: {speedup:.2}x");
        }

        println!("\n📋 Detailed Results:");
        for result in &self.results {
            println!(
                "  {} - {:.2}ms",
                result.operation_name,
                result.sequential_time.as_millis()
            );
            if let Some(parallel_time) = result.parallel_time {
                println!("    Parallel: {:.2}ms", parallel_time.as_millis());
            }
            if let Some(speedup) = result.speedup_factor {
                println!("    Speedup: {speedup:.2}x");
            }
            println!(
                "    Throughput: {:.1} items/sec",
                result.throughput_items_per_second
            );
        }

        println!("\n💡 Performance Recommendations:");
        for recommendation in &self.performance_recommendations {
            println!("  {recommendation}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_creation() {
        let config = BenchmarkConfig::default();
        let benchmark = PerformanceBenchmark::new(config);
        assert_eq!(benchmark.results.len(), 0);
    }

    #[test]
    fn test_benchmark_data_builder() {
        let data = BenchmarkData::new()
            .with_text_samples(vec!["test".to_string()])
            .with_query_samples(vec!["query".to_string()]);

        assert_eq!(data.text_samples.len(), 1);
        assert_eq!(data.query_samples.len(), 1);
    }

    #[test]
    fn test_benchmark_summary() {
        let results = vec![BenchmarkResult {
            operation_name: "Test".to_string(),
            sequential_time: Duration::from_millis(100),
            parallel_time: Some(Duration::from_millis(50)),
            speedup_factor: Some(2.0),
            throughput_items_per_second: 10.0,
            memory_usage_estimate: None,
        }];

        let summary = BenchmarkSummary::new(results);
        assert_eq!(summary.total_operations, 1);
        assert_eq!(summary.average_sequential_time.as_millis(), 100);
        assert_eq!(summary.average_parallel_speedup, Some(2.0));
    }
}
