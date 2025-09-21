use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Pipeline stage for performance tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Query expansion stage
    QueryExpansion,
    /// Hybrid retrieval stage
    HybridRetrieval,
    /// BM25 keyword search
    BM25Search,
    /// Vector semantic search
    VectorSearch,
    /// Result fusion
    ResultFusion,
    /// Cross-encoder reranking
    Reranking,
    /// Confidence filtering
    ConfidenceFiltering,
    /// Total pipeline execution
    TotalPipeline,
}

/// Detailed timing breakdown for a query
#[derive(Debug, Clone)]
pub struct TimingBreakdown {
    /// Time spent in each pipeline stage
    pub stage_timings: HashMap<PipelineStage, Duration>,
    /// Total query processing time
    pub total_time: Duration,
    /// Number of expanded queries generated
    pub expanded_query_count: usize,
    /// Number of results retrieved before filtering
    pub raw_result_count: usize,
    /// Number of results after filtering
    pub final_result_count: usize,
    /// Average confidence score of final results
    pub average_confidence: f32,
}

impl TimingBreakdown {
    /// Create a new timing breakdown
    pub fn new() -> Self {
        Self {
            stage_timings: HashMap::new(),
            total_time: Duration::new(0, 0),
            expanded_query_count: 0,
            raw_result_count: 0,
            final_result_count: 0,
            average_confidence: 0.0,
        }
    }

    /// Add timing for a specific stage
    pub fn add_stage_timing(&mut self, stage: PipelineStage, duration: Duration) {
        self.stage_timings.insert(stage, duration);
    }

    /// Get timing for a specific stage
    pub fn get_stage_timing(&self, stage: &PipelineStage) -> Option<Duration> {
        self.stage_timings.get(stage).cloned()
    }

    /// Calculate percentage of total time for each stage
    pub fn get_stage_percentages(&self) -> HashMap<PipelineStage, f32> {
        let mut percentages = HashMap::new();
        let total_ms = self.total_time.as_millis() as f32;

        if total_ms > 0.0 {
            for (stage, duration) in &self.stage_timings {
                let stage_ms = duration.as_millis() as f32;
                let percentage = (stage_ms / total_ms) * 100.0;
                percentages.insert(stage.clone(), percentage);
            }
        }

        percentages
    }

    /// Print detailed breakdown
    pub fn print(&self) {
        println!("Query Performance Breakdown:");
        println!("  Total time: {:?}", self.total_time);
        println!("  Expanded queries: {}", self.expanded_query_count);
        println!("  Raw results: {}", self.raw_result_count);
        println!("  Final results: {}", self.final_result_count);
        println!("  Average confidence: {:.3}", self.average_confidence);
        println!();

        let percentages = self.get_stage_percentages();
        println!("  Stage timings:");

        for (stage, duration) in &self.stage_timings {
            let percentage = percentages.get(stage).unwrap_or(&0.0);
            println!("    {stage:?}: {duration:?} ({percentage:.1}%)");
        }
    }
}

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for the GraphRAG system
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Retrieval time in milliseconds
    pub retrieval_time_ms: u64,
    /// Reranking time in milliseconds
    pub reranking_time_ms: u64,
    /// Total query processing time in milliseconds
    pub total_time_ms: u64,
    /// Number of results returned
    pub results_count: usize,
    /// Confidence scores of results
    pub confidence_scores: Vec<f32>,
    /// Query expansion metrics
    pub expansion_metrics: ExpansionMetrics,
    /// Fusion metrics
    pub fusion_metrics: FusionMetrics,
    /// Filtering metrics
    pub filtering_metrics: FilteringMetrics,
}

/// Metrics for query expansion
#[derive(Debug, Clone)]
pub struct ExpansionMetrics {
    /// Number of expanded queries generated
    pub expanded_count: usize,
    /// Time spent on expansion
    pub expansion_time_ms: u64,
    /// Types of expansions used
    pub expansion_types: HashMap<String, usize>,
}

/// Metrics for result fusion
#[derive(Debug, Clone)]
pub struct FusionMetrics {
    /// Number of semantic results
    pub semantic_results: usize,
    /// Number of keyword results
    pub keyword_results: usize,
    /// Number of fused results
    pub fused_results: usize,
    /// Fusion method used
    pub fusion_method: String,
}

/// Metrics for confidence filtering
#[derive(Debug, Clone)]
pub struct FilteringMetrics {
    /// Number of results before filtering
    pub input_count: usize,
    /// Number of results after filtering
    pub output_count: usize,
    /// Filter pass rate as percentage
    pub pass_rate: f32,
    /// Average confidence of filtered results
    pub average_confidence: f32,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            retrieval_time_ms: 0,
            reranking_time_ms: 0,
            total_time_ms: 0,
            results_count: 0,
            confidence_scores: Vec::new(),
            expansion_metrics: ExpansionMetrics {
                expanded_count: 0,
                expansion_time_ms: 0,
                expansion_types: HashMap::new(),
            },
            fusion_metrics: FusionMetrics {
                semantic_results: 0,
                keyword_results: 0,
                fused_results: 0,
                fusion_method: "unknown".to_string(),
            },
            filtering_metrics: FilteringMetrics {
                input_count: 0,
                output_count: 0,
                pass_rate: 0.0,
                average_confidence: 0.0,
            },
        }
    }

    /// Log performance metrics to console
    pub fn log_performance(&self, query: &str) {
        println!("🔍 Query Performance Report");
        println!("{}", "=".repeat(50));
        println!("Query: {query}");
        println!("Total time: {}ms", self.total_time_ms);
        println!("Retrieval: {}ms", self.retrieval_time_ms);
        println!("Reranking: {}ms", self.reranking_time_ms);
        println!("Results count: {}", self.results_count);

        if !self.confidence_scores.is_empty() {
            println!("Average confidence: {:.2}", self.average_confidence());
            println!("Min confidence: {:.2}", self.min_confidence());
            println!("Max confidence: {:.2}", self.max_confidence());
        }

        println!();
        println!("📈 Detailed Breakdown:");

        // Expansion metrics
        println!("  Query Expansion:");
        println!(
            "    Expanded queries: {}",
            self.expansion_metrics.expanded_count
        );
        println!(
            "    Expansion time: {}ms",
            self.expansion_metrics.expansion_time_ms
        );

        // Fusion metrics
        println!("  Result Fusion:");
        println!(
            "    Semantic results: {}",
            self.fusion_metrics.semantic_results
        );
        println!(
            "    Keyword results: {}",
            self.fusion_metrics.keyword_results
        );
        println!("    Fused results: {}", self.fusion_metrics.fused_results);
        println!("    Fusion method: {}", self.fusion_metrics.fusion_method);

        // Filtering metrics
        println!("  Confidence Filtering:");
        println!("    Input count: {}", self.filtering_metrics.input_count);
        println!("    Output count: {}", self.filtering_metrics.output_count);
        println!("    Pass rate: {:.1}%", self.filtering_metrics.pass_rate);
        println!(
            "    Avg confidence: {:.3}",
            self.filtering_metrics.average_confidence
        );
    }

    /// Calculate average confidence score
    pub fn average_confidence(&self) -> f32 {
        if self.confidence_scores.is_empty() {
            0.0
        } else {
            self.confidence_scores.iter().sum::<f32>() / self.confidence_scores.len() as f32
        }
    }

    /// Get minimum confidence score
    pub fn min_confidence(&self) -> f32 {
        self.confidence_scores
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Get maximum confidence score
    pub fn max_confidence(&self) -> f32 {
        self.confidence_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Calculate throughput in queries per second
    pub fn throughput(&self) -> f32 {
        if self.total_time_ms > 0 {
            1000.0 / self.total_time_ms as f32
        } else {
            0.0
        }
    }

    /// Get performance efficiency score (0-1)
    pub fn efficiency_score(&self) -> f32 {
        let time_efficiency = if self.total_time_ms <= 500 {
            1.0
        } else if self.total_time_ms <= 2000 {
            1.0 - ((self.total_time_ms - 500) as f32 / 1500.0) * 0.5
        } else {
            0.5
        };

        let confidence_efficiency = self.average_confidence();
        let result_efficiency = if self.results_count >= 5 {
            1.0
        } else {
            self.results_count as f32 / 5.0
        };

        (time_efficiency + confidence_efficiency + result_efficiency) / 3.0
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics collector for tracking performance across queries
pub struct MetricsCollector {
    /// Historical metrics for queries
    query_metrics: Vec<(String, PerformanceMetrics)>,
    /// Current session start time
    session_start: Instant,
    /// Aggregate statistics
    aggregate_stats: AggregateStats,
}

/// Aggregate statistics across all queries
#[derive(Debug, Clone)]
pub struct AggregateStats {
    /// Total queries processed
    pub total_queries: usize,
    /// Average query time
    pub avg_query_time_ms: f32,
    /// Average results per query
    pub avg_results_count: f32,
    /// Overall average confidence
    pub avg_confidence: f32,
    /// Query time percentiles
    pub time_percentiles: HashMap<String, u64>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            query_metrics: Vec::new(),
            session_start: Instant::now(),
            aggregate_stats: AggregateStats {
                total_queries: 0,
                avg_query_time_ms: 0.0,
                avg_results_count: 0.0,
                avg_confidence: 0.0,
                time_percentiles: HashMap::new(),
            },
        }
    }

    /// Record metrics for a query
    pub fn record_query(&mut self, query: String, metrics: PerformanceMetrics) {
        self.query_metrics.push((query, metrics));
        self.update_aggregate_stats();
    }

    /// Get metrics for recent queries
    pub fn get_recent_metrics(&self, count: usize) -> &[(String, PerformanceMetrics)] {
        let start_idx = if self.query_metrics.len() > count {
            self.query_metrics.len() - count
        } else {
            0
        };
        &self.query_metrics[start_idx..]
    }

    /// Get aggregate statistics
    pub fn get_aggregate_stats(&self) -> &AggregateStats {
        &self.aggregate_stats
    }

    /// Update aggregate statistics
    fn update_aggregate_stats(&mut self) {
        if self.query_metrics.is_empty() {
            return;
        }

        let total = self.query_metrics.len();

        // Calculate averages
        let total_time: u64 = self
            .query_metrics
            .iter()
            .map(|(_, m)| m.total_time_ms)
            .sum();
        let total_results: usize = self
            .query_metrics
            .iter()
            .map(|(_, m)| m.results_count)
            .sum();
        let total_confidence: f32 = self
            .query_metrics
            .iter()
            .map(|(_, m)| m.average_confidence())
            .sum();

        self.aggregate_stats.total_queries = total;
        self.aggregate_stats.avg_query_time_ms = total_time as f32 / total as f32;
        self.aggregate_stats.avg_results_count = total_results as f32 / total as f32;
        self.aggregate_stats.avg_confidence = total_confidence / total as f32;

        // Calculate time percentiles
        let mut times: Vec<u64> = self
            .query_metrics
            .iter()
            .map(|(_, m)| m.total_time_ms)
            .collect();
        times.sort();

        if !times.is_empty() {
            self.aggregate_stats
                .time_percentiles
                .insert("p50".to_string(), times[times.len() / 2]);
            self.aggregate_stats
                .time_percentiles
                .insert("p90".to_string(), times[times.len() * 9 / 10]);
            self.aggregate_stats
                .time_percentiles
                .insert("p95".to_string(), times[times.len() * 95 / 100]);
            self.aggregate_stats
                .time_percentiles
                .insert("p99".to_string(), times[times.len() * 99 / 100]);
        }
    }

    /// Print session summary
    pub fn print_session_summary(&self) {
        let session_duration = self.session_start.elapsed();

        println!("📊 GraphRAG Session Summary");
        println!("{}", "=".repeat(50));
        println!("Session duration: {session_duration:?}");
        println!("Total queries: {}", self.aggregate_stats.total_queries);

        if self.aggregate_stats.total_queries > 0 {
            println!(
                "Average query time: {:.1}ms",
                self.aggregate_stats.avg_query_time_ms
            );
            println!(
                "Average results per query: {:.1}",
                self.aggregate_stats.avg_results_count
            );
            println!(
                "Average confidence: {:.3}",
                self.aggregate_stats.avg_confidence
            );

            println!();
            println!("⏱️  Query Time Percentiles:");
            for (percentile, time) in &self.aggregate_stats.time_percentiles {
                println!("  {percentile}: {time}ms");
            }

            // Calculate queries per minute
            let queries_per_minute = if session_duration.as_secs() > 0 {
                (self.aggregate_stats.total_queries as f32 * 60.0)
                    / session_duration.as_secs() as f32
            } else {
                0.0
            };

            println!();
            println!("🚀 Performance Insights:");
            println!("  Queries per minute: {queries_per_minute:.1}");

            // Performance recommendations
            if self.aggregate_stats.avg_query_time_ms > 1000.0 {
                println!("  ⚠️  Consider optimizing - queries are taking >1s on average");
            } else if self.aggregate_stats.avg_query_time_ms < 200.0 {
                println!("  ✅ Excellent performance - queries under 200ms");
            }

            if self.aggregate_stats.avg_confidence < 0.6 {
                println!("  ⚠️  Low confidence scores - consider adjusting filters");
            } else if self.aggregate_stats.avg_confidence > 0.8 {
                println!("  ✅ High confidence results");
            }
        }
    }

    /// Export metrics to JSON
    pub fn export_to_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut json_data = json::JsonValue::new_object();

        json_data["session_start"] = self.session_start.elapsed().as_secs().into();
        json_data["total_queries"] = self.aggregate_stats.total_queries.into();
        json_data["aggregate_stats"] = json::object! {
            "avg_query_time_ms" => self.aggregate_stats.avg_query_time_ms,
            "avg_results_count" => self.aggregate_stats.avg_results_count,
            "avg_confidence" => self.aggregate_stats.avg_confidence,
            "time_percentiles" => json::JsonValue::new_object()
        };

        // Add percentiles
        for (percentile, time) in &self.aggregate_stats.time_percentiles {
            json_data["aggregate_stats"]["time_percentiles"][percentile] = (*time).into();
        }

        // Add recent queries (last 10)
        let mut recent_queries = json::JsonValue::new_array();
        for (query, metrics) in self.get_recent_metrics(10) {
            let query_data = json::object! {
                "query" => query.clone(),
                "total_time_ms" => metrics.total_time_ms,
                "results_count" => metrics.results_count,
                "avg_confidence" => metrics.average_confidence()
            };
            recent_queries.push(query_data)?;
        }
        json_data["recent_queries"] = recent_queries;

        Ok(json_data.dump())
    }

    /// Clear all collected metrics
    pub fn clear(&mut self) {
        self.query_metrics.clear();
        self.session_start = Instant::now();
        self.aggregate_stats = AggregateStats {
            total_queries: 0,
            avg_query_time_ms: 0.0,
            avg_results_count: 0.0,
            avg_confidence: 0.0,
            time_percentiles: HashMap::new(),
        };
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer utility for measuring performance
pub struct PerformanceTimer {
    start_time: Instant,
    stage_timers: HashMap<PipelineStage, Instant>,
}

impl PerformanceTimer {
    /// Start a new performance timer
    pub fn start() -> Self {
        Self {
            start_time: Instant::now(),
            stage_timers: HashMap::new(),
        }
    }

    /// Start timing a specific stage
    pub fn start_stage(&mut self, stage: PipelineStage) {
        self.stage_timers.insert(stage, Instant::now());
    }

    /// End timing a specific stage and return duration
    pub fn end_stage(&mut self, stage: PipelineStage) -> Duration {
        if let Some(start_time) = self.stage_timers.remove(&stage) {
            start_time.elapsed()
        } else {
            Duration::new(0, 0)
        }
    }

    /// Get total elapsed time
    pub fn total_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Create timing breakdown from collected measurements
    pub fn create_breakdown(
        &self,
        stage_durations: HashMap<PipelineStage, Duration>,
    ) -> TimingBreakdown {
        let mut breakdown = TimingBreakdown::new();
        breakdown.total_time = self.total_elapsed();
        breakdown.stage_timings = stage_durations;
        breakdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_metrics_creation() {
        let metrics = PerformanceMetrics::new();
        assert_eq!(metrics.results_count, 0);
        assert_eq!(metrics.total_time_ms, 0);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        let metrics = PerformanceMetrics {
            total_time_ms: 100,
            results_count: 5,
            confidence_scores: vec![0.8, 0.9, 0.7],
            ..Default::default()
        };

        collector.record_query("test query".to_string(), metrics);

        let stats = collector.get_aggregate_stats();
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.avg_query_time_ms, 100.0);
    }

    #[test]
    fn test_timing_breakdown() {
        let mut breakdown = TimingBreakdown::new();
        breakdown.add_stage_timing(PipelineStage::QueryExpansion, Duration::from_millis(50));
        breakdown.add_stage_timing(PipelineStage::HybridRetrieval, Duration::from_millis(100));
        breakdown.total_time = Duration::from_millis(200);

        let percentages = breakdown.get_stage_percentages();
        assert_eq!(percentages.get(&PipelineStage::QueryExpansion), Some(&25.0));
        assert_eq!(
            percentages.get(&PipelineStage::HybridRetrieval),
            Some(&50.0)
        );
    }

    #[test]
    fn test_performance_timer() {
        let mut timer = PerformanceTimer::start();

        timer.start_stage(PipelineStage::QueryExpansion);
        thread::sleep(Duration::from_millis(10));
        let duration = timer.end_stage(PipelineStage::QueryExpansion);

        assert!(duration.as_millis() >= 10);
        assert!(timer.total_elapsed().as_millis() >= 10);
    }

    #[test]
    fn test_efficiency_score() {
        let mut metrics = PerformanceMetrics::new();
        metrics.total_time_ms = 200;
        metrics.results_count = 8;
        metrics.confidence_scores = vec![0.9, 0.8, 0.85];

        let efficiency = metrics.efficiency_score();
        assert!(efficiency > 0.8); // Should be high with good performance
    }
}
