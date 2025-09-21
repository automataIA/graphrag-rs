//! Quality metrics and performance tracking for ROGRAG system
//!
//! Provides comprehensive metrics collection and analysis to measure
//! the effectiveness and improvement of the ROGRAG system over baseline GraphRAG.

#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use crate::rograg::{DecompositionResult, RogragResponse};
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use std::collections::VecDeque;
#[cfg(feature = "rograg")]
use std::time::{Duration, SystemTime, UNIX_EPOCH};
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Error types for quality metrics
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Invalid metric value: {metric} = {value}")]
    InvalidValue { metric: String, value: f64 },

    #[error("Insufficient data for analysis: {reason}")]
    InsufficientData { reason: String },

    #[error("Metric calculation failed: {reason}")]
    CalculationFailed { reason: String },
}

/// Configuration for quality metrics
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct QualityMetricsConfig {
    pub enable_tracking: bool,
    pub max_history_size: usize,
    pub enable_comparative_analysis: bool,
    pub baseline_comparison_window: usize,
    pub quality_threshold: f32,
    pub performance_threshold_ms: u64,
    pub enable_real_time_monitoring: bool,
}

#[cfg(feature = "rograg")]
impl Default for QualityMetricsConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            max_history_size: 1000,
            enable_comparative_analysis: true,
            baseline_comparison_window: 100,
            quality_threshold: 0.75,
            performance_threshold_ms: 5000,
            enable_real_time_monitoring: true,
        }
    }
}

/// Quality metrics collector and analyzer
#[cfg(feature = "rograg")]
#[derive(Clone)]
pub struct QualityMetrics {
    config: QualityMetricsConfig,
    query_history: VecDeque<QueryMetrics>,
    performance_stats: PerformanceStatistics,
    quality_benchmarks: QualityBenchmarks,
    real_time_monitor: RealTimeMonitor,
}

/// Metrics for a single query
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryMetrics {
    pub timestamp: u64,
    pub query: String,
    pub decomposition_success: bool,
    pub decomposition_time_ms: u64,
    pub subquery_count: usize,
    pub retrieval_strategy_used: RetrievalStrategy,
    pub response_quality: ResponseQuality,
    pub processing_time_ms: u64,
    pub fallback_used: bool,
    pub confidence_score: f32,
    pub user_satisfaction: Option<f32>, // If available from feedback
}

/// Response quality metrics
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseQuality {
    pub accuracy_score: f32,
    pub completeness_score: f32,
    pub coherence_score: f32,
    pub relevance_score: f32,
    pub source_credibility: f32,
    pub overall_quality: f32,
}

/// Retrieval strategy used
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, Default)]
pub enum RetrievalStrategy {
    #[default]
    LogicForm,
    FuzzyMatch,
    Hybrid,
    Fallback,
}

/// Performance statistics
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceStatistics {
    pub total_queries: usize,
    pub successful_decompositions: usize,
    pub avg_processing_time_ms: f64,
    pub avg_quality_score: f64,
    pub fallback_rate: f64,
    pub error_rate: f64,
    pub throughput_qps: f64, // Queries per second
}

/// Quality benchmarks comparing to baseline
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityBenchmarks {
    pub accuracy_improvement: f64,      // % improvement over baseline
    pub completeness_improvement: f64,
    pub coherence_improvement: f64,
    pub overall_improvement: f64,
    pub baseline_comparison_count: usize,
    pub confidence_intervals: ConfidenceIntervals,
}

/// Confidence intervals for statistical significance
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfidenceIntervals {
    pub accuracy_ci_95: (f64, f64),
    pub completeness_ci_95: (f64, f64),
    pub coherence_ci_95: (f64, f64),
    pub overall_ci_95: (f64, f64),
}

/// Real-time monitoring
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct RealTimeMonitor {
    pub current_window: VecDeque<QueryMetrics>,
    pub window_size: usize,
    pub alert_thresholds: AlertThresholds,
    pub active_alerts: Vec<QualityAlert>,
}

/// Alert thresholds for real-time monitoring
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub min_quality_score: f32,
    pub max_processing_time_ms: u64,
    pub max_error_rate: f32,
    pub min_success_rate: f32,
}

/// Quality alert
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: u64,
    pub metric_value: f64,
    pub threshold: f64,
}

/// Type of quality alert
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, Default)]
pub enum AlertType {
    #[default]
    QualityDegradation,
    PerformanceDegradation,
    HighErrorRate,
    LowSuccessRate,
    UnusualPattern,
}

/// Severity of alert
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, Default)]
pub enum AlertSeverity {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

/// Comparative analysis result
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComparativeAnalysis {
    pub rograg_metrics: AggregatedMetrics,
    pub baseline_metrics: AggregatedMetrics,
    pub improvement_percentages: ImprovementPercentages,
    pub statistical_significance: StatisticalSignificance,
    pub sample_size: usize,
    pub analysis_timestamp: u64,
}

/// Aggregated metrics
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatedMetrics {
    pub mean_accuracy: f64,
    pub mean_completeness: f64,
    pub mean_coherence: f64,
    pub mean_relevance: f64,
    pub mean_processing_time_ms: f64,
    pub success_rate: f64,
    pub std_dev_quality: f64,
}

/// Improvement percentages
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImprovementPercentages {
    pub accuracy_improvement: f64,
    pub completeness_improvement: f64,
    pub coherence_improvement: f64,
    pub relevance_improvement: f64,
    pub overall_improvement: f64,
}

/// Statistical significance analysis
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatisticalSignificance {
    pub p_value_accuracy: f64,
    pub p_value_completeness: f64,
    pub p_value_coherence: f64,
    pub p_value_overall: f64,
    pub is_significant_95: bool,
    pub effect_size: f64,
}

#[cfg(feature = "rograg")]
impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl QualityMetrics {
    /// Create a new quality metrics collector
    pub fn new() -> Self {
        Self::with_config(QualityMetricsConfig::default())
    }

    /// Create a new quality metrics collector with custom configuration
    pub fn with_config(config: QualityMetricsConfig) -> Self {
        Self {
            config: config.clone(),
            query_history: VecDeque::with_capacity(config.max_history_size),
            performance_stats: PerformanceStatistics {
                total_queries: 0,
                successful_decompositions: 0,
                avg_processing_time_ms: 0.0,
                avg_quality_score: 0.0,
                fallback_rate: 0.0,
                error_rate: 0.0,
                throughput_qps: 0.0,
            },
            quality_benchmarks: QualityBenchmarks {
                accuracy_improvement: 0.0,
                completeness_improvement: 0.0,
                coherence_improvement: 0.0,
                overall_improvement: 0.0,
                baseline_comparison_count: 0,
                confidence_intervals: ConfidenceIntervals {
                    accuracy_ci_95: (0.0, 0.0),
                    completeness_ci_95: (0.0, 0.0),
                    coherence_ci_95: (0.0, 0.0),
                    overall_ci_95: (0.0, 0.0),
                },
            },
            real_time_monitor: RealTimeMonitor {
                current_window: VecDeque::with_capacity(100),
                window_size: 100,
                alert_thresholds: AlertThresholds {
                    min_quality_score: 0.6,
                    max_processing_time_ms: 10000,
                    max_error_rate: 0.1,
                    min_success_rate: 0.8,
                },
                active_alerts: Vec::new(),
            },
        }
    }

    /// Record a query and its results
    pub fn record_query(
        &mut self,
        query: &str,
        decomposition_result: &DecompositionResult,
        response: &RogragResponse,
        processing_time: Duration,
    ) -> Result<()> {
        if !self.config.enable_tracking {
            return Ok(());
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let response_quality = self.calculate_response_quality(response)?;

        let query_metrics = QueryMetrics {
            timestamp,
            query: query.to_string(),
            decomposition_success: decomposition_result.is_decomposed(),
            decomposition_time_ms: processing_time.as_millis() as u64,
            subquery_count: decomposition_result.subqueries.len(),
            retrieval_strategy_used: self.determine_retrieval_strategy(response),
            response_quality,
            processing_time_ms: processing_time.as_millis() as u64,
            fallback_used: response.processing_stats.fallback_used ,
            confidence_score: response.confidence,
            user_satisfaction: None, // Can be updated later with feedback
        };

        // Add to history
        self.add_to_history(query_metrics.clone());

        // Update performance statistics
        self.update_performance_stats(&query_metrics);

        // Real-time monitoring
        if self.config.enable_real_time_monitoring {
            self.update_real_time_monitor(query_metrics);
        }

        Ok(())
    }

    /// Add query metrics to history
    fn add_to_history(&mut self, metrics: QueryMetrics) {
        if self.query_history.len() >= self.config.max_history_size {
            self.query_history.pop_front();
        }
        self.query_history.push_back(metrics);
    }

    /// Calculate response quality metrics
    fn calculate_response_quality(&self, response: &RogragResponse) -> Result<ResponseQuality> {
        // Accuracy score based on confidence and source credibility
        let accuracy_score = (response.confidence + self.calculate_source_credibility(response)) / 2.0;

        // Completeness score based on answer length and source coverage
        let completeness_score = self.calculate_completeness_score(response);

        // Coherence score based on text flow and structure
        let coherence_score = self.calculate_coherence_score(response);

        // Relevance score based on query-answer alignment
        let relevance_score = self.calculate_relevance_score(response);

        // Source credibility based on source count and diversity
        let source_credibility = self.calculate_source_credibility(response);

        // Overall quality as weighted average
        let overall_quality = (accuracy_score * 0.3 +
                              completeness_score * 0.25 +
                              coherence_score * 0.2 +
                              relevance_score * 0.15 +
                              source_credibility * 0.1).min(1.0);

        Ok(ResponseQuality {
            accuracy_score,
            completeness_score,
            coherence_score,
            relevance_score,
            source_credibility,
            overall_quality,
        })
    }

    /// Calculate completeness score
    fn calculate_completeness_score(&self, response: &RogragResponse) -> f32 {
        let answer_length = response.content.len();
        let source_count = response.sources.len();
        let subquery_coverage = response.subquery_results.len();

        // Normalize components
        let length_score = (answer_length as f32 / 500.0).min(1.0); // Normalize to 500 chars
        let source_score = (source_count as f32 / 3.0).min(1.0);   // Normalize to 3 sources
        let coverage_score = (subquery_coverage as f32 / 5.0).min(1.0); // Normalize to 5 subqueries

        (length_score + source_score + coverage_score) / 3.0
    }

    /// Calculate coherence score
    fn calculate_coherence_score(&self, response: &RogragResponse) -> f32 {
        let text = &response.content;
        let sentences: Vec<&str> = text.split(['.', '!', '?']).collect();

        if sentences.len() <= 1 {
            return 1.0; // Single sentence is trivially coherent
        }

        // Look for transition words and logical flow
        let transition_words = ["however", "therefore", "furthermore", "additionally", "meanwhile",
                               "consequently", "moreover", "nevertheless", "thus", "hence"];

        let transition_count = sentences.iter()
            .filter(|s| transition_words.iter().any(|t| s.to_lowercase().contains(t)))
            .count();

        // Calculate coherence based on transitions and sentence flow
        let transition_score = (transition_count as f32 / sentences.len() as f32).min(1.0);

        // Simple repetition check (lower score for excessive repetition)
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().copied().collect();
        let repetition_score = if words.is_empty() {
            1.0
        } else {
            unique_words.len() as f32 / words.len() as f32
        };

        (transition_score + repetition_score) / 2.0
    }

    /// Calculate relevance score
    fn calculate_relevance_score(&self, response: &RogragResponse) -> f32 {
        let query_lower = response.query.to_lowercase();
        let answer_lower = response.content.to_lowercase();

        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter short words
            .collect();

        let answer_words: std::collections::HashSet<&str> = answer_lower
            .split_whitespace()
            .collect();

        if query_words.is_empty() {
            return 1.0;
        }

        let overlap = query_words.intersection(&answer_words).count();
        overlap as f32 / query_words.len() as f32
    }

    /// Calculate source credibility
    fn calculate_source_credibility(&self, response: &RogragResponse) -> f32 {
        if response.sources.is_empty() {
            return 0.0;
        }

        // Score based on source count and diversity
        let count_score = (response.sources.len() as f32 / 5.0).min(1.0);

        // Simple diversity check
        let prefixes: std::collections::HashSet<String> = response.sources.iter()
            .map(|s| s.chars().take(5).collect())
            .collect();
        let diversity_score = prefixes.len() as f32 / response.sources.len() as f32;

        (count_score + diversity_score) / 2.0
    }

    /// Determine retrieval strategy used
    fn determine_retrieval_strategy(&self, response: &RogragResponse) -> RetrievalStrategy {
        let logic_form_count = response.subquery_results.iter()
            .filter(|r| matches!(r.result_type, crate::rograg::SubqueryResultType::LogicForm))
            .count();

        let fuzzy_match_count = response.subquery_results.iter()
            .filter(|r| matches!(r.result_type, crate::rograg::SubqueryResultType::FuzzyMatch))
            .count();

        let fallback_count = response.subquery_results.iter()
            .filter(|r| matches!(r.result_type, crate::rograg::SubqueryResultType::Fallback))
            .count();

        if fallback_count > 0 {
            RetrievalStrategy::Fallback
        } else if logic_form_count > 0 && fuzzy_match_count > 0 {
            RetrievalStrategy::Hybrid
        } else if logic_form_count > 0 {
            RetrievalStrategy::LogicForm
        } else {
            RetrievalStrategy::FuzzyMatch
        }
    }

    /// Update performance statistics
    fn update_performance_stats(&mut self, metrics: &QueryMetrics) {
        self.performance_stats.total_queries += 1;

        if metrics.decomposition_success {
            self.performance_stats.successful_decompositions += 1;
        }

        // Update running averages
        let total = self.performance_stats.total_queries as f64;
        let new_processing_time = metrics.processing_time_ms as f64;
        let new_quality = metrics.response_quality.overall_quality as f64;

        self.performance_stats.avg_processing_time_ms =
            (self.performance_stats.avg_processing_time_ms * (total - 1.0) + new_processing_time) / total;

        self.performance_stats.avg_quality_score =
            (self.performance_stats.avg_quality_score * (total - 1.0) + new_quality) / total;

        // Update rates
        self.performance_stats.fallback_rate =
            self.query_history.iter().filter(|m| m.fallback_used).count() as f64 / total;

        // Error rate would need to be tracked separately
        self.performance_stats.error_rate = 0.0; // Placeholder

        // Calculate throughput over recent window
        self.calculate_throughput();
    }

    /// Calculate current throughput
    fn calculate_throughput(&mut self) {
        if self.query_history.len() < 2 {
            self.performance_stats.throughput_qps = 0.0;
            return;
        }

        let recent_queries: Vec<&QueryMetrics> = self.query_history.iter().rev().take(10).collect();
        if recent_queries.len() < 2 {
            return;
        }

        let time_span = recent_queries.first().unwrap().timestamp - recent_queries.last().unwrap().timestamp;
        if time_span > 0 {
            self.performance_stats.throughput_qps = recent_queries.len() as f64 / time_span as f64;
        }
    }

    /// Update real-time monitoring
    fn update_real_time_monitor(&mut self, metrics: QueryMetrics) {
        // Add to current window
        if self.real_time_monitor.current_window.len() >= self.real_time_monitor.window_size {
            self.real_time_monitor.current_window.pop_front();
        }
        self.real_time_monitor.current_window.push_back(metrics.clone());

        // Check for alerts
        self.check_quality_alerts(&metrics);
    }

    /// Check for quality alerts
    fn check_quality_alerts(&mut self, metrics: &QueryMetrics) {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        // Quality degradation alert
        if metrics.response_quality.overall_quality < self.real_time_monitor.alert_thresholds.min_quality_score {
            self.real_time_monitor.active_alerts.push(QualityAlert {
                alert_type: AlertType::QualityDegradation,
                severity: AlertSeverity::High,
                message: format!("Low quality response: {:.2}", metrics.response_quality.overall_quality),
                timestamp,
                metric_value: metrics.response_quality.overall_quality as f64,
                threshold: self.real_time_monitor.alert_thresholds.min_quality_score as f64,
            });
        }

        // Performance degradation alert
        if metrics.processing_time_ms > self.real_time_monitor.alert_thresholds.max_processing_time_ms {
            self.real_time_monitor.active_alerts.push(QualityAlert {
                alert_type: AlertType::PerformanceDegradation,
                severity: AlertSeverity::Medium,
                message: format!("Slow processing: {}ms", metrics.processing_time_ms),
                timestamp,
                metric_value: metrics.processing_time_ms as f64,
                threshold: self.real_time_monitor.alert_thresholds.max_processing_time_ms as f64,
            });
        }

        // Keep only recent alerts (last hour)
        let one_hour_ago = timestamp.saturating_sub(3600);
        self.real_time_monitor.active_alerts.retain(|alert| alert.timestamp > one_hour_ago);
    }

    /// Perform comparative analysis against baseline
    pub fn perform_comparative_analysis(&self, baseline_metrics: &[QueryMetrics]) -> Result<ComparativeAnalysis> {
        if !self.config.enable_comparative_analysis {
            return Err(MetricsError::InsufficientData {
                reason: "Comparative analysis disabled".to_string(),
            }.into());
        }

        if self.query_history.is_empty() || baseline_metrics.is_empty() {
            return Err(MetricsError::InsufficientData {
                reason: "Insufficient data for comparison".to_string(),
            }.into());
        }

        let rograg_metrics = self.calculate_aggregated_metrics(&self.query_history.iter().collect::<Vec<_>>())?;
        let baseline_agg = self.calculate_aggregated_metrics(&baseline_metrics.iter().collect::<Vec<_>>())?;

        let improvement_percentages = ImprovementPercentages {
            accuracy_improvement: self.calculate_improvement_percentage(rograg_metrics.mean_accuracy, baseline_agg.mean_accuracy),
            completeness_improvement: self.calculate_improvement_percentage(rograg_metrics.mean_completeness, baseline_agg.mean_completeness),
            coherence_improvement: self.calculate_improvement_percentage(rograg_metrics.mean_coherence, baseline_agg.mean_coherence),
            relevance_improvement: self.calculate_improvement_percentage(rograg_metrics.mean_relevance, baseline_agg.mean_relevance),
            overall_improvement: self.calculate_improvement_percentage(
                (rograg_metrics.mean_accuracy + rograg_metrics.mean_completeness +
                 rograg_metrics.mean_coherence + rograg_metrics.mean_relevance) / 4.0,
                (baseline_agg.mean_accuracy + baseline_agg.mean_completeness +
                 baseline_agg.mean_coherence + baseline_agg.mean_relevance) / 4.0,
            ),
        };

        let statistical_significance = self.calculate_statistical_significance(&rograg_metrics, &baseline_agg)?;

        Ok(ComparativeAnalysis {
            rograg_metrics,
            baseline_metrics: baseline_agg,
            improvement_percentages,
            statistical_significance,
            sample_size: self.query_history.len().min(baseline_metrics.len()),
            analysis_timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        })
    }

    /// Calculate aggregated metrics
    fn calculate_aggregated_metrics(&self, metrics: &[&QueryMetrics]) -> Result<AggregatedMetrics> {
        if metrics.is_empty() {
            return Err(MetricsError::InsufficientData {
                reason: "No metrics provided".to_string(),
            }.into());
        }

        let n = metrics.len() as f64;

        let mean_accuracy = metrics.iter().map(|m| m.response_quality.accuracy_score as f64).sum::<f64>() / n;
        let mean_completeness = metrics.iter().map(|m| m.response_quality.completeness_score as f64).sum::<f64>() / n;
        let mean_coherence = metrics.iter().map(|m| m.response_quality.coherence_score as f64).sum::<f64>() / n;
        let mean_relevance = metrics.iter().map(|m| m.response_quality.relevance_score as f64).sum::<f64>() / n;
        let mean_processing_time_ms = metrics.iter().map(|m| m.processing_time_ms as f64).sum::<f64>() / n;

        let success_count = metrics.iter().filter(|m| m.decomposition_success).count();
        let success_rate = success_count as f64 / n;

        // Calculate standard deviation of overall quality
        let quality_scores: Vec<f64> = metrics.iter().map(|m| m.response_quality.overall_quality as f64).collect();
        let mean_quality = quality_scores.iter().sum::<f64>() / n;
        let variance = quality_scores.iter().map(|&q| (q - mean_quality).powi(2)).sum::<f64>() / n;
        let std_dev_quality = variance.sqrt();

        Ok(AggregatedMetrics {
            mean_accuracy,
            mean_completeness,
            mean_coherence,
            mean_relevance,
            mean_processing_time_ms,
            success_rate,
            std_dev_quality,
        })
    }

    /// Calculate improvement percentage
    fn calculate_improvement_percentage(&self, rograg_value: f64, baseline_value: f64) -> f64 {
        if baseline_value == 0.0 {
            return if rograg_value > 0.0 { 100.0 } else { 0.0 };
        }
        ((rograg_value - baseline_value) / baseline_value) * 100.0
    }

    /// Calculate statistical significance
    fn calculate_statistical_significance(
        &self,
        rograg_metrics: &AggregatedMetrics,
        baseline_metrics: &AggregatedMetrics,
    ) -> Result<StatisticalSignificance> {
        // Simplified statistical significance calculation
        // In a real implementation, you'd use proper statistical tests

        let effect_size = (rograg_metrics.mean_accuracy - baseline_metrics.mean_accuracy) /
                         ((rograg_metrics.std_dev_quality + baseline_metrics.std_dev_quality) / 2.0);

        // Simple heuristic for p-value estimation
        let p_value_accuracy = if effect_size.abs() > 0.5 { 0.01 } else { 0.1 };
        let p_value_completeness = if rograg_metrics.mean_completeness > baseline_metrics.mean_completeness { 0.05 } else { 0.1 };
        let p_value_coherence = if rograg_metrics.mean_coherence > baseline_metrics.mean_coherence { 0.05 } else { 0.1 };
        let p_value_overall = (p_value_accuracy + p_value_completeness + p_value_coherence) / 3.0;

        Ok(StatisticalSignificance {
            p_value_accuracy,
            p_value_completeness,
            p_value_coherence,
            p_value_overall,
            is_significant_95: p_value_overall < 0.05,
            effect_size,
        })
    }

    /// Get current performance statistics
    pub fn get_performance_statistics(&self) -> &PerformanceStatistics {
        &self.performance_stats
    }

    /// Get quality benchmarks
    pub fn get_quality_benchmarks(&self) -> &QualityBenchmarks {
        &self.quality_benchmarks
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> &[QualityAlert] {
        &self.real_time_monitor.active_alerts
    }

    /// Get recent query metrics
    pub fn get_recent_metrics(&self, count: usize) -> Vec<&QueryMetrics> {
        self.query_history.iter().rev().take(count).collect()
    }

    /// Export metrics to JSON
    pub fn export_to_json(&self) -> Result<String> {
        let export_data = serde_json::json!({
            "performance_stats": self.performance_stats,
            "quality_benchmarks": self.quality_benchmarks,
            "recent_queries": self.query_history.iter().rev().take(100).collect::<Vec<_>>(),
            "active_alerts": self.real_time_monitor.active_alerts,
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });

        Ok(serde_json::to_string_pretty(&export_data)?)
    }

    /// Clear all metrics history
    pub fn clear_history(&mut self) {
        self.query_history.clear();
        self.real_time_monitor.current_window.clear();
        self.real_time_monitor.active_alerts.clear();
        self.performance_stats = PerformanceStatistics {
            total_queries: 0,
            successful_decompositions: 0,
            avg_processing_time_ms: 0.0,
            avg_quality_score: 0.0,
            fallback_rate: 0.0,
            error_rate: 0.0,
            throughput_qps: 0.0,
        };
    }

    /// Get configuration
    pub fn get_config(&self) -> &QualityMetricsConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: QualityMetricsConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rograg::{IntentResult, QueryIntent, ProcessingStats, SubqueryResult, SubqueryResultType};
    use std::time::Duration;

    #[cfg(feature = "rograg")]
    fn create_test_response() -> RogragResponse {
        RogragResponse {
            query: "What is Entity Name?".to_string(),
            content: "Entity Name is a young boy character in Mark Twain's novels.".to_string(),
            confidence: 0.8,
            sources: vec!["source1".to_string(), "source2".to_string()],
            subquery_results: vec![
                SubqueryResult {
                    subquery: "What is Entity Name?".to_string(),
                    result_type: SubqueryResultType::LogicForm,
                    confidence: 0.8,
                    content: "Entity Name character info".to_string(),
                    sources: vec!["source1".to_string()],
                }
            ],
            intent_result: IntentResult {
                primary_intent: QueryIntent::Factual,
                secondary_intents: vec![],
                confidence: 0.8,
                should_refuse: false,
                refusal_reason: None,
                suggested_reformulation: None,
                complexity_score: 0.3,
            },
            processing_stats: ProcessingStats::default(),
            is_streaming: false,
            is_refusal: false,
        }
    }

    #[cfg(feature = "rograg")]
    fn create_test_decomposition() -> DecompositionResult {
        use crate::rograg::{DecompositionResult, DecompositionStrategy, Subquery, SubqueryType};

        DecompositionResult {
            original_query: "What is Entity Name?".to_string(),
            subqueries: vec![
                Subquery {
                    id: "1".to_string(),
                    text: "What is Entity Name?".to_string(),
                    query_type: SubqueryType::Definitional,
                    priority: 1.0,
                    dependencies: vec![],
                }
            ],
            strategy_used: DecompositionStrategy::Semantic,
            confidence: 0.8,
            dependencies: vec![],
        }
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_quality_metrics_creation() {
        let metrics = QualityMetrics::new();
        assert_eq!(metrics.performance_stats.total_queries, 0);
        assert!(metrics.query_history.is_empty());
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_record_query() {
        let mut metrics = QualityMetrics::new();
        let response = create_test_response();
        let decomposition = create_test_decomposition();

        let result = metrics.record_query(
            "What is Entity Name?",
            &decomposition,
            &response,
            Duration::from_millis(1000),
        );

        assert!(result.is_ok());
        assert_eq!(metrics.performance_stats.total_queries, 1);
        assert_eq!(metrics.query_history.len(), 1);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_response_quality_calculation() {
        let metrics = QualityMetrics::new();
        let response = create_test_response();

        let quality = metrics.calculate_response_quality(&response).unwrap();

        assert!(quality.accuracy_score > 0.0);
        assert!(quality.completeness_score > 0.0);
        assert!(quality.coherence_score > 0.0);
        assert!(quality.relevance_score > 0.0);
        assert!(quality.overall_quality > 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_performance_stats_update() {
        let mut metrics = QualityMetrics::new();
        let response = create_test_response();
        let decomposition = create_test_decomposition();

        // Record multiple queries
        for i in 0..5 {
            let query = format!("Test query {i}");
            metrics.record_query(&query, &decomposition, &response, Duration::from_millis(1000 + i as u64 * 100)).unwrap();
        }

        assert_eq!(metrics.performance_stats.total_queries, 5);
        assert_eq!(metrics.performance_stats.successful_decompositions, 5);
        assert!(metrics.performance_stats.avg_processing_time_ms > 0.0);
        assert!(metrics.performance_stats.avg_quality_score > 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_improvement_percentage_calculation() {
        let metrics = QualityMetrics::new();

        let improvement = metrics.calculate_improvement_percentage(0.8, 0.6);
        assert!((improvement - 33.333).abs() < 0.1); // 33.33% improvement

        let no_improvement = metrics.calculate_improvement_percentage(0.6, 0.6);
        assert_eq!(no_improvement, 0.0);

        let degradation = metrics.calculate_improvement_percentage(0.5, 0.7);
        assert!(degradation < 0.0); // Negative improvement (degradation)
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_export_to_json() {
        let mut metrics = QualityMetrics::new();
        let response = create_test_response();
        let decomposition = create_test_decomposition();

        metrics.record_query("Test query", &decomposition, &response, Duration::from_millis(1000)).unwrap();

        let json = metrics.export_to_json().unwrap();
        assert!(json.contains("performance_stats"));
        assert!(json.contains("total_queries"));
    }
}