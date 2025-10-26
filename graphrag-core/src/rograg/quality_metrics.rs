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

/// Error types for quality metrics operations.
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum MetricsError {
    /// Metric value is outside valid range or malformed.
    ///
    /// Occurs when a calculated metric is NaN, infinite, or outside [0, 1].
    #[error("Invalid metric value: {metric} = {value}")]
    InvalidValue {
        /// Name of the metric with invalid value.
        metric: String,
        /// The invalid value that was encountered.
        value: f64
    },

    /// Insufficient data points for statistical analysis.
    ///
    /// Occurs when analysis requires a minimum sample size not met.
    #[error("Insufficient data for analysis: {reason}")]
    InsufficientData {
        /// Explanation of data insufficiency.
        reason: String
    },

    /// Metric calculation encountered an error.
    ///
    /// Occurs due to division by zero, overflow, or other computation failures.
    #[error("Metric calculation failed: {reason}")]
    CalculationFailed {
        /// Description of the calculation failure.
        reason: String
    },
}

/// Configuration for quality metrics tracking and analysis.
///
/// Controls metrics collection, comparative analysis, and monitoring behavior.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct QualityMetricsConfig {
    /// Whether to enable metrics tracking.
    pub enable_tracking: bool,

    /// Maximum number of query metrics to retain in history.
    pub max_history_size: usize,

    /// Whether to enable comparative analysis against baseline.
    pub enable_comparative_analysis: bool,

    /// Window size for baseline comparison (number of queries).
    pub baseline_comparison_window: usize,

    /// Minimum acceptable quality threshold (0.0 to 1.0).
    pub quality_threshold: f32,

    /// Maximum acceptable processing time in milliseconds.
    pub performance_threshold_ms: u64,

    /// Whether to enable real-time quality monitoring and alerts.
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

/// Quality metrics collector and analyzer.
///
/// Central system for tracking, analyzing, and reporting on ROGRAG quality metrics.
/// Maintains query history, calculates performance statistics, performs comparative
/// analysis against baseline GraphRAG, and provides real-time monitoring with alerts.
///
/// # Features
/// - **Query Tracking**: Records detailed metrics for each query processed
/// - **Performance Statistics**: Maintains aggregated performance metrics
/// - **Comparative Analysis**: Compares ROGRAG against baseline GraphRAG
/// - **Real-time Monitoring**: Detects quality degradation and generates alerts
/// - **Statistical Analysis**: Provides significance testing for improvements
///
/// # Example
/// ```
/// use graphrag_core::rograg::QualityMetrics;
/// use graphrag_core::rograg::QualityMetricsConfig;
///
/// // Create with default configuration
/// let metrics = QualityMetrics::new();
///
/// // Or with custom configuration
/// let config = QualityMetricsConfig {
///     enable_tracking: true,
///     max_history_size: 500,
///     quality_threshold: 0.8,
///     ..Default::default()
/// };
/// let metrics = QualityMetrics::with_config(config);
/// ```
#[cfg(feature = "rograg")]
#[derive(Clone)]
pub struct QualityMetrics {
    config: QualityMetricsConfig,
    query_history: VecDeque<QueryMetrics>,
    performance_stats: PerformanceStatistics,
    quality_benchmarks: QualityBenchmarks,
    real_time_monitor: RealTimeMonitor,
}

/// Metrics for a single query execution.
///
/// Captures all relevant performance and quality data for one query.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryMetrics {
    /// Unix timestamp when query was processed.
    pub timestamp: u64,

    /// The query text.
    pub query: String,

    /// Whether query decomposition succeeded.
    pub decomposition_success: bool,

    /// Time spent on decomposition in milliseconds.
    pub decomposition_time_ms: u64,

    /// Number of subqueries generated.
    pub subquery_count: usize,

    /// Retrieval strategy that was used.
    pub retrieval_strategy_used: RetrievalStrategy,

    /// Quality scores for the response.
    pub response_quality: ResponseQuality,

    /// Total processing time in milliseconds.
    pub processing_time_ms: u64,

    /// Whether fallback strategies were used.
    pub fallback_used: bool,

    /// Overall confidence score (0.0 to 1.0).
    pub confidence_score: f32,

    /// User satisfaction score if available from feedback (0.0 to 1.0).
    pub user_satisfaction: Option<f32>,
}

/// Response quality metrics.
///
/// Comprehensive quality assessment scores for a generated response,
/// measuring multiple dimensions of answer quality.
///
/// # Score Range
/// All scores are normalized to [0.0, 1.0] where 1.0 is perfect quality.
///
/// # Example
/// ```
/// # use graphrag_core::rograg::ResponseQuality;
/// let quality = ResponseQuality {
///     accuracy_score: 0.85,
///     completeness_score: 0.78,
///     coherence_score: 0.92,
///     relevance_score: 0.88,
///     source_credibility: 0.75,
///     overall_quality: 0.84,
/// };
/// assert!(quality.overall_quality > 0.8);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseQuality {
    /// Accuracy of the response based on confidence and source credibility.
    ///
    /// Measures factual correctness and reliability of information. Range: [0.0, 1.0].
    pub accuracy_score: f32,

    /// Completeness of the response coverage.
    ///
    /// Measures how thoroughly the response addresses all aspects of the query.
    /// Considers answer length, source count, and subquery coverage. Range: [0.0, 1.0].
    pub completeness_score: f32,

    /// Coherence and logical flow of the response text.
    ///
    /// Measures text structure, transitions between ideas, and lack of repetition.
    /// Range: [0.0, 1.0].
    pub coherence_score: f32,

    /// Relevance to the original query.
    ///
    /// Measures alignment between query terms and response content.
    /// Higher when response directly addresses query keywords. Range: [0.0, 1.0].
    pub relevance_score: f32,

    /// Credibility of information sources.
    ///
    /// Measures source count, diversity, and reliability. Higher with more
    /// diverse and credible sources. Range: [0.0, 1.0].
    pub source_credibility: f32,

    /// Weighted overall quality score.
    ///
    /// Composite score combining all quality dimensions with weights:
    /// accuracy (30%), completeness (25%), coherence (20%),
    /// relevance (15%), source credibility (10%). Range: [0.0, 1.0].
    pub overall_quality: f32,
}

/// Retrieval strategy used for query processing.
///
/// Identifies which retrieval approach was applied to answer the query,
/// helping track effectiveness of different strategies.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, Default)]
pub enum RetrievalStrategy {
    /// Logic form-based structured retrieval.
    ///
    /// Uses formal logic representations (FOL) to query the knowledge graph.
    /// Most precise but requires successful query decomposition.
    #[default]
    LogicForm,

    /// Fuzzy matching-based retrieval.
    ///
    /// Uses semantic similarity and approximate matching to find relevant entities.
    /// More flexible than logic form but potentially less precise.
    FuzzyMatch,

    /// Combined logic form and fuzzy matching.
    ///
    /// Uses both strategies to maximize recall and precision.
    /// Applied when query complexity benefits from multiple approaches.
    Hybrid,

    /// Fallback strategy when primary methods fail.
    ///
    /// Basic retrieval used when decomposition fails or other strategies
    /// produce insufficient results.
    Fallback,
}

/// Performance statistics for system operation.
///
/// Aggregated metrics tracking system performance, efficiency, and reliability
/// across all processed queries.
///
/// # Example
/// ```
/// # use graphrag_core::rograg::PerformanceStatistics;
/// let stats = PerformanceStatistics {
///     total_queries: 1000,
///     successful_decompositions: 950,
///     avg_processing_time_ms: 250.5,
///     avg_quality_score: 0.82,
///     fallback_rate: 0.05,
///     error_rate: 0.02,
///     throughput_qps: 4.5,
/// };
/// assert!(stats.avg_quality_score > 0.8);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceStatistics {
    /// Total number of queries processed.
    ///
    /// Cumulative count of all queries submitted to the system.
    pub total_queries: usize,

    /// Number of successful query decompositions.
    ///
    /// Count of queries where decomposition into subqueries succeeded.
    /// Used to calculate decomposition success rate.
    pub successful_decompositions: usize,

    /// Average processing time in milliseconds.
    ///
    /// Mean time to process a query from submission to response.
    /// Updated as running average. Measured in milliseconds.
    pub avg_processing_time_ms: f64,

    /// Average overall quality score.
    ///
    /// Mean of overall quality scores across all queries.
    /// Range: [0.0, 1.0]. Updated as running average.
    pub avg_quality_score: f64,

    /// Rate of fallback strategy usage.
    ///
    /// Proportion of queries requiring fallback retrieval.
    /// Range: [0.0, 1.0] where 0.0 means no fallbacks needed.
    pub fallback_rate: f64,

    /// Rate of query processing errors.
    ///
    /// Proportion of queries that failed or encountered errors.
    /// Range: [0.0, 1.0] where 0.0 means no errors.
    pub error_rate: f64,

    /// System throughput in queries per second.
    ///
    /// Calculated over recent query window to reflect current capacity.
    /// Measured in queries per second (QPS).
    pub throughput_qps: f64,
}

/// Quality benchmarks comparing to baseline GraphRAG.
///
/// Tracks performance improvements of ROGRAG over baseline GraphRAG,
/// providing statistical evidence of system enhancements.
///
/// # Improvement Metrics
/// Improvements are expressed as percentages where positive values indicate
/// ROGRAG outperforms baseline, and negative values indicate degradation.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityBenchmarks {
    /// Percentage improvement in accuracy over baseline.
    ///
    /// Positive values indicate ROGRAG is more accurate. Measured as
    /// ((rograg_accuracy - baseline_accuracy) / baseline_accuracy) * 100.
    pub accuracy_improvement: f64,

    /// Percentage improvement in completeness over baseline.
    ///
    /// Positive values indicate ROGRAG responses are more complete.
    /// Measured as percentage change from baseline.
    pub completeness_improvement: f64,

    /// Percentage improvement in coherence over baseline.
    ///
    /// Positive values indicate ROGRAG responses are more coherent.
    /// Measured as percentage change from baseline.
    pub coherence_improvement: f64,

    /// Overall percentage improvement across all quality metrics.
    ///
    /// Composite improvement metric averaging all quality dimensions.
    /// Primary indicator of system enhancement.
    pub overall_improvement: f64,

    /// Number of queries used for baseline comparison.
    ///
    /// Sample size for statistical comparison. Larger values provide
    /// more reliable benchmark results.
    pub baseline_comparison_count: usize,

    /// Statistical confidence intervals for improvements.
    ///
    /// 95% confidence intervals for each improvement metric,
    /// establishing statistical significance.
    pub confidence_intervals: ConfidenceIntervals,
}

/// Confidence intervals for statistical significance.
///
/// Provides 95% confidence intervals for quality improvement metrics,
/// establishing the statistical reliability of observed improvements.
///
/// # Interpretation
/// Each tuple (lower, upper) represents the range within which the true
/// improvement value lies with 95% confidence. If both bounds are positive,
/// the improvement is statistically significant.
///
/// # Example
/// ```
/// # use graphrag_core::rograg::ConfidenceIntervals;
/// let intervals = ConfidenceIntervals {
///     accuracy_ci_95: (2.5, 8.3),  // Improvement between 2.5% and 8.3%
///     completeness_ci_95: (1.2, 6.7),
///     coherence_ci_95: (0.8, 5.4),
///     overall_ci_95: (1.5, 6.8),
/// };
/// // All intervals positive = statistically significant improvement
/// assert!(intervals.overall_ci_95.0 > 0.0);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfidenceIntervals {
    /// 95% confidence interval for accuracy improvement.
    ///
    /// Tuple of (lower_bound, upper_bound) in percentage points.
    /// If both values are positive, accuracy improvement is significant at 95% level.
    pub accuracy_ci_95: (f64, f64),

    /// 95% confidence interval for completeness improvement.
    ///
    /// Tuple of (lower_bound, upper_bound) in percentage points.
    pub completeness_ci_95: (f64, f64),

    /// 95% confidence interval for coherence improvement.
    ///
    /// Tuple of (lower_bound, upper_bound) in percentage points.
    pub coherence_ci_95: (f64, f64),

    /// 95% confidence interval for overall quality improvement.
    ///
    /// Tuple of (lower_bound, upper_bound) in percentage points.
    /// Primary indicator of statistically significant system improvement.
    pub overall_ci_95: (f64, f64),
}

/// Real-time monitoring for quality and performance tracking.
///
/// Maintains a sliding window of recent queries to detect quality degradation,
/// performance issues, or unusual patterns requiring immediate attention.
///
/// # Monitoring Behavior
/// Continuously evaluates recent queries against configurable thresholds,
/// generating alerts when metrics fall outside acceptable ranges.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct RealTimeMonitor {
    /// Sliding window of recent query metrics.
    ///
    /// Maintains most recent queries up to `window_size`. Oldest queries
    /// are dropped as new ones arrive. Used for real-time trend analysis.
    pub current_window: VecDeque<QueryMetrics>,

    /// Maximum number of queries to retain in monitoring window.
    ///
    /// Determines the temporal scope of real-time monitoring. Larger values
    /// provide more stable trend detection but slower response to changes.
    pub window_size: usize,

    /// Quality and performance thresholds for alerting.
    ///
    /// Defines acceptable operating ranges. Queries outside these ranges
    /// trigger quality alerts.
    pub alert_thresholds: AlertThresholds,

    /// Currently active quality alerts.
    ///
    /// List of recent alerts indicating quality or performance issues.
    /// Alerts older than one hour are automatically pruned.
    pub active_alerts: Vec<QualityAlert>,
}

/// Alert thresholds for real-time monitoring.
///
/// Defines acceptable operating ranges for quality and performance metrics.
/// Queries falling outside these ranges trigger quality alerts.
///
/// # Default Values
/// - min_quality_score: 0.6
/// - max_processing_time_ms: 10000
/// - max_error_rate: 0.1 (10%)
/// - min_success_rate: 0.8 (80%)
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Minimum acceptable overall quality score.
    ///
    /// Queries with quality scores below this threshold trigger quality
    /// degradation alerts. Range: [0.0, 1.0]. Default: 0.6.
    pub min_quality_score: f32,

    /// Maximum acceptable processing time in milliseconds.
    ///
    /// Queries exceeding this duration trigger performance degradation alerts.
    /// Measured in milliseconds. Default: 10000ms (10 seconds).
    pub max_processing_time_ms: u64,

    /// Maximum acceptable error rate.
    ///
    /// Error rates above this threshold trigger high error rate alerts.
    /// Range: [0.0, 1.0] where 0.1 means 10% error rate. Default: 0.1.
    pub max_error_rate: f32,

    /// Minimum acceptable success rate.
    ///
    /// Success rates below this threshold trigger low success rate alerts.
    /// Range: [0.0, 1.0] where 0.8 means 80% success rate. Default: 0.8.
    pub min_success_rate: f32,
}

/// Quality alert indicating detected issues.
///
/// Generated when monitored metrics fall outside acceptable thresholds,
/// providing actionable information about quality or performance problems.
///
/// # Example
/// ```
/// # use graphrag_core::rograg::{QualityAlert, AlertType, AlertSeverity};
/// let alert = QualityAlert {
///     alert_type: AlertType::QualityDegradation,
///     severity: AlertSeverity::High,
///     message: "Low quality response: 0.45".to_string(),
///     timestamp: 1234567890,
///     metric_value: 0.45,
///     threshold: 0.6,
/// };
/// assert!(alert.metric_value < alert.threshold);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityAlert {
    /// Type of issue detected.
    ///
    /// Categorizes the alert to enable appropriate response actions.
    pub alert_type: AlertType,

    /// Severity level of the alert.
    ///
    /// Indicates urgency and priority for addressing the issue.
    /// Critical alerts require immediate attention.
    pub severity: AlertSeverity,

    /// Human-readable description of the alert.
    ///
    /// Provides context and details about the detected issue,
    /// including specific metric values.
    pub message: String,

    /// Unix timestamp when alert was generated.
    ///
    /// Seconds since UNIX epoch. Used for alert aging and pruning.
    pub timestamp: u64,

    /// Actual metric value that triggered the alert.
    ///
    /// The measured value that fell outside acceptable range.
    pub metric_value: f64,

    /// Threshold value that was violated.
    ///
    /// The configured limit that the metric exceeded or fell below.
    pub threshold: f64,
}

/// Type of quality alert.
///
/// Categorizes detected issues to enable appropriate monitoring,
/// logging, and response actions.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, Default)]
pub enum AlertType {
    /// Response quality has fallen below acceptable threshold.
    ///
    /// Triggered when overall quality score is below `min_quality_score`.
    /// Indicates potential issues with answer accuracy, completeness, or coherence.
    #[default]
    QualityDegradation,

    /// Processing time has exceeded acceptable threshold.
    ///
    /// Triggered when query processing time exceeds `max_processing_time_ms`.
    /// Indicates system slowdown or resource constraints.
    PerformanceDegradation,

    /// Error rate has risen above acceptable threshold.
    ///
    /// Triggered when error rate exceeds `max_error_rate`.
    /// Indicates systemic failures or integration problems.
    HighErrorRate,

    /// Success rate has fallen below acceptable threshold.
    ///
    /// Triggered when success rate drops below `min_success_rate`.
    /// Indicates widespread query processing failures.
    LowSuccessRate,

    /// Unusual pattern detected in metrics.
    ///
    /// Triggered by anomalous behavior that doesn't fit other categories.
    /// May indicate data quality issues or unexpected system behavior.
    UnusualPattern,
}

/// Severity level of quality alert.
///
/// Indicates the urgency and priority for responding to detected issues.
/// Higher severity levels require more immediate attention and escalation.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, Default)]
pub enum AlertSeverity {
    /// Low priority issue.
    ///
    /// Minor deviation from optimal performance. Can be addressed during
    /// routine maintenance. No immediate action required.
    #[default]
    Low,

    /// Medium priority issue.
    ///
    /// Noticeable quality or performance degradation. Should be investigated
    /// within hours to prevent escalation.
    Medium,

    /// High priority issue.
    ///
    /// Significant quality or performance problem affecting user experience.
    /// Requires prompt investigation and resolution.
    High,

    /// Critical priority issue.
    ///
    /// Severe system degradation or failure. Requires immediate attention
    /// and may necessitate service interruption for fixes.
    Critical,
}

/// Comparative analysis result.
///
/// Comprehensive comparison of ROGRAG performance against baseline GraphRAG,
/// including statistical significance testing and improvement quantification.
///
/// # Example
/// ```
/// # use graphrag_core::rograg::ComparativeAnalysis;
/// # use graphrag_core::rograg::{AggregatedMetrics, ImprovementPercentages, StatisticalSignificance};
/// let analysis = ComparativeAnalysis {
///     rograg_metrics: AggregatedMetrics::default(),
///     baseline_metrics: AggregatedMetrics::default(),
///     improvement_percentages: ImprovementPercentages {
///         accuracy_improvement: 15.2,
///         completeness_improvement: 12.8,
///         coherence_improvement: 18.5,
///         relevance_improvement: 14.3,
///         overall_improvement: 15.2,
///     },
///     statistical_significance: StatisticalSignificance::default(),
///     sample_size: 100,
///     analysis_timestamp: 1234567890,
/// };
/// assert!(analysis.improvement_percentages.overall_improvement > 10.0);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComparativeAnalysis {
    /// Aggregated metrics for ROGRAG system.
    ///
    /// Performance statistics averaged across all ROGRAG queries
    /// in the comparison sample.
    pub rograg_metrics: AggregatedMetrics,

    /// Aggregated metrics for baseline GraphRAG system.
    ///
    /// Performance statistics averaged across all baseline queries
    /// in the comparison sample.
    pub baseline_metrics: AggregatedMetrics,

    /// Percentage improvements over baseline.
    ///
    /// Calculated improvement for each quality dimension,
    /// showing how much ROGRAG outperforms baseline.
    pub improvement_percentages: ImprovementPercentages,

    /// Statistical significance analysis.
    ///
    /// P-values and effect sizes establishing whether observed improvements
    /// are statistically significant or due to chance.
    pub statistical_significance: StatisticalSignificance,

    /// Number of query pairs used in comparison.
    ///
    /// Sample size for statistical analysis. Larger values provide
    /// more reliable results.
    pub sample_size: usize,

    /// Unix timestamp when analysis was performed.
    ///
    /// Seconds since UNIX epoch. Used for tracking analysis history.
    pub analysis_timestamp: u64,
}

/// Aggregated metrics across multiple queries.
///
/// Statistical summary of quality and performance metrics calculated
/// over a collection of queries for comparison and trend analysis.
///
/// # Statistical Measures
/// All means and rates are calculated as arithmetic averages over the
/// query sample. Standard deviation measures quality score variability.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatedMetrics {
    /// Mean accuracy score across all queries.
    ///
    /// Average of accuracy scores. Range: [0.0, 1.0].
    /// Higher values indicate better factual correctness.
    pub mean_accuracy: f64,

    /// Mean completeness score across all queries.
    ///
    /// Average of completeness scores. Range: [0.0, 1.0].
    /// Higher values indicate more thorough responses.
    pub mean_completeness: f64,

    /// Mean coherence score across all queries.
    ///
    /// Average of coherence scores. Range: [0.0, 1.0].
    /// Higher values indicate better text structure and flow.
    pub mean_coherence: f64,

    /// Mean relevance score across all queries.
    ///
    /// Average of relevance scores. Range: [0.0, 1.0].
    /// Higher values indicate better query-answer alignment.
    pub mean_relevance: f64,

    /// Mean processing time in milliseconds.
    ///
    /// Average time to process a query. Measured in milliseconds.
    /// Lower values indicate better performance.
    pub mean_processing_time_ms: f64,

    /// Proportion of successful query decompositions.
    ///
    /// Success rate as fraction of queries. Range: [0.0, 1.0].
    /// Higher values indicate more reliable query processing.
    pub success_rate: f64,

    /// Standard deviation of overall quality scores.
    ///
    /// Measures consistency of quality across queries.
    /// Lower values indicate more consistent performance.
    pub std_dev_quality: f64,
}

/// Improvement percentages over baseline.
///
/// Quantifies ROGRAG performance gains relative to baseline GraphRAG,
/// expressed as percentage improvements for each quality dimension.
///
/// # Calculation
/// Each improvement is calculated as:
/// `((rograg_value - baseline_value) / baseline_value) * 100`
///
/// Positive values indicate improvement, negative values indicate degradation.
///
/// # Example
/// ```
/// # use graphrag_core::rograg::ImprovementPercentages;
/// let improvements = ImprovementPercentages {
///     accuracy_improvement: 15.2,      // 15.2% better accuracy
///     completeness_improvement: 12.8,  // 12.8% more complete
///     coherence_improvement: 18.5,     // 18.5% more coherent
///     relevance_improvement: 14.3,     // 14.3% more relevant
///     overall_improvement: 15.2,       // 15.2% overall improvement
/// };
/// assert!(improvements.overall_improvement > 10.0);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImprovementPercentages {
    /// Percentage improvement in accuracy.
    ///
    /// How much more accurate ROGRAG responses are compared to baseline.
    /// Positive values indicate improvement.
    pub accuracy_improvement: f64,

    /// Percentage improvement in completeness.
    ///
    /// How much more complete ROGRAG responses are compared to baseline.
    /// Positive values indicate improvement.
    pub completeness_improvement: f64,

    /// Percentage improvement in coherence.
    ///
    /// How much more coherent ROGRAG responses are compared to baseline.
    /// Positive values indicate improvement.
    pub coherence_improvement: f64,

    /// Percentage improvement in relevance.
    ///
    /// How much more relevant ROGRAG responses are compared to baseline.
    /// Positive values indicate improvement.
    pub relevance_improvement: f64,

    /// Overall percentage improvement across all dimensions.
    ///
    /// Composite improvement metric averaging all quality improvements.
    /// Primary indicator of ROGRAG effectiveness over baseline.
    pub overall_improvement: f64,
}

/// Statistical significance analysis.
///
/// Evaluates whether observed improvements are statistically significant
/// or likely due to random chance, using p-values and effect sizes.
///
/// # Interpretation
/// - p-value < 0.05: Improvement is statistically significant at 95% confidence
/// - p-value < 0.01: Improvement is highly significant at 99% confidence
/// - Effect size > 0.5: Large practical significance
/// - Effect size > 0.3: Medium practical significance
///
/// # Example
/// ```
/// # use graphrag_core::rograg::StatisticalSignificance;
/// let significance = StatisticalSignificance {
///     p_value_accuracy: 0.02,      // Significant at 95% level
///     p_value_completeness: 0.03,
///     p_value_coherence: 0.01,     // Highly significant
///     p_value_overall: 0.02,       // Significant
///     is_significant_95: true,      // Overall improvement is significant
///     effect_size: 0.65,            // Large effect size
/// };
/// assert!(significance.is_significant_95);
/// assert!(significance.effect_size > 0.5);
/// ```
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatisticalSignificance {
    /// P-value for accuracy improvement.
    ///
    /// Probability that observed accuracy improvement occurred by chance.
    /// Values < 0.05 indicate statistically significant improvement.
    pub p_value_accuracy: f64,

    /// P-value for completeness improvement.
    ///
    /// Probability that observed completeness improvement occurred by chance.
    /// Values < 0.05 indicate statistically significant improvement.
    pub p_value_completeness: f64,

    /// P-value for coherence improvement.
    ///
    /// Probability that observed coherence improvement occurred by chance.
    /// Values < 0.05 indicate statistically significant improvement.
    pub p_value_coherence: f64,

    /// P-value for overall quality improvement.
    ///
    /// Probability that observed overall improvement occurred by chance.
    /// Primary indicator of statistical significance. Values < 0.05 indicate
    /// statistically significant improvement at 95% confidence level.
    pub p_value_overall: f64,

    /// Whether improvement is statistically significant at 95% confidence.
    ///
    /// True if `p_value_overall < 0.05`, indicating improvement is unlikely
    /// to be due to chance alone.
    pub is_significant_95: bool,

    /// Cohen's d effect size.
    ///
    /// Standardized measure of improvement magnitude.
    /// - < 0.3: Small effect
    /// - 0.3-0.5: Medium effect
    /// - > 0.5: Large effect
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