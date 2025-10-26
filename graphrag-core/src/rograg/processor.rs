//! Main ROGRAG processor implementation
//!
//! Orchestrates the complete ROGRAG pipeline including query decomposition,
//! dual-level retrieval, and robust response generation.

#[cfg(feature = "rograg")]
use crate::core::KnowledgeGraph;
#[cfg(feature = "rograg")]
use crate::Result;

use crate::rograg::{
    DecompositionResult, FuzzyMatchResult, FuzzyMatcher, HybridQueryDecomposer, IntentClassifier,
    IntentResult, LogicFormResult, LogicFormRetriever, QueryDecomposer, QueryValidator,
    StreamingResponseBuilder,
};

use crate::rograg::quality_metrics::QualityMetrics;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use std::sync::Arc;
#[cfg(feature = "rograg")]
use std::time::{Duration, Instant};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Error types for ROGRAG processing.
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum ProcessingError {
    /// Query processing exceeded the configured timeout.
    ///
    /// Occurs when query processing takes longer than `max_processing_time`.
    #[error("Query processing timeout after {duration:?}")]
    Timeout {
        /// Duration elapsed before timeout.
        duration: Duration
    },

    /// All retrieval strategies failed to produce results.
    ///
    /// Occurs when both logic form and fuzzy matching fail.
    #[error("All retrieval strategies failed: {reason}")]
    AllStrategiesFailed {
        /// Reason why strategies failed.
        reason: String
    },

    /// Query is malformed, empty, or otherwise invalid.
    ///
    /// Occurs during query validation before processing begins.
    #[error("Invalid query: {reason}")]
    InvalidQuery {
        /// Description of what makes the query invalid.
        reason: String
    },

    /// ROGRAG configuration is invalid or incomplete.
    ///
    /// Occurs when configuration parameters are outside valid ranges.
    #[error("Configuration error: {reason}")]
    ConfigurationError {
        /// Description of the configuration issue.
        reason: String
    },

    /// Component initialization failed during processor setup.
    ///
    /// Occurs when decomposer, matcher, or classifier fails to initialize.
    #[error("Component initialization failed: {component}: {reason}")]
    InitializationFailed {
        /// Name of the component that failed.
        component: String,
        /// Reason for initialization failure.
        reason: String
    },
}

/// ROGRAG processor implementation
#[cfg(feature = "rograg")]
pub struct RogragProcessor {
    decomposer: Arc<dyn QueryDecomposer>,
    fuzzy_matcher: Arc<FuzzyMatcher>,
    intent_classifier: Arc<IntentClassifier>,
    logic_form_retriever: Arc<LogicFormRetriever>,
    streaming_builder: Arc<StreamingResponseBuilder>,
    validator: Arc<QueryValidator>,
    quality_metrics: Arc<std::sync::Mutex<QualityMetrics>>,
    config: RogragConfig,
}

/// Builder for RogragProcessor
#[cfg(feature = "rograg")]
pub struct RogragProcessorBuilder {
    decomposer: Option<Arc<dyn QueryDecomposer>>,
    fuzzy_matcher: Option<Arc<FuzzyMatcher>>,
    intent_classifier: Option<Arc<IntentClassifier>>,
    logic_form_retriever: Option<Arc<LogicFormRetriever>>,
    streaming_builder: Option<Arc<StreamingResponseBuilder>>,
    validator: Option<Arc<QueryValidator>>,
    quality_metrics: Option<Arc<std::sync::Mutex<QualityMetrics>>>,
    config: RogragConfig,
}

/// Processing context for a single query
#[cfg(feature = "rograg")]
#[derive(Debug)]
struct ProcessingContext {
    #[allow(dead_code)] query: String,
    start_time: Instant,
    decomposition_time: Option<Duration>,
    retrieval_time: Option<Duration>,
    response_time: Option<Duration>,
    fallback_count: usize,
    errors_encountered: Vec<String>,
}

#[cfg(feature = "rograg")]
impl ProcessingContext {
    fn new(query: String) -> Self {
        Self {
            query,
            start_time: Instant::now(),
            decomposition_time: None,
            retrieval_time: None,
            response_time: None,
            fallback_count: 0,
            errors_encountered: Vec::new(),
        }
    }

    fn record_decomposition_time(&mut self, duration: Duration) {
        self.decomposition_time = Some(duration);
    }

    fn record_retrieval_time(&mut self, duration: Duration) {
        self.retrieval_time = Some(duration);
    }

    fn record_response_time(&mut self, duration: Duration) {
        self.response_time = Some(duration);
    }

    fn increment_fallback(&mut self) {
        self.fallback_count += 1;
    }

    fn add_error(&mut self, error: String) {
        self.errors_encountered.push(error);
    }

    fn total_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    fn to_processing_stats(&self) -> ProcessingStats {
        ProcessingStats {
            total_time_ms: self.total_time().as_millis() as u64,
            decomposition_time_ms: self
                .decomposition_time
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            retrieval_time_ms: self
                .retrieval_time
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            synthesis_time_ms: self
                .response_time
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            intent_classification_time_ms: 0,
            subqueries_processed: 0, // Will be set by caller
            fallback_used: self.fallback_count > 0,
            validation_time_ms: 0,
        }
    }
}

#[cfg(feature = "rograg")]
impl Default for RogragProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RogragProcessorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            decomposer: None,
            fuzzy_matcher: None,
            intent_classifier: None,
            logic_form_retriever: None,
            streaming_builder: None,
            validator: None,
            quality_metrics: None,
            config: RogragConfig::default(),
        }
    }

    /// Set custom decomposer
    pub fn with_decomposer(mut self, decomposer: Arc<dyn QueryDecomposer>) -> Self {
        self.decomposer = Some(decomposer);
        self
    }

    /// Set custom fuzzy matcher
    pub fn with_fuzzy_matcher(mut self, fuzzy_matcher: Arc<FuzzyMatcher>) -> Self {
        self.fuzzy_matcher = Some(fuzzy_matcher);
        self
    }

    /// Set custom intent classifier
    pub fn with_intent_classifier(mut self, intent_classifier: Arc<IntentClassifier>) -> Self {
        self.intent_classifier = Some(intent_classifier);
        self
    }

    /// Set custom logic form retriever
    pub fn with_logic_form_retriever(
        mut self,
        logic_form_retriever: Arc<LogicFormRetriever>,
    ) -> Self {
        self.logic_form_retriever = Some(logic_form_retriever);
        self
    }

    /// Set custom streaming builder
    pub fn with_streaming_builder(
        mut self,
        streaming_builder: Arc<StreamingResponseBuilder>,
    ) -> Self {
        self.streaming_builder = Some(streaming_builder);
        self
    }

    /// Set custom validator
    pub fn with_validator(mut self, validator: Arc<QueryValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Set custom quality metrics
    pub fn with_quality_metrics(
        mut self,
        quality_metrics: Arc<std::sync::Mutex<QualityMetrics>>,
    ) -> Self {
        self.quality_metrics = Some(quality_metrics);
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: RogragConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the processor
    pub fn build(self) -> Result<RogragProcessor> {
        // Use provided components or create defaults
        let decomposer = self
            .decomposer
            .unwrap_or_else(|| Arc::new(HybridQueryDecomposer::new().unwrap()));

        let fuzzy_matcher = self
            .fuzzy_matcher
            .unwrap_or_else(|| Arc::new(FuzzyMatcher::new()));

        let intent_classifier = self
            .intent_classifier
            .unwrap_or_else(|| Arc::new(IntentClassifier::new().unwrap()));

        let logic_form_retriever = self
            .logic_form_retriever
            .unwrap_or_else(|| Arc::new(LogicFormRetriever::new()));

        let streaming_builder = self
            .streaming_builder
            .unwrap_or_else(|| Arc::new(StreamingResponseBuilder::new()));

        let validator = self
            .validator
            .unwrap_or_else(|| Arc::new(QueryValidator::new()));

        let quality_metrics = self
            .quality_metrics
            .unwrap_or_else(|| Arc::new(std::sync::Mutex::new(QualityMetrics::new())));

        Ok(RogragProcessor {
            decomposer,
            fuzzy_matcher,
            intent_classifier,
            logic_form_retriever,
            streaming_builder,
            validator,
            quality_metrics,
            config: self.config,
        })
    }
}

#[cfg(feature = "rograg")]
impl RogragProcessor {
    /// Create a new ROGRAG processor with default configuration
    pub fn new() -> Result<Self> {
        RogragProcessorBuilder::new().build()
    }

    /// Create a new ROGRAG processor with custom configuration
    pub fn with_config(config: RogragConfig) -> Result<Self> {
        RogragProcessorBuilder::new().with_config(config).build()
    }

    /// Get a builder for custom configuration
    pub fn builder() -> RogragProcessorBuilder {
        RogragProcessorBuilder::new()
    }

    /// Process a query using the complete ROGRAG pipeline
    pub async fn process_query(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
    ) -> Result<RogragResponse> {
        let mut context = ProcessingContext::new(query.to_string());

        // Process query (timeout handling would require tokio feature)
        let result = self
            .process_query_internal(query, graph, &mut context)
            .await;

        // Record metrics if tracking is enabled
        if self.config.quality_tracking {
            if let Ok(ref response) = result {
                if let Some(decomposition_result) = self.get_decomposition_for_metrics(query).await
                {
                    if let Ok(mut metrics) = self.quality_metrics.lock() {
                        let _ = metrics.record_query(
                            query,
                            &decomposition_result,
                            response,
                            context.total_time(),
                        );
                    }
                }
            }
        }

        result
    }

    /// Internal query processing implementation
    async fn process_query_internal(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
        context: &mut ProcessingContext,
    ) -> Result<RogragResponse> {
        // Step 1: Validate query
        let query_validation = self.validator.validate_query(query)?;
        if !query_validation.is_valid {
            return Ok(RogragResponse::refusal(
                query_validation
                    .issues
                    .first()
                    .map(|issue| issue.description.clone())
                    .unwrap_or_else(|| "Query validation failed".to_string()),
                "Invalid query".to_string(),
            ));
        }

        // Step 2: Classify intent
        let intent_result = self.intent_classifier.classify(query)?;
        if intent_result.should_refuse {
            return Ok(RogragResponse::refusal(
                intent_result
                    .refusal_reason
                    .unwrap_or_else(|| "Query cannot be answered safely".to_string()),
                "Safety refusal".to_string(),
            ));
        }

        // Step 3: Query decomposition
        let decomposition_start = Instant::now();
        let decomposition_result = self.decompose_query_with_fallback(query, context).await?;
        context.record_decomposition_time(decomposition_start.elapsed());

        // Step 4: Dual-level retrieval
        let retrieval_start = Instant::now();
        let subquery_results = self
            .execute_dual_level_retrieval(&decomposition_result, graph, context)
            .await?;
        context.record_retrieval_time(retrieval_start.elapsed());

        // Step 5: Response generation
        let response_start = Instant::now();
        let mut response = self
            .generate_response(query, subquery_results, intent_result, context)
            .await?;
        context.record_response_time(response_start.elapsed());

        // Step 6: Response validation and enhancement
        response = self.validator.validate_response(&response)?;

        // Step 7: Update processing statistics
        response.processing_stats = context.to_processing_stats();
        response.processing_stats.subqueries_processed = decomposition_result.subqueries.len();

        Ok(response)
    }

    /// Decompose query with fallback handling
    async fn decompose_query_with_fallback(
        &self,
        query: &str,
        context: &mut ProcessingContext,
    ) -> Result<DecompositionResult> {
        match self.decomposer.decompose(query).await {
            Ok(result) => Ok(result),
            Err(error) if self.config.enable_fallbacks => {
                context.add_error(format!("Decomposition failed: {error}"));
                context.increment_fallback();

                // Fallback to single query
                Ok(DecompositionResult::single_query(query.to_string()))
            }
            Err(error) => Err(error),
        }
    }

    /// Execute dual-level retrieval for all subqueries
    async fn execute_dual_level_retrieval(
        &self,
        decomposition_result: &DecompositionResult,
        graph: &KnowledgeGraph,
        context: &mut ProcessingContext,
    ) -> Result<Vec<SubqueryResult>> {
        let mut all_results = Vec::new();

        for subquery in &decomposition_result.subqueries {
            let subquery_result = self
                .process_single_subquery(&subquery.text, graph, context)
                .await;

            match subquery_result {
                Ok(result) => all_results.push(result),
                Err(error) if self.config.enable_fallbacks => {
                    context.add_error(format!("Subquery '{}' failed: {}", subquery.text, error));
                    context.increment_fallback();

                    // Create fallback result
                    all_results.push(SubqueryResult {
                        subquery: subquery.text.clone(),
                        result_type: SubqueryResultType::Fallback,
                        confidence: 0.1,
                        content: "Unable to process this part of the query".to_string(),
                        sources: vec![],
                    });
                }
                Err(error) => return Err(error),
            }
        }

        if all_results.is_empty() {
            return Err(ProcessingError::AllStrategiesFailed {
                reason: "No subqueries could be processed".to_string(),
            }
            .into());
        }

        Ok(all_results)
    }

    /// Process a single subquery using dual-level retrieval
    async fn process_single_subquery(
        &self,
        subquery: &str,
        graph: &KnowledgeGraph,
        context: &mut ProcessingContext,
    ) -> Result<SubqueryResult> {
        // Level 1: Try logic form retrieval
        match self.logic_form_retriever.retrieve(subquery, graph).await {
            Ok(logic_result) => {
                return Ok(SubqueryResult::logic_form(
                    subquery.to_string(),
                    logic_result,
                ));
            }
            Err(error) if self.config.enable_fallbacks => {
                context.add_error(format!(
                    "Logic form retrieval failed for '{subquery}': {error}"
                ));
            }
            Err(error) => return Err(error),
        }

        // Level 2: Fallback to fuzzy matching
        let fuzzy_result = self
            .fuzzy_matcher
            .match_query(subquery, graph)
            .map_err(|error| {
                context.add_error(format!("Fuzzy matching failed for '{subquery}': {error}"));
                error
            })?;

        context.increment_fallback();
        Ok(SubqueryResult::fuzzy_match(
            subquery.to_string(),
            fuzzy_result,
        ))
    }

    /// Generate response from subquery results
    async fn generate_response(
        &self,
        query: &str,
        subquery_results: Vec<SubqueryResult>,
        intent_result: IntentResult,
        _context: &mut ProcessingContext,
    ) -> Result<RogragResponse> {
        if self.config.response_streaming {
            self.streaming_builder
                .build_streaming_response(query.to_string(), subquery_results, intent_result)
                .await
        } else {
            self.streaming_builder
                .build_complete_response(query.to_string(), subquery_results, intent_result)
                .await
        }
    }

    /// Get decomposition result for metrics (helper method)
    async fn get_decomposition_for_metrics(&self, query: &str) -> Option<DecompositionResult> {
        self.decomposer.decompose(query).await.ok()
    }

    /// Get quality metrics
    pub fn get_quality_metrics(&self) -> Result<QualityMetrics> {
        self.quality_metrics
            .lock()
            .map(|metrics| metrics.clone())
            .map_err(|_| {
                ProcessingError::ConfigurationError {
                    reason: "Failed to access quality metrics".to_string(),
                }
                .into()
            })
    }

    /// Get current configuration
    pub fn get_config(&self) -> &RogragConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RogragConfig) {
        self.config = config;
    }

    /// Check system health
    pub async fn health_check(&self) -> HealthCheckResult {
        let mut issues = Vec::new();
        let mut overall_health = HealthStatus::Healthy;

        // Check decomposer
        if !self.decomposer.can_decompose("test query") {
            issues.push("Query decomposer may not be functioning properly".to_string());
            overall_health = HealthStatus::Degraded;
        }

        // Check quality metrics
        if let Ok(metrics) = self.quality_metrics.lock() {
            let stats = metrics.get_performance_statistics();
            if stats.error_rate > 0.1 {
                issues.push(format!("High error rate: {:.1}%", stats.error_rate * 100.0));
                overall_health = HealthStatus::Unhealthy;
            }
            if stats.avg_processing_time_ms > 10000.0 {
                issues.push(format!(
                    "Slow processing: {:.0}ms average",
                    stats.avg_processing_time_ms
                ));
                if overall_health == HealthStatus::Healthy {
                    overall_health = HealthStatus::Degraded;
                }
            }
        } else {
            issues.push("Cannot access quality metrics".to_string());
            overall_health = HealthStatus::Degraded;
        }

        HealthCheckResult {
            status: overall_health,
            issues,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Get system statistics
    pub fn get_system_statistics(&self) -> SystemStatistics {
        let performance_stats = self
            .quality_metrics
            .lock()
            .map(|metrics| metrics.get_performance_statistics().clone())
            .unwrap_or_default();

        SystemStatistics {
            total_queries_processed: performance_stats.total_queries,
            average_processing_time_ms: performance_stats.avg_processing_time_ms,
            current_throughput_qps: performance_stats.throughput_qps,
            error_rate: performance_stats.error_rate,
            fallback_rate: performance_stats.fallback_rate,
            average_quality_score: performance_stats.avg_quality_score,
            decomposition_success_rate: if performance_stats.total_queries > 0 {
                performance_stats.successful_decompositions as f64
                    / performance_stats.total_queries as f64
            } else {
                0.0
            },
        }
    }

    /// Process multiple queries in batch
    pub async fn batch_process(
        &self,
        queries: &[&str],
        graph: &KnowledgeGraph,
    ) -> Vec<Result<RogragResponse>> {
        let mut results = Vec::with_capacity(queries.len());

        for query in queries {
            let result = self.process_query(query, graph).await;
            results.push(result);
        }

        results
    }

    /// Process multiple queries concurrently
    pub async fn concurrent_batch_process(
        &self,
        queries: &[&str],
        graph: &KnowledgeGraph,
        max_concurrent: usize,
    ) -> Vec<Result<RogragResponse>> {
        use futures::stream::{self, StreamExt};

        stream::iter(queries)
            .map(|query| async move { self.process_query(query, graph).await })
            .buffer_unordered(max_concurrent)
            .collect()
            .await
    }
}

/// Health check result for system monitoring.
///
/// Provides current system health status with any detected issues.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Current health status of the system.
    pub status: HealthStatus,

    /// List of detected issues or warnings.
    pub issues: Vec<String>,

    /// Unix timestamp when health check was performed.
    pub timestamp: u64,
}

/// Health status levels for system monitoring.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    /// System is operating normally.
    Healthy,

    /// System is operational but with degraded performance or warnings.
    Degraded,

    /// System has critical issues requiring attention.
    Unhealthy,
}

/// System statistics for performance monitoring.
///
/// Aggregated metrics across all processed queries for system analysis.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    /// Total number of queries processed since system start.
    pub total_queries_processed: usize,

    /// Average processing time per query in milliseconds.
    pub average_processing_time_ms: f64,

    /// Current throughput in queries per second.
    pub current_throughput_qps: f64,

    /// Error rate as a fraction (0.0 to 1.0).
    pub error_rate: f64,

    /// Rate of fallback usage as a fraction (0.0 to 1.0).
    pub fallback_rate: f64,

    /// Average quality score across all responses (0.0 to 1.0).
    pub average_quality_score: f64,

    /// Success rate of query decomposition (0.0 to 1.0).
    pub decomposition_success_rate: f64,
}

#[cfg(feature = "rograg")]
impl Default for SystemStatistics {
    fn default() -> Self {
        Self {
            total_queries_processed: 0,
            average_processing_time_ms: 0.0,
            current_throughput_qps: 0.0,
            error_rate: 0.0,
            fallback_rate: 0.0,
            average_quality_score: 0.0,
            decomposition_success_rate: 0.0,
        }
    }
}

/// Configuration for ROGRAG processing behavior.
///
/// Controls all aspects of the ROGRAG pipeline including feature toggles,
/// thresholds, and performance limits.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RogragConfig {
    /// Maximum time allowed for processing a single query.
    pub max_processing_time: Duration,

    /// Whether to classify query intent before processing.
    pub enable_intent_classification: bool,

    /// Whether to decompose complex queries into subqueries.
    pub enable_query_decomposition: bool,

    /// Whether to attempt logic form retrieval.
    pub enable_logic_form_retrieval: bool,

    /// Whether to use fuzzy matching as a fallback.
    pub enable_fuzzy_matching: bool,

    /// Confidence threshold for triggering fallback strategies (0.0 to 1.0).
    pub fallback_threshold: f32,

    /// Minimum acceptable quality score for responses (0.0 to 1.0).
    pub quality_threshold: f32,

    /// Maximum number of subqueries to generate per query.
    pub max_subqueries: usize,

    /// Whether to enable streaming response generation.
    pub response_streaming: bool,

    /// Whether to track quality metrics for processed queries.
    pub quality_tracking: bool,

    /// Whether to enable automatic fallback on failures.
    pub enable_fallbacks: bool,
}

#[cfg(feature = "rograg")]
impl Default for RogragConfig {
    fn default() -> Self {
        Self {
            max_processing_time: Duration::from_secs(30),
            enable_intent_classification: true,
            enable_query_decomposition: true,
            enable_logic_form_retrieval: true,
            enable_fuzzy_matching: true,
            fallback_threshold: 0.6,
            quality_threshold: 0.7,
            max_subqueries: 5,
            response_streaming: true,
            quality_tracking: true,
            enable_fallbacks: true,
        }
    }
}

/// Response from ROGRAG processing.
///
/// Contains the generated answer, confidence scores, sources, and processing metadata.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RogragResponse {
    /// The original query that was processed.
    pub query: String,

    /// Generated response content.
    pub content: String,

    /// Overall confidence in the response (0.0 to 1.0).
    pub confidence: f32,

    /// Source entity/chunk IDs used to generate the response.
    pub sources: Vec<String>,

    /// Intent classification result for the query.
    pub intent_result: IntentResult,

    /// Results from processing individual subqueries.
    pub subquery_results: Vec<SubqueryResult>,

    /// Processing performance statistics.
    pub processing_stats: ProcessingStats,

    /// Whether this is a refusal to answer.
    pub is_refusal: bool,

    /// Whether response was generated with streaming.
    pub is_streaming: bool,
}

#[cfg(feature = "rograg")]
impl RogragResponse {
    /// Create a refusal response.
    ///
    /// Used when the system determines a query should not be answered.
    ///
    /// # Arguments
    ///
    /// * `query` - The original query being refused
    /// * `reason` - Explanation for the refusal
    pub fn refusal(query: String, reason: String) -> Self {
        Self {
            query,
            content: format!("Unable to provide an answer: {reason}"),
            confidence: 0.0,
            sources: vec![],
            intent_result: IntentResult::default(),
            subquery_results: vec![],
            processing_stats: ProcessingStats::default(),
            is_refusal: true,
            is_streaming: false,
        }
    }
}

/// Result from processing a single subquery.
///
/// Contains the answer, confidence, and metadata for one decomposed subquery.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubqueryResult {
    /// The subquery text that was processed.
    pub subquery: String,

    /// Generated content for this subquery.
    pub content: String,

    /// Confidence in this subquery result (0.0 to 1.0).
    pub confidence: f32,

    /// Source IDs used for this subquery.
    pub sources: Vec<String>,

    /// Type of retrieval strategy used.
    pub result_type: SubqueryResultType,
}

#[cfg(feature = "rograg")]
impl SubqueryResult {
    /// Create a subquery result from logic form retrieval.
    ///
    /// # Arguments
    ///
    /// * `subquery` - The subquery text
    /// * `logic_result` - Result from logic form execution
    pub fn logic_form(subquery: String, logic_result: LogicFormResult) -> Self {
        Self {
            subquery,
            content: logic_result.answer,
            confidence: logic_result.confidence,
            sources: logic_result.sources,
            result_type: SubqueryResultType::LogicForm,
        }
    }

    /// Create a subquery result from fuzzy matching.
    ///
    /// # Arguments
    ///
    /// * `subquery` - The subquery text
    /// * `fuzzy_result` - Result from fuzzy matching
    pub fn fuzzy_match(subquery: String, fuzzy_result: FuzzyMatchResult) -> Self {
        Self {
            subquery,
            content: fuzzy_result.content,
            confidence: fuzzy_result.confidence,
            sources: fuzzy_result.sources,
            result_type: SubqueryResultType::FuzzyMatch,
        }
    }
}

/// Type of retrieval strategy used for subquery processing.
///
/// Indicates which retrieval method successfully answered the subquery.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SubqueryResultType {
    /// Result from logic form-based structured retrieval.
    LogicForm,

    /// Result from fuzzy matching fallback.
    FuzzyMatch,

    /// Generic fallback when other strategies failed.
    Fallback,
}

/// Processing statistics for a query.
///
/// Detailed timing breakdown and metrics for performance analysis.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ProcessingStats {
    /// Total processing time in milliseconds.
    pub total_time_ms: u64,

    /// Time spent on query decomposition in milliseconds.
    pub decomposition_time_ms: u64,

    /// Time spent on intent classification in milliseconds.
    pub intent_classification_time_ms: u64,

    /// Time spent on retrieval operations in milliseconds.
    pub retrieval_time_ms: u64,

    /// Time spent on response synthesis in milliseconds.
    pub synthesis_time_ms: u64,

    /// Time spent on validation in milliseconds.
    pub validation_time_ms: u64,

    /// Number of subqueries that were processed.
    pub subqueries_processed: usize,

    /// Whether fallback strategies were used.
    pub fallback_used: bool,
}

#[cfg(feature = "rograg")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Entity, EntityId, KnowledgeGraph};

    #[cfg(feature = "rograg")]
    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        let entity = Entity {
            id: EntityId::new("entity_1".to_string()),
            name: "Entity Name".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        };

        graph.add_entity(entity).unwrap();
        graph
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_processor_creation() {
        let processor = RogragProcessor::new();
        assert!(processor.is_ok());
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_processor_with_config() {
        let config = RogragConfig {
            max_subqueries: 3,
            quality_threshold: 0.8,
            ..Default::default()
        };

        let processor = RogragProcessor::with_config(config);
        assert!(processor.is_ok());
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_query_processing() {
        let processor = RogragProcessor::new().unwrap();
        let graph = create_test_graph();

        let result = processor.process_query("What is Entity Name?", &graph).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.content.is_empty());
        assert!(response.confidence >= 0.0);
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_batch_processing() {
        let processor = RogragProcessor::new().unwrap();
        let graph = create_test_graph();

        let queries = vec!["What is Entity Name?", "Who is the entity?"];
        let results = processor.batch_process(&queries, &graph).await;

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_health_check() {
        let processor = RogragProcessor::new().unwrap();
        let health = processor.health_check().await;

        // Should be healthy for a new processor
        assert!(matches!(
            health.status,
            HealthStatus::Healthy | HealthStatus::Degraded
        ));
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_system_statistics() {
        let processor = RogragProcessor::new().unwrap();
        let stats = processor.get_system_statistics();

        assert_eq!(stats.total_queries_processed, 0);
        assert_eq!(stats.average_processing_time_ms, 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_builder_pattern() {
        let processor = RogragProcessor::builder()
            .with_config(RogragConfig {
                max_subqueries: 10,
                ..Default::default()
            })
            .build();

        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().get_config().max_subqueries, 10);
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_concurrent_batch_processing() {
        let processor = RogragProcessor::new().unwrap();
        let graph = create_test_graph();

        let queries = vec![
            "What is Entity Name?",
            "Who is the entity?",
            "Tell me about entities",
        ];
        let results = processor
            .concurrent_batch_process(&queries, &graph, 2)
            .await;

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
