//! Query validation and verification for ROGRAG system
//!
//! Provides comprehensive validation of queries and responses to ensure
//! quality, safety, and appropriateness.

#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use crate::rograg::RogragResponse;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Error types for validation
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Query validation failed: {reason}")]
    QueryValidationFailed { reason: String },

    #[error("Response validation failed: {reason}")]
    ResponseValidationFailed { reason: String },

    #[error("Content safety violation: {violation_type}")]
    SafetyViolation { violation_type: String },

    #[error("Quality threshold not met: {metric} = {value}, required >= {threshold}")]
    QualityThresholdNotMet { metric: String, value: f32, threshold: f32 },
}

/// Configuration for validation
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub enable_query_validation: bool,
    pub enable_response_validation: bool,
    pub enable_safety_checks: bool,
    pub enable_quality_checks: bool,
    pub min_confidence_threshold: f32,
    pub max_response_length: usize,
    pub min_response_length: usize,
    pub enable_coherence_check: bool,
    pub enable_relevance_check: bool,
    pub enable_factual_consistency_check: bool,
}

#[cfg(feature = "rograg")]
impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_query_validation: true,
            enable_response_validation: true,
            enable_safety_checks: true,
            enable_quality_checks: true,
            min_confidence_threshold: 0.3,
            max_response_length: 4096,
            min_response_length: 10,
            enable_coherence_check: true,
            enable_relevance_check: true,
            enable_factual_consistency_check: true,
        }
    }
}

/// Result of validation
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub validation_score: f32,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
    pub quality_metrics: QualityMetrics,
}

/// A validation issue
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub recommendation: Option<String>,
}

/// Type of validation issue
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum IssueType {
    Safety,
    Quality,
    Coherence,
    Relevance,
    Length,
    Confidence,
    Consistency,
    Format,
}

/// Severity of validation issue
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality metrics for validation
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityMetrics {
    pub coherence_score: f32,
    pub relevance_score: f32,
    pub factual_consistency_score: f32,
    pub completeness_score: f32,
    pub readability_score: f32,
    pub source_credibility_score: f32,
}

/// Query validator implementation
#[cfg(feature = "rograg")]
pub struct QueryValidator {
    config: ValidationConfig,
    safety_patterns: Vec<regex::Regex>,
    quality_checks: Vec<Box<dyn QualityCheck>>,
    coherence_checker: CoherenceChecker,
    relevance_checker: RelevanceChecker,
}

/// Trait for quality checks
#[cfg(feature = "rograg")]
pub trait QualityCheck: Send + Sync {
    fn check(&self, response: &RogragResponse) -> Result<QualityCheckResult>;
    fn name(&self) -> &str;
}

/// Result of a quality check
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct QualityCheckResult {
    pub passed: bool,
    pub score: f32,
    pub issues: Vec<ValidationIssue>,
}

#[cfg(feature = "rograg")]
impl Default for QueryValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl QueryValidator {
    /// Create a new query validator
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new query validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        let mut validator = Self {
            config,
            safety_patterns: vec![],
            quality_checks: vec![],
            coherence_checker: CoherenceChecker::new(),
            relevance_checker: RelevanceChecker::new(),
        };

        validator.initialize_safety_patterns().unwrap();
        validator.initialize_quality_checks();
        validator
    }

    /// Initialize safety patterns
    fn initialize_safety_patterns(&mut self) -> Result<()> {
        if self.config.enable_safety_checks {
            self.safety_patterns = vec![
                regex::Regex::new(r"\b(?:harm|violence|illegal|inappropriate|offensive)\b")?,
                regex::Regex::new(r"\b(?:hate|discrimination|bias|prejudice)\b")?,
                regex::Regex::new(r"\b(?:private|confidential|secret|classified)\b")?,
                // Add more safety patterns as needed
            ];
        }
        Ok(())
    }

    /// Initialize quality checks
    fn initialize_quality_checks(&mut self) {
        if self.config.enable_quality_checks {
            self.quality_checks.push(Box::new(LengthCheck::new(
                self.config.min_response_length,
                self.config.max_response_length,
            )));
            self.quality_checks.push(Box::new(ConfidenceCheck::new(
                self.config.min_confidence_threshold,
            )));
            self.quality_checks.push(Box::new(SourceCredibilityCheck::new()));
            self.quality_checks.push(Box::new(CompletenessCheck::new()));
        }
    }

    /// Validate a query before processing
    pub fn validate_query(&self, query: &str) -> Result<ValidationResult> {
        if !self.config.enable_query_validation {
            return Ok(ValidationResult {
                is_valid: true,
                validation_score: 1.0,
                issues: vec![],
                recommendations: vec![],
                quality_metrics: QualityMetrics::default(),
            });
        }

        let mut issues = Vec::new();

        // Check query length
        if query.trim().is_empty() {
            issues.push(ValidationIssue {
                issue_type: IssueType::Length,
                severity: IssueSeverity::Critical,
                description: "Query is empty".to_string(),
                recommendation: Some("Please provide a non-empty query".to_string()),
            });
        } else if query.len() > 1000 {
            issues.push(ValidationIssue {
                issue_type: IssueType::Length,
                severity: IssueSeverity::Medium,
                description: "Query is very long".to_string(),
                recommendation: Some("Consider shortening your query for better results".to_string()),
            });
        }

        // Safety checks
        if self.config.enable_safety_checks {
            for pattern in &self.safety_patterns {
                if pattern.is_match(&query.to_lowercase()) {
                    issues.push(ValidationIssue {
                        issue_type: IssueType::Safety,
                        severity: IssueSeverity::High,
                        description: "Query contains potentially inappropriate content".to_string(),
                        recommendation: Some("Please rephrase your query appropriately".to_string()),
                    });
                    break;
                }
            }
        }

        // Check for basic structure
        if !query.contains(char::is_alphabetic) {
            issues.push(ValidationIssue {
                issue_type: IssueType::Format,
                severity: IssueSeverity::High,
                description: "Query contains no alphabetic characters".to_string(),
                recommendation: Some("Please provide a text-based query".to_string()),
            });
        }

        let has_critical_issues = issues.iter().any(|i| matches!(i.severity, IssueSeverity::Critical));
        let validation_score = if has_critical_issues {
            0.0
        } else {
            1.0 - (issues.len() as f32 * 0.1).min(0.8)
        };

        Ok(ValidationResult {
            is_valid: !has_critical_issues,
            validation_score,
            issues,
            recommendations: vec![],
            quality_metrics: QualityMetrics::default(),
        })
    }

    /// Validate a response after generation
    pub fn validate_response(&self, response: &RogragResponse) -> Result<RogragResponse> {
        if !self.config.enable_response_validation {
            return Ok(response.clone());
        }

        let mut issues = Vec::new();
        let mut quality_metrics = QualityMetrics::default();

        // Run quality checks
        for check in &self.quality_checks {
            let result = check.check(response)?;
            if !result.passed {
                issues.extend(result.issues);
            }
        }

        // Coherence check
        if self.config.enable_coherence_check {
            let coherence_result = self.coherence_checker.check_coherence(response)?;
            quality_metrics.coherence_score = coherence_result.score;
            if !coherence_result.passed {
                issues.extend(coherence_result.issues);
            }
        }

        // Relevance check
        if self.config.enable_relevance_check {
            let relevance_result = self.relevance_checker.check_relevance(response)?;
            quality_metrics.relevance_score = relevance_result.score;
            if !relevance_result.passed {
                issues.extend(relevance_result.issues);
            }
        }

        // Factual consistency check
        if self.config.enable_factual_consistency_check {
            let consistency_score = self.check_factual_consistency(response)?;
            quality_metrics.factual_consistency_score = consistency_score;
            if consistency_score < 0.5 {
                issues.push(ValidationIssue {
                    issue_type: IssueType::Consistency,
                    severity: IssueSeverity::Medium,
                    description: "Response may contain factual inconsistencies".to_string(),
                    recommendation: Some("Verify information with additional sources".to_string()),
                });
            }
        }

        // Safety check on response content
        if self.config.enable_safety_checks {
            for pattern in &self.safety_patterns {
                if pattern.is_match(&response.content.to_lowercase()) {
                    issues.push(ValidationIssue {
                        issue_type: IssueType::Safety,
                        severity: IssueSeverity::High,
                        description: "Response contains potentially inappropriate content".to_string(),
                        recommendation: Some("Response should be reviewed before delivery".to_string()),
                    });
                    break;
                }
            }
        }

        // Calculate overall quality metrics
        quality_metrics.completeness_score = self.calculate_completeness_score(response);
        quality_metrics.readability_score = self.calculate_readability_score(response);
        quality_metrics.source_credibility_score = self.calculate_source_credibility_score(response);

        // Create validated response
        let mut validated_response = response.clone();

        // Apply any necessary modifications based on validation results
        if issues.iter().any(|i| matches!(i.severity, IssueSeverity::Critical)) {
            // For critical issues, modify the response
            validated_response.content = "I apologize, but I cannot provide a response to this query due to safety or quality concerns.".to_string();
            validated_response.confidence = 0.0;
            validated_response.is_refusal = true;
        }

        Ok(validated_response)
    }

    /// Check factual consistency
    fn check_factual_consistency(&self, response: &RogragResponse) -> Result<f32> {
        // Simple heuristic: check for contradictions within the response
        let sentences: Vec<&str> = response.content.split(['.', '!', '?']).collect();

        // Look for obvious contradictions
        let mut contradiction_count = 0;
        let contradiction_patterns = [
            ("not", "is"),
            ("never", "always"),
            ("none", "all"),
            ("impossible", "possible"),
        ];

        for sentence in &sentences {
            let sentence_lower = sentence.to_lowercase();
            for (neg, pos) in &contradiction_patterns {
                if sentence_lower.contains(neg) && sentence_lower.contains(pos) {
                    contradiction_count += 1;
                }
            }
        }

        // Simple scoring based on contradiction density
        let max_contradictions = sentences.len().max(1);
        let consistency_score = 1.0 - (contradiction_count as f32 / max_contradictions as f32).min(1.0);

        Ok(consistency_score)
    }

    /// Calculate completeness score
    fn calculate_completeness_score(&self, response: &RogragResponse) -> f32 {
        // Heuristic based on response length, source count, and subquery coverage
        let length_score = if response.content.len() > 100 { 1.0 } else { response.content.len() as f32 / 100.0 };
        let source_score = if response.sources.len() > 2 { 1.0 } else { response.sources.len() as f32 / 2.0 };
        let subquery_score = if response.subquery_results.len() > 1 { 1.0 } else { response.subquery_results.len() as f32 };

        (length_score + source_score + subquery_score) / 3.0
    }

    /// Calculate readability score
    fn calculate_readability_score(&self, response: &RogragResponse) -> f32 {
        let text = &response.content;
        let word_count = text.split_whitespace().count();
        let sentence_count = text.chars().filter(|&c| c == '.' || c == '!' || c == '?').count().max(1);

        // Simple readability heuristic
        let avg_words_per_sentence = word_count as f32 / sentence_count as f32;
        let avg_word_length = text.chars().filter(|c| c.is_alphabetic()).count() as f32 / word_count.max(1) as f32;

        // Score based on reasonable sentence and word lengths
        let sentence_score = if avg_words_per_sentence > 30.0 { 0.5 } else { 1.0 };
        let word_score = if avg_word_length > 8.0 { 0.7 } else { 1.0 };

        (sentence_score + word_score) / 2.0
    }

    /// Calculate source credibility score
    fn calculate_source_credibility_score(&self, response: &RogragResponse) -> f32 {
        if response.sources.is_empty() {
            return 0.0;
        }

        // Simple heuristic: more sources = higher credibility, up to a point
        let source_count_score = (response.sources.len() as f32 / 5.0).min(1.0);

        // Check for source diversity (simple heuristic)
        let unique_source_prefixes: std::collections::HashSet<String> = response.sources.iter()
            .map(|s| s.chars().take(10).collect())
            .collect();
        let diversity_score = unique_source_prefixes.len() as f32 / response.sources.len() as f32;

        (source_count_score + diversity_score) / 2.0
    }

    /// Get configuration
    pub fn get_config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ValidationConfig) -> Result<()> {
        self.config = config;
        self.initialize_safety_patterns()?;
        self.quality_checks.clear();
        self.initialize_quality_checks();
        Ok(())
    }
}

/// Length check implementation
#[cfg(feature = "rograg")]
pub struct LengthCheck {
    min_length: usize,
    max_length: usize,
}

#[cfg(feature = "rograg")]
impl LengthCheck {
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self { min_length, max_length }
    }
}

#[cfg(feature = "rograg")]
impl QualityCheck for LengthCheck {
    fn check(&self, response: &RogragResponse) -> Result<QualityCheckResult> {
        let length = response.content.len();
        let mut issues = Vec::new();

        if length < self.min_length {
            issues.push(ValidationIssue {
                issue_type: IssueType::Length,
                severity: IssueSeverity::Medium,
                description: format!("Response too short: {length} characters"),
                recommendation: Some("Response should be more detailed".to_string()),
            });
        }

        if length > self.max_length {
            issues.push(ValidationIssue {
                issue_type: IssueType::Length,
                severity: IssueSeverity::Medium,
                description: format!("Response too long: {length} characters"),
                recommendation: Some("Response should be more concise".to_string()),
            });
        }

        let score = if issues.is_empty() { 1.0 } else { 0.5 };

        Ok(QualityCheckResult {
            passed: issues.is_empty(),
            score,
            issues,
        })
    }

    fn name(&self) -> &str {
        "length_check"
    }
}

/// Confidence check implementation
#[cfg(feature = "rograg")]
pub struct ConfidenceCheck {
    min_confidence: f32,
}

#[cfg(feature = "rograg")]
impl ConfidenceCheck {
    pub fn new(min_confidence: f32) -> Self {
        Self { min_confidence }
    }
}

#[cfg(feature = "rograg")]
impl QualityCheck for ConfidenceCheck {
    fn check(&self, response: &RogragResponse) -> Result<QualityCheckResult> {
        let mut issues = Vec::new();

        if response.confidence < self.min_confidence {
            issues.push(ValidationIssue {
                issue_type: IssueType::Confidence,
                severity: IssueSeverity::High,
                description: format!("Low confidence: {:.2}", response.confidence),
                recommendation: Some("Consider gathering more information".to_string()),
            });
        }

        Ok(QualityCheckResult {
            passed: issues.is_empty(),
            score: response.confidence,
            issues,
        })
    }

    fn name(&self) -> &str {
        "confidence_check"
    }
}

/// Source credibility check implementation
#[cfg(feature = "rograg")]
pub struct SourceCredibilityCheck;

#[cfg(feature = "rograg")]
impl Default for SourceCredibilityCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl SourceCredibilityCheck {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "rograg")]
impl QualityCheck for SourceCredibilityCheck {
    fn check(&self, response: &RogragResponse) -> Result<QualityCheckResult> {
        let mut issues = Vec::new();

        if response.sources.is_empty() {
            issues.push(ValidationIssue {
                issue_type: IssueType::Quality,
                severity: IssueSeverity::Medium,
                description: "No sources provided".to_string(),
                recommendation: Some("Response should cite sources".to_string()),
            });
        }

        let score = if response.sources.is_empty() { 0.0 } else { 0.8 };

        Ok(QualityCheckResult {
            passed: !response.sources.is_empty(),
            score,
            issues,
        })
    }

    fn name(&self) -> &str {
        "source_credibility_check"
    }
}

/// Completeness check implementation
#[cfg(feature = "rograg")]
pub struct CompletenessCheck;

#[cfg(feature = "rograg")]
impl Default for CompletenessCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl CompletenessCheck {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "rograg")]
impl QualityCheck for CompletenessCheck {
    fn check(&self, response: &RogragResponse) -> Result<QualityCheckResult> {
        let mut issues = Vec::new();

        // Check if response addresses the query
        let query_lower = response.query.to_lowercase();
        let answer_lower = response.content.to_lowercase();

        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter out short words
            .collect();

        let answer_words: std::collections::HashSet<&str> = answer_lower
            .split_whitespace()
            .collect();

        let overlap = query_words.intersection(&answer_words).count();
        let relevance = if query_words.is_empty() { 1.0 } else { overlap as f32 / query_words.len() as f32 };

        if relevance < 0.3 {
            issues.push(ValidationIssue {
                issue_type: IssueType::Relevance,
                severity: IssueSeverity::Medium,
                description: "Response may not fully address the query".to_string(),
                recommendation: Some("Ensure response directly answers the question".to_string()),
            });
        }

        Ok(QualityCheckResult {
            passed: relevance >= 0.3,
            score: relevance,
            issues,
        })
    }

    fn name(&self) -> &str {
        "completeness_check"
    }
}

/// Coherence checker
#[cfg(feature = "rograg")]
pub struct CoherenceChecker;

#[cfg(feature = "rograg")]
impl Default for CoherenceChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl CoherenceChecker {
    pub fn new() -> Self {
        Self
    }

    pub fn check_coherence(&self, response: &RogragResponse) -> Result<QualityCheckResult> {
        let mut issues = Vec::new();

        // Simple coherence check: look for logical flow
        let sentences: Vec<&str> = response.content.split(['.', '!', '?']).collect();

        // Check for abrupt topic changes (very simple heuristic)
        let mut coherence_score = 1.0;

        if sentences.len() > 1 {
            // Look for connecting words that indicate good flow
            let connectors = ["however", "therefore", "furthermore", "additionally", "meanwhile", "consequently"];
            let connector_count = sentences.iter()
                .filter(|s| connectors.iter().any(|c| s.to_lowercase().contains(c)))
                .count();

            coherence_score = (connector_count as f32 / sentences.len() as f32).min(1.0);

            if coherence_score < 0.3 {
                issues.push(ValidationIssue {
                    issue_type: IssueType::Coherence,
                    severity: IssueSeverity::Low,
                    description: "Response may lack logical flow".to_string(),
                    recommendation: Some("Consider improving transitions between ideas".to_string()),
                });
            }
        }

        Ok(QualityCheckResult {
            passed: coherence_score >= 0.3,
            score: coherence_score,
            issues,
        })
    }
}

/// Relevance checker
#[cfg(feature = "rograg")]
pub struct RelevanceChecker;

#[cfg(feature = "rograg")]
impl Default for RelevanceChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl RelevanceChecker {
    pub fn new() -> Self {
        Self
    }

    pub fn check_relevance(&self, response: &RogragResponse) -> Result<QualityCheckResult> {
        let mut issues = Vec::new();

        // Check semantic relevance between query and answer
        let query_lower = response.query.to_lowercase();
        let answer_lower = response.content.to_lowercase();

        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let answer_words: Vec<&str> = answer_lower.split_whitespace().collect();

        // Calculate word overlap
        let query_set: std::collections::HashSet<&str> = query_words.iter().copied().collect();
        let answer_set: std::collections::HashSet<&str> = answer_words.iter().copied().collect();

        let intersection = query_set.intersection(&answer_set).count();
        let union = query_set.union(&answer_set).count();

        let relevance_score = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };

        if relevance_score < 0.2 {
            issues.push(ValidationIssue {
                issue_type: IssueType::Relevance,
                severity: IssueSeverity::High,
                description: "Response appears unrelated to query".to_string(),
                recommendation: Some("Ensure response directly addresses the query".to_string()),
            });
        }

        Ok(QualityCheckResult {
            passed: relevance_score >= 0.2,
            score: relevance_score,
            issues,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rograg::{IntentResult, QueryIntent, ProcessingStats, SubqueryResult, SubqueryResultType};

    #[cfg(feature = "rograg")]
    fn create_test_response() -> RogragResponse {
        RogragResponse {
            query: "What is Entity Name?".to_string(),
            content: "Entity Name is a young boy character in Mark Twain's novels. He is adventurous and mischievous.".to_string(),
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
    #[test]
    fn test_query_validation_valid() {
        let validator = QueryValidator::new();
        let result = validator.validate_query("What is Entity Name?").unwrap();

        assert!(result.is_valid);
        assert!(result.validation_score > 0.8);
        assert!(result.issues.is_empty());
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_query_validation_empty() {
        let validator = QueryValidator::new();
        let result = validator.validate_query("").unwrap();

        assert!(!result.is_valid);
        assert_eq!(result.validation_score, 0.0);
        assert!(!result.issues.is_empty());
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_response_validation() {
        let validator = QueryValidator::new();
        let response = create_test_response();

        let validated = validator.validate_response(&response).unwrap();

        assert!(!validated.is_refusal);
        assert!(validated.confidence > 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_length_check() {
        let check = LengthCheck::new(10, 100);
        let response = create_test_response();

        let result = check.check(&response).unwrap();
        assert!(result.passed); // Should pass as the response is reasonable length
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_confidence_check() {
        let check = ConfidenceCheck::new(0.5);
        let response = create_test_response();

        let result = check.check(&response).unwrap();
        assert!(result.passed); // Should pass as confidence is 0.8 > 0.5
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_source_credibility_check() {
        let check = SourceCredibilityCheck::new();
        let response = create_test_response();

        let result = check.check(&response).unwrap();
        assert!(result.passed); // Should pass as response has sources
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_coherence_check() {
        let checker = CoherenceChecker::new();
        let response = create_test_response();

        let result = checker.check_coherence(&response).unwrap();
        assert!(result.score >= 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_relevance_check() {
        let checker = RelevanceChecker::new();
        let response = create_test_response();

        let result = checker.check_relevance(&response).unwrap();
        assert!(result.score > 0.0); // Should have some relevance
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_factual_consistency() {
        let validator = QueryValidator::new();
        let response = create_test_response();

        let consistency_score = validator.check_factual_consistency(&response).unwrap();
        assert!((0.0..=1.0).contains(&consistency_score));
    }
}