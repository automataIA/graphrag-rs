//! Intent classification for ROGRAG system
//!
//! Classifies query intent and determines whether the system should attempt
//! to answer the query or refuse to answer based on confidence and appropriateness.

#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use std::collections::HashMap;
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Errors that can occur during intent classification.
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum IntentClassificationError {
    /// Unable to determine the query's intent from available patterns.
    ///
    /// Occurs when the query doesn't match any known intent patterns and
    /// cannot be reliably classified. Consider query reformulation.
    #[error("Unable to classify query intent: {query}")]
    CannotClassify {
        /// The query text that could not be classified.
        query: String
    },

    /// Multiple intents detected with similar confidence scores.
    ///
    /// Occurs when the query could be interpreted in multiple ways with
    /// similar likelihood. The query should be clarified or rephrased.
    #[error("Ambiguous intent detected: {intents:?}")]
    AmbiguousIntent {
        /// List of competing intents with similar confidence scores.
        intents: Vec<QueryIntent>
    },

    /// Classification confidence below acceptable threshold.
    ///
    /// Occurs when the best intent match has low confidence (<0.7 by default).
    /// The system may refuse to answer or suggest query reformulation.
    #[error("Insufficient confidence for classification: {confidence}")]
    InsufficientConfidence {
        /// The confidence score that was below the threshold.
        confidence: f32
    },
}

/// Classification of query intent types.
///
/// Each intent type represents a different information-seeking goal and may
/// require different retrieval and generation strategies.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Factual information requests about specific attributes or facts.
    ///
    /// Examples: "What is X?", "How many Y are there?"
    Factual,

    /// Requests for definitions, explanations, or conceptual understanding.
    ///
    /// Examples: "Define X", "Explain what Y means"
    Definitional,

    /// Requests about relationships or connections between entities.
    ///
    /// Examples: "How are X and Y related?", "What is the connection between A and B?"
    Relational,

    /// Temporal information requests about timing, sequence, or duration.
    ///
    /// Examples: "When did X happen?", "What year was Y?"
    Temporal,

    /// Causal information requests about reasons, causes, or effects.
    ///
    /// Examples: "Why did X happen?", "What caused Y?"
    Causal,

    /// Comparative analysis requests between two or more entities.
    ///
    /// Examples: "Compare X and Y", "What's the difference between A and B?"
    Comparative,

    /// Exploratory or open-ended questions requiring broader analysis.
    ///
    /// Examples: General questions without clear structure
    Exploratory,

    /// Requests for summaries, overviews, or high-level descriptions.
    ///
    /// Examples: "Summarize X", "Tell me about Y", "Give an overview of Z"
    Summary,

    /// Inappropriate or harmful requests that should be refused.
    ///
    /// Examples: Queries containing hate speech, violence, illegal content
    Inappropriate,

    /// Ambiguous or unclear requests that cannot be reliably interpreted.
    ///
    /// Occurs when intent is unclear or multiple intents have similar confidence.
    Ambiguous,
}

/// Result of classifying a query's intent.
///
/// Contains the primary intent, confidence scores, and guidance on whether
/// the system should refuse to answer.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    /// The primary classified intent with highest confidence.
    pub primary_intent: QueryIntent,

    /// Secondary intents with their confidence scores.
    ///
    /// Useful for detecting multi-intent queries or ambiguous cases.
    pub secondary_intents: Vec<(QueryIntent, f32)>,

    /// Confidence in the primary intent classification (0.0 to 1.0).
    ///
    /// Higher values indicate more reliable classification.
    pub confidence: f32,

    /// Whether the system should refuse to answer this query.
    ///
    /// Set to `true` for inappropriate content, low confidence, or ambiguity.
    pub should_refuse: bool,

    /// Reason for refusal, if `should_refuse` is true.
    pub refusal_reason: Option<String>,

    /// Suggested query reformulation for better results.
    ///
    /// Provided when confidence is low or intent is ambiguous.
    pub suggested_reformulation: Option<String>,

    /// Query complexity score (0.0 to 1.0).
    ///
    /// Based on length, structure, and linguistic complexity.
    pub complexity_score: f32,
}

#[cfg(feature = "rograg")]
impl Default for IntentResult {
    fn default() -> Self {
        Self {
            primary_intent: QueryIntent::Exploratory,
            secondary_intents: vec![],
            confidence: 0.0,
            should_refuse: false,
            refusal_reason: None,
            suggested_reformulation: None,
            complexity_score: 0.0,
        }
    }
}

/// Configuration for intent classification behavior.
///
/// Controls thresholds, detection features, and reformulation suggestions.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct IntentClassificationConfig {
    /// Minimum confidence required for classification (0.0 to 1.0).
    ///
    /// Default: 0.7. Classifications below this threshold are considered unreliable.
    pub confidence_threshold: f32,

    /// Threshold for refusing to answer (0.0 to 1.0).
    ///
    /// Default: 0.8. Queries with confidence below this threshold trigger refusal.
    pub refusal_threshold: f32,

    /// Whether to detect and refuse inappropriate content.
    ///
    /// Default: true. When enabled, scans for hate speech, violence, etc.
    pub enable_inappropriate_detection: bool,

    /// Whether to detect ambiguous intent.
    ///
    /// Default: true. When enabled, identifies queries with multiple competing intents.
    pub enable_ambiguity_detection: bool,

    /// Whether to suggest query reformulations.
    ///
    /// Default: true. When enabled, provides suggestions for improving unclear queries.
    pub suggest_reformulations: bool,
}

#[cfg(feature = "rograg")]
impl Default for IntentClassificationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            refusal_threshold: 0.8,
            enable_inappropriate_detection: true,
            enable_ambiguity_detection: true,
            suggest_reformulations: true,
        }
    }
}

/// Intent classifier for query analysis and refusal detection.
///
/// Uses pattern matching (keywords and regex) to classify query intent and
/// determine whether queries should be answered or refused. Supports detection
/// of inappropriate content, ambiguous queries, and provides reformulation suggestions.
///
/// # Classification Process
///
/// 1. Check for inappropriate content (if enabled)
/// 2. Score query against each intent pattern
/// 3. Detect ambiguity if secondary intents have high confidence
/// 4. Determine if query should be refused based on thresholds
/// 5. Generate reformulation suggestions if needed
///
/// # Supported Intents
///
/// - Factual, Definitional, Relational
/// - Temporal, Causal, Comparative
/// - Exploratory, Summary
/// - Inappropriate, Ambiguous (special cases)
///
/// # Example
///
/// ```rust,ignore
/// use graphrag_core::rograg::IntentClassifier;
///
/// let classifier = IntentClassifier::new()?;
/// let result = classifier.classify("What is X?")?;
///
/// if result.should_refuse {
///     println!("Refusing: {}", result.refusal_reason.unwrap());
/// } else {
///     println!("Intent: {:?}", result.primary_intent);
/// }
/// ```
#[cfg(feature = "rograg")]
pub struct IntentClassifier {
    config: IntentClassificationConfig,
    intent_patterns: HashMap<QueryIntent, Vec<IntentPattern>>,
    inappropriate_patterns: Vec<regex::Regex>,
}

#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
struct IntentPattern {
    keywords: Vec<String>,
    patterns: Vec<regex::Regex>,
    weight: f32,
    requires_all: bool, // If true, all keywords must be present
}

#[cfg(feature = "rograg")]
impl IntentClassifier {
    /// Create a new intent classifier with default configuration.
    ///
    /// Initializes pattern databases for all supported intent types.
    ///
    /// # Returns
    ///
    /// Returns an `IntentClassifier` with default thresholds and all detection
    /// features enabled, or an error if pattern initialization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if regex pattern compilation fails during initialization.
    pub fn new() -> Result<Self> {
        let config = IntentClassificationConfig::default();
        let mut classifier = Self {
            config,
            intent_patterns: HashMap::new(),
            inappropriate_patterns: vec![],
        };

        classifier.initialize_patterns()?;
        Ok(classifier)
    }

    /// Create a new intent classifier with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom configuration for classification behavior
    ///
    /// # Returns
    ///
    /// Returns an `IntentClassifier` with the specified configuration, or an
    /// error if pattern initialization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if regex pattern compilation fails during initialization.
    pub fn with_config(config: IntentClassificationConfig) -> Result<Self> {
        let mut classifier = Self {
            config,
            intent_patterns: HashMap::new(),
            inappropriate_patterns: vec![],
        };

        classifier.initialize_patterns()?;
        Ok(classifier)
    }

    /// Initialize intent patterns
    fn initialize_patterns(&mut self) -> Result<()> {
        // Factual intent patterns
        self.add_intent_pattern(QueryIntent::Factual, IntentPattern {
            keywords: ["what", "which", "how many", "how much"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\bwhat (?:is|are|was|were)\b")?,
                regex::Regex::new(r"\bwhich (?:is|are|was|were)\b")?,
                regex::Regex::new(r"\bhow many\b")?,
                regex::Regex::new(r"\bhow much\b")?,
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Definitional intent patterns
        self.add_intent_pattern(QueryIntent::Definitional, IntentPattern {
            keywords: ["define", "definition", "meaning", "explain", "what is"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\bdefine\b")?,
                regex::Regex::new(r"\bdefinition of\b")?,
                regex::Regex::new(r"\bmeaning of\b")?,
                regex::Regex::new(r"\bexplain what\b")?,
                regex::Regex::new(r"\bwhat (?:is|are) (?:the )?(?:concept|idea|notion) of\b")?,
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Relational intent patterns
        self.add_intent_pattern(QueryIntent::Relational, IntentPattern {
            keywords: ["relationship", "related", "connection", "between", "and"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\brelationship between\b")?,
                regex::Regex::new(r"\bhow (?:is|are) .+ related to\b")?,
                regex::Regex::new(r"\bconnection between\b")?,
                regex::Regex::new(r"\b\w+ and \w+\b")?, // Simple pattern for "X and Y"
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Temporal intent patterns
        self.add_intent_pattern(QueryIntent::Temporal, IntentPattern {
            keywords: ["when", "time", "date", "year", "before", "after", "during"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\bwhen (?:did|was|were|will|is|are)\b")?,
                regex::Regex::new(r"\bwhat (?:time|date|year)\b")?,
                regex::Regex::new(r"\bbefore .+ happened\b")?,
                regex::Regex::new(r"\bafter .+ happened\b")?,
                regex::Regex::new(r"\bduring .+ period\b")?,
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Causal intent patterns
        self.add_intent_pattern(QueryIntent::Causal, IntentPattern {
            keywords: ["why", "because", "cause", "reason", "result", "due to"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\bwhy (?:did|was|were|is|are|do|does)\b")?,
                regex::Regex::new(r"\bwhat (?:caused|causes)\b")?,
                regex::Regex::new(r"\breason for\b")?,
                regex::Regex::new(r"\bdue to what\b")?,
                regex::Regex::new(r"\bwhat led to\b")?,
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Comparative intent patterns
        self.add_intent_pattern(QueryIntent::Comparative, IntentPattern {
            keywords: ["compare", "difference", "versus", "vs", "better", "worse", "similar"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\bcompare .+ (?:to|with|and)\b")?,
                regex::Regex::new(r"\bdifference between\b")?,
                regex::Regex::new(r"\b.+ (?:versus|vs) .+\b")?,
                regex::Regex::new(r"\bwhich is (?:better|worse)\b")?,
                regex::Regex::new(r"\bhow (?:similar|different)\b")?,
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Summary intent patterns
        self.add_intent_pattern(QueryIntent::Summary, IntentPattern {
            keywords: ["summarize", "overview", "summary", "tell me about", "describe"].iter().map(|s| s.to_string()).collect(),
            patterns: vec![
                regex::Regex::new(r"\bsummarize\b")?,
                regex::Regex::new(r"\bgive (?:me )?(?:an )?overview\b")?,
                regex::Regex::new(r"\btell me about\b")?,
                regex::Regex::new(r"\bdescribe .+\b")?,
                regex::Regex::new(r"\bwhat (?:can you tell me )?about\b")?,
            ],
            weight: 1.0,
            requires_all: false,
        });

        // Initialize inappropriate content patterns
        if self.config.enable_inappropriate_detection {
            self.inappropriate_patterns = vec![
                regex::Regex::new(r"\b(?:hate|violence|harm|illegal|inappropriate)\b")?,
                // Add more patterns as needed
            ];
        }

        Ok(())
    }

    /// Add an intent pattern
    fn add_intent_pattern(&mut self, intent: QueryIntent, pattern: IntentPattern) {
        self.intent_patterns.entry(intent).or_default().push(pattern);
    }

    /// Classify the intent of a query.
    ///
    /// Analyzes the query to determine its primary intent, detect ambiguity,
    /// identify inappropriate content, and decide whether to refuse answering.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to classify
    ///
    /// # Returns
    ///
    /// Returns an `IntentResult` containing:
    /// - Primary and secondary intents with confidence scores
    /// - Whether to refuse answering
    /// - Refusal reason (if applicable)
    /// - Suggested reformulation (if configured)
    /// - Query complexity score
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = classifier.classify("What is entity X?")?;
    /// println!("Intent: {:?}, Confidence: {:.2}",
    ///          result.primary_intent, result.confidence);
    /// ```
    pub fn classify(&self, query: &str) -> Result<IntentResult> {
        let query_lower = query.to_lowercase();

        // Check for inappropriate content first
        if self.config.enable_inappropriate_detection && self.is_inappropriate(&query_lower) {
            return Ok(IntentResult {
                primary_intent: QueryIntent::Inappropriate,
                secondary_intents: vec![],
                confidence: 1.0,
                should_refuse: true,
                refusal_reason: Some("Query contains inappropriate content".to_string()),
                suggested_reformulation: None,
                complexity_score: 0.0,
            });
        }

        // Calculate scores for each intent
        let mut intent_scores: HashMap<QueryIntent, f32> = HashMap::new();

        for (intent, patterns) in &self.intent_patterns {
            let score = self.calculate_intent_score(&query_lower, patterns);
            if score > 0.0 {
                intent_scores.insert(intent.clone(), score);
            }
        }

        // Determine primary and secondary intents
        let mut sorted_intents: Vec<(QueryIntent, f32)> = intent_scores.into_iter().collect();
        sorted_intents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted_intents.is_empty() {
            return Ok(IntentResult {
                primary_intent: QueryIntent::Ambiguous,
                secondary_intents: vec![],
                confidence: 0.0,
                should_refuse: true,
                refusal_reason: Some("Unable to understand the query intent".to_string()),
                suggested_reformulation: self.suggest_reformulation(query),
                complexity_score: self.calculate_complexity(&query_lower),
            });
        }

        let (primary_intent, primary_score) = sorted_intents[0].clone();
        let secondary_intents: Vec<(QueryIntent, f32)> = sorted_intents.into_iter().skip(1).take(2).collect();

        // Check for ambiguity
        let is_ambiguous = if self.config.enable_ambiguity_detection {
            secondary_intents.iter().any(|(_, score)| *score > primary_score * 0.8)
        } else {
            false
        };

        let final_intent = if is_ambiguous {
            QueryIntent::Ambiguous
        } else {
            primary_intent
        };

        let should_refuse = primary_score < self.config.refusal_threshold || is_ambiguous;

        let refusal_reason = if should_refuse {
            if is_ambiguous {
                Some("Query intent is ambiguous - please be more specific".to_string())
            } else {
                Some("Insufficient confidence in understanding the query".to_string())
            }
        } else {
            None
        };

        Ok(IntentResult {
            primary_intent: final_intent,
            secondary_intents,
            confidence: primary_score,
            should_refuse,
            refusal_reason,
            suggested_reformulation: if should_refuse && self.config.suggest_reformulations {
                self.suggest_reformulation(query)
            } else {
                None
            },
            complexity_score: self.calculate_complexity(&query_lower),
        })
    }

    /// Calculate score for a specific intent
    fn calculate_intent_score(&self, query: &str, patterns: &[IntentPattern]) -> f32 {
        let mut total_score = 0.0;

        for pattern in patterns {
            let mut pattern_score = 0.0;

            // Check keyword matches
            let keyword_matches = pattern.keywords.iter()
                .filter(|keyword| query.contains(&keyword.to_lowercase()))
                .count();

            if pattern.requires_all && keyword_matches != pattern.keywords.len() {
                continue; // Skip if all keywords are required but not all are present
            }

            if keyword_matches > 0 {
                pattern_score += (keyword_matches as f32 / pattern.keywords.len() as f32) * 0.5;
            }

            // Check regex pattern matches
            let regex_matches = pattern.patterns.iter()
                .filter(|regex| regex.is_match(query))
                .count();

            if regex_matches > 0 {
                pattern_score += (regex_matches as f32 / pattern.patterns.len() as f32) * 0.5;
            }

            total_score += pattern_score * pattern.weight;
        }

        total_score.min(1.0) // Cap at 1.0
    }

    /// Check if query contains inappropriate content
    fn is_inappropriate(&self, query: &str) -> bool {
        self.inappropriate_patterns.iter().any(|pattern| pattern.is_match(query))
    }

    /// Calculate query complexity
    fn calculate_complexity(&self, query: &str) -> f32 {
        let word_count = query.split_whitespace().count();
        let sentence_count = query.chars().filter(|&c| c == '.' || c == '?' || c == '!').count().max(1);
        let avg_word_length = query.chars().filter(|c| c.is_alphabetic()).count() as f32 / word_count.max(1) as f32;

        // Complexity factors
        let length_complexity = (word_count as f32 / 20.0).min(1.0); // Normalize to 20 words
        let sentence_complexity = (sentence_count as f32 / 3.0).min(1.0); // Normalize to 3 sentences
        let word_length_complexity = (avg_word_length / 8.0).min(1.0); // Normalize to 8 chars per word

        // Check for complex constructs
        let has_conjunctions = query.contains(" and ") || query.contains(" or ") || query.contains(" but ");
        let has_subordination = query.contains(" because ") || query.contains(" since ") || query.contains(" although ");
        let construct_complexity = if has_conjunctions || has_subordination { 0.3 } else { 0.0 };

        (length_complexity * 0.3 + sentence_complexity * 0.2 + word_length_complexity * 0.2 + construct_complexity).min(1.0)
    }

    /// Suggest query reformulation
    fn suggest_reformulation(&self, query: &str) -> Option<String> {
        if !self.config.suggest_reformulations {
            return None;
        }

        let query_lower = query.to_lowercase();

        // Suggest more specific reformulations based on common patterns
        if query_lower.starts_with("tell me about") {
            Some("Try asking a more specific question like 'What is...?' or 'How does...?'".to_string())
        } else if query_lower.contains(" and ") {
            Some("Try breaking your question into separate parts or focus on one aspect".to_string())
        } else if query.split_whitespace().count() > 20 {
            Some("Try using a shorter, more focused question".to_string())
        } else if !query.ends_with('?') && !query.ends_with('.') && !query.ends_with('!') {
            Some("Try phrasing your request as a clear question".to_string())
        } else {
            Some("Try being more specific about what information you're looking for".to_string())
        }
    }

    /// Get configuration
    pub fn get_config(&self) -> &IntentClassificationConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: IntentClassificationConfig) -> Result<()> {
        let old_inappropriate_detection = self.config.enable_inappropriate_detection;
        self.config = config;
        // Re-initialize patterns if needed
        if self.config.enable_inappropriate_detection != old_inappropriate_detection {
            self.initialize_patterns()?;
        }
        Ok(())
    }

    /// Get intent statistics
    pub fn get_statistics(&self) -> IntentClassificationStats {
        let total_patterns = self.intent_patterns.values()
            .map(|patterns| patterns.len())
            .sum();

        IntentClassificationStats {
            supported_intents: self.intent_patterns.keys().cloned().collect(),
            total_patterns,
            inappropriate_patterns: self.inappropriate_patterns.len(),
            confidence_threshold: self.config.confidence_threshold,
            refusal_threshold: self.config.refusal_threshold,
        }
    }
}

/// Statistics about intent classification capabilities and configuration.
///
/// Provides a snapshot of the classifier's capabilities, thresholds, and
/// pattern coverage for monitoring and debugging purposes.
#[cfg(feature = "rograg")]
#[derive(Debug)]
pub struct IntentClassificationStats {
    /// List of query intents supported by the classifier.
    pub supported_intents: Vec<QueryIntent>,

    /// Total number of patterns across all intents.
    pub total_patterns: usize,

    /// Number of patterns for detecting inappropriate content.
    pub inappropriate_patterns: usize,

    /// Minimum confidence threshold for classification (0.0 to 1.0).
    pub confidence_threshold: f32,

    /// Threshold for refusing to answer queries (0.0 to 1.0).
    pub refusal_threshold: f32,
}

#[cfg(feature = "rograg")]
impl IntentClassificationStats {
    /// Print statistics to the log using structured logging.
    ///
    /// Outputs intent count, pattern counts, and thresholds at INFO level.
    pub fn print(&self) {
        tracing::info!(
            supported_intents = self.supported_intents.len(),
            total_patterns = self.total_patterns,
            inappropriate_patterns = self.inappropriate_patterns,
            confidence_threshold = format!("{:.2}", self.confidence_threshold),
            refusal_threshold = format!("{:.2}", self.refusal_threshold),
            "Intent classification statistics"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rograg")]
    #[test]
    fn test_factual_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("What is Entity Name?").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Factual);
        assert!(result.confidence > 0.5);
        assert!(!result.should_refuse);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_definitional_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("Define the concept of friendship").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Definitional);
        assert!(result.confidence > 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_relational_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("How is Entity Name related to Second Entity?").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Relational);
        assert!(result.confidence > 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_temporal_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("When did Entity Name main activity?").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Temporal);
        assert!(result.confidence > 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_causal_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("Why did Entity Name trick his friends?").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Causal);
        assert!(result.confidence > 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_comparative_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("Compare Entity Name and Second Entity").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Comparative);
        assert!(result.confidence > 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_summary_intent() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("Tell me about Entity Name").unwrap();

        assert_eq!(result.primary_intent, QueryIntent::Summary);
        assert!(result.confidence > 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_ambiguous_query() {
        let classifier = IntentClassifier::new().unwrap();
        let result = classifier.classify("something unclear").unwrap();

        // Should either be ambiguous or have low confidence
        assert!(result.primary_intent == QueryIntent::Ambiguous || result.confidence < 0.5);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_complexity_calculation() {
        let classifier = IntentClassifier::new().unwrap();

        let simple_result = classifier.classify("What is Tom?").unwrap();
        let complex_result = classifier.classify("What is the intricate relationship between Entity Name and Second Entity, and how does it evolve throughout their various adventures and escapades?").unwrap();

        assert!(complex_result.complexity_score > simple_result.complexity_score);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_reformulation_suggestions() {
        let config = IntentClassificationConfig {
            suggest_reformulations: true,
            refusal_threshold: 0.9, // High threshold to trigger suggestions
            ..Default::default()
        };

        let classifier = IntentClassifier::with_config(config).unwrap();
        let result = classifier.classify("tell me about stuff").unwrap();

        assert!(result.suggested_reformulation.is_some());
    }
}