//! Query decomposition for ROGRAG system
//!
//! Implements multiple strategies for breaking complex queries into simpler subqueries:
//! - Semantic decomposition using linguistic patterns
//! - Syntactic decomposition using grammatical structure
//! - Hybrid approach combining both methods

#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use async_trait::async_trait;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Errors that can occur during query decomposition.
///
/// These errors indicate problems with parsing or structuring complex queries
/// into simpler subqueries.
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum DecompositionError {
    /// The query is too complex to be decomposed by available strategies.
    ///
    /// Occurs when the query contains deeply nested structures, excessive
    /// conjunctions, or ambiguous grammatical constructs that prevent reliable
    /// decomposition.
    #[error("Query too complex to decompose: {message}")]
    TooComplex {
        /// Error message describing the complexity issue.
        message: String
    },

    /// The query structure is invalid or malformed.
    ///
    /// Occurs when the query lacks proper sentence structure, has unbalanced
    /// clauses, or contains patterns that cannot be parsed.
    #[error("Invalid query structure: {message}")]
    InvalidStructure {
        /// Error message describing the structural problem.
        message: String
    },

    /// A specific decomposition strategy encountered an error.
    ///
    /// Occurs when a strategy (semantic, syntactic, etc.) fails during execution
    /// due to pattern matching failures or internal errors.
    #[error("Decomposition strategy failed: {strategy}: {reason}")]
    StrategyFailed {
        /// Name of the strategy that failed.
        strategy: String,
        /// Reason for the failure.
        reason: String
    },

    /// No valid subqueries could be generated from the input.
    ///
    /// Occurs when all decomposition attempts result in empty or invalid
    /// subquery sets, typically indicating an unsupported query type.
    #[error("No valid subqueries generated")]
    NoValidSubqueries,
}

/// Strategy used for decomposing complex queries into subqueries.
///
/// Different strategies apply different heuristics for identifying query boundaries
/// and extracting meaningful subqueries.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, PartialEq)]
pub enum DecompositionStrategy {
    /// Decomposition based on semantic patterns and linguistic analysis.
    ///
    /// Uses regex patterns to identify question types and entity relationships.
    Semantic,

    /// Decomposition based on grammatical structure and clause boundaries.
    ///
    /// Splits on conjunctions, punctuation, and other syntactic separators.
    Syntactic,

    /// Combination of semantic and syntactic approaches.
    ///
    /// Tries semantic decomposition first, falls back to syntactic if needed.
    Hybrid,

    /// Decomposition based on logical operators and query structure.
    ///
    /// Reserved for future use with formal logic-based decomposition.
    Logical,
}

/// Result of decomposing a query into subqueries.
///
/// Contains the original query, extracted subqueries, confidence scores,
/// and dependency information for proper execution ordering.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// The original query text that was decomposed.
    pub original_query: String,

    /// List of subqueries extracted from the original query.
    ///
    /// May contain a single element if the query could not be meaningfully decomposed.
    pub subqueries: Vec<Subquery>,

    /// The strategy that was used to perform the decomposition.
    pub strategy_used: DecompositionStrategy,

    /// Confidence score for the decomposition quality (0.0 to 1.0).
    ///
    /// Higher values indicate more reliable decomposition. Values below 0.5
    /// suggest the decomposition may be unreliable.
    pub confidence: f32,

    /// Dependencies between subqueries that affect execution order.
    pub dependencies: Vec<QueryDependency>,
}

/// A single subquery extracted from a complex query.
///
/// Each subquery represents an atomic question that can be answered independently
/// or with reference to other subquery results.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subquery {
    /// Unique identifier for this subquery within the decomposition result.
    pub id: String,

    /// The text of the subquery.
    pub text: String,

    /// Classification of the subquery type for specialized handling.
    pub query_type: SubqueryType,

    /// Priority for execution (0.0 to 1.0).
    ///
    /// Higher priority subqueries should be executed first. Priority decreases
    /// for later clauses in compound queries.
    pub priority: f32,

    /// IDs of other subqueries that must be processed before this one.
    ///
    /// Empty if this subquery has no dependencies.
    pub dependencies: Vec<String>,
}

/// Classification of subquery types for specialized processing.
///
/// Each type corresponds to a different information need and may require
/// different retrieval strategies.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum SubqueryType {
    /// Entity identification query.
    ///
    /// Example: "Who is X?"
    Entity,

    /// Relationship query between entities.
    ///
    /// Example: "How are X and Y related?"
    Relationship,

    /// Attribute or property query.
    ///
    /// Example: "What is X's property?"
    Attribute,

    /// Temporal information query.
    ///
    /// Example: "When did X happen?"
    Temporal,

    /// Causal reasoning query.
    ///
    /// Example: "Why did X happen?"
    Causal,

    /// Comparative analysis query.
    ///
    /// Example: "Compare X and Y"
    Comparative,

    /// Definition or explanation query.
    ///
    /// Example: "What is X?"
    Definitional,
}

/// Dependency relationship between two subqueries.
///
/// Specifies how one subquery depends on another for proper execution.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDependency {
    /// ID of the subquery that has the dependency.
    pub dependent_id: String,

    /// ID of the subquery that must be processed first.
    pub prerequisite_id: String,

    /// Type of dependency relationship.
    pub dependency_type: DependencyType,
}

/// Type of dependency between subqueries.
///
/// Determines how results should be combined and in what order.
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum DependencyType {
    /// Sequential dependency - must be processed in strict order.
    ///
    /// The dependent query cannot start until the prerequisite completes.
    Sequential,

    /// Reference dependency - uses results from another query.
    ///
    /// The dependent query incorporates the prerequisite's results.
    Reference,

    /// Context dependency - provides background for another query.
    ///
    /// The prerequisite establishes context for the dependent query.
    Context,
}

/// Trait for implementing query decomposition strategies.
///
/// Implementors provide different approaches to breaking complex queries into
/// simpler subqueries based on semantic, syntactic, or hybrid analysis.
#[cfg(feature = "rograg")]
#[async_trait]
pub trait QueryDecomposer: Send + Sync {
    /// Decompose a query into subqueries.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to decompose
    ///
    /// # Returns
    ///
    /// Returns a `DecompositionResult` containing the subqueries, dependencies,
    /// and confidence score. If decomposition is not possible, returns a single
    /// subquery containing the original query.
    async fn decompose(&self, query: &str) -> Result<DecompositionResult>;

    /// Check if a query can be decomposed by this strategy.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to check
    ///
    /// # Returns
    ///
    /// Returns `true` if this strategy recognizes patterns in the query that
    /// it can decompose, `false` otherwise.
    fn can_decompose(&self, query: &str) -> bool;

    /// Get the strategy name.
    ///
    /// # Returns
    ///
    /// Returns a string identifier for this decomposition strategy.
    fn strategy_name(&self) -> &str;
}

/// Semantic query decomposer using linguistic patterns.
///
/// This decomposer uses regex patterns to identify question types, entity mentions,
/// and relationship patterns in queries. It excels at breaking down queries with
/// clear semantic structure like "Who is X and what is Y?".
///
/// # Pattern Matching
///
/// The semantic decomposer recognizes patterns such as:
/// - "Who/What is X and Y?" → separate entity queries
/// - "How are X and Y related?" → entity + relationship queries
/// - Conjunction-based splitting as fallback
///
/// # Confidence Scoring
///
/// - 0.8: Pattern-based decomposition successful
/// - 0.5: Conjunction-based fallback decomposition
/// - 1.0: Single query (no decomposition)
#[cfg(feature = "rograg")]
pub struct SemanticQueryDecomposer {
    patterns: Vec<SemanticPattern>,
}

#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
struct SemanticPattern {
    pattern: regex::Regex,
    extractor: fn(&str) -> Vec<String>,
    subquery_type: SubqueryType,
}

#[cfg(feature = "rograg")]
impl SemanticQueryDecomposer {
    /// Create a new semantic query decomposer with default patterns.
    ///
    /// # Returns
    ///
    /// Returns a `SemanticQueryDecomposer` initialized with predefined patterns
    /// for common query structures, or an error if pattern compilation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if regex pattern compilation fails during initialization.
    pub fn new() -> Result<Self> {
        let patterns = vec![
            SemanticPattern {
                pattern: regex::Regex::new(r"\b(who|what) is (.+?) and (.+)")?,
                extractor: |text| {
                    if let Some(caps) = regex::Regex::new(r"\b(who|what) is (.+?) and (.+)")
                        .unwrap()
                        .captures(text)
                    {
                        vec![
                            format!(
                                "{} is {}",
                                caps.get(1).unwrap().as_str(),
                                caps.get(2).unwrap().as_str()
                            ),
                            caps.get(3).unwrap().as_str().to_string(),
                        ]
                    } else {
                        vec![]
                    }
                },
                subquery_type: SubqueryType::Entity,
            },
            SemanticPattern {
                pattern: regex::Regex::new(
                    r"\bhow (?:is|are) (.+?) (?:related to|connected to) (.+)",
                )?,
                extractor: |text| {
                    if let Some(caps) = regex::Regex::new(
                        r"\bhow (?:is|are) (.+?) (?:related to|connected to) (.+)",
                    )
                    .unwrap()
                    .captures(text)
                    {
                        vec![
                            format!("What is {}", caps.get(1).unwrap().as_str()),
                            format!("What is {}", caps.get(2).unwrap().as_str()),
                            format!(
                                "How are {} and {} related",
                                caps.get(1).unwrap().as_str(),
                                caps.get(2).unwrap().as_str()
                            ),
                        ]
                    } else {
                        vec![]
                    }
                },
                subquery_type: SubqueryType::Relationship,
            },
        ];

        Ok(Self { patterns })
    }
}

#[cfg(feature = "rograg")]
#[async_trait]
impl QueryDecomposer for SemanticQueryDecomposer {
    async fn decompose(&self, query: &str) -> Result<DecompositionResult> {
        let mut all_subqueries = Vec::new();
        let mut strategy_confidence = 0.0;

        for pattern in &self.patterns {
            if pattern.pattern.is_match(query) {
                let subquery_texts = (pattern.extractor)(query);

                for (idx, text) in subquery_texts.into_iter().enumerate() {
                    if !text.trim().is_empty() {
                        all_subqueries.push(Subquery {
                            id: format!("sem_{idx}"),
                            text: text.trim().to_string(),
                            query_type: pattern.subquery_type.clone(),
                            priority: 1.0 - (idx as f32 * 0.1),
                            dependencies: if idx > 0 {
                                vec![format!("sem_{}", idx - 1)]
                            } else {
                                vec![]
                            },
                        });
                    }
                }

                strategy_confidence = 0.8;
                break;
            }
        }

        if all_subqueries.is_empty() {
            // Fallback: split on conjunctions
            let conjunctions = ["and", "or", "but", "also", "furthermore"];
            for conjunction in &conjunctions {
                if query.to_lowercase().contains(conjunction) {
                    let parts: Vec<&str> = query.split(conjunction).collect();
                    if parts.len() > 1 {
                        for (idx, part) in parts.iter().enumerate() {
                            let text = part.trim();
                            if !text.is_empty() {
                                all_subqueries.push(Subquery {
                                    id: format!("sem_fallback_{idx}"),
                                    text: text.to_string(),
                                    query_type: SubqueryType::Entity, // Default
                                    priority: 1.0 - (idx as f32 * 0.2),
                                    dependencies: vec![],
                                });
                            }
                        }
                        strategy_confidence = 0.5;
                        break;
                    }
                }
            }
        }

        if all_subqueries.is_empty() {
            return Ok(DecompositionResult::single_query(query.to_string()));
        }

        Ok(DecompositionResult {
            original_query: query.to_string(),
            subqueries: all_subqueries,
            strategy_used: DecompositionStrategy::Semantic,
            confidence: strategy_confidence,
            dependencies: vec![], // TODO: Implement dependency analysis
        })
    }

    fn can_decompose(&self, query: &str) -> bool {
        self.patterns.iter().any(|p| p.pattern.is_match(query))
    }

    fn strategy_name(&self) -> &str {
        "semantic"
    }
}

/// Syntactic query decomposer using grammatical structure.
///
/// This decomposer splits queries based on clause boundaries identified through
/// punctuation and conjunctions. It works well for queries with clear syntactic
/// structure but may split inappropriately for complex nested clauses.
///
/// # Clause Separators
///
/// Recognizes the following as clause boundaries:
/// - Conjunctions: "and", "or", "but", "also", "furthermore", "moreover"
/// - Punctuation: commas, semicolons
/// - Discourse markers: "however", "therefore"
///
/// # Confidence Scoring
///
/// - 0.7: Successfully split into multiple clauses
/// - 0.3: Single clause or minimal splitting
#[cfg(feature = "rograg")]
pub struct SyntacticQueryDecomposer {
    clause_separators: Vec<String>,
}

#[cfg(feature = "rograg")]
impl Default for SyntacticQueryDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntacticQueryDecomposer {
    /// Create a new syntactic query decomposer with default clause separators.
    ///
    /// # Returns
    ///
    /// Returns a `SyntacticQueryDecomposer` configured with common English
    /// clause separators and conjunctions.
    pub fn new() -> Self {
        Self {
            clause_separators: vec![
                "and".to_string(),
                "or".to_string(),
                "but".to_string(),
                ",".to_string(),
                ";".to_string(),
                "also".to_string(),
                "furthermore".to_string(),
                "moreover".to_string(),
                "however".to_string(),
                "therefore".to_string(),
            ],
        }
    }

    fn identify_clause_boundaries(&self, query: &str) -> Vec<usize> {
        let mut boundaries = vec![0];

        for separator in &self.clause_separators {
            let separator_lower = separator.to_lowercase();
            let query_lower = query.to_lowercase();

            let mut start = 0;
            while let Some(pos) = query_lower[start..].find(&separator_lower) {
                let absolute_pos = start + pos;
                if !boundaries.contains(&absolute_pos) {
                    boundaries.push(absolute_pos);
                }
                start = absolute_pos + separator.len();
            }
        }

        boundaries.push(query.len());
        boundaries.sort();
        boundaries.dedup();
        boundaries
    }

    fn extract_clauses(&self, query: &str) -> Vec<String> {
        let boundaries = self.identify_clause_boundaries(query);
        let mut clauses = Vec::new();

        for window in boundaries.windows(2) {
            if let [start, end] = window {
                let clause = query[*start..*end].trim();

                // Remove leading separators
                let clause = self
                    .clause_separators
                    .iter()
                    .fold(clause.to_string(), |acc, sep| {
                        if acc.to_lowercase().starts_with(&sep.to_lowercase()) {
                            acc[sep.len()..].trim().to_string()
                        } else {
                            acc
                        }
                    });

                if !clause.is_empty() && clause.len() > 3 {
                    clauses.push(clause);
                }
            }
        }

        clauses
    }

    fn classify_clause_type(&self, clause: &str) -> SubqueryType {
        let clause_lower = clause.to_lowercase();

        if clause_lower.starts_with("who") || clause_lower.starts_with("what person") {
            SubqueryType::Entity
        } else if clause_lower.starts_with("what") {
            SubqueryType::Definitional
        } else if clause_lower.starts_with("when") {
            SubqueryType::Temporal
        } else if clause_lower.starts_with("why") || clause_lower.contains("because") {
            SubqueryType::Causal
        } else if clause_lower.contains("relation") || clause_lower.contains("connect") {
            SubqueryType::Relationship
        } else if clause_lower.contains("compare") || clause_lower.contains("versus") {
            SubqueryType::Comparative
        } else {
            SubqueryType::Attribute
        }
    }
}

#[cfg(feature = "rograg")]
#[async_trait]
impl QueryDecomposer for SyntacticQueryDecomposer {
    async fn decompose(&self, query: &str) -> Result<DecompositionResult> {
        let clauses = self.extract_clauses(query);

        if clauses.len() <= 1 {
            return Ok(DecompositionResult::single_query(query.to_string()));
        }

        let subqueries: Vec<Subquery> = clauses
            .into_iter()
            .enumerate()
            .map(|(idx, clause)| Subquery {
                id: format!("syn_{idx}"),
                text: clause.clone(),
                query_type: self.classify_clause_type(&clause),
                priority: 1.0 - (idx as f32 * 0.1),
                dependencies: vec![],
            })
            .collect();

        let confidence = if subqueries.len() > 1 { 0.7 } else { 0.3 };

        Ok(DecompositionResult {
            original_query: query.to_string(),
            subqueries,
            strategy_used: DecompositionStrategy::Syntactic,
            confidence,
            dependencies: vec![],
        })
    }

    fn can_decompose(&self, query: &str) -> bool {
        self.clause_separators
            .iter()
            .any(|sep| query.to_lowercase().contains(&sep.to_lowercase()))
    }

    fn strategy_name(&self) -> &str {
        "syntactic"
    }
}

/// Hybrid decomposer that combines semantic and syntactic approaches.
///
/// This decomposer attempts semantic decomposition first (which is more accurate
/// for well-formed queries) and falls back to syntactic decomposition if semantic
/// patterns don't match or produce low confidence results.
///
/// # Strategy Selection
///
/// 1. Try semantic decomposition if patterns match
/// 2. If semantic confidence > 0.6, use semantic result
/// 3. Otherwise, try syntactic decomposition
/// 4. If neither works, return single query
///
/// # Recommended Use
///
/// This is the recommended decomposer for general use as it balances accuracy
/// and coverage across different query styles.
#[cfg(feature = "rograg")]
pub struct HybridQueryDecomposer {
    semantic: SemanticQueryDecomposer,
    syntactic: SyntacticQueryDecomposer,
}

#[cfg(feature = "rograg")]
impl HybridQueryDecomposer {
    /// Create a new hybrid query decomposer.
    ///
    /// # Returns
    ///
    /// Returns a `HybridQueryDecomposer` with initialized semantic and syntactic
    /// decomposers, or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if semantic decomposer initialization fails (typically
    /// due to regex compilation errors).
    pub fn new() -> Result<Self> {
        Ok(Self {
            semantic: SemanticQueryDecomposer::new()?,
            syntactic: SyntacticQueryDecomposer::new(),
        })
    }
}

#[cfg(feature = "rograg")]
#[async_trait]
impl QueryDecomposer for HybridQueryDecomposer {
    async fn decompose(&self, query: &str) -> Result<DecompositionResult> {
        // Try semantic decomposition first
        if self.semantic.can_decompose(query) {
            let semantic_result = self.semantic.decompose(query).await?;
            if semantic_result.confidence > 0.6 {
                return Ok(DecompositionResult {
                    strategy_used: DecompositionStrategy::Hybrid,
                    ..semantic_result
                });
            }
        }

        // Fall back to syntactic decomposition
        if self.syntactic.can_decompose(query) {
            let syntactic_result = self.syntactic.decompose(query).await?;
            return Ok(DecompositionResult {
                strategy_used: DecompositionStrategy::Hybrid,
                ..syntactic_result
            });
        }

        // If neither works, return single query
        Ok(DecompositionResult::single_query(query.to_string()))
    }

    fn can_decompose(&self, query: &str) -> bool {
        self.semantic.can_decompose(query) || self.syntactic.can_decompose(query)
    }

    fn strategy_name(&self) -> &str {
        "hybrid"
    }
}

#[cfg(feature = "rograg")]
impl DecompositionResult {
    /// Create a result with a single query (no decomposition).
    ///
    /// Used as a fallback when decomposition is not possible or not beneficial.
    /// The result will have confidence 1.0 since the original query is preserved.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to wrap as a single subquery
    ///
    /// # Returns
    ///
    /// Returns a `DecompositionResult` with a single subquery containing the
    /// original query text.
    pub fn single_query(query: String) -> Self {
        Self {
            original_query: query.clone(),
            subqueries: vec![Subquery {
                id: "single".to_string(),
                text: query,
                query_type: SubqueryType::Entity,
                priority: 1.0,
                dependencies: vec![],
            }],
            strategy_used: DecompositionStrategy::Semantic,
            confidence: 1.0,
            dependencies: vec![],
        }
    }

    /// Check if decomposition was successful.
    ///
    /// # Returns
    ///
    /// Returns `true` if the query was decomposed into multiple subqueries,
    /// `false` if it remains as a single query.
    pub fn is_decomposed(&self) -> bool {
        self.subqueries.len() > 1
    }

    /// Get subqueries ordered by priority (highest first).
    ///
    /// # Returns
    ///
    /// Returns a vector of subquery references sorted in descending order
    /// by priority. Higher priority subqueries should be processed first.
    pub fn ordered_subqueries(&self) -> Vec<&Subquery> {
        let mut subqueries: Vec<&Subquery> = self.subqueries.iter().collect();
        subqueries.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        subqueries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_semantic_decomposition() {
        let decomposer = SemanticQueryDecomposer::new().unwrap();

        let result = decomposer
            .decompose("Who is Entity Name and what is his relationship with Second Entity?")
            .await
            .unwrap();

        assert!(result.is_decomposed());
        assert!(result.subqueries.len() >= 2);
        assert_eq!(result.strategy_used, DecompositionStrategy::Semantic);
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_syntactic_decomposition() {
        let decomposer = SyntacticQueryDecomposer::new();

        let result = decomposer
            .decompose("Tell me about Entity Name, and also describe Second Entity")
            .await
            .unwrap();

        assert!(result.is_decomposed());
        assert_eq!(result.strategy_used, DecompositionStrategy::Syntactic);
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_hybrid_decomposition() {
        let decomposer = HybridQueryDecomposer::new().unwrap();

        let result = decomposer
            .decompose("What is friendship and how are Tom and Huck related?")
            .await
            .unwrap();

        assert_eq!(result.strategy_used, DecompositionStrategy::Hybrid);
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_single_query_fallback() {
        let decomposer = HybridQueryDecomposer::new().unwrap();

        let result = decomposer.decompose("Simple query").await.unwrap();

        assert!(!result.is_decomposed());
        assert_eq!(result.subqueries.len(), 1);
    }
}
