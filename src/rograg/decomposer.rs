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

/// Error types for query decomposition
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum DecompositionError {
    #[error("Query too complex to decompose: {message}")]
    TooComplex { message: String },

    #[error("Invalid query structure: {message}")]
    InvalidStructure { message: String },

    #[error("Decomposition strategy failed: {strategy}: {reason}")]
    StrategyFailed { strategy: String, reason: String },

    #[error("No valid subqueries generated")]
    NoValidSubqueries,
}

/// Strategy for query decomposition
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize, PartialEq)]
pub enum DecompositionStrategy {
    Semantic,
    Syntactic,
    Hybrid,
    Logical,
}

/// Result of query decomposition
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    pub original_query: String,
    pub subqueries: Vec<Subquery>,
    pub strategy_used: DecompositionStrategy,
    pub confidence: f32,
    pub dependencies: Vec<QueryDependency>,
}

/// A decomposed subquery
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subquery {
    pub id: String,
    pub text: String,
    pub query_type: SubqueryType,
    pub priority: f32,
    pub dependencies: Vec<String>, // IDs of other subqueries this depends on
}

/// Type of subquery
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum SubqueryType {
    Entity,       // "Who is X?"
    Relationship, // "How are X and Y related?"
    Attribute,    // "What is X's property?"
    Temporal,     // "When did X happen?"
    Causal,       // "Why did X happen?"
    Comparative,  // "Compare X and Y"
    Definitional, // "What is X?"
}

/// Dependency between subqueries
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDependency {
    pub dependent_id: String,
    pub prerequisite_id: String,
    pub dependency_type: DependencyType,
}

#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum DependencyType {
    Sequential, // Must be processed in order
    Reference,  // Uses result from another query
    Context,    // Provides context for another query
}

/// Trait for query decomposition strategies
#[cfg(feature = "rograg")]
#[async_trait]
pub trait QueryDecomposer: Send + Sync {
    /// Decompose a query into subqueries
    async fn decompose(&self, query: &str) -> Result<DecompositionResult>;

    /// Check if a query can be decomposed by this strategy
    fn can_decompose(&self, query: &str) -> bool;

    /// Get the strategy name
    fn strategy_name(&self) -> &str;
}

/// Semantic query decomposer using linguistic patterns
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

/// Syntactic query decomposer using grammatical structure
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

/// Hybrid decomposer that combines semantic and syntactic approaches
#[cfg(feature = "rograg")]
pub struct HybridQueryDecomposer {
    semantic: SemanticQueryDecomposer,
    syntactic: SyntacticQueryDecomposer,
}

#[cfg(feature = "rograg")]
impl HybridQueryDecomposer {
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
    /// Create a result with a single query (no decomposition)
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

    /// Check if decomposition was successful
    pub fn is_decomposed(&self) -> bool {
        self.subqueries.len() > 1
    }

    /// Get high-priority subqueries first
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
