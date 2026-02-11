//! Adaptive Query Routing for Hierarchical GraphRAG
//!
//! Automatically selects the appropriate hierarchical level based on query complexity.
//!
//! # Strategy
//! - **Broad queries** (overview, themes, summary) → Higher levels (2-3)
//! - **Specific queries** (relationships, details, entities) → Lower levels (0-1)
//! - **Adaptive routing** based on query analysis

use serde::{Deserialize, Serialize};

/// Configuration for adaptive query routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRoutingConfig {
    /// Enable adaptive query routing
    pub enabled: bool,

    /// Default level when query complexity is unclear
    pub default_level: usize,

    /// Maximum hierarchical level available
    pub max_level: usize,

    /// Weight for keyword-based level selection (0.0-1.0)
    pub keyword_weight: f32,

    /// Weight for query length-based selection (0.0-1.0)
    pub length_weight: f32,

    /// Weight for entity mention-based selection (0.0-1.0)
    pub entity_weight: f32,
}

impl Default for AdaptiveRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_level: 1,
            max_level: 3,
            keyword_weight: 0.5,
            length_weight: 0.3,
            entity_weight: 0.2,
        }
    }
}

/// Query complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryComplexity {
    /// Very broad query (overview, themes)
    VeryBroad,
    /// Broad query (general understanding)
    Broad,
    /// Medium complexity
    Medium,
    /// Specific query (detailed information)
    Specific,
    /// Very specific query (precise relationships)
    VerySpecific,
}

impl QueryComplexity {
    /// Convert complexity to hierarchical level
    pub fn to_level(self, max_level: usize) -> usize {
        match self {
            QueryComplexity::VeryBroad => max_level.max(2),
            QueryComplexity::Broad => (max_level - 1).max(1),
            QueryComplexity::Medium => 1,
            QueryComplexity::Specific => 0,
            QueryComplexity::VerySpecific => 0,
        }
    }
}

/// Analyzes query complexity and suggests appropriate hierarchical level
#[derive(Debug)]
pub struct QueryComplexityAnalyzer {
    config: AdaptiveRoutingConfig,

    // Keyword sets for classification
    broad_keywords: Vec<&'static str>,
    specific_keywords: Vec<&'static str>,
}

impl QueryComplexityAnalyzer {
    /// Create a new query complexity analyzer
    pub fn new(config: AdaptiveRoutingConfig) -> Self {
        Self {
            config,
            broad_keywords: vec![
                "overview", "summary", "summarize", "main", "general", "all",
                "themes", "topics", "overall", "broadly", "big picture",
                "what are", "list all", "show me all",
            ],
            specific_keywords: vec![
                "relationship between", "how does", "why does", "specific",
                "detail", "exactly", "precisely", "what is the connection",
                "explain how", "describe the", "between", "and",
            ],
        }
    }

    /// Analyze query and determine complexity
    pub fn analyze(&self, query: &str) -> QueryComplexity {
        let query_lower = query.to_lowercase();

        // Score components
        let keyword_score = self.analyze_keywords(&query_lower);
        let length_score = self.analyze_length(query);
        let entity_score = self.analyze_entity_mentions(&query_lower);

        // Weighted combination
        let total_score =
            keyword_score * self.config.keyword_weight +
            length_score * self.config.length_weight +
            entity_score * self.config.entity_weight;

        // Map score to complexity level
        if total_score >= 0.7 {
            QueryComplexity::VeryBroad
        } else if total_score >= 0.4 {
            QueryComplexity::Broad
        } else if total_score >= -0.2 {
            QueryComplexity::Medium
        } else if total_score >= -0.5 {
            QueryComplexity::Specific
        } else {
            QueryComplexity::VerySpecific
        }
    }

    /// Analyze query keywords (-1.0 = very specific, +1.0 = very broad)
    fn analyze_keywords(&self, query_lower: &str) -> f32 {
        let mut score = 0.0;
        let mut matches = 0;

        // Check broad keywords (positive score)
        for keyword in &self.broad_keywords {
            if query_lower.contains(keyword) {
                score += 1.0;
                matches += 1;
            }
        }

        // Check specific keywords (negative score)
        for keyword in &self.specific_keywords {
            if query_lower.contains(keyword) {
                score -= 1.0;
                matches += 1;
            }
        }

        // Normalize
        if matches > 0 {
            score / matches as f32
        } else {
            0.0 // No keywords found = medium
        }
    }

    /// Analyze query length (short = specific, long = broad)
    fn analyze_length(&self, query: &str) -> f32 {
        let words: Vec<&str> = query.split_whitespace().collect();
        let word_count = words.len();

        // Short queries (1-3 words) tend to be broad ("AI overview")
        // Long queries (8+ words) tend to be specific
        match word_count {
            1..=3 => 0.5,      // Short → broad
            4..=5 => 0.2,      // Medium-short
            6..=7 => 0.0,      // Medium
            8..=10 => -0.3,    // Medium-long → specific
            _ => -0.5,         // Long → very specific
        }
    }

    /// Analyze entity mentions (many entities = specific query)
    fn analyze_entity_mentions(&self, query_lower: &str) -> f32 {
        // Count capital words (potential entity names in original query)
        // and quoted phrases (explicit entity references)
        let quoted_count = query_lower.matches('"').count() / 2;
        let and_between = query_lower.matches(" and ").count();
        let between_count = query_lower.matches("between").count();

        let entity_indicators = quoted_count + and_between + between_count;

        // More entity indicators = more specific
        match entity_indicators {
            0 => 0.3,      // No entities → broad
            1 => 0.0,      // One entity → medium
            2 => -0.4,     // Two entities → specific
            _ => -0.7,     // Multiple entities → very specific
        }
    }

    /// Suggest hierarchical level for query
    pub fn suggest_level(&self, query: &str) -> usize {
        let complexity = self.analyze(query);
        complexity.to_level(self.config.max_level)
    }

    /// Get detailed analysis with explanation
    pub fn analyze_detailed(&self, query: &str) -> QueryAnalysis {
        let query_lower = query.to_lowercase();

        let keyword_score = self.analyze_keywords(&query_lower);
        let length_score = self.analyze_length(query);
        let entity_score = self.analyze_entity_mentions(&query_lower);

        let complexity = self.analyze(query);
        let suggested_level = complexity.to_level(self.config.max_level);

        QueryAnalysis {
            query: query.to_string(),
            complexity,
            suggested_level,
            keyword_score,
            length_score,
            entity_score,
            explanation: self.generate_explanation(complexity, suggested_level),
        }
    }

    /// Generate explanation for the routing decision
    fn generate_explanation(&self, complexity: QueryComplexity, level: usize) -> String {
        match complexity {
            QueryComplexity::VeryBroad => format!(
                "Very broad query detected → using level {} for high-level overview",
                level
            ),
            QueryComplexity::Broad => format!(
                "Broad query detected → using level {} for general understanding",
                level
            ),
            QueryComplexity::Medium => format!(
                "Medium complexity query → using level {} for balanced detail",
                level
            ),
            QueryComplexity::Specific => format!(
                "Specific query detected → using level {} for detailed information",
                level
            ),
            QueryComplexity::VerySpecific => format!(
                "Very specific query detected → using level {} for precise relationships",
                level
            ),
        }
    }
}

/// Detailed query analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// Original query
    pub query: String,
    /// Detected complexity level
    pub complexity: QueryComplexity,
    /// Suggested hierarchical level
    pub suggested_level: usize,
    /// Keyword analysis score
    pub keyword_score: f32,
    /// Length analysis score
    pub length_score: f32,
    /// Entity mention score
    pub entity_score: f32,
    /// Human-readable explanation
    pub explanation: String,
}

impl QueryAnalysis {
    /// Print detailed analysis
    pub fn print(&self) {
        println!("Query Analysis:");
        println!("  Query: \"{}\"", self.query);
        println!("  Complexity: {:?}", self.complexity);
        println!("  Suggested Level: {}", self.suggested_level);
        println!("  Scores:");
        println!("    - Keywords: {:.2}", self.keyword_score);
        println!("    - Length: {:.2}", self.length_score);
        println!("    - Entities: {:.2}", self.entity_score);
        println!("  {}", self.explanation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broad_query() {
        let config = AdaptiveRoutingConfig::default();
        let analyzer = QueryComplexityAnalyzer::new(config);

        let query = "Give me an overview of AI technologies";
        let complexity = analyzer.analyze(query);
        let level = analyzer.suggest_level(query);

        // Should be broad → high level
        assert!(matches!(complexity, QueryComplexity::VeryBroad | QueryComplexity::Broad));
        assert!(level >= 1);
    }

    #[test]
    fn test_specific_query() {
        let config = AdaptiveRoutingConfig::default();
        let analyzer = QueryComplexityAnalyzer::new(config);

        let query = "What is the relationship between Transformers and GPT?";
        let complexity = analyzer.analyze(query);
        let level = analyzer.suggest_level(query);

        // Should be specific → low level
        assert!(matches!(complexity, QueryComplexity::Specific | QueryComplexity::VerySpecific));
        assert_eq!(level, 0);
    }

    #[test]
    fn test_medium_query() {
        let config = AdaptiveRoutingConfig::default();
        let analyzer = QueryComplexityAnalyzer::new(config);

        let query = "How does machine learning work?";
        let level = analyzer.suggest_level(query);

        // Should be medium → level 1
        assert!(level <= 1);
    }

    #[test]
    fn test_detailed_analysis() {
        let config = AdaptiveRoutingConfig::default();
        let analyzer = QueryComplexityAnalyzer::new(config);

        let query = "Summarize the main themes";
        let analysis = analyzer.analyze_detailed(query);

        assert!(analysis.keyword_score > 0.0); // Contains "summarize" and "main"
        assert!(!analysis.explanation.is_empty());
    }
}
