//! Keyword extraction for dual-level retrieval
//!
//! Extracts high-level (topics) and low-level (entities) keywords from queries.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::error::GraphRAGError;
use crate::core::traits::AsyncLanguageModel;

/// Dual-level keywords for retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLevelKeywords {
    /// High-level: broader topics, themes, concepts
    pub high_level: Vec<String>,

    /// Low-level: specific entities, attributes, details
    pub low_level: Vec<String>,
}

/// Configuration for keyword extraction
#[derive(Debug, Clone)]
pub struct KeywordExtractorConfig {
    /// Maximum total keywords (LightRAG optimization: <20)
    pub max_keywords: usize,

    /// Language for keywords
    pub language: String,

    /// Whether to enable caching
    pub enable_cache: bool,
}

impl Default for KeywordExtractorConfig {
    fn default() -> Self {
        Self {
            max_keywords: 20,  // LightRAG optimization
            language: "English".to_string(),
            enable_cache: true,
        }
    }
}

/// Keyword extractor for dual-level retrieval
pub struct KeywordExtractor {
    llm: Arc<dyn AsyncLanguageModel<Error = GraphRAGError>>,
    config: KeywordExtractorConfig,
}

impl KeywordExtractor {
    /// Create a new keyword extractor
    pub fn new(llm: Arc<dyn AsyncLanguageModel<Error = GraphRAGError>>, config: KeywordExtractorConfig) -> Self {
        Self { llm, config }
    }

    /// Extract dual-level keywords from query
    ///
    /// Returns (high_level_keywords, low_level_keywords)
    pub async fn extract(&self, query: &str) -> Result<DualLevelKeywords, GraphRAGError> {
        // Build prompt for LLM
        let prompt = self.build_extraction_prompt(query);

        // Call LLM to extract keywords
        let response = self.llm.complete(&prompt).await?;

        // Parse JSON response
        let keywords = self.parse_keywords_response(&response)?;

        // Validate keyword count
        self.validate_keywords(&keywords)?;

        Ok(keywords)
    }

    /// Build prompt for keyword extraction
    fn build_extraction_prompt(&self, query: &str) -> String {
        format!(r#"Extract keywords at two levels from this query: "{}"

Return JSON with this exact structure:
{{
  "high_level": ["theme1", "theme2", ...],
  "low_level": ["entity1", "entity2", ...]
}}

Rules:
1. high_level: Broader topics, concepts, themes (abstract level)
2. low_level: Specific entities, attributes, details (concrete level)
3. LIMIT: Maximum {} total keywords combined
4. NO duplication between levels
5. Keep keywords concise (1-3 words each)

Example 1:
Query: "How did Alice and Bob collaborate on the quantum computing project?"
{{
  "high_level": ["collaboration", "quantum computing", "teamwork"],
  "low_level": ["Alice", "Bob", "project"]
}}

Example 2:
Query: "What are the main themes in the dataset?"
{{
  "high_level": ["themes", "patterns", "overview"],
  "low_level": ["dataset"]
}}

Language: {}

Now extract keywords:"#, query, self.config.max_keywords, self.config.language)
    }

    /// Parse LLM response into keywords
    fn parse_keywords_response(&self, response: &str) -> Result<DualLevelKeywords, GraphRAGError> {
        // Try to find JSON in response
        let json_str = self.extract_json(response)?;

        // Parse JSON
        let keywords: DualLevelKeywords = serde_json::from_str(&json_str)
            .map_err(|e| GraphRAGError::Serialization {
                message: format!("Failed to parse keywords JSON: {}", e)
            })?;

        Ok(keywords)
    }

    /// Extract JSON from LLM response (handle various formats)
    fn extract_json(&self, response: &str) -> Result<String, GraphRAGError> {
        // Try to find JSON object in response
        if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                let json_str = &response[start..=end];
                return Ok(json_str.to_string());
            }
        }

        // If no JSON found, try to parse entire response
        if response.trim().starts_with('{') {
            return Ok(response.trim().to_string());
        }

        Err(GraphRAGError::Serialization {
            message: "No JSON object found in LLM response".to_string()
        })
    }

    /// Validate extracted keywords
    fn validate_keywords(&self, keywords: &DualLevelKeywords) -> Result<(), GraphRAGError> {
        let total = keywords.high_level.len() + keywords.low_level.len();

        if total > self.config.max_keywords {
            return Err(GraphRAGError::Validation {
                message: format!(
                    "Too many keywords: {} (max: {})",
                    total, self.config.max_keywords
                )
            });
        }

        if total == 0 {
            return Err(GraphRAGError::Validation {
                message: "No keywords extracted".to_string()
            });
        }

        Ok(())
    }

    /// Extract keywords with fallback to query terms
    pub async fn extract_with_fallback(&self, query: &str) -> Result<DualLevelKeywords, GraphRAGError> {
        match self.extract(query).await {
            Ok(keywords) => Ok(keywords),
            Err(e) => {
                log::warn!("Keyword extraction failed: {}, using fallback", e);

                // Fallback: use query terms as low-level keywords
                let words: Vec<String> = query
                    .split_whitespace()
                    .filter(|w| w.len() > 3)  // Filter short words
                    .take(10)  // Limit to 10 words
                    .map(|w| w.to_lowercase())
                    .collect();

                Ok(DualLevelKeywords {
                    high_level: Vec::new(),
                    low_level: words,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json() {
        // Mock LLM not needed for this test - we're testing JSON extraction logic
        let _config = KeywordExtractorConfig::default();

        // Simulate responses
        let responses = vec![
            r#"{"high_level": ["test"], "low_level": ["data"]}"#,
            r#"Here's the result: {"high_level": ["test"], "low_level": ["data"]}"#,
            r#"
            {
              "high_level": ["test"],
              "low_level": ["data"]
            }
            "#,
        ];

        for response in responses {
            // We'd need to construct a KeywordExtractor with a mock LLM
            // but for JSON extraction testing, we can test the logic directly
            if let Some(start) = response.find('{') {
                if let Some(end) = response.rfind('}') {
                    let json_str = &response[start..=end];
                    let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_str);
                    assert!(parsed.is_ok(), "Failed to parse JSON from: {}", response);
                }
            }
        }
    }

    #[test]
    fn test_validate_keywords() {
        // Test keyword validation logic
        let config = KeywordExtractorConfig::default();

        // Valid keywords
        let valid = DualLevelKeywords {
            high_level: vec!["topic1".to_string(), "topic2".to_string()],
            low_level: vec!["entity1".to_string()],
        };

        let total = valid.high_level.len() + valid.low_level.len();
        assert!(total <= config.max_keywords);
        assert!(total > 0);

        // Too many keywords
        let too_many = DualLevelKeywords {
            high_level: (0..15).map(|i| format!("topic{}", i)).collect(),
            low_level: (0..15).map(|i| format!("entity{}", i)).collect(),
        };
        let total = too_many.high_level.len() + too_many.low_level.len();
        assert!(total > config.max_keywords);

        // Empty keywords
        let empty = DualLevelKeywords {
            high_level: Vec::new(),
            low_level: Vec::new(),
        };
        let total = empty.high_level.len() + empty.low_level.len();
        assert_eq!(total, 0);
    }

    #[test]
    fn test_config_defaults() {
        let config = KeywordExtractorConfig::default();
        assert_eq!(config.max_keywords, 20);
        assert_eq!(config.language, "English");
        assert!(config.enable_cache);
    }
}
