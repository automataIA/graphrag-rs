//! Evaluation framework for GraphRAG system
//!
//! This module provides two complementary evaluation approaches:
//!
//! ## 1. LLM-based Query Result Evaluation
//! Evaluate GraphRAG query results using LLM-based metrics:
//! - Relevance: How relevant is the answer to the query?
//! - Faithfulness: Is the answer grounded in the retrieved context?
//! - Completeness: Does the answer address all aspects of the query?
//! - Coherence: Is the answer well-structured and easy to understand?
//! - Groundedness: Are entity names and relationships correctly mentioned?
//!
//! ## 2. Pipeline Phase Validation
//! Validate each phase of the GraphRAG pipeline:
//! - Document Processing: Chunking and enrichment validation
//! - Entity Extraction: Entity quality and coverage checks
//! - Relationship Extraction: Relationship validity and connectivity
//! - Graph Construction: Overall graph structure validation

pub mod pipeline_validation;

pub use pipeline_validation::{
    PhaseValidation, ValidationCheck, PipelineValidationReport,
    DocumentProcessingValidator, EntityExtractionValidator,
    RelationshipExtractionValidator, GraphConstructionValidator,
};

use crate::{Entity, Relationship, Result, GraphRAGError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A query result from GraphRAG that can be evaluated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluableQueryResult {
    /// The original user query
    pub query: String,
    /// The generated answer/response
    pub answer: String,
    /// Retrieved entities used in the answer
    pub retrieved_entities: Vec<Entity>,
    /// Retrieved relationships used in the answer
    pub retrieved_relationships: Vec<Relationship>,
    /// Relevant text chunks/context
    pub context_chunks: Vec<String>,
    /// Metadata about the retrieval process
    pub metadata: ResultMetadata,
}

/// Metadata about how the result was generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    /// Number of entities retrieved
    pub entities_count: usize,
    /// Number of relationships retrieved
    pub relationships_count: usize,
    /// Number of context chunks used
    pub chunks_count: usize,
    /// Retrieval strategy used (semantic, keyword, hybrid)
    pub retrieval_strategy: String,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Additional custom fields
    pub custom: HashMap<String, String>,
}

/// LLM evaluation prompt template for query results
#[derive(Debug, Clone)]
pub struct LLMEvaluationPrompt {
    /// Template for the evaluation prompt
    pub template: String,
}

impl Default for LLMEvaluationPrompt {
    fn default() -> Self {
        Self {
            template: Self::default_template(),
        }
    }
}

impl LLMEvaluationPrompt {
    /// Default evaluation prompt template
    fn default_template() -> String {
        r#"You are an expert evaluator for question-answering systems. Evaluate the following GraphRAG query result.

## Query
{query}

## Generated Answer
{answer}

## Retrieved Context
### Entities ({entities_count} total)
{entities}

### Relationships ({relationships_count} total)
{relationships}

### Text Chunks ({chunks_count} total)
{chunks}

## Evaluation Criteria
Please evaluate the answer on the following dimensions (score 1-5, where 5 is best):

1. **Relevance**: How well does the answer address the query?
   - 5: Perfectly addresses the query
   - 3: Partially addresses the query
   - 1: Not relevant to the query

2. **Faithfulness**: Is the answer grounded in the provided context?
   - 5: Fully supported by context, no hallucination
   - 3: Mostly supported, minor extrapolation
   - 1: Contains unsupported claims

3. **Completeness**: Does the answer cover all aspects of the query?
   - 5: Comprehensive, addresses all aspects
   - 3: Covers main points, misses some details
   - 1: Incomplete, misses key information

4. **Coherence**: Is the answer well-structured and clear?
   - 5: Excellent structure, very clear
   - 3: Adequate structure, somewhat clear
   - 1: Poor structure, confusing

5. **Groundedness**: Are entity names and relationships correctly mentioned?
   - 5: All entities/relationships accurate
   - 3: Minor inaccuracies
   - 1: Significant errors in entity/relationship mentions

## Output Format
Provide your evaluation in the following JSON format:

```json
{{
  "relevance": {{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
  }},
  "faithfulness": {{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
  }},
  "completeness": {{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
  }},
  "coherence": {{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
  }},
  "groundedness": {{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
  }},
  "overall_score": <average of all scores>,
  "summary": "<overall assessment in 2-3 sentences>"
}}
```

Evaluate now:"#.to_string()
    }

    /// Generate evaluation prompt for a query result
    pub fn generate(&self, result: &EvaluableQueryResult) -> String {
        let entities_str = self.format_entities(&result.retrieved_entities);
        let relationships_str = self.format_relationships(&result.retrieved_relationships);
        let chunks_str = self.format_chunks(&result.context_chunks);

        self.template
            .replace("{query}", &result.query)
            .replace("{answer}", &result.answer)
            .replace("{entities_count}", &result.metadata.entities_count.to_string())
            .replace("{relationships_count}", &result.metadata.relationships_count.to_string())
            .replace("{chunks_count}", &result.metadata.chunks_count.to_string())
            .replace("{entities}", &entities_str)
            .replace("{relationships}", &relationships_str)
            .replace("{chunks}", &chunks_str)
    }

    fn format_entities(&self, entities: &[Entity]) -> String {
        if entities.is_empty() {
            return "No entities retrieved.".to_string();
        }

        entities
            .iter()
            .take(10) // Limit to top 10 for prompt length
            .map(|e| format!("- {} (type: {}, confidence: {:.2})", e.name, e.entity_type, e.confidence))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_relationships(&self, relationships: &[Relationship]) -> String {
        if relationships.is_empty() {
            return "No relationships retrieved.".to_string();
        }

        relationships
            .iter()
            .take(10) // Limit to top 10 for prompt length
            .map(|r| format!("- {} --[{}]--> {} (confidence: {:.2})",
                r.source, r.relation_type, r.target, r.confidence))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_chunks(&self, chunks: &[String]) -> String {
        if chunks.is_empty() {
            return "No context chunks retrieved.".to_string();
        }

        chunks
            .iter()
            .take(5) // Limit to top 5 chunks
            .enumerate()
            .map(|(i, chunk)| {
                let preview = if chunk.len() > 200 {
                    format!("{}...", &chunk[..200])
                } else {
                    chunk.clone()
                };
                format!("Chunk {}:\n{}\n", i + 1, preview)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Parsed LLM evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMEvaluation {
    /// Relevance score and reasoning
    pub relevance: DimensionScore,
    /// Faithfulness score and reasoning
    pub faithfulness: DimensionScore,
    /// Completeness score and reasoning
    pub completeness: DimensionScore,
    /// Coherence score and reasoning
    pub coherence: DimensionScore,
    /// Groundedness score and reasoning
    pub groundedness: DimensionScore,
    /// Overall average score
    pub overall_score: f32,
    /// Summary assessment
    pub summary: String,
}

/// Score for a single evaluation dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    /// Score 1-5
    pub score: u8,
    /// Reasoning for the score
    pub reasoning: String,
}

impl LLMEvaluation {
    /// Parse LLM response JSON into structured evaluation
    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str).map_err(|e| GraphRAGError::Serialization {
            message: format!("Failed to parse LLM evaluation JSON: {}", e),
        })
    }

    /// Check if the evaluation passes a minimum quality threshold
    pub fn passes_threshold(&self, min_score: f32) -> bool {
        self.overall_score >= min_score
    }

    /// Get the dimension with the lowest score
    pub fn weakest_dimension(&self) -> (&str, &DimensionScore) {
        let dimensions = [
            ("relevance", &self.relevance),
            ("faithfulness", &self.faithfulness),
            ("completeness", &self.completeness),
            ("coherence", &self.coherence),
            ("groundedness", &self.groundedness),
        ];

        dimensions
            .iter()
            .min_by_key(|(_, score)| score.score)
            .map(|(name, score)| (*name, *score))
            .unwrap_or(("unknown", &self.relevance))
    }

    /// Generate a report string
    pub fn report(&self) -> String {
        format!(
            r#"## LLM Evaluation Report

**Overall Score**: {:.2}/5.0

### Dimension Scores
- Relevance:     {}/5 - {}
- Faithfulness:  {}/5 - {}
- Completeness:  {}/5 - {}
- Coherence:     {}/5 - {}
- Groundedness:  {}/5 - {}

### Summary
{}

### Weakest Dimension
{}: {} (score {}/5)
"#,
            self.overall_score,
            self.relevance.score, self.relevance.reasoning,
            self.faithfulness.score, self.faithfulness.reasoning,
            self.completeness.score, self.completeness.reasoning,
            self.coherence.score, self.coherence.reasoning,
            self.groundedness.score, self.groundedness.reasoning,
            self.summary,
            self.weakest_dimension().0,
            self.weakest_dimension().1.reasoning,
            self.weakest_dimension().1.score
        )
    }
}

/// Builder for creating evaluable query results
pub struct EvaluableQueryResultBuilder {
    query: Option<String>,
    answer: Option<String>,
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    chunks: Vec<String>,
    retrieval_strategy: String,
    processing_time_ms: u64,
    custom: HashMap<String, String>,
}

impl EvaluableQueryResultBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            query: None,
            answer: None,
            entities: Vec::new(),
            relationships: Vec::new(),
            chunks: Vec::new(),
            retrieval_strategy: "unknown".to_string(),
            processing_time_ms: 0,
            custom: HashMap::new(),
        }
    }

    /// Set the query
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Set the answer
    pub fn answer(mut self, answer: impl Into<String>) -> Self {
        self.answer = Some(answer.into());
        self
    }

    /// Add retrieved entities
    pub fn entities(mut self, entities: Vec<Entity>) -> Self {
        self.entities = entities;
        self
    }

    /// Add retrieved relationships
    pub fn relationships(mut self, relationships: Vec<Relationship>) -> Self {
        self.relationships = relationships;
        self
    }

    /// Add context chunks
    pub fn chunks(mut self, chunks: Vec<String>) -> Self {
        self.chunks = chunks;
        self
    }

    /// Set retrieval strategy
    pub fn retrieval_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.retrieval_strategy = strategy.into();
        self
    }

    /// Set processing time
    pub fn processing_time_ms(mut self, time_ms: u64) -> Self {
        self.processing_time_ms = time_ms;
        self
    }

    /// Add custom metadata
    pub fn custom_metadata(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Build the evaluable query result
    pub fn build(self) -> Result<EvaluableQueryResult> {
        let query = self.query.ok_or_else(|| GraphRAGError::Config {
            message: "Query is required".to_string(),
        })?;
        let answer = self.answer.ok_or_else(|| GraphRAGError::Config {
            message: "Answer is required".to_string(),
        })?;

        Ok(EvaluableQueryResult {
            query,
            answer,
            metadata: ResultMetadata {
                entities_count: self.entities.len(),
                relationships_count: self.relationships.len(),
                chunks_count: self.chunks.len(),
                retrieval_strategy: self.retrieval_strategy,
                processing_time_ms: self.processing_time_ms,
                custom: self.custom,
            },
            retrieved_entities: self.entities,
            retrieved_relationships: self.relationships,
            context_chunks: self.chunks,
        })
    }
}

impl Default for EvaluableQueryResultBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityId, ChunkId};

    #[test]
    fn test_prompt_generation() {
        let entity = Entity {
            id: EntityId::new("e1".to_string()),
            name: "Alice".to_string(),
            entity_type: "person".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        };

        let result = EvaluableQueryResultBuilder::new()
            .query("Who is Alice?")
            .answer("Alice is a person mentioned in the context.")
            .entities(vec![entity])
            .chunks(vec!["Alice works at Stanford.".to_string()])
            .retrieval_strategy("semantic")
            .build()
            .unwrap();

        let prompt = LLMEvaluationPrompt::default();
        let generated = prompt.generate(&result);

        assert!(generated.contains("Who is Alice?"));
        assert!(generated.contains("Alice is a person"));
        assert!(generated.contains("Alice (type: person"));
        assert!(generated.contains("Evaluate now:"));
    }

    #[test]
    fn test_evaluation_parsing() {
        let json = r#"{
            "relevance": {
                "score": 5,
                "reasoning": "Perfectly answers the question"
            },
            "faithfulness": {
                "score": 4,
                "reasoning": "Mostly grounded in context"
            },
            "completeness": {
                "score": 4,
                "reasoning": "Covers main points"
            },
            "coherence": {
                "score": 5,
                "reasoning": "Well structured"
            },
            "groundedness": {
                "score": 5,
                "reasoning": "All entities accurate"
            },
            "overall_score": 4.6,
            "summary": "High quality answer"
        }"#;

        let eval = LLMEvaluation::from_json(json).unwrap();
        assert_eq!(eval.relevance.score, 5);
        assert_eq!(eval.faithfulness.score, 4);
        assert!(eval.passes_threshold(4.0));
        assert!(!eval.passes_threshold(5.0));
    }

    #[test]
    fn test_weakest_dimension() {
        let json = r#"{
            "relevance": {"score": 5, "reasoning": "Perfect"},
            "faithfulness": {"score": 3, "reasoning": "Some issues"},
            "completeness": {"score": 4, "reasoning": "Good"},
            "coherence": {"score": 5, "reasoning": "Excellent"},
            "groundedness": {"score": 4, "reasoning": "Accurate"},
            "overall_score": 4.2,
            "summary": "Good overall"
        }"#;

        let eval = LLMEvaluation::from_json(json).unwrap();
        let (name, score) = eval.weakest_dimension();
        assert_eq!(name, "faithfulness");
        assert_eq!(score.score, 3);
    }

    #[test]
    fn test_report_generation() {
        let json = r#"{
            "relevance": {"score": 5, "reasoning": "Perfect"},
            "faithfulness": {"score": 4, "reasoning": "Good"},
            "completeness": {"score": 4, "reasoning": "Complete"},
            "coherence": {"score": 5, "reasoning": "Clear"},
            "groundedness": {"score": 4, "reasoning": "Accurate"},
            "overall_score": 4.4,
            "summary": "Excellent answer"
        }"#;

        let eval = LLMEvaluation::from_json(json).unwrap();
        let report = eval.report();

        // Check for numeric score (format may vary: 4.40 or 4.4)
        assert!(report.contains("4.4") || report.contains("4.40"),
                "Expected score 4.4 not found in report: {}", report);
        assert!(report.contains("5/5") && report.contains("Relevance"),
                "Expected 'Relevance: 5/5' not found");
        assert!(report.contains("Excellent answer"));

        // Verify the actual overall_score value
        assert!((eval.overall_score - 4.4).abs() < 0.01);
    }
}
