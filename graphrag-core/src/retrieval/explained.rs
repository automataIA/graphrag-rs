//! Structured answer types with reasoning trace.
//!
//! `ExplainedAnswer` and its supporting types (`SourceReference`, `SourceType`,
//! `ReasoningStep`) were extracted from `retrieval/mod.rs` to keep that module
//! focused on retrieval orchestration.

use super::{ResultType, SearchResult};
use std::collections::HashSet;

/// An answer with detailed explanation of the reasoning process
///
/// This struct provides transparency into how the GraphRAG system
/// arrived at its answer, including confidence scores, source references,
/// and step-by-step reasoning.
///
/// # Example
/// ```no_run
/// use graphrag_core::prelude::*;
///
/// # async fn example() -> graphrag_core::Result<()> {
/// let mut graphrag = GraphRAG::quick_start("Your document").await?;
/// let explained = graphrag.ask_explained("What is the main topic?").await?;
///
/// println!("Answer: {}", explained.answer);
/// println!("Confidence: {:.0}%", explained.confidence * 100.0);
///
/// for step in &explained.reasoning_steps {
///     println!("Step {}: {} (confidence: {:.0}%)",
///         step.step_number, step.description, step.confidence * 100.0);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ExplainedAnswer {
    /// The answer text
    pub answer: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Sources used to generate the answer
    pub sources: Vec<SourceReference>,
    /// Step-by-step reasoning trace
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Entities that were key to the answer
    pub key_entities: Vec<String>,
    /// Query analysis that guided retrieval
    pub query_analysis: Option<super::QueryAnalysis>,
}

/// Reference to a source document or chunk used in the answer
#[derive(Debug, Clone)]
pub struct SourceReference {
    /// Identifier of the source (chunk ID, document ID, or entity ID)
    pub id: String,
    /// Type of source
    pub source_type: SourceType,
    /// Relevant excerpt from the source
    pub excerpt: String,
    /// Relevance score to the query
    pub relevance_score: f32,
}

/// Type of source reference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceType {
    /// A text chunk from a document
    TextChunk,
    /// An entity in the knowledge graph
    Entity,
    /// A relationship between entities
    Relationship,
    /// A document-level summary
    Summary,
}

/// A single step in the reasoning process
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step number (1-indexed)
    pub step_number: u8,
    /// Description of what was done in this step
    pub description: String,
    /// IDs of entities involved in this step
    pub entities_used: Vec<String>,
    /// Evidence snippet that supports this step
    pub evidence_snippet: Option<String>,
    /// Confidence for this specific step
    pub confidence: f32,
}

impl ExplainedAnswer {
    /// Create a new explained answer from search results
    pub fn from_results(answer: String, search_results: &[SearchResult], query: &str) -> Self {
        // Calculate overall confidence from result scores
        let confidence = if search_results.is_empty() {
            0.0
        } else {
            let total_score: f32 = search_results.iter().map(|r| r.score).sum();
            let avg_score = total_score / search_results.len() as f32;
            // Normalize to 0-1 range (assuming scores are already somewhat normalized)
            (avg_score * 0.7 + 0.3).clamp(0.0, 1.0)
        };

        // Build source references
        let sources: Vec<SourceReference> = search_results
            .iter()
            .take(5) // Top 5 sources
            .map(|r| SourceReference {
                id: r.id.clone(),
                source_type: match r.result_type {
                    ResultType::Entity => SourceType::Entity,
                    ResultType::Chunk => SourceType::TextChunk,
                    ResultType::GraphPath => SourceType::Relationship,
                    ResultType::HierarchicalSummary => SourceType::Summary,
                    ResultType::Hybrid => SourceType::TextChunk,
                },
                excerpt: if r.content.len() > 200 {
                    format!("{}...", &r.content[..200])
                } else {
                    r.content.clone()
                },
                relevance_score: r.score,
            })
            .collect();

        // Build reasoning steps
        let mut reasoning_steps = Vec::new();
        let mut step_num = 1u8;

        // Step 1: Query analysis
        reasoning_steps.push(ReasoningStep {
            step_number: step_num,
            description: format!("Analyzed query: \"{}\"", query),
            entities_used: vec![],
            evidence_snippet: None,
            confidence: 0.95,
        });
        step_num += 1;

        // Step 2: Entity retrieval
        let unique_entities: HashSet<_> = search_results
            .iter()
            .flat_map(|r| r.entities.iter().cloned())
            .collect();
        if !unique_entities.is_empty() {
            reasoning_steps.push(ReasoningStep {
                step_number: step_num,
                description: format!("Found {} relevant entities", unique_entities.len()),
                entities_used: unique_entities.iter().take(5).cloned().collect(),
                evidence_snippet: None,
                confidence: 0.85,
            });
            step_num += 1;
        }

        // Step 3: Chunk retrieval
        let chunk_count = search_results
            .iter()
            .filter(|r| r.result_type == ResultType::Chunk || r.result_type == ResultType::Hybrid)
            .count();
        if chunk_count > 0 {
            reasoning_steps.push(ReasoningStep {
                step_number: step_num,
                description: format!("Retrieved {} relevant text chunks", chunk_count),
                entities_used: vec![],
                evidence_snippet: search_results.first().map(|r| {
                    if r.content.len() > 100 {
                        format!("{}...", &r.content[..100])
                    } else {
                        r.content.clone()
                    }
                }),
                confidence,
            });
            step_num += 1;
        }

        // Step 4: Answer synthesis
        reasoning_steps.push(ReasoningStep {
            step_number: step_num,
            description: "Synthesized answer from retrieved information".to_string(),
            entities_used: unique_entities.into_iter().take(3).collect(),
            evidence_snippet: None,
            confidence,
        });

        // Collect key entities
        let key_entities: Vec<String> = search_results
            .iter()
            .flat_map(|r| r.entities.iter().cloned())
            .take(10)
            .collect();

        Self {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        }
    }

    /// Format the explained answer for display
    pub fn format_display(&self) -> String {
        let mut output = String::new();

        // Answer
        output.push_str(&format!("**Answer:** {}\n\n", self.answer));

        // Confidence
        output.push_str(&format!(
            "**Confidence:** {:.0}%\n\n",
            self.confidence * 100.0
        ));

        // Reasoning steps
        if !self.reasoning_steps.is_empty() {
            output.push_str("**Reasoning:**\n");
            for step in &self.reasoning_steps {
                output.push_str(&format!(
                    "{}. {} (confidence: {:.0}%)\n",
                    step.step_number,
                    step.description,
                    step.confidence * 100.0
                ));
                if let Some(evidence) = &step.evidence_snippet {
                    output.push_str(&format!("   Evidence: \"{}\"\n", evidence));
                }
            }
            output.push('\n');
        }

        // Sources
        if !self.sources.is_empty() {
            output.push_str("**Sources:**\n");
            for (i, source) in self.sources.iter().enumerate() {
                output.push_str(&format!(
                    "{}. [{:?}] {} (relevance: {:.0}%)\n",
                    i + 1,
                    source.source_type,
                    source.id,
                    source.relevance_score * 100.0
                ));
            }
        }

        output
    }
}
