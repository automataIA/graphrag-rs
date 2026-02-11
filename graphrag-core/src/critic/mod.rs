use crate::core::{GraphRAGError, Result};
use crate::ollama::OllamaClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Result of a critic evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Score from 0.0 to 1.0 (1.0 = perfect)
    pub score: f32,
    /// Whether the answer is grounded in the context
    pub grounded: bool,
    /// Detailed feedback on why the score was given
    pub feedback: String,
}

/// Critic for evaluating RAG answers
pub struct Critic {
    client: Arc<OllamaClient>,
}

impl Critic {
    /// Create a new critic with an Ollama client
    pub fn new(client: Arc<OllamaClient>) -> Self {
        Self { client }
    }

    /// Evaluate the quality of a RAG answer against the query and context
    ///
    /// Returns an evaluation result with score, grounding information, and feedback
    pub async fn evaluate(
        &self,
        query: &str,
        context: &[String],
        answer: &str,
    ) -> Result<EvaluationResult> {
        let context_text = context.join("\n\n");

        let prompt = format!(
            "You are a strict critic for a RAG system. Your job is to evaluate the quality of a generated answer based on the provided query and retrieved context.\n\
            \n\
            Query: '{}'\n\
            \n\
            Retrieved Context:\n\
            {}\n\
            \n\
            Generated Answer:\n\
            {}\n\
            \n\
            Evaluate the answer on: \n\
            1. Grounding: Is every claim in the answer supported by the context? \n\
            2. Relevance: Does it answer the user's query? \n\
            3. Completeness: Is it missing critical info present in the context? \n\
            \n\
            Return ONLY a raw JSON object with these keys: \n\
            - 'score': float between 0.0 and 1.0 \n\
            - 'grounded': boolean \n\
            - 'feedback': string explanation \n\
            \n\
            JSON Response:",
            query, context_text, answer
        );

        let response_text = self.client.generate(&prompt).await?;

        // Clean up markdown code blocks if present
        let cleaned_json = response_text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let evaluation: EvaluationResult =
            serde_json::from_str(cleaned_json).map_err(|e| GraphRAGError::Generation {
                message: format!(
                    "Failed to parse critic response: {}. Text: {}",
                    e, cleaned_json
                ),
            })?;

        Ok(evaluation)
    }

    /// Refine an answer based on specific feedback
    pub async fn refine(
        &self,
        query: &str,
        current_answer: &str,
        feedback: &str,
    ) -> Result<String> {
        let prompt = format!(
            "You are an expert editor refining an answer for a RAG system.\n\
            \n\
            Original Query: '{}'\n\
            \n\
            Current Answer:\n\
            {}\n\
            \n\
            Critique/Feedback:\n\
            {}\n\
            \n\
            Please rewrite the answer to address the critique while maintaining accuracy and relevance. \n\
            Return ONLY the refined answer text.",
            query, current_answer, feedback
        );

        self.client.generate(&prompt).await
    }
}
