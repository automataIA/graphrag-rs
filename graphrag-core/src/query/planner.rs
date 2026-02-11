use crate::core::Result;
use crate::ollama::OllamaClient;
use serde::{Deserialize, Serialize};

/// Planner for breaking down complex queries
pub struct QueryPlanner {
    client: OllamaClient,
}

#[derive(Debug, Serialize, Deserialize)]
struct DecompositionResponse {
    sub_queries: Vec<String>,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(client: OllamaClient) -> Self {
        Self { client }
    }

    /// Decompose a complex query into simpler sub-queries
    pub async fn decompose(&self, query: &str) -> Result<Vec<String>> {
        let prompt = format!(
            "You are an expert query planner for a RAG system. \
            Your task is to decompose the following complex user query into a list of simple, independent sub-queries \
            that can be answered using vector search. \
            \
            Return ONLY a raw JSON object with a single key 'sub_queries' containing the list of strings. \
            Do not include any explanation, markdown formatting, or preamble. \
            \
            Query: '{}' \
            \
            JSON Response:",
            query
        );

        let response_text = self.client.generate(&prompt).await?;

        // Clean up potential markdown formatting (```json ... ```)
        let cleaned_json = response_text
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        // Parse JSON response
        let response: DecompositionResponse = serde_json::from_str(cleaned_json).map_err(|e| {
            crate::core::GraphRAGError::Generation {
                message: format!(
                    "Failed to parse planner response: {}. Text: {}",
                    e, cleaned_json
                ),
            }
        })?;

        Ok(response.sub_queries)
    }
}
