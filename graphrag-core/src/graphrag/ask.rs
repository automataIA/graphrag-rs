#![allow(unused_imports)]

use crate::config::Config;
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use crate::{critic, ollama, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

use super::GraphRAG;

impl GraphRAG {
    /// Query the system associated with reasoning (Query Decomposition)
    /// This splits the query into sub-queries, gathers context for all of them, and synthesizes an answer.
    #[cfg(feature = "async")]
    pub async fn ask_with_reasoning(&mut self, query: &str) -> Result<String> {
        // If planner is not available, fallback to standard ask
        if self.query_planner.is_none() {
            return self.ask(query).await;
        }

        self.ensure_initialized()?;
        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        let planner = self.query_planner.as_ref().expect("checked above");
        tracing::info!("Decomposing query: {}", query);

        // Decompose query
        let sub_queries = match planner.decompose(query).await {
            Ok(sq) => sq,
            Err(e) => {
                tracing::warn!(
                    "Query decomposition failed, falling back to standard query: {}",
                    e
                );
                vec![query.to_string()]
            },
        };

        tracing::info!("Sub-queries: {:?}", sub_queries);

        // Gather results for all sub-queries
        let mut all_results = Vec::new();
        for sub_query in sub_queries {
            match self.query_internal_with_results(&sub_query).await {
                Ok(results) => all_results.extend(results),
                Err(e) => tracing::warn!("Failed to execute sub-query '{}': {}", sub_query, e),
            }
        }

        if all_results.is_empty() {
            return Ok("No relevant information found for the decomposed queries.".to_string());
        }

        // Deduplicate results by ID
        // (Simple optimization to avoid duplicate context)
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut unique_results = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for result in all_results {
            if !seen_ids.contains(&result.id) {
                seen_ids.insert(result.id.clone());
                unique_results.push(result);
            }
        }

        if self.config.ollama.enabled {
            // Initial synthesis
            let mut current_answer = self
                .generate_semantic_answer_from_results(query, &unique_results)
                .await?;

            // Critic refinement loop
            if let Some(critic) = &self.critic {
                let mut attempts = 0;
                let max_retries = 3;

                while attempts < max_retries {
                    let context_strings: Vec<String> =
                        unique_results.iter().map(|r| r.content.clone()).collect();

                    let evaluation = match critic
                        .evaluate(query, &context_strings, &current_answer)
                        .await
                    {
                        Ok(eval) => eval,
                        Err(e) => {
                            tracing::warn!("Critic evaluation failed: {}", e);
                            break;
                        },
                    };

                    tracing::info!(
                        "Critic Evaluation (Attempt {}): Score={:.2}, Grounded={}, Feedback='{}'",
                        attempts + 1,
                        evaluation.score,
                        evaluation.grounded,
                        evaluation.feedback
                    );

                    if evaluation.score >= 0.7 && evaluation.grounded {
                        tracing::info!("Answer accepted by critic.");
                        break;
                    }

                    tracing::warn!("Answer rejected by critic. Refining...");

                    // Refine the answer using the feedback
                    current_answer = critic
                        .refine(query, &current_answer, &evaluation.feedback)
                        .await?;
                    attempts += 1;
                }
            }

            return Ok(current_answer);
        }

        // Fallback formatting
        let formatted: Vec<String> = unique_results
            .into_iter()
            .take(10)
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();
        Ok(formatted.join("\n"))
    }

    /// Query the system for relevant information
    #[cfg(feature = "async")]
    pub async fn ask(&mut self, query: &str) -> Result<String> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        // Get full search results with metadata
        let search_results = self.query_internal_with_results(query).await?;

        // If Ollama is enabled, generate semantic answer using LLM
        if self.config.ollama.enabled {
            return self
                .generate_semantic_answer_from_results(query, &search_results)
                .await;
        }

        // Fallback: return formatted search results
        let formatted: Vec<String> = search_results
            .into_iter()
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();
        Ok(formatted.join("\n"))
    }

    /// Query the system for relevant information (synchronous version)
    #[cfg(not(feature = "async"))]
    pub fn ask(&mut self, query: &str) -> Result<String> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        let retrieval = self
            .retrieval_system
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        let results = retrieval.query(query)?;
        Ok(results.join("\n"))
    }

    /// Query the system and return an explained answer with reasoning trace
    ///
    /// Unlike `ask()`, this method returns detailed information about:
    /// - Confidence score
    /// - Source references
    /// - Step-by-step reasoning
    /// - Key entities used
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::prelude::*;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let mut graphrag = GraphRAG::quick_start("Your document text").await?;
    /// let explained = graphrag.ask_explained("What is the main topic?").await?;
    ///
    /// println!("Answer: {}", explained.answer);
    /// println!("Confidence: {:.0}%", explained.confidence * 100.0);
    ///
    /// for step in &explained.reasoning_steps {
    ///     println!("Step {}: {}", step.step_number, step.description);
    /// }
    ///
    /// for source in &explained.sources {
    ///     println!("Source: {} (relevance: {:.0}%)",
    ///         source.id, source.relevance_score * 100.0);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn ask_explained(&mut self, query: &str) -> Result<retrieval::ExplainedAnswer> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        // Get search results
        let search_results = self.query_internal_with_results(query).await?;

        // Generate the answer
        let answer = if self.config.ollama.enabled {
            self.generate_semantic_answer_from_results(query, &search_results)
                .await?
        } else {
            // Fallback: concatenate top results
            search_results
                .iter()
                .take(3)
                .map(|r| r.content.clone())
                .collect::<Vec<_>>()
                .join(" ")
        };

        // Build the explained answer
        let explained = retrieval::ExplainedAnswer::from_results(answer, &search_results, query);

        Ok(explained)
    }

    /// Internal query method (public for CLI access to raw results)
    pub async fn query_internal(&mut self, query: &str) -> Result<Vec<String>> {
        let retrieval = self
            .retrieval_system
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Add embeddings to graph if not already present
        retrieval.add_embeddings_to_graph(graph).await?;

        // Use hybrid query for real semantic search
        let search_results = retrieval.hybrid_query(query, graph).await?;

        // Convert search results to strings
        let result_strings: Vec<String> = search_results
            .into_iter()
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect();

        Ok(result_strings)
    }

    /// Internal query method that returns full SearchResult objects
    #[cfg(feature = "async")]
    async fn query_internal_with_results(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::SearchResult>> {
        let retrieval = self
            .retrieval_system
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Add embeddings to graph if not already present
        retrieval.add_embeddings_to_graph(graph).await?;

        // Use hybrid query for real semantic search
        retrieval.hybrid_query(query, graph).await
    }

    /// Generate semantic answer from SearchResult objects
    #[cfg(feature = "async")]
    async fn generate_semantic_answer_from_results(
        &self,
        query: &str,
        search_results: &[retrieval::SearchResult],
    ) -> Result<String> {
        use crate::ollama::OllamaClient;

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        // Build context from search results by fetching actual chunk content.
        // We track chunk IDs to avoid duplicating the same chunk from multiple entity results.
        let mut context_parts = Vec::new();
        let mut seen_chunk_ids = std::collections::HashSet::new();

        for result in search_results.iter() {
            // For entity results, fetch the chunks where the entity appears
            if result.result_type == retrieval::ResultType::Entity
                && !result.source_chunks.is_empty()
            {
                let entity_label = result
                    .content
                    .split(" (score:")
                    .next()
                    .unwrap_or(&result.content);
                for chunk_id_str in &result.source_chunks {
                    if seen_chunk_ids.contains(chunk_id_str) {
                        continue;
                    }
                    let chunk_id = ChunkId::new(chunk_id_str.clone());
                    if let Some(chunk) = graph.chunks().find(|c| c.id == chunk_id) {
                        seen_chunk_ids.insert(chunk_id_str.clone());
                        context_parts.push((
                            result.score,
                            format!(
                                "[Entity: {} | Relevance: {:.2}]\n{}",
                                entity_label, result.score, chunk.content
                            ),
                        ));
                    }
                }
            }
            // For chunk results, use the full content directly
            else if result.result_type == retrieval::ResultType::Chunk {
                if !seen_chunk_ids.contains(&result.id) {
                    seen_chunk_ids.insert(result.id.clone());
                    context_parts.push((
                        result.score,
                        format!(
                            "[Chunk | Relevance: {:.2}]\n{}",
                            result.score, result.content
                        ),
                    ));
                }
            }
            // For other result types, use content as-is
            else {
                context_parts.push((
                    result.score,
                    format!(
                        "[{:?} | Relevance: {:.2}]\n{}",
                        result.result_type, result.score, result.content
                    ),
                ));
            }
        }

        // Sort by relevance descending, then join
        context_parts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let context = context_parts
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        if context.trim().is_empty() {
            return Ok("No relevant information found in the knowledge graph.".to_string());
        }

        // Create Ollama client
        let client = OllamaClient::new(self.config.ollama.clone());

        // Build prompt for semantic answer generation with RAG best practices (2025)
        let prompt = format!(
            "You are a knowledgeable assistant specialized in answering questions based on a knowledge graph.\n\n\
            IMPORTANT INSTRUCTIONS:\n\
            - Answer ONLY using information from the provided context below\n\
            - Synthesize information from ALL context sections to give a comprehensive answer\n\
            - Provide direct, conversational, and natural responses\n\
            - Do NOT show your reasoning process or use <think> tags\n\
            - If the context lacks sufficient information, clearly state: \"I don't have enough information to answer this question.\"\n\
            - Aim for a complete answer (3-6 sentences) that covers different aspects found across the context\n\
            - Use a natural, helpful tone as if speaking to a person\n\n\
            CONTEXT:\n\
            {}\n\n\
            QUESTION: {}\n\n\
            ANSWER (direct response only, no reasoning):",
            context, query
        );

        // Dynamic num_ctx: prompt tokens + generous output budget + 20% margin
        let max_answer_tokens: u32 = 800;
        let prompt_tokens = (prompt.len() / 4) as u32;
        let total = prompt_tokens + max_answer_tokens;
        let with_margin = (total as f32 * 1.20) as u32;
        let num_ctx = (((with_margin + 1023) / 1024) * 1024).clamp(4096, 131_072);

        let params = crate::ollama::OllamaGenerationParams {
            num_predict: Some(max_answer_tokens),
            temperature: self.config.ollama.temperature,
            num_ctx: Some(num_ctx),
            keep_alive: self.config.ollama.keep_alive.clone(),
            ..Default::default()
        };

        // Generate answer using LLM with dynamic context window
        match client.generate_with_params(&prompt, params).await {
            Ok(answer) => {
                // Post-processing: Remove <think> tags if present (Qwen3)
                let cleaned_answer = Self::remove_thinking_tags(&answer);
                Ok(cleaned_answer.trim().to_string())
            },
            Err(e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    "LLM generation failed: {}. Falling back to search results.",
                    e
                );

                // Fallback: return formatted search results
                Ok(format!(
                    "Relevant information from knowledge graph:\n\n{}",
                    context
                ))
            },
        }
    }

    /// Remove thinking tags from LLM output (for Qwen3 and similar models)
    ///
    /// Qwen3 often outputs <think>...</think> tags showing internal reasoning.
    /// This function removes all such tags and their content.
    #[cfg(feature = "async")]
    fn remove_thinking_tags(text: &str) -> String {
        // Remove all <think>...</think> blocks (including nested ones)
        // Use a simple approach: repeatedly remove until no more found
        let mut result = text.to_string();

        while let Some(start) = result.find("<think>") {
            // Find corresponding closing tag
            if let Some(end) = result[start..].find("</think>") {
                // Remove the entire block
                let end_pos = start + end + "</think>".len();
                result.replace_range(start..end_pos, "");
            } else {
                // No closing tag found, just remove opening tag
                result.replace_range(start..start + "<think>".len(), "");
                break;
            }
        }

        result.trim().to_string()
    }

    /// Query using PageRank-based retrieval (when pagerank feature is enabled)
    #[cfg(all(feature = "pagerank", feature = "async"))]
    pub async fn ask_with_pagerank(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::pagerank_retrieval::ScoredResult>> {
        use crate::retrieval::pagerank_retrieval::PageRankRetrievalSystem;

        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let pagerank_system = PageRankRetrievalSystem::new(10);
        pagerank_system.search_with_pagerank(query, graph, Some(5))
    }

    /// Query using PageRank-based retrieval (when pagerank feature is enabled, sync version)
    #[cfg(all(feature = "pagerank", not(feature = "async")))]
    pub fn ask_with_pagerank(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::pagerank_retrieval::ScoredResult>> {
        use crate::retrieval::pagerank_retrieval::PageRankRetrievalSystem;

        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph()?;
        }

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let pagerank_system = PageRankRetrievalSystem::new(10);
        pagerank_system.search_with_pagerank(query, graph, Some(5))
    }
}
