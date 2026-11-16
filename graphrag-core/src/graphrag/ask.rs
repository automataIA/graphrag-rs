#![allow(unused_imports)]

use crate::config::Config;
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use crate::{critic, llm, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

use super::summaries::{is_summary_chunk, is_summary_chunk_id};
use super::GraphRAG;

#[cfg(feature = "async")]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Provider-reported token counts associated with a generated answer.
pub struct TokenUsage {
    /// Number of prompt tokens reported by the configured provider.
    pub prompt_tokens: u64,
    /// Number of completion tokens reported by the configured provider.
    pub completion_tokens: u64,
    /// Total number of tokens reported by the configured provider.
    pub total_tokens: u64,
}

/// An answer together with optional provider-measured token usage.
#[cfg(feature = "async")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerResponse {
    /// The generated answer or formatted local retrieval result.
    pub answer: String,
    /// Provider-reported usage. Local retrieval and providers that omit usage return `None`.
    pub usage: Option<TokenUsage>,
}

#[cfg(feature = "async")]
struct GeneratedAnswer {
    text: String,
    usage: Option<TokenUsage>,
}

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

        if self.config.llm.enabled {
            // Initial synthesis
            let mut current_answer = self
                .generate_semantic_answer_from_results(query, None, &unique_results)
                .await?
                .text;

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
        self.ask_with_results(query).await.map(|(answer, _)| answer)
    }

    /// Run retrieval only (no LLM), building the graph first if documents are pending.
    #[cfg(feature = "async")]
    pub async fn search(&mut self, query: &str) -> Result<Vec<retrieval::SearchResult>> {
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }

        self.query_internal_with_results(query).await
    }

    /// Answer a query and return the retrieval results used for it, from a single retrieval pass.
    #[cfg(feature = "async")]
    pub async fn ask_with_results(
        &mut self,
        query: &str,
    ) -> Result<(String, Vec<retrieval::SearchResult>)> {
        // HyDE generation runs concurrently with retrieval (retrieval never depends on it).
        let hyde = self.hyde_retrieval_query(query);
        let (hyde_result, search_results) = tokio::join!(hyde, self.search(query));
        let (hyde_query, _) = hyde_result;
        let search_results = search_results?;

        // If the LLM is enabled, generate semantic answer using LLM
        let answer = if self.config.llm.enabled {
            self.generate_semantic_answer_from_results(
                query,
                hyde_query.as_deref(),
                &search_results,
            )
            .await?
            .text
        } else {
            // Fallback: return formatted search results
            search_results
                .iter()
                .map(|r| format!("{} (score: {:.2})", r.content, r.score))
                .collect::<Vec<_>>()
                .join("\n")
        };
        Ok((answer, search_results))
    }

    /// Query the system and preserve measured token usage when the provider reports it.
    ///
    /// This is the rich counterpart to [`Self::ask`]. Existing callers can keep using
    /// `ask()` when they only need text; `usage` is `None` for local retrieval, LLM
    /// responses without usage, and any provider response that omits the usage object.
    #[cfg(feature = "async")]
    pub async fn ask_with_usage(&mut self, query: &str) -> Result<AnswerResponse> {
        self.ensure_initialized()?;
        if self.has_documents() && !self.has_graph() {
            self.build_graph().await?;
        }
        let hyde = self.hyde_retrieval_query(query);
        let (hyde_result, search_results) =
            tokio::join!(hyde, self.query_internal_with_results(query));
        let (hyde_query, hyde_usage) = hyde_result;
        let search_results = search_results?;
        if self.config.llm.enabled {
            let generated = self
                .generate_semantic_answer_from_results(
                    query,
                    hyde_query.as_deref(),
                    &search_results,
                )
                .await?;
            return Ok(AnswerResponse {
                answer: generated.text,
                usage: sum_usage(hyde_usage, generated.usage),
            });
        }
        let answer = search_results
            .into_iter()
            .map(|r| format!("{} (score: {:.2})", r.content, r.score))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(AnswerResponse {
            answer,
            usage: None,
        })
    }

    /// Query the system synchronously using local lexical and graph retrieval.
    pub fn ask_sync(&mut self, query: &str) -> Result<String> {
        self.ensure_sync_supported()?;
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph_sync()?;
        }

        let retrieval = self
            .retrieval_system
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;
        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let results = retrieval.query(query, graph)?;
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

        // Get search results (HyDE expands only the retrieval query, never the answer prompt)
        let hyde = self.hyde_retrieval_query(query);
        let (hyde_result, search_results) =
            tokio::join!(hyde, self.query_internal_with_results(query));
        let (hyde_query, _) = hyde_result;
        let search_results = search_results?;

        // Generate the answer
        let answer = if self.config.llm.enabled {
            self.generate_semantic_answer_from_results(
                query,
                hyde_query.as_deref(),
                &search_results,
            )
            .await?
            .text
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
    #[cfg(feature = "async")]
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

    /// Return raw local retrieval results synchronously.
    pub fn query_internal_sync(&mut self, query: &str) -> Result<Vec<String>> {
        self.ensure_sync_supported()?;
        let retrieval = self
            .retrieval_system
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Retrieval system not initialized".to_string(),
            })?;
        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;
        retrieval.query(query, graph)
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
        &mut self,
        query: &str,
        hyde_query: Option<&str>,
        search_results: &[retrieval::SearchResult],
    ) -> Result<GeneratedAnswer> {
        use crate::llm::LlmClient;

        // Build the context parts (direct chunk hits first, entity chunks capped).
        let query_embedding = match self.retrieval_system.as_mut() {
            Some(retrieval) => retrieval.embed_query(query).await.ok(),
            None => None,
        };
        let hyde_embedding = match (hyde_query, self.retrieval_system.as_mut()) {
            (Some(hq), Some(retrieval)) => retrieval.embed_query(hq).await.ok(),
            _ => None,
        };
        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;
        let mut reranked_ids = std::collections::HashSet::new();
        let mut texts: Vec<String> = Vec::new();
        let rerank_cfg = &self.config.reranker;
        // Chapter summaries: at most `max_in_context` of them are added after the original
        // chunks; everything else is selected as if the index held no summary chunk.
        let summaries_in_graph = graph.chunks().filter(|c| is_summary_chunk(c)).count();
        let summary_cap = if summaries_in_graph > 0 {
            self.config.chapter_summaries.max_in_context
        } else {
            0
        };
        let mut summary_texts: Vec<String> = Vec::new();
        let mut fused_ids: Vec<String> = Vec::new();
        let unfiltered_results = search_results;
        let search_results_without_summaries: Vec<retrieval::SearchResult>;
        let search_results = if summaries_in_graph > 0 {
            search_results_without_summaries = search_results
                .iter()
                .filter(|r| !is_summary_chunk_id(&r.id))
                .cloned()
                .collect();
            search_results_without_summaries.as_slice()
        } else {
            search_results
        };
        // Map-reduce (opt-in): chunk candidates for the per-chunk map calls, best first.
        let map_active = self.config.retrieval.map_reduce && self.config.llm.enabled;
        let map_cap = self.config.retrieval.map_reduce_chunks.max(1);
        let mut map_pool: Vec<PoolChunk> = Vec::new();
        if rerank_cfg.enabled {
            let max_pool = if map_active {
                rerank_cfg.candidates.max(1).max(map_cap)
            } else {
                rerank_cfg.candidates.max(1)
            };
            let mut pool =
                build_candidate_pool(graph, query_embedding.as_deref(), search_results, max_pool);
            let base_pool = pool.len();
            if pool.len() < max_pool || summary_cap > 0 {
                if let (Some(retrieval), Some(embedding)) =
                    (self.retrieval_system.as_mut(), query_embedding.as_deref())
                {
                    // over-fetch: summary chunks in the fused list must not displace originals
                    match retrieval
                        .fused_chunk_candidates(
                            query,
                            embedding,
                            graph,
                            max_pool + summaries_in_graph,
                        )
                        .await
                    {
                        Ok(ids) => {
                            extend_pool_with_chunk_ids(graph, &mut pool, &ids, max_pool);
                            fused_ids.extend(ids);
                        },
                        Err(e) => tracing::warn!("Fused chunk candidates failed: {e}"),
                    }
                }
            }
            let base_pool_before_hyde = pool.len();
            if let (Some(hq), Some(retrieval), Some(embedding)) = (
                hyde_query,
                self.retrieval_system.as_mut(),
                hyde_embedding.as_deref(),
            ) {
                match retrieval
                    .fused_chunk_candidates(hq, embedding, graph, max_pool + summaries_in_graph)
                    .await
                {
                    Ok(ids) => {
                        extend_pool_with_chunk_ids(
                            graph,
                            &mut pool,
                            &ids,
                            max_pool.saturating_mul(2),
                        );
                        fused_ids.extend(ids);
                    },
                    Err(e) => tracing::warn!("HyDE fused chunk candidates failed: {e}"),
                }
            }
            let hyde_added = pool.len() - base_pool_before_hyde;
            tracing::info!(
                "Reranker pool: {} chunks ({} from results, {} from fused dense+BM25)",
                pool.len(),
                base_pool,
                base_pool_before_hyde - base_pool
            );
            // Summary candidates join the pool after the originals, so the original
            // chunks are the same as without them; they are only ranked together.
            let original_len = pool.len();
            if summary_cap > 0 {
                pool.extend(summary_candidates(
                    graph,
                    unfiltered_results,
                    &fused_ids,
                    max_pool,
                ));
            }
            let mut hyde_won = std::collections::HashSet::new();
            let ranking = match hyde_query {
                None => {
                    // With the map (or summary candidates) the ranking must go past `top_k`.
                    let top_n = if map_active || pool.len() > original_len {
                        pool.len().max(1)
                    } else {
                        rerank_cfg.top_k.max(1)
                    };
                    rerank_pool(rerank_cfg, query, &pool, top_n).await
                },
                Some(hq) => {
                    let all = pool.len().max(1);
                    let orig = rerank_pool(rerank_cfg, query, &pool, all).await;
                    let hyde = rerank_pool(rerank_cfg, hq, &pool, all).await;
                    fuse_rankings(orig, hyde).map(|fused| {
                        hyde_won = fused
                            .iter()
                            .take(rerank_cfg.top_k.max(1))
                            .filter(|(_, _, won)| *won)
                            .map(|(idx, _, _)| *idx)
                            .collect();
                        fused.into_iter().map(|(i, s, _)| (i, s)).collect()
                    })
                },
            };
            if hyde_query.is_some() {
                tracing::info!(
                    "HyDE fusion: pool {} chunks (+{} from HyDE), {} top chunks from the HyDE ranking",
                    pool.len(),
                    hyde_added,
                    hyde_won.len()
                );
            }
            if map_active {
                // the map calls only get original chunks (summary entries sit past `original_len`)
                let order = ranking.as_ref().ok().map(|r| r.as_slice());
                map_pool = order_map_pool(&pool[..original_len], order, map_cap);
            }
            let summary_picks = match &ranking {
                Ok(r) if original_len < pool.len() => {
                    pick_summaries(r, original_len, rerank_cfg.top_k, summary_cap)
                },
                _ => Vec::new(),
            };
            let ranking = ranking.map(|r| {
                r.into_iter()
                    .filter(|(idx, _)| *idx < original_len)
                    .collect()
            });
            for (idx, score) in select_reranked(&pool, ranking, rerank_cfg.top_k) {
                reranked_ids.insert(pool[idx].id.clone());
                texts.push(format!(
                    "[Chunk | Rerank: {score:.2}]\n{}",
                    pool[idx].content
                ));
            }
            for (idx, score) in summary_picks {
                summary_texts.push(format!(
                    "[Chunk | Rerank: {score:.2}]\n{}",
                    pool[idx].content
                ));
            }
        } else if map_active {
            map_pool =
                build_candidate_pool(graph, query_embedding.as_deref(), search_results, map_cap);
            if map_pool.len() < map_cap {
                if let (Some(retrieval), Some(embedding)) =
                    (self.retrieval_system.as_mut(), query_embedding.as_deref())
                {
                    match retrieval
                        .fused_chunk_candidates(
                            query,
                            embedding,
                            graph,
                            map_cap + summaries_in_graph,
                        )
                        .await
                    {
                        Ok(ids) => extend_pool_with_chunk_ids(graph, &mut map_pool, &ids, map_cap),
                        Err(e) => tracing::warn!("Fused chunk candidates failed: {e}"),
                    }
                }
            }
        }
        texts.extend(build_context_parts_excluding(
            graph,
            query_embedding.as_deref(),
            search_results,
            &reranked_ids,
        ));
        if !rerank_cfg.enabled {
            summary_texts = summary_hit_parts(unfiltered_results, summary_cap);
        }
        // Summaries come last: when the budget trims, they go first.
        texts.extend(summary_texts);
        let total_parts = texts.len();
        // Graph facts (opt-in) are never dropped by the budget: a fixed token reserve
        // (min(2000, max_context_tokens / 3)) is taken off the chunk budget, the chunk
        // parts are fitted in what is left, then the facts (ranked against the query and
        // the kept chunks) are rendered to fit the reserve and placed FIRST. Without any
        // chunk part the context stays empty (no-answer path).
        let max_ctx = self.config.llm.max_context_tokens;
        let reserve = if self.config.retrieval.graph_facts {
            GRAPH_FACTS_RESERVE_TOKENS.min(max_ctx / 3)
        } else {
            0
        };
        // Map-reduce notes: the reserve is what the notes part actually uses (<= the cap).
        let (notes_part, map_usage) = if map_active && !texts.is_empty() && !map_pool.is_empty() {
            let (notes, usage) = run_map_calls(
                &self.config.llm,
                query,
                &map_pool,
                self.config.llm.max_concurrent_requests,
            )
            .await;
            let cap = MAP_REDUCE_RESERVE_TOKENS.min(max_ctx / 4);
            (build_notes_part(&notes, cap as usize), usage)
        } else {
            (None, None)
        };
        let map_reserve = notes_part
            .as_deref()
            .map_or(0, |part| estimate_tokens(part) as u32);
        let mut texts = fit_context_to_budget(
            texts,
            max_ctx.saturating_sub(reserve).saturating_sub(map_reserve),
        );
        tracing::info!(
            "Context budget {} tokens: kept {} of {} parts",
            max_ctx,
            texts.len(),
            total_parts
        );
        let mut facts_inserted = false;
        if self.config.retrieval.graph_facts && !texts.is_empty() {
            if let Some(facts) = build_graph_facts_part(
                graph,
                query,
                search_results,
                &texts,
                rerank_cfg,
                reserve as usize,
            )
            .await
            {
                texts.insert(0, facts);
                facts_inserted = true;
            }
        }
        if let Some(notes) = notes_part {
            if !texts.is_empty() {
                texts.insert(usize::from(facts_inserted), notes);
            }
        }
        let context = texts.join("\n\n---\n\n");

        if context.trim().is_empty() {
            return Ok(GeneratedAnswer {
                text: "No relevant information found in the knowledge graph.".to_string(),
                usage: None,
            });
        }

        // Build prompt for semantic answer generation with RAG best practices (2025)
        let prompt = build_answer_prompt(&context, query);
        dump_answer_context(query, &context, &prompt);

        let params = crate::llm::LlmGenerationParams {
            max_tokens: Some(answer_max_tokens(&self.config.llm)),
            temperature: self.config.llm.temperature,
            ..Default::default()
        };

        // A configured provider must either answer successfully or return its error;
        // returning retrieved context here would make a failed provider look healthy.
        let response = LlmClient::new(self.config.llm.clone())
            .generate_with_full_response(&prompt, params)
            .await?;
        let usage = response.usage.map(|value| TokenUsage {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        });
        let usage = sum_usage(map_usage, usage);
        let answer = response.text;

        // Post-processing: Remove <think> tags if present (Qwen3)
        let cleaned_answer = Self::remove_thinking_tags(&answer);
        Ok(GeneratedAnswer {
            text: cleaned_answer.trim().to_string(),
            usage,
        })
    }

    /// Build the HyDE query (`"{query}\n{passage}"`) that is fused with the original query
    /// at rerank time (retrieval itself always uses the original query).
    ///
    /// With `retrieval.hyde` and `llm.enabled`, one extra LLM call writes a hypothetical
    /// passage. Returns `None` when HyDE is off, the reranker is disabled (HyDE only acts
    /// through it: warning logged, no LLM call), or the call fails (warning logged). The
    /// token usage of the call is returned too (it is also counted in the global usage totals).
    ///
    /// Returns a future that owns everything it needs (no borrow of `self`), so callers can
    /// run it concurrently with the `&mut self` retrieval via `tokio::join!`. With HyDE off
    /// nothing is cloned and no LLM call is made.
    #[cfg(feature = "async")]
    fn hyde_retrieval_query(
        &self,
        query: &str,
    ) -> impl std::future::Future<Output = (Option<String>, Option<TokenUsage>)> {
        let active = self.config.retrieval.hyde && self.config.llm.enabled;
        let reranker_on = self.config.reranker.enabled;
        let llm = (active && reranker_on).then(|| self.config.llm.clone());
        let query = query.to_string();
        async move {
            if !active {
                return (None, None);
            }
            let Some(llm) = llm else {
                tracing::warn!("retrieval.hyde needs reranker.enabled: skipping HyDE");
                return (None, None);
            };
            Self::hyde_query_from_llm(&llm, &query).await
        }
    }

    #[cfg(feature = "async")]
    async fn hyde_query_from_llm(
        llm: &crate::llm::LlmConfig,
        query: &str,
    ) -> (Option<String>, Option<TokenUsage>) {
        match Self::generate_hyde_passage(llm, query).await {
            Ok((passage, usage)) => {
                let passage = clean_hyde_passage(&passage);
                tracing::info!("HyDE passage: {}", hyde_log_preview(&passage));
                let combined = combine_retrieval_query(query, Some(&passage));
                if combined == query {
                    (None, usage)
                } else {
                    (Some(combined), usage)
                }
            },
            Err(e) => {
                tracing::warn!("HyDE generation failed, using the original query: {e}");
                (None, None)
            },
        }
    }

    #[cfg(all(feature = "async", feature = "llm"))]
    async fn generate_hyde_passage(
        llm: &crate::llm::LlmConfig,
        query: &str,
    ) -> Result<(String, Option<TokenUsage>)> {
        let params = crate::llm::LlmGenerationParams {
            // Reasoning models need room for the thinking block before the passage.
            max_tokens: Some(if llm.disable_thinking { 200 } else { 4096 }),
            temperature: llm.temperature,
            ..Default::default()
        };
        let response = crate::llm::LlmClient::new(llm.clone())
            .generate_with_full_response(&build_hyde_prompt(query), params)
            .await?;
        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
        Ok((response.text, usage))
    }

    #[cfg(all(feature = "async", not(feature = "llm")))]
    async fn generate_hyde_passage(
        _llm: &crate::llm::LlmConfig,
        _query: &str,
    ) -> Result<(String, Option<TokenUsage>)> {
        Err(GraphRAGError::Config {
            message: "retrieval.hyde needs the `llm` feature".to_string(),
        })
    }

    /// Remove thinking tags from LLM output (for Qwen3 and similar models)
    ///
    /// Qwen3 often outputs <think>...</think> tags showing internal reasoning.
    /// This function removes all such tags and their content.
    #[cfg(feature = "async")]
    pub(super) fn remove_thinking_tags(text: &str) -> String {
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
    #[cfg(feature = "pagerank")]
    pub fn ask_with_pagerank_sync(
        &mut self,
        query: &str,
    ) -> Result<Vec<retrieval::pagerank_retrieval::ScoredResult>> {
        use crate::retrieval::pagerank_retrieval::PageRankRetrievalSystem;

        self.ensure_sync_supported()?;
        self.ensure_initialized()?;

        if self.has_documents() && !self.has_graph() {
            self.build_graph_sync()?;
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

/// Conservative token estimate (chars / 3, rounded up) for prompt budgeting.
#[cfg(feature = "async")]
fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(3)
}

/// Maximum number of chunks one entity result may contribute to the answer context.
#[cfg(feature = "async")]
const MAX_CHUNKS_PER_ENTITY: usize = 2;

/// Build the ordered context parts sent to the LLM from retrieval results.
///
/// Order: direct chunk hits (by score), other result types, then entity-derived chunks.
/// Each entity contributes at most [`MAX_CHUNKS_PER_ENTITY`] chunks, ranked by cosine
/// similarity to the query embedding when available (otherwise in stored order), so a hub
/// entity cannot crowd the budget with its early chapters. Chunks are deduplicated by id.
#[cfg(all(test, feature = "async"))]
fn build_context_parts(
    graph: &KnowledgeGraph,
    query_embedding: Option<&[f32]>,
    search_results: &[retrieval::SearchResult],
) -> Vec<String> {
    build_context_parts_excluding(
        graph,
        query_embedding,
        search_results,
        &std::collections::HashSet::new(),
    )
}

/// Like [`build_context_parts`], but chunks whose id is in `exclude` (already placed in
/// the context, e.g. by the reranker) are skipped everywhere.
#[cfg(feature = "async")]
fn build_context_parts_excluding(
    graph: &KnowledgeGraph,
    query_embedding: Option<&[f32]>,
    search_results: &[retrieval::SearchResult],
    exclude: &std::collections::HashSet<String>,
) -> Vec<String> {
    use crate::vector::VectorUtils;
    use retrieval::ResultType;

    let mut seen_chunk_ids = exclude.clone();
    // Texts already in the context: identical text must enter it only once, whatever
    // result type carried it (e.g. an entity result whose content is a chunk's text).
    let mut seen_text: std::collections::HashSet<String> = exclude
        .iter()
        .filter_map(|id| graph.get_chunk(&ChunkId::new(id.clone())))
        .map(|c| c.content.trim().to_string())
        .collect();

    let mut chunk_hits: Vec<(f32, String)> = Vec::new();
    for result in search_results
        .iter()
        .filter(|r| r.result_type == ResultType::Chunk)
    {
        if seen_chunk_ids.insert(result.id.clone())
            && seen_text.insert(result.content.trim().to_string())
        {
            chunk_hits.push((
                result.score,
                format!(
                    "[Chunk | Relevance: {:.2}]\n{}",
                    result.score, result.content
                ),
            ));
        }
    }
    chunk_hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut parts: Vec<String> = chunk_hits.into_iter().map(|(_, text)| text).collect();

    for result in search_results.iter().filter(|r| {
        r.result_type != ResultType::Chunk
            && (r.result_type != ResultType::Entity || r.source_chunks.is_empty())
    }) {
        if !seen_text.insert(result.content.trim().to_string()) {
            continue;
        }
        parts.push(format!(
            "[{:?} | Relevance: {:.2}]\n{}",
            result.result_type, result.score, result.content
        ));
    }

    for result in search_results
        .iter()
        .filter(|r| r.result_type == ResultType::Entity && !r.source_chunks.is_empty())
    {
        let entity_label = result
            .content
            .split(" (score:")
            .next()
            .unwrap_or(&result.content);
        let mut candidates: Vec<(f32, &TextChunk)> = result
            .source_chunks
            .iter()
            .filter(|id| !seen_chunk_ids.contains(*id))
            .filter_map(|id| {
                let chunk_id = ChunkId::new(id.clone());
                graph.chunks().find(|c| c.id == chunk_id)
            })
            .map(|chunk| {
                let similarity = match (query_embedding, chunk.embedding.as_deref()) {
                    (Some(q), Some(e)) => VectorUtils::cosine_similarity(q, e),
                    _ => 0.0,
                };
                (similarity, chunk)
            })
            .collect();
        if query_embedding.is_some() {
            // Stable sort: ties (e.g. no chunk embedding) keep the stored order.
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        for (_, chunk) in candidates.into_iter().take(MAX_CHUNKS_PER_ENTITY) {
            seen_chunk_ids.insert(chunk.id.to_string());
            if !seen_text.insert(chunk.content.trim().to_string()) {
                continue;
            }
            parts.push(format!(
                "[Entity: {} | Relevance: {:.2}]\n{}",
                entity_label, result.score, chunk.content
            ));
        }
    }

    parts
}

/// Max chunks one entity (or other non-chunk result) adds to the reranker candidate pool,
/// so a hub entity cannot fill the whole pool.
#[cfg(feature = "async")]
const POOL_CHUNKS_PER_RESULT: usize = 5;

/// A distinct chunk offered to the reranker.
#[cfg(feature = "async")]
#[derive(Debug, Clone, PartialEq)]
struct PoolChunk {
    id: String,
    content: String,
}

/// Build the reranker candidate pool: direct chunk hits (by score), then chunks reachable
/// through `source_chunks` of the other results (per result ranked by query similarity
/// when embeddings exist, capped at [`POOL_CHUNKS_PER_RESULT`]). Deduplicated by chunk id,
/// at most `max` chunks, in current priority order.
#[cfg(feature = "async")]
fn build_candidate_pool(
    graph: &KnowledgeGraph,
    query_embedding: Option<&[f32]>,
    search_results: &[retrieval::SearchResult],
    max: usize,
) -> Vec<PoolChunk> {
    use crate::vector::VectorUtils;
    use retrieval::ResultType;

    let mut seen = std::collections::HashSet::new();
    let mut pool: Vec<PoolChunk> = Vec::new();

    let mut hits: Vec<&retrieval::SearchResult> = search_results
        .iter()
        .filter(|r| r.result_type == ResultType::Chunk)
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for hit in hits {
        if pool.len() >= max {
            return pool;
        }
        if is_summary_chunk_id(&hit.id) {
            continue;
        }
        if seen.insert(hit.id.clone()) {
            let content = graph
                .chunks()
                .find(|c| c.id.to_string() == hit.id)
                .map(|c| c.content.clone())
                .unwrap_or_else(|| hit.content.clone());
            pool.push(PoolChunk {
                id: hit.id.clone(),
                content,
            });
        }
    }

    for result in search_results
        .iter()
        .filter(|r| r.result_type != ResultType::Chunk && !r.source_chunks.is_empty())
    {
        let mut candidates: Vec<(f32, &TextChunk)> = result
            .source_chunks
            .iter()
            .filter(|id| !seen.contains(*id) && !is_summary_chunk_id(id))
            .filter_map(|id| {
                let chunk_id = ChunkId::new(id.clone());
                graph.chunks().find(|c| c.id == chunk_id)
            })
            .map(|chunk| {
                let similarity = match (query_embedding, chunk.embedding.as_deref()) {
                    (Some(q), Some(e)) => VectorUtils::cosine_similarity(q, e),
                    _ => 0.0,
                };
                (similarity, chunk)
            })
            .collect();
        if query_embedding.is_some() {
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        for (_, chunk) in candidates.into_iter().take(POOL_CHUNKS_PER_RESULT) {
            if pool.len() >= max {
                return pool;
            }
            if seen.insert(chunk.id.to_string()) {
                pool.push(PoolChunk {
                    id: chunk.id.to_string(),
                    content: chunk.content.clone(),
                });
            }
        }
    }
    pool
}

/// Positions of the summary entries that join the answer context, given the ranking as
/// "is this entry a summary" flags (best first): summaries ranked above the last selected
/// original (the `top_k`-th original, or the last one when there are fewer), in ranking
/// order, at most `cap`. No original in the ranking means no summary either.
#[cfg(feature = "async")]
fn summaries_above_last_original(
    is_summary: impl Iterator<Item = bool>,
    top_k: usize,
    cap: usize,
) -> Vec<usize> {
    let flags: Vec<bool> = is_summary.collect();
    let originals: Vec<usize> = (0..flags.len()).filter(|&p| !flags[p]).collect();
    let Some(&last) = originals.get(top_k.max(1) - 1).or(originals.last()) else {
        return Vec::new();
    };
    (0..last).filter(|&p| flags[p]).take(cap).collect()
}

/// Summary entries of a reranker ranking (pool indexes at or after `original_len` are
/// summaries) that join the context; see [`summaries_above_last_original`].
#[cfg(feature = "async")]
fn pick_summaries(
    ranking: &[(usize, f32)],
    original_len: usize,
    top_k: usize,
    cap: usize,
) -> Vec<(usize, f32)> {
    summaries_above_last_original(
        ranking.iter().map(|(idx, _)| *idx >= original_len),
        top_k,
        cap,
    )
    .into_iter()
    .map(|pos| ranking[pos])
    .collect()
}

/// Summary chunks to rank together with the originals: chunk hits of the search results
/// (by score), then the ids of the fused dense+keyword lists, deduplicated, at most `max`.
#[cfg(feature = "async")]
fn summary_candidates(
    graph: &KnowledgeGraph,
    search_results: &[retrieval::SearchResult],
    fused_ids: &[String],
    max: usize,
) -> Vec<PoolChunk> {
    let mut hits: Vec<&retrieval::SearchResult> = search_results
        .iter()
        .filter(|r| r.result_type == retrieval::ResultType::Chunk && is_summary_chunk_id(&r.id))
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ids = hits.into_iter().map(|r| r.id.as_str()).chain(
        fused_ids
            .iter()
            .map(String::as_str)
            .filter(|id| is_summary_chunk_id(id)),
    );
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for id in ids {
        if out.len() >= max {
            break;
        }
        if !seen.insert(id) {
            continue;
        }
        if let Some(chunk) = graph.get_chunk(&ChunkId::new(id.to_string())) {
            out.push(PoolChunk {
                id: id.to_string(),
                content: chunk.content.clone(),
            });
        }
    }
    out
}

/// Context parts of the summary chunk hits when there is no reranker: the summaries
/// ranked above the last original chunk hit, at most `cap`, formatted like chunk hits.
#[cfg(feature = "async")]
fn summary_hit_parts(search_results: &[retrieval::SearchResult], cap: usize) -> Vec<String> {
    if cap == 0 {
        return Vec::new();
    }
    let mut seen = std::collections::HashSet::new();
    let mut hits: Vec<&retrieval::SearchResult> = search_results
        .iter()
        .filter(|r| r.result_type == retrieval::ResultType::Chunk && seen.insert(r.id.as_str()))
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    summaries_above_last_original(
        hits.iter().map(|r| is_summary_chunk_id(&r.id)),
        usize::MAX,
        cap,
    )
    .into_iter()
    .map(|pos| {
        format!(
            "[Chunk | Relevance: {:.2}]\n{}",
            hits[pos].score, hits[pos].content
        )
    })
    .collect()
}

/// Append chunks (by id, best first) to `pool` after the existing entries, skipping ids
/// already present or unknown to the graph, until the pool holds `max` chunks.
#[cfg(feature = "async")]
fn extend_pool_with_chunk_ids(
    graph: &KnowledgeGraph,
    pool: &mut Vec<PoolChunk>,
    ids: &[String],
    max: usize,
) {
    let mut seen: std::collections::HashSet<String> = pool.iter().map(|c| c.id.clone()).collect();
    for id in ids {
        if pool.len() >= max {
            break;
        }
        if seen.contains(id) || is_summary_chunk_id(id) {
            continue;
        }
        if let Some(chunk) = graph.get_chunk(&ChunkId::new(id.clone())) {
            seen.insert(id.clone());
            pool.push(PoolChunk {
                id: id.clone(),
                content: chunk.content.clone(),
            });
        }
    }
}

/// Rerank `pool` with the configured llama.cpp reranker. Returns `(pool_index, score)` by
/// descending score.
#[cfg(all(feature = "async", feature = "llm"))]
async fn rerank_pool(
    cfg: &crate::config::RerankerConfig,
    query: &str,
    pool: &[PoolChunk],
    top_n: usize,
) -> Result<Vec<(usize, f32)>> {
    if pool.is_empty() {
        return Ok(Vec::new());
    }
    let started = std::time::Instant::now();
    let documents: Vec<String> = pool.iter().map(|c| c.content.clone()).collect();
    let ranking = crate::reranker::LlamaCppReranker::from_config(cfg)
        .rerank(query, documents, Some(top_n))
        .await;
    if ranking.is_ok() {
        tracing::info!(
            "Reranked {} candidate chunks in {} ms (top_k {})",
            pool.len(),
            started.elapsed().as_millis(),
            cfg.top_k
        );
    }
    ranking
}

#[cfg(all(feature = "async", not(feature = "llm")))]
async fn rerank_pool(
    _cfg: &crate::config::RerankerConfig,
    _query: &str,
    _pool: &[PoolChunk],
    _top_n: usize,
) -> Result<Vec<(usize, f32)>> {
    Err(GraphRAGError::Config {
        message: "reranker.enabled requires the `llm` feature".to_string(),
    })
}

/// Output-token budget for the final answer: `llm.max_tokens` (default 800). With
/// thinking enabled, reasoning tokens count toward the limit, so keep at least
/// 4096 or a hosted reasoning model can run out before emitting any `content`.
#[cfg(feature = "async")]
fn answer_max_tokens(llm: &crate::llm::LlmConfig) -> u32 {
    let base = llm.max_tokens.unwrap_or(800);
    if llm.disable_thinking {
        base
    } else {
        base.max(4096)
    }
}

/// Prompt for the final RAG answer (CONTEXT / QUESTION / ANSWER layout).
#[cfg(feature = "async")]
fn build_answer_prompt(context: &str, query: &str) -> String {
    format!(
        "You are a knowledgeable assistant specialized in answering questions based on a knowledge graph.\n\n\
        IMPORTANT INSTRUCTIONS:\n\
        - Answer using only information from the provided context below\n\
        - Answer in the same language as the question\n\
        - Synthesize information from ALL context sections to give a comprehensive answer\n\
        - If the context only partially covers the question, answer with what the context supports and briefly note what is missing\n\
        - Only if the context contains nothing relevant to the question, say so in one short sentence, in the language of the question\n\
        - Provide direct, conversational, and natural responses\n\
        - Do NOT show your reasoning process or use <think> tags\n\
        - Aim for a complete answer (3-6 sentences) that covers different aspects found across the context\n\
        - Use a natural, helpful tone as if speaking to a person\n\n\
        CONTEXT:\n\
        {context}\n\n\
        QUESTION: {query}\n\n\
        ANSWER (direct response only, no reasoning):"
    )
}

/// Prompt for the HyDE hypothetical passage (language-agnostic: mirrors the question).
#[cfg(all(feature = "async", any(feature = "llm", test)))]
fn build_hyde_prompt(query: &str) -> String {
    format!(
        "Write a short passage (3-4 sentences) in the style and language of the indexed text \
        that would answer the question. Write it in the language of the question. \
        Do not say you don't know; write a plausible passage. \
        Output only the passage, with no preamble and no reasoning.\n\n\
        QUESTION: {query}\n\nPASSAGE:"
    )
}

/// Remove reasoning blocks and surrounding whitespace from a raw HyDE completion.
#[cfg(feature = "async")]
fn clean_hyde_passage(raw: &str) -> String {
    GraphRAG::remove_thinking_tags(raw)
}

/// `"{query}\n{passage}"`, or the plain query when the passage is absent or blank.
#[cfg(feature = "async")]
fn combine_retrieval_query(query: &str, passage: Option<&str>) -> String {
    match passage.map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => format!("{query}\n{p}"),
        None => query.to_string(),
    }
}

/// First 120 chars of the passage, for logs.
#[cfg(feature = "async")]
fn hyde_log_preview(passage: &str) -> String {
    passage.chars().take(120).collect()
}

/// Sum two optional usages (HyDE call + answer call).
#[cfg(feature = "async")]
fn sum_usage(a: Option<TokenUsage>, b: Option<TokenUsage>) -> Option<TokenUsage> {
    match (a, b) {
        (Some(a), Some(b)) => Some(TokenUsage {
            prompt_tokens: a.prompt_tokens + b.prompt_tokens,
            completion_tokens: a.completion_tokens + b.completion_tokens,
            total_tokens: a.total_tokens + b.total_tokens,
        }),
        (a, b) => a.or(b),
    }
}

/// Pick the top `top_k` reranked pool entries, or nothing (with a warning) when the
/// reranker failed so the caller keeps the default ordering.
#[cfg(feature = "async")]
fn select_reranked(
    pool: &[PoolChunk],
    ranking: Result<Vec<(usize, f32)>>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    match ranking {
        Ok(ranked) => ranked
            .into_iter()
            .filter(|(idx, _)| *idx < pool.len())
            .take(top_k.max(1))
            .collect(),
        Err(e) => {
            tracing::warn!("Reranker failed, using default context order: {e}");
            Vec::new()
        },
    }
}

/// Fuse the rankings of the original query and of the HyDE query by rank, not by score
/// (the reranker scores of the longer HyDE query are systematically higher, so the two
/// scales are not comparable): both lists are sorted by descending score, then taken
/// alternately (original first, then HyDE, ...) skipping indices already taken, until both
/// are exhausted. Each item keeps the score of the list it came from; the flag is true when
/// it came from the HyDE list. A failed HyDE ranking falls back to the original one
/// (warning); a failed original ranking is returned as the error.
#[cfg(feature = "async")]
fn fuse_rankings(
    orig: Result<Vec<(usize, f32)>>,
    hyde: Result<Vec<(usize, f32)>>,
) -> Result<Vec<(usize, f32, bool)>> {
    let mut orig = orig?;
    let mut hyde = match hyde {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("HyDE rerank failed, using the original ranking: {e}");
            Vec::new()
        },
    };
    orig.sort_by(|a, b| b.1.total_cmp(&a.1));
    hyde.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut seen = std::collections::HashSet::new();
    let mut fused = Vec::with_capacity(orig.len() + hyde.len());
    let (mut o, mut h) = (orig.into_iter(), hyde.into_iter());
    loop {
        let mut progressed = false;
        for (iter, from_hyde) in [(&mut o, false), (&mut h, true)] {
            for (idx, score) in iter.by_ref() {
                progressed = true;
                if seen.insert(idx) {
                    fused.push((idx, score, from_hyde));
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    Ok(fused)
}

/// Keep context parts (already sorted by relevance) while the cumulative estimated
/// tokens fit `max_tokens`. The first part is always kept, truncated on a char
/// boundary when it alone exceeds the budget.
#[cfg(feature = "async")]
fn fit_context_to_budget(parts: Vec<String>, max_tokens: u32) -> Vec<String> {
    let budget = max_tokens as usize;
    let mut used = 0usize;
    let mut kept = Vec::with_capacity(parts.len());
    for part in parts {
        let cost = estimate_tokens(&part);
        if kept.is_empty() && cost > budget {
            // Always keep the best part, cut to the budget (chars, never mid-codepoint).
            kept.push(part.chars().take(budget.saturating_mul(3)).collect());
            used = budget;
        } else if used + cost <= budget {
            used += cost;
            kept.push(part);
        } else {
            break;
        }
    }
    kept
}

/// Max seed entities for the graph-facts context part.
#[cfg(feature = "async")]
const GRAPH_FACTS_MAX_SEEDS: usize = 8;
/// Max lines of the graph-facts context part.
#[cfg(feature = "async")]
const GRAPH_FACTS_MAX_LINES: usize = 60;
/// Max candidate facts sent to the reranker (cheap pre-filter above this).
#[cfg(feature = "async")]
const GRAPH_FACTS_MAX_CANDIDATES: usize = 1000;
/// Token budget reserved for the graph-facts part (capped at a third of the context).
#[cfg(feature = "async")]
const GRAPH_FACTS_RESERVE_TOKENS: u32 = 2000;
/// Max chars of a relationship description in a graph-facts line.
#[cfg(feature = "async")]
const GRAPH_FACTS_MAX_DESC_CHARS: usize = 150;
/// Title of the graph-facts context part.
#[cfg(feature = "async")]
const GRAPH_FACTS_TITLE: &str = "Fatti dal grafo (relazioni tra entità):";

/// One relationship ready to be printed as a graph-facts line.
#[cfg(feature = "async")]
#[derive(Debug, Clone)]
struct GraphFact {
    source: String,
    relation: String,
    target: String,
    description: String,
    /// Both endpoints are seed entities.
    both_seeds: bool,
    /// Name of the non-seed endpoint (empty when both are seeds).
    other: String,
}

/// True when `name` occurs in `query` as a whole word (case-insensitive; the characters
/// around the match must not be alphanumeric). Names under 3 chars never match.
#[cfg(feature = "async")]
fn name_in_query(name: &str, query: &str) -> bool {
    let name = name.trim().to_lowercase();
    if name.chars().count() < 3 {
        return false;
    }
    let query = query.to_lowercase();
    query.match_indices(&name).any(|(start, m)| {
        let before = query[..start].chars().next_back();
        let after = query[start + m.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

/// The rendered line of a fact: `- Source —[REL_TYPE]→ Target: description` (description
/// cut to 150 chars, omitted with its colon when empty).
#[cfg(feature = "async")]
fn fact_line(f: &GraphFact) -> String {
    let mut line = format!("- {} —[{}]→ {}", f.source, f.relation, f.target);
    let desc = f.description.trim();
    if !desc.is_empty() {
        line.push_str(": ");
        line.extend(desc.chars().take(GRAPH_FACTS_MAX_DESC_CHARS));
    }
    line
}

/// Lowercase query words of at least 4 chars.
#[cfg(feature = "async")]
fn query_words(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
        .map(str::to_string)
        .collect()
}

/// Cheap relevance of a fact (pre-filter and reranker fallback): +2 when both endpoints
/// are seeds, +1 when the non-seed endpoint's name occurs in the kept chunk context
/// (`context_lower`, already lowercase), +1 per query word occurring in the line.
#[cfg(feature = "async")]
fn cheap_fact_score(f: &GraphFact, context_lower: &str, words: &[String]) -> u32 {
    let mut score = 0;
    if f.both_seeds {
        score += 2;
    }
    let other = f.other.trim().to_lowercase();
    if !other.is_empty() && context_lower.contains(&other) {
        score += 1;
    }
    let line = fact_line(f).to_lowercase();
    score + words.iter().filter(|w| line.contains(w.as_str())).count() as u32
}

/// Keep facts by round-robin over neighbours. In the given (rank) order, identical triples
/// and self-loops are dropped and facts are grouped by neighbour (lowercase `other`, or
/// lowercase `source|target` when both endpoints are seeds, so those never share a bucket).
/// The output is the best fact of every neighbour (in rank order), then the second best of
/// every neighbour, and so on, at most `max_lines` in total: a hub with hundreds of
/// relationships cannot crowd out its many different neighbours.
#[cfg(feature = "async")]
fn select_facts(ranked: Vec<GraphFact>, max_lines: usize) -> Vec<GraphFact> {
    let mut seen = std::collections::HashSet::new();
    let mut bucket_of: std::collections::HashMap<String, usize> = Default::default();
    let mut buckets: Vec<Vec<GraphFact>> = Vec::new();
    for f in ranked {
        if f.source == f.target
            || !seen.insert((f.source.clone(), f.relation.clone(), f.target.clone()))
        {
            continue;
        }
        let key = if f.other.trim().is_empty() {
            format!("{}|{}", f.source, f.target).to_lowercase()
        } else {
            f.other.to_lowercase()
        };
        let idx = *bucket_of.entry(key).or_insert_with(|| {
            buckets.push(Vec::new());
            buckets.len() - 1
        });
        buckets[idx].push(f);
    }
    let mut iters: Vec<_> = buckets.into_iter().map(Vec::into_iter).collect();
    let mut out = Vec::new();
    while out.len() < max_lines {
        let before = out.len();
        for it in iters.iter_mut() {
            if out.len() >= max_lines {
                break;
            }
            out.extend(it.next());
        }
        if out.len() == before {
            break;
        }
    }
    out
}

/// Render the part (title + lines in the given order), dropping tail lines until the
/// estimated tokens fit `max_tokens`. `None` when no line fits.
#[cfg(feature = "async")]
fn render_graph_facts(facts: &[GraphFact], max_tokens: usize) -> Option<String> {
    let mut lines: Vec<String> = facts.iter().map(fact_line).collect();
    while !lines.is_empty() {
        let text = format!("{GRAPH_FACTS_TITLE}\n{}", lines.join("\n"));
        if estimate_tokens(&text) <= max_tokens {
            return Some(text);
        }
        lines.pop();
    }
    None
}

/// Candidate facts (`retrieval.graph_facts`): every relationship (both directions) touching
/// a seed entity, deduplicated. Seeds, capped at 8: the entity-type search results in rank
/// order, then the entities whose name appears in the query.
#[cfg(feature = "async")]
fn collect_graph_facts(
    graph: &KnowledgeGraph,
    query: &str,
    search_results: &[retrieval::SearchResult],
) -> Vec<GraphFact> {
    let mut seeds: Vec<&Entity> = Vec::new();
    let mut seed_idx: std::collections::HashMap<String, usize> = Default::default();
    for r in search_results
        .iter()
        .filter(|r| r.result_type == retrieval::ResultType::Entity)
    {
        if let Some(e) = graph.get_entity_by_id(&r.id) {
            if seeds.len() < GRAPH_FACTS_MAX_SEEDS && !seed_idx.contains_key(&e.id.to_string()) {
                seed_idx.insert(e.id.to_string(), seeds.len());
                seeds.push(e);
            }
        }
    }
    for e in graph.entities() {
        if seeds.len() >= GRAPH_FACTS_MAX_SEEDS {
            break;
        }
        if name_in_query(&e.name, query) && !seed_idx.contains_key(&e.id.to_string()) {
            seed_idx.insert(e.id.to_string(), seeds.len());
            seeds.push(e);
        }
    }
    if seeds.is_empty() {
        return Vec::new();
    }
    let name_of = |id: &EntityId| {
        graph
            .get_entity(id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| id.to_string())
    };
    let mut seen = std::collections::HashSet::new();
    let mut facts = Vec::new();
    for r in graph.relationships() {
        let s = seed_idx.get(&r.source.to_string());
        let t = seed_idx.get(&r.target.to_string());
        if s.is_none() && t.is_none() {
            continue;
        }
        let fact = GraphFact {
            source: name_of(&r.source),
            relation: r.relation_type.clone(),
            target: name_of(&r.target),
            description: r.description.clone().unwrap_or_default(),
            both_seeds: s.is_some() && t.is_some(),
            other: match (s, t) {
                (Some(_), None) => name_of(&r.target),
                (None, Some(_)) => name_of(&r.source),
                _ => String::new(),
            },
        };
        if seen.insert((
            fact.source.clone(),
            fact.relation.clone(),
            fact.target.clone(),
        )) {
            facts.push(fact);
        }
    }
    facts
}

/// Build the graph-facts context part, ranked by relevance to the query: candidates above
/// 1000 are first cut by the cheap score, then ranked with the configured cross-encoder
/// reranker (lines as documents; the reranker client splits them in batches and merges the scores into one global ranking). Without reranker, or if it fails (warning),
/// the cheap score decides. Never fails: worst case the facts are less relevant.
#[cfg(feature = "async")]
async fn build_graph_facts_part(
    graph: &KnowledgeGraph,
    query: &str,
    search_results: &[retrieval::SearchResult],
    kept_chunks: &[String],
    rerank_cfg: &crate::config::RerankerConfig,
    max_tokens: usize,
) -> Option<String> {
    let mut facts = collect_graph_facts(graph, query, search_results);
    if facts.is_empty() {
        return None;
    }
    let context_lower = kept_chunks.join("\n").to_lowercase();
    let words = query_words(query);
    let mut scored: Vec<(u32, GraphFact)> = facts
        .drain(..)
        .map(|f| (cheap_fact_score(&f, &context_lower, &words), f))
        .collect();
    // Stable sort: ties keep the graph order.
    scored.sort_by_key(|(s, _)| std::cmp::Reverse(*s));
    scored.truncate(GRAPH_FACTS_MAX_CANDIDATES);
    let mut ranked: Vec<GraphFact> = scored.into_iter().map(|(_, f)| f).collect();
    if rerank_cfg.enabled {
        let pool: Vec<PoolChunk> = ranked
            .iter()
            .map(|f| PoolChunk {
                id: String::new(),
                content: fact_line(f),
            })
            .collect();
        match rerank_pool(rerank_cfg, query, &pool, pool.len()).await {
            Ok(order) => {
                let mut by_rank: Vec<GraphFact> = Vec::with_capacity(ranked.len());
                let mut used = std::collections::HashSet::new();
                for (idx, _) in order {
                    if idx < ranked.len() && used.insert(idx) {
                        by_rank.push(ranked[idx].clone());
                    }
                }
                // Anything the reranker did not return keeps its cheap order, after.
                for (idx, f) in ranked.iter().enumerate() {
                    if !used.contains(&idx) {
                        by_rank.push(f.clone());
                    }
                }
                ranked = by_rank;
            },
            Err(e) => tracing::warn!("Graph facts reranking failed, using cheap score: {e}"),
        }
    }
    let selected = select_facts(ranked, GRAPH_FACTS_MAX_LINES);
    render_graph_facts(&selected, max_tokens)
}

/// Token reserve for the map-reduce notes part, capped at a quarter of the context budget.
#[cfg(feature = "async")]
const MAP_REDUCE_RESERVE_TOKENS: u32 = 1500;

/// Title of the map-reduce notes part (not bracketed: never parsed as a chunk part).
#[cfg(feature = "async")]
const MAP_REDUCE_TITLE: &str = "Fatti estratti dai passaggi:";

/// Map pool in priority order: the reranker ranking first (when it succeeded), then the
/// pool entries it did not rank, in pool order. Distinct chunk ids, at most `cap`.
#[cfg(feature = "async")]
fn order_map_pool(
    pool: &[PoolChunk],
    ranking: Option<&[(usize, f32)]>,
    cap: usize,
) -> Vec<PoolChunk> {
    let ranked = ranking.unwrap_or(&[]).iter().map(|(i, _)| *i);
    let mut seen_idx = std::collections::HashSet::new();
    let mut seen_id = std::collections::HashSet::new();
    let mut out = Vec::new();
    for idx in ranked.chain(0..pool.len()) {
        if out.len() >= cap {
            break;
        }
        if idx >= pool.len() || !seen_idx.insert(idx) {
            continue;
        }
        if seen_id.insert(pool[idx].id.clone()) {
            out.push(pool[idx].clone());
        }
    }
    out
}

/// Map prompt: static instructions and the question first (shared by the N calls of one
/// question, so the server can reuse its KV cache), the passage last.
#[cfg(feature = "async")]
fn build_map_prompt(query: &str, chunk: &str) -> String {
    format!(
        "Extract from the passage below only the facts that help answer the question.\n\
        - Write short bullet lines, one fact per line.\n\
        - Quote names exactly as they appear in the passage.\n\
        - Use only what the passage states: no outside knowledge, no inference.\n\
        - Answer in the language of the passage.\n\
        - If the passage has nothing relevant to the question, reply with exactly NONE.\n\
        - Output only the bullet lines or NONE, with no preamble and no reasoning.\n\n\
        QUESTION: {query}\n\n\
        PASSAGE:\n{chunk}"
    )
}

/// Note from a raw map reply: `None` for an empty reply or `NONE` (any case, with trailing
/// punctuation, quotes or markdown emphasis around it).
#[cfg(feature = "async")]
fn parse_map_note(raw: &str) -> Option<String> {
    let text = raw.trim();
    let core = text.trim_matches(|c: char| c.is_whitespace() || "*_\"'`.!:;,()[]".contains(c));
    if core.is_empty() || core.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(text.to_string())
    }
}

/// One map call per pool chunk, `limit` at a time, notes in pool order as
/// `(chunk_id, note)`. Failed calls are skipped (warning); usage of all calls is summed.
#[cfg(feature = "async")]
async fn run_map_calls(
    llm: &crate::llm::LlmConfig,
    query: &str,
    pool: &[PoolChunk],
    limit: usize,
) -> (Vec<(String, String)>, Option<TokenUsage>) {
    let client = crate::llm::LlmClient::new(llm.clone());
    let params = crate::llm::LlmGenerationParams {
        // Reasoning models need room for the thinking block before the notes.
        max_tokens: Some(if llm.disable_thinking { 200 } else { 4096 }),
        temperature: Some(0.0),
        ..Default::default()
    };
    let results = crate::entity::llm_extractor::run_ordered_concurrent(
        pool,
        limit,
        |chunk| {
            let client = &client;
            let params = params.clone();
            async move {
                client
                    .generate_with_full_response(&build_map_prompt(query, &chunk.content), params)
                    .await
            }
        },
        |_| {},
    )
    .await;
    let mut notes = Vec::new();
    let mut usage = None;
    for (chunk, result) in pool.iter().zip(results) {
        match result {
            Ok(response) => {
                let call_usage = response.usage.map(|u| TokenUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                });
                usage = sum_usage(usage, call_usage);
                if let Some(note) = parse_map_note(&GraphRAG::remove_thinking_tags(&response.text))
                {
                    notes.push((chunk.id.clone(), note));
                }
            },
            Err(e) => tracing::warn!("Map-reduce call failed for chunk {}: {e}", chunk.id),
        }
    }
    tracing::info!(
        "Map-reduce: {} notes from {} chunks",
        notes.len(),
        pool.len()
    );
    (notes, usage)
}

/// The notes part (title + one `[chunk id] note` entry per note, in the given order),
/// dropping tail notes until the estimated tokens fit `max_tokens`. `None` when no note fits.
#[cfg(feature = "async")]
fn build_notes_part(notes: &[(String, String)], max_tokens: usize) -> Option<String> {
    let mut entries: Vec<String> = notes
        .iter()
        .map(|(id, note)| format!("[{id}]\n{note}"))
        .collect();
    while !entries.is_empty() {
        let text = format!("{MAP_REDUCE_TITLE}\n{}", entries.join("\n"));
        if estimate_tokens(&text) <= max_tokens {
            return Some(text);
        }
        entries.pop();
    }
    None
}

/// Debug aid: when `GRAPHRAG_DUMP_CONTEXT` names a file, append one JSON line
/// `{query, context, prompt}` per answer. Write errors are logged, never returned.
#[cfg(feature = "async")]
fn dump_answer_context(query: &str, context: &str, prompt: &str) {
    use std::io::Write;
    let Some(path) = std::env::var_os("GRAPHRAG_DUMP_CONTEXT") else {
        return;
    };
    let line = serde_json::json!({ "query": query, "context": context, "prompt": prompt });
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = result {
        tracing::warn!("GRAPHRAG_DUMP_CONTEXT write failed: {e}");
    }
}

#[cfg(all(test, feature = "async"))]
mod graph_facts_tests {
    use super::*;

    fn fact(s: &str, r: &str, t: &str, d: &str) -> GraphFact {
        GraphFact {
            source: s.into(),
            relation: r.into(),
            target: t.into(),
            description: d.into(),
            both_seeds: false,
            other: t.into(),
        }
    }

    #[test]
    fn line_format_with_and_without_description() {
        let facts = [
            fact("Geppetto", "PARENT_OF", "Pinocchio", ""),
            fact("Fata", "HELPS", "Pinocchio", "lo salva"),
        ];
        assert_eq!(
            render_graph_facts(&facts, 1000).unwrap(),
            "Fatti dal grafo (relazioni tra entità):\n\
             - Geppetto —[PARENT_OF]→ Pinocchio\n\
             - Fata —[HELPS]→ Pinocchio: lo salva"
        );
        assert!(render_graph_facts(&[], 1000).is_none());
    }

    #[test]
    fn said_marker_is_shown_in_rendered_facts() {
        let facts = [fact(
            "Pinocchio",
            "LOCATED_IN",
            "Paese dei Balocchi",
            "[detto da Pinocchio] il Gatto gli ha mangiato i piedi",
        )];
        let text = render_graph_facts(&facts, 1000).unwrap();
        assert!(text.contains(
            "- Pinocchio —[LOCATED_IN]→ Paese dei Balocchi: [detto da Pinocchio] il Gatto"
        ));
    }

    #[test]
    fn description_is_truncated() {
        let f = fact("C", "R", "D", &"x".repeat(400));
        assert_eq!(fact_line(&f), format!("- C —[R]→ D: {}", "x".repeat(150)));
    }

    #[test]
    fn cheap_score_components() {
        let mut f = fact("Fata", "HELPS", "Lumaca", "porta aiuto");
        let words = query_words("Chi ha aiutato la Fata? aiuto");
        assert_eq!(words, vec!["aiutato", "fata", "aiuto"]);
        // query words: "fata" and "aiuto" occur in the line; context lacks "lumaca"
        assert_eq!(cheap_fact_score(&f, "niente", &words), 2);
        // other endpoint present in the kept chunks: +1
        assert_eq!(cheap_fact_score(&f, "la lumaca arriva", &words), 3);
        // both endpoints seeds: +2
        f.both_seeds = true;
        f.other = String::new();
        assert_eq!(cheap_fact_score(&f, "niente", &words), 4);
    }

    #[test]
    fn select_round_robin_dedups_and_caps() {
        let mut v = vec![fact("A", "R", "B", ""), fact("A", "R", "B", "dup")];
        v.push(fact("A", "R", "A", "")); // self-loop
        for i in 0..3 {
            v.push(fact("Hub", "R", &format!("T{i}"), ""));
        }
        // second fact of neighbour T0 (same `other`, different relation)
        v.push(fact("Hub", "S", "T0", ""));
        let out = select_facts(v.clone(), 60);
        let order: Vec<String> = out
            .iter()
            .map(|f| format!("{}-{}", f.relation, f.target))
            .collect();
        // dup triple and self-loop gone; best of each neighbour first, T0's second last
        assert_eq!(order, ["R-B", "R-T0", "R-T1", "R-T2", "S-T0"]);
        assert_eq!(select_facts(v, 2).len(), 2);
    }

    #[test]
    fn select_keeps_both_seed_facts_in_separate_buckets() {
        let both = |s: &str, t: &str| {
            let mut f = fact(s, "R", t, "");
            f.both_seeds = true;
            f.other = String::new();
            f
        };
        let v = vec![both("A", "B"), both("A", "C"), fact("A", "R", "Z", "")];
        let out = select_facts(v, 3);
        assert_eq!(out.len(), 3);
        // one bucket each: rank order preserved, nothing deferred to a second round
        assert_eq!(
            out.iter().map(|f| f.target.as_str()).collect::<Vec<_>>(),
            ["B", "C", "Z"]
        );
    }

    #[test]
    fn render_drops_tail_lines_to_fit_budget() {
        let facts: Vec<GraphFact> = (0..10)
            .map(|i| fact("Sorgente", "REL", &format!("Target{i}"), ""))
            .collect();
        let full = render_graph_facts(&facts, 10_000).unwrap();
        assert_eq!(full.lines().count(), 11);
        let budget = estimate_tokens(&full) - 5;
        let trimmed = render_graph_facts(&facts, budget).unwrap();
        assert!(estimate_tokens(&trimmed) <= budget);
        assert!(trimmed.lines().count() < 11);
        assert!(trimmed.contains("Target0"));
        assert!(render_graph_facts(&facts, 3).is_none());
    }

    #[test]
    fn name_match_is_whole_word_case_insensitive() {
        assert!(name_in_query("Pinocchio", "Chi ha aiutato pinocchio?"));
        assert!(!name_in_query("Pino", "chi è Pinocchio"));
        assert!(!name_in_query("Al", "al mercato"));
    }
}

#[cfg(all(test, feature = "async"))]
mod budget_tests {
    use super::*;

    fn part(n: usize) -> String {
        "è".repeat(n) // n chars, 2 bytes each: 3 chars ~ 1 token
    }

    #[test]
    fn estimate_is_chars_over_three_rounded_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("àèì"), 1);
        assert_eq!(estimate_tokens("àèìò"), 2);
    }

    #[test]
    fn under_budget_keeps_all() {
        let parts = vec![part(30), part(30)];
        assert_eq!(fit_context_to_budget(parts.clone(), 100), parts);
    }

    #[test]
    fn over_budget_drops_tail() {
        let parts = vec![part(30), part(30), part(30)]; // 10 tokens each
        let out = fit_context_to_budget(parts.clone(), 25);
        assert_eq!(out, parts[..2].to_vec());
    }

    #[test]
    fn oversized_first_part_is_truncated_on_char_boundary() {
        let parts = vec![part(300), part(3)];
        let out = fit_context_to_budget(parts, 10);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].chars().count(), 30);
        assert!(out[0].chars().all(|c| c == 'è'));
    }

    #[test]
    fn empty_input_and_zero_budget() {
        assert!(fit_context_to_budget(vec![], 10).is_empty());
        let out = fit_context_to_budget(vec![part(9)], 0);
        assert_eq!(out.len(), 1);
    }
}

#[cfg(all(test, feature = "async"))]
mod context_tests {
    use super::*;
    use crate::core::DocumentId;
    use crate::retrieval::{ResultType, SearchResult};

    fn graph_with_chunks(n: usize, embeddings: bool) -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();
        for i in 0..n {
            let mut chunk = TextChunk::new(
                ChunkId::new(format!("c{i}")),
                DocumentId::new("d".to_string()),
                format!("chunk-{i} {}", "x".repeat(300)),
                0,
                10,
            );
            if embeddings {
                // chunk i points further from [1, 0] as i grows; the last is the closest
                let angle = (n - 1 - i) as f32 * 0.1;
                chunk.embedding = Some(vec![angle.cos(), angle.sin()]);
            }
            graph.add_chunk(chunk).unwrap();
        }
        graph
    }

    fn entity_result(chunks: &[&str]) -> SearchResult {
        SearchResult {
            id: "person_pinocchio".to_string(),
            content: "Pinocchio (score: 27.00)".to_string(),
            score: 27.0,
            result_type: ResultType::Entity,
            entities: vec!["Pinocchio".to_string()],
            source_chunks: chunks.iter().map(|c| c.to_string()).collect(),
        }
    }

    fn chunk_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: format!("direct hit {id}"),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: vec![id.to_string()],
        }
    }

    #[test]
    fn identical_text_enters_the_context_once() {
        let graph = graph_with_chunks(2, false);
        let mut dup = chunk_result("c0", 0.9);
        dup.content = "stesso testo".to_string();
        let mut other_type = chunk_result("hybrid_x", 0.8);
        other_type.content = "stesso testo ".to_string();
        other_type.result_type = ResultType::Hybrid;
        let parts = build_context_parts(&graph, None, &[dup, other_type]);
        assert_eq!(
            parts.iter().filter(|p| p.contains("stesso testo")).count(),
            1
        );
    }

    #[test]
    fn entity_expansion_skips_text_already_in_a_chunk_hit() {
        let graph = graph_with_chunks(2, false);
        let mut hit = chunk_result("other", 0.9);
        hit.content = format!("chunk-0 {}", "x".repeat(300));
        let parts = build_context_parts(&graph, None, &[hit, entity_result(&["c0"])]);
        assert_eq!(parts.iter().filter(|p| p.contains("chunk-0")).count(), 1);
    }

    #[test]
    fn entity_expansion_is_capped_at_two_chunks() {
        let graph = graph_with_chunks(6, false);
        let parts = build_context_parts(
            &graph,
            None,
            &[entity_result(&["c0", "c1", "c2", "c3", "c4", "c5"])],
        );
        assert_eq!(parts.len(), 2);
        // Without embeddings the original order is kept.
        assert!(parts[0].contains("chunk-0"));
        assert!(parts[1].contains("chunk-1"));
    }

    #[test]
    fn entity_chunks_are_ranked_by_query_similarity() {
        let graph = graph_with_chunks(6, true);
        let parts = build_context_parts(
            &graph,
            Some(&[1.0, 0.0]),
            &[entity_result(&["c0", "c1", "c2", "c3", "c4", "c5"])],
        );
        assert_eq!(parts.len(), 2);
        assert!(parts[0].contains("chunk-5"), "{}", parts[0]);
        assert!(parts[1].contains("chunk-4"), "{}", parts[1]);
    }

    #[test]
    fn direct_chunk_hits_stay_within_budget_despite_dominant_entity() {
        let graph = graph_with_chunks(6, false);
        let results = vec![
            entity_result(&["c0", "c1", "c2", "c3", "c4", "c5"]),
            chunk_result("hit1", 0.2),
            chunk_result("hit2", 0.15),
        ];
        let parts = build_context_parts(&graph, None, &results);
        // Direct hits come first even though the entity score is far higher.
        assert!(parts[0].contains("direct hit hit1"));
        assert!(parts[1].contains("direct hit hit2"));
        // A tight budget (~ two direct hits) still keeps both of them.
        let kept = fit_context_to_budget(parts, 40);
        assert!(kept.len() >= 2);
        assert!(kept[0].contains("hit1") && kept[1].contains("hit2"));
    }

    #[test]
    fn chunk_shared_by_direct_hit_and_entity_is_not_duplicated() {
        let graph = graph_with_chunks(3, false);
        let mut hit = chunk_result("c0", 0.2);
        hit.content = "direct hit c0".to_string();
        let parts = build_context_parts(&graph, None, &[entity_result(&["c0", "c1", "c2"]), hit]);
        let c0 = parts.iter().filter(|p| p.contains("direct hit c0")).count();
        assert_eq!(c0, 1);
        assert!(!parts.iter().any(|p| p.contains("chunk-0 ")));
        assert_eq!(parts.len(), 3); // direct c0 + entity c1 + c2
    }
}

#[cfg(all(test, feature = "async"))]
mod rerank_tests {
    use super::*;
    use crate::core::DocumentId;
    use crate::retrieval::{ResultType, SearchResult};

    fn graph(n: usize) -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        for i in 0..n {
            g.add_chunk(TextChunk::new(
                ChunkId::new(format!("c{i}")),
                DocumentId::new("d".to_string()),
                format!("chunk-{i}"),
                0,
                10,
            ))
            .unwrap();
        }
        g
    }

    fn chunk_hit(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: format!("hit {id}"),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: vec![id.to_string()],
        }
    }

    fn entity(chunks: &[&str]) -> SearchResult {
        SearchResult {
            id: "e".to_string(),
            content: "E (score: 1.00)".to_string(),
            score: 1.0,
            result_type: ResultType::Entity,
            entities: vec!["E".to_string()],
            source_chunks: chunks.iter().map(|c| c.to_string()).collect(),
        }
    }

    #[test]
    fn pool_is_deduped_ordered_and_capped() {
        let g = graph(8);
        let results = vec![
            entity(&["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]),
            chunk_hit("c1", 0.1),
            chunk_hit("c5", 0.9),
            chunk_hit("c1", 0.1), // duplicate hit
        ];
        let pool = build_candidate_pool(&g, None, &results, 100);
        let ids: Vec<&str> = pool.iter().map(|c| c.id.as_str()).collect();
        // direct hits by score first, then entity chunks (cap 5 per entity, minus seen)
        assert_eq!(ids, ["c5", "c1", "c0", "c2", "c3", "c4", "c6"][..ids.len()]);
        assert_eq!(ids.len(), 2 + POOL_CHUNKS_PER_RESULT);
        // direct hit content comes from the graph chunk
        assert_eq!(pool[0].content, "chunk-5");
        let capped = build_candidate_pool(&g, None, &results, 3);
        assert_eq!(capped.len(), 3);
    }

    #[test]
    fn pool_extension_keeps_order_dedups_skips_unknown_and_caps() {
        let g = graph(8);
        let results = vec![chunk_hit("c3", 0.9), chunk_hit("c1", 0.5)];
        let mut pool = build_candidate_pool(&g, None, &results, 30);
        let ids: Vec<String> = ["c1", "zzz", "c7", "c3", "c0", "c5"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        extend_pool_with_chunk_ids(&g, &mut pool, &ids, 5);
        let got: Vec<&str> = pool.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(got, ["c3", "c1", "c7", "c0", "c5"]);
        assert_eq!(pool[2].content, "chunk-7");
        extend_pool_with_chunk_ids(&g, &mut pool, &ids, 5);
        assert_eq!(pool.len(), 5);
    }

    #[test]
    fn reranked_chunks_lead_and_are_not_repeated() {
        let g = graph(4);
        let results = vec![entity(&["c0", "c1", "c2", "c3"]), chunk_hit("c0", 0.5)];
        let pool = build_candidate_pool(&g, None, &results, 30);
        let sel = select_reranked(&pool, Ok(vec![(3, 5.0), (1, 2.0), (0, -1.0)]), 2);
        assert_eq!(sel, vec![(3, 5.0), (1, 2.0)]);
        let exclude: std::collections::HashSet<String> =
            sel.iter().map(|(i, _)| pool[*i].id.clone()).collect();
        let rest = build_context_parts_excluding(&g, None, &results, &exclude);
        assert!(rest
            .iter()
            .all(|p| !p.contains("chunk-3") && !p.contains("chunk-1")));
        assert!(rest.iter().any(|p| p.contains("hit c0")));
    }

    #[test]
    fn reranker_error_falls_back_to_default_order() {
        let g = graph(3);
        let results = vec![chunk_hit("c0", 0.5)];
        let pool = build_candidate_pool(&g, None, &results, 30);
        let err = Err(GraphRAGError::Retrieval {
            message: "boom".to_string(),
        });
        assert!(select_reranked(&pool, err, 8).is_empty());
        // nothing excluded -> identical to the default context
        let default = build_context_parts(&g, None, &results);
        let fallback = build_context_parts_excluding(&g, None, &results, &Default::default());
        assert_eq!(default, fallback);
    }

    #[test]
    fn out_of_range_indices_are_ignored() {
        let pool = vec![PoolChunk {
            id: "c0".into(),
            content: "x".into(),
        }];
        assert_eq!(
            select_reranked(&pool, Ok(vec![(7, 1.0), (0, 0.5)]), 8),
            vec![(0, 0.5)]
        );
    }
}

#[cfg(all(test, feature = "async"))]
mod answer_prompt_tests {
    use super::*;

    #[test]
    fn answer_max_tokens_follows_config_and_gives_reasoning_headroom() {
        let mut llm = crate::llm::LlmConfig {
            disable_thinking: true,
            max_tokens: None,
            ..Default::default()
        };
        assert_eq!(answer_max_tokens(&llm), 800);
        llm.max_tokens = Some(1200);
        assert_eq!(answer_max_tokens(&llm), 1200);

        llm.disable_thinking = false;
        llm.max_tokens = Some(800);
        assert_eq!(answer_max_tokens(&llm), 4096);
        llm.max_tokens = None;
        assert_eq!(answer_max_tokens(&llm), 4096);
        llm.max_tokens = Some(8000);
        assert_eq!(answer_max_tokens(&llm), 8000);
    }

    #[test]
    fn answer_prompt_has_language_and_partial_rules_without_canned_refusal() {
        let prompt = build_answer_prompt("CTX-BODY", "Chi è Geppetto?");
        assert!(!prompt.contains("I don't have enough information"));
        assert!(prompt.contains("same language as the question"));
        assert!(prompt.contains("only partially"));
        assert!(prompt.contains("nothing relevant"));
        assert!(prompt.contains("<think>"));
        let ctx = prompt.find("CONTEXT:\nCTX-BODY").unwrap();
        let q = prompt.find("QUESTION: Chi è Geppetto?").unwrap();
        let a = prompt.find("ANSWER").unwrap();
        assert!(ctx < q && q < a);
    }
}

#[cfg(all(test, feature = "async"))]
mod hyde_tests {
    use super::fuse_rankings;
    use crate::core::GraphRAGError;

    #[test]
    fn fuse_interleaves_original_first_and_dedups() {
        // orig order: 0,1,2 ; hyde order: 2,1,3
        let f = fuse_rankings(
            Ok(vec![(1, 1.0), (0, 3.0), (2, -2.0)]),
            Ok(vec![(3, 0.5), (2, 4.0), (1, 2.0)]),
        )
        .unwrap();
        assert_eq!(
            f,
            vec![
                (0, 3.0, false),
                (2, 4.0, true),
                (1, 1.0, false),
                (3, 0.5, true)
            ]
        );
    }

    #[test]
    fn fuse_handles_unequal_lengths() {
        let f = fuse_rankings(Ok(vec![(0, 1.0)]), Ok(vec![(5, 2.0), (6, 1.0), (7, 0.0)])).unwrap();
        assert_eq!(
            f,
            vec![
                (0, 1.0, false),
                (5, 2.0, true),
                (6, 1.0, true),
                (7, 0.0, true)
            ]
        );
        let f = fuse_rankings(Ok(vec![(0, 2.0), (1, 1.0), (2, 0.0)]), Ok(vec![(0, 9.0)])).unwrap();
        assert_eq!(f, vec![(0, 2.0, false), (1, 1.0, false), (2, 0.0, false)]);
    }

    #[test]
    fn fuse_falls_back_to_original_on_hyde_error() {
        let err = Err(GraphRAGError::Config {
            message: "x".into(),
        });
        let f = fuse_rankings(Ok(vec![(1, 2.0), (0, 1.0)]), err).unwrap();
        assert_eq!(f, vec![(1, 2.0, false), (0, 1.0, false)]);
    }

    #[test]
    fn fuse_propagates_original_error() {
        let err = Err(GraphRAGError::Config {
            message: "x".into(),
        });
        assert!(fuse_rankings(err, Ok(vec![(0, 1.0)])).is_err());
    }

    use super::*;

    #[test]
    fn combine_appends_passage_on_new_line() {
        assert_eq!(
            combine_retrieval_query("Chi è Geppetto?", Some("Geppetto è un falegname.")),
            "Chi è Geppetto?\nGeppetto è un falegname."
        );
    }

    #[test]
    fn combine_falls_back_to_original_query() {
        assert_eq!(combine_retrieval_query("Q?", None), "Q?");
        assert_eq!(combine_retrieval_query("Q?", Some("")), "Q?");
        assert_eq!(combine_retrieval_query("Q?", Some("  \n\t ")), "Q?");
    }

    #[test]
    fn passage_is_trimmed_before_combining() {
        assert_eq!(combine_retrieval_query("Q?", Some("\n P \n")), "Q?\nP");
    }

    #[test]
    fn clean_passage_strips_think_tags() {
        assert_eq!(
            clean_hyde_passage("<think>ragiono...</think>\nUna Volpe zoppa e un Gatto cieco."),
            "Una Volpe zoppa e un Gatto cieco."
        );
        // Only reasoning: nothing left, so the combine step falls back to the query.
        let cleaned = clean_hyde_passage("<think>solo ragionamento</think>");
        assert_eq!(combine_retrieval_query("Q?", Some(&cleaned)), "Q?");
    }

    #[test]
    fn prompt_contains_question_and_language_instructions() {
        let p = build_hyde_prompt("Quali animali ingannano Pinocchio?");
        assert!(p.contains("Quali animali ingannano Pinocchio?"));
        assert!(p.contains("3-4 sentences"));
        assert!(p.contains("language of the question"));
    }

    #[test]
    fn log_preview_is_char_safe() {
        assert_eq!(hyde_log_preview(&"è".repeat(300)).chars().count(), 120);
        assert_eq!(hyde_log_preview("breve"), "breve");
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn hyde_off_returns_nothing_without_llm_call() {
        let g = GraphRAG::new(Config::default()).unwrap();
        assert!(!g.config.retrieval.hyde);
        assert!(matches!(g.hyde_retrieval_query("Q?").await, (None, None)));
    }
}

#[cfg(all(test, feature = "async"))]
mod map_reduce_tests {
    use super::*;

    fn pc(id: &str) -> PoolChunk {
        PoolChunk {
            id: id.to_string(),
            content: format!("text {id}"),
        }
    }

    #[test]
    fn map_prompt_has_question_chunk_as_suffix_and_none_rule() {
        let prompt = build_map_prompt("Chi è Geppetto?", "Geppetto era un vecchio falegname.");
        assert!(prompt.contains("QUESTION: Chi è Geppetto?"));
        assert!(prompt.ends_with("Geppetto era un vecchio falegname."));
        assert!(prompt.contains("NONE"));
        assert!(prompt.contains("language of the passage"));
        // the shared prefix does not depend on the chunk
        let other = build_map_prompt("Chi è Geppetto?", "Altro testo.");
        let common = prompt.find("PASSAGE:").unwrap();
        assert_eq!(prompt[..common], other[..common]);
    }

    #[test]
    fn none_replies_yield_no_note() {
        for raw in [
            "NONE",
            "none.",
            " **NONE** ",
            "",
            "   \n",
            "\"None\"",
            "NONE.",
        ] {
            assert_eq!(parse_map_note(raw), None, "{raw:?}");
        }
        assert_eq!(
            parse_map_note("  - Geppetto è un falegname\n- Vive a casa  "),
            Some("- Geppetto è un falegname\n- Vive a casa".to_string())
        );
        assert!(parse_map_note("None of the above are true, says Tom").is_some());
    }

    #[test]
    fn notes_part_keeps_order_and_titles() {
        let notes = vec![
            ("c2".to_string(), "- b".to_string()),
            ("c1".to_string(), "- a".to_string()),
        ];
        let part = build_notes_part(&notes, 1000).unwrap();
        assert_eq!(part, format!("{MAP_REDUCE_TITLE}\n[c2]\n- b\n[c1]\n- a"));
        assert!(!part.starts_with('['), "must not look like a chunk part");
    }

    #[test]
    fn notes_part_drops_whole_tail_notes_to_fit_budget() {
        let notes: Vec<(String, String)> = (0..5)
            .map(|i| (format!("c{i}"), format!("- fact {i} {}", "x".repeat(90))))
            .collect();
        let full = build_notes_part(&notes, 10_000).unwrap();
        let budget = estimate_tokens(&full) - 5;
        let trimmed = build_notes_part(&notes, budget).unwrap();
        assert!(estimate_tokens(&trimmed) <= budget);
        assert!(trimmed.contains("[c0]") && !trimmed.contains("[c4]"));
        assert!(trimmed.ends_with(&"x".repeat(90)), "whole notes only");
        assert!(build_notes_part(&notes, 3).is_none());
    }

    #[test]
    fn empty_notes_give_no_part() {
        assert!(build_notes_part(&[], 1000).is_none());
    }

    #[test]
    fn map_pool_follows_ranking_then_pool_order_dedup_and_cap() {
        let pool = vec![pc("a"), pc("b"), pc("c"), pc("d")];
        let ranking = [(2usize, 0.9f32), (9, 0.8), (2, 0.7), (0, 0.5)];
        let ids = |v: Vec<PoolChunk>| v.into_iter().map(|c| c.id).collect::<Vec<_>>();
        assert_eq!(
            ids(order_map_pool(&pool, Some(&ranking), 10)),
            ["c", "a", "b", "d"]
        );
        assert_eq!(ids(order_map_pool(&pool, Some(&ranking), 2)), ["c", "a"]);
        assert_eq!(ids(order_map_pool(&pool, None, 3)), ["a", "b", "c"]);
        let dup = vec![pc("a"), pc("a"), pc("b")];
        assert_eq!(ids(order_map_pool(&dup, None, 10)), ["a", "b"]);
    }

    /// Fake chat server: records every request body; map prompts containing `BOOM` get an
    /// HTTP 500, other map prompts a bullet note (`NONE` for chunks mentioning `EMPTY`),
    /// everything else the final answer.
    #[cfg(feature = "llm")]
    fn chat_stub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let shared = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let log = shared.clone();
                std::thread::spawn(move || {
                    let mut raw_req = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        let n = stream.read(&mut buf).unwrap_or(0);
                        raw_req.extend_from_slice(&buf[..n]);
                        let text = String::from_utf8_lossy(&raw_req).to_string();
                        if let Some(pos) = text.find("\r\n\r\n") {
                            let len = text[..pos]
                                .to_ascii_lowercase()
                                .split("content-length:")
                                .nth(1)
                                .and_then(|r| r.lines().next())
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            if raw_req.len() >= pos + 4 + len || n == 0 {
                                break;
                            }
                        } else if n == 0 {
                            break;
                        }
                    }
                    let text = String::from_utf8_lossy(&raw_req).to_string();
                    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                    log.lock().unwrap().push(body.clone());
                    let is_map = body.contains("PASSAGE:");
                    if is_map && body.contains("BOOM") {
                        let _ = stream.write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom",
                        );
                        return;
                    }
                    let content = if !is_map {
                        "FINAL ANSWER"
                    } else if body.contains("EMPTY") {
                        "NONE"
                    } else {
                        "- map fact"
                    };
                    let json = serde_json::json!({
                        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                    })
                    .to_string();
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                            json.len()
                        )
                        .as_bytes(),
                    );
                });
            }
        });
        (url, log)
    }

    #[cfg(feature = "llm")]
    fn graphrag_with_chunks(
        url: &str,
        map_reduce: bool,
    ) -> (GraphRAG, Vec<retrieval::SearchResult>) {
        let mut config = Config::default().with_llm_base_url(url);
        config.llm.max_retries = 1;
        config.llm.enable_caching = false;
        config.retrieval.map_reduce = map_reduce;
        let mut graph = KnowledgeGraph::new();
        let texts = ["alpha fact", "BOOM beta", "EMPTY gamma", "delta fact"];
        let mut results = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let id = format!("c{i}");
            graph
                .add_chunk(TextChunk::new(
                    ChunkId::new(id.clone()),
                    DocumentId::new("d".to_string()),
                    text.to_string(),
                    0,
                    10,
                ))
                .unwrap();
            results.push(retrieval::SearchResult {
                id,
                content: text.to_string(),
                score: 1.0 - i as f32 * 0.1,
                result_type: retrieval::ResultType::Chunk,
                entities: Vec::new(),
                source_chunks: Vec::new(),
            });
        }
        let mut g = GraphRAG::new(config).unwrap();
        g.knowledge_graph = Some(graph);
        (g, results)
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn flag_off_makes_one_call_with_the_unchanged_prompt() {
        let (url, log) = chat_stub();
        let (mut g, results) = graphrag_with_chunks(&url, false);
        let answer = g
            .generate_semantic_answer_from_results("Q?", None, &results)
            .await
            .unwrap();
        assert_eq!(answer.text, "FINAL ANSWER");
        let bodies = log.lock().unwrap().clone();
        assert_eq!(bodies.len(), 1, "no map calls with the flag off");
        let expected = build_context_parts(g.knowledge_graph.as_ref().unwrap(), None, &results)
            .join("\n\n---\n\n");
        let prompt = build_answer_prompt(&expected, "Q?");
        assert!(
            bodies[0].contains(&serde_json::to_string(&prompt).unwrap()),
            "answer prompt must be byte-identical to the one without map-reduce"
        );
        assert!(!bodies[0].contains("Fatti estratti"));
        assert_eq!(answer.usage.unwrap().total_tokens, 15);
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn flag_on_maps_every_chunk_skips_failures_and_sums_usage() {
        let (url, log) = chat_stub();
        let (mut g, results) = graphrag_with_chunks(&url, true);
        let answer = g
            .generate_semantic_answer_from_results("Q?", None, &results)
            .await
            .unwrap();
        assert_eq!(answer.text, "FINAL ANSWER");
        let bodies = log.lock().unwrap().clone();
        let maps: Vec<_> = bodies.iter().filter(|b| b.contains("PASSAGE:")).collect();
        assert_eq!(
            maps.len(),
            4,
            "one map call per chunk, even the failing one"
        );
        let finals: Vec<_> = bodies.iter().filter(|b| !b.contains("PASSAGE:")).collect();
        assert_eq!(finals.len(), 1);
        // c0 and c3 give notes, c1 failed (500), c2 answered NONE
        let ctx = finals[0];
        assert!(ctx.contains("Fatti estratti dai passaggi"));
        assert!(ctx.contains("[c0]") && ctx.contains("[c3]"));
        assert!(!ctx.contains("[c1]\\n") && !ctx.contains("[c2]\\n"));
        // 2 successful + 1 NONE map calls (15 tokens each) + the answer; the 500 has no usage
        assert_eq!(answer.usage.unwrap().total_tokens, 15 * 4);
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn map_with_every_call_failing_answers_like_flag_off() {
        let (url, log) = chat_stub();
        let (mut g, mut results) = graphrag_with_chunks(&url, true);
        results.retain(|r| r.id == "c1");
        let answer = g
            .generate_semantic_answer_from_results("Q?", None, &results)
            .await
            .unwrap();
        assert_eq!(answer.text, "FINAL ANSWER");
        let bodies = log.lock().unwrap().clone();
        let last = bodies.iter().find(|b| !b.contains("PASSAGE:")).unwrap();
        assert!(!last.contains("Fatti estratti"));
    }
}

#[cfg(all(test, feature = "async"))]
mod summary_context_tests {
    use super::*;
    use crate::core::DocumentId;
    use crate::retrieval::{ResultType, SearchResult};

    #[test]
    fn summaries_above_last_original_follow_the_rule() {
        // S S O S O O S: top_k 2 -> last selected original is position 4
        let flags = [true, true, false, true, false, false, true];
        let pick = |top_k, cap| summaries_above_last_original(flags.iter().copied(), top_k, cap);
        assert_eq!(pick(2, 2), [0, 1]);
        assert_eq!(pick(2, 3), [0, 1, 3]);
        assert_eq!(pick(2, 9), [0, 1, 3]);
        assert_eq!(pick(1, 9), [0, 1]);
        // fewer originals than top_k: the last original is the limit (the final S is below it)
        assert_eq!(pick(9, 9), [0, 1, 3]);
        assert!(pick(2, 0).is_empty());
        // no original at all, or no summary: nothing
        assert!(summaries_above_last_original([true, true].into_iter(), 2, 2).is_empty());
        assert!(summaries_above_last_original([false, false].into_iter(), 2, 2).is_empty());
    }

    #[test]
    fn pick_summaries_returns_ranking_entries_in_order() {
        // pool: 0..3 originals, 3..5 summaries
        let ranking = [(4, 9.0), (0, 8.0), (3, 7.0), (1, 6.0), (2, 5.0)];
        assert_eq!(pick_summaries(&ranking, 3, 2, 5), [(4, 9.0), (3, 7.0)]);
        assert_eq!(pick_summaries(&ranking, 3, 1, 5), [(4, 9.0)]);
        assert_eq!(pick_summaries(&ranking, 3, 2, 1), [(4, 9.0)]);
        assert!(pick_summaries(&ranking, 3, 2, 0).is_empty());
        assert!(pick_summaries(&ranking, 5, 2, 5).is_empty());
    }

    fn hit(id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        }
    }

    #[test]
    fn summary_hit_parts_without_reranker() {
        let hits = [
            hit("d_summary_0", "[Riassunto del capitolo: I.]\na", 0.9),
            hit("o1", "one", 0.8),
            hit("d_summary_1", "[Riassunto del capitolo: II.]\nb", 0.7),
            hit("o2", "two", 0.6),
            hit("d_summary_2", "[Riassunto del capitolo: III.]\nc", 0.5),
        ];
        let parts = summary_hit_parts(&hits, 5);
        assert_eq!(
            parts.len(),
            2,
            "the summary below the last original is dropped"
        );
        assert!(parts[0].starts_with("[Chunk | Relevance: 0.90]\n[Riassunto del capitolo: I.]"));
        assert!(parts[1].contains("capitolo: II."));
        assert_eq!(summary_hit_parts(&hits, 1).len(), 1);
        assert!(summary_hit_parts(&hits, 0).is_empty());
        assert!(summary_hit_parts(&hits[1..2], 3).is_empty());
    }

    #[test]
    fn candidates_and_pool_builders_handle_summaries() {
        let mut graph = KnowledgeGraph::new();
        for (id, text) in [
            ("o1", "one"),
            ("d_summary_0", "[Riassunto del capitolo: I.]\na"),
            ("d_summary_1", "[Riassunto del capitolo: II.]\nb"),
        ] {
            graph
                .add_chunk(TextChunk::new(
                    ChunkId::new(id.to_string()),
                    DocumentId::new("d".to_string()),
                    text.to_string(),
                    0,
                    1,
                ))
                .unwrap();
        }
        let hits = [hit("o1", "one", 0.5), hit("d_summary_1", "x", 0.9)];
        let pool = build_candidate_pool(&graph, None, &hits, 10);
        assert_eq!(
            pool.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            ["o1"]
        );
        let mut pool = pool;
        extend_pool_with_chunk_ids(
            &graph,
            &mut pool,
            &["d_summary_0".to_string(), "o1".to_string()],
            10,
        );
        assert_eq!(pool.len(), 1);
        let fused = [
            "d_summary_0".to_string(),
            "d_summary_1".to_string(),
            "o1".to_string(),
        ];
        let cands = summary_candidates(&graph, &hits, &fused, 10);
        // search-result summary first, then the fused ones, deduplicated
        assert_eq!(
            cands.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            ["d_summary_1", "d_summary_0"]
        );
        assert_eq!(summary_candidates(&graph, &hits, &fused, 1).len(), 1);
    }

    #[cfg(feature = "llm")]
    type RequestLog = std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>;

    /// Stub for `/v1/chat/completions` (answers FINAL ANSWER, or a map fact for prompts with
    /// `PASSAGE:`) and `/v1/rerank` (score = the number after `s=` in the document, honours
    /// `top_n`). Every request body is logged with its path.
    #[cfg(feature = "llm")]
    fn stub() -> (String, RequestLog) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let shared = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let log = shared.clone();
                std::thread::spawn(move || {
                    let mut raw = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        let n = stream.read(&mut buf).unwrap_or(0);
                        raw.extend_from_slice(&buf[..n]);
                        let text = String::from_utf8_lossy(&raw).to_string();
                        if let Some(pos) = text.find("\r\n\r\n") {
                            let len = text[..pos]
                                .to_ascii_lowercase()
                                .split("content-length:")
                                .nth(1)
                                .and_then(|r| r.lines().next())
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            if raw.len() >= pos + 4 + len || n == 0 {
                                break;
                            }
                        } else if n == 0 {
                            break;
                        }
                    }
                    let text = String::from_utf8_lossy(&raw).to_string();
                    let path = text
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or_default()
                        .to_string();
                    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                    log.lock().unwrap().push((path.clone(), body.clone()));
                    let json = if path.ends_with("/rerank") {
                        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
                        let mut scored: Vec<(usize, f64)> = value["documents"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .enumerate()
                            .map(|(i, d)| {
                                let d = d.as_str().unwrap();
                                let score = d
                                    .split("s=")
                                    .nth(1)
                                    .and_then(|r| r.split_whitespace().next())
                                    .and_then(|v| v.parse::<f64>().ok())
                                    .unwrap_or(0.0);
                                (i, score)
                            })
                            .collect();
                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        let top_n = value["top_n"].as_u64().map_or(scored.len(), |n| n as usize);
                        let results: Vec<_> = scored
                            .iter()
                            .take(top_n)
                            .map(|(i, s)| serde_json::json!({"index": i, "relevance_score": s}))
                            .collect();
                        serde_json::json!({ "results": results }).to_string()
                    } else {
                        let content = if body.contains("PASSAGE:") {
                            "- map fact"
                        } else {
                            "FINAL ANSWER"
                        };
                        serde_json::json!({
                            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                        })
                        .to_string()
                    };
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                            json.len()
                        )
                        .as_bytes(),
                    );
                });
            }
        });
        (base, log)
    }

    #[cfg(feature = "llm")]
    const SEP: &str = "\n\n---\n\n";

    /// Originals `o0..o3` scored 5,4,3,2 (all returned as chunk hits, in that order) and,
    /// when `summaries` is given, summary chunks with the given reranker scores.
    #[cfg(feature = "llm")]
    fn setup(
        base: &str,
        reranker: bool,
        summaries: &[f32],
        cap: usize,
    ) -> (GraphRAG, Vec<SearchResult>) {
        let mut config = Config::default().with_llm_base_url(format!("{base}/v1"));
        config.llm.max_retries = 1;
        config.llm.enable_caching = false;
        config.reranker.enabled = reranker;
        config.reranker.api_endpoint = base.to_string();
        config.reranker.top_k = 2;
        config.chapter_summaries.max_in_context = cap;
        let mut graph = KnowledgeGraph::new();
        let mut results = Vec::new();
        let mut add = |id: String, content: String, score: f32| {
            graph
                .add_chunk(TextChunk::new(
                    ChunkId::new(id.clone()),
                    DocumentId::new("d".to_string()),
                    content.clone(),
                    0,
                    1,
                ))
                .unwrap();
            results.push(hit(&id, &content, score));
        };
        for (i, s) in [5, 4, 3, 2].iter().enumerate() {
            add(
                format!("o{i}"),
                format!("original {i} s={s}"),
                1.0 - i as f32 * 0.1,
            );
        }
        for (i, s) in summaries.iter().enumerate() {
            add(
                format!("d_summary_{i}"),
                format!("[Riassunto del capitolo: {i}.]\nsummary {i} s={s}"),
                // summaries lead the retrieval ranking too
                2.0 - i as f32 * 0.1,
            );
        }
        let mut g = GraphRAG::new(config).unwrap();
        g.knowledge_graph = Some(graph);
        (g, results)
    }

    /// The context the answer prompt must carry when no summary joins it.
    #[cfg(feature = "llm")]
    fn expected_originals(g: &GraphRAG, results: &[SearchResult], reranker: bool) -> Vec<String> {
        let graph = g.knowledge_graph.as_ref().unwrap();
        let originals: Vec<SearchResult> = results
            .iter()
            .filter(|r| !is_summary_chunk_id(&r.id))
            .cloned()
            .collect();
        let mut ids = std::collections::HashSet::new();
        let mut parts = Vec::new();
        if reranker {
            // scores 5 and 4: o0, o1
            parts.push("[Chunk | Rerank: 5.00]\noriginal 0 s=5".to_string());
            parts.push("[Chunk | Rerank: 4.00]\noriginal 1 s=4".to_string());
            ids.insert("o0".to_string());
            ids.insert("o1".to_string());
        }
        parts.extend(build_context_parts_excluding(graph, None, &originals, &ids));
        parts
    }

    #[cfg(feature = "llm")]
    async fn final_prompt(
        g: &mut GraphRAG,
        results: &[SearchResult],
        log: &std::sync::Mutex<Vec<(String, String)>>,
    ) -> String {
        g.generate_semantic_answer_from_results("Q?", None, results)
            .await
            .unwrap();
        let bodies = log.lock().unwrap().clone();
        let (_, body) = bodies
            .iter()
            .rev()
            .find(|(p, b)| p.ends_with("/chat/completions") && !b.contains("PASSAGE:"))
            .unwrap();
        body.clone()
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn index_without_summaries_gives_the_same_context_as_before() {
        for reranker in [true, false] {
            let (base, log) = stub();
            let (mut g, results) = setup(&base, reranker, &[], 2);
            let body = final_prompt(&mut g, &results, &log).await;
            let expected = expected_originals(&g, &results, reranker).join(SEP);
            let prompt = build_answer_prompt(&expected, "Q?");
            assert!(
                body.contains(&serde_json::to_string(&prompt).unwrap()),
                "reranker {reranker}: context changed"
            );
            if reranker {
                let rerank: Vec<_> = log
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(p, _)| p.ends_with("/rerank"))
                    .cloned()
                    .collect();
                assert_eq!(rerank.len(), 1);
                assert!(
                    rerank[0].1.contains("\"top_n\":2"),
                    "still asks for top_k only"
                );
            }
        }
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn summaries_are_added_after_the_originals_up_to_the_cap() {
        for reranker in [true, false] {
            // reranker scores: three summaries; the first two beat the weakest selected
            // original (4), the third (s=1) does not
            let (base, log) = stub();
            let (mut g, results) = setup(&base, reranker, &[9.0, 8.0, 1.0], 2);
            let body = final_prompt(&mut g, &results, &log).await;
            let mut parts = expected_originals(&g, &results, reranker);
            let marker = |i: usize| {
                if reranker {
                    format!(
                        "[Chunk | Rerank: {:.2}]\n[Riassunto del capitolo: {i}.]\nsummary {i} s={}",
                        [9.0, 8.0][i],
                        [9, 8][i]
                    )
                } else {
                    format!(
                        "[Chunk | Relevance: {:.2}]\n[Riassunto del capitolo: {i}.]\nsummary {i} s={}",
                        2.0 - i as f32 * 0.1,
                        [9, 8][i]
                    )
                }
            };
            parts.extend([marker(0), marker(1)]);
            let prompt = build_answer_prompt(&parts.join(SEP), "Q?");
            assert!(
                body.contains(&serde_json::to_string(&prompt).unwrap()),
                "reranker {reranker}: originals must be unchanged, summaries last"
            );
            assert!(!body.contains("capitolo: 2."));
            if reranker {
                // the reranker was asked for every candidate to rank the summaries
                let req = log
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(p, _)| p.ends_with("/rerank"))
                    .cloned()
                    .unwrap();
                assert!(req.1.contains("\"top_n\":7"));
            }
        }
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn cap_zero_keeps_summaries_out_of_the_context() {
        for reranker in [true, false] {
            let (base, log) = stub();
            let (mut g, results) = setup(&base, reranker, &[9.0, 8.0], 0);
            let body = final_prompt(&mut g, &results, &log).await;
            let expected = expected_originals(&g, &results, reranker).join(SEP);
            let prompt = build_answer_prompt(&expected, "Q?");
            assert!(body.contains(&serde_json::to_string(&prompt).unwrap()));
            assert!(!body.contains("Riassunto"));
        }
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn summaries_ranked_below_the_last_original_are_dropped() {
        let (base, log) = stub();
        let (mut g, results) = setup(&base, true, &[3.5, 1.0], 2);
        let body = final_prompt(&mut g, &results, &log).await;
        let expected = expected_originals(&g, &results, true).join(SEP);
        let prompt = build_answer_prompt(&expected, "Q?");
        assert!(body.contains(&serde_json::to_string(&prompt).unwrap()));
        assert!(!body.contains("Riassunto"));
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn map_reduce_never_maps_summary_chunks() {
        let (base, log) = stub();
        let (mut g, results) = setup(&base, true, &[9.0], 2);
        g.config.retrieval.map_reduce = true;
        g.config.retrieval.map_reduce_chunks = 10;
        let body = final_prompt(&mut g, &results, &log).await;
        assert!(body.contains("Fatti estratti"));
        assert!(body.contains("[Riassunto del capitolo: 0.]"));
        let bodies = log.lock().unwrap().clone();
        let maps: Vec<_> = bodies
            .iter()
            .filter(|(_, b)| b.contains("PASSAGE:"))
            .collect();
        assert_eq!(maps.len(), 4, "one map call per original chunk");
        assert!(maps.iter().all(|(_, b)| !b.contains("Riassunto")));
    }
}
