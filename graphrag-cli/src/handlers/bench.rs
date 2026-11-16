use super::bench_score::{score_answer, summarize, GoldFile, Score, Summary};
use color_eyre::eyre::{eyre, Result};
use graphrag_core::llm::{global_llm_usage, LlmUsageTotals};
use graphrag_core::GraphRAG;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tracing::info;

#[derive(Serialize, Deserialize)]
struct BenchResult {
    config_file: String,
    book_file: String,
    timing: BenchTiming,
    stats: BenchStats,
    questions_and_answers: Vec<QAItem>,
    llm_usage: LlmUsageReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<Summary>,
}

/// Server-reported LLM token usage, split by phase (process-wide deltas).
#[derive(Serialize, Deserialize)]
struct LlmUsageReport {
    build: LlmUsageTotals,
    queries: LlmUsageTotals,
}

impl LlmUsageReport {
    fn from_snapshots(
        start: &LlmUsageTotals,
        after_build: &LlmUsageTotals,
        after_queries: &LlmUsageTotals,
    ) -> Self {
        Self {
            build: after_build.since(start),
            queries: after_queries.since(after_build),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BenchTiming {
    init_ms: u128,
    build_ms: u128,
    total_query_ms: u128,
    total_ms: u128,
}

#[derive(Serialize, Deserialize, Default)]
struct BenchStats {
    entities: usize,
    relationships: usize,
    chunks: usize,
}

#[derive(Serialize, Deserialize)]
struct QAItem {
    index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    question: String,
    answer: String,
    confidence: Option<f32>,
    sources: Vec<SourceInfo>,
    query_time_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score: Option<Score>,
}

#[derive(Serialize, Deserialize)]
struct SourceInfo {
    id: String,
    excerpt: String,
    relevance_score: f32,
}

/// Index the book only when no persisted workspace was reloaded (graph has no chunks).
fn should_index(existing_chunks: usize) -> bool {
    existing_chunks == 0
}

pub async fn run_benchmark(
    config_path: &Path,
    book_path: &Path,
    questions: Vec<String>,
    gold_path: Option<&Path>,
) -> Result<()> {
    let gold = match gold_path {
        Some(p) => {
            let raw = tokio::fs::read_to_string(p)
                .await
                .map_err(|e| eyre!("Failed to read gold file: {}", e))?;
            let g: GoldFile = serde_json::from_str(&raw)
                .map_err(|e| eyre!("Invalid gold file {}: {}", p.display(), e))?;
            Some(g.questions)
        },
        None => None,
    };
    let questions: Vec<String> = match &gold {
        Some(g) => g.iter().map(|q| q.question.clone()).collect(),
        None => questions,
    };
    if questions.iter().all(|q| q.trim().is_empty()) {
        return Err(eyre!("No questions: pass --questions or --gold"));
    }

    // 1. Load config
    let config = crate::config::load_config(config_path).await?;
    let config_file_str = config_path.to_string_lossy().to_string();
    let book_file_str = book_path.to_string_lossy().to_string();

    // 2. Init
    let start_all = Instant::now();
    let start_init = Instant::now();

    // We instantiate GraphRAG directly, not via the handler, to keep it simple and local
    let mut graphrag = GraphRAG::new(config)?;
    graphrag.initialize()?;

    let init_ms = start_init.elapsed().as_millis();
    info!("Init done in {}ms", init_ms);

    let usage_start = global_llm_usage();

    // 3. Load & Build (skipped when initialize() reloaded a persisted workspace)
    let reloaded_chunks = graphrag
        .knowledge_graph()
        .map(|kg| kg.chunks().count())
        .unwrap_or(0);
    let build_ms = if should_index(reloaded_chunks) {
        let start_build = Instant::now();
        let content = tokio::fs::read_to_string(book_path)
            .await
            .map_err(|e| eyre!("Failed to read book file: {}", e))?;

        graphrag.add_document_from_text(&content)?;
        graphrag.build_graph().await?;

        let build_ms = start_build.elapsed().as_millis();
        info!("Build done in {}ms", build_ms);
        build_ms
    } else {
        eprintln!("Index reloaded from workspace ({reloaded_chunks} chunks): skipping indexing");
        0 // no build happened; the reload cost is part of init_ms
    };

    let usage_after_build = global_llm_usage();

    // Get stats
    let kg = graphrag
        .knowledge_graph()
        .ok_or(eyre!("Knowledge graph not initialized"))?;
    let stats = BenchStats {
        entities: kg.entities().count(),
        relationships: kg.relationships().count(),
        chunks: kg.chunks().count(),
    };

    // 4. Query
    let mut qa_results = Vec::new();
    let start_query_total = Instant::now();

    for (i, q) in questions.iter().enumerate() {
        let q_start = Instant::now();
        // Use ask_explained() which returns answer + source references
        let (answer, confidence, sources) = match graphrag.ask_explained(q).await {
            Ok(explained) => {
                let source_infos: Vec<SourceInfo> = explained
                    .sources
                    .iter()
                    .map(|s| SourceInfo {
                        id: s.id.clone(),
                        excerpt: s.excerpt.clone(),
                        relevance_score: s.relevance_score,
                    })
                    .collect();
                (explained.answer, Some(explained.confidence), source_infos)
            },
            Err(e) => (format!("Error: {e}"), None, vec![]),
        };
        let q_ms = q_start.elapsed().as_millis();

        let gold_q = gold.as_ref().map(|g| &g[i]);
        let score = gold_q.map(|g| score_answer(&answer, &g.expected_keywords));
        qa_results.push(QAItem {
            index: i + 1,
            id: gold_q.map(|g| g.id.clone()),
            question: q.clone(),
            answer,
            confidence,
            sources,
            query_time_ms: q_ms,
            score,
        });
        info!("Q{} done in {}ms", i + 1, q_ms);
    }

    let llm_usage =
        LlmUsageReport::from_snapshots(&usage_start, &usage_after_build, &global_llm_usage());
    let total_query_ms = start_query_total.elapsed().as_millis();
    let total_ms = start_all.elapsed().as_millis();

    let summary = gold.as_ref().map(|_| {
        let scores: Vec<Score> = qa_results.iter().filter_map(|q| q.score.clone()).collect();
        let lat: Vec<u128> = qa_results.iter().map(|q| q.query_time_ms).collect();
        summarize(&scores, &lat)
    });

    // Output JSON
    let result = BenchResult {
        config_file: config_file_str,
        book_file: book_file_str,
        timing: BenchTiming {
            init_ms,
            build_ms,
            total_query_ms,
            total_ms,
        },
        stats,
        questions_and_answers: qa_results,
        llm_usage,
        summary,
    };

    println!("{}", serde_json::to_string(&result)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{should_index, LlmUsageReport};
    use graphrag_core::llm::LlmUsageTotals;

    #[test]
    fn indexes_only_when_graph_is_empty() {
        assert!(should_index(0));
        assert!(!should_index(108));
    }

    #[test]
    fn llm_usage_report_splits_build_and_queries() {
        let totals = |requests, prompt_tokens| LlmUsageTotals {
            requests,
            prompt_tokens,
            ..Default::default()
        };
        let report =
            LlmUsageReport::from_snapshots(&totals(1, 10), &totals(5, 110), &totals(7, 150));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["build"]["requests"], 4);
        assert_eq!(json["build"]["prompt_tokens"], 100);
        assert_eq!(json["queries"]["requests"], 2);
        assert_eq!(json["queries"]["prompt_tokens"], 40);
        assert!(json["queries"].get("cached_prompt_tokens").is_some());
    }
}
