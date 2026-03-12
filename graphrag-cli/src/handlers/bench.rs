use color_eyre::eyre::{eyre, Result};
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
    question: String,
    answer: String,
    confidence: Option<f32>,
    sources: Vec<SourceInfo>,
    query_time_ms: u128,
}

#[derive(Serialize, Deserialize)]
struct SourceInfo {
    id: String,
    excerpt: String,
    relevance_score: f32,
}

pub async fn run_benchmark(
    config_path: &Path,
    book_path: &Path,
    questions: Vec<String>,
) -> Result<()> {
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

    // 3. Load & Build
    let start_build = Instant::now();
    let content = tokio::fs::read_to_string(book_path)
        .await
        .map_err(|e| eyre!("Failed to read book file: {}", e))?;

    graphrag.add_document_from_text(&content)?;
    graphrag.build_graph().await?;

    let build_ms = start_build.elapsed().as_millis();
    info!("Build done in {}ms", build_ms);

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
            Err(e) => (format!("Error: {}", e), None, vec![]),
        };
        let q_ms = q_start.elapsed().as_millis();

        qa_results.push(QAItem {
            index: i + 1,
            question: q.clone(),
            answer,
            confidence,
            sources,
            query_time_ms: q_ms,
        });
        info!("Q{} done in {}ms", i + 1, q_ms);
    }

    let total_query_ms = start_query_total.elapsed().as_millis();
    let total_ms = start_all.elapsed().as_millis();

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
    };

    println!("{}", serde_json::to_string(&result)?);

    Ok(())
}
