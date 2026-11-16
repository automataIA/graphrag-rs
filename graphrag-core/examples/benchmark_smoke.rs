use graphrag_core::config::Config;
use graphrag_core::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, KnowledgeGraph, Relationship,
    TextChunk,
};
use graphrag_core::evaluation::{
    evaluate_benchmark, AnswerRecord, BenchmarkEvaluationInput, BenchmarkEvaluationReport,
    ProviderTokenUsage, RetrievalRecord,
};
use graphrag_core::llama_cpp::LlamaCppEmbeddingProvider;
use graphrag_core::llm::{LlmClient, LlmGenerationParams, LlmUsage};
use graphrag_core::retrieval::{BM25Document, BM25Retriever};
use graphrag_core::Result as CoreResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct CorpusFile {
    schema_version: String,
    corpus_id: String,
    documents: Vec<CorpusDocument>,
    entities: Vec<CorpusEntity>,
    relationships: Vec<CorpusRelationship>,
}

#[derive(Debug, Deserialize)]
struct CorpusDocument {
    id: String,
    title: String,
    content: String,
    chunk_id: String,
}

#[derive(Debug, Deserialize)]
struct CorpusEntity {
    id: String,
    name: String,
    #[serde(rename = "type")]
    entity_type: String,
    mentions: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CorpusRelationship {
    source: String,
    target: String,
    #[serde(rename = "type")]
    relation_type: String,
    context: Vec<String>,
    confidence: f32,
}

#[derive(Debug, Deserialize)]
struct QuestionsFile {
    questions: Vec<graphrag_core::evaluation::BenchmarkQuestion>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkConfig {
    schema_version: String,
    config_version: String,
    runner_version: String,
    seed: u64,
    top_k: usize,
    context_limit_chars: usize,
    chunking: ChunkingConfig,
    hybrid_fusion: HybridFusionConfig,
    embedding: EmbeddingBenchmarkConfig,
    generation: GenerationBenchmarkConfig,
    provider_revisions: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ChunkingConfig {
    strategy: String,
    chunk_size: usize,
    chunk_overlap: usize,
}

#[derive(Debug, Deserialize)]
struct HybridFusionConfig {
    kind: String,
    k: f64,
    lexical_weight: f64,
    vector_weight: f64,
}

#[derive(Debug, Deserialize)]
struct EmbeddingBenchmarkConfig {
    provider: String,
    model: String,
    dimension: usize,
    max_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct GenerationBenchmarkConfig {
    provider: String,
    model: String,
    context_budget_chars: usize,
    max_tokens: u32,
    temperature: f64,
}

fn validate_benchmark_config(config: &BenchmarkConfig) -> Result<(), Box<dyn Error>> {
    if config.schema_version != "graphrag-0.3"
        || config.config_version != "graphrag-0.3-benchmark-config-v1"
        || config.runner_version != "graphrag-core-example-v1"
    {
        return Err("benchmark config has an unsupported schema, config, or runner version".into());
    }
    if config.seed != 42 {
        return Err("benchmark config seed must be 42 (server-side/provider seed)".into());
    }
    if config.top_k == 0 || config.context_limit_chars == 0 {
        return Err("benchmark top_k and context_limit_chars must be greater than zero".into());
    }
    if config.chunking.chunk_size == 0
        || config.chunking.chunk_overlap >= config.chunking.chunk_size
    {
        return Err("benchmark chunking requires chunk_size > 0 and overlap < size".into());
    }
    if config.chunking.strategy != "versioned one-source-per-chunk"
        || config.chunking.chunk_size != 512
        || config.chunking.chunk_overlap != 0
    {
        return Err("benchmark chunking fixture settings are unsupported".into());
    }
    if config.hybrid_fusion.kind != "reciprocal_rank"
        || !config.hybrid_fusion.k.is_finite()
        || config.hybrid_fusion.k <= 0.0
        || !config.hybrid_fusion.lexical_weight.is_finite()
        || config.hybrid_fusion.lexical_weight < 0.0
        || !config.hybrid_fusion.vector_weight.is_finite()
        || config.hybrid_fusion.vector_weight < 0.0
        || config.hybrid_fusion.lexical_weight + config.hybrid_fusion.vector_weight <= 0.0
    {
        return Err("benchmark hybrid fusion settings are invalid".into());
    }
    if config.embedding.provider != "llama_cpp"
        || config.embedding.model.trim().is_empty()
        || config.embedding.dimension != 384
        || config.embedding.max_tokens != 512
    {
        return Err(
            "benchmark embedding must use llama_cpp E5 dimension 384 and 512 tokens".into(),
        );
    }
    if config.generation.provider != "llama_cpp"
        || config.generation.model.trim().is_empty()
        || config.generation.context_budget_chars != config.context_limit_chars
        || config.generation.max_tokens == 0
        || !config.generation.temperature.is_finite()
        || !(config.generation.temperature as f32).is_finite()
    {
        return Err(
            "benchmark generation settings are invalid or use a different context budget".into(),
        );
    }
    for provider in ["llama_cpp", "e5_source"] {
        if config
            .provider_revisions
            .get(provider)
            .is_none_or(|revision| revision.trim().is_empty())
        {
            return Err(format!("benchmark provider revision {provider} is missing").into());
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Lexical,
    Vector,
    Hybrid,
    Graph,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Vector => "vector",
            Self::Hybrid => "hybrid",
            Self::Graph => "graph",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct RankedChunk {
    source_id: String,
    score: f64,
    content: String,
}

#[derive(Debug, Serialize)]
struct RetrievalOutput {
    question_id: String,
    source_ids: Vec<String>,
    scores: Vec<f64>,
    latency_ms: f64,
    /// Raw chunks supplied as context, retained beside evaluator fields.
    retrieved: Vec<RankedChunk>,
}

#[derive(Debug, Serialize)]
struct RunOutput {
    run: u32,
    mode: Mode,
    retrieval: Vec<RetrievalOutput>,
    answers: Vec<AnswerRecord>,
    report: BenchmarkEvaluationReport,
}

#[derive(Debug, Serialize)]
struct ModeSummary {
    mode: Mode,
    run_count: u32,
    question_count: usize,
    mean_recall_at_k: Option<f64>,
    mean_reciprocal_rank: Option<f64>,
    mean_retrieved_source_precision: f64,
    mean_expected_term_coverage: Option<f64>,
    abstention_rate: Option<f64>,
    mean_retrieval_latency_ms: f64,
    mean_answer_latency_ms: f64,
    measured_usage_count: usize,
    missing_usage_count: usize,
    measured_prompt_tokens: u64,
    measured_completion_tokens: u64,
    measured_total_tokens: u64,
}

#[derive(Debug, Serialize)]
struct Manifest {
    schema_version: String,
    config_version: String,
    runner_version: String,
    corpus_id: String,
    corpus_sha256: String,
    questions_sha256: String,
    config_sha256: String,
    e5_gguf_path: String,
    e5_gguf_sha256: Option<String>,
    ornith_gguf_path: String,
    ornith_gguf_sha256: Option<String>,
    modes: Vec<&'static str>,
    top_k: usize,
    context_limit_chars: usize,
    hybrid_fusion: String,
    seed: u64,
    seed_semantics: &'static str,
    run_count: u32,
    e5_model: String,
    e5_endpoint: String,
    ornith_model: String,
    ornith_endpoint: String,
    provider_revisions: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct SmokeOutput {
    manifest: Manifest,
    runs: Vec<RunOutput>,
    summaries: Vec<ModeSummary>,
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn runtime_config(benchmark: &BenchmarkConfig) -> Config {
    let mut config = Config::default();
    config.embeddings.backend = "llama_cpp".to_string();
    config.embeddings.dimension = benchmark.embedding.dimension;
    config.embeddings.model = Some(benchmark.embedding.model.clone());
    config.embeddings.api_endpoint = Some(format!(
        "{}/v1/embeddings",
        env_or("GRAPHRAG_E5_ENDPOINT", "http://127.0.0.1:18081").trim_end_matches('/')
    ));
    config.embeddings.fallback_to_hash = false;
    config.llm.enabled = true;
    config.llm.base_url = format!(
        "{}/v1",
        env_or("GRAPHRAG_ORNITH_ENDPOINT", "http://127.0.0.1:18080").trim_end_matches('/')
    );
    // Every run must hit the server: the in-process response cache would hide latency.
    config.llm.enable_caching = false;
    config.llm.chat_model = env_or("GRAPHRAG_ORNITH_MODEL", &benchmark.generation.model);
    config.llm.timeout_seconds = 120;
    config.llm.max_tokens = Some(benchmark.generation.max_tokens);
    config.llm.fallback_to_hash = false;
    config.llm.temperature = Some(benchmark.generation.temperature as f32);
    config
}

fn graph_from_corpus(corpus: &CorpusFile) -> CoreResult<KnowledgeGraph> {
    let mut graph = KnowledgeGraph::new();
    let documents: HashMap<_, _> = corpus
        .documents
        .iter()
        .map(|document| (document.chunk_id.clone(), document))
        .collect();
    for document in &corpus.documents {
        let document_id = DocumentId::new(document.id.clone());
        let chunk = TextChunk::new(
            ChunkId::new(document.chunk_id.clone()),
            document_id.clone(),
            document.content.clone(),
            0,
            document.content.len(),
        );
        graph.add_document(
            Document::new(
                document_id,
                document.title.clone(),
                document.content.clone(),
            )
            .with_chunks(vec![chunk]),
        )?;
    }
    for entity in &corpus.entities {
        let mentions = entity
            .mentions
            .iter()
            .filter_map(|chunk_id| documents.get(chunk_id).map(|document| (chunk_id, document)))
            .map(|(chunk_id, document)| {
                let start = document.content.find(&entity.name).unwrap_or(0);
                EntityMention {
                    chunk_id: ChunkId::new(chunk_id.clone()),
                    start_offset: start,
                    end_offset: start + entity.name.len(),
                    confidence: 1.0,
                }
            })
            .collect();
        graph.add_entity(
            Entity::new(
                EntityId::new(entity.id.clone()),
                entity.name.clone(),
                entity.entity_type.clone(),
                1.0,
            )
            .with_mentions(mentions),
        )?;
    }
    for relationship in &corpus.relationships {
        graph.add_relationship(
            Relationship::new(
                EntityId::new(relationship.source.clone()),
                EntityId::new(relationship.target.clone()),
                relationship.relation_type.clone(),
                relationship.confidence,
            )
            .with_context(
                relationship
                    .context
                    .iter()
                    .map(|id| ChunkId::new(id.clone()))
                    .collect(),
            ),
        )?;
    }
    Ok(graph)
}

fn lexical_index(graph: &KnowledgeGraph) -> Result<BM25Retriever, Box<dyn Error>> {
    let mut index = BM25Retriever::new();
    for chunk in graph.chunks() {
        index.index_document(BM25Document {
            id: chunk.id.to_string(),
            content: chunk.content.clone(),
            metadata: HashMap::new(),
        })?;
    }
    Ok(index)
}

fn lexical(index: &BM25Retriever, query: &str, top_k: usize) -> Vec<RankedChunk> {
    let mut rows: Vec<_> = index
        .search(query, top_k)
        .into_iter()
        .map(|row| RankedChunk {
            source_id: row.doc_id,
            score: f64::from(row.score),
            content: row.content,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    rows
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn document_embeddings(
    provider: &LlamaCppEmbeddingProvider,
    graph: &KnowledgeGraph,
) -> Result<BTreeMap<String, Vec<f32>>, Box<dyn Error>> {
    let mut chunks = graph.chunks().collect::<Vec<_>>();
    chunks.sort_by_key(|chunk| chunk.id.to_string());
    let texts = chunks
        .iter()
        .map(|chunk| chunk.content.as_str())
        .collect::<Vec<_>>();
    let vectors = provider.embed_batch(&texts, graphrag_core::llama_cpp::E5Intent::Document)?;
    let embeddings = chunks
        .into_iter()
        .zip(vectors)
        .map(|(chunk, vector)| (chunk.id.to_string(), vector))
        .collect();
    Ok(embeddings)
}

fn vector(
    graph: &KnowledgeGraph,
    document_embeddings: &BTreeMap<String, Vec<f32>>,
    query_embedding: &[f32],
    top_k: usize,
) -> Result<Vec<RankedChunk>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for chunk in graph.chunks() {
        let embedding = document_embeddings
            .get(&chunk.id.to_string())
            .ok_or_else(|| format!("missing embedding for {}", chunk.id))?;
        rows.push(RankedChunk {
            source_id: chunk.id.to_string(),
            score: cosine(query_embedding, embedding),
            content: chunk.content.clone(),
        });
    }
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    rows.truncate(top_k);
    Ok(rows)
}

fn hybrid(
    lexical: &[RankedChunk],
    vector: &[RankedChunk],
    top_k: usize,
    fusion_k: f64,
    lexical_weight: f64,
    vector_weight: f64,
) -> Vec<RankedChunk> {
    let mut by_id: HashMap<String, RankedChunk> = HashMap::new();
    let mut scores: HashMap<String, f64> = HashMap::new();
    for (rank, row) in lexical.iter().enumerate() {
        by_id
            .entry(row.source_id.clone())
            .or_insert_with(|| row.clone());
        *scores.entry(row.source_id.clone()).or_default() +=
            lexical_weight / (fusion_k + rank as f64 + 1.0);
    }
    for (rank, row) in vector.iter().enumerate() {
        by_id
            .entry(row.source_id.clone())
            .or_insert_with(|| row.clone());
        *scores.entry(row.source_id.clone()).or_default() +=
            vector_weight / (fusion_k + rank as f64 + 1.0);
    }
    let mut rows: Vec<_> = scores
        .into_iter()
        .filter_map(|(id, score)| {
            by_id.remove(&id).map(|mut row| {
                row.score = score;
                row
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    rows.truncate(top_k);
    rows
}

fn terms(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

fn graph_mode(graph: &KnowledgeGraph, query: &str, top_k: usize) -> Vec<RankedChunk> {
    let query_terms = terms(query);
    let mut chunks: HashMap<String, (f64, String)> = HashMap::new();
    let mut entities: Vec<_> = graph.entities().collect();
    entities.sort_by_key(|entity| entity.id.to_string());
    for entity in entities {
        if !terms(&entity.name)
            .iter()
            .any(|term| query_terms.contains(term))
        {
            continue;
        }
        for (target, relationship) in graph.get_neighbors(&entity.id) {
            let relation_terms = terms(&relationship.relation_type);
            let relation_boost = if relation_terms.iter().any(|term| query_terms.contains(term)) {
                0.25
            } else {
                0.0
            };
            for source in &relationship.context {
                if let Some(chunk) = graph.get_chunk(source) {
                    let score = f64::from(relationship.confidence) + relation_boost;
                    let entry = chunks
                        .entry(source.to_string())
                        .or_insert((score, chunk.content.clone()));
                    if score > entry.0 {
                        entry.0 = score;
                    }
                }
            }
            // Include target mentions: this is relation expansion, not a fabricated fact.
            for mention in &target.mentions {
                if let Some(chunk) = graph.get_chunk(&mention.chunk_id) {
                    let score = f64::from(relationship.confidence) * 0.9;
                    let entry = chunks
                        .entry(mention.chunk_id.to_string())
                        .or_insert((score, chunk.content.clone()));
                    if score > entry.0 {
                        entry.0 = score;
                    }
                }
            }
        }
    }
    let mut rows: Vec<_> = chunks
        .into_iter()
        .map(|(source_id, (score, content))| RankedChunk {
            source_id,
            score,
            content,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.source_id.cmp(&b.source_id))
    });
    rows.truncate(top_k);
    rows
}

fn bounded_context(rows: &[RankedChunk], context_limit: usize) -> String {
    let mut result = String::new();
    for row in rows {
        let addition = format!("[{}] {}\n", row.source_id, row.content);
        if result.len() + addition.len() <= context_limit {
            result.push_str(&addition);
        } else {
            let remaining = context_limit.saturating_sub(result.len());
            let mut end = addition.len().min(remaining);
            while end > 0 && !addition.is_char_boundary(end) {
                end -= 1;
            }
            result.push_str(&addition[..end]);
            break;
        }
    }
    result
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn optional_file_sha256(path: &str) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

fn provider_usage(usage: Option<LlmUsage>) -> Option<ProviderTokenUsage> {
    usage.map(|usage| ProviderTokenUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    })
}

fn parse_runs(args: &[String]) -> Result<(u32, bool), Box<dyn Error>> {
    let mut smoke = false;
    let mut runs = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--smoke" => {
                if smoke {
                    return Err("--smoke may only be provided once".into());
                }
                smoke = true;
            },
            "--runs" => {
                if runs.is_some() {
                    return Err("--runs may only be provided once".into());
                }
                let value = args
                    .get(index + 1)
                    .ok_or("--runs requires a positive integer")?;
                let parsed = value.parse::<u32>().map_err(|_| "invalid --runs value")?;
                runs = Some(parsed);
                index += 1;
            },
            other => return Err(format!("unknown benchmark argument: {other}").into()),
        }
        index += 1;
    }
    match (smoke, runs) {
        (true, None) => Ok((1, true)),
        (false, Some(runs)) if runs > 0 => Ok((runs, false)),
        (true, Some(_)) => Err("--smoke and --runs are mutually exclusive".into()),
        (false, None) => Err("use --smoke or --runs N".into()),
        (false, Some(_)) => Err("--runs must be greater than zero".into()),
    }
}

fn average(values: impl Iterator<Item = f64>, count: usize) -> Result<f64, Box<dyn Error>> {
    if count == 0 {
        return Err("cannot average an empty collection".into());
    }
    let total: f64 = values.sum();
    if !total.is_finite() {
        return Err("non-finite aggregate metric".into());
    }
    Ok(total / count as f64)
}

fn average_optional(
    values: impl Iterator<Item = Option<f64>>,
) -> Result<Option<f64>, Box<dyn Error>> {
    let values = values.flatten().collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(None);
    }
    let count = values.len();
    Ok(Some(average(values.into_iter(), count)?))
}

fn display_optional(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "N/A".to_string())
}

fn mode_summary(
    mode: Mode,
    reports: &[&BenchmarkEvaluationReport],
) -> Result<ModeSummary, Box<dyn Error>> {
    let count = reports.len();
    if count == 0 {
        return Err(format!("no evaluation reports for {}", mode.name()).into());
    }
    let aggregate = reports.iter().map(|report| &report.aggregate);
    let mut measured_prompt_tokens = 0_u64;
    let mut measured_completion_tokens = 0_u64;
    let mut measured_total_tokens = 0_u64;
    let mut measured_usage_count = 0_usize;
    let mut missing_usage_count = 0_usize;
    for item in aggregate {
        measured_prompt_tokens = measured_prompt_tokens
            .checked_add(item.measured_prompt_tokens)
            .ok_or("prompt token aggregate overflow")?;
        measured_completion_tokens = measured_completion_tokens
            .checked_add(item.measured_completion_tokens)
            .ok_or("completion token aggregate overflow")?;
        measured_total_tokens = measured_total_tokens
            .checked_add(item.measured_total_tokens)
            .ok_or("total token aggregate overflow")?;
        measured_usage_count = measured_usage_count
            .checked_add(item.measured_usage_count)
            .ok_or("usage count overflow")?;
        missing_usage_count = missing_usage_count
            .checked_add(item.missing_usage_count)
            .ok_or("missing usage count overflow")?;
    }
    let aggregates = reports.iter().map(|report| &report.aggregate);
    Ok(ModeSummary {
        mode,
        run_count: count as u32,
        question_count: reports[0].question_count,
        mean_recall_at_k: average_optional(aggregates.clone().map(|item| item.mean_recall_at_k))?,
        mean_reciprocal_rank: average_optional(
            aggregates.clone().map(|item| item.mean_reciprocal_rank),
        )?,
        mean_retrieved_source_precision: average(
            aggregates
                .clone()
                .map(|item| item.mean_retrieved_source_precision),
            count,
        )?,
        mean_expected_term_coverage: average_optional(
            aggregates
                .clone()
                .map(|item| item.mean_expected_term_coverage),
        )?,
        abstention_rate: average_optional(aggregates.clone().map(|item| item.abstention_rate))?,
        mean_retrieval_latency_ms: average(
            aggregates
                .clone()
                .map(|item| item.mean_retrieval_latency_ms),
            count,
        )?,
        mean_answer_latency_ms: average(aggregates.map(|item| item.mean_answer_latency_ms), count)?,
        measured_usage_count,
        missing_usage_count,
        measured_prompt_tokens,
        measured_completion_tokens,
        measured_total_tokens,
    })
}

fn write_report(
    path: &std::path::Path,
    manifest: &Manifest,
    summaries: &[ModeSummary],
) -> Result<(), Box<dyn Error>> {
    let mut report = String::new();
    report.push_str("# GraphRAG-rs 0.3 benchmark report\n\n");
    report.push_str("This is a local benchmark and is not a SOTA certification.\n\n");
    report.push_str(&format!(
        "Manifest `{}`; corpus `{}`; seed {} (server-side/provider seed); runs: {}.\n\n",
        manifest.schema_version, manifest.corpus_id, manifest.seed, manifest.run_count
    ));
    report.push_str("## Retrieval quality\n\n");
    report.push_str("| mode | Recall@3 | MRR | retrieved-source precision | retrieval latency ms |\n|---|---:|---:|---:|---:|\n");
    for summary in summaries {
        report.push_str(&format!(
            "| {} | {} | {} | {:.4} | {:.3} |\n",
            summary.mode.name(),
            display_optional(summary.mean_recall_at_k),
            display_optional(summary.mean_reciprocal_rank),
            summary.mean_retrieved_source_precision,
            summary.mean_retrieval_latency_ms
        ));
    }
    report.push_str("\n## Answer quality and provider usage\n\n");
    report.push_str("| mode | expected-term coverage | abstention rate | answer latency ms | measured usage | missing usage | prompt/completion/total tokens |\n|---|---:|---:|---:|---:|---:|---:|\n");
    for summary in summaries {
        report.push_str(&format!(
            "| {} | {} | {} | {:.3} | {} | {} | {}/{}/{} |\n",
            summary.mode.name(),
            display_optional(summary.mean_expected_term_coverage),
            display_optional(summary.abstention_rate),
            summary.mean_answer_latency_ms,
            summary.measured_usage_count,
            summary.missing_usage_count,
            summary.measured_prompt_tokens,
            summary.measured_completion_tokens,
            summary.measured_total_tokens
        ));
    }
    report.push_str("\nRaw retrieval, answers, context source IDs, optional provider usage, latencies, and per-run reports are stored in the JSON output. Retrieved context IDs are not asserted to be model citations.\n");
    fs::write(path, report)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let (run_count, smoke) = parse_runs(&args[1..])?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/graphrag-0.3");
    let corpus_path = root.join("corpus.json");
    let questions_path = root.join("questions.json");
    let config_path = root.join("config.json");
    let corpus_bytes = fs::read(&corpus_path)?;
    let questions_bytes = fs::read(&questions_path)?;
    let config_bytes = fs::read(&config_path)?;
    let corpus: CorpusFile = serde_json::from_slice(&corpus_bytes)?;
    let questions_file: QuestionsFile = serde_json::from_slice(&questions_bytes)?;
    let benchmark_config: BenchmarkConfig = serde_json::from_slice(&config_bytes)?;
    validate_benchmark_config(&benchmark_config)?;
    if corpus.schema_version != "graphrag-0.3"
        || corpus.documents.len() < 8
        || questions_file.questions.len() != 24
    {
        return Err(
            "benchmark fixture must contain exactly 24 questions and at least 8 documents".into(),
        );
    }
    let graph = graph_from_corpus(&corpus)?;
    let lexical_index = lexical_index(&graph)?;
    let config = runtime_config(&benchmark_config);
    let embedding_provider =
        LlamaCppEmbeddingProvider::from_config(&config.embeddings, config.llm.timeout_seconds)?;
    let generation_client = LlmClient::new(config.llm.clone());
    // Fixed corpus work is done once per process and shared by vector/hybrid.
    let document_embeddings = document_embeddings(&embedding_provider, &graph)?;
    let benchmark_questions = if smoke {
        questions_file.questions.iter().take(1).cloned().collect()
    } else {
        questions_file.questions.clone()
    };
    let mut outputs = Vec::new();
    for run in 1..=run_count {
        let mut query_embeddings = BTreeMap::new();
        for question in &benchmark_questions {
            let started = Instant::now();
            let vector = embedding_provider.embed_query(&question.query)?;
            let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
            query_embeddings.insert(question.id.clone(), (vector, latency_ms));
        }
        for mode in [Mode::Lexical, Mode::Vector, Mode::Hybrid, Mode::Graph] {
            let mut retrieval = Vec::with_capacity(benchmark_questions.len());
            let mut retrieval_output = Vec::with_capacity(benchmark_questions.len());
            let mut answers = Vec::with_capacity(benchmark_questions.len());
            for question in &benchmark_questions {
                let query_embedding = query_embeddings
                    .get(&question.id)
                    .ok_or("missing query embedding")?;
                let retrieval_start = Instant::now();
                let retrieved = match mode {
                    Mode::Lexical => {
                        lexical(&lexical_index, &question.query, benchmark_config.top_k)
                    },
                    Mode::Vector => vector(
                        &graph,
                        &document_embeddings,
                        &query_embedding.0,
                        benchmark_config.top_k,
                    )?,
                    Mode::Hybrid => {
                        let lexical_rows =
                            lexical(&lexical_index, &question.query, benchmark_config.top_k);
                        let vector_rows = vector(
                            &graph,
                            &document_embeddings,
                            &query_embedding.0,
                            benchmark_config.top_k,
                        )?;
                        hybrid(
                            &lexical_rows,
                            &vector_rows,
                            benchmark_config.top_k,
                            benchmark_config.hybrid_fusion.k,
                            benchmark_config.hybrid_fusion.lexical_weight,
                            benchmark_config.hybrid_fusion.vector_weight,
                        )
                    },
                    Mode::Graph => graph_mode(&graph, &question.query, benchmark_config.top_k),
                };
                let retrieval_latency_ms = retrieval_start.elapsed().as_secs_f64() * 1_000.0
                    + if matches!(mode, Mode::Vector | Mode::Hybrid) {
                        query_embedding.1
                    } else {
                        0.0
                    };
                let source_ids = retrieved
                    .iter()
                    .map(|row| row.source_id.clone())
                    .collect::<Vec<_>>();
                let scores = retrieved.iter().map(|row| row.score).collect::<Vec<_>>();
                retrieval.push(RetrievalRecord {
                    question_id: question.id.clone(),
                    source_ids: source_ids.clone(),
                    scores,
                    latency_ms: retrieval_latency_ms,
                });
                retrieval_output.push(RetrievalOutput {
                    question_id: question.id.clone(),
                    source_ids: source_ids.clone(),
                    scores: retrieved.iter().map(|row| row.score).collect(),
                    latency_ms: retrieval_latency_ms,
                    retrieved: retrieved.clone(),
                });
                let prompt = format!(
                    "Answer using only the source context. If it is insufficient, say I don't know.\nQuestion: {}\nContext:\n{}",
                    question.query,
                    bounded_context(&retrieved, benchmark_config.context_limit_chars)
                );
                let answer_start = Instant::now();
                let generated = generation_client.generate_blocking(
                    &prompt,
                    &LlmGenerationParams {
                        max_tokens: config.llm.max_tokens,
                        temperature: config.llm.temperature,
                        ..Default::default()
                    },
                )?;
                let answer_latency_ms = answer_start.elapsed().as_secs_f64() * 1_000.0;
                answers.push(AnswerRecord {
                    question_id: question.id.clone(),
                    answer: generated.text,
                    context_source_ids: source_ids,
                    usage: provider_usage(generated.usage),
                    latency_ms: answer_latency_ms,
                });
            }
            let report = evaluate_benchmark(&BenchmarkEvaluationInput {
                questions: benchmark_questions.clone(),
                retrieval,
                answers: answers.clone(),
                top_k: benchmark_config.top_k,
            })?;
            outputs.push(RunOutput {
                run,
                mode,
                retrieval: retrieval_output,
                answers,
                report,
            });
            println!("completed run {run}/{run_count} {}", mode.name());
        }
    }
    let revisions = benchmark_config.provider_revisions.clone();
    let manifest = Manifest {
        schema_version: if smoke {
            "graphrag-0.3-smoke-v1"
        } else {
            "graphrag-0.3-evaluation-v1"
        }
        .to_string(),
        config_version: benchmark_config.config_version.clone(),
        runner_version: benchmark_config.runner_version.clone(),
        corpus_id: corpus.corpus_id,
        corpus_sha256: sha256_hex(&corpus_bytes),
        questions_sha256: sha256_hex(&questions_bytes),
        config_sha256: sha256_hex(&config_bytes),
        e5_gguf_path: env_or(
            "GRAPHRAG_E5_GGUF",
            "/home/dio/modelli/graphrag-03/multilingual-e5-small-f16-probe.gguf",
        ),
        e5_gguf_sha256: optional_file_sha256(&env_or(
            "GRAPHRAG_E5_GGUF",
            "/home/dio/modelli/graphrag-03/multilingual-e5-small-f16-probe.gguf",
        )),
        ornith_gguf_path: env_or(
            "GRAPHRAG_ORNITH_GGUF",
            "/home/dio/modelli/Ornith-1.5-9B-MTP-Q6_K.gguf",
        ),
        ornith_gguf_sha256: optional_file_sha256(&env_or(
            "GRAPHRAG_ORNITH_GGUF",
            "/home/dio/modelli/Ornith-1.5-9B-MTP-Q6_K.gguf",
        )),
        modes: vec!["lexical", "vector", "hybrid", "graph"],
        top_k: benchmark_config.top_k,
        context_limit_chars: benchmark_config.context_limit_chars,
        hybrid_fusion: format!(
            "weighted reciprocal-rank fusion, k={}, lexical_weight={}, vector_weight={}",
            benchmark_config.hybrid_fusion.k,
            benchmark_config.hybrid_fusion.lexical_weight,
            benchmark_config.hybrid_fusion.vector_weight
        ),
        seed: benchmark_config.seed,
        seed_semantics: "server-side/provider seed",
        run_count,
        e5_model: benchmark_config.embedding.model.clone(),
        e5_endpoint: config.embeddings.api_endpoint.unwrap_or_default(),
        ornith_model: benchmark_config.generation.model.clone(),
        ornith_endpoint: config.llm.chat_completions_url(),
        provider_revisions: revisions,
    };
    let mut summaries = Vec::new();
    for mode in [Mode::Lexical, Mode::Vector, Mode::Hybrid, Mode::Graph] {
        let reports = outputs
            .iter()
            .filter(|output| output.mode == mode)
            .map(|output| &output.report)
            .collect::<Vec<_>>();
        summaries.push(mode_summary(mode, &reports)?);
    }
    let output = SmokeOutput {
        manifest,
        runs: outputs,
        summaries,
    };
    let output_path = root.join(if smoke {
        "results/smoke.json"
    } else {
        "results/evaluation.json"
    });
    fs::write(&output_path, serde_json::to_vec_pretty(&output)?)?;
    let report_path = root.join(if smoke {
        "results/smoke.md"
    } else {
        "results/report.md"
    });
    write_report(&report_path, &output.manifest, &output.summaries)?;
    println!("wrote {}", output_path.display());
    println!("wrote {}", report_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_runs;

    #[test]
    fn accepts_only_explicit_smoke_or_runs() {
        assert_eq!(parse_runs(&["--smoke".into()]).unwrap(), (1, true));
        assert_eq!(
            parse_runs(&["--runs".into(), "3".into()]).unwrap(),
            (3, false)
        );
    }

    #[test]
    fn rejects_unknown_and_repeated_arguments() {
        assert!(parse_runs(&["--unknown".into()]).is_err());
        assert!(parse_runs(&["--runs".into(), "1".into(), "--runs".into(), "2".into()]).is_err());
        assert!(parse_runs(&["--smoke".into(), "--smoke".into()]).is_err());
    }
}
