//! Multi-Document GraphRAG REST API Server
//!
//! This server provides REST endpoints for multi-document knowledge graph operations:
//! - Batch document upload
//! - Incremental document merging
//! - Cross-document queries with RRF ranking
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example graphrag_multi_doc_server
//! ```
//!
//! ## API Endpoints
//!
//! - POST   /api/collections/:name/documents/batch
//! - POST   /api/collections/:name/merge
//! - POST   /api/query/multi
//! - GET    /api/collections/:name/stats
//! - GET    /health
//!
//! ## Example Requests
//!
//! ```bash
//! # 1. Batch upload Symposium
//! curl -X POST http://localhost:3000/api/collections/classics/documents/batch \
//!   -H "Content-Type: application/json" \
//!   -d '{"documents": [{"id": "symposium", "text": "..."}]}'
//!
//! # 2. Incremental merge Tom Sawyer
//! curl -X POST http://localhost:3000/api/collections/classics/merge \
//!   -H "Content-Type: application/json" \
//!   -d '{"document_id": "tom_sawyer", "text": "..."}'
//!
//! # 3. Cross-document query
//! curl -X POST http://localhost:3000/api/query/multi \
//!   -H "Content-Type: application/json" \
//!   -d '{"query": "philosophy and freedom", "collections": ["classics"]}'
//! ```

use axum::{
    extract::{Path, State},
    http::{Method, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use rayon::prelude::*;

// Include the handlers module inline (for standalone example)
// In a real app, this would be in a separate file

// ============================================================================
// Types
// ============================================================================

#[derive(Clone)]
struct AppState {
    collections: Arc<RwLock<HashMap<String, Collection>>>,
}

#[derive(Debug, Clone)]
struct Collection {
    documents: Vec<Document>,
    chunks: Vec<Chunk>,
    entities: HashMap<String, Entity>,
}

#[derive(Debug, Clone)]
struct Document {
    id: String,
    text: String,
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct Chunk {
    doc_id: String,
    chunk_id: usize,
    text: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Entity {
    #[allow(dead_code)]
    id: String,
    name: String,
    #[allow(dead_code)]
    entity_type: String,
    source_docs: Vec<String>,
    #[allow(dead_code)]
    mentions: usize,
}

#[derive(Debug, Deserialize)]
struct BatchUploadRequest {
    documents: Vec<DocumentInput>,
}

#[derive(Debug, Deserialize)]
struct DocumentInput {
    id: String,
    text: String,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
struct BatchUploadResponse {
    collection: String,
    processed: usize,
    total_chunks: usize,
    total_entities: usize,
    elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
struct IncrementalMergeRequest {
    document_id: String,
    text: String,
    #[allow(dead_code)]
    strategy: Option<String>,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
struct MergeResponse {
    collection: String,
    document_id: String,
    new_chunks: usize,
    new_entities: usize,
    merged_entities: usize,
    total_entities: usize,
    elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
struct CrossDocumentQueryRequest {
    query: String,
    collections: Vec<String>,
    top_k: Option<usize>,
    #[allow(dead_code)]
    strategy: Option<String>,
}

#[derive(Debug, Serialize)]
struct CrossDocumentQueryResponse {
    query: String,
    results: Vec<QueryResult>,
    source_distribution: HashMap<String, usize>,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct QueryResult {
    rank: usize,
    doc_id: String,
    chunk_id: usize,
    text: String,
    similarity: f32,
    source: String,
}

#[derive(Debug, Serialize)]
struct CollectionStatsResponse {
    collection: String,
    documents: usize,
    chunks: usize,
    entities: usize,
    relationships: usize,
    memory_mb: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_seconds: u64,
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let state = AppState {
        collections: Arc::new(RwLock::new(HashMap::new())),
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route(
            "/api/collections/:name/documents/batch",
            post(batch_upload),
        )
        .route("/api/collections/:name/merge", post(incremental_merge))
        .route("/api/query/multi", post(cross_document_query))
        .route("/api/collections/:name/stats", get(get_collection_stats))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers(Any),
        )
        .with_state(state);

    let addr = "0.0.0.0:3000";
    println!("\n🚀 GraphRAG Multi-Document Server");
    println!("📡 Listening on http://{}\n", addr);
    println!("API Endpoints:");
    println!("  POST   /api/collections/:name/documents/batch");
    println!("  POST   /api/collections/:name/merge");
    println!("  POST   /api/query/multi");
    println!("  GET    /api/collections/:name/stats");
    println!("  GET    /health\n");
    println!("Example: Load Symposium and Tom Sawyer");
    println!("  ./examples/load_documents.sh\n");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ============================================================================
// Handlers
// ============================================================================

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // TODO: Track actual uptime
    })
}

async fn batch_upload(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<BatchUploadRequest>,
) -> Result<Json<BatchUploadResponse>, StatusCode> {
    let start = Instant::now();

    tracing::info!(
        "Batch upload: collection={}, documents={}",
        collection_name,
        req.documents.len()
    );

    let mut collections = state.collections.write().await;
    let collection = collections
        .entry(collection_name.clone())
        .or_insert_with(|| Collection {
            documents: Vec::new(),
            chunks: Vec::new(),
            entities: HashMap::new(),
        });

    let mut total_chunks = 0;
    let mut total_entities = 0;
    let num_documents = req.documents.len();

    for doc_input in req.documents {
        let doc = Document {
            id: doc_input.id.clone(),
            text: doc_input.text.clone(),
            metadata: doc_input.metadata.unwrap_or_default(),
        };

        let chunks = chunk_document(&doc.text, 200, 50);
        total_chunks += chunks.len();

        let chunks_with_embeddings: Vec<Chunk> = chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_id, text)| Chunk {
                doc_id: doc.id.clone(),
                chunk_id,
                text: text.to_string(),
                embedding: hash_embedding(text, 384),
            })
            .collect();

        let entities = extract_entities(&doc.text, &doc.id);
        total_entities += entities.len();

        collection.documents.push(doc);
        collection.chunks.extend(chunks_with_embeddings);
        collection.entities.extend(entities);
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(Json(BatchUploadResponse {
        collection: collection_name,
        processed: num_documents,
        total_chunks,
        total_entities,
        elapsed_ms,
    }))
}

async fn incremental_merge(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<IncrementalMergeRequest>,
) -> Result<Json<MergeResponse>, StatusCode> {
    let start = Instant::now();

    let mut collections = state.collections.write().await;
    let collection = collections
        .entry(collection_name.clone())
        .or_insert_with(|| Collection {
            documents: Vec::new(),
            chunks: Vec::new(),
            entities: HashMap::new(),
        });

    let entities_before = collection.entities.len();
    let chunks_before = collection.chunks.len();

    let doc = Document {
        id: req.document_id.clone(),
        text: req.text.clone(),
        metadata: req.metadata.unwrap_or_default(),
    };

    let chunks = chunk_document(&doc.text, 200, 50);
    let chunks_with_embeddings: Vec<Chunk> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, text)| Chunk {
            doc_id: doc.id.clone(),
            chunk_id,
            text: text.to_string(),
            embedding: hash_embedding(text, 384),
        })
        .collect();

    let new_entities = extract_entities(&doc.text, &doc.id);

    let mut merged_count = 0;
    for (new_id, new_entity) in &new_entities {
        let mut is_duplicate = false;
        for existing_entity in collection.entities.values() {
            if new_entity.name.to_lowercase() == existing_entity.name.to_lowercase()
                && !existing_entity.source_docs.contains(&doc.id)
            {
                is_duplicate = true;
                merged_count += 1;
                break;
            }
        }
        if !is_duplicate {
            collection.entities.insert(new_id.clone(), new_entity.clone());
        }
    }

    collection.documents.push(doc);
    collection.chunks.extend(chunks_with_embeddings);

    Ok(Json(MergeResponse {
        collection: collection_name,
        document_id: req.document_id,
        new_chunks: collection.chunks.len() - chunks_before,
        new_entities: collection.entities.len() - entities_before,
        merged_entities: merged_count,
        total_entities: collection.entities.len(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn cross_document_query(
    State(state): State<AppState>,
    Json(req): Json<CrossDocumentQueryRequest>,
) -> Result<Json<CrossDocumentQueryResponse>, StatusCode> {
    let start = Instant::now();

    let collections = state.collections.read().await;
    let top_k = req.top_k.unwrap_or(10);
    let query_embedding = hash_embedding(&req.query, 384);

    let mut result_sets = Vec::new();

    for collection_name in &req.collections {
        if let Some(collection) = collections.get(collection_name) {
            let results = query_collection(&query_embedding, collection, collection_name, top_k);
            result_sets.push(results);
        }
    }

    let merged_results = apply_rrf(result_sets, top_k);

    let mut source_distribution: HashMap<String, usize> = HashMap::new();
    for result in &merged_results {
        *source_distribution.entry(result.source.clone()).or_insert(0) += 1;
    }

    Ok(Json(CrossDocumentQueryResponse {
        query: req.query,
        results: merged_results,
        source_distribution,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn get_collection_stats(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CollectionStatsResponse>, StatusCode> {
    let collections = state.collections.read().await;
    let collection = collections.get(&collection_name).ok_or(StatusCode::NOT_FOUND)?;

    let memory_mb = estimate_memory_mb(collection);

    Ok(Json(CollectionStatsResponse {
        collection: collection_name,
        documents: collection.documents.len(),
        chunks: collection.chunks.len(),
        entities: collection.entities.len(),
        relationships: 0,
        memory_mb,
    }))
}

// ============================================================================
// Helper Functions
// ============================================================================

fn chunk_document(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();

    if words.len() < chunk_size {
        return vec![words.join(" ")];
    }

    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        if end - i < 50 {
            break;
        }
        chunks.push(words[i..end].join(" "));
        i += chunk_size - overlap;
    }

    chunks
}

fn hash_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dimension];
    let tokens: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 2)
        .map(|s| s.to_string())
        .collect();

    if tokens.is_empty() {
        return embedding;
    }

    for token in &tokens {
        let hash = hash_token(token);
        let idx = (hash % dimension as u64) as usize;
        embedding[idx] += 1.0;
    }

    for value in &mut embedding {
        if *value > 0.0 {
            *value = (1.0 + *value).ln();
        }
    }

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }

    embedding
}

fn hash_token(token: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in token.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }
    dot_product / (magnitude_a * magnitude_b)
}

fn extract_entities(text: &str, doc_id: &str) -> HashMap<String, Entity> {
    let mut entity_counts: HashMap<String, usize> = HashMap::new();

    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphabetic());
        if clean.len() > 3 && clean.chars().next().unwrap().is_uppercase() {
            *entity_counts.entry(clean.to_string()).or_insert(0) += 1;
        }
    }

    let mut entities = HashMap::new();
    for (name, count) in entity_counts {
        if count > 1 {
            let entity_id = format!("{}:{}", doc_id, name.to_lowercase());
            entities.insert(
                entity_id.clone(),
                Entity {
                    id: entity_id,
                    name,
                    entity_type: "PERSON".to_string(),
                    source_docs: vec![doc_id.to_string()],
                    mentions: count,
                },
            );
        }
    }

    entities
}

fn query_collection(
    query_embedding: &[f32],
    collection: &Collection,
    collection_name: &str,
    top_k: usize,
) -> Vec<QueryResult> {
    let mut results: Vec<QueryResult> = collection
        .chunks
        .iter()
        .map(|chunk| QueryResult {
            rank: 0,
            doc_id: chunk.doc_id.clone(),
            chunk_id: chunk.chunk_id,
            text: chunk.text.clone(),
            similarity: cosine_similarity(query_embedding, &chunk.embedding),
            source: collection_name.to_string(),
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.iter_mut().enumerate().for_each(|(i, r)| r.rank = i + 1);
    results.truncate(top_k);
    results
}

fn apply_rrf(result_sets: Vec<Vec<QueryResult>>, top_k: usize) -> Vec<QueryResult> {
    const K: f32 = 60.0;
    let mut rrf_scores: HashMap<String, f32> = HashMap::new();
    let mut result_map: HashMap<String, QueryResult> = HashMap::new();

    for results in result_sets {
        for result in results {
            let key = format!("{}:{}:{}", result.source, result.doc_id, result.chunk_id);
            let rrf_score = 1.0 / (K + result.rank as f32);
            *rrf_scores.entry(key.clone()).or_insert(0.0) += rrf_score;
            result_map.entry(key).or_insert(result);
        }
    }

    let mut merged: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    merged
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(i, (key, _))| {
            let mut result = result_map.remove(&key).unwrap();
            result.rank = i + 1;
            result
        })
        .collect()
}

fn estimate_memory_mb(collection: &Collection) -> f64 {
    let chunks_size = collection.chunks.len() * (std::mem::size_of::<Chunk>() + 384 * 4);
    let entities_size = collection.entities.len() * std::mem::size_of::<Entity>();
    let docs_size: usize = collection.documents.iter().map(|d| d.text.len()).sum();
    (chunks_size + entities_size + docs_size) as f64 / (1024.0 * 1024.0)
}
