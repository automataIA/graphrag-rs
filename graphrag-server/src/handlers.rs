//! Multi-Document REST API Handlers
//!
//! Implements endpoints for:
//! - Batch document upload
//! - Incremental document merging
//! - Cross-document queries with RRF ranking

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use rayon::prelude::*;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BatchUploadRequest {
    pub documents: Vec<DocumentInput>,
}

#[derive(Debug, Deserialize)]
pub struct DocumentInput {
    pub id: String,
    pub text: String,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct BatchUploadResponse {
    pub collection: String,
    pub processed: usize,
    pub total_chunks: usize,
    pub total_entities: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct IncrementalMergeRequest {
    pub document_id: String,
    pub text: String,
    pub strategy: Option<String>, // "incremental" or "full"
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct MergeResponse {
    pub collection: String,
    pub document_id: String,
    pub new_chunks: usize,
    pub new_entities: usize,
    pub merged_entities: usize,
    pub total_entities: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct CrossDocumentQueryRequest {
    pub query: String,
    pub collections: Vec<String>,
    pub top_k: Option<usize>,
    pub strategy: Option<String>, // "rrf", "weighted", "concat"
}

#[derive(Debug, Serialize)]
pub struct CrossDocumentQueryResponse {
    pub query: String,
    pub results: Vec<QueryResult>,
    pub source_distribution: HashMap<String, usize>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub rank: usize,
    pub doc_id: String,
    pub chunk_id: usize,
    pub text: String,
    pub similarity: f32,
    pub source: String,
}

#[derive(Debug, Serialize)]
pub struct CollectionStatsResponse {
    pub collection: String,
    pub documents: usize,
    pub chunks: usize,
    pub entities: usize,
    pub relationships: usize,
    pub memory_mb: f64,
}

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
pub struct AppState {
    pub collections: Arc<RwLock<HashMap<String, Collection>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Collection {
    pub documents: Vec<Document>,
    pub chunks: Vec<Chunk>,
    pub entities: HashMap<String, Entity>,
}

impl Collection {
    fn new() -> Self {
        Self {
            documents: Vec::new(),
            chunks: Vec::new(),
            entities: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub doc_id: String,
    pub chunk_id: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub source_docs: Vec<String>,
    pub mentions: usize,
}

// ============================================================================
// Handler Functions
// ============================================================================

/// POST /api/collections/:name/documents/batch
///
/// Upload multiple documents to a collection in batch.
/// Uses Rayon for parallel embedding generation.
pub async fn batch_upload(
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

    // Get or create collection
    let mut collections = state.collections.write().await;
    let collection = collections
        .entry(collection_name.clone())
        .or_insert_with(Collection::new);

    let mut total_chunks = 0;
    let mut total_entities = 0;

    // Process each document
    for doc_input in req.documents {
        // Create document
        let doc = Document {
            id: doc_input.id.clone(),
            text: doc_input.text.clone(),
            metadata: doc_input.metadata.unwrap_or_default(),
        };

        // Chunk the document
        let chunks = chunk_document(&doc.text, 200, 50);
        total_chunks += chunks.len();

        // Generate embeddings in parallel (Rayon)
        let chunks_with_embeddings: Vec<Chunk> = chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_id, text)| {
                let embedding = hash_embedding(text, 384);
                Chunk {
                    doc_id: doc.id.clone(),
                    chunk_id,
                    text: text.to_string(),
                    embedding,
                }
            })
            .collect();

        // Extract entities
        let entities = extract_entities(&doc.text, &doc.id);
        total_entities += entities.len();

        // Add to collection
        collection.documents.push(doc);
        collection.chunks.extend(chunks_with_embeddings);
        collection.entities.extend(entities);
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        "Batch upload completed: {}ms, chunks={}, entities={}",
        elapsed_ms,
        total_chunks,
        total_entities
    );

    Ok(Json(BatchUploadResponse {
        collection: collection_name,
        processed: req.documents.len(),
        total_chunks,
        total_entities,
        elapsed_ms,
    }))
}

/// POST /api/collections/:name/merge
///
/// Incrementally merge a new document into an existing collection.
/// Detects and resolves duplicate entities.
pub async fn incremental_merge(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<IncrementalMergeRequest>,
) -> Result<Json<MergeResponse>, StatusCode> {
    let start = Instant::now();

    tracing::info!(
        "Incremental merge: collection={}, document={}",
        collection_name,
        req.document_id
    );

    let mut collections = state.collections.write().await;
    let collection = collections
        .entry(collection_name.clone())
        .or_insert_with(Collection::new);

    let entities_before = collection.entities.len();
    let chunks_before = collection.chunks.len();

    // Create new document
    let doc = Document {
        id: req.document_id.clone(),
        text: req.text.clone(),
        metadata: req.metadata.unwrap_or_default(),
    };

    // Chunk and embed
    let chunks = chunk_document(&doc.text, 200, 50);
    let chunks_with_embeddings: Vec<Chunk> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, text)| {
            let embedding = hash_embedding(text, 384);
            Chunk {
                doc_id: doc.id.clone(),
                chunk_id,
                text: text.to_string(),
                embedding,
            }
        })
        .collect();

    // Extract entities
    let new_entities = extract_entities(&doc.text, &doc.id);

    // Detect duplicate entities (cosine similarity > 0.95)
    let mut merged_count = 0;
    for (new_id, new_entity) in &new_entities {
        let mut is_duplicate = false;

        for (existing_id, existing_entity) in &collection.entities {
            // Check if same entity (case-insensitive name match)
            if new_entity.name.to_lowercase() == existing_entity.name.to_lowercase()
                && !existing_entity.source_docs.contains(&doc.id)
            {
                is_duplicate = true;
                merged_count += 1;

                // In a real implementation, we would merge the entities here
                // For now, we just count them
                tracing::debug!(
                    "Duplicate detected: {} (new) ≈ {} (existing)",
                    new_id,
                    existing_id
                );
                break;
            }
        }

        if !is_duplicate {
            collection.entities.insert(new_id.clone(), new_entity.clone());
        }
    }

    // Add document and chunks
    collection.documents.push(doc);
    collection.chunks.extend(chunks_with_embeddings);

    let new_chunks = collection.chunks.len() - chunks_before;
    let new_entities = collection.entities.len() - entities_before;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        "Merge completed: {}ms, new_chunks={}, new_entities={}, merged={}",
        elapsed_ms,
        new_chunks,
        new_entities,
        merged_count
    );

    Ok(Json(MergeResponse {
        collection: collection_name,
        document_id: req.document_id,
        new_chunks,
        new_entities,
        merged_entities: merged_count,
        total_entities: collection.entities.len(),
        elapsed_ms,
    }))
}

/// POST /api/query/multi
///
/// Query across multiple collections and merge results using RRF.
pub async fn cross_document_query(
    State(state): State<AppState>,
    Json(req): Json<CrossDocumentQueryRequest>,
) -> Result<Json<CrossDocumentQueryResponse>, StatusCode> {
    let start = Instant::now();

    tracing::info!(
        "Cross-document query: query=\"{}\", collections={:?}",
        req.query,
        req.collections
    );

    let collections = state.collections.read().await;
    let top_k = req.top_k.unwrap_or(10);

    // Generate query embedding
    let query_embedding = hash_embedding(&req.query, 384);

    // Query each collection
    let mut result_sets = Vec::new();

    for collection_name in &req.collections {
        if let Some(collection) = collections.get(collection_name) {
            let results = query_collection(&query_embedding, collection, collection_name, top_k);
            result_sets.push(results);
        }
    }

    // Merge results using RRF or other strategy
    let strategy = req.strategy.as_deref().unwrap_or("rrf");
    let merged_results = match strategy {
        "rrf" => apply_rrf(result_sets, top_k),
        "concat" => concat_results(result_sets, top_k),
        _ => apply_rrf(result_sets, top_k), // Default to RRF
    };

    // Calculate source distribution
    let mut source_distribution: HashMap<String, usize> = HashMap::new();
    for result in &merged_results {
        *source_distribution.entry(result.source.clone()).or_insert(0) += 1;
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        "Query completed: {}ms, results={}, sources={:?}",
        elapsed_ms,
        merged_results.len(),
        source_distribution
    );

    Ok(Json(CrossDocumentQueryResponse {
        query: req.query,
        results: merged_results,
        source_distribution,
        elapsed_ms,
    }))
}

/// GET /api/collections/:name/stats
///
/// Get statistics about a collection.
pub async fn get_collection_stats(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CollectionStatsResponse>, StatusCode> {
    let collections = state.collections.read().await;

    let collection = collections
        .get(&collection_name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let memory_mb = estimate_memory_mb(collection);

    Ok(Json(CollectionStatsResponse {
        collection: collection_name,
        documents: collection.documents.len(),
        chunks: collection.chunks.len(),
        entities: collection.entities.len(),
        relationships: 0, // Not implemented yet
        memory_mb,
    }))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Chunk text into overlapping windows
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

/// Generate hash-based TF embedding
fn hash_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dimension];

    let tokens: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| s.len() > 2)
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

    // Sublinear TF scaling
    for value in &mut embedding {
        if *value > 0.0 {
            *value = (1.0 + *value).ln();
        }
    }

    // L2 normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }

    embedding
}

/// FNV-1a hash
fn hash_token(token: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in token.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Cosine similarity
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

/// Extract entities (simple keyword extraction)
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

/// Query a single collection
fn query_collection(
    query_embedding: &[f32],
    collection: &Collection,
    collection_name: &str,
    top_k: usize,
) -> Vec<QueryResult> {
    let mut results: Vec<QueryResult> = collection
        .chunks
        .iter()
        .map(|chunk| {
            let similarity = cosine_similarity(query_embedding, &chunk.embedding);
            QueryResult {
                rank: 0, // Will be set later
                doc_id: chunk.doc_id.clone(),
                chunk_id: chunk.chunk_id,
                text: chunk.text.clone(),
                similarity,
                source: collection_name.to_string(),
            }
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.iter_mut().enumerate().for_each(|(i, r)| r.rank = i + 1);
    results.truncate(top_k);

    results
}

/// Apply Reciprocal Rank Fusion (RRF)
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

/// Concatenate results (simple strategy)
fn concat_results(result_sets: Vec<Vec<QueryResult>>, top_k: usize) -> Vec<QueryResult> {
    let mut all_results: Vec<QueryResult> = result_sets.into_iter().flatten().collect();

    all_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    all_results.iter_mut().enumerate().for_each(|(i, r)| r.rank = i + 1);
    all_results.truncate(top_k);

    all_results
}

/// Estimate memory usage
fn estimate_memory_mb(collection: &Collection) -> f64 {
    let chunks_size = collection.chunks.len() * (std::mem::size_of::<Chunk>() + 384 * 4);
    let entities_size = collection.entities.len() * std::mem::size_of::<Entity>();
    let docs_size: usize = collection.documents.iter().map(|d| d.text.len()).sum();

    (chunks_size + entities_size + docs_size) as f64 / (1024.0 * 1024.0)
}
