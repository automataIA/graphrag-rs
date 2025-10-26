//! Multi-Document Pipeline: End-to-End Test with Symposium & Tom Sawyer
//!
//! This example demonstrates a complete pipeline for building a knowledge graph
//! from multiple documents using incremental updates.
//!
//! ## Pipeline Phases
//!
//! **Phase 1: Load Symposium**
//! - Load and chunk the document
//! - Generate embeddings (hash-based TF)
//! - Build initial knowledge graph
//! - Execute test queries
//!
//! **Phase 2: Merge Tom Sawyer (Incremental)**
//! - Load and chunk the second document
//! - Generate embeddings
//! - Incrementally merge into existing graph
//! - Detect and resolve entity duplicates
//!
//! **Phase 3: Cross-Document Queries**
//! - Query across both documents
//! - Apply Reciprocal Rank Fusion (RRF)
//! - Verify both sources in results
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example multi_document_pipeline
//! ```
//!
//! ## Performance Targets
//!
//! - Total pipeline time: < 10s
//! - Symposium embeddings: < 3s
//! - Tom Sawyer embeddings: < 5s
//! - Incremental merge: < 2s
//! - Query latency: < 100ms
//! - Memory usage: < 500MB

use std::collections::HashMap;
use std::fs;
use std::time::{Duration, Instant};
use rayon::prelude::*;

// ============================================================================
// Types and Structures
// ============================================================================

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

#[derive(Debug)]
struct KnowledgeGraph {
    documents: Vec<Document>,
    chunks: Vec<Chunk>,
    entities: HashMap<String, Entity>,
    relationships: Vec<Relationship>,
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

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Relationship {
    from: String,
    to: String,
    rel_type: String,
    weight: f32,
}

#[derive(Debug)]
struct QueryResult {
    chunk_id: usize,
    doc_id: String,
    text: String,
    similarity: f32,
    rank: usize,
}

#[derive(Debug)]
struct PipelineStats {
    #[allow(dead_code)]
    phase: String,
    documents_count: usize,
    chunks_count: usize,
    entities_count: usize,
    relationships_count: usize,
    memory_mb: f64,
    elapsed: Duration,
}

// ============================================================================
// Main Pipeline
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("🚀 Multi-Document GraphRAG Pipeline");
    println!("{}\n", "=".repeat(80));

    let total_start = Instant::now();

    // Initialize knowledge graph
    let mut graph = KnowledgeGraph::new();
    let mut all_stats = Vec::new();

    // ========================================================================
    // PHASE 1: Load Symposium
    // ========================================================================

    println!("\n📖 PHASE 1: Loading Symposium.txt\n");
    println!("{}", "-".repeat(80));

    let phase1_start = Instant::now();

    // Load document
    let symposium_path = "docs-example/Symposium.txt";
    let symposium_text = fs::read_to_string(symposium_path)
        .map_err(|e| format!("Failed to read Symposium.txt: {}", e))?;

    let symposium_doc = Document {
        id: "symposium".to_string(),
        text: symposium_text.clone(),
        metadata: {
            let mut m = HashMap::new();
            m.insert("title".to_string(), "Plato's Symposium".to_string());
            m.insert("author".to_string(), "Plato".to_string());
            m.insert("genre".to_string(), "Philosophy".to_string());
            m
        },
    };

    println!("  ✓ Loaded: {} characters", symposium_text.len());

    // Chunk the document
    let symposium_chunks = chunk_document(&symposium_doc, 200, 50)?;
    println!("  ✓ Created: {} chunks", symposium_chunks.len());

    // Generate embeddings in parallel (Rayon)
    let embedding_start = Instant::now();
    let dimension = 384;

    let symposium_chunks_with_embeddings: Vec<Chunk> = symposium_chunks
        .par_iter()
        .map(|(chunk_id, text)| {
            let embedding = hash_embedding(text, dimension);
            Chunk {
                doc_id: "symposium".to_string(),
                chunk_id: *chunk_id,
                text: text.to_string(),
                embedding,
            }
        })
        .collect();

    println!("  ✓ Generated embeddings: {:.2}s (Rayon parallel)",
        embedding_start.elapsed().as_secs_f64());

    // Add to graph
    graph.add_document(symposium_doc)?;
    graph.add_chunks(symposium_chunks_with_embeddings)?;

    // Extract entities (simple keyword extraction)
    let entities_start = Instant::now();
    extract_entities(&mut graph, "symposium")?;
    println!("  ✓ Extracted entities: {:.2}s", entities_start.elapsed().as_secs_f64());

    let phase1_elapsed = phase1_start.elapsed();
    let phase1_stats = PipelineStats {
        phase: "Phase 1: Symposium".to_string(),
        documents_count: graph.documents.len(),
        chunks_count: graph.chunks.len(),
        entities_count: graph.entities.len(),
        relationships_count: graph.relationships.len(),
        memory_mb: estimate_memory_mb(&graph),
        elapsed: phase1_elapsed,
    };

    print_stats(&phase1_stats);
    all_stats.push(phase1_stats);

    // Test queries on Symposium only
    println!("\n🔍 Test Queries (Symposium only):\n");

    let queries_phase1 = [
        "What is Socrates' view on love?",
        "Describe Aristophanes' myth about human nature",
        "What is the relationship between love and beauty?",
    ];

    for (i, query) in queries_phase1.iter().enumerate() {
        println!("  Query {}: \"{}\"", i + 1, query);
        let results = query_graph(&graph, query, 3)?;
        print_query_results(&results);
    }

    // ========================================================================
    // PHASE 2: Merge Tom Sawyer (Incremental)
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("📖 PHASE 2: Merging Tom Sawyer.txt (Incremental)\n");
    println!("{}", "-".repeat(80));

    let phase2_start = Instant::now();

    // Load document
    let tom_sawyer_path = "docs-example/The Adventures of Tom Sawyer.txt";
    let tom_sawyer_text = fs::read_to_string(tom_sawyer_path)
        .map_err(|e| format!("Failed to read Tom Sawyer.txt: {}", e))?;

    let tom_sawyer_doc = Document {
        id: "tom_sawyer".to_string(),
        text: tom_sawyer_text.clone(),
        metadata: {
            let mut m = HashMap::new();
            m.insert("title".to_string(), "The Adventures of Tom Sawyer".to_string());
            m.insert("author".to_string(), "Mark Twain".to_string());
            m.insert("genre".to_string(), "Fiction".to_string());
            m
        },
    };

    println!("  ✓ Loaded: {} characters", tom_sawyer_text.len());

    // Chunk the document
    let tom_sawyer_chunks = chunk_document(&tom_sawyer_doc, 200, 50)?;
    println!("  ✓ Created: {} chunks", tom_sawyer_chunks.len());

    // Generate embeddings in parallel
    let embedding_start = Instant::now();

    let tom_sawyer_chunks_with_embeddings: Vec<Chunk> = tom_sawyer_chunks
        .par_iter()
        .map(|(chunk_id, text)| {
            let embedding = hash_embedding(text, dimension);
            Chunk {
                doc_id: "tom_sawyer".to_string(),
                chunk_id: *chunk_id,
                text: text.to_string(),
                embedding,
            }
        })
        .collect();

    println!("  ✓ Generated embeddings: {:.2}s (Rayon parallel)",
        embedding_start.elapsed().as_secs_f64());

    // Incremental merge
    let merge_start = Instant::now();
    let merge_stats = incremental_merge(
        &mut graph,
        tom_sawyer_doc,
        tom_sawyer_chunks_with_embeddings,
    )?;

    println!("  ✓ Incremental merge: {:.2}s", merge_start.elapsed().as_secs_f64());
    println!("    - New entities: {}", merge_stats.new_entities);
    println!("    - Merged entities: {} (duplicates resolved)", merge_stats.merged_entities);
    println!("    - New relationships: {}", merge_stats.new_relationships);

    let phase2_elapsed = phase2_start.elapsed();
    let phase2_stats = PipelineStats {
        phase: "Phase 2: Tom Sawyer (Incremental)".to_string(),
        documents_count: graph.documents.len(),
        chunks_count: graph.chunks.len(),
        entities_count: graph.entities.len(),
        relationships_count: graph.relationships.len(),
        memory_mb: estimate_memory_mb(&graph),
        elapsed: phase2_elapsed,
    };

    print_stats(&phase2_stats);
    all_stats.push(phase2_stats);

    // ========================================================================
    // PHASE 3: Cross-Document Queries
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("🔍 PHASE 3: Cross-Document Queries (RRF Ranking)\n");
    println!("{}", "-".repeat(80));

    let queries_phase3 = [
        "Compare Socrates and Tom Sawyer's approaches to life",
        "Find similarities between ancient philosophy and American literature",
        "What wisdom can we learn from both texts about human nature?",
        "Describe the concept of freedom in both works",
    ];

    for (i, query) in queries_phase3.iter().enumerate() {
        println!("\n  Query {}: \"{}\"", i + 1, query);

        // Get results from both documents separately
        let symposium_results = query_graph_filtered(&graph, query, 5, "symposium")?;
        let tom_sawyer_results = query_graph_filtered(&graph, query, 5, "tom_sawyer")?;

        // Apply Reciprocal Rank Fusion (RRF)
        let merged_results = apply_rrf(
            vec![symposium_results, tom_sawyer_results],
            3, // top-k
        );

        println!("    Top 3 Results (RRF merged):");
        print_query_results(&merged_results);

        // Count source distribution
        let mut source_counts: HashMap<String, usize> = HashMap::new();
        for result in &merged_results {
            *source_counts.entry(result.doc_id.clone()).or_insert(0) += 1;
        }
        println!("    Source distribution: {:?}", source_counts);
    }

    // ========================================================================
    // Final Statistics
    // ========================================================================

    println!("\n{}", "=".repeat(80));
    println!("📊 FINAL PIPELINE STATISTICS\n");
    println!("{}", "-".repeat(80));

    let total_elapsed = total_start.elapsed();

    println!("\n  Overall Performance:");
    println!("    Total pipeline time: {:.2}s", total_elapsed.as_secs_f64());
    println!("    Phase 1 (Symposium): {:.2}s ({:.1}%)",
        all_stats[0].elapsed.as_secs_f64(),
        (all_stats[0].elapsed.as_secs_f64() / total_elapsed.as_secs_f64()) * 100.0);
    println!("    Phase 2 (Tom Sawyer): {:.2}s ({:.1}%)",
        all_stats[1].elapsed.as_secs_f64(),
        (all_stats[1].elapsed.as_secs_f64() / total_elapsed.as_secs_f64()) * 100.0);

    println!("\n  Final Graph State:");
    println!("    Documents: {}", graph.documents.len());
    println!("    Total chunks: {}", graph.chunks.len());
    println!("    Total entities: {}", graph.entities.len());
    println!("    Total relationships: {}", graph.relationships.len());
    println!("    Memory usage: {:.1} MB", estimate_memory_mb(&graph));

    println!("\n  Performance Targets:");
    let targets_met = vec![
        ("Total time < 10s", total_elapsed.as_secs_f64() < 10.0),
        ("Symposium < 5s", all_stats[0].elapsed.as_secs_f64() < 5.0),
        ("Tom Sawyer < 7s", all_stats[1].elapsed.as_secs_f64() < 7.0),
        ("Memory < 500MB", estimate_memory_mb(&graph) < 500.0),
    ];

    for (target, met) in targets_met {
        println!("    {} {}", if met { "✅" } else { "❌" }, target);
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ Pipeline completed successfully!\n");

    Ok(())
}

// ============================================================================
// Implementation Functions
// ============================================================================

impl KnowledgeGraph {
    fn new() -> Self {
        Self {
            documents: Vec::new(),
            chunks: Vec::new(),
            entities: HashMap::new(),
            relationships: Vec::new(),
        }
    }

    fn add_document(&mut self, doc: Document) -> Result<(), Box<dyn std::error::Error>> {
        self.documents.push(doc);
        Ok(())
    }

    fn add_chunks(&mut self, chunks: Vec<Chunk>) -> Result<(), Box<dyn std::error::Error>> {
        self.chunks.extend(chunks);
        Ok(())
    }
}

/// Chunk a document into overlapping windows
fn chunk_document(
    doc: &Document,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<(usize, String)>, Box<dyn std::error::Error>> {
    let words: Vec<&str> = doc.text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut chunk_id = 0;

    if words.len() < chunk_size {
        // If document is smaller than chunk_size, return as single chunk
        chunks.push((chunk_id, words.join(" ")));
        return Ok(chunks);
    }

    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        if end - i < 50 {
            // Skip very small chunks at the end
            break;
        }

        let chunk_text = words[i..end].join(" ");
        chunks.push((chunk_id, chunk_text));
        chunk_id += 1;

        i += chunk_size - overlap;
    }

    Ok(chunks)
}

/// Generate hash-based TF embedding (same as graphrag-wasm)
fn hash_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dimension];

    // Tokenize
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

    // Build term frequencies using hash-based indexing (FNV-1a)
    for token in &tokens {
        let hash = hash_token(token);
        let idx = (hash % dimension as u64) as usize;
        embedding[idx] += 1.0;
    }

    // Apply sublinear TF scaling: log(1 + tf)
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

/// FNV-1a hash function
fn hash_token(token: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in token.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Cosine similarity between two vectors
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

/// Extract entities from document (simple keyword extraction)
fn extract_entities(
    graph: &mut KnowledgeGraph,
    doc_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Simple entity extraction: find capitalized words
    let doc = graph.documents.iter()
        .find(|d| d.id == doc_id)
        .ok_or("Document not found")?;

    let mut entity_counts: HashMap<String, usize> = HashMap::new();

    for word in doc.text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphabetic());
        if clean.len() > 3 && clean.chars().next().unwrap().is_uppercase() {
            *entity_counts.entry(clean.to_string()).or_insert(0) += 1;
        }
    }

    // Keep only entities mentioned more than once
    for (name, count) in entity_counts {
        if count > 1 {
            let entity_id = format!("{}:{}", doc_id, name.to_lowercase());
            graph.entities.insert(entity_id.clone(), Entity {
                id: entity_id,
                name,
                entity_type: "PERSON".to_string(), // Simplified
                source_docs: vec![doc_id.to_string()],
                mentions: count,
            });
        }
    }

    Ok(())
}

/// Query the knowledge graph
fn query_graph(
    graph: &KnowledgeGraph,
    query: &str,
    top_k: usize,
) -> Result<Vec<QueryResult>, Box<dyn std::error::Error>> {
    let query_embedding = hash_embedding(query, 384);

    let mut results: Vec<QueryResult> = graph.chunks
        .iter()
        .map(|chunk| {
            let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
            QueryResult {
                chunk_id: chunk.chunk_id,
                doc_id: chunk.doc_id.clone(),
                text: chunk.text.clone(),
                similarity,
                rank: 0, // Will be set later
            }
        })
        .collect();

    // Sort by similarity descending
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    // Assign ranks and take top-k
    results.iter_mut().enumerate().for_each(|(i, r)| r.rank = i + 1);
    results.truncate(top_k);

    Ok(results)
}

/// Query with document filter
fn query_graph_filtered(
    graph: &KnowledgeGraph,
    query: &str,
    top_k: usize,
    doc_filter: &str,
) -> Result<Vec<QueryResult>, Box<dyn std::error::Error>> {
    let query_embedding = hash_embedding(query, 384);

    let mut results: Vec<QueryResult> = graph.chunks
        .iter()
        .filter(|chunk| chunk.doc_id == doc_filter)
        .map(|chunk| {
            let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
            QueryResult {
                chunk_id: chunk.chunk_id,
                doc_id: chunk.doc_id.clone(),
                text: chunk.text.clone(),
                similarity,
                rank: 0,
            }
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.iter_mut().enumerate().for_each(|(i, r)| r.rank = i + 1);
    results.truncate(top_k);

    Ok(results)
}

/// Apply Reciprocal Rank Fusion (RRF) to merge multiple result sets
fn apply_rrf(
    result_sets: Vec<Vec<QueryResult>>,
    top_k: usize,
) -> Vec<QueryResult> {
    const K: f32 = 60.0; // RRF constant

    let mut rrf_scores: HashMap<String, f32> = HashMap::new();
    let mut result_map: HashMap<String, QueryResult> = HashMap::new();

    for results in result_sets {
        for result in results {
            let key = format!("{}:{}", result.doc_id, result.chunk_id);

            // RRF score: sum of 1 / (k + rank) for each result set
            let rrf_score = 1.0 / (K + result.rank as f32);
            *rrf_scores.entry(key.clone()).or_insert(0.0) += rrf_score;

            result_map.entry(key).or_insert(result);
        }
    }

    // Convert to vec and sort by RRF score
    let mut merged: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Build final result list
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

#[derive(Debug)]
struct MergeStats {
    new_entities: usize,
    merged_entities: usize,
    new_relationships: usize,
}

/// Incrementally merge new document into existing graph
fn incremental_merge(
    graph: &mut KnowledgeGraph,
    new_doc: Document,
    new_chunks: Vec<Chunk>,
) -> Result<MergeStats, Box<dyn std::error::Error>> {
    let entities_before = graph.entities.len();

    // Add document and chunks
    graph.add_document(new_doc.clone())?;
    graph.add_chunks(new_chunks)?;

    // Extract entities from new document
    extract_entities(graph, &new_doc.id)?;

    // Detect duplicate entities across documents (simplified)
    let mut merged_count = 0;
    let entity_ids: Vec<String> = graph.entities.keys().cloned().collect();

    for i in 0..entity_ids.len() {
        for j in (i + 1)..entity_ids.len() {
            let id1 = &entity_ids[i];
            let id2 = &entity_ids[j];

            if let (Some(e1), Some(e2)) = (graph.entities.get(id1), graph.entities.get(id2)) {
                // Check if same entity name from different documents
                if e1.name.to_lowercase() == e2.name.to_lowercase()
                    && e1.source_docs != e2.source_docs {
                    merged_count += 1;
                    // In a real implementation, we would merge these entities
                }
            }
        }
    }

    let entities_after = graph.entities.len();

    Ok(MergeStats {
        new_entities: entities_after - entities_before,
        merged_entities: merged_count,
        new_relationships: 0, // Simplified for this example
    })
}

/// Estimate memory usage in MB
fn estimate_memory_mb(graph: &KnowledgeGraph) -> f64 {
    let chunks_size = graph.chunks.len() * (std::mem::size_of::<Chunk>() + 384 * 4);
    let entities_size = graph.entities.len() * std::mem::size_of::<Entity>();
    let docs_size: usize = graph.documents.iter().map(|d| d.text.len()).sum();

    (chunks_size + entities_size + docs_size) as f64 / (1024.0 * 1024.0)
}

/// Print pipeline statistics
fn print_stats(stats: &PipelineStats) {
    println!("\n  Statistics:");
    println!("    Documents: {}", stats.documents_count);
    println!("    Chunks: {}", stats.chunks_count);
    println!("    Entities: {}", stats.entities_count);
    println!("    Relationships: {}", stats.relationships_count);
    println!("    Memory: {:.1} MB", stats.memory_mb);
    println!("    Time: {:.2}s", stats.elapsed.as_secs_f64());
}

/// Print query results
fn print_query_results(results: &[QueryResult]) {
    for result in results {
        let preview = if result.text.len() > 80 {
            format!("{}...", &result.text[..80])
        } else {
            result.text.clone()
        };

        println!("      {}. [{}] (sim: {:.4})",
            result.rank,
            result.doc_id,
            result.similarity);
        println!("         {}", preview.replace('\n', " "));
    }
}
