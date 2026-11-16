//! Retrieval system: finds the chunks and entities relevant to a query.
//!
//! Combines keyword search (BM25), vector similarity, and PageRank-weighted graph
//! traversal, and can return `explained` answers with a source/reasoning trace.

pub mod adaptive;
/// BM25 text retrieval implementation for keyword-based search
pub mod bm25;
/// Structured answer types with reasoning trace (`ExplainedAnswer`, `SourceReference`, `ReasoningStep`)
pub mod explained;
/// Core retrieval data types (`SearchResult`, `RetrievalConfig`, query analysis enums, statistics).
mod types;
pub use types::*;
/// Causal chain analysis for discovering cause-effect paths (Phase 2.3)
pub mod causal_analysis;
/// Enriched metadata-aware retrieval
pub mod enriched;
/// HippoRAG Personalized PageRank retrieval
#[cfg(feature = "pagerank")]
pub mod hipporag_ppr;
/// Hybrid retrieval combining multiple search strategies
pub mod hybrid;
pub mod pagerank_retrieval;
/// Symbolic anchoring for conceptual queries (Phase 2.1 - CatRAG)
pub mod symbolic_anchoring;

#[cfg(feature = "parallel-processing")]
use crate::parallel::ParallelProcessor;
use crate::{
    config::Config,
    core::{ChunkId, EntityId, KnowledgeGraph},
    summarization::DocumentTree,
    vector::{EmbeddingGenerator, VectorUtils},
    Result,
};
use std::collections::{HashMap, HashSet};

pub use bm25::{BM25Result, BM25Retriever, Document as BM25Document};
pub use enriched::{EnrichedRetrievalConfig, EnrichedRetriever};
pub use explained::{ExplainedAnswer, ReasoningStep, SourceReference, SourceType};
pub use hybrid::{FusionMethod, HybridConfig, HybridRetriever, HybridSearchResult};

#[cfg(feature = "pagerank")]
pub use pagerank_retrieval::{PageRankRetrievalSystem, ScoredResult};

#[cfg(feature = "pagerank")]
pub use hipporag_ppr::{Fact, HippoRAGConfig, HippoRAGRetriever};

use crate::vector::store::VectorStore;

fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Reciprocal Rank Fusion constant (standard value).
const RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion over several rankings of ids (best first).
///
/// Each id scores `sum(1 / (k + rank))` over the rankings that contain it (rank is
/// 1-based). Returns ids sorted by descending fused score; ties keep first-seen order.
fn reciprocal_rank_fusion(rankings: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut order: Vec<String> = Vec::new();
    let mut scores: HashMap<String, f32> = HashMap::new();
    for ranking in rankings {
        for (idx, id) in ranking.iter().enumerate() {
            let entry = scores.entry(id.clone()).or_insert_with(|| {
                order.push(id.clone());
                0.0
            });
            *entry += 1.0 / (k + idx as f32 + 1.0);
        }
    }
    let mut fused: Vec<(String, f32)> = order
        .into_iter()
        .map(|id| {
            let score = scores[&id];
            (id, score)
        })
        .collect();
    // Stable sort keeps first-seen order for equal scores.
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// BM25 index over the graph's chunks, tagged with a fingerprint of the chunks it was
/// built from so it is rebuilt whenever the graph changes (new build or workspace reload).
struct ChunkBm25Index {
    fingerprint: u64,
    retriever: BM25Retriever,
}

/// Cheap fingerprint of the graph's chunks (ids + content), independent of map order.
fn chunk_fingerprint(graph: &KnowledgeGraph) -> u64 {
    use std::hash::{Hash, Hasher};
    graph
        .chunks()
        .map(|chunk| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            chunk.id.to_string().hash(&mut hasher);
            chunk.content.hash(&mut hasher);
            hasher.finish()
        })
        .fold(0u64, |acc, h| acc.wrapping_add(h))
}

/// Retrieval system for querying the knowledge graph
pub struct RetrievalSystem {
    vector_store: std::sync::Arc<dyn VectorStore>,
    embedding_generator: EmbeddingGenerator,
    #[cfg(feature = "llm")]
    llama_cpp_provider: Option<std::sync::Arc<crate::llama_cpp::LlamaCppEmbeddingProvider>>,
    config: RetrievalConfig,
    #[cfg(feature = "parallel-processing")]
    parallel_processor: Option<ParallelProcessor>,
    #[cfg(feature = "pagerank")]
    pagerank_retriever: Option<PageRankRetrievalSystem>,
    enriched_retriever: Option<EnrichedRetriever>,
    /// Lazily built keyword index over chunk text (see `keyword_chunk_ranking`)
    chunk_bm25: Option<ChunkBm25Index>,
}

impl RetrievalSystem {
    /// Create a new retrieval system
    pub fn new(config: &Config) -> Result<Self> {
        let retrieval_config = RetrievalConfig {
            top_k: config.retrieval.top_k,
            similarity_threshold: 0.35,
            max_expansion_depth: 2,
            entity_weight: 0.4,
            chunk_weight: 0.4,
            graph_weight: 0.2,
            #[cfg(feature = "lazygraphrag")]
            use_concept_filtering: false,
            #[cfg(feature = "lazygraphrag")]
            concept_top_k: 20,
        };

        // Default to MemoryVectorStore for now (mimics old behavior)
        // In the future, this will select based on Config (LanceDB, Qdrant, etc.)
        let vector_store =
            std::sync::Arc::new(crate::vector::memory_store::MemoryVectorStore::new());

        #[cfg(feature = "llm")]
        let llama_cpp_provider = if config.embeddings.backend.eq_ignore_ascii_case("llama_cpp")
            || config.embeddings.backend.eq_ignore_ascii_case("llama.cpp")
        {
            Some(std::sync::Arc::new(
                crate::llama_cpp::LlamaCppEmbeddingProvider::from_config(
                    &config.embeddings,
                    config.llm.timeout_seconds,
                )?,
            ))
        } else {
            None
        };

        #[cfg(not(feature = "llm"))]
        if config.embeddings.backend.eq_ignore_ascii_case("llama_cpp")
            || config.embeddings.backend.eq_ignore_ascii_case("llama.cpp")
        {
            return Err(crate::core::GraphRAGError::Unsupported {
                operation: "llama.cpp embeddings".to_string(),
                reason: "enable the graphrag-core `llm` feature".to_string(),
            });
        }

        if !config.embeddings.backend.eq_ignore_ascii_case("hash")
            && !config.embeddings.backend.eq_ignore_ascii_case("llama_cpp")
            && !config.embeddings.backend.eq_ignore_ascii_case("llama.cpp")
        {
            return Err(crate::core::GraphRAGError::Unsupported {
                operation: "embedding provider".to_string(),
                reason: format!(
                    "embedding backend '{}' is not wired into the retrieval pipeline",
                    config.embeddings.backend
                ),
            });
        }

        Ok(Self {
            vector_store,
            embedding_generator: EmbeddingGenerator::new(config.embeddings.dimension),
            #[cfg(feature = "llm")]
            llama_cpp_provider,
            config: retrieval_config,
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
            #[cfg(feature = "pagerank")]
            pagerank_retriever: None,
            enriched_retriever: None,
            chunk_bm25: None,
        })
    }
}

impl RetrievalSystem {
    /// Create a new retrieval system with parallel processing support
    #[cfg(feature = "parallel-processing")]
    pub fn with_parallel_processing(
        vector_store: std::sync::Arc<dyn VectorStore>,
        embedding_generator: EmbeddingGenerator,
        parallel_processor: ParallelProcessor,
    ) -> Result<Self> {
        // VectorStore trait is already Send + Sync and wrapped in Arc
        // Can be safely used across threads for parallel operations
        // EmbeddingGenerator operations can be parallelized with rayon

        let retrieval_config = RetrievalConfig::default();

        Ok(Self {
            vector_store,
            embedding_generator,
            #[cfg(feature = "llm")]
            llama_cpp_provider: None,
            config: retrieval_config,
            parallel_processor: Some(parallel_processor),
            #[cfg(feature = "pagerank")]
            pagerank_retriever: None,
            enriched_retriever: None,
            chunk_bm25: None,
        })
    }

    fn embed_query_text(&mut self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "llm")]
        if let Some(provider) = &self.llama_cpp_provider {
            return provider.embed_query(text);
        }
        Ok(self.embedding_generator.generate_embedding(text))
    }

    fn embed_document_text(&mut self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "llm")]
        if let Some(provider) = &self.llama_cpp_provider {
            return provider.embed_document(text);
        }
        Ok(self.embedding_generator.generate_embedding(text))
    }

    async fn embed_query_text_async(&mut self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "llm")]
        if let Some(provider) = self.llama_cpp_provider.clone() {
            let text = text.to_string();
            #[cfg(feature = "async")]
            return tokio::task::spawn_blocking(move || provider.embed_query(&text))
                .await
                .map_err(|e| crate::core::GraphRAGError::Embedding {
                    message: format!("llama.cpp embedding worker failed: {e}"),
                })?;
            #[cfg(not(feature = "async"))]
            return provider.embed_query(&text);
        }
        self.embed_query_text(text)
    }

    async fn embed_document_text_async(&mut self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "llm")]
        if let Some(provider) = self.llama_cpp_provider.clone() {
            let text = text.to_string();
            #[cfg(feature = "async")]
            return tokio::task::spawn_blocking(move || provider.embed_document(&text))
                .await
                .map_err(|e| crate::core::GraphRAGError::Embedding {
                    message: format!("llama.cpp embedding worker failed: {e}"),
                })?;
            #[cfg(not(feature = "async"))]
            return provider.embed_document(&text);
        }
        self.embed_document_text(text)
    }

    /// Embed a query with the configured embedding backend (used to rank context chunks).
    pub async fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_query_text_async(text).await
    }

    /// Index a knowledge graph for retrieval
    pub async fn index_graph(&self, graph: &KnowledgeGraph) -> Result<()> {
        // Index entity embeddings
        for entity in graph.entities() {
            if let Some(embedding) = &entity.embedding {
                let id = format!("entity:{}", entity.id);
                // Simple empty metadata for now, could add name/type
                self.vector_store
                    .add_vector(&id, embedding.clone(), HashMap::new())
                    .await?;
            }
        }

        // Index chunk embeddings
        for chunk in graph.chunks() {
            if let Some(embedding) = &chunk.embedding {
                let id = format!("chunk:{}", chunk.id);
                self.vector_store
                    .add_vector(&id, embedding.clone(), HashMap::new())
                    .await?;
            }
        }

        // Initialize/Build if needed (some stores might need explicit commit)
        self.vector_store.initialize().await?;

        Ok(())
    }

    /// Initialize PageRank retrieval system (feature-gated)
    #[cfg(feature = "pagerank")]
    pub fn initialize_pagerank(&mut self, graph: &KnowledgeGraph) -> Result<()> {
        use crate::graph::pagerank::{PageRankConfig, ScoreWeights};

        #[cfg(feature = "tracing")]
        tracing::debug!("Initializing high-performance PageRank retrieval system...");

        let pagerank_config = PageRankConfig {
            damping_factor: 0.85,
            max_iterations: 50, // Reduced for faster convergence
            tolerance: 1e-5,    // Slightly relaxed for speed
            personalized: true,
            #[cfg(feature = "parallel-processing")]
            parallel_enabled: self.parallel_processor.is_some(),
            #[cfg(not(feature = "parallel-processing"))]
            parallel_enabled: false,
            cache_size: 2000, // Large cache for better performance
            sparse_threshold: 500,
            incremental_updates: true,
            simd_block_size: 64, // Optimized for modern CPUs
        };

        let score_weights = ScoreWeights {
            vector_weight: 0.3,
            pagerank_weight: 0.5, // Higher weight for PageRank like fast-GraphRAG
            chunk_weight: 0.15,
            relationship_weight: 0.05,
        };

        let mut pagerank_retriever = PageRankRetrievalSystem::new(self.config.top_k)
            .with_pagerank_config(pagerank_config)
            .with_score_weights(score_weights)
            .with_incremental_mode(true)
            .with_min_threshold(0.05);

        // Initialize vector index
        // pagerank_retriever.initialize_vector_index(graph)?;

        // Pre-compute global PageRank scores for faster queries
        pagerank_retriever.precompute_global_pagerank(graph)?;

        self.pagerank_retriever = Some(pagerank_retriever);

        #[cfg(feature = "tracing")]
        tracing::debug!("PageRank retrieval system initialized with 27x performance optimizations");
        Ok(())
    }

    /// Initialize enriched metadata-aware retrieval system
    pub fn initialize_enriched(&mut self, config: Option<EnrichedRetrievalConfig>) -> Result<()> {
        #[cfg(feature = "tracing")]
        tracing::debug!("Initializing enriched metadata-aware retrieval system...");

        let enriched_config = config.unwrap_or_default();
        let enriched_retriever = EnrichedRetriever::with_config(enriched_config);

        self.enriched_retriever = Some(enriched_retriever);

        #[cfg(feature = "tracing")]
        tracing::debug!("Enriched retrieval system initialized with metadata boosting");
        Ok(())
    }

    /// Query using PageRank-enhanced retrieval (feature-gated)
    #[cfg(feature = "pagerank")]
    pub fn pagerank_query(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
        max_results: Option<usize>,
    ) -> Result<Vec<ScoredResult>> {
        if let Some(pagerank_retriever) = &self.pagerank_retriever {
            pagerank_retriever.search_with_pagerank(query, graph, max_results)
        } else {
            Err(crate::core::GraphRAGError::Retrieval {
                message: "PageRank retriever not initialized. Call initialize_pagerank() first."
                    .to_string(),
            })
        }
    }

    /// Batch PageRank queries for high throughput (feature-gated)
    #[cfg(feature = "pagerank")]
    pub fn pagerank_batch_query(
        &self,
        queries: &[&str],
        graph: &KnowledgeGraph,
        max_results_per_query: Option<usize>,
    ) -> Result<Vec<Vec<ScoredResult>>> {
        if let Some(pagerank_retriever) = &self.pagerank_retriever {
            pagerank_retriever.batch_search(queries, graph, max_results_per_query)
        } else {
            Err(crate::core::GraphRAGError::Retrieval {
                message: "PageRank retriever not initialized. Call initialize_pagerank() first."
                    .to_string(),
            })
        }
    }

    /// Query an explicit graph using deterministic local retrieval.
    ///
    /// This path intentionally does not consult a hidden graph or an LLM.  It scores chunks
    /// by query-term overlap, boosts chunks mentioning matching entities, and includes the
    /// matching graph facts as source context.
    pub fn query(&self, query: &str, graph: &KnowledgeGraph) -> Result<Vec<String>> {
        let query_terms = Self::local_query_terms(query);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut matching_entities: Vec<_> = graph
            .entities()
            .filter(|entity| {
                let entity_terms = Self::local_query_terms(&entity.name);
                entity_terms.iter().any(|term| query_terms.contains(term))
            })
            .collect();
        // KnowledgeGraph stores entities in petgraph, whose iteration order is an
        // implementation detail and may differ after persistence.  Sort every graph
        // derived collection before using it to construct the answer.
        matching_entities.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        let mut ranked: Vec<(f32, String, String)> = graph
            .chunks()
            .filter_map(|chunk| {
                let chunk_terms = Self::local_query_terms(&chunk.content);
                let overlap = query_terms
                    .iter()
                    .filter(|term| chunk_terms.contains(term))
                    .count();
                let mentioned_entity_boost = matching_entities
                    .iter()
                    .filter(|entity| {
                        entity
                            .mentions
                            .iter()
                            .any(|mention| mention.chunk_id == chunk.id)
                    })
                    .count() as f32
                    * 0.25;
                let score = overlap as f32 / query_terms.len() as f32 + mentioned_entity_boost;
                (score > 0.0).then(|| (score, chunk.id.to_string(), chunk.content.clone()))
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        ranked.truncate(self.config.top_k);

        let mut results = ranked
            .into_iter()
            .map(|(score, id, content)| {
                format!("[Source chunk {id} | score: {score:.3}] {content}")
            })
            .collect::<Vec<_>>();

        // A query can match an entity name while its source chunk has no exact query term
        // (for example an inflected or abbreviated question). Include its graph facts so the
        // answer remains grounded in the explicit graph supplied by the caller.
        for entity in matching_entities {
            let mut neighbors = graph
                .get_neighbors(&entity.id)
                .into_iter()
                .map(|(target, relationship)| {
                    format!(
                        "{} {} {}",
                        entity.name, relationship.relation_type, target.name
                    )
                })
                .collect::<Vec<_>>();
            neighbors.sort();
            neighbors.dedup();

            let mut source_chunks = entity
                .mentions
                .iter()
                .map(|mention| mention.chunk_id.to_string())
                .collect::<Vec<_>>();
            source_chunks.sort();
            source_chunks.dedup();
            let fact = if neighbors.is_empty() {
                entity.name.clone()
            } else {
                neighbors.join("; ")
            };
            let suffix = if source_chunks.is_empty() {
                String::new()
            } else {
                format!(" | sources: {}", source_chunks.join(", "))
            };
            let marker = format!("[Graph entity {}", entity.id);
            if !results.iter().any(|result| result.contains(&marker)) {
                results.push(format!("[Graph entity {}{suffix}] {fact}", entity.id));
            }
        }

        // Keep the local result deterministic and bounded even when many entities match.
        results.truncate(self.config.top_k.max(1));
        Ok(results)
    }

    fn local_query_terms(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|term| term.len() >= 2)
            .map(str::to_lowercase)
            .filter(|term| {
                !matches!(
                    term.as_str(),
                    "the"
                        | "and"
                        | "are"
                        | "who"
                        | "what"
                        | "how"
                        | "does"
                        | "is"
                        | "il"
                        | "la"
                        | "chi"
                        | "che"
                )
            })
            .collect()
    }

    /// Advanced hybrid query with strategy selection and hierarchical integration
    pub async fn hybrid_query(
        &mut self,
        query: &str,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        self.hybrid_query_with_trees(query, graph, &HashMap::new())
            .await
    }

    /// Hybrid query with access to document trees for hierarchical retrieval
    pub async fn hybrid_query_with_trees(
        &mut self,
        query: &str,
        graph: &KnowledgeGraph,
        document_trees: &HashMap<crate::core::DocumentId, DocumentTree>,
    ) -> Result<Vec<SearchResult>> {
        // 1. Analyze query to determine optimal strategy
        let analysis = self.analyze_query(query, graph)?;

        // 2. Generate query embedding
        let query_embedding = self.embed_query_text_async(query).await?;

        // 3. Execute multi-strategy retrieval based on analysis
        let mut results = self
            .execute_adaptive_retrieval(query, &query_embedding, graph, document_trees, &analysis)
            .await?;

        // 4. Apply enriched metadata-aware boosting and filtering if enabled
        if let Some(enriched_retriever) = &self.enriched_retriever {
            // First apply metadata boosting to enhance relevance
            results = enriched_retriever.boost_with_metadata(results, query, graph)?;

            // Then apply structure filtering if query mentions chapters/sections
            results = enriched_retriever.filter_by_structure(query, results, graph)?;
        }

        Ok(results)
    }

    /// Query the system using hybrid retrieval (vector + graph) - legacy method
    pub async fn legacy_hybrid_query(
        &mut self,
        query: &str,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        // 1. Generate query embedding
        let query_embedding = self.embed_query_text_async(query).await?;

        // 2. Perform comprehensive search
        let results = self.comprehensive_search(&query_embedding, graph).await?;

        Ok(results)
    }

    /// Add embeddings to chunks and entities in the graph with parallel processing
    pub async fn add_embeddings_to_graph(&mut self, graph: &mut KnowledgeGraph) -> Result<()> {
        #[cfg(feature = "parallel-processing")]
        if let Some(processor) = self.parallel_processor.clone() {
            return self.add_embeddings_parallel(graph, &processor).await;
        }

        self.add_embeddings_sequential(graph).await
    }

    /// Parallel embedding generation with proper error handling and work-stealing
    #[cfg(feature = "parallel-processing")]
    async fn add_embeddings_parallel(
        &mut self,
        graph: &mut KnowledgeGraph,
        processor: &ParallelProcessor,
    ) -> Result<()> {
        // Extract texts for embedding generation
        let mut chunk_texts = Vec::new();
        let mut entity_texts = Vec::new();

        // Collect chunk texts that need embeddings
        for chunk in graph.chunks() {
            if chunk.embedding.is_none() {
                chunk_texts.push((chunk.id.clone(), chunk.content.clone()));
            }
        }

        // Collect entity texts that need embeddings
        for entity in graph.entities() {
            if entity.embedding.is_none() {
                let entity_text = format!("{} {}", entity.name, entity.entity_type);
                entity_texts.push((entity.id.clone(), entity_text));
            }
        }

        // For parallel processing, we need to use a different approach since
        // generate_embedding requires &mut self. We'll fall back to enhanced sequential
        // processing with better chunking and monitoring for now.

        let total_items = chunk_texts.len() + entity_texts.len();
        if processor.should_use_parallel(total_items) {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "Processing {total_items} embeddings with enhanced sequential approach"
            );
        }

        // Process chunks
        for (chunk_id, text) in chunk_texts {
            let embedding = self.embed_document_text_async(&text).await?;
            if let Some(chunk) = graph.get_chunk_mut(&chunk_id) {
                chunk.embedding = Some(embedding);
            }
        }

        // Process entities
        for (entity_id, text) in entity_texts {
            let embedding = self.embed_document_text_async(&text).await?;
            if let Some(entity) = graph.get_entity_mut(&entity_id) {
                entity.embedding = Some(embedding);
            }
        }

        // Re-index the graph with new embeddings
        self.index_graph(graph).await?;

        Ok(())
    }

    /// Sequential embedding generation (fallback)
    #[cfg_attr(not(feature = "tracing"), allow(unused_assignments, unused_variables))]
    async fn add_embeddings_sequential(&mut self, graph: &mut KnowledgeGraph) -> Result<()> {
        // Debug: Check total counts first (uncomment for debugging)
        let _total_chunks = graph.chunks().count();
        let _total_entities = graph.entities().count();
        // println!("DEBUG: Found {} total chunks and {} total entities in graph", _total_chunks, _total_entities);

        // Generate embeddings for all chunks
        let mut chunk_count = 0;
        for chunk in graph.chunks_mut() {
            if chunk.embedding.is_none() {
                let embedding = self.embed_document_text_async(&chunk.content).await?;
                chunk.embedding = Some(embedding);
                chunk_count += 1;
            }
        }

        // Generate embeddings for all entities (using their name and context)
        let mut entity_count = 0;
        for entity in graph.entities_mut() {
            if entity.embedding.is_none() {
                // Create entity text from name and entity type
                let entity_text = format!("{} {}", entity.name, entity.entity_type);
                let embedding = self.embed_document_text_async(&entity_text).await?;
                entity.embedding = Some(embedding);
                entity_count += 1;
            }
        }

        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Generated embeddings for {chunk_count} chunks and {entity_count} entities"
        );

        // Re-index the graph with new embeddings
        // Re-index the graph with new embeddings
        self.index_graph(graph).await?;

        Ok(())
    }

    /// Parallel batch query processing with optimized workload distribution
    /// Batch process multiple queries efficiently
    #[cfg(feature = "parallel-processing")]
    pub async fn batch_query(
        &mut self,
        queries: &[&str],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let processor =
            self.parallel_processor
                .as_ref()
                .ok_or_else(|| crate::core::GraphRAGError::Config {
                    message: "Parallel processor not initialized".to_string(),
                })?;

        if !processor.should_use_parallel(queries.len()) {
            let mut results = Vec::new();
            for &query in queries {
                results.push(self.hybrid_query(query, graph).await?);
            }
            return Ok(results);
        }

        let chunk_size = processor.config().chunk_batch_size.min(queries.len());
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Processing {} queries with enhanced sequential approach (chunk size: {})",
            queries.len(),
            chunk_size
        );

        let mut all_results = Vec::new();
        for &query in queries {
            match self.hybrid_query(query, graph).await {
                Ok(results) => all_results.push(results),
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("Error processing query '{query}': {e}");
                    all_results.push(Vec::new());
                },
            }
        }

        Ok(all_results)
    }

    /// Sequential batch query (fallback when parallel-processing is disabled)
    #[cfg(not(feature = "parallel-processing"))]
    pub async fn batch_query(
        &mut self,
        queries: &[&str],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let mut results = Vec::new();
        for &query in queries {
            results.push(self.hybrid_query(query, graph).await?);
        }
        Ok(results)
    }

    /// Analyze query to determine optimal retrieval strategy
    pub fn analyze_query(&self, query: &str, graph: &KnowledgeGraph) -> Result<QueryAnalysis> {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        // Detect key entities mentioned in the query (word-level match, deduplicated)
        let key_entities = self.detect_key_entities(&query_lower, graph);

        // Extract concepts (non-entity meaningful words)
        let concepts: Vec<String> = words
            .iter()
            .filter(|&&word| word.len() > 3 && !self.is_stop_word(word))
            .filter(|&&word| {
                !key_entities.iter().any(|entity| {
                    entity.to_lowercase().contains(word) || word.contains(&entity.to_lowercase())
                })
            })
            .map(|&word| word.to_string())
            .collect();

        // Determine query type
        let query_type = if !key_entities.is_empty() && key_entities.len() > 1 {
            QueryType::Relationship
        } else if !key_entities.is_empty() {
            QueryType::EntityFocused
        } else if self.has_abstract_concepts(&words) {
            QueryType::Conceptual
        } else if self.has_question_words(&words) {
            QueryType::Exploratory
        } else {
            QueryType::Factual
        };

        // Determine intent
        let intent = if words
            .iter()
            .any(|&w| ["overview", "summary", "general", "about"].contains(&w))
        {
            QueryIntent::Overview
        } else if words
            .iter()
            .any(|&w| ["detailed", "specific", "exactly", "precise"].contains(&w))
        {
            QueryIntent::Detailed
        } else if words
            .iter()
            .any(|&w| ["compare", "vs", "versus", "between", "difference"].contains(&w))
        {
            QueryIntent::Comparative
        } else if words
            .iter()
            .any(|&w| ["cause", "why", "because", "lead", "result"].contains(&w))
        {
            QueryIntent::Causal
        } else if words
            .iter()
            .any(|&w| ["when", "time", "before", "after", "during"].contains(&w))
        {
            QueryIntent::Temporal
        } else {
            QueryIntent::Detailed
        };

        // Calculate complexity score
        let complexity_score = (words.len() as f32 * 0.1
            + key_entities.len() as f32 * 0.3
            + concepts.len() as f32 * 0.2)
            .min(1.0);

        Ok(QueryAnalysis {
            query_type,
            key_entities,
            concepts,
            intent,
            complexity_score,
        })
    }

    /// Execute adaptive retrieval based on query analysis
    pub async fn execute_adaptive_retrieval(
        &mut self,
        query: &str,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
        document_trees: &HashMap<crate::core::DocumentId, DocumentTree>,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        let mut all_results = Vec::new();

        // Strategy weights based on query analysis
        let (vector_weight, graph_weight, hierarchical_weight) =
            self.calculate_strategy_weights(analysis);

        // 1. Vector similarity search (always included)
        if vector_weight > 0.0 {
            let vector_results = self
                .vector_similarity_search(query_embedding, graph)
                .await?;
            // Fuse the dense chunk ranking with a BM25 keyword ranking (RRF); the
            // fused chunk scores are already on the weighted scale.
            let keyword_ids =
                self.keyword_chunk_ranking(query, graph, self.config.top_k.max(1) * 2);
            let mut vector_results =
                self.fuse_chunk_results(vector_results, keyword_ids, graph, vector_weight);
            for result in &mut vector_results {
                if result.result_type != ResultType::Chunk {
                    result.score *= vector_weight;
                }
            }
            all_results.extend(vector_results);
        }

        // 2. Graph-based search (emphasized for entity and relationship queries)
        if graph_weight > 0.0 {
            let mut graph_results = match analysis.query_type {
                QueryType::EntityFocused | QueryType::Relationship => {
                    self.entity_centric_search(query_embedding, graph, &analysis.key_entities)?
                },
                _ => self.entity_based_search(query_embedding, graph)?,
            };
            for result in &mut graph_results {
                result.score *= graph_weight;
            }
            all_results.extend(graph_results);
        }

        // 3. Hierarchical search (emphasized for overview and conceptual queries)
        if hierarchical_weight > 0.0 && !document_trees.is_empty() {
            let mut hierarchical_results =
                self.hierarchical_search(query, document_trees, analysis)?;
            for result in &mut hierarchical_results {
                result.score *= hierarchical_weight;
            }
            all_results.extend(hierarchical_results);
        }

        // 4. Advanced graph traversal for complex queries
        if analysis.complexity_score > 0.7 {
            let traversal_results =
                self.advanced_graph_traversal(query_embedding, graph, analysis)?;
            all_results.extend(traversal_results);
        }

        // 5. Cross-strategy fusion for hybrid results
        Self::cross_strategy_fusion(&mut all_results);

        // Final ranking and deduplication
        let final_results = self.adaptive_rank_and_deduplicate(all_results, analysis)?;

        Ok(final_results.into_iter().take(self.config.top_k).collect())
    }

    /// Comprehensive search that combines multiple retrieval strategies (legacy)
    pub async fn comprehensive_search(
        &self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        let mut all_results = Vec::new();

        // 1. Vector similarity search
        let vector_results = self
            .vector_similarity_search(query_embedding, graph)
            .await?;
        all_results.extend(vector_results);

        // 2. Entity-based search
        let entity_results = self.entity_based_search(query_embedding, graph)?;
        all_results.extend(entity_results);

        // 3. Graph traversal search
        let graph_results = self.graph_traversal_search(query_embedding, graph)?;
        all_results.extend(graph_results);

        // Deduplicate and rank results
        let final_results = self.rank_and_deduplicate(all_results)?;

        Ok(final_results.into_iter().take(self.config.top_k).collect())
    }

    /// Vector similarity search
    async fn vector_similarity_search(
        &self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Search for similar vectors
        // Note: vector_store returns SearchResult struct from store module, we need to convert or us it
        // The store::SearchResult is slightly different from retrieval::SearchResult (metadata map vs specific fields)
        let similar_vectors = self
            .vector_store
            .search(query_embedding, self.config.top_k * 2)
            .await?;

        for store_result in similar_vectors {
            let id = store_result.id;
            let similarity = store_result.score;
            if similarity >= self.config.similarity_threshold {
                let result = if id.starts_with("entity:") {
                    let entity_id = EntityId::new(
                        id.strip_prefix("entity:")
                            .expect("prefix checked")
                            .to_string(),
                    );
                    graph.get_entity(&entity_id).map(|entity| SearchResult {
                        id: entity.id.to_string(),
                        content: entity.name.clone(),
                        score: similarity * self.config.entity_weight,
                        result_type: ResultType::Entity,
                        entities: vec![entity.name.clone()],
                        source_chunks: entity
                            .mentions
                            .iter()
                            .map(|m| m.chunk_id.to_string())
                            .collect(),
                    })
                } else if id.starts_with("chunk:") {
                    let chunk_id = ChunkId::new(
                        id.strip_prefix("chunk:")
                            .expect("prefix checked")
                            .to_string(),
                    );
                    if let Some(chunk) = graph.get_chunk(&chunk_id) {
                        let entity_names: Vec<String> = chunk
                            .entities
                            .iter()
                            .filter_map(|eid| graph.get_entity(eid))
                            .map(|e| e.name.clone())
                            .collect();

                        Some(SearchResult {
                            id: chunk.id.to_string(),
                            content: chunk.content.clone(),
                            score: similarity * self.config.chunk_weight,
                            result_type: ResultType::Chunk,
                            entities: entity_names,
                            source_chunks: vec![chunk.id.to_string()],
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(search_result) = result {
                    results.push(search_result);
                }
            }
        }

        Ok(results)
    }

    /// RRF-fused dense + BM25 chunk ranking for a query, best first, at most `limit` ids.
    ///
    /// Unlike `hybrid_query` this is not cut to `top_k` and ignores entity results: it is
    /// meant to widen the candidate pool of a reranker. `query_embedding` must come from
    /// the same query (see `embed_query`).
    pub async fn fused_chunk_candidates(
        &mut self,
        query: &str,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
        limit: usize,
    ) -> Result<Vec<String>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        // Over-fetch: the store also holds entity vectors, which are dropped here.
        let dense_ids: Vec<String> = self
            .vector_store
            .search(query_embedding, limit * 3)
            .await?
            .into_iter()
            .filter_map(|r| r.id.strip_prefix("chunk:").map(str::to_string))
            .collect();
        let keyword_ids = self.keyword_chunk_ranking(query, graph, limit);
        Ok(reciprocal_rank_fusion(&[dense_ids, keyword_ids], RRF_K)
            .into_iter()
            .take(limit)
            .map(|(id, _)| id)
            .collect())
    }

    /// Rank chunks by BM25 over their text; returns chunk ids, best first.
    ///
    /// The index is built lazily from the graph's chunks and rebuilt whenever their
    /// fingerprint changes, so it works after `build_graph` and after a workspace reload.
    fn keyword_chunk_ranking(
        &mut self,
        query: &str,
        graph: &KnowledgeGraph,
        limit: usize,
    ) -> Vec<String> {
        let fingerprint = chunk_fingerprint(graph);
        if self
            .chunk_bm25
            .as_ref()
            .is_none_or(|index| index.fingerprint != fingerprint)
        {
            let mut retriever = BM25Retriever::new();
            for chunk in graph.chunks() {
                // Ids are unique per chunk, so indexing cannot fail in practice.
                let _ = retriever.index_document(BM25Document {
                    id: chunk.id.to_string(),
                    content: chunk.content.clone(),
                    metadata: HashMap::new(),
                });
            }
            self.chunk_bm25 = Some(ChunkBm25Index {
                fingerprint,
                retriever,
            });
        }
        self.chunk_bm25
            .as_ref()
            .map(|index| {
                index
                    .retriever
                    .search(query, limit)
                    .into_iter()
                    .map(|r| r.doc_id)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Replace the dense chunk results with an RRF fusion of the dense and keyword
    /// chunk rankings. Non-chunk results pass through unchanged.
    ///
    /// Fused scores are `rrf / (2 / (k + 1))` (1.0 = rank 1 in both rankings) times the
    /// chunk and strategy weights, i.e. the same scale as dense-only chunk scores.
    fn fuse_chunk_results(
        &self,
        vector_results: Vec<SearchResult>,
        keyword_ids: Vec<String>,
        graph: &KnowledgeGraph,
        vector_weight: f32,
    ) -> Vec<SearchResult> {
        if keyword_ids.is_empty() {
            return vector_results;
        }
        let (mut dense_chunks, others): (Vec<_>, Vec<_>) = vector_results
            .into_iter()
            .partition(|r| r.result_type == ResultType::Chunk);
        dense_chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let dense_ids: Vec<String> = dense_chunks.iter().map(|r| r.id.clone()).collect();

        let max_rrf = 2.0 / (RRF_K + 1.0);
        let scale = self.config.chunk_weight * vector_weight / max_rrf;
        let mut by_id: HashMap<String, SearchResult> = dense_chunks
            .into_iter()
            .map(|r| (r.id.clone(), r))
            .collect();

        let mut fused_results = others;
        for (id, rrf) in reciprocal_rank_fusion(&[dense_ids, keyword_ids], RRF_K) {
            let existing = by_id.remove(&id);
            let result = existing.or_else(|| {
                graph
                    .get_chunk(&ChunkId::new(id.clone()))
                    .map(|chunk| SearchResult {
                        id: chunk.id.to_string(),
                        content: chunk.content.clone(),
                        score: 0.0,
                        result_type: ResultType::Chunk,
                        entities: chunk
                            .entities
                            .iter()
                            .filter_map(|eid| graph.get_entity(eid))
                            .map(|e| e.name.clone())
                            .collect(),
                        source_chunks: vec![chunk.id.to_string()],
                    })
            });
            if let Some(mut result) = result {
                result.score = rrf * scale;
                fused_results.push(result);
            }
        }
        fused_results
    }

    /// Entity-based search with graph expansion
    fn entity_based_search(
        &self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();

        // Find most relevant entities
        let entity_similarities = self.find_relevant_entities(query_embedding, graph)?;

        for (entity_id, similarity) in entity_similarities.into_iter().take(5) {
            if visited.contains(&entity_id) {
                continue;
            }

            // Expand through graph relationships
            let expanded_entities = self.expand_through_relationships(
                &entity_id,
                graph,
                self.config.max_expansion_depth,
                &mut visited,
            )?;

            for expanded_entity_id in expanded_entities {
                if let Some(entity) = graph.get_entity(&expanded_entity_id) {
                    let expansion_penalty = if expanded_entity_id == entity_id {
                        1.0
                    } else {
                        0.8
                    };

                    results.push(SearchResult {
                        id: entity.id.to_string(),
                        content: format!("{} ({})", entity.name, entity.entity_type),
                        score: similarity * expansion_penalty * self.config.entity_weight,
                        result_type: ResultType::Entity,
                        entities: vec![entity.name.clone()],
                        source_chunks: entity
                            .mentions
                            .iter()
                            .map(|m| m.chunk_id.to_string())
                            .collect(),
                    });
                }
            }
        }

        Ok(results)
    }

    /// Calculate strategy weights based on query analysis
    fn calculate_strategy_weights(&self, analysis: &QueryAnalysis) -> (f32, f32, f32) {
        match (&analysis.query_type, &analysis.intent) {
            // For entity-focused queries, balance vector (chunks) and graph (entities) equally
            // This ensures we get both entity information AND contextual chunks
            (QueryType::EntityFocused, _) => (0.5, 0.4, 0.1),
            (QueryType::Relationship, _) => (0.3, 0.6, 0.1),
            (QueryType::Conceptual, QueryIntent::Overview) => (0.2, 0.2, 0.6),
            (QueryType::Conceptual, _) => (0.4, 0.3, 0.3),
            (QueryType::Exploratory, QueryIntent::Overview) => (0.3, 0.2, 0.5),
            (QueryType::Exploratory, _) => (0.4, 0.4, 0.2),
            (QueryType::Factual, _) => (0.6, 0.3, 0.1),
        }
    }

    /// Entity-centric search focusing on specific entities
    fn entity_centric_search(
        &mut self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
        key_entities: &[String],
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();

        for entity_name in key_entities {
            // Find the entity in the graph
            if let Some(entity) = graph
                .entities()
                .find(|e| e.name.eq_ignore_ascii_case(entity_name))
            {
                // Add the entity itself
                results.push(SearchResult {
                    id: entity.id.to_string(),
                    content: format!("{} ({})", entity.name, entity.entity_type),
                    score: 0.9, // High score for exact entity match
                    result_type: ResultType::Entity,
                    entities: vec![entity.name.clone()],
                    source_chunks: entity
                        .mentions
                        .iter()
                        .map(|m| m.chunk_id.to_string())
                        .collect(),
                });

                // Get entity neighbors with weighted scores
                let neighbors = graph.get_neighbors(&entity.id);
                for (neighbor, relationship) in neighbors {
                    if !visited.contains(&neighbor.id) {
                        visited.insert(neighbor.id.clone());

                        // Calculate relationship relevance
                        let rel_embedding =
                            self.embed_document_text(&relationship.relation_type)?;
                        let rel_similarity =
                            VectorUtils::cosine_similarity(query_embedding, &rel_embedding);

                        results.push(SearchResult {
                            id: neighbor.id.to_string(),
                            content: format!("{} ({})", neighbor.name, neighbor.entity_type),
                            score: 0.7 * relationship.confidence * (1.0 + rel_similarity),
                            result_type: ResultType::Entity,
                            entities: vec![neighbor.name.clone()],
                            source_chunks: neighbor
                                .mentions
                                .iter()
                                .map(|m| m.chunk_id.to_string())
                                .collect(),
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Hierarchical search using document trees
    fn hierarchical_search(
        &self,
        query: &str,
        document_trees: &HashMap<crate::core::DocumentId, DocumentTree>,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let max_results_per_tree = match analysis.intent {
            QueryIntent::Overview => 3,
            QueryIntent::Detailed => 8,
            _ => 5,
        };

        for (doc_id, tree) in document_trees.iter() {
            let tree_summaries = tree.query(query, max_results_per_tree)?;

            for (idx, summary) in tree_summaries.iter().enumerate() {
                // Convert tree query result to search result
                let level_bonus = match analysis.intent {
                    QueryIntent::Overview => 0.3,
                    QueryIntent::Detailed => 0.2,
                    _ => 0.0,
                };

                results.push(SearchResult {
                    id: format!("{doc_id}:summary:{idx}"),
                    content: summary.summary.clone(),
                    score: summary.score + level_bonus,
                    result_type: ResultType::HierarchicalSummary,
                    entities: Vec::new(),
                    source_chunks: vec![doc_id.to_string()],
                });
            }
        }

        Ok(results)
    }

    /// Advanced graph traversal for complex queries
    fn advanced_graph_traversal(
        &self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        if analysis.query_type == QueryType::Relationship && analysis.key_entities.len() >= 2 {
            // Find paths between entities
            results.extend(self.find_entity_paths(graph, &analysis.key_entities)?);
        }

        if analysis.complexity_score > 0.8 {
            // Community detection for exploratory queries
            results.extend(self.community_based_search(query_embedding, graph)?);
        }

        Ok(results)
    }

    /// Cross-strategy agreement boost.
    ///
    /// Results with identical (trimmed) content found by more than one strategy are the
    /// same evidence seen twice: boost the best-scoring one in place and merge the
    /// entities / source chunks of the others into it. No copy is appended, so the same
    /// text never reaches the answer context more than once.
    fn cross_strategy_fusion(all_results: &mut [SearchResult]) {
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, result) in all_results.iter().enumerate() {
            groups
                .entry(result.content.trim().to_string())
                .or_default()
                .push(i);
        }

        for group in groups.into_values() {
            if group.len() < 2 {
                continue;
            }
            let types: HashSet<_> = group
                .iter()
                .map(|&i| all_results[i].result_type.clone())
                .collect();
            if types.len() < 2 {
                continue;
            }
            let avg_score =
                group.iter().map(|&i| all_results[i].score).sum::<f32>() / group.len() as f32;
            let boost = 0.2 * (types.len() - 1) as f32;
            let best = group
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    all_results[a]
                        .score
                        .partial_cmp(&all_results[b].score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("group is non-empty");

            let mut entities: Vec<String> = Vec::new();
            let mut chunks: Vec<String> = Vec::new();
            for &i in &group {
                for e in &all_results[i].entities {
                    if !entities.contains(e) {
                        entities.push(e.clone());
                    }
                }
                for c in &all_results[i].source_chunks {
                    if !chunks.contains(c) {
                        chunks.push(c.clone());
                    }
                }
            }
            let winner = &mut all_results[best];
            winner.score = winner.score.max((avg_score + boost).min(1.0));
            winner.entities = entities;
            winner.source_chunks = chunks;
        }
    }

    /// Adaptive ranking and deduplication based on query analysis
    fn adaptive_rank_and_deduplicate(
        &self,
        mut results: Vec<SearchResult>,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        // Apply query-specific score adjustments
        for result in &mut results {
            match analysis.query_type {
                QueryType::EntityFocused if result.result_type == ResultType::Entity => {
                    result.score *= 1.2;
                },
                QueryType::Conceptual if result.result_type == ResultType::HierarchicalSummary => {
                    result.score *= 1.1;
                },
                QueryType::Relationship if result.entities.len() > 1 => {
                    result.score *= 1.15;
                },
                _ => {},
            }

            // Boost results that contain a key entity (once, so the score stays bounded)
            if analysis
                .key_entities
                .iter()
                .any(|key| result.entities.iter().any(|e| e.eq_ignore_ascii_case(key)))
            {
                result.score *= 1.1;
            }
        }

        // Sort by adjusted scores
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Diversity-aware deduplication
        let mut deduplicated = Vec::new();
        let mut seen_content = HashSet::new();
        let mut type_counts: HashMap<ResultType, usize> = HashMap::new();

        for result in results {
            let content_signature = self.create_content_signature(&result.content);

            if !seen_content.contains(&content_signature) {
                let type_count = type_counts.get(&result.result_type).unwrap_or(&0);

                // Ensure diversity across result types
                let max_per_type = match result.result_type {
                    ResultType::Entity => self.config.top_k / 3,
                    ResultType::Chunk => self.config.top_k / 2,
                    ResultType::HierarchicalSummary => self.config.top_k / 4,
                    ResultType::Hybrid => self.config.top_k / 4,
                    ResultType::GraphPath => self.config.top_k / 5,
                };

                if *type_count < max_per_type {
                    seen_content.insert(content_signature);
                    *type_counts.entry(result.result_type.clone()).or_insert(0) += 1;
                    deduplicated.push(result);
                }
            }
        }

        Ok(deduplicated)
    }

    /// Find paths between entities in the graph
    fn find_entity_paths(
        &self,
        graph: &KnowledgeGraph,
        key_entities: &[String],
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        if key_entities.len() < 2 {
            return Ok(results);
        }

        // Simple path finding between first two entities
        if let (Some(source), Some(target)) = (
            graph
                .entities()
                .find(|e| e.name.eq_ignore_ascii_case(&key_entities[0])),
            graph
                .entities()
                .find(|e| e.name.eq_ignore_ascii_case(&key_entities[1])),
        ) {
            let path_description =
                format!("Connection between {} and {}", source.name, target.name);
            let neighbors_source = graph.get_neighbors(&source.id);
            let neighbors_target = graph.get_neighbors(&target.id);

            // Check for direct connection
            if neighbors_source
                .iter()
                .any(|(neighbor, _)| neighbor.id == target.id)
            {
                results.push(SearchResult {
                    id: format!("path_{}_{}", source.id, target.id),
                    content: format!("Direct relationship: {path_description}"),
                    score: 0.8,
                    result_type: ResultType::GraphPath,
                    entities: vec![source.name.clone(), target.name.clone()],
                    source_chunks: Vec::new(),
                });
            }

            // Check for indirect connections through common neighbors
            for (neighbor_s, rel_s) in &neighbors_source {
                for (neighbor_t, rel_t) in &neighbors_target {
                    if neighbor_s.id == neighbor_t.id {
                        results.push(SearchResult {
                            id: format!("path_{}_{}_{}", source.id, neighbor_s.id, target.id),
                            content: format!(
                                "Indirect relationship via {}: {} -> {} -> {}",
                                neighbor_s.name, source.name, neighbor_s.name, target.name
                            ),
                            score: 0.6 * rel_s.confidence * rel_t.confidence,
                            result_type: ResultType::GraphPath,
                            entities: vec![
                                source.name.clone(),
                                neighbor_s.name.clone(),
                                target.name.clone(),
                            ],
                            source_chunks: Vec::new(),
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Community-based search for exploratory queries
    fn community_based_search(
        &self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let mut entity_scores: HashMap<String, f32> = HashMap::new();

        // Degree centrality normalized by the max degree (0..1): a small secondary term,
        // so query similarity decides the ranking and scores stay bounded by 1.0.
        let degrees: Vec<(&crate::core::Entity, usize)> = graph
            .entities()
            .map(|e| (e, graph.get_neighbors(&e.id).len()))
            .collect();
        let max_degree = degrees.iter().map(|(_, d)| *d).max().unwrap_or(0).max(1) as f32;

        for (entity, degree) in degrees {
            let centrality_score = degree as f32 / max_degree;

            // Combine with embedding similarity
            if let Some(embedding) = &entity.embedding {
                let similarity = VectorUtils::cosine_similarity(query_embedding, embedding);
                entity_scores.insert(
                    entity.id.to_string(),
                    similarity * 0.8 + centrality_score * 0.2,
                );
            }
        }

        // Select top entities by combined score
        let mut sorted_entities: Vec<_> = entity_scores.iter().collect();
        sorted_entities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (entity_id, score) in sorted_entities.iter().take(3) {
            if let Some(entity) = graph.entities().find(|e| e.id.to_string() == **entity_id) {
                // Get context from chunks where this entity is mentioned
                let mut entity_context = String::new();
                for mention in entity.mentions.iter().take(2) {
                    if let Some(chunk) = graph.chunks().find(|c| c.id == mention.chunk_id) {
                        let chunk_excerpt = if chunk.content.len() > 200 {
                            format!("{}...", truncate_utf8(&chunk.content, 200))
                        } else {
                            chunk.content.clone()
                        };
                        entity_context.push_str(&chunk_excerpt);
                        entity_context.push(' ');
                    }
                }

                // If no context found, provide a meaningful description
                if entity_context.is_empty() {
                    entity_context = format!(
                        "{} is a {} character in the story.",
                        entity.name, entity.entity_type
                    );
                }

                results.push(SearchResult {
                    id: entity.id.to_string(),
                    content: entity_context,
                    score: **score,
                    result_type: ResultType::Entity,
                    entities: vec![entity.name.clone()],
                    source_chunks: entity
                        .mentions
                        .iter()
                        .map(|m| m.chunk_id.to_string())
                        .collect(),
                });
            }
        }

        Ok(results)
    }

    /// Helper method to detect abstract concepts
    fn has_abstract_concepts(&self, words: &[&str]) -> bool {
        const ABSTRACT_INDICATORS: &[&str] = &[
            "concept",
            "idea",
            "theory",
            "principle",
            "philosophy",
            "meaning",
            "understanding",
            "knowledge",
            "wisdom",
            "truth",
            "beauty",
            "justice",
        ];
        words
            .iter()
            .any(|&word| ABSTRACT_INDICATORS.contains(&word))
    }

    /// Helper method to detect question words
    fn has_question_words(&self, words: &[&str]) -> bool {
        const QUESTION_WORDS: &[&str] = &[
            "what", "how", "why", "when", "where", "who", "which", "explain", "describe",
        ];
        words.iter().any(|&word| QUESTION_WORDS.contains(&word))
    }

    /// Create content signature for deduplication
    fn create_content_signature(&self, content: &str) -> String {
        // Simple signature based on first 50 characters and length. Trimmed so the same
        // chunk surfaced by two strategies with different trailing whitespace dedups.
        let content = content.trim();
        let prefix = Self::safe_truncate(content, 50);
        format!(
            "{}_{}",
            prefix
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>(),
            content.len()
        )
    }

    /// Graph traversal search for path-based results (legacy)
    fn graph_traversal_search(
        &self,
        _query_embedding: &[f32],
        _graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        // Placeholder for graph traversal algorithms
        // This would implement algorithms like:
        // - Random walks
        // - Shortest paths between relevant entities
        // - Community detection
        // - PageRank-style scoring

        Ok(Vec::new())
    }

    /// Find entities most relevant to the query
    fn find_relevant_entities(
        &self,
        query_embedding: &[f32],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<(EntityId, f32)>> {
        let mut similarities = Vec::new();

        for entity in graph.entities() {
            if let Some(embedding) = &entity.embedding {
                let similarity = VectorUtils::cosine_similarity(query_embedding, embedding);
                if similarity >= self.config.similarity_threshold {
                    similarities.push((entity.id.clone(), similarity));
                }
            }
        }

        // Sort by similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities)
    }

    /// Expand search through graph relationships
    fn expand_through_relationships(
        &self,
        start_entity: &EntityId,
        graph: &KnowledgeGraph,
        max_depth: usize,
        visited: &mut HashSet<EntityId>,
    ) -> Result<Vec<EntityId>> {
        let mut results = Vec::new();
        let mut current_level = vec![start_entity.clone()];
        visited.insert(start_entity.clone());

        for _depth in 0..max_depth {
            let mut next_level = Vec::new();

            for entity_id in &current_level {
                results.push(entity_id.clone());

                // Get neighbors through graph relationships
                let neighbors = graph.get_neighbors(entity_id);
                for (neighbor_entity, _relationship) in neighbors {
                    if !visited.contains(&neighbor_entity.id) {
                        visited.insert(neighbor_entity.id.clone());
                        next_level.push(neighbor_entity.id.clone());
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            current_level = next_level;
        }

        Ok(results)
    }

    /// Simple stop word detection (English)
    /// Split lowercase text into unicode alphanumeric words.
    fn alnum_words(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Entities named in the query. A query word (>= 3 chars, not a stop word) must equal
    /// one of the entity name's words, or the full entity name must appear in the query.
    /// Names are deduplicated case-insensitively (the graph holds many same-named entities).
    fn detect_key_entities(&self, query_lower: &str, graph: &KnowledgeGraph) -> Vec<String> {
        let all_words = Self::alnum_words(query_lower);
        let content_words: HashSet<&str> = all_words
            .iter()
            .filter(|w| w.chars().count() >= 3 && !self.is_stop_word(w))
            .map(String::as_str)
            .collect();
        let padded_query = format!(" {} ", all_words.join(" "));

        let mut seen = HashSet::new();
        let mut key_entities = Vec::new();
        for entity in graph.entities() {
            let name_lower = entity.name.to_lowercase();
            if seen.contains(&name_lower) {
                continue;
            }
            let name_words = Self::alnum_words(&name_lower);
            let matches = name_words
                .iter()
                .any(|w| content_words.contains(w.as_str()))
                || (name_lower.chars().count() >= 3
                    && padded_query.contains(&format!(" {} ", name_words.join(" "))));
            if matches {
                seen.insert(name_lower);
                key_entities.push(entity.name.clone());
            }
        }
        key_entities
    }

    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", // Italian
            "chi", "che", "cosa", "come", "dove", "quando", "perché", "perche", "quale", "quali",
            "il", "lo", "la", "gli", "le", "un", "una", "uno", "del", "della", "dei", "delle",
            "di", "da", "in", "con", "su", "per", "tra", "fra", "non", "sono", "era", "è", "ha",
            "hanno", "suo", "sua", "loro", "questo", "quello", "anche", "ma", "se", "poi", "mai",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Rank and deduplicate search results (legacy)
    fn rank_and_deduplicate(&self, mut results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate by ID
        let mut seen_ids = HashSet::new();
        let mut deduplicated = Vec::new();

        for result in results {
            if !seen_ids.contains(&result.id) {
                seen_ids.insert(result.id.clone());
                deduplicated.push(result);
            }
        }

        Ok(deduplicated)
    }

    /// Vector-based search
    pub async fn vector_search(
        &mut self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embed_query_text_async(query).await?;
        let similar_vectors = self
            .vector_store
            .search(&query_embedding, max_results)
            .await?;

        let mut results = Vec::new();
        for store_result in similar_vectors {
            results.push(SearchResult {
                id: store_result.id.clone(),
                content: format!("Vector result for: {}", store_result.id),
                score: store_result.score,
                result_type: ResultType::Chunk,
                entities: Vec::new(),
                source_chunks: vec![store_result.id],
            });
        }

        Ok(results)
    }

    /// Graph-based search
    pub fn graph_search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        // Simplified graph search - in a real implementation this would traverse the graph
        let mut results = Vec::new();
        results.push(SearchResult {
            id: format!("graph_result_{}", query.len()),
            content: format!("Graph-based result for: {query}"),
            score: 0.7,
            result_type: ResultType::GraphPath,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        });

        Ok(results.into_iter().take(max_results).collect())
    }

    /// Hierarchical search (public wrapper)
    pub fn public_hierarchical_search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>> {
        // Simplified hierarchical search - in a real implementation this would use document trees
        let mut results = Vec::new();
        results.push(SearchResult {
            id: format!("hierarchical_result_{}", query.len()),
            content: format!("Hierarchical result for: {query}"),
            score: 0.8,
            result_type: ResultType::HierarchicalSummary,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        });

        Ok(results.into_iter().take(max_results).collect())
    }

    /// BM25-based search
    pub fn bm25_search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        // Simplified BM25 search - in a real implementation this would use proper BM25 scoring
        let mut results = Vec::new();
        results.push(SearchResult {
            id: format!("bm25_result_{}", query.len()),
            content: format!("BM25 result for: {query}"),
            score: 0.75,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        });

        Ok(results.into_iter().take(max_results).collect())
    }

    /// Get retrieval statistics
    pub fn get_statistics(&self) -> RetrievalStatistics {
        // let vector_stats = self.vector_index.statistics();

        RetrievalStatistics {
            indexed_vectors: 0,  // vector_stats.vector_count,
            vector_dimension: 0, // vector_stats.dimension,
            index_built: false,  // vector_stats.index_built,
            config: self.config.clone(),
        }
    }

    /// Safely truncate a string to a maximum byte length, respecting UTF-8 character boundaries
    fn safe_truncate(s: &str, max_bytes: usize) -> String {
        if s.len() <= max_bytes {
            return s.to_string();
        }

        // Find the largest valid character boundary <= max_bytes
        let mut end_idx = max_bytes;
        while end_idx > 0 && !s.is_char_boundary(end_idx) {
            end_idx -= 1;
        }

        s[..end_idx].to_string()
    }

    /// Save retrieval system state to JSON file
    pub fn save_state_to_json(&self, file_path: &str) -> Result<()> {
        use std::fs;

        let mut json_data = json::JsonValue::new_object();

        // Add metadata
        json_data["metadata"] = json::object! {
            "format_version" => "1.0",
            "created_at" => chrono::Utc::now().to_rfc3339(),
            "config" => json::object! {
                "top_k" => self.config.top_k,
                "similarity_threshold" => self.config.similarity_threshold,
                "max_expansion_depth" => self.config.max_expansion_depth,
                "entity_weight" => self.config.entity_weight,
                "chunk_weight" => self.config.chunk_weight,
                "graph_weight" => self.config.graph_weight
            }
        };

        // Add vector index statistics
        // let vector_stats = self.vector_index.statistics();
        json_data["vector_index"] = json::object! {
            "vector_count" => 0, // vector_stats.vector_count,
            "dimension" => 0, // vector_stats.dimension,
            "index_built" => false, // vector_stats.index_built,
            "min_norm" => 0.0, // vector_stats.min_norm,
            "max_norm" => 0.0, // vector_stats.max_norm,
            "avg_norm" => 0.0 // vector_stats.avg_norm
        };

        // Add embedding generator info
        json_data["embedding_generator"] = json::object! {
            "dimension" => self.embedding_generator.dimension(),
            "cached_words" => self.embedding_generator.cached_words()
        };

        // Add parallel processing info
        #[cfg(feature = "parallel-processing")]
        {
            json_data["parallel_enabled"] = self.parallel_processor.is_some().into();
        }
        #[cfg(not(feature = "parallel-processing"))]
        {
            json_data["parallel_enabled"] = false.into();
        }

        // Save to file
        fs::write(file_path, json_data.dump())?;
        #[cfg(feature = "tracing")]
        tracing::info!("Retrieval system state saved to {file_path}");

        Ok(())
    }
}

/// Statistics about the retrieval system
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, core::KnowledgeGraph};

    #[test]
    fn fusion_boosts_best_result_without_appending_a_copy() {
        let text = "Geppetto intaglia un pezzo di legno che parla.";
        let mk = |id: &str, ty: ResultType, score: f32, ents: &[&str]| SearchResult {
            id: id.to_string(),
            content: text.to_string(),
            score,
            result_type: ty,
            entities: ents.iter().map(|s| s.to_string()).collect(),
            source_chunks: vec![format!("c-{id}")],
        };
        let mut results = vec![
            mk("chunk_1", ResultType::Chunk, 0.6, &[]),
            mk("person_pinocchio", ResultType::Entity, 0.4, &["Pinocchio"]),
            SearchResult {
                content: "altro testo".to_string(),
                ..mk("chunk_2", ResultType::Chunk, 0.5, &[])
            },
        ];
        RetrievalSystem::cross_strategy_fusion(&mut results);
        assert_eq!(results.len(), 3, "no copy appended");
        assert!(results.iter().all(|r| !r.id.starts_with("fusion_")));
        let copies = results.iter().filter(|r| r.content == text).count();
        assert_eq!(copies, 2, "originals untouched");
        let best = results.iter().find(|r| r.id == "chunk_1").unwrap();
        assert!(best.score > 0.6, "agreement boost applied: {}", best.score);
        assert!(best.entities.contains(&"Pinocchio".to_string()));
        assert_eq!(best.source_chunks.len(), 2);
        let other = results.iter().find(|r| r.id == "chunk_2").unwrap();
        assert!((other.score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dedup_ignores_trailing_whitespace_differences() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let graph = KnowledgeGraph::new();
        let analysis = retrieval
            .analyze_query("Chi minaccia Don Abbondio?", &graph)
            .unwrap();
        let result = |id: &str, content: &str, score: f32| SearchResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: vec!["chunk-1".to_string()],
        };
        let results = retrieval
            .adaptive_rank_and_deduplicate(
                vec![
                    result("a", "Don Rodrigo minaccia Don Abbondio.\n", 0.9),
                    result("b", "Don Rodrigo minaccia Don Abbondio.\n ", 0.5),
                ],
                &analysis,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    fn graph_with_entities(names: &[&str]) -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();
        for (i, name) in names.iter().enumerate() {
            graph
                .add_entity(crate::core::Entity::new(
                    crate::core::EntityId::new(format!("e{i}")),
                    name.to_string(),
                    "PERSON".to_string(),
                    0.9,
                ))
                .unwrap();
        }
        graph
    }

    #[test]
    fn italian_query_detects_only_the_named_entity() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let graph = graph_with_entities(&["Pinocchio", "Geppetto", "Medoro"]);
        let analysis = retrieval.analyze_query("Chi è Medoro?", &graph).unwrap();
        assert_eq!(analysis.key_entities, vec!["Medoro".to_string()]);
    }

    #[test]
    fn short_and_stop_words_do_not_match_entities() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let graph = graph_with_entities(&["Pinocchio", "Geppetto", "Di"]);
        let analysis = retrieval
            .analyze_query("Che cosa è successo a Geppetto e di chi?", &graph)
            .unwrap();
        assert_eq!(analysis.key_entities, vec!["Geppetto".to_string()]);
    }

    #[test]
    fn multiword_entity_name_matches_by_word_or_full_name() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let graph = graph_with_entities(&["Fata Turchina", "Mastro Ciliegia"]);
        let analysis = retrieval
            .analyze_query("Chi è la Fata Turchina?", &graph)
            .unwrap();
        assert_eq!(analysis.key_entities, vec!["Fata Turchina".to_string()]);
    }

    #[test]
    fn same_named_entities_yield_one_key_entity() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let graph = graph_with_entities(&["Medoro", "Medoro", "medoro", "Geppetto"]);
        let analysis = retrieval.analyze_query("Chi è Medoro?", &graph).unwrap();
        assert_eq!(analysis.key_entities, vec!["Medoro".to_string()]);
    }

    #[test]
    fn key_entity_boost_is_applied_once_regardless_of_duplicates() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let analysis = QueryAnalysis {
            query_type: QueryType::Factual,
            key_entities: vec!["Medoro".to_string(); 10],
            concepts: Vec::new(),
            intent: QueryIntent::Detailed,
            complexity_score: 0.1,
        };
        let results = retrieval
            .adaptive_rank_and_deduplicate(
                vec![SearchResult {
                    id: "r".to_string(),
                    content: "Medoro compare nel capitolo".to_string(),
                    score: 1.0,
                    result_type: ResultType::Chunk,
                    entities: vec!["Medoro".to_string()],
                    source_chunks: vec!["c".to_string()],
                }],
                &analysis,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].score <= 1.1 + 1e-6, "score {}", results[0].score);
        assert!(results[0].score > 1.0);
    }

    #[test]
    fn community_search_ranks_by_similarity_with_bounded_centrality() {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let mut graph = KnowledgeGraph::new();
        let mut add = |id: &str, name: &str, emb: Vec<f32>| {
            let mut e = crate::core::Entity::new(
                crate::core::EntityId::new(id.to_string()),
                name.to_string(),
                "PERSON".to_string(),
                0.9,
            );
            e.embedding = Some(emb);
            graph.add_entity(e).unwrap();
        };
        // Hub: low similarity to the query [1, 0]; target: high similarity.
        add("hub", "Pinocchio", vec![0.1, 1.0]);
        add("target", "Medoro", vec![1.0, 0.1]);
        for i in 0..250 {
            add(&format!("n{i}"), &format!("N{i}"), vec![0.0, 1.0]);
        }
        let mut link = |a: &str, b: &str| {
            graph
                .add_relationship(crate::core::Relationship::new(
                    crate::core::EntityId::new(a.to_string()),
                    crate::core::EntityId::new(b.to_string()),
                    "KNOWS".to_string(),
                    0.9,
                ))
                .unwrap();
        };
        for i in 0..250 {
            link("hub", &format!("n{i}"));
        }
        link("hub", "target");

        let results = retrieval
            .community_based_search(&[1.0, 0.0], &graph)
            .unwrap();
        assert_eq!(results[0].id, "target");
        assert!(results.iter().all(|r| r.score <= 1.0), "{results:?}");
    }
    #[test]
    fn test_sync_query_uses_explicit_graph() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config).unwrap();
        let mut graph = KnowledgeGraph::new();
        graph
            .add_chunk(crate::core::TextChunk::new(
                crate::core::ChunkId::new("chunk-1".to_string()),
                crate::core::DocumentId::new("doc-1".to_string()),
                "Maria directs the Prisma project".to_string(),
                0,
                34,
            ))
            .unwrap();

        let results = retrieval.query("Who directs Prisma?", &graph).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("Maria directs the Prisma project"));
    }

    #[test]
    fn excerpt_truncation_preserves_utf8_boundaries() {
        let text = format!("{} end", "文".repeat(100));
        let excerpt = truncate_utf8(&text, 200);
        assert!(excerpt.is_char_boundary(excerpt.len()));
        assert_eq!(excerpt.len(), 198);
    }

    #[test]
    fn rrf_fuses_two_rankings_and_rewards_agreement() {
        let dense = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let keyword = vec!["c".to_string(), "d".to_string()];
        let fused = reciprocal_rank_fusion(&[dense, keyword], 60.0);

        let score = |id: &str| {
            fused
                .iter()
                .find(|(i, _)| i == id)
                .map(|(_, s)| *s)
                .unwrap()
        };
        assert!((score("a") - 1.0 / 61.0).abs() < 1e-6);
        assert!((score("c") - (1.0 / 63.0 + 1.0 / 61.0)).abs() < 1e-6);
        assert!((score("d") - 1.0 / 62.0).abs() < 1e-6);
        // "c" is found by both rankings, so it outranks everything else
        assert_eq!(fused[0].0, "c");
        assert_eq!(fused.len(), 4);
        // sorted by descending score
        assert!(fused.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn rrf_of_empty_rankings_is_empty() {
        assert!(reciprocal_rank_fusion(&[], 60.0).is_empty());
        assert!(reciprocal_rank_fusion(&[Vec::new()], 60.0).is_empty());
    }

    #[cfg(feature = "async")]
    fn chunk_with_embedding(id: &str, text: &str, embedding: Vec<f32>) -> crate::core::TextChunk {
        let mut chunk = crate::core::TextChunk::new(
            ChunkId::new(id.to_string()),
            crate::core::DocumentId::new("doc".to_string()),
            text.to_string(),
            0,
            text.len(),
        );
        chunk.embedding = Some(embedding);
        chunk
    }

    /// 30 decoy chunks that dense search prefers, plus one chunk with a rare exact
    /// keyword whose embedding is orthogonal to the query.
    #[cfg(feature = "async")]
    async fn keyword_only_fixture() -> (RetrievalSystem, KnowledgeGraph) {
        let retrieval = RetrievalSystem::new(&Config::default()).unwrap();
        let mut graph = KnowledgeGraph::new();
        for i in 0..30 {
            graph
                .add_chunk(chunk_with_embedding(
                    &format!("decoy-{i}"),
                    &format!("Pinocchio cammina per la strada con Geppetto, episodio {i}"),
                    vec![1.0, 0.01 * i as f32],
                ))
                .unwrap();
        }
        graph
            .add_chunk(chunk_with_embedding(
                "target",
                "Seppellirono le monete nel Campo dei miracoli, vicino alla città",
                vec![0.0, 1.0],
            ))
            .unwrap();
        retrieval.index_graph(&graph).await.unwrap();
        (retrieval, graph)
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn keyword_match_missed_by_dense_ranking_reaches_results() {
        let (mut retrieval, graph) = keyword_only_fixture().await;
        let query = "Come si chiama il Campo dei miracoli?";
        let analysis = retrieval.analyze_query(query, &graph).unwrap();
        let results = retrieval
            .execute_adaptive_retrieval(query, &[1.0, 0.0], &graph, &HashMap::new(), &analysis)
            .await
            .unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.result_type == ResultType::Chunk && r.id == "target"),
            "keyword-only chunk missing from {:?}",
            results.iter().map(|r| &r.id).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn fused_chunk_candidates_merge_dense_and_keyword_beyond_top_k() {
        let (mut retrieval, graph) = keyword_only_fixture().await;
        let ids = retrieval
            .fused_chunk_candidates(
                "Come si chiama il Campo dei miracoli?",
                &[1.0, 0.0],
                &graph,
                20,
            )
            .await
            .unwrap();
        assert_eq!(ids.len(), 20);
        assert!(ids.iter().any(|i| i == "target"), "{ids:?}");
        assert!(ids.iter().all(|i| !i.starts_with("chunk:")));
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), ids.len());
        assert!(retrieval
            .fused_chunk_candidates("x", &[1.0, 0.0], &graph, 0)
            .await
            .unwrap()
            .is_empty());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn bm25_index_rebuilds_when_graph_chunks_change() {
        let (mut retrieval, mut graph) = keyword_only_fixture().await;
        assert_eq!(
            retrieval.keyword_chunk_ranking("pescatore verde", &graph, 5),
            Vec::<String>::new()
        );
        graph
            .add_chunk(chunk_with_embedding(
                "late",
                "Il pescatore verde sta per friggere Pinocchio",
                vec![0.0, 1.0],
            ))
            .unwrap();
        assert_eq!(
            retrieval.keyword_chunk_ranking("pescatore verde", &graph, 5),
            vec!["late".to_string()]
        );
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_graph_indexing() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config).unwrap();
        let graph = KnowledgeGraph::new();

        let result = retrieval.index_graph(&graph).await;
        assert!(result.is_ok());
    }

    // ============================================================================
    // ExplainedAnswer Tests
    // ============================================================================

    #[test]
    fn test_explained_answer_creation() {
        let search_results = vec![
            SearchResult {
                id: "chunk_1".to_string(),
                content: "This is the first relevant chunk about climate change.".to_string(),
                score: 0.85,
                result_type: ResultType::Chunk,
                entities: vec!["climate".to_string(), "environment".to_string()],
                source_chunks: vec!["doc1_chunk1".to_string()],
            },
            SearchResult {
                id: "chunk_2".to_string(),
                content: "Another chunk discussing environmental policies.".to_string(),
                score: 0.72,
                result_type: ResultType::Chunk,
                entities: vec!["policy".to_string(), "environment".to_string()],
                source_chunks: vec!["doc1_chunk2".to_string()],
            },
        ];

        let explained = ExplainedAnswer::from_results(
            "Climate change is a major environmental concern.".to_string(),
            &search_results,
            "What is climate change?",
        );

        assert!(!explained.answer.is_empty());
        assert!(explained.confidence > 0.0 && explained.confidence <= 1.0);
        assert!(!explained.sources.is_empty());
        assert!(!explained.reasoning_steps.is_empty());
    }

    #[test]
    fn test_explained_answer_empty_results() {
        let explained = ExplainedAnswer::from_results(
            "No relevant information found.".to_string(),
            &[],
            "What is something unknown?",
        );

        assert_eq!(explained.confidence, 0.0);
        assert!(explained.sources.is_empty());
        assert!(!explained.reasoning_steps.is_empty()); // Should still have query analysis step
    }

    #[test]
    fn test_explained_answer_format_display() {
        let search_results = vec![SearchResult {
            id: "test_chunk".to_string(),
            content: "Test content about technology.".to_string(),
            score: 0.9,
            result_type: ResultType::Chunk,
            entities: vec!["technology".to_string()],
            source_chunks: vec!["doc1_chunk1".to_string()],
        }];

        let explained = ExplainedAnswer::from_results(
            "Technology is important.".to_string(),
            &search_results,
            "Why is technology important?",
        );

        let formatted = explained.format_display();

        assert!(formatted.contains("**Answer:**"));
        assert!(formatted.contains("**Confidence:**"));
        assert!(formatted.contains("**Reasoning:**"));
        assert!(formatted.contains("**Sources:**"));
    }

    #[test]
    fn test_reasoning_steps_structure() {
        let search_results = vec![SearchResult {
            id: "entity_1".to_string(),
            content: "Entity description".to_string(),
            score: 0.8,
            result_type: ResultType::Entity,
            entities: vec!["person".to_string(), "organization".to_string()],
            source_chunks: vec![],
        }];

        let explained = ExplainedAnswer::from_results(
            "Answer text".to_string(),
            &search_results,
            "Who are the key people?",
        );

        // Check reasoning steps are numbered correctly
        for (i, step) in explained.reasoning_steps.iter().enumerate() {
            assert_eq!(step.step_number as usize, i + 1);
            assert!(!step.description.is_empty());
            assert!(step.confidence >= 0.0 && step.confidence <= 1.0);
        }
    }

    #[test]
    fn test_source_reference_types() {
        let search_results = vec![
            SearchResult {
                id: "chunk".to_string(),
                content: "Chunk content".to_string(),
                score: 0.7,
                result_type: ResultType::Chunk,
                entities: vec![],
                source_chunks: vec![],
            },
            SearchResult {
                id: "entity".to_string(),
                content: "Entity content".to_string(),
                score: 0.6,
                result_type: ResultType::Entity,
                entities: vec![],
                source_chunks: vec![],
            },
            SearchResult {
                id: "path".to_string(),
                content: "Graph path content".to_string(),
                score: 0.5,
                result_type: ResultType::GraphPath,
                entities: vec![],
                source_chunks: vec![],
            },
        ];

        let explained =
            ExplainedAnswer::from_results("Answer".to_string(), &search_results, "Query");

        let source_types: Vec<_> = explained.sources.iter().map(|s| &s.source_type).collect();
        assert!(source_types.contains(&&SourceType::TextChunk));
        assert!(source_types.contains(&&SourceType::Entity));
        assert!(source_types.contains(&&SourceType::Relationship));
    }
}
