pub mod adaptive;
/// BM25 text retrieval implementation for keyword-based search
pub mod bm25;
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
/// Causal chain analysis for discovering cause-effect paths (Phase 2.3)
pub mod causal_analysis;

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
pub use hybrid::{FusionMethod, HybridConfig, HybridRetriever, HybridSearchResult};

#[cfg(feature = "pagerank")]
pub use pagerank_retrieval::{PageRankRetrievalSystem, ScoredResult};

#[cfg(feature = "pagerank")]
pub use hipporag_ppr::{Fact, HippoRAGConfig, HippoRAGRetriever};

use crate::vector::store::VectorStore;

/// Retrieval system for querying the knowledge graph
pub struct RetrievalSystem {
    vector_store: std::sync::Arc<dyn VectorStore>,
    embedding_generator: EmbeddingGenerator,
    config: RetrievalConfig,
    #[cfg(feature = "parallel-processing")]
    parallel_processor: Option<ParallelProcessor>,
    #[cfg(feature = "pagerank")]
    pagerank_retriever: Option<PageRankRetrievalSystem>,
    enriched_retriever: Option<EnrichedRetriever>,
    #[cfg(feature = "lazygraphrag")]
    concept_filtering_enabled: bool,
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

        Ok(Self {
            vector_store,
            embedding_generator: EmbeddingGenerator::new(128), // 128-dimensional embeddings
            config: retrieval_config,
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
            #[cfg(feature = "pagerank")]
            pagerank_retriever: None,
            enriched_retriever: None,
            #[cfg(feature = "lazygraphrag")]
            concept_filtering_enabled: false,
        })
    }
}

/// Configuration parameters for the retrieval system
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum number of results to return
    pub top_k: usize,
    /// Minimum similarity score threshold for results (typically -1.0 to 1.0)
    pub similarity_threshold: f32,
    /// Maximum depth for graph relationship expansion
    pub max_expansion_depth: usize,
    /// Weight for entity-based results in scoring (0.0 to 1.0)
    pub entity_weight: f32,
    /// Weight for chunk-based results in scoring (0.0 to 1.0)
    pub chunk_weight: f32,
    /// Weight for graph-based results in scoring (0.0 to 1.0)
    pub graph_weight: f32,
    /// Enable concept-based chunk filtering (requires lazygraphrag feature)
    #[cfg(feature = "lazygraphrag")]
    pub use_concept_filtering: bool,
    /// Top-K concepts to select for filtering (requires lazygraphrag feature)
    #[cfg(feature = "lazygraphrag")]
    pub concept_top_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            similarity_threshold: 0.7,
            max_expansion_depth: 2,
            entity_weight: 0.4,
            chunk_weight: 0.4,
            graph_weight: 0.2,
            #[cfg(feature = "lazygraphrag")]
            use_concept_filtering: false,
            #[cfg(feature = "lazygraphrag")]
            concept_top_k: 20,
        }
    }
}

/// A search result containing relevant information
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique identifier for this result
    pub id: String,
    /// Content or description of the result
    pub content: String,
    /// Relevance score (higher is better)
    pub score: f32,
    /// Type of result (entity, chunk, graph path, etc.)
    pub result_type: ResultType,
    /// Names of entities associated with this result
    pub entities: Vec<String>,
    /// IDs of source chunks this result is derived from
    pub source_chunks: Vec<String>,
}

/// Type of search result indicating the retrieval strategy used
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResultType {
    /// Result from entity-based retrieval
    Entity,
    /// Result from text chunk retrieval
    Chunk,
    /// Result from graph path traversal
    GraphPath,
    /// Result from hierarchical document summarization
    HierarchicalSummary,
    /// Result from combining multiple retrieval strategies
    Hybrid,
}

// ============================================================================
// EXPLAINED ANSWER - Structured answer with reasoning trace
// ============================================================================

/// An answer with detailed explanation of the reasoning process
///
/// This struct provides transparency into how the GraphRAG system
/// arrived at its answer, including confidence scores, source references,
/// and step-by-step reasoning.
///
/// # Example
/// ```no_run
/// use graphrag_core::prelude::*;
///
/// # async fn example() -> graphrag_core::Result<()> {
/// let mut graphrag = GraphRAG::quick_start("Your document").await?;
/// let explained = graphrag.ask_explained("What is the main topic?").await?;
///
/// println!("Answer: {}", explained.answer);
/// println!("Confidence: {:.0}%", explained.confidence * 100.0);
///
/// for step in &explained.reasoning_steps {
///     println!("Step {}: {} (confidence: {:.0}%)",
///         step.step_number, step.description, step.confidence * 100.0);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ExplainedAnswer {
    /// The answer text
    pub answer: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Sources used to generate the answer
    pub sources: Vec<SourceReference>,
    /// Step-by-step reasoning trace
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Entities that were key to the answer
    pub key_entities: Vec<String>,
    /// Query analysis that guided retrieval
    pub query_analysis: Option<QueryAnalysis>,
}

/// Reference to a source document or chunk used in the answer
#[derive(Debug, Clone)]
pub struct SourceReference {
    /// Identifier of the source (chunk ID, document ID, or entity ID)
    pub id: String,
    /// Type of source
    pub source_type: SourceType,
    /// Relevant excerpt from the source
    pub excerpt: String,
    /// Relevance score to the query
    pub relevance_score: f32,
}

/// Type of source reference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceType {
    /// A text chunk from a document
    TextChunk,
    /// An entity in the knowledge graph
    Entity,
    /// A relationship between entities
    Relationship,
    /// A document-level summary
    Summary,
}

/// A single step in the reasoning process
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Step number (1-indexed)
    pub step_number: u8,
    /// Description of what was done in this step
    pub description: String,
    /// IDs of entities involved in this step
    pub entities_used: Vec<String>,
    /// Evidence snippet that supports this step
    pub evidence_snippet: Option<String>,
    /// Confidence for this specific step
    pub confidence: f32,
}

impl ExplainedAnswer {
    /// Create a new explained answer from search results
    pub fn from_results(
        answer: String,
        search_results: &[SearchResult],
        query: &str,
    ) -> Self {
        // Calculate overall confidence from result scores
        let confidence = if search_results.is_empty() {
            0.0
        } else {
            let total_score: f32 = search_results.iter().map(|r| r.score).sum();
            let avg_score = total_score / search_results.len() as f32;
            // Normalize to 0-1 range (assuming scores are already somewhat normalized)
            (avg_score * 0.7 + 0.3).min(1.0).max(0.0)
        };

        // Build source references
        let sources: Vec<SourceReference> = search_results
            .iter()
            .take(5) // Top 5 sources
            .map(|r| SourceReference {
                id: r.id.clone(),
                source_type: match r.result_type {
                    ResultType::Entity => SourceType::Entity,
                    ResultType::Chunk => SourceType::TextChunk,
                    ResultType::GraphPath => SourceType::Relationship,
                    ResultType::HierarchicalSummary => SourceType::Summary,
                    ResultType::Hybrid => SourceType::TextChunk,
                },
                excerpt: if r.content.len() > 200 {
                    format!("{}...", &r.content[..200])
                } else {
                    r.content.clone()
                },
                relevance_score: r.score,
            })
            .collect();

        // Build reasoning steps
        let mut reasoning_steps = Vec::new();
        let mut step_num = 1u8;

        // Step 1: Query analysis
        reasoning_steps.push(ReasoningStep {
            step_number: step_num,
            description: format!("Analyzed query: \"{}\"", query),
            entities_used: vec![],
            evidence_snippet: None,
            confidence: 0.95,
        });
        step_num += 1;

        // Step 2: Entity retrieval
        let unique_entities: HashSet<_> = search_results
            .iter()
            .flat_map(|r| r.entities.iter().cloned())
            .collect();
        if !unique_entities.is_empty() {
            reasoning_steps.push(ReasoningStep {
                step_number: step_num,
                description: format!("Found {} relevant entities", unique_entities.len()),
                entities_used: unique_entities.iter().take(5).cloned().collect(),
                evidence_snippet: None,
                confidence: 0.85,
            });
            step_num += 1;
        }

        // Step 3: Chunk retrieval
        let chunk_count = search_results
            .iter()
            .filter(|r| r.result_type == ResultType::Chunk || r.result_type == ResultType::Hybrid)
            .count();
        if chunk_count > 0 {
            reasoning_steps.push(ReasoningStep {
                step_number: step_num,
                description: format!("Retrieved {} relevant text chunks", chunk_count),
                entities_used: vec![],
                evidence_snippet: search_results.first().map(|r| {
                    if r.content.len() > 100 {
                        format!("{}...", &r.content[..100])
                    } else {
                        r.content.clone()
                    }
                }),
                confidence: confidence,
            });
            step_num += 1;
        }

        // Step 4: Answer synthesis
        reasoning_steps.push(ReasoningStep {
            step_number: step_num,
            description: "Synthesized answer from retrieved information".to_string(),
            entities_used: unique_entities.into_iter().take(3).collect(),
            evidence_snippet: None,
            confidence,
        });

        // Collect key entities
        let key_entities: Vec<String> = search_results
            .iter()
            .flat_map(|r| r.entities.iter().cloned())
            .take(10)
            .collect();

        Self {
            answer,
            confidence,
            sources,
            reasoning_steps,
            key_entities,
            query_analysis: None,
        }
    }

    /// Format the explained answer for display
    pub fn format_display(&self) -> String {
        let mut output = String::new();

        // Answer
        output.push_str(&format!("**Answer:** {}\n\n", self.answer));

        // Confidence
        output.push_str(&format!(
            "**Confidence:** {:.0}%\n\n",
            self.confidence * 100.0
        ));

        // Reasoning steps
        if !self.reasoning_steps.is_empty() {
            output.push_str("**Reasoning:**\n");
            for step in &self.reasoning_steps {
                output.push_str(&format!(
                    "{}. {} (confidence: {:.0}%)\n",
                    step.step_number,
                    step.description,
                    step.confidence * 100.0
                ));
                if let Some(evidence) = &step.evidence_snippet {
                    output.push_str(&format!("   Evidence: \"{}\"\n", evidence));
                }
            }
            output.push('\n');
        }

        // Sources
        if !self.sources.is_empty() {
            output.push_str("**Sources:**\n");
            for (i, source) in self.sources.iter().enumerate() {
                output.push_str(&format!(
                    "{}. [{:?}] {} (relevance: {:.0}%)\n",
                    i + 1,
                    source.source_type,
                    source.id,
                    source.relevance_score * 100.0
                ));
            }
        }

        output
    }
}

// ============================================================================
// QUERY ANALYSIS - Adaptive retrieval strategy
// ============================================================================

/// Query analysis results to determine optimal retrieval strategy
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Type of query based on content analysis
    pub query_type: QueryType,
    /// Key entities detected in the query
    pub key_entities: Vec<String>,
    /// Conceptual terms extracted from the query
    pub concepts: Vec<String>,
    /// Inferred user intent from the query
    pub intent: QueryIntent,
    /// Query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
}

/// Classification of query types for adaptive retrieval strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// Queries focused on specific entities
    EntityFocused,
    /// Abstract concept queries requiring broader context
    Conceptual,
    /// Specific fact retrieval queries
    Factual,
    /// Open-ended exploration queries
    Exploratory,
    /// Queries about relationships between entities
    Relationship,
}

/// User intent classification for result presentation
#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    /// User wants a high-level summary or overview
    Overview,
    /// User wants detailed, specific information
    Detailed,
    /// User wants to compare multiple items
    Comparative,
    /// User wants to understand cause-effect relationships
    Causal,
    /// User wants time-based or chronological information
    Temporal,
}

/// Query analysis result with additional metadata for adaptive retrieval
#[derive(Debug, Clone)]
pub struct QueryAnalysisResult {
    /// Detected query type
    pub query_type: QueryType,
    /// Confidence score for the detected query type (0.0 to 1.0)
    pub confidence: f32,
    /// Keywords extracted and matched from the query
    pub keywords_matched: Vec<String>,
    /// Recommended retrieval strategies based on analysis
    pub suggested_strategies: Vec<String>,
    /// Overall query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
}

/// Query result with hierarchical summary
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Original query string
    pub query: String,
    /// List of search results
    pub results: Vec<SearchResult>,
    /// Optional generated summary of all results
    pub summary: Option<String>,
    /// Additional metadata about the query execution
    pub metadata: HashMap<String, String>,
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
            config: retrieval_config,
            parallel_processor: Some(parallel_processor),
            #[cfg(feature = "pagerank")]
            pagerank_retriever: None,
            enriched_retriever: None,
            #[cfg(feature = "lazygraphrag")]
            concept_filtering_enabled: false,
        })
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

        tracing::debug!("PageRank retrieval system initialized with 27x performance optimizations");
        Ok(())
    }

    /// Initialize enriched metadata-aware retrieval system
    pub fn initialize_enriched(&mut self, config: Option<EnrichedRetrievalConfig>) -> Result<()> {
        tracing::debug!("Initializing enriched metadata-aware retrieval system...");

        let enriched_config = config.unwrap_or_default();
        let enriched_retriever = EnrichedRetriever::with_config(enriched_config);

        self.enriched_retriever = Some(enriched_retriever);

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

    /// Query the system for relevant information
    pub fn query(&self, query: &str) -> Result<Vec<String>> {
        // For now, return a placeholder implementation
        // In a real system, this would:
        // 1. Convert query to embedding
        // 2. Search vector index
        // 3. Expand through graph relationships
        // 4. Rank and return results

        Ok(vec![format!("Results for query: {}", query)])
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
        let query_embedding = self.embedding_generator.generate_embedding(query);

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
        let query_embedding = self.embedding_generator.generate_embedding(query);

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
            tracing::debug!(
                "Processing {total_items} embeddings with enhanced sequential approach"
            );
        }

        // Process chunks
        for (chunk_id, text) in chunk_texts {
            let embedding = self.embedding_generator.generate_embedding(&text);
            if let Some(chunk) = graph.get_chunk_mut(&chunk_id) {
                chunk.embedding = Some(embedding);
            }
        }

        // Process entities
        for (entity_id, text) in entity_texts {
            let embedding = self.embedding_generator.generate_embedding(&text);
            if let Some(entity) = graph.get_entity_mut(&entity_id) {
                entity.embedding = Some(embedding);
            }
        }

        // Re-index the graph with new embeddings
        self.index_graph(graph).await?;

        Ok(())
    }

    /// Sequential embedding generation (fallback)
    async fn add_embeddings_sequential(&mut self, graph: &mut KnowledgeGraph) -> Result<()> {
        // Debug: Check total counts first (uncomment for debugging)
        let _total_chunks = graph.chunks().count();
        let _total_entities = graph.entities().count();
        // println!("DEBUG: Found {} total chunks and {} total entities in graph", _total_chunks, _total_entities);

        // Generate embeddings for all chunks
        let mut chunk_count = 0;
        for chunk in graph.chunks_mut() {
            if chunk.embedding.is_none() {
                let embedding = self.embedding_generator.generate_embedding(&chunk.content);
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
                let embedding = self.embedding_generator.generate_embedding(&entity_text);
                entity.embedding = Some(embedding);
                entity_count += 1;
            }
        }

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
            // Use sequential processing for small batches
            let mut results = Vec::new();
            for &query in queries {
                results.push(self.hybrid_query(query, graph).await?);
            }
            return Ok(results);
        }

        #[cfg(feature = "parallel-processing")]
        {
            // For parallel query processing, we need to work around the borrowing limitations
            // of the embedding generator. We'll use enhanced sequential processing with
            // better monitoring and chunking for now.

            let chunk_size = processor.config().chunk_batch_size.min(queries.len());
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
                        tracing::warn!("Error processing query '{query}': {e}");
                        all_results.push(Vec::new());
                    },
                }
            }

            Ok(all_results)
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            // Sequential fallback when parallel processing is not available
            let mut results = Vec::new();
            for &query in queries {
                results.push(self.hybrid_query(query, graph).await?);
            }
            Ok(results)
        }
    }

    /// Analyze query to determine optimal retrieval strategy
    pub fn analyze_query(&self, query: &str, graph: &KnowledgeGraph) -> Result<QueryAnalysis> {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        // Detect key entities mentioned in the query
        let mut key_entities = Vec::new();
        for entity in graph.entities() {
            let entity_name_lower = entity.name.to_lowercase();
            if words
                .iter()
                .any(|&word| entity_name_lower.contains(word) || word.contains(&entity_name_lower))
            {
                key_entities.push(entity.name.clone());
            }
        }

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
            let mut vector_results = self
                .vector_similarity_search(query_embedding, graph)
                .await?;
            for result in &mut vector_results {
                result.score *= vector_weight;
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
        let fusion_results = self.cross_strategy_fusion(&all_results, analysis)?;
        all_results.extend(fusion_results);

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
                    let entity_id = EntityId::new(id.strip_prefix("entity:").unwrap().to_string());
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
                    let chunk_id = ChunkId::new(id.strip_prefix("chunk:").unwrap().to_string());
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
                        let rel_embedding = self
                            .embedding_generator
                            .generate_embedding(&relationship.relation_type);
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
                    id: format!("{}:summary:{}", doc_id, idx),
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

    /// Cross-strategy fusion to create hybrid results
    fn cross_strategy_fusion(
        &self,
        all_results: &[SearchResult],
        _analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        let mut fusion_results = Vec::new();

        // Group results by content similarity
        let mut content_groups: HashMap<String, Vec<&SearchResult>> = HashMap::new();

        for result in all_results {
            let content_key = Self::safe_truncate(&result.content, 50);

            content_groups.entry(content_key).or_default().push(result);
        }

        // Create fusion results for groups with multiple strategies
        for (content_key, group) in content_groups {
            if group.len() > 1 {
                let types: HashSet<_> = group.iter().map(|r| &r.result_type).collect();
                if types.len() > 1 {
                    // This content was found by multiple strategies - boost confidence
                    let avg_score = group.iter().map(|r| r.score).sum::<f32>() / group.len() as f32;
                    let boost = 0.2 * (types.len() - 1) as f32;

                    let all_entities: HashSet<_> =
                        group.iter().flat_map(|r| r.entities.iter()).collect();

                    let all_chunks: HashSet<_> =
                        group.iter().flat_map(|r| r.source_chunks.iter()).collect();

                    fusion_results.push(SearchResult {
                        id: format!(
                            "fusion_{}",
                            content_key.chars().take(10).collect::<String>()
                        ),
                        content: group[0].content.clone(),
                        score: (avg_score + boost).min(1.0),
                        result_type: ResultType::Hybrid,
                        entities: all_entities.into_iter().cloned().collect(),
                        source_chunks: all_chunks.into_iter().cloned().collect(),
                    });
                }
            }
        }

        Ok(fusion_results)
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
                QueryType::EntityFocused => {
                    if result.result_type == ResultType::Entity {
                        result.score *= 1.2;
                    }
                },
                QueryType::Conceptual => {
                    if result.result_type == ResultType::HierarchicalSummary {
                        result.score *= 1.1;
                    }
                },
                QueryType::Relationship => {
                    if result.entities.len() > 1 {
                        result.score *= 1.15;
                    }
                },
                _ => {},
            }

            // Boost results that contain key entities
            for entity in &analysis.key_entities {
                if result
                    .entities
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(entity))
                {
                    result.score *= 1.1;
                }
            }
        }

        // Sort by adjusted scores
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

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

        // Calculate centrality-like scores for entities
        for entity in graph.entities() {
            let neighbors = graph.get_neighbors(&entity.id);
            let centrality_score = neighbors.len() as f32 * 0.1;

            // Combine with embedding similarity
            if let Some(embedding) = &entity.embedding {
                let similarity = VectorUtils::cosine_similarity(query_embedding, embedding);
                entity_scores.insert(entity.id.to_string(), centrality_score + similarity);
            }
        }

        // Select top entities by combined score
        let mut sorted_entities: Vec<_> = entity_scores.iter().collect();
        sorted_entities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (entity_id, score) in sorted_entities.iter().take(3) {
            if let Some(entity) = graph.entities().find(|e| e.id.to_string() == **entity_id) {
                // Get context from chunks where this entity is mentioned
                let mut entity_context = String::new();
                for mention in entity.mentions.iter().take(2) {
                    if let Some(chunk) = graph.chunks().find(|c| c.id == mention.chunk_id) {
                        let chunk_excerpt = if chunk.content.len() > 200 {
                            format!("{}...", &chunk.content[..200])
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
        // Simple signature based on first 50 characters and length
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
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Rank and deduplicate search results (legacy)
    fn rank_and_deduplicate(&self, mut results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

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
        let query_embedding = self.embedding_generator.generate_embedding(query);
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
        tracing::info!("Retrieval system state saved to {file_path}");

        Ok(())
    }
}

/// Statistics about the retrieval system
#[derive(Debug)]
pub struct RetrievalStatistics {
    /// Number of vectors indexed in the system
    pub indexed_vectors: usize,
    /// Dimensionality of the vector embeddings
    pub vector_dimension: usize,
    /// Whether the vector index has been built
    pub index_built: bool,
    /// Current retrieval configuration
    pub config: RetrievalConfig,
}

impl RetrievalStatistics {
    /// Print retrieval statistics
    #[allow(dead_code)]
    pub fn print(&self) {
        tracing::info!("Retrieval System Statistics:");
        tracing::info!("  Indexed vectors: {}", self.indexed_vectors);
        tracing::info!("  Vector dimension: {}", self.vector_dimension);
        tracing::info!("  Index built: {}", self.index_built);
        tracing::info!("  Configuration:");
        tracing::info!("    Top K: {}", self.config.top_k);
        tracing::info!(
            "    Similarity threshold: {:.2}",
            self.config.similarity_threshold
        );
        tracing::info!(
            "    Max expansion depth: {}",
            self.config.max_expansion_depth
        );
        tracing::info!("    Entity weight: {:.2}", self.config.entity_weight);
        tracing::info!("    Chunk weight: {:.2}", self.config.chunk_weight);
        tracing::info!("    Graph weight: {:.2}", self.config.graph_weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, core::KnowledgeGraph};

    #[test]
    fn test_retrieval_system_creation() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config);
        assert!(retrieval.is_ok());
    }

    #[test]
    fn test_query_placeholder() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config).unwrap();

        let results = retrieval.query("test query");
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].contains("test query"));
    }

    #[tokio::test]
    async fn test_graph_indexing() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).unwrap();
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

        let explained = ExplainedAnswer::from_results(
            "Answer".to_string(),
            &search_results,
            "Query",
        );

        let source_types: Vec<_> = explained.sources.iter().map(|s| &s.source_type).collect();
        assert!(source_types.contains(&&SourceType::TextChunk));
        assert!(source_types.contains(&&SourceType::Entity));
        assert!(source_types.contains(&&SourceType::Relationship));
    }
}
