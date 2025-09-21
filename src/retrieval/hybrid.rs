use crate::{
    core::KnowledgeGraph,
    retrieval::{
        bm25::{BM25Result, BM25Retriever},
        ResultType,
    },
    vector::{EmbeddingGenerator, VectorIndex},
    GraphRAGError, Result,
};
use std::collections::HashMap;

/// Hybrid search result combining multiple retrieval strategies
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub result_type: ResultType,
    pub entities: Vec<String>,
    pub source_chunks: Vec<String>,
    pub fusion_method: FusionMethod,
}

/// Method used to combine scores
#[derive(Debug, Clone, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion
    RRF,
    /// Weighted combination
    Weighted,
    /// CombSUM scoring
    CombSum,
    /// Maximum score
    MaxScore,
}

/// Configuration for hybrid retrieval
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Weight for semantic search results (0.0 to 1.0)
    pub semantic_weight: f32,
    /// Weight for keyword search results (0.0 to 1.0)
    pub keyword_weight: f32,
    /// Fusion method to combine results
    pub fusion_method: FusionMethod,
    /// RRF parameter (used when fusion_method is RRF)
    pub rrf_k: f32,
    /// Maximum results to retrieve from each method before fusion
    pub max_candidates: usize,
    /// Minimum score threshold for final results
    pub min_score_threshold: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.7,
            keyword_weight: 0.3,
            fusion_method: FusionMethod::RRF,
            rrf_k: 60.0,
            max_candidates: 100,
            min_score_threshold: 0.1,
        }
    }
}

/// Hybrid retriever that combines semantic and keyword search
pub struct HybridRetriever {
    /// Vector-based retrieval system
    vector_index: VectorIndex,
    /// Embedding generator
    embedding_generator: EmbeddingGenerator,
    /// BM25-based keyword retrieval
    bm25_retriever: BM25Retriever,
    /// Configuration for hybrid retrieval
    config: HybridConfig,
    /// Flag indicating whether the system is initialized
    initialized: bool,
}

impl HybridRetriever {
    /// Create a new hybrid retriever with default configuration
    pub fn new() -> Self {
        Self {
            vector_index: VectorIndex::new(),
            embedding_generator: EmbeddingGenerator::new(128),
            bm25_retriever: BM25Retriever::new(),
            config: HybridConfig::default(),
            initialized: false,
        }
    }

    /// Create a new hybrid retriever with custom configuration
    pub fn with_config(config: HybridConfig) -> Self {
        Self {
            vector_index: VectorIndex::new(),
            embedding_generator: EmbeddingGenerator::new(128),
            bm25_retriever: BM25Retriever::new(),
            config,
            initialized: false,
        }
    }

    /// Initialize the hybrid retriever with a knowledge graph
    pub fn initialize_with_graph(&mut self, graph: &KnowledgeGraph) -> Result<()> {
        // Index entities and chunks for vector search
        for entity in graph.entities() {
            if let Some(ref embedding) = entity.embedding {
                let id = format!("entity:{}", entity.id);
                self.vector_index.add_vector(id, embedding.clone())?;
            }
        }

        for chunk in graph.chunks() {
            if let Some(ref embedding) = chunk.embedding {
                let id = format!("chunk:{}", chunk.id);
                self.vector_index.add_vector(id, embedding.clone())?;
            }
        }

        // Build vector index
        if !self.vector_index.is_empty() {
            self.vector_index.build_index()?;
        }

        // Index documents for BM25 search
        let mut bm25_documents = Vec::new();

        // Add entities as documents
        for entity in graph.entities() {
            let doc = crate::retrieval::bm25::Document {
                id: format!("entity:{}", entity.id),
                content: format!("{} {}", entity.name, entity.entity_type),
                metadata: HashMap::new(),
            };
            bm25_documents.push(doc);
        }

        // Add chunks as documents
        for chunk in graph.chunks() {
            let doc = crate::retrieval::bm25::Document {
                id: format!("chunk:{}", chunk.id),
                content: chunk.content.clone(),
                metadata: HashMap::new(),
            };
            bm25_documents.push(doc);
        }

        self.bm25_retriever.index_documents(&bm25_documents)?;
        self.initialized = true;

        Ok(())
    }

    /// Perform hybrid search combining semantic and keyword retrieval
    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<HybridSearchResult>> {
        if !self.initialized {
            return Err(GraphRAGError::Retrieval {
                message: "Hybrid retriever not initialized. Call initialize_with_graph() first."
                    .to_string(),
            });
        }

        // Get semantic results
        let semantic_results = self.semantic_search(query, self.config.max_candidates)?;

        // Get keyword results
        let keyword_results = self.keyword_search(query, self.config.max_candidates);

        // Combine results using configured fusion method
        let combined_results = self.combine_results(semantic_results, keyword_results, limit)?;

        Ok(combined_results)
    }

    /// Perform semantic search using vector similarity
    fn semantic_search(&mut self, query: &str, limit: usize) -> Result<Vec<(String, f32, String)>> {
        let query_embedding = self.embedding_generator.generate_embedding(query);
        let similar_vectors = self.vector_index.search(&query_embedding, limit)?;

        let mut results = Vec::new();
        for (id, score) in similar_vectors {
            // For now, use the ID as content - in a real implementation,
            // you would fetch the actual content from the knowledge graph
            results.push((id.clone(), score, id));
        }

        Ok(results)
    }

    /// Perform keyword search using BM25
    fn keyword_search(&self, query: &str, limit: usize) -> Vec<BM25Result> {
        self.bm25_retriever.search(query, limit)
    }

    /// Combine semantic and keyword results using the configured fusion method
    fn combine_results(
        &mut self,
        semantic_results: Vec<(String, f32, String)>,
        keyword_results: Vec<BM25Result>,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        match self.config.fusion_method {
            FusionMethod::RRF => {
                self.reciprocal_rank_fusion(semantic_results, keyword_results, limit)
            }
            FusionMethod::Weighted => {
                self.weighted_combination(semantic_results, keyword_results, limit)
            }
            FusionMethod::CombSum => self.comb_sum_fusion(semantic_results, keyword_results, limit),
            FusionMethod::MaxScore => {
                self.max_score_fusion(semantic_results, keyword_results, limit)
            }
        }
    }

    /// Reciprocal Rank Fusion (RRF)
    fn reciprocal_rank_fusion(
        &mut self,
        semantic_results: Vec<(String, f32, String)>,
        keyword_results: Vec<BM25Result>,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        let mut combined_scores: HashMap<String, (f32, f32, f32)> = HashMap::new();
        let mut content_map: HashMap<String, String> = HashMap::new();

        // Process semantic results
        for (rank, (id, score, content)) in semantic_results.iter().enumerate() {
            let rrf_score = 1.0 / (self.config.rrf_k + rank as f32 + 1.0);
            combined_scores.insert(
                id.clone(),
                (rrf_score * self.config.semantic_weight, *score, 0.0),
            );
            content_map.insert(id.clone(), content.clone());
        }

        // Process keyword results
        for (rank, result) in keyword_results.iter().enumerate() {
            let rrf_score = 1.0 / (self.config.rrf_k + rank as f32 + 1.0);
            let entry = combined_scores
                .entry(result.doc_id.clone())
                .or_insert((0.0, 0.0, 0.0));
            entry.0 += rrf_score * self.config.keyword_weight;
            entry.2 = result.score;
            content_map.insert(result.doc_id.clone(), result.content.clone());
        }

        self.create_hybrid_results(combined_scores, content_map, limit, FusionMethod::RRF)
    }

    /// Weighted combination of scores
    fn weighted_combination(
        &mut self,
        semantic_results: Vec<(String, f32, String)>,
        keyword_results: Vec<BM25Result>,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        let mut combined_scores: HashMap<String, (f32, f32, f32)> = HashMap::new();
        let mut content_map: HashMap<String, String> = HashMap::new();

        // Normalize semantic scores
        let max_semantic = semantic_results
            .iter()
            .map(|(_, score, _)| *score)
            .fold(f32::NEG_INFINITY, f32::max);

        for (id, score, content) in semantic_results {
            let normalized_score = if max_semantic > 0.0 {
                score / max_semantic
            } else {
                0.0
            };
            combined_scores.insert(
                id.clone(),
                (normalized_score * self.config.semantic_weight, score, 0.0),
            );
            content_map.insert(id, content);
        }

        // Normalize keyword scores
        let max_keyword = keyword_results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);

        for result in keyword_results {
            let normalized_score = if max_keyword > 0.0 {
                result.score / max_keyword
            } else {
                0.0
            };
            let entry = combined_scores
                .entry(result.doc_id.clone())
                .or_insert((0.0, 0.0, 0.0));
            entry.0 += normalized_score * self.config.keyword_weight;
            entry.2 = result.score;
            content_map.insert(result.doc_id.clone(), result.content.clone());
        }

        self.create_hybrid_results(combined_scores, content_map, limit, FusionMethod::Weighted)
    }

    /// CombSUM fusion (simple addition of normalized scores)
    fn comb_sum_fusion(
        &mut self,
        semantic_results: Vec<(String, f32, String)>,
        keyword_results: Vec<BM25Result>,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        let mut combined_scores: HashMap<String, (f32, f32, f32)> = HashMap::new();
        let mut content_map: HashMap<String, String> = HashMap::new();

        // Process semantic results
        for (id, score, content) in semantic_results {
            combined_scores.insert(id.clone(), (score, score, 0.0));
            content_map.insert(id, content);
        }

        // Process keyword results
        for result in keyword_results {
            let entry = combined_scores
                .entry(result.doc_id.clone())
                .or_insert((0.0, 0.0, 0.0));
            entry.0 += result.score;
            entry.2 = result.score;
            content_map.insert(result.doc_id.clone(), result.content.clone());
        }

        self.create_hybrid_results(combined_scores, content_map, limit, FusionMethod::CombSum)
    }

    /// Maximum score fusion
    fn max_score_fusion(
        &mut self,
        semantic_results: Vec<(String, f32, String)>,
        keyword_results: Vec<BM25Result>,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        let mut combined_scores: HashMap<String, (f32, f32, f32)> = HashMap::new();
        let mut content_map: HashMap<String, String> = HashMap::new();

        // Process semantic results
        for (id, score, content) in semantic_results {
            combined_scores.insert(id.clone(), (score, score, 0.0));
            content_map.insert(id, content);
        }

        // Process keyword results
        for result in keyword_results {
            let entry = combined_scores
                .entry(result.doc_id.clone())
                .or_insert((0.0, 0.0, 0.0));
            entry.0 = entry.0.max(result.score);
            entry.2 = result.score;
            content_map.insert(result.doc_id.clone(), result.content.clone());
        }

        self.create_hybrid_results(combined_scores, content_map, limit, FusionMethod::MaxScore)
    }

    /// Create hybrid results from combined scores
    fn create_hybrid_results(
        &self,
        combined_scores: HashMap<String, (f32, f32, f32)>,
        content_map: HashMap<String, String>,
        limit: usize,
        fusion_method: FusionMethod,
    ) -> Result<Vec<HybridSearchResult>> {
        let mut results: Vec<HybridSearchResult> = combined_scores
            .into_iter()
            .filter_map(|(id, (combined_score, semantic_score, keyword_score))| {
                if combined_score >= self.config.min_score_threshold {
                    let content = content_map.get(&id).cloned().unwrap_or_else(|| id.clone());

                    // Determine result type based on ID prefix
                    let result_type = if id.starts_with("entity:") {
                        ResultType::Entity
                    } else if id.starts_with("chunk:") {
                        ResultType::Chunk
                    } else {
                        ResultType::Hybrid
                    };

                    // Extract entities (simplified)
                    let entities = if result_type == ResultType::Entity {
                        vec![content.clone()]
                    } else {
                        Vec::new()
                    };

                    Some(HybridSearchResult {
                        id: id.clone(),
                        content,
                        score: combined_score,
                        semantic_score,
                        keyword_score,
                        result_type,
                        entities,
                        source_chunks: vec![id],
                        fusion_method: fusion_method.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by combined score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    /// Get configuration
    pub fn get_config(&self) -> &HybridConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: HybridConfig) {
        self.config = config;
    }

    /// Check if the retriever is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get statistics about the hybrid retriever
    pub fn get_statistics(&self) -> HybridStatistics {
        let vector_stats = self.vector_index.statistics();
        let bm25_stats = self.bm25_retriever.get_statistics();

        HybridStatistics {
            vector_count: vector_stats.vector_count,
            bm25_document_count: bm25_stats.total_documents,
            bm25_term_count: bm25_stats.total_terms,
            config: self.config.clone(),
            initialized: self.initialized,
        }
    }

    /// Clear all indexed data
    pub fn clear(&mut self) {
        self.vector_index = VectorIndex::new();
        self.bm25_retriever.clear();
        self.initialized = false;
    }
}

impl Default for HybridRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the hybrid retriever
#[derive(Debug, Clone)]
pub struct HybridStatistics {
    pub vector_count: usize,
    pub bm25_document_count: usize,
    pub bm25_term_count: usize,
    pub config: HybridConfig,
    pub initialized: bool,
}

impl HybridStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Hybrid Retriever Statistics:");
        println!("  Initialized: {}", self.initialized);
        println!("  Vector index: {} vectors", self.vector_count);
        println!(
            "  BM25 index: {} documents, {} terms",
            self.bm25_document_count, self.bm25_term_count
        );
        println!("  Fusion method: {:?}", self.config.fusion_method);
        println!(
            "  Weights: semantic={:.2}, keyword={:.2}",
            self.config.semantic_weight, self.config.keyword_weight
        );
        println!("  Score threshold: {:.3}", self.config.min_score_threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::KnowledgeGraph;

    #[test]
    fn test_hybrid_retriever_creation() {
        let retriever = HybridRetriever::new();
        assert!(!retriever.is_initialized());
    }

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert_eq!(config.semantic_weight, 0.7);
        assert_eq!(config.keyword_weight, 0.3);
        assert_eq!(config.fusion_method, FusionMethod::RRF);
    }

    #[test]
    fn test_fusion_method_variants() {
        assert_eq!(FusionMethod::RRF, FusionMethod::RRF);
        assert_ne!(FusionMethod::RRF, FusionMethod::Weighted);
    }

    #[test]
    fn test_hybrid_retriever_with_empty_graph() {
        let mut retriever = HybridRetriever::new();
        let graph = KnowledgeGraph::new();

        let result = retriever.initialize_with_graph(&graph);
        assert!(result.is_ok());
        assert!(retriever.is_initialized());
    }

    #[test]
    fn test_search_without_initialization() {
        let mut retriever = HybridRetriever::new();
        let result = retriever.search("test", 10);
        assert!(result.is_err());
    }
}
