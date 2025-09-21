use crate::core::{ChunkId, DocumentId, Entity, EntityId, KnowledgeGraph, Result, TextChunk};
use crate::entity::gleaning_extractor::GleaningEntityExtractor;
#[cfg(feature = "pagerank")]
use crate::graph::pagerank::PersonalizedPageRank;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Advanced query pipeline for multi-modal GraphRAG query processing
pub struct AdvancedQueryPipeline {
    entity_extractor: Arc<GleaningEntityExtractor>,
    #[cfg(feature = "pagerank")]
    pagerank_calculator: Option<Arc<PersonalizedPageRank>>,
    score_combiner: ScoreCombiner,
    ranking_policies: Vec<Box<dyn RankingPolicy>>,
}

/// Result of query analysis containing extracted entities, concepts, and intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysisResult {
    pub query_entities: Vec<Entity>,
    pub query_concepts: Vec<String>,
    pub query_intent: QueryIntent,
    pub temporal_references: Vec<TemporalReference>,
}

/// Classification of query intent to guide processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryIntent {
    Factual,     // "What is X?"
    Relational,  // "How is X related to Y?"
    Temporal,    // "When did X happen?"
    Causal,      // "Why did X happen?"
    Comparative, // "Compare X and Y"
    Exploratory, // "Tell me about X"
}

/// Temporal reference extracted from query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalReference {
    pub text: String,
    pub reference_type: TemporalType,
    pub parsed_date: Option<DateTime<Utc>>,
}

/// Type of temporal reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalType {
    Absolute, // "January 2020"
    Relative, // "last year"
    Duration, // "for 3 months"
}

/// Complete query result with scored entities and execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub results: Vec<ScoredResult>,
    pub analysis: QueryAnalysisResult,
    pub execution_stats: ExecutionStats,
}

/// Single scored result entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    pub entity_id: EntityId,
    pub score: f64,
    pub breakdown: ScoreBreakdown,
    pub content: ResultContent,
}

/// Breakdown of different scoring components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub vector_score: f64,
    pub pagerank_score: f64,
    pub relationship_score: f64,
    pub chunk_score: f64,
    pub combined_score: f64,
}

/// Content associated with a result entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultContent {
    pub entity_name: String,
    pub entity_type: String,
    pub description: String,
    pub relevant_chunks: Vec<String>,
    pub related_entities: Vec<EntityId>,
}

/// Execution statistics for query performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub total_time_ms: u64,
    pub analysis_time_ms: u64,
    pub vector_search_time_ms: u64,
    pub pagerank_time_ms: u64,
    pub ranking_time_ms: u64,
    pub entities_analyzed: usize,
    pub candidates_found: usize,
}

/// Trait for implementing custom ranking policies
pub trait RankingPolicy: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn apply(&self, scores: &mut Vec<ScoredResult>) -> Result<()>;
    fn is_applicable(&self, query_analysis: &QueryAnalysisResult) -> bool;
}

/// Configurable score combiner for multi-modal scoring
#[derive(Debug, Clone)]
pub struct ScoreCombiner {
    pub vector_weight: f64,
    pub pagerank_weight: f64,
    pub relationship_weight: f64,
    pub chunk_weight: f64,
}

impl Default for ScoreCombiner {
    fn default() -> Self {
        Self {
            vector_weight: 0.3,
            pagerank_weight: 0.4,
            relationship_weight: 0.2,
            chunk_weight: 0.1,
        }
    }
}

impl AdvancedQueryPipeline {
    /// Create a new advanced query pipeline
    pub fn new(entity_extractor: Arc<GleaningEntityExtractor>) -> Self {
        Self {
            entity_extractor,
            #[cfg(feature = "pagerank")]
            pagerank_calculator: None,
            score_combiner: ScoreCombiner::default(),
            ranking_policies: Vec::new(),
        }
    }

    /// Configure PageRank calculator for graph-based scoring
    #[cfg(feature = "pagerank")]
    pub fn with_pagerank(mut self, pagerank_calculator: Arc<PersonalizedPageRank>) -> Self {
        self.pagerank_calculator = Some(pagerank_calculator);
        self
    }

    /// Configure custom score combiner weights
    pub fn with_score_combiner(mut self, score_combiner: ScoreCombiner) -> Self {
        self.score_combiner = score_combiner;
        self
    }

    /// Add a custom ranking policy
    pub fn add_ranking_policy(mut self, policy: Box<dyn RankingPolicy>) -> Self {
        self.ranking_policies.push(policy);
        self
    }

    /// Execute complete multi-modal query pipeline
    pub async fn execute_query(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
        max_results: usize,
    ) -> Result<QueryResult> {
        let start_time = Instant::now();
        println!("Executing advanced query: {query}");

        // Step 1: Analyze query and extract entities/concepts
        let analysis_start = Instant::now();
        let query_analysis = self.analyze_query(query).await?;
        let analysis_time = analysis_start.elapsed().as_millis() as u64;

        println!(
            "Found {} query entities, intent: {:?}",
            query_analysis.query_entities.len(),
            query_analysis.query_intent
        );

        // Step 2: Vector similarity search for initial candidates
        let vector_start = Instant::now();
        let vector_scores = self.vector_similarity_search(query, graph).await?;
        let vector_time = vector_start.elapsed().as_millis() as u64;

        println!("Vector search found {} candidates", vector_scores.len());

        // Step 3: Personalized PageRank propagation
        let pagerank_start = Instant::now();
        let pagerank_scores = self
            .pagerank_propagation(&query_analysis, &vector_scores, graph)
            .await?;
        let pagerank_time = pagerank_start.elapsed().as_millis() as u64;

        println!("PageRank propagation completed");

        // Step 4: Relationship scoring
        let relationship_scores =
            self.score_relationships(&pagerank_scores, &query_analysis, graph)?;

        // Step 5: Chunk relevance scoring
        let chunk_scores = self.score_chunks(&relationship_scores, &query_analysis, graph)?;

        // Step 6: Multi-modal score combination
        let combined_scores = self.score_combiner.combine_scores(
            &vector_scores,
            &pagerank_scores,
            &chunk_scores,
            &query_analysis,
        )?;

        // Step 7: Policy-based ranking and selection
        let ranking_start = Instant::now();
        let final_results =
            self.apply_ranking_policies(combined_scores, &query_analysis, max_results)?;
        let ranking_time = ranking_start.elapsed().as_millis() as u64;

        let total_time = start_time.elapsed().as_millis() as u64;

        Ok(QueryResult {
            query: query.to_string(),
            results: final_results.clone(),
            analysis: query_analysis,
            execution_stats: ExecutionStats {
                total_time_ms: total_time,
                analysis_time_ms: analysis_time,
                vector_search_time_ms: vector_time,
                pagerank_time_ms: pagerank_time,
                ranking_time_ms: ranking_time,
                entities_analyzed: graph.entity_count(),
                candidates_found: final_results.len(),
            },
        })
    }

    /// Analyze query to extract entities, concepts, intent, and temporal references
    async fn analyze_query(&self, query: &str) -> Result<QueryAnalysisResult> {
        // Extract entities from the query using gleaning extractor
        let query_chunk = TextChunk::new(
            ChunkId::new("query_chunk".to_string()),
            DocumentId::new("query_doc".to_string()),
            query.to_string(),
            0,
            query.len(),
        );

        let (query_entities, _) = self
            .entity_extractor
            .extract_with_gleaning(&query_chunk)
            .await?;

        // Extract concepts (important words/phrases)
        let query_concepts = self.extract_concepts(query);

        // Determine query intent
        let query_intent = self.classify_intent(query);

        // Extract temporal references
        let temporal_references = self.extract_temporal_references(query);

        Ok(QueryAnalysisResult {
            query_entities,
            query_concepts,
            query_intent,
            temporal_references,
        })
    }

    /// Extract important concepts from the query text
    fn extract_concepts(&self, query: &str) -> Vec<String> {
        // Simple concept extraction based on important words
        let stop_words = [
            "the", "is", "are", "was", "were", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "a", "an", "and", "or", "but", "in",
            "on", "at", "to", "for", "of", "with", "by",
        ];

        query
            .to_lowercase()
            .split_whitespace()
            .map(|word| {
                // Remove punctuation from the word
                word.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
            })
            .filter(|word| word.len() > 2 && !stop_words.contains(&word.as_str()))
            .collect()
    }

    /// Classify the intent of the query
    fn classify_intent(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();

        if query_lower.starts_with("what is") || query_lower.starts_with("what are") {
            QueryIntent::Factual
        } else if query_lower.contains("related")
            || query_lower.contains("connection")
            || query_lower.contains("relationship")
        {
            QueryIntent::Relational
        } else if query_lower.starts_with("when")
            || query_lower.contains("time")
            || query_lower.contains("date")
        {
            QueryIntent::Temporal
        } else if query_lower.starts_with("why")
            || query_lower.contains("cause")
            || query_lower.contains("reason")
        {
            QueryIntent::Causal
        } else if query_lower.contains("compare")
            || query_lower.contains("difference")
            || query_lower.contains("versus")
        {
            QueryIntent::Comparative
        } else {
            QueryIntent::Exploratory
        }
    }

    /// Extract temporal references from the query
    fn extract_temporal_references(&self, _query: &str) -> Vec<TemporalReference> {
        // Simplified implementation - could use regex patterns for real temporal extraction
        Vec::new()
    }

    /// Create a basic query result for testing/demonstration purposes
    pub fn create_demo_result(query: &str) -> QueryResult {
        QueryResult {
            query: query.to_string(),
            results: vec![ScoredResult {
                entity_id: EntityId::new("demo_entity_1".to_string()),
                score: 0.85,
                breakdown: ScoreBreakdown {
                    vector_score: 0.8,
                    pagerank_score: 0.9,
                    relationship_score: 0.7,
                    chunk_score: 0.6,
                    combined_score: 0.85,
                },
                content: ResultContent {
                    entity_name: "Demo Entity 1".to_string(),
                    entity_type: "ORGANIZATION".to_string(),
                    description: "A demonstration entity for testing".to_string(),
                    relevant_chunks: vec!["Chunk 1".to_string()],
                    related_entities: vec![EntityId::new("related_entity".to_string())],
                },
            }],
            analysis: QueryAnalysisResult {
                query_entities: Vec::new(),
                query_concepts: vec!["demo".to_string(), "entity".to_string()],
                query_intent: QueryIntent::Factual,
                temporal_references: Vec::new(),
            },
            execution_stats: ExecutionStats {
                total_time_ms: 100,
                analysis_time_ms: 20,
                vector_search_time_ms: 30,
                pagerank_time_ms: 25,
                ranking_time_ms: 15,
                entities_analyzed: 1000,
                candidates_found: 1,
            },
        }
    }

    /// Perform vector similarity search for initial candidate selection
    async fn vector_similarity_search(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
    ) -> Result<HashMap<EntityId, f64>> {
        // Simple text-based similarity search using Jaccard similarity
        // In a full implementation, this would use actual vector embeddings
        let mut scores = HashMap::new();

        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower.split_whitespace().collect();

        for entity in graph.entities() {
            let entity_name_lower = entity.name.to_lowercase();
            let entity_words: HashSet<&str> = entity_name_lower.split_whitespace().collect();

            let intersection = query_words.intersection(&entity_words).count();
            let union = query_words.union(&entity_words).count();

            if union > 0 {
                let jaccard_similarity = intersection as f64 / union as f64;
                if jaccard_similarity > 0.1 {
                    scores.insert(entity.id.clone(), jaccard_similarity);
                }
            }
        }

        Ok(scores)
    }

    /// Perform PageRank propagation to score entities based on graph structure
    async fn pagerank_propagation(
        &self,
        _query_analysis: &QueryAnalysisResult,
        vector_scores: &HashMap<EntityId, f64>,
        _graph: &KnowledgeGraph,
    ) -> Result<HashMap<EntityId, f64>> {
        #[cfg(feature = "pagerank")]
        {
            if let Some(ref pagerank_calc) = self.pagerank_calculator {
                // Use query entities and vector scores as reset probabilities for personalized PageRank
                let mut reset_probabilities = HashMap::new();

                // Add query entities with high reset probability
                for entity in &_query_analysis.query_entities {
                    reset_probabilities.insert(entity.id.clone(), 0.8);
                }

                // Add vector search results with their scores as reset probabilities
                for (entity_id, &score) in vector_scores {
                    let existing_prob = reset_probabilities.get(entity_id).unwrap_or(&0.0);
                    reset_probabilities
                        .insert(entity_id.clone(), f64::max(*existing_prob, score * 0.5));
                }

                let pagerank_scores = pagerank_calc.calculate_scores(&reset_probabilities)?;
                return Ok(pagerank_scores);
            }
        }

        // If no PageRank calculator available, return vector scores
        Ok(vector_scores.clone())
    }

    /// Score entities based on their relationships to highly scored entities
    fn score_relationships(
        &self,
        pagerank_scores: &HashMap<EntityId, f64>,
        query_analysis: &QueryAnalysisResult,
        graph: &KnowledgeGraph,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut relationship_scores = HashMap::new();

        // Score entities based on their relationships to highly scored entities
        for entity in graph.entities() {
            let mut total_relationship_score = 0.0;
            let mut relationship_count = 0;

            // Get relationships for this entity
            for relationship in graph.get_entity_relationships(&entity.id.0) {
                let other_entity_id = if relationship.source == entity.id {
                    &relationship.target
                } else {
                    &relationship.source
                };

                if let Some(&other_score) = pagerank_scores.get(other_entity_id) {
                    let relationship_strength = relationship.confidence as f64;
                    total_relationship_score += other_score * relationship_strength;
                    relationship_count += 1;
                }
            }

            if relationship_count > 0 {
                let avg_relationship_score = total_relationship_score / relationship_count as f64;
                relationship_scores.insert(entity.id.clone(), avg_relationship_score);
            }
        }

        // Boost scores for entities that match query intent
        if let QueryIntent::Relational = query_analysis.query_intent {
            // Boost entities with many relationships
            for (entity_id, score) in relationship_scores.iter_mut() {
                let relationship_count = graph.get_entity_relationships(&entity_id.0).count();
                *score *= 1.0 + (relationship_count as f64 * 0.1);
            }
        }

        Ok(relationship_scores)
    }

    /// Score entities based on chunk relevance
    fn score_chunks(
        &self,
        _relationship_scores: &HashMap<EntityId, f64>,
        _query_analysis: &QueryAnalysisResult,
        graph: &KnowledgeGraph,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut chunk_scores = HashMap::new();

        // Score entities based on the relevance of chunks they appear in
        for entity in graph.entities() {
            // Use a simple heuristic based on mention frequency
            let mention_count = entity.mentions.len();
            if mention_count > 0 {
                let chunk_score = (mention_count as f64).ln() + 1.0;
                chunk_scores.insert(entity.id.clone(), chunk_score);
            }
        }

        Ok(chunk_scores)
    }

    /// Apply ranking policies and return top results
    fn apply_ranking_policies(
        &self,
        mut combined_scores: Vec<ScoredResult>,
        query_analysis: &QueryAnalysisResult,
        max_results: usize,
    ) -> Result<Vec<ScoredResult>> {
        // Apply applicable ranking policies
        for policy in &self.ranking_policies {
            if policy.is_applicable(query_analysis) {
                println!("Applying ranking policy: {}", policy.name());
                policy.apply(&mut combined_scores)?;
            }
        }

        // Apply simple top-k if results exceed max_results
        if combined_scores.len() > max_results {
            combined_scores.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            combined_scores.truncate(max_results);
        }

        Ok(combined_scores)
    }
}

impl ScoreCombiner {
    /// Combine multiple scoring modalities into final ranked results
    pub fn combine_scores(
        &self,
        vector_scores: &HashMap<EntityId, f64>,
        pagerank_scores: &HashMap<EntityId, f64>,
        chunk_scores: &HashMap<EntityId, f64>,
        _query_analysis: &QueryAnalysisResult,
    ) -> Result<Vec<ScoredResult>> {
        let mut results = Vec::new();

        // Get all unique entity IDs from all scoring modalities
        let all_entities: HashSet<EntityId> = vector_scores
            .keys()
            .chain(pagerank_scores.keys())
            .chain(chunk_scores.keys())
            .cloned()
            .collect();

        for entity_id in all_entities {
            let vector_score = vector_scores.get(&entity_id).copied().unwrap_or(0.0);
            let pagerank_score = pagerank_scores.get(&entity_id).copied().unwrap_or(0.0);
            let relationship_score = 0.0; // Could be enhanced with relationship-specific scoring
            let chunk_score = chunk_scores.get(&entity_id).copied().unwrap_or(0.0);

            let combined_score = self.vector_weight * vector_score
                + self.pagerank_weight * pagerank_score
                + self.relationship_weight * relationship_score
                + self.chunk_weight * chunk_score;

            // Only include entities with meaningful scores
            if combined_score > 0.01 {
                results.push(ScoredResult {
                    entity_id: entity_id.clone(),
                    score: combined_score,
                    breakdown: ScoreBreakdown {
                        vector_score,
                        pagerank_score,
                        relationship_score,
                        chunk_score,
                        combined_score,
                    },
                    content: ResultContent {
                        entity_name: entity_id.0.clone(),
                        entity_type: "ENTITY".to_string(),
                        description: format!("Entity: {}", entity_id.0),
                        relevant_chunks: Vec::new(),
                        related_entities: Vec::new(),
                    },
                });
            }
        }

        // Sort by combined score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_pipeline() -> AdvancedQueryPipeline {
        // Create a mock entity extractor for testing
        use crate::entity::{gleaning_extractor::GleaningConfig, EntityExtractor};

        let entity_extractor = Arc::new(GleaningEntityExtractor::new(
            EntityExtractor::new(0.7).unwrap(),
            GleaningConfig::default(),
        ));

        AdvancedQueryPipeline::new(entity_extractor)
    }

    #[test]
    fn test_intent_classification() {
        let pipeline = create_mock_pipeline();

        assert_eq!(
            pipeline.classify_intent("What is Apple?"),
            QueryIntent::Factual
        );
        assert_eq!(
            pipeline.classify_intent("How is Apple related to Microsoft?"),
            QueryIntent::Relational
        );
        assert_eq!(
            pipeline.classify_intent("When was Apple founded?"),
            QueryIntent::Temporal
        );
        assert_eq!(
            pipeline.classify_intent("Why did Apple succeed?"),
            QueryIntent::Causal
        );
        assert_eq!(
            pipeline.classify_intent("Compare Apple and Microsoft"),
            QueryIntent::Comparative
        );
        assert_eq!(
            pipeline.classify_intent("Tell me about Apple"),
            QueryIntent::Exploratory
        );
    }

    #[test]
    fn test_concept_extraction() {
        let pipeline = create_mock_pipeline();

        let concepts =
            pipeline.extract_concepts("What is the relationship between Apple and Microsoft?");

        assert!(concepts.contains(&"relationship".to_string()));
        assert!(concepts.contains(&"apple".to_string()));
        assert!(concepts.contains(&"microsoft".to_string()));
        assert!(!concepts.contains(&"the".to_string())); // Stop word should be filtered
    }

    #[test]
    fn test_score_combiner() {
        let combiner = ScoreCombiner::default();

        let mut vector_scores = HashMap::new();
        let mut pagerank_scores = HashMap::new();
        let mut chunk_scores = HashMap::new();

        let entity_id = EntityId::new("test_entity".to_string());
        vector_scores.insert(entity_id.clone(), 0.8);
        pagerank_scores.insert(entity_id.clone(), 0.6);
        chunk_scores.insert(entity_id.clone(), 0.4);

        let query_analysis = QueryAnalysisResult {
            query_entities: Vec::new(),
            query_concepts: Vec::new(),
            query_intent: QueryIntent::Factual,
            temporal_references: Vec::new(),
        };

        let results = combiner
            .combine_scores(
                &vector_scores,
                &pagerank_scores,
                &chunk_scores,
                &query_analysis,
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.0);

        // Check that the combined score is a weighted combination
        let expected_score = 0.3 * 0.8 + 0.4 * 0.6 + 0.0 * 0.0 + 0.1 * 0.4;
        assert!((results[0].score - expected_score).abs() < 0.001);
    }

    #[test]
    fn test_score_combiner_default_weights() {
        let combiner = ScoreCombiner::default();

        // Verify default weights sum to 1.0
        let total_weight = combiner.vector_weight
            + combiner.pagerank_weight
            + combiner.relationship_weight
            + combiner.chunk_weight;

        assert!((total_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temporal_reference_creation() {
        let temporal_ref = TemporalReference {
            text: "last year".to_string(),
            reference_type: TemporalType::Relative,
            parsed_date: None,
        };

        assert_eq!(temporal_ref.text, "last year");
        assert_eq!(temporal_ref.reference_type, TemporalType::Relative);
        assert!(temporal_ref.parsed_date.is_none());
    }

    #[test]
    fn test_query_intent_equality() {
        assert_eq!(QueryIntent::Factual, QueryIntent::Factual);
        assert_ne!(QueryIntent::Factual, QueryIntent::Relational);
    }
}
