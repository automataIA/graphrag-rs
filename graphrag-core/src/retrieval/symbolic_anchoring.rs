//! Symbolic Anchoring for Conceptual Queries
//!
//! This module implements symbolic anchoring from CatRAG methodology.
//! It helps with abstract/conceptual queries by grounding concepts to concrete entities.
//!
//! Example: Query "What is the nature of love?" should find entities like:
//! - Phaedrus (dialog about love)
//! - Symposium (discusses love)
//! - Socrates (taught about love)
//!
//! Instead of just matching "love" keyword.

use crate::{
    core::{EntityId, KnowledgeGraph},
    retrieval::SearchResult,
};
use std::collections::HashMap;
use std::sync::Arc;

/// A symbolic anchor linking an abstract concept to concrete entities
///
/// Anchors help ground conceptual queries in the knowledge graph by finding
/// entities that embody or discuss the concept.
#[derive(Debug, Clone)]
pub struct SymbolicAnchor {
    /// The abstract concept (e.g., "love", "virtue", "justice")
    pub concept: String,

    /// Entities in the graph that embody or discuss this concept
    pub grounded_entities: Vec<EntityId>,

    /// Relevance score indicating how important this anchor is for the query (0.0-1.0)
    pub relevance_score: f32,

    /// Semantic similarity between concept and anchor (from embeddings)
    pub similarity_score: f32,
}

impl SymbolicAnchor {
    /// Create a new symbolic anchor
    pub fn new(concept: String, relevance_score: f32) -> Self {
        Self {
            concept,
            grounded_entities: Vec::new(),
            relevance_score,
            similarity_score: 0.0,
        }
    }

    /// Add a grounded entity to this anchor
    pub fn add_entity(&mut self, entity_id: EntityId) {
        if !self.grounded_entities.contains(&entity_id) {
            self.grounded_entities.push(entity_id);
        }
    }

    /// Set similarity score
    pub fn with_similarity(mut self, score: f32) -> Self {
        self.similarity_score = score.clamp(0.0, 1.0);
        self
    }
}

/// Strategy for symbolic anchoring in retrieval
///
/// Identifies abstract concepts in queries and grounds them to concrete entities
/// in the knowledge graph.
pub struct SymbolicAnchoringStrategy {
    /// Reference to the knowledge graph
    graph: Arc<KnowledgeGraph>,

    /// Minimum relevance score to keep an anchor
    min_relevance: f32,

    /// Maximum number of anchors to extract per query
    max_anchors: usize,

    /// Maximum entities per anchor
    max_entities_per_anchor: usize,

    /// Optional PageRank scores for entities (for importance boosting)
    pagerank_scores: Option<HashMap<EntityId, f32>>,
}

impl SymbolicAnchoringStrategy {
    /// Create a new symbolic anchoring strategy
    ///
    /// # Arguments
    ///
    /// * `graph` - Reference to the knowledge graph
    pub fn new(graph: Arc<KnowledgeGraph>) -> Self {
        Self {
            graph,
            min_relevance: 0.3,
            max_anchors: 5,
            max_entities_per_anchor: 10,
            pagerank_scores: None,
        }
    }

    /// Set PageRank scores for importance-based boosting
    ///
    /// # Arguments
    ///
    /// * `scores` - Map of entity IDs to their PageRank scores
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::HashMap;
    /// let mut scores = HashMap::new();
    /// scores.insert(EntityId("socrates".to_string()), 0.85);
    /// scores.insert(EntityId("plato".to_string()), 0.72);
    ///
    /// let strategy = SymbolicAnchoringStrategy::new(graph)
    ///     .with_pagerank_scores(scores);
    /// ```
    pub fn with_pagerank_scores(mut self, scores: HashMap<EntityId, f32>) -> Self {
        self.pagerank_scores = Some(scores);
        self
    }

    /// Set minimum relevance threshold
    pub fn with_min_relevance(mut self, min_relevance: f32) -> Self {
        self.min_relevance = min_relevance.clamp(0.0, 1.0);
        self
    }

    /// Set maximum number of anchors
    pub fn with_max_anchors(mut self, max_anchors: usize) -> Self {
        self.max_anchors = max_anchors;
        self
    }

    /// Extract symbolic anchors from a query
    ///
    /// # Arguments
    ///
    /// * `query` - The user's query string
    ///
    /// # Returns
    ///
    /// Vector of symbolic anchors grounding concepts to entities
    pub fn extract_anchors(&self, query: &str) -> Vec<SymbolicAnchor> {
        let mut anchors = Vec::new();

        // Step 1: Extract potential concepts from query
        let concepts = self.extract_concepts(query);

        // Step 2: For each concept, find grounded entities
        for concept in concepts {
            let mut anchor = SymbolicAnchor::new(concept.clone(), 1.0);

            // Find entities related to this concept
            let grounded = self.ground_concept(&concept);

            for entity_id in grounded.into_iter().take(self.max_entities_per_anchor) {
                anchor.add_entity(entity_id);
            }

            // Only keep anchors with entities
            if !anchor.grounded_entities.is_empty() {
                // Calculate relevance based on number of groundings and PageRank
                let relevance = self.calculate_relevance(&anchor);
                anchor.relevance_score = relevance;

                if anchor.relevance_score >= self.min_relevance {
                    anchors.push(anchor);
                }
            }
        }

        // Sort by relevance and take top-K
        anchors.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        anchors.truncate(self.max_anchors);

        anchors
    }

    /// Extract conceptual terms from query
    ///
    /// Identifies abstract nouns and philosophical terms.
    fn extract_concepts(&self, query: &str) -> Vec<String> {
        let mut concepts = Vec::new();

        // Simple heuristic: Look for question words + abstract nouns
        let words: Vec<&str> = query.split_whitespace().collect();

        // Conceptual query patterns
        let conceptual_patterns = [
            "what is", "nature of", "meaning of", "definition of",
            "concept of", "idea of", "philosophy of", "theory of"
        ];

        let query_lower = query.to_lowercase();
        let is_conceptual = conceptual_patterns.iter()
            .any(|pattern| query_lower.contains(pattern));

        if is_conceptual {
            // Extract nouns after conceptual markers
            for (i, word) in words.iter().enumerate() {
                let word_lower = word.to_lowercase();

                // Check if this follows a conceptual marker
                if i > 0 {
                    let prev_lower = words[i - 1].to_lowercase();
                    if ["is", "of", "about"].contains(&prev_lower.as_str()) {
                        // Extract the concept (remove punctuation)
                        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                        if !clean.is_empty() && clean.len() > 2 {
                            concepts.push(clean.to_string());
                        }
                    }
                }

                // Also extract words that look like abstract concepts
                if Self::is_likely_concept(&word_lower) {
                    let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if !clean.is_empty() && !concepts.contains(&clean.to_string()) {
                        concepts.push(clean.to_string());
                    }
                }
            }
        }

        // Fallback: extract important nouns
        if concepts.is_empty() {
            for word in words {
                if word.len() > 4 && word.chars().next().unwrap().is_uppercase() {
                    let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if !clean.is_empty() {
                        concepts.push(clean.to_string());
                    }
                }
            }
        }

        concepts
    }

    /// Check if a word is likely an abstract concept
    fn is_likely_concept(word: &str) -> bool {
        // Common abstract concept patterns
        let concept_words = [
            "love", "virtue", "justice", "truth", "beauty", "good", "evil",
            "knowledge", "wisdom", "courage", "philosophy", "ethics", "morality",
            "freedom", "happiness", "meaning", "purpose", "existence", "reality",
            "consciousness", "mind", "soul", "spirit", "nature", "essence"
        ];

        concept_words.contains(&word)
    }

    /// Ground a concept to concrete entities in the graph
    ///
    /// # Arguments
    ///
    /// * `concept` - The concept to ground
    ///
    /// # Returns
    ///
    /// Vector of entity IDs that are related to this concept
    fn ground_concept(&self, concept: &str) -> Vec<EntityId> {
        let mut grounded = Vec::new();
        let concept_lower = concept.to_lowercase();

        // Search entities by name/type matching
        for entity in self.graph.entities() {
            let entity_name_lower = entity.name.to_lowercase();
            let entity_type_lower = entity.entity_type.to_lowercase();

            // Direct match
            if entity_name_lower.contains(&concept_lower) {
                grounded.push(entity.id.clone());
                continue;
            }

            // Type match (e.g., concept "love" matches entity type "CONCEPT")
            if entity_type_lower == "concept" && entity_name_lower.contains(&concept_lower) {
                grounded.push(entity.id.clone());
                continue;
            }

            // Relationship match: entities that have relationships mentioning this concept
            for rel in self.graph.get_entity_relationships(&entity.id.0) {
                if rel.relation_type.to_lowercase().contains(&concept_lower) {
                    grounded.push(entity.id.clone());
                    break;
                }
            }
        }

        grounded
    }

    /// Calculate relevance score for an anchor
    ///
    /// Combines entity count, PageRank scores, and semantic similarity
    fn calculate_relevance(&self, anchor: &SymbolicAnchor) -> f32 {
        if anchor.grounded_entities.is_empty() {
            return 0.0;
        }

        // Base score from number of groundings (normalized)
        let count_score = (anchor.grounded_entities.len() as f32 / 10.0).min(1.0);

        // PageRank-based boost if scores are available
        if let Some(ref pagerank) = self.pagerank_scores {
            // Average PageRank score of grounded entities
            let mut total_pr = 0.0;
            let mut found_count = 0;

            for entity_id in &anchor.grounded_entities {
                if let Some(&pr_score) = pagerank.get(entity_id) {
                    total_pr += pr_score;
                    found_count += 1;
                }
            }

            if found_count > 0 {
                let avg_pr = total_pr / found_count as f32;
                // Combine count score (40%) and PageRank score (60%)
                // PageRank weighted higher as it indicates importance
                return (count_score * 0.4) + (avg_pr * 0.6);
            }
        }

        // Fallback: use simple count-based relevance
        count_score
    }

    /// Boost search results using symbolic anchors
    ///
    /// # Arguments
    ///
    /// * `results` - Original search results
    /// * `anchors` - Symbolic anchors extracted from query
    ///
    /// # Returns
    ///
    /// Search results with boosted scores for anchor-matched entities
    pub fn boost_with_anchors(
        &self,
        mut results: Vec<SearchResult>,
        anchors: &[SymbolicAnchor],
    ) -> Vec<SearchResult> {
        if anchors.is_empty() {
            return results;
        }

        // Create entity name -> anchor mapping for fast lookup
        let mut entity_anchors: HashMap<String, Vec<&SymbolicAnchor>> = HashMap::new();
        for anchor in anchors {
            for entity_id in &anchor.grounded_entities {
                // Convert EntityId to string for matching
                let entity_str = entity_id.0.clone();
                entity_anchors
                    .entry(entity_str)
                    .or_insert_with(Vec::new)
                    .push(anchor);
            }
        }

        // Boost results that match anchors
        for result in &mut results {
            // Check if any of the result's entities match our anchors
            let mut total_boost = 0.0;
            let mut match_count = 0;

            for entity_name in &result.entities {
                // Try to match by entity name
                if let Some(matching_anchors) = entity_anchors.get(entity_name) {
                    let boost: f32 = matching_anchors
                        .iter()
                        .map(|a| a.relevance_score)
                        .sum::<f32>()
                        / matching_anchors.len() as f32;

                    total_boost += boost;
                    match_count += 1;
                }
            }

            // Apply accumulated boost
            if match_count > 0 {
                let avg_boost = total_boost / match_count as f32;
                let original_score = result.score;
                result.score *= 1.0 + avg_boost;

                #[cfg(feature = "tracing")]
                tracing::debug!(
                    result_id = %result.id,
                    original_score = original_score,
                    boost = avg_boost,
                    boosted_score = result.score,
                    matched_entities = match_count,
                    "Applied symbolic anchor boost"
                );
            }
        }

        // Re-sort by boosted scores
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

/// Detect if a query is conceptual vs factual
///
/// Conceptual queries ask about abstract ideas ("What is love?")
/// Factual queries ask about specific facts ("Who taught Plato?")
pub fn is_conceptual_query(query: &str) -> bool {
    let query_lower = query.to_lowercase();

    // Conceptual question patterns
    let conceptual_patterns = [
        "what is",
        "what are",
        "nature of",
        "meaning of",
        "definition of",
        "concept of",
        "idea of",
        "philosophy of",
        "theory of",
        "how does",
        "why does",
        "explain",
    ];

    conceptual_patterns
        .iter()
        .any(|pattern| query_lower.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Entity;

    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Add test entities
        let love_entity = Entity::new(
            EntityId::new("concept_love".to_string()),
            "love".to_string(),
            "CONCEPT".to_string(),
            0.9,
        );
        graph.add_entity(love_entity).unwrap();

        let phaedrus = Entity::new(
            EntityId::new("dialog_phaedrus".to_string()),
            "Phaedrus".to_string(),
            "DIALOG".to_string(),
            0.95,
        );
        graph.add_entity(phaedrus).unwrap();

        graph
    }

    #[test]
    fn test_is_conceptual_query() {
        assert!(is_conceptual_query("What is the nature of love?"));
        assert!(is_conceptual_query("Explain the concept of virtue"));
        assert!(is_conceptual_query("What are the key ideas in Platonism?"));

        assert!(!is_conceptual_query("Who taught Plato?"));
        assert!(!is_conceptual_query("When was Socrates born?"));
    }

    #[test]
    fn test_extract_concepts() {
        let graph = Arc::new(create_test_graph());
        let strategy = SymbolicAnchoringStrategy::new(graph);

        let concepts = strategy.extract_concepts("What is the nature of love?");
        assert!(concepts.contains(&"love".to_string()));
    }

    #[test]
    fn test_is_likely_concept() {
        assert!(SymbolicAnchoringStrategy::is_likely_concept("love"));
        assert!(SymbolicAnchoringStrategy::is_likely_concept("virtue"));
        assert!(SymbolicAnchoringStrategy::is_likely_concept("justice"));

        assert!(!SymbolicAnchoringStrategy::is_likely_concept("table"));
        assert!(!SymbolicAnchoringStrategy::is_likely_concept("book"));
    }

    #[test]
    fn test_symbolic_anchor_creation() {
        let mut anchor = SymbolicAnchor::new("love".to_string(), 0.8);
        anchor.add_entity(EntityId::new("entity1".to_string()));
        anchor.add_entity(EntityId::new("entity2".to_string()));

        assert_eq!(anchor.concept, "love");
        assert_eq!(anchor.grounded_entities.len(), 2);
        assert_eq!(anchor.relevance_score, 0.8);
    }

    #[test]
    fn test_extract_anchors() {
        let graph = Arc::new(create_test_graph());
        let strategy = SymbolicAnchoringStrategy::new(graph);

        let anchors = strategy.extract_anchors("What is the nature of love?");

        // Should extract "love" as anchor
        assert!(!anchors.is_empty());
        assert!(anchors.iter().any(|a| a.concept == "love"));
    }

    #[test]
    fn test_pagerank_boost() {
        let graph = Arc::new(create_test_graph());

        // Create PageRank scores - "dialog_phaedrus" is more important
        let mut pagerank_scores = HashMap::new();
        pagerank_scores.insert(EntityId::new("concept_love".to_string()), 0.3);
        pagerank_scores.insert(EntityId::new("dialog_phaedrus".to_string()), 0.9);

        let strategy = SymbolicAnchoringStrategy::new(graph)
            .with_pagerank_scores(pagerank_scores);

        // Create anchor with both entities
        let mut anchor = SymbolicAnchor::new("love".to_string(), 0.8);
        anchor.add_entity(EntityId::new("concept_love".to_string()));
        anchor.add_entity(EntityId::new("dialog_phaedrus".to_string()));

        let relevance = strategy.calculate_relevance(&anchor);

        // With PageRank: count_score=0.2 (2/10), avg_pr=0.6 (0.3+0.9)/2
        // Expected: 0.2*0.4 + 0.6*0.6 = 0.08 + 0.36 = 0.44
        assert!(relevance > 0.4 && relevance < 0.5, "Expected ~0.44, got {}", relevance);
    }

    #[test]
    fn test_pagerank_boost_fallback() {
        let graph = Arc::new(create_test_graph());

        // No PageRank scores provided
        let strategy = SymbolicAnchoringStrategy::new(graph);

        let mut anchor = SymbolicAnchor::new("love".to_string(), 0.8);
        anchor.add_entity(EntityId::new("concept_love".to_string()));
        anchor.add_entity(EntityId::new("dialog_phaedrus".to_string()));

        let relevance = strategy.calculate_relevance(&anchor);

        // Without PageRank: just count_score = 2/10 = 0.2
        assert!((relevance - 0.2).abs() < 0.01, "Expected 0.2, got {}", relevance);
    }
}
