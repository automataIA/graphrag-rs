//! Fuzzy matching for ROGRAG system
//!
//! Provides semantic similarity matching as a fallback when query decomposition
//! fails or when logic form retrieval is not applicable.

#[cfg(feature = "rograg")]
use crate::core::{EntityId, KnowledgeGraph};
#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use derive_more::Display;
#[cfg(feature = "rograg")]
use itertools::Itertools;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use tap::Pipe;
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Error types for fuzzy matching
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum FuzzyMatchError {
    #[error("No matching entities found for query: {query}")]
    NoMatches { query: String },

    #[error("Similarity threshold too low: {threshold}")]
    ThresholdTooLow { threshold: f32 },

    #[error("Graph is empty or invalid")]
    InvalidGraph,
}

/// Configuration for fuzzy matching
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct FuzzyMatchConfig {
    pub similarity_threshold: f32,
    pub max_matches: usize,
    pub enable_entity_matching: bool,
    pub enable_chunk_matching: bool,
    pub enable_semantic_expansion: bool,
    pub expansion_depth: usize,
    pub boost_exact_matches: bool,
}

#[cfg(feature = "rograg")]
impl Default for FuzzyMatchConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.6,
            max_matches: 10,
            enable_entity_matching: true,
            enable_chunk_matching: true,
            enable_semantic_expansion: true,
            expansion_depth: 2,
            boost_exact_matches: true,
        }
    }
}

/// Result from fuzzy matching
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatchResult {
    pub query: String,
    pub matches: Vec<FuzzyMatch>,
    pub confidence: f32,
    pub content: String,
    pub sources: Vec<String>,
    pub match_strategy: MatchStrategy,
}

/// A single fuzzy match
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatch {
    pub id: String,
    pub content: String,
    pub similarity: f32,
    pub match_type: MatchType,
    pub explanation: String,
    pub source_ids: Vec<String>,
}

/// Type of match found
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Display, PartialEq)]
pub enum MatchType {
    ExactEntity,    // Exact entity name match
    PartialEntity,  // Partial entity name match
    SemanticEntity, // Semantic similarity to entity
    ChunkContent,   // Content within text chunks
    RelatedConcept, // Related concept through graph traversal
}

/// Strategy used for matching
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Display)]
pub enum MatchStrategy {
    DirectMatching,
    SemanticExpansion,
    GraphTraversal,
    HybridApproach,
}

/// Fuzzy matcher implementation
#[cfg(feature = "rograg")]
pub struct FuzzyMatcher {
    config: FuzzyMatchConfig,
    // Cache removed for Arc compatibility
}

#[cfg(feature = "rograg")]
impl Default for FuzzyMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl FuzzyMatcher {
    /// Create a new fuzzy matcher with default configuration
    pub fn new() -> Self {
        Self::with_config(FuzzyMatchConfig::default())
    }

    /// Create a new fuzzy matcher with custom configuration
    pub fn with_config(config: FuzzyMatchConfig) -> Self {
        Self { config }
    }

    /// Match a query against the knowledge graph
    pub fn match_query(&self, query: &str, graph: &KnowledgeGraph) -> Result<FuzzyMatchResult> {
        if graph.entities().count() == 0 && graph.chunks().count() == 0 {
            return Err(FuzzyMatchError::InvalidGraph.into());
        }

        let mut all_matches = Vec::new();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        // Strategy 1: Direct entity matching
        if self.config.enable_entity_matching {
            let entity_matches = self.match_entities(query, &query_words, graph)?;
            all_matches.extend(entity_matches);
        }

        // Strategy 2: Chunk content matching
        if self.config.enable_chunk_matching {
            let chunk_matches = self.match_chunks(query, &query_words, graph)?;
            all_matches.extend(chunk_matches);
        }

        // Strategy 3: Semantic expansion through graph
        if self.config.enable_semantic_expansion && all_matches.len() < self.config.max_matches {
            let expanded_matches =
                self.semantic_expansion(query, &query_words, graph, &all_matches)?;
            all_matches.extend(expanded_matches);
        }

        // Sort by similarity and take top matches
        all_matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_matches.truncate(self.config.max_matches);

        if all_matches.is_empty() {
            return Err(FuzzyMatchError::NoMatches {
                query: query.to_string(),
            }
            .into());
        }

        // Calculate overall confidence and generate response
        let confidence = self.calculate_overall_confidence(&all_matches);
        let content = self.generate_response_content(&all_matches);
        let sources = self.extract_sources(&all_matches);
        let strategy = self.determine_strategy(&all_matches);

        Ok(FuzzyMatchResult {
            query: query.to_string(),
            matches: all_matches,
            confidence,
            content,
            sources,
            match_strategy: strategy,
        })
    }

    /// Match against entity names and types
    fn match_entities(
        &self,
        query: &str,
        query_words: &[&str],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<FuzzyMatch>> {
        let mut matches = Vec::new();

        for entity in graph.entities() {
            let entity_name_lower = entity.name.to_lowercase();
            let entity_type_lower = entity.entity_type.to_lowercase();

            // Exact name match (highest priority)
            if entity_name_lower == query.to_lowercase() {
                matches.push(FuzzyMatch {
                    id: entity.id.to_string(),
                    content: format!("{} ({})", entity.name, entity.entity_type),
                    similarity: if self.config.boost_exact_matches {
                        1.0
                    } else {
                        0.95
                    },
                    match_type: MatchType::ExactEntity,
                    explanation: "Exact entity name match".to_string(),
                    source_ids: vec![entity.id.to_string()],
                });
                continue;
            }

            // Partial name match
            let name_similarity = self.calculate_text_similarity(query, &entity.name);
            if name_similarity >= self.config.similarity_threshold {
                matches.push(FuzzyMatch {
                    id: entity.id.to_string(),
                    content: format!("{} ({})", entity.name, entity.entity_type),
                    similarity: name_similarity * 0.9, // Slight penalty for partial match
                    match_type: MatchType::PartialEntity,
                    explanation: format!("Partial name match: {name_similarity:.2}"),
                    source_ids: vec![entity.id.to_string()],
                });
            }

            // Entity type match
            let type_similarity =
                self.calculate_word_overlap(query_words, &[entity_type_lower.as_str()]);
            if type_similarity > 0.0 {
                matches.push(FuzzyMatch {
                    id: format!("type_{}", entity.id),
                    content: format!("{} ({})", entity.name, entity.entity_type),
                    similarity: type_similarity * 0.7, // Lower weight for type matches
                    match_type: MatchType::SemanticEntity,
                    explanation: format!("Entity type match: {}", entity.entity_type),
                    source_ids: vec![entity.id.to_string()],
                });
            }
        }

        Ok(matches)
    }

    /// Match against chunk content
    fn match_chunks(
        &self,
        query: &str,
        query_words: &[&str],
        graph: &KnowledgeGraph,
    ) -> Result<Vec<FuzzyMatch>> {
        let mut matches = Vec::new();

        for chunk in graph.chunks() {
            let chunk_content_lower = chunk.content.to_lowercase();

            // Direct content similarity
            let content_similarity = self.calculate_text_similarity(query, &chunk.content);
            if content_similarity >= self.config.similarity_threshold {
                let preview = Self::create_content_preview(&chunk.content, query, 200);

                matches.push(FuzzyMatch {
                    id: chunk.id.to_string(),
                    content: preview,
                    similarity: content_similarity,
                    match_type: MatchType::ChunkContent,
                    explanation: format!("Content similarity: {content_similarity:.2}"),
                    source_ids: vec![chunk.id.to_string()],
                });
            }

            // Keyword overlap
            let chunk_words: Vec<&str> = chunk_content_lower.split_whitespace().collect();
            let word_overlap = self.calculate_word_overlap(query_words, &chunk_words);
            if word_overlap >= self.config.similarity_threshold * 0.8 {
                let preview = Self::create_content_preview(&chunk.content, query, 200);

                matches.push(FuzzyMatch {
                    id: format!("keywords_{}", chunk.id),
                    content: preview,
                    similarity: word_overlap,
                    match_type: MatchType::ChunkContent,
                    explanation: format!("Keyword overlap: {word_overlap:.2}"),
                    source_ids: vec![chunk.id.to_string()],
                });
            }
        }

        Ok(matches)
    }

    /// Perform semantic expansion through graph relationships
    fn semantic_expansion(
        &self,
        _query: &str,
        _query_words: &[&str],
        graph: &KnowledgeGraph,
        existing_matches: &[FuzzyMatch],
    ) -> Result<Vec<FuzzyMatch>> {
        let mut matches = Vec::new();

        // Get entity IDs from existing matches
        let matched_entity_ids: Vec<String> = existing_matches
            .iter()
            .filter_map(|m| {
                if m.match_type == MatchType::ExactEntity
                    || m.match_type == MatchType::PartialEntity
                {
                    Some(m.id.clone())
                } else {
                    None
                }
            })
            .collect();

        // Expand through relationships
        for entity_id in &matched_entity_ids {
            if let Ok(entity_id_typed) =
                EntityId::new(entity_id.clone()).pipe(Ok::<EntityId, crate::GraphRAGError>)
            {
                let neighbors = graph.get_neighbors(&entity_id_typed);

                for (neighbor, relationship) in neighbors.iter().take(5) {
                    // Limit expansion
                    let relationship_similarity = relationship.confidence * 0.6; // Discount for indirect match

                    if relationship_similarity >= self.config.similarity_threshold * 0.7 {
                        matches.push(FuzzyMatch {
                            id: format!("expanded_{}", neighbor.id),
                            content: format!(
                                "{} ({}): {} via {}",
                                neighbor.name,
                                neighbor.entity_type,
                                relationship.relation_type,
                                graph
                                    .get_entity(&entity_id_typed)
                                    .map(|e| e.name.as_str())
                                    .unwrap_or("unknown")
                            ),
                            similarity: relationship_similarity,
                            match_type: MatchType::RelatedConcept,
                            explanation: format!(
                                "Related via {} (confidence: {:.2})",
                                relationship.relation_type, relationship.confidence
                            ),
                            source_ids: vec![neighbor.id.to_string(), entity_id.clone()],
                        });
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Calculate text similarity using multiple metrics
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();

        // Calculate multiple similarity metrics
        let jaccard = self.jaccard_similarity(&text1_lower, &text2_lower);
        let containment = self.containment_similarity(&text1_lower, &text2_lower);
        let levenshtein = self.levenshtein_similarity(&text1_lower, &text2_lower);

        // Weighted combination
        jaccard * 0.4 + containment * 0.4 + levenshtein * 0.2
    }

    /// Calculate Jaccard similarity between two texts
    fn jaccard_similarity(&self, text1: &str, text2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculate containment similarity
    fn containment_similarity(&self, text1: &str, text2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let smaller = words1.len().min(words2.len());
        if smaller == 0 {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        intersection as f32 / smaller as f32
    }

    /// Calculate Levenshtein-based similarity
    fn levenshtein_similarity(&self, text1: &str, text2: &str) -> f32 {
        let distance = self.levenshtein_distance(text1, text2);
        let max_len = text1.len().max(text2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f32 / max_len as f32)
        }
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, item) in matrix.iter_mut().enumerate().take(len1 + 1) {
            item[0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Calculate word overlap between two word lists
    fn calculate_word_overlap(&self, words1: &[&str], words2: &[&str]) -> f32 {
        let set1: std::collections::HashSet<&str> = words1.iter().copied().collect();
        let set2: std::collections::HashSet<&str> = words2.iter().copied().collect();

        let intersection = set1.intersection(&set2).count();
        let min_size = words1.len().min(words2.len());

        if min_size == 0 {
            0.0
        } else {
            intersection as f32 / min_size as f32
        }
    }

    /// Calculate overall confidence from matches
    fn calculate_overall_confidence(&self, matches: &[FuzzyMatch]) -> f32 {
        if matches.is_empty() {
            return 0.0;
        }

        // Weight by match type and position
        let weighted_sum: f32 = matches
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let position_weight = 1.0 / (idx as f32 + 1.0); // Higher weight for earlier matches
                let type_weight = match m.match_type {
                    MatchType::ExactEntity => 1.0,
                    MatchType::PartialEntity => 0.8,
                    MatchType::SemanticEntity => 0.7,
                    MatchType::ChunkContent => 0.6,
                    MatchType::RelatedConcept => 0.5,
                };
                m.similarity * position_weight * type_weight
            })
            .sum();

        let total_weight: f32 = matches
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let position_weight = 1.0 / (idx as f32 + 1.0);
                let type_weight = match m.match_type {
                    MatchType::ExactEntity => 1.0,
                    MatchType::PartialEntity => 0.8,
                    MatchType::SemanticEntity => 0.7,
                    MatchType::ChunkContent => 0.6,
                    MatchType::RelatedConcept => 0.5,
                };
                position_weight * type_weight
            })
            .sum();

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Generate response content from matches
    fn generate_response_content(&self, matches: &[FuzzyMatch]) -> String {
        if matches.is_empty() {
            return "No relevant information found.".to_string();
        }

        let top_matches = matches.iter().take(3).collect::<Vec<_>>();

        let content = top_matches
            .iter()
            .map(|m| format!("• {} (similarity: {:.2})", m.content, m.similarity))
            .join("\n");

        if matches.len() > 3 {
            format!("{}\n... and {} more results", content, matches.len() - 3)
        } else {
            content
        }
    }

    /// Extract source IDs from matches
    fn extract_sources(&self, matches: &[FuzzyMatch]) -> Vec<String> {
        matches
            .iter()
            .flat_map(|m| m.source_ids.iter())
            .cloned()
            .unique()
            .collect()
    }

    /// Determine the primary strategy used
    fn determine_strategy(&self, matches: &[FuzzyMatch]) -> MatchStrategy {
        if matches.is_empty() {
            return MatchStrategy::DirectMatching;
        }

        let has_entity_matches = matches.iter().any(|m| {
            matches!(
                m.match_type,
                MatchType::ExactEntity | MatchType::PartialEntity
            )
        });
        let has_chunk_matches = matches
            .iter()
            .any(|m| matches!(m.match_type, MatchType::ChunkContent));
        let has_expanded_matches = matches
            .iter()
            .any(|m| matches!(m.match_type, MatchType::RelatedConcept));

        match (has_entity_matches, has_chunk_matches, has_expanded_matches) {
            (true, true, true) => MatchStrategy::HybridApproach,
            (_, _, true) => MatchStrategy::SemanticExpansion,
            (true, true, false) => MatchStrategy::HybridApproach,
            _ => MatchStrategy::DirectMatching,
        }
    }

    /// Create a content preview with query context
    fn create_content_preview(content: &str, query: &str, max_length: usize) -> String {
        let content_lower = content.to_lowercase();
        let query_lower = query.to_lowercase();

        // Try to find query terms in content
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let mut best_position = 0;
        let mut best_score = 0;

        // Find the position with the most query words
        for (pos, window) in content_lower
            .chars()
            .collect::<Vec<_>>()
            .windows(max_length)
            .enumerate()
        {
            let window_str: String = window.iter().collect();
            let score = query_words
                .iter()
                .filter(|&&word| window_str.contains(word))
                .count();

            if score > best_score {
                best_score = score;
                best_position = pos;
            }
        }

        // Extract content around the best position
        let chars: Vec<char> = content.chars().collect();
        let start = best_position;
        let end = (start + max_length).min(chars.len());

        let preview: String = chars[start..end].iter().collect();

        if start > 0 || end < chars.len() {
            format!("...{}...", preview.trim())
        } else {
            preview.trim().to_string()
        }
    }

    /// Get configuration
    pub fn get_config(&self) -> &FuzzyMatchConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: FuzzyMatchConfig) {
        self.config = config;
        // Clear cache if thresholds changed
    }

    /// Clear similarity cache
    pub fn clear_cache(&self) {}

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (0, 0) // Cache disabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId, Entity, KnowledgeGraph, TextChunk};

    #[cfg(feature = "rograg")]
    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Add test entities
        let entity1 = Entity {
            id: EntityId::new("entity_1".to_string()),
            name: "Entity Name".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 1.0,
            mentions: vec![],
            embedding: None,
        };

        let entity2 = Entity {
            id: EntityId::new("entity_2".to_string()),
            name: "Second Entity".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 1.0,
            mentions: vec![],
            embedding: None,
        };

        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();

        // Add test chunks
        let chunk = TextChunk::new(
            ChunkId::new("chunk1".to_string()),
            DocumentId::new("doc1".to_string()),
            "Entity Name is a character who interacts with other entities. It is associated with Second Entity and they have various relationships.".to_string(),
            0,
            100,
        );

        graph.add_chunk(chunk).unwrap();

        graph
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_exact_entity_match() {
        let matcher = FuzzyMatcher::new();
        let graph = create_test_graph();

        let result = matcher.match_query("Entity Name", &graph).unwrap();

        assert!(!result.matches.is_empty());
        assert!(result
            .matches
            .iter()
            .any(|m| m.match_type == MatchType::ExactEntity));
        assert!(result.confidence > 0.9);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_partial_entity_match() {
        let matcher = FuzzyMatcher::new();
        let graph = create_test_graph();

        let result = matcher.match_query("Entity", &graph).unwrap();

        assert!(!result.matches.is_empty());
        assert!(result.confidence > 0.6);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_chunk_content_match() {
        let matcher = FuzzyMatcher::new();
        let graph = create_test_graph();

        let result = matcher.match_query("character relationships", &graph).unwrap();

        assert!(!result.matches.is_empty());
        assert!(result
            .matches
            .iter()
            .any(|m| m.match_type == MatchType::ChunkContent));
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_similarity_calculations() {
        let matcher = FuzzyMatcher::new();

        let jaccard = matcher.jaccard_similarity("hello world", "hello universe");
        assert!(jaccard > 0.0 && jaccard < 1.0);

        let levenshtein = matcher.levenshtein_similarity("kitten", "sitting");
        assert!(levenshtein > 0.0 && levenshtein < 1.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_no_matches() {
        let matcher = FuzzyMatcher::new();
        let graph = create_test_graph();

        let result = matcher.match_query("completely unrelated query", &graph);
        assert!(result.is_err());
    }
}
