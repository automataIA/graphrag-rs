//! Implicit relationship inference system

use crate::core::{Entity, EntityId, KnowledgeGraph, TextChunk};
use std::collections::HashMap;

/// Represents a relationship inferred between two entities
///
/// This structure contains information about a relationship discovered through
/// co-occurrence analysis and contextual pattern matching.
#[derive(Debug, Clone)]
pub struct InferredRelation {
    /// Source entity in the relationship
    pub source: EntityId,
    /// Target entity in the relationship
    pub target: EntityId,
    /// Type of relationship (e.g., "FRIENDS", "DISCUSSES", "LOCATED_IN")
    pub relation_type: String,
    /// Confidence score for this inference (0.0-1.0)
    pub confidence: f32,
    /// Number of text chunks providing evidence for this relationship
    pub evidence_count: usize,
}

/// Configuration for the relationship inference engine
///
/// Controls the behavior and thresholds used when inferring implicit relationships
/// between entities based on their co-occurrence in text.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Minimum confidence threshold for accepting an inferred relationship
    pub min_confidence: f32,
    /// Maximum number of candidate relationships to return per query
    pub max_candidates: usize,
    /// Threshold for determining if entities co-occur frequently enough
    pub co_occurrence_threshold: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            max_candidates: 10,
            co_occurrence_threshold: 0.4,
        }
    }
}

/// Engine for inferring implicit relationships between entities
///
/// The inference engine analyzes entity co-occurrence patterns, proximity,
/// and contextual clues to discover relationships that may not be explicitly
/// stated in the text.
pub struct InferenceEngine {
    /// Configuration controlling inference behavior
    config: InferenceConfig,
}

impl InferenceEngine {
    /// Create a new inference engine with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration controlling inference thresholds and limits
    pub fn new(config: InferenceConfig) -> Self {
        Self { config }
    }

    /// Infer relationships for a target entity
    ///
    /// Analyzes the knowledge graph to find entities that frequently co-occur with
    /// the target entity and have contextual evidence of a relationship.
    ///
    /// # Arguments
    ///
    /// * `target_entity` - The entity to find relationships for
    /// * `relation_type` - The type of relationship to infer (e.g., "FRIENDS")
    /// * `knowledge_graph` - The knowledge graph containing entities and chunks
    ///
    /// # Returns
    ///
    /// Returns a vector of inferred relationships, sorted by confidence score,
    /// limited to `max_candidates` from the configuration.
    pub fn infer_relationships(
        &self,
        target_entity: &EntityId,
        relation_type: &str,
        knowledge_graph: &KnowledgeGraph,
    ) -> Vec<InferredRelation> {
        let mut inferred_relations = Vec::new();

        // Find target entity
        let target_ent = knowledge_graph.entities().find(|e| &e.id == target_entity);

        if target_ent.is_none() {
            return inferred_relations;
        }

        // Get chunks containing target entity
        let target_chunks: Vec<_> = knowledge_graph
            .chunks()
            .filter(|chunk| chunk.entities.contains(target_entity))
            .collect();

        // Find co-occurring entities
        let mut entity_scores: HashMap<EntityId, f32> = HashMap::new();

        for chunk in &target_chunks {
            for entity_id in &chunk.entities {
                if entity_id != target_entity {
                    let evidence_score =
                        self.calculate_evidence_score(chunk, target_entity, entity_id);
                    *entity_scores.entry(entity_id.clone()).or_insert(0.0) += evidence_score;
                }
            }
        }

        // Create inferred relations for high-scoring entities
        for (entity_id, score) in entity_scores {
            let normalized_score = (score / target_chunks.len() as f32).min(1.0);

            if normalized_score >= self.config.min_confidence {
                inferred_relations.push(InferredRelation {
                    source: target_entity.clone(),
                    target: entity_id,
                    relation_type: relation_type.to_string(),
                    confidence: normalized_score,
                    evidence_count: target_chunks.len(),
                });
            }
        }

        // Sort by confidence and limit results
        inferred_relations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        inferred_relations.truncate(self.config.max_candidates);

        inferred_relations
    }

    /// Calculate evidence score for a potential relationship
    ///
    /// Analyzes a text chunk to determine how strongly it suggests a relationship
    /// between two entities. Uses proximity analysis, pattern matching, and
    /// contextual clues.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The text chunk containing both entities
    /// * `entity_a` - First entity ID
    /// * `entity_b` - Second entity ID
    ///
    /// # Returns
    ///
    /// Returns a score between 0.0 and 1.0 indicating relationship strength.
    /// Higher scores indicate stronger evidence of a relationship.
    fn calculate_evidence_score(
        &self,
        chunk: &TextChunk,
        entity_a: &EntityId,
        entity_b: &EntityId,
    ) -> f32 {
        let content = &chunk.content.to_lowercase();
        let mut score: f32 = 0.2; // Lower base co-occurrence score

        // Get entity names for contextual analysis
        let entity_a_name = self.extract_entity_name(entity_a);
        let entity_b_name = self.extract_entity_name(entity_b);

        // Calculate proximity score between entities in text
        let proximity_bonus =
            self.calculate_proximity_score(content, &entity_a_name, &entity_b_name);
        score += proximity_bonus;

        // Enhanced friendship indicators with contextual patterns
        let friendship_patterns = [
            // Direct friendship terms
            ("best friend", 0.8),
            ("close friend", 0.7),
            ("good friend", 0.6),
            ("friend", 0.4),
            ("friends", 0.4),
            ("friendship", 0.5),
            // Activity-based friendship indicators
            ("played together", 0.6),
            ("went together", 0.5),
            ("talked with", 0.4),
            ("helped each other", 0.7),
            ("shared", 0.3),
            ("together", 0.3),
            // Emotional bonding indicators
            ("trusted", 0.6),
            ("loyal", 0.5),
            ("bond", 0.5),
            ("close", 0.4),
            ("cared for", 0.6),
            ("looked after", 0.5),
            ("protected", 0.6),
            // Adventure/activity companionship
            ("adventure", 0.4),
            ("explore", 0.3),
            ("journey", 0.3),
            ("companion", 0.6),
            ("partner", 0.5),
            ("ally", 0.5),
        ];

        // Contextual pattern matching with weighted scores
        for (pattern, weight) in &friendship_patterns {
            if content.contains(pattern) {
                // Additional context bonus if entities are mentioned near the pattern
                let context_bonus =
                    if self.entities_near_pattern(content, &entity_a_name, &entity_b_name, pattern)
                    {
                        weight * 0.5
                    } else {
                        *weight * 0.3
                    };
                score += context_bonus;
            }
        }

        // Enhanced negative indicators with contextual analysis
        let negative_patterns = [
            ("enemy", -0.8),
            ("enemies", -0.8),
            ("rival", -0.6),
            ("rivals", -0.6),
            ("fought", -0.5),
            ("fight", -0.4),
            ("battle", -0.4),
            ("conflict", -0.5),
            ("angry at", -0.6),
            ("hate", -0.7),
            ("hated", -0.7),
            ("despise", -0.6),
            ("betrayed", -0.8),
            ("betrayal", -0.7),
            ("argued", -0.3),
            ("quarrel", -0.4),
            ("against", -0.2),
            ("opposed", -0.4),
            ("disagree", -0.2),
        ];

        for (pattern, weight) in &negative_patterns {
            if content.contains(pattern) {
                let context_penalty =
                    if self.entities_near_pattern(content, &entity_a_name, &entity_b_name, pattern)
                    {
                        weight * 1.2
                    } else {
                        weight * 0.8
                    };
                score += context_penalty; // weight is already negative
            }
        }

        // Family relationship indicators (neutral for friendship)
        let family_patterns = ["brother", "sister", "cousin", "aunt", "uncle", "family"];
        let mut has_family_relation = false;
        for pattern in &family_patterns {
            if content.contains(pattern) {
                has_family_relation = true;
                break;
            }
        }

        // Family relations can still be friendships, but lower weight
        if has_family_relation {
            score *= 0.8;
        }

        score.clamp(0.0, 1.0)
    }

    /// Extract clean entity name from an entity ID
    ///
    /// Entity IDs typically have format "TYPE_normalized_name". This method
    /// extracts just the name portion and formats it for matching.
    ///
    /// # Arguments
    ///
    /// * `entity_id` - The entity ID to extract the name from
    ///
    /// # Returns
    ///
    /// Returns the cleaned, lowercase entity name with underscores replaced by spaces
    fn extract_entity_name(&self, entity_id: &EntityId) -> String {
        // EntityId format is typically "TYPE_normalized_name"
        let id_str = &entity_id.0;
        if let Some(underscore_pos) = id_str.find('_') {
            id_str[underscore_pos + 1..]
                .replace('_', " ")
                .to_lowercase()
        } else {
            id_str.to_lowercase()
        }
    }

    /// Calculate proximity score between entities in text
    ///
    /// Determines how close two entities are mentioned in the text. Closer proximity
    /// suggests a stronger relationship between the entities.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to analyze
    /// * `entity_a` - Name of the first entity
    /// * `entity_b` - Name of the second entity
    ///
    /// # Returns
    ///
    /// Returns a proximity score:
    /// - 0.4 for very close (0-2 words apart)
    /// - 0.3 for close (3-5 words apart)
    /// - 0.2 for medium distance (6-10 words apart)
    /// - 0.1 for far (11-20 words apart)
    /// - 0.05 for very far (20+ words apart)
    fn calculate_proximity_score(&self, content: &str, entity_a: &str, entity_b: &str) -> f32 {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut positions_a = Vec::new();
        let mut positions_b = Vec::new();

        // Find all positions of entity mentions
        for (i, word) in words.iter().enumerate() {
            if word.to_lowercase().contains(entity_a) {
                positions_a.push(i);
            }
            if word.to_lowercase().contains(entity_b) {
                positions_b.push(i);
            }
        }

        if positions_a.is_empty() || positions_b.is_empty() {
            return 0.0;
        }

        // Find minimum distance between any mentions
        let mut min_distance = usize::MAX;
        for &pos_a in &positions_a {
            for &pos_b in &positions_b {
                let distance = pos_a.abs_diff(pos_b);
                min_distance = min_distance.min(distance);
            }
        }

        // Convert distance to proximity score (closer = higher score)
        match min_distance {
            0..=2 => 0.4,   // Very close (same sentence likely)
            3..=5 => 0.3,   // Close
            6..=10 => 0.2,  // Medium distance
            11..=20 => 0.1, // Far
            _ => 0.05,      // Very far
        }
    }

    /// Check if entities are mentioned near a relationship pattern
    ///
    /// Determines if both entities appear within a 200-character window
    /// around a relationship keyword or pattern. This helps determine if
    /// a relationship pattern actually applies to these specific entities.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to search
    /// * `entity_a` - Name of the first entity
    /// * `entity_b` - Name of the second entity
    /// * `pattern` - The relationship pattern to search for (e.g., "friend", "enemy")
    ///
    /// # Returns
    ///
    /// Returns `true` if both entities are found within 100 characters before and
    /// after the pattern, `false` otherwise.
    fn entities_near_pattern(
        &self,
        content: &str,
        entity_a: &str,
        entity_b: &str,
        pattern: &str,
    ) -> bool {
        if let Some(pattern_pos) = content.find(pattern) {
            let start = pattern_pos.saturating_sub(100); // 100 chars before
            let end = (pattern_pos + pattern.len() + 100).min(content.len()); // 100 chars after
            let context = &content[start..end];

            context.contains(entity_a) && context.contains(entity_b)
        } else {
            false
        }
    }

    /// Find an entity in the knowledge graph by name
    ///
    /// Performs a case-insensitive substring search to find an entity whose
    /// name contains the given search string.
    ///
    /// # Arguments
    ///
    /// * `knowledge_graph` - The knowledge graph to search
    /// * `name` - The name (or partial name) to search for
    ///
    /// # Returns
    ///
    /// Returns `Some(&Entity)` if a matching entity is found, `None` otherwise.
    pub fn find_entity_by_name<'a>(
        &self,
        knowledge_graph: &'a KnowledgeGraph,
        name: &str,
    ) -> Option<&'a Entity> {
        knowledge_graph
            .entities()
            .find(|e| e.name.to_lowercase().contains(&name.to_lowercase()))
    }
}
