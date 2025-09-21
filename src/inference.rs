//! Implicit relationship inference system

use crate::core::{Entity, EntityId, KnowledgeGraph, TextChunk};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct InferredRelation {
    pub source: EntityId,
    pub target: EntityId,
    pub relation_type: String,
    pub confidence: f32,
    pub evidence_count: usize,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub min_confidence: f32,
    pub max_candidates: usize,
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

pub struct InferenceEngine {
    config: InferenceConfig,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        Self { config }
    }

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

    // Helper function to extract clean entity name from ID
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

    // Calculate proximity score between entities in text
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

    // Check if entities are mentioned near a relationship pattern
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
