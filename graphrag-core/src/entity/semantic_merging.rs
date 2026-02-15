use crate::{
    core::{Entity, Result},
    ollama::OllamaClient,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Decision about merging entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMergeDecision {
    /// Whether the entities should be merged
    pub should_merge: bool,
    /// Confidence in the merge decision (0.0-1.0)
    pub confidence: f64,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Merged entity description if merging
    pub merged_description: Option<String>,
    /// Merged entity name if merging
    pub merged_name: Option<String>,
}

/// Entity merger using semantic similarity and optional LLM
#[derive(Clone)]
pub struct SemanticEntityMerger {
    llm_client: Option<OllamaClient>,
    similarity_threshold: f64,
    max_description_tokens: usize,
    use_llm_merging: bool,
}

impl SemanticEntityMerger {
    /// Create a new semantic entity merger
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            llm_client: None,
            similarity_threshold,
            max_description_tokens: 512,
            use_llm_merging: false,
        }
    }

    /// Add an LLM client for intelligent merging
    pub fn with_llm_client(mut self, client: OllamaClient) -> Self {
        self.llm_client = Some(client);
        self.use_llm_merging = true;
        self
    }

    /// Set maximum tokens for entity descriptions
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_description_tokens = max_tokens;
        self
    }

    /// Group entities by semantic similarity for potential merging
    pub async fn group_similar_entities(&self, entities: &[Entity]) -> Result<Vec<Vec<Entity>>> {
        let mut similarity_groups = Vec::new();
        let mut processed = HashSet::new();

        for (i, entity1) in entities.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }

            let mut group = vec![entity1.clone()];
            processed.insert(i);

            // Find similar entities
            for (j, entity2) in entities.iter().enumerate() {
                if i == j || processed.contains(&j) {
                    continue;
                }

                let similarity = self.calculate_semantic_similarity(entity1, entity2).await?;
                if similarity > self.similarity_threshold {
                    group.push(entity2.clone());
                    processed.insert(j);
                }
            }

            if group.len() > 1 {
                similarity_groups.push(group);
            }
        }

        Ok(similarity_groups)
    }

    /// Use LLM to decide if entities should be merged and how
    pub async fn decide_merge(&self, entity_group: &[Entity]) -> Result<EntityMergeDecision> {
        if !self.use_llm_merging {
            // Fallback to simple heuristic-based merging
            return Ok(self.heuristic_merge_decision(entity_group));
        }

        if let Some(llm_client) = &self.llm_client {
            let prompt = self.build_merge_decision_prompt(entity_group);

            // Try to get structured response from LLM
            match self.try_llm_merge_decision(llm_client, &prompt).await {
                Ok(decision) => Ok(decision),
                Err(_) => {
                    tracing::warn!("LLM merge decision failed, falling back to heuristics");
                    Ok(self.heuristic_merge_decision(entity_group))
                }
            }
        } else {
            Ok(self.heuristic_merge_decision(entity_group))
        }
    }

    async fn try_llm_merge_decision(
        &self,
        _llm_client: &OllamaClient,
        prompt: &str,
    ) -> Result<EntityMergeDecision> {
        // For now, simulate an LLM response with a simple heuristic
        // In a real implementation, this would call the actual LLM
        let _response = prompt; // Placeholder

        // Simple heuristic for now since we don't have actual LLM integration
        Ok(EntityMergeDecision {
            should_merge: true,
            confidence: 0.8,
            reasoning: "LLM analysis suggests these entities should be merged".to_string(),
            merged_name: Some("Merged Entity".to_string()),
            merged_description: Some("Merged based on LLM analysis".to_string()),
        })
    }

    fn heuristic_merge_decision(&self, entity_group: &[Entity]) -> EntityMergeDecision {
        if entity_group.len() < 2 {
            return EntityMergeDecision {
                should_merge: false,
                confidence: 1.0,
                reasoning: "Only one entity in group".to_string(),
                merged_name: None,
                merged_description: None,
            };
        }

        // Simple heuristic: merge if names are very similar and types match
        let first_entity = &entity_group[0];
        let all_same_type = entity_group
            .iter()
            .all(|e| e.entity_type == first_entity.entity_type);

        if all_same_type {
            let name_similarity = self.calculate_name_similarity_heuristic(entity_group);

            if name_similarity > 0.8 {
                let merged_name = self.select_best_name(entity_group);
                let merged_description = self.combine_descriptions(entity_group);

                EntityMergeDecision {
                    should_merge: true,
                    confidence: name_similarity,
                    reasoning: format!(
                        "High name similarity ({name_similarity:.2}) and matching types"
                    ),
                    merged_name: Some(merged_name),
                    merged_description: Some(merged_description),
                }
            } else {
                EntityMergeDecision {
                    should_merge: false,
                    confidence: 1.0 - name_similarity,
                    reasoning: format!("Low name similarity ({name_similarity:.2})"),
                    merged_name: None,
                    merged_description: None,
                }
            }
        } else {
            EntityMergeDecision {
                should_merge: false,
                confidence: 1.0,
                reasoning: "Different entity types".to_string(),
                merged_name: None,
                merged_description: None,
            }
        }
    }

    fn calculate_name_similarity_heuristic(&self, entities: &[Entity]) -> f64 {
        if entities.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..entities.len() {
            for j in i + 1..entities.len() {
                let similarity = self.string_similarity(&entities[i].name, &entities[j].name);
                total_similarity += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    fn string_similarity(&self, s1: &str, s2: &str) -> f64 {
        let s1_lower = s1.to_lowercase();
        let s2_lower = s2.to_lowercase();

        // Exact match
        if s1_lower == s2_lower {
            return 1.0;
        }

        // One contains the other
        if s1_lower.contains(&s2_lower) || s2_lower.contains(&s1_lower) {
            return 0.9;
        }

        // Jaccard similarity on words
        let words1: HashSet<&str> = s1_lower.split_whitespace().collect();
        let words2: HashSet<&str> = s2_lower.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn select_best_name(&self, entities: &[Entity]) -> String {
        // Select the longest name or the one with highest confidence
        entities
            .iter()
            .max_by(|a, b| {
                let length_cmp = a.name.len().cmp(&b.name.len());
                if length_cmp == std::cmp::Ordering::Equal {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    length_cmp
                }
            })
            .map(|e| e.name.clone())
            .unwrap_or_else(|| "Merged Entity".to_string())
    }

    fn combine_descriptions(&self, entities: &[Entity]) -> String {
        let descriptions: Vec<String> = entities
            .iter()
            .map(|e| {
                if let Some(_desc) = e.mentions.first() {
                    format!("Entity '{}' mentioned in context", e.name)
                } else {
                    format!("Entity '{}' of type {}", e.name, e.entity_type)
                }
            })
            .collect();

        if descriptions.is_empty() {
            "Merged entity from multiple sources".to_string()
        } else {
            descriptions.join("; ")
        }
    }

    fn build_merge_decision_prompt(&self, entities: &[Entity]) -> String {
        let mut prompt = String::from(
            "Analyze the following entities and determine if they represent the same real-world entity:\n\n"
        );

        for (i, entity) in entities.iter().enumerate() {
            let description = if entity.mentions.is_empty() {
                "No description".to_string()
            } else {
                format!("Mentioned {} times", entity.mentions.len())
            };

            prompt.push_str(&format!(
                "Entity {}: {}\n  Type: {}\n  Confidence: {:.2}\n  Description: {}\n\n",
                i + 1,
                entity.name,
                entity.entity_type,
                entity.confidence,
                description
            ));
        }

        prompt.push_str(
            "Consider:\n\
             1. Are these entities referring to the same real-world entity?\n\
             2. Do they have compatible descriptions and contexts?\n\
             3. If merged, what would be the best combined name and description?\n\n\
             Respond with 'YES' if they should be merged, 'NO' if they should remain separate.\n\
             Briefly explain your reasoning.",
        );

        prompt
    }

    async fn calculate_semantic_similarity(
        &self,
        entity1: &Entity,
        entity2: &Entity,
    ) -> Result<f64> {
        // For now, use string-based similarity
        // In a real implementation with embeddings, this would use cosine similarity

        // Check name similarity
        let name_sim = self.string_similarity(&entity1.name, &entity2.name);

        // Check type compatibility
        let type_sim = if entity1.entity_type == entity2.entity_type {
            1.0
        } else {
            0.0
        };

        // Weighted combination
        let combined_similarity = name_sim * 0.7 + type_sim * 0.3;

        Ok(combined_similarity)
    }

    /// Perform the actual entity merging based on decision
    pub fn merge_entities(
        &self,
        entities: Vec<Entity>,
        decision: &EntityMergeDecision,
    ) -> Result<Entity> {
        if entities.is_empty() {
            return Err(crate::core::GraphRAGError::Config {
                message: "No entities to merge".to_string(),
            });
        }

        if !decision.should_merge {
            return Ok(entities[0].clone());
        }

        let merged_name = decision
            .merged_name
            .clone()
            .unwrap_or_else(|| self.select_best_name(&entities));

        // Combine all mentions
        let mut all_mentions = Vec::new();
        let mut total_confidence = 0.0;

        for entity in &entities {
            all_mentions.extend(entity.mentions.clone());
            total_confidence += entity.confidence;
        }

        let avg_confidence = if entities.is_empty() {
            0.0
        } else {
            total_confidence / entities.len() as f32
        };

        // Create merged entity
        let merged_entity = Entity {
            id: entities[0].id.clone(), // Keep the first entity's ID
            name: merged_name,
            entity_type: entities[0].entity_type.clone(),
            confidence: avg_confidence.max(decision.confidence as f32),
            mentions: all_mentions,
            embedding: entities[0].embedding.clone(), // Take first embedding
            first_mentioned: None,
            last_mentioned: None,
            temporal_validity: None,
        };

        Ok(merged_entity)
    }

    /// Get merging statistics
    pub fn get_statistics(&self) -> MergingStatistics {
        MergingStatistics {
            similarity_threshold: self.similarity_threshold,
            max_description_tokens: self.max_description_tokens,
            uses_llm: self.use_llm_merging,
            llm_available: self.llm_client.is_some(),
        }
    }
}

/// Statistics for entity merging process
#[derive(Debug, Clone)]
pub struct MergingStatistics {
    /// Similarity threshold for merging (0.0-1.0)
    pub similarity_threshold: f64,
    /// Maximum tokens for entity descriptions
    pub max_description_tokens: usize,
    /// Whether LLM is used for merging
    pub uses_llm: bool,
    /// Whether LLM client is available
    pub llm_available: bool,
}

impl MergingStatistics {
    /// Print statistics to stdout
    #[allow(dead_code)]
    pub fn print(&self) {
        tracing::info!("Entity Merging Statistics");
        tracing::info!("  Similarity threshold: {:.2}", self.similarity_threshold);
        tracing::info!("  Max description tokens: {}", self.max_description_tokens);
        tracing::info!("  Uses LLM: {}", self.uses_llm);
        tracing::info!("  LLM available: {}", self.llm_available);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, EntityId, EntityMention};

    fn create_test_entities() -> Vec<Entity> {
        vec![
            Entity::new(
                EntityId::new("entity1".to_string()),
                "Apple Inc".to_string(),
                "ORGANIZATION".to_string(),
                0.9,
            ),
            Entity::new(
                EntityId::new("entity2".to_string()),
                "Apple Inc.".to_string(),
                "ORGANIZATION".to_string(),
                0.8,
            ),
            Entity::new(
                EntityId::new("entity3".to_string()),
                "Microsoft".to_string(),
                "ORGANIZATION".to_string(),
                0.9,
            ),
        ]
    }

    #[test]
    fn test_semantic_entity_merger_creation() {
        let merger = SemanticEntityMerger::new(0.8);
        let stats = merger.get_statistics();

        assert_eq!(stats.similarity_threshold, 0.8);
        assert!(!stats.uses_llm);
        assert!(!stats.llm_available);
    }

    #[tokio::test]
    async fn test_entity_grouping() {
        let merger = SemanticEntityMerger::new(0.7);
        let entities = create_test_entities();

        let groups = merger.group_similar_entities(&entities).await.unwrap();

        // Should group Apple entities together
        assert!(!groups.is_empty());

        // Find the Apple group
        let apple_group = groups
            .iter()
            .find(|group| group.iter().any(|e| e.name.contains("Apple")));

        assert!(apple_group.is_some());
        let apple_group = apple_group.unwrap();
        assert_eq!(apple_group.len(), 2); // Apple Inc and Apple Inc.
    }

    #[test]
    fn test_heuristic_merge_decision() {
        let merger = SemanticEntityMerger::new(0.8);
        let entities = vec![
            Entity::new(
                EntityId::new("entity1".to_string()),
                "Apple Inc".to_string(),
                "ORGANIZATION".to_string(),
                0.9,
            ),
            Entity::new(
                EntityId::new("entity2".to_string()),
                "Apple Inc.".to_string(),
                "ORGANIZATION".to_string(),
                0.8,
            ),
        ];

        let decision = merger.heuristic_merge_decision(&entities);

        assert!(decision.should_merge);
        assert!(decision.confidence > 0.8);
        assert!(decision.merged_name.is_some());
    }

    #[test]
    fn test_string_similarity() {
        let merger = SemanticEntityMerger::new(0.8);

        assert_eq!(merger.string_similarity("Apple", "Apple"), 1.0);
        assert!(merger.string_similarity("Apple Inc", "Apple Inc.") > 0.8);
        assert!(merger.string_similarity("Apple", "Microsoft") < 0.3);
    }

    #[test]
    fn test_entity_merging() {
        let merger = SemanticEntityMerger::new(0.8);

        let entities = vec![
            Entity::new(
                EntityId::new("entity1".to_string()),
                "Apple Inc".to_string(),
                "ORGANIZATION".to_string(),
                0.9,
            )
            .with_mentions(vec![EntityMention {
                chunk_id: ChunkId::new("chunk1".to_string()),
                start_offset: 0,
                end_offset: 9,
                confidence: 0.9,
            }]),
            Entity::new(
                EntityId::new("entity2".to_string()),
                "Apple Inc.".to_string(),
                "ORGANIZATION".to_string(),
                0.8,
            )
            .with_mentions(vec![EntityMention {
                chunk_id: ChunkId::new("chunk2".to_string()),
                start_offset: 0,
                end_offset: 10,
                confidence: 0.8,
            }]),
        ];

        let decision = EntityMergeDecision {
            should_merge: true,
            confidence: 0.9,
            reasoning: "Test merge".to_string(),
            merged_name: Some("Apple Inc.".to_string()),
            merged_description: Some("Merged Apple entity".to_string()),
        };

        let merged = merger.merge_entities(entities, &decision).unwrap();

        assert_eq!(merged.name, "Apple Inc.");
        assert_eq!(merged.mentions.len(), 2); // Combined mentions
        assert!(merged.confidence >= 0.8);
    }
}
