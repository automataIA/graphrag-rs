//! Incremental update system for LightRAG

use std::collections::{HashMap, VecDeque};
use crate::core::Result;
use crate::lightrag::graph_indexer::{EntityReference, ConceptSummary, ExtractionResult};

/// Conflict resolution strategies for incremental updates
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictStrategy {
    /// Merge conflicting entities/relations
    Merge,
    /// Replace old with new
    Replace,
    /// Skip new if conflict exists
    Skip,
}

/// Types of graph updates
#[derive(Debug, Clone)]
pub enum GraphUpdate {
    AddEntity(EntityReference),
    UpdateEntity(EntityReference),
    AddConcept(ConceptSummary),
    UpdateConcept(ConceptSummary),
    DeleteEntity(String),
    DeleteConcept(String),
}

/// Statistics from an incremental update
#[derive(Debug, Clone)]
pub struct UpdateStats {
    pub entities_added: usize,
    pub entities_updated: usize,
    pub concepts_added: usize,
    pub concepts_updated: usize,
    pub entities_deleted: usize,
    pub concepts_deleted: usize,
    pub tokens_used: usize,
    pub conflicts_resolved: usize,
}

/// Incremental graph updater implementing LightRAG's union-based approach
pub struct IncrementalGraphUpdater {
    conflict_strategy: ConflictStrategy,
    update_queue: VecDeque<GraphUpdate>,
    entity_index: HashMap<String, EntityReference>,
    concept_index: HashMap<String, ConceptSummary>,
    update_history: Vec<UpdateBatch>,
}

impl IncrementalGraphUpdater {
    pub fn new(conflict_strategy: ConflictStrategy) -> Result<Self> {
        Ok(Self {
            conflict_strategy,
            update_queue: VecDeque::new(),
            entity_index: HashMap::new(),
            concept_index: HashMap::new(),
            update_history: Vec::new(),
        })
    }

    /// Apply incremental update using union operation (not reconstruction)
    pub fn apply_incremental_update(&mut self, new_data: ExtractionResult) -> Result<UpdateStats> {
        let mut stats = UpdateStats {
            entities_added: 0,
            entities_updated: 0,
            concepts_added: 0,
            concepts_updated: 0,
            entities_deleted: 0,
            concepts_deleted: 0,
            tokens_used: new_data.tokens_used,
            conflicts_resolved: 0,
        };

        println!("🔄 Applying incremental update...");
        println!("   New entities: {}", new_data.entities.len());
        println!("   New concepts: {}", new_data.concepts.len());

        // Phase 1: Process new entities
        for entity in new_data.entities {
            match self.process_entity_update(entity)? {
                EntityUpdateResult::Added => stats.entities_added += 1,
                EntityUpdateResult::Updated => stats.entities_updated += 1,
                EntityUpdateResult::ConflictResolved => {
                    stats.entities_updated += 1;
                    stats.conflicts_resolved += 1;
                }
                EntityUpdateResult::Skipped => {
                    // Entity was skipped due to conflict strategy
                }
            }
        }

        // Phase 2: Process new concepts
        for concept in new_data.concepts {
            match self.process_concept_update(concept)? {
                ConceptUpdateResult::Added => stats.concepts_added += 1,
                ConceptUpdateResult::Updated => stats.concepts_updated += 1,
                ConceptUpdateResult::ConflictResolved => {
                    stats.concepts_updated += 1;
                    stats.conflicts_resolved += 1;
                }
                ConceptUpdateResult::Skipped => {
                    // Concept was skipped due to conflict strategy
                }
            }
        }

        // Phase 3: Process queued updates
        self.process_update_queue(&mut stats)?;

        // Phase 4: Record update batch
        let batch = UpdateBatch {
            timestamp: std::time::SystemTime::now(),
            stats: stats.clone(),
        };
        self.update_history.push(batch);

        println!("✅ Incremental update completed:");
        println!("   Entities added: {}", stats.entities_added);
        println!("   Entities updated: {}", stats.entities_updated);
        println!("   Concepts added: {}", stats.concepts_added);
        println!("   Concepts updated: {}", stats.concepts_updated);
        println!("   Conflicts resolved: {}", stats.conflicts_resolved);
        println!("   Tokens used: {}", stats.tokens_used);

        Ok(stats)
    }

    fn process_entity_update(&mut self, entity: EntityReference) -> Result<EntityUpdateResult> {
        let entity_key = self.generate_entity_key(&entity);

        match self.entity_index.get(&entity_key) {
            Some(existing_entity) => {
                // Entity exists, handle conflict
                match self.conflict_strategy {
                    ConflictStrategy::Merge => {
                        let merged_entity = self.merge_entities(existing_entity, &entity)?;
                        self.entity_index.insert(entity_key, merged_entity);
                        self.update_queue.push_back(GraphUpdate::UpdateEntity(entity));
                        Ok(EntityUpdateResult::ConflictResolved)
                    }
                    ConflictStrategy::Replace => {
                        self.entity_index.insert(entity_key, entity.clone());
                        self.update_queue.push_back(GraphUpdate::UpdateEntity(entity));
                        Ok(EntityUpdateResult::Updated)
                    }
                    ConflictStrategy::Skip => {
                        Ok(EntityUpdateResult::Skipped)
                    }
                }
            }
            None => {
                // New entity
                self.entity_index.insert(entity_key, entity.clone());
                self.update_queue.push_back(GraphUpdate::AddEntity(entity));
                Ok(EntityUpdateResult::Added)
            }
        }
    }

    fn process_concept_update(&mut self, concept: ConceptSummary) -> Result<ConceptUpdateResult> {
        let concept_key = self.generate_concept_key(&concept);

        match self.concept_index.get(&concept_key) {
            Some(existing_concept) => {
                // Concept exists, handle conflict
                match self.conflict_strategy {
                    ConflictStrategy::Merge => {
                        let merged_concept = self.merge_concepts(existing_concept, &concept)?;
                        self.concept_index.insert(concept_key, merged_concept);
                        self.update_queue.push_back(GraphUpdate::UpdateConcept(concept));
                        Ok(ConceptUpdateResult::ConflictResolved)
                    }
                    ConflictStrategy::Replace => {
                        self.concept_index.insert(concept_key, concept.clone());
                        self.update_queue.push_back(GraphUpdate::UpdateConcept(concept));
                        Ok(ConceptUpdateResult::Updated)
                    }
                    ConflictStrategy::Skip => {
                        Ok(ConceptUpdateResult::Skipped)
                    }
                }
            }
            None => {
                // New concept
                self.concept_index.insert(concept_key, concept.clone());
                self.update_queue.push_back(GraphUpdate::AddConcept(concept));
                Ok(ConceptUpdateResult::Added)
            }
        }
    }

    fn process_update_queue(&mut self, stats: &mut UpdateStats) -> Result<()> {
        while let Some(update) = self.update_queue.pop_front() {
            match update {
                GraphUpdate::AddEntity(_) | GraphUpdate::UpdateEntity(_) => {
                    // Entity updates already processed
                }
                GraphUpdate::AddConcept(_) | GraphUpdate::UpdateConcept(_) => {
                    // Concept updates already processed
                }
                GraphUpdate::DeleteEntity(entity_id) => {
                    if self.delete_entity(&entity_id)? {
                        stats.entities_deleted += 1;
                    }
                }
                GraphUpdate::DeleteConcept(concept_id) => {
                    if self.delete_concept(&concept_id)? {
                        stats.concepts_deleted += 1;
                    }
                }
            }
        }

        Ok(())
    }

    fn merge_entities(&self, existing: &EntityReference, new: &EntityReference) -> Result<EntityReference> {
        let mut merged = existing.clone();

        // Merge attributes
        for attr in &new.attributes {
            if !merged.attributes.contains(attr) {
                merged.attributes.push(attr.clone());
            }
        }

        // Merge mentions
        merged.mentions.extend(new.mentions.clone());

        // Update confidence (weighted average)
        let total_mentions = merged.mentions.len() as f32;
        let existing_weight = (existing.mentions.len() as f32) / total_mentions;
        let new_weight = (new.mentions.len() as f32) / total_mentions;
        merged.confidence = existing.confidence * existing_weight + new.confidence * new_weight;

        Ok(merged)
    }

    fn merge_concepts(&self, existing: &ConceptSummary, new: &ConceptSummary) -> Result<ConceptSummary> {
        let mut merged = existing.clone();

        // Merge related entities
        for entity in &new.related_entities {
            if !merged.related_entities.contains(entity) {
                merged.related_entities.push(entity.clone());
            }
        }

        // Combine summaries if different
        if existing.summary != new.summary {
            merged.summary = format!("{} Additionally, {}", existing.summary, new.summary);
        }

        // Update confidence (average)
        merged.confidence = (existing.confidence + new.confidence) / 2.0;

        Ok(merged)
    }

    fn delete_entity(&mut self, entity_id: &str) -> Result<bool> {
        // Find and remove entity from index
        let mut removed = false;
        self.entity_index.retain(|_, entity| {
            if entity.id == entity_id {
                removed = true;
                false
            } else {
                true
            }
        });

        Ok(removed)
    }

    fn delete_concept(&mut self, concept_id: &str) -> Result<bool> {
        // Find and remove concept from index
        let mut removed = false;
        self.concept_index.retain(|_, concept| {
            if concept.id == concept_id {
                removed = true;
                false
            } else {
                true
            }
        });

        Ok(removed)
    }

    fn generate_entity_key(&self, entity: &EntityReference) -> String {
        format!("{}:{}", entity.entity_type, entity.name.to_lowercase())
    }

    fn generate_concept_key(&self, concept: &ConceptSummary) -> String {
        concept.theme.to_lowercase().replace(' ', "_")
    }

    /// Get update statistics
    pub fn get_statistics(&self) -> IncrementalStatistics {
        IncrementalStatistics {
            total_entities: self.entity_index.len(),
            total_concepts: self.concept_index.len(),
            total_updates: self.update_history.len(),
            total_conflicts_resolved: self.update_history.iter()
                .map(|batch| batch.stats.conflicts_resolved)
                .sum(),
            average_tokens_per_update: if self.update_history.is_empty() {
                0.0
            } else {
                self.update_history.iter()
                    .map(|batch| batch.stats.tokens_used as f32)
                    .sum::<f32>() / self.update_history.len() as f32
            },
        }
    }

    /// Get all entities
    pub fn get_all_entities(&self) -> Vec<&EntityReference> {
        self.entity_index.values().collect()
    }

    /// Get all concepts
    pub fn get_all_concepts(&self) -> Vec<&ConceptSummary> {
        self.concept_index.values().collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.entity_index.clear();
        self.concept_index.clear();
        self.update_queue.clear();
        self.update_history.clear();
    }
}

// Result types for internal processing

#[derive(Debug)]
enum EntityUpdateResult {
    Added,
    Updated,
    ConflictResolved,
    Skipped,
}

#[derive(Debug)]
enum ConceptUpdateResult {
    Added,
    Updated,
    ConflictResolved,
    Skipped,
}

#[derive(Debug, Clone)]
struct UpdateBatch {
    #[allow(dead_code)] timestamp: std::time::SystemTime,
    stats: UpdateStats,
}

/// Statistics for incremental processing
#[derive(Debug, Clone)]
pub struct IncrementalStatistics {
    pub total_entities: usize,
    pub total_concepts: usize,
    pub total_updates: usize,
    pub total_conflicts_resolved: usize,
    pub average_tokens_per_update: f32,
}

impl IncrementalStatistics {
    pub fn print(&self) {
        println!("📊 Incremental Update Statistics:");
        println!("   Total entities: {}", self.total_entities);
        println!("   Total concepts: {}", self.total_concepts);
        println!("   Total updates: {}", self.total_updates);
        println!("   Conflicts resolved: {}", self.total_conflicts_resolved);
        println!("   Avg tokens/update: {:.1}", self.average_tokens_per_update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_updater() {
        let mut updater = IncrementalGraphUpdater::new(ConflictStrategy::Merge).unwrap();

        let entity = EntityReference {
            id: "test_entity".to_string(),
            name: "Test Entity".to_string(),
            entity_type: "person".to_string(),
            attributes: vec!["engineer".to_string()],
            mentions: vec![],
            confidence: 0.8,
        };

        let concept = ConceptSummary {
            id: "test_concept".to_string(),
            theme: "technology".to_string(),
            summary: "Technology-related concepts".to_string(),
            related_entities: vec!["Test Entity".to_string()],
            confidence: 0.7,
        };

        let extraction_result = ExtractionResult {
            entities: vec![entity],
            concepts: vec![concept],
            tokens_used: 100,
        };

        let stats = updater.apply_incremental_update(extraction_result).unwrap();

        assert_eq!(stats.entities_added, 1);
        assert_eq!(stats.concepts_added, 1);
        assert_eq!(stats.tokens_used, 100);

        let incremental_stats = updater.get_statistics();
        assert_eq!(incremental_stats.total_entities, 1);
        assert_eq!(incremental_stats.total_concepts, 1);
    }
}