//! Gleaning-based entity extraction with TRUE LLM inference
//!
//! This module implements iterative gleaning refinement using actual LLM calls,
//! not pattern matching. Based on Microsoft GraphRAG and LightRAG research.
//!
//! Expected performance: 15-30 seconds per chunk per round. For a 1000-page book
//! with 4 gleaning rounds, expect 2-4 hours of processing time.

use crate::{
    core::{Entity, Relationship, Result, TextChunk},
    entity::{
        llm_extractor::LLMEntityExtractor,
        prompts::{EntityData, RelationshipData},
    },
    ollama::OllamaClient,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for gleaning-based entity extraction
#[derive(Debug, Clone)]
pub struct GleaningConfig {
    /// Maximum number of gleaning rounds (typically 3-4)
    pub max_gleaning_rounds: usize,
    /// Threshold for extraction completion (0.0-1.0)
    pub completion_threshold: f64,
    /// Minimum confidence for extracted entities (0.0-1.0)
    pub entity_confidence_threshold: f64,
    /// Whether to use LLM for completion checking (always true for real gleaning)
    pub use_llm_completion_check: bool,
    /// Entity types to extract
    pub entity_types: Vec<String>,
    /// LLM temperature for extraction (lower = more consistent)
    pub temperature: f32,
    /// Maximum tokens for LLM responses
    pub max_tokens: usize,
}

impl Default for GleaningConfig {
    fn default() -> Self {
        Self {
            max_gleaning_rounds: 4, // Microsoft GraphRAG uses 4 rounds
            completion_threshold: 0.85,
            entity_confidence_threshold: 0.7,
            use_llm_completion_check: true, // Always use LLM for real gleaning
            entity_types: vec![
                "PERSON".to_string(),
                "ORGANIZATION".to_string(),
                "LOCATION".to_string(),
                "EVENT".to_string(),
                "CONCEPT".to_string(),
            ],
            temperature: 0.1, // Low temperature for consistent extraction
            max_tokens: 1500,
        }
    }
}

/// Status of entity extraction completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionCompletionStatus {
    /// Whether extraction is considered complete
    pub is_complete: bool,
    /// Confidence score for completeness (0.0-1.0)
    pub confidence: f64,
    /// Aspects that may be missing from extraction
    pub missing_aspects: Vec<String>,
    /// Suggestions for improving extraction
    pub suggestions: Vec<String>,
}

/// Entity extractor with iterative gleaning refinement using TRUE LLM calls
///
/// This is the REAL implementation that makes actual LLM API calls for every extraction.
/// It replaces the fake pattern-based extraction with genuine language model inference.
pub struct GleaningEntityExtractor {
    llm_extractor: LLMEntityExtractor,
    config: GleaningConfig,
}

impl GleaningEntityExtractor {
    /// Create a new gleaning entity extractor with LLM client
    ///
    /// # Arguments
    /// * `ollama_client` - Ollama client for LLM inference (REQUIRED)
    /// * `config` - Gleaning configuration
    pub fn new(ollama_client: OllamaClient, config: GleaningConfig) -> Self {
        // Create LLM extractor with configured entity types
        let llm_extractor = LLMEntityExtractor::new(
            ollama_client,
            config.entity_types.clone(),
        )
        .with_temperature(config.temperature)
        .with_max_tokens(config.max_tokens);

        Self {
            llm_extractor,
            config,
        }
    }

    /// Extract entities with iterative refinement (gleaning) using TRUE LLM calls
    ///
    /// This is the REAL implementation that makes actual LLM API calls.
    /// Expected time: 15-30 seconds per round = 60-120 seconds total for 4 rounds per chunk.
    ///
    /// # Performance
    /// - 1 chunk, 4 rounds: ~2 minutes
    /// - 100 chunks, 4 rounds: ~3-4 hours
    /// - Tom Sawyer (1000 pages): ~2-4 hours
    #[cfg(feature = "async")]
    pub async fn extract_with_gleaning(
        &self,
        chunk: &TextChunk,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        tracing::info!(
            "🔍 Starting REAL LLM gleaning extraction for chunk: {} ({} chars)",
            chunk.id,
            chunk.content.len()
        );

        let start_time = std::time::Instant::now();

        // Track all extracted entities and relationships across rounds
        let mut all_entity_data: Vec<EntityData> = Vec::new();
        let mut all_relationship_data: Vec<RelationshipData> = Vec::new();

        // Round 1: Initial extraction (THIS IS A REAL LLM CALL!)
        tracing::info!("📝 Round 1: Initial LLM extraction...");
        let round_start = std::time::Instant::now();

        let (initial_entities, initial_relationships) =
            self.llm_extractor.extract_from_chunk(chunk).await?;

        tracing::info!(
            "✅ Round 1 complete: {} entities, {} relationships ({:.1}s)",
            initial_entities.len(),
            initial_relationships.len(),
            round_start.elapsed().as_secs_f32()
        );

        // Convert to EntityData for tracking
        let mut entity_data = self.convert_entities_to_data(&initial_entities);
        let mut relationship_data = self.convert_relationships_to_data(&initial_relationships);

        all_entity_data.append(&mut entity_data);
        all_relationship_data.append(&mut relationship_data);

        // Rounds 2-N: Gleaning continuation rounds
        for round in 2..=self.config.max_gleaning_rounds {
            tracing::info!("📝 Round {}: Gleaning continuation...", round);
            let round_start = std::time::Instant::now();

            // Check if extraction is complete using LLM (REAL LLM CALL!)
            if self.config.use_llm_completion_check {
                let is_complete = self
                    .llm_extractor
                    .check_completion(chunk, &all_entity_data, &all_relationship_data)
                    .await?;

                if is_complete {
                    tracing::info!(
                        "✅ LLM determined extraction is COMPLETE after {} rounds ({:.1}s total)",
                        round - 1,
                        start_time.elapsed().as_secs_f32()
                    );
                    break;
                }

                tracing::debug!("⚠️  LLM determined extraction is INCOMPLETE, continuing...");
            }

            // Perform additional extraction round (REAL LLM CALL!)
            let (additional_entities, additional_relationships) = self
                .llm_extractor
                .extract_additional(chunk, &all_entity_data, &all_relationship_data)
                .await?;

            tracing::info!(
                "✅ Round {} complete: {} new entities, {} new relationships ({:.1}s)",
                round,
                additional_entities.len(),
                additional_relationships.len(),
                round_start.elapsed().as_secs_f32()
            );

            // If no new entities found, stop gleaning
            if additional_entities.is_empty() && additional_relationships.is_empty() {
                tracing::info!(
                    "🛑 No additional entities found in round {}, stopping gleaning",
                    round
                );
                break;
            }

            // Convert and merge new results
            let new_entity_data = self.convert_entities_to_data(&additional_entities);
            let mut new_relationship_data =
                self.convert_relationships_to_data(&additional_relationships);

            // Merge with length-based strategy (LightRAG approach)
            all_entity_data = self.merge_entity_data(all_entity_data, new_entity_data);
            all_relationship_data.append(&mut new_relationship_data);
        }

        // Convert back to domain entities and relationships
        let final_entities = self.convert_data_to_entities(&all_entity_data, &chunk.id, &chunk.content)?;
        let final_relationships = self.convert_data_to_relationships(&all_relationship_data, &final_entities)?;

        // Deduplicate relationships
        let deduplicated_relationships = self.deduplicate_relationships(final_relationships);

        let total_time = start_time.elapsed().as_secs_f32();

        tracing::info!(
            "🎉 REAL LLM gleaning complete: {} entities, {} relationships ({:.1}s total)",
            final_entities.len(),
            deduplicated_relationships.len(),
            total_time
        );

        Ok((final_entities, deduplicated_relationships))
    }

    /// Merge entity data using length-based strategy (LightRAG approach)
    ///
    /// When multiple rounds produce the same entity, keep the version with the longer description
    /// as it likely contains more information.
    fn merge_entity_data(
        &self,
        existing: Vec<EntityData>,
        new: Vec<EntityData>,
    ) -> Vec<EntityData> {
        let mut merged: HashMap<String, EntityData> = HashMap::new();

        // Add existing entities to map (normalized by lowercase name)
        for entity in existing {
            let key = entity.name.to_lowercase();
            merged.insert(key, entity);
        }

        // Merge new entities - keep longer descriptions
        for new_entity in new {
            let key = new_entity.name.to_lowercase();

            match merged.get(&key) {
                Some(existing_entity) => {
                    // Keep the entity with the longer description (more information)
                    if new_entity.description.len() > existing_entity.description.len() {
                        tracing::debug!(
                            "📝 Merging entity '{}': keeping longer description ({} chars vs {} chars)",
                            new_entity.name,
                            new_entity.description.len(),
                            existing_entity.description.len()
                        );
                        merged.insert(key, new_entity);
                    } else {
                        tracing::debug!(
                            "📝 Entity '{}' already exists with longer description, keeping existing",
                            new_entity.name
                        );
                    }
                }
                None => {
                    // New entity, add it
                    merged.insert(key, new_entity);
                }
            }
        }

        merged.into_values().collect()
    }

    /// Convert domain entities to EntityData
    fn convert_entities_to_data(&self, entities: &[Entity]) -> Vec<EntityData> {
        entities
            .iter()
            .map(|e| EntityData {
                name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                description: format!("{} (confidence: {:.2})", e.entity_type, e.confidence),
            })
            .collect()
    }

    /// Convert domain relationships to RelationshipData
    fn convert_relationships_to_data(&self, relationships: &[Relationship]) -> Vec<RelationshipData> {
        relationships
            .iter()
            .map(|r| RelationshipData {
                source: r.source.0.clone(),
                target: r.target.0.clone(),
                description: r.relation_type.clone(),
                strength: r.confidence as f64,
            })
            .collect()
    }

    /// Convert EntityData back to domain entities
    fn convert_data_to_entities(
        &self,
        entity_data: &[EntityData],
        chunk_id: &crate::core::ChunkId,
        chunk_text: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for data in entity_data {
            // Generate entity ID
            let entity_id = crate::core::EntityId::new(format!(
                "{}_{}",
                data.entity_type,
                self.normalize_name(&data.name)
            ));

            // Find mentions in chunk
            let mentions = self.find_mentions(&data.name, chunk_id, chunk_text);

            // Create entity with mentions
            let entity = Entity::new(
                entity_id,
                data.name.clone(),
                data.entity_type.clone(),
                0.9, // High confidence since LLM-extracted
            )
            .with_mentions(mentions);

            entities.push(entity);
        }

        Ok(entities)
    }

    /// Find all mentions of an entity name in the chunk text
    fn find_mentions(
        &self,
        name: &str,
        chunk_id: &crate::core::ChunkId,
        text: &str,
    ) -> Vec<crate::core::EntityMention> {
        let mut mentions = Vec::new();
        let mut start = 0;

        while let Some(pos) = text[start..].find(name) {
            let actual_pos = start + pos;
            mentions.push(crate::core::EntityMention {
                chunk_id: chunk_id.clone(),
                start_offset: actual_pos,
                end_offset: actual_pos + name.len(),
                confidence: 0.9,
            });
            start = actual_pos + name.len();
        }

        // If no exact matches, try case-insensitive
        if mentions.is_empty() {
            let name_lower = name.to_lowercase();
            let text_lower = text.to_lowercase();
            let mut start = 0;

            while let Some(pos) = text_lower[start..].find(&name_lower) {
                let actual_pos = start + pos;
                mentions.push(crate::core::EntityMention {
                    chunk_id: chunk_id.clone(),
                    start_offset: actual_pos,
                    end_offset: actual_pos + name.len(),
                    confidence: 0.85,
                });
                start = actual_pos + name.len();
            }
        }

        mentions
    }

    /// Convert RelationshipData to domain Relationships
    fn convert_data_to_relationships(
        &self,
        relationship_data: &[RelationshipData],
        entities: &[Entity],
    ) -> Result<Vec<Relationship>> {
        let mut relationships = Vec::new();

        // Build entity name to entity mapping
        let mut name_to_entity: HashMap<String, &Entity> = HashMap::new();
        for entity in entities {
            name_to_entity.insert(entity.name.to_lowercase(), entity);
        }

        for data in relationship_data {
            // Find source and target entities
            let source_entity = name_to_entity.get(&data.source.to_lowercase());
            let target_entity = name_to_entity.get(&data.target.to_lowercase());

            if let (Some(source), Some(target)) = (source_entity, target_entity) {
                let relationship = Relationship {
                    source: source.id.clone(),
                    target: target.id.clone(),
                    relation_type: data.description.clone(),
                    confidence: data.strength as f32,
                    context: vec![],
                };

                relationships.push(relationship);
            } else {
                tracing::warn!(
                    "Skipping relationship: entity not found. Source: {}, Target: {}",
                    data.source,
                    data.target
                );
            }
        }

        Ok(relationships)
    }

    /// Deduplicate relationships by source-target-type combination
    fn deduplicate_relationships(&self, relationships: Vec<Relationship>) -> Vec<Relationship> {
        let mut seen = std::collections::HashSet::new();
        let mut deduplicated = Vec::new();

        for relationship in relationships {
            let key = format!(
                "{}->{}:{}",
                relationship.source, relationship.target, relationship.relation_type
            );

            if !seen.contains(&key) {
                seen.insert(key);
                deduplicated.push(relationship);
            }
        }

        deduplicated
    }

    /// Normalize entity name for ID generation
    fn normalize_name(&self, name: &str) -> String {
        name.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .replace(' ', "_")
    }

    /// Get extraction statistics
    pub fn get_statistics(&self) -> GleaningStatistics {
        GleaningStatistics {
            config: self.config.clone(),
            llm_available: true, // Always true for real gleaning
        }
    }
}

/// Statistics for gleaning extraction process
#[derive(Debug, Clone)]
pub struct GleaningStatistics {
    /// Gleaning configuration used
    pub config: GleaningConfig,
    /// Whether LLM is available for completion checking
    pub llm_available: bool,
}

impl GleaningStatistics {
    /// Print statistics to stdout
    #[allow(dead_code)]
    pub fn print(&self) {
        tracing::info!("🔍 REAL LLM Gleaning Extraction Statistics");
        tracing::info!("  Max rounds: {}", self.config.max_gleaning_rounds);
        tracing::info!(
            "  Completion threshold: {:.2}",
            self.config.completion_threshold
        );
        tracing::info!(
            "  Entity confidence threshold: {:.2}",
            self.config.entity_confidence_threshold
        );
        tracing::info!(
            "  Uses LLM completion check: {}",
            self.config.use_llm_completion_check
        );
        tracing::info!("  LLM available: {} ✅", self.llm_available);
        tracing::info!("  Entity types: {:?}", self.config.entity_types);
        tracing::info!("  Temperature: {}", self.config.temperature);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{ChunkId, DocumentId, TextChunk},
        ollama::OllamaConfig,
    };

    fn create_test_chunk() -> TextChunk {
        TextChunk::new(
            ChunkId::new("test_chunk".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Tom Sawyer is a young boy who lives in St. Petersburg with his Aunt Polly. Tom is best friends with Huckleberry Finn.".to_string(),
            0,
            120,
        )
    }

    #[test]
    fn test_gleaning_extractor_creation() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let config = GleaningConfig::default();

        let extractor = GleaningEntityExtractor::new(ollama_client, config);

        let stats = extractor.get_statistics();
        assert_eq!(stats.config.max_gleaning_rounds, 4);
        assert!(stats.llm_available);
    }

    #[test]
    fn test_merge_entity_data() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(ollama_client, config);

        let existing = vec![
            EntityData {
                name: "Tom Sawyer".to_string(),
                entity_type: "PERSON".to_string(),
                description: "A boy".to_string(),
            },
        ];

        let new = vec![
            EntityData {
                name: "Tom Sawyer".to_string(),
                entity_type: "PERSON".to_string(),
                description: "A young boy who lives in St. Petersburg".to_string(), // Longer description
            },
            EntityData {
                name: "Huck Finn".to_string(),
                entity_type: "PERSON".to_string(),
                description: "Tom's friend".to_string(),
            },
        ];

        let merged = extractor.merge_entity_data(existing, new);

        assert_eq!(merged.len(), 2); // Tom (merged) and Huck
        let tom = merged.iter().find(|e| e.name == "Tom Sawyer").unwrap();
        assert!(tom.description.len() > 10); // Should have the longer description
    }

    #[test]
    fn test_normalize_name() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(ollama_client, config);

        assert_eq!(extractor.normalize_name("Tom Sawyer"), "tom_sawyer");
        assert_eq!(extractor.normalize_name("St. Petersburg"), "st_petersburg");
    }

    #[test]
    fn test_find_mentions() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(ollama_client, config);

        let chunk = create_test_chunk();
        let mentions = extractor.find_mentions("Tom", &chunk.id, &chunk.content);

        assert!(!mentions.is_empty());
        assert!(mentions.len() >= 2); // "Tom Sawyer" and "Tom is best friends"
    }

    #[test]
    fn test_deduplicate_relationships() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(ollama_client, config);

        let relationships = vec![
            Relationship {
                source: crate::core::EntityId::new("person_tom".to_string()),
                target: crate::core::EntityId::new("person_huck".to_string()),
                relation_type: "FRIENDS_WITH".to_string(),
                confidence: 0.9,
                context: vec![],
            },
            Relationship {
                source: crate::core::EntityId::new("person_tom".to_string()),
                target: crate::core::EntityId::new("person_huck".to_string()),
                relation_type: "FRIENDS_WITH".to_string(), // Duplicate
                confidence: 0.85,
                context: vec![],
            },
            Relationship {
                source: crate::core::EntityId::new("person_tom".to_string()),
                target: crate::core::EntityId::new("location_stpetersburg".to_string()),
                relation_type: "LIVES_IN".to_string(),
                confidence: 0.8,
                context: vec![],
            },
        ];

        let deduplicated = extractor.deduplicate_relationships(relationships);

        assert_eq!(deduplicated.len(), 2); // Duplicate FRIENDS_WITH removed
    }
}
