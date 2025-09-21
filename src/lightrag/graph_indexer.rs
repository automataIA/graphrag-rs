//! Graph-based indexing for LightRAG dual-level system

use std::collections::HashMap;
use crate::core::Result;
use crate::lightrag::kv_store::EntityMention;

/// Entity reference for low-level indexing
#[derive(Debug, Clone)]
pub struct EntityReference {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub attributes: Vec<String>,
    pub mentions: Vec<EntityMention>,
    pub confidence: f32,
}

/// Concept summary for high-level indexing
#[derive(Debug, Clone)]
pub struct ConceptSummary {
    pub id: String,
    pub theme: String,
    pub summary: String,
    pub related_entities: Vec<String>,
    pub confidence: f32,
}

/// Graph indexer for extracting entities and building dual-level indexes
pub struct GraphIndexer {
    entity_types: Vec<String>,
    max_gleaning_iterations: usize,
    #[allow(dead_code)] confidence_threshold: f32,
}

impl GraphIndexer {
    pub fn new(entity_types: Vec<String>, max_gleaning_iterations: usize) -> Result<Self> {
        Ok(Self {
            entity_types,
            max_gleaning_iterations,
            confidence_threshold: 0.5,
        })
    }

    /// Extract entities and concepts from text using gleaning loops
    pub fn extract_from_text(&self, text: &str) -> Result<ExtractionResult> {
        let mut entities = Vec::new();
        let mut concepts = Vec::new();
        let mut total_tokens = 0;

        // Phase 1: Multi-pass entity extraction (gleaning loops)
        let entity_results = self.extract_entities_with_gleaning(text)?;
        entities.extend(entity_results.entities);
        total_tokens += entity_results.tokens_used;

        // Phase 2: Concept and theme extraction
        let concept_results = self.extract_concepts(text, &entities)?;
        concepts.extend(concept_results.concepts);
        total_tokens += concept_results.tokens_used;

        // Phase 3: Entity type categorization
        self.categorize_entities(&mut entities)?;

        // Phase 4: Cross-document deduplication (placeholder)
        self.deduplicate_entities(&mut entities)?;

        Ok(ExtractionResult {
            entities: self.convert_to_entity_references(entities),
            concepts,
            tokens_used: total_tokens,
        })
    }

    fn extract_entities_with_gleaning(&self, text: &str) -> Result<EntityExtractionResult> {
        let mut all_entities = Vec::new();
        let mut context = text.to_string();
        let mut total_tokens = 0;

        for iteration in 0..self.max_gleaning_iterations {
            println!("🔍 Gleaning iteration {}/{}", iteration + 1, self.max_gleaning_iterations);

            let extraction_result = self.extract_single_pass(&context)?;
            total_tokens += extraction_result.tokens_used;

            if extraction_result.entities.is_empty() {
                println!("   No new entities found, stopping early");
                break;
            }

            println!("   Found {} new entities", extraction_result.entities.len());
            all_entities.extend(extraction_result.entities);

            // Update context for next iteration
            context = self.update_context_for_next_iteration(&context, &all_entities);
        }

        // Deduplicate entities found across iterations
        let deduplicated_entities = self.deduplicate_within_extraction(all_entities)?;

        Ok(EntityExtractionResult {
            entities: deduplicated_entities,
            tokens_used: total_tokens,
        })
    }

    fn extract_single_pass(&self, text: &str) -> Result<EntityExtractionResult> {
        // This is a simplified extraction for demo purposes
        // In a real implementation, this would use an LLM or NLP pipeline

        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let tokens_used = words.len(); // Simplified token counting

        // Simple pattern-based entity extraction
        for (i, word) in words.iter().enumerate() {
            // Detect potential person names (capitalized words)
            if word.chars().next().is_some_and(|c| c.is_uppercase()) && word.len() > 2 {
                if let Some(next_word) = words.get(i + 1) {
                    if next_word.chars().next().is_some_and(|c| c.is_uppercase()) {
                        // Potential person name
                        let full_name = format!("{word} {next_word}");
                        let entity = ExtractedEntity {
                            name: full_name,
                            entity_type: "person".to_string(),
                            attributes: vec![],
                            confidence: 0.7,
                            context: self.get_context_around_word(text, i, 5),
                            position: i,
                        };
                        entities.push(entity);
                    }
                }
            }

            // Detect organizations (words ending with Inc, Corp, LLC, etc.)
            if word.ends_with("Inc.") || word.ends_with("Corp.") || word.ends_with("LLC") {
                let entity = ExtractedEntity {
                    name: word.to_string(),
                    entity_type: "organization".to_string(),
                    attributes: vec!["company".to_string()],
                    confidence: 0.8,
                    context: self.get_context_around_word(text, i, 5),
                    position: i,
                };
                entities.push(entity);
            }
        }

        Ok(EntityExtractionResult {
            entities,
            tokens_used,
        })
    }

    fn extract_concepts(&self, text: &str, entities: &[ExtractedEntity]) -> Result<ConceptExtractionResult> {
        let mut concepts = Vec::new();
        let tokens_used = text.split_whitespace().count(); // Simplified

        // Extract themes based on entity clustering and text analysis
        let themes = self.identify_themes(text, entities)?;

        for (i, theme) in themes.iter().enumerate() {
            let related_entities: Vec<String> = entities
                .iter()
                .filter(|e| e.context.to_lowercase().contains(&theme.to_lowercase()))
                .map(|e| e.name.clone())
                .collect();

            if !related_entities.is_empty() {
                let concept = ConceptSummary {
                    id: format!("concept_{i}"),
                    theme: theme.clone(),
                    summary: format!("Theme '{}' involves entities: {}", theme, related_entities.join(", ")),
                    related_entities,
                    confidence: 0.6,
                };
                concepts.push(concept);
            }
        }

        Ok(ConceptExtractionResult {
            concepts,
            tokens_used,
        })
    }

    fn identify_themes(&self, text: &str, entities: &[ExtractedEntity]) -> Result<Vec<String>> {
        // Simplified theme identification
        let mut themes = Vec::new();

        // Look for common business/organizational themes
        if text.to_lowercase().contains("meeting") || text.to_lowercase().contains("conference") {
            themes.push("business_meeting".to_string());
        }

        if text.to_lowercase().contains("project") || text.to_lowercase().contains("development") {
            themes.push("project_development".to_string());
        }

        if text.to_lowercase().contains("research") || text.to_lowercase().contains("study") {
            themes.push("research_activity".to_string());
        }

        // Entity-based themes
        let person_count = entities.iter().filter(|e| e.entity_type == "person").count();
        let org_count = entities.iter().filter(|e| e.entity_type == "organization").count();

        if person_count >= 2 && org_count >= 1 {
            themes.push("organizational_collaboration".to_string());
        }

        Ok(themes)
    }

    fn categorize_entities(&self, entities: &mut [ExtractedEntity]) -> Result<()> {
        for entity in entities.iter_mut() {
            // Refine entity type categorization
            if !self.entity_types.contains(&entity.entity_type) {
                // Map to closest supported type or mark as "concept"
                entity.entity_type = match entity.entity_type.as_str() {
                    "company" | "corporation" => "organization".to_string(),
                    "individual" => "person".to_string(),
                    _ => "concept".to_string(),
                };
            }
        }
        Ok(())
    }

    fn deduplicate_entities(&self, entities: &mut Vec<ExtractedEntity>) -> Result<()> {
        // Simple deduplication based on name similarity
        let mut deduplicated = Vec::new();
        let mut seen_names = std::collections::HashSet::new();

        for entity in entities.drain(..) {
            let normalized_name = entity.name.to_lowercase().trim().to_string();

            if !seen_names.contains(&normalized_name) {
                seen_names.insert(normalized_name);
                deduplicated.push(entity);
            }
        }

        *entities = deduplicated;
        Ok(())
    }

    fn deduplicate_within_extraction(&self, entities: Vec<ExtractedEntity>) -> Result<Vec<ExtractedEntity>> {
        let mut deduplicated = Vec::new();
        let mut entity_map: HashMap<String, ExtractedEntity> = HashMap::new();

        for entity in entities {
            let key = format!("{}:{}", entity.entity_type, entity.name.to_lowercase());

            match entity_map.get_mut(&key) {
                Some(existing) => {
                    // Merge with existing entity
                    existing.confidence = (existing.confidence + entity.confidence) / 2.0;
                    existing.attributes.extend(entity.attributes);
                    existing.attributes.sort();
                    existing.attributes.dedup();
                }
                None => {
                    entity_map.insert(key, entity);
                }
            }
        }

        deduplicated.extend(entity_map.into_values());
        Ok(deduplicated)
    }

    fn convert_to_entity_references(&self, entities: Vec<ExtractedEntity>) -> Vec<EntityReference> {
        entities
            .into_iter()
            .enumerate()
            .map(|(i, entity)| {
                let mention = EntityMention {
                    text: entity.name.clone(),
                    context: entity.context,
                    position: entity.position,
                };

                EntityReference {
                    id: format!("entity_{i}"),
                    name: entity.name,
                    entity_type: entity.entity_type,
                    attributes: entity.attributes,
                    mentions: vec![mention],
                    confidence: entity.confidence,
                }
            })
            .collect()
    }

    fn update_context_for_next_iteration(&self, _original_context: &str, _entities: &[ExtractedEntity]) -> String {
        // In a real implementation, this would modify the context to help the LLM
        // find additional entities by providing previously found entities as context
        // For now, return the original context
        _original_context.to_string()
    }

    fn get_context_around_word(&self, text: &str, word_index: usize, window: usize) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let start = word_index.saturating_sub(window);
        let end = std::cmp::min(word_index + window + 1, words.len());

        words[start..end].join(" ")
    }
}

// Internal types for extraction

#[derive(Debug, Clone)]
struct ExtractedEntity {
    name: String,
    entity_type: String,
    attributes: Vec<String>,
    confidence: f32,
    context: String,
    position: usize,
}

#[derive(Debug)]
struct EntityExtractionResult {
    entities: Vec<ExtractedEntity>,
    tokens_used: usize,
}

#[derive(Debug)]
struct ConceptExtractionResult {
    concepts: Vec<ConceptSummary>,
    tokens_used: usize,
}

// Public result type
#[derive(Debug)]
pub struct ExtractionResult {
    pub entities: Vec<EntityReference>,
    pub concepts: Vec<ConceptSummary>,
    pub tokens_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_indexer() {
        let entity_types = vec!["person".to_string(), "organization".to_string()];
        let indexer = GraphIndexer::new(entity_types, 2).unwrap();

        let text = "John Doe works at Microsoft Corp. He met with Jane Smith from Google Inc.";
        let result = indexer.extract_from_text(text).unwrap();

        assert!(!result.entities.is_empty());
        assert!(result.tokens_used > 0);
    }
}