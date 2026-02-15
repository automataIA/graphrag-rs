//! Graph indexer for entity extraction
//!
//! Provides entity and relationship extraction from text using pattern matching
//! and heuristic-based detection. Supports 23 relationship patterns and entity
//! type classification (person, organization, location).
//!
//! Future enhancements could include:
//! - Advanced NLP models (NER, dependency parsing)
//! - Coreference resolution
//! - Multilingual support

use crate::core::Result;

/// Extraction result containing entities and relationships
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// List of entities extracted from the text
    pub entities: Vec<ExtractedEntity>,
    /// List of relationships between entities extracted from the text
    pub relationships: Vec<ExtractedRelationship>,
}

/// Entity extracted from text
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    /// Unique identifier for the entity
    pub id: String,
    /// Name or label of the entity
    pub name: String,
    /// Type/category of the entity (e.g., "person", "organization", "location")
    pub entity_type: String,
    /// Confidence score for the extraction (0.0 to 1.0)
    pub confidence: f32,
}

/// Relationship extracted from text
#[derive(Debug, Clone)]
pub struct ExtractedRelationship {
    /// ID or name of the source entity
    pub source: String,
    /// ID or name of the target entity
    pub target: String,
    /// Type of relationship (e.g., "works_at", "located_in", "manages")
    pub relation_type: String,
    /// Confidence score for the relationship extraction (0.0 to 1.0)
    pub confidence: f32,
}

/// Graph indexer for extracting entities and relationships from text
pub struct GraphIndexer {
    /// List of entity types to recognize during extraction
    entity_types: Vec<String>,
    /// Maximum depth for relationship traversal (reserved for future implementation)
    #[allow(dead_code)]
    max_depth: usize,
}

impl GraphIndexer {
    /// Create a new graph indexer with specified entity types and depth
    pub fn new(entity_types: Vec<String>, max_depth: usize) -> Result<Self> {
        Ok(Self {
            entity_types,
            max_depth,
        })
    }

    /// Extract entities and relationships from text
    pub fn extract_from_text(&self, text: &str) -> Result<ExtractionResult> {
        // Simple stub implementation - extract basic patterns
        let mut entities = Vec::new();
        let mut entity_id = 0;

        // Extract capitalized words as potential entities
        let words: Vec<&str> = text.split_whitespace().collect();

        for window in words.windows(3) {
            let phrase = window.join(" ");

            // Look for capitalized phrases
            if window.iter().all(|w| w.chars().next().map_or(false, |c| c.is_uppercase())) {
                entities.push(ExtractedEntity {
                    id: format!("entity_{}", entity_id),
                    name: phrase.clone(),
                    entity_type: self.guess_entity_type(&phrase),
                    confidence: 0.6,
                });
                entity_id += 1;
            }
        }

        // Single capitalized words
        for word in words {
            if word.len() > 2 && word.chars().next().map_or(false, |c| c.is_uppercase()) {
                entities.push(ExtractedEntity {
                    id: format!("entity_{}", entity_id),
                    name: word.to_string(),
                    entity_type: self.guess_entity_type(word),
                    confidence: 0.5,
                });
                entity_id += 1;
            }
        }

        // Deduplicate entities by name
        entities.sort_by(|a, b| a.name.cmp(&b.name));
        entities.dedup_by(|a, b| a.name == b.name);

        // Extract relationships using pattern matching
        let relationships = self.extract_relationships(text, &entities);

        Ok(ExtractionResult {
            entities,
            relationships,
        })
    }

    /// Extract relationships between entities using pattern matching
    fn extract_relationships(&self, text: &str, entities: &[ExtractedEntity]) -> Vec<ExtractedRelationship> {
        let mut relationships = Vec::new();
        let text_lower = text.to_lowercase();

        // Common relationship patterns
        let patterns = [
            // Employment relationships
            ("works at", "works_at", 0.7),
            ("works for", "works_at", 0.7),
            ("employed by", "works_at", 0.7),
            ("employee of", "works_at", 0.7),
            ("works as", "works_as", 0.6),
            // Location relationships
            ("located in", "located_in", 0.8),
            ("based in", "located_in", 0.7),
            ("in", "located_in", 0.4),
            ("from", "from", 0.5),
            // Organizational relationships
            ("founded", "founded", 0.8),
            ("created", "created", 0.7),
            ("manages", "manages", 0.8),
            ("leads", "leads", 0.7),
            ("owns", "owns", 0.8),
            ("part of", "part_of", 0.7),
            ("subsidiary of", "subsidiary_of", 0.8),
            // Association relationships
            ("collaborates with", "collaborates_with", 0.7),
            ("partners with", "partners_with", 0.7),
            ("associated with", "associated_with", 0.6),
            ("related to", "related_to", 0.5),
            ("knows", "knows", 0.6),
        ];

        // Try to find relationships between pairs of entities
        for (i, entity1) in entities.iter().enumerate() {
            for entity2 in entities.iter().skip(i + 1) {
                // Check if both entities appear in text
                let e1_lower = entity1.name.to_lowercase();
                let e2_lower = entity2.name.to_lowercase();

                if !text_lower.contains(&e1_lower) || !text_lower.contains(&e2_lower) {
                    continue;
                }

                // Find positions of entities in text
                if let Some(pos1) = text_lower.find(&e1_lower) {
                    if let Some(pos2) = text_lower.find(&e2_lower) {
                        let (first, second, forward) = if pos1 < pos2 {
                            (entity1, entity2, true)
                        } else {
                            (entity2, entity1, false)
                        };

                        let first_pos = pos1.min(pos2);
                        let second_pos = pos1.max(pos2);

                        // Extract text between entities
                        let between_text = &text_lower[first_pos..second_pos];

                        // Check for relationship patterns
                        for (pattern, rel_type, base_confidence) in &patterns {
                            if between_text.contains(pattern) {
                                // Adjust confidence based on entity types
                                let mut confidence: f32 = *base_confidence;

                                // Higher confidence for type-appropriate relationships
                                match (*rel_type, first.entity_type.as_str(), second.entity_type.as_str()) {
                                    ("works_at", "person", "organization") => confidence += 0.2,
                                    ("located_in", _, "location") => confidence += 0.2,
                                    ("founded", "person", "organization") => confidence += 0.2,
                                    ("manages", "person", _) => confidence += 0.1,
                                    _ => {}
                                }

                                confidence = confidence.min(1.0);

                                if forward {
                                    relationships.push(ExtractedRelationship {
                                        source: first.name.clone(),
                                        target: second.name.clone(),
                                        relation_type: rel_type.to_string(),
                                        confidence,
                                    });
                                } else {
                                    // Some relationships are bidirectional or should be reversed
                                    let (final_source, final_target) = match *rel_type {
                                        "works_at" | "located_in" | "from" => (second.name.clone(), first.name.clone()),
                                        _ => (first.name.clone(), second.name.clone()),
                                    };
                                    relationships.push(ExtractedRelationship {
                                        source: final_source,
                                        target: final_target,
                                        relation_type: rel_type.to_string(),
                                        confidence,
                                    });
                                }
                                break; // Take first matching pattern
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate relationships
        relationships.sort_by(|a, b| {
            a.source.cmp(&b.source)
                .then(a.target.cmp(&b.target))
                .then(a.relation_type.cmp(&b.relation_type))
        });
        relationships.dedup_by(|a, b| {
            a.source == b.source && a.target == b.target && a.relation_type == b.relation_type
        });

        relationships
    }

    /// Guess entity type based on simple heuristics
    fn guess_entity_type(&self, text: &str) -> String {
        // Check if it's one of our known types
        for entity_type in &self.entity_types {
            if text.to_lowercase().contains(entity_type) {
                return entity_type.clone();
            }
        }

        // Simple heuristics
        let lower = text.to_lowercase();
        if lower.ends_with("company") || lower.ends_with("corp") || lower.ends_with("inc") {
            "organization".to_string()
        } else if lower.contains("city") || lower.contains("country") || lower.contains("state") {
            "location".to_string()
        } else if text.split_whitespace().count() == 1 && text.len() < 20 {
            "person".to_string()
        } else {
            "other".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_indexer_creation() {
        let entity_types = vec!["person".to_string(), "organization".to_string()];
        let indexer = GraphIndexer::new(entity_types, 3);
        assert!(indexer.is_ok());
    }

    #[test]
    fn test_basic_extraction() {
        let entity_types = vec!["person".to_string(), "organization".to_string()];
        let indexer = GraphIndexer::new(entity_types, 3).unwrap();

        let text = "John Smith works at Microsoft Corporation in Seattle.";
        let result = indexer.extract_from_text(text);

        assert!(result.is_ok());
        let extraction = result.unwrap();
        assert!(!extraction.entities.is_empty());
    }
}
