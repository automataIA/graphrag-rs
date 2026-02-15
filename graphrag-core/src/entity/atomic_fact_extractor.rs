//! ATOM Atomic Fact Extraction
//!
//! This module implements atomic fact extraction following the ATOM methodology
//! (itext2kg - https://github.com/AuvaLab/itext2kg)
//!
//! ATOM extracts self-contained facts as 5-tuples:
//! (Subject, Predicate, Object, TemporalMarker, Confidence)
//!
//! Benefits:
//! - More granular than entity-relationship pairs
//! - Better temporal grounding
//! - Easier to validate and verify
//! - Natural fit for knowledge graphs

use crate::{
    core::{Entity, EntityId, Relationship, Result, TextChunk, GraphRAGError},
    ollama::OllamaClient,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Atomic fact extracted from text
///
/// Represents a single, self-contained factual statement.
/// Each fact should be verifiable and stand alone without context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicFact {
    /// Subject of the fact (entity performing action or being described)
    pub subject: String,
    /// Predicate describing the relationship or property
    pub predicate: String,
    /// Object of the fact (entity being acted upon or value)
    pub object: String,
    /// Optional temporal marker (e.g., "in 1876", "during summer", "380 BC")
    pub temporal_marker: Option<String>,
    /// Confidence score for this fact (0.0-1.0)
    pub confidence: f32,
}

impl AtomicFact {
    /// Check if this fact has temporal information
    pub fn is_temporal(&self) -> bool {
        self.temporal_marker.is_some()
    }

    /// Extract approximate Unix timestamp from temporal marker if possible
    ///
    /// This is a best-effort extraction and may not work for all formats.
    /// Returns None if temporal marker is missing or cannot be parsed.
    pub fn extract_timestamp(&self) -> Option<i64> {
        let marker = self.temporal_marker.as_ref()?;

        // Try to extract year from common formats
        // "in 1876", "during 1876", "1876", "380 BC", etc.

        // Check for BC/BCE dates
        if marker.contains("BC") || marker.contains("BCE") {
            // Extract the number before BC/BCE
            let num_str: String = marker.chars()
                .filter(|c| c.is_ascii_digit())
                .collect();

            if let Ok(year) = num_str.parse::<i64>() {
                // Negative for BC years
                // Approximate: 365.25 days per year * 24 hours * 3600 seconds
                return Some(-year * 365 * 24 * 3600);
            }
        }

        // Check for AD dates (positive years)
        let num_str: String = marker.chars()
            .filter(|c| c.is_ascii_digit())
            .collect();

        if let Ok(year) = num_str.parse::<i64>() {
            if year > 1000 && year < 3000 {
                // Approximate Unix timestamp for year
                // Unix epoch is 1970, so subtract that
                return Some((year - 1970) * 365 * 24 * 3600);
            }
        }

        None
    }
}

/// Extractor for atomic facts from text
///
/// Uses LLM to decompose text into self-contained factual statements.
pub struct AtomicFactExtractor {
    /// Ollama client for LLM-based extraction
    ollama_client: OllamaClient,
    /// Maximum tokens per fact (default: 400)
    max_fact_tokens: usize,
}

impl AtomicFactExtractor {
    /// Create a new atomic fact extractor
    ///
    /// # Arguments
    ///
    /// * `ollama_client` - Ollama client for LLM calls
    pub fn new(ollama_client: OllamaClient) -> Self {
        Self {
            ollama_client,
            max_fact_tokens: 400,
        }
    }

    /// Set maximum tokens per fact
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_fact_tokens = max_tokens;
        self
    }

    /// Extract atomic facts from a text chunk
    ///
    /// # Arguments
    ///
    /// * `chunk` - Text chunk to extract facts from
    ///
    /// # Returns
    ///
    /// Vector of atomic facts extracted from the text
    #[cfg(feature = "async")]
    pub async fn extract_atomic_facts(&self, chunk: &TextChunk) -> Result<Vec<AtomicFact>> {
        let prompt = format!(
            r#"Extract atomic facts from the following text. Each fact should be:
- Self-contained and verifiable (< {} tokens)
- In the format: (Subject, Predicate, Object, TemporalMarker, Confidence)
- TemporalMarker should capture time expressions like "in 1876", "during summer", "380 BC" (or null if none)
- Confidence should be 0.0-1.0

Respond ONLY with valid JSON array:
[
  {{
    "subject": "entity or concept",
    "predicate": "relationship or property",
    "object": "entity, value, or concept",
    "temporal_marker": "time expression or null",
    "confidence": 0.0-1.0
  }}
]

Text: "{}"

JSON:"#,
            self.max_fact_tokens, chunk.content
        );

        #[cfg(feature = "tracing")]
        tracing::debug!(
            chunk_id = %chunk.id,
            "Extracting atomic facts from chunk"
        );

        match self.ollama_client.generate(&prompt).await {
            Ok(response) => {
                // Extract JSON from response
                let json_str = response.trim();
                let json_str = if let Some(start) = json_str.find('[') {
                    if let Some(end) = json_str.rfind(']') {
                        &json_str[start..=end]
                    } else {
                        json_str
                    }
                } else {
                    json_str
                };

                #[derive(Deserialize)]
                struct AtomicFactJson {
                    subject: String,
                    predicate: String,
                    object: String,
                    temporal_marker: Option<String>,
                    confidence: f32,
                }

                match serde_json::from_str::<Vec<AtomicFactJson>>(json_str) {
                    Ok(facts_json) => {
                        let facts: Vec<AtomicFact> = facts_json
                            .into_iter()
                            .map(|f| AtomicFact {
                                subject: f.subject,
                                predicate: f.predicate,
                                object: f.object,
                                temporal_marker: f.temporal_marker.filter(|s| !s.is_empty() && s != "null"),
                                confidence: f.confidence.clamp(0.0, 1.0),
                            })
                            .collect();

                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            chunk_id = %chunk.id,
                            fact_count = facts.len(),
                            "Extracted atomic facts"
                        );

                        Ok(facts)
                    }
                    Err(e) => {
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            chunk_id = %chunk.id,
                            error = %e,
                            response = %json_str,
                            "Failed to parse atomic facts JSON"
                        );

                        // Return empty vector on parse failure
                        Ok(Vec::new())
                    }
                }
            }
            Err(e) => {
                #[cfg(feature = "tracing")]
                tracing::error!(
                    chunk_id = %chunk.id,
                    error = %e,
                    "Atomic fact extraction failed"
                );

                Err(GraphRAGError::EntityExtraction {
                    message: format!("Atomic fact extraction failed: {}", e),
                })
            }
        }
    }

    /// Convert atomic facts to graph elements (entities and relationships)
    ///
    /// # Arguments
    ///
    /// * `facts` - Vector of atomic facts to convert
    /// * `chunk_id` - ID of the source chunk for context
    ///
    /// # Returns
    ///
    /// Tuple of (entities, relationships) extracted from facts
    pub fn atomics_to_graph_elements(
        &self,
        facts: Vec<AtomicFact>,
        chunk_id: &crate::core::ChunkId,
    ) -> (Vec<Entity>, Vec<Relationship>) {
        let mut entities: HashMap<String, Entity> = HashMap::new();
        let mut relationships = Vec::new();

        for fact in facts {
            // Create or update subject entity
            let subject_id = EntityId::new(Self::normalize_entity_name(&fact.subject));
            entities.entry(subject_id.0.clone()).or_insert_with(|| {
                let mut entity = Entity::new(
                    subject_id.clone(),
                    fact.subject.clone(),
                    Self::infer_entity_type(&fact.subject),
                    fact.confidence,
                );

                // Add temporal information if available
                if let Some(timestamp) = fact.extract_timestamp() {
                    entity.first_mentioned = Some(timestamp);
                    entity.last_mentioned = Some(timestamp);
                }

                entity
            });

            // Create or update object entity
            let object_id = EntityId::new(Self::normalize_entity_name(&fact.object));
            entities.entry(object_id.0.clone()).or_insert_with(|| {
                let mut entity = Entity::new(
                    object_id.clone(),
                    fact.object.clone(),
                    Self::infer_entity_type(&fact.object),
                    fact.confidence,
                );

                // Add temporal information if available
                if let Some(timestamp) = fact.extract_timestamp() {
                    entity.first_mentioned = Some(timestamp);
                    entity.last_mentioned = Some(timestamp);
                }

                entity
            });

            // Create relationship
            let mut relationship = Relationship::new(
                subject_id,
                object_id,
                fact.predicate.to_uppercase(),
                fact.confidence,
            )
            .with_context(vec![chunk_id.clone()]);

            // Add temporal information if available
            if let Some(timestamp) = fact.extract_timestamp() {
                relationship.temporal_range = Some(
                    crate::graph::temporal::TemporalRange::new(timestamp, timestamp)
                );

                // Infer temporal relationship type based on predicate
                if fact.predicate.to_lowercase().contains("caused")
                    || fact.predicate.to_lowercase().contains("led to")
                {
                    relationship.temporal_type = Some(
                        crate::graph::temporal::TemporalRelationType::Caused
                    );
                    relationship.causal_strength = Some(fact.confidence);
                } else if fact.predicate.to_lowercase().contains("enabled")
                    || fact.predicate.to_lowercase().contains("allowed")
                {
                    relationship.temporal_type = Some(
                        crate::graph::temporal::TemporalRelationType::Enabled
                    );
                    relationship.causal_strength = Some(fact.confidence * 0.6);
                }
            }

            relationships.push(relationship);
        }

        (entities.into_values().collect(), relationships)
    }

    /// Normalize entity name for consistent ID generation
    fn normalize_entity_name(name: &str) -> String {
        name.trim()
            .to_lowercase()
            .replace(' ', "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    }

    /// Infer entity type from name (simple heuristic)
    fn infer_entity_type(name: &str) -> String {
        let lower = name.to_lowercase();

        // Check for proper nouns (capitalized)
        if name.chars().next().map_or(false, |c| c.is_uppercase()) {
            if lower.ends_with("ia") || lower.ends_with("land") || lower.ends_with("istan") {
                return "LOCATION".to_string();
            }
            return "PERSON".to_string();
        }

        // Check for numbers/dates
        if name.chars().any(|c| c.is_ascii_digit()) {
            return "DATE".to_string();
        }

        // Default to concept
        "CONCEPT".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_fact_creation() {
        let fact = AtomicFact {
            subject: "Socrates".to_string(),
            predicate: "taught".to_string(),
            object: "Plato".to_string(),
            temporal_marker: Some("in 380 BC".to_string()),
            confidence: 0.9,
        };

        assert_eq!(fact.subject, "Socrates");
        assert!(fact.is_temporal());
    }

    #[test]
    fn test_timestamp_extraction_bc() {
        let fact = AtomicFact {
            subject: "Event".to_string(),
            predicate: "occurred".to_string(),
            object: "Athens".to_string(),
            temporal_marker: Some("380 BC".to_string()),
            confidence: 0.9,
        };

        let timestamp = fact.extract_timestamp();
        assert!(timestamp.is_some());
        assert!(timestamp.unwrap() < 0); // BC should be negative
    }

    #[test]
    fn test_timestamp_extraction_ad() {
        let fact = AtomicFact {
            subject: "Event".to_string(),
            predicate: "occurred".to_string(),
            object: "Rome".to_string(),
            temporal_marker: Some("in 1876".to_string()),
            confidence: 0.9,
        };

        let timestamp = fact.extract_timestamp();
        assert!(timestamp.is_some());
    }

    #[test]
    fn test_normalize_entity_name() {
        assert_eq!(
            AtomicFactExtractor::normalize_entity_name("Socrates the Philosopher"),
            "socrates_the_philosopher"
        );
        assert_eq!(
            AtomicFactExtractor::normalize_entity_name("New York"),
            "new_york"
        );
    }

    #[test]
    fn test_infer_entity_type() {
        assert_eq!(
            AtomicFactExtractor::infer_entity_type("Socrates"),
            "PERSON"
        );
        assert_eq!(
            AtomicFactExtractor::infer_entity_type("Athens"),
            "PERSON" // Would be LOCATION if we had more sophisticated logic
        );
        assert_eq!(
            AtomicFactExtractor::infer_entity_type("love"),
            "CONCEPT"
        );
        assert_eq!(
            AtomicFactExtractor::infer_entity_type("1876"),
            "DATE"
        );
    }
}
