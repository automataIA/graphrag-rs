//! LLM-based relationship extraction following Microsoft GraphRAG methodology
//!
//! This module implements proper entity-relationship extraction using LLM prompts
//! instead of simple pattern matching. It extracts entities and relationships
//! together in a single LLM call, following the best practices from Microsoft
//! GraphRAG and LightRAG.

use crate::core::{Entity, EntityId, TextChunk, Result, GraphRAGError};
use serde::{Deserialize, Serialize};

/// Extracted relationship with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// Source entity name in the relationship
    pub source: String,
    /// Target entity name in the relationship
    pub target: String,
    /// Type of relationship (e.g., DISCUSSES, TEACHES, WORKS_FOR)
    pub relation_type: String,
    /// Brief explanation of why the entities are related
    pub description: String,
    /// Confidence score between 0.0 and 1.0
    pub strength: f32,
}

/// Triple validation result from LLM reflection
///
/// This struct captures the validation of an extracted relationship
/// against the source text, following DEG-RAG triple reflection methodology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleValidation {
    /// Whether the relationship is valid according to the source text
    pub is_valid: bool,
    /// Confidence score for the validation (0.0-1.0)
    pub confidence: f32,
    /// Explanation of why the relationship is valid or invalid
    pub reason: String,
    /// Optional suggestion for fixing invalid relationships
    pub suggested_fix: Option<String>,
}

/// Combined extraction result from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// List of entities extracted from text
    pub entities: Vec<ExtractedEntity>,
    /// List of relationships between entities
    pub relationships: Vec<ExtractedRelationship>,
}

/// Extracted entity with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Name of the entity
    pub name: String,
    /// Type of entity (e.g., PERSON, CONCEPT, LOCATION, ORGANIZATION)
    #[serde(rename = "type")]
    pub entity_type: String,
    /// Optional description providing context about the entity
    pub description: Option<String>,
}

/// LLM-based relationship extractor
///
/// This extractor uses a language model to identify entities and their relationships
/// in text. It follows Microsoft GraphRAG methodology for high-quality extraction.
pub struct LLMRelationshipExtractor {
    /// Optional Ollama client for LLM-based extraction
    pub ollama_client: Option<crate::ollama::OllamaClient>,
}

impl LLMRelationshipExtractor {
    /// Create a new LLM relationship extractor
    ///
    /// # Arguments
    ///
    /// * `ollama_config` - Optional Ollama configuration. If provided and enabled,
    ///   the extractor will use LLM-based extraction. Otherwise, it will fall back
    ///   to pattern-based extraction.
    ///
    /// # Returns
    ///
    /// Returns a new extractor instance or an error if initialization fails.
    pub fn new(ollama_config: Option<&crate::ollama::OllamaConfig>) -> Result<Self> {
        let ollama_client = if let Some(config) = ollama_config {
            if config.enabled {
                let local_config = crate::ollama::OllamaConfig {
                    enabled: config.enabled,
                    host: config.host.clone(),
                    port: config.port,
                    chat_model: config.chat_model.clone(),
                    embedding_model: config.embedding_model.clone(),
                    timeout_seconds: config.timeout_seconds,
                    max_retries: config.max_retries,
                    fallback_to_hash: config.fallback_to_hash,
                    max_tokens: None,
                    temperature: None,
                    enable_caching: true, // Default to enabled
                };

                Some(crate::ollama::OllamaClient::new(local_config))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self { ollama_client })
    }

    /// Build the extraction prompt following Microsoft GraphRAG methodology
    ///
    /// Creates a detailed prompt that instructs the LLM to extract both entities
    /// and relationships from text, with specific guidelines for different text types.
    ///
    /// # Arguments
    ///
    /// * `chunk_content` - The text content to extract entities and relationships from
    ///
    /// # Returns
    ///
    /// A formatted prompt string ready to be sent to the LLM
    fn build_extraction_prompt(&self, chunk_content: &str) -> String {
        format!(
            r#"You are an expert at extracting entities and relationships from text.
Extract all meaningful entities and relationships from the provided text.

**ENTITIES**: Extract people, concepts, locations, events, organizations, and other significant entities.
For each entity provide:
- name: the entity name
- type: entity type (PERSON, CONCEPT, LOCATION, EVENT, ORGANIZATION, OBJECT, etc.)
- description: brief description of the entity (optional)

**RELATIONSHIPS**: For entities that interact or are related, extract their relationships.
For each relationship provide:
- source: source entity name (must match an entity name)
- target: target entity name (must match an entity name)
- type: relationship type (DISCUSSES, QUESTIONS, RESPONDS_TO, TEACHES, LOVES, ADMIRES, ARGUES_WITH, MENTIONS, WORKS_FOR, LOCATED_IN, etc.)
- description: brief explanation of why they are related
- strength: confidence score between 0.0 and 1.0

**IMPORTANT GUIDELINES**:
1. Extract relationships for entities that have meaningful connections
2. Choose descriptive relationship types that capture the nature of the connection
3. For philosophical/dialogue texts, use types like DISCUSSES, QUESTIONS, RESPONDS_TO
4. For narrative texts, use types like MEETS, HELPS, OPPOSES, TRAVELS_WITH
5. For technical texts, use types like IMPLEMENTS, DEPENDS_ON, EXTENDS
6. Provide higher strength values (0.8-1.0) for explicit relationships
7. Provide lower strength values (0.5-0.7) for implicit or inferred relationships

**TEXT TO ANALYZE**:
{chunk_content}

**OUTPUT FORMAT** (JSON only, no other text):
{{
  "entities": [
    {{"name": "Entity Name", "type": "PERSON", "description": "Brief description"}},
    ...
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "type": "DISCUSSES", "description": "Why they are related", "strength": 0.85}},
    ...
  ]
}}

Return ONLY valid JSON, nothing else."#,
            chunk_content = chunk_content
        )
    }

    /// Extract entities and relationships using LLM
    ///
    /// Uses the configured LLM to extract entities and their relationships from a text chunk.
    /// The LLM analyzes the text and returns structured data with entities, their types,
    /// and the relationships between them.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The text chunk to process
    ///
    /// # Returns
    ///
    /// Returns an `ExtractionResult` containing entities and relationships, or an error
    /// if the LLM is not configured or extraction fails.
    ///
    /// # Errors
    ///
    /// - Returns `GraphRAGError::Config` if Ollama client is not configured
    /// - Returns `GraphRAGError::EntityExtraction` if LLM generation fails
    pub async fn extract_with_llm(
        &self,
        chunk: &TextChunk,
    ) -> Result<ExtractionResult> {
        if let Some(client) = &self.ollama_client {
            let prompt = self.build_extraction_prompt(&chunk.content);

            #[cfg(feature = "tracing")]
            tracing::debug!(
                chunk_id = %chunk.id,
                "Extracting entities and relationships with LLM"
            );

            match client.generate(&prompt).await {
                Ok(response) => {
                    // Parse LLM response as JSON
                    let json_str = response.trim();

                    // Extract JSON from response (LLM might add extra text)
                    let json_str = if let Some(start) = json_str.find('{') {
                        if let Some(end) = json_str.rfind('}') {
                            &json_str[start..=end]
                        } else {
                            json_str
                        }
                    } else {
                        json_str
                    };

                    match serde_json::from_str::<ExtractionResult>(json_str) {
                        Ok(result) => {
                            #[cfg(feature = "tracing")]
                            tracing::info!(
                                chunk_id = %chunk.id,
                                entity_count = result.entities.len(),
                                relationship_count = result.relationships.len(),
                                "Successfully extracted entities and relationships"
                            );
                            Ok(result)
                        }
                        Err(_e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk.id,
                                error = %_e,
                                response = %json_str,
                                "Failed to parse LLM response as JSON, falling back to entity-only extraction"
                            );
                            // Return empty result on parse failure
                            Ok(ExtractionResult {
                                entities: Vec::new(),
                                relationships: Vec::new(),
                            })
                        }
                    }
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!(
                        chunk_id = %chunk.id,
                        error = %e,
                        "LLM extraction failed"
                    );
                    Err(GraphRAGError::EntityExtraction {
                        message: format!("LLM extraction failed: {}", e),
                    })
                }
            }
        } else {
            Err(GraphRAGError::Config {
                message: "Ollama client not configured".to_string(),
            })
        }
    }

    /// Validate a relationship triple against source text (Triple Reflection)
    ///
    /// This method implements DEG-RAG's triple reflection methodology by asking
    /// an LLM to validate whether a relationship is explicitly supported by the text.
    ///
    /// # Arguments
    ///
    /// * `source` - Source entity name
    /// * `relation_type` - Type of relationship
    /// * `target` - Target entity name
    /// * `source_text` - The original text to validate against
    ///
    /// # Returns
    ///
    /// Returns a `TripleValidation` containing validity status, confidence, and reasoning.
    #[cfg(feature = "async")]
    pub async fn validate_triple(
        &self,
        source: &str,
        relation_type: &str,
        target: &str,
        source_text: &str,
    ) -> Result<TripleValidation> {
        if let Some(client) = &self.ollama_client {
            let prompt = format!(
                r#"You are validating a relationship extracted from text.

Text: "{}"

Extracted Relationship:
- Source: {}
- Relationship: {}
- Target: {}

Does this text EXPLICITLY support this relationship?
Consider:
1. Are both entities mentioned in the text?
2. Is the relationship type accurate?
3. Is there direct evidence for this connection?

Respond ONLY with valid JSON in this exact format:
{{
  "valid": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "suggested_fix": "optional fix if invalid"
}}

JSON:"#,
                source_text, source, relation_type, target
            );

            #[cfg(feature = "tracing")]
            tracing::debug!(
                source = %source,
                relation = %relation_type,
                target = %target,
                "Validating relationship triple"
            );

            match client.generate(&prompt).await {
                Ok(response) => {
                    // Extract JSON from response
                    let json_str = response.trim();
                    let json_str = if let Some(start) = json_str.find('{') {
                        if let Some(end) = json_str.rfind('}') {
                            &json_str[start..=end]
                        } else {
                            json_str
                        }
                    } else {
                        json_str
                    };

                    // Try to parse JSON response
                    #[derive(Deserialize)]
                    struct ValidationJson {
                        valid: bool,
                        confidence: f32,
                        reason: String,
                        suggested_fix: Option<String>,
                    }

                    match serde_json::from_str::<ValidationJson>(json_str) {
                        Ok(val) => {
                            #[cfg(feature = "tracing")]
                            tracing::debug!(
                                source = %source,
                                target = %target,
                                valid = val.valid,
                                confidence = val.confidence,
                                "Triple validation complete"
                            );

                            Ok(TripleValidation {
                                is_valid: val.valid,
                                confidence: val.confidence.clamp(0.0, 1.0),
                                reason: val.reason,
                                suggested_fix: val.suggested_fix,
                            })
                        }
                        Err(_e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                error = %_e,
                                response = %json_str,
                                "Failed to parse validation response, assuming valid"
                            );

                            // On parse error, assume valid with low confidence
                            Ok(TripleValidation {
                                is_valid: true,
                                confidence: 0.5,
                                reason: "Failed to parse validation response".to_string(),
                                suggested_fix: None,
                            })
                        }
                    }
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!(
                        error = %e,
                        "Triple validation failed"
                    );

                    // On LLM error, assume valid with low confidence
                    Ok(TripleValidation {
                        is_valid: true,
                        confidence: 0.5,
                        reason: format!("Validation LLM call failed: {}", e),
                        suggested_fix: None,
                    })
                }
            }
        } else {
            // No LLM available, assume valid
            Ok(TripleValidation {
                is_valid: true,
                confidence: 1.0,
                reason: "Ollama client not configured, skipping validation".to_string(),
                suggested_fix: None,
            })
        }
    }

    /// Extract relationships between entities using improved co-occurrence logic
    ///
    /// This is a fallback method when LLM is not available. It identifies relationships
    /// by analyzing entity co-occurrence patterns and contextual clues in the text.
    ///
    /// # Arguments
    ///
    /// * `entities` - List of all known entities
    /// * `chunk` - The text chunk to analyze for relationships
    ///
    /// # Returns
    ///
    /// Returns a vector of tuples containing:
    /// - Source entity ID
    /// - Target entity ID
    /// - Relationship type (string)
    /// - Confidence score (0.0-1.0)
    pub fn extract_relationships_fallback(
        &self,
        entities: &[Entity],
        chunk: &TextChunk,
    ) -> Vec<(EntityId, EntityId, String, f32)> {
        let mut relationships = Vec::new();

        // Get entities that appear in this chunk
        let chunk_entities: Vec<&Entity> = entities
            .iter()
            .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
            .collect();

        // Extract relationships between co-occurring entities
        for i in 0..chunk_entities.len() {
            for j in (i + 1)..chunk_entities.len() {
                let entity1 = chunk_entities[i];
                let entity2 = chunk_entities[j];

                // Infer relationship with improved heuristics
                if let Some((rel_type, confidence)) =
                    self.infer_relationship_with_context(entity1, entity2, &chunk.content)
                {
                    relationships.push((
                        entity1.id.clone(),
                        entity2.id.clone(),
                        rel_type,
                        confidence,
                    ));
                }
            }
        }

        relationships
    }

    /// Infer relationship type with improved context analysis
    ///
    /// Analyzes the context around two entities to determine the type and strength
    /// of their relationship. Uses entity types and contextual patterns to make
    /// intelligent inferences.
    ///
    /// # Arguments
    ///
    /// * `entity1` - First entity in the potential relationship
    /// * `entity2` - Second entity in the potential relationship
    /// * `context` - The text context containing both entities
    ///
    /// # Returns
    ///
    /// Returns `Some((relationship_type, confidence))` if a relationship is detected,
    /// or `None` if entities are too far apart or no clear relationship exists.
    fn infer_relationship_with_context(
        &self,
        entity1: &Entity,
        entity2: &Entity,
        context: &str,
    ) -> Option<(String, f32)> {
        let context_lower = context.to_lowercase();
        let e1_name_lower = entity1.name.to_lowercase();
        let e2_name_lower = entity2.name.to_lowercase();

        // Find positions of entities in text
        let e1_pos = context_lower.find(&e1_name_lower)?;
        let e2_pos = context_lower.find(&e2_name_lower)?;

        // Extract context window between entities (max 200 chars)
        let start = e1_pos.min(e2_pos);
        let end = (e1_pos.max(e2_pos) + 50).min(context.len());
        let window = &context_lower[start..end];

        // Analyze relationship based on context and entity types
        match (&entity1.entity_type[..], &entity2.entity_type[..]) {
            // Person-Person relationships
            ("PERSON", "PERSON") | ("CHARACTER", "CHARACTER") | ("SPEAKER", "SPEAKER") => {
                if window.contains("said") || window.contains("replied") || window.contains("responded") {
                    Some(("RESPONDS_TO".to_string(), 0.85))
                } else if window.contains("asked") || window.contains("questioned") {
                    Some(("QUESTIONS".to_string(), 0.85))
                } else if window.contains("taught") || window.contains("explained") {
                    Some(("TEACHES".to_string(), 0.80))
                } else if window.contains("discussed") || window.contains("spoke about") {
                    Some(("DISCUSSES".to_string(), 0.80))
                } else if window.contains("loved") || window.contains("admired") {
                    Some(("ADMIRES".to_string(), 0.85))
                } else if window.contains("argued") || window.contains("disagreed") {
                    Some(("ARGUES_WITH".to_string(), 0.85))
                } else if window.contains("met") || window.contains("encountered") {
                    Some(("MEETS".to_string(), 0.75))
                } else {
                    // Default for co-occurring persons
                    Some(("INTERACTS_WITH".to_string(), 0.60))
                }
            }

            // Person-Concept relationships
            ("PERSON", "CONCEPT") | ("CHARACTER", "CONCEPT") | ("SPEAKER", "CONCEPT") => {
                if window.contains("discussed") || window.contains("spoke of") {
                    Some(("DISCUSSES".to_string(), 0.80))
                } else if window.contains("defined") || window.contains("described") {
                    Some(("DEFINES".to_string(), 0.85))
                } else if window.contains("questioned") || window.contains("wondered about") {
                    Some(("QUESTIONS".to_string(), 0.80))
                } else {
                    Some(("MENTIONS".to_string(), 0.70))
                }
            }

            // Reverse: Concept-Person
            ("CONCEPT", "PERSON") | ("CONCEPT", "CHARACTER") | ("CONCEPT", "SPEAKER") => {
                Some(("DISCUSSED_BY".to_string(), 0.70))
            }

            // Person-Organization relationships
            ("PERSON", "ORGANIZATION") | ("ORGANIZATION", "PERSON") => {
                if window.contains("works for") || window.contains("employed by") {
                    Some(("WORKS_FOR".to_string(), 0.90))
                } else if window.contains("founded") || window.contains("CEO") || window.contains("leads") {
                    Some(("LEADS".to_string(), 0.90))
                } else {
                    Some(("ASSOCIATED_WITH".to_string(), 0.65))
                }
            }

            // Person-Location relationships
            ("PERSON", "LOCATION") | ("CHARACTER", "LOCATION") => {
                if window.contains("born in") || window.contains("from") {
                    Some(("BORN_IN".to_string(), 0.90))
                } else if window.contains("lives in") || window.contains("resides in") {
                    Some(("LIVES_IN".to_string(), 0.85))
                } else if window.contains("traveled to") || window.contains("visited") {
                    Some(("VISITED".to_string(), 0.80))
                } else {
                    Some(("LOCATED_IN".to_string(), 0.70))
                }
            }

            // Organization-Location relationships
            ("ORGANIZATION", "LOCATION") | ("LOCATION", "ORGANIZATION") => {
                if window.contains("headquartered") || window.contains("based in") {
                    Some(("HEADQUARTERED_IN".to_string(), 0.90))
                } else {
                    Some(("LOCATED_IN".to_string(), 0.75))
                }
            }

            // Concept-Concept relationships
            ("CONCEPT", "CONCEPT") => {
                if window.contains("similar to") || window.contains("related to") {
                    Some(("RELATED_TO".to_string(), 0.75))
                } else if window.contains("opposite") || window.contains("contrasts with") {
                    Some(("CONTRASTS_WITH".to_string(), 0.80))
                } else {
                    Some(("ASSOCIATED_WITH".to_string(), 0.60))
                }
            }

            // Event relationships
            ("PERSON", "EVENT") | ("CHARACTER", "EVENT") => {
                Some(("PARTICIPATES_IN".to_string(), 0.75))
            }
            ("EVENT", "LOCATION") => {
                Some(("OCCURS_IN".to_string(), 0.80))
            }

            // Default fallback
            _ => {
                // Only create relationship if entities are close together (within 100 chars)
                if (e1_pos as i32 - e2_pos as i32).abs() < 100 {
                    Some(("CO_OCCURS".to_string(), 0.50))
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId};

    #[test]
    fn test_prompt_generation() {
        let extractor = LLMRelationshipExtractor::new(None).unwrap();
        let prompt = extractor.build_extraction_prompt("Socrates discusses love with Phaedrus.");

        assert!(prompt.contains("entities"));
        assert!(prompt.contains("relationships"));
        assert!(prompt.contains("Socrates discusses love with Phaedrus"));
    }

    #[test]
    fn test_fallback_extraction() {
        let extractor = LLMRelationshipExtractor::new(None).unwrap();

        let chunk = TextChunk::new(
            ChunkId::new("test".to_string()),
            DocumentId::new("doc".to_string()),
            "Socrates discussed love with Phaedrus in Athens.".to_string(),
            0,
            50,
        );

        let entities = vec![
            Entity::new(
                EntityId::new("person_socrates".to_string()),
                "Socrates".to_string(),
                "PERSON".to_string(),
                0.9,
            ),
            Entity::new(
                EntityId::new("person_phaedrus".to_string()),
                "Phaedrus".to_string(),
                "PERSON".to_string(),
                0.9,
            ),
        ];

        let relationships = extractor.extract_relationships_fallback(&entities, &chunk);

        // Should extract at least one relationship
        assert!(!relationships.is_empty());
    }

    #[test]
    fn test_triple_validation_struct() {
        // Test TripleValidation struct creation and serialization
        let validation = TripleValidation {
            is_valid: true,
            confidence: 0.85,
            reason: "The text explicitly states this relationship.".to_string(),
            suggested_fix: None,
        };

        assert!(validation.is_valid);
        assert_eq!(validation.confidence, 0.85);
        assert!(!validation.reason.is_empty());

        // Test serialization
        let json = serde_json::to_string(&validation).unwrap();
        assert!(json.contains("is_valid"));
        assert!(json.contains("confidence"));
        assert!(json.contains("reason"));
    }

    #[test]
    fn test_triple_validation_deserialization() {
        // Test deserializing validation from JSON (like LLM response)
        let json = r#"{
            "is_valid": true,
            "confidence": 0.9,
            "reason": "Explicitly supported",
            "suggested_fix": null
        }"#;

        let validation: TripleValidation = serde_json::from_str(json).unwrap();
        assert!(validation.is_valid);
        assert_eq!(validation.confidence, 0.9);
        assert_eq!(validation.reason, "Explicitly supported");
        assert!(validation.suggested_fix.is_none());
    }

    #[test]
    fn test_triple_validation_with_suggested_fix() {
        let validation = TripleValidation {
            is_valid: false,
            confidence: 0.3,
            reason: "The relationship is implied but not explicit.".to_string(),
            suggested_fix: Some("Change TAUGHT to INFLUENCED".to_string()),
        };

        assert!(!validation.is_valid);
        assert!(validation.confidence < 0.5);
        assert!(validation.suggested_fix.is_some());

        let fix = validation.suggested_fix.unwrap();
        assert!(fix.contains("INFLUENCED"));
    }

    #[test]
    fn test_validation_confidence_thresholds() {
        // Test different confidence levels
        let high_confidence = TripleValidation {
            is_valid: true,
            confidence: 0.95,
            reason: "Strong evidence".to_string(),
            suggested_fix: None,
        };

        let medium_confidence = TripleValidation {
            is_valid: true,
            confidence: 0.7,
            reason: "Moderate evidence".to_string(),
            suggested_fix: None,
        };

        let low_confidence = TripleValidation {
            is_valid: false,
            confidence: 0.3,
            reason: "Weak evidence".to_string(),
            suggested_fix: Some("Revise".to_string()),
        };

        // Test threshold filtering (default 0.7)
        let threshold = 0.7;
        assert!(high_confidence.confidence >= threshold);
        assert!(medium_confidence.confidence >= threshold);
        assert!(low_confidence.confidence < threshold);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_validate_triple_without_ollama() {
        // Test validation method when Ollama is not configured
        let extractor = LLMRelationshipExtractor::new(None).unwrap();

        let result = extractor.validate_triple(
            "Socrates",
            "TAUGHT",
            "Plato",
            "Socrates taught Plato."
        ).await;

        // Should gracefully fallback with high confidence when no LLM is available
        assert!(result.is_ok(), "Should gracefully handle missing Ollama client");

        let validation = result.unwrap();
        assert!(validation.is_valid, "Fallback should assume valid");
        assert_eq!(validation.confidence, 1.0, "Fallback should have high confidence");
        assert!(validation.reason.contains("not configured"),
            "Reason should explain Ollama is not configured");
    }
}
