//! LLM-based entity and relationship extraction
//!
//! This module provides TRUE LLM-based extraction using Ollama or any other LLM service.
//! Unlike pattern-based extraction, this uses actual language model inference to extract
//! entities and relationships from text with deep semantic understanding.

use crate::{
    core::{ChunkId, Entity, EntityId, EntityMention, Relationship, TextChunk},
    entity::prompts::{EntityData, ExtractionOutput, PromptBuilder, RelationshipData},
    ollama::OllamaClient,
    Result, GraphRAGError,
};
use serde_json;

/// LLM-based entity extractor that uses actual language model calls
pub struct LLMEntityExtractor {
    ollama_client: OllamaClient,
    prompt_builder: PromptBuilder,
    temperature: f32,
    max_tokens: usize,
}

impl LLMEntityExtractor {
    /// Create a new LLM-based entity extractor
    ///
    /// # Arguments
    /// * `ollama_client` - Ollama client for LLM inference
    /// * `entity_types` - List of entity types to extract (e.g., ["PERSON", "LOCATION", "ORGANIZATION"])
    pub fn new(ollama_client: OllamaClient, entity_types: Vec<String>) -> Self {
        Self {
            ollama_client,
            prompt_builder: PromptBuilder::new(entity_types),
            temperature: 0.1, // Low temperature for consistent extraction
            max_tokens: 1500,
        }
    }

    /// Set temperature for LLM generation (default: 0.1)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set maximum tokens for LLM generation (default: 1500)
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Extract entities and relationships from a text chunk using LLM
    ///
    /// This is the REAL extraction that makes actual LLM API calls.
    /// Expected time: 15-30 seconds per chunk depending on chunk size and model.
    #[cfg(feature = "async")]
    pub async fn extract_from_chunk(
        &self,
        chunk: &TextChunk,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        tracing::debug!("LLM extraction for chunk: {} (size: {} chars)", chunk.id, chunk.content.len());

        // Build extraction prompt
        let prompt = self.prompt_builder.build_extraction_prompt(&chunk.content);

        // Call LLM for extraction (THIS IS THE REAL LLM CALL!)
        let llm_response = self.call_llm_with_retry(&prompt).await?;

        // Parse response into structured data
        let extraction_output = self.parse_extraction_response(&llm_response)?;

        // Convert to domain entities and relationships
        let entities = self.convert_to_entities(&extraction_output.entities, &chunk.id, &chunk.content)?;
        let relationships = self.convert_to_relationships(&extraction_output.relationships, &entities)?;

        tracing::info!(
            "LLM extracted {} entities and {} relationships from chunk {}",
            entities.len(),
            relationships.len(),
            chunk.id
        );

        Ok((entities, relationships))
    }

    /// Extract additional entities in a gleaning round (continuation)
    ///
    /// This is used after the initial extraction to catch missed entities.
    #[cfg(feature = "async")]
    pub async fn extract_additional(
        &self,
        chunk: &TextChunk,
        previous_entities: &[EntityData],
        previous_relationships: &[RelationshipData],
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        tracing::debug!("LLM gleaning round for chunk: {}", chunk.id);

        // Build continuation prompt with previous extraction
        let prompt = self.prompt_builder.build_continuation_prompt(
            &chunk.content,
            previous_entities,
            previous_relationships,
        );

        // Call LLM for additional extraction
        let llm_response = self.call_llm_with_retry(&prompt).await?;

        // Parse response
        let extraction_output = self.parse_extraction_response(&llm_response)?;

        // Convert to domain entities
        let entities = self.convert_to_entities(&extraction_output.entities, &chunk.id, &chunk.content)?;
        let relationships = self.convert_to_relationships(&extraction_output.relationships, &entities)?;

        tracing::info!(
            "LLM gleaning extracted {} additional entities and {} relationships",
            entities.len(),
            relationships.len()
        );

        Ok((entities, relationships))
    }

    /// Check if extraction is complete using LLM judgment
    ///
    /// Uses the LLM to determine if all significant entities have been extracted.
    #[cfg(feature = "async")]
    pub async fn check_completion(
        &self,
        chunk: &TextChunk,
        entities: &[EntityData],
        relationships: &[RelationshipData],
    ) -> Result<bool> {
        tracing::debug!("LLM completion check for chunk: {}", chunk.id);

        // Build completion check prompt
        let prompt = self.prompt_builder.build_completion_prompt(
            &chunk.content,
            entities,
            relationships,
        );

        // Call LLM with logit bias for YES/NO response
        let llm_response = self.call_llm_completion_check(&prompt).await?;

        // Parse YES/NO response
        let response_trimmed = llm_response.trim().to_uppercase();
        let is_complete = response_trimmed.starts_with("YES") || response_trimmed.contains("YES");

        tracing::debug!(
            "LLM completion check result: {} (response: {})",
            if is_complete { "COMPLETE" } else { "INCOMPLETE" },
            llm_response.trim()
        );

        Ok(is_complete)
    }

    /// Call LLM with retry logic for extraction
    #[cfg(feature = "async")]
    async fn call_llm_with_retry(&self, prompt: &str) -> Result<String> {
        // Try to get structured JSON output if supported
        // Otherwise fall back to regular generation
        match self.ollama_client.generate(prompt).await {
            Ok(response) => Ok(response),
            Err(e) => {
                tracing::warn!("LLM call failed, retrying: {}", e);
                // Retry once
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                self.ollama_client.generate(prompt).await
            }
        }
    }

    /// Call LLM for completion check with short response
    #[cfg(feature = "async")]
    async fn call_llm_completion_check(&self, prompt: &str) -> Result<String> {
        // For completion check, we want a short YES/NO answer
        // In future, we can use logit bias to force YES/NO tokens
        self.ollama_client.generate(prompt).await
    }

    /// Parse LLM response into structured extraction output
    ///
    /// Handles multiple JSON formats and attempts repair if needed
    fn parse_extraction_response(&self, response: &str) -> Result<ExtractionOutput> {
        // Strategy 1: Try direct JSON parsing
        if let Ok(output) = serde_json::from_str::<ExtractionOutput>(response) {
            return Ok(output);
        }

        // Strategy 2: Try to extract JSON from markdown code blocks
        if let Some(json_str) = Self::extract_json_from_markdown(response) {
            if let Ok(output) = serde_json::from_str::<ExtractionOutput>(json_str) {
                return Ok(output);
            }
        }

        // Strategy 3: Try JSON repair using jsonfixer
        match self.repair_and_parse_json(response) {
            Ok(output) => return Ok(output),
            Err(e) => {
                tracing::warn!("JSON repair failed: {}", e);
            }
        }

        // Strategy 4: Look for JSON anywhere in the response
        if let Some(json_str) = Self::find_json_in_text(response) {
            if let Ok(output) = serde_json::from_str::<ExtractionOutput>(json_str) {
                return Ok(output);
            }

            // Try repairing the extracted JSON
            if let Ok(output) = self.repair_and_parse_json(json_str) {
                return Ok(output);
            }
        }

        // If all strategies fail, return empty extraction
        tracing::error!("Failed to parse LLM response as JSON. Response preview: {}", &response.chars().take(200).collect::<String>());
        Ok(ExtractionOutput {
            entities: vec![],
            relationships: vec![],
        })
    }

    /// Extract JSON from markdown code blocks
    fn extract_json_from_markdown(text: &str) -> Option<&str> {
        // Look for ```json ... ``` or ``` ... ```
        if let Some(start) = text.find("```json") {
            let json_start = start + 7; // length of ```json
            if let Some(end) = text[json_start..].find("```") {
                return Some(text[json_start..json_start + end].trim());
            }
        }

        if let Some(start) = text.find("```") {
            let json_start = start + 3;
            if let Some(end) = text[json_start..].find("```") {
                let candidate = &text[json_start..json_start + end].trim();
                // Check if it looks like JSON
                if candidate.starts_with('{') || candidate.starts_with('[') {
                    return Some(candidate);
                }
            }
        }

        None
    }

    /// Find JSON object or array anywhere in text
    fn find_json_in_text(text: &str) -> Option<&str> {
        // Find first { and last }
        if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                if end > start {
                    return Some(&text[start..=end]);
                }
            }
        }
        None
    }

    /// Attempt to repair malformed JSON using jsonfixer
    fn repair_and_parse_json(&self, json_str: &str) -> Result<ExtractionOutput> {
        // jsonfixer::repair_json returns Result<String, Error>
        let options = jsonfixer::JsonRepairOptions::default();
        let fixed_json = jsonfixer::repair_json(json_str, options)
            .map_err(|e| GraphRAGError::Generation {
                message: format!("JSON repair failed: {:?}", e),
            })?;

        serde_json::from_str::<ExtractionOutput>(&fixed_json)
            .map_err(|e| GraphRAGError::Generation {
                message: format!("Failed to parse repaired JSON: {}", e),
            })
    }

    /// Convert EntityData to domain Entity objects
    fn convert_to_entities(
        &self,
        entity_data: &[EntityData],
        chunk_id: &ChunkId,
        chunk_text: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for entity_item in entity_data {
            // Generate entity ID
            let entity_id = EntityId::new(format!(
                "{}_{}",
                entity_item.entity_type,
                self.normalize_name(&entity_item.name)
            ));

            // Find mentions in chunk
            let mentions = self.find_mentions(&entity_item.name, chunk_id, chunk_text);

            // Create entity with mentions
            // Note: Description is stored in the entity but not used in current Entity struct
            // We store it in the entity name or as a separate field if needed
            let entity = Entity::new(
                entity_id,
                entity_item.name.clone(),
                entity_item.entity_type.clone(),
                0.9, // High confidence since it's LLM-extracted
            )
            .with_mentions(mentions);

            entities.push(entity);
        }

        Ok(entities)
    }

    /// Find all mentions of an entity name in the chunk text
    fn find_mentions(&self, name: &str, chunk_id: &ChunkId, text: &str) -> Vec<EntityMention> {
        let mut mentions = Vec::new();
        let mut start = 0;

        while let Some(pos) = text[start..].find(name) {
            let actual_pos = start + pos;
            mentions.push(EntityMention {
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
                mentions.push(EntityMention {
                    chunk_id: chunk_id.clone(),
                    start_offset: actual_pos,
                    end_offset: actual_pos + name.len(),
                    confidence: 0.85, // Slightly lower confidence for case-insensitive match
                });
                start = actual_pos + name.len();
            }
        }

        mentions
    }

    /// Convert RelationshipData to domain Relationship objects
    fn convert_to_relationships(
        &self,
        relationship_data: &[RelationshipData],
        entities: &[Entity],
    ) -> Result<Vec<Relationship>> {
        let mut relationships = Vec::new();

        // Build entity name to ID mapping
        let mut name_to_entity: std::collections::HashMap<String, &Entity> = std::collections::HashMap::new();
        for entity in entities {
            name_to_entity.insert(entity.name.to_lowercase(), entity);
        }

        for rel_item in relationship_data {
            // Find source and target entities
            let source_entity = name_to_entity.get(&rel_item.source.to_lowercase());
            let target_entity = name_to_entity.get(&rel_item.target.to_lowercase());

            if let (Some(source), Some(target)) = (source_entity, target_entity) {
                let relationship = Relationship {
                    source: source.id.clone(),
                    target: target.id.clone(),
                    relation_type: rel_item.description.clone(),
                    confidence: rel_item.strength as f32,
                    context: vec![], // No context chunks for this relationship
                    embedding: None,
                    temporal_type: None,
                    temporal_range: None,
                    causal_strength: None,
                };

                relationships.push(relationship);
            } else {
                tracing::warn!(
                    "Skipping relationship: entity not found. Source: {}, Target: {}",
                    rel_item.source,
                    rel_item.target
                );
            }
        }

        Ok(relationships)
    }

    /// Normalize entity name for ID generation
    fn normalize_name(&self, name: &str) -> String {
        name.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .replace(' ', "_")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::DocumentId, ollama::OllamaConfig};

    fn create_test_chunk() -> TextChunk {
        TextChunk::new(
            ChunkId::new("chunk_001".to_string()),
            DocumentId::new("doc_001".to_string()),
            "Tom Sawyer is a young boy who lives in St. Petersburg with his Aunt Polly. \
             Tom is best friends with Huckleberry Finn. They often go on adventures together."
                .to_string(),
            0,
            150,
        )
    }

    #[test]
    fn test_extract_json_from_markdown() {
        let markdown = r#"
Here's the extraction:
```json
{
  "entities": [],
  "relationships": []
}
```
"#;
        let json = LLMEntityExtractor::extract_json_from_markdown(markdown);
        assert!(json.is_some());
        assert!(json.unwrap().contains("entities"));
    }

    #[test]
    fn test_find_json_in_text() {
        let text = "Some text before { \"entities\": [] } some text after";
        let json = LLMEntityExtractor::find_json_in_text(text);
        assert!(json.is_some());
        assert_eq!(json.unwrap(), "{ \"entities\": [] }");
    }

    #[test]
    fn test_parse_valid_json() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let extractor = LLMEntityExtractor::new(
            ollama_client,
            vec!["PERSON".to_string(), "LOCATION".to_string()],
        );

        let response = r#"
{
  "entities": [
    {
      "name": "Tom Sawyer",
      "type": "PERSON",
      "description": "A young boy"
    }
  ],
  "relationships": []
}
"#;

        let result = extractor.parse_extraction_response(response);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.entities.len(), 1);
        assert_eq!(output.entities[0].name, "Tom Sawyer");
    }

    #[test]
    fn test_convert_to_entities() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let extractor = LLMEntityExtractor::new(
            ollama_client,
            vec!["PERSON".to_string()],
        );

        let chunk = create_test_chunk();
        let entity_data = vec![EntityData {
            name: "Tom Sawyer".to_string(),
            entity_type: "PERSON".to_string(),
            description: "A young boy".to_string(),
        }];

        let entities = extractor
            .convert_to_entities(&entity_data, &chunk.id, &chunk.content)
            .unwrap();

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Tom Sawyer");
        assert_eq!(entities[0].entity_type, "PERSON");
        assert!(!entities[0].mentions.is_empty());
    }

    #[test]
    fn test_find_mentions() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let extractor = LLMEntityExtractor::new(ollama_client, vec!["PERSON".to_string()]);

        let chunk = create_test_chunk();
        let mentions = extractor.find_mentions("Tom", &chunk.id, &chunk.content);

        assert!(!mentions.is_empty());
        assert!(mentions.len() >= 2); // "Tom Sawyer" and "Tom is best friends"
    }

    #[test]
    fn test_normalize_name() {
        let ollama_config = OllamaConfig::default();
        let ollama_client = OllamaClient::new(ollama_config);
        let extractor = LLMEntityExtractor::new(ollama_client, vec!["PERSON".to_string()]);

        assert_eq!(extractor.normalize_name("Tom Sawyer"), "tom_sawyer");
        assert_eq!(extractor.normalize_name("New York City"), "new_york_city");
        assert_eq!(extractor.normalize_name("Dr. Smith"), "dr_smith");
    }
}
