//! Prompt templates for LLM-based entity and relationship extraction
//!
//! Based on Microsoft GraphRAG prompts with structured JSON output

use serde::{Deserialize, Serialize};

/// Entity extraction prompt template (Microsoft GraphRAG style)
pub const ENTITY_EXTRACTION_PROMPT: &str = r#"-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in JSON format with the following structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "description": "entity description"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "description": "relationship description",
      "strength": 0.8
    }}
  ]
}}

-Real Data-
######################
Entity Types: {entity_types}
Text: {input_text}
######################
Output:
"#;

/// Gleaning continuation prompt for additional rounds
pub const GLEANING_CONTINUATION_PROMPT: &str = r#"-Goal-
You previously extracted entities and relationships from a text document. Review your previous extraction and the original text to identify any additional entities or relationships you may have missed in the first pass.

-Steps-
1. Review the entities you previously identified:
{previous_entities}

2. Review the relationships you previously identified:
{previous_relationships}

3. Carefully review the original text again and identify:
- Any entities you may have missed
- Any relationships between entities you may have overlooked
- Any entities that need better descriptions

4. Return ONLY the NEW entities and relationships you discovered in this pass, using the same JSON format:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "description": "entity description"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "description": "relationship description",
      "strength": 0.8
    }}
  ]
}}

If you found no additional entities or relationships, return empty arrays.

-Real Data-
######################
Entity Types: {entity_types}
Text: {input_text}
######################
Output:
"#;

/// Completion check prompt to determine if extraction is complete
pub const COMPLETION_CHECK_PROMPT: &str = r#"Based on the text below and the entities/relationships already extracted, are there any significant entities or relationships that have been missed?

Text:
{input_text}

Current Entities ({entity_count}):
{entities_summary}

Current Relationships ({relationship_count}):
{relationships_summary}

Think carefully about:
1. Are all important characters, people, organizations mentioned in the text captured?
2. Are all significant locations, places, settings identified?
3. Are all key events, objects, concepts extracted?
4. Are all meaningful relationships between entities documented?

Respond with ONLY "YES" if the extraction is complete and thorough, or "NO" if there are still significant entities or relationships missing.

Answer (YES or NO):"#;

/// JSON schema for entity extraction output
pub const ENTITY_EXTRACTION_JSON_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "description": {"type": "string"}
        },
        "required": ["name", "type", "description"]
      }
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source": {"type": "string"},
          "target": {"type": "string"},
          "description": {"type": "string"},
          "strength": {"type": "number"}
        },
        "required": ["source", "target", "description", "strength"]
      }
    }
  },
  "required": ["entities", "relationships"]
}"#;

/// Structured extraction output from LLM entity and relationship analysis.
///
/// This structure contains the results from LLM-based entity extraction,
/// including both discovered entities and their relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionOutput {
    /// List of entities extracted from the text
    pub entities: Vec<EntityData>,
    /// List of relationships between extracted entities
    pub relationships: Vec<RelationshipData>,
}

/// Represents an entity extracted from text with its metadata.
///
/// Contains the entity's name, type classification, and a description
/// of its role or significance in the context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityData {
    /// The name/identifier of the extracted entity
    pub name: String,
    /// The type/category of the entity (e.g., "PERSON", "ORGANIZATION", "CONCEPT")
    #[serde(rename = "type")]
    pub entity_type: String,
    /// Description of the entity's role or significance in the context
    pub description: String,
}

/// Represents a relationship between two extracted entities.
///
/// Defines how entities are connected with a description and strength
/// indicating the relationship's importance or confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipData {
    /// The source entity in the relationship
    pub source: String,
    /// The target entity in the relationship
    pub target: String,
    /// Description of the relationship type and context
    pub description: String,
    /// Strength/confidence score of the relationship (0.0-1.0)
    pub strength: f64,
}

/// Prompt builder for entity extraction
pub struct PromptBuilder {
    entity_types: Vec<String>,
    tuple_delimiter: String,
}

impl PromptBuilder {
    /// Create a new prompt builder
    pub fn new(entity_types: Vec<String>) -> Self {
        Self {
            entity_types,
            tuple_delimiter: "|".to_string(),
        }
    }

    /// Build initial entity extraction prompt
    pub fn build_extraction_prompt(&self, text: &str) -> String {
        let entity_types_str = self.entity_types.join(", ");

        ENTITY_EXTRACTION_PROMPT
            .replace("{entity_types}", &entity_types_str)
            .replace("{tuple_delimiter}", &self.tuple_delimiter)
            .replace("{input_text}", text)
    }

    /// Build gleaning continuation prompt
    pub fn build_continuation_prompt(
        &self,
        text: &str,
        previous_entities: &[EntityData],
        previous_relationships: &[RelationshipData],
    ) -> String {
        let entity_types_str = self.entity_types.join(", ");

        // Format previous entities for display
        let entities_summary = previous_entities
            .iter()
            .map(|e| format!("- {} ({}): {}", e.name, e.entity_type, e.description))
            .collect::<Vec<_>>()
            .join("\n");

        // Format previous relationships for display
        let relationships_summary = previous_relationships
            .iter()
            .map(|r| format!("- {} -> {}: {} (strength: {:.2})", r.source, r.target, r.description, r.strength))
            .collect::<Vec<_>>()
            .join("\n");

        GLEANING_CONTINUATION_PROMPT
            .replace("{entity_types}", &entity_types_str)
            .replace("{input_text}", text)
            .replace("{previous_entities}", &entities_summary)
            .replace("{previous_relationships}", &relationships_summary)
    }

    /// Build completion check prompt
    pub fn build_completion_prompt(
        &self,
        text: &str,
        entities: &[EntityData],
        relationships: &[RelationshipData],
    ) -> String {
        // Create concise summary of entities
        let entities_summary = entities
            .iter()
            .take(20)  // Limit to first 20 to keep prompt manageable
            .map(|e| format!("- {} ({})", e.name, e.entity_type))
            .collect::<Vec<_>>()
            .join("\n");

        let entities_summary = if entities.len() > 20 {
            format!("{}...\n(showing 20 of {} entities)", entities_summary, entities.len())
        } else {
            entities_summary
        };

        // Create concise summary of relationships
        let relationships_summary = relationships
            .iter()
            .take(20)  // Limit to first 20
            .map(|r| format!("- {} -> {}", r.source, r.target))
            .collect::<Vec<_>>()
            .join("\n");

        let relationships_summary = if relationships.len() > 20 {
            format!("{}...\n(showing 20 of {} relationships)", relationships_summary, relationships.len())
        } else {
            relationships_summary
        };

        COMPLETION_CHECK_PROMPT
            .replace("{input_text}", text)
            .replace("{entity_count}", &entities.len().to_string())
            .replace("{entities_summary}", &entities_summary)
            .replace("{relationship_count}", &relationships.len().to_string())
            .replace("{relationships_summary}", &relationships_summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_extraction_prompt() {
        let builder = PromptBuilder::new(vec![
            "PERSON".to_string(),
            "LOCATION".to_string(),
            "ORGANIZATION".to_string(),
        ]);

        let prompt = builder.build_extraction_prompt("Tom and Huck went to the cave.");

        assert!(prompt.contains("PERSON"));
        assert!(prompt.contains("LOCATION"));
        assert!(prompt.contains("ORGANIZATION"));
        assert!(prompt.contains("Tom and Huck went to the cave."));
    }

    #[test]
    fn test_build_continuation_prompt() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]);

        let previous_entities = vec![
            EntityData {
                name: "Tom".to_string(),
                entity_type: "PERSON".to_string(),
                description: "A young boy".to_string(),
            },
        ];

        let previous_relationships = vec![
            RelationshipData {
                source: "Tom".to_string(),
                target: "Huck".to_string(),
                description: "friends".to_string(),
                strength: 0.9,
            },
        ];

        let prompt = builder.build_continuation_prompt(
            "Tom and Huck are best friends.",
            &previous_entities,
            &previous_relationships,
        );

        assert!(prompt.contains("Tom"));
        assert!(prompt.contains("Huck"));
        assert!(prompt.contains("friends"));
    }

    #[test]
    fn test_build_completion_prompt() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]);

        let entities = vec![
            EntityData {
                name: "Tom".to_string(),
                entity_type: "PERSON".to_string(),
                description: "A young boy".to_string(),
            },
        ];

        let relationships = vec![
            RelationshipData {
                source: "Tom".to_string(),
                target: "Huck".to_string(),
                description: "friends".to_string(),
                strength: 0.9,
            },
        ];

        let prompt = builder.build_completion_prompt("Test text", &entities, &relationships);

        assert!(prompt.contains("Tom"));
        assert!(prompt.contains("YES or NO"));
    }

    #[test]
    fn test_extraction_output_serialization() {
        let output = ExtractionOutput {
            entities: vec![
                EntityData {
                    name: "Tom Sawyer".to_string(),
                    entity_type: "PERSON".to_string(),
                    description: "The protagonist".to_string(),
                },
            ],
            relationships: vec![
                RelationshipData {
                    source: "Tom Sawyer".to_string(),
                    target: "Huck Finn".to_string(),
                    description: "best friends".to_string(),
                    strength: 0.95,
                },
            ],
        };

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("Tom Sawyer"));
        assert!(json.contains("PERSON"));

        let deserialized: ExtractionOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entities.len(), 1);
        assert_eq!(deserialized.relationships.len(), 1);
    }
}
