//! Prompt templates for LLM-based entity and relationship extraction
//!
//! Based on Microsoft GraphRAG prompts with structured JSON output

use crate::core::SAID_MARKER_PREFIX;
use crate::ontology::Ontology;
use serde::{Deserialize, Serialize};

/// Entity extraction prompt template (Microsoft GraphRAG style)
pub const ENTITY_EXTRACTION_PROMPT: &str = r#"-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Language-
Work in the language of the text. Copy every entity name EXACTLY as it appears in the text, in the same language: never translate, paraphrase or add articles that are not in the text (e.g. keep "Bottega di Geppetto", never "Geppetto's Shop"). Write entity and relationship descriptions in the language of the text. Only the type labels and relationship type labels listed below stay in English.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, copied exactly as written in the text (same language, not translated)
- entity_type: One of the following types: [{entity_types}], or OTHER
- entity_description: Comprehensive description of the entity's attributes and activities, written in the language of the text

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other, written in the language of the text (never in English unless the text itself is in English)
- relationship_type: the relationship type from the list below that best matches the description
- relationship_strength: a numeric score between 0 and 1 indicating strength of the relationship

{ontology}

3. Return output in JSON format with the following structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "description": "entity description, in the language of the text"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "description": "relationship description, in the language of the text",
      "type": "RELATIONSHIP_TYPE",
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

-Language-
Work in the language of the text. Copy every entity name EXACTLY as it appears in the text, in the same language: never translate, paraphrase or add articles that are not in the text (e.g. keep "Bottega di Geppetto", never "Geppetto's Shop"). Write entity and relationship descriptions in the language of the text. Only the type labels and relationship type labels listed below stay in English.

-Steps-
1. Review the entities you previously identified:
{previous_entities}

2. Review the relationships you previously identified:
{previous_relationships}

3. Carefully review the original text again and identify:
- Any entities you may have missed
- Any relationships between entities you may have overlooked
- Any entities that need better descriptions

{ontology}

4. Return ONLY the NEW entities and relationships you discovered in this pass, using the same JSON format:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "description": "entity description, in the language of the text"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "description": "relationship description, in the language of the text",
      "type": "RELATIONSHIP_TYPE",
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

/// Relation-only extraction prompt (GLiNER to LLM cascade): the entities are given,
/// the model must only link them. Static prefix first, per-chunk data after `-Real Data-`.
pub const RELATION_EXTRACTION_PROMPT: &str = r#"-Goal-
Given a text and the list of entities already identified in it, identify all pairs of entities from the list that are *clearly related* to each other in the text. Do not add entities that are not in the list.

-Language-
Work in the language of the text. Copy every source and target name EXACTLY as written in the entity list: never translate, paraphrase or invent names. Write relationship descriptions in the language of the text. Only the relationship type labels listed below stay in English.

-Steps-
For each pair of related entities, extract the following information:
- source: name of the source entity, exactly as in the list
- target: name of the target entity, exactly as in the list (different from the source)
- description: explanation as to why the source and the target are related, written in the language of the text (never in English unless the text itself is in English)
- type: the relationship type from the list below that best matches the description
- strength: a numeric score between 0 and 1 indicating strength of the relationship

{relationship_types}

Return output in JSON format with the following structure:
{
  "relationships": [
    {
      "source": "source entity name",
      "target": "target entity name",
      "description": "relationship description, in the language of the text",
      "type": "RELATIONSHIP_TYPE",
      "strength": 0.8
    }
  ]
}

If no pair is clearly related, return an empty array.

-Real Data-
######################
Entities:
{entities}
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
          "type": {"type": "string"},
          "strength": {"type": "number"}
        },
        "required": ["source", "target", "description", "type", "strength"]
      }
    }
  },
  "required": ["entities", "relationships"]
}"#;

/// Extra instruction block (`entity_extraction.epistemic_status`), inserted right before
/// `-Real Data-` so the static prefix before the chunk text stays chunk-independent.
pub const EPISTEMIC_PROMPT_BLOCK: &str = r#"-Epistemic status-
Add two more fields to every relationship object:
- status: "narrated" when the narrator states the fact as happening in the story; "said" when the fact is only asserted, claimed, promised, threatened or told by a character in dialogue (including lies and stories told by characters)
- said_by: when status is "said", the name of the character who says it, exactly as written in the text; otherwise an empty string
When unsure, use "narrated".

"#;

/// Output of relation-only extraction: just the relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationExtractionOutput {
    /// Relationships between the entities given in the prompt
    pub relationships: Vec<RelationshipData>,
}

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
    #[serde(default)]
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
    #[serde(default)]
    pub description: String,
    /// Relationship type from the ontology (empty when the model omitted it)
    #[serde(default, rename = "type")]
    pub relation_type: String,
    /// Strength/confidence score of the relationship (0.0-1.0)
    #[serde(default = "default_strength")]
    pub strength: f64,
    /// `"narrated"` (default) or `"said"` (only asserted by a character); see
    /// [`EPISTEMIC_PROMPT_BLOCK`]. Missing or unknown values count as narrated.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    /// Character who says the fact when `status` is `"said"` (empty otherwise)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub said_by: String,
}

impl RelationshipData {
    /// True when the model marked the relationship as only said by a character
    /// (case-insensitive; anything else, including unknown values, is narrated).
    pub fn is_said(&self) -> bool {
        self.status.trim().eq_ignore_ascii_case("said")
    }
}

/// Description stored for a relationship the model marked as `said`: the original
/// description prefixed with `[detto da <said_by>]` (`[detto da un personaggio]` when the
/// name is empty). Marker only, no new field on `Relationship`.
pub fn said_description(description: Option<String>, said_by: &str) -> Option<String> {
    let who = said_by.trim();
    let marker = if who.is_empty() {
        format!("{SAID_MARKER_PREFIX}un personaggio]")
    } else {
        format!("{SAID_MARKER_PREFIX}{who}]")
    };
    match description {
        Some(d) if !d.trim().is_empty() => Some(format!("{marker} {d}")),
        _ => Some(marker),
    }
}

fn default_strength() -> f64 {
    0.5
}

/// Prompt builder for entity extraction
pub struct PromptBuilder {
    ontology: Ontology,
    epistemic: bool,
}

impl PromptBuilder {
    /// Create a prompt builder over the generic ontology restricted to `entity_types`
    pub fn new(entity_types: Vec<String>) -> Self {
        Self::with_ontology(Ontology::generic().with_entity_types(&entity_types))
    }

    /// Create a prompt builder for an explicit ontology
    pub fn with_ontology(ontology: Ontology) -> Self {
        Self {
            ontology,
            epistemic: false,
        }
    }

    /// Ask for `status` / `said_by` on every relationship (default off: prompts unchanged)
    pub fn with_epistemic_status(mut self, enabled: bool) -> Self {
        self.epistemic = enabled;
        self
    }

    /// Whether the epistemic status instructions are part of the prompts
    pub fn epistemic_status(&self) -> bool {
        self.epistemic
    }

    /// Ontology used by this builder
    pub fn ontology(&self) -> &Ontology {
        &self.ontology
    }

    /// Fill the ontology placeholders shared by all extraction templates.
    /// They precede `{input_text}`, so the rendered prefix is identical for
    /// every chunk and stays in the llama.cpp prompt cache.
    fn render(&self, template: &str) -> String {
        let with_block;
        let template = if self.epistemic {
            with_block = template.replacen(
                "-Real Data-",
                &format!("{EPISTEMIC_PROMPT_BLOCK}-Real Data-"),
                1,
            );
            with_block.as_str()
        } else {
            template
        };
        template
            .replace(
                "{entity_types}",
                &self.ontology.entity_type_names().join(", "),
            )
            .replace("{ontology}", &self.ontology.prompt_section())
    }

    /// Build initial entity extraction prompt
    pub fn build_extraction_prompt(&self, text: &str) -> String {
        self.render(ENTITY_EXTRACTION_PROMPT)
            .replace("{input_text}", text)
    }

    /// Build the relation-only prompt for `entities` found in `text`.
    ///
    /// The ontology section is rendered before `-Real Data-`, so the prefix is
    /// identical for every chunk (KV-cache friendly).
    pub fn build_relation_prompt(&self, text: &str, entities: &[crate::core::Entity]) -> String {
        let list = entities
            .iter()
            .map(|e| format!("- {} ({})", e.name, e.entity_type))
            .collect::<Vec<_>>()
            .join("\n");
        let template = if self.epistemic {
            RELATION_EXTRACTION_PROMPT.replacen(
                "-Real Data-",
                &format!("{EPISTEMIC_PROMPT_BLOCK}-Real Data-"),
                1,
            )
        } else {
            RELATION_EXTRACTION_PROMPT.to_string()
        };
        template
            .replace(
                "{relationship_types}",
                &self.ontology.relation_prompt_section(),
            )
            .replace("{entities}", &list)
            .replace("{input_text}", text)
    }

    /// Build gleaning continuation prompt
    pub fn build_continuation_prompt(
        &self,
        text: &str,
        previous_entities: &[EntityData],
        previous_relationships: &[RelationshipData],
    ) -> String {
        // Format previous entities for display
        let entities_summary = previous_entities
            .iter()
            .map(|e| format!("- {} ({}): {}", e.name, e.entity_type, e.description))
            .collect::<Vec<_>>()
            .join("\n");

        // Format previous relationships for display
        let relationships_summary = previous_relationships
            .iter()
            .map(|r| {
                format!(
                    "- {} -[{}]-> {}: {} (strength: {:.2})",
                    r.source, r.relation_type, r.target, r.description, r.strength
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        self.render(GLEANING_CONTINUATION_PROMPT)
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
            format!(
                "{}...\n(showing 20 of {} entities)",
                entities_summary,
                entities.len()
            )
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
            format!(
                "{}...\n(showing 20 of {} relationships)",
                relationships_summary,
                relationships.len()
            )
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
    fn extraction_prompt_lists_ontology_and_keeps_static_prefix() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]);
        let a = builder.build_extraction_prompt("chunk one");
        let b = builder.build_extraction_prompt("another chunk");
        assert!(a.contains("- WORKS_FOR:"));
        assert!(a.contains("\"type\": \"RELATIONSHIP_TYPE\""));
        assert!(!a.contains("{ontology}") && !a.contains("{entity_types}"));
        let prefix = a.find("chunk one").unwrap();
        assert_eq!(a[..prefix], b[..prefix]);
    }

    #[test]
    fn prompts_keep_source_language_in_static_prefix() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]);
        let text = "La Bottega di Geppetto.";
        let extraction = builder.build_extraction_prompt(text);
        let continuation = builder.build_continuation_prompt(text, &[], &[]);
        for prompt in [&extraction, &continuation] {
            let lang = prompt.find("-Language-").expect("language rule");
            assert!(prompt.contains("never translate"));
            assert!(prompt[lang..].contains("same language"));
            assert!(
                lang < prompt.find(text).unwrap(),
                "rule is in the static part"
            );
            // The rule is repeated next to the description field: the general
            // rule alone was ignored for most relationship descriptions.
            let example = prompt
                .find("\"relationship description, in the language of the text\"")
                .expect("language reminder in the JSON example");
            assert!(example < prompt.find(text).unwrap());
        }
        assert!(extraction.contains("never in English unless the text itself is in English"));
    }

    #[test]
    fn relationship_data_accepts_legacy_output_without_type() {
        let r: RelationshipData =
            serde_json::from_str(r#"{"source":"A","target":"B","description":"WORKS_FOR"}"#)
                .unwrap();
        assert_eq!(r.relation_type, "");
        assert_eq!(r.strength, 0.5);
    }

    #[test]
    fn test_build_continuation_prompt() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]);

        let previous_entities = vec![EntityData {
            name: "Tom".to_string(),
            entity_type: "PERSON".to_string(),
            description: "A young boy".to_string(),
        }];

        let previous_relationships = vec![RelationshipData {
            source: "Tom".to_string(),
            target: "Huck".to_string(),
            description: "friends".to_string(),
            relation_type: "FRIEND_OF".to_string(),
            strength: 0.9,
            status: String::new(),
            said_by: String::new(),
        }];

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

        let entities = vec![EntityData {
            name: "Tom".to_string(),
            entity_type: "PERSON".to_string(),
            description: "A young boy".to_string(),
        }];

        let relationships = vec![RelationshipData {
            source: "Tom".to_string(),
            target: "Huck".to_string(),
            description: "friends".to_string(),
            relation_type: "FRIEND_OF".to_string(),
            strength: 0.9,
            status: String::new(),
            said_by: String::new(),
        }];

        let prompt = builder.build_completion_prompt("Test text", &entities, &relationships);

        assert!(prompt.contains("Tom"));
        assert!(prompt.contains("YES or NO"));
    }

    #[test]
    fn test_extraction_output_serialization() {
        let output = ExtractionOutput {
            entities: vec![EntityData {
                name: "Tom Sawyer".to_string(),
                entity_type: "PERSON".to_string(),
                description: "The protagonist".to_string(),
            }],
            relationships: vec![RelationshipData {
                source: "Tom Sawyer".to_string(),
                target: "Huck Finn".to_string(),
                description: "best friends".to_string(),
                relation_type: "FRIEND_OF".to_string(),
                strength: 0.95,
                status: String::new(),
                said_by: String::new(),
            }],
        };

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("Tom Sawyer"));
        assert!(json.contains("PERSON"));

        let deserialized: ExtractionOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entities.len(), 1);
        assert_eq!(deserialized.relationships.len(), 1);
    }

    fn relation_entities() -> Vec<crate::core::Entity> {
        use crate::core::{Entity, EntityId};
        ["Pinocchio", "Geppetto"]
            .iter()
            .map(|n| {
                Entity::new(
                    EntityId::from_type_and_name("PERSON", n),
                    n.to_string(),
                    "PERSON".to_string(),
                    0.9,
                )
            })
            .collect()
    }

    #[test]
    fn relation_prompt_has_static_prefix_before_entities_and_text() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]);
        let a = builder.build_relation_prompt("Testo uno.", &relation_entities());
        let b = builder.build_relation_prompt("Altro testo.", &relation_entities()[..1]);
        let split = a.find("-Real Data-").expect("dynamic part marker");
        assert_eq!(
            a[..split],
            b[..b.find("-Real Data-").unwrap()],
            "prefix must be chunk-independent"
        );
        let tail = &a[split..];
        assert!(tail.contains("- Pinocchio (PERSON)"));
        assert!(tail.contains("- Geppetto (PERSON)"));
        assert!(tail.contains("Testo uno."));
        let prefix = &a[..split];
        assert!(prefix.contains("-Language-"));
        assert!(prefix.contains("FRIEND_OF"));
        assert!(prefix.contains("\"relationships\""));
        assert!(!prefix.contains("Pinocchio"));
        assert!(!a.contains("{{") && !a.contains("{entities}") && !a.contains("{input_text}"));
    }

    #[test]
    fn relation_output_parses_without_entities_field() {
        let out: RelationExtractionOutput = serde_json::from_str(
            r#"{"relationships":[{"source":"A","target":"B","description":"d","type":"FRIEND_OF","strength":0.8}]}"#,
        )
        .unwrap();
        assert_eq!(out.relationships.len(), 1);
    }

    fn old_render(template: &str, ontology: &Ontology) -> String {
        template
            .replace("{entity_types}", &ontology.entity_type_names().join(", "))
            .replace("{ontology}", &ontology.prompt_section())
    }

    #[test]
    fn epistemic_off_prompts_are_byte_identical_to_the_templates() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string(), "LOCATION".to_string()]);
        let onto = builder.ontology().clone();
        assert!(!builder.epistemic_status());
        let text = "Il Gatto gli ha mangiato i piedi.";
        assert_eq!(
            builder.build_extraction_prompt(text),
            old_render(ENTITY_EXTRACTION_PROMPT, &onto).replace("{input_text}", text)
        );
        let prev_e = [EntityData {
            name: "Tom".into(),
            entity_type: "PERSON".into(),
            description: "d".into(),
        }];
        let prev_r = [RelationshipData {
            source: "Tom".into(),
            target: "Huck".into(),
            description: "friends".into(),
            relation_type: "FRIEND_OF".into(),
            strength: 0.9,
            status: String::new(),
            said_by: String::new(),
        }];
        assert_eq!(
            builder.build_continuation_prompt(text, &prev_e, &prev_r),
            old_render(GLEANING_CONTINUATION_PROMPT, &onto)
                .replace("{input_text}", text)
                .replace("{previous_entities}", "- Tom (PERSON): d")
                .replace(
                    "{previous_relationships}",
                    "- Tom -[FRIEND_OF]-> Huck: friends (strength: 0.90)"
                )
        );
        let entities = relation_entities();
        assert_eq!(
            builder.build_relation_prompt(text, &entities),
            RELATION_EXTRACTION_PROMPT
                .replace("{relationship_types}", &onto.relation_prompt_section())
                .replace("{entities}", "- Pinocchio (PERSON)\n- Geppetto (PERSON)")
                .replace("{input_text}", text)
        );
    }

    #[test]
    fn epistemic_on_adds_block_before_real_data_and_keeps_prefix_static() {
        let builder = PromptBuilder::new(vec!["PERSON".to_string()]).with_epistemic_status(true);
        let off = PromptBuilder::new(vec!["PERSON".to_string()]);
        let a = builder.build_extraction_prompt("chunk one");
        let b = builder.build_extraction_prompt("another chunk");
        let block = a.find("-Epistemic status-").expect("instruction block");
        assert!(a.contains("\"narrated\"") && a.contains("said_by"));
        assert!(block < a.find("-Real Data-").unwrap());
        let prefix = a.find("chunk one").unwrap();
        assert_eq!(a[..prefix], b[..prefix]);
        // everything but the block is the flag-off prompt
        assert_eq!(
            a.replacen(EPISTEMIC_PROMPT_BLOCK, "", 1),
            off.build_extraction_prompt("chunk one")
        );
        let cont = builder.build_continuation_prompt("chunk", &[], &[]);
        assert!(cont.contains("-Epistemic status-"));
        let rel = builder.build_relation_prompt("chunk", &relation_entities());
        assert!(rel.contains("-Epistemic status-"));
    }

    #[test]
    fn relationship_data_epistemic_fields_are_optional_and_tolerant() {
        let plain: RelationshipData =
            serde_json::from_str(r#"{"source":"A","target":"B","description":"d"}"#).unwrap();
        assert!(!plain.is_said());
        let said: RelationshipData = serde_json::from_str(
            r#"{"source":"A","target":"B","status":" Said ","said_by":"Pinocchio"}"#,
        )
        .unwrap();
        assert!(said.is_said());
        let unknown: RelationshipData =
            serde_json::from_str(r#"{"source":"A","target":"B","status":"rumoured"}"#).unwrap();
        assert!(!unknown.is_said());
        // not serialized when empty: cached normalized responses keep their old shape
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("status") && !json.contains("said_by"));
    }

    #[test]
    fn said_description_builds_marker() {
        assert_eq!(
            said_description(Some("ha perso i piedi".into()), "Pinocchio").as_deref(),
            Some("[detto da Pinocchio] ha perso i piedi")
        );
        assert_eq!(
            said_description(Some("x".into()), "  ").as_deref(),
            Some("[detto da un personaggio] x")
        );
        assert_eq!(
            said_description(None, "Pinocchio").as_deref(),
            Some("[detto da Pinocchio]")
        );
    }
}
