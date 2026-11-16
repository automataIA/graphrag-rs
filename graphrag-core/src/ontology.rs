//! Shared, versioned extraction ontology.
//!
//! One [`Ontology`] defines the closed set of entity types and relation types
//! every extractor (LLM single-pass, gleaning, GLiNER) maps its output onto.
//! Labels outside the ontology are never stored verbatim as types: unknown
//! entity types become [`OTHER`], unknown relation types become [`OTHER`] and
//! keep the original label and description in `Relationship::description`.
//!
//! [`Ontology::id`] combines name, version and a content fingerprint, so any
//! change to the type lists yields a new id (used to detect stale workspaces
//! and extraction caches).

use crate::core::{GraphRAGError, Result};
use serde::{Deserialize, Serialize};

/// Catch-all type for entities and relations outside the ontology.
pub const OTHER: &str = "OTHER";

/// One entity or relation type with its accepted aliases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeDef {
    /// Canonical name (e.g. `PERSON`, `WORKS_FOR`).
    pub name: String,
    /// One-line definition shown to the LLM.
    #[serde(default)]
    pub description: String,
    /// Alternative labels mapped onto `name` (case and separators are ignored).
    #[serde(default)]
    pub aliases: Vec<String>,
}

impl TypeDef {
    fn new(name: &str, description: &str, aliases: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
        }
    }

    fn matches(&self, key: &str) -> bool {
        normalize_key(&self.name) == key || self.aliases.iter().any(|a| normalize_key(a) == key)
    }
}

/// Closed set of entity and relation types shared by all extractors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ontology {
    /// Ontology name.
    pub name: String,
    /// Ontology version (bump on any semantic change).
    pub version: String,
    /// Allowed entity types (`OTHER` is implicit).
    pub entity_types: Vec<TypeDef>,
    /// Allowed relation types (`OTHER` is implicit).
    pub relation_types: Vec<TypeDef>,
}

impl Default for Ontology {
    fn default() -> Self {
        Self::generic()
    }
}

impl Ontology {
    /// Built-in domain-agnostic ontology (`generic@1.0.0`).
    pub fn generic() -> Self {
        let e = TypeDef::new;
        let entity_types = vec![
            e(
                "PERSON",
                "a real or fictional individual",
                &[
                    "PER",
                    "PEOPLE",
                    "CHARACTER",
                    "HUMAN",
                    "PERSONA",
                    "PERSONAGGIO",
                ],
            ),
            e(
                "ORGANIZATION",
                "company, institution, group or political body",
                &[
                    "ORG",
                    "COMPANY",
                    "INSTITUTION",
                    "GROUP",
                    "ORGANIZZAZIONE",
                    "SOCIETA",
                    "AZIENDA",
                ],
            ),
            e(
                "LOCATION",
                "geographic or physical place",
                &[
                    "LOC", "PLACE", "GPE", "CITY", "COUNTRY", "REGION", "LUOGO", "CITTA",
                ],
            ),
            e(
                "EVENT",
                "something that happens at a time: battle, meeting, ceremony",
                &["EVENTO", "INCIDENT"],
            ),
            e(
                "CONCEPT",
                "abstract idea, theory, topic or value",
                &["IDEA", "TOPIC", "THEORY", "CONCETTO", "TEMA"],
            ),
            e(
                "WORK",
                "creative or intellectual work: book, poem, artwork, law",
                &["BOOK", "ARTWORK", "CREATIVE_WORK", "OPERA", "LIBRO"],
            ),
            e(
                "OBJECT",
                "physical artifact or product",
                &["ARTIFACT", "PRODUCT", "ITEM", "OGGETTO"],
            ),
            e(
                "DATE",
                "date, time or period",
                &["TIME", "PERIOD", "DATA", "PERIODO"],
            ),
        ];
        let r = TypeDef::new;
        let relation_types = vec![
            r(
                "FAMILY_OF",
                "kinship or marriage",
                &[
                    "PARENT_OF",
                    "CHILD_OF",
                    "SIBLING_OF",
                    "MARRIED_TO",
                    "SPOUSE_OF",
                    "RELATIVE_OF",
                    "BROTHER_OF",
                    "SISTER_OF",
                ],
            ),
            r(
                "FRIEND_OF",
                "friendship, acquaintance or alliance between people or groups",
                &["FRIENDS_WITH", "KNOWS", "ALLY_OF", "COMPANION_OF"],
            ),
            r(
                "LOVES",
                "romantic love or strong affection",
                &["IN_LOVE_WITH", "LOVER_OF", "ADMIRES"],
            ),
            r(
                "OPPOSES",
                "conflict, rivalry, hostility or disagreement",
                &[
                    "ENEMY_OF",
                    "RIVAL_OF",
                    "CONFLICTS_WITH",
                    "FIGHTS",
                    "DISAGREES_WITH",
                ],
            ),
            r(
                "WORKS_FOR",
                "employed by or serving an organization or person",
                &["EMPLOYED_BY", "WORKS_AT", "SERVES"],
            ),
            r(
                "LEADS",
                "heads, directs, manages or rules an organization, group or place",
                &[
                    "HEADS",
                    "DIRECTS",
                    "MANAGES",
                    "RULES",
                    "CEO_OF",
                    "LEADER_OF",
                ],
            ),
            r(
                "MEMBER_OF",
                "belongs to an organization or group",
                &["BELONGS_TO", "AFFILIATED_WITH"],
            ),
            r(
                "FOUNDED",
                "established an organization or place",
                &["ESTABLISHED", "FOUNDER_OF"],
            ),
            r(
                "CREATED",
                "authored, wrote, made or invented a work or object",
                &[
                    "AUTHOR_OF",
                    "WROTE",
                    "AUTHORED",
                    "PRODUCED",
                    "INVENTED",
                    "CREATOR_OF",
                ],
            ),
            r(
                "LOCATED_IN",
                "is situated in, lives in, is based in or was born in a place",
                &[
                    "LIVES_IN",
                    "BASED_IN",
                    "SITUATED_IN",
                    "BORN_IN",
                    "RESIDES_IN",
                ],
            ),
            r(
                "PART_OF",
                "is a component, subdivision or subset of",
                &["COMPONENT_OF", "SUBSET_OF", "DIVISION_OF"],
            ),
            r(
                "PARTICIPATED_IN",
                "took part in or attended an event",
                &["ATTENDED", "TOOK_PART_IN", "INVOLVED_IN"],
            ),
            r(
                "OCCURRED_AT",
                "an event happened at a place or time",
                &["HAPPENED_IN", "TOOK_PLACE_IN", "OCCURRED_IN", "HAPPENED_AT"],
            ),
            r(
                "CAUSES",
                "causes, leads to or results in",
                &["CAUSED", "LEADS_TO", "RESULTS_IN"],
            ),
            r(
                "INFLUENCES",
                "influences, inspires, teaches or mentors",
                &["INSPIRED", "AFFECTS", "TAUGHT", "MENTOR_OF", "TEACHES"],
            ),
            r(
                "DISCUSSES",
                "talks about, describes, praises or refers to a topic, work or entity",
                &[
                    "ABOUT",
                    "DESCRIBES",
                    "MENTIONS",
                    "PRAISES",
                    "REFERS_TO",
                    "SPEAKS_ABOUT",
                ],
            ),
            r("OWNS", "owns or possesses", &["POSSESSES", "OWNER_OF"]),
        ];
        Self {
            name: "generic".to_string(),
            version: "1.0.0".to_string(),
            entity_types,
            relation_types,
        }
    }

    /// Load an ontology from a TOML or JSON file (chosen by extension).
    pub fn from_path(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let parsed: std::result::Result<Self, String> = if path.ends_with(".json") {
            serde_json::from_str(&content).map_err(|e| e.to_string())
        } else {
            toml::from_str(&content).map_err(|e| e.to_string())
        };
        let ontology = parsed.map_err(|e| GraphRAGError::Config {
            message: format!("invalid ontology file {path}: {e}"),
        })?;
        ontology.validate()?;
        Ok(ontology)
    }

    /// Restrict the entity types to `types`, keeping built-in aliases.
    ///
    /// A configured name that matches a built-in type or alias (e.g. `ORG`)
    /// replaces that type's canonical name; the old name becomes an alias.
    /// Unknown names are added as new types without aliases.
    pub fn with_entity_types(mut self, types: &[String]) -> Self {
        if types.is_empty() {
            return self;
        }
        let entity_types = types
            .iter()
            .map(|configured| {
                let key = normalize_key(configured);
                match self.entity_types.iter().find(|t| t.matches(&key)) {
                    Some(def) => {
                        let mut def = def.clone();
                        if normalize_key(&def.name) != key {
                            def.aliases.push(def.name.clone());
                        }
                        def.name = configured.to_uppercase();
                        def
                    },
                    None => TypeDef::new(&configured.to_uppercase(), "", &[]),
                }
            })
            .collect();
        self.entity_types = entity_types;
        self
    }

    /// Check names are non-empty and unique and do not shadow `OTHER`.
    pub fn validate(&self) -> Result<()> {
        for (kind, defs) in [
            ("entity", &self.entity_types),
            ("relation", &self.relation_types),
        ] {
            let mut seen = std::collections::HashSet::new();
            for def in defs {
                let key = normalize_key(&def.name);
                if key.is_empty() || key == OTHER {
                    return Err(GraphRAGError::Config {
                        message: format!(
                            "ontology {kind} type name {:?} is empty or reserved",
                            def.name
                        ),
                    });
                }
                for label in std::iter::once(&def.name).chain(&def.aliases) {
                    if !seen.insert(normalize_key(label)) {
                        return Err(GraphRAGError::Config {
                            message: format!("ontology {kind} label {label:?} is defined twice"),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Stable identifier: `name@version#fingerprint` over the full content.
    pub fn id(&self) -> String {
        let content = serde_json::to_string(self).unwrap_or_default();
        format!(
            "{}@{}#{:016x}",
            self.name,
            self.version,
            fnv1a64(content.as_bytes())
        )
    }

    /// Canonical entity type names, without `OTHER`.
    pub fn entity_type_names(&self) -> Vec<String> {
        self.entity_types.iter().map(|t| t.name.clone()).collect()
    }

    /// Canonical relation type names, without `OTHER`.
    pub fn relation_type_names(&self) -> Vec<String> {
        self.relation_types.iter().map(|t| t.name.clone()).collect()
    }

    /// Map a raw entity label onto its canonical type, or `OTHER`.
    pub fn normalize_entity_type(&self, raw: &str) -> String {
        let key = normalize_key(raw);
        self.entity_types
            .iter()
            .find(|t| t.matches(&key))
            .map_or_else(|| OTHER.to_string(), |t| t.name.clone())
    }

    /// Map a raw relation label onto `(canonical type, description)`.
    ///
    /// When `raw_type` is empty the description is tried as a label (older
    /// prompts returned the relation label in `description`). For `OTHER`
    /// the original label is kept in front of the description, so no
    /// information is lost.
    pub fn normalize_relation(
        &self,
        raw_type: &str,
        description: &str,
    ) -> (String, Option<String>) {
        let raw_type = raw_type.trim();
        let description = description.trim();
        let label = if raw_type.is_empty() {
            description
        } else {
            raw_type
        };
        let key = normalize_key(label);
        if let Some(def) = self.relation_types.iter().find(|t| t.matches(&key)) {
            let kept = (!description.is_empty() && normalize_key(description) != key)
                .then(|| description.to_string());
            return (def.name.clone(), kept);
        }
        let kept = match (key.as_str(), description.is_empty()) {
            ("", true) => None,
            ("", false) | (OTHER, false) => Some(description.to_string()),
            (OTHER, true) => None,
            (_, true) => Some(label.to_string()),
            (_, false) if normalize_key(description) == key => Some(description.to_string()),
            (_, false) => Some(format!("{raw_type}: {description}")),
        };
        (OTHER.to_string(), kept)
    }

    /// Prompt block listing entity and relation types with definitions.
    pub fn prompt_section(&self) -> String {
        format!(
            "Entity types (use OTHER only if none fits):\n{}\n- OTHER: none of the above\n\n{}",
            Self::type_list(&self.entity_types),
            self.relation_prompt_section()
        )
    }

    /// Prompt block listing only the relation types (relation-only extraction).
    pub fn relation_prompt_section(&self) -> String {
        format!(
            "Relationship types (pick the most specific; OTHER only if none fits, then explain it in description):\n{}\n- OTHER: none of the above",
            Self::type_list(&self.relation_types)
        )
    }

    fn type_list(defs: &[TypeDef]) -> String {
        defs.iter()
            .map(|t| {
                if t.description.is_empty() {
                    format!("- {}", t.name)
                } else {
                    format!("- {}: {}", t.name, t.description)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// JSON schema for relation-only extraction over a fixed entity list:
    /// `source`/`target` are an enum of `entity_names` (deduplicated, order
    /// kept) and `type` an enum of the relation types plus `OTHER`. Property
    /// order matters for the llama.cpp grammar (`description` before `type`).
    pub fn relation_extraction_json_schema(&self, entity_names: &[String]) -> serde_json::Value {
        let mut names: Vec<&String> = Vec::with_capacity(entity_names.len());
        for name in entity_names {
            if !names.contains(&name) {
                names.push(name);
            }
        }
        let mut types = self.relation_type_names();
        types.push(OTHER.to_string());
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "relationships": {"type": "array", "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "source": {"type": "string", "enum": names},
                        "target": {"type": "string", "enum": names},
                        "description": {"type": "string"},
                        "type": {"type": "string", "enum": types},
                        "strength": {"type": "number"}
                    },
                    "required": ["source", "target", "description", "type", "strength"]
                }}
            },
            "required": ["relationships"]
        })
    }

    /// JSON schema for the extraction output, constraining `type` fields to
    /// the ontology. `description` precedes `type` so the model states the
    /// relation before classifying it.
    pub fn extraction_json_schema(&self) -> serde_json::Value {
        let with_other = |mut names: Vec<String>| {
            names.push(OTHER.to_string());
            names
        };
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "entities": {"type": "array", "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": with_other(self.entity_type_names())},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "description"]
                }},
                "relationships": {"type": "array", "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "description": {"type": "string"},
                        "type": {"type": "string", "enum": with_other(self.relation_type_names())},
                        "strength": {"type": "number"}
                    },
                    "required": ["source", "target", "description", "type", "strength"]
                }}
            },
            "required": ["entities", "relationships"]
        })
    }
}

/// Add the epistemic fields (`entity_extraction.epistemic_status`) to the relationship
/// objects of an extraction schema: `status` (`"narrated"` | `"said"`) and `said_by`,
/// appended after the existing properties (grammar order matters) and required like them.
pub fn with_epistemic_status(mut schema: serde_json::Value) -> serde_json::Value {
    if let Some(item) = schema.pointer_mut("/properties/relationships/items") {
        if let Some(props) = item.get_mut("properties").and_then(|p| p.as_object_mut()) {
            props.insert(
                "status".to_string(),
                serde_json::json!({"type": "string", "enum": ["narrated", "said"]}),
            );
            props.insert("said_by".to_string(), serde_json::json!({"type": "string"}));
        }
        if let Some(required) = item.get_mut("required").and_then(|r| r.as_array_mut()) {
            required.push("status".into());
            required.push("said_by".into());
        }
    }
    schema
}

/// Uppercase, ASCII-fold common Italian accents, and collapse every
/// non-alphanumeric run into a single `_`.
fn normalize_key(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        let ch = match ch {
            'à' | 'á' | 'À' | 'Á' => 'A',
            'è' | 'é' | 'È' | 'É' => 'E',
            'ì' | 'í' | 'Ì' | 'Í' => 'I',
            'ò' | 'ó' | 'Ò' | 'Ó' => 'O',
            'ù' | 'ú' | 'Ù' | 'Ú' => 'U',
            c => c,
        };
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_uppercase());
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_is_valid_and_id_is_stable() {
        let o = Ontology::generic();
        o.validate().unwrap();
        assert_eq!(o.id(), Ontology::generic().id());
        assert!(o.id().starts_with("generic@1.0.0#"));
        let changed = o.clone().with_entity_types(&["PERSON".to_string()]);
        assert_ne!(changed.id(), o.id());
    }

    #[test]
    fn entity_types_normalize_through_aliases() {
        let o = Ontology::generic();
        assert_eq!(o.normalize_entity_type("org"), "ORGANIZATION");
        assert_eq!(o.normalize_entity_type("Personaggio"), "PERSON");
        assert_eq!(o.normalize_entity_type("città"), "LOCATION");
        assert_eq!(o.normalize_entity_type("spaceship"), OTHER);
    }

    #[test]
    fn configured_entity_names_win_over_canonical() {
        let o = Ontology::generic().with_entity_types(&[
            "PERSON".into(),
            "ORG".into(),
            "CLAUSE".into(),
        ]);
        assert_eq!(o.entity_type_names(), vec!["PERSON", "ORG", "CLAUSE"]);
        assert_eq!(o.normalize_entity_type("organization"), "ORG");
        assert_eq!(o.normalize_entity_type("clause"), "CLAUSE");
        assert_eq!(o.normalize_entity_type("location"), OTHER);
    }

    #[test]
    fn relations_map_to_closed_list_or_other_with_description() {
        let o = Ontology::generic();
        assert_eq!(
            o.normalize_relation("works at", "Anna works at Acme"),
            ("WORKS_FOR".into(), Some("Anna works at Acme".into()))
        );
        assert_eq!(
            o.normalize_relation("", "FRIENDS_WITH"),
            ("FRIEND_OF".into(), None)
        );
        assert_eq!(
            o.normalize_relation("LEADS", "LEADS"),
            ("LEADS".into(), None)
        );
        assert_eq!(
            o.normalize_relation("OTHER", "Socrates questions Agathon"),
            (OTHER.into(), Some("Socrates questions Agathon".into()))
        );
        assert_eq!(
            o.normalize_relation("questions", "Socrates questions Agathon"),
            (
                OTHER.into(),
                Some("questions: Socrates questions Agathon".into())
            )
        );
        assert_eq!(
            o.normalize_relation("questions", ""),
            (OTHER.into(), Some("questions".into()))
        );
        assert_eq!(o.normalize_relation("", ""), (OTHER.into(), None));
    }

    #[test]
    fn schema_enums_include_other_and_order_description_first() {
        let schema = Ontology::generic().extraction_json_schema();
        let rel = &schema["properties"]["relationships"]["items"];
        let enums = rel["properties"]["type"]["enum"].as_array().unwrap();
        assert_eq!(enums.last().unwrap(), OTHER);
        let keys: Vec<_> = rel["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            keys,
            ["source", "target", "description", "type", "strength"]
        );
    }

    #[test]
    fn relation_schema_constrains_entities_and_keeps_property_order() {
        let names = vec![
            "Pinocchio".to_string(),
            "Geppetto".to_string(),
            "Pinocchio".to_string(),
        ];
        let schema = Ontology::generic().relation_extraction_json_schema(&names);
        assert_eq!(schema["required"], serde_json::json!(["relationships"]));
        assert!(schema["properties"].get("entities").is_none());
        let rel = &schema["properties"]["relationships"]["items"];
        let props = rel["properties"].as_object().unwrap();
        assert_eq!(
            props.keys().cloned().collect::<Vec<_>>(),
            ["source", "target", "description", "type", "strength"]
        );
        let expected = serde_json::json!(["Pinocchio", "Geppetto"]); // deduplicated, order kept
        assert_eq!(props["source"]["enum"], expected);
        assert_eq!(props["target"]["enum"], expected);
        let types = props["type"]["enum"].as_array().unwrap();
        assert!(types.contains(&"FRIEND_OF".into()));
        assert_eq!(types.last().unwrap(), OTHER);
    }

    #[test]
    fn relation_prompt_section_lists_relations_only() {
        let section = Ontology::generic().relation_prompt_section();
        assert!(section.contains("- FRIEND_OF:"));
        assert!(section.contains("- OTHER:"));
        assert!(!section.contains("- PERSON"));
    }

    #[test]
    fn validate_rejects_duplicates_and_reserved_names() {
        let mut o = Ontology::generic();
        o.relation_types.push(TypeDef::new("Works-For", "", &[]));
        assert!(o.validate().is_err());
        let mut o = Ontology::generic();
        o.entity_types.push(TypeDef::new("other", "", &[]));
        assert!(o.validate().is_err());
    }

    #[test]
    fn config_ontology_follows_active_extractor() {
        let mut config = crate::config::Config::default();
        config.entities.entity_types = vec!["PERSON".into(), "ORG".into()];
        assert_eq!(
            config.ontology().unwrap().entity_type_names(),
            ["PERSON", "ORG"]
        );

        config.gliner.enabled = true;
        config.gliner.entity_labels = vec!["person".into(), "event".into()];
        assert_eq!(
            config.ontology().unwrap().entity_type_names(),
            ["PERSON", "EVENT"]
        );

        config.llm.enabled = true; // GLiNER extraction wins over the LLM (extraction_path)
        assert_eq!(
            config.ontology().unwrap().entity_type_names(),
            ["PERSON", "EVENT"]
        );

        config.gliner.llm_relations = true; // cascade: GLiNER labels define the entity types
        assert_eq!(
            config.ontology().unwrap().entity_type_names(),
            ["PERSON", "EVENT"]
        );
        config.gliner.llm_relations = false;

        config.entities.ontology_path = Some("/nonexistent/ontology.toml".into());
        assert!(config.ontology().is_err());
    }

    #[test]
    fn loads_toml_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("o.toml");
        std::fs::write(
            &path,
            "name = \"legal\"\nversion = \"0.1.0\"\n\
             [[entity_types]]\nname = \"PARTY\"\naliases = [\"PARTE\"]\n\
             [[relation_types]]\nname = \"SIGNED\"\ndescription = \"signed a contract\"\n",
        )
        .unwrap();
        let o = Ontology::from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(o.normalize_entity_type("parte"), "PARTY");
        assert_eq!(o.normalize_relation("signed", "").0, "SIGNED");
        assert!(o.id().starts_with("legal@0.1.0#"));
    }

    #[test]
    fn epistemic_schema_appends_status_and_said_by_after_existing_properties() {
        let names = vec!["Pinocchio".to_string(), "Geppetto".to_string()];
        for schema in [
            with_epistemic_status(Ontology::generic().extraction_json_schema()),
            with_epistemic_status(Ontology::generic().relation_extraction_json_schema(&names)),
        ] {
            let rel = &schema["properties"]["relationships"]["items"];
            let keys: Vec<_> = rel["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            assert_eq!(
                keys,
                [
                    "source",
                    "target",
                    "description",
                    "type",
                    "strength",
                    "status",
                    "said_by"
                ]
            );
            assert_eq!(
                rel["properties"]["status"]["enum"],
                serde_json::json!(["narrated", "said"])
            );
            assert_eq!(rel["properties"]["said_by"]["type"], "string");
            let required = rel["required"].as_array().unwrap();
            assert_eq!(&required[required.len() - 2..], ["status", "said_by"]);
        }
    }
}
