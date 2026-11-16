//! GLiNER2.5 entity + relation extractor (pure-Rust candle, no ONNX Runtime).
//!
//! Uses `gliner2-candle` (`fastino/gliner2.5-multi-v1`): one schema (entity labels +
//! relation labels) per chunk, one forward pass. The raw output is noisy (duplicates,
//! self-loops, low-confidence relations, places as tails of person relations), so it is
//! cleaned by the pure function [`postprocess`] before being converted to graph types.
//!
//! Feature-gated: only compiled when `--features gliner2` is active.
//!
//! **Model loading.** `GLiNER2::from_pretrained(local_dir)` in gliner2-candle 0.3.0 passes
//! the directory (not `model.safetensors`) to the weight loader and ignores
//! `encoder_config/config.json`. We therefore replicate its remote-model setup by hand with
//! the public low-level API: `ExtractorConfig` (+ architecture and boundary config from config.json,
//! encoder dimensions), `GLiNER2::new`, then `GLiNER2::load_weights(<dir>/model.safetensors)`.
//! `GlinerConfig.model_path` is the model *directory* (`config.json`,
//! `encoder_config/config.json`, `model.safetensors`, `tokenizer.json`).

#![cfg(feature = "gliner2")]

use gliner2_candle::config::{Architecture, BoundaryConfig, ExtractorConfig};
use gliner2_candle::schema::types::{EntityDef, RelationDef, Schema};
use gliner2_candle::GLiNER2;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use crate::{
    config::GlinerConfig,
    core::{error::GraphRAGError, Entity, EntityId, EntityMention, Relationship, TextChunk},
};

/// Entity as returned by the model, before graph conversion.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawEntity {
    pub text: String,
    pub label: String,
    pub confidence: f32,
}

/// Relation as returned by the model (`head --rel--> tail`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawRelation {
    pub head: String,
    pub rel: String,
    pub tail: String,
    pub confidence: f32,
}

/// Lowercase + collapse whitespace: the identity of a mention inside a chunk.
fn norm(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Entity labels that denote places (matched on the raw, lowercased GLiNER label).
fn is_place_label(label: &str) -> bool {
    matches!(
        label.to_lowercase().as_str(),
        "place" | "location" | "loc" | "city" | "country" | "gpe" | "setting"
    )
}

/// Relations that legitimately point at a place.
fn is_location_relation(rel: &str) -> bool {
    let r = rel.to_lowercase();
    r.contains("located") || r.contains("lives_in")
}

/// Clean the raw model output of one chunk (pure, order-preserving).
///
/// Entities: drop `confidence < entity_threshold`; dedup by (normalized text, label)
/// keeping the highest confidence.
///
/// Relations, in order: drop `confidence < relation_threshold`; drop self-loops
/// (normalized head == tail); drop relations whose head or tail is not among the
/// surviving entities; type constraint (a tail that is only ever extracted as a place
/// is allowed only for location relations, i.e. label containing `located` or
/// `lives_in`); dedup (head, rel, tail) keeping the highest confidence.
pub(crate) fn postprocess(
    entities: Vec<RawEntity>,
    relations: Vec<RawRelation>,
    entity_threshold: f32,
    relation_threshold: f32,
) -> (Vec<RawEntity>, Vec<RawRelation>) {
    let mut ents: Vec<RawEntity> = Vec::new();
    let mut index: HashMap<(String, String), usize> = HashMap::new();
    for e in entities {
        if e.confidence < entity_threshold {
            continue;
        }
        let key = (norm(&e.text), e.label.to_lowercase());
        match index.get(&key) {
            Some(&i) => {
                if e.confidence > ents[i].confidence {
                    ents[i].confidence = e.confidence;
                }
            },
            None => {
                index.insert(key, ents.len());
                ents.push(e);
            },
        }
    }

    // text -> "is only ever a place"
    let mut only_place: HashMap<String, bool> = HashMap::new();
    for e in &ents {
        let p = only_place.entry(norm(&e.text)).or_insert(true);
        *p &= is_place_label(&e.label);
    }

    let mut rels: Vec<RawRelation> = Vec::new();
    let mut seen: HashMap<(String, String, String), usize> = HashMap::new();
    for r in relations {
        if r.confidence < relation_threshold {
            continue;
        }
        let (h, t) = (norm(&r.head), norm(&r.tail));
        if h == t || !only_place.contains_key(&h) {
            continue;
        }
        match only_place.get(&t) {
            None => continue,
            Some(true) if !is_location_relation(&r.rel) => continue,
            _ => {},
        }
        let key = (h, r.rel.to_lowercase(), t);
        match seen.get(&key) {
            Some(&i) => {
                if r.confidence > rels[i].confidence {
                    rels[i].confidence = r.confidence;
                }
            },
            None => {
                seen.insert(key, rels.len());
                rels.push(r);
            },
        }
    }
    (ents, rels)
}

fn confidence_of(v: &Value) -> f32 {
    v.get("confidence").and_then(Value::as_f64).unwrap_or(1.0) as f32
}

/// Text of a mention that is either `{"text": ..}` or a bare string.
fn text_of(v: &Value) -> Option<String> {
    v.get("text")
        .and_then(Value::as_str)
        .or_else(|| v.as_str())
        .map(str::to_string)
}

/// Parse the engine's JSON (`entities: {label: [{text, confidence}]}`,
/// `relation_extraction: {rel: [{head, tail}]}`). Relation confidence is the minimum of
/// the two endpoint confidences. Malformed items are skipped.
pub(crate) fn parse_output(output: &Value) -> (Vec<RawEntity>, Vec<RawRelation>) {
    let mut entities = Vec::new();
    if let Some(map) = output.get("entities").and_then(Value::as_object) {
        for (label, items) in map {
            for item in items.as_array().into_iter().flatten() {
                if let Some(text) = text_of(item) {
                    entities.push(RawEntity {
                        text,
                        label: label.clone(),
                        confidence: confidence_of(item),
                    });
                }
            }
        }
    }
    let mut relations = Vec::new();
    if let Some(map) = output.get("relation_extraction").and_then(Value::as_object) {
        for (rel, items) in map {
            for item in items.as_array().into_iter().flatten() {
                let (Some(h), Some(t)) = (item.get("head"), item.get("tail")) else {
                    continue;
                };
                let (Some(head), Some(tail)) = (text_of(h), text_of(t)) else {
                    continue;
                };
                relations.push(RawRelation {
                    head,
                    rel: rel.clone(),
                    tail,
                    confidence: confidence_of(h).min(confidence_of(t)),
                });
            }
        }
    }
    (entities, relations)
}

/// Convert cleaned raw output to graph types (ontology normalization, ids, provenance).
pub(crate) fn to_graph(
    chunk: &TextChunk,
    entities: &[RawEntity],
    relations: &[RawRelation],
    ontology: Option<&crate::ontology::Ontology>,
) -> (Vec<Entity>, Vec<Relationship>) {
    let mut out_entities: Vec<Entity> = Vec::new();
    let mut by_text: HashMap<String, usize> = HashMap::new();
    for e in entities {
        let class = match ontology {
            Some(o) => o.normalize_entity_type(&e.label),
            None => e.label.clone(),
        };
        let id = EntityId::from_type_and_name(&class, &e.text);
        if out_entities.iter().any(|x| x.id == id) {
            continue;
        }
        // Several labels for one text: relations attach to the most confident one.
        let key = norm(&e.text);
        match by_text.get(&key) {
            Some(&i) if out_entities[i].confidence >= e.confidence => {},
            _ => {
                by_text.insert(key, out_entities.len());
            },
        }
        out_entities.push(
            Entity::new(id, e.text.clone(), class, e.confidence).with_mentions(vec![
                EntityMention {
                    chunk_id: chunk.id.clone(),
                    start_offset: 0,
                    end_offset: 0,
                    confidence: e.confidence,
                },
            ]),
        );
    }

    let mut out_rels: Vec<Relationship> = Vec::new();
    for r in relations {
        let (Some(&h), Some(&t)) = (by_text.get(&norm(&r.head)), by_text.get(&norm(&r.tail)))
        else {
            continue;
        };
        let (src, tgt) = (out_entities[h].id.clone(), out_entities[t].id.clone());
        if src == tgt {
            continue;
        }
        let (relation_type, description) = match ontology {
            Some(o) => o.normalize_relation(&r.rel, ""),
            None => (r.rel.clone(), None),
        };
        let mut rel = Relationship::new(src, tgt, relation_type, r.confidence);
        rel.description = description;
        rel.context.push(chunk.id.clone());
        out_rels.push(rel);
    }
    (out_entities, out_rels)
}

/// GLiNER2.5 extractor. The model is loaded in [`Gliner2Extractor::new`] (fail fast);
/// inference is `&self`, so one instance is shared across `spawn_blocking` tasks.
pub struct Gliner2Extractor {
    config: GlinerConfig,
    engine: GLiNER2,
    schema: Schema,
    ontology: Option<crate::ontology::Ontology>,
}

impl std::fmt::Debug for Gliner2Extractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gliner2Extractor")
            .field("config", &self.config)
            .finish()
    }
}

fn load_error(msg: impl std::fmt::Display) -> GraphRAGError {
    GraphRAGError::Config {
        message: format!("GLiNER2 model load failed: {msg}"),
    }
}

/// Load a GLiNER2.5 model from a local directory (see module docs for why not
/// `GLiNER2::from_pretrained`).
fn load_engine(dir: &Path) -> Result<GLiNER2, GraphRAGError> {
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        return Err(load_error(format!("{} not found", weights.display())));
    }
    let mut config = ExtractorConfig::new(dir.to_string_lossy().to_string());
    let hf_config = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| load_error(format!("config.json: {e}")))?;
    config.architecture = Architecture::from_hf_config_json(&hf_config);
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);
    config.tokenizer_path = Some(dir.to_path_buf());

    // Encoder dimensions live in encoder_config/config.json.
    let enc = std::fs::read_to_string(dir.join("encoder_config").join("config.json"))
        .map_err(|e| load_error(format!("encoder_config/config.json: {e}")))?;
    let enc: Value = serde_json::from_str(&enc).map_err(|e| load_error(e.to_string()))?;
    let dim = |k: &str| enc.get(k).and_then(Value::as_u64).map(|v| v as usize);
    if let Some(v) = dim("vocab_size") {
        config.vocab_size = v;
    }
    if let Some(v) = dim("hidden_size") {
        config.hidden_size = v;
    }
    if let Some(v) = dim("num_hidden_layers") {
        config.num_hidden_layers = v;
    }
    if let Some(v) = dim("num_attention_heads") {
        config.num_attention_heads = v;
    }
    if let Some(v) = dim("intermediate_size") {
        config.intermediate_size = v;
    }

    let mut engine = GLiNER2::new(&config).map_err(load_error)?;
    engine.load_weights(&weights).map_err(load_error)?;
    Ok(engine)
}

impl Gliner2Extractor {
    /// Load the model from `config.model_path` (a directory) and build the schema from
    /// `entity_labels` / `relation_labels` (empty relation list = NER only).
    pub fn new(config: GlinerConfig) -> Result<Self, GraphRAGError> {
        let dir = Path::new(&config.model_path);
        if !dir.is_dir() {
            return Err(GraphRAGError::Config {
                message: format!(
                    "GLiNER2 model directory not found: {} (gliner.model_path must be a directory)",
                    config.model_path
                ),
            });
        }
        let engine = load_engine(dir)?;
        let mut schema = Schema::new().entities(
            config
                .entity_labels
                .iter()
                .map(|l| EntityDef::new(l.as_str()))
                .collect(),
        );
        if !config.relation_labels.is_empty() {
            schema = schema.relations(
                config
                    .relation_labels
                    .iter()
                    .map(|l| RelationDef::new(l.as_str()))
                    .collect(),
            );
        }
        Ok(Self {
            config,
            engine,
            schema,
            ontology: None,
        })
    }

    /// Map GLiNER2 labels onto `ontology` (same contract as the gline extractor).
    pub fn with_ontology(mut self, ontology: crate::ontology::Ontology) -> Self {
        self.ontology = Some(ontology);
        self
    }

    /// Extract entities and relationships from one chunk (blocking, CPU-heavy: call it
    /// inside `tokio::task::spawn_blocking`).
    pub fn extract_from_chunk(
        &self,
        chunk: &TextChunk,
    ) -> Result<(Vec<Entity>, Vec<Relationship>), GraphRAGError> {
        // The engine applies one threshold to entities and relations: use the lower, then
        // re-apply each threshold in `postprocess`.
        let engine_threshold = self
            .config
            .entity_threshold
            .min(self.config.relation_threshold);
        let output = self
            .engine
            .extract(
                &chunk.content,
                &self.schema,
                engine_threshold,
                true,
                false,
                None,
            )
            .map_err(|e| GraphRAGError::EntityExtraction {
                message: format!("GLiNER2 inference error: {e}"),
            })?;
        let (entities, relations) = parse_output(&output);
        let (entities, relations) = postprocess(
            entities,
            relations,
            self.config.entity_threshold,
            self.config.relation_threshold,
        );
        Ok(to_graph(
            chunk,
            &entities,
            &relations,
            self.ontology.as_ref(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId};

    fn ent(text: &str, label: &str, c: f32) -> RawEntity {
        RawEntity {
            text: text.into(),
            label: label.into(),
            confidence: c,
        }
    }
    fn rel(h: &str, r: &str, t: &str, c: f32) -> RawRelation {
        RawRelation {
            head: h.into(),
            rel: r.into(),
            tail: t.into(),
            confidence: c,
        }
    }
    fn chunk() -> TextChunk {
        TextChunk::new(
            ChunkId::new("c1".to_string()),
            DocumentId::new("d1".to_string()),
            "testo".to_string(),
            0,
            5,
        )
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GLiNER2>();
        assert_send_sync::<Gliner2Extractor>();
    }

    #[test]
    fn drops_low_confidence_relations() {
        let es = vec![
            ent("Geppetto", "character", 0.9),
            ent("Pinocchio", "character", 0.9),
        ];
        let rs = vec![
            rel("Geppetto", "father_of", "Pinocchio", 0.64),
            rel("Pinocchio", "friend_of", "Geppetto", 0.65),
        ];
        let (_, r) = postprocess(es, rs, 0.5, 0.65);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rel, "friend_of");
    }

    #[test]
    fn drops_low_confidence_entities() {
        let (e, _) = postprocess(vec![ent("A", "character", 0.4)], vec![], 0.5, 0.65);
        assert!(e.is_empty());
    }

    #[test]
    fn drops_self_loops_after_normalization() {
        let es = vec![ent("Pinocchio", "character", 0.9)];
        let rs = vec![rel("Pinocchio", "friend_of", " pinocchio ", 0.9)];
        assert!(postprocess(es, rs, 0.5, 0.5).1.is_empty());
    }

    #[test]
    fn dedups_triples_keeping_max_confidence() {
        let es = vec![ent("A", "character", 0.9), ent("B", "character", 0.9)];
        let rs = vec![
            rel("A", "meets", "B", 0.7),
            rel("a", "Meets", "b", 0.9),
            rel("A", "meets", "B", 0.8),
        ];
        let (_, r) = postprocess(es, rs, 0.5, 0.5);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].confidence, 0.9);
    }

    #[test]
    fn dedups_entities_keeping_max_confidence() {
        let es = vec![
            ent("Pinocchio", "character", 0.6),
            ent("pinocchio ", "character", 0.9),
            ent("Pinocchio", "place", 0.7),
        ];
        let (e, _) = postprocess(es, vec![], 0.5, 0.5);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].confidence, 0.9);
    }

    #[test]
    fn drops_relations_with_unknown_endpoints() {
        let es = vec![ent("A", "character", 0.9)];
        let rs = vec![rel("A", "meets", "Z", 0.9), rel("Z", "meets", "A", 0.9)];
        assert!(postprocess(es, rs, 0.5, 0.5).1.is_empty());
    }

    #[test]
    fn endpoint_below_entity_threshold_drops_relation() {
        let es = vec![ent("A", "character", 0.9), ent("B", "character", 0.3)];
        let rs = vec![rel("A", "meets", "B", 0.9)];
        assert!(postprocess(es, rs, 0.5, 0.5).1.is_empty());
    }

    #[test]
    fn place_tail_only_allowed_for_location_relations() {
        let es = vec![
            ent("Pinocchio", "character", 0.9),
            ent("Roma", "place", 0.9),
        ];
        let rs = vec![
            rel("Pinocchio", "friend_of", "Roma", 0.9),
            rel("Pinocchio", "located_in", "Roma", 0.9),
            rel("Pinocchio", "lives_in", "Roma", 0.9),
        ];
        let (_, r) = postprocess(es, rs, 0.5, 0.5);
        let labels: Vec<_> = r.iter().map(|r| r.rel.as_str()).collect();
        assert_eq!(labels, ["located_in", "lives_in"]);
    }

    #[test]
    fn text_that_is_also_a_character_is_not_a_place_tail() {
        let es = vec![
            ent("Fata", "character", 0.9),
            ent("Fata", "place", 0.9),
            ent("A", "character", 0.9),
        ];
        let rs = vec![rel("A", "friend_of", "Fata", 0.9)];
        assert_eq!(postprocess(es, rs, 0.5, 0.5).1.len(), 1);
    }

    #[test]
    fn parse_output_reads_engine_json() {
        let v: Value = serde_json::from_str(
            r#"{"entities":{"character":[{"text":"Geppetto","confidence":0.9}],"place":[]},
                "relation_extraction":{"father_of":[{"head":{"text":"Geppetto","confidence":0.8},
                "tail":{"text":"Pinocchio","confidence":0.7}}],"meets":[]}}"#,
        )
        .unwrap();
        let (e, r) = parse_output(&v);
        assert_eq!(e, vec![ent("Geppetto", "character", 0.9)]);
        assert_eq!(r, vec![rel("Geppetto", "father_of", "Pinocchio", 0.7)]);
        assert_eq!(parse_output(&Value::Null), (vec![], vec![]));
    }

    #[test]
    fn to_graph_builds_ids_mentions_and_provenance() {
        let es = vec![
            ent("Geppetto", "character", 0.9),
            ent("Pinocchio", "character", 0.8),
        ];
        let rs = vec![rel("Geppetto", "father_of", "Pinocchio", 0.7)];
        let (e, r) = to_graph(&chunk(), &es, &rs, None);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].id.0, "character_geppetto");
        assert_eq!(e[0].mentions[0].chunk_id, chunk().id);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].source.0, "character_geppetto");
        assert_eq!(r[0].target.0, "character_pinocchio");
        assert_eq!(r[0].context, vec![chunk().id]);
    }

    #[test]
    fn to_graph_normalizes_onto_ontology() {
        let o = crate::ontology::Ontology::generic();
        let es = vec![ent("Geppetto", "character", 0.9), ent("Roma", "place", 0.9)];
        let rs = vec![rel("Geppetto", "located_in", "Roma", 0.9)];
        let (e, r) = to_graph(&chunk(), &es, &rs, Some(&o));
        assert_eq!(e[0].entity_type, o.normalize_entity_type("character"));
        assert_ne!(e[0].entity_type, "character");
        assert_eq!(r[0].relation_type, o.normalize_relation("located_in", "").0);
    }

    #[test]
    fn missing_model_dir_is_a_config_error() {
        let cfg = GlinerConfig {
            model_path: "/nonexistent/gliner2".into(),
            ..GlinerConfig::default()
        };
        let msg = Gliner2Extractor::new(cfg).unwrap_err().to_string();
        assert!(msg.contains("not found"), "unexpected: {msg}");
    }
}
