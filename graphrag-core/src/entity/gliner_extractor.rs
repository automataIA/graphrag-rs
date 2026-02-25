//! GLiNER-Relex entity + relation extractor via ONNX Runtime.
//!
//! Uses `gline-rs` (v1.0.1) for joint NER + RE in a single forward pass.
//! ~1.5 GB VRAM vs 8+ GB for generative LLMs; no hallucinations on structure.
//!
//! Feature-gated: only compiled when `--features gliner` is active.

#![cfg(feature = "gliner")]

use composable::Composable;
use gliner::model::{
    input::{relation::schema::RelationSchema, text::TextInput},
    params::Parameters,
    pipeline::{relation::RelationPipeline, span::SpanPipeline, token::TokenPipeline},
};
use orp::{model::Model, params::RuntimeParameters, pipeline::Pipeline};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::{
    config::GlinerConfig,
    core::{
        error::GraphRAGError, Entity, EntityId, EntityMention, Relationship, TextChunk,
    },
};

/// Joint NER + RE extractor backed by GLiNER-Relex via ONNX Runtime.
///
/// The model is loaded lazily on the first call to [`extract_from_chunk`];
/// the constructor only validates that the files exist (fail-fast).
///
/// `GLiNERExtractor` is `Send + Sync` and can be safely moved into
/// `tokio::task::spawn_blocking` closures.
pub struct GLiNERExtractor {
    config: GlinerConfig,
    /// Lazy-loaded ONNX model.  `None` until the first extraction call.
    model: Arc<RwLock<Option<Model>>>,
}

impl std::fmt::Debug for GLiNERExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLiNERExtractor")
            .field("config", &self.config)
            .field("model_loaded", &self.model.read().is_some())
            .finish()
    }
}

impl GLiNERExtractor {
    /// Create a new extractor, validating that model and tokenizer files exist.
    pub fn new(config: GlinerConfig) -> Result<Self, GraphRAGError> {
        if !std::path::Path::new(&config.model_path).exists() {
            return Err(GraphRAGError::Config {
                message: format!("GLiNER model not found: {}", config.model_path),
            });
        }
        let tokenizer = Self::resolve_tokenizer_path(&config);
        if !std::path::Path::new(&tokenizer).exists() {
            return Err(GraphRAGError::Config {
                message: format!("GLiNER tokenizer not found: {}", tokenizer),
            });
        }
        Ok(Self {
            config,
            model: Arc::new(RwLock::new(None)),
        })
    }

    /// Resolve the tokenizer path: use `config.tokenizer_path` if set,
    /// otherwise default to `tokenizer.json` in the same directory as the model.
    fn resolve_tokenizer_path(config: &GlinerConfig) -> String {
        if !config.tokenizer_path.is_empty() {
            return config.tokenizer_path.clone();
        }
        std::path::Path::new(&config.model_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("tokenizer.json")
            .to_string_lossy()
            .to_string()
    }

    /// Ensure the ONNX model is loaded, loading it if necessary.
    fn ensure_model_loaded(&self) -> Result<(), GraphRAGError> {
        let mut guard = self.model.write();
        if guard.is_none() {
            #[allow(unused_mut)]
            let mut rt_params = RuntimeParameters::default();
            if self.config.use_gpu {
                #[cfg(feature = "cuda")]
                {
                    use ort::execution_providers::CUDAExecutionProvider;
                    rt_params = rt_params
                        .with_execution_providers([CUDAExecutionProvider::default().build()]);
                }
            }
            let model = Model::new(&self.config.model_path, rt_params).map_err(|e| {
                GraphRAGError::EntityExtraction {
                    message: format!("Failed to load GLiNER model: {e}"),
                }
            })?;
            *guard = Some(model);
        }
        Ok(())
    }

    /// Perform joint NER + RE on a single text chunk (synchronous / blocking).
    ///
    /// In async contexts, wrap with `tokio::task::spawn_blocking`:
    /// ```ignore
    /// let (ents, rels) = tokio::task::spawn_blocking({
    ///     let ext = extractor.clone();
    ///     let ch  = chunk.clone();
    ///     move || ext.extract_from_chunk(&ch)
    /// }).await??;
    /// ```
    pub fn extract_from_chunk(
        &self,
        chunk: &TextChunk,
    ) -> Result<(Vec<Entity>, Vec<Relationship>), GraphRAGError> {
        self.ensure_model_loaded()?;

        let guard = self.model.read();
        let model = guard.as_ref().expect("model loaded");
        let tokenizer = Self::resolve_tokenizer_path(&self.config);
        let params = Parameters::default();

        let entity_refs: Vec<&str> =
            self.config.entity_labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&[chunk.content.as_str()], &entity_refs).map_err(|e| {
            GraphRAGError::EntityExtraction {
                message: format!("GLiNER TextInput error: {e}"),
            }
        })?;

        // ── Stage 1: NER ──────────────────────────────────────────────────────
        let span_output = match self.config.mode.to_lowercase().as_str() {
            "token" => TokenPipeline::new(&tokenizer)
                .map_err(|e| GraphRAGError::EntityExtraction {
                    message: format!("GLiNER TokenPipeline error: {e}"),
                })?
                .to_composable(model, &params)
                .apply(input)
                .map_err(|e| GraphRAGError::EntityExtraction {
                    message: format!("GLiNER token inference error: {e}"),
                })?,
            _ => SpanPipeline::new(&tokenizer)
                .map_err(|e| GraphRAGError::EntityExtraction {
                    message: format!("GLiNER SpanPipeline error: {e}"),
                })?
                .to_composable(model, &params)
                .apply(input)
                .map_err(|e| GraphRAGError::EntityExtraction {
                    message: format!("GLiNER span inference error: {e}"),
                })?,
        };

        // Convert spans → Entity (dedup by (text, class))
        let mut entities: Vec<Entity> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        if let Some(seq) = span_output.spans.first() {
            for span in seq {
                if span.probability() < self.config.entity_threshold {
                    continue;
                }
                let key = (span.text().to_string(), span.class().to_string());
                if !seen.insert(key) {
                    continue;
                }
                let entity_id = Self::make_entity_id(span.class(), span.text());
                entities.push(
                    Entity::new(
                        entity_id,
                        span.text().to_string(),
                        span.class().to_string(),
                        span.probability(),
                    )
                    .with_mentions(vec![EntityMention {
                        chunk_id: chunk.id.clone(),
                        // gline-rs does not expose character offsets in its current
                        // public API; set to 0 for now (TODO: update when available).
                        start_offset: 0,
                        end_offset: 0,
                        confidence: span.probability(),
                    }]),
                );
            }
        }

        // ── Stage 2: RE (optional) ────────────────────────────────────────────
        let mut relationships: Vec<Relationship> = Vec::new();
        if !self.config.relation_labels.is_empty() {
            let mut schema = RelationSchema::new();
            for rel in &self.config.relation_labels {
                schema.push(rel.as_str());
            }

            let rel_output = RelationPipeline::default(&tokenizer, &schema)
                .map_err(|e| GraphRAGError::EntityExtraction {
                    message: format!("GLiNER RelationPipeline error: {e}"),
                })?
                .to_composable(model, &params)
                .apply(span_output)
                .map_err(|e| GraphRAGError::EntityExtraction {
                    message: format!("GLiNER relation inference error: {e}"),
                })?;

            if let Some(seq) = rel_output.relations.first() {
                for rel in seq {
                    if rel.probability() < self.config.relation_threshold {
                        continue;
                    }
                    let src = Self::find_entity_id(&entities, rel.subject());
                    let tgt = Self::find_entity_id(&entities, rel.object());
                    if let (Some(src_id), Some(tgt_id)) = (src, tgt) {
                        if src_id != tgt_id {
                            relationships.push(Relationship::new(
                                src_id,
                                tgt_id,
                                rel.class().to_string(),
                                rel.probability(),
                            ));
                            // Add chunk context to last inserted relationship
                            if let Some(r) = relationships.last_mut() {
                                r.context.push(chunk.id.clone());
                            }
                        }
                    }
                }
            }
        }

        Ok((entities, relationships))
    }

    /// Build a deterministic entity ID from type and name.
    fn make_entity_id(entity_type: &str, name: &str) -> EntityId {
        let normalized = name.to_lowercase().replace(' ', "_");
        EntityId::new(format!("{}_{}", entity_type.to_lowercase(), normalized))
    }

    /// Find an entity by exact name match and return its ID.
    fn find_entity_id(entities: &[Entity], text: &str) -> Option<EntityId> {
        entities
            .iter()
            .find(|e| e.name == text)
            .map(|e| e.id.clone())
    }
}

// ---------------------------------------------------------------------------
// Tests (do not require a real ONNX model)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GlinerConfig;

    #[test]
    fn test_normalize_entity_id() {
        let id = GLiNERExtractor::make_entity_id("PERSON", "John Doe");
        assert_eq!(id.0, "person_john_doe");
    }

    #[test]
    fn test_config_defaults() {
        let cfg = GlinerConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.entity_threshold, 0.4);
        assert_eq!(cfg.mode, "span");
    }

    #[test]
    fn test_extractor_new_missing_model() {
        let cfg = GlinerConfig {
            enabled: true,
            model_path: "/nonexistent/model.onnx".to_string(),
            ..GlinerConfig::default()
        };
        let result = GLiNERExtractor::new(cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found"), "unexpected error: {msg}");
    }

    #[test]
    fn test_resolve_tokenizer_default() {
        let cfg = GlinerConfig {
            model_path: "/models/gliner/model.onnx".to_string(),
            tokenizer_path: String::new(),
            ..GlinerConfig::default()
        };
        let tok = GLiNERExtractor::resolve_tokenizer_path(&cfg);
        assert!(tok.ends_with("tokenizer.json"));
        assert!(tok.contains("/models/gliner/"));
    }

    #[test]
    fn test_resolve_tokenizer_explicit() {
        let cfg = GlinerConfig {
            model_path: "/models/gliner/model.onnx".to_string(),
            tokenizer_path: "/custom/tok.json".to_string(),
            ..GlinerConfig::default()
        };
        let tok = GLiNERExtractor::resolve_tokenizer_path(&cfg);
        assert_eq!(tok, "/custom/tok.json");
    }
}
