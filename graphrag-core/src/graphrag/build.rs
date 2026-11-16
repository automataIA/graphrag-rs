#![allow(unused_imports)]

use crate::config::Config;
use crate::core::{
    is_said_description, ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError,
    KnowledgeGraph, Relationship, Result, TextChunk,
};
use crate::{critic, llm, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

use super::GraphRAG;

/// Entity extraction pipeline chosen by `build_graph` from the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "async"), allow(dead_code))]
pub(super) enum ExtractionPath {
    /// LLM extraction with gleaning rounds
    Gleaning,
    /// One LLM call per chunk for entities and relationships
    LlmSinglePass,
    /// GLiNER finds the entities, the LLM extracts only the relationships among them
    GlinerCascade,
    /// GLiNER (optionally with its own relation stage)
    Gliner,
    /// Regex / capitalization extraction
    Pattern,
}

/// GLiNER inference backend (`gliner.backend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(feature = "gliner", feature = "gliner2")), allow(dead_code))]
pub(super) enum GlinerBackend {
    /// ONNX Runtime via gline-rs (feature `gliner`)
    Gline,
    /// GLiNER2.5 via candle (feature `gliner2`)
    Gliner2,
}

/// Whether a chunk with these first-pass counts looks under-extracted.
#[cfg_attr(not(feature = "async"), allow(dead_code))]
fn needs_relation_pass(
    entities: usize,
    relationships: usize,
    min_entities: usize,
    max_relationships: usize,
) -> bool {
    entities >= min_entities && relationships <= max_relationships
}

/// Append the relationships of `extra` that `existing` does not already hold (same source,
/// target and relation type); returns how many were added. On a duplicate, a narrated
/// description replaces a "said" one (same rule as `KnowledgeGraph::add_relationship`).
#[cfg_attr(not(feature = "async"), allow(dead_code))]
fn merge_new_relationships(existing: &mut Vec<Relationship>, extra: Vec<Relationship>) -> usize {
    let mut added = 0;
    for rel in extra {
        let duplicate = existing.iter_mut().find(|r| {
            r.source == rel.source && r.target == rel.target && r.relation_type == rel.relation_type
        });
        match duplicate {
            None => {
                existing.push(rel);
                added += 1;
            },
            Some(dup) => {
                if is_said_description(dup.description.as_deref())
                    && rel
                        .description
                        .as_deref()
                        .is_some_and(|d| !d.trim().is_empty())
                    && !is_said_description(rel.description.as_deref())
                {
                    dup.description = rel.description;
                }
            },
        }
    }
    added
}

impl GraphRAG {
    /// Output cap for extraction calls: `entities.extraction_max_tokens`, else the answer
    /// setting `llm.max_tokens` (1500 when unset).
    #[cfg_attr(not(feature = "async"), allow(dead_code))]
    fn extraction_max_tokens(config: &Config) -> usize {
        config
            .entities
            .extraction_max_tokens
            .unwrap_or_else(|| config.llm.max_tokens.unwrap_or(1500) as usize)
    }

    /// Pick the extraction pipeline. GLiNER (`gliner.enabled`) takes precedence over the LLM
    /// extractors, so "GLiNER-only index + LLM answers" is expressible: the LLM is then used
    /// for answers only. With `gliner.llm_relations` GLiNER finds the entities and the LLM the
    /// relationships (cascade).
    #[cfg_attr(not(feature = "async"), allow(dead_code))]
    pub(super) fn extraction_path(config: &Config) -> ExtractionPath {
        if config.gliner.enabled && config.gliner.llm_relations {
            ExtractionPath::GlinerCascade
        } else if config.gliner.enabled {
            ExtractionPath::Gliner
        } else if config.entities.use_gleaning && config.llm.enabled {
            ExtractionPath::Gleaning
        } else if config.llm.enabled {
            ExtractionPath::LlmSinglePass
        } else {
            ExtractionPath::Pattern
        }
    }

    /// Backend selected by `gliner.backend` ("gline" or "gliner2"); unknown values are a
    /// configuration error rather than a silent fallback.
    #[cfg_attr(not(any(feature = "gliner", feature = "gliner2")), allow(dead_code))]
    pub(super) fn gliner_backend(config: &Config) -> Result<GlinerBackend> {
        match config.gliner.backend.trim().to_lowercase().as_str() {
            "gline" => Ok(GlinerBackend::Gline),
            "gliner2" => Ok(GlinerBackend::Gliner2),
            other => Err(GraphRAGError::Config {
                message: format!(
                    "unknown gliner.backend \"{other}\" (expected \"gline\" or \"gliner2\")"
                ),
            }),
        }
    }

    /// Chunks that still need entity extraction: those of documents added since the last
    /// successful build. Reloaded workspace documents are skipped (already extracted).
    pub(super) fn chunks_to_extract(
        graph: &KnowledgeGraph,
        pending: &indexmap::IndexSet<DocumentId>,
    ) -> Vec<TextChunk> {
        pending
            .iter()
            .filter_map(|id| graph.get_document(id))
            .flat_map(|doc| doc.chunks.iter())
            .filter(|c| !super::summaries::is_summary_chunk(c))
            .map(|c| {
                super::chunk_context::with_original_content(graph.get_chunk(&c.id).unwrap_or(c))
            })
            .collect()
    }

    /// On-disk extraction cache: `<base_dir>/<workspace>/extraction_cache/`.
    /// `None` unless `llm.extraction_cache` is on and `auto_save` provides a directory.
    #[cfg(feature = "async")]
    fn extraction_cache(config: &Config) -> Option<crate::entity::ExtractionCache> {
        if !config.llm.extraction_cache || !config.auto_save.enabled {
            return None;
        }
        let base_dir = config.auto_save.base_dir.as_deref()?;
        let workspace = config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");
        Some(crate::entity::ExtractionCache::new(
            std::path::Path::new(base_dir)
                .join(workspace)
                .join("extraction_cache"),
        ))
    }

    /// Build the knowledge graph from added documents
    ///
    /// This method implements dynamic pipeline selection based on the configured approach:
    /// - **Semantic** (config.approach = "semantic"): Uses LLM-based entity extraction with gleaning
    ///   for high-quality results. Requires the LLM (`llm.enabled`).
    /// - **Algorithmic** (config.approach = "algorithmic"): Uses pattern-based entity extraction
    ///   (regex + capitalization) for fast, resource-efficient processing.
    /// - **Hybrid** (config.approach = "hybrid"): Combines both approaches with weighted fusion.
    ///
    /// The selection is controlled by `config.approach` and mapped from TomlConfig's [mode] section.
    #[cfg(feature = "async")]
    pub async fn build_graph(&mut self) -> Result<()> {
        use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

        // When running inside a TUI, suppress indicatif output to avoid corrupting
        // ratatui's raw-mode terminal (the default draw target writes to stderr).
        let suppress = self.config.suppress_progress_bars;
        let make_pb = move |total: u64, style: ProgressStyle| -> ProgressBar {
            let pb = ProgressBar::new(total).with_style(style);
            if suppress {
                pb.set_draw_target(ProgressDrawTarget::hidden());
            }
            pb
        };

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let chunks = Self::chunks_to_extract(graph, &self.pending_documents);
        let total_chunks = chunks.len();

        // Chunks whose extraction failed in the skip-and-continue branches below. Any
        // failure aborts the build after the phase (nothing cleared, nothing saved) so
        // the documents stay pending and a rerun retries them.
        let mut failed_chunks = 0usize;
        let mut first_failure: Option<String> = None;

        // PHASE 1: Extract and add all entities
        // Pipeline selection based on config.approach (semantic/algorithmic/hybrid)
        // - Semantic: config.entities.use_gleaning = true (LLM-based with iterative refinement)
        // - Algorithmic: config.entities.use_gleaning = false (pattern-based extraction)
        // - Hybrid: config.entities.use_gleaning = true (uses LLM + pattern fusion)

        // DEBUG: Log current configuration state
        #[cfg(feature = "tracing")]
        tracing::info!(
            "build_graph() - Config state: approach='{}', use_gleaning={}, llm.enabled={}",
            self.config.approach,
            self.config.entities.use_gleaning,
            self.config.llm.enabled
        );

        let extraction_path = Self::extraction_path(&self.config);
        #[cfg(feature = "tracing")]
        if self.config.entities.selective_gleaning
            && extraction_path != ExtractionPath::LlmSinglePass
        {
            tracing::info!(
                "entity_extraction.selective_gleaning only applies to single-pass LLM \
                 extraction; ignored for this extraction path"
            );
        }
        if extraction_path == ExtractionPath::Gleaning {
            // LLM-based extraction with gleaning
            #[cfg(feature = "async")]
            {
                use crate::entity::GleaningEntityExtractor;
                use crate::llm::LlmClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM-based entity extraction with gleaning (max_rounds: {})",
                    self.config.entities.max_gleaning_rounds
                );

                // Create LLM client
                let client = LlmClient::new(self.config.llm.clone());

                // Create gleaning config from our config
                let ontology = self.config.ontology()?;
                let gleaning_config = crate::entity::GleaningConfig {
                    max_gleaning_rounds: self.config.entities.max_gleaning_rounds,
                    completion_threshold: 0.8,
                    entity_confidence_threshold: self.config.entities.min_confidence as f64,
                    use_llm_completion_check: true,
                    entity_types: ontology.entity_type_names(),
                    temperature: 0.1,
                    max_tokens: 1500,
                    ontology: Some(ontology),
                    epistemic_status: self.config.entities.epistemic_status,
                };

                // Create gleaning extractor with LLM client
                let extractor = GleaningEntityExtractor::new(client.clone(), gleaning_config);
                let extractor = match Self::extraction_cache(&self.config) {
                    Some(cache) => extractor.with_cache(cache),
                    None => extractor,
                };

                // Create relationship extractor for triple validation (if enabled)
                let rel_extractor = if self.config.entities.enable_triple_reflection {
                    Some(crate::entity::LLMRelationshipExtractor::new(Some(
                        &self.config.llm,
                    ))?)
                } else {
                    None
                };

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-")
                );
                pb.set_message("Extracting entities with LLM");

                // Gleaning rounds of different chunks are independent: run up to
                // `llm.max_concurrent_requests` chunks at once, results in chunk order.
                let limit = self.config.llm.max_concurrent_requests.max(1);
                let results = crate::entity::llm_extractor::run_ordered_concurrent(
                    &chunks,
                    limit,
                    |chunk| extractor.extract_with_gleaning(chunk),
                    |idx| {
                        pb.set_message(format!(
                            "Chunk {}/{} (gleaning with {} rounds)",
                            idx + 1,
                            total_chunks,
                            self.config.entities.max_gleaning_rounds
                        ));
                        pb.inc(1);
                    },
                )
                .await;

                for (chunk, result) in chunks.iter().zip(results) {
                    let (entities, relationships) = result?;

                    // Build entity ID to name mapping for validation
                    let entity_map: std::collections::HashMap<_, _> = entities
                        .iter()
                        .map(|e| (e.id.clone(), e.name.clone()))
                        .collect();

                    // Add extracted entities
                    for entity in entities {
                        graph.add_entity(entity)?;
                    }

                    // Add extracted relationships with optional triple reflection validation
                    if let Some(ref validator) = rel_extractor {
                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            "Triple reflection enabled: validating {} relationships",
                            relationships.len()
                        );

                        let mut validated_count = 0;
                        let mut filtered_count = 0;

                        for relationship in relationships {
                            // Get entity names for validation
                            let source_name = entity_map
                                .get(&relationship.source)
                                .or_else(|| {
                                    graph
                                        .entities()
                                        .find(|e| e.id == relationship.source)
                                        .map(|e| &e.name)
                                })
                                .map(|s| s.as_str())
                                .unwrap_or(relationship.source.0.as_str());
                            let target_name = entity_map
                                .get(&relationship.target)
                                .or_else(|| {
                                    graph
                                        .entities()
                                        .find(|e| e.id == relationship.target)
                                        .map(|e| &e.name)
                                })
                                .map(|s| s.as_str())
                                .unwrap_or(relationship.target.0.as_str());

                            // Validate triple with LLM
                            match validator
                                .validate_triple(
                                    source_name,
                                    &relationship.relation_type,
                                    target_name,
                                    &chunk.content,
                                )
                                .await
                            {
                                Ok(validation) => {
                                    if validation.is_valid
                                        && validation.confidence
                                            >= self.config.entities.validation_min_confidence
                                    {
                                        // Valid relationship, add to graph
                                        if let Err(e) = graph.add_relationship(relationship) {
                                            #[cfg(feature = "tracing")]
                                            tracing::debug!(
                                                "Failed to add validated relationship: {}",
                                                e
                                            );
                                        } else {
                                            validated_count += 1;
                                        }
                                    } else {
                                        // Invalid or low-confidence, filter out
                                        filtered_count += 1;
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!(
                                            "Filtered relationship {} --[{}]--> {} (valid={}, conf={:.2}): {}",
                                            source_name, relationship.relation_type, target_name,
                                            validation.is_valid, validation.confidence, validation.reason
                                        );
                                    }
                                },
                                Err(e) => {
                                    // Validation failed, add anyway with warning
                                    #[cfg(feature = "tracing")]
                                    tracing::warn!(
                                        "Validation error, adding relationship anyway: {}",
                                        e
                                    );
                                    let _ = graph.add_relationship(relationship);
                                },
                            }
                        }

                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            "Triple reflection complete: {} validated, {} filtered",
                            validated_count,
                            filtered_count
                        );
                    } else {
                        // No validation, add all relationships
                        for relationship in relationships {
                            if let Err(e) = graph.add_relationship(relationship) {
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    "Failed to add relationship: {} -> {} ({}). Error: {}",
                                    e.to_string().split("entity ").nth(1).unwrap_or("unknown"),
                                    e.to_string().split("entity ").nth(2).unwrap_or("unknown"),
                                    "relationship",
                                    e
                                );
                            }
                        }
                    }
                }

                pb.finish_with_message("Entity extraction complete");

                // Phase 1.3: ATOM Atomic Fact Extraction (if enabled)
                if self.config.entities.use_atomic_facts {
                    use crate::entity::AtomicFactExtractor;

                    #[cfg(feature = "tracing")]
                    tracing::info!("Starting atomic fact extraction (ATOM methodology)");

                    let atomic_extractor = AtomicFactExtractor::new(client.clone())
                        .with_max_tokens(self.config.entities.max_fact_tokens);

                    let pb_atomic = make_pb(total_chunks as u64,
                        ProgressStyle::default_bar()
                            .template("   [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} atomic facts ({eta})")
                            .expect("Invalid progress bar template")
                            .progress_chars("=>-")
                    );
                    pb_atomic.set_message("Extracting atomic facts");

                    let mut total_facts = 0;
                    let mut total_atomic_entities = 0;
                    let mut total_atomic_relationships = 0;

                    for (idx, chunk) in chunks.iter().enumerate() {
                        pb_atomic.set_message(format!(
                            "Chunk {}/{} (extracting atomic facts)",
                            idx + 1,
                            total_chunks
                        ));

                        #[cfg(feature = "tracing")]
                        tracing::info!("Processing chunk {}/{} (Atomic)", idx + 1, total_chunks);

                        match atomic_extractor.extract_atomic_facts(chunk).await {
                            Ok(facts) => {
                                total_facts += facts.len();

                                // Convert atomic facts to graph elements
                                let (atomic_entities, atomic_relationships) =
                                    atomic_extractor.atomics_to_graph_elements(facts, &chunk.id);

                                total_atomic_entities += atomic_entities.len();
                                total_atomic_relationships += atomic_relationships.len();

                                // Add atomic entities to graph
                                for entity in atomic_entities {
                                    if let Err(e) = graph.add_entity(entity) {
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!("Failed to add atomic entity: {}", e);
                                    }
                                }

                                // Add atomic relationships to graph
                                for relationship in atomic_relationships {
                                    if let Err(e) = graph.add_relationship(relationship) {
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!("Failed to add atomic relationship: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                failed_chunks += 1;
                                first_failure.get_or_insert_with(|| e.to_string());
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    chunk_id = %chunk.id,
                                    error = %e,
                                    "Atomic fact extraction failed for chunk"
                                );
                            },
                        }

                        pb_atomic.inc(1);
                    }

                    pb_atomic.finish_with_message(format!(
                        "Atomic extraction complete: {total_facts} facts → {total_atomic_entities} entities, {total_atomic_relationships} relationships"
                    ));

                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        facts_extracted = total_facts,
                        atomic_entities = total_atomic_entities,
                        atomic_relationships = total_atomic_relationships,
                        "ATOM atomic fact extraction complete"
                    );
                }
            }
        } else if extraction_path == ExtractionPath::LlmSinglePass {
            // LLM single-pass extraction (LLM enabled, gleaning disabled)
            //
            // Uses LLMEntityExtractor directly for one extraction round per chunk.
            // The server prompt cache reuses the shared prompt prefix between chunks.
            #[cfg(feature = "async")]
            {
                use crate::entity::llm_extractor::LLMEntityExtractor;
                use crate::llm::LlmClient;

                #[cfg(feature = "tracing")]
                tracing::info!("Using LLM single-pass entity extraction (no gleaning)");

                let client = LlmClient::new(self.config.llm.clone());
                let ontology = self.config.ontology()?;

                let extractor = LLMEntityExtractor::new(client, ontology.entity_type_names())
                    .with_ontology(ontology)
                    .with_epistemic_status(self.config.entities.epistemic_status)
                    .with_temperature(self.config.llm.temperature.unwrap_or(0.1))
                    .with_max_tokens(Self::extraction_max_tokens(&self.config));
                let extractor = match Self::extraction_cache(&self.config) {
                    Some(cache) => extractor.with_cache(cache),
                    None => extractor,
                };

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-"),
                );
                pb.set_message("Extracting entities with LLM (single-pass)");

                // Up to `llm.max_concurrent_requests` requests in flight (match the server
                // slots); results come back in chunk order so the graph is deterministic.
                let limit = self.config.llm.max_concurrent_requests.max(1);
                let results = extractor
                    .extract_chunks_concurrent(&chunks, limit, |idx| {
                        pb.set_message(format!(
                            "Chunk {}/{} (LLM single-pass)",
                            idx + 1,
                            total_chunks
                        ));
                        pb.inc(1);
                    })
                    .await;
                let mut results = results;

                if self.config.entities.selective_gleaning {
                    let min_entities = self.config.entities.selective_min_entities;
                    let max_relationships = self.config.entities.selective_max_relationships;
                    // Selective gleaning: chunks the single-pass call under-extracted get
                    // one relation-only call over their own entities (same extractor, so
                    // the same cache). A failure keeps the first-pass result.
                    let selected: Vec<usize> = results
                        .iter()
                        .enumerate()
                        .filter_map(|(i, r)| match r {
                            Ok((ents, rels))
                                if needs_relation_pass(
                                    ents.len(),
                                    rels.len(),
                                    min_entities,
                                    max_relationships,
                                ) =>
                            {
                                Some(i)
                            },
                            _ => None,
                        })
                        .collect();
                    let second = {
                        let jobs: Vec<(&TextChunk, &[Entity])> = selected
                            .iter()
                            .filter_map(|&i| match &results[i] {
                                Ok((ents, _)) => Some((&chunks[i], ents.as_slice())),
                                Err(_) => None,
                            })
                            .collect();
                        extractor
                            .extract_relationships_concurrent(&jobs, limit, |_| {})
                            .await
                    };
                    let (mut added, mut failures) = (0usize, 0usize);
                    for (&i, result) in selected.iter().zip(second) {
                        match result {
                            Ok(extra) => {
                                if let Ok((_, rels)) = &mut results[i] {
                                    added += merge_new_relationships(rels, extra);
                                }
                            },
                            Err(_e) => {
                                failures += 1;
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    chunk_id = %chunks[i].id,
                                    error = %_e,
                                    "selective gleaning failed for chunk, keeping the first-pass result"
                                );
                            },
                        }
                    }
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        "Selective gleaning (min_entities={}, max_relationships={}): {}/{} chunks selected, +{} relationships{}",
                        min_entities,
                        max_relationships,
                        selected.len(),
                        total_chunks,
                        added,
                        if failures > 0 {
                            format!(", {failures} failed")
                        } else {
                            String::new()
                        }
                    );
                    let _ = (added, failures);
                }

                for (chunk, result) in chunks.iter().zip(results) {
                    match result {
                        Ok((entities, relationships)) => {
                            for entity in entities {
                                if let Err(e) = graph.add_entity(entity) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("Failed to add entity: {}", e);
                                }
                            }
                            for relationship in relationships {
                                if let Err(e) = graph.add_relationship(relationship) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("Failed to add relationship: {}", e);
                                }
                            }
                        },
                        Err(e) => {
                            failed_chunks += 1;
                            first_failure.get_or_insert_with(|| e.to_string());
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk.id,
                                error = %e,
                                "LLM extraction failed for chunk"
                            );
                        },
                    }
                }

                pb.finish_with_message("LLM single-pass extraction complete");
            }
        } else if matches!(
            extraction_path,
            ExtractionPath::Gliner | ExtractionPath::GlinerCascade
        ) {
            // GLiNER joint NER + RE extraction (`gliner.backend`: gline-rs ONNX or
            // GLiNER2.5 candle)
            //
            // Both backends are synchronous and CPU-heavy, so each chunk runs in
            // `spawn_blocking` to avoid stalling the Tokio runtime. One extractor is
            // created outside the loop and shared across blocking tasks behind an `Arc`.
            #[cfg(any(feature = "gliner", feature = "gliner2"))]
            {
                use std::sync::Arc;

                type ChunkExtractFn =
                    dyn Fn(&TextChunk) -> Result<(Vec<Entity>, Vec<Relationship>)> + Send + Sync;

                let backend = Self::gliner_backend(&self.config)?;

                let cascade = extraction_path == ExtractionPath::GlinerCascade;
                let mut gliner_config = self.config.gliner.clone();
                if cascade {
                    if !self.config.llm.enabled {
                        return Err(crate::core::error::GraphRAGError::Config {
                            message: "gliner.llm_relations requires llm.enabled = true".into(),
                        });
                    }
                    if !gliner_config.relation_labels.is_empty() {
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            "gliner.llm_relations is set: ignoring gliner.relation_labels \
                             (the LLM extracts the relationships)"
                        );
                        gliner_config.relation_labels.clear();
                    }
                }

                let ontology = self.config.ontology()?;
                let extractor: Arc<ChunkExtractFn> = match backend {
                    GlinerBackend::Gline => {
                        #[cfg(feature = "gliner")]
                        {
                            let ext = Arc::new(
                                crate::entity::GLiNERExtractor::new(gliner_config)
                                    .map_err(|e| {
                                        crate::core::error::GraphRAGError::EntityExtraction {
                                            message: format!("GLiNER init failed: {e}"),
                                        }
                                    })?
                                    .with_ontology(ontology),
                            );
                            Arc::new(move |c: &TextChunk| ext.extract_from_chunk(c))
                        }
                        #[cfg(not(feature = "gliner"))]
                        {
                            return Err(crate::core::error::GraphRAGError::Config {
                                message: "gliner.backend = \"gline\" but graphrag-core was \
                                          compiled without --features gliner"
                                    .into(),
                            });
                        }
                    },
                    GlinerBackend::Gliner2 => {
                        #[cfg(feature = "gliner2")]
                        {
                            let ext = Arc::new(
                                crate::entity::Gliner2Extractor::new(gliner_config)
                                    .map_err(|e| {
                                        crate::core::error::GraphRAGError::EntityExtraction {
                                            message: format!("GLiNER2 init failed: {e}"),
                                        }
                                    })?
                                    .with_ontology(ontology),
                            );
                            Arc::new(move |c: &TextChunk| ext.extract_from_chunk(c))
                        }
                        #[cfg(not(feature = "gliner2"))]
                        {
                            return Err(crate::core::error::GraphRAGError::Config {
                                message: "gliner.backend = \"gliner2\" but graphrag-core was \
                                          compiled without --features gliner2"
                                    .into(),
                            });
                        }
                    },
                };

                // g5 mode: Jev classifies entity types and relations of the GLiNER entities.
                #[cfg(feature = "ureq")]
                let jev: Option<Arc<crate::entity::jev::JevClassifier>> =
                    if self.config.gliner.jev.enabled {
                        if cascade {
                            return Err(crate::core::error::GraphRAGError::Config {
                                message: "gliner.jev.enabled and gliner.llm_relations are \
                                          mutually exclusive"
                                    .into(),
                            });
                        }
                        let default_log = if self.config.auto_save.enabled {
                            self.config.auto_save.base_dir.as_deref().map(|b| {
                                std::path::Path::new(b)
                                    .join(
                                        self.config
                                            .auto_save
                                            .workspace_name
                                            .as_deref()
                                            .unwrap_or("default"),
                                    )
                                    .join("jev_raw.jsonl")
                            })
                        } else {
                            None
                        };
                        Some(Arc::new(crate::entity::jev::JevClassifier::new(
                            self.config.gliner.jev.clone(),
                            self.config.ontology()?,
                            default_log,
                        )?))
                    } else {
                        None
                    };
                #[cfg(not(feature = "ureq"))]
                if self.config.gliner.jev.enabled {
                    return Err(crate::core::error::GraphRAGError::Config {
                        message: "gliner.jev.enabled but graphrag-core was compiled without \
                                  --features ureq"
                            .into(),
                    });
                }
                #[cfg(feature = "ureq")]
                let jev_active = jev.is_some();
                #[cfg(not(feature = "ureq"))]
                let jev_active = false;

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template(
                            "   [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} chunks ({eta})",
                        )
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-"),
                );
                pb.set_message("Extracting entities with GLiNER-Relex");

                // Parallelism cap — ORT GLiNER is thread-safe via Arc<Model>.
                // Default 4 concurrent inferences (RTX-class CPU/GPU sweet spot).
                let parallelism = self.config.gliner.max_concurrent_chunks.unwrap_or(4).max(1);

                use futures::stream::{self, StreamExt};
                let mut stream = stream::iter(chunks.iter().cloned().enumerate())
                    .map(|(idx, chunk)| {
                        let ext = Arc::clone(&extractor);
                        let chunk_id = chunk.id.clone();
                        async move {
                            let r = tokio::task::spawn_blocking(move || ext(&chunk)).await;
                            (idx, chunk_id, r)
                        }
                    })
                    .buffer_unordered(parallelism);

                // Cascade: GLiNER entities per chunk (by chunk index), input of the LLM stage.
                let mut chunk_entities: Vec<Option<Vec<Entity>>> = vec![None; chunks.len()];

                while let Some((idx, chunk_id, join_result)) = stream.next().await {
                    let result = join_result.map_err(|e| {
                        crate::core::error::GraphRAGError::EntityExtraction {
                            message: format!("spawn_blocking join error: {e}"),
                        }
                    })?;
                    match result {
                        Ok((entities, relationships)) => {
                            if cascade || jev_active {
                                chunk_entities[idx] = Some(entities.clone());
                            }
                            // With Jev the entities are added after re-typing, and
                            // GLiNER's own relations are replaced.
                            let (entities, relationships) = if jev_active {
                                (Vec::new(), Vec::new())
                            } else {
                                (entities, relationships)
                            };
                            for entity in entities {
                                if let Err(e) = graph.add_entity(entity) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("GLiNER: failed to add entity: {}", e);
                                }
                            }
                            for rel in relationships {
                                if let Err(e) = graph.add_relationship(rel) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("GLiNER: failed to add relationship: {}", e);
                                }
                            }
                        },
                        Err(e) => {
                            failed_chunks += 1;
                            first_failure.get_or_insert_with(|| e.to_string());
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk_id,
                                error = %e,
                                "GLiNER extraction failed for chunk"
                            );
                        },
                    }
                    pb.inc(1);
                }

                pb.finish_with_message("GLiNER-Relex extraction complete");

                #[cfg(feature = "ureq")]
                if let Some(jev) = jev.clone() {
                    use crate::entity::jev::JevStats;
                    let jobs: Vec<(TextChunk, Vec<Entity>)> = chunks
                        .iter()
                        .zip(&chunk_entities)
                        .filter_map(|(c, e)| {
                            let e = e.as_ref()?;
                            (!e.is_empty()).then(|| (c.clone(), e.clone()))
                        })
                        .collect();
                    let pb = make_pb(jobs.len() as u64,
                        ProgressStyle::default_bar()
                            .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                            .expect("Invalid progress bar template")
                            .progress_chars("=>-"),
                    );
                    pb.set_message("Classifying entities and relations with Jev");
                    let mut total = JevStats::default();
                    let mut jstream = stream::iter(jobs)
                        .map(|(chunk, ents)| {
                            let jev = Arc::clone(&jev);
                            async move {
                                tokio::task::spawn_blocking(move || {
                                    jev.classify_chunk(&chunk, &ents)
                                })
                                .await
                            }
                        })
                        .buffer_unordered(jev.max_concurrent());
                    while let Some(joined) = jstream.next().await {
                        let res = joined.map_err(|e| {
                            crate::core::error::GraphRAGError::EntityExtraction {
                                message: format!("Jev spawn_blocking join error: {e}"),
                            }
                        })?;
                        total.add(&res.stats);
                        for entity in res.entities {
                            if let Err(e) = graph.add_entity(entity) {
                                #[cfg(feature = "tracing")]
                                tracing::debug!("Jev: failed to add entity: {}", e);
                            }
                        }
                        for rel in res.relationships {
                            if let Err(e) = graph.add_relationship(rel) {
                                #[cfg(feature = "tracing")]
                                tracing::debug!("Jev: failed to add relationship: {}", e);
                            }
                        }
                        pb.inc(1);
                    }
                    pb.finish_with_message("Jev classification complete");
                    #[cfg(feature = "tracing")]
                    tracing::info!("{}", total.summary());
                    #[cfg(not(feature = "tracing"))]
                    eprintln!("{}", total.summary());
                    if total.filtered_entities > 0 {
                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            "Jev: {} entities filtered out by gliner.jev.min_entity={}",
                            total.filtered_entities,
                            self.config.gliner.jev.min_entity
                        );
                        #[cfg(not(feature = "tracing"))]
                        eprintln!(
                            "Jev: {} entities filtered out by gliner.jev.min_entity={}",
                            total.filtered_entities, self.config.gliner.jev.min_entity
                        );
                    }
                    if total.dropped_entities > 0 {
                        #[cfg(feature = "tracing")]
                        tracing::info!(
                            "Jev: {} entities beyond gliner.jev.max_entities_per_chunk kept with no relations (GLiNER type, or the Jev type when gliner.jev.filter_entities is on)",
                            total.dropped_entities
                        );
                    }
                }

                if cascade {
                    // Stage 2: the LLM extracts only the relationships among each chunk's
                    // GLiNER entities (shorter output than full extraction). Chunks with
                    // fewer than 2 distinct entities cost no call. Chunks whose GLiNER stage
                    // failed (counted above) have no entities and are skipped.
                    use crate::entity::llm_extractor::LLMEntityExtractor;
                    use crate::llm::LlmClient;

                    let ontology = self.config.ontology()?;
                    let llm_extractor = LLMEntityExtractor::new(
                        LlmClient::new(self.config.llm.clone()),
                        ontology.entity_type_names(),
                    )
                    .with_ontology(ontology)
                    .with_epistemic_status(self.config.entities.epistemic_status)
                    .with_temperature(self.config.llm.temperature.unwrap_or(0.1))
                    .with_max_tokens(Self::extraction_max_tokens(&self.config));
                    let llm_extractor = match Self::extraction_cache(&self.config) {
                        Some(cache) => llm_extractor.with_cache(cache),
                        None => llm_extractor,
                    };

                    let jobs: Vec<(&TextChunk, &[Entity])> = chunks
                        .iter()
                        .zip(&chunk_entities)
                        .filter_map(|(chunk, ents)| {
                            let ents = ents.as_deref()?;
                            (ents.len() >= 2).then_some((chunk, ents))
                        })
                        .collect();

                    let pb = make_pb(jobs.len() as u64,
                        ProgressStyle::default_bar()
                            .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                            .expect("Invalid progress bar template")
                            .progress_chars("=>-"),
                    );
                    pb.set_message("Extracting relationships with LLM (GLiNER cascade)");

                    let limit = self.config.llm.max_concurrent_requests.max(1);
                    let results = llm_extractor
                        .extract_relationships_concurrent(&jobs, limit, |_| pb.inc(1))
                        .await;

                    for ((chunk, _), result) in jobs.iter().zip(results) {
                        match result {
                            Ok(relationships) => {
                                for rel in relationships {
                                    if let Err(e) = graph.add_relationship(rel) {
                                        #[cfg(feature = "tracing")]
                                        tracing::debug!(
                                            "Cascade: failed to add relationship: {}",
                                            e
                                        );
                                    }
                                }
                            },
                            Err(e) => {
                                failed_chunks += 1;
                                first_failure.get_or_insert_with(|| e.to_string());
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    chunk_id = %chunk.id,
                                    error = %e,
                                    "LLM relation extraction failed for chunk"
                                );
                            },
                        }
                    }
                    pb.finish_with_message("LLM relation extraction complete");
                }
            }
            #[cfg(not(any(feature = "gliner", feature = "gliner2")))]
            return Err(crate::core::error::GraphRAGError::Config {
                message: "GLiNER enabled in config but crate compiled without --features gliner \
                          or --features gliner2"
                    .into(),
            });
        } else {
            // Pattern-based extraction (regex + capitalization)
            use crate::entity::EntityExtractor;

            #[cfg(feature = "tracing")]
            tracing::info!("Using pattern-based entity extraction");

            let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

            // Create progress bar for pattern-based extraction
            let pb = make_pb(
                total_chunks as u64,
                ProgressStyle::default_bar()
                    .template(
                        "   [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} chunks ({eta})",
                    )
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-"),
            );
            pb.set_message("Extracting entities (pattern-based)");

            for (idx, chunk) in chunks.iter().enumerate() {
                pb.set_message(format!(
                    "Chunk {}/{} (pattern-based)",
                    idx + 1,
                    total_chunks
                ));

                #[cfg(feature = "tracing")]
                tracing::info!("Processing chunk {}/{} (Pattern)", idx + 1, total_chunks);

                let entities = extractor.extract_from_chunk(chunk)?;
                for entity in entities {
                    graph.add_entity(entity)?;
                }

                pb.inc(1);
            }

            pb.finish_with_message("Entity extraction complete");

            // PHASE 2: Extract and add relationships between entities (for pattern-based only)
            // Gleaning extractor already extracts relationships in Phase 1
            // Only proceed if graph construction config enables relationship extraction
            if self.config.graph.extract_relationships {
                let all_entities: Vec<_> = graph.entities().cloned().collect();

                // Create progress bar for relationship extraction
                let rel_pb = make_pb(total_chunks as u64,
                ProgressStyle::default_bar()
                    .template("   [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} chunks ({eta})")
                    .expect("Invalid progress bar template")
                    .progress_chars("=>-")
            );
                rel_pb.set_message("Extracting relationships");

                for (idx, chunk) in chunks.iter().enumerate() {
                    rel_pb.set_message(format!(
                        "Chunk {}/{} (relationships)",
                        idx + 1,
                        total_chunks
                    ));
                    // Get entities that appear in this chunk
                    let chunk_entities: Vec<_> = all_entities
                        .iter()
                        .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                        .cloned()
                        .collect();

                    if chunk_entities.len() < 2 {
                        rel_pb.inc(1);
                        continue; // Need at least 2 entities for relationships
                    }

                    // Extract relationships
                    let relationships = extractor.extract_relationships(&chunk_entities, chunk)?;

                    // Add relationships to graph
                    for (source_id, target_id, relation_type) in relationships {
                        let relationship = Relationship {
                            description: None,
                            source: source_id.clone(),
                            target: target_id.clone(),
                            relation_type: relation_type.clone(),
                            confidence: self.config.graph.relationship_confidence_threshold,
                            context: vec![chunk.id.clone()],
                            embedding: None,
                            temporal_type: None,
                            temporal_range: None,
                            causal_strength: None,
                        };

                        // Log errors for debugging relationship extraction issues
                        if let Err(_e) = graph.add_relationship(relationship) {
                            #[cfg(feature = "tracing")]
                            tracing::debug!(
                                "Failed to add relationship: {} -> {} ({}). Error: {}",
                                source_id,
                                target_id,
                                relation_type,
                                _e
                            );
                        }
                    }

                    rel_pb.inc(1);
                }

                rel_pb.finish_with_message("Relationship extraction complete");
            } // End of extract_relationships check
        } // End of pattern-based extraction

        if failed_chunks > 0 {
            return Err(GraphRAGError::EntityExtraction {
                message: format!(
                    "entity extraction failed for {failed_chunks} of {total_chunks} chunks \
                     (first error: {}); successful chunks are in the extraction cache when \
                     `llm.extraction_cache` is on, rerun to retry the failed ones",
                    first_failure.unwrap_or_default()
                ),
            });
        }

        // Same name extracted under different types -> one entity (before embeddings,
        // which are computed later at retrieval index time).
        if let Some(graph) = self.knowledge_graph.as_mut() {
            let merged = graph.merge_same_name_entities();
            #[cfg(feature = "tracing")]
            tracing::info!("Merged {} same-name entities of different type", merged);
            #[cfg(not(feature = "tracing"))]
            let _ = merged;
        }

        // Optional per-chapter summaries as extra chunks (after a successful extraction, so a
        // failed build costs no summary calls; before the save so they are persisted).
        self.write_chapter_summaries().await;
        // Optional context line per chunk (after the summaries and the extraction, which
        // read the original text; before the save so it is persisted).
        self.write_chunk_contexts().await;
        self.pending_documents.clear();

        // Persist to workspace if storage is configured
        self.save_to_workspace()?;

        Ok(())
    }

    /// Build the knowledge graph synchronously.
    ///
    /// The synchronous pipeline deliberately uses the local pattern extractor.  LLM-backed
    /// extraction is asynchronous and therefore reports an explicit error here instead of
    /// silently changing the configured pipeline.
    pub fn build_graph_sync(&mut self) -> Result<()> {
        use crate::entity::EntityExtractor;

        self.ensure_sync_supported()?;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let chunks = Self::chunks_to_extract(graph, &self.pending_documents);

        #[cfg(feature = "tracing")]
        tracing::info!("Using pattern-based entity extraction (sync mode)");

        let extractor = EntityExtractor::new(self.config.entities.min_confidence)?;

        for chunk in &chunks {
            let entities = extractor.extract_from_chunk(chunk)?;
            for entity in entities {
                graph.add_entity(entity)?;
            }
        }

        // Extract relationships if enabled
        if self.config.graph.extract_relationships {
            let all_entities: Vec<_> = graph.entities().cloned().collect();

            for chunk in &chunks {
                let chunk_entities: Vec<_> = all_entities
                    .iter()
                    .filter(|e| e.mentions.iter().any(|m| m.chunk_id == chunk.id))
                    .cloned()
                    .collect();

                if chunk_entities.len() < 2 {
                    continue;
                }

                let relationships = extractor.extract_relationships(&chunk_entities, chunk)?;

                for (source_id, target_id, relation_type) in relationships {
                    let relationship = Relationship {
                        description: None,
                        source: source_id.clone(),
                        target: target_id.clone(),
                        relation_type: relation_type.clone(),
                        confidence: self.config.graph.relationship_confidence_threshold,
                        context: vec![chunk.id.clone()],
                        embedding: None,
                        temporal_type: None,
                        temporal_range: None,
                        causal_strength: None,
                    };

                    if let Err(_e) = graph.add_relationship(relationship) {
                        #[cfg(feature = "tracing")]
                        tracing::debug!(
                            "Failed to add relationship: {} -> {} ({}). Error: {}",
                            source_id,
                            target_id,
                            relation_type,
                            _e
                        );
                    }
                }
            }
        }

        if let Some(graph) = self.knowledge_graph.as_mut() {
            graph.merge_same_name_entities();
        }
        self.pending_documents.clear();

        self.save_to_workspace()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(gleaning: bool, llm: bool, gliner: bool, cascade: bool) -> Config {
        let mut config = Config::default();
        config.entities.use_gleaning = gleaning;
        config.llm.enabled = llm;
        config.gliner.enabled = gliner;
        config.gliner.llm_relations = cascade;
        config
    }

    #[test]
    fn extraction_path_selection() {
        use ExtractionPath::*;
        let path = |g, l, n, c| GraphRAG::extraction_path(&config(g, l, n, c));
        assert_eq!(path(true, true, false, false), Gleaning);
        assert_eq!(path(false, true, false, false), LlmSinglePass);
        assert_eq!(path(false, false, true, false), Gliner);
        // GLiNER on beats the LLM extractors (the LLM then only answers): GLiNER-only index
        assert_eq!(path(false, true, true, false), Gliner);
        assert_eq!(path(true, true, true, false), Gliner);
        assert_eq!(path(false, false, false, false), Pattern);
        // the cascade takes precedence over both LLM-only branches
        assert_eq!(path(false, true, true, true), GlinerCascade);
        assert_eq!(path(true, true, true, true), GlinerCascade);
        // llm_relations without llm.enabled still selects the cascade branch, which then
        // fails with a config error (see `build_graph`)
        assert_eq!(path(false, false, true, true), GlinerCascade);
        // llm_relations alone does nothing unless GLiNER is enabled
        assert_eq!(path(false, true, false, true), LlmSinglePass);
    }

    #[test]
    fn gliner_backend_selection() {
        let mut config = Config::default();
        assert_eq!(
            GraphRAG::gliner_backend(&config).unwrap(),
            GlinerBackend::Gline
        );
        config.gliner.backend = " GLiNER2 ".into();
        assert_eq!(
            GraphRAG::gliner_backend(&config).unwrap(),
            GlinerBackend::Gliner2
        );
        config.gliner.backend = "onnx".into();
        let err = GraphRAG::gliner_backend(&config).unwrap_err().to_string();
        assert!(err.contains("unknown gliner.backend"), "{err}");
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn gliner2_cerebras_bench_config_routes_to_gliner_only_extraction() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/e2e/configs/it_pinocchio_gliner2_cerebras.json5"
        );
        let config = crate::config::loader::load_config(path).unwrap();
        assert!(config.gliner.enabled && !config.gliner.llm_relations);
        // the LLM is on (answers only) yet extraction stays GLiNER
        assert!(config.llm.enabled);
        assert_eq!(GraphRAG::extraction_path(&config), ExtractionPath::Gliner);
        assert_eq!(
            GraphRAG::gliner_backend(&config).unwrap(),
            GlinerBackend::Gliner2
        );
        assert_eq!(config.gliner.relation_threshold, 0.65);
        assert_eq!(config.gliner.entity_threshold, 0.5);
        assert_eq!(config.gliner.relation_labels.len(), 7);
        assert_eq!(
            config.auto_save.workspace_name.as_deref(),
            Some("it_pinocchio_gliner2_cerebras")
        );
        let ontology = config.ontology().unwrap();
        assert_eq!(
            ontology.entity_type_names(),
            ["PERSON", "LOCATION", "EVENT"]
        );
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn gliner_cascade_bench_config_reaches_the_cascade() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/e2e/configs/it_pinocchio_gliner_cascade.json5"
        );
        let config = crate::config::loader::load_config(path).unwrap();
        assert!(config.gliner.enabled && config.gliner.llm_relations);
        assert!(config.llm.enabled);
        assert!(config.gliner.relation_labels.is_empty());
        assert_eq!(config.gliner.mode, "span");
        assert_eq!(
            config.auto_save.workspace_name.as_deref(),
            Some("it_pinocchio_gliner_cascade")
        );
        assert_eq!(
            GraphRAG::extraction_path(&config),
            ExtractionPath::GlinerCascade
        );
        // every GLiNER label maps onto a real ontology type (none falls to OTHER)
        let ontology = config.ontology().unwrap();
        for label in &config.gliner.entity_labels {
            assert_ne!(ontology.normalize_entity_type(label), "OTHER", "{label}");
        }
    }

    #[cfg(feature = "json5-support")]
    #[test]
    fn gliner_cascade_v2_bench_config_uses_canonical_types() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/e2e/configs/it_pinocchio_gliner_cascade_v2.json5"
        );
        let config = crate::config::loader::load_config(path).unwrap();
        assert_eq!(
            GraphRAG::extraction_path(&config),
            ExtractionPath::GlinerCascade
        );
        // labels character/place/event become PERSON/LOCATION/EVENT, not renamed types
        let ontology = config.ontology().unwrap();
        assert_eq!(
            ontology.entity_type_names(),
            ["PERSON", "LOCATION", "EVENT"]
        );
    }
}

#[cfg(test)]
mod selective_gleaning_unit_tests {
    use super::*;
    use crate::core::EntityId;

    fn rel(source: &str, target: &str, kind: &str, desc: &str) -> Relationship {
        Relationship {
            source: EntityId::new(source.to_string()),
            target: EntityId::new(target.to_string()),
            relation_type: kind.to_string(),
            description: Some(desc.to_string()),
            confidence: 0.5,
            context: vec![],
            embedding: None,
            temporal_type: None,
            temporal_range: None,
            causal_strength: None,
        }
    }

    #[test]
    fn extraction_max_tokens_resolution() {
        let mut config = Config::default();
        config.llm.max_tokens = None;
        assert_eq!(GraphRAG::extraction_max_tokens(&config), 1500);
        config.llm.max_tokens = Some(800);
        assert_eq!(GraphRAG::extraction_max_tokens(&config), 800);
        config.entities.extraction_max_tokens = Some(4096);
        assert_eq!(GraphRAG::extraction_max_tokens(&config), 4096);
    }

    #[test]
    fn selection_rule_boundaries() {
        assert!(!needs_relation_pass(2, 0, 3, 1));
        assert!(needs_relation_pass(3, 0, 3, 1));
        assert!(needs_relation_pass(3, 1, 3, 1));
        assert!(!needs_relation_pass(3, 2, 3, 1));
        assert!(needs_relation_pass(10, 1, 3, 1));
        assert!(!needs_relation_pass(0, 0, 3, 1));
    }

    #[test]
    fn selection_rule_custom_thresholds() {
        // min 2 / max 1
        assert!(needs_relation_pass(2, 1, 2, 1));
        assert!(!needs_relation_pass(1, 0, 2, 1));
        assert!(!needs_relation_pass(2, 2, 2, 1));
        // min 3 / max 2
        assert!(needs_relation_pass(3, 2, 3, 2));
        assert!(!needs_relation_pass(3, 3, 3, 2));
        assert!(!needs_relation_pass(2, 0, 3, 2));
    }

    #[test]
    fn merge_skips_same_source_target_and_type_only() {
        let mut first = vec![rel("a", "b", "FRIEND_OF", "first")];
        let added = merge_new_relationships(
            &mut first,
            vec![
                rel("a", "b", "FRIEND_OF", "dup"),     // duplicate
                rel("a", "b", "ENEMY_OF", "other"),    // other type
                rel("b", "a", "FRIEND_OF", "reverse"), // other direction
                rel("a", "c", "FRIEND_OF", "new"),
                rel("a", "c", "FRIEND_OF", "new again"), // duplicate within the batch
            ],
        );
        assert_eq!(added, 3);
        assert_eq!(first.len(), 4);
        assert_eq!(first[0].description.as_deref(), Some("first"));
    }

    #[test]
    fn merge_duplicate_replaces_said_description_only_with_narrated() {
        let said = "[detto da Socrate] x";
        // said then narrated: replaced, not counted
        let mut v = vec![rel("a", "b", "K", said)];
        assert_eq!(
            merge_new_relationships(&mut v, vec![rel("a", "b", "K", "narrated")]),
            0
        );
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].description.as_deref(), Some("narrated"));
        // narrated then said: unchanged
        let mut v = vec![rel("a", "b", "K", "narrated")];
        assert_eq!(
            merge_new_relationships(&mut v, vec![rel("a", "b", "K", said)]),
            0
        );
        assert_eq!(v[0].description.as_deref(), Some("narrated"));
        // said then None / blank: unchanged
        let mut none = rel("a", "b", "K", "");
        none.description = None;
        for incoming in [none, rel("a", "b", "K", "  ")] {
            let mut v = vec![rel("a", "b", "K", said)];
            assert_eq!(merge_new_relationships(&mut v, vec![incoming]), 0);
            assert_eq!(v[0].description.as_deref(), Some(said));
        }
    }
}

#[cfg(all(test, feature = "llm", feature = "async"))]
mod selective_gleaning_e2e_tests {
    use super::*;
    use crate::core::Document;
    use std::sync::{Arc, Mutex};

    /// Marker the relation-only prompt carries and the single-pass prompt does not.
    const RELATION_MARKER: &str = "list of entities already identified";

    fn names(n: &[&str]) -> String {
        n.iter()
            .map(|x| format!(r#"{{"name":"{x}","type":"PERSON","description":"d"}}"#))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn relation(s: &str, t: &str, desc: &str) -> String {
        format!(
            r#"{{"source":"{s}","target":"{t}","description":"{desc}","type":"FRIEND_OF","strength":0.9}}"#
        )
    }

    /// Chunk text decides the answer. Single-pass: every chunk gets 3 entities; ALFA/FAIL
    /// 0 relationships, DUPL 1, BETA 2. Relation-only: ALFA 1 new, DUPL 1 duplicate + 1 new,
    /// FAIL HTTP 500.
    fn answer(body: &str) -> Option<String> {
        let is_relation = body.contains(RELATION_MARKER);
        let (ents, first, second): (&[&str], Vec<String>, Vec<String>) = if body.contains("ALFA") {
            (
                &["Anna", "Bruno", "Carla"],
                vec![],
                vec![relation("Anna", "Bruno", "second")],
            )
        } else if body.contains("BETA") {
            (
                &["Dario", "Elena", "Fabio"],
                vec![
                    relation("Dario", "Elena", "uno"),
                    relation("Elena", "Fabio", "due"),
                ],
                vec![relation("Dario", "Fabio", "must not be asked")],
            )
        } else if body.contains("DUPL") {
            (
                &["Gino", "Hana", "Ivo"],
                vec![relation("Gino", "Hana", "first")],
                vec![
                    relation("Gino", "Hana", "duplicate"),
                    relation("Gino", "Ivo", "new"),
                ],
            )
        } else if body.contains("FAIL") {
            if is_relation {
                return None;
            }
            (&["Luca", "Maria", "Nico"], vec![], vec![])
        } else {
            (&[], vec![], vec![])
        };
        Some(if is_relation {
            format!(r#"{{"relationships":[{}]}}"#, second.join(","))
        } else {
            format!(
                r#"{{"entities":[{}],"relationships":[{}]}}"#,
                names(ents),
                first.join(",")
            )
        })
    }

    fn chat_stub() -> (String, Arc<Mutex<Vec<String>>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::new()));
        let shared = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let log = shared.clone();
                std::thread::spawn(move || {
                    let mut raw = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        let n = stream.read(&mut buf).unwrap_or(0);
                        raw.extend_from_slice(&buf[..n]);
                        let text = String::from_utf8_lossy(&raw).to_string();
                        if let Some(pos) = text.find("\r\n\r\n") {
                            let len = text[..pos]
                                .to_ascii_lowercase()
                                .split("content-length:")
                                .nth(1)
                                .and_then(|r| r.lines().next())
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            if raw.len() >= pos + 4 + len || n == 0 {
                                break;
                            }
                        } else if n == 0 {
                            break;
                        }
                    }
                    let text = String::from_utf8_lossy(&raw).to_string();
                    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                    log.lock().unwrap().push(body.clone());
                    let Some(content) = answer(&body) else {
                        let _ = stream.write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom",
                        );
                        return;
                    };
                    let json = serde_json::json!({
                        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                    })
                    .to_string();
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                            json.len()
                        )
                        .as_bytes(),
                    );
                });
            }
        });
        (url, log)
    }

    fn relation_calls(log: &Mutex<Vec<String>>) -> usize {
        log.lock()
            .unwrap()
            .iter()
            .filter(|b| b.contains(RELATION_MARKER))
            .count()
    }

    fn doc(markers: &[&str]) -> Document {
        let chunks = markers
            .iter()
            .enumerate()
            .map(|(n, m)| {
                TextChunk::new(
                    ChunkId::new(format!("doc_{n}")),
                    DocumentId::new("doc".to_string()),
                    format!("testo {m}"),
                    n * 100,
                    n * 100 + 50,
                )
            })
            .collect();
        Document {
            id: DocumentId::new("doc".to_string()),
            title: "doc".to_string(),
            content: String::new(),
            metadata: indexmap::IndexMap::new(),
            chunks,
        }
    }

    async fn build(url: &str, selective: bool, markers: &[&str]) -> GraphRAG {
        let mut config = Config::default().with_llm_base_url(url);
        config.llm.max_retries = 1;
        config.llm.enable_caching = false;
        config.entities.use_gleaning = false;
        config.entities.selective_gleaning = selective;
        let mut g = GraphRAG::new(config).unwrap();
        g.initialize().unwrap();
        g.add_document(doc(markers)).unwrap();
        g.build_graph().await.expect("build must succeed");
        g
    }

    fn edges(g: &GraphRAG) -> Vec<(String, String, String)> {
        let graph = g.knowledge_graph().unwrap();
        let name = |id: &crate::core::EntityId| graph.get_entity(id).unwrap().name.clone();
        let mut v: Vec<_> = graph
            .relationships()
            .map(|r| (name(&r.source), name(&r.target), r.relation_type.clone()))
            .collect();
        v.sort();
        v
    }

    #[tokio::test]
    async fn flag_on_calls_only_for_under_extracted_chunks() {
        let (url, log) = chat_stub();
        let g = build(&url, true, &["ALFA", "BETA"]).await;
        // 2 first-pass calls + exactly one relation-only call (ALFA; BETA has 2 relationships)
        assert_eq!(log.lock().unwrap().len(), 3);
        assert_eq!(relation_calls(&log), 1);
        let e = edges(&g);
        assert_eq!(e.len(), 3, "{e:?}");
        assert!(e.iter().any(|(s, t, _)| s == "Anna" && t == "Bruno"));
        assert!(!e.iter().any(|(s, t, _)| s == "Dario" && t == "Fabio"));
    }

    #[tokio::test]
    async fn flag_off_makes_no_extra_call_and_keeps_the_old_graph() {
        let (url, log) = chat_stub();
        let g = build(&url, false, &["ALFA", "BETA"]).await;
        assert_eq!(log.lock().unwrap().len(), 2);
        assert_eq!(relation_calls(&log), 0);
        let e = edges(&g);
        assert_eq!(e.len(), 2, "{e:?}");
        assert!(!e.iter().any(|(s, _, _)| s == "Anna"));
    }

    #[tokio::test]
    async fn second_pass_failure_keeps_first_pass_data_and_the_build_succeeds() {
        let (url, log) = chat_stub();
        let g = build(&url, true, &["FAIL", "ALFA"]).await;
        // the failing chunk is retried by the client, so it can show more than one call
        let by_chunk = |m: &str| {
            log.lock()
                .unwrap()
                .iter()
                .filter(|b| b.contains(RELATION_MARKER) && b.contains(m))
                .count()
        };
        assert_eq!(by_chunk("ALFA"), 1);
        assert!(by_chunk("FAIL") >= 1);
        let graph = g.knowledge_graph().unwrap();
        // entities of the failed chunk are still in the graph
        assert!(graph.entities().any(|e| e.name == "Luca"));
        let e = edges(&g);
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!((e[0].0.as_str(), e[0].1.as_str()), ("Anna", "Bruno"));
    }

    #[tokio::test]
    async fn duplicate_from_second_pass_is_not_added_twice() {
        let (url, log) = chat_stub();
        let g = build(&url, true, &["DUPL"]).await;
        assert_eq!(relation_calls(&log), 1);
        let e = edges(&g);
        assert_eq!(e.len(), 2, "{e:?}");
        let graph = g.knowledge_graph().unwrap();
        let kept = graph
            .relationships()
            .find(|r| {
                graph.get_entity(&r.source).unwrap().name == "Gino"
                    && graph.get_entity(&r.target).unwrap().name == "Hana"
            })
            .unwrap();
        assert_eq!(kept.description.as_deref(), Some("first"));
    }
}
