#![allow(unused_imports)]

use crate::config::Config;
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use crate::{critic, ollama, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

use super::GraphRAG;

impl GraphRAG {

    /// Build the knowledge graph from added documents
    ///
    /// This method implements dynamic pipeline selection based on the configured approach:
    /// - **Semantic** (config.approach = "semantic"): Uses LLM-based entity extraction with gleaning
    ///   for high-quality results. Requires Ollama to be enabled.
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

        let chunks: Vec<_> = graph.chunks().cloned().collect();
        let total_chunks = chunks.len();

        // PHASE 1: Extract and add all entities
        // Pipeline selection based on config.approach (semantic/algorithmic/hybrid)
        // - Semantic: config.entities.use_gleaning = true (LLM-based with iterative refinement)
        // - Algorithmic: config.entities.use_gleaning = false (pattern-based extraction)
        // - Hybrid: config.entities.use_gleaning = true (uses LLM + pattern fusion)

        // DEBUG: Log current configuration state
        #[cfg(feature = "tracing")]
        tracing::info!(
            "build_graph() - Config state: approach='{}', use_gleaning={}, ollama.enabled={}",
            self.config.approach,
            self.config.entities.use_gleaning,
            self.config.ollama.enabled
        );

        if self.config.entities.use_gleaning && self.config.ollama.enabled {
            // LLM-based extraction with gleaning
            #[cfg(feature = "async")]
            {
                use crate::entity::GleaningEntityExtractor;
                use crate::ollama::OllamaClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM-based entity extraction with gleaning (max_rounds: {})",
                    self.config.entities.max_gleaning_rounds
                );

                // Create Ollama client
                let client = OllamaClient::new(self.config.ollama.clone());

                // Create gleaning config from our config
                let gleaning_config = crate::entity::GleaningConfig {
                    max_gleaning_rounds: self.config.entities.max_gleaning_rounds,
                    completion_threshold: 0.8,
                    entity_confidence_threshold: self.config.entities.min_confidence as f64,
                    use_llm_completion_check: true,
                    entity_types: if self.config.entities.entity_types.is_empty() {
                        vec![
                            "PERSON".to_string(),
                            "ORGANIZATION".to_string(),
                            "LOCATION".to_string(),
                        ]
                    } else {
                        self.config.entities.entity_types.clone()
                    },
                    temperature: 0.1,
                    max_tokens: 1500,
                };

                // Create gleaning extractor with LLM client
                let extractor = GleaningEntityExtractor::new(client.clone(), gleaning_config);

                // Create relationship extractor for triple validation (if enabled)
                let rel_extractor = if self.config.entities.enable_triple_reflection {
                    Some(crate::entity::LLMRelationshipExtractor::new(Some(
                        &self.config.ollama,
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

                // Extract entities using async gleaning
                for (idx, chunk) in chunks.iter().enumerate() {
                    pb.set_message(format!(
                        "Chunk {}/{} (gleaning with {} rounds)",
                        idx + 1,
                        total_chunks,
                        self.config.entities.max_gleaning_rounds
                    ));

                    #[cfg(feature = "tracing")]
                    tracing::info!("Processing chunk {}/{} (LLM)", idx + 1, total_chunks);

                    let (entities, relationships) = extractor.extract_with_gleaning(chunk).await?;

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

                    pb.inc(1);
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
                        "Atomic extraction complete: {} facts → {} entities, {} relationships",
                        total_facts, total_atomic_entities, total_atomic_relationships
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
        } else if self.config.ollama.enabled {
            // LLM single-pass extraction (Ollama enabled, gleaning disabled)
            //
            // Uses LLMEntityExtractor directly for one extraction round per chunk.
            // num_ctx is calculated dynamically from the built prompt + 20% margin,
            // and keep_alive is forwarded so Ollama preserves the KV cache between chunks.
            #[cfg(feature = "async")]
            {
                use crate::entity::llm_extractor::LLMEntityExtractor;
                use crate::ollama::OllamaClient;

                #[cfg(feature = "tracing")]
                tracing::info!(
                    "Using LLM single-pass entity extraction (no gleaning, keep_alive={:?})",
                    self.config.ollama.keep_alive,
                );

                let client = OllamaClient::new(self.config.ollama.clone());
                let entity_types = if self.config.entities.entity_types.is_empty() {
                    vec![
                        "PERSON".to_string(),
                        "ORGANIZATION".to_string(),
                        "LOCATION".to_string(),
                    ]
                } else {
                    self.config.entities.entity_types.clone()
                };

                let extractor = LLMEntityExtractor::new(client, entity_types)
                    .with_temperature(self.config.ollama.temperature.unwrap_or(0.1))
                    .with_max_tokens(self.config.ollama.max_tokens.unwrap_or(1500) as usize)
                    .with_keep_alive(self.config.ollama.keep_alive.clone());

                let pb = make_pb(total_chunks as u64,
                    ProgressStyle::default_bar()
                        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                        .expect("Invalid progress bar template")
                        .progress_chars("=>-"),
                );
                pb.set_message("Extracting entities with LLM (single-pass)");

                for (idx, chunk) in chunks.iter().enumerate() {
                    pb.set_message(format!(
                        "Chunk {}/{} (LLM single-pass)",
                        idx + 1,
                        total_chunks
                    ));

                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        "Processing chunk {}/{} (LLM single-pass)",
                        idx + 1,
                        total_chunks
                    );

                    match extractor.extract_from_chunk(chunk).await {
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
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk.id,
                                error = %e,
                                "LLM extraction failed for chunk, skipping"
                            );
                        },
                    }

                    pb.inc(1);
                }

                pb.finish_with_message("LLM single-pass extraction complete");
            }
        } else if self.config.gliner.enabled {
            // GLiNER-Relex joint NER + RE extraction
            //
            // gline-rs is synchronous (ONNX Runtime blocks the calling thread),
            // so we wrap each chunk in `spawn_blocking` to avoid stalling the
            // Tokio runtime.  A new `GLiNERExtractor` (with lazy model loading)
            // is created once outside the loop; the `Arc` inside it makes it
            // cheaply cloneable across blocking tasks.
            #[cfg(feature = "gliner")]
            {
                use crate::entity::GLiNERExtractor;
                use std::sync::Arc;

                let extractor = Arc::new(
                    GLiNERExtractor::new(self.config.gliner.clone()).map_err(|e| {
                        crate::core::error::GraphRAGError::EntityExtraction {
                            message: format!("GLiNER init failed: {e}"),
                        }
                    })?,
                );

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
                let parallelism = self
                    .config
                    .gliner
                    .max_concurrent_chunks
                    .unwrap_or(4)
                    .max(1);

                use futures::stream::{self, StreamExt};
                let mut stream = stream::iter(chunks.iter().cloned())
                    .map(|chunk| {
                        let ext = Arc::clone(&extractor);
                        let chunk_id = chunk.id.clone();
                        async move {
                            let r = tokio::task::spawn_blocking(move || {
                                ext.extract_from_chunk(&chunk)
                            })
                            .await;
                            (chunk_id, r)
                        }
                    })
                    .buffer_unordered(parallelism);

                while let Some((chunk_id, join_result)) = stream.next().await {
                    let result = join_result.map_err(|e| {
                        crate::core::error::GraphRAGError::EntityExtraction {
                            message: format!("spawn_blocking join error: {e}"),
                        }
                    })?;
                    match result {
                        Ok((entities, relationships)) => {
                            for entity in entities {
                                if let Err(e) = graph.add_entity(entity) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!("GLiNER: failed to add entity: {}", e);
                                }
                            }
                            for rel in relationships {
                                if let Err(e) = graph.add_relationship(rel) {
                                    #[cfg(feature = "tracing")]
                                    tracing::debug!(
                                        "GLiNER: failed to add relationship: {}",
                                        e
                                    );
                                }
                            }
                        },
                        Err(e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                chunk_id = %chunk_id,
                                error = %e,
                                "GLiNER extraction failed for chunk, skipping"
                            );
                        },
                    }
                    pb.inc(1);
                }

                pb.finish_with_message("GLiNER-Relex extraction complete");
            }
            #[cfg(not(feature = "gliner"))]
            return Err(crate::core::error::GraphRAGError::Config {
                message: "GLiNER enabled in config but crate compiled without --features gliner"
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

        // Persist to workspace if storage is configured
        self.save_to_workspace()?;

        Ok(())
    }

    /// Build the knowledge graph from added documents (synchronous fallback)
    ///
    /// This is a synchronous version for when the async feature is not enabled.
    /// Only supports pattern-based entity extraction.
    #[cfg(not(feature = "async"))]
    pub fn build_graph(&mut self) -> Result<()> {
        use crate::entity::EntityExtractor;

        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let chunks: Vec<_> = graph.chunks().cloned().collect();

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

        Ok(())
    }
}
