//! Conversions between graphrag-core's in-memory `Entity`/`Relationship`
//! types and the wire-format envelopes (`PersistedEntity`/
//! `PersistedRelationship`) we store in Qdrant for cross-restart
//! survival.
//!
//! These helpers exist so the call sites in `main.rs` (build_graph,
//! append_graph) and `config_endpoints.rs` (set_config hydration)
//! don't have to know the serde plumbing.

#[cfg(feature = "qdrant")]
use crate::qdrant_store::{PersistedEntity, PersistedRelationship, QdrantError, QdrantStore};
#[cfg(feature = "qdrant")]
use graphrag_core::core::{Entity, Relationship};

/// Current persistence schema version. Bump when the wire envelope
/// changes incompatibly so loaders can migrate or drop old payloads.
const SCHEMA_VERSION: u32 = 1;

/// Convert a graphrag-core `Entity` into the wire envelope.
#[cfg(feature = "qdrant")]
pub fn entity_to_persisted(entity: &Entity) -> PersistedEntity {
    PersistedEntity {
        schema_version: SCHEMA_VERSION,
        id: entity.id.0.clone(),
        name: entity.name.clone(),
        entity_type: entity.entity_type.clone(),
        entity_json: serde_json::to_value(entity).unwrap_or(serde_json::json!({})),
    }
}

/// Convert a graphrag-core `Relationship` into the wire envelope.
#[cfg(feature = "qdrant")]
pub fn relationship_to_persisted(rel: &Relationship) -> PersistedRelationship {
    PersistedRelationship {
        schema_version: SCHEMA_VERSION,
        source: rel.source.0.clone(),
        target: rel.target.0.clone(),
        relation_type: rel.relation_type.clone(),
        relationship_json: serde_json::to_value(rel).unwrap_or(serde_json::json!({})),
    }
}

/// Round-trip a wire envelope back into a graphrag-core `Entity`.
/// Returns `None` if the stored JSON is malformed (e.g. schema drift
/// across versions); callers should log and skip rather than fail
/// hydration outright.
#[cfg(feature = "qdrant")]
pub fn persisted_to_entity(p: &PersistedEntity) -> Option<Entity> {
    serde_json::from_value(p.entity_json.clone()).ok()
}

/// Round-trip a wire envelope back into a graphrag-core `Relationship`.
#[cfg(feature = "qdrant")]
pub fn persisted_to_relationship(p: &PersistedRelationship) -> Option<Relationship> {
    serde_json::from_value(p.relationship_json.clone()).ok()
}

/// Convenience: dump every entity + relationship currently in `graphrag`
/// to Qdrant, with a description embedding per row. Idempotent (the
/// underlying `QdrantStore::persist_graph` clear-and-repopulates).
/// Returns `(entities_persisted, relationships_persisted)` for telemetry.
///
/// Embedding strategy mirrors Microsoft GraphRAG's `description_embedding`
/// convention: each entity is embedded as `"{name} ({entity_type})"`;
/// each relationship as `"{source_name} {relation_type} {target_name}"`.
/// Reuses `Entity.embedding` / `Relationship.embedding` if already
/// populated by the extractor (saves a round-trip); otherwise batches
/// through the supplied `EmbeddingService` (the same service the
/// document path uses, so vectors live in one consistent space).
///
/// Safe to call from inside a write-locked GraphRAG: only does
/// Embedding + Qdrant I/O, doesn't mutate the in-memory state.
#[cfg(feature = "qdrant")]
pub async fn persist_in_memory_graph(
    graphrag: &graphrag_core::GraphRAG,
    qdrant: &QdrantStore,
    embeddings: &crate::embeddings::EmbeddingService,
) -> Result<(usize, usize), QdrantError> {
    let Some(kg) = graphrag.knowledge_graph() else {
        return Ok((0, 0));
    };

    let dim = embeddings.dimension() as u64;

    // ---- Entities ----------------------------------------------------
    let entities_with_text: Vec<(graphrag_core::core::Entity, String)> = kg
        .entities()
        .map(|e| {
            let text = format!("{} ({})", e.name, e.entity_type);
            (e.clone(), text)
        })
        .collect();

    // Entities already carrying an embedding from the extractor pass
    // skip the round-trip; collect the rest into a batch call.
    let (texts_to_embed, indices_to_embed): (Vec<String>, Vec<usize>) = entities_with_text
        .iter()
        .enumerate()
        .filter_map(|(i, (e, t))| {
            if e.embedding.is_some() {
                None
            } else {
                Some((t.clone(), i))
            }
        })
        .unzip();

    let mut entity_embeddings: Vec<Option<Vec<f32>>> =
        entities_with_text.iter().map(|(e, _)| e.embedding.clone()).collect();

    if !texts_to_embed.is_empty() {
        let refs: Vec<&str> = texts_to_embed.iter().map(String::as_str).collect();
        let computed = embeddings
            .generate(&refs)
            .await
            .map_err(|e| QdrantError::OperationError(format!("entity embed failed: {}", e)))?;
        for (i, vec) in indices_to_embed.iter().zip(computed) {
            entity_embeddings[*i] = Some(vec);
        }
    }

    let entity_payloads_with_vec: Vec<(PersistedEntity, Vec<f32>)> = entities_with_text
        .iter()
        .zip(entity_embeddings.iter())
        .map(|((e, _), emb)| {
            let vec = emb.clone().unwrap_or_else(|| vec![0.0_f32; dim as usize]);
            (entity_to_persisted(e), vec)
        })
        .collect();

    // ---- Relationships ----------------------------------------------
    let relationships_with_text: Vec<(graphrag_core::core::Relationship, String)> = kg
        .relationships()
        .map(|r| {
            let src_name = kg
                .get_entity(&r.source)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let tgt_name = kg
                .get_entity(&r.target)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let text = format!("{} {} {}", src_name, r.relation_type, tgt_name);
            (r.clone(), text)
        })
        .collect();

    let (rel_texts, rel_indices): (Vec<String>, Vec<usize>) = relationships_with_text
        .iter()
        .enumerate()
        .filter_map(|(i, (r, t))| {
            if r.embedding.is_some() {
                None
            } else {
                Some((t.clone(), i))
            }
        })
        .unzip();

    let mut rel_embeddings: Vec<Option<Vec<f32>>> = relationships_with_text
        .iter()
        .map(|(r, _)| r.embedding.clone())
        .collect();

    if !rel_texts.is_empty() {
        let refs: Vec<&str> = rel_texts.iter().map(String::as_str).collect();
        let computed = embeddings.generate(&refs).await.map_err(|e| {
            QdrantError::OperationError(format!("relationship embed failed: {}", e))
        })?;
        for (i, vec) in rel_indices.iter().zip(computed) {
            rel_embeddings[*i] = Some(vec);
        }
    }

    let rel_payloads_with_vec: Vec<(PersistedRelationship, Vec<f32>)> = relationships_with_text
        .iter()
        .zip(rel_embeddings.iter())
        .map(|((r, _), emb)| {
            let vec = emb.clone().unwrap_or_else(|| vec![0.0_f32; dim as usize]);
            (relationship_to_persisted(r), vec)
        })
        .collect();

    qdrant
        .persist_graph(entity_payloads_with_vec, rel_payloads_with_vec, dim)
        .await
}

/// Hydrate the in-memory KnowledgeGraph from Qdrant's persisted entities
/// + relationships. Order matters: entities go in first so each
/// relationship's `add_relationship` call finds its source/target.
///
/// Returns `(entities_restored, relationships_restored,
/// relationships_skipped_orphan)` for telemetry. Orphan-skip happens
/// when a stored relationship references an entity id that isn't
/// in the persisted entity set (e.g. deleted between persist + restore);
/// we log and drop the row rather than fail hydration.
#[cfg(feature = "qdrant")]
pub async fn hydrate_in_memory_graph(
    graphrag: &mut graphrag_core::GraphRAG,
    qdrant: &QdrantStore,
) -> Result<(usize, usize, usize), QdrantError> {
    let entities = qdrant.load_persisted_entities().await?;
    let relationships = qdrant.load_persisted_relationships().await?;

    let Some(kg) = graphrag.knowledge_graph_mut() else {
        return Ok((0, 0, 0));
    };

    let mut entities_restored = 0usize;
    for p in &entities {
        if let Some(entity) = persisted_to_entity(p) {
            // add_entity ignores duplicates if the id is already present?
            // No — it always adds a new node. To stay idempotent across
            // re-hydrations, check first.
            if kg.get_entity(&entity.id).is_none() {
                if kg.add_entity(entity).is_ok() {
                    entities_restored += 1;
                }
            }
        }
    }

    let mut relationships_restored = 0usize;
    let mut relationships_skipped = 0usize;
    for p in &relationships {
        if let Some(rel) = persisted_to_relationship(p) {
            if kg.get_entity(&rel.source).is_none() || kg.get_entity(&rel.target).is_none() {
                relationships_skipped += 1;
                continue;
            }
            if kg.add_relationship(rel).is_ok() {
                relationships_restored += 1;
            }
        }
    }

    Ok((entities_restored, relationships_restored, relationships_skipped))
}
