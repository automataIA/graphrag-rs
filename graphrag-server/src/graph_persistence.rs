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
/// to Qdrant. Idempotent (the underlying `QdrantStore::persist_graph`
/// clear-and-repopulates). Returns `(entities_persisted,
/// relationships_persisted)` for telemetry.
///
/// Safe to call from inside a write-locked GraphRAG: only does Qdrant
/// I/O, does not touch the in-memory state.
#[cfg(feature = "qdrant")]
pub async fn persist_in_memory_graph(
    graphrag: &graphrag_core::GraphRAG,
    qdrant: &QdrantStore,
) -> Result<(usize, usize), QdrantError> {
    let Some(kg) = graphrag.knowledge_graph() else {
        return Ok((0, 0));
    };
    let entities: Vec<PersistedEntity> = kg.entities().map(entity_to_persisted).collect();
    let relationships: Vec<PersistedRelationship> =
        kg.relationships().map(relationship_to_persisted).collect();
    qdrant.persist_graph(entities, relationships).await
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
