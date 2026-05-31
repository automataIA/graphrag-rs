//! Comprehensive incremental updates architecture for GraphRAG-RS
//!
//! This module provides zero-downtime incremental updates with ACID-like guarantees,
//! intelligent cache invalidation, conflict resolution, and comprehensive monitoring.
//!
//! ## Architecture Goals
//!
//! - **Zero-downtime updates**: System remains available during modifications
//! - **Consistency guarantees**: ACID-like properties for graph operations
//! - **Performance**: Updates should be 10x+ faster than full reconstruction
//! - **Scalability**: Handle thousands of concurrent updates per second
//! - **Observability**: Complete audit trail of all changes
//!
//! ## Key Components
//!
//! - `IncrementalGraphStore` trait for atomic update operations
//! - `ChangeRecord` and `ChangeLog` for tracking modifications
//! - `GraphDelta` for representing atomic change sets
//! - `ConflictResolver` for handling concurrent modifications
//! - `SelectiveInvalidation` for cache management
//! - `UpdateMonitor` for change tracking and metrics
//! - `IncrementalPageRank` for efficient graph algorithm updates
//!
//! Phase 4 file split: this previously 2905-LOC monolithic file is now a directory
//! module with focused sub-files (`types`, `helpers`, `manager`, `store`). Public
//! items are re-exported here so existing paths (`crate::graph::incremental::*`)
//! resolve unchanged.

mod helpers;
mod manager;
mod store;
mod types;

pub use helpers::*;
pub use manager::*;
pub use store::*;
pub use types::*;

// Imports surfaced for the inline `mod tests` below (which uses `super::*` to
// glob-pull both the re-exported items above and these external types — the
// same way the original monolithic file's tests resolved them).
#[cfg(test)]
#[allow(unused_imports)]
use {
    crate::core::{Entity, EntityId, KnowledgeGraph, Relationship},
    chrono::Utc,
    std::collections::{HashMap, HashSet},
    std::sync::Arc,
    std::time::Duration,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_id_generation() {
        let id1 = UpdateId::new();
        let id2 = UpdateId::new();
        assert_ne!(id1.as_str(), id2.as_str());
    }

    #[test]
    fn test_transaction_id_generation() {
        let tx1 = TransactionId::new();
        let tx2 = TransactionId::new();
        assert_ne!(tx1.as_str(), tx2.as_str());
    }

    #[test]
    fn test_change_record_creation() {
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        let config = IncrementalConfig::default();
        let graph = KnowledgeGraph::new();
        let manager = IncrementalGraphManager::new(graph, config);

        let change = manager.create_change_record(
            ChangeType::EntityAdded,
            Operation::Insert,
            ChangeData::Entity(entity.clone()),
            Some(entity.id.clone()),
            None,
        );

        assert_eq!(change.change_type, ChangeType::EntityAdded);
        assert_eq!(change.operation, Operation::Insert);
        assert_eq!(change.entity_id, Some(entity.id));
    }

    #[test]
    fn test_conflict_resolver_creation() {
        let resolver = ConflictResolver::new(ConflictStrategy::KeepExisting);
        assert!(matches!(resolver.strategy, ConflictStrategy::KeepExisting));
    }

    #[test]
    fn test_incremental_config_default() {
        let config = IncrementalConfig::default();
        assert_eq!(config.max_change_log_size, 10000);
        assert_eq!(config.batch_size, 100);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_statistics_creation() {
        let stats = IncrementalStatistics::empty();
        assert_eq!(stats.total_updates, 0);
        assert_eq!(stats.entities_added, 0);
        assert_eq!(stats.average_update_time_ms, 0.0);
    }

    #[cfg(feature = "incremental")]

    #[cfg(feature = "incremental")]
    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(100, Duration::from_millis(500), 10);
        let metrics = processor.get_metrics();
        assert_eq!(metrics.total_batches_processed, 0);
    }

    #[cfg(feature = "incremental")]
    #[tokio::test]
    async fn test_selective_invalidation() {
        let invalidation = SelectiveInvalidation::new();

        let region = CacheRegion {
            region_id: "test_region".to_string(),
            entity_ids: [EntityId::new("entity1".to_string())].into_iter().collect(),
            relationship_types: ["KNOWS".to_string()].into_iter().collect(),
            document_ids: HashSet::new(),
            last_modified: Utc::now(),
        };

        invalidation.register_cache_region(region);

        let entity = Entity::new(
            EntityId::new("entity1".to_string()),
            "Entity 1".to_string(),
            "Person".to_string(),
            0.9,
        );

        let ent_id_for_log = entity.id.clone();
        let change = ChangeRecord {
            change_id: UpdateId::new(),
            timestamp: Utc::now(),
            change_type: ChangeType::EntityUpdated,
            entity_id: Some(ent_id_for_log),
            document_id: None,
            operation: Operation::Update,
            data: ChangeData::Entity(entity),
            metadata: HashMap::new(),
        };

        let strategies = invalidation.invalidate_for_changes(&[change]);
        assert!(!strategies.is_empty());
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_conflict_resolver_merge() {
        let resolver = ConflictResolver::new(ConflictStrategy::Merge);

        let entity1 = Entity::new(
            EntityId::new("entity1".to_string()),
            "Entity 1".to_string(),
            "Person".to_string(),
            0.8,
        );

        let entity2 = Entity::new(
            EntityId::new("entity1".to_string()),
            "Entity 1 Updated".to_string(),
            "Person".to_string(),
            0.9,
        );

        let merged = resolver.merge_entities(&entity1, &entity2).unwrap();
        assert_eq!(merged.confidence, 0.9); // Should take higher confidence
        assert_eq!(merged.name, "Entity 1 Updated");
    }

    #[test]
    fn test_graph_statistics_creation() {
        let stats = GraphStatistics {
            node_count: 100,
            edge_count: 150,
            average_degree: 3.0,
            max_degree: 10,
            connected_components: 1,
            clustering_coefficient: 0.3,
            last_updated: Utc::now(),
        };

        assert_eq!(stats.node_count, 100);
        assert_eq!(stats.edge_count, 150);
    }

    #[test]
    fn test_consistency_report_creation() {
        let report = ConsistencyReport {
            is_consistent: true,
            orphaned_entities: vec![],
            broken_relationships: vec![],
            missing_embeddings: vec![],
            validation_time: Utc::now(),
            issues_found: 0,
        };

        assert!(report.is_consistent);
        assert_eq!(report.issues_found, 0);
    }

    #[test]
    fn test_change_event_creation() {
        let event = ChangeEvent {
            event_id: UpdateId::new(),
            event_type: ChangeEventType::EntityUpserted,
            entity_id: Some(EntityId::new("entity1".to_string())),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        assert!(matches!(event.event_type, ChangeEventType::EntityUpserted));
        assert!(event.entity_id.is_some());
    }
}
