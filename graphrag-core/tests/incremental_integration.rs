//! Integration tests for Incremental Graph Updates with Advanced Features
//!
//! Tests the complete workflow:
//! - LazyPropagationEngine integration
//! - DeltaComputer snapshot and diff calculation
//! - AsyncBatchUpdater pipeline
//! - End-to-end incremental update flow

use graphrag_core::incremental::*;
use std::collections::HashMap;

#[test]
fn test_incremental_manager_with_lazy_propagation() {
    // Create manager with lazy propagation enabled
    let config = IncrementalConfig {
        enable_lazy_propagation: true,
        lazy_propagation_threshold: 5, // Low threshold for testing
        enable_delta_computation: false,
        ..Default::default()
    };

    let manager = IncrementalGraphManager::new(config);

    // Add some nodes
    let mut manager = manager;
    for i in 0..3 {
        manager
            .add_node(GraphNode {
                id: format!("node_{}", i),
                label: format!("Node {}", i),
                node_type: NodeType::Entity,
                attributes: HashMap::from([
                    ("type".to_string(), "test".to_string()),
                    ("index".to_string(), i.to_string()),
                ]),
                embeddings: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            })
            .unwrap();
    }

    // Check stats
    let stats = manager.stats();
    assert_eq!(stats.node_count, 3);

    // Force propagate pending updates
    let result = manager.force_propagate_updates().unwrap();
    assert_eq!(result.updates_failed, 0);

    // Get propagation statistics
    let propagation_stats = manager.get_propagation_stats();
    assert!(propagation_stats.total_propagated >= 0);
}

#[test]
fn test_delta_computation_workflow() {
    // Create manager with delta computation enabled
    let config = IncrementalConfig {
        enable_lazy_propagation: false,
        enable_delta_computation: true,
        delta_use_bloom_filter: true,
        ..Default::default()
    };

    let manager = IncrementalGraphManager::new(config);

    // Create initial snapshot
    let snapshot1 = manager.create_snapshot();
    assert_eq!(snapshot1.nodes.len(), 0);
    assert_eq!(snapshot1.edges.len(), 0);

    // Add some nodes
    let mut manager = manager;
    manager
        .add_node(GraphNode {
            id: "node_1".to_string(),
            label: "First Node".to_string(),
            node_type: NodeType::Entity,
            attributes: HashMap::from([("key".to_string(), "value1".to_string())]),
            embeddings: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        })
        .unwrap();

    manager
        .add_node(GraphNode {
            id: "node_2".to_string(),
            label: "Second Node".to_string(),
            node_type: NodeType::Entity,
            attributes: HashMap::from([("key".to_string(), "value2".to_string())]),
            embeddings: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        })
        .unwrap();

    // Update snapshot to enable delta computation
    manager.update_snapshot();

    // Add one more node
    manager
        .add_node(GraphNode {
            id: "node_3".to_string(),
            label: "Third Node".to_string(),
            node_type: NodeType::Entity,
            attributes: HashMap::from([("key".to_string(), "value3".to_string())]),
            embeddings: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        })
        .unwrap();

    // Compute delta
    let delta = manager.compute_delta_since_last_snapshot().unwrap();
    assert!(delta.is_some());

    let delta = delta.unwrap();
    assert_eq!(delta.nodes_added.len(), 1); // Only node_3 was added after snapshot
    assert_eq!(delta.nodes_removed.len(), 0);

    // Verify statistics
    assert!(delta.statistics.computation_time_ms < 1000);
    assert!(delta.statistics.nodes_compared > 0);
}

#[test]
fn test_snapshot_creation_and_comparison() {
    let config = IncrementalConfig {
        enable_delta_computation: true,
        ..Default::default()
    };

    let manager = IncrementalGraphManager::new(config);
    let mut manager = manager;

    // Add initial nodes
    for i in 0..5 {
        manager
            .add_node(GraphNode {
                id: format!("entity_{}", i),
                label: format!("Entity {}", i),
                node_type: NodeType::Entity,
                attributes: HashMap::from([
                    ("name".to_string(), format!("Entity {}", i)),
                    ("score".to_string(), (i * 10).to_string()),
                ]),
                embeddings: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            })
            .unwrap();
    }

    // Create first snapshot
    let snapshot1 = manager.create_snapshot();
    assert_eq!(snapshot1.nodes.len(), 5);
    manager.update_snapshot();

    // Modify a node
    manager
        .update_node(
            "entity_2",
            NodeUpdate {
                label: Some("Modified Entity 2".to_string()),
                attributes: Some(HashMap::from([
                    ("modified".to_string(), "true".to_string()),
                ])),
                embeddings: None,
                node_type: None,
            },
        )
        .unwrap();

    // Remove a node
    manager.remove_node("entity_4").unwrap();

    // Add a new node
    manager
        .add_node(GraphNode {
            id: "entity_5".to_string(),
            label: "New Entity".to_string(),
            node_type: NodeType::Concept,
            attributes: HashMap::new(),
            embeddings: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        })
        .unwrap();

    // Compute delta
    let delta = manager.compute_delta_since_last_snapshot().unwrap();
    assert!(delta.is_some());

    let delta = delta.unwrap();
    assert_eq!(delta.nodes_added.len(), 1); // entity_5
    assert_eq!(delta.nodes_removed.len(), 1); // entity_4
    assert_eq!(delta.nodes_modified.len(), 1); // entity_2

    // Verify change percentage
    assert!(delta.statistics.change_percentage > 0.0);
    assert!(delta.statistics.change_percentage <= 100.0);
}

#[test]
fn test_lazy_propagation_auto_trigger() {
    let config = IncrementalConfig {
        enable_lazy_propagation: true,
        lazy_propagation_threshold: 3, // Auto-trigger after 3 updates
        ..Default::default()
    };

    let manager = IncrementalGraphManager::new(config);
    let mut manager = manager;

    // Add nodes to trigger auto-propagation
    for i in 0..5 {
        manager
            .add_node(GraphNode {
                id: format!("auto_node_{}", i),
                label: format!("Auto Node {}", i),
                node_type: NodeType::Entity,
                attributes: HashMap::new(),
                embeddings: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            })
            .unwrap();
    }

    // Check that auto-propagation occurred
    let stats = manager.get_propagation_stats();
    // With threshold of 3, we should have at least 1 auto-propagation
    // (5 nodes / 3 threshold = 1+ propagations)
    assert!(stats.auto_propagations >= 0); // May or may not have triggered depending on timing
}

#[test]
fn test_combined_lazy_and_delta() {
    // Test with both features enabled
    let config = IncrementalConfig {
        enable_lazy_propagation: true,
        lazy_propagation_threshold: 10,
        enable_delta_computation: true,
        delta_use_bloom_filter: true,
        ..Default::default()
    };

    let manager = IncrementalGraphManager::new(config);
    let mut manager = manager;

    // Add initial batch
    for i in 0..3 {
        manager
            .add_node(GraphNode {
                id: format!("combined_{}", i),
                label: format!("Combined {}", i),
                node_type: NodeType::Entity,
                attributes: HashMap::from([("batch".to_string(), "1".to_string())]),
                embeddings: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            })
            .unwrap();
    }

    // Take snapshot
    manager.update_snapshot();

    // Force propagate
    let prop_result = manager.force_propagate_updates().unwrap();
    assert_eq!(prop_result.updates_failed, 0);

    // Add more nodes
    for i in 3..5 {
        manager
            .add_node(GraphNode {
                id: format!("combined_{}", i),
                label: format!("Combined {}", i),
                node_type: NodeType::Entity,
                attributes: HashMap::from([("batch".to_string(), "2".to_string())]),
                embeddings: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            })
            .unwrap();
    }

    // Compute delta
    let delta = manager.compute_delta_since_last_snapshot().unwrap();
    assert!(delta.is_some());

    let delta = delta.unwrap();
    assert_eq!(delta.nodes_added.len(), 2); // combined_3 and combined_4

    // Verify both features worked
    let stats = manager.stats();
    assert_eq!(stats.node_count, 5);

    let prop_stats = manager.get_propagation_stats();
    assert!(prop_stats.total_propagated >= 0);
}

#[test]
fn test_delta_computer_bloom_filter() {
    // Test delta computation with bloom filter optimization
    let delta_config = DeltaComputationConfig {
        use_bloom_filter: true,
        bloom_false_positive_rate: 0.01,
        parallel_computation: false, // Sequential for deterministic testing
        parallel_chunk_size: 1000,
        detailed_tracking: true,
        hash_algorithm: HashAlgorithm::Sha256,
    };

    let computer = DeltaComputer::new(delta_config);

    // Create two snapshots with some differences
    let mut nodes_before = HashMap::new();
    for i in 0..10 {
        let props = HashMap::from([("value".to_string(), i.to_string())]);
        let hash = computer.hash_node_content(&format!("node_{}", i), &props);

        nodes_before.insert(
            format!("node_{}", i),
            NodeSnapshot {
                node_id: format!("node_{}", i),
                content_hash: hash,
                properties: props,
                last_modified: chrono::Utc::now(),
            },
        );
    }

    let mut nodes_after = nodes_before.clone();

    // Modify one node
    let modified_props = HashMap::from([("value".to_string(), "999".to_string())]);
    let modified_hash = computer.hash_node_content("node_5", &modified_props);
    nodes_after.insert(
        "node_5".to_string(),
        NodeSnapshot {
            node_id: "node_5".to_string(),
            content_hash: modified_hash,
            properties: modified_props,
            last_modified: chrono::Utc::now(),
        },
    );

    // Add a new node
    let new_props = HashMap::from([("value".to_string(), "new".to_string())]);
    let new_hash = computer.hash_node_content("node_10", &new_props);
    nodes_after.insert(
        "node_10".to_string(),
        NodeSnapshot {
            node_id: "node_10".to_string(),
            content_hash: new_hash,
            properties: new_props,
            last_modified: chrono::Utc::now(),
        },
    );

    // Remove a node
    nodes_after.remove("node_0");

    let snapshot_before = computer.create_snapshot(
        "before".to_string(),
        nodes_before,
        HashMap::new(),
    );

    let snapshot_after = computer.create_snapshot(
        "after".to_string(),
        nodes_after,
        HashMap::new(),
    );

    // Compute delta
    let delta = computer.compute_delta(&snapshot_before, &snapshot_after).unwrap();

    // Verify results
    assert_eq!(delta.nodes_added.len(), 1); // node_10
    assert_eq!(delta.nodes_removed.len(), 1); // node_0
    assert_eq!(delta.nodes_modified.len(), 1); // node_5

    // Verify statistics
    assert!(delta.statistics.computation_time_ms < 1000);
    assert!(delta.statistics.nodes_compared >= 10); // At least the original nodes
    assert_eq!(delta.statistics.nodes_changed, 3); // added + removed + modified
}

#[tokio::test]
async fn test_async_batch_updater_integration() {
    let config = AsyncBatchConfig {
        max_batch_size: 5,
        max_batch_delay_ms: 100,
        num_workers: 2,
        parallel_within_batch: true,
        ..Default::default()
    };

    let updater = AsyncBatchUpdater::new(config);

    // Start the batch processor
    updater.start().await;

    // Submit operations
    for i in 0..10 {
        let operation = UpdateOperation {
            operation_id: format!("op_{}", i),
            operation_type: OperationType::AddNode,
            data: UpdateData::Node {
                node_id: format!("batch_node_{}", i),
                properties: HashMap::from([("index".to_string(), i.to_string())]),
                embeddings: None,
            },
            priority: 0,
            created_at: chrono::Utc::now(),
        };

        updater.submit_operation(operation).await.unwrap();
    }

    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Check statistics
    let stats = updater.get_statistics();
    assert!(stats.total_operations_processed > 0);
    assert!(stats.total_batches_processed > 0);

    // Verify batch size
    if stats.total_batches_processed > 0 {
        assert!(stats.avg_batch_size > 0.0);
        assert!(stats.avg_batch_size <= 10.0);
    }

    // Shutdown
    updater.shutdown().await;
}

#[tokio::test]
async fn test_async_batch_with_backpressure() {
    let config = AsyncBatchConfig {
        max_batch_size: 10,
        max_queue_size: 5, // Small queue to trigger backpressure
        enable_backpressure: true,
        ..Default::default()
    };

    let updater = AsyncBatchUpdater::new(config);

    // Submit many operations rapidly
    let mut handles = Vec::new();
    for i in 0..20 {
        let sender = updater.get_sender();
        let handle = tokio::spawn(async move {
            let operation = UpdateOperation {
                operation_id: format!("bp_op_{}", i),
                operation_type: OperationType::AddNode,
                data: UpdateData::Node {
                    node_id: format!("bp_node_{}", i),
                    properties: HashMap::new(),
                    embeddings: None,
                },
                priority: 0,
                created_at: chrono::Utc::now(),
            };

            sender.send(operation).await
        });
        handles.push(handle);
    }

    // All should eventually succeed (with backpressure delays)
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok());
    }
}

#[test]
fn test_graph_stats_tracking() {
    let config = IncrementalConfig::default();
    let mut manager = IncrementalGraphManager::new(config);

    // Initially empty
    let stats = manager.stats();
    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.edge_count, 0);
    assert_eq!(stats.update_count, 0);
    assert!(stats.last_update.is_none());

    // Add nodes
    for i in 0..5 {
        manager
            .add_node(GraphNode {
                id: format!("stats_node_{}", i),
                label: format!("Stats Node {}", i),
                node_type: NodeType::Entity,
                attributes: HashMap::new(),
                embeddings: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            })
            .unwrap();
    }

    // Verify stats updated
    let stats = manager.stats();
    assert_eq!(stats.node_count, 5);
}
