//! Complete Persistence Test
//!
//! Tests that entities and relationships are properly saved and loaded

use wasm_bindgen_test::*;
use graphrag_wasm::GraphRAG;

wasm_bindgen_test_configure!(run_in_browser);

/// Test: Save and load complete knowledge graph with entities and relationships
#[wasm_bindgen_test]
async fn test_complete_persistence() {
    // Create graph with all data
    let mut graph = GraphRAG::new(384).unwrap();

    // Add documents
    let doc1_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    graph.add_document(
        "doc1".to_string(),
        "GraphRAG combines knowledge graphs and RAG".to_string(),
        doc1_embedding
    ).await.unwrap();

    // Build index
    graph.build_index().await.unwrap();

    // Manually add entities and relationships (simulating extraction)
    use graphrag_wasm::entity_extractor::{Entity, Relationship};

    let entities = vec![
        Entity {
            name: "GraphRAG".to_string(),
            entity_type: "TECHNOLOGY".to_string(),
            description: "Knowledge graph RAG system".to_string(),
        },
        Entity {
            name: "RAG".to_string(),
            entity_type: "CONCEPT".to_string(),
            description: "Retrieval-Augmented Generation".to_string(),
        },
    ];

    let relationships = vec![
        Relationship {
            from: "GraphRAG".to_string(),
            relation: "USES".to_string(),
            to: "RAG".to_string(),
        },
    ];

    graph.add_entities(entities.clone());
    graph.add_relationships(relationships.clone());

    // Verify counts before save
    assert_eq!(graph.document_count(), 1);
    assert_eq!(graph.entity_count(), 2);
    assert_eq!(graph.relationship_count(), 1);

    // Save complete graph
    let db_name = "test-complete-persistence";
    graph.save_to_storage(db_name).await.unwrap();

    web_sys::console::log_1(&"✅ Saved complete graph".into());

    // Create new instance and load
    let mut loaded_graph = GraphRAG::new(384).unwrap();
    loaded_graph.load_from_storage(db_name).await.unwrap();

    web_sys::console::log_1(&"✅ Loaded complete graph".into());

    // Verify all data was loaded
    assert_eq!(loaded_graph.document_count(), 1, "Documents count mismatch");
    assert_eq!(loaded_graph.entity_count(), 2, "Entities count mismatch");
    assert_eq!(loaded_graph.relationship_count(), 1, "Relationships count mismatch");
    assert_eq!(loaded_graph.get_dimension(), 384, "Dimension mismatch");
    assert!(loaded_graph.is_index_built(), "Index should be rebuilt");

    // Verify entities content
    let loaded_entities = loaded_graph.entities();
    assert_eq!(loaded_entities.len(), 2);
    assert_eq!(loaded_entities[0].name, "GraphRAG");
    assert_eq!(loaded_entities[1].name, "RAG");

    // Verify relationships content
    let loaded_relationships = loaded_graph.relationships();
    assert_eq!(loaded_relationships.len(), 1);
    assert_eq!(loaded_relationships[0].from, "GraphRAG");
    assert_eq!(loaded_relationships[0].relation, "USES");
    assert_eq!(loaded_relationships[0].to, "RAG");

    web_sys::console::log_1(&"✅ All assertions passed!".into());

    // Test get_stats() method
    let stats = loaded_graph.get_stats();
    web_sys::console::log_1(&format!("📊 Stats: {}", stats).into());
    assert!(stats.contains("\"documents\":1"));
    assert!(stats.contains("\"entities\":2"));
    assert!(stats.contains("\"relationships\":1"));

    web_sys::console::log_1(&"🎉 Complete persistence test PASSED!".into());
}

/// Test: Backward compatibility - load old format without entities
#[wasm_bindgen_test]
async fn test_backward_compatibility() {
    // Create "old format" graph (no entities/relationships)
    let mut old_graph = GraphRAG::new(384).unwrap();

    let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    old_graph.add_document(
        "doc1".to_string(),
        "Old format document".to_string(),
        embedding
    ).await.unwrap();

    old_graph.build_index().await.unwrap();

    // Save without entities (simulating old version)
    let db_name = "test-backward-compat";

    // Use internal storage API to save only docs and embeddings (old format)
    use graphrag_wasm::storage::IndexedDBStore;
    let db = IndexedDBStore::new(db_name, 1).await.unwrap();
    db.put("documents", "all_docs", &vec!["Old format document".to_string()]).await.unwrap();
    db.put("metadata", "embeddings", &vec![vec![0.1f32; 384]]).await.unwrap();
    db.put("metadata", "dimension", &384usize).await.unwrap();

    // Try to load with new version
    let mut new_graph = GraphRAG::new(384).unwrap();
    let load_result = new_graph.load_from_storage(db_name).await;

    // Should succeed with empty entities/relationships
    assert!(load_result.is_ok(), "Should load old format successfully");
    assert_eq!(new_graph.document_count(), 1);
    assert_eq!(new_graph.entity_count(), 0, "Should have 0 entities (backward compat)");
    assert_eq!(new_graph.relationship_count(), 0, "Should have 0 relationships (backward compat)");

    web_sys::console::log_1(&"✅ Backward compatibility test PASSED!".into());
}

/// Test: Get entities and relationships as JSON
#[wasm_bindgen_test]
async fn test_json_export() {
    let mut graph = GraphRAG::new(384).unwrap();

    // Add entities
    use graphrag_wasm::entity_extractor::{Entity, Relationship};
    graph.add_entities(vec![
        Entity {
            name: "Test Entity".to_string(),
            entity_type: "CONCEPT".to_string(),
            description: "Test".to_string(),
        },
    ]);

    graph.add_relationships(vec![
        Relationship {
            from: "A".to_string(),
            relation: "RELATES_TO".to_string(),
            to: "B".to_string(),
        },
    ]);

    // Get as JSON
    let entities_json = graph.get_entities_json().unwrap();
    let relationships_json = graph.get_relationships_json().unwrap();

    web_sys::console::log_1(&format!("Entities JSON: {}", entities_json).into());
    web_sys::console::log_1(&format!("Relationships JSON: {}", relationships_json).into());

    // Should be valid JSON
    assert!(entities_json.contains("Test Entity"));
    assert!(relationships_json.contains("RELATES_TO"));

    web_sys::console::log_1(&"✅ JSON export test PASSED!".into());
}
