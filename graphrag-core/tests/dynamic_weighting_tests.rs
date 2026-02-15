//! Tests for Dynamic Edge Weighting (Phase 2.2)
//!
//! These tests verify that dynamic weights are calculated correctly based on query context.

use graphrag_core::{
    core::{Entity, EntityId, KnowledgeGraph, Relationship},
    graph::temporal::{TemporalRange, TemporalRelationType},
};

fn create_test_graph() -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::new();

    // Create test entities
    let socrates = Entity::new(
        EntityId::new("socrates".to_string()),
        "Socrates".to_string(),
        "PERSON".to_string(),
        0.9,
    );

    let plato = Entity::new(
        EntityId::new("plato".to_string()),
        "Plato".to_string(),
        "PERSON".to_string(),
        0.9,
    );

    let love = Entity::new(
        EntityId::new("love".to_string()),
        "love".to_string(),
        "CONCEPT".to_string(),
        0.85,
    );

    graph.add_entity(socrates).unwrap();
    graph.add_entity(plato).unwrap();
    graph.add_entity(love).unwrap();

    // Create relationships
    let taught_rel = Relationship::new(
        EntityId::new("socrates".to_string()),
        EntityId::new("plato".to_string()),
        "TAUGHT".to_string(),
        0.8,
    );

    let discussed_rel = Relationship::new(
        EntityId::new("socrates".to_string()),
        EntityId::new("love".to_string()),
        "DISCUSSED".to_string(),
        0.75,
    )
    .with_temporal_range(
        -380 * 365 * 24 * 3600, // 380 BC
        -380 * 365 * 24 * 3600,
    );

    graph.add_relationship(taught_rel).unwrap();
    graph.add_relationship(discussed_rel).unwrap();

    graph
}

#[test]
fn test_dynamic_weight_base_score() {
    let graph = create_test_graph();

    // Get a relationship
    let rels = graph.get_all_relationships();
    let rel = &rels[0];

    // Calculate dynamic weight with no query context
    let query_concepts: Vec<String> = vec![];
    let weight = graph.dynamic_weight(rel, None, &query_concepts);

    // With no context, weight should equal base confidence
    assert_eq!(weight, rel.confidence);
}

#[test]
fn test_dynamic_weight_conceptual_boost() {
    let graph = create_test_graph();

    // Find the TAUGHT relationship
    let taught_rel = graph
        .get_all_relationships()
        .iter()
        .find(|r| r.relation_type == "TAUGHT")
        .unwrap()
        .clone();

    // Query concepts that match the relationship type
    let query_concepts = vec!["taught".to_string(), "teaching".to_string()];

    let weight = graph.dynamic_weight(&taught_rel, None, &query_concepts);

    // Should have conceptual boost applied
    assert!(
        weight > taught_rel.confidence,
        "Expected weight {} to be greater than base confidence {}",
        weight,
        taught_rel.confidence
    );

    // Boost should be proportional to number of matching concepts
    let expected_boost = 1.0 + (1.0 * 0.15); // 1 match * 15% boost per match
    let expected_weight = taught_rel.confidence * expected_boost;

    assert!(
        (weight - expected_weight).abs() < 0.01,
        "Expected weight ~{} but got {}",
        expected_weight,
        weight
    );
}

#[test]
fn test_dynamic_weight_temporal_boost() {
    let graph = create_test_graph();

    // Find the DISCUSSED relationship (has temporal range)
    let discussed_rel = graph
        .get_all_relationships()
        .iter()
        .find(|r| r.relation_type == "DISCUSSED")
        .unwrap()
        .clone();

    let query_concepts: Vec<String> = vec![];
    let weight = graph.dynamic_weight(&discussed_rel, None, &query_concepts);

    // Should have temporal boost applied (ancient event gets small boost)
    // Note: Ancient events get lower boost than recent ones
    assert!(
        weight >= discussed_rel.confidence,
        "Expected weight {} to be >= base confidence {}",
        weight,
        discussed_rel.confidence
    );
}

#[test]
fn test_dynamic_weight_causal_boost() {
    let mut graph = KnowledgeGraph::new();

    // Create entities
    let cause = Entity::new(
        EntityId::new("event1".to_string()),
        "Event1".to_string(),
        "EVENT".to_string(),
        0.9,
    );

    let effect = Entity::new(
        EntityId::new("event2".to_string()),
        "Event2".to_string(),
        "EVENT".to_string(),
        0.9,
    );

    graph.add_entity(cause).unwrap();
    graph.add_entity(effect).unwrap();

    // Create causal relationship
    let causal_rel = Relationship::new(
        EntityId::new("event1".to_string()),
        EntityId::new("event2".to_string()),
        "CAUSED".to_string(),
        0.8,
    )
    .with_temporal_type(TemporalRelationType::Caused)
    .with_causal_strength(0.95);

    graph.add_relationship(causal_rel.clone()).unwrap();

    let query_concepts: Vec<String> = vec![];
    let weight = graph.dynamic_weight(&causal_rel, None, &query_concepts);

    // Should have causal boost applied
    let expected_boost = 0.95 * 0.2; // causal_strength * 20%
    let expected_weight = causal_rel.confidence * (1.0 + expected_boost);

    assert!(
        (weight - expected_weight).abs() < 0.01,
        "Expected weight ~{} but got {}",
        expected_weight,
        weight
    );
}

#[test]
fn test_dynamic_weight_semantic_boost() {
    let mut graph = KnowledgeGraph::new();

    // Create entities
    let e1 = Entity::new(
        EntityId::new("e1".to_string()),
        "Entity1".to_string(),
        "CONCEPT".to_string(),
        0.9,
    );

    let e2 = Entity::new(
        EntityId::new("e2".to_string()),
        "Entity2".to_string(),
        "CONCEPT".to_string(),
        0.9,
    );

    graph.add_entity(e1).unwrap();
    graph.add_entity(e2).unwrap();

    // Create relationship with embedding
    let rel_embedding = vec![0.1, 0.2, 0.3, 0.4]; // Sample embedding
    let rel = Relationship::new(
        EntityId::new("e1".to_string()),
        EntityId::new("e2".to_string()),
        "RELATES_TO".to_string(),
        0.7,
    )
    .with_embedding(rel_embedding.clone());

    graph.add_relationship(rel.clone()).unwrap();

    // Query embedding with high similarity (same vector)
    let query_embedding = vec![0.1, 0.2, 0.3, 0.4];

    let query_concepts: Vec<String> = vec![];
    let weight = graph.dynamic_weight(&rel, Some(&query_embedding), &query_concepts);

    // Should have semantic boost applied (cosine similarity = 1.0 for identical vectors)
    let expected_weight = rel.confidence * (1.0 + 1.0); // base * (1 + similarity)

    assert!(
        (weight - expected_weight).abs() < 0.01,
        "Expected weight ~{} but got {}",
        expected_weight,
        weight
    );
}

#[test]
fn test_dynamic_weight_combined_boosts() {
    let mut graph = KnowledgeGraph::new();

    // Create entities
    let e1 = Entity::new(
        EntityId::new("e1".to_string()),
        "Entity1".to_string(),
        "CONCEPT".to_string(),
        0.9,
    );

    let e2 = Entity::new(
        EntityId::new("e2".to_string()),
        "Entity2".to_string(),
        "CONCEPT".to_string(),
        0.9,
    );

    graph.add_entity(e1).unwrap();
    graph.add_entity(e2).unwrap();

    // Create relationship with multiple boost factors
    let rel = Relationship::new(
        EntityId::new("e1".to_string()),
        EntityId::new("e2".to_string()),
        "DISCUSSED".to_string(),
        0.7,
    )
    .with_embedding(vec![0.5, 0.5, 0.5, 0.5])
    .with_temporal_range(
        -380 * 365 * 24 * 3600, // 380 BC
        -380 * 365 * 24 * 3600,
    )
    .with_causal_strength(0.8);

    graph.add_relationship(rel.clone()).unwrap();

    // Apply all boost factors
    let query_embedding = vec![0.5, 0.5, 0.5, 0.5]; // Perfect match
    let query_concepts = vec!["discussed".to_string()];

    let weight = graph.dynamic_weight(&rel, Some(&query_embedding), &query_concepts);

    // Should have all boosts combined
    // semantic_boost = 1.0 (cosine similarity)
    // temporal_boost = ~0.05 (ancient event)
    // concept_boost = 0.15 (1 match)
    // causal_boost = 0.16 (0.8 * 0.2)
    // total_boost = 1.0 + 0.05 + 0.15 + 0.16 = 1.36
    let expected_weight = rel.confidence * (1.0 + 1.0 + 0.15 + 0.16);

    assert!(
        weight > rel.confidence * 2.0,
        "Expected significant boost with all factors, got weight {}",
        weight
    );

    // Verify weight is within reasonable range
    assert!(
        weight > expected_weight * 0.9 && weight < expected_weight * 1.1,
        "Expected weight ~{} but got {}",
        expected_weight,
        weight
    );
}

#[test]
fn test_cosine_similarity_identical_vectors() {
    let graph = KnowledgeGraph::new();

    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];

    // Access cosine_similarity through dynamic_weight with minimal relationship
    let mut test_graph = KnowledgeGraph::new();
    let e1 = Entity::new(
        EntityId::new("e1".to_string()),
        "E1".to_string(),
        "TEST".to_string(),
        0.5,
    );
    let e2 = Entity::new(
        EntityId::new("e2".to_string()),
        "E2".to_string(),
        "TEST".to_string(),
        0.5,
    );
    test_graph.add_entity(e1).unwrap();
    test_graph.add_entity(e2).unwrap();

    let rel = Relationship::new(
        EntityId::new("e1".to_string()),
        EntityId::new("e2".to_string()),
        "TEST".to_string(),
        0.5,
    )
    .with_embedding(v1);

    test_graph.add_relationship(rel.clone()).unwrap();

    let weight = test_graph.dynamic_weight(&rel, Some(&v2), &vec![]);

    // Identical vectors should give similarity = 1.0
    // weight = 0.5 * (1.0 + 1.0) = 1.0
    assert!(
        (weight - 1.0).abs() < 0.01,
        "Expected weight 1.0 for identical vectors, got {}",
        weight
    );
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    let mut test_graph = KnowledgeGraph::new();
    let e1 = Entity::new(
        EntityId::new("e1".to_string()),
        "E1".to_string(),
        "TEST".to_string(),
        0.5,
    );
    let e2 = Entity::new(
        EntityId::new("e2".to_string()),
        "E2".to_string(),
        "TEST".to_string(),
        0.5,
    );
    test_graph.add_entity(e1).unwrap();
    test_graph.add_entity(e2).unwrap();

    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];

    let rel = Relationship::new(
        EntityId::new("e1".to_string()),
        EntityId::new("e2".to_string()),
        "TEST".to_string(),
        0.5,
    )
    .with_embedding(v1);

    test_graph.add_relationship(rel.clone()).unwrap();

    let weight = test_graph.dynamic_weight(&rel, Some(&v2), &vec![]);

    // Orthogonal vectors should give similarity = 0.0
    // weight = 0.5 * (1.0 + 0.0) = 0.5
    assert!(
        (weight - 0.5).abs() < 0.01,
        "Expected weight 0.5 for orthogonal vectors, got {}",
        weight
    );
}
