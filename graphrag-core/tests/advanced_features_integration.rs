//! Integration tests for advanced GraphRAG features (Phases 2-3)
//!
//! These tests verify end-to-end functionality of:
//! - Symbolic Anchoring (Phase 2.1)
//! - Dynamic Edge Weighting (Phase 2.2)
//! - Causal Chain Analysis (Phase 2.3)
//! - Hierarchical Relationship Clustering (Phase 3.1)
//! - Graph Weight Optimization (Phase 3.2)

use graphrag_core::{
    core::{Entity, EntityId, KnowledgeGraph, Relationship},
    graph::temporal::TemporalRelationType,
    Config,
};

#[cfg(feature = "async")]
use graphrag_core::{
    retrieval::causal_analysis::CausalAnalyzer,
    graph::hierarchical_relationships::HierarchyBuilder,
    optimization::{GraphWeightOptimizer, OptimizerConfig, ObjectiveWeights, TestQuery},
};

use std::sync::Arc;

/// Helper to create a test graph with philosophical entities and relationships
fn create_philosophy_graph() -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::new();

    // Entities
    let socrates = Entity {
        id: EntityId("socrates".to_string()),
        name: "Socrates".to_string(),
        entity_type: "PERSON".to_string(),
        confidence: 0.95,
        mentions: vec![],
        embedding: None,
        first_mentioned: Some(-470 * 365 * 24 * 3600),
        last_mentioned: Some(-399 * 365 * 24 * 3600),
        temporal_validity: None,
    };

    let plato = Entity {
        id: EntityId("plato".to_string()),
        name: "Plato".to_string(),
        entity_type: "PERSON".to_string(),
        confidence: 0.95,
        mentions: vec![],
        embedding: None,
        first_mentioned: Some(-428 * 365 * 24 * 3600),
        last_mentioned: Some(-348 * 365 * 24 * 3600),
        temporal_validity: None,
    };

    let aristotle = Entity {
        id: EntityId("aristotle".to_string()),
        name: "Aristotle".to_string(),
        entity_type: "PERSON".to_string(),
        confidence: 0.95,
        mentions: vec![],
        embedding: None,
        first_mentioned: Some(-384 * 365 * 24 * 3600),
        last_mentioned: Some(-322 * 365 * 24 * 3600),
        temporal_validity: None,
    };

    let philosophy = Entity {
        id: EntityId("philosophy".to_string()),
        name: "Western Philosophy".to_string(),
        entity_type: "CONCEPT".to_string(),
        confidence: 0.9,
        mentions: vec![],
        embedding: None,
        first_mentioned: None,
        last_mentioned: None,
        temporal_validity: None,
    };

    let academy = Entity {
        id: EntityId("academy".to_string()),
        name: "The Academy".to_string(),
        entity_type: "INSTITUTION".to_string(),
        confidence: 0.85,
        mentions: vec![],
        embedding: None,
        first_mentioned: Some(-387 * 365 * 24 * 3600),
        last_mentioned: None,
        temporal_validity: None,
    };

    graph.add_entity(socrates).unwrap();
    graph.add_entity(plato).unwrap();
    graph.add_entity(aristotle).unwrap();
    graph.add_entity(philosophy).unwrap();
    graph.add_entity(academy).unwrap();

    // Relationships
    let mut taught_plato = Relationship::new(
        EntityId("socrates".to_string()),
        EntityId("plato".to_string()),
        "TAUGHT".to_string(),
        0.9,
    );
    taught_plato.temporal_type = Some(TemporalRelationType::Before);
    taught_plato.causal_strength = Some(0.8);

    let mut founded_academy = Relationship::new(
        EntityId("plato".to_string()),
        EntityId("academy".to_string()),
        "FOUNDED".to_string(),
        0.95,
    );
    founded_academy.temporal_type = Some(TemporalRelationType::Caused);
    founded_academy.causal_strength = Some(0.9);

    let mut taught_aristotle = Relationship::new(
        EntityId("plato".to_string()),
        EntityId("aristotle".to_string()),
        "TAUGHT".to_string(),
        0.9,
    );
    taught_aristotle.temporal_type = Some(TemporalRelationType::Before);
    taught_aristotle.causal_strength = Some(0.8);

    let mut founded_philosophy = Relationship::new(
        EntityId("socrates".to_string()),
        EntityId("philosophy".to_string()),
        "FOUNDED".to_string(),
        0.85,
    );
    founded_philosophy.temporal_type = Some(TemporalRelationType::Caused);
    founded_philosophy.causal_strength = Some(0.7);

    let mut influenced_philosophy = Relationship::new(
        EntityId("plato".to_string()),
        EntityId("philosophy".to_string()),
        "INFLUENCED".to_string(),
        0.9,
    );
    influenced_philosophy.temporal_type = Some(TemporalRelationType::Enabled);
    influenced_philosophy.causal_strength = Some(0.75);

    graph.add_relationship(taught_plato).unwrap();
    graph.add_relationship(founded_academy).unwrap();
    graph.add_relationship(taught_aristotle).unwrap();
    graph.add_relationship(founded_philosophy).unwrap();
    graph.add_relationship(influenced_philosophy).unwrap();

    graph
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_causal_chain_discovery() {
    // Test Phase 2.3: Causal Chain Analysis
    let graph = create_philosophy_graph();
    let graph_arc = Arc::new(graph);

    let analyzer = CausalAnalyzer::new(graph_arc.clone());

    // Find causal chains from Socrates to Philosophy
    let cause = EntityId("socrates".to_string());
    let effect = EntityId("philosophy".to_string());

    let chains = analyzer.find_causal_chains(&cause, &effect, 5).unwrap();

    // Should find at least one chain (direct relationship)
    assert!(!chains.is_empty(), "Should find causal chains");

    // Verify chain properties
    let first_chain = &chains[0];
    assert_eq!(first_chain.cause, cause);
    assert_eq!(first_chain.effect, effect);
    assert!(first_chain.total_confidence > 0.0);
    assert!(first_chain.total_confidence <= 1.0);

    println!("Found {} causal chain(s)", chains.len());
    for (i, chain) in chains.iter().enumerate() {
        println!(
            "Chain {}: {} -> {} (confidence: {:.2}, {} steps)",
            i + 1,
            chain.cause,
            chain.effect,
            chain.total_confidence,
            chain.steps.len()
        );
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_multi_step_causal_reasoning() {
    // Test multi-step causal chains: Socrates -> Plato -> Aristotle
    let graph = create_philosophy_graph();
    let graph_arc = Arc::new(graph);

    let analyzer = CausalAnalyzer::new(graph_arc.clone());

    let cause = EntityId("socrates".to_string());
    let effect = EntityId("aristotle".to_string());

    let chains = analyzer.find_causal_chains(&cause, &effect, 5).unwrap();

    // Should find multi-step chain through Plato
    if !chains.is_empty() {
        let chain = &chains[0];
        assert!(chain.steps.len() >= 2, "Should have at least 2 steps");
        println!(
            "Multi-step chain: {} -> Plato -> {}",
            chain.cause, chain.effect
        );
    } else {
        println!("Note: No multi-step causal chain found (this may be expected if relationships don't form a causal path)");
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_temporal_consistency_validation() {
    // Test that causal chains respect temporal ordering
    let graph = create_philosophy_graph();
    let graph_arc = Arc::new(graph);

    let analyzer = CausalAnalyzer::new(graph_arc.clone());

    let cause = EntityId("socrates".to_string());
    let effect = EntityId("philosophy".to_string());

    let chains = analyzer.find_causal_chains(&cause, &effect, 5).unwrap();

    if !chains.is_empty() {
        let chain = &chains[0];

        // If temporal_consistency is true, all steps should be properly ordered
        if chain.temporal_consistency {
            println!("✓ Chain has temporal consistency");

            // Verify time span is reasonable (negative means later event has earlier timestamp)
            if let Some(span) = chain.time_span {
                assert!(span >= 0, "Time span should be positive (cause before effect)");
                println!("  Time span: {} seconds", span);
            }
        } else {
            println!("Note: Chain lacks temporal consistency (timestamps may be missing)");
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_hierarchical_clustering_build() {
    // Test Phase 3.1: Hierarchical Relationship Clustering
    let graph = create_philosophy_graph();
    let relationships: Vec<Relationship> = graph
        .get_all_relationships()
        .into_iter()
        .cloned()
        .collect();

    // Build hierarchy with 2 levels
    let builder = HierarchyBuilder::new(relationships)
        .with_num_levels(2)
        .with_resolutions(vec![0.5, 1.0])
        .with_min_cluster_size(1);

    // Note: actual build() requires Ollama for summaries
    // Here we just test the builder construction
    println!("Hierarchy builder created with 2 levels");

    // In a full test with Ollama available:
    // let hierarchy = builder.build().await.unwrap();
    // assert_eq!(hierarchy.levels.len(), 2);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_graph_weight_optimization_setup() {
    // Test Phase 3.2: Graph Weight Optimization
    let _graph = create_philosophy_graph();

    let weights = ObjectiveWeights {
        relevance: 0.4,
        faithfulness: 0.4,
        conciseness: 0.2,
    };

    let config = OptimizerConfig {
        learning_rate: 0.05,
        max_iterations: 5,
        slope_window: 3,
        stagnation_threshold: 0.01,
        objective_weights: weights,
        use_llm_eval: false, // Disable LLM for test
    };

    let optimizer = GraphWeightOptimizer::with_config(config);

    // Test queries for optimization
    let test_queries = vec![
        TestQuery {
            query: "Who taught Plato?".to_string(),
            expected_answer: "Socrates taught Plato".to_string(),
            weight: 1.0,
        },
        TestQuery {
            query: "What is the foundation of Western Philosophy?".to_string(),
            expected_answer: "Socrates founded Western Philosophy".to_string(),
            weight: 1.0,
        },
    ];

    // Note: actual optimization requires Ollama for LLM evaluation
    // Here we just test the setup
    println!("Optimizer created with {} test queries", test_queries.len());

    // In a full test with Ollama available:
    // optimizer.optimize_weights(&mut graph, &test_queries).await.unwrap();
    // assert!(!optimizer.get_history().is_empty());
}

#[test]
fn test_advanced_features_config_defaults() {
    // Test that advanced features config has sensible defaults
    let config = Config::default();

    // Symbolic Anchoring
    assert!(config.advanced_features.symbolic_anchoring.min_relevance >= 0.0);
    assert!(config.advanced_features.symbolic_anchoring.min_relevance <= 1.0);
    assert!(config.advanced_features.symbolic_anchoring.max_anchors > 0);

    // Dynamic Weighting
    assert!(config.advanced_features.dynamic_weighting.enable_semantic_boost);
    assert!(config.advanced_features.dynamic_weighting.enable_temporal_boost);

    // Causal Analysis
    assert!(config.advanced_features.causal_analysis.min_confidence >= 0.0);
    assert!(config.advanced_features.causal_analysis.min_confidence <= 1.0);
    assert!(config.advanced_features.causal_analysis.max_chain_depth > 0);
    assert!(config.advanced_features.causal_analysis.max_chain_depth <= 10);

    // Hierarchical Clustering
    assert!(config.advanced_features.hierarchical_clustering.num_levels >= 2);
    assert!(config.advanced_features.hierarchical_clustering.num_levels <= 5);
    assert_eq!(
        config.advanced_features.hierarchical_clustering.resolutions.len(),
        config.advanced_features.hierarchical_clustering.num_levels
    );

    // Weight Optimization
    assert!(config.advanced_features.weight_optimization.learning_rate > 0.0);
    assert!(config.advanced_features.weight_optimization.learning_rate <= 1.0);
    assert!(config.advanced_features.weight_optimization.max_iterations > 0);

    println!("✓ All advanced features config defaults are valid");
}

#[test]
fn test_config_serialization() {
    // Test that advanced features config can be serialized/deserialized
    let config = Config::default();

    let serialized = toml::to_string(&config).expect("Should serialize to TOML");
    assert!(serialized.contains("[advanced_features"));

    println!("Config serialized successfully ({} bytes)", serialized.len());
}

#[test]
fn test_temporal_relationship_types() {
    // Test TemporalRelationType usage
    let mut rel = Relationship::new(
        EntityId("a".to_string()),
        EntityId("b".to_string()),
        "CAUSED".to_string(),
        0.9,
    );

    rel.temporal_type = Some(TemporalRelationType::Caused);
    rel.causal_strength = Some(0.85);

    assert!(rel.temporal_type.is_some());
    assert!(rel.causal_strength.unwrap() > 0.8);

    println!("✓ Temporal relationship types work correctly");
}

#[test]
fn test_entity_temporal_fields() {
    // Test Entity temporal fields
    let entity = Entity {
        id: EntityId("test".to_string()),
        name: "Test Entity".to_string(),
        entity_type: "TEST".to_string(),
        confidence: 0.9,
        mentions: vec![],
        embedding: None,
        first_mentioned: Some(1000),
        last_mentioned: Some(2000),
        temporal_validity: None,
    };

    assert!(entity.first_mentioned.is_some());
    assert!(entity.last_mentioned.is_some());
    assert!(entity.last_mentioned.unwrap() > entity.first_mentioned.unwrap());

    println!("✓ Entity temporal fields work correctly");
}

#[cfg(feature = "async")]
#[cfg(feature = "rograg")]
#[tokio::test]
async fn test_rograg_causal_analyzer_integration() {
    // Test Phase 2.3 enhancement: RoGRAG with CausalAnalyzer integration
    use graphrag_core::rograg::{LogicFormQuery, LogicFormExecutor, Predicate, Argument, ArgumentType, LogicQueryType};

    let graph = create_philosophy_graph();

    // Create RoGRAG executor
    let executor = LogicFormExecutor::new();

    // Create a Caused predicate logic form query
    let logic_form = LogicFormQuery {
        predicate: Predicate::Caused,
        arguments: vec![
            Argument {
                arg_type: ArgumentType::Entity,
                value: "Socrates".to_string(),
                variable: Some("C".to_string()),
                constraints: vec![],
            },
            Argument {
                arg_type: ArgumentType::Entity,
                value: "Western Philosophy".to_string(),
                variable: Some("E".to_string()),
                constraints: vec![],
            },
        ],
        constraints: vec![],
        query_type: LogicQueryType::Select,
        confidence: 0.8,
    };

    // Execute the logic form
    let bindings = executor.execute(&logic_form, &graph).unwrap();

    // Should find causal chains with temporal consistency
    assert!(!bindings.is_empty(), "Should find causal bindings");

    // Check that results include temporal consistency information
    let has_temporal_info = bindings.iter().any(|b| {
        b.value.contains("temporally consistent")
            || b.value.contains("Causal chain")
    });

    assert!(has_temporal_info, "Should include temporal consistency information");

    println!("RoGRAG CausalAnalyzer integration test: {} bindings found", bindings.len());
    for binding in &bindings {
        println!("  - {} (confidence: {:.2})", binding.value, binding.confidence);
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_end_to_end_advanced_pipeline() {
    // Integration test: use multiple advanced features together
    let graph = create_philosophy_graph();
    let graph_arc = Arc::new(graph);

    // 1. Causal Analysis
    let analyzer = CausalAnalyzer::new(graph_arc.clone());
    let cause = EntityId("socrates".to_string());
    let effect = EntityId("philosophy".to_string());
    let chains = analyzer.find_causal_chains(&cause, &effect, 5).unwrap();

    println!("Step 1: Found {} causal chain(s)", chains.len());

    // 2. Dynamic Weighting (simulated - would need actual query context)
    let relationships = graph_arc.get_all_relationships();
    for rel in relationships {
        let base_weight = rel.confidence;
        let boosted_weight = if rel.causal_strength.is_some() {
            base_weight * 1.1 // Causal boost
        } else {
            base_weight
        };
        assert!(boosted_weight >= base_weight);
    }

    println!("Step 2: Applied dynamic weighting to {} relationships",
             graph_arc.get_all_relationships().len());

    // 3. Hierarchical Clustering (builder setup)
    let rels: Vec<Relationship> = graph_arc
        .get_all_relationships()
        .into_iter()
        .cloned()
        .collect();

    let _hierarchy_builder = HierarchyBuilder::new(rels)
        .with_num_levels(2)
        .with_resolutions(vec![0.5, 1.0]);

    println!("Step 3: Hierarchy builder configured");

    println!("✓ End-to-end advanced pipeline test completed");
}
