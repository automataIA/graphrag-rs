use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use graphrag_core::core::{EntityId, KnowledgeGraph, Entity, Relationship};

#[cfg(feature = "async")]
use graphrag_core::{
    retrieval::causal_analysis::CausalAnalyzer,
    graph::hierarchical_relationships::HierarchyBuilder,
    optimization::OptimizerConfig,
};

use std::sync::Arc;

// Sample text for benchmarking
const SAMPLE_TEXT: &str = r#"
Socrates was a classical Greek philosopher born in Athens around 470 BC.
He is credited as one of the founders of Western philosophy. Socrates taught Plato,
who in turn taught Aristotle, forming a crucial chain of philosophical influence.
His method of inquiry, now called the Socratic method, involved asking probing questions
to stimulate critical thinking. In 399 BC, Socrates was sentenced to death by drinking
hemlock after being found guilty of corrupting the youth and impiety. His death caused
widespread debate about justice and the role of philosophy in society.
"#;

const CONCEPTUAL_QUERY: &str = "What is the nature of philosophical inquiry?";
const CAUSAL_QUERY: &str = "What caused Socrates' death?";
const FACTUAL_QUERY: &str = "Who taught Plato?";

/// Helper to create a test graph with relationships
fn create_test_graph() -> KnowledgeGraph {
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
        first_mentioned: None,
        last_mentioned: None,
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

    graph.add_entity(socrates).unwrap();
    graph.add_entity(plato).unwrap();
    graph.add_entity(philosophy).unwrap();

    // Relationships
    let taught_rel = Relationship::new(
        EntityId("socrates".to_string()),
        EntityId("plato".to_string()),
        "TAUGHT".to_string(),
        0.9,
    );

    let founded_rel = Relationship::new(
        EntityId("socrates".to_string()),
        EntityId("philosophy".to_string()),
        "FOUNDED".to_string(),
        0.85,
    );

    graph.add_relationship(taught_rel).unwrap();
    graph.add_relationship(founded_rel).unwrap();

    graph
}

/// Benchmark baseline retrieval (no advanced features)
fn bench_baseline_retrieval(c: &mut Criterion) {
    let graph = create_test_graph();
    let graph_arc = Arc::new(graph);

    c.bench_function("baseline_retrieval", |b| {
        b.iter(|| {
            // Simple entity lookup
            let id = EntityId("socrates".to_string());
            let result = graph_arc.get_entity(&id);
            black_box(result);
        });
    });
}

/// Benchmark symbolic anchoring (Phase 2.1)
#[cfg(feature = "async")]
fn bench_symbolic_anchoring(c: &mut Criterion) {
    let graph = create_test_graph();
    let graph_arc = Arc::new(graph);

    // Mock embedding provider (in real benchmark, use actual provider)
    // For now, just test the structure overhead

    c.bench_function("symbolic_anchoring_extract", |b| {
        b.iter(|| {
            // Simulate anchor extraction overhead
            let query = black_box(CONCEPTUAL_QUERY);
            let concepts: Vec<String> = query
                .split_whitespace()
                .filter(|w| w.len() > 5)
                .map(|s| s.to_string())
                .collect();
            black_box(concepts);
        });
    });
}

/// Benchmark dynamic edge weighting (Phase 2.2)
fn bench_dynamic_weighting(c: &mut Criterion) {
    let graph = create_test_graph();

    c.bench_function("dynamic_weight_calculation", |b| {
        b.iter(|| {
            for rel in graph.get_all_relationships() {
                // Simulate dynamic weight calculation
                let base_weight = rel.confidence;
                let semantic_boost = 0.15;
                let temporal_boost = 0.10;
                let concept_boost = 0.05;

                let dynamic_weight = base_weight * (1.0 + semantic_boost + temporal_boost + concept_boost);
                black_box(dynamic_weight);
            }
        });
    });
}

/// Benchmark causal chain analysis (Phase 2.3)
#[cfg(feature = "async")]
fn bench_causal_analysis(c: &mut Criterion) {
    let graph = create_test_graph();
    let graph_arc = Arc::new(graph);

    c.bench_function("causal_chain_finding", |b| {
        b.iter(|| {
            let analyzer = CausalAnalyzer::new(graph_arc.clone());

            // Find chains between two entities
            let cause = EntityId("socrates".to_string());
            let effect = EntityId("philosophy".to_string());

            let chains = analyzer.find_causal_chains(&cause, &effect, 5);
            black_box(chains);
        });
    });
}

/// Benchmark hierarchical clustering (Phase 3.1)
#[cfg(feature = "async")]
fn bench_hierarchical_clustering(c: &mut Criterion) {
    let graph = create_test_graph();
    let relationships: Vec<Relationship> = graph.get_all_relationships().into_iter().cloned().collect();

    c.bench_function("hierarchical_clustering_build", |b| {
        b.iter(|| {
            let builder = HierarchyBuilder::new(relationships.clone())
                .with_num_levels(3)
                .with_resolutions(vec![0.8, 1.0, 1.5])
                .with_min_cluster_size(2);

            // Note: actual build() is async and requires Ollama
            // This benchmarks just the builder setup
            black_box(builder);
        });
    });
}

/// Benchmark graph weight optimization (Phase 3.2)
#[cfg(feature = "async")]
fn bench_weight_optimization(c: &mut Criterion) {
    let _graph = create_test_graph();

    c.bench_function("weight_optimization_step", |b| {
        b.iter(|| {
            // Simulate one optimization step
            let learning_rate = 0.05;
            let adjustment_factor = 1.0 + learning_rate;

            // Simulate weight adjustment
            let relationships = _graph.get_all_relationships();
            for rel in relationships {
                let adjusted = rel.confidence * adjustment_factor;
                black_box(adjusted);
            }
        });
    });
}

/// Comparative benchmark: baseline vs all features
fn bench_feature_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_comparison");

    let graph = create_test_graph();

    // Baseline
    group.bench_function(BenchmarkId::new("baseline", "simple_lookup"), |b| {
        b.iter(|| {
            let id = EntityId("socrates".to_string());
            let result = graph.get_entity(&id);
            black_box(result);
        });
    });

    // With dynamic weighting
    group.bench_function(BenchmarkId::new("dynamic_weighting", "weighted_lookup"), |b| {
        b.iter(|| {
            let id = EntityId("socrates".to_string());
            let entity = graph.get_entity(&id);
            if let Some(e) = entity {
                // Simulate dynamic weighting overhead
                for rel in graph.get_all_relationships() {
                    if rel.source == e.id {
                        let weight = rel.confidence * 1.15;
                        black_box(weight);
                    }
                }
            }
        });
    });

    // With triple reflection (simulated)
    group.bench_function(BenchmarkId::new("triple_reflection", "validated_lookup"), |b| {
        b.iter(|| {
            let id = EntityId("socrates".to_string());
            let entity = graph.get_entity(&id);
            if let Some(e) = entity {
                // Simulate validation overhead
                for rel in graph.get_all_relationships() {
                    if rel.source == e.id {
                        let is_valid = rel.confidence > 0.7;
                        black_box(is_valid);
                    }
                }
            }
        });
    });

    group.finish();
}

/// Benchmark scaling with graph size
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    for num_entities in [10, 50, 100, 500].iter() {
        let mut graph = KnowledgeGraph::new();

        // Create entities
        for i in 0..*num_entities {
            let entity = Entity {
                id: EntityId(format!("entity_{}", i)),
                name: format!("Entity {}", i),
                entity_type: "TEST".to_string(),
                confidence: 0.9,
                mentions: vec![],
                embedding: None,
                first_mentioned: None,
                last_mentioned: None,
                temporal_validity: None,
            };
            graph.add_entity(entity).unwrap();
        }

        // Create relationships (each entity connected to next)
        for i in 0..*num_entities - 1 {
            let rel = Relationship::new(
                EntityId(format!("entity_{}", i)),
                EntityId(format!("entity_{}", i + 1)),
                "RELATES_TO".to_string(),
                0.8,
            );
            graph.add_relationship(rel).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("entity_lookup", num_entities),
            num_entities,
            |b, _| {
                b.iter(|| {
                    let id = EntityId("entity_0".to_string());
                    let result = graph.get_entity(&id);
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("relationship_traversal", num_entities),
            num_entities,
            |b, _| {
                b.iter(|| {
                    let relationships = graph.get_all_relationships();
                    black_box(relationships.len());
                });
            },
        );
    }

    group.finish();
}

// Configure criterion groups
criterion_group!(
    benches,
    bench_baseline_retrieval,
    bench_dynamic_weighting,
    bench_feature_comparison,
    bench_scaling,
);

#[cfg(feature = "async")]
criterion_group!(
    async_benches,
    bench_symbolic_anchoring,
    bench_causal_analysis,
    bench_hierarchical_clustering,
    bench_weight_optimization,
);

// Main benchmark runner
#[cfg(not(feature = "async"))]
criterion_main!(benches);

#[cfg(feature = "async")]
criterion_main!(benches, async_benches);
