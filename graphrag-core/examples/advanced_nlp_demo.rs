//! Advanced NLP Features Demo
//!
//! Demonstrates the newly implemented deterministic NLP capabilities:
//! - Syntax Analysis (POS tagging, dependency parsing)
//! - Graph Traversal (BFS, DFS, ego networks)
//! - Query Optimization (cost-based planning)

use graphrag_core::{
    KnowledgeGraph, Entity, EntityId, Relationship,
    Result,
};

use graphrag_core::nlp::{SyntaxAnalyzer, SyntaxAnalyzerConfig};
use graphrag_core::graph::{GraphTraversal, TraversalConfig};
use graphrag_core::query::{QueryOptimizer, QueryOp, GraphStatistics, JoinType};

fn main() -> Result<()> {
    println!("\n🚀 Advanced NLP Features Demo");
    println!("{}\n", "=".repeat(70));

    demo_syntax_analysis()?;
    demo_graph_traversal()?;
    demo_query_optimization()?;

    println!("\n{}", "=".repeat(70));
    println!("✅ All demos completed successfully!\n");

    Ok(())
}

fn demo_syntax_analysis() -> Result<()> {
    println!("📝 DEMO 1: Syntax Analysis");
    println!("{}", "-".repeat(70));

    let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
    let text = "The good brown fox jumps over the lazy dog.";

    println!("\nInput: \"{}\"", text);

    // POS Tagging
    let tokens = analyzer.pos_tag(text)?;
    println!("\n1. POS Tags ({} tokens):", tokens.len());
    for token in tokens.iter().take(5) {
        println!("   {} → {:?}", token.text, token.pos);
    }

    // Noun Phrases
    let noun_phrases = analyzer.extract_noun_phrases(&tokens)?;
    println!("\n2. Noun Phrases ({} found):", noun_phrases.len());
    for np in &noun_phrases {
        let head = &tokens[np.head_idx];
        println!("   '{}' (head: {})", np.text, head.text);
    }

    // Dependencies
    let deps = analyzer.parse_dependencies(&tokens)?;
    println!("\n3. Dependencies ({} found):", deps.len());
    for dep in deps.iter().take(3) {
        println!("   {} → {}",
            tokens[dep.head].text,
            tokens[dep.dependent].text
        );
    }

    Ok(())
}

fn demo_graph_traversal() -> Result<()> {
    println!("\n\n🗺️  DEMO 2: Graph Traversal");
    println!("{}", "-".repeat(70));

    // Build sample graph
    let mut graph = KnowledgeGraph::new();

    for (id, name, etype) in &[
        ("alice", "Alice", "person"),
        ("bob", "Bob", "person"),
        ("charlie", "Charlie", "person"),
        ("stanford", "Stanford", "organization"),
    ] {
        graph.add_entity(Entity {
            id: EntityId::new(id.to_string()),
            name: name.to_string(),
            entity_type: etype.to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        })?;
    }

    graph.add_relationship(Relationship {
        source: EntityId::new("alice".to_string()),
        target: EntityId::new("bob".to_string()),
        relation_type: "knows".to_string(),
        confidence: 1.0,
        context: vec![],
    })?;

    graph.add_relationship(Relationship {
        source: EntityId::new("bob".to_string()),
        target: EntityId::new("charlie".to_string()),
        relation_type: "knows".to_string(),
        confidence: 1.0,
        context: vec![],
    })?;

    println!("\nGraph: {} entities, {} relationships",
        graph.entities().count(),
        graph.relationships().count()
    );

    let traversal = GraphTraversal::new(TraversalConfig {
        max_depth: 3,
        max_paths: 10,
        use_edge_weights: true,
        min_relationship_strength: 0.5,
    });

    let alice = EntityId::new("alice".to_string());

    // BFS
    let bfs = traversal.bfs(&graph, &alice)?;
    println!("\n1. BFS from Alice: {} entities discovered", bfs.entities.len());
    for entity in &bfs.entities {
        let depth = bfs.distances.get(&entity.id).unwrap_or(&0);
        println!("   Depth {}: {}", depth, entity.name);
    }

    // DFS
    let dfs = traversal.dfs(&graph, &alice)?;
    println!("\n2. DFS from Alice: {} entities", dfs.entities.len());

    // Ego network
    let ego = traversal.ego_network(&graph, &alice, Some(2))?;
    println!("\n3. Ego Network (2-hops): {} entities", ego.entities.len());
    for entity in &ego.entities {
        println!("   - {}", entity.name);
    }

    Ok(())
}

fn demo_query_optimization() -> Result<()> {
    println!("\n\n⚡ DEMO 3: Query Optimization");
    println!("{}", "-".repeat(70));

    // Build sample graph with statistics
    let mut graph = KnowledgeGraph::new();

    for i in 0..50 {
        let etype = match i % 3 {
            0 => "person",
            1 => "organization",
            _ => "location",
        };
        graph.add_entity(Entity {
            id: EntityId::new(format!("e{}", i)),
            name: format!("Entity {}", i),
            entity_type: etype.to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        })?;
    }

    let stats = GraphStatistics::from_graph(&graph);
    println!("\nGraph Statistics:");
    println!("   Total entities: {}", stats.total_entities);
    println!("   Average degree: {:.2}", stats.average_degree);

    let optimizer = QueryOptimizer::new(stats);

    // Simple scan query
    let scan_query = QueryOp::EntityScan {
        entity_type: "person".to_string(),
    };

    let cost = optimizer.estimate_cost(&scan_query)?;
    println!("\n1. Entity Scan Query:");
    println!("   Cost: {}, Cardinality: {}", cost.cost, cost.cardinality);

    // Join query
    let join_query = QueryOp::Join {
        left: Box::new(QueryOp::EntityScan {
            entity_type: "person".to_string(),
        }),
        right: Box::new(QueryOp::EntityScan {
            entity_type: "organization".to_string(),
        }),
        join_type: JoinType::Inner,
    };

    let join_cost = optimizer.estimate_cost(&join_query)?;
    println!("\n2. Join Query:");
    println!("   Cost: {}, Cardinality: {}", join_cost.cost, join_cost.cardinality);

    // Optimization
    let optimized = optimizer.optimize(join_query.clone())?;
    let opt_cost = optimizer.estimate_cost(&optimized)?;

    if opt_cost.cost < join_cost.cost {
        let improvement = (join_cost.cost - opt_cost.cost) / join_cost.cost * 100.0;
        println!("\n3. After Optimization:");
        println!("   Cost: {} ({}% improvement)", opt_cost.cost, improvement as i32);
    }

    Ok(())
}
