//! Hierarchical GraphRAG Example
//!
//! Demonstrates the complete hierarchical GraphRAG pipeline with Leiden community detection:
//! 1. Build knowledge graph from documents
//! 2. Extract entities and relationships
//! 3. Detect hierarchical communities using Leiden algorithm
//! 4. Generate entity-enriched community summaries
//! 5. Perform multi-level retrieval
//!
//! Run with: cargo run --example hierarchical_graphrag_demo --features leiden

use graphrag_core::{
    core::{Document, DocumentId, Entity, EntityId, KnowledgeGraph, Relationship},
    graph::LeidenConfig,
};

fn main() -> graphrag_core::Result<()> {
    println!("🚀 Hierarchical GraphRAG Demo");
    println!("{}", "=".repeat(80));

    // Step 1: Create Knowledge Graph
    println!("\n📚 Step 1: Building Knowledge Graph...");
    let mut graph = KnowledgeGraph::new();

    // Add sample documents
    let doc1 = Document::new(
        DocumentId::new("doc1".to_string()),
        "AI Research Paper".to_string(),
        "Artificial Intelligence and Machine Learning are transforming technology.".to_string(),
    );
    graph.add_document(doc1)?;

    // Step 2: Add sample entities
    println!("\n🏷️  Step 2: Adding entities...");

    let entities = vec![
        Entity::new(
            EntityId::new("e1".to_string()),
            "Artificial Intelligence".to_string(),
            "TECHNOLOGY".to_string(),
            0.95,
        ),
        Entity::new(
            EntityId::new("e2".to_string()),
            "Machine Learning".to_string(),
            "TECHNOLOGY".to_string(),
            0.92,
        ),
        Entity::new(
            EntityId::new("e3".to_string()),
            "Deep Learning".to_string(),
            "TECHNOLOGY".to_string(),
            0.88,
        ),
        Entity::new(
            EntityId::new("e4".to_string()),
            "Neural Networks".to_string(),
            "CONCEPT".to_string(),
            0.85,
        ),
        Entity::new(
            EntityId::new("e5".to_string()),
            "Natural Language Processing".to_string(),
            "TECHNOLOGY".to_string(),
            0.90,
        ),
        Entity::new(
            EntityId::new("e6".to_string()),
            "Computer Vision".to_string(),
            "TECHNOLOGY".to_string(),
            0.87,
        ),
        Entity::new(
            EntityId::new("e7".to_string()),
            "Transformers".to_string(),
            "ARCHITECTURE".to_string(),
            0.93,
        ),
        Entity::new(
            EntityId::new("e8".to_string()),
            "GPT".to_string(),
            "MODEL".to_string(),
            0.91,
        ),
    ];

    for entity in entities {
        println!("  + {} ({})", entity.name, entity.entity_type);
        graph.add_entity(entity)?;
    }

    // Step 3: Add relationships
    println!("\n🔗 Step 3: Adding relationships...");

    let relationships = vec![
        ("e1", "e2", "INCLUDES", 0.95), // AI includes ML
        ("e2", "e3", "INCLUDES", 0.90), // ML includes DL
        ("e3", "e4", "USES", 0.88),     // DL uses Neural Networks
        ("e2", "e5", "INCLUDES", 0.85), // ML includes NLP
        ("e2", "e6", "INCLUDES", 0.83), // ML includes CV
        ("e5", "e7", "USES", 0.92),     // NLP uses Transformers
        ("e7", "e8", "IMPLEMENTS", 0.89), // Transformers implements GPT
        ("e3", "e6", "USED_IN", 0.80),  // DL used in CV
    ];

    for (src, tgt, rel_type, confidence) in relationships {
        let rel = Relationship {
            source: EntityId::new(src.to_string()),
            target: EntityId::new(tgt.to_string()),
            relation_type: rel_type.to_string(),
            confidence,
            context: Vec::new(),
        };
        println!("  + {} --[{}]--> {}", src, rel_type, tgt);
        graph.add_relationship(rel)?;
    }

    // Step 4: Convert to Leiden graph and detect hierarchical communities
    println!("\n🔍 Step 4: Detecting hierarchical communities with Leiden...");

    let config = LeidenConfig {
        max_cluster_size: 4,
        use_lcc: true,
        seed: Some(42),
        resolution: 1.0,
        max_levels: 3,
        min_improvement: 0.001,
    };

    println!("  Configuration:");
    println!("    - Max cluster size: {}", config.max_cluster_size);
    println!("    - Resolution: {}", config.resolution);
    println!("    - Max levels: {}", config.max_levels);
    println!("    - Seed: {:?}", config.seed);

    let communities = graph.detect_hierarchical_communities(config)?;

    println!("\n  Community detection complete!");
    println!("  - Total levels: {}", communities.levels.len());
    println!("  - Entity mapping: {}",
        if communities.entity_mapping.is_some() { "✅ Enabled" } else { "❌ Disabled" });

    // Step 5: Analyze communities at each level
    println!("\n📊 Step 5: Analyzing community structure...");

    let leiden_graph = graph.to_leiden_graph();

    for level in 0..communities.levels.len() {
        if let Some(level_map) = communities.levels.get(&level) {
            let unique_communities: std::collections::HashSet<_> =
                level_map.values().collect();

            println!("\n  Level {}: {} communities", level, unique_communities.len());

            for &community_id in unique_communities {
                let entities = communities.get_community_entities(
                    community_id,
                    level,
                    &leiden_graph
                );

                if !entities.is_empty() {
                    println!("\n    Community {} (Level {}):", community_id, level);
                    println!("      Entities: {:?}", entities);

                    // Get entity metadata
                    let metadata_list = communities.get_entities_metadata(&entities);
                    for metadata in metadata_list {
                        println!("        - {} [{}] (confidence: {:.2})",
                            metadata.name,
                            metadata.entity_type,
                            metadata.confidence
                        );
                    }

                    // Get community statistics
                    let (count, avg_conf, types) = communities.get_community_stats(
                        community_id,
                        level,
                        &leiden_graph
                    );
                    println!("      Stats: {} entities, avg confidence: {:.2}",
                        count, avg_conf);
                    println!("      Entity types: {:?}", types);
                }
            }
        }
    }

    // Step 6: Generate hierarchical summaries
    println!("\n📝 Step 6: Generating hierarchical summaries...");

    let mut communities_mut = communities;
    communities_mut.generate_hierarchical_summaries(&leiden_graph, 3);

    println!("  Generated {} summaries", communities_mut.summaries.len());

    for (community_id, summary) in &communities_mut.summaries {
        println!("\n  Community {} summary:", community_id);
        println!("    {}", summary);
    }

    // Step 7: Demonstrate adaptive query routing (NEW!)
    println!("\n🔎 Step 7: Demonstrating ADAPTIVE query routing...");
    println!("  (Automatically selects best level based on query complexity)\n");

    use graphrag_core::query::AdaptiveRoutingConfig;

    let routing_config = AdaptiveRoutingConfig::default();

    // Test different query types
    let test_queries = vec![
        "Give me an overview of AI",              // Broad → high level
        "Transformers",                            // Specific → low level
        "What is the relationship between Deep Learning and Neural Networks?", // Very specific
    ];

    for query in test_queries {
        println!("\n  Query: \"{}\"", query);

        // Use adaptive retrieval (automatic level selection)
        let (analysis, results) = communities_mut.adaptive_retrieve_detailed(
            query,
            &leiden_graph,
            routing_config.clone(),
        );

        println!("  Complexity: {:?}", analysis.complexity);
        println!("  Suggested Level: {} (auto-selected)", analysis.suggested_level);
        println!("  Analysis scores:");
        println!("    - Keywords: {:.2}", analysis.keyword_score);
        println!("    - Length: {:.2}", analysis.length_score);
        println!("    - Entities: {:.2}", analysis.entity_score);

        if results.is_empty() {
            println!("    No relevant communities found");
        } else {
            println!("    Found {} relevant community/communities:", results.len());
            for (level, community_id, summary) in &results {
                println!("      [Level {}, Community {}]", level, community_id);
                let preview = if summary.len() > 80 {
                    format!("{}...", &summary[..77])
                } else {
                    summary.clone()
                };
                println!("      {}", preview);
            }
        }
    }

    println!("\n📊 Step 8: Comparing manual vs adaptive routing...");

    let query = "Neural Networks";

    // Manual retrieval at level 0
    println!("\n  Manual (level 0):");
    let manual_results = communities_mut.retrieve_at_level(query, &leiden_graph, 0);
    println!("    Found {} results at level 0", manual_results.len());

    // Adaptive retrieval (automatic)
    println!("\n  Adaptive (auto-select):");
    let adaptive_results = communities_mut.adaptive_retrieve(
        query,
        &leiden_graph,
        routing_config,
    );
    println!("    Found {} results (auto-selected level)", adaptive_results.len());

    println!("\n✅ Hierarchical GraphRAG with Adaptive Routing demo completed!");
    println!("{}", "=".repeat(80));

    Ok(())
}
