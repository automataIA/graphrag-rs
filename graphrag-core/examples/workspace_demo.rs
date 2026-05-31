//! Workspace persistence demo
//!
//! Demonstrates saving and loading knowledge graphs using the WorkspaceManager.
//!
//! Run with: cargo run --example workspace_demo

use graphrag_core::{
    persistence::WorkspaceManager, ChunkId, Document, DocumentId, Entity, EntityId, KnowledgeGraph,
    Relationship, TextChunk,
};
use indexmap::IndexMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GraphRAG Workspace Demo\n");

    // Create workspace manager
    let workspace_dir = "./workspace_demo";
    let workspace = WorkspaceManager::new(workspace_dir)?;
    println!("✅ Created workspace manager at: {}\n", workspace_dir);

    // Create a simple knowledge graph
    let mut graph = KnowledgeGraph::new();

    // Add a document
    let doc = Document {
        id: DocumentId::new("doc1".to_string()),
        title: "Test Document".to_string(),
        content: "This is a test document about Rust and GraphRAG.".to_string(),
        metadata: IndexMap::new(),
        chunks: vec![],
    };
    graph.add_document(doc)?;

    // Add a chunk
    let chunk = TextChunk::new(
        ChunkId::new("chunk1".to_string()),
        DocumentId::new("doc1".to_string()),
        "This is a test document about Rust and GraphRAG.".to_string(),
        0,
        47,
    );
    graph.add_chunk(chunk)?;

    // Add entities
    let entity1 = Entity::new(
        EntityId::new("rust".to_string()),
        "Rust".to_string(),
        "TECHNOLOGY".to_string(),
        0.95,
    );
    graph.add_entity(entity1)?;

    let entity2 = Entity::new(
        EntityId::new("graphrag".to_string()),
        "GraphRAG".to_string(),
        "SYSTEM".to_string(),
        0.9,
    );
    graph.add_entity(entity2)?;

    // Add a relationship
    let relationship = Relationship {
        source: EntityId::new("rust".to_string()),
        target: EntityId::new("graphrag".to_string()),
        relation_type: "USES".to_string(),
        confidence: 0.85,
        context: vec![ChunkId::new("chunk1".to_string())],
        embedding: None,
        temporal_type: None,
        temporal_range: None,
        causal_strength: None,
    };
    graph.add_relationship(relationship)?;

    println!("📊 Created knowledge graph:");
    println!("   - Entities: {}", graph.entity_count());
    println!("   - Relationships: {}", graph.relationship_count());
    println!("   - Documents: {}", graph.document_count());
    println!("   - Chunks: {}\n", graph.chunks().count());

    // Save to workspace
    println!("💾 Saving to workspace 'demo'...");
    workspace.save_graph(&graph, "demo")?;
    println!("✅ Graph saved successfully!\n");

    // List workspaces
    println!("📁 Available workspaces:");
    let workspaces = workspace.list_workspaces()?;
    for (i, ws) in workspaces.iter().enumerate() {
        println!(
            "   {}. {} ({} entities, {} relationships) - {} bytes",
            i + 1,
            ws.name,
            ws.metadata.entity_count,
            ws.metadata.relationship_count,
            ws.size_bytes
        );
    }
    println!();

    // Load from workspace
    println!("📥 Loading from workspace 'demo'...");
    let loaded_graph = workspace.load_graph("demo")?;
    println!("✅ Graph loaded successfully!\n");

    println!("📊 Loaded knowledge graph:");
    println!("   - Entities: {}", loaded_graph.entity_count());
    println!("   - Relationships: {}", loaded_graph.relationship_count());
    println!("   - Documents: {}", loaded_graph.document_count());
    println!("   - Chunks: {}\n", loaded_graph.chunks().count());

    // Verify data integrity
    println!("🔍 Verifying data integrity...");
    assert_eq!(graph.entity_count(), loaded_graph.entity_count());
    assert_eq!(
        graph.relationship_count(),
        loaded_graph.relationship_count()
    );
    assert_eq!(graph.document_count(), loaded_graph.document_count());
    println!("✅ Data integrity verified!\n");

    // Show entities
    println!("🏷️  Entities in loaded graph:");
    for entity in loaded_graph.entities() {
        println!(
            "   - {} ({}) [confidence: {:.2}]",
            entity.name, entity.entity_type, entity.confidence
        );
    }
    println!();

    // Show relationships
    println!("🔗 Relationships in loaded graph:");
    for rel in loaded_graph.relationships() {
        println!(
            "   - {} --[{}]--> {} [confidence: {:.2}]",
            rel.source.0, rel.relation_type, rel.target.0, rel.confidence
        );
    }
    println!();

    println!("🎉 Demo completed successfully!");
    println!("\nℹ️  Workspace saved at: {}", workspace_dir);
    println!("   You can inspect the files:");
    println!("   - {}/demo/graph.json (knowledge graph)", workspace_dir);
    println!(
        "   - {}/demo/metadata.toml (workspace metadata)",
        workspace_dir
    );

    Ok(())
}
