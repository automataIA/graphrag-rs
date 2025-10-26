//! Tom Sawyer workspace persistence test
//!
//! Tests workspace save/load with a real document (Tom Sawyer).
//! Demonstrates full persistence lifecycle:
//! 1. Load document
//! 2. Process into chunks
//! 3. Extract entities (mock)
//! 4. Save to workspace
//! 5. Load from workspace
//! 6. Verify integrity
//!
//! Run with: cargo run --example tom_sawyer_workspace

use graphrag_core::{
    persistence::WorkspaceManager, ChunkId, Document, DocumentId, Entity, EntityId,
    KnowledgeGraph, Relationship, TextChunk,
};
use indexmap::IndexMap;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Tom Sawyer Workspace Persistence Test\n");

    // === PHASE 1: Load and process document ===
    println!("📖 Phase 1: Loading document...");
    let tom_sawyer_path = "./docs-example/The_Adventures_of_Tom_Sawyer.txt";

    if !std::path::Path::new(tom_sawyer_path).exists() {
        eprintln!("❌ Tom Sawyer text file not found at: {}", tom_sawyer_path);
        eprintln!("   Please ensure the file exists.");
        return Ok(());
    }

    let content = fs::read_to_string(tom_sawyer_path)?;
    let content_size = content.len();
    println!("   ✅ Loaded {} bytes", content_size);

    let mut graph = KnowledgeGraph::new();

    // Create document
    let doc = Document {
        id: DocumentId::new("tom_sawyer".to_string()),
        title: "The Adventures of Tom Sawyer".to_string(),
        content: content.clone(),
        metadata: {
            let mut meta = IndexMap::new();
            meta.insert("author".to_string(), "Mark Twain".to_string());
            meta.insert("year".to_string(), "1876".to_string());
            meta.insert("genre".to_string(), "Fiction".to_string());
            meta
        },
        chunks: vec![],
    };
    graph.add_document(doc)?;

    // === PHASE 2: Create chunks ===
    println!("\n🔪 Phase 2: Creating chunks...");
    let chunk_size = 1000;
    let mut chunk_count = 0;

    for (i, chunk_text) in content.as_bytes().chunks(chunk_size).enumerate() {
        let chunk_text_str = String::from_utf8_lossy(chunk_text).to_string();
        let start_offset = i * chunk_size;
        let end_offset = start_offset + chunk_text_str.len();

        let chunk = TextChunk::new(
            ChunkId::new(format!("chunk_{}", i)),
            DocumentId::new("tom_sawyer".to_string()),
            chunk_text_str,
            start_offset,
            end_offset,
        );
        graph.add_chunk(chunk)?;
        chunk_count += 1;
    }
    println!("   ✅ Created {} chunks", chunk_count);

    // === PHASE 3: Extract mock entities ===
    println!("\n👤 Phase 3: Extracting entities (mock)...");

    // Mock entities from Tom Sawyer
    let entities_data = vec![
        ("tom_sawyer_entity", "Tom Sawyer", "PERSON", 0.95),
        ("huck_finn", "Huckleberry Finn", "PERSON", 0.92),
        ("becky_thatcher", "Becky Thatcher", "PERSON", 0.90),
        ("aunt_polly", "Aunt Polly", "PERSON", 0.88),
        ("injun_joe", "Injun Joe", "PERSON", 0.87),
        ("st_petersburg", "St. Petersburg", "LOCATION", 0.85),
        ("mississippi_river", "Mississippi River", "LOCATION", 0.83),
    ];

    for (id, name, entity_type, confidence) in &entities_data {
        let entity = Entity::new(
            EntityId::new(id.to_string()),
            name.to_string(),
            entity_type.to_string(),
            *confidence,
        );
        graph.add_entity(entity)?;
    }
    println!("   ✅ Added {} entities", entities_data.len());

    // === PHASE 4: Create relationships ===
    println!("\n🔗 Phase 4: Creating relationships...");

    let relationships_data = vec![
        ("tom_sawyer_entity", "huck_finn", "FRIEND_OF", 0.90),
        ("tom_sawyer_entity", "becky_thatcher", "LOVES", 0.85),
        ("tom_sawyer_entity", "aunt_polly", "NEPHEW_OF", 0.95),
        ("huck_finn", "tom_sawyer_entity", "FRIEND_OF", 0.90),
        ("injun_joe", "st_petersburg", "LOCATED_IN", 0.80),
        ("st_petersburg", "mississippi_river", "NEAR", 0.88),
    ];

    for (source, target, rel_type, confidence) in &relationships_data {
        let relationship = Relationship {
            source: EntityId::new(source.to_string()),
            target: EntityId::new(target.to_string()),
            relation_type: rel_type.to_string(),
            confidence: *confidence,
            context: vec![ChunkId::new("chunk_0".to_string())],
        };
        graph.add_relationship(relationship)?;
    }
    println!("   ✅ Added {} relationships", relationships_data.len());

    // === PHASE 5: Display graph stats ===
    println!("\n📊 Knowledge Graph Statistics:");
    println!("   - Documents: {}", graph.document_count());
    println!("   - Chunks: {}", graph.chunks().count());
    println!("   - Entities: {}", graph.entity_count());
    println!("   - Relationships: {}", graph.relationship_count());

    // === PHASE 6: Save to workspace ===
    println!("\n💾 Phase 6: Saving to workspace...");
    let workspace_dir = "./workspace_test";
    let workspace = WorkspaceManager::new(workspace_dir)?;

    workspace.save_graph(&graph, "tom_sawyer")?;

    let workspaces = workspace.list_workspaces()?;
    if let Some(ws_info) = workspaces.iter().find(|w| w.name == "tom_sawyer") {
        println!("   ✅ Saved workspace: {} ({:.2} KB)",
                 ws_info.name,
                 ws_info.size_bytes as f64 / 1024.0);
        println!("      Created: {}", ws_info.metadata.created_at.format("%Y-%m-%d %H:%M:%S"));
    }

    // === PHASE 7: Load from workspace ===
    println!("\n📥 Phase 7: Loading from workspace...");
    let loaded_graph = workspace.load_graph("tom_sawyer")?;

    println!("   ✅ Loaded successfully");
    println!("\n📊 Loaded Knowledge Graph Statistics:");
    println!("   - Documents: {}", loaded_graph.document_count());
    println!("   - Chunks: {}", loaded_graph.chunks().count());
    println!("   - Entities: {}", loaded_graph.entity_count());
    println!("   - Relationships: {}", loaded_graph.relationship_count());

    // === PHASE 8: Verify integrity ===
    println!("\n🔍 Phase 8: Verifying data integrity...");

    let checks = vec![
        ("Documents", graph.document_count(), loaded_graph.document_count()),
        ("Chunks", graph.chunks().count(), loaded_graph.chunks().count()),
        ("Entities", graph.entity_count(), loaded_graph.entity_count()),
        ("Relationships", graph.relationship_count(), loaded_graph.relationship_count()),
    ];

    let mut all_passed = true;
    for (name, original, loaded) in &checks {
        if original == loaded {
            println!("   ✅ {} match: {}", name, original);
        } else {
            println!("   ❌ {} mismatch: {} != {}", name, original, loaded);
            all_passed = false;
        }
    }

    // Verify document content
    if let Some(original_doc) = graph.get_document(&DocumentId::new("tom_sawyer".to_string())) {
        if let Some(loaded_doc) = loaded_graph.get_document(&DocumentId::new("tom_sawyer".to_string())) {
            if original_doc.content.len() == loaded_doc.content.len() {
                println!("   ✅ Document content size matches: {} bytes", original_doc.content.len());
            } else {
                println!("   ❌ Document content size mismatch: {} != {}",
                         original_doc.content.len(), loaded_doc.content.len());
                all_passed = false;
            }
        }
    }

    // === PHASE 9: Sample data ===
    println!("\n📝 Phase 9: Sample loaded data...");

    println!("\n   Entities (first 3):");
    for (i, entity) in loaded_graph.entities().take(3).enumerate() {
        println!("      {}. {} ({}) [confidence: {:.2}]",
                 i + 1, entity.name, entity.entity_type, entity.confidence);
    }

    println!("\n   Relationships (first 3):");
    for (i, rel) in loaded_graph.relationships().take(3).enumerate() {
        println!("      {}. {} --[{}]--> {} [confidence: {:.2}]",
                 i + 1, rel.source.0, rel.relation_type, rel.target.0, rel.confidence);
    }

    // === FINAL RESULT ===
    println!("\n{}", "=".repeat(60));
    if all_passed {
        println!("✅ ALL TESTS PASSED!");
        println!("\n🎉 Tom Sawyer workspace persistence test completed successfully!");
    } else {
        println!("❌ SOME TESTS FAILED!");
        return Err("Data integrity check failed".into());
    }

    println!("\n📂 Workspace location: {}", workspace_dir);
    println!("   To inspect: cat {}/tom_sawyer/metadata.toml", workspace_dir);
    println!("   To list: ls -lh {}/", workspace_dir);

    Ok(())
}
