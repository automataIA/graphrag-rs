//! Builder API example of GraphRAG-rs
//!
//! This example demonstrates the configurable Builder API for advanced use cases
//! with custom configuration options.

use graphrag_rs::{GraphRAG, ConfigPreset, Document, DocumentId};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("GraphRAG-rs Builder API Example\n");
    println!("===============================\n");

    // Sample technical documentation
    let technical_doc = r#"
        GraphRAG Architecture Overview

        GraphRAG is a retrieval-augmented generation system that combines
        knowledge graphs with language models. The system consists of several
        key components:

        1. Text Processor: Chunks documents into manageable segments with
           configurable overlap to preserve context across boundaries.

        2. Entity Extractor: Identifies named entities (people, places,
           organizations, concepts) using NLP techniques.

        3. Graph Builder: Constructs a knowledge graph by discovering
           relationships between entities based on co-occurrence and
           semantic similarity.

        4. Vector Index: Creates embeddings for entities and chunks using
           transformer models, enabling semantic search.

        5. Query Engine: Processes natural language queries by combining
           graph traversal, vector similarity, and LLM generation.

        The system supports multiple retrieval strategies including:
        - Pure semantic search using vector similarity
        - Graph-based traversal for relationship queries
        - Hybrid approaches combining multiple strategies

        Performance optimizations include parallel processing, caching,
        and incremental updates for real-time applications.
    "#;

    // Example 1: Using presets
    println!("1. Building with Basic Preset:");
    println!("----------------------------------");

    let mut balanced_graph = GraphRAG::builder()
        .with_preset(ConfigPreset::Basic)
        .auto_detect_llm()
        .build()?;

    balanced_graph.add_document_from_text(technical_doc)?;
    let answer = balanced_graph.ask("What are the main components of GraphRAG?")?;
    println!("Q: What are the main components of GraphRAG?");
    println!("A: {}\n", answer);

    // Example 2: Custom configuration
    println!("2. Building with Custom Configuration:");
    println!("--------------------------------------");

    let mut custom_graph = GraphRAG::builder()
        .with_text_config(500, 100)    // chunk_size: 500, chunk_overlap: 100
        .with_parallel_processing(true, Some(4)) // Enable with 4 threads
        .auto_detect_llm()
        .build()?;

    custom_graph.add_document_from_text(technical_doc)?;
    let answer = custom_graph.ask("What retrieval strategies are available?")?;
    println!("Q: What retrieval strategies are available?");
    println!("A: {}\n", answer);

    // Example 3: Performance-optimized configuration
    println!("3. Building with Performance Optimization:");
    println!("------------------------------------------");

    #[cfg(feature = "caching")]
    let mut performance_graph = GraphRAG::builder()
        .with_preset(ConfigPreset::PerformanceOptimized)
        .with_caching()                // Enable caching with defaults
        .with_parallel_processing(true, None) // Auto-detect thread count
        .auto_detect_llm()
        .build()?;

    #[cfg(not(feature = "caching"))]
    let mut performance_graph = GraphRAG::builder()
        .with_preset(ConfigPreset::PerformanceOptimized)
        .with_parallel_processing(true, None) // Auto-detect thread count
        .auto_detect_llm()
        .build()?;

    // Create a document with metadata
    let document = Document::new(
        DocumentId::new("doc1".to_string()),
        "Technical Documentation".to_string(),
        technical_doc.to_string(),
    )
    .with_metadata("source".to_string(), "technical_documentation".to_string())
    .with_metadata("version".to_string(), "1.0".to_string())
    .with_metadata("category".to_string(), "architecture".to_string());

    performance_graph.add_document(document)?;

    let answer = performance_graph.ask("How does GraphRAG optimize performance?")?;
    println!("Q: How does GraphRAG optimize performance?");
    println!("A: {}\n", answer);

    // Example 4: Memory-optimized configuration for large documents
    println!("4. Building with Memory Optimization:");
    println!("-------------------------------------");

    let mut memory_graph = GraphRAG::builder()
        .with_preset(ConfigPreset::MemoryOptimized)
        .with_text_config(300, 30)     // Small chunks for less memory
        .without_parallel_processing() // Sequential to save memory
        .auto_detect_llm()
        .build()?;

    memory_graph.add_document_from_text(technical_doc)?;
    let answer = memory_graph.ask("Explain the Vector Index component")?;
    println!("Q: Explain the Vector Index component");
    println!("A: {}\n", answer);

    println!("✅ Builder API example completed successfully!");

    Ok(())
}