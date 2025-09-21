//! Ollama integration example for GraphRAG-rs
//!
//! This example shows how to use GraphRAG-rs with Ollama for local LLM processing.
//! Prerequisites: Ollama must be installed and running with appropriate models.

use graphrag_rs::GraphRAG;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("GraphRAG-rs with Ollama Example\n");
    println!("================================\n");

    // Check if Ollama is available
    println!("Note: This example requires Ollama to be running locally.");
    println!("Install Ollama from: https://ollama.ai");
    println!("Required models: llama3.1:8b and nomic-embed-text\n");

    // Scientific article sample
    let scientific_text = r#"
        Research Paper: Advances in Quantum Computing

        Abstract:
        Quantum computing represents a fundamental shift in computational
        paradigms, leveraging quantum mechanical phenomena such as
        superposition and entanglement to process information in ways
        impossible for classical computers.

        Introduction:
        The field of quantum computing has seen remarkable progress in
        recent years. Major breakthroughs include the achievement of
        quantum supremacy by Google's Sycamore processor in 2019, and
        IBM's development of 127-qubit quantum processors.

        Key Concepts:
        1. Qubits: Unlike classical bits that exist in states 0 or 1,
           qubits can exist in superposition of both states simultaneously.

        2. Entanglement: Quantum particles can become correlated in ways
           that measuring one instantly affects the other, regardless of
           distance.

        3. Quantum Gates: Operations that manipulate qubits, forming the
           basis of quantum circuits and algorithms.

        Applications:
        - Cryptography: Quantum computers could break current encryption
          but also enable quantum-safe cryptography.
        - Drug Discovery: Simulating molecular interactions at quantum level.
        - Optimization: Solving complex optimization problems exponentially
          faster than classical computers.
        - Machine Learning: Quantum machine learning algorithms promise
          significant speedups for certain tasks.

        Challenges:
        The main obstacles include quantum decoherence, error rates,
        and the need for near-absolute zero temperatures for operation.
        Current quantum computers are in the NISQ (Noisy Intermediate-Scale
        Quantum) era, with limited coherence times and high error rates.

        Conclusion:
        While practical quantum computers remain years away, the rapid
        progress suggests revolutionary changes in computing are imminent.
    "#;

    // Configure Ollama with specific models
    println!("Configuring Ollama integration...\n");

    // Create GraphRAG with Ollama using builder pattern
    let mut graphrag = GraphRAG::builder()
        .with_ollama()                         // Enable Ollama with defaults
        .with_text_config(600, 150)            // chunk_size: 600, chunk_overlap: 150
        .build()?;

    println!("Processing document with Ollama...\n");
    graphrag.add_document_from_text(scientific_text)?;

    // Ask various types of questions
    let questions = vec![
        ("What is quantum computing?", "Basic definition question"),
        ("What are the main challenges facing quantum computing?", "Specific detail question"),
        ("How do qubits differ from classical bits?", "Comparison question"),
        ("What applications could benefit from quantum computing?", "Application question"),
        ("What breakthrough did Google achieve in 2019?", "Fact retrieval question"),
        ("How might quantum computing impact cryptography?", "Impact analysis question"),
    ];

    println!("Asking questions using Ollama LLM:\n");
    println!("{}", "=".repeat(50));

    for (question, question_type) in questions {
        println!("\nQuestion Type: {question_type}");
        println!("Q: {question}");

        match graphrag.ask(question) {
            Ok(answer) => {
                println!("A: {answer}");
            }
            Err(e) => {
                println!("Error: {e}. Make sure Ollama is running and models are installed.");
                println!("Install models with:");
                println!("  ollama pull llama3.1:8b");
                println!("  ollama pull nomic-embed-text");
            }
        }
        println!("{}", "-".repeat(50));
    }

    // Demonstrate additional GraphRAG features
    println!("\n\nAdditional Features:");
    println!("===================\n");

    // Using multiple GraphRAG instances for different types of analysis
    println!("1. Creating specialized GraphRAG instances:\n");

    // Analysis focused on factual extraction
    let mut factual_graph = GraphRAG::builder()
        .with_ollama()
        .build()?;

    factual_graph.add_document_from_text(scientific_text)?;

    println!("Factual analysis - List key concepts:");
    if let Ok(answer) = factual_graph.ask("List the key concepts in quantum computing") {
        println!("{answer}\n");
    }

    // Analysis focused on creative interpretation
    let mut creative_graph = GraphRAG::builder()
        .with_ollama()
        .build()?;

    creative_graph.add_document_from_text(scientific_text)?;

    println!("Creative analysis - Future implications:");
    if let Ok(answer) = creative_graph.ask("Imagine the future impact of quantum computing") {
        println!("{answer}\n");
    }

    println!("✅ Ollama integration example completed!");
    println!("\nNote: If you encountered errors, ensure:");
    println!("1. Ollama is installed and running (ollama serve)");
    println!("2. Required models are installed:");
    println!("   - ollama pull llama3.1:8b");
    println!("   - ollama pull nomic-embed-text");

    Ok(())
}