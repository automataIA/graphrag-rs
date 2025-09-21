//! Basic usage example of GraphRAG-rs
//!
//! This example demonstrates the simplest way to use GraphRAG-rs
//! to process a document and ask questions about it.

use graphrag_rs::simple;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("GraphRAG-rs Basic Usage Example\n");
    println!("================================\n");

    // Sample document text
    let document = r#"
        Machine learning is a field of computer science that gives computers
        the ability to learn without being explicitly programmed. It evolved
        from the study of pattern recognition and computational learning theory
        in artificial intelligence.

        Deep learning is part of a broader family of machine learning methods
        based on artificial neural networks. The adjective "deep" refers to
        the use of multiple layers in the network.

        Natural language processing (NLP) is a subfield of linguistics, computer
        science, and artificial intelligence concerned with the interactions
        between computers and human language.
    "#;

    println!("Processing document...\n");

    // Example 1: One-line API
    println!("1. Using One-Line API:");
    println!("----------------------");

    let answer = simple::answer(document, "What is machine learning?")?;
    println!("Q: What is machine learning?");
    println!("A: {}\n", answer);

    // Example 2: Another question
    let answer2 = simple::answer(document, "How does deep learning relate to machine learning?")?;
    println!("Q: How does deep learning relate to machine learning?");
    println!("A: {}\n", answer2);

    // Example 3: Asking about relationships
    let answer3 = simple::answer(document, "What are the main AI fields mentioned?")?;
    println!("Q: What are the main AI fields mentioned?");
    println!("A: {}\n", answer3);

    println!("✅ Basic usage example completed successfully!");

    Ok(())
}