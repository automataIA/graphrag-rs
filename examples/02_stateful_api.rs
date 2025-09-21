//! Stateful API example of GraphRAG-rs
//!
//! This example shows how to use the stateful API to process a document once
//! and ask multiple questions without reprocessing.

use graphrag_rs::easy::SimpleGraphRAG;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("GraphRAG-rs Stateful API Example\n");
    println!("=================================\n");

    // Sample document about a fictional story
    let document = r#"
        Tom is a software engineer who works at TechCorp in San Francisco.
        He specializes in machine learning and has been working on a project
        called "SmartAssist" for the past year. The project uses natural
        language processing to help users automate their daily tasks.

        Sarah, Tom's colleague, is a data scientist who joined the team
        six months ago. She brought expertise in deep learning models and
        helped improve the accuracy of SmartAssist by 40%.

        The team recently presented their work at the AI Conference in
        New York, where they won the Best Innovation Award. The project
        is scheduled to launch next quarter.
    "#;

    println!("Creating knowledge graph from document...\n");

    // Create a stateful GraphRAG instance
    let mut graph = SimpleGraphRAG::from_text(document)?;

    println!("Knowledge graph created! Now asking multiple questions:\n");

    // Ask multiple questions without reprocessing
    let questions = [
        "Who works at TechCorp?",
        "What is SmartAssist?",
        "What did Sarah contribute to the project?",
        "Where was the AI Conference held?",
        "What award did they win?",
    ];

    for (i, question) in questions.iter().enumerate() {
        println!("{}. Q: {}", i + 1, question);
        let answer = graph.ask(question)?;
        println!("   A: {answer}\n");
    }

    // Demonstrate adding more content
    println!("Adding additional information to the graph...\n");

    let additional_info = r#"
        After the conference, Tom and Sarah received funding from VentureCapital
        to expand SmartAssist into a full platform. They hired five new engineers
        and plan to open an office in Austin, Texas.
    "#;

    graph.add_text(additional_info)?;

    // Ask questions about the new information
    println!("Q: What happened after the conference?");
    let answer = graph.ask("What happened after the conference?")?;
    println!("A: {answer}\n");

    println!("Q: Where will the new office be?");
    let answer = graph.ask("Where will the new office be?")?;
    println!("A: {answer}\n");

    println!("✅ Stateful API example completed successfully!");

    Ok(())
}