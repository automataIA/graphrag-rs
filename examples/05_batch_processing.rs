//! Batch processing example for GraphRAG-rs
//!
//! This example demonstrates how to process multiple documents in batch
//! and query across all of them.

use graphrag_rs::{GraphRAG, Document};
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("GraphRAG-rs Batch Processing Example\n");
    println!("====================================\n");

    // Multiple related documents about a company
    let documents = vec![
        (
            "company_overview",
            r#"
            TechVision Inc. Company Overview

            TechVision Inc. is a leading technology company founded in 2015
            by CEO Jane Smith and CTO Robert Johnson. Headquartered in
            Seattle, Washington, the company specializes in artificial
            intelligence and cloud computing solutions.

            The company has grown from a small startup of 5 employees to
            over 500 employees across three offices in Seattle, London,
            and Tokyo. TechVision's flagship product, CloudAI Platform,
            serves over 10,000 enterprise customers worldwide.
            "#,
            "corporate",
        ),
        (
            "product_lineup",
            r#"
            TechVision Product Portfolio

            1. CloudAI Platform: Our main enterprise AI solution that provides
               automated data analysis, predictive modeling, and natural
               language processing capabilities.

            2. VisionAnalytics: A business intelligence tool that transforms
               raw data into actionable insights using advanced visualization
               and machine learning algorithms.

            3. SecureCloud: An enterprise-grade cloud storage solution with
               end-to-end encryption and compliance with major regulatory
               standards including GDPR and HIPAA.

            4. DevOps Suite: A comprehensive set of tools for continuous
               integration and deployment, container orchestration, and
               infrastructure automation.
            "#,
            "products",
        ),
        (
            "financial_report",
            r#"
            TechVision Inc. Q4 2023 Financial Summary

            Revenue: TechVision reported record revenue of $450 million in
            Q4 2023, representing a 35% year-over-year growth. The CloudAI
            Platform contributed 60% of total revenue.

            Profitability: The company achieved its first profitable quarter
            with a net income of $25 million. Operating margin improved to
            15%, up from -5% in Q4 2022.

            Customer Growth: Added 1,500 new enterprise customers in Q4,
            bringing the total customer base to over 10,000. Customer
            retention rate remained strong at 95%.

            R&D Investment: Invested $75 million in research and development,
            focusing on next-generation AI capabilities and quantum computing
            research.
            "#,
            "financial",
        ),
        (
            "future_plans",
            r#"
            TechVision Strategic Roadmap 2024-2026

            Expansion Plans:
            - Opening new offices in Singapore and São Paulo by Q2 2024
            - Targeting 1,000 employees by end of 2024
            - Launching in 15 new markets across Asia and South America

            Product Development:
            - Release of CloudAI 3.0 with advanced multimodal capabilities
            - Launch of QuantumReady initiative for quantum computing services
            - Integration of blockchain technology for secure data sharing

            Partnerships:
            - Strategic alliance with major cloud providers (AWS, Azure, GCP)
            - Collaboration with leading universities for AI research
            - Partnership with healthcare providers for AI-driven diagnostics

            Sustainability:
            - Carbon neutral operations by 2025
            - 100% renewable energy for all data centers
            - Employee sustainability programs and green commute initiatives
            "#,
            "strategy",
        ),
    ];

    println!("Creating GraphRAG instance for batch processing...\n");

    // Configure for efficient batch processing
    let mut graphrag = GraphRAG::builder()
        .with_chunk_size(600)
        .with_chunk_overlap(100)
        .with_parallel_processing(true)  // Process documents in parallel
        .with_batch_size(50)             // Process chunks in batches
        .with_caching(true)              // Enable caching for repeated queries
        .auto_detect_llm()
        .build()?;

    println!("Processing {} documents in batch...\n", documents.len());

    // Process all documents
    for (doc_id, content, category) in &documents {
        println!("Adding document: {} (category: {})", doc_id, category);

        // Create document with metadata
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), category.to_string());
        metadata.insert("doc_id".to_string(), doc_id.to_string());

        let document = Document::new(doc_id, content.to_string())
            .with_metadata(metadata);

        graphrag.add_document(document)?;
    }

    println!("\nAll documents processed. Knowledge graph built.\n");
    println!("=" .repeat(60));

    // Query across all documents
    let cross_document_queries = vec![
        "Who founded TechVision Inc and when?",
        "What products does TechVision offer?",
        "How many employees does the company have?",
        "What was the Q4 2023 revenue?",
        "What are the expansion plans for 2024?",
        "Which cities have TechVision offices?",
        "What is the company's main product and its contribution to revenue?",
        "What sustainability goals has TechVision set?",
        "How has the company's profitability changed?",
        "What technologies is TechVision investing in for the future?",
    ];

    println!("\nQuerying across all documents:\n");

    for (i, query) in cross_document_queries.iter().enumerate() {
        println!("{}. Q: {}", i + 1, query);

        match graphrag.query_with_sources(query) {
            Ok(result) => {
                println!("   A: {}", result.answer);
                if !result.sources.is_empty() {
                    println!("   Sources: {:?}", result.sources);
                }
                println!("   Confidence: {:.2}%\n", result.confidence * 100.0);
            }
            Err(_) => {
                // Fallback to simple query
                if let Ok(answer) = graphrag.ask(query) {
                    println!("   A: {}\n", answer);
                }
            }
        }
    }

    // Demonstrate category-specific queries
    println!("=" .repeat(60));
    println!("\nCategory-Specific Analysis:\n");

    // Financial analysis
    println!("Financial Analysis:");
    let financial_query = "Provide a financial summary of TechVision";
    if let Ok(answer) = graphrag.ask(financial_query) {
        println!("Q: {}", financial_query);
        println!("A: {}\n", answer);
    }

    // Product analysis
    println!("Product Analysis:");
    let product_query = "What is TechVision's product strategy?";
    if let Ok(answer) = graphrag.ask(product_query) {
        println!("Q: {}", product_query);
        println!("A: {}\n", answer);
    }

    // Relationship analysis
    println!("Relationship Analysis:");
    let relationship_query = "How do TechVision's products relate to their financial performance?";
    if let Ok(answer) = graphrag.ask(relationship_query) {
        println!("Q: {}", relationship_query);
        println!("A: {}\n", answer);
    }

    // Save the knowledge graph for later use
    println!("=" .repeat(60));
    println!("\nSaving knowledge graph for future use...");

    if let Ok(()) = graphrag.save("./output/techvision_graph") {
        println!("✅ Knowledge graph saved to ./output/techvision_graph");
    }

    // Demonstrate loading and querying saved graph
    println!("\nLoading saved knowledge graph...");

    let mut loaded_graphrag = GraphRAG::builder()
        .auto_detect_llm()
        .build()?;

    if let Ok(()) = loaded_graphrag.load("./output/techvision_graph") {
        println!("✅ Knowledge graph loaded successfully");

        // Query the loaded graph
        let test_query = "What is TechVision's main office location?";
        if let Ok(answer) = loaded_graphrag.ask(test_query) {
            println!("\nVerification query on loaded graph:");
            println!("Q: {}", test_query);
            println!("A: {}", answer);
        }
    }

    println!("\n✅ Batch processing example completed successfully!");

    // Print statistics
    println!("\nProcessing Statistics:");
    println!("----------------------");
    println!("Documents processed: {}", documents.len());
    println!("Total text size: {} characters",
             documents.iter().map(|(_, content, _)| content.len()).sum::<usize>());
    println!("Queries executed: {}", cross_document_queries.len() + 3);

    Ok(())
}