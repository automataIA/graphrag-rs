//! Document Enrichment Demo
//!
//! This example demonstrates the complete document structure recognition and
//! metadata enrichment pipeline, showing how to:
//! 1. Parse document structure (headings, sections, hierarchy)
//! 2. Extract semantic metadata (keywords, summaries)
//! 3. Enrich text chunks with contextual information
//! 4. Display enrichment statistics

use graphrag_core::{
    core::{Document, DocumentId},
    text::{ChunkEnricher, LayoutParser, TextProcessor, parsers::*},
    Result,
};

fn main() -> Result<()> {
    println!("=== Document Enrichment Demo ===\n");

    // Demo 1: Markdown Document
    demo_markdown_document()?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 2: HTML Document
    demo_html_document()?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 3: Plain Text Document
    demo_plaintext_document()?;

    println!("\n{}\n", "=".repeat(80));

    // Demo 4: Comparison of enrichment strategies
    demo_enrichment_comparison()?;

    Ok(())
}

fn demo_markdown_document() -> Result<()> {
    println!("Demo 1: Markdown Document Enrichment\n");

    let markdown_content = r#"# Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms
that learn from data. Deep learning and neural networks are powerful techniques
used in modern machine learning applications.

## Section 1.1: Supervised Learning

Supervised learning algorithms learn from labeled training data. Common techniques
include linear regression, decision trees, and support vector machines. These
methods are used extensively in classification and regression tasks.

### Subsection 1.1.1: Classification Tasks

Classification involves predicting categorical labels. Popular algorithms include
logistic regression, random forests, and gradient boosting machines.

## Section 1.2: Unsupervised Learning

Unsupervised learning discovers patterns in unlabeled data. Clustering algorithms
like k-means and hierarchical clustering are commonly used.

# Chapter 2: Deep Learning

Deep learning uses artificial neural networks with multiple layers. Convolutional
neural networks excel at computer vision tasks, while recurrent neural networks
are effective for sequential data.
"#;

    let document = Document::new(
        DocumentId::new("ml_guide".to_string()),
        "machine_learning_guide.md".to_string(),
        markdown_content.to_string(),
    );

    // Create processor and enricher
    let processor = TextProcessor::new(300, 50)?;

    println!("📄 Processing: {}", document.title);
    println!("📏 Document size: {} characters\n", document.content.len());

    // Use convenience method for automatic enrichment
    let chunks = processor.chunk_and_enrich(&document)?;

    println!("✅ Generated {} enriched chunks\n", chunks.len());

    // Display first few chunks with their metadata
    for (idx, chunk) in chunks.iter().take(3).enumerate() {
        println!("Chunk #{}", idx + 1);
        println!("  Content: {}...", &chunk.content[..chunk.content.len().min(80)]);

        if let Some(chapter) = &chunk.metadata.chapter {
            println!("  📖 Chapter: {}", chapter);
        }
        if let Some(section) = &chunk.metadata.section {
            println!("  📑 Section: {}", section);
        }
        if !chunk.metadata.keywords.is_empty() {
            println!("  🔑 Keywords: {}", chunk.metadata.keywords.join(", "));
        }
        if let Some(summary) = &chunk.metadata.summary {
            println!("  📝 Summary: {}", summary);
        }
        if let Some(pos) = chunk.metadata.position_in_document {
            println!("  📍 Position: {:.1}%", pos * 100.0);
        }
        println!();
    }

    // Create enricher manually for statistics
    let parser = Box::new(MarkdownLayoutParser::new());
    let mut enricher = ChunkEnricher::new_default(parser);
    let mut chunks_copy = processor.chunk_text(&document)?;
    enricher.enrich_chunks(&mut chunks_copy, &document)?;

    // Display statistics
    let stats = enricher.get_statistics(&chunks_copy);
    stats.print_summary();

    Ok(())
}

fn demo_html_document() -> Result<()> {
    println!("Demo 2: HTML Document Enrichment\n");

    let html_content = r#"
<!DOCTYPE html>
<html>
<head><title>Web Development Guide</title></head>
<body>
<h1>Introduction to Web Development</h1>
<p>Web development encompasses frontend and backend technologies. HTML, CSS, and
JavaScript form the foundation of modern web applications.</p>

<h2>Frontend Technologies</h2>
<p>Frontend development focuses on user interface and user experience. React, Vue,
and Angular are popular JavaScript frameworks for building interactive applications.</p>

<h3>React Framework</h3>
<p>React is a component-based library developed by Facebook. It uses virtual DOM
for efficient rendering and has a large ecosystem of tools and libraries.</p>

<h2>Backend Technologies</h2>
<p>Backend development handles server-side logic, databases, and APIs. Node.js,
Python Django, and Ruby on Rails are common backend frameworks.</p>

<h1>Database Design</h1>
<p>Database design is crucial for application performance. SQL and NoSQL databases
each have their strengths for different use cases.</p>
</body>
</html>
"#;

    let document = Document::new(
        DocumentId::new("web_guide".to_string()),
        "web_development.html".to_string(),
        html_content.to_string(),
    );

    let processor = TextProcessor::new(250, 40)?;

    println!("📄 Processing: {}", document.title);

    // Manual enrichment with HTML parser
    let parser = Box::new(HtmlLayoutParser::new());
    let mut enricher = ChunkEnricher::new_default(parser);
    let chunks = processor.chunk_text_with_enrichment(&document, &mut enricher)?;

    println!("✅ Generated {} enriched chunks\n", chunks.len());

    // Show hierarchy
    println!("📊 Document Structure:");
    for (idx, chunk) in chunks.iter().enumerate() {
        if !chunk.metadata.heading_path.is_empty() {
            let indent = "  ".repeat(chunk.metadata.heading_path.len() - 1);
            println!("  {}└─ Chunk {} at: {}", indent, idx + 1, chunk.metadata.heading_path.last().unwrap());
        }
    }

    Ok(())
}

fn demo_plaintext_document() -> Result<()> {
    println!("Demo 3: Plain Text Document Enrichment\n");

    let plaintext_content = r#"INTRODUCTION

This document covers programming fundamentals. Variables, data structures, and
algorithms are essential concepts for software development.

Basic Data Structures
=====================

Arrays store elements in contiguous memory locations. Linked lists provide dynamic
memory allocation and efficient insertion operations.

ALGORITHMS

Sorting and searching algorithms are fundamental operations. Binary search provides
logarithmic time complexity for sorted arrays.

1. Sorting Algorithms

QuickSort uses divide-and-conquer strategy. MergeSort guarantees stable sorting
with n log n time complexity.

2. Search Algorithms

Linear search checks each element sequentially. Hash tables provide constant-time
average case lookups.
"#;

    let document = Document::new(
        DocumentId::new("prog_guide".to_string()),
        "programming_fundamentals.txt".to_string(),
        plaintext_content.to_string(),
    );

    let processor = TextProcessor::new(200, 30)?;

    println!("📄 Processing: {}", document.title);

    // Use automatic parser detection
    let chunks = processor.chunk_hierarchical_and_enrich(&document)?;

    println!("✅ Generated {} enriched chunks with hierarchical boundaries\n", chunks.len());

    // Show detected structure
    println!("🔍 Detected Structure:");
    let parser = Box::new(PlainTextLayoutParser::new());
    let structure = parser.parse(&document.content);

    for heading in structure.headings.iter() {
        let indent = "  ".repeat(heading.level as usize);
        println!("  {}Level {}: {}", indent, heading.level, heading.text);
    }

    println!("\n📊 Chunk Metadata Coverage:");
    let with_chapter = chunks.iter().filter(|c| c.metadata.chapter.is_some()).count();
    let with_keywords = chunks.iter().filter(|c| !c.metadata.keywords.is_empty()).count();
    let with_summary = chunks.iter().filter(|c| c.metadata.summary.is_some()).count();

    println!("  Chunks with chapter info: {}/{}", with_chapter, chunks.len());
    println!("  Chunks with keywords: {}/{}", with_keywords, chunks.len());
    println!("  Chunks with summaries: {}/{}", with_summary, chunks.len());

    Ok(())
}

fn demo_enrichment_comparison() -> Result<()> {
    println!("Demo 4: Enrichment Strategy Comparison\n");

    let content = r#"# Technical Documentation

## Overview

This section provides technical specifications and implementation details.

## Architecture

The system uses microservices architecture with containerized deployments.
Docker and Kubernetes orchestrate service management and scaling operations.

### Database Layer

PostgreSQL handles relational data storage. Redis provides caching functionality
for frequently accessed data and session management.
"#;

    let document = Document::new(
        DocumentId::new("tech_doc".to_string()),
        "technical_doc.md".to_string(),
        content.to_string(),
    );

    let processor = TextProcessor::new(150, 25)?;

    // Strategy 1: Standard chunking without enrichment
    let standard_chunks = processor.chunk_text(&document)?;
    println!("📦 Standard Chunking:");
    println!("  Chunks: {}", standard_chunks.len());
    println!("  Metadata: None");
    println!();

    // Strategy 2: Hierarchical chunking with enrichment
    let enriched_chunks = processor.chunk_hierarchical_and_enrich(&document)?;
    println!("🎯 Hierarchical + Enrichment:");
    println!("  Chunks: {}", enriched_chunks.len());

    let keywords_count: usize = enriched_chunks.iter()
        .map(|c| c.metadata.keywords.len())
        .sum();
    let avg_keywords = keywords_count as f32 / enriched_chunks.len() as f32;

    println!("  Avg keywords per chunk: {:.1}", avg_keywords);
    println!("  Chunks with structure: {}",
        enriched_chunks.iter().filter(|c| c.metadata.chapter.is_some()).count());

    // Show metadata completeness
    println!("\n📈 Metadata Completeness:");
    for (idx, chunk) in enriched_chunks.iter().enumerate() {
        let completeness = chunk.metadata.completeness_score();
        println!("  Chunk {}: {:.0}%", idx + 1, completeness * 100.0);
    }

    Ok(())
}
