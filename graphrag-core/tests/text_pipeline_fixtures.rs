//! Fixture-Based Integration Tests for Text Processing Pipeline
//!
//! These tests use REAL text from fixture files to verify that the entire
//! text processing pipeline works correctly on realistic data.
//!
//! Test Type: Integration Tests (multiple components, no external services)
//! Data Source: Real text documents from tests/fixtures/documents/

use graphrag_core::{Document, DocumentId};
use graphrag_core::text::{TextProcessor, ChunkEnricher};
use graphrag_core::text::parsers::MarkdownLayoutParser;
use std::fs;

/// Helper function to load fixture files
fn load_fixture(filename: &str) -> String {
    let path = format!("tests/fixtures/documents/{}", filename);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to load fixture '{}': {}", filename, e))
}

/// Test 1: Complete pipeline on real article
///
/// This test verifies the entire text processing workflow:
/// - Document creation
/// - Text chunking with overlap
/// - Chunk enrichment (keywords, summaries, metadata)
/// - Heading detection and hierarchy
#[test]
fn test_complete_pipeline_on_real_article() {
    // 1. Load REAL text from fixture
    let content = load_fixture("sample_article.txt");

    // Verify fixture loaded correctly
    assert!(content.len() > 1000, "Fixture should contain substantial content");
    assert!(content.contains("Knowledge Graphs"), "Fixture should contain expected content");

    // 2. Create document
    let document = Document::new(
        DocumentId::new("fixture_test_article".to_string()),
        "sample_article.txt".to_string(),
        content,
    );

    // 3. Run text processing pipeline (INTEGRATION of multiple components)
    let processor = TextProcessor::new(500, 50)
        .expect("TextProcessor should initialize");

    let chunks = processor.chunk_and_enrich(&document)
        .expect("Pipeline should process document successfully");

    // 4. Verify results against expectations for this specific document

    // Should create multiple chunks from this ~3000 char document
    assert!(
        chunks.len() >= 4,
        "Expected at least 4 chunks for this document, got {}",
        chunks.len()
    );

    // At least some chunks should have chapter metadata (heading detected)
    let chunks_with_headings: Vec<_> = chunks
        .iter()
        .filter(|c| c.metadata.chapter.is_some())
        .collect();

    assert!(
        !chunks_with_headings.is_empty(),
        "Expected some chunks to have detected headings"
    );

    // Check that headings contain expected content
    let has_intro_heading = chunks.iter().any(|c| {
        c.metadata.chapter.as_ref()
            .map(|ch| ch.contains("Introduction"))
            .unwrap_or(false)
    });

    assert!(
        has_intro_heading,
        "Expected to find 'Introduction' heading in chunk metadata"
    );

    // Verify chunk content integrity
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            !chunk.content.is_empty(),
            "Chunk {} should have non-empty content",
            i
        );
        assert!(
            chunk.content.len() <= 600, // 500 + some tolerance for word boundaries
            "Chunk {} exceeds expected size: {} chars",
            i,
            chunk.content.len()
        );
    }

    // Verify keywords were extracted for at least some chunks
    let chunks_with_keywords: Vec<_> = chunks
        .iter()
        .filter(|c| !c.metadata.keywords.is_empty())
        .collect();

    assert!(
        !chunks_with_keywords.is_empty(),
        "Expected some chunks to have extracted keywords"
    );
}

/// Test 2: Markdown parsing with real technical document
///
/// Verifies that Markdown heading hierarchy is correctly detected
/// from a real technical document with multiple heading levels.
#[test]
fn test_markdown_parsing_on_technical_doc() {
    // Load Markdown fixture
    let content = load_fixture("markdown_technical.md");

    assert!(content.contains("# GraphRAG System Architecture"));
    assert!(content.contains("## Overview"));
    assert!(content.contains("### System Components"));

    let document = Document::new(
        DocumentId::new("fixture_markdown".to_string()),
        "markdown_technical.md".to_string(),
        content,
    );

    // Use TextProcessor with Markdown-aware enricher
    let parser = Box::new(MarkdownLayoutParser::new());
    let mut enricher = ChunkEnricher::new_default(parser);
    let processor = TextProcessor::new(400, 50)
        .expect("TextProcessor should initialize");

    let chunks = processor.chunk_text_with_enrichment(&document, &mut enricher)
        .expect("Should process Markdown document");

    // Verify multiple chunks created
    assert!(chunks.len() >= 3, "Should create multiple chunks");

    // Verify heading hierarchy detection
    let h1_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| {
            c.metadata.chapter.as_ref()
                .map(|ch| ch.starts_with("GraphRAG System"))
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !h1_chunks.is_empty(),
        "Should detect H1 heading 'GraphRAG System Architecture'"
    );

    // Check for nested heading structure
    let has_h2 = chunks.iter().any(|c| {
        c.metadata.section.as_ref()
            .map(|s| s.contains("Overview") || s.contains("Implementation"))
            .unwrap_or(false)
    });

    assert!(
        has_h2,
        "Should detect H2 headings like 'Overview' or 'Implementation Details'"
    );

    // Verify subsections detected
    let has_subsections = chunks.iter().any(|c| {
        c.metadata.subsection.is_some()
    });

    assert!(
        has_subsections,
        "Should detect H3/H4 subsections in the document"
    );
}

/// Test 3: Keyword extraction quality on domain-specific content
///
/// Verifies that TF-IDF keyword extraction produces meaningful
/// keywords from technical content.
#[test]
fn test_keyword_extraction_quality() {
    let content = load_fixture("sample_article.txt");

    let document = Document::new(
        DocumentId::new("fixture_keywords".to_string()),
        "sample_article.txt".to_string(),
        content,
    );

    let processor = TextProcessor::new(800, 50)
        .expect("TextProcessor should initialize");

    let chunks = processor.chunk_and_enrich(&document)
        .expect("Should process document");

    // Find chunks about "Knowledge Graphs"
    let kg_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.content.contains("knowledge graph"))
        .collect();

    assert!(!kg_chunks.is_empty(), "Should have chunks about knowledge graphs");

    // Check keyword quality in these chunks
    for chunk in &kg_chunks {
        if chunk.metadata.keywords.is_empty() {
            continue; // Skip chunks without keywords
        }

        let keywords_str = chunk.metadata.keywords.join(" ");

        // Keywords should be lowercase
        assert!(
            keywords_str.chars().all(|c| !c.is_uppercase() || !c.is_alphabetic()),
            "Keywords should be normalized to lowercase"
        );

        // Should extract domain-relevant terms (not just stop words)
        let has_relevant_term = chunk.metadata.keywords.iter().any(|kw| {
            kw.len() > 3 && // Not too short
            !["about", "there", "their", "would", "could"].contains(&kw.as_str())
        });

        assert!(
            has_relevant_term,
            "Keywords should include meaningful domain terms, got: {:?}",
            chunk.metadata.keywords
        );
    }
}

/// Test 4: Chunk overlap consistency
///
/// Verifies that overlapping chunks maintain continuity
/// and don't lose information at boundaries.
#[test]
fn test_chunk_overlap_on_real_text() {
    let content = load_fixture("markdown_technical.md");

    let document = Document::new(
        DocumentId::new("fixture_overlap".to_string()),
        "markdown_technical.md".to_string(),
        content.clone(),
    );

    let overlap = 100;
    let processor = TextProcessor::new(500, overlap)
        .expect("TextProcessor should initialize");

    let chunks = processor.chunk_text(&document)
        .expect("Should chunk document");

    assert!(chunks.len() >= 2, "Need at least 2 chunks to test overlap");

    // Verify overlap between consecutive chunks
    for i in 0..(chunks.len() - 1) {
        let current = &chunks[i];
        let next = &chunks[i + 1];

        // Extract last N chars of current chunk
        let current_end = if current.content.len() > overlap {
            &current.content[current.content.len() - overlap..]
        } else {
            &current.content
        };

        // Extract first N chars of next chunk
        let next_start = if next.content.len() > overlap {
            &next.content[..overlap]
        } else {
            &next.content
        };

        // Check for meaningful overlap (not necessarily exact due to word boundaries)
        // At least some content should be similar
        let has_overlap = next.content.contains(&current_end[..50])
            || current_end.contains(&next_start[..50]);

        assert!(
            has_overlap,
            "Chunks {} and {} should have overlapping content",
            i,
            i + 1
        );
    }
}

/// Test 5: Document statistics accuracy
///
/// Verifies that computed statistics match the actual document properties.
#[test]
fn test_document_statistics_on_real_content() {
    let content = load_fixture("sample_article.txt");

    // Manually count expected properties
    let line_count = content.lines().count();
    let word_count = content.split_whitespace().count();
    let char_count = content.len();

    assert!(line_count > 50, "Fixture should have substantial line count");
    assert!(word_count > 400, "Fixture should have substantial word count");

    let document = Document::new(
        DocumentId::new("fixture_stats".to_string()),
        "sample_article.txt".to_string(),
        content,
    );

    let processor = TextProcessor::new(500, 50)
        .expect("TextProcessor should initialize");

    let chunks = processor.chunk_text(&document)
        .expect("Should process document");

    // Verify total chunk content roughly equals original
    let total_chunk_chars: usize = chunks.iter().map(|c| c.content.len()).sum();

    // Should be roughly similar (allowing for overlap)
    assert!(
        total_chunk_chars >= char_count,
        "Total chunk content should cover original document (with overlap)"
    );

    // Verify each chunk has position information
    for chunk in &chunks {
        assert!(
            chunk.start_offset <= chunk.end_offset,
            "Chunk offsets should be valid"
        );
        assert!(
            chunk.end_offset <= char_count + 1000, // Allow for some tolerance
            "Chunk end offset should be within document bounds"
        );
    }
}

/// Test 6: Empty and edge case handling
///
/// Verifies graceful handling of edge cases with real file structure.
#[test]
fn test_edge_cases_with_fixtures() {
    // Test with minimal content
    let minimal = "# Title\n\nSingle paragraph.";

    let document = Document::new(
        DocumentId::new("edge_minimal".to_string()),
        "minimal.txt".to_string(),
        minimal.to_string(),
    );

    let processor = TextProcessor::new(100, 20)
        .expect("TextProcessor should initialize");

    let chunks = processor.chunk_and_enrich(&document)
        .expect("Should handle minimal document");

    assert!(chunks.len() >= 1, "Should create at least one chunk");
    assert!(chunks[0].metadata.chapter.is_some(), "Should detect title heading");
}
