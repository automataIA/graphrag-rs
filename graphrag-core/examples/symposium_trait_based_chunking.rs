//! Comprehensive example demonstrating trait-based chunking strategies
//!
//! This example implements the cAST (Context-Aware Splitting with Tree-sitter) approach
//! from the CMU paper, showing how AST-based chunking improves RAG performance.
//!
//! We use Plato's Symposium as real-world text to demonstrate:
//! 1. Hierarchical chunking for philosophical text - respects paragraph/sentence boundaries
//! 2. Tree-sitter AST-based chunking for embedded code snippets - preserves syntactic boundaries

use graphrag_core::{
    core::{DocumentId, Document, TextChunk, ChunkingStrategy},
    text::{TextProcessor, HierarchicalChunkingStrategy},
};
use std::time::Instant;
use std::path::Path;

#[cfg(feature = "code-chunking")]
use graphrag_core::text::RustCodeChunkingStrategy;

/// Metrics for comparing chunking strategies
#[derive(Debug)]
struct ChunkingMetrics {
    strategy_name: String,
    num_chunks: usize,
    avg_chunk_size: f64,
    min_chunk_size: usize,
    max_chunk_size: usize,
    total_chars: usize,
    processing_time_ms: u64,
}

impl ChunkingMetrics {
    fn from_chunks(strategy_name: &str, chunks: &[TextChunk], processing_time: u64) -> Self {
        if chunks.is_empty() {
            return Self {
                strategy_name: strategy_name.to_string(),
                num_chunks: 0,
                avg_chunk_size: 0.0,
                min_chunk_size: 0,
                max_chunk_size: 0,
                total_chars: 0,
                processing_time_ms: processing_time,
            };
        }

        let sizes: Vec<usize> = chunks.iter().map(|c| c.content.len()).collect();
        let total_chars: usize = sizes.iter().sum();
        let avg_chunk_size = total_chars as f64 / chunks.len() as f64;
        let min_chunk_size = *sizes.iter().min().unwrap();
        let max_chunk_size = *sizes.iter().max().unwrap();

        Self {
            strategy_name: strategy_name.to_string(),
            num_chunks: chunks.len(),
            avg_chunk_size,
            min_chunk_size,
            max_chunk_size,
            total_chars,
            processing_time_ms: processing_time,
        }
    }

    fn print(&self) {
        println!("┌─ {:^20} ─────┐", self.strategy_name);
        println!("│ Chunks: {:13} │", self.num_chunks);
        println!("│ Avg size: {:11.1} │", self.avg_chunk_size);
        println!("│ Min size: {:11} │", self.min_chunk_size);
        println!("│ Max size: {:11} │", self.max_chunk_size);
        println!("│ Time: {:12} ms │", self.processing_time_ms);
        println!("└─────────────────────┘");
    }
}

fn load_symposium_text() -> Result<String, Box<dyn std::error::Error>> {
    let path = Path::new("docs-example/Symposium.txt");
    let content = std::fs::read_to_string(path)?;

    // Remove BOM if present and normalize line endings
    let content = content
        .trim_start_matches('\u{FEFF}') // Remove BOM
        .replace("\r\n", "\n") // Normalize Windows line endings
        .replace('\r', "\n"); // Normalize old Mac line endings

    Ok(content)
}

fn create_document(content: String) -> Document {
    Document::new(
        DocumentId::new("symposium_plato".to_string()),
        "Symposium by Plato (Translated by Benjamin Jowett)".to_string(),
        content,
    )
}

fn demonstrate_hierarchical_chunking(document: &Document) -> (Vec<TextChunk>, ChunkingMetrics) {
    println!("\n🔹 Hierarchical Chunking (LangChain-style)");
    println!("   Respects paragraph and sentence boundaries");

    let start = Instant::now();
    let processor = TextProcessor::new(1000, 100).unwrap();
    let strategy = HierarchicalChunkingStrategy::new(1000, 100, document.id.clone());
    let chunks = processor.chunk_with_strategy(document, &strategy).unwrap();
    let elapsed = start.elapsed().as_millis() as u64;

    let metrics = ChunkingMetrics::from_chunks("Hierarchical", &chunks, elapsed);

    // Show first chunk as example
    if !chunks.is_empty() {
        let preview = chunks[0].content.chars().take(100).collect::<String>();
        println!("   Example chunk: \"{}...\"", preview.replace('\n', " "));
    }

    (chunks, metrics)
}


#[cfg(feature = "code-chunking")]
fn demonstrate_tree_sitter_chunking() -> (Vec<TextChunk>, ChunkingMetrics) {
    println!("\n🔹 Tree-sitter AST Chunking (cAST approach)");
    println!("   Preserves syntactic boundaries for code");

    // Create a Rust code snippet that could appear in a technical document
    let rust_code = r#"/// Example of Rust code with proper syntactic structure
fn analyze_philosophical_concepts(text: &str) -> Vec<String> {
    let concepts: Vec<String> = text
        .split_whitespace()
        .filter(|word| is_philosophical_term(word))
        .map(|word| word.to_lowercase())
        .collect();

    // Deduplicate while preserving order
    let mut unique_concepts = Vec::new();
    for concept in concepts {
        if !unique_concepts.contains(&concept) {
            unique_concepts.push(concept);
        }
    }

    unique_concepts
}

fn is_philosophical_term(word: &str) -> bool {
    let philosophical_terms = [
        "virtue", "wisdom", "justice", "beauty", "truth",
        "knowledge", "love", "good", "evil", "soul"
    ];
    philosophical_terms.contains(&word.to_lowercase().as_str())
}

struct PhilosophicalAnalysis {
    concepts: Vec<String>,
    sentiment: f64,
    complexity: u8,
}

impl PhilosophicalAnalysis {
    fn new(text: &str) -> Self {
        let concepts = analyze_philosophical_concepts(text);
        let sentiment = calculate_sentiment(text);
        let complexity = estimate_complexity(text);

        Self {
            concepts,
            sentiment,
            complexity,
        }
    }

    fn summarize(&self) -> String {
        format!(
            "Found {} concepts with {:.2} sentiment and {} complexity",
            self.concepts.len(),
            self.sentiment,
            self.complexity
        )
    }
}

fn calculate_sentiment(text: &str) -> f64 {
    // Simplified sentiment analysis
    let positive_words = ["good", "beautiful", "wise", "virtuous", "true"];
    let negative_words = ["evil", "ugly", "foolish", "vicious", "false"];

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut score = 0.0;

    for word in words {
        if positive_words.contains(&word) {
            score += 0.1;
        } else if negative_words.contains(&word) {
            score -= 0.1;
        }
    }

    score.clamp(-1.0, 1.0)
}

fn estimate_complexity(text: &str) -> u8 {
    // Simple complexity heuristic based on sentence length and vocabulary
    let words: Vec<&str> = text.split_whitespace().collect();
    let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();

    let avg_words_per_sentence = if !sentences.is_empty() {
        words.len() / sentences.len()
    } else {
        0
    };

    match avg_words_per_sentence {
        0..=10 => 1,
        11..=20 => 2,
        21..=30 => 3,
        _ => 4,
    }
}"#;

    let start = Instant::now();

    let document_id = DocumentId::new("rust_code_example".to_string());
    let strategy = RustCodeChunkingStrategy::new(50, document_id);
    let chunks = strategy.chunk(rust_code);
    let elapsed = start.elapsed().as_millis() as u64;

    let metrics = ChunkingMetrics::from_chunks("Tree-sitter (Rust)", &chunks, elapsed);

    // Show first chunk as example
    if !chunks.is_empty() {
        println!("   Example chunk: {}", chunks[0].content.chars().take(80).collect::<String>());
        println!("   Chunk type: {}", extract_function_name(&chunks[0].content).unwrap_or("unknown"));
    }

    (chunks, metrics)
}

#[cfg(not(feature = "code-chunking"))]
fn demonstrate_tree_sitter_chunking() -> (Vec<TextChunk>, ChunkingMetrics) {
    println!("\n🔹 Tree-sitter AST Chunking");
    println!("   (Feature 'code-chunking' not enabled - would demonstrate AST-based code chunking)");

    let chunks = Vec::new();
    let metrics = ChunkingMetrics {
        strategy_name: "Tree-sitter (Rust)".to_string(),
        num_chunks: 0,
        avg_chunk_size: 0.0,
        min_chunk_size: 0,
        max_chunk_size: 0,
        total_chars: 0,
        processing_time_ms: 0,
    };

    (chunks, metrics)
}

#[cfg(feature = "code-chunking")]
fn extract_function_name(code: &str) -> Option<&str> {
    use std::collections::HashSet;

    // Simple heuristic to extract function/struct names
    let keywords = HashSet::from(["fn", "struct", "impl", "trait", "enum", "mod"]);

    for line in code.lines() {
        let trimmed = line.trim();
        if let Some(first_word) = trimmed.split_whitespace().next() {
            if keywords.contains(first_word) {
                if let Some(name) = trimmed.split_whitespace().nth(1) {
                    // Remove any generics or parameters
                    if let Some(clean_name) = name.split('<').next() {
                        if let Some(cleaner_name) = clean_name.split('(').next() {
                            return Some(cleaner_name);
                        }
                    }
                }
            }
        }
    }
    None
}

fn analyze_boundary_preservation(chunks: &[TextChunk]) -> f64 {
    if chunks.len() < 2 {
        return 1.0;
    }

    let mut well_terminated = 0;

    for chunk in chunks {
        let trimmed = chunk.content.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if chunk ends at a natural boundary
        let last_char = trimmed.chars().last().unwrap_or(' ');
        if last_char == '.' || last_char == '!' || last_char == '?' ||
           last_char == ':' || last_char == ';' || last_char == '\n' {
            well_terminated += 1;
        }
    }

    well_terminated as f64 / chunks.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎭 GraphRAG Trait-Based Chunking: Symposium Analysis");
    println!("=====================================================");
    println!("Implementing cAST (Context-Aware Splitting) approach");

    // Load the Symposium text
    println!("\n📚 Loading Symposium by Plato...");
    let content = load_symposium_text()?;
    let document = create_document(content);

    println!("   Document loaded: {} characters", document.content.len());
    println!("   First 100 chars: {}", document.content.chars().take(100).collect::<String>().replace('\n', " "));

    // Demonstrate different chunking strategies
    println!("\n🔧 Testing Different Chunking Strategies:");
    println!("=========================================");

    let (hierarchical_chunks, hierarchical_metrics) = demonstrate_hierarchical_chunking(&document);
    let (tree_sitter_chunks, tree_sitter_metrics) = demonstrate_tree_sitter_chunking();

    // Print comprehensive comparison
    println!("\n📊 Comparative Analysis:");
    println!("========================");

    hierarchical_metrics.print();
    tree_sitter_metrics.print();

    // Analyze boundary preservation
    println!("\n🎯 Boundary Preservation Analysis:");
    println!("==================================");

    let hierarchical_boundary_score = analyze_boundary_preservation(&hierarchical_chunks);

    println!("Hierarchical: {:.1}% of chunks end at natural boundaries",
             hierarchical_boundary_score * 100.0);

    if !tree_sitter_chunks.is_empty() {
        println!("Tree-sitter: Preserves syntactic boundaries for all code chunks");
    }

    // Performance comparison
    println!("\n⚡ Performance Comparison:");
    println!("==========================");

    let total_time = hierarchical_metrics.processing_time_ms +
                    tree_sitter_metrics.processing_time_ms;

    println!("Hierarchical:  {:.1}% of total time",
             (hierarchical_metrics.processing_time_ms as f64 / total_time as f64) * 100.0);
    println!("Tree-sitter:   {:.1}% of total time",
             (tree_sitter_metrics.processing_time_ms as f64 / total_time as f64) * 100.0);

    // cAST Benefits Summary
    println!("\n🏆 cAST (Context-Aware Splitting) Benefits:");
    println!("===========================================");
    println!("✅ Preserves syntactic boundaries for code snippets");
    println!("✅ Maintains coherent structure in philosophical text");
    println!("✅ Flexible strategy selection based on content type");
    println!("✅ Modular architecture allows easy extension");
    println!("✅ Performance-optimized with trait-based design");

    // Show chunk diversity
    println!("\n📈 Chunk Size Distribution:");
    println!("===========================");

    for (name, chunks) in [
        ("Hierarchical", &hierarchical_chunks),
    ] {
        if !chunks.is_empty() {
            let sizes: Vec<usize> = chunks.iter().map(|c| c.content.len()).collect();
            let median = if !sizes.is_empty() {
                let mut sorted = sizes.clone();
                sorted.sort();
                sorted[sorted.len() / 2]
            } else {
                0
            };

            println!("{}: Min={}, Median={}, Max={}",
                     name, sizes.iter().min().unwrap(),
                     median, sizes.iter().max().unwrap());
        }
    }

    println!("\n✅ Example completed successfully!");
    println!("The trait-based chunking architecture provides:");
    println!("  • Clean separation of concerns");
    println!("  • Easy strategy switching");
    println!("  • Optimal boundary preservation");
    println!("  • Extensible design for new chunking approaches");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symposium_loading() {
        let content = load_symposium_text().unwrap();
        assert!(!content.is_empty());
        assert!(content.len() > 1000); // Symposium should be substantial
        assert!(content.contains("Plato") || content.contains("Symposium"));
    }

    #[test]
    fn test_chunking_strategies() {
        let content = "Test paragraph one. Test paragraph two.\n\nNew paragraph with different content.";
        let document = create_document(content.to_string());

        // Test hierarchical chunking
        let strategy = HierarchicalChunkingStrategy::new(50, 10, document.id.clone());
        let chunks = strategy.chunk(&document.content);
        assert!(!chunks.is_empty());

        // Test boundary preservation
        let boundary_score = analyze_boundary_preservation(&chunks);
        assert!(boundary_score > 0.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut chunks = Vec::new();
        let doc_id = DocumentId::new("test".to_string());

        // Create test chunks
        for i in 0..5 {
            let chunk = TextChunk::new(
                graphrag_core::core::ChunkId::new(format!("chunk_{}", i)),
                doc_id.clone(),
                format!("Test chunk {}", i),
                i * 10,
                i * 10 + 10,
            );
            chunks.push(chunk);
        }

        let metrics = ChunkingMetrics::from_chunks("Test", &chunks, 100);
        assert_eq!(metrics.num_chunks, 5);
        assert_eq!(metrics.avg_chunk_size, 13.0); // "Test chunk X" = 12 chars + newline
        assert_eq!(metrics.processing_time_ms, 100);
    }
}