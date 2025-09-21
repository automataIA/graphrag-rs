use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graphrag_rs::{
    core::{Document, DocumentId},
    graph::GraphBuilder,
};

fn benchmark_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");

    // Different document sizes
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        let documents = generate_test_documents(size, 10);

        group.bench_with_input(
            BenchmarkId::new("build_graph", format!("{}_docs", documents.len())),
            &documents,
            |b, docs| {
                b.iter(|| {
                    let mut builder = GraphBuilder::new(500, 100, 0.7, 0.8, 5).unwrap();
                    black_box(builder.build_graph(docs.clone()).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn benchmark_text_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_processing");

    use graphrag_rs::text::TextProcessor;
    let processor = TextProcessor::new(1000, 200).unwrap();

    // Different text sizes
    let text_sizes = vec![1000, 5000, 10000, 50000];

    for size in text_sizes {
        let text = generate_test_text(size);
        let document = Document::new(
            DocumentId::new("test".to_string()),
            "Test Document".to_string(),
            text,
        );

        group.bench_with_input(
            BenchmarkId::new("chunk_text", format!("{size}_chars")),
            &document,
            |b, doc| b.iter(|| black_box(processor.chunk_text(doc).unwrap())),
        );
    }

    group.finish();
}

fn benchmark_entity_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_extraction");

    use graphrag_rs::{
        core::{ChunkId, DocumentId, TextChunk},
        entity::EntityExtractor,
    };

    let extractor = EntityExtractor::new(0.7).unwrap();

    let generated_text_500 = generate_entity_rich_text(500);
    let generated_text_1000 = generate_entity_rich_text(1000);
    let test_texts = [
        "Dr. John Smith works at Microsoft Corporation in Seattle.",
        "Alice Johnson is a professor at Stanford University. She conducts research on artificial intelligence and machine learning. The university is located in Palo Alto, California.",
        &generated_text_500,
        &generated_text_1000,
    ];

    for (i, text) in test_texts.iter().enumerate() {
        let chunk = TextChunk::new(
            ChunkId::new(format!("chunk_{i}")),
            DocumentId::new("test".to_string()),
            text.to_string(),
            0,
            text.len(),
        );

        group.bench_with_input(
            BenchmarkId::new("extract_entities", format!("text_{}_chars", text.len())),
            &chunk,
            |b, chunk| b.iter(|| black_box(extractor.extract_from_chunk(chunk).unwrap())),
        );
    }

    group.finish();
}

// Helper functions for generating test data

fn generate_test_documents(word_count: usize, num_docs: usize) -> Vec<Document> {
    let mut documents = Vec::new();

    for i in 0..num_docs {
        let content = generate_test_text(word_count);
        let document = Document::new(
            DocumentId::new(format!("doc_{i}")),
            format!("Test Document {i}"),
            content,
        );
        documents.push(document);
    }

    documents
}

fn generate_test_text(word_count: usize) -> String {
    let words = vec![
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "data",
        "science",
        "computer",
        "vision",
        "natural",
        "language",
        "processing",
        "algorithm",
        "model",
        "training",
        "dataset",
        "prediction",
        "classification",
        "regression",
        "neural",
        "network",
        "deep",
        "learning",
        "transformer",
        "attention",
    ];

    let mut text = String::new();
    for i in 0..word_count {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(words[i % words.len()]);

        // Add punctuation occasionally
        if (i + 1) % 15 == 0 {
            text.push('.');
        } else if (i + 1) % 7 == 0 {
            text.push(',');
        }
    }

    text
}

fn generate_entity_rich_text(word_count: usize) -> String {
    let people = [
        "Dr. John Smith",
        "Alice Johnson",
        "Prof. Robert Brown",
        "Ms. Sarah Wilson",
        "Mr. David Lee",
        "Dr. Emily Chen",
        "Prof. Michael Davis",
        "Jane Anderson",
    ];

    let organizations = [
        "Microsoft Corporation",
        "Google Inc",
        "Stanford University",
        "MIT",
        "OpenAI",
        "Meta Platforms",
        "Apple Inc",
        "Amazon Web Services",
    ];

    let locations = [
        "Seattle, Washington",
        "Palo Alto, California",
        "New York, NY",
        "Boston, Massachusetts",
        "San Francisco, California",
        "Austin, Texas",
        "London, England",
        "Tokyo, Japan",
    ];

    let base_words = vec![
        "research",
        "development",
        "innovation",
        "technology",
        "science",
        "collaboration",
        "project",
        "study",
        "analysis",
        "implementation",
        "algorithm",
        "system",
        "platform",
        "framework",
        "methodology",
    ];

    let mut text = String::new();
    let mut word_count_remaining = word_count;

    while word_count_remaining > 0 {
        // Randomly insert entities
        if word_count_remaining > 10 && text.len() % 100 == 0 {
            match text.len() % 3 {
                0 => {
                    text.push_str(people[text.len() % people.len()]);
                    word_count_remaining = word_count_remaining.saturating_sub(2);
                }
                1 => {
                    text.push_str(organizations[text.len() % organizations.len()]);
                    word_count_remaining = word_count_remaining.saturating_sub(2);
                }
                2 => {
                    text.push_str(locations[text.len() % locations.len()]);
                    word_count_remaining = word_count_remaining.saturating_sub(2);
                }
                _ => {}
            }
        } else {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(base_words[word_count_remaining % base_words.len()]);
            word_count_remaining -= 1;
        }

        // Add punctuation
        if text.len() % 80 == 0 {
            text.push('.');
        }
    }

    text
}

criterion_group!(
    benches,
    benchmark_graph_construction,
    benchmark_text_processing,
    benchmark_entity_extraction
);
criterion_main!(benches);
