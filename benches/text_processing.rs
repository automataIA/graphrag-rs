use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use graphrag_rs::{
    core::{Document, DocumentId},
    text::{LanguageDetector, TextProcessor},
};

fn benchmark_text_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_chunking");

    let chunk_sizes = vec![500, 1000, 2000];
    let overlaps = vec![50, 100, 200];
    let text_sizes = vec![1000, 5000, 10000, 50000];

    for chunk_size in chunk_sizes {
        for overlap in &overlaps {
            if *overlap < chunk_size / 2 {
                let processor = TextProcessor::new(chunk_size, *overlap).unwrap();

                for text_size in &text_sizes {
                    let document = create_test_document(*text_size);

                    group.bench_with_input(
                        BenchmarkId::new(
                            "chunk_text",
                            format!("chunk_{chunk_size}_overlap_{overlap}_text_{text_size}"),
                        ),
                        &(&processor, &document),
                        |b, (processor, document)| {
                            b.iter(|| black_box(processor.chunk_text(document).unwrap()))
                        },
                    );
                }
            }
        }
    }

    group.finish();
}

fn benchmark_text_cleaning(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_cleaning");

    let processor = TextProcessor::new(1000, 100).unwrap();
    let messy_text_100 = create_messy_text(100);
    let messy_text_1000 = create_messy_text(1000);
    let messy_text_10000 = create_messy_text(10000);
    let test_texts = [
        "  Simple   text   with   extra   spaces  ",
        &messy_text_100,
        &messy_text_1000,
        &messy_text_10000,
    ];

    for text in test_texts.iter() {
        group.bench_with_input(
            BenchmarkId::new("clean_text", format!("text_{}_chars", text.len())),
            &(&processor, text),
            |b, (processor, text)| b.iter(|| black_box(processor.clean_text(text))),
        );
    }

    group.finish();
}

fn benchmark_keyword_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyword_extraction");

    let processor = TextProcessor::new(1000, 100).unwrap();
    let keyword_counts = vec![5, 10, 20, 50];
    let text_sizes = vec![100, 500, 1000, 5000];

    for keyword_count in keyword_counts {
        for text_size in &text_sizes {
            let text = create_keyword_rich_text(*text_size);

            group.bench_with_input(
                BenchmarkId::new(
                    "extract_keywords",
                    format!("top_{keyword_count}_from_{text_size}_words"),
                ),
                &(&processor, &text, keyword_count),
                |b, (processor, text, count)| {
                    b.iter(|| black_box(processor.extract_keywords(text, *count)))
                },
            );
        }
    }

    group.finish();
}

fn benchmark_sentence_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_extraction");

    let processor = TextProcessor::new(1000, 100).unwrap();
    let text_sizes = vec![100, 500, 1000, 5000, 10000];

    for text_size in text_sizes {
        let text = create_sentence_rich_text(text_size);

        group.bench_with_input(
            BenchmarkId::new("extract_sentences", format!("{text_size}_chars")),
            &(&processor, &text),
            |b, (processor, text)| b.iter(|| black_box(processor.extract_sentences(text))),
        );
    }

    group.finish();
}

fn benchmark_language_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("language_detection");

    let multilingual_100 = create_multilingual_text(100);
    let multilingual_500 = create_multilingual_text(500);
    let multilingual_1000 = create_multilingual_text(1000);
    let test_texts = vec![
        ("English text with common words and phrases", "en"),
        ("Este es un texto en español con palabras específicas", "es"),
        (
            "Este texto está em português com características únicas",
            "pt",
        ),
        (
            "Ce texte est en français avec des caractères spéciaux",
            "fr",
        ),
        (multilingual_100.as_str(), "mixed"),
        (multilingual_500.as_str(), "mixed"),
        (multilingual_1000.as_str(), "mixed"),
    ];

    for (text, lang) in test_texts {
        group.bench_with_input(
            BenchmarkId::new("detect_language", format!("{}_{}_chars", lang, text.len())),
            &text,
            |b, text| b.iter(|| black_box(LanguageDetector::detect_language(text))),
        );
    }

    group.finish();
}

fn benchmark_word_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("word_counting");

    let processor = TextProcessor::new(1000, 100).unwrap();
    let text_sizes = vec![100, 500, 1000, 5000, 10000, 50000];

    for text_size in text_sizes {
        let text = create_test_text(text_size);

        group.bench_with_input(
            BenchmarkId::new("word_count", format!("{text_size}_chars")),
            &(&processor, &text),
            |b, (processor, text)| b.iter(|| black_box(processor.word_count(text))),
        );
    }

    group.finish();
}

// Helper functions for generating test data

fn create_test_document(word_count: usize) -> Document {
    let content = create_test_text(word_count);
    Document::new(
        DocumentId::new("test_doc".to_string()),
        "Test Document".to_string(),
        content,
    )
}

fn create_test_text(word_count: usize) -> String {
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
        "computer",
        "science",
        "data",
        "analysis",
        "neural",
        "networks",
        "deep",
        "learning",
        "natural",
        "language",
        "processing",
        "algorithm",
        "model",
        "training",
        "prediction",
        "classification",
        "regression",
        "clustering",
        "optimization",
        "research",
        "development",
        "innovation",
        "technology",
        "system",
    ];

    let mut text = String::new();
    for i in 0..word_count {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(words[i % words.len()]);

        // Add sentence endings
        if (i + 1) % 12 == 0 {
            text.push('.');
        } else if (i + 1) % 8 == 0 {
            text.push(',');
        }
    }

    text
}

fn create_messy_text(word_count: usize) -> String {
    let words = [
        "word",
        "with",
        "extra",
        "spaces",
        "and",
        "punctuation",
        "mixed",
        "content",
        "various",
        "symbols",
        "numbers",
        "123",
    ];

    let mut text = String::new();
    for i in 0..word_count {
        // Add random amounts of whitespace
        for _ in 0..(i % 3 + 1) {
            text.push(' ');
        }

        text.push_str(words[i % words.len()]);

        // Add random punctuation
        match i % 7 {
            0 => text.push('.'),
            1 => text.push(','),
            2 => text.push('!'),
            3 => text.push('?'),
            4 => text.push(';'),
            _ => {}
        }
    }

    text
}

fn create_keyword_rich_text(word_count: usize) -> String {
    let keywords = vec![
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "data",
        "science",
        "neural",
        "network",
        "deep",
        "learning",
        "algorithm",
        "model",
        "training",
        "prediction",
        "classification",
        "regression",
        "clustering",
        "optimization",
        "feature",
        "extraction",
        "dimensionality",
        "reduction",
    ];

    let filler_words = vec![
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "about", "into", "through", "during", "before", "after", "above", "below", "up", "down",
        "out", "off", "over", "under",
    ];

    let mut text = String::new();
    for i in 0..word_count {
        if i > 0 {
            text.push(' ');
        }

        // Mix keywords with filler words
        if i % 3 == 0 {
            text.push_str(keywords[i % keywords.len()]);
        } else {
            text.push_str(filler_words[i % filler_words.len()]);
        }

        // Add punctuation
        if (i + 1) % 15 == 0 {
            text.push('.');
        } else if (i + 1) % 7 == 0 {
            text.push(',');
        }
    }

    text
}

fn create_sentence_rich_text(char_count: usize) -> String {
    let sentences = [
        "This is a simple sentence.",
        "Machine learning is a subset of artificial intelligence!",
        "Data science involves statistics, programming, and domain expertise?",
        "Neural networks are inspired by biological neural systems.",
        "Deep learning uses multiple layers to model complex patterns;",
        "Natural language processing enables computers to understand human language:",
        "Algorithms are step-by-step procedures for solving problems...",
        "Model training requires large amounts of data and computational resources!",
    ];

    let mut text = String::new();
    let mut current_length = 0;

    while current_length < char_count {
        if !text.is_empty() {
            text.push(' ');
            current_length += 1;
        }

        let sentence = sentences[current_length % sentences.len()];
        text.push_str(sentence);
        current_length += sentence.len();
    }

    // Truncate to desired length
    if text.len() > char_count {
        text.truncate(char_count);
    }

    text
}

fn create_multilingual_text(word_count: usize) -> String {
    let english_words = ["machine", "learning", "data", "science", "algorithm"];
    let spanish_words = ["máquina", "aprendizaje", "datos", "ciencia", "algoritmo"];
    let french_words = [
        "machine",
        "apprentissage",
        "données",
        "science",
        "algorithme",
    ];
    let portuguese_words = ["máquina", "aprendizado", "dados", "ciência", "algoritmo"];

    let mut text = String::new();
    for i in 0..word_count {
        if i > 0 {
            text.push(' ');
        }

        match i % 4 {
            0 => text.push_str(english_words[i % english_words.len()]),
            1 => text.push_str(spanish_words[i % spanish_words.len()]),
            2 => text.push_str(french_words[i % french_words.len()]),
            3 => text.push_str(portuguese_words[i % portuguese_words.len()]),
            _ => unreachable!(),
        }

        if (i + 1) % 10 == 0 {
            text.push('.');
        }
    }

    text
}

criterion_group!(
    benches,
    benchmark_text_chunking,
    benchmark_text_cleaning,
    benchmark_keyword_extraction,
    benchmark_sentence_extraction,
    benchmark_language_detection,
    benchmark_word_counting
);
criterion_main!(benches);
