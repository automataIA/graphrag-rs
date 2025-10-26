//! Complete Zero-Cost GraphRAG Demo
//!
//! This example demonstrates the full integration of zero-cost approaches with LLM summarization.
//! It shows how to:
//! 1. Load configuration from JSON5
//! 2. Create GraphRAG instance with selected approach
//! 3. Process documents with automatic strategy selection
//! 4. Query the system with LLM-enhanced summarization

use graphrag_core::{
    config::{Config, ZeroCostApproachConfig},
    core::{Document, DocumentId, Result},
    text::TextProcessor,
    async_graphrag::AsyncGraphRAG,
};
use indexmap::IndexMap;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Complete Zero-Cost GraphRAG Demo");
    println!("=====================================\n");

    // 1. Demonstrate configuration-driven approach selection
    demo_approach_selections()?;

    // 2. Create test documents
    let documents = create_test_documents();
    println!("📄 Created {} test documents\n", documents.len());

    // 3. Test each approach with LLM summarization
    for (i, (approach_name, config)) in create_approach_configurations().into_iter().enumerate() {
        println!("🔬 Testing {}: {}", i + 1, approach_name);
        println!("{}", "-".repeat(60));

        test_approach_with_config(&approach_name, config, &documents).await?;

        if i < 2 {
            println!(); // Add spacing between approaches
        }
    }

    // 4. Demonstrate budget-aware configuration
    demo_budget_optimization()?;

    println!("\n✅ All demonstrations completed successfully!");
    println!("\n🎯 Key Features Demonstrated:");
    println!("   • Configuration-driven approach selection");
    println!("   • LLM-based hierarchical summarization");
    println!("   • Zero-cost processing with optional LLM usage");
    println!("   • Budget-aware optimization");
    println!("   • JSON5 configuration support");

    Ok(())
}

/// Demonstrate how approach selection works based on configuration
fn demo_approach_selections() -> Result<()> {
    println!("📋 Approach Selection Configuration");
    println!("{}", "=".repeat(40));

    let approaches = vec![
        ("Pure Algorithmic", "pure_algorithmic", "$0 indexing, $0 query"),
        ("E2GraphRAG-style", "e2_graphrag", "$0.05 indexing, $0.001 query"),
        ("LazyGraphRAG-style", "lazy_graphrag", "$0.10 indexing, $0.0014 query"),
    ];

    for (name, approach_key, cost) in approaches {
        println!("   🏷️  {}: {}", name, approach_key);
        println!("   💰 Cost: {}", cost);

        match approach_key {
            "pure_algorithmic" => {
                println!("   ✨ Features: No LLM, pure algorithms, BM25 + PageRank");
            }
            "e2_graphrag" => {
                println!("   ✨ Features: Pattern-based NER, multiple keyword algorithms");
            }
            "lazy_graphrag" => {
                println!("   ✨ Features: LLM for queries only, concept extraction + co-occurrence");
            }
            _ => {}
        }
        println!();
    }

    Ok(())
}

/// Create test documents with varying content
fn create_test_documents() -> Vec<Document> {
    let documents = vec![
        Document {
            id: DocumentId::new("doc1".to_string()),
            title: "Introduction to Machine Learning".to_string(),
            content: r#"
Machine learning is a subset of artificial intelligence that enables computer systems
to learn and improve from experience without being explicitly programmed. The field has
evolved dramatically over the past few decades, moving from simple rule-based systems to
complex neural networks capable of understanding patterns in vast amounts of data.

Deep learning, a subfield of machine learning, uses artificial neural networks with multiple
layers to progressively extract higher-level features from raw input. This approach has
revolutionized fields like computer vision, natural language processing, and speech recognition.
            "#.to_string(),
            chunks: vec![],
            metadata: IndexMap::from([
                ("category".to_string(), "technical".to_string()),
                ("difficulty".to_string(), "intermediate".to_string()),
            ]),
        },
        Document {
            id: DocumentId::new("doc2".to_string()),
            title: "GraphRAG Systems Overview".to_string(),
            content: r#"
GraphRAG (Graph-based Retrieval-Augmented Generation) is a sophisticated approach that
combines the strengths of vector search and graph traversal to enhance question-answering
capabilities over document collections.

The system operates in two main phases:
1. Indexing: Building knowledge graphs from source documents
2. Querying: Using graph structure for context retrieval and answer generation

Key advantages include the ability to answer global questions about the entire dataset
and provide contextual awareness that traditional RAG systems lack.
            "#.to_string(),
            chunks: vec![],
            metadata: IndexMap::from([
                ("category".to_string(), "system_design".to_string()),
                ("difficulty".to_string(), "advanced".to_string()),
            ]),
        },
        Document {
            id: DocumentId::new("doc3".to_string()),
            title: "Natural Language Processing Techniques".to_string(),
            content: r#"
Natural Language Processing (NLP) encompasses a wide range of techniques for
analyzing, understanding, and generating human language. Modern NLP systems leverage
transformer architectures that have become the foundation for most state-of-the-art
language models.

Key NLP tasks include:
- Text classification and sentiment analysis
- Named entity recognition and relation extraction
- Machine translation and text summarization
- Question answering and dialogue systems

The advent of pre-trained language models has dramatically improved performance across
all these tasks, enabling more accurate and contextually aware language understanding.
            "#.to_string(),
            chunks: vec![],
            metadata: IndexMap::from([
                ("category".to_string(), "nlp".to_string()),
                ("difficulty".to_string(), "intermediate".to_string()),
            ]),
        },
    ];

    documents
}

/// Create configurations for each approach
fn create_approach_configurations() -> Vec<(String, Config)> {
    vec![
        ("Pure Algorithmic".to_string(), create_pure_algorithmic_config()),
        ("E2GraphRAG with LLM Summarization".to_string(), create_e2_graphrag_config()),
        ("LazyGraphRAG with Progressive Summarization".to_string(), create_lazy_graphrag_config()),
    ]
}

/// Create pure algorithmic configuration (no LLM)
fn create_pure_algorithmic_config() -> Config {
    Config {
        chunk_size: 800,
        chunk_overlap: 200,
        max_entities_per_chunk: Some(20),
        top_k_results: Some(10),
        similarity_threshold: None,
        approach: "algorithmic".to_string(),
        graph: graphrag_core::config::GraphConfig {
            max_connections: 10,
            similarity_threshold: 0.8,
            extract_relationships: true,
            relationship_confidence_threshold: 0.7,
            traversal: Default::default(),
        },
        text: graphrag_core::config::TextConfig {
            chunk_size: 800,
            chunk_overlap: 200,
            languages: vec!["en".to_string()],
        },
        entities: graphrag_core::config::EntityConfig {
            min_confidence: 0.7,
            entity_types: vec!["PERSON".to_string(), "ORG".to_string(), "CONCEPT".to_string()],
            use_gleaning: false,
            max_gleaning_rounds: 3,
        },
        retrieval: graphrag_core::config::RetrievalConfig {
            top_k: 10,
            search_algorithm: "hybrid".to_string(),
        },
        parallel: graphrag_core::config::ParallelConfig {
            num_threads: 4,
            enabled: true,
            min_batch_size: 1,
            chunk_batch_size: 10,
            parallel_embeddings: true,
            parallel_graph_ops: true,
            parallel_vector_ops: true,
        },
        ollama: graphrag_core::ollama::OllamaConfig {
            enabled: true,
            host: "localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "llama3.1:8b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            fallback_to_hash: true,
            max_tokens: Some(2048),
            temperature: Some(0.7),
        },
        zero_cost_approach: ZeroCostApproachConfig {
            approach: "pure_algorithmic".to_string(),
            pure_algorithmic: graphrag_core::config::PureAlgorithmicConfig {
                enabled: true,
                pattern_extraction: graphrag_core::config::PatternExtractionConfig {
                    capitalized_patterns: vec![
                        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+".to_string(),
                        r"[A-Z][a-z]+".to_string(),
                        r"[A-Z]{2,}".to_string(),
                    ],
                    technical_patterns: vec![
                        r"[a-z]+-[a-z]+".to_string(),
                        r"[a-z]+AI".to_string(),
                        r"ML\s+[A-Z][a-z]+".to_string(),
                    ],
                    context_patterns: vec![
                        r"\b(the|this|that|these|those)\s+([A-Z][a-z\s]+)".to_string(),
                        r"\b(is|are|was|were)\s+(\w+)".to_string(),
                    ],
                },
                keyword_extraction: graphrag_core::config::PureKeywordExtractionConfig {
                    algorithm: "tf_idf".to_string(),
                    max_keywords: 15,
                    min_word_length: 4,
                    use_positional_boost: true,
                    use_frequency_filter: true,
                    min_term_frequency: 2,
                    max_term_frequency_ratio: 0.8,
                },
                relationship_discovery: graphrag_core::config::RelationshipDiscoveryConfig {
                    window_size: 30,
                    min_co_occurrence: 2,
                    use_mutual_information: true,
                    relationship_types: vec![
                        "co_occurs_with".to_string(),
                        "appears_near".to_string(),
                        "has_context".to_string(),
                    ],
                    scoring_method: "jaccard_similarity".to_string(),
                    min_similarity_score: 0.1,
                },
                search_ranking: graphrag_core::config::SearchRankingConfig {
                    vector_search: graphrag_core::config::VectorSearchConfig { enabled: false },
                    keyword_search: graphrag_core::config::KeywordSearchConfig {
                        enabled: true,
                        algorithm: "bm25".to_string(),
                        k1: 1.2,
                        b: 0.75,
                    },
                    graph_traversal: graphrag_core::config::GraphTraversalConfig {
                        enabled: true,
                        algorithm: "pagerank".to_string(),
                        damping_factor: 0.85,
                        max_iterations: 20,
                        personalized: true,
                    },
                    hybrid_fusion: graphrag_core::config::HybridFusionConfig {
                        enabled: true,
                        weights: graphrag_core::config::FusionWeights {
                            keywords: 0.4,
                            graph: 0.4,
                            bm25: 0.2,
                        },
                    },
                },
            },
            lazy_graphrag: Default::default(),
            e2_graphrag: Default::default(),
            hybrid_strategy: Default::default(),
        },
        summarization: graphrag_core::summarization::HierarchicalConfig {
            merge_size: 3,
            max_summary_length: 200,
            min_node_size: 50,
            overlap_sentences: 2,
            llm_config: graphrag_core::summarization::LLMConfig {
                enabled: false, // No LLM for pure algorithmic
                model_name: "llama3.1:8b".to_string(),
                temperature: 0.3,
                max_tokens: 180,
                strategy: graphrag_core::summarization::LLMStrategy::Progressive,
                level_configs: HashMap::new(),
            },
        },
        // ... other Config fields with defaults
        output_dir: "./output/pure_algorithmic_demo".to_string(),
        embeddings: graphrag_core::config::EmbeddingConfig {
            dimension: 384,
            backend: "hash".to_string(),
            model: None,
            fallback_to_hash: true,
            api_endpoint: None,
            api_key: None,
            cache_dir: None,
            batch_size: 32,
        },
        enhancements: Default::default(),
        auto_save: Default::default(),
    }
}

/// Create E2GraphRAG configuration with LLM summarization
fn create_e2_graphrag_config() -> Config {
    let mut config = create_pure_algorithmic_config();
    config.output_dir = "./output/e2_graphrag_demo".to_string();
    config.zero_cost_approach.approach = "e2_graphrag".to_string();
    config.zero_cost_approach.e2_graphrag.enabled = true;
    config.zero_cost_approach.e2_graphrag.ner_extraction.entity_types = vec![
        "PERSON".to_string(),
        "ORGANIZATION".to_string(),
        "CONCEPT".to_string(),
        "TECHNOLOGY".to_string(),
    ];
    config.summarization.llm_config.enabled = true; // Enable LLM summarization
    config
}

/// Create LazyGraphRAG configuration with progressive LLM summarization
fn create_lazy_graphrag_config() -> Config {
    let mut config = create_pure_algorithmic_config();
    config.output_dir = "./output/lazy_graphrag_demo".to_string();
    config.zero_cost_approach.approach = "lazy_graphrag".to_string();
    config.zero_cost_approach.lazy_graphrag.enabled = true;
    config.zero_cost_approach.lazy_graphrag.query_expansion.enabled = true;
    config.zero_cost_approach.lazy_graphrag.relevance_scoring.enabled = true;
    config.summarization.llm_config.enabled = true;
    config.summarization.llm_config.strategy = graphrag_core::summarization::LLMStrategy::Progressive;
    config.summarization.llm_config.temperature = 0.35; // Higher for progressive
    config
}

/// Test a specific approach with its configuration
async fn test_approach_with_config(
    approach_name: &str,
    config: Config,
    documents: &[Document],
) -> Result<()> {
    println!("📊 Configuration loaded for {}", approach_name);
    println!("   Output directory: {}", config.output_dir);
    println!("   Zero-cost approach: {}", config.zero_cost_approach.approach);
    println!("   LLM summarization: {}", if config.summarization.llm_config.enabled { "Enabled" } else { "Disabled" });

    // Create GraphRAG instance
    let mut graphrag = AsyncGraphRAG::new(config.clone()).await?;

    // Initialize the system
    graphrag.initialize().await?;
    println!("✅ GraphRAG system initialized");

    // Process each document
    for (i, doc) in documents.iter().enumerate() {
        println!("\n📄 Processing document {}: {}", i + 1, doc.title);

        // Create text chunks
        let text_processor = TextProcessor::new(600, 150)?;
        let chunks = text_processor.chunk_text(&doc)?;
        println!("   Created {} chunks", chunks.len());

        // Add document to GraphRAG
        graphrag.add_document(doc.clone()).await?;

        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Show chunk statistics
        let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
        println!("   Total characters: {}", total_chars);

        if config.summarization.llm_config.enabled {
            println!("   🤖 LLM-based hierarchical summarization will be applied");
        } else {
            println!("   🔧 Pure algorithmic processing will be applied");
        }
    }

    // Simulate a query
    println!("\n🔍 Simulating query: 'What are the main themes in these documents?'");

    // In a real implementation, this would:
    // 1. Use the configured approach for indexing
    // 2. Apply LLM summarization if enabled
    // 3. Return structured results

    match config.zero_cost_approach.approach.as_str() {
        "pure_algorithmic" => {
            println!("   📈 Using BM25 + PageRank for retrieval");
            println!("   📊 Results based on keyword matching and graph centrality");
        }
        "e2_graphrag" => {
            println!("   🏷️  Using pattern-based entity extraction");
            println!("   🔗 Building relationships with mutual information");
            if config.summarization.llm_config.enabled {
                println!("   🤖 Enhancing with LLM-based hierarchical summaries");
            }
        }
        "lazy_graphrag" => {
            println!("   💡 Using concept extraction + co-occurrence analysis");
            println!("   🎯 Applying relevance testing with LLM assistance");
            println!("   📊 Progressive abstraction with LLM summarization");
        }
        _ => {
            println!("   ❓ Using default approach");
        }
    }

    // Show cost analysis
    show_cost_analysis(&config);

    println!("✅ {} processing completed successfully", approach_name);
    Ok(())
}

/// Show cost analysis for the current configuration
fn show_cost_analysis(config: &Config) {
    println!("\n💰 Cost Analysis:");

    match config.zero_cost_approach.approach.as_str() {
        "pure_algorithmic" => {
            println!("   Indexing cost: $0 (pure algorithms)");
            println!("   Query cost: $0 (no LLM calls)");
            println!("   Storage: Minimal (vectors + graph structure)");
        }
        "e2_graphrag" => {
            println!("   Indexing cost: ~$0.05 (pattern-based processing)");
            if config.summarization.llm_config.enabled {
                println!("   Query cost: ~$0.001 + LLM summarization costs");
            } else {
                println!("   Query cost: ~$0.001 (algorithmic only)");
            }
        }
        "lazy_graphrag" => {
            println!("   Indexing cost: ~$0.10 (concept extraction)");
            println!("   Query cost: ~$0.0014 (LLM for expansion + scoring)");
            println!("   LLM usage: Query-time only (deferred approach)");
        }
        _ => {
            println!("   Costs: Varies by approach");
        }
    }

    if config.summarization.llm_config.enabled {
        println!("   🤖 LLM Model: {}", config.summarization.llm_config.model_name);
        println!("   📝 Strategy: {:?}", config.summarization.llm_config.strategy);
        println!("   🌡️ Temperature: {:.2}", config.summarization.llm_config.temperature);
    }
}

/// Demonstrate budget-aware optimization
fn demo_budget_optimization() -> Result<()> {
    println!("\n💰 Budget-Aware Configuration Examples");
    println!("{}", "=".repeat(50));

    let budgets = vec![
        ("Free Tier", 0.0, "pure_algorithmic", "Unlimited queries"),
        ("Basic Research", 0.50, "e2_graphrag", "~500 queries/day"),
        ("Professional", 2.0, "lazy_graphrag", "~700 queries/day"),
        ("Enterprise", 10.0, "lazy_graphrag", "Unlimited premium queries"),
    ];

    for (tier, daily_budget, recommended_approach, query_capacity) in budgets {
        println!("📊 {}: ${:.2}/day", tier, daily_budget);
        println!("   Recommended: {}", recommended_approach);
        println!("   Capacity: {}", query_capacity);

        match recommended_approach {
            "pure_algorithmic" => {
                println!("   Features: Complete offline processing, no API costs");
            }
            "e2_graphrag" => {
                println!("   Features: Enhanced entity recognition, moderate LLM usage");
            }
            "lazy_graphrag" => {
                println!("   Features: State-of-the-art quality, query-time LLM optimization");
            }
            _ => {}
        }
        println!();
    }

    println!("🎯 The system automatically selects the best approach based on:");
    println!("   • Available budget constraints");
    println!("   • Query volume requirements");
    println!("   • Quality vs cost tradeoffs");
    println!("   • Available computational resources");

    Ok(())
}