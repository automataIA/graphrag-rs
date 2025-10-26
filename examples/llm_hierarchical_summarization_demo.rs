//! LLM-based Hierarchical Summarization Demo
//!
//! This example demonstrates how to use the new LLM-powered hierarchical summarization
//! system to create multi-level abstractions of document content.
//!
//! Features demonstrated:
//! - LLM-based vs extractive summarization
//! - Progressive abstraction across tree levels
//! - Level-specific configuration
//! - Batch processing with LLM
//! - Fallback to extractive when LLM fails

use graphrag_core::{
    core::{DocumentId, TextChunk},
    summarization::{
        DocumentTree, HierarchicalConfig, LLMConfig, LLMStrategy, LevelConfig, LLMClient
    },
    text::TextProcessor,
    Result,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Example LLM client implementation
/// In a real implementation, this would connect to Ollama, OpenAI, or another LLM provider
struct ExampleLLMClient {
    model_name: String,
}

#[async_trait::async_trait]
impl LLMClient for ExampleLLMClient {
    async fn generate_summary(
        &self,
        text: &str,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<String> {
        // Simulate LLM call with a simple extractive approach
        println!("🤖 LLM Call: {} (max_tokens: {})", self.model_name, max_tokens);
        println!("   Prompt: {}", prompt.chars().take(100).collect::<String>());

        // Simple mock implementation - in reality this would call the actual LLM
        let sentences: Vec<&str> = text
            .split('.')
            .filter(|s| s.trim().len() > 10)
            .collect();

        let mut summary = String::new();
        let mut char_count = 0;

        for sentence in sentences {
            let sentence = sentence.trim();
            if char_count + sentence.len() + 1 <= max_tokens * 4 { // Rough char estimation
                if !summary.is_empty() {
                    summary.push(' ');
                }
                summary.push_str(sentence);
                char_count += sentence.len() + 1;
            } else {
                break;
            }
        }

        if summary.is_empty() && !sentences.is_empty() {
            summary = sentences[0].trim().to_string();
        }

        Ok(summary)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 LLM-based Hierarchical Summarization Demo\n");

    // Example document content
    let document_text = r#"
    Machine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed.
    The field has evolved dramatically over the past few decades, moving from simple rule-based systems to complex neural networks capable of understanding patterns in vast amounts of data.

    Deep learning, a subfield of machine learning, uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. This approach has revolutionized fields like computer vision, natural language processing, and speech recognition.

    Modern machine learning models, particularly large language models, have demonstrated remarkable capabilities in understanding and generating human-like text. These models are trained on enormous datasets and can perform tasks ranging from translation to creative writing.

    The future of machine learning includes developments in areas like reinforcement learning, few-shot learning, and the integration of machine learning systems into everyday applications and decision-making processes.
    "#;

    println!("📄 Document length: {} characters\n", document_text.len());

    // Demonstrate different summarization strategies
    demo_extractive_vs_llm(document_text).await?;
    demo_progressive_summarization(document_text).await?;
    demo_level_specific_configuration(document_text).await?;
    demo_batch_processing(document_text).await?;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

/// Demonstrate the difference between extractive and LLM-based summarization
async fn demo_extractive_vs_llm(document_text: &str) -> Result<()> {
    println!("📊 Demo 1: Extractive vs LLM-based Summarization");
    println!("=" .repeat(50));

    // Create chunks
    let mut text_processor = TextProcessor::new(800, 200)?;
    let chunks = text_processor.chunk_text(document_text)?;
    println!("Created {} chunks\n", chunks.len());

    // Configuration 1: Extractive only (LLM disabled)
    let extractive_config = HierarchicalConfig {
        merge_size: 3,
        max_summary_length: 200,
        min_node_size: 50,
        overlap_sentences: 2,
        llm_config: LLMConfig {
            enabled: false,
            model_name: "extractive".to_string(),
            temperature: 0.0,
            max_tokens: 100,
            strategy: LLMStrategy::Uniform,
            level_configs: HashMap::new(),
        },
    };

    // Configuration 2: LLM-based
    let llm_client = Arc::new(ExampleLLMClient {
        model_name: "llama3.1:8b".to_string(),
    });

    let llm_config = HierarchicalConfig {
        merge_size: 3,
        max_summary_length: 200,
        min_node_size: 50,
        overlap_sentences: 2,
        llm_config: LLMConfig {
            enabled: true,
            model_name: "llama3.1:8b".to_string(),
            temperature: 0.3,
            max_tokens: 150,
            strategy: LLMStrategy::Progressive,
            level_configs: HashMap::new(),
        },
    };

    // Build trees with both approaches
    let mut extractive_tree = DocumentTree::new(
        DocumentId::new("extractive_demo".to_string()),
        extractive_config,
    )?;
    extractive_tree.build_from_chunks(chunks.clone())?;

    let mut llm_tree = DocumentTree::with_llm_client(
        DocumentId::new("llm_demo".to_string()),
        llm_config,
        llm_client,
    )?;
    llm_tree.build_from_chunks(chunks).await?;

    // Compare results
    println!("📈 Extractive Tree Statistics:");
    let extractive_stats = extractive_tree.get_statistics();
    extractive_stats.print();

    println!("\n🤖 LLM Tree Statistics:");
    let llm_stats = llm_tree.get_statistics();
    llm_stats.print();

    // Show sample summaries
    println!("\n📝 Sample Summary Comparison:");
    if let (Some(extractive_root), Some(llm_root)) = (
        extractive_tree.get_root_nodes().first(),
        llm_tree.get_root_nodes().first()
    ) {
        println!("Extractive Root (Level {}): {}", extractive_root.level,
                   extractive_root.summary.chars().take(100).collect::<String>());
        println!("LLM Root (Level {}): {}", llm_root.level,
                   llm_root.summary.chars().take(100).collect::<String>());
    }

    println!("\n");
    Ok(())
}

/// Demonstrate progressive summarization across levels
async fn demo_progressive_summarization(document_text: &str) -> Result<()> {
    println!("🔄 Demo 2: Progressive Abstraction Across Levels");
    println!("=" .repeat(50));

    let llm_client = Arc::new(ExampleLLMClient {
        model_name: "llama3.1:8b".to_string(),
    });

    // Progressive strategy configuration
    let progressive_config = HierarchicalConfig {
        merge_size: 3,
        max_summary_length: 250,
        min_node_size: 80,
        overlap_sentences: 2,
        llm_config: LLMConfig {
            enabled: true,
            model_name: "llama3.1:8b".to_string(),
            temperature: 0.3,
            max_tokens: 180,
            strategy: LLMStrategy::Progressive,
            level_configs: {
                let mut level_configs = HashMap::new();

                // Level 0: Extractive
                level_configs.insert(0, LevelConfig {
                    max_length: 180,
                    use_abstractive: false,
                    prompt_template: None,
                    temperature: Some(0.2),
                });

                // Level 1: Still extractive
                level_configs.insert(1, LevelConfig {
                    max_length: 200,
                    use_abstractive: false,
                    prompt_template: Some(
                        "Extract the key information from this text segment. Keep it factual and under {max_length} characters.\n\n{text}".to_string()
                    ),
                    temperature: Some(0.25),
                });

                // Level 2: Begin abstractive
                level_configs.insert(2, LevelConfig {
                    max_length: 220,
                    use_abstractive: true,
                    prompt_template: Some(
                        "Create a summary that synthesizes the key concepts from these related segments. Focus on main themes.\n\n{text}".to_string()
                    ),
                    temperature: Some(0.3),
                });

                // Level 3+: Fully abstractive
                level_configs.insert(3, LevelConfig {
                    max_length: 250,
                    use_abstractive: true,
                    prompt_template: Some(
                        "Generate a high-level abstract summary of this content. Focus on essential themes and insights.\n\n{text}".to_string()
                    ),
                    temperature: Some(0.35),
                });

                level_configs
            },
        },
    };

    let mut text_processor = TextProcessor::new(600, 150)?;
    let chunks = text_processor.chunk_text(document_text)?;

    let mut progressive_tree = DocumentTree::with_llm_client(
        DocumentId::new("progressive_demo".to_string()),
        progressive_config,
        llm_client,
    )?;
    progressive_tree.build_from_chunks(chunks).await?;

    // Show progression across levels
    println!("📊 Progressive Summarization Results:");
    let stats = progressive_tree.get_statistics();

    for level in 0..=stats.max_level {
        if let Some(nodes) = progressive_tree.get_level_nodes(level) {
            if let Some(node) = nodes.first() {
                println!("Level {}: {} ({} chars)",
                          level,
                          node.summary.chars().take(80).collect::<String>(),
                          node.summary.len());
            }
        }
    }

    println!("\n");
    Ok(())
}

/// Demonstrate level-specific configuration
async fn demo_level_specific_configuration(document_text: &str) -> Result<()> {
    println!("⚙️  Demo 3: Level-Specific Configuration");
    println!("=" .repeat(50));

    let llm_client = Arc::new(ExampleLLMClient {
        model_name: "llama3.1:8b".to_string(),
    });

    // Adaptive strategy with custom level configurations
    let adaptive_config = HierarchicalConfig {
        merge_size: 4,
        max_summary_length: 300,
        min_node_size: 60,
        overlap_sentences: 2,
        llm_config: LLMConfig {
            enabled: true,
            model_name: "llama3.1:8b".to_string(),
            temperature: 0.3,
            max_tokens: 200,
            strategy: LLMStrategy::Adaptive,
            level_configs: {
                let mut configs = HashMap::new();

                // Custom templates for different purposes
                configs.insert(0, LevelConfig {
                    max_length: 150,
                    use_abstractive: false,
                    prompt_template: Some(
                        "🔍 FACT EXTRACTION\nExtract the most important facts from this text segment:\n\n{text}\n\nKey facts:".to_string()
                    ),
                    temperature: Some(0.1),
                });

                configs.insert(1, LevelConfig {
                    max_length: 200,
                    use_abstractive: false,
                    prompt_template: Some(
                        "📋 SEGMENT SUMMARY\nCreate a concise summary of this document segment:\n\n{text}\n\nSummary:".to_string()
                    ),
                    temperature: Some(0.2),
                });

                configs.insert(2, LevelConfig {
                    max_length: 250,
                    use_abstractive: true,
                    prompt_template: Some(
                        "🎯 THEME SYNTHESIS\nIdentify and synthesize the main themes in this content:\n\n{text}\n\nMain themes:".to_string()
                    ),
                    temperature: Some(0.3),
                });

                configs.insert(3, LevelConfig {
                    max_length: 300,
                    use_abstractive: true,
                    prompt_template: Some(
                        "🌟 INSIGHTS OVERVIEW\nGenerate high-level insights and abstract understanding:\n\n{text}\n\nKey insights:".to_string()
                    ),
                    temperature: Some(0.4),
                });

                configs
            },
        },
    };

    let mut text_processor = TextProcessor::new(500, 100)?;
    let chunks = text_processor.chunk_text(document_text)?;

    let mut adaptive_tree = DocumentTree::with_llm_client(
        DocumentId::new("adaptive_demo".to_string()),
        adaptive_config,
        llm_client,
    )?;
    adaptive_tree.build_from_chunks(chunks).await?;

    println!("🎨 Adaptive Summarization with Custom Templates:");
    let stats = adaptive_tree.get_statistics();

    for level in 0..=stats.max_level {
        if let Some(nodes) = adaptive_tree.get_level_nodes(level) {
            if let Some(node) = nodes.first() {
                println!("Level {}: {} chars", level, node.summary.len());
                if node.summary.len() < 100 {
                    println!("  \"{}\"", node.summary);
                } else {
                    println!("  \"{}...\"", node.summary.chars().take(97).collect::<String>());
                }
            }
        }
    }

    println!("\n");
    Ok(())
}

/// Demonstrate batch processing capabilities
async fn demo_batch_processing(document_text: &str) -> Result<()> {
    println!("⚡ Demo 4: Batch Processing with LLM");
    println!("=" .repeat(50));

    let llm_client = Arc::new(ExampleLLMClient {
        model_name: "llama3.1:8b".to_string(),
    });

    let batch_config = HierarchicalConfig {
        merge_size: 5,
        max_summary_length: 200,
        min_node_size: 100,
        overlap_sentences: 2,
        llm_config: LLMConfig {
            enabled: true,
            model_name: "llama3.1:8b".to_string(),
            temperature: 0.25,
            max_tokens: 150,
            strategy: LLMStrategy::Uniform,
            level_configs: HashMap::new(),
        },
    };

    let mut text_processor = TextProcessor::new(400, 100)?;
    let chunks = text_processor.chunk_text(document_text)?;

    println!("Processing {} chunks with batch LLM summarization...", chunks.len());

    let start_time = std::time::Instant::now();

    let mut batch_tree = DocumentTree::with_llm_client(
        DocumentId::new("batch_demo".to_string()),
        batch_config,
        llm_client,
    )?;
    batch_tree.build_from_chunks(chunks).await?;

    let elapsed = start_time.elapsed();
    println!("✅ Batch processing completed in: {:?}", elapsed);

    // Show efficiency
    let stats = batch_tree.get_statistics();
    println!("📊 Batch Processing Results:");
    stats.print();

    println!("\n");
    Ok(())
}