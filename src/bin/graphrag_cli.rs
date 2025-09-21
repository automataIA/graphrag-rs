//! Enhanced GraphRAG CLI with TOML configuration support
//! Usage:
//!   cargo run --bin graphrag_cli -- <config.toml> build
//!   cargo run --bin graphrag_cli -- <config.toml> query "Your question here"

use graphrag_rs::{
    config::TomlConfig,
    ollama::{OllamaClient, OllamaEmbeddings},
};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

// Enhanced GraphRAG implementation using TOML config
struct TomlGraphRAG {
    config: TomlConfig,
    ollama_embeddings: OllamaEmbeddings,
    chunks: Vec<String>,
    chunk_embeddings: Vec<Vec<f32>>,
    entities: HashMap<String, Vec<usize>>,
}

impl TomlGraphRAG {
    async fn from_config_file(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("📝 Loading TOML configuration from: {config_path}");

        // Load TOML configuration
        let config = TomlConfig::from_file(config_path)?;

        println!("✅ TOML configuration loaded successfully");
        println!("  📊 Workflows: {:?}", config.pipeline.workflows);
        println!(
            "  🔧 Chunk size: {}",
            config.pipeline.text_extraction.chunk_size
        );
        println!("  🧮 Ollama model: {}", config.ollama.chat_model);
        println!("  📁 Output dir: {}", config.general.output_dir);

        // Create Ollama embeddings from config
        let ollama_config = graphrag_rs::ollama::OllamaConfig {
            enabled: config.ollama.enabled,
            host: config.ollama.host.clone(),
            port: config.ollama.port,
            chat_model: config.ollama.chat_model.clone(),
            embedding_model: config.ollama.embedding_model.clone(),
            timeout_seconds: config.ollama.timeout_seconds,
            max_retries: config.ollama.max_retries,
            fallback_to_hash: config.ollama.fallback_to_hash,
            max_tokens: config.ollama.max_tokens,
            temperature: config.ollama.temperature,
        };

        let ollama_embeddings = OllamaEmbeddings::new(ollama_config)?;

        Ok(Self {
            config,
            ollama_embeddings,
            chunks: Vec::new(),
            chunk_embeddings: Vec::new(),
            entities: HashMap::new(),
        })
    }

    async fn build_knowledge_graph(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🔧 Step 2: Testing Ollama Connection");

        // Test Ollama connection
        let test_ollama_config = graphrag_rs::ollama::OllamaConfig {
            enabled: self.config.ollama.enabled,
            host: self.config.ollama.host.clone(),
            port: self.config.ollama.port,
            chat_model: self.config.ollama.chat_model.clone(),
            embedding_model: self.config.ollama.embedding_model.clone(),
            timeout_seconds: self.config.ollama.timeout_seconds,
            max_retries: self.config.ollama.max_retries,
            fallback_to_hash: self.config.ollama.fallback_to_hash,
            max_tokens: self.config.ollama.max_tokens,
            temperature: self.config.ollama.temperature,
        };

        let ollama_client = OllamaClient::new(test_ollama_config)?;

        if !ollama_client.health_check().await? {
            println!("❌ Ollama connection failed");
            return Err("Ollama not available".into());
        }

        println!(
            "✅ Ollama connected: {} + {}",
            self.config.ollama.chat_model, self.config.ollama.embedding_model
        );

        println!("\n📚 Step 3: Loading Document from Configuration");

        // Load document from configured path
        let document_path = self
            .config
            .general
            .input_document_path
            .as_ref()
            .ok_or("No input document path configured")?;

        let content = fs::read_to_string(document_path)?;
        println!("✅ Loaded document from: {document_path}");
        println!("✅ Processing document: {} characters", content.len());

        println!("\n🔄 Step 4: Processing Document with TOML Settings");

        // Create chunks
        let chunk_size = self.config.pipeline.text_extraction.chunk_size;
        let chunk_overlap = self.config.pipeline.text_extraction.chunk_overlap;
        let min_chunk_size = 200; // from config or default

        println!("🔄 Processing document using TOML configuration...");
        println!("  📏 Chunk size: {chunk_size} characters");
        println!("  🔗 Overlap: {chunk_overlap} characters");
        println!("  📐 Min chunk size: {min_chunk_size} characters");

        // Simple chunking implementation
        self.chunks = create_chunks(&content, chunk_size, chunk_overlap, min_chunk_size);
        println!("✅ Created {} chunks", self.chunks.len());

        // Generate embeddings
        println!("🧮 Generating embeddings with Ollama...");
        let start_time = Instant::now();

        for (i, chunk) in self.chunks.iter().enumerate() {
            let chunk_start = Instant::now();
            let embedding = self
                .ollama_embeddings
                .generate_embedding_async(chunk)
                .await?;
            let chunk_duration = chunk_start.elapsed();

            self.chunk_embeddings.push(embedding.clone());

            // Print progress for every 5th chunk
            if i == 0 || (i + 1) % 5 == 0 {
                println!(
                    "  ✅ Chunk {} embedded in {:?} ({} dims)",
                    i + 1,
                    chunk_duration,
                    embedding.len()
                );
            }
        }

        let total_duration = start_time.elapsed();
        println!("✅ All embeddings generated in {total_duration:?}");

        // Extract entities (simplified)
        println!("👤 Extracting entities...");
        self.entities = extract_simple_entities(&self.chunks);
        let entity_names: Vec<String> = self.entities.keys().cloned().collect();
        println!(
            "✅ Extracted {} entities: {:?}",
            entity_names.len(),
            entity_names
        );

        // Save results
        self.save_results().await?;

        let processing_duration = start_time.elapsed();
        println!("✅ Total processing time: {processing_duration:?}");

        Ok(())
    }

    async fn save_results(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Create output directory
        let output_dir = &self.config.general.output_dir;
        fs::create_dir_all(output_dir)?;
        println!("📁 Output directory ready: {output_dir}");

        // Save entities
        let entities_path = format!("{output_dir}/entities.json");
        let entities_json = serde_json::to_string_pretty(&self.entities)?;
        fs::write(&entities_path, entities_json)?;

        // Save config used (dynamically determined from the config that was actually loaded)
        let config_path = format!("{output_dir}/used_config.toml");
        self.config.save_to_file(&config_path)?;

        Ok(())
    }

}

// Advanced text chunking with semantic boundary preservation (2024 best practices)
fn create_chunks(text: &str, chunk_size: usize, overlap: usize, min_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;

    // Hierarchical separators following LangChain RecursiveCharacterTextSplitter approach
    let separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""];

    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());

        // Ensure we're on a UTF-8 character boundary first
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }

        // If we're at the exact end, no need to adjust
        if end >= text.len() {
            let chunk = &text[start..];
            if chunk.trim().len() >= min_size {
                chunks.push(chunk.to_string());
            }
            break;
        }

        // Find the best boundary to avoid word truncation
        let optimal_end = find_optimal_boundary(text, start, end, &separators);

        // If we found a good boundary, use it
        if optimal_end > start {
            end = optimal_end;
        }

        let chunk = &text[start..end];

        if chunk.trim().len() >= min_size {
            chunks.push(chunk.to_string());
        }

        if end >= text.len() {
            break;
        }

        // Calculate next start with overlap, preserving word boundaries
        let mut next_start = end.saturating_sub(overlap);

        // Ensure next start is on a UTF-8 boundary
        while next_start > 0 && !text.is_char_boundary(next_start) {
            next_start -= 1;
        }

        // Try to align next start with word boundary
        next_start = find_word_boundary_backward(text, next_start);

        start = next_start;
    }

    chunks
}

// Find optimal boundary using hierarchical separators (inspired by LangChain)
fn find_optimal_boundary(text: &str, start: usize, max_end: usize, separators: &[&str]) -> usize {
    let search_text = &text[start..max_end];

    // Try each separator in order of preference
    for &separator in separators {
        if separator.is_empty() {
            continue;
        }

        // Find the last occurrence of this separator within our range
        if let Some(sep_pos) = search_text.rfind(separator) {
            let boundary = start + sep_pos + separator.len();

            // Make sure we're not too close to the start (maintain minimum chunk size)
            if boundary > start + (max_end - start) / 4 {
                return boundary;
            }
        }
    }

    // If no good separator found, try to at least end at a word boundary
    find_word_boundary_backward(text, max_end)
}

// Find the nearest word boundary going backward from the given position
fn find_word_boundary_backward(text: &str, mut pos: usize) -> usize {
    // Ensure we're on a UTF-8 boundary
    while pos > 0 && !text.is_char_boundary(pos) {
        pos -= 1;
    }

    // Look for whitespace (word boundary) going backward
    while pos > 0 {
        if let Some(ch) = text.chars().nth(pos.saturating_sub(1)) {
            if ch.is_whitespace() {
                return pos;
            }
        }
        pos = pos.saturating_sub(1);

        // Ensure we stay on UTF-8 boundaries
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }
    }

    pos
}

// Simple entity extraction
fn extract_simple_entities(chunks: &[String]) -> HashMap<String, Vec<usize>> {
    let mut entities = HashMap::new();

    // Simple pattern matching for proper nouns (starting with capital letter)
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let words: Vec<&str> = chunk.split_whitespace().collect();

        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
            if clean_word.len() > 2
                && clean_word.chars().next().unwrap().is_uppercase()
                && clean_word.chars().all(|c| c.is_alphabetic())
            {
                entities
                    .entry(clean_word.to_string())
                    .or_insert_with(Vec::new)
                    .push(chunk_idx);
            }
        }
    }

    // Keep only entities that appear in multiple chunks
    entities.retain(|_, chunks| chunks.len() > 1);

    entities
}

// Cosine similarity calculation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

// Query interface for pre-computed GraphRAG
struct GraphRAGQuery {
    config: TomlConfig,
    ollama_embeddings: OllamaEmbeddings,
    chunks: Vec<String>,
    chunk_embeddings: Vec<Vec<f32>>,
    entities: HashMap<String, Vec<usize>>,
}

impl GraphRAGQuery {
    async fn load_from_config(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("📝 Loading pre-computed GraphRAG from configuration...");

        // Load TOML configuration
        let config = TomlConfig::from_file(config_path)?;
        let output_dir = &config.general.output_dir;

        // Check if GraphRAG files exist
        let entities_path = format!("{output_dir}/entities.json");

        if !std::path::Path::new(&entities_path).exists() {
            return Err(format!(
                "No pre-computed GraphRAG found in {output_dir}. Please run 'build' command first."
            )
            .into());
        }

        println!("✅ Found pre-computed GraphRAG in: {output_dir}");

        // Load entities
        let entities_content = fs::read_to_string(&entities_path)?;
        let entities: HashMap<String, Vec<usize>> = serde_json::from_str(&entities_content)
            .map_err(|e| format!("Failed to parse entities: {e}"))?;
        println!("📚 Loaded {} entities", entities.len());

        // Recreate chunks by re-reading the document
        let document_path = config
            .general
            .input_document_path
            .as_ref()
            .ok_or("No input document path configured")?;
        let content = fs::read_to_string(document_path)?;
        let chunks = create_chunks(
            &content,
            config.pipeline.text_extraction.chunk_size,
            config.pipeline.text_extraction.chunk_overlap,
            200,
        );
        println!("📄 Recreated {} chunks for querying", chunks.len());

        // Create Ollama embeddings from config
        let ollama_config = graphrag_rs::ollama::OllamaConfig {
            enabled: config.ollama.enabled,
            host: config.ollama.host.clone(),
            port: config.ollama.port,
            chat_model: config.ollama.chat_model.clone(),
            embedding_model: config.ollama.embedding_model.clone(),
            timeout_seconds: config.ollama.timeout_seconds,
            max_retries: config.ollama.max_retries,
            fallback_to_hash: config.ollama.fallback_to_hash,
            max_tokens: config.ollama.max_tokens,
            temperature: config.ollama.temperature,
        };

        let ollama_embeddings = OllamaEmbeddings::new(ollama_config)?;
        println!(
            "✅ Connected to Ollama: {} + {}",
            config.ollama.chat_model, config.ollama.embedding_model
        );

        Ok(Self {
            config,
            ollama_embeddings,
            chunks,
            chunk_embeddings: Vec::new(), // Will be generated on-demand
            entities,
        })
    }

    async fn query(&mut self, query_text: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🎯 Query: \"{query_text}\"");
        println!("🔍 Searching for: \"{query_text}\"");

        // Generate embeddings for all chunks if not already done
        if self.chunk_embeddings.is_empty() {
            println!("⚡ Generating embeddings for similarity search...");
            for (i, chunk) in self.chunks.iter().enumerate() {
                let embedding = self
                    .ollama_embeddings
                    .generate_embedding_async(chunk)
                    .await?;
                self.chunk_embeddings.push(embedding);

                if (i + 1) % 10 == 0 {
                    println!("   Processed {}/{} chunks", i + 1, self.chunks.len());
                }
            }
        }

        let start_time = Instant::now();

        // Generate query embedding
        let query_embedding = self
            .ollama_embeddings
            .generate_embedding_async(query_text)
            .await?;

        // Calculate similarities
        let mut similarities: Vec<(usize, f32)> = Vec::new();

        for (i, chunk_embedding) in self.chunk_embeddings.iter().enumerate() {
            let similarity = cosine_similarity(&query_embedding, chunk_embedding);
            similarities.push((i, similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let search_duration = start_time.elapsed();
        println!("✅ Found 5 results in {search_duration:?}");

        // Display top results
        println!("\n📊 Top Results:");
        for (rank, (chunk_idx, score)) in similarities.iter().take(5).enumerate() {
            let chunk = &self.chunks[*chunk_idx];
            let preview = if chunk.len() > 150 {
                format!("{}...", &chunk[..147])
            } else {
                chunk.clone()
            };

            println!("  {}. Score: {:.3}", rank + 1, score);
            println!("      {preview}");
            println!();
        }

        // Find relevant entities in the query
        for (entity_name, _chunk_indices) in &self.entities {
            if query_text.to_lowercase().contains(&entity_name.to_lowercase()) {
                println!("📍 Entity '{}' found in {} chunks", entity_name, _chunk_indices.len());
                break; // Just show the first matching entity to avoid spam
            }
        }

        // Generate final answer using LLM
        println!("🤖 Generating answer using LLM...");

        // Collect top 5 chunks as context
        let context_chunks: Vec<String> = similarities
            .iter()
            .take(5)
            .map(|(chunk_idx, _score)| self.chunks[*chunk_idx].clone())
            .collect();

        let context = context_chunks.join("\n\n");

        // Create a prompt for the LLM
        let prompt = format!(
            "Based on the following context from the document, please answer this question: {query_text}\n\nContext:\n{context}\n\nAnswer:"
        );

        // Create Ollama client for chat completion
        let ollama_config = graphrag_rs::ollama::OllamaConfig {
            enabled: self.config.ollama.enabled,
            host: self.config.ollama.host.clone(),
            port: self.config.ollama.port,
            chat_model: self.config.ollama.chat_model.clone(),
            embedding_model: self.config.ollama.embedding_model.clone(),
            timeout_seconds: self.config.ollama.timeout_seconds,
            max_retries: self.config.ollama.max_retries,
            fallback_to_hash: self.config.ollama.fallback_to_hash,
            max_tokens: self.config.ollama.max_tokens,
            temperature: self.config.ollama.temperature,
        };

        let ollama_client = graphrag_rs::ollama::OllamaClient::new(ollama_config)?;

        match ollama_client.generate_response(&prompt).await {
            Ok(answer) => {
                println!("\n💬 Answer:");
                println!("==========================================");
                println!("{answer}");
                println!("==========================================");
            }
            Err(e) => {
                println!("❌ Error generating answer: {e}");
                println!("💡 Context is available above for manual review");
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  {} <config.toml> build", args[0]);
        eprintln!("  {} <config.toml> query \"Your question here\"", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} config_symposium_working.toml build", args[0]);
        eprintln!(
            "  {} config_symposium_working.toml query \"Who is Socrates?\"",
            args[0]
        );
        std::process::exit(1);
    }

    let config_path = &args[1];
    let command = &args[2];

    match command.as_str() {
        "build" => {
            println!("🎯 GraphRAG Builder");
            println!("===================");

            let mut graphrag = TomlGraphRAG::from_config_file(config_path).await?;
            graphrag.build_knowledge_graph().await?;

            println!("\n🎉 GraphRAG Build Complete!");
            println!("✅ Knowledge graph ready for queries");
            println!(
                "💡 Use: {} {} query \"Your question\"",
                args[0], config_path
            );
        }
        "query" => {
            if args.len() < 4 {
                eprintln!("Error: No query provided");
                eprintln!(
                    "Usage: {} {} query \"Your question here\"",
                    args[0], config_path
                );
                std::process::exit(1);
            }

            let query_text = &args[3];

            println!("🔍 GraphRAG Query Tool");
            println!("======================");

            let mut query_system = GraphRAGQuery::load_from_config(config_path).await?;
            query_system.query(query_text).await?;
        }
        _ => {
            eprintln!("Error: Unknown command '{command}'");
            eprintln!("Available commands: build, query");
            std::process::exit(1);
        }
    }

    Ok(())
}
