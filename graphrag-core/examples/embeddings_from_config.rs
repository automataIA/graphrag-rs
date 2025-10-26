///! Load embedding configuration from TOML file
///!
///! This example demonstrates how to configure embedding providers
///! using a TOML configuration file.
///!
///! Run with:
///! ```bash
///! cargo run --example embeddings_from_config --features "ureq,huggingface-hub"
///! ```

use graphrag_core::embeddings::config::EmbeddingProviderConfig;
use graphrag_core::embeddings::EmbeddingProvider;

#[cfg(feature = "ureq")]
use graphrag_core::embeddings::api_providers::HttpEmbeddingProvider;

#[cfg(feature = "huggingface-hub")]
use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;

#[tokio::main]
async fn main() -> graphrag_core::core::error::Result<()> {
    println!("📝 GraphRAG - Load Embeddings from TOML Config\n");

    // Example 1: Load from TOML file
    println!("1️⃣  Loading config from embeddings.toml...");

    let config_path = "examples/embeddings.toml";
    let config = match EmbeddingProviderConfig::from_toml_file(config_path) {
        Ok(cfg) => {
            println!("   ✅ Config loaded successfully");
            println!("   Provider: {}", cfg.provider);
            println!("   Model: {}", cfg.model);
            println!("   Batch size: {}", cfg.batch_size);
            cfg
        }
        Err(e) => {
            println!("   ⚠️  Failed to load config: {}", e);
            println!("   Using default configuration instead\n");
            EmbeddingProviderConfig::default()
        }
    };

    // Convert to EmbeddingConfig
    let embedding_config = config.to_embedding_config()?;
    println!("   Provider type: {:?}\n", embedding_config.provider);

    // Example 2: Create provider from config
    println!("2️⃣  Creating provider from config...");

    match embedding_config.provider {
        graphrag_core::embeddings::EmbeddingProviderType::HuggingFace => {
            #[cfg(feature = "huggingface-hub")]
            {
                println!("   Creating HuggingFace provider");
                let mut hf = HuggingFaceEmbeddings::from_config(&embedding_config);

                if std::env::var("ENABLE_DOWNLOAD_TESTS").is_ok() {
                    println!("   Initializing (downloading model if needed)...");
                    match hf.initialize().await {
                        Ok(_) => {
                            println!("   ✅ Provider initialized");
                            println!("   Dimensions: {}", hf.dimensions());

                            // Test embedding
                            let text = "GraphRAG combines knowledge graphs with RAG.";
                            match hf.embed(text).await {
                                Ok(embedding) => {
                                    println!(
                                        "   ✅ Generated embedding: {} dimensions",
                                        embedding.len()
                                    );
                                }
                                Err(e) => println!("   ⚠️  Embedding failed: {}", e),
                            }
                        }
                        Err(e) => println!("   ⚠️  Initialization failed: {}", e),
                    }
                } else {
                    println!("   ℹ️  Set ENABLE_DOWNLOAD_TESTS=1 to test downloads");
                }
            }

            #[cfg(not(feature = "huggingface-hub"))]
            {
                println!("   ⚠️  HuggingFace feature not enabled");
                println!("   Enable with: --features huggingface-hub");
            }
        }
        _ => {
            #[cfg(feature = "ureq")]
            {
                println!("   Creating HTTP API provider");
                match HttpEmbeddingProvider::from_config(&embedding_config) {
                    Ok(provider) => {
                        println!("   ✅ Provider created: {}", provider.provider_name());
                        println!("   Dimensions: {}", provider.dimensions());

                        // Test if provider is ready (has API key)
                        if provider.is_available() {
                            let text = "GraphRAG combines knowledge graphs with RAG.";
                            match provider.embed(text).await {
                                Ok(embedding) => {
                                    println!(
                                        "   ✅ Generated embedding: {} dimensions",
                                        embedding.len()
                                    );
                                }
                                Err(e) => println!("   ⚠️  API call failed: {}", e),
                            }
                        } else {
                            println!("   ⚠️  API key not set");
                            println!("   Set api_key in config or use environment variable");
                        }
                    }
                    Err(e) => println!("   ⚠️  Failed to create provider: {}", e),
                }
            }

            #[cfg(not(feature = "ureq"))]
            {
                println!("   ⚠️  HTTP features not enabled");
                println!("   Enable with: --features ureq");
            }
        }
    }

    println!("\n3️⃣  Example configurations:");
    for (name, example_config) in EmbeddingProviderConfig::examples() {
        println!("\n   📌 {}", name);
        println!("      Provider: {}", example_config.provider);
        println!("      Model: {}", example_config.model);
        println!("      Batch size: {}", example_config.batch_size);
        if let Some(dims) = example_config.dimensions {
            println!("      Dimensions: {}", dims);
        }
    }

    println!("\n4️⃣  Generating example TOML configs:");

    // Generate example configs
    let examples_to_generate = vec![
        (
            "config-huggingface.toml",
            EmbeddingProviderConfig {
                provider: "huggingface".to_string(),
                model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                api_key: None,
                cache_dir: Some("~/.cache/huggingface".to_string()),
                batch_size: 32,
                dimensions: Some(384),
            },
        ),
        (
            "config-openai.toml",
            EmbeddingProviderConfig {
                provider: "openai".to_string(),
                model: "text-embedding-3-small".to_string(),
                api_key: Some("sk-...".to_string()),
                cache_dir: None,
                batch_size: 100,
                dimensions: Some(1536),
            },
        ),
        (
            "config-voyage.toml",
            EmbeddingProviderConfig {
                provider: "voyage".to_string(),
                model: "voyage-3-large".to_string(),
                api_key: Some("pa-...".to_string()),
                cache_dir: None,
                batch_size: 128,
                dimensions: Some(1024),
            },
        ),
    ];

    for (filename, config) in examples_to_generate {
        match config.to_toml_file(filename) {
            Ok(_) => println!("   ✅ Generated {}", filename),
            Err(e) => println!("   ⚠️  Failed to generate {}: {}", filename, e),
        }
    }

    println!("\n✨ Demo complete!");
    println!("\n💡 Tips:");
    println!("   - Edit examples/embeddings.toml to configure your provider");
    println!("   - Set API keys via environment variables (recommended)");
    println!("   - Use HuggingFace for free, offline embeddings");
    println!("   - Use API providers for production deployments");

    Ok(())
}
