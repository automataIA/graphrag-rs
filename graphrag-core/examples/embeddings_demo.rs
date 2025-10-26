///! Demonstration of embedding providers in graphrag-core
///!
///! This example shows how to use different embedding providers:
///! - Hugging Face Hub (free, downloadable models)
///! - OpenAI API
///! - Voyage AI API
///! - Cohere API
///! - Jina AI API
///! - Mistral AI API
///! - Together AI API

use graphrag_core::embeddings::{
    EmbeddingConfig, EmbeddingProvider, EmbeddingProviderType,
};

#[cfg(feature = "huggingface-hub")]
use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;

#[cfg(feature = "ureq")]
use graphrag_core::embeddings::api_providers::HttpEmbeddingProvider;

#[tokio::main]
async fn main() -> graphrag_core::core::error::Result<()> {
    println!("🚀 GraphRAG Core - Embedding Providers Demo\n");

    // Example text to embed
    let text = "GraphRAG combines knowledge graphs with retrieval-augmented generation.";

    // 1. Hugging Face Hub - Free, downloadable models
    #[cfg(feature = "huggingface-hub")]
    {
        println!("📦 Hugging Face Hub Provider");
        println!("   Model: sentence-transformers/all-MiniLM-L6-v2");

        let mut hf_embeddings = HuggingFaceEmbeddings::new(
            "sentence-transformers/all-MiniLM-L6-v2",
            None, // Use default cache directory
        );

        // Note: This will download the model on first use
        // Set ENABLE_DOWNLOAD_TESTS=1 to actually run this
        if std::env::var("ENABLE_DOWNLOAD_TESTS").is_ok() {
            match hf_embeddings.initialize().await {
                Ok(_) => println!("   ✅ Model downloaded and initialized"),
                Err(e) => println!("   ⚠️  Download skipped: {}", e),
            }

            match hf_embeddings.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  Embedding failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set ENABLE_DOWNLOAD_TESTS=1 to test downloads");
        }
        println!();
    }

    // 2. OpenAI Embeddings
    #[cfg(feature = "ureq")]
    {
        println!("🔵 OpenAI Provider");
        println!("   Model: text-embedding-3-small");

        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            let openai = HttpEmbeddingProvider::openai(
                api_key,
                "text-embedding-3-small".to_string(),
            );

            match openai.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  API call failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set OPENAI_API_KEY to test OpenAI embeddings");
        }
        println!();
    }

    // 3. Voyage AI Embeddings
    #[cfg(feature = "ureq")]
    {
        println!("🟣 Voyage AI Provider (Recommended by Anthropic)");
        println!("   Model: voyage-3-large");

        if let Ok(api_key) = std::env::var("VOYAGE_API_KEY") {
            let voyage = HttpEmbeddingProvider::voyage_ai(
                api_key,
                "voyage-3-large".to_string(),
            );

            match voyage.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  API call failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set VOYAGE_API_KEY to test Voyage AI embeddings");
        }
        println!();
    }

    // 4. Cohere Embeddings
    #[cfg(feature = "ureq")]
    {
        println!("🟢 Cohere Provider");
        println!("   Model: embed-english-v3.0");

        if let Ok(api_key) = std::env::var("COHERE_API_KEY") {
            let cohere = HttpEmbeddingProvider::cohere(
                api_key,
                "embed-english-v3.0".to_string(),
            );

            match cohere.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  API call failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set COHERE_API_KEY to test Cohere embeddings");
        }
        println!();
    }

    // 5. Jina AI Embeddings
    #[cfg(feature = "ureq")]
    {
        println!("🔴 Jina AI Provider");
        println!("   Model: jina-embeddings-v3");

        if let Ok(api_key) = std::env::var("JINA_API_KEY") {
            let jina = HttpEmbeddingProvider::jina_ai(
                api_key,
                "jina-embeddings-v3".to_string(),
            );

            match jina.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  API call failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set JINA_API_KEY to test Jina AI embeddings");
        }
        println!();
    }

    // 6. Mistral AI Embeddings
    #[cfg(feature = "ureq")]
    {
        println!("🟠 Mistral AI Provider");
        println!("   Model: mistral-embed");

        if let Ok(api_key) = std::env::var("MISTRAL_API_KEY") {
            let mistral = HttpEmbeddingProvider::mistral(
                api_key,
                "mistral-embed".to_string(),
            );

            match mistral.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  API call failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set MISTRAL_API_KEY to test Mistral AI embeddings");
        }
        println!();
    }

    // 7. Together AI Embeddings
    #[cfg(feature = "ureq")]
    {
        println!("🟡 Together AI Provider");
        println!("   Model: BAAI/bge-large-en-v1.5");

        if let Ok(api_key) = std::env::var("TOGETHER_API_KEY") {
            let together = HttpEmbeddingProvider::together_ai(
                api_key,
                "BAAI/bge-large-en-v1.5".to_string(),
            );

            match together.embed(text).await {
                Ok(embedding) => println!(
                    "   ✅ Generated embedding: {} dimensions",
                    embedding.len()
                ),
                Err(e) => println!("   ⚠️  API call failed: {}", e),
            }
        } else {
            println!("   ℹ️  Set TOGETHER_API_KEY to test Together AI embeddings");
        }
        println!();
    }

    // 8. Using EmbeddingConfig (for production use)
    #[cfg(feature = "ureq")]
    {
        println!("⚙️  Using EmbeddingConfig");

        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            let config = EmbeddingConfig {
                provider: EmbeddingProviderType::OpenAI,
                model: "text-embedding-3-small".to_string(),
                api_key: Some(api_key),
                cache_dir: None,
                batch_size: 32,
            };

            match HttpEmbeddingProvider::from_config(&config) {
                Ok(provider) => {
                    println!("   ✅ Provider created from config");
                    println!("   Provider: {}", provider.provider_name());
                    println!("   Dimensions: {}", provider.dimensions());
                }
                Err(e) => println!("   ⚠️  Config error: {}", e),
            }
        }
    }

    println!("\n✨ Demo complete!");
    println!("\n💡 Tips:");
    println!("   - Hugging Face: Free, download models once");
    println!("   - API providers: Require API keys, pay-per-use");
    println!("   - See LLM_PROVIDERS.md for detailed comparison");

    Ok(())
}
