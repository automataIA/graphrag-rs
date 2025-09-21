//! Ollama integration for GraphRAG-rs
//!
//! This module provides integration with Ollama for:
//! - Local LLM inference for answer generation
//! - Semantic embeddings via local models
//! - Fallback support when Ollama is unavailable

pub mod client;
pub mod config;
pub mod embeddings;
pub mod error;
#[cfg(feature = "function-calling")]
pub mod function_calling;
pub mod generation;

// Async implementation module
#[cfg(feature = "async-traits")]
pub mod async_generation;

pub use client::OllamaClient;
pub use config::OllamaConfig;
pub use embeddings::OllamaEmbeddings;
pub use error::{OllamaError, OllamaResult};
#[cfg(feature = "function-calling")]
pub use function_calling::{FunctionCallingConfig, FunctionCallingSession, OllamaFunctionAgent};
pub use generation::OllamaGenerator;

// Async exports
#[cfg(feature = "async-traits")]
pub use async_generation::{AsyncOllamaGenerator, AsyncOllamaGeneratorBuilder};

/// Check if Ollama service is available
pub async fn is_ollama_available(host: &str) -> bool {
    (ollama_rs::Ollama::new(host.to_string(), 11434)
        .list_local_models()
        .await)
        .is_ok()
}

/// Get default Ollama configuration
pub fn default_config() -> OllamaConfig {
    OllamaConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_availability() {
        // This test will only pass if Ollama is running locally
        let available = is_ollama_available("http://localhost").await;
        println!("Ollama available: {available}");
        // Don't assert, just print status
    }
}
