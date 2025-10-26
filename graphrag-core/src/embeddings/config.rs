//! Configuration for embedding providers via TOML
//!
//! This module provides TOML-based configuration for all embedding providers.

use crate::core::error::{GraphRAGError, Result};
use crate::embeddings::{EmbeddingConfig, EmbeddingProviderType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// TOML configuration for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsTomlConfig {
    /// Embedding provider configuration
    #[serde(default)]
    pub embeddings: EmbeddingProviderConfig,
}

/// Embedding provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProviderConfig {
    /// Provider type: "huggingface", "openai", "voyage", "cohere", "jina", "mistral", "together"
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Model identifier
    /// - HuggingFace: "sentence-transformers/all-MiniLM-L6-v2"
    /// - OpenAI: "text-embedding-3-small" or "text-embedding-3-large"
    /// - Voyage: "voyage-3-large", "voyage-code-3", etc.
    /// - Cohere: "embed-english-v3.0"
    /// - Jina: "jina-embeddings-v3"
    /// - Mistral: "mistral-embed"
    /// - Together: "BAAI/bge-large-en-v1.5"
    #[serde(default = "default_model")]
    pub model: String,

    /// API key (for API providers)
    /// Can also be set via environment variables:
    /// - OPENAI_API_KEY
    /// - VOYAGE_API_KEY
    /// - COHERE_API_KEY
    /// - JINA_API_KEY
    /// - MISTRAL_API_KEY
    /// - TOGETHER_API_KEY
    pub api_key: Option<String>,

    /// Cache directory for downloaded models (HuggingFace)
    /// Default: ~/.cache/huggingface/hub
    pub cache_dir: Option<String>,

    /// Batch size for processing multiple texts
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Embedding dimensions (read-only, determined by model)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
}

impl Default for EmbeddingProviderConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            api_key: None,
            cache_dir: None,
            batch_size: default_batch_size(),
            dimensions: None,
        }
    }
}

fn default_provider() -> String {
    "huggingface".to_string()
}

fn default_model() -> String {
    "sentence-transformers/all-MiniLM-L6-v2".to_string()
}

fn default_batch_size() -> usize {
    32
}

impl EmbeddingProviderConfig {
    /// Convert TOML config to EmbeddingConfig
    pub fn to_embedding_config(&self) -> Result<EmbeddingConfig> {
        // Parse provider type
        let provider = match self.provider.to_lowercase().as_str() {
            "huggingface" | "hf" => EmbeddingProviderType::HuggingFace,
            "openai" => EmbeddingProviderType::OpenAI,
            "voyage" | "voyageai" | "voyage-ai" => EmbeddingProviderType::VoyageAI,
            "cohere" => EmbeddingProviderType::Cohere,
            "jina" | "jinaai" | "jina-ai" => EmbeddingProviderType::JinaAI,
            "mistral" | "mistralai" | "mistral-ai" => EmbeddingProviderType::Mistral,
            "together" | "togetherai" | "together-ai" => EmbeddingProviderType::TogetherAI,
            "onnx" => EmbeddingProviderType::Onnx,
            "candle" => EmbeddingProviderType::Candle,
            _ => {
                return Err(GraphRAGError::Config {
                    message: format!("Unknown embedding provider: {}", self.provider),
                })
            },
        };

        // Get API key from config or environment
        let api_key = self.api_key.clone().or_else(|| match provider {
            EmbeddingProviderType::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
            EmbeddingProviderType::VoyageAI => std::env::var("VOYAGE_API_KEY").ok(),
            EmbeddingProviderType::Cohere => std::env::var("COHERE_API_KEY").ok(),
            EmbeddingProviderType::JinaAI => std::env::var("JINA_API_KEY").ok(),
            EmbeddingProviderType::Mistral => std::env::var("MISTRAL_API_KEY").ok(),
            EmbeddingProviderType::TogetherAI => std::env::var("TOGETHER_API_KEY").ok(),
            _ => None,
        });

        Ok(EmbeddingConfig {
            provider,
            model: self.model.clone(),
            api_key,
            cache_dir: self.cache_dir.clone(),
            batch_size: self.batch_size,
        })
    }

    /// Load from TOML file
    pub fn from_toml_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to read config file {:?}: {}", path, e),
        })?;

        let config: EmbeddingsTomlConfig =
            toml::from_str(&content).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to parse TOML config: {}", e),
            })?;

        Ok(config.embeddings)
    }

    /// Save to TOML file
    pub fn to_toml_file(&self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        let config = EmbeddingsTomlConfig {
            embeddings: self.clone(),
        };

        let toml_string = toml::to_string_pretty(&config).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to serialize TOML: {}", e),
        })?;

        std::fs::write(&path, toml_string).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write config file {:?}: {}", path, e),
        })?;

        Ok(())
    }

    /// Create example configurations for different use cases
    pub fn examples() -> Vec<(String, Self)> {
        vec![
            (
                "HuggingFace (Free, Offline)".to_string(),
                Self {
                    provider: "huggingface".to_string(),
                    model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    api_key: None,
                    cache_dir: Some("~/.cache/huggingface".to_string()),
                    batch_size: 32,
                    dimensions: Some(384),
                },
            ),
            (
                "HuggingFace (High Quality)".to_string(),
                Self {
                    provider: "huggingface".to_string(),
                    model: "BAAI/bge-large-en-v1.5".to_string(),
                    api_key: None,
                    cache_dir: None,
                    batch_size: 16,
                    dimensions: Some(1024),
                },
            ),
            (
                "OpenAI (Production)".to_string(),
                Self {
                    provider: "openai".to_string(),
                    model: "text-embedding-3-small".to_string(),
                    api_key: Some("sk-...".to_string()),
                    cache_dir: None,
                    batch_size: 100,
                    dimensions: Some(1536),
                },
            ),
            (
                "Voyage AI (Recommended by Anthropic)".to_string(),
                Self {
                    provider: "voyage".to_string(),
                    model: "voyage-3-large".to_string(),
                    api_key: Some("pa-...".to_string()),
                    cache_dir: None,
                    batch_size: 128,
                    dimensions: Some(1024),
                },
            ),
            (
                "Voyage AI (Code Search)".to_string(),
                Self {
                    provider: "voyage".to_string(),
                    model: "voyage-code-3".to_string(),
                    api_key: Some("pa-...".to_string()),
                    cache_dir: None,
                    batch_size: 64,
                    dimensions: Some(1024),
                },
            ),
            (
                "Cohere (Multilingual)".to_string(),
                Self {
                    provider: "cohere".to_string(),
                    model: "embed-multilingual-v3.0".to_string(),
                    api_key: Some("...".to_string()),
                    cache_dir: None,
                    batch_size: 96,
                    dimensions: Some(1024),
                },
            ),
            (
                "Jina AI (Cost Optimized)".to_string(),
                Self {
                    provider: "jina".to_string(),
                    model: "jina-embeddings-v3".to_string(),
                    api_key: Some("jina_...".to_string()),
                    cache_dir: None,
                    batch_size: 200,
                    dimensions: Some(1024),
                },
            ),
            (
                "Mistral (RAG Optimized)".to_string(),
                Self {
                    provider: "mistral".to_string(),
                    model: "mistral-embed".to_string(),
                    api_key: Some("...".to_string()),
                    cache_dir: None,
                    batch_size: 50,
                    dimensions: Some(1024),
                },
            ),
            (
                "Together AI (Cheapest)".to_string(),
                Self {
                    provider: "together".to_string(),
                    model: "BAAI/bge-large-en-v1.5".to_string(),
                    api_key: Some("...".to_string()),
                    cache_dir: None,
                    batch_size: 128,
                    dimensions: Some(1024),
                },
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingProviderConfig::default();
        assert_eq!(config.provider, "huggingface");
        assert_eq!(config.model, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_to_embedding_config() {
        let toml_config = EmbeddingProviderConfig {
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("sk-test".to_string()),
            cache_dir: None,
            batch_size: 50,
            dimensions: None,
        };

        let config = toml_config.to_embedding_config().unwrap();
        assert_eq!(config.provider, EmbeddingProviderType::OpenAI);
        assert_eq!(config.model, "text-embedding-3-small");
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_provider_aliases() {
        let configs = vec![
            ("huggingface", EmbeddingProviderType::HuggingFace),
            ("hf", EmbeddingProviderType::HuggingFace),
            ("openai", EmbeddingProviderType::OpenAI),
            ("voyage", EmbeddingProviderType::VoyageAI),
            ("voyageai", EmbeddingProviderType::VoyageAI),
            ("voyage-ai", EmbeddingProviderType::VoyageAI),
            ("cohere", EmbeddingProviderType::Cohere),
            ("jina", EmbeddingProviderType::JinaAI),
            ("jinaai", EmbeddingProviderType::JinaAI),
            ("mistral", EmbeddingProviderType::Mistral),
            ("together", EmbeddingProviderType::TogetherAI),
        ];

        for (alias, expected) in configs {
            let config = EmbeddingProviderConfig {
                provider: alias.to_string(),
                ..Default::default()
            };
            let result = config.to_embedding_config().unwrap();
            assert_eq!(result.provider, expected, "Failed for alias: {}", alias);
        }
    }

    #[test]
    fn test_toml_serialization() {
        let config = EmbeddingProviderConfig {
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("sk-test".to_string()),
            cache_dir: Some("/custom/cache".to_string()),
            batch_size: 100,
            dimensions: Some(1536),
        };

        let toml_string = toml::to_string_pretty(&EmbeddingsTomlConfig {
            embeddings: config.clone(),
        })
        .unwrap();

        assert!(toml_string.contains("provider = \"openai\""));
        assert!(toml_string.contains("model = \"text-embedding-3-small\""));
        assert!(toml_string.contains("batch_size = 100"));
    }

    #[test]
    fn test_examples() {
        let examples = EmbeddingProviderConfig::examples();
        assert!(!examples.is_empty());

        for (name, config) in examples {
            println!("Testing example: {}", name);
            assert!(!config.provider.is_empty());
            assert!(!config.model.is_empty());
            assert!(config.batch_size > 0);

            // Should convert successfully
            let embedding_config = config.to_embedding_config();
            assert!(embedding_config.is_ok(), "Failed for: {}", name);
        }
    }
}
