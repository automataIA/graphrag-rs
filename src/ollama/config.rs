//! Configuration for Ollama integration

// Note: Removed serde dependency to keep dependencies minimal
// Can be added back if needed for serialization

/// Ollama-specific configuration
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Whether Ollama integration is enabled
    pub enabled: bool,
    /// Ollama service host URL
    pub host: String,
    /// Port for Ollama service
    pub port: u16,
    /// Model to use for embeddings
    pub embedding_model: String,
    /// Model to use for chat/generation
    pub chat_model: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Whether to fallback to hash-based embeddings if Ollama fails
    pub fallback_to_hash: bool,
    /// Maximum tokens for chat responses
    pub max_tokens: Option<u32>,
    /// Temperature for chat generation
    pub temperature: Option<f32>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default to not break existing setups
            host: "http://localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "llama3.2:3b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            fallback_to_hash: true,
            max_tokens: Some(2000),
            temperature: Some(0.7),
        }
    }
}

impl OllamaConfig {
    /// Get full Ollama URL
    pub fn url(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Check if configuration is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled {
            if self.host.is_empty() {
                return Err("Host cannot be empty when Ollama is enabled".to_string());
            }
            if self.embedding_model.is_empty() {
                return Err("Embedding model cannot be empty when Ollama is enabled".to_string());
            }
            if self.chat_model.is_empty() {
                return Err("Chat model cannot be empty when Ollama is enabled".to_string());
            }
            if self.timeout_seconds == 0 {
                return Err("Timeout must be greater than 0".to_string());
            }
        }
        Ok(())
    }

    /// Create config with custom models
    pub fn with_models(embedding_model: String, chat_model: String) -> Self {
        Self {
            enabled: true,
            embedding_model,
            chat_model,
            ..Default::default()
        }
    }

    /// Enable Ollama with default models
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OllamaConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.host, "http://localhost");
        assert_eq!(config.port, 11434);
        assert_eq!(config.url(), "http://localhost:11434");
    }

    #[test]
    fn test_config_validation() {
        let mut config = OllamaConfig::default();

        // Valid disabled config
        assert!(config.validate().is_ok());

        // Invalid enabled config (empty models)
        config.enabled = true;
        config.embedding_model.clear();
        assert!(config.validate().is_err());

        // Valid enabled config
        config.embedding_model = "nomic-embed-text".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_with_models() {
        let config =
            OllamaConfig::with_models("custom-embed".to_string(), "custom-chat".to_string());

        assert!(config.enabled);
        assert_eq!(config.embedding_model, "custom-embed");
        assert_eq!(config.chat_model, "custom-chat");
    }
}
