//! Error handling for Ollama integration

use std::fmt;

/// Ollama-specific error types
#[derive(Debug)]
pub enum OllamaError {
    /// Connection to Ollama service failed
    ConnectionFailed(String),
    /// Model not available
    ModelNotFound(String),
    /// Request timeout
    Timeout,
    /// API response error
    ApiError(String),
    /// JSON parsing error
    JsonError(String),
    /// Configuration error
    ConfigError(String),
    /// Request timeout
    RequestTimeout,
    /// Generation error
    GenerationError(String),
    /// Embedding dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for OllamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OllamaError::ConnectionFailed(msg) => write!(f, "Ollama connection failed: {msg}"),
            OllamaError::ModelNotFound(model) => write!(f, "Model '{model}' not found in Ollama"),
            OllamaError::Timeout => write!(f, "Ollama request timeout"),
            OllamaError::ApiError(msg) => write!(f, "Ollama API error: {msg}"),
            OllamaError::JsonError(msg) => write!(f, "JSON parsing error: {msg}"),
            OllamaError::ConfigError(msg) => write!(f, "Ollama configuration error: {msg}"),
            OllamaError::RequestTimeout => write!(f, "Ollama request timeout"),
            OllamaError::GenerationError(msg) => write!(f, "Ollama generation error: {msg}"),
            OllamaError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Embedding dimension mismatch: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for OllamaError {}

impl From<ollama_rs::error::OllamaError> for OllamaError {
    fn from(err: ollama_rs::error::OllamaError) -> Self {
        OllamaError::ApiError(err.to_string())
    }
}

impl From<crate::GraphRAGError> for OllamaError {
    fn from(err: crate::GraphRAGError) -> Self {
        OllamaError::ConfigError(err.to_string())
    }
}

/// Result type for Ollama operations
pub type OllamaResult<T> = Result<T, OllamaError>;
