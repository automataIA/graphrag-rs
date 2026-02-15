//! Unified error handling for the GraphRAG system
//!
//! This module provides a centralized error type that encompasses all possible
//! errors that can occur throughout the GraphRAG pipeline.

use std::fmt;

/// Main error type for the GraphRAG system
#[derive(Debug)]
pub enum GraphRAGError {
    /// Configuration-related errors
    Config {
        /// Error message
        message: String
    },

    /// System not initialized error with helpful guidance
    NotInitialized,

    /// No documents added error with helpful guidance
    NoDocuments,

    /// I/O errors from file operations
    Io(std::io::Error),

    /// HTTP request errors
    #[cfg(feature = "ureq")]
    Http(Box<ureq::Error>),

    /// HTTP request errors (WASM-compatible)
    #[cfg(not(feature = "ureq"))]
    Http(String),

    /// JSON parsing/serialization errors
    Json(json::Error),

    /// Serde JSON errors
    SerdeJson(serde_json::Error),

    /// Text processing errors
    TextProcessing {
        /// Error message
        message: String
    },

    /// Graph construction and manipulation errors
    GraphConstruction {
        /// Error message
        message: String
    },

    /// Vector search and embedding errors
    VectorSearch {
        /// Error message
        message: String
    },

    /// Entity extraction errors
    EntityExtraction {
        /// Error message
        message: String
    },

    /// Retrieval system errors
    Retrieval {
        /// Error message
        message: String
    },

    /// Answer generation errors
    Generation {
        /// Error message
        message: String
    },

    /// Function calling errors
    FunctionCall {
        /// Error message
        message: String
    },

    /// Storage backend errors
    Storage {
        /// Error message
        message: String
    },

    /// Embedding model errors
    Embedding {
        /// Error message
        message: String
    },

    /// Language model errors
    LanguageModel {
        /// Error message
        message: String
    },

    /// Parallel processing errors
    Parallel {
        /// Error message
        message: String
    },

    /// Serialization errors
    Serialization {
        /// Error message
        message: String
    },

    /// Validation errors
    Validation {
        /// Error message
        message: String
    },

    /// Network connectivity errors
    Network {
        /// Error message
        message: String
    },

    /// Authentication/authorization errors
    Auth {
        /// Error message
        message: String
    },

    /// Resource not found errors
    NotFound {
        /// Resource type
        resource: String,
        /// Resource identifier
        id: String
    },

    /// Already exists errors
    AlreadyExists {
        /// Resource type
        resource: String,
        /// Resource identifier
        id: String
    },

    /// Operation timeout errors
    Timeout {
        /// Operation name
        operation: String,
        /// Timeout duration
        duration: std::time::Duration,
    },

    /// Capacity/resource limit errors
    ResourceLimit {
        /// Resource name
        resource: String,
        /// Limit value
        limit: usize
    },

    /// Data corruption or integrity errors
    DataCorruption {
        /// Error message
        message: String
    },

    /// Unsupported operation errors
    Unsupported {
        /// Operation name
        operation: String,
        /// Reason for not supporting
        reason: String
    },

    /// Rate limiting errors
    RateLimit {
        /// Error message
        message: String
    },

    /// Conflict resolution errors
    ConflictResolution {
        /// Error message
        message: String
    },

    /// Incremental update errors
    IncrementalUpdate {
        /// Error message
        message: String
    },
}

impl fmt::Display for GraphRAGError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphRAGError::Config { message } => {
                write!(f, "Configuration error: {message}. \
                          Solution: Check your config file or use default settings with GraphRAG::builder()")
            }
            GraphRAGError::NotInitialized => {
                write!(
                    f,
                    "GraphRAG not initialized. \
                          Solution: Call .initialize() or use .ask() which auto-initializes"
                )
            }
            GraphRAGError::NoDocuments => {
                write!(f, "No documents added. \
                          Solution: Use .add_document(), .add_document_from_text(), or .from_file() to add content")
            }
            GraphRAGError::Io(err) => {
                write!(
                    f,
                    "I/O error: {err}. \
                          Solution: Check file permissions and that paths exist"
                )
            }
            #[cfg(feature = "ureq")]
            GraphRAGError::Http(err) => {
                write!(
                    f,
                    "HTTP request error: {err}. \
                          Solution: Check network connectivity and service availability"
                )
            }
            #[cfg(not(feature = "ureq"))]
            GraphRAGError::Http(msg) => {
                write!(
                    f,
                    "HTTP request error: {msg}. \
                          Solution: Check network connectivity and service availability"
                )
            }
            GraphRAGError::Json(err) => {
                write!(
                    f,
                    "JSON parsing error: {err}. \
                          Solution: Verify JSON format or use default configuration"
                )
            }
            GraphRAGError::SerdeJson(err) => {
                write!(
                    f,
                    "JSON serialization error: {err}. \
                          Solution: Verify data structure compatibility"
                )
            }
            GraphRAGError::TextProcessing { message } => {
                write!(
                    f,
                    "Text processing error: {message}. \
                          Solution: Check text content and chunk size configuration"
                )
            }
            GraphRAGError::GraphConstruction { message } => {
                write!(
                    f,
                    "Graph construction error: {message}. \
                          Solution: Initialize GraphRAG system and add documents first"
                )
            }
            GraphRAGError::VectorSearch { message } => {
                write!(f, "Vector search error: {message}. \
                          Solution: Ensure embeddings are initialized with .initialize_embeddings()")
            }
            GraphRAGError::EntityExtraction { message } => {
                write!(f, "Entity extraction error: {message}. \
                          Solution: Check entity extraction configuration or use lower confidence threshold")
            }
            GraphRAGError::Retrieval { message } => {
                write!(
                    f,
                    "Retrieval error: {message}. \
                          Solution: Ensure documents are added and graph is built"
                )
            }
            GraphRAGError::Generation { message } => {
                write!(f, "Answer generation error: {message}. \
                          Solution: Check LLM provider configuration or use GraphRAG::builder().auto_detect_llm()")
            }
            GraphRAGError::FunctionCall { message } => {
                write!(f, "Function call error: {message}")
            }
            GraphRAGError::Storage { message } => {
                write!(f, "Storage error: {message}")
            }
            GraphRAGError::Embedding { message } => {
                write!(f, "Embedding error: {message}")
            }
            GraphRAGError::LanguageModel { message } => {
                write!(f, "Language model error: {message}")
            }
            GraphRAGError::Parallel { message } => {
                write!(f, "Parallel processing error: {message}")
            }
            GraphRAGError::Serialization { message } => {
                write!(f, "Serialization error: {message}")
            }
            GraphRAGError::Validation { message } => {
                write!(f, "Validation error: {message}")
            }
            GraphRAGError::Network { message } => {
                write!(f, "Network error: {message}")
            }
            GraphRAGError::Auth { message } => {
                write!(f, "Authentication error: {message}")
            }
            GraphRAGError::NotFound { resource, id } => {
                write!(f, "{resource} not found: {id}")
            }
            GraphRAGError::AlreadyExists { resource, id } => {
                write!(f, "{resource} already exists: {id}")
            }
            GraphRAGError::Timeout {
                operation,
                duration,
            } => {
                write!(f, "Operation '{operation}' timed out after {duration:?}")
            }
            GraphRAGError::ResourceLimit { resource, limit } => {
                write!(f, "Resource limit exceeded for {resource}: {limit}")
            }
            GraphRAGError::DataCorruption { message } => {
                write!(f, "Data corruption detected: {message}")
            }
            GraphRAGError::Unsupported { operation, reason } => {
                write!(f, "Unsupported operation '{operation}': {reason}")
            }
            GraphRAGError::RateLimit { message } => {
                write!(f, "Rate limit error: {message}")
            }
            GraphRAGError::ConflictResolution { message } => {
                write!(f, "Conflict resolution error: {message}")
            }
            GraphRAGError::IncrementalUpdate { message } => {
                write!(f, "Incremental update error: {message}")
            }
        }
    }
}

impl std::error::Error for GraphRAGError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GraphRAGError::Io(err) => Some(err),
            #[cfg(feature = "ureq")]
            GraphRAGError::Http(err) => Some(err.as_ref()),
            #[cfg(not(feature = "ureq"))]
            GraphRAGError::Http(_) => None,
            GraphRAGError::Json(err) => Some(err),
            GraphRAGError::SerdeJson(err) => Some(err),
            _ => None,
        }
    }
}

// Automatic conversions from common error types
impl From<std::io::Error> for GraphRAGError {
    fn from(err: std::io::Error) -> Self {
        GraphRAGError::Io(err)
    }
}

#[cfg(feature = "ureq")]
impl From<ureq::Error> for GraphRAGError {
    fn from(err: ureq::Error) -> Self {
        GraphRAGError::Http(Box::new(err))
    }
}

impl From<json::Error> for GraphRAGError {
    fn from(err: json::Error) -> Self {
        GraphRAGError::Json(err)
    }
}

impl From<serde_json::Error> for GraphRAGError {
    fn from(err: serde_json::Error) -> Self {
        GraphRAGError::SerdeJson(err)
    }
}

// ROGRAG error conversions
#[cfg(feature = "rograg")]
impl From<crate::rograg::logic_form::LogicFormError> for GraphRAGError {
    fn from(err: crate::rograg::logic_form::LogicFormError) -> Self {
        GraphRAGError::Retrieval {
            message: format!("Logic form error: {err}"),
        }
    }
}

#[cfg(feature = "rograg")]
impl From<crate::rograg::processor::ProcessingError> for GraphRAGError {
    fn from(err: crate::rograg::processor::ProcessingError) -> Self {
        GraphRAGError::Generation {
            message: format!("Processing error: {err}"),
        }
    }
}

#[cfg(feature = "rograg")]
impl From<crate::rograg::quality_metrics::MetricsError> for GraphRAGError {
    fn from(err: crate::rograg::quality_metrics::MetricsError) -> Self {
        GraphRAGError::Validation {
            message: format!("Metrics error: {err}"),
        }
    }
}

#[cfg(feature = "rograg")]
impl From<crate::rograg::streaming::StreamingError> for GraphRAGError {
    fn from(err: crate::rograg::streaming::StreamingError) -> Self {
        GraphRAGError::Generation {
            message: format!("Streaming error: {err}"),
        }
    }
}

#[cfg(feature = "rograg")]
impl From<crate::rograg::fuzzy_matcher::FuzzyMatchError> for GraphRAGError {
    fn from(err: crate::rograg::fuzzy_matcher::FuzzyMatchError) -> Self {
        GraphRAGError::Retrieval {
            message: format!("Fuzzy match error: {err}"),
        }
    }
}

/// Convenient Result type alias
pub type Result<T> = std::result::Result<T, GraphRAGError>;

/// Trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context(self, context: &str) -> Result<T>;

    /// Add context using a closure
    fn with_context_lazy<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<GraphRAGError>,
{
    fn with_context(self, context: &str) -> Result<T> {
        self.map_err(|e| {
            let base_error = e.into();
            match base_error {
                GraphRAGError::Config { message } => GraphRAGError::Config {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::TextProcessing { message } => GraphRAGError::TextProcessing {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::GraphConstruction { message } => GraphRAGError::GraphConstruction {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::VectorSearch { message } => GraphRAGError::VectorSearch {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::EntityExtraction { message } => GraphRAGError::EntityExtraction {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Retrieval { message } => GraphRAGError::Retrieval {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Generation { message } => GraphRAGError::Generation {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::FunctionCall { message } => GraphRAGError::FunctionCall {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Storage { message } => GraphRAGError::Storage {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Embedding { message } => GraphRAGError::Embedding {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::LanguageModel { message } => GraphRAGError::LanguageModel {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Parallel { message } => GraphRAGError::Parallel {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Serialization { message } => GraphRAGError::Serialization {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Validation { message } => GraphRAGError::Validation {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Network { message } => GraphRAGError::Network {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::Auth { message } => GraphRAGError::Auth {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::DataCorruption { message } => GraphRAGError::DataCorruption {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::RateLimit { message } => GraphRAGError::RateLimit {
                    message: format!("{context}: {message}"),
                },
                GraphRAGError::ConflictResolution { message } => {
                    GraphRAGError::ConflictResolution {
                        message: format!("{context}: {message}"),
                    }
                }
                GraphRAGError::IncrementalUpdate { message } => GraphRAGError::IncrementalUpdate {
                    message: format!("{context}: {message}"),
                },
                other => other, // For errors that don't have a message field
            }
        })
    }

    fn with_context_lazy<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        match self {
            Ok(value) => Ok(value),
            Err(e) => {
                let context = f();
                Err(e).with_context(&context)
            }
        }
    }
}

/// Helper macros for creating specific error types
///
/// Creates a configuration error with a message
#[macro_export]
macro_rules! config_error {
    ($msg:expr) => {
        $crate::GraphRAGError::Config {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::GraphRAGError::Config {
            message: format!($fmt, $($arg)*),
        }
    };
}

/// Creates a storage error with a message
#[macro_export]
macro_rules! storage_error {
    ($msg:expr) => {
        $crate::GraphRAGError::Storage {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::GraphRAGError::Storage {
            message: format!($fmt, $($arg)*),
        }
    };
}

/// Creates a retrieval error with a message
#[macro_export]
macro_rules! retrieval_error {
    ($msg:expr) => {
        $crate::GraphRAGError::Retrieval {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::GraphRAGError::Retrieval {
            message: format!($fmt, $($arg)*),
        }
    };
}

/// Creates a generation error with a message
#[macro_export]
macro_rules! generation_error {
    ($msg:expr) => {
        $crate::GraphRAGError::Generation {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::GraphRAGError::Generation {
            message: format!($fmt, $($arg)*),
        }
    };
}

/// Error severity levels for logging and monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Informational - not actually an error
    Info,
    /// Warning - something unexpected but recoverable
    Warning,
    /// Error - operation failed but system can continue
    Error,
    /// Critical - system integrity compromised
    Critical,
}

/// Structured suggestion for error recovery
#[derive(Debug, Clone)]
pub struct ErrorSuggestion {
    /// Short description of what to do
    pub action: String,
    /// Example code snippet (if applicable)
    pub code_example: Option<String>,
    /// Link to documentation (if applicable)
    pub doc_link: Option<String>,
}

impl GraphRAGError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            GraphRAGError::Config { .. } => ErrorSeverity::Critical,
            GraphRAGError::NotInitialized => ErrorSeverity::Warning,
            GraphRAGError::NoDocuments => ErrorSeverity::Warning,
            GraphRAGError::Io(_) => ErrorSeverity::Error,
            GraphRAGError::Http(_) => ErrorSeverity::Warning,
            GraphRAGError::Json(_) | GraphRAGError::SerdeJson(_) => ErrorSeverity::Error,
            GraphRAGError::TextProcessing { .. } => ErrorSeverity::Warning,
            GraphRAGError::GraphConstruction { .. } => ErrorSeverity::Error,
            GraphRAGError::VectorSearch { .. } => ErrorSeverity::Warning,
            GraphRAGError::EntityExtraction { .. } => ErrorSeverity::Warning,
            GraphRAGError::Retrieval { .. } => ErrorSeverity::Warning,
            GraphRAGError::Generation { .. } => ErrorSeverity::Warning,
            GraphRAGError::FunctionCall { .. } => ErrorSeverity::Warning,
            GraphRAGError::Storage { .. } => ErrorSeverity::Error,
            GraphRAGError::Embedding { .. } => ErrorSeverity::Warning,
            GraphRAGError::LanguageModel { .. } => ErrorSeverity::Warning,
            GraphRAGError::Parallel { .. } => ErrorSeverity::Error,
            GraphRAGError::Serialization { .. } => ErrorSeverity::Error,
            GraphRAGError::Validation { .. } => ErrorSeverity::Error,
            GraphRAGError::Network { .. } => ErrorSeverity::Warning,
            GraphRAGError::Auth { .. } => ErrorSeverity::Error,
            GraphRAGError::NotFound { .. } => ErrorSeverity::Warning,
            GraphRAGError::AlreadyExists { .. } => ErrorSeverity::Warning,
            GraphRAGError::Timeout { .. } => ErrorSeverity::Warning,
            GraphRAGError::ResourceLimit { .. } => ErrorSeverity::Error,
            GraphRAGError::DataCorruption { .. } => ErrorSeverity::Critical,
            GraphRAGError::Unsupported { .. } => ErrorSeverity::Error,
            GraphRAGError::RateLimit { .. } => ErrorSeverity::Warning,
            GraphRAGError::ConflictResolution { .. } => ErrorSeverity::Error,
            GraphRAGError::IncrementalUpdate { .. } => ErrorSeverity::Error,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self.severity() {
            ErrorSeverity::Info | ErrorSeverity::Warning => true,
            ErrorSeverity::Error => false,
            ErrorSeverity::Critical => false,
        }
    }

    /// Get error category for metrics/monitoring
    pub fn category(&self) -> &'static str {
        match self {
            GraphRAGError::Config { .. } => "config",
            GraphRAGError::NotInitialized => "initialization",
            GraphRAGError::NoDocuments => "usage",
            GraphRAGError::Io(_) => "io",
            GraphRAGError::Http(_) => "http",
            GraphRAGError::Json(_) | GraphRAGError::SerdeJson(_) => "serialization",
            GraphRAGError::TextProcessing { .. } => "text_processing",
            GraphRAGError::GraphConstruction { .. } => "graph",
            GraphRAGError::VectorSearch { .. } => "vector_search",
            GraphRAGError::EntityExtraction { .. } => "entity_extraction",
            GraphRAGError::Retrieval { .. } => "retrieval",
            GraphRAGError::Generation { .. } => "generation",
            GraphRAGError::FunctionCall { .. } => "function_calling",
            GraphRAGError::Storage { .. } => "storage",
            GraphRAGError::Embedding { .. } => "embedding",
            GraphRAGError::LanguageModel { .. } => "language_model",
            GraphRAGError::Parallel { .. } => "parallel",
            GraphRAGError::Serialization { .. } => "serialization",
            GraphRAGError::Validation { .. } => "validation",
            GraphRAGError::Network { .. } => "network",
            GraphRAGError::Auth { .. } => "auth",
            GraphRAGError::NotFound { .. } => "not_found",
            GraphRAGError::AlreadyExists { .. } => "already_exists",
            GraphRAGError::Timeout { .. } => "timeout",
            GraphRAGError::ResourceLimit { .. } => "resource_limit",
            GraphRAGError::DataCorruption { .. } => "data_corruption",
            GraphRAGError::Unsupported { .. } => "unsupported",
            GraphRAGError::RateLimit { .. } => "rate_limit",
            GraphRAGError::ConflictResolution { .. } => "conflict_resolution",
            GraphRAGError::IncrementalUpdate { .. } => "incremental_update",
        }
    }

    /// Get structured suggestions for recovering from this error
    ///
    /// Returns actionable suggestions with optional code examples.
    ///
    /// # Example
    /// ```rust
    /// use graphrag_core::GraphRAGError;
    ///
    /// let error = GraphRAGError::NotInitialized;
    /// let suggestion = error.suggestion();
    /// println!("Action: {}", suggestion.action);
    /// if let Some(code) = suggestion.code_example {
    ///     println!("Example:\n{}", code);
    /// }
    /// ```
    pub fn suggestion(&self) -> ErrorSuggestion {
        match self {
            GraphRAGError::Config { message } => {
                let (action, code) = if message.contains("not found") || message.contains("load") {
                    (
                        "Create a config file or use defaults with Config::default()".to_string(),
                        Some(r#"// Option 1: Use defaults
let config = Config::default();

// Option 2: Load with hierarchy (user -> project -> env)
let config = Config::load()?;

// Option 3: Use TypedBuilder for compile-time safety
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_ollama()
    .build()?;"#.to_string())
                    )
                } else {
                    (
                        "Check configuration values and TOML syntax".to_string(),
                        Some(r#"// Validate config before use
let config = Config::from_toml_file("config.toml")?;

// Or use hierarchical loading with env var overrides
// Set GRAPHRAG_OLLAMA_HOST=localhost to override
let config = Config::load()?;"#.to_string())
                    )
                };
                ErrorSuggestion { action, code_example: code, doc_link: None }
            }
            GraphRAGError::NotInitialized => ErrorSuggestion {
                action: "Initialize the GraphRAG system before querying".to_string(),
                code_example: Some(r#"// Option 1: Manual initialization
let mut graphrag = GraphRAG::new(config)?;
graphrag.initialize()?;

// Option 2: Use quick_start (recommended)
let mut graphrag = GraphRAG::quick_start("Your document text").await?;

// Option 3: Use builder with auto-init
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_ollama()
    .build_and_init()?;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::NoDocuments => ErrorSuggestion {
                action: "Add documents before building the graph or querying".to_string(),
                code_example: Some(r#"// Add text directly
graphrag.add_document_from_text("Your document content here")?;

// Add from file
graphrag.add_document_from_file("document.txt")?;

// Add multiple documents
for file in glob("docs/*.txt")? {
    graphrag.add_document_from_file(file)?;
}

// Then build the graph
graphrag.build_graph().await?;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::Http(_) => ErrorSuggestion {
                action: "Check network connectivity and service availability".to_string(),
                code_example: Some(r#"// Check Ollama is running
// Terminal: curl http://localhost:11434/api/tags

// Verify Ollama config
let config = Config::default();
assert!(config.ollama.enabled);
assert_eq!(config.ollama.host, "localhost");
assert_eq!(config.ollama.port, 11434);

// Or use hash embeddings for offline mode
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_hash_embeddings()  // No network needed
    .build()?;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::LanguageModel { message } => {
                let action = if message.contains("not found") || message.contains("model") {
                    "Ensure the LLM model is pulled and available".to_string()
                } else {
                    "Check LLM provider configuration".to_string()
                };
                ErrorSuggestion {
                    action,
                    code_example: Some(r#"// Pull the model first (in terminal)
// ollama pull llama3.2:latest
// ollama pull nomic-embed-text:latest

// Or specify a different model
let graphrag = GraphRAGBuilder::new()
    .with_ollama_enabled(true)
    .with_chat_model("mistral:latest")
    .with_ollama_embedding_model("nomic-embed-text:latest")
    .build()?;"#.to_string()),
                    doc_link: None,
                }
            }
            GraphRAGError::Embedding { message } => {
                let action = if message.contains("dimension") {
                    "Check embedding dimension matches your model".to_string()
                } else {
                    "Verify embedding provider configuration".to_string()
                };
                ErrorSuggestion {
                    action,
                    code_example: Some(r#"// Use matching dimension for your model
// nomic-embed-text: 768, all-MiniLM-L6-v2: 384
let graphrag = GraphRAGBuilder::new()
    .with_embedding_dimension(768)
    .with_embedding_backend("ollama")
    .build()?;

// Or use hash embeddings (dimension-agnostic)
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_hash_embeddings()
    .build()?;"#.to_string()),
                    doc_link: None,
                }
            }
            GraphRAGError::Retrieval { .. } => ErrorSuggestion {
                action: "Ensure documents are added and graph is built before querying".to_string(),
                code_example: Some(r#"// Full workflow
let mut graphrag = GraphRAG::new(config)?;
graphrag.initialize()?;
graphrag.add_document_from_text("Your content")?;
graphrag.build_graph().await?;

// Now you can query
let answer = graphrag.ask("Your question").await?;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::VectorSearch { .. } => ErrorSuggestion {
                action: "Initialize embeddings and ensure vectors are indexed".to_string(),
                code_example: Some(r#"// Make sure embeddings are configured
let graphrag = GraphRAGBuilder::new()
    .with_embedding_backend("ollama")
    .with_embedding_model("nomic-embed-text:latest")
    .build()?;

// Or check if documents need reindexing
graphrag.build_graph().await?;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::Timeout { operation, duration } => ErrorSuggestion {
                action: format!("Operation '{}' took too long ({:?}). Consider increasing timeout or optimizing.", operation, duration),
                code_example: Some(r#"// Increase timeout in config
let mut config = Config::default();
config.ollama.timeout_seconds = 120;  // 2 minutes

// Or process smaller chunks
config.chunk_size = 500;  // Smaller chunks
config.parallel.enabled = true;  // Parallel processing"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::RateLimit { .. } => ErrorSuggestion {
                action: "You've hit rate limits. Wait and retry, or use local models.".to_string(),
                code_example: Some(r#"// Switch to local Ollama (no rate limits)
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_ollama()  // Local, no API limits
    .build()?;

// Or enable caching to reduce API calls
let mut config = Config::default();
config.caching.enabled = true;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::Storage { .. } => ErrorSuggestion {
                action: "Check file permissions and disk space".to_string(),
                code_example: Some(r#"// Verify output directory exists and is writable
std::fs::create_dir_all("./output")?;

// Use a different output directory
let graphrag = GraphRAGBuilder::new()
    .with_output_dir("/tmp/graphrag_output")
    .build()?;"#.to_string()),
                doc_link: None,
            },
            GraphRAGError::NotFound { resource, id } => ErrorSuggestion {
                action: format!("{} '{}' not found. Verify it exists.", resource, id),
                code_example: None,
                doc_link: None,
            },
            GraphRAGError::AlreadyExists { resource, id } => ErrorSuggestion {
                action: format!("{} '{}' already exists. Use a different ID or update existing.", resource, id),
                code_example: None,
                doc_link: None,
            },
            // Default suggestions for other errors
            _ => ErrorSuggestion {
                action: "Check the error message for details".to_string(),
                code_example: None,
                doc_link: None,
            },
        }
    }

    /// Get a formatted error message with suggestion
    ///
    /// Returns a user-friendly error message including the suggestion.
    pub fn display_with_suggestion(&self) -> String {
        let suggestion = self.suggestion();
        let mut output = format!("{}\n\n💡 Suggestion: {}", self, suggestion.action);
        if let Some(code) = &suggestion.code_example {
            output.push_str(&format!("\n\nExample:\n```rust\n{}\n```", code));
        }
        output
    }
}

impl From<regex::Error> for GraphRAGError {
    fn from(err: regex::Error) -> Self {
        GraphRAGError::Validation {
            message: format!("Regex error: {err}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = GraphRAGError::Config {
            message: "Invalid configuration".to_string(),
        };
        assert_eq!(
            format!("{error}"),
            "Configuration error: Invalid configuration. Solution: Check your config file or use default settings with GraphRAG::builder()"
        );
    }

    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));

        let error = result.with_context("loading configuration").unwrap_err();
        assert!(matches!(error, GraphRAGError::Io(_)));
    }

    #[test]
    fn test_error_macros() {
        let error = config_error!("test message");
        assert!(matches!(error, GraphRAGError::Config { .. }));

        let error = storage_error!("test {} {}", "formatted", "message");
        assert!(matches!(error, GraphRAGError::Storage { .. }));
    }

    #[test]
    fn test_error_severity() {
        let config_error = GraphRAGError::Config {
            message: "test".to_string(),
        };
        assert_eq!(config_error.severity(), ErrorSeverity::Critical);
        assert!(!config_error.is_recoverable());

        let warning_error = GraphRAGError::Retrieval {
            message: "test".to_string(),
        };
        assert_eq!(warning_error.severity(), ErrorSeverity::Warning);
        assert!(warning_error.is_recoverable());
    }

    #[test]
    fn test_error_suggestion_not_initialized() {
        let error = GraphRAGError::NotInitialized;
        let suggestion = error.suggestion();

        assert!(suggestion.action.contains("Initialize"));
        assert!(suggestion.code_example.is_some());
        let code = suggestion.code_example.unwrap();
        assert!(code.contains("initialize()") || code.contains("quick_start"));
    }

    #[test]
    fn test_error_suggestion_no_documents() {
        let error = GraphRAGError::NoDocuments;
        let suggestion = error.suggestion();

        assert!(suggestion.action.contains("Add documents"));
        assert!(suggestion.code_example.is_some());
        let code = suggestion.code_example.unwrap();
        assert!(code.contains("add_document"));
    }

    #[test]
    fn test_error_suggestion_config() {
        let error = GraphRAGError::Config {
            message: "File not found".to_string(),
        };
        let suggestion = error.suggestion();

        assert!(suggestion.code_example.is_some());
        let code = suggestion.code_example.unwrap();
        assert!(code.contains("Config::default()") || code.contains("Config::load"));
    }

    #[test]
    fn test_error_suggestion_not_found() {
        let error = GraphRAGError::NotFound {
            resource: "Document".to_string(),
            id: "test-123".to_string(),
        };
        let suggestion = error.suggestion();

        assert!(suggestion.action.contains("Document"));
        assert!(suggestion.action.contains("test-123"));
    }

    #[test]
    fn test_display_with_suggestion() {
        let error = GraphRAGError::NotInitialized;
        let display = error.display_with_suggestion();

        assert!(display.contains("not initialized"));
        assert!(display.contains("Suggestion:"));
        assert!(display.contains("```rust"));
    }
}
