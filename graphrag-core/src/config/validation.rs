use crate::config::{Config, SetConfig};
use crate::{GraphRAGError, Result};
use std::path::Path;

/// Validate the endpoint shape shared by the optional HTTP providers.
///
/// This intentionally validates only the URL parts that are meaningful to a
/// local provider (scheme, authority and an optional path). DNS resolution is
/// left to the provider at initialization/request time.
fn valid_http_endpoint(endpoint: &str) -> bool {
    let Some(rest) = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
    else {
        return false;
    };
    if rest.is_empty() || rest.chars().any(char::is_whitespace) {
        return false;
    }
    let authority = rest
        .split_once('/')
        .map_or(rest, |(authority, _)| authority);
    if authority.is_empty() || authority.chars().any(|ch| matches!(ch, '@' | '?' | '#')) {
        return false;
    }
    if authority.starts_with('[') {
        let Some(end) = authority.find(']') else {
            return false;
        };
        if end <= 1 {
            return false;
        }
        let port = &authority[end + 1..];
        return port.is_empty() || (port.starts_with(':') && port[1..].parse::<u16>().is_ok());
    }
    match authority.split_once(':') {
        Some((host, port)) => !host.is_empty() && port.parse::<u16>().is_ok(),
        None => true,
    }
}

/// Result of configuration validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the configuration is valid
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// List of validation warnings
    pub warnings: Vec<String>,
    /// List of optimization suggestions
    pub suggestions: Vec<String>,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error and mark validation as failed
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a warning (doesn't affect validity)
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Add an optimization suggestion
    pub fn add_suggestion(&mut self, suggestion: String) {
        self.suggestions.push(suggestion);
    }
}

/// Trait for configuration validation
pub trait Validatable {
    /// Validate configuration with standard checks
    fn validate(&self) -> ValidationResult;
    /// Validate configuration with strict checks (includes warnings and suggestions)
    fn validate_strict(&self) -> ValidationResult;
}

impl Validatable for Config {
    fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate output directory
        if self.output_dir.is_empty() {
            result.add_error("Output directory cannot be empty".to_string());
        }

        if self.embeddings.dimension == 0 {
            result.add_error("Embedding dimension must be greater than 0".to_string());
        }
        let embedding_backend = self.embeddings.backend.to_ascii_lowercase();
        if embedding_backend == "llama_cpp" || embedding_backend == "llama.cpp" {
            if self.embeddings.dimension != 384 {
                result.add_error(format!(
                    "llama.cpp multilingual-e5-small embeddings require dimension 384, got {}",
                    self.embeddings.dimension
                ));
            }
            if self
                .embeddings
                .api_endpoint
                .as_deref()
                .unwrap_or_default()
                .trim()
                .is_empty()
            {
                result
                    .add_error("llama.cpp embeddings require embeddings.api_endpoint".to_string());
            } else if !valid_http_endpoint(self.embeddings.api_endpoint.as_deref().unwrap()) {
                result.add_error(
                    "llama.cpp embeddings api_endpoint must be a valid http or https URL"
                        .to_string(),
                );
            }
            if self
                .embeddings
                .model
                .as_deref()
                .unwrap_or_default()
                .trim()
                .is_empty()
            {
                result.add_error("llama.cpp embedding model must not be empty".to_string());
            }
            if self.embeddings.fallback_to_hash {
                result.add_error(
                    "llama.cpp embeddings cannot fall back to hash embeddings".to_string(),
                );
            }
        }
        if self.llm.enabled {
            if !valid_http_endpoint(&self.llm.base_url) {
                result.add_error("llm.base_url must be a valid http or https URL".to_string());
            }
            if self.llm.chat_model.trim().is_empty() {
                result.add_error("llm.chat_model must not be empty".to_string());
            }
            if self.llm.timeout_seconds == 0 {
                result.add_error("llm.timeout_seconds must be greater than 0".to_string());
            }
            if self
                .llm
                .api_key_env
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                result.add_error("llm.api_key_env must name an environment variable".to_string());
            }
        }
        if self.graph.max_connections == 0 {
            result.add_error("Graph max_connections must be greater than 0".to_string());
        }
        for (name, threshold) in [
            (
                "Graph similarity threshold",
                self.graph.similarity_threshold,
            ),
            (
                "Graph relationship confidence threshold",
                self.graph.relationship_confidence_threshold,
            ),
            (
                "Graph traversal minimum relationship strength",
                self.graph.traversal.min_relationship_strength,
            ),
            ("Entity minimum confidence", self.entities.min_confidence),
            (
                "Entity validation minimum confidence",
                self.entities.validation_min_confidence,
            ),
        ] {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                result.add_error(format!("{name} must be finite and between 0.0 and 1.0"));
            }
        }

        // Validate chunk size
        if self.text.chunk_size == 0 {
            result.add_error("Chunk size must be greater than 0".to_string());
        } else if self.text.chunk_size < 100 {
            result.add_warning(
                "Chunk size is very small (<100), this may affect performance".to_string(),
            );
        } else if self.text.chunk_size > 10000 {
            result.add_warning(
                "Chunk size is very large (>10000), this may affect quality".to_string(),
            );
        } else {
            // Chunk size is in acceptable range (100-10000)
        }

        // Validate chunk overlap
        if self.text.chunk_overlap >= self.text.chunk_size {
            result.add_error("Chunk overlap must be less than chunk size".to_string());
        } else if self.text.chunk_overlap > self.text.chunk_size / 2 {
            result.add_warning(
                "Chunk overlap is more than 50% of chunk size, this may be inefficient".to_string(),
            );
        } else {
            // Chunk overlap is in acceptable range
        }

        // Validate entity extraction settings
        if let Some(max_entities) = self.max_entities_per_chunk {
            if max_entities == 0 {
                result.add_error("Max entities per chunk must be greater than 0".to_string());
            } else if max_entities > 100 {
                result.add_warning("Max entities per chunk is very high (>100)".to_string());
            } else {
                // Max entities is in acceptable range
            }
        }

        // Validate retrieval settings
        if self.retrieval.top_k == 0 {
            result.add_error("Top-k results must be greater than 0".to_string());
        } else if self.retrieval.top_k > 100 {
            result.add_warning(
                "Top-k results is very high (>100), this may affect performance".to_string(),
            );
        }

        // Validate similarity threshold
        if let Some(threshold) = self.similarity_threshold {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                result.add_error("Similarity threshold must be between 0.0 and 1.0".to_string());
            } else if threshold < 0.1 {
                result.add_warning(
                    "Similarity threshold is very low (<0.1), this may return irrelevant results"
                        .to_string(),
                );
            } else if threshold > 0.9 {
                result.add_warning(
                    "Similarity threshold is very high (>0.9), this may return too few results"
                        .to_string(),
                );
            } else {
                // Similarity threshold is in acceptable range (0.1-0.9)
            }
        }

        if self.graph.traversal.max_depth == 0 {
            result.add_error("Graph traversal max_depth must be greater than 0".to_string());
        }
        if self.graph.traversal.max_paths == 0 {
            result.add_error("Graph traversal max_paths must be greater than 0".to_string());
        }

        // Add suggestions based on configuration
        if self.text.chunk_size > 1000 && self.text.chunk_overlap < 100 {
            result.add_suggestion("Consider increasing chunk overlap for better context preservation with large chunks".to_string());
        }

        result
    }

    fn validate_strict(&self) -> ValidationResult {
        let mut result = self.validate();

        // Additional strict validations

        // Ensure all paths exist
        let output_path = Path::new(&self.output_dir);
        if !output_path.exists() {
            result.add_warning(format!(
                "Output directory does not exist: {}",
                self.output_dir
            ));
            result.add_suggestion("Directory will be created automatically".to_string());
        }

        // Validate feature compatibility
        #[cfg(not(feature = "llm"))]
        {
            result.add_warning("`llm` feature is not enabled, LLM support unavailable".to_string());
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            result.add_warning(
                "Parallel processing is not enabled, performance may be reduced".to_string(),
            );
        }

        // Check for optimal settings
        let optimal_chunk_size = 800;
        let optimal_overlap = 200;

        if self.text.chunk_size.abs_diff(optimal_chunk_size as usize) > 300 {
            result.add_suggestion(format!(
                "Consider using chunk size around {optimal_chunk_size} for optimal performance"
            ));
        }

        if self.text.chunk_overlap.abs_diff(optimal_overlap as usize) > 100 {
            result.add_suggestion(format!(
                "Consider using chunk overlap around {optimal_overlap} for optimal context preservation"
            ));
        }

        result
    }
}

/// Validate pipeline approach configuration (semantic/algorithmic/hybrid)
fn validate_pipeline_approach(config: &SetConfig, result: &mut ValidationResult) {
    let approach = &config.mode.approach;

    // Validate approach value
    match approach.as_str() {
        "semantic" | "algorithmic" | "hybrid" => {},
        invalid => {
            result.add_error(format!(
                "Invalid pipeline approach: '{invalid}'. Must be 'semantic', 'algorithmic', or 'hybrid'"
            ));
            return;
        },
    }

    // Validate semantic pipeline
    if approach == "semantic" {
        match &config.semantic {
            None => {
                // The converter supplies the documented semantic defaults.
            },
            Some(semantic) => {
                if !semantic.enabled {
                    result.add_error(
                        "Semantic pipeline approach selected but semantic.enabled = false"
                            .to_string(),
                    );
                }

                // Validate semantic embeddings
                let valid_backends = [
                    "huggingface",
                    "openai",
                    "voyage",
                    "cohere",
                    "jina",
                    "mistral",
                    "together",
                    "llama_cpp",
                ];
                if !valid_backends.contains(&semantic.embeddings.backend.as_str()) {
                    result.add_error(format!(
                        "Invalid semantic embedding backend: '{}'. Must be one of: {}",
                        semantic.embeddings.backend,
                        valid_backends.join(", ")
                    ));
                }

                if semantic.embeddings.dimension == 0 {
                    result.add_error(
                        "Semantic embedding dimension must be greater than 0".to_string(),
                    );
                }

                // Validate semantic entity extraction
                if semantic.entity_extraction.confidence_threshold < 0.0
                    || semantic.entity_extraction.confidence_threshold > 1.0
                {
                    result.add_error("Semantic entity extraction confidence threshold must be between 0.0 and 1.0".to_string());
                }

                if semantic.entity_extraction.temperature < 0.0
                    || semantic.entity_extraction.temperature > 2.0
                {
                    result.add_error(
                        "Semantic entity extraction temperature must be between 0.0 and 2.0"
                            .to_string(),
                    );
                }

                // Validate semantic retrieval
                if semantic.retrieval.similarity_threshold < 0.0
                    || semantic.retrieval.similarity_threshold > 1.0
                {
                    result.add_error(
                        "Semantic retrieval similarity threshold must be between 0.0 and 1.0"
                            .to_string(),
                    );
                }

                if semantic.retrieval.top_k == 0 {
                    result.add_error("Semantic retrieval top_k must be greater than 0".to_string());
                }
            },
        }
    }

    // Validate algorithmic pipeline
    if approach == "algorithmic" {
        match &config.algorithmic {
            None => {
                // The converter supplies the documented algorithmic defaults.
            },
            Some(algorithmic) => {
                if !algorithmic.enabled {
                    result.add_error(
                        "Algorithmic pipeline approach selected but algorithmic.enabled = false"
                            .to_string(),
                    );
                }

                // Validate algorithmic embeddings
                if algorithmic.embeddings.backend != "hash" {
                    result.add_warning(format!(
                        "Algorithmic pipeline typically uses 'hash' backend, but '{}' is configured",
                        algorithmic.embeddings.backend
                    ));
                }

                if algorithmic.embeddings.dimension == 0 {
                    result.add_error(
                        "Algorithmic embedding dimension must be greater than 0".to_string(),
                    );
                }

                if algorithmic.embeddings.max_document_frequency < 0.0
                    || algorithmic.embeddings.max_document_frequency > 1.0
                {
                    result.add_error(
                        "Algorithmic max_document_frequency must be between 0.0 and 1.0"
                            .to_string(),
                    );
                }

                // Validate algorithmic entity extraction
                if algorithmic.entity_extraction.confidence_threshold < 0.0
                    || algorithmic.entity_extraction.confidence_threshold > 1.0
                {
                    result.add_error("Algorithmic entity extraction confidence threshold must be between 0.0 and 1.0".to_string());
                }

                // Validate algorithmic retrieval (BM25 parameters)
                if algorithmic.retrieval.k1 < 0.0 {
                    result.add_error("BM25 k1 parameter must be non-negative".to_string());
                }

                if algorithmic.retrieval.b < 0.0 || algorithmic.retrieval.b > 1.0 {
                    result.add_error("BM25 b parameter must be between 0.0 and 1.0".to_string());
                }

                if algorithmic.retrieval.top_k == 0 {
                    result.add_error(
                        "Algorithmic retrieval top_k must be greater than 0".to_string(),
                    );
                }
            },
        }
    }

    // Validate hybrid pipeline
    if approach == "hybrid" {
        match &config.hybrid {
            None => {
                // The converter supplies the documented hybrid defaults.
            },
            Some(hybrid) => {
                if !hybrid.enabled {
                    result.add_error(
                        "Hybrid pipeline approach selected but hybrid.enabled = false".to_string(),
                    );
                }

                // Validate hybrid weights
                let weight_sum = hybrid.weights.semantic_weight + hybrid.weights.algorithmic_weight;
                if (weight_sum - 1.0).abs() > 0.01 {
                    result.add_warning(format!(
                        "Hybrid weights should sum to 1.0 (currently: {weight_sum:.2})"
                    ));
                }

                if hybrid.weights.semantic_weight < 0.0 || hybrid.weights.semantic_weight > 1.0 {
                    result.add_error(
                        "Hybrid semantic_weight must be between 0.0 and 1.0".to_string(),
                    );
                }

                if hybrid.weights.algorithmic_weight < 0.0
                    || hybrid.weights.algorithmic_weight > 1.0
                {
                    result.add_error(
                        "Hybrid algorithmic_weight must be between 0.0 and 1.0".to_string(),
                    );
                }

                // Validate hybrid entity extraction weights
                let entity_weight_sum =
                    hybrid.entity_extraction.llm_weight + hybrid.entity_extraction.pattern_weight;
                if (entity_weight_sum - 1.0).abs() > 0.01 {
                    result.add_warning(format!(
                        "Hybrid entity extraction weights should sum to 1.0 (currently: {entity_weight_sum:.2})"
                    ));
                }

                // Validate hybrid retrieval weights
                let retrieval_weight_sum =
                    hybrid.retrieval.vector_weight + hybrid.retrieval.bm25_weight;
                if (retrieval_weight_sum - 1.0).abs() > 0.01 {
                    result.add_warning(format!(
                        "Hybrid retrieval weights should sum to 1.0 (currently: {retrieval_weight_sum:.2})"
                    ));
                }

                if hybrid.retrieval.rrf_constant == 0 {
                    result.add_error(
                        "Hybrid RRF constant must be greater than 0 (typically 60)".to_string(),
                    );
                }

                // Validate confidence boost
                if hybrid.entity_extraction.confidence_boost < 0.0
                    || hybrid.entity_extraction.confidence_boost > 1.0
                {
                    result.add_warning(
                        "Hybrid confidence_boost should typically be between 0.0 and 1.0"
                            .to_string(),
                    );
                }
            },
        }
    }

    // Add suggestions based on approach
    match approach.as_str() {
        "semantic" => {
            result.add_suggestion("Semantic pipeline uses neural embeddings and LLM-based extraction for high-quality results".to_string());
            if config.llm.enabled {
                result.add_suggestion(
                    "Consider using 'llama3.1:8b' for entity extraction with gleaning enabled"
                        .to_string(),
                );
            }
        },
        "algorithmic" => {
            result.add_suggestion("Algorithmic pipeline uses pattern matching and TF-IDF for fast, resource-efficient processing".to_string());
            result.add_suggestion("Algorithmic pipeline works well for structured documents and doesn't require an LLM".to_string());
        },
        "hybrid" => {
            result.add_suggestion("Hybrid pipeline combines semantic and algorithmic approaches for balanced quality and performance".to_string());
            result.add_suggestion(
                "Fine-tune hybrid weights based on your specific use case and evaluation metrics"
                    .to_string(),
            );
        },
        _ => {},
    }
}

impl Validatable for SetConfig {
    fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate pipeline approach configuration
        validate_pipeline_approach(self, &mut result);

        if self.general.output_dir.is_empty() {
            result.add_error("Output directory cannot be empty".to_string());
        }

        // Validate the canonical runtime settings as well as source-specific sections.
        let canonical = self.to_graphrag_config_unchecked();
        for error in canonical.validate().errors {
            if !result.errors.contains(&error) {
                result.add_error(error);
            }
        }

        // Base URL / timeout are validated on the canonical config above.
        let llm = &self.llm;
        if llm.enabled && llm.embedding_model.trim().is_empty() {
            result.add_error("llm.embedding_model must be specified".to_string());
        }

        // Validate storage settings
        let storage = &self.storage;
        match storage.database_type.as_str() {
            "memory" | "file" | "sqlite" | "postgresql" | "neo4j" => {},
            db_type => {
                result.add_error(format!("Unknown database type: {db_type}"));
                result.add_suggestion(
                    "Supported types: memory, file, sqlite, postgresql, neo4j".to_string(),
                );
            },
        }

        result
    }

    fn validate_strict(&self) -> ValidationResult {
        let mut result = self.validate();

        // Additional strict checks
        if !self.llm.enabled {
            result.add_warning("LLM is not enabled, will use mock LLM".to_string());
        }

        result
    }
}

/// Check a document path when a command actually needs to read a document.
/// Configuration validation intentionally does not require an input document.
pub fn validate_document_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() {
        return Err(GraphRAGError::Config {
            message: "Input document path cannot be empty".to_string(),
        });
    }
    if !path.exists() {
        return Err(GraphRAGError::Config {
            message: format!("Input document not found: {}", path.display()),
        });
    }
    if !path.is_file() {
        return Err(GraphRAGError::Config {
            message: format!("Input path is not a file: {}", path.display()),
        });
    }
    Ok(())
}

/// Load and validate a configuration file using the same format dispatcher as
/// the CLI and server. File readability/parsing errors are returned separately
/// from the semantic [`ValidationResult`].
pub fn validate_config_file(path: &Path, strict: bool) -> Result<ValidationResult> {
    if !path.exists() {
        return Err(GraphRAGError::Config {
            message: format!("Configuration file not found: {}", path.display()),
        });
    }
    if !path.is_file() {
        return Err(GraphRAGError::Config {
            message: format!("Configuration path is not a file: {}", path.display()),
        });
    }
    let config = crate::config::loader::load_config(path.to_str().ok_or_else(|| {
        GraphRAGError::Config {
            message: "Configuration path is not valid UTF-8".to_string(),
        }
    })?)?;

    Ok(if strict {
        config.validate_strict()
    } else {
        config.validate()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TextConfig;

    #[test]
    fn default_validation_result_is_valid_and_empty() {
        let result = ValidationResult::default();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_config_validation() {
        let config = Config {
            text: TextConfig {
                chunk_size: 0,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = config.validate();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_chunk_overlap_validation() {
        let config = Config {
            text: TextConfig {
                chunk_size: 100,
                chunk_overlap: 150,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = config.validate();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("overlap")));
    }

    #[test]
    fn test_suggestions() {
        let config = Config {
            text: TextConfig {
                chunk_size: 2000,
                chunk_overlap: 50,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = config.validate();
        assert!(!result.suggestions.is_empty());
    }
}
