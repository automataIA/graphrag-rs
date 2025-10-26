use crate::{GraphRAGError, Result};
use crate::config::{Config, SetConfig};
use std::path::Path;

/// Result of configuration validation
#[derive(Debug, Clone, Default)]
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

        // Validate chunk size
        if self.chunk_size == 0 {
            result.add_error("Chunk size must be greater than 0".to_string());
        } else if self.chunk_size < 100 {
            result.add_warning("Chunk size is very small (<100), this may affect performance".to_string());
        } else if self.chunk_size > 10000 {
            result.add_warning("Chunk size is very large (>10000), this may affect quality".to_string());
        } else {
            // Chunk size is in acceptable range (100-10000)
        }

        // Validate chunk overlap
        if self.chunk_overlap >= self.chunk_size {
            result.add_error("Chunk overlap must be less than chunk size".to_string());
        } else if self.chunk_overlap > self.chunk_size / 2 {
            result.add_warning("Chunk overlap is more than 50% of chunk size, this may be inefficient".to_string());
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
        if let Some(top_k) = self.top_k_results {
            if top_k == 0 {
                result.add_error("Top-k results must be greater than 0".to_string());
            } else if top_k > 100 {
                result.add_warning("Top-k results is very high (>100), this may affect performance".to_string());
            } else {
                // Top-k is in acceptable range
            }
        }

        // Validate similarity threshold
        if let Some(threshold) = self.similarity_threshold {
            if !(0.0..=1.0).contains(&threshold) {
                result.add_error("Similarity threshold must be between 0.0 and 1.0".to_string());
            } else if threshold < 0.1 {
                result.add_warning("Similarity threshold is very low (<0.1), this may return irrelevant results".to_string());
            } else if threshold > 0.9 {
                result.add_warning("Similarity threshold is very high (>0.9), this may return too few results".to_string());
            } else {
                // Similarity threshold is in acceptable range (0.1-0.9)
            }
        }

        // Add suggestions based on configuration
        if self.chunk_size > 1000 && self.chunk_overlap < 100 {
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
            result.add_warning(format!("Output directory does not exist: {}", self.output_dir));
            result.add_suggestion("Directory will be created automatically".to_string());
        }

        // Validate feature compatibility
        #[cfg(not(feature = "ollama"))]
        {
            result.add_warning("Ollama feature is not enabled, local LLM support unavailable".to_string());
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            result.add_warning("Parallel processing is not enabled, performance may be reduced".to_string());
        }

        // Check for optimal settings
        let optimal_chunk_size = 800;
        let optimal_overlap = 200;

        if (self.chunk_size as i32 - optimal_chunk_size).abs() > 300 {
            result.add_suggestion(format!(
                "Consider using chunk size around {} for optimal performance",
                optimal_chunk_size
            ));
        }

        if (self.chunk_overlap as i32 - optimal_overlap).abs() > 100 {
            result.add_suggestion(format!(
                "Consider using chunk overlap around {} for optimal context preservation",
                optimal_overlap
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
            result.add_error(format!("Invalid pipeline approach: '{}'. Must be 'semantic', 'algorithmic', or 'hybrid'", invalid));
            return;
        }
    }

    // Validate semantic pipeline
    if approach == "semantic" {
        match &config.semantic {
            None => {
                result.add_error("Semantic pipeline approach selected but [semantic] configuration is missing".to_string());
            }
            Some(semantic) => {
                if !semantic.enabled {
                    result.add_error("Semantic pipeline approach selected but semantic.enabled = false".to_string());
                }

                // Validate semantic embeddings
                let valid_backends = ["huggingface", "openai", "voyage", "cohere", "jina", "mistral", "together", "ollama"];
                if !valid_backends.contains(&semantic.embeddings.backend.as_str()) {
                    result.add_error(format!(
                        "Invalid semantic embedding backend: '{}'. Must be one of: {}",
                        semantic.embeddings.backend,
                        valid_backends.join(", ")
                    ));
                }

                if semantic.embeddings.dimension == 0 {
                    result.add_error("Semantic embedding dimension must be greater than 0".to_string());
                }

                // Validate semantic entity extraction
                if semantic.entity_extraction.confidence_threshold < 0.0 || semantic.entity_extraction.confidence_threshold > 1.0 {
                    result.add_error("Semantic entity extraction confidence threshold must be between 0.0 and 1.0".to_string());
                }

                if semantic.entity_extraction.temperature < 0.0 || semantic.entity_extraction.temperature > 2.0 {
                    result.add_error("Semantic entity extraction temperature must be between 0.0 and 2.0".to_string());
                }

                // Validate semantic retrieval
                if semantic.retrieval.similarity_threshold < 0.0 || semantic.retrieval.similarity_threshold > 1.0 {
                    result.add_error("Semantic retrieval similarity threshold must be between 0.0 and 1.0".to_string());
                }

                if semantic.retrieval.top_k == 0 {
                    result.add_error("Semantic retrieval top_k must be greater than 0".to_string());
                }
            }
        }
    }

    // Validate algorithmic pipeline
    if approach == "algorithmic" {
        match &config.algorithmic {
            None => {
                result.add_error("Algorithmic pipeline approach selected but [algorithmic] configuration is missing".to_string());
            }
            Some(algorithmic) => {
                if !algorithmic.enabled {
                    result.add_error("Algorithmic pipeline approach selected but algorithmic.enabled = false".to_string());
                }

                // Validate algorithmic embeddings
                if algorithmic.embeddings.backend != "hash" {
                    result.add_warning(format!(
                        "Algorithmic pipeline typically uses 'hash' backend, but '{}' is configured",
                        algorithmic.embeddings.backend
                    ));
                }

                if algorithmic.embeddings.dimension == 0 {
                    result.add_error("Algorithmic embedding dimension must be greater than 0".to_string());
                }

                if algorithmic.embeddings.max_document_frequency < 0.0 || algorithmic.embeddings.max_document_frequency > 1.0 {
                    result.add_error("Algorithmic max_document_frequency must be between 0.0 and 1.0".to_string());
                }

                // Validate algorithmic entity extraction
                if algorithmic.entity_extraction.confidence_threshold < 0.0 || algorithmic.entity_extraction.confidence_threshold > 1.0 {
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
                    result.add_error("Algorithmic retrieval top_k must be greater than 0".to_string());
                }
            }
        }
    }

    // Validate hybrid pipeline
    if approach == "hybrid" {
        match &config.hybrid {
            None => {
                result.add_error("Hybrid pipeline approach selected but [hybrid] configuration is missing".to_string());
            }
            Some(hybrid) => {
                if !hybrid.enabled {
                    result.add_error("Hybrid pipeline approach selected but hybrid.enabled = false".to_string());
                }

                // Validate hybrid weights
                let weight_sum = hybrid.weights.semantic_weight + hybrid.weights.algorithmic_weight;
                if (weight_sum - 1.0).abs() > 0.01 {
                    result.add_warning(format!(
                        "Hybrid weights should sum to 1.0 (currently: {:.2})",
                        weight_sum
                    ));
                }

                if hybrid.weights.semantic_weight < 0.0 || hybrid.weights.semantic_weight > 1.0 {
                    result.add_error("Hybrid semantic_weight must be between 0.0 and 1.0".to_string());
                }

                if hybrid.weights.algorithmic_weight < 0.0 || hybrid.weights.algorithmic_weight > 1.0 {
                    result.add_error("Hybrid algorithmic_weight must be between 0.0 and 1.0".to_string());
                }

                // Validate hybrid entity extraction weights
                let entity_weight_sum = hybrid.entity_extraction.llm_weight + hybrid.entity_extraction.pattern_weight;
                if (entity_weight_sum - 1.0).abs() > 0.01 {
                    result.add_warning(format!(
                        "Hybrid entity extraction weights should sum to 1.0 (currently: {:.2})",
                        entity_weight_sum
                    ));
                }

                // Validate hybrid retrieval weights
                let retrieval_weight_sum = hybrid.retrieval.vector_weight + hybrid.retrieval.bm25_weight;
                if (retrieval_weight_sum - 1.0).abs() > 0.01 {
                    result.add_warning(format!(
                        "Hybrid retrieval weights should sum to 1.0 (currently: {:.2})",
                        retrieval_weight_sum
                    ));
                }

                if hybrid.retrieval.rrf_constant == 0 {
                    result.add_error("Hybrid RRF constant must be greater than 0 (typically 60)".to_string());
                }

                // Validate confidence boost
                if hybrid.entity_extraction.confidence_boost < 0.0 || hybrid.entity_extraction.confidence_boost > 1.0 {
                    result.add_warning("Hybrid confidence_boost should typically be between 0.0 and 1.0".to_string());
                }
            }
        }
    }

    // Add suggestions based on approach
    match approach.as_str() {
        "semantic" => {
            result.add_suggestion("Semantic pipeline uses neural embeddings and LLM-based extraction for high-quality results".to_string());
            if config.ollama.enabled {
                result.add_suggestion("Consider using 'llama3.1:8b' for entity extraction with gleaning enabled".to_string());
            }
        }
        "algorithmic" => {
            result.add_suggestion("Algorithmic pipeline uses pattern matching and TF-IDF for fast, resource-efficient processing".to_string());
            result.add_suggestion("Algorithmic pipeline works well for structured documents and doesn't require an LLM".to_string());
        }
        "hybrid" => {
            result.add_suggestion("Hybrid pipeline combines semantic and algorithmic approaches for balanced quality and performance".to_string());
            result.add_suggestion("Fine-tune hybrid weights based on your specific use case and evaluation metrics".to_string());
        }
        _ => {}
    }
}

impl Validatable for SetConfig {
    fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate pipeline approach configuration
        validate_pipeline_approach(self, &mut result);

        // Validate general settings
        if let Some(input_path) = &self.general.input_document_path {
            if input_path.is_empty() {
                result.add_error("Input document path cannot be empty".to_string());
            } else {
                let path = Path::new(input_path);
                if !path.exists() {
                    result.add_error(format!("Input document not found: {}", input_path));
                } else if !path.is_file() {
                    result.add_error(format!("Input path is not a file: {}", input_path));
                } else {
                    // Input path exists and is a valid file
                }
            }
        } else {
            result.add_error("Input document path is required".to_string());
        }

        if self.general.output_dir.is_empty() {
            result.add_error("Output directory cannot be empty".to_string());
        }

        // Validate pipeline settings
        let pipeline = &self.pipeline;
        if pipeline.text_extraction.chunk_size == 0 {
            result.add_error("Chunk size must be greater than 0".to_string());
        }

        if pipeline.text_extraction.chunk_overlap >= pipeline.text_extraction.chunk_size {
            result.add_error("Chunk overlap must be less than chunk size".to_string());
        }

        // Validate Ollama settings if enabled
        let ollama = &self.ollama;
        if ollama.enabled {
            if ollama.host.is_empty() {
                result.add_error("Ollama host cannot be empty when enabled".to_string());
            }

            if ollama.port == 0 {
                result.add_error("Ollama port must be valid".to_string());
            }

            if ollama.chat_model.is_empty() {
                result.add_error("Ollama chat model must be specified".to_string());
            }

            if ollama.embedding_model.is_empty() {
                result.add_error("Ollama embedding model must be specified".to_string());
            }

            // Suggest common models if using defaults
            if ollama.chat_model == "llama2" {
                result.add_suggestion("Consider using 'llama3.1:8b' for better performance".to_string());
            }
        }

        // Validate storage settings
        let storage = &self.storage;
        match storage.database_type.as_str() {
            "memory" | "file" | "sqlite" | "postgresql" | "neo4j" => {},
            db_type => {
                result.add_error(format!("Unknown database type: {}", db_type));
                result.add_suggestion("Supported types: memory, file, sqlite, postgresql, neo4j".to_string());
            }
        }

        result
    }

    fn validate_strict(&self) -> ValidationResult {
        let mut result = self.validate();

        // Additional strict checks
        if !self.ollama.enabled {
            result.add_warning("Ollama is not enabled, will use mock LLM".to_string());
        }

        result
    }
}

/// Validate a TOML configuration file
pub fn validate_config_file(path: &Path, strict: bool) -> Result<ValidationResult> {
    let config_str = std::fs::read_to_string(path)?;
    let config: SetConfig = toml::from_str(&config_str)
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse TOML config: {}", e)
        })?;

    let result = if strict {
        config.validate_strict()
    } else {
        config.validate()
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = Config {
            chunk_size: 0,
            ..Default::default()
        };

        let result = config.validate();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_chunk_overlap_validation() {
        let config = Config {
            chunk_size: 100,
            chunk_overlap: 150,
            ..Default::default()
        };

        let result = config.validate();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("overlap")));
    }

    #[test]
    fn test_suggestions() {
        let config = Config {
            chunk_size: 2000,
            chunk_overlap: 50,
            ..Default::default()
        };

        let result = config.validate();
        assert!(!result.suggestions.is_empty());
    }
}