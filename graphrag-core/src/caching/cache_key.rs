//! Intelligent cache key generation for optimal hit rates

use super::CacheResult;
use crate::core::traits::GenerationParams;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// A cache key that uniquely identifies a request-response pair
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CacheKey {
    /// The hashed key for fast lookup
    pub key_hash: String,
    /// Human-readable representation for debugging
    pub display_key: String,
    /// Optional metadata for the key
    pub metadata: HashMap<String, String>,
}

impl std::hash::Hash for CacheKey {
    /// Compute the hash of this cache key based on key_hash and display_key
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key_hash.hash(state);
        self.display_key.hash(state);
        // Don't hash metadata as it's not part of the key identity
    }
}

impl CacheKey {
    /// Create a new cache key from a hash and display string
    pub fn new(key_hash: String, display_key: String) -> Self {
        Self {
            key_hash,
            display_key,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the cache key
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the hash for cache lookup
    pub fn hash(&self) -> &str {
        &self.key_hash
    }

    /// Get human-readable display format
    pub fn display(&self) -> &str {
        &self.display_key
    }
}

/// Cache key generation strategies
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KeyStrategy {
    /// Simple prompt-only hashing
    Simple,
    /// Include generation parameters in the key
    WithParameters,
    /// Semantic-aware key generation (normalize whitespace, case, etc.)
    Semantic,
    /// Content-based hashing with advanced normalization
    ContentBased,
}

impl Default for KeyStrategy {
    /// Returns the default key strategy (ContentBased)
    fn default() -> Self {
        Self::ContentBased
    }
}

/// Type alias for custom normalizer functions
type CustomNormalizer = Box<dyn Fn(&str) -> String + Send + Sync>;

/// Cache key generator with configurable strategies
pub struct CacheKeyGenerator {
    strategy: KeyStrategy,
    normalize_whitespace: bool,
    ignore_case: bool,
    include_model_info: bool,

    custom_normalizers: Vec<CustomNormalizer>,
}

impl CacheKeyGenerator {
    /// Create a new cache key generator with default settings
    pub fn new() -> Self {
        Self {
            strategy: KeyStrategy::default(),
            normalize_whitespace: true,
            ignore_case: false,
            include_model_info: true,
            custom_normalizers: Vec::new(),
        }
    }

    /// Create a generator with a specific strategy
    pub fn with_strategy(strategy: KeyStrategy) -> Self {
        Self {
            strategy,
            normalize_whitespace: true,
            ignore_case: false,
            include_model_info: true,
            custom_normalizers: Vec::new(),
        }
    }

    /// Configure whitespace normalization
    pub fn normalize_whitespace(mut self, enabled: bool) -> Self {
        self.normalize_whitespace = enabled;
        self
    }

    /// Configure case sensitivity
    pub fn ignore_case(mut self, enabled: bool) -> Self {
        self.ignore_case = enabled;
        self
    }

    /// Configure whether to include model information in keys
    pub fn include_model_info(mut self, enabled: bool) -> Self {
        self.include_model_info = enabled;
        self
    }

    /// Add a custom text normalizer
    pub fn add_normalizer<F>(mut self, normalizer: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.custom_normalizers.push(Box::new(normalizer));
        self
    }

    /// Generate a cache key for a prompt
    pub fn generate_key(&self, prompt: &str) -> CacheResult<CacheKey> {
        self.generate_key_with_params(prompt, None, None)
    }

    /// Generate a cache key with generation parameters
    pub fn generate_key_with_params(
        &self,
        prompt: &str,
        params: Option<&GenerationParams>,
        model_name: Option<&str>,
    ) -> CacheResult<CacheKey> {
        let normalized_prompt = self.normalize_text(prompt);

        let key_components = match self.strategy {
            KeyStrategy::Simple => {
                vec![normalized_prompt.clone()]
            }
            KeyStrategy::WithParameters => {
                let mut components = vec![normalized_prompt.clone()];
                if let Some(params) = params {
                    components.push(self.serialize_params(params)?);
                }
                components
            }
            KeyStrategy::Semantic => {
                let semantic_prompt = self.apply_semantic_normalization(&normalized_prompt);
                let mut components = vec![semantic_prompt];
                if let Some(params) = params {
                    components.push(self.serialize_params(params)?);
                }
                components
            }
            KeyStrategy::ContentBased => {
                let content_prompt = self.apply_content_normalization(&normalized_prompt);
                let mut components = vec![content_prompt];
                if let Some(params) = params {
                    components.push(self.serialize_params(params)?);
                }
                if self.include_model_info {
                    if let Some(model) = model_name {
                        components.push(model.to_string());
                    }
                }
                components
            }
        };

        let combined_input = key_components.join("|");
        let key_hash = self.hash_string(&combined_input);

        // Create a shortened display key for readability
        let display_key = self.create_display_key(prompt, params, model_name);

        let mut cache_key = CacheKey::new(key_hash, display_key);

        // Add metadata
        cache_key = cache_key
            .with_metadata("strategy".to_string(), format!("{:?}", self.strategy))
            .with_metadata("prompt_length".to_string(), prompt.len().to_string());

        if let Some(model) = model_name {
            cache_key = cache_key.with_metadata("model".to_string(), model.to_string());
        }

        Ok(cache_key)
    }

    /// Normalize text according to configuration
    fn normalize_text(&self, text: &str) -> String {
        let mut normalized = text.to_string();

        // Apply whitespace normalization
        if self.normalize_whitespace {
            normalized = self.normalize_whitespace_internal(&normalized);
        }

        // Apply case normalization
        if self.ignore_case {
            normalized = normalized.to_lowercase();
        }

        // Apply custom normalizers
        for normalizer in &self.custom_normalizers {
            normalized = normalizer(&normalized);
        }

        normalized
    }

    /// Normalize whitespace by collapsing multiple spaces and trimming
    fn normalize_whitespace_internal(&self, text: &str) -> String {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string()
    }

    /// Apply semantic normalization (more aggressive)
    fn apply_semantic_normalization(&self, text: &str) -> String {
        let mut normalized = text.to_string();

        // Remove common punctuation that doesn't affect meaning
        normalized = normalized.replace(['.', ',', '!', '?', ';', ':'], "");

        // Normalize quotes
        normalized = normalized.replace(['"', '"', '"', '\'', '\''], "'");

        // Normalize dashes
        normalized = normalized.replace(['–', '—'], "-");

        // Collapse multiple spaces again after punctuation removal
        normalized = self.normalize_whitespace_internal(&normalized);

        normalized
    }

    /// Apply content-based normalization (most aggressive)
    fn apply_content_normalization(&self, text: &str) -> String {
        let mut normalized = self.apply_semantic_normalization(text);

        // Remove common stop words that don't affect LLM responses
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        let words: Vec<&str> = normalized.split_whitespace().collect();
        let filtered_words: Vec<&str> = words
            .into_iter()
            .filter(|word| !stop_words.contains(&word.to_lowercase().as_str()))
            .collect();

        if !filtered_words.is_empty() {
            normalized = filtered_words.join(" ");
        }

        // Additional normalizations for better content matching
        normalized = normalized.replace("what's", "what is");
        normalized = normalized.replace("won't", "will not");
        normalized = normalized.replace("can't", "cannot");
        normalized = normalized.replace("don't", "do not");

        normalized
    }

    /// Serialize generation parameters for inclusion in cache key
    fn serialize_params(&self, params: &GenerationParams) -> CacheResult<String> {
        // Create a deterministic string representation of parameters
        let mut param_parts = Vec::new();

        if let Some(max_tokens) = params.max_tokens {
            param_parts.push(format!("max_tokens:{max_tokens}"));
        }
        if let Some(temperature) = params.temperature {
            param_parts.push(format!("temp:{temperature:.3}"));
        }
        if let Some(top_p) = params.top_p {
            param_parts.push(format!("top_p:{top_p:.3}"));
        }
        if let Some(stop_sequences) = &params.stop_sequences {
            param_parts.push(format!("stop:{}", stop_sequences.join(",")));
        }

        Ok(param_parts.join(";"))
    }

    /// Create a hash of the input string
    fn hash_string(&self, input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Create a human-readable display key
    fn create_display_key(
        &self,
        prompt: &str,
        params: Option<&GenerationParams>,
        model_name: Option<&str>,
    ) -> String {
        let truncated_prompt = if prompt.len() > 50 {
            format!("{}...", &prompt[..47])
        } else {
            prompt.to_string()
        };

        let mut display_parts = vec![truncated_prompt];

        if let Some(params) = params {
            if let Some(temperature) = params.temperature {
                display_parts.push(format!("T:{temperature:.1}"));
            }
        }

        if let Some(model) = model_name {
            display_parts.push(format!("M:{model}"));
        }

        display_parts.join(" | ")
    }

    /// Get statistics about key generation
    pub fn key_statistics(&self, keys: &[CacheKey]) -> KeyStatistics {
        let total_keys = keys.len();
        let unique_keys = keys
            .iter()
            .map(|k| &k.key_hash)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let avg_prompt_length = if total_keys > 0 {
            keys.iter()
                .filter_map(|k| k.metadata.get("prompt_length"))
                .filter_map(|s| s.parse::<usize>().ok())
                .sum::<usize>() as f32
                / total_keys as f32
        } else {
            0.0
        };

        let collision_rate = if total_keys > 0 {
            1.0 - (unique_keys as f32 / total_keys as f32)
        } else {
            0.0
        };

        KeyStatistics {
            total_keys,
            unique_keys,
            collision_rate,
            avg_prompt_length,
            strategy: self.strategy.clone(),
        }
    }
}

impl Default for CacheKeyGenerator {
    /// Returns a new cache key generator with default settings
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about cache key generation
#[derive(Debug, Clone)]
pub struct KeyStatistics {
    /// Total number of keys generated
    pub total_keys: usize,
    /// Number of unique keys (after deduplication)
    pub unique_keys: usize,
    /// Rate of key collisions (0.0 to 1.0)
    pub collision_rate: f32,
    /// Average length of prompts that keys were generated from
    pub avg_prompt_length: f32,
    /// Key generation strategy used
    pub strategy: KeyStrategy,
}

impl KeyStatistics {
    /// Print cache key statistics to the log
    pub fn print(&self) {
        tracing::info!(
            strategy = ?self.strategy,
            total_keys = self.total_keys,
            unique_keys = self.unique_keys,
            collision_rate = format!("{:.2}%", self.collision_rate * 100.0),
            avg_prompt_length = format!("{:.1}", self.avg_prompt_length),
            "Cache key statistics"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key = CacheKey::new("hash123".to_string(), "display key".to_string())
            .with_metadata("test".to_string(), "value".to_string());

        assert_eq!(key.hash(), "hash123");
        assert_eq!(key.display(), "display key");
        assert_eq!(key.metadata.get("test"), Some(&"value".to_string()));
    }

    #[test]
    fn test_key_generator_simple() {
        let generator = CacheKeyGenerator::with_strategy(KeyStrategy::Simple);
        let key = generator.generate_key("Hello world").unwrap();

        assert!(!key.key_hash.is_empty());
        assert!(key.display_key.contains("Hello world"));
    }

    #[test]
    fn test_key_generator_with_params() {
        let generator = CacheKeyGenerator::with_strategy(KeyStrategy::WithParameters);
        let params = GenerationParams {
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: None,
        };

        let key1 = generator
            .generate_key_with_params("Hello", Some(&params), None)
            .unwrap();
        let key2 = generator
            .generate_key_with_params("Hello", None, None)
            .unwrap();

        // Keys should be different due to different parameters
        assert_ne!(key1.key_hash, key2.key_hash);
    }

    #[test]
    fn test_whitespace_normalization() {
        let generator = CacheKeyGenerator::new().normalize_whitespace(true);

        let key1 = generator.generate_key("Hello    world").unwrap();
        let key2 = generator.generate_key("Hello world").unwrap();

        // Should generate same key after whitespace normalization
        assert_eq!(key1.key_hash, key2.key_hash);
    }

    #[test]
    fn test_semantic_normalization() {
        let generator = CacheKeyGenerator::with_strategy(KeyStrategy::Semantic);

        let key1 = generator.generate_key("Hello, world!").unwrap();
        let key2 = generator.generate_key("Hello world").unwrap();

        // Should generate same key after semantic normalization
        assert_eq!(key1.key_hash, key2.key_hash);
    }

    #[test]
    fn test_case_sensitivity() {
        let generator = CacheKeyGenerator::new().ignore_case(true);

        let key1 = generator.generate_key("Hello World").unwrap();
        let key2 = generator.generate_key("hello world").unwrap();

        assert_eq!(key1.key_hash, key2.key_hash);
    }

    #[test]
    fn test_custom_normalizer() {
        let generator = CacheKeyGenerator::new()
            .add_normalizer(|text| text.replace("AI", "artificial intelligence"));

        let key1 = generator.generate_key("What is AI?").unwrap();
        let key2 = generator
            .generate_key("What is artificial intelligence?")
            .unwrap();

        assert_eq!(key1.key_hash, key2.key_hash);
    }

    #[test]
    fn test_key_statistics() {
        let generator = CacheKeyGenerator::new();
        let keys = vec![
            generator.generate_key("test 1").unwrap(),
            generator.generate_key("test 2").unwrap(),
            generator.generate_key("test 1").unwrap(), // Duplicate
        ];

        let stats = generator.key_statistics(&keys);
        assert_eq!(stats.total_keys, 3);
        assert_eq!(stats.unique_keys, 2);
        assert!(stats.collision_rate > 0.0);
    }
}
