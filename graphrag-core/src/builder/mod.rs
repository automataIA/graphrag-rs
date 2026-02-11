//! GraphRAG builder module
//!
//! This module provides a builder pattern for constructing GraphRAG instances.
//!
//! ## Example
//!
//! ```no_run
//! use graphrag_core::builder::GraphRAGBuilder;
//!
//! # fn example() -> graphrag_core::Result<()> {
//! let graphrag = GraphRAGBuilder::new()
//!     .with_output_dir("./my_output")
//!     .with_chunk_size(512)
//!     .with_ollama_base_url("http://localhost:11434")
//!     .with_embedding_model("nomic-embed-text:latest")
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use crate::config::Config;
use crate::core::Result;

/// Builder for GraphRAG instances
///
/// Provides a fluent API for configuring GraphRAG with various options.
#[derive(Debug, Clone)]
pub struct GraphRAGBuilder {
    config: Config,
}

impl Default for GraphRAGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphRAGBuilder {
    /// Create a new builder with default configuration
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::builder::GraphRAGBuilder;
    ///
    /// let builder = GraphRAGBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set the output directory for storing graphs and data
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_output_dir("./my_workspace");
    /// ```
    pub fn with_output_dir(mut self, dir: &str) -> Self {
        self.config.output_dir = dir.to_string();
        self
    }

    /// Set the chunk size for text processing
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_chunk_size(512);
    /// ```
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self.config.text.chunk_size = size;
        self
    }

    /// Set the chunk overlap for text processing
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_chunk_overlap(50);
    /// ```
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.config.chunk_overlap = overlap;
        self.config.text.chunk_overlap = overlap;
        self
    }

    /// Set the embedding dimension
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_embedding_dimension(384);
    /// ```
    pub fn with_embedding_dimension(mut self, dimension: usize) -> Self {
        self.config.embeddings.dimension = dimension;
        self
    }

    /// Set the embedding model name
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_embedding_model("nomic-embed-text:latest");
    /// ```
    pub fn with_embedding_model(mut self, model: &str) -> Self {
        self.config.embeddings.model = Some(model.to_string());
        self
    }

    /// Set the embedding backend ("candle", "fastembed", "api", "hash")
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_embedding_backend("candle");
    /// ```
    pub fn with_embedding_backend(mut self, backend: &str) -> Self {
        self.config.embeddings.backend = backend.to_string();
        self
    }

    /// Set the Ollama host
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_ollama_host("localhost");
    /// ```
    pub fn with_ollama_host(mut self, host: &str) -> Self {
        self.config.ollama.host = host.to_string();
        self
    }

    /// Set the Ollama port
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_ollama_port(11434);
    /// ```
    pub fn with_ollama_port(mut self, port: u16) -> Self {
        self.config.ollama.port = port;
        self
    }

    /// Enable Ollama integration
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_ollama_enabled(true);
    /// ```
    pub fn with_ollama_enabled(mut self, enabled: bool) -> Self {
        self.config.ollama.enabled = enabled;
        self
    }

    /// Set the Ollama chat/generation model
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_chat_model("llama3.2:latest");
    /// ```
    pub fn with_chat_model(mut self, model: &str) -> Self {
        self.config.ollama.chat_model = model.to_string();
        self
    }

    /// Set the Ollama embedding model
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_ollama_embedding_model("nomic-embed-text:latest");
    /// ```
    pub fn with_ollama_embedding_model(mut self, model: &str) -> Self {
        self.config.ollama.embedding_model = model.to_string();
        self
    }

    /// Set the top-k results for retrieval
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_top_k(10);
    /// ```
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.config.top_k_results = Some(k);
        self.config.retrieval.top_k = k;
        self
    }

    /// Set the similarity threshold for retrieval
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_similarity_threshold(0.7);
    /// ```
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.config.similarity_threshold = Some(threshold);
        self.config.graph.similarity_threshold = threshold;
        self
    }

    /// Set the pipeline approach ("semantic", "algorithmic", or "hybrid")
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_approach("hybrid");
    /// ```
    pub fn with_approach(mut self, approach: &str) -> Self {
        self.config.approach = approach.to_string();
        self
    }

    /// Enable or disable parallel processing
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_parallel_processing(true);
    /// ```
    pub fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.config.parallel.enabled = enabled;
        self
    }

    /// Set the number of parallel threads
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_num_threads(4);
    /// ```
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.config.parallel.num_threads = num_threads;
        self
    }

    /// Enable or disable auto-save functionality
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_auto_save(true, 300); // Save every 5 minutes
    /// ```
    pub fn with_auto_save(mut self, enabled: bool, interval_seconds: u64) -> Self {
        self.config.auto_save.enabled = enabled;
        self.config.auto_save.interval_seconds = interval_seconds;
        self
    }

    /// Set the auto-save workspace name
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_auto_save_workspace("my_project");
    /// ```
    pub fn with_auto_save_workspace(mut self, name: &str) -> Self {
        self.config.auto_save.workspace_name = Some(name.to_string());
        self
    }

    /// Configure for local zero-config setup using Ollama
    ///
    /// Sets up:
    /// - Ollama enabled with localhost:11434
    /// - Default models (nomic-embed-text for embeddings, llama3.2 for chat)
    /// - Candle backend for local embeddings
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// let builder = GraphRAGBuilder::new()
    ///     .with_local_defaults();
    /// ```
    pub fn with_local_defaults(mut self) -> Self {
        self.config.ollama.enabled = true;
        self.config.ollama.host = "localhost".to_string();
        self.config.ollama.port = 11434;
        self.config.embeddings.backend = "candle".to_string();
        self
    }

    /// Build the GraphRAG instance with the configured settings
    ///
    /// # Errors
    /// Returns an error if the GraphRAG initialization fails
    ///
    /// # Example
    /// ```no_run
    /// # use graphrag_core::builder::GraphRAGBuilder;
    /// # fn example() -> graphrag_core::Result<()> {
    /// let graphrag = GraphRAGBuilder::new()
    ///     .with_output_dir("./workspace")
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<crate::GraphRAG> {
        crate::GraphRAG::new(self.config)
    }

    /// Get a reference to the current configuration
    ///
    /// Useful for inspecting the configuration before building
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get a mutable reference to the current configuration
    ///
    /// Allows direct manipulation of the config for advanced use cases
    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let builder = GraphRAGBuilder::new();
        assert_eq!(builder.config().output_dir, "./output");
    }

    #[test]
    fn test_builder_with_output_dir() {
        let builder = GraphRAGBuilder::new().with_output_dir("./custom");
        assert_eq!(builder.config().output_dir, "./custom");
    }

    #[test]
    fn test_builder_with_chunk_size() {
        let builder = GraphRAGBuilder::new().with_chunk_size(512);
        assert_eq!(builder.config().chunk_size, 512);
        assert_eq!(builder.config().text.chunk_size, 512);
    }

    #[test]
    fn test_builder_with_embedding_config() {
        let builder = GraphRAGBuilder::new()
            .with_embedding_dimension(384)
            .with_embedding_model("test-model");

        assert_eq!(builder.config().embeddings.dimension, 384);
        assert_eq!(
            builder.config().embeddings.model,
            Some("test-model".to_string())
        );
    }

    #[test]
    fn test_builder_with_ollama() {
        let builder = GraphRAGBuilder::new()
            .with_ollama_enabled(true)
            .with_ollama_host("custom-host")
            .with_ollama_port(8080)
            .with_chat_model("custom-model");

        assert!(builder.config().ollama.enabled);
        assert_eq!(builder.config().ollama.host, "custom-host");
        assert_eq!(builder.config().ollama.port, 8080);
        assert_eq!(builder.config().ollama.chat_model, "custom-model");
    }

    #[test]
    fn test_builder_with_retrieval() {
        let builder = GraphRAGBuilder::new()
            .with_top_k(20)
            .with_similarity_threshold(0.8);

        assert_eq!(builder.config().top_k_results, Some(20));
        assert_eq!(builder.config().retrieval.top_k, 20);
        assert_eq!(builder.config().similarity_threshold, Some(0.8));
        assert_eq!(builder.config().graph.similarity_threshold, 0.8);
    }

    #[test]
    fn test_builder_with_parallel() {
        let builder = GraphRAGBuilder::new()
            .with_parallel_processing(false)
            .with_num_threads(8);

        assert!(!builder.config().parallel.enabled);
        assert_eq!(builder.config().parallel.num_threads, 8);
    }

    #[test]
    fn test_builder_with_auto_save() {
        let builder = GraphRAGBuilder::new()
            .with_auto_save(true, 600)
            .with_auto_save_workspace("test");

        assert!(builder.config().auto_save.enabled);
        assert_eq!(builder.config().auto_save.interval_seconds, 600);
        assert_eq!(
            builder.config().auto_save.workspace_name,
            Some("test".to_string())
        );
    }

    #[test]
    fn test_builder_local_defaults() {
        let builder = GraphRAGBuilder::new().with_local_defaults();

        assert!(builder.config().ollama.enabled);
        assert_eq!(builder.config().ollama.host, "localhost");
        assert_eq!(builder.config().ollama.port, 11434);
        assert_eq!(builder.config().embeddings.backend, "candle");
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = GraphRAGBuilder::new()
            .with_output_dir("./test")
            .with_chunk_size(256)
            .with_chunk_overlap(32)
            .with_top_k(15)
            .with_approach("hybrid");

        assert_eq!(builder.config().output_dir, "./test");
        assert_eq!(builder.config().chunk_size, 256);
        assert_eq!(builder.config().chunk_overlap, 32);
        assert_eq!(builder.config().top_k_results, Some(15));
        assert_eq!(builder.config().approach, "hybrid");
    }
}
