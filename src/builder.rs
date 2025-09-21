//! GraphRAG Builder Pattern for Simplified API
//!
//! This module provides a fluent builder API to dramatically simplify GraphRAG initialization
//! from 6+ steps to 2-3 lines of code while maintaining full backward compatibility.

use crate::{
    config::Config,
    generation::{AnswerGenerator, GenerationConfig, MockLLM},
    GraphRAG, GraphRAGError, Result,
};

#[cfg(feature = "ollama")]
use crate::ollama::{is_ollama_available, OllamaGenerator};

#[cfg(feature = "parallel-processing")]
use num_cpus;

use std::path::Path;

/// Provider types for auto-detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LLMProvider {
    /// Mock LLM for testing
    Mock,
    /// Ollama local LLM server
    #[cfg(feature = "ollama")]
    Ollama { host: String, port: u16 },
    /// OpenAI API (future implementation)
    #[cfg(feature = "openai")]
    OpenAI { api_key: String },
}

/// Preset configuration templates
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigPreset {
    /// Basic configuration for getting started
    Basic,
    /// Advanced configuration with all features enabled
    Advanced,
    /// Production-ready configuration with optimizations
    Production,
    /// Memory-optimized configuration for resource-constrained environments
    MemoryOptimized,
    /// Performance-optimized configuration for high-throughput scenarios
    PerformanceOptimized,
}

/// GraphRAG Builder for fluent API construction
pub struct GraphRAGBuilder {
    config: Config,
    llm_provider: Option<LLMProvider>,
    auto_initialize: bool,
    preset: Option<ConfigPreset>,
    #[cfg(feature = "caching")]
    cache_config: Option<crate::caching::CacheConfig>,
    #[cfg(feature = "caching")]
    warming_config: Option<crate::caching::WarmingConfig>,
}

impl GraphRAGBuilder {
    /// Create a new GraphRAG builder with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            llm_provider: None,
            auto_initialize: true,
            preset: None,
            #[cfg(feature = "caching")]
            cache_config: None,
            #[cfg(feature = "caching")]
            warming_config: None,
        }
    }

    /// Load configuration from a JSON file path
    pub fn from_config_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let config = Config::from_file(path_ref.to_str().unwrap())?;

        Ok(Self {
            config,
            llm_provider: None,
            auto_initialize: true,
            preset: None,
            #[cfg(feature = "caching")]
            cache_config: None,
            #[cfg(feature = "caching")]
            warming_config: None,
        })
    }

    /// Apply a preset configuration template
    pub fn with_preset(mut self, preset: ConfigPreset) -> Self {
        self.preset = Some(preset.clone());
        self.config = self.apply_preset(preset);
        self
    }

    /// Enable automatic LLM provider detection
    ///
    /// This will attempt to detect available LLM providers in order:
    /// 1. Ollama (if feature enabled and server available)
    /// 2. OpenAI (if feature enabled and API key provided)
    /// 3. Mock LLM (fallback)
    pub fn auto_detect_llm(mut self) -> Self {
        // Store for later async detection during build
        self.llm_provider = None; // Will trigger auto-detection
        self
    }

    /// Manually set the LLM provider
    pub fn with_llm_provider(mut self, provider: LLMProvider) -> Self {
        self.llm_provider = Some(provider);
        self
    }

    /// Enable Ollama with default configuration
    #[cfg(feature = "ollama")]
    pub fn with_ollama(mut self) -> Self {
        self.config.ollama.enabled = true;
        self.llm_provider = Some(LLMProvider::Ollama {
            host: self.config.ollama.host.clone(),
            port: self.config.ollama.port,
        });
        self
    }

    /// Enable Ollama with custom host and port
    #[cfg(feature = "ollama")]
    pub fn with_ollama_at(mut self, host: impl Into<String>, port: u16) -> Self {
        let host = host.into();
        self.config.ollama.host = host.clone();
        self.config.ollama.port = port;
        self.config.ollama.enabled = true;
        self.llm_provider = Some(LLMProvider::Ollama { host, port });
        self
    }

    /// Configure text processing settings
    pub fn with_text_config(mut self, chunk_size: usize, chunk_overlap: usize) -> Self {
        self.config.text.chunk_size = chunk_size;
        self.config.text.chunk_overlap = chunk_overlap;
        self
    }

    /// Configure embedding settings
    pub fn with_embedding_config(
        mut self,
        dimension: usize,
        backend: impl Into<String>,
    ) -> Self {
        self.config.embeddings.dimension = dimension;
        self.config.embeddings.backend = backend.into();
        self
    }

    /// Configure entity extraction settings
    pub fn with_entity_config(
        mut self,
        min_confidence: f32,
        entity_types: Vec<String>,
    ) -> Self {
        self.config.entities.min_confidence = min_confidence;
        self.config.entities.entity_types = entity_types;
        self
    }

    /// Configure parallel processing
    pub fn with_parallel_processing(mut self, enabled: bool, num_threads: Option<usize>) -> Self {
        self.config.parallel.enabled = enabled;
        if let Some(threads) = num_threads {
            self.config.parallel.num_threads = threads;
        }
        self
    }

    /// Disable parallel processing
    pub fn without_parallel_processing(mut self) -> Self {
        self.config.parallel.enabled = false;
        self
    }

    /// Control whether to automatically initialize the GraphRAG system
    pub fn auto_initialize(mut self, auto_init: bool) -> Self {
        self.auto_initialize = auto_init;
        self
    }

    /// Enable LLM response caching with default configuration
    #[cfg(feature = "caching")]
    pub fn with_caching(mut self) -> Self {
        self.cache_config = Some(crate::caching::CacheConfig::default());
        self
    }

    /// Enable LLM response caching with custom configuration
    #[cfg(feature = "caching")]
    pub fn with_cache_config(mut self, config: crate::caching::CacheConfig) -> Self {
        self.cache_config = Some(config);
        self
    }

    /// Enable cache warming with default configuration
    #[cfg(feature = "caching")]
    pub fn with_cache_warming(mut self) -> Self {
        if self.cache_config.is_none() {
            self.cache_config = Some(crate::caching::CacheConfig::default());
        }
        self.warming_config = Some(crate::caching::WarmingConfig::default());
        self
    }

    /// Enable cache warming with custom configuration
    #[cfg(feature = "caching")]
    pub fn with_warming_config(mut self, config: crate::caching::WarmingConfig) -> Self {
        if self.cache_config.is_none() {
            self.cache_config = Some(crate::caching::CacheConfig::default());
        }
        self.warming_config = Some(config);
        self
    }

    /// Configure production-ready caching (high capacity, long TTL, warming enabled)
    #[cfg(feature = "caching")]
    pub fn with_production_caching(mut self) -> Self {
        self.cache_config = Some(crate::caching::CacheConfig::production());
        self.warming_config = Some(
            crate::caching::WarmingConfig::builder()
                .strategy(crate::caching::WarmingStrategy::PredefinedQueries)
                .max_queries(100)
                .background_warming(true)
                .build()
        );
        self
    }

    /// Configure high-performance caching (optimized for speed)
    #[cfg(feature = "caching")]
    pub fn with_high_performance_caching(mut self) -> Self {
        self.cache_config = Some(crate::caching::CacheConfig::high_performance());
        self.warming_config = Some(
            crate::caching::WarmingConfig::builder()
                .strategy(crate::caching::WarmingStrategy::FrequencyBased)
                .max_queries(50)
                .background_warming(true)
                .build()
        );
        self
    }

    /// Build the GraphRAG instance
    ///
    /// This method handles:
    /// - LLM provider auto-detection (if not manually set)
    /// - Component initialization (if auto_initialize is true)
    /// - Error handling with meaningful messages
    pub fn build(self) -> Result<GraphRAG> {
        // Apply preset if specified
        let mut config = if let Some(ref preset) = self.preset {
            self.apply_preset(preset.clone())
        } else {
            self.config.clone()
        };

        // Auto-detect LLM provider if not set
        let provider = if let Some(provider) = self.llm_provider {
            provider
        } else {
            self.detect_available_llm_provider(&config)?
        };

        // Update config based on detected provider
        Self::configure_for_provider(&mut config, &provider)?;

        // Create GraphRAG instance
        let mut graphrag = GraphRAG::new(config)?;

        // Initialize if requested
        if self.auto_initialize {
            graphrag.initialize()?;

            // Set up LLM provider-specific components
            Self::setup_llm_components(&mut graphrag, provider)?;
        }

        Ok(graphrag)
    }

    /// Build asynchronously (for async LLM provider detection)
    #[cfg(feature = "ollama")]
    pub async fn build_async(self) -> Result<GraphRAG> {
        // Apply preset if specified
        let mut config = if let Some(ref preset) = self.preset {
            self.apply_preset(preset.clone())
        } else {
            self.config.clone()
        };

        // Auto-detect LLM provider if not set (async version)
        let provider = if let Some(provider) = self.llm_provider {
            provider
        } else {
            self.detect_available_llm_provider_async(&config).await?
        };

        // Update config based on detected provider
        Self::configure_for_provider(&mut config, &provider)?;

        // Create GraphRAG instance
        let mut graphrag = GraphRAG::new(config)?;

        // Initialize if requested
        if self.auto_initialize {
            graphrag.initialize()?;

            // Set up LLM provider-specific components
            Self::setup_llm_components(&mut graphrag, provider)?;
        }

        Ok(graphrag)
    }

    /// Build async GraphRAG system (new async implementation)
    #[cfg(feature = "async-traits")]
    pub async fn build_async_graphrag(self) -> Result<crate::async_graphrag::AsyncGraphRAG> {
        // Apply preset if specified
        let config = if let Some(ref preset) = self.preset {
            self.apply_preset(preset.clone())
        } else {
            self.config.clone()
        };

        // Set up async language model based on provider
        let provider = if let Some(provider) = self.llm_provider {
            provider
        } else {
            #[cfg(feature = "ollama")]
            {
                self.detect_available_llm_provider_async(&config).await?
            }
            #[cfg(not(feature = "ollama"))]
            {
                LLMProvider::Mock
            }
        };

        let mut builder = crate::async_graphrag::AsyncGraphRAGBuilder::new().config(config);

        builder = match provider {
            LLMProvider::Mock => {
                #[cfg(feature = "async-traits")]
                {
                    builder.with_async_mock_llm().await?
                }
                #[cfg(not(feature = "async-traits"))]
                {
                    return Err(GraphRAGError::Config {
                        message: "async-traits feature required for async GraphRAG".to_string(),
                    });
                }
            }
            #[cfg(all(feature = "ollama", feature = "async-traits"))]
            LLMProvider::Ollama { host, port } => {
                let ollama_config = crate::ollama::OllamaConfig {
                    host,
                    port,
                    enabled: true,
                    ..crate::ollama::OllamaConfig::default()
                };
                builder.with_async_ollama(ollama_config).await?
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI { api_key: _ } => {
                // TODO: Implement async OpenAI support
                builder.with_async_mock_llm().await?
            }
        };

        builder.build().await
    }

    /// Apply preset configuration
    fn apply_preset(&self, preset: ConfigPreset) -> Config {
        let mut config = self.config.clone();

        match preset {
            ConfigPreset::Basic => {
                // Basic configuration - minimal features for getting started
                config.parallel.enabled = false;
                config.retrieval.top_k = 5;
                config.entities.min_confidence = 0.8;
                config.text.chunk_size = 500;
                config.text.chunk_overlap = 50;
                config.enhancements.enabled = false;
            }
            ConfigPreset::Advanced => {
                // Advanced configuration - all features enabled
                config.parallel.enabled = true;
                config.retrieval.top_k = 20;
                config.entities.min_confidence = 0.6;
                config.text.chunk_size = 1000;
                config.text.chunk_overlap = 200;
                config.enhancements.enabled = true;
                config.enhancements.query_analysis.enabled = true;
                config.enhancements.adaptive_retrieval.enabled = true;
            }
            ConfigPreset::Production => {
                // Production-ready configuration
                config.parallel.enabled = true;
                config.parallel.num_threads = 0; // Auto-detect
                config.retrieval.top_k = 15;
                config.entities.min_confidence = 0.7;
                config.text.chunk_size = 800;
                config.text.chunk_overlap = 160;
                config.enhancements.enabled = true;
                config.enhancements.performance_benchmarking.enabled = true;
                config.ollama.timeout_seconds = 60;
                config.ollama.max_retries = 5;
            }
            ConfigPreset::MemoryOptimized => {
                // Memory-optimized for resource-constrained environments
                config.parallel.enabled = false;
                config.retrieval.top_k = 5;
                config.entities.min_confidence = 0.8;
                config.text.chunk_size = 300;
                config.text.chunk_overlap = 30;
                config.embeddings.dimension = 256; // Smaller embeddings
                config.enhancements.enabled = false;
            }
            ConfigPreset::PerformanceOptimized => {
                // Performance-optimized for high throughput
                config.parallel.enabled = true;
                #[cfg(feature = "parallel-processing")]
                {
                    config.parallel.num_threads = num_cpus::get().max(4);
                }
                #[cfg(not(feature = "parallel-processing"))]
                {
                    config.parallel.num_threads = 4; // fallback to 4 threads
                }
                config.parallel.parallel_embeddings = true;
                config.parallel.parallel_graph_ops = true;
                config.parallel.parallel_vector_ops = true;
                config.retrieval.top_k = 10;
                config.entities.min_confidence = 0.75;
                config.text.chunk_size = 1200;
                config.text.chunk_overlap = 240;
                config.enhancements.enabled = true;
                config.enhancements.adaptive_retrieval.enabled = true;
            }
        }

        config
    }

    /// Detect available LLM provider (synchronous version)
    fn detect_available_llm_provider(&self, config: &Config) -> Result<LLMProvider> {
        // For sync version, we can't check Ollama availability
        // So we prefer based on feature flags and config

        #[cfg(feature = "ollama")]
        if config.ollama.enabled {
            return Ok(LLMProvider::Ollama {
                host: config.ollama.host.clone(),
                port: config.ollama.port,
            });
        }

        // Fallback to mock
        Ok(LLMProvider::Mock)
    }

    /// Detect available LLM provider (async version)
    #[cfg(feature = "ollama")]
    async fn detect_available_llm_provider_async(&self, config: &Config) -> Result<LLMProvider> {
        // Try Ollama first if feature is enabled
        let ollama_url = format!("{}:{}", config.ollama.host, config.ollama.port);
        if is_ollama_available(&ollama_url).await {
            return Ok(LLMProvider::Ollama {
                host: config.ollama.host.clone(),
                port: config.ollama.port,
            });
        }

        // Try with localhost if configured host fails
        if config.ollama.host != "http://localhost"
            && is_ollama_available("http://localhost:11434").await
        {
            return Ok(LLMProvider::Ollama {
                host: "http://localhost".to_string(),
                port: 11434,
            });
        }

        // TODO: Check OpenAI when feature is available
        #[cfg(feature = "openai")]
        {
            // Check for OpenAI API key in environment
            if std::env::var("OPENAI_API_KEY").is_ok() {
                return Ok(LLMProvider::OpenAI {
                    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
                });
            }
        }

        // Fallback to mock
        Ok(LLMProvider::Mock)
    }

    /// Configure settings based on detected provider
    fn configure_for_provider(config: &mut Config, provider: &LLMProvider) -> Result<()> {
        match provider {
            LLMProvider::Mock => {
                // Mock LLM configuration
                config.embeddings.backend = "hash".to_string();
                config.embeddings.fallback_to_hash = true;
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama { host, port } => {
                // Ollama configuration
                config.ollama.enabled = true;
                config.ollama.host = host.clone();
                config.ollama.port = *port;
                config.embeddings.backend = "ollama".to_string();
                config.embeddings.fallback_to_hash = true;

                // Validate Ollama config
                config.ollama.validate().map_err(|e| GraphRAGError::Config {
                    message: format!("Ollama configuration invalid: {e}"),
                })?;
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI { api_key: _ } => {
                // OpenAI configuration
                config.embeddings.backend = "openai".to_string();
                config.embeddings.fallback_to_hash = true;
            }
        }

        Ok(())
    }

    /// Set up LLM-specific components
    fn setup_llm_components(graphrag: &mut GraphRAG, provider: LLMProvider) -> Result<()> {
        match provider {
            LLMProvider::Mock => {
                // Mock LLM is already set up by default
                let llm = Box::new(MockLLM::new()?);
                let generation_config = GenerationConfig::default();
                let generator = AnswerGenerator::new(llm, generation_config)?;
                graphrag.set_answer_generator(generator);
            }
            #[cfg(feature = "ollama")]
            LLMProvider::Ollama { host, port } => {
                // Set up Ollama components
                let ollama_url = format!("{host}:{port}");
                let llm = Box::new(OllamaGenerator::new(ollama_url)?);
                let generation_config = GenerationConfig::default();
                let generator = AnswerGenerator::new(llm, generation_config)?;
                graphrag.set_answer_generator(generator);
            }
            #[cfg(feature = "openai")]
            LLMProvider::OpenAI { api_key: _ } => {
                // TODO: Set up OpenAI components when implemented
                let llm = Box::new(MockLLM::new()?);
                let generation_config = GenerationConfig::default();
                let generator = AnswerGenerator::new(llm, generation_config)?;
                graphrag.set_answer_generator(generator);
            }
        }

        Ok(())
    }
}

impl Default for GraphRAGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience methods for GraphRAG struct
impl GraphRAG {
    /// Create a new GraphRAG builder
    pub fn builder() -> GraphRAGBuilder {
        GraphRAGBuilder::new()
    }

    /// Quick setup with auto-detection (async)
    #[cfg(feature = "ollama")]
    pub async fn quick_setup() -> Result<Self> {
        Self::builder()
            .auto_detect_llm()
            .with_preset(ConfigPreset::Basic)
            .build_async()
            .await
    }

    /// Quick setup with mock LLM (sync)
    pub fn quick_setup_mock() -> Result<Self> {
        Self::builder()
            .with_llm_provider(LLMProvider::Mock)
            .with_preset(ConfigPreset::Basic)
            .build()
    }

    /// Production setup with auto-detection (async)
    #[cfg(feature = "ollama")]
    pub async fn production_setup() -> Result<Self> {
        Self::builder()
            .auto_detect_llm()
            .with_preset(ConfigPreset::Production)
            .build_async()
            .await
    }

    /// Quick setup for async GraphRAG with auto-detection
    #[cfg(all(feature = "async-traits", feature = "ollama"))]
    pub async fn quick_setup_async() -> Result<crate::async_graphrag::AsyncGraphRAG> {
        Self::builder()
            .auto_detect_llm()
            .with_preset(ConfigPreset::Basic)
            .build_async_graphrag()
            .await
    }

    /// Quick setup for async GraphRAG with mock LLM
    #[cfg(feature = "async-traits")]
    pub async fn quick_setup_async_mock() -> Result<crate::async_graphrag::AsyncGraphRAG> {
        Self::builder()
            .with_llm_provider(LLMProvider::Mock)
            .with_preset(ConfigPreset::Basic)
            .build_async_graphrag()
            .await
    }

    /// Production setup for async GraphRAG
    #[cfg(all(feature = "async-traits", feature = "ollama"))]
    pub async fn production_setup_async() -> Result<crate::async_graphrag::AsyncGraphRAG> {
        Self::builder()
            .auto_detect_llm()
            .with_preset(ConfigPreset::Production)
            .build_async_graphrag()
            .await
    }

    /// Create instance with service registry (backward compatibility)
    pub fn with_services(config: Config, _registry: crate::ServiceRegistry) -> Result<Self> {
        // For backward compatibility, create GraphRAG with config
        // The registry pattern is being phased out in favor of the builder pattern
        Self::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = GraphRAGBuilder::new();
        assert!(builder.auto_initialize);
        assert!(builder.llm_provider.is_none());
        assert!(builder.preset.is_none());
    }

    #[test]
    fn test_preset_application() {
        let builder = GraphRAGBuilder::new().with_preset(ConfigPreset::Basic);
        assert_eq!(builder.preset, Some(ConfigPreset::Basic));
    }

    #[test]
    fn test_mock_llm_build() {
        let result = GraphRAGBuilder::new()
            .with_llm_provider(LLMProvider::Mock)
            .with_preset(ConfigPreset::Basic)
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_text_config() {
        let builder = GraphRAGBuilder::new()
            .with_text_config(800, 160);

        assert_eq!(builder.config.text.chunk_size, 800);
        assert_eq!(builder.config.text.chunk_overlap, 160);
    }

    #[test]
    fn test_parallel_processing() {
        let builder = GraphRAGBuilder::new()
            .with_parallel_processing(true, Some(4));

        assert!(builder.config.parallel.enabled);
        assert_eq!(builder.config.parallel.num_threads, 4);
    }

    #[test]
    fn test_without_parallel_processing() {
        let builder = GraphRAGBuilder::new()
            .without_parallel_processing();

        assert!(!builder.config.parallel.enabled);
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_ollama_configuration() {
        let builder = GraphRAGBuilder::new()
            .with_ollama_at("http://192.168.1.100", 11434);

        assert_eq!(builder.config.ollama.host, "http://192.168.1.100");
        assert_eq!(builder.config.ollama.port, 11434);
        assert!(builder.config.ollama.enabled);

        if let Some(LLMProvider::Ollama { host, port }) = builder.llm_provider {
            assert_eq!(host, "http://192.168.1.100");
            assert_eq!(port, 11434);
        } else {
            panic!("Expected Ollama provider");
        }
    }

    #[test]
    fn test_quick_setup_mock() {
        let result = GraphRAG::quick_setup_mock();
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[cfg(feature = "ollama")]
    async fn test_async_build() {
        let result = GraphRAGBuilder::new()
            .with_llm_provider(LLMProvider::Mock)
            .build_async()
            .await;

        assert!(result.is_ok());
    }
}