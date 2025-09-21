//! # Neural Embeddings Module
//!
//! This module provides neural embedding capabilities with sentence transformers,
//! model management, and hybrid fallback systems for GraphRAG-RS.

pub mod neural;
pub mod hybrid;

pub use neural::{SentenceTransformer, ModelManager, PretrainedModel};
pub use hybrid::{HybridEmbedder, EmbeddingProvider};

use crate::core::Result;

/// Embedding configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub primary_provider: EmbeddingProviderType,
    pub fallback_providers: Vec<EmbeddingProviderType>,
    pub model_name: String,
    pub cache_size: usize,
    pub batch_size: usize,
    pub max_length: usize,
    pub device: Device,
    pub pooling_strategy: PoolingStrategy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingProviderType {
    Neural,
    Ollama,
    Hash,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    Cuda,
    Cpu,
    Auto,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PoolingStrategy {
    Mean,
    Max,
    Cls,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            primary_provider: EmbeddingProviderType::Neural,
            fallback_providers: vec![
                EmbeddingProviderType::Ollama,
                EmbeddingProviderType::Hash,
            ],
            model_name: "all-MiniLM-L6-v2".to_string(),
            cache_size: 10000,
            batch_size: 32,
            max_length: 512,
            device: Device::Auto,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
}

/// Unified embedding dimensions for different models
#[derive(Debug, Clone)]
pub struct EmbeddingDimensions {
    pub all_mini_lm_l6_v2: usize,     // 384
    pub all_mpnet_base_v2: usize,     // 768
    pub multilingual_e5_large: usize, // 1024
    pub distil_use_base: usize,       // 512
}

impl Default for EmbeddingDimensions {
    fn default() -> Self {
        Self {
            all_mini_lm_l6_v2: 384,
            all_mpnet_base_v2: 768,
            multilingual_e5_large: 1024,
            distil_use_base: 512,
        }
    }
}

impl EmbeddingDimensions {
    pub fn get_dimensions(&self, model_name: &str) -> usize {
        match model_name {
            "all-MiniLM-L6-v2" => self.all_mini_lm_l6_v2,
            "all-mpnet-base-v2" => self.all_mpnet_base_v2,
            "multilingual-e5-large" => self.multilingual_e5_large,
            "distiluse-base-multilingual-cased" => self.distil_use_base,
            _ => self.all_mini_lm_l6_v2, // Default fallback
        }
    }
}

/// Statistics for embedding operations
#[derive(Debug, Clone)]
pub struct EmbeddingStats {
    pub total_embeddings_generated: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub provider_fallbacks: usize,
    pub average_embedding_time_ms: f32,
    pub batch_operations: usize,
}

impl Default for EmbeddingStats {
    fn default() -> Self {
        Self {
            total_embeddings_generated: 0,
            cache_hits: 0,
            cache_misses: 0,
            provider_fallbacks: 0,
            average_embedding_time_ms: 0.0,
            batch_operations: 0,
        }
    }
}

impl EmbeddingStats {
    pub fn cache_hit_rate(&self) -> f32 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
        }
    }

    pub fn print(&self) {
        println!("🧮 Embedding Statistics:");
        println!("   Total embeddings: {}", self.total_embeddings_generated);
        println!("   Cache hit rate: {:.1}%", self.cache_hit_rate() * 100.0);
        println!("   Provider fallbacks: {}", self.provider_fallbacks);
        println!("   Avg embedding time: {:.1}ms", self.average_embedding_time_ms);
        println!("   Batch operations: {}", self.batch_operations);
    }
}

/// Main embedding manager that coordinates all embedding providers
pub struct EmbeddingManager {
    config: EmbeddingConfig,
    hybrid_embedder: HybridEmbedder,
    stats: EmbeddingStats,
}

impl EmbeddingManager {
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let hybrid_embedder = HybridEmbedder::new(&config)?;

        Ok(Self {
            config,
            hybrid_embedder,
            stats: EmbeddingStats::default(),
        })
    }

    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();
        let result = self.hybrid_embedder.embed(text).await;
        let duration = start_time.elapsed();

        // Update statistics
        self.stats.total_embeddings_generated += 1;
        self.stats.average_embedding_time_ms =
            (self.stats.average_embedding_time_ms * (self.stats.total_embeddings_generated - 1) as f32
             + duration.as_millis() as f32) / self.stats.total_embeddings_generated as f32;

        result
    }

    pub async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let start_time = std::time::Instant::now();
        let result = self.hybrid_embedder.batch_embed(texts).await;
        let duration = start_time.elapsed();

        // Update statistics
        self.stats.total_embeddings_generated += texts.len();
        self.stats.batch_operations += 1;
        self.stats.average_embedding_time_ms =
            (self.stats.average_embedding_time_ms * (self.stats.total_embeddings_generated - texts.len()) as f32
             + duration.as_millis() as f32) / self.stats.total_embeddings_generated as f32;

        result
    }

    pub fn get_stats(&self) -> &EmbeddingStats {
        &self.stats
    }

    pub fn get_config(&self) -> &EmbeddingConfig {
        &self.config
    }

    pub fn clear_cache(&mut self) -> Result<()> {
        self.hybrid_embedder.clear_cache()
    }
}