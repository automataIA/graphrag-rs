//! Hybrid embedding system with multiple provider fallbacks

use std::sync::Arc;
use std::time::Instant;
use crate::core::{Result, GraphRAGError};
use crate::embeddings::{EmbeddingConfig, EmbeddingProviderType, EmbeddingStats};
use crate::embeddings::neural::{NeuralEmbedder, NeuralConfig};
use crate::embeddings::neural::cache::EmbeddingCache;

/// Trait for embedding providers
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>>;
    async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn get_embedding_dimension(&self) -> usize;
    fn get_provider_name(&self) -> &str;
    fn is_available(&self) -> bool;
}

/// Neural embedding provider wrapper
pub struct NeuralProvider {
    embedder: NeuralEmbedder,
    dimension: usize,
}

impl NeuralProvider {
    pub async fn new(config: &EmbeddingConfig) -> Result<Self> {
        let neural_config = NeuralConfig {
            model_name: config.model_name.clone(),
            device: config.device.clone(),
            max_length: config.max_length,
            pooling: config.pooling_strategy.clone(),
            cache_size: config.cache_size,
            batch_size: config.batch_size,
        };

        let mut embedder = NeuralEmbedder::new(neural_config)?;

        // Try to initialize - if it fails, we'll mark as unavailable
        let dimension = match embedder.initialize().await {
            Ok(_) => {
                if let Some(info) = embedder.get_model_info() {
                    info.embedding_dimension
                } else {
                    384 // Default fallback
                }
            }
            Err(_) => return Err(GraphRAGError::VectorSearch {
                message: "Failed to initialize neural embedder".to_string(),
            }),
        };

        Ok(Self { embedder, dimension })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for NeuralProvider {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embedder.embed(text).await
    }

    async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder.batch_embed(texts).await
    }

    fn get_embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn get_provider_name(&self) -> &str {
        "neural"
    }

    fn is_available(&self) -> bool {
        self.embedder.is_initialized()
    }
}

/// Ollama embedding provider (placeholder)
pub struct OllamaProvider {
    #[allow(dead_code)] model_name: String,
    dimension: usize,
    available: bool,
}

impl OllamaProvider {
    pub fn new(model_name: String) -> Self {
        // In a real implementation, this would check if Ollama is running
        // and the model is available
        Self {
            model_name,
            dimension: 384, // Most Ollama embedding models use 384 or 768
            available: false, // Default to false for placeholder
        }
    }

    async fn generate_ollama_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Placeholder implementation
        // In real implementation, this would call Ollama API

        // Generate a simple hash-based embedding for consistency
        let mut embedding = vec![0.0f32; self.dimension];
        let text_bytes = text.as_bytes();

        for (i, chunk) in text_bytes.chunks(4).enumerate() {
            if i >= self.dimension { break; }

            let mut value = 0u32;
            for (j, &byte) in chunk.iter().enumerate() {
                value |= (byte as u32) << (j * 8);
            }

            embedding[i] = (value as f32 / u32::MAX as f32) - 0.5;
        }

        // Normalize
        let magnitude = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        Ok(embedding)
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if !self.available {
            return Err(GraphRAGError::VectorSearch {
                message: "Ollama provider not available".to_string(),
            });
        }
        self.generate_ollama_embedding(text).await
    }

    async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if !self.available {
            return Err(GraphRAGError::VectorSearch {
                message: "Ollama provider not available".to_string(),
            });
        }

        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.generate_ollama_embedding(text).await?);
        }
        Ok(embeddings)
    }

    fn get_embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn get_provider_name(&self) -> &str {
        "ollama"
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

/// Simple hash-based embedding provider (always available fallback)
pub struct HashProvider {
    dimension: usize,
}

impl HashProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn generate_hash_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0f32; self.dimension];

        // Generate multiple hash values to fill the embedding
        for (i, value) in embedding.iter_mut().enumerate().take(self.dimension) {
            let mut hasher = DefaultHasher::new();
            format!("{text}{i}").hash(&mut hasher);
            let hash_value = hasher.finish();

            // Convert to normalized float in range [-1, 1]
            *value = ((hash_value % 1000) as f32 / 500.0) - 1.0;
        }

        // Normalize the vector
        let magnitude = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        embedding
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for HashProvider {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.generate_hash_embedding(text))
    }

    async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| self.generate_hash_embedding(text)).collect())
    }

    fn get_embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn get_provider_name(&self) -> &str {
        "hash"
    }

    fn is_available(&self) -> bool {
        true // Always available
    }
}

/// Hybrid embedder that tries multiple providers with fallback
pub struct HybridEmbedder {
    providers: Vec<Box<dyn EmbeddingProvider>>,
    cache: Arc<std::sync::RwLock<EmbeddingCache>>,
    stats: EmbeddingStats,
    primary_dimension: usize,
}

impl HybridEmbedder {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        let cache = Arc::new(std::sync::RwLock::new(EmbeddingCache::new(config.cache_size)));
        let mut providers: Vec<Box<dyn EmbeddingProvider>> = Vec::new();

        // Determine primary dimension from the first available provider
        let primary_dimension = 384; // Default fallback

        // Always add hash provider as final fallback
        let hash_provider = HashProvider::new(primary_dimension);
        providers.push(Box::new(hash_provider));

        Ok(Self {
            providers,
            cache,
            stats: EmbeddingStats::default(),
            primary_dimension,
        })
    }

    pub async fn initialize(&mut self, config: &EmbeddingConfig) -> Result<()> {
        let mut new_providers: Vec<Box<dyn EmbeddingProvider>> = Vec::new();

        // Try to initialize providers in order of preference
        for provider_type in &config.fallback_providers {
            match provider_type {
                EmbeddingProviderType::Neural => {
                    match NeuralProvider::new(config).await {
                        Ok(provider) => {
                            self.primary_dimension = provider.get_embedding_dimension();
                            println!("✅ Neural embedder initialized ({}D)", self.primary_dimension);
                            new_providers.push(Box::new(provider));
                        }
                        Err(e) => {
                            println!("⚠️  Neural embedder failed: {e}");
                        }
                    }
                }
                EmbeddingProviderType::Ollama => {
                    let provider = OllamaProvider::new(config.model_name.clone());
                    if provider.is_available() {
                        println!("✅ Ollama embedder available");
                        new_providers.push(Box::new(provider));
                    } else {
                        println!("⚠️  Ollama embedder not available");
                    }
                }
                EmbeddingProviderType::Hash => {
                    let provider = HashProvider::new(self.primary_dimension);
                    println!("✅ Hash embedder initialized ({}D)", self.primary_dimension);
                    new_providers.push(Box::new(provider));
                }
            }
        }

        // Ensure we always have at least the hash provider
        if new_providers.is_empty() {
            let provider = HashProvider::new(self.primary_dimension);
            new_providers.push(Box::new(provider));
        }

        self.providers = new_providers;
        Ok(())
    }

    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        let cached_result = {
            let cache_read = self.cache.read().map_err(|_| GraphRAGError::VectorSearch {
                message: "Cache lock error".to_string(),
            })?;

            cache_read.get(text).map(|entry| entry.embedding.clone())
        };

        if let Some(embedding) = cached_result {
            self.stats.cache_hits += 1;
            return Ok(embedding);
        }

        self.stats.cache_misses += 1;

        // Try providers in order
        let start_time = Instant::now();
        for (i, provider) in self.providers.iter_mut().enumerate() {
            if !provider.is_available() {
                continue;
            }

            match provider.embed(text).await {
                Ok(embedding) => {
                    // Update stats
                    self.stats.total_embeddings_generated += 1;
                    let duration = start_time.elapsed();
                    self.stats.average_embedding_time_ms =
                        (self.stats.average_embedding_time_ms * (self.stats.total_embeddings_generated - 1) as f32
                         + duration.as_millis() as f32) / self.stats.total_embeddings_generated as f32;

                    if i > 0 {
                        self.stats.provider_fallbacks += 1;
                        println!("🔄 Fallback to provider: {}", provider.get_provider_name());
                    }

                    // Cache the result
                    {
                        let mut cache_write = self.cache.write().map_err(|_| GraphRAGError::VectorSearch {
                            message: "Cache lock error".to_string(),
                        })?;
                        cache_write.insert(text.to_string(), embedding.clone());
                    }

                    return Ok(embedding);
                }
                Err(e) => {
                    println!("⚠️  Provider {} failed: {}", provider.get_provider_name(), e);
                    continue;
                }
            }
        }

        Err(GraphRAGError::VectorSearch {
            message: "All embedding providers failed".to_string(),
        })
    }

    pub async fn batch_embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for each text
        {
            let cache_read = self.cache.read().map_err(|_| GraphRAGError::VectorSearch {
                message: "Cache lock error".to_string(),
            })?;

            for (i, text) in texts.iter().enumerate() {
                if let Some(entry) = cache_read.get(text.as_ref()) {
                    results.push(Some(entry.embedding));
                    self.stats.cache_hits += 1;
                } else {
                    results.push(None);
                    uncached_texts.push(*text);
                    uncached_indices.push(i);
                    self.stats.cache_misses += 1;
                }
            }
        }

        // Process uncached texts
        if !uncached_texts.is_empty() {
            let start_time = Instant::now();

            for provider in &mut self.providers {
                if !provider.is_available() {
                    continue;
                }

                match provider.batch_embed(&uncached_texts).await {
                    Ok(embeddings) => {
                        // Update stats
                        self.stats.total_embeddings_generated += embeddings.len();
                        self.stats.batch_operations += 1;
                        let duration = start_time.elapsed();
                        self.stats.average_embedding_time_ms =
                            (self.stats.average_embedding_time_ms * (self.stats.total_embeddings_generated - embeddings.len()) as f32
                             + duration.as_millis() as f32) / self.stats.total_embeddings_generated as f32;

                        // Cache and fill results
                        {
                            let mut cache_write = self.cache.write().map_err(|_| GraphRAGError::VectorSearch {
                                message: "Cache lock error".to_string(),
                            })?;

                            for (text, embedding) in uncached_texts.iter().zip(embeddings.iter()) {
                                cache_write.insert(text.to_string(), embedding.clone());
                            }
                        }

                        // Fill in the results
                        for (local_idx, global_idx) in uncached_indices.iter().enumerate() {
                            if local_idx < embeddings.len() {
                                results[*global_idx] = Some(embeddings[local_idx].clone());
                            }
                        }

                        break;
                    }
                    Err(e) => {
                        println!("⚠️  Batch provider {} failed: {}", provider.get_provider_name(), e);
                        continue;
                    }
                }
            }
        }

        // Convert Option<Vec<f32>> to Vec<f32>
        results
            .into_iter()
            .map(|opt| opt.ok_or_else(|| GraphRAGError::VectorSearch {
                message: "Failed to generate embedding".to_string(),
            }))
            .collect()
    }

    pub fn get_embedding_dimension(&self) -> usize {
        self.primary_dimension
    }

    pub fn get_stats(&self) -> &EmbeddingStats {
        &self.stats
    }

    pub fn get_available_providers(&self) -> Vec<String> {
        self.providers
            .iter()
            .filter(|p| p.is_available())
            .map(|p| p.get_provider_name().to_string())
            .collect()
    }

    pub fn clear_cache(&mut self) -> Result<()> {
        let mut cache_write = self.cache.write().map_err(|_| GraphRAGError::VectorSearch {
            message: "Cache lock error".to_string(),
        })?;
        cache_write.clear();
        Ok(())
    }

    pub fn get_cache_stats(&self) -> Result<(usize, usize)> {
        let cache_read = self.cache.read().map_err(|_| GraphRAGError::VectorSearch {
            message: "Cache lock error".to_string(),
        })?;
        Ok(cache_read.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingConfig;

    #[test]
    fn test_hash_provider() {
        let provider = HashProvider::new(384);

        assert!(provider.is_available());
        assert_eq!(provider.get_embedding_dimension(), 384);
        assert_eq!(provider.get_provider_name(), "hash");
    }

    #[tokio::test]
    async fn test_hash_embedding_generation() {
        let mut provider = HashProvider::new(128);

        let embedding = provider.embed("test text").await.unwrap();
        assert_eq!(embedding.len(), 128);

        // Test deterministic nature
        let embedding2 = provider.embed("test text").await.unwrap();
        assert_eq!(embedding, embedding2);

        // Test different inputs produce different outputs
        let embedding3 = provider.embed("different text").await.unwrap();
        assert_ne!(embedding, embedding3);
    }

    #[tokio::test]
    async fn test_hash_batch_embedding() {
        let mut provider = HashProvider::new(64);

        let texts = vec!["text1", "text2", "text3"];
        let embeddings = provider.batch_embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 64);
        }

        // Test consistency with single embeddings
        for (i, text) in texts.iter().enumerate() {
            let single = provider.embed(text).await.unwrap();
            assert_eq!(single, embeddings[i]);
        }
    }

    #[tokio::test]
    async fn test_hybrid_embedder_creation() {
        let config = EmbeddingConfig {
            fallback_providers: vec![EmbeddingProviderType::Hash],
            ..Default::default()
        };

        let mut embedder = HybridEmbedder::new(&config).unwrap();
        embedder.initialize(&config).await.unwrap();

        assert_eq!(embedder.get_embedding_dimension(), 384);
        assert!(!embedder.get_available_providers().is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_embedder_fallback() {
        let config = EmbeddingConfig {
            fallback_providers: vec![
                EmbeddingProviderType::Neural, // Will fail
                EmbeddingProviderType::Ollama,  // Will fail
                EmbeddingProviderType::Hash,    // Will succeed
            ],
            ..Default::default()
        };

        let mut embedder = HybridEmbedder::new(&config).unwrap();
        embedder.initialize(&config).await.unwrap();

        // Should fallback to hash provider
        let embedding = embedder.embed("test text").await.unwrap();
        assert_eq!(embedding.len(), embedder.get_embedding_dimension());

        // Check that fallback was recorded in stats
        let stats = embedder.get_stats();
        assert_eq!(stats.total_embeddings_generated, 1);
    }

    #[tokio::test]
    async fn test_hybrid_embedder_caching() {
        let config = EmbeddingConfig {
            fallback_providers: vec![EmbeddingProviderType::Hash],
            cache_size: 100,
            ..Default::default()
        };

        let mut embedder = HybridEmbedder::new(&config).unwrap();
        embedder.initialize(&config).await.unwrap();

        let text = "test caching";

        // First call should miss cache
        let embedding1 = embedder.embed(text).await.unwrap();
        let stats1 = embedder.get_stats();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);

        // Second call should hit cache
        let embedding2 = embedder.embed(text).await.unwrap();
        let stats2 = embedder.get_stats();
        assert_eq!(stats2.cache_misses, 1);
        assert_eq!(stats2.cache_hits, 1);

        // Embeddings should be identical
        assert_eq!(embedding1, embedding2);
    }
}