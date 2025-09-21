//! Ollama-based embeddings for GraphRAG

use super::{OllamaClient, OllamaConfig, OllamaError, OllamaResult};
use crate::vector::EmbeddingGenerator;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use std::collections::HashMap;
use std::time::Instant;

/// Embedding generator using Ollama local models
pub struct OllamaEmbeddings {
    client: OllamaClient,
    cache: HashMap<String, Vec<f32>>,
    stats: EmbeddingStats,
    fallback_generator: Option<EmbeddingGenerator>,
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddingStats {
    pub requests_made: usize,
    pub cache_hits: usize,
    pub total_response_time: std::time::Duration,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub fallback_used: usize,
}

impl EmbeddingStats {
    pub fn record_request(
        &mut self,
        duration: std::time::Duration,
        success: bool,
        from_cache: bool,
    ) {
        if from_cache {
            self.cache_hits += 1;
        } else {
            self.requests_made += 1;
            self.total_response_time += duration;

            if success {
                self.successful_requests += 1;
            } else {
                self.failed_requests += 1;
            }
        }
    }

    pub fn record_fallback(&mut self) {
        self.fallback_used += 1;
    }

    pub fn average_response_time(&self) -> std::time::Duration {
        if self.requests_made > 0 {
            self.total_response_time / self.requests_made as u32
        } else {
            std::time::Duration::ZERO
        }
    }

    pub fn cache_hit_rate(&self) -> f32 {
        let total_queries = self.requests_made + self.cache_hits;
        if total_queries > 0 {
            self.cache_hits as f32 / total_queries as f32
        } else {
            0.0
        }
    }

    pub fn success_rate(&self) -> f32 {
        if self.requests_made > 0 {
            self.successful_requests as f32 / self.requests_made as f32
        } else {
            0.0
        }
    }
}

impl OllamaEmbeddings {
    /// Create new Ollama embeddings generator
    pub fn new(config: OllamaConfig) -> OllamaResult<Self> {
        let client = OllamaClient::new(config.clone())?;

        // Create fallback generator if fallback is enabled
        let fallback_generator = if config.fallback_to_hash {
            Some(EmbeddingGenerator::new(384)) // Default dimension for hash-based
        } else {
            None
        };

        Ok(Self {
            client,
            cache: HashMap::new(),
            stats: EmbeddingStats::default(),
            fallback_generator,
        })
    }

    /// Generate embedding for text using Ollama
    pub async fn generate_embedding_async(&mut self, text: &str) -> OllamaResult<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            self.stats
                .record_request(std::time::Duration::ZERO, true, true);
            return Ok(cached.clone());
        }

        let start_time = Instant::now();

        // Create embedding request
        let request = GenerateEmbeddingsRequest::new(
            self.client.embedding_model().to_string(),
            text.to_string().into(),
        );

        // Execute request with timeout
        let response = match self
            .client
            .with_timeout(self.client.inner().generate_embeddings(request))
            .await
        {
            Ok(response) => {
                let duration = start_time.elapsed();
                self.stats.record_request(duration, true, false);
                response
            }
            Err(e) => {
                let duration = start_time.elapsed();
                self.stats.record_request(duration, false, false);

                // Try fallback if available
                if let Some(ref mut fallback) = self.fallback_generator {
                    self.stats.record_fallback();
                    let embedding = fallback.generate_embedding(text);
                    self.cache.insert(text.to_string(), embedding.clone());
                    return Ok(embedding);
                }

                return Err(e);
            }
        };

        // Extract embedding from response
        let embedding = response
            .embeddings
            .first()
            .ok_or_else(|| OllamaError::ApiError("No embeddings in response".to_string()))?
            .clone();

        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn batch_generate_async(&mut self, texts: &[&str]) -> OllamaResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.generate_embedding_async(text).await?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Generate embedding (sync wrapper for compatibility)
    pub fn generate_embedding(&mut self, text: &str) -> Vec<f32> {
        // Use tokio to run async code in sync context
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => {
                // Fallback to hash-based if runtime creation fails
                if let Some(ref mut fallback) = self.fallback_generator {
                    self.stats.record_fallback();
                    return fallback.generate_embedding(text);
                }
                return vec![0.0; 384]; // Return zero vector as last resort
            }
        };

        rt.block_on(async {
            self.generate_embedding_async(text)
                .await
                .unwrap_or_else(|_| {
                    // Final fallback
                    if let Some(ref mut fallback) = self.fallback_generator {
                        self.stats.record_fallback();
                        fallback.generate_embedding(text)
                    } else {
                        vec![0.0; 384]
                    }
                })
        })
    }

    /// Generate batch embeddings (sync wrapper)
    pub fn batch_generate(&mut self, texts: &[&str]) -> Vec<Vec<f32>> {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => {
                // Fallback to hash-based
                if let Some(ref mut fallback) = self.fallback_generator {
                    self.stats.fallback_used += texts.len();
                    return fallback.batch_generate(texts);
                }
                return texts.iter().map(|_| vec![0.0; 384]).collect();
            }
        };

        rt.block_on(async {
            self.batch_generate_async(texts).await.unwrap_or_else(|_| {
                // Fallback
                if let Some(ref mut fallback) = self.fallback_generator {
                    self.stats.fallback_used += texts.len();
                    fallback.batch_generate(texts)
                } else {
                    texts.iter().map(|_| vec![0.0; 384]).collect()
                }
            })
        })
    }

    /// Get the expected embedding dimension from Ollama model
    pub async fn get_dimension(&self) -> OllamaResult<usize> {
        // Generate a test embedding to determine dimension
        let test_request = GenerateEmbeddingsRequest::new(
            self.client.embedding_model().to_string(),
            "test".to_string().into(),
        );

        let response = self
            .client
            .with_timeout(self.client.inner().generate_embeddings(test_request))
            .await?;

        let dimension = response
            .embeddings
            .first()
            .map(|emb| emb.len())
            .ok_or_else(|| OllamaError::ApiError("No embeddings in response".to_string()))?;

        Ok(dimension)
    }

    /// Check if Ollama service is available for embeddings
    pub async fn is_available(&self) -> bool {
        self.client.health_check().await.unwrap_or(false)
    }

    /// Validate that embedding model is available
    pub async fn validate_setup(&self) -> OllamaResult<()> {
        if !self
            .client
            .model_exists(self.client.embedding_model())
            .await?
        {
            return Err(OllamaError::ModelNotFound(
                self.client.embedding_model().to_string(),
            ));
        }
        Ok(())
    }

    /// Get embedding statistics
    pub fn get_stats(&self) -> &EmbeddingStats {
        &self.stats
    }

    /// Clear embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get cached embeddings count
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Enable or disable fallback
    pub fn set_fallback_enabled(&mut self, enabled: bool, dimension: Option<usize>) {
        if enabled && self.fallback_generator.is_none() {
            let dim = dimension.unwrap_or(384);
            self.fallback_generator = Some(EmbeddingGenerator::new(dim));
        } else if !enabled {
            self.fallback_generator = None;
        }
    }
}

impl Clone for OllamaEmbeddings {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            cache: HashMap::new(), // Don't clone cache to avoid memory issues
            stats: EmbeddingStats::default(), // Reset stats for cloned instance
            fallback_generator: self
                .fallback_generator
                .as_ref()
                .map(|fg| EmbeddingGenerator::new(fg.dimension())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings_creation() {
        let config = OllamaConfig::default();
        let embeddings = OllamaEmbeddings::new(config);
        assert!(embeddings.is_ok());
    }

    #[test]
    fn test_sync_embedding_generation() {
        let config = OllamaConfig::default();
        let mut embeddings = OllamaEmbeddings::new(config).unwrap();

        // This will use fallback if Ollama is not available
        let embedding = embeddings.generate_embedding("test text");
        assert!(!embedding.is_empty());
        println!("Generated embedding dimension: {}", embedding.len());
    }

    #[test]
    fn test_batch_generation() {
        let config = OllamaConfig::default();
        let mut embeddings = OllamaEmbeddings::new(config).unwrap();

        let texts = vec!["text 1", "text 2", "text 3"];
        let results = embeddings.batch_generate(&texts);

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|emb| !emb.is_empty()));
    }

    #[tokio::test]
    async fn test_availability_check() {
        let config = OllamaConfig::default();
        let embeddings = OllamaEmbeddings::new(config).unwrap();

        let available = embeddings.is_available().await;
        println!("Ollama embeddings available: {available}");
    }

    #[tokio::test]
    async fn test_dimension_detection() {
        let config = OllamaConfig::enabled();
        if let Ok(embeddings) = OllamaEmbeddings::new(config) {
            if embeddings.is_available().await {
                match embeddings.get_dimension().await {
                    Ok(dim) => println!("Ollama embedding dimension: {dim}"),
                    Err(e) => println!("Could not get dimension: {e}"),
                }
            }
        }
    }
}
