//! Semantic Coherence Scoring for Boundary-Aware Chunking
//!
//! This module implements semantic coherence analysis using sentence embeddings
//! to optimize chunk boundaries for maximum semantic unity.
//!
//! Key capabilities:
//! - Cosine similarity calculation between sentence embeddings
//! - Intra-chunk coherence scoring
//! - Optimal split-point detection via binary search
//! - Adaptive threshold based on embedding distances
//!
//! ## References
//!
//! - BAR-RAG Paper: "Boundary-Aware Retrieval-Augmented Generation"
//! - Target: +40% semantic coherence improvement

use crate::core::error::{GraphRAGError, Result};
use crate::embeddings::EmbeddingProvider;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for semantic coherence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Minimum coherence score threshold (0.0-1.0)
    pub min_coherence_threshold: f32,

    /// Maximum sentences per chunk for coherence analysis
    pub max_sentences_per_chunk: usize,

    /// Minimum sentences per chunk
    pub min_sentences_per_chunk: usize,

    /// Window size for local coherence calculation
    pub coherence_window_size: usize,

    /// Weight for adjacent sentence similarity (vs all pairs)
    pub adjacency_weight: f32,

    /// Enable adaptive threshold based on content
    pub adaptive_threshold: bool,

    /// Batch size for embedding generation
    pub embedding_batch_size: usize,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            min_coherence_threshold: 0.65,
            max_sentences_per_chunk: 20,
            min_sentences_per_chunk: 2,
            coherence_window_size: 3,
            adjacency_weight: 0.7,
            adaptive_threshold: true,
            embedding_batch_size: 32,
        }
    }
}

/// Represents a candidate chunk with coherence score
#[derive(Debug, Clone)]
pub struct ScoredChunk {
    /// Text content
    pub text: String,

    /// Start position in original text (byte offset)
    pub start_pos: usize,

    /// End position in original text (byte offset)
    pub end_pos: usize,

    /// Coherence score (0.0-1.0, higher = more coherent)
    pub coherence_score: f32,

    /// Number of sentences in chunk
    pub sentence_count: usize,

    /// Average embedding similarity
    pub avg_similarity: f32,
}

/// Result of split-point optimization
#[derive(Debug, Clone)]
pub struct OptimalSplit {
    /// Split positions (byte offsets)
    pub split_positions: Vec<usize>,

    /// Resulting chunks with scores
    pub chunks: Vec<ScoredChunk>,

    /// Overall coherence score
    pub overall_coherence: f32,

    /// Number of iterations needed
    pub optimization_iterations: usize,
}

/// Semantic coherence scorer using sentence embeddings
pub struct SemanticCoherenceScorer {
    config: CoherenceConfig,
    embedding_provider: Arc<dyn EmbeddingProvider>,
}

impl SemanticCoherenceScorer {
    /// Create a new semantic coherence scorer
    pub fn new(
        config: CoherenceConfig,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            config,
            embedding_provider,
        }
    }

    /// Score the semantic coherence of a text chunk
    ///
    /// Returns a score between 0.0 (incoherent) and 1.0 (highly coherent).
    /// High coherence = high cosine similarity between sentence embeddings.
    pub async fn score_chunk_coherence(&self, text: &str) -> Result<f32> {
        // Split into sentences
        let sentences = self.split_sentences(text);

        if sentences.len() < 2 {
            // Single sentence = perfect coherence
            return Ok(1.0);
        }

        // Limit to max sentences for efficiency
        let sentences: Vec<&str> = sentences
            .iter()
            .take(self.config.max_sentences_per_chunk)
            .map(|s| s.as_str())
            .collect();

        // Generate embeddings for all sentences
        let embeddings = self
            .embedding_provider
            .embed_batch(&sentences)
            .await
            .map_err(|e| GraphRAGError::Embedding {
                message: e.to_string(),
            })?;

        if embeddings.len() != sentences.len() {
            return Err(GraphRAGError::TextProcessing {
                message: "Embedding count mismatch".to_string(),
            });
        }

        // Calculate coherence score
        let coherence = self.calculate_coherence(&embeddings);

        Ok(coherence)
    }

    /// Calculate coherence from sentence embeddings
    ///
    /// Uses a combination of:
    /// 1. Adjacent sentence similarity (weighted higher)
    /// 2. All-pairs average similarity
    fn calculate_coherence(&self, embeddings: &[Vec<f32>]) -> f32 {
        if embeddings.len() < 2 {
            return 1.0;
        }

        // Calculate adjacent sentence similarities
        let mut adjacent_similarities = Vec::new();
        for i in 0..embeddings.len() - 1 {
            let sim = self.cosine_similarity(&embeddings[i], &embeddings[i + 1]);
            adjacent_similarities.push(sim);
        }

        let adjacent_avg = adjacent_similarities.iter().sum::<f32>()
            / adjacent_similarities.len() as f32;

        // Calculate window-based similarities
        let window_avg = if self.config.coherence_window_size > 1 {
            let mut window_similarities = Vec::new();
            for i in 0..embeddings.len() {
                let window_start = i.saturating_sub(self.config.coherence_window_size / 2);
                let window_end =
                    (i + self.config.coherence_window_size / 2 + 1).min(embeddings.len());

                for j in window_start..window_end {
                    if i != j {
                        let sim = self.cosine_similarity(&embeddings[i], &embeddings[j]);
                        window_similarities.push(sim);
                    }
                }
            }

            if window_similarities.is_empty() {
                adjacent_avg
            } else {
                window_similarities.iter().sum::<f32>() / window_similarities.len() as f32
            }
        } else {
            adjacent_avg
        };

        // Weighted combination
        let coherence = self.config.adjacency_weight * adjacent_avg
            + (1.0 - self.config.adjacency_weight) * window_avg;

        coherence.clamp(0.0, 1.0)
    }

    /// Find optimal split points in text to maximize chunk coherence
    ///
    /// Uses a greedy algorithm:
    /// 1. Start with no splits
    /// 2. Try all candidate split points
    /// 3. Pick split that maximizes average chunk coherence
    /// 4. Repeat until coherence stops improving
    pub async fn find_optimal_split(
        &self,
        text: &str,
        candidate_boundaries: &[usize],
    ) -> Result<OptimalSplit> {
        if candidate_boundaries.is_empty() {
            // No boundaries = single chunk
            let score = self.score_chunk_coherence(text).await?;
            return Ok(OptimalSplit {
                split_positions: vec![],
                chunks: vec![ScoredChunk {
                    text: text.to_string(),
                    start_pos: 0,
                    end_pos: text.len(),
                    coherence_score: score,
                    sentence_count: self.split_sentences(text).len(),
                    avg_similarity: score,
                }],
                overall_coherence: score,
                optimization_iterations: 1,
            });
        }

        // Greedy split optimization
        let mut current_splits: Vec<usize> = vec![];
        let mut iterations = 0;
        let max_iterations = 100;

        loop {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            // Generate candidate chunks with current splits
            let current_chunks = self.create_chunks(text, &current_splits).await?;
            let current_score = current_chunks.iter().map(|c| c.coherence_score).sum::<f32>()
                / current_chunks.len() as f32;

            // Try adding each candidate boundary
            let mut best_new_split: Option<usize> = None;
            let mut best_score = current_score;

            for &boundary in candidate_boundaries {
                if current_splits.contains(&boundary) {
                    continue;
                }

                // Try this split
                let mut test_splits = current_splits.clone();
                test_splits.push(boundary);
                test_splits.sort_unstable();

                let test_chunks = self.create_chunks(text, &test_splits).await?;
                let test_score = test_chunks.iter().map(|c| c.coherence_score).sum::<f32>()
                    / test_chunks.len() as f32;

                if test_score > best_score {
                    best_score = test_score;
                    best_new_split = Some(boundary);
                }
            }

            // If no improvement, stop
            if best_new_split.is_none() {
                break;
            }

            // Add best split
            current_splits.push(best_new_split.unwrap());
            current_splits.sort_unstable();

            // Check minimum chunk size constraint
            if !self.validate_splits(text, &current_splits) {
                current_splits.pop();
                break;
            }
        }

        // Generate final chunks
        let final_chunks = self.create_chunks(text, &current_splits).await?;
        let overall_coherence = final_chunks.iter().map(|c| c.coherence_score).sum::<f32>()
            / final_chunks.len() as f32;

        Ok(OptimalSplit {
            split_positions: current_splits,
            chunks: final_chunks,
            overall_coherence,
            optimization_iterations: iterations,
        })
    }

    /// Create scored chunks from text and split positions
    async fn create_chunks(&self, text: &str, splits: &[usize]) -> Result<Vec<ScoredChunk>> {
        let mut chunks = Vec::new();
        let mut boundaries = vec![0];
        boundaries.extend_from_slice(splits);
        boundaries.push(text.len());

        for i in 0..boundaries.len() - 1 {
            let start = boundaries[i];
            let end = boundaries[i + 1];
            let chunk_text = &text[start..end];

            let coherence = self.score_chunk_coherence(chunk_text).await?;
            let sentences = self.split_sentences(chunk_text);

            chunks.push(ScoredChunk {
                text: chunk_text.to_string(),
                start_pos: start,
                end_pos: end,
                coherence_score: coherence,
                sentence_count: sentences.len(),
                avg_similarity: coherence,
            });
        }

        Ok(chunks)
    }

    /// Validate that splits create chunks meeting minimum size requirements
    fn validate_splits(&self, text: &str, splits: &[usize]) -> bool {
        let mut boundaries = vec![0];
        boundaries.extend_from_slice(splits);
        boundaries.push(text.len());

        for i in 0..boundaries.len() - 1 {
            let start = boundaries[i];
            let end = boundaries[i + 1];
            let chunk_text = &text[start..end];
            let sentences = self.split_sentences(chunk_text);

            if sentences.len() < self.config.min_sentences_per_chunk {
                return false;
            }
        }

        true
    }

    /// Calculate cosine similarity between two embedding vectors
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Split text into sentences (simple implementation)
    ///
    /// This is a basic sentence splitter. For production, consider using
    /// a dedicated NLP library or more sophisticated tokenization.
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            current_sentence.push(ch);

            // Check for sentence endings
            if matches!(ch, '.' | '!' | '?') {
                // Look ahead for whitespace
                if let Some(&next_ch) = chars.peek() {
                    if next_ch.is_whitespace() || next_ch == '\n' {
                        let trimmed = current_sentence.trim();
                        if !trimmed.is_empty() && trimmed.len() > 3 {
                            sentences.push(trimmed.to_string());
                            current_sentence.clear();
                        }
                    }
                } else {
                    // End of text
                    let trimmed = current_sentence.trim();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed.to_string());
                        current_sentence.clear();
                    }
                }
            }
        }

        // Add remaining text as final sentence
        let trimmed = current_sentence.trim();
        if !trimmed.is_empty() && trimmed.len() > 3 {
            sentences.push(trimmed.to_string());
        }

        sentences
    }

    /// Calculate adaptive threshold based on content characteristics
    pub fn calculate_adaptive_threshold(&self, text: &str) -> f32 {
        if !self.config.adaptive_threshold {
            return self.config.min_coherence_threshold;
        }

        let sentences = self.split_sentences(text);
        let sentence_count = sentences.len();

        // Adjust threshold based on document characteristics
        let base_threshold = self.config.min_coherence_threshold;

        // Longer documents = slightly more tolerant
        let length_factor = (sentence_count as f32 / 50.0).min(1.0);
        let adjusted = base_threshold - (length_factor * 0.05);

        adjusted.clamp(0.5, 0.9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingProvider;
    use async_trait::async_trait;
    use std::sync::Arc;

    /// Mock embedding provider for testing
    struct MockEmbeddingProvider {
        dimension: usize,
    }

    impl MockEmbeddingProvider {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Generate deterministic embedding based on text length and content
            let mut embedding = vec![0.0; self.dimension];
            let hash = text.len() as f32;
            for (i, val) in embedding.iter_mut().enumerate() {
                *val = ((hash + i as f32) * 0.1).sin();
            }
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut embedding {
                *val /= norm;
            }
            Ok(embedding)
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut results = Vec::new();
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }

        fn dimensions(&self) -> usize {
            self.dimension
        }

        fn is_available(&self) -> bool {
            true
        }

        fn provider_name(&self) -> &str {
            "MockProvider"
        }
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        // Identical vectors = 1.0
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let sim = scorer.cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors = 0.0
        let v3 = vec![1.0, 0.0, 0.0];
        let v4 = vec![0.0, 1.0, 0.0];
        let sim = scorer.cosine_similarity(&v3, &v4);
        assert!(sim.abs() < 0.001);

        // Opposite vectors = -1.0
        let v5 = vec![1.0, 0.0, 0.0];
        let v6 = vec![-1.0, 0.0, 0.0];
        let sim = scorer.cosine_similarity(&v5, &v6);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_sentence_splitting() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        let text = "This is sentence one. This is sentence two! Is this sentence three?";
        let sentences = scorer.split_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("sentence one"));
        assert!(sentences[1].contains("sentence two"));
        assert!(sentences[2].contains("sentence three"));
    }

    #[tokio::test]
    async fn test_score_chunk_coherence() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        let text = "This is a test. This is another test. Testing continues here.";
        let score = scorer.score_chunk_coherence(text).await.unwrap();

        // Should return a valid score between 0 and 1
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_single_sentence_coherence() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        let text = "This is a single sentence.";
        let score = scorer.score_chunk_coherence(text).await.unwrap();

        // Single sentence = perfect coherence
        assert_eq!(score, 1.0);
    }

    #[tokio::test]
    async fn test_find_optimal_split_no_boundaries() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        let text = "First sentence. Second sentence. Third sentence.";
        let result = scorer.find_optimal_split(text, &[]).await.unwrap();

        // No boundaries = single chunk
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.split_positions.len(), 0);
    }

    #[tokio::test]
    async fn test_create_chunks() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        let text = "First part. Second part. Third part.";
        let splits = vec![12, 25]; // Split after "First part." and "Second part."

        let chunks = scorer.create_chunks(text, &splits).await.unwrap();

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[1].text.contains("Second"));
        assert!(chunks[2].text.contains("Third"));
    }

    #[tokio::test]
    async fn test_validate_splits() {
        let config = CoherenceConfig {
            min_sentences_per_chunk: 2,
            ..Default::default()
        };
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        let text =
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.";

        // Valid splits (each chunk has 2+ sentences)
        let splits = vec![26]; // After "Sentence two."
        assert!(scorer.validate_splits(text, &splits));

        // Invalid splits (would create chunk with 1 sentence)
        let splits = vec![14]; // After "Sentence one." only
        assert!(!scorer.validate_splits(text, &splits));
    }

    #[tokio::test]
    async fn test_adaptive_threshold() {
        let config = CoherenceConfig {
            adaptive_threshold: true,
            ..Default::default()
        };
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        // Short text
        let short_text = "One. Two. Three.";
        let threshold_short = scorer.calculate_adaptive_threshold(short_text);

        // Long text
        let long_text = (0..100)
            .map(|i| format!("Sentence {}.", i))
            .collect::<Vec<_>>()
            .join(" ");
        let threshold_long = scorer.calculate_adaptive_threshold(&long_text);

        // Longer text should have slightly lower threshold (more tolerant)
        assert!(threshold_long <= threshold_short);
        assert!(threshold_short >= 0.5 && threshold_short <= 0.9);
        assert!(threshold_long >= 0.5 && threshold_long <= 0.9);
    }

    #[tokio::test]
    async fn test_coherence_calculation() {
        let config = CoherenceConfig::default();
        let provider = Arc::new(MockEmbeddingProvider::new(384));
        let scorer = SemanticCoherenceScorer::new(config, provider);

        // Similar embeddings (high coherence)
        let emb1 = vec![1.0, 0.1, 0.1];
        let emb2 = vec![0.9, 0.15, 0.15];
        let emb3 = vec![0.95, 0.12, 0.12];
        let embeddings = vec![emb1, emb2, emb3];

        let coherence = scorer.calculate_coherence(&embeddings);
        assert!(coherence > 0.5); // Should be high

        // Dissimilar embeddings (low coherence)
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0];
        let emb3 = vec![0.0, 0.0, 1.0];
        let embeddings = vec![emb1, emb2, emb3];

        let coherence = scorer.calculate_coherence(&embeddings);
        assert!(coherence < 0.5); // Should be low
    }
}
