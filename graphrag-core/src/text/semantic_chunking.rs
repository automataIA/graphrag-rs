//! Semantic Chunking for RAG
//!
//! This module implements semantic chunking that splits text based on
//! semantic similarity rather than fixed character/token counts.
//!
//! Key innovation: Uses sentence embeddings and cosine similarity to
//! determine natural breakpoints, creating semantically cohesive chunks.
//!
//! Reference: LangChain SemanticChunker, Greg Kamradt's 5 Levels of Text Splitting

use crate::core::Result;
use crate::vector::EmbeddingGenerator;

/// Chunk of semantically similar sentences
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    /// The text content of the chunk
    pub content: String,

    /// Start sentence index
    pub start_sentence: usize,

    /// End sentence index (exclusive)
    pub end_sentence: usize,

    /// Number of sentences in this chunk
    pub sentence_count: usize,
}

/// Strategy for determining chunk breakpoints
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BreakpointStrategy {
    /// Use percentile of similarity differences (e.g., 95th percentile)
    Percentile,

    /// Use standard deviation of similarity differences (e.g., 3σ)
    StandardDeviation,

    /// Use absolute threshold (e.g., similarity < 0.5)
    Absolute,
}

/// Configuration for semantic chunking
#[derive(Debug, Clone)]
pub struct SemanticChunkerConfig {
    /// Strategy for determining breakpoints
    pub breakpoint_strategy: BreakpointStrategy,

    /// Threshold amount:
    /// - Percentile: 0-100 (default: 95.0)
    /// - StandardDeviation: number of std devs (default: 3.0)
    /// - Absolute: similarity threshold (default: 0.5)
    pub threshold_amount: f32,

    /// Minimum chunk size in sentences
    pub min_chunk_size: usize,

    /// Maximum chunk size in sentences (0 = unlimited)
    pub max_chunk_size: usize,

    /// Buffer size for comparing sentences (default: 1 = compare consecutive)
    pub buffer_size: usize,
}

impl Default for SemanticChunkerConfig {
    fn default() -> Self {
        Self {
            breakpoint_strategy: BreakpointStrategy::Percentile,
            threshold_amount: 95.0,
            min_chunk_size: 1,
            max_chunk_size: 0, // unlimited
            buffer_size: 1,
        }
    }
}

/// Semantic text chunker that splits based on embedding similarity
pub struct SemanticChunker {
    config: SemanticChunkerConfig,
    embedding_generator: EmbeddingGenerator,
}

impl SemanticChunker {
    /// Create a new semantic chunker
    pub fn new(config: SemanticChunkerConfig, embedding_generator: EmbeddingGenerator) -> Self {
        Self {
            config,
            embedding_generator,
        }
    }

    /// Split text into semantic chunks
    pub fn chunk(&mut self, text: &str) -> Result<Vec<SemanticChunk>> {
        // 1. Split into sentences
        let sentences = self.split_sentences(text);

        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        if sentences.len() == 1 {
            return Ok(vec![SemanticChunk {
                content: text.to_string(),
                start_sentence: 0,
                end_sentence: 1,
                sentence_count: 1,
            }]);
        }

        // 2. Generate embeddings for each sentence
        let embeddings = self.embed_sentences(&sentences)?;

        // 3. Calculate similarity differences between consecutive sentences
        let similarity_diffs = self.calculate_similarity_differences(&embeddings);

        // 4. Determine breakpoints based on strategy
        let breakpoints = self.determine_breakpoints(&similarity_diffs)?;

        // 5. Create chunks from sentences using breakpoints
        let chunks = self.create_chunks(&sentences, &breakpoints);

        Ok(chunks)
    }

    /// Split text into sentences using simple sentence tokenization
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                if !current_sentence.is_empty() {
                    sentences.push(current_sentence.clone());
                    current_sentence.clear();
                }
                continue;
            }

            // Split on sentence boundaries: . ! ?
            for part in line.split_inclusive(&['.', '!', '?']) {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }

                current_sentence.push_str(part);
                current_sentence.push(' ');

                // Check if this looks like end of sentence
                if part.ends_with('.') || part.ends_with('!') || part.ends_with('?') {
                    sentences.push(current_sentence.trim().to_string());
                    current_sentence.clear();
                }
            }
        }

        // Add any remaining text
        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }

        sentences
    }

    /// Generate embeddings for all sentences
    fn embed_sentences(&mut self, sentences: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for sentence in sentences {
            let embedding = self.embedding_generator.generate_embedding(sentence);
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Calculate cosine similarity differences between consecutive sentences
    fn calculate_similarity_differences(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        let mut diffs = Vec::new();

        for i in 0..embeddings.len().saturating_sub(self.config.buffer_size) {
            let sim = self.cosine_similarity(&embeddings[i], &embeddings[i + self.config.buffer_size]);

            // Convert similarity to difference (distance)
            // Higher distance = more dissimilar = potential breakpoint
            let distance = 1.0 - sim;
            diffs.push(distance);
        }

        diffs
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    /// Determine chunk breakpoints based on similarity differences
    fn determine_breakpoints(&self, diffs: &[f32]) -> Result<Vec<usize>> {
        if diffs.is_empty() {
            return Ok(Vec::new());
        }

        let threshold = match self.config.breakpoint_strategy {
            BreakpointStrategy::Percentile => self.calculate_percentile_threshold(diffs),
            BreakpointStrategy::StandardDeviation => self.calculate_std_threshold(diffs),
            BreakpointStrategy::Absolute => self.config.threshold_amount,
        };

        // Find indices where difference exceeds threshold
        let mut breakpoints = Vec::new();
        for (i, &diff) in diffs.iter().enumerate() {
            if diff > threshold {
                // +1 because diff[i] is between sentence[i] and sentence[i+1]
                breakpoints.push(i + 1);
            }
        }

        Ok(breakpoints)
    }

    /// Calculate threshold based on percentile
    fn calculate_percentile_threshold(&self, diffs: &[f32]) -> f32 {
        let mut sorted = diffs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = self.config.threshold_amount / 100.0;
        let index = ((sorted.len() as f32 * percentile) as usize).min(sorted.len() - 1);

        sorted[index]
    }

    /// Calculate threshold based on standard deviation
    fn calculate_std_threshold(&self, diffs: &[f32]) -> f32 {
        let mean: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;

        let variance: f32 = diffs.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / diffs.len() as f32;

        let std_dev = variance.sqrt();

        mean + (self.config.threshold_amount * std_dev)
    }

    /// Create chunks from sentences using breakpoints
    fn create_chunks(&self, sentences: &[String], breakpoints: &[usize]) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();
        let mut start_idx = 0;

        let mut all_breakpoints = breakpoints.to_vec();
        all_breakpoints.push(sentences.len()); // Add final breakpoint

        for &end_idx in &all_breakpoints {
            if end_idx <= start_idx {
                continue;
            }

            let sentence_count = end_idx - start_idx;

            // Check size constraints
            if sentence_count < self.config.min_chunk_size {
                continue;
            }

            if self.config.max_chunk_size > 0 && sentence_count > self.config.max_chunk_size {
                // Split large chunk into smaller ones
                let mut sub_start = start_idx;
                while sub_start < end_idx {
                    let sub_end = (sub_start + self.config.max_chunk_size).min(end_idx);
                    let content = sentences[sub_start..sub_end].join(" ");

                    chunks.push(SemanticChunk {
                        content,
                        start_sentence: sub_start,
                        end_sentence: sub_end,
                        sentence_count: sub_end - sub_start,
                    });

                    sub_start = sub_end;
                }
            } else {
                let content = sentences[start_idx..end_idx].join(" ");

                chunks.push(SemanticChunk {
                    content,
                    start_sentence: start_idx,
                    end_sentence: end_idx,
                    sentence_count,
                });
            }

            start_idx = end_idx;
        }

        chunks
    }

    /// Get configuration
    pub fn config(&self) -> &SemanticChunkerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_splitting() {
        let config = SemanticChunkerConfig::default();
        let embedding_gen = EmbeddingGenerator::new(384); // Use simple hash-based for testing
        let chunker = SemanticChunker::new(config, embedding_gen);

        let text = "This is sentence one. This is sentence two! Is this sentence three?";
        let sentences = chunker.split_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("sentence one"));
        assert!(sentences[1].contains("sentence two"));
        assert!(sentences[2].contains("sentence three"));
    }

    #[test]
    fn test_cosine_similarity() {
        let config = SemanticChunkerConfig::default();
        let embedding_gen = EmbeddingGenerator::new(384);
        let chunker = SemanticChunker::new(config, embedding_gen);

        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = chunker.cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = chunker.cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);

        // Opposite vectors
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = chunker.cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_threshold() {
        let config = SemanticChunkerConfig {
            breakpoint_strategy: BreakpointStrategy::Percentile,
            threshold_amount: 95.0,
            ..Default::default()
        };
        let embedding_gen = EmbeddingGenerator::new(384);
        let chunker = SemanticChunker::new(config, embedding_gen);

        let diffs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let threshold = chunker.calculate_percentile_threshold(&diffs);

        // 95th percentile of 10 values should be around 0.95
        assert!(threshold >= 0.9);
    }

    #[test]
    fn test_std_threshold() {
        let config = SemanticChunkerConfig {
            breakpoint_strategy: BreakpointStrategy::StandardDeviation,
            threshold_amount: 3.0,
            ..Default::default()
        };
        let embedding_gen = EmbeddingGenerator::new(384);
        let chunker = SemanticChunker::new(config, embedding_gen);

        let diffs = vec![0.5, 0.5, 0.5, 0.5, 0.5]; // All same = zero std dev
        let threshold = chunker.calculate_std_threshold(&diffs);

        assert!((threshold - 0.5).abs() < 0.001); // Should be mean when std=0
    }

    #[test]
    fn test_semantic_chunking_basic() {
        let config = SemanticChunkerConfig {
            breakpoint_strategy: BreakpointStrategy::Percentile,
            threshold_amount: 50.0, // Lower threshold for testing
            min_chunk_size: 1,
            max_chunk_size: 0,
            buffer_size: 1,
        };

        let embedding_gen = EmbeddingGenerator::new(384);
        let mut chunker = SemanticChunker::new(config, embedding_gen);

        let text = "Alice loves programming. Bob also codes daily. \
                    The weather is sunny. Rain is expected tomorrow.";

        let chunks = chunker.chunk(text).unwrap();

        // Should create at least 1 chunk
        assert!(!chunks.is_empty());

        // Each chunk should have content
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.sentence_count > 0);
        }
    }
}
