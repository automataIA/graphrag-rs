//! Semantic Chunking
//!
//! This module provides intelligent text chunking strategies that respect:
//! - Sentence boundaries
//! - Paragraph structure
//! - Topic coherence
//! - Semantic similarity
//!
//! ## Chunking Strategies
//!
//! 1. **Sentence-based**: Chunks at sentence boundaries
//! 2. **Paragraph-based**: Chunks at paragraph breaks
//! 3. **Topic-based**: Chunks when topic shifts detected
//! 4. **Semantic**: Chunks based on embedding similarity
//! 5. **Hybrid**: Combines multiple strategies

use serde::{Deserialize, Serialize};

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks (character count)
    FixedSize,
    /// Sentence boundary-based chunks
    Sentence,
    /// Paragraph boundary-based chunks
    Paragraph,
    /// Topic shift detection
    Topic,
    /// Semantic similarity-based
    Semantic,
    /// Hybrid approach (combines multiple strategies)
    Hybrid,
}

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Chunking strategy
    pub strategy: ChunkingStrategy,
    /// Target chunk size (characters or sentences)
    pub target_size: usize,
    /// Minimum chunk size
    pub min_size: usize,
    /// Maximum chunk size
    pub max_size: usize,
    /// Overlap size (characters or sentences)
    pub overlap: usize,
    /// Similarity threshold for semantic chunking (0.0 to 1.0)
    pub similarity_threshold: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::Sentence,
            target_size: 500,
            min_size: 100,
            max_size: 1000,
            overlap: 50,
            similarity_threshold: 0.7,
        }
    }
}

/// Text chunk with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    /// Chunk text
    pub text: String,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Sentence count in chunk
    pub sentence_count: usize,
    /// Paragraph count in chunk
    pub paragraph_count: usize,
    /// Coherence score (higher = more coherent)
    pub coherence: f32,
}

/// Semantic chunker
pub struct SemanticChunker {
    config: ChunkingConfig,
}

impl SemanticChunker {
    /// Create new semantic chunker
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: ChunkingConfig::default(),
        }
    }

    /// Chunk text according to strategy
    pub fn chunk(&self, text: &str) -> Vec<SemanticChunk> {
        match self.config.strategy {
            ChunkingStrategy::FixedSize => self.chunk_fixed_size(text),
            ChunkingStrategy::Sentence => self.chunk_by_sentences(text),
            ChunkingStrategy::Paragraph => self.chunk_by_paragraphs(text),
            ChunkingStrategy::Topic => self.chunk_by_topic(text),
            ChunkingStrategy::Semantic => self.chunk_by_similarity(text),
            ChunkingStrategy::Hybrid => self.chunk_hybrid(text),
        }
    }

    /// Fixed-size chunking (baseline)
    fn chunk_fixed_size(&self, text: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_len = chars.len();
        let mut start = 0;

        while start < total_len {
            let end = (start + self.config.target_size).min(total_len);
            let chunk_text: String = chars[start..end].iter().collect();

            chunks.push(SemanticChunk {
                text: chunk_text,
                start,
                end,
                sentence_count: 0, // Will be calculated if needed
                paragraph_count: 0,
                coherence: 1.0,
            });

            start += self.config.target_size - self.config.overlap;
        }

        chunks
    }

    /// Sentence-based chunking
    fn chunk_by_sentences(&self, text: &str) -> Vec<SemanticChunk> {
        let sentences = self.split_sentences(text);
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        let mut chunk_start = 0;

        for sentence in sentences.iter() {
            let sentence_len = sentence.len();

            // Check if adding this sentence exceeds max size
            if current_size + sentence_len > self.config.max_size && !current_chunk.is_empty() {
                // Create chunk from accumulated sentences
                let chunk_text = current_chunk.join(" ");
                let chunk_end = chunk_start + chunk_text.len();

                chunks.push(SemanticChunk {
                    text: chunk_text,
                    start: chunk_start,
                    end: chunk_end,
                    sentence_count: current_chunk.len(),
                    paragraph_count: self.count_paragraphs(&current_chunk.join(" ")),
                    coherence: self.calculate_coherence(&current_chunk),
                });

                // Start new chunk with overlap
                let overlap_sentences = if current_chunk.len() > 1 {
                    vec![current_chunk.last().unwrap().clone()]
                } else {
                    Vec::new()
                };

                chunk_start = chunk_end - overlap_sentences.join(" ").len();
                current_chunk = overlap_sentences;
                current_size = current_chunk.iter().map(|s| s.len()).sum();
            }

            current_chunk.push(sentence.clone());
            current_size += sentence_len;

            // Check if we've reached target size
            if current_size >= self.config.target_size {
                let chunk_text = current_chunk.join(" ");
                let chunk_end = chunk_start + chunk_text.len();

                chunks.push(SemanticChunk {
                    text: chunk_text,
                    start: chunk_start,
                    end: chunk_end,
                    sentence_count: current_chunk.len(),
                    paragraph_count: self.count_paragraphs(&current_chunk.join(" ")),
                    coherence: self.calculate_coherence(&current_chunk),
                });

                // Start new chunk with overlap
                let overlap_sentences = if current_chunk.len() > 1 {
                    vec![current_chunk.last().unwrap().clone()]
                } else {
                    Vec::new()
                };

                chunk_start = chunk_end - overlap_sentences.join(" ").len();
                current_chunk = overlap_sentences;
                current_size = current_chunk.iter().map(|s| s.len()).sum();
            }
        }

        // Add remaining sentences as final chunk
        if !current_chunk.is_empty() && current_chunk.join(" ").len() >= self.config.min_size {
            let chunk_text = current_chunk.join(" ");
            let chunk_end = chunk_start + chunk_text.len();

            chunks.push(SemanticChunk {
                text: chunk_text,
                start: chunk_start,
                end: chunk_end,
                sentence_count: current_chunk.len(),
                paragraph_count: self.count_paragraphs(&current_chunk.join(" ")),
                coherence: self.calculate_coherence(&current_chunk),
            });
        }

        chunks
    }

    /// Paragraph-based chunking
    fn chunk_by_paragraphs(&self, text: &str) -> Vec<SemanticChunk> {
        let paragraphs: Vec<&str> = text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;
        let mut chunk_start = 0;

        for paragraph in paragraphs {
            let para_len = paragraph.len();

            if current_size + para_len > self.config.max_size && !current_chunk.is_empty() {
                // Create chunk
                let chunk_text = current_chunk.join("\n\n");
                let chunk_end = chunk_start + chunk_text.len();

                chunks.push(SemanticChunk {
                    text: chunk_text.clone(),
                    start: chunk_start,
                    end: chunk_end,
                    sentence_count: self.count_sentences(&chunk_text),
                    paragraph_count: current_chunk.len(),
                    coherence: self.calculate_coherence(&current_chunk),
                });

                chunk_start = chunk_end;
                current_chunk = Vec::new();
                current_size = 0;
            }

            current_chunk.push(paragraph.to_string());
            current_size += para_len;
        }

        // Add remaining chunk
        if !current_chunk.is_empty() {
            let chunk_text = current_chunk.join("\n\n");
            let chunk_end = chunk_start + chunk_text.len();

            chunks.push(SemanticChunk {
                text: chunk_text.clone(),
                start: chunk_start,
                end: chunk_end,
                sentence_count: self.count_sentences(&chunk_text),
                paragraph_count: current_chunk.len(),
                coherence: self.calculate_coherence(&current_chunk),
            });
        }

        chunks
    }

    /// Topic-based chunking (simplified TextTiling algorithm)
    fn chunk_by_topic(&self, text: &str) -> Vec<SemanticChunk> {
        let sentences = self.split_sentences(text);
        let mut chunks = Vec::new();

        // Calculate lexical cohesion scores between adjacent sentences
        let mut boundaries = vec![0]; // Start of text is always a boundary

        for i in 1..sentences.len() {
            let cohesion = self.lexical_cohesion(&sentences[i-1], &sentences[i]);

            // If cohesion is low, mark as potential boundary
            if cohesion < self.config.similarity_threshold {
                boundaries.push(i);
            }
        }

        boundaries.push(sentences.len()); // End of text is always a boundary

        // Create chunks from boundaries
        let mut text_pos = 0;
        for window in boundaries.windows(2) {
            let start_idx = window[0];
            let end_idx = window[1];

            let chunk_sentences = &sentences[start_idx..end_idx];
            let chunk_text = chunk_sentences.join(" ");
            let chunk_len = chunk_text.len();

            if chunk_len >= self.config.min_size {
                chunks.push(SemanticChunk {
                    text: chunk_text,
                    start: text_pos,
                    end: text_pos + chunk_len,
                    sentence_count: chunk_sentences.len(),
                    paragraph_count: self.count_paragraphs(&chunk_sentences.join(" ")),
                    coherence: self.calculate_coherence(chunk_sentences),
                });
            }

            text_pos += chunk_len;
        }

        chunks
    }

    /// Semantic similarity-based chunking
    fn chunk_by_similarity(&self, text: &str) -> Vec<SemanticChunk> {
        // For now, fall back to sentence-based chunking
        // TODO: Implement proper embedding-based similarity chunking
        // This requires:
        // 1. Generate embeddings for each sentence
        // 2. Calculate cosine similarity between adjacent sentences
        // 3. Create boundaries where similarity drops below threshold
        // 4. Merge small chunks that are below min_size

        self.chunk_by_sentences(text)
    }

    /// Hybrid chunking strategy
    fn chunk_hybrid(&self, text: &str) -> Vec<SemanticChunk> {
        // Start with paragraph boundaries
        let para_chunks = self.chunk_by_paragraphs(text);

        // Further split large paragraphs by sentences
        let mut final_chunks = Vec::new();

        for chunk in para_chunks {
            if chunk.text.len() > self.config.max_size {
                // Split by sentences
                let mut temp_config = self.config.clone();
                temp_config.strategy = ChunkingStrategy::Sentence;
                let sub_chunker = SemanticChunker::new(temp_config);
                let sub_chunks = sub_chunker.chunk(&chunk.text);
                final_chunks.extend(sub_chunks);
            } else {
                final_chunks.push(chunk);
            }
        }

        final_chunks
    }

    /// Split text into sentences (simple heuristic)
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);

            // Simple sentence boundary detection
            if matches!(c, '.' | '!' | '?') {
                if let Some(next) = current.chars().last() {
                    if next.is_whitespace() || !current.trim().is_empty() {
                        sentences.push(current.trim().to_string());
                        current = String::new();
                    }
                }
            }
        }

        // Add remaining text
        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        sentences
    }

    /// Count sentences in text
    fn count_sentences(&self, text: &str) -> usize {
        text.chars().filter(|c| matches!(c, '.' | '!' | '?')).count()
    }

    /// Count paragraphs in text
    fn count_paragraphs(&self, text: &str) -> usize {
        text.split("\n\n").filter(|p| !p.trim().is_empty()).count().max(1)
    }

    /// Calculate lexical cohesion between two texts (word overlap)
    fn lexical_cohesion(&self, text1: &str, text2: &str) -> f32 {
        let text1_lower = text1.to_lowercase();
        let words1: std::collections::HashSet<_> = text1_lower
            .split_whitespace()
            .collect();

        let text2_lower = text2.to_lowercase();
        let words2: std::collections::HashSet<_> = text2_lower
            .split_whitespace()
            .collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculate coherence score for a chunk (based on word overlap)
    fn calculate_coherence(&self, sentences: &[String]) -> f32 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut total_cohesion = 0.0;
        for window in sentences.windows(2) {
            total_cohesion += self.lexical_cohesion(&window[0], &window[1]);
        }

        total_cohesion / (sentences.len() - 1) as f32
    }
}

/// Chunking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingStats {
    /// Total chunks created
    pub total_chunks: usize,
    /// Average chunk size (characters)
    pub avg_chunk_size: f32,
    /// Minimum chunk size
    pub min_chunk_size: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
    /// Average coherence score
    pub avg_coherence: f32,
    /// Average sentences per chunk
    pub avg_sentences_per_chunk: f32,
}

impl ChunkingStats {
    /// Calculate statistics from chunks
    pub fn from_chunks(chunks: &[SemanticChunk]) -> Self {
        if chunks.is_empty() {
            return Self {
                total_chunks: 0,
                avg_chunk_size: 0.0,
                min_chunk_size: 0,
                max_chunk_size: 0,
                avg_coherence: 0.0,
                avg_sentences_per_chunk: 0.0,
            };
        }

        let total_chunks = chunks.len();
        let sizes: Vec<usize> = chunks.iter().map(|c| c.text.len()).collect();
        let avg_chunk_size = sizes.iter().sum::<usize>() as f32 / total_chunks as f32;
        let min_chunk_size = *sizes.iter().min().unwrap();
        let max_chunk_size = *sizes.iter().max().unwrap();

        let avg_coherence = chunks.iter().map(|c| c.coherence).sum::<f32>() / total_chunks as f32;
        let avg_sentences_per_chunk = chunks.iter().map(|c| c.sentence_count).sum::<usize>() as f32
            / total_chunks as f32;

        Self {
            total_chunks,
            avg_chunk_size,
            min_chunk_size,
            max_chunk_size,
            avg_coherence,
            avg_sentences_per_chunk,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TEXT: &str = "This is the first sentence. This is the second sentence. \
                              This is the third sentence.\n\n\
                              This is a new paragraph with different content. \
                              It has multiple sentences too. \
                              And here is another one.";

    #[test]
    fn test_fixed_size_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::FixedSize,
            target_size: 50,
            min_size: 10,
            max_size: 100,
            overlap: 10,
            similarity_threshold: 0.7,
        };

        let chunker = SemanticChunker::new(config);
        let chunks = chunker.chunk(TEST_TEXT);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.text.len() <= 100);
        }
    }

    #[test]
    fn test_sentence_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Sentence,
            target_size: 100,
            min_size: 20,
            max_size: 200,
            overlap: 20,
            similarity_threshold: 0.7,
        };

        let chunker = SemanticChunker::new(config);
        let chunks = chunker.chunk(TEST_TEXT);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.sentence_count > 0);
            assert!(chunk.text.len() >= 20);
        }
    }

    #[test]
    fn test_paragraph_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Paragraph,
            target_size: 100,
            min_size: 20,
            max_size: 500,
            overlap: 0,
            similarity_threshold: 0.7,
        };

        let chunker = SemanticChunker::new(config);
        let chunks = chunker.chunk(TEST_TEXT);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.paragraph_count > 0);
        }
    }

    #[test]
    fn test_topic_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Topic,
            target_size: 100,
            min_size: 20,
            max_size: 300,
            overlap: 0,
            similarity_threshold: 0.3,
        };

        let chunker = SemanticChunker::new(config);
        let chunks = chunker.chunk(TEST_TEXT);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_hybrid_chunking() {
        let config = ChunkingConfig {
            strategy: ChunkingStrategy::Hybrid,
            target_size: 100,
            min_size: 20,
            max_size: 150,
            overlap: 10,
            similarity_threshold: 0.7,
        };

        let chunker = SemanticChunker::new(config);
        let chunks = chunker.chunk(TEST_TEXT);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunking_stats() {
        let chunker = SemanticChunker::default_config();
        let chunks = chunker.chunk(TEST_TEXT);
        let stats = ChunkingStats::from_chunks(&chunks);

        assert_eq!(stats.total_chunks, chunks.len());
        assert!(stats.avg_chunk_size > 0.0);
        assert!(stats.avg_coherence >= 0.0 && stats.avg_coherence <= 1.0);
    }

    #[test]
    fn test_sentence_splitting() {
        let chunker = SemanticChunker::default_config();
        let sentences = chunker.split_sentences("Hello world. How are you? I am fine!");

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
    }

    #[test]
    fn test_lexical_cohesion() {
        let chunker = SemanticChunker::default_config();

        let cohesion1 = chunker.lexical_cohesion(
            "The cat sat on the mat",
            "The cat was very happy"
        );
        assert!(cohesion1 > 0.0);

        let cohesion2 = chunker.lexical_cohesion(
            "The cat sat on the mat",
            "Quantum physics is complex"
        );
        assert!(cohesion2 < cohesion1);
    }
}
