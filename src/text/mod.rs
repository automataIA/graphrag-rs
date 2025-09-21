pub mod chunking;

use crate::{
    core::{ChunkId, Document, TextChunk},
    parallel::{ParallelProcessor, PerformanceMonitor},
    Result,
};
use chunking::HierarchicalChunker;

/// Text processing utilities for chunking and preprocessing
#[derive(Debug)]
pub struct TextProcessor {
    chunk_size: usize,
    chunk_overlap: usize,
    parallel_processor: Option<ParallelProcessor>,
    performance_monitor: PerformanceMonitor,
}

impl TextProcessor {
    /// Create a new text processor
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        Ok(Self {
            chunk_size,
            chunk_overlap,
            parallel_processor: None,
            performance_monitor: PerformanceMonitor::new(),
        })
    }

    /// Create a new text processor with parallel processing support
    pub fn with_parallel_processing(
        chunk_size: usize,
        chunk_overlap: usize,
        parallel_processor: ParallelProcessor,
    ) -> Result<Self> {
        Ok(Self {
            chunk_size,
            chunk_overlap,
            parallel_processor: Some(parallel_processor),
            performance_monitor: PerformanceMonitor::new(),
        })
    }

    /// Split text into chunks with overlap using hierarchical boundary preservation
    pub fn chunk_text_hierarchical(&self, document: &Document) -> Result<Vec<TextChunk>> {
        let chunker = HierarchicalChunker::new().with_min_size(50);
        let chunks_text = chunker.chunk_text(&document.content, self.chunk_size, self.chunk_overlap);

        let mut chunks = Vec::new();
        let mut chunk_counter = 0;
        let mut current_pos = 0;

        for chunk_content in chunks_text {
            if !chunk_content.trim().is_empty() {
                let chunk_id = ChunkId::new(format!("{}_{}", document.id, chunk_counter));
                let chunk_start = current_pos;
                let chunk_end = chunk_start + chunk_content.len();

                current_pos += chunk_content.len();

                let chunk = TextChunk::new(
                    chunk_id,
                    document.id.clone(),
                    chunk_content,
                    chunk_start,
                    chunk_end,
                );
                chunks.push(chunk);
                chunk_counter += 1;
            } else {
                current_pos += chunk_content.len();
            }
        }

        Ok(chunks)
    }

    /// Split text into chunks with overlap (legacy method)
    pub fn chunk_text(&self, document: &Document) -> Result<Vec<TextChunk>> {
        let text = &document.content;
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_counter = 0;

        while start < text.len() {
            let end = std::cmp::min(start + self.chunk_size, text.len());

            // Try to find a good breaking point (sentence boundary)
            let actual_end = if end < text.len() {
                self.find_sentence_boundary(text, start, end)
                    .unwrap_or_else(|| self.find_char_boundary(text, end))
            } else {
                end
            };

            let chunk_content = text[start..actual_end].to_string();

            if !chunk_content.trim().is_empty() {
                let chunk_id = ChunkId::new(format!("{}_{}", document.id, chunk_counter));
                let chunk = TextChunk::new(
                    chunk_id,
                    document.id.clone(),
                    chunk_content,
                    start,
                    actual_end,
                );
                chunks.push(chunk);
                chunk_counter += 1;
            }

            // Calculate next start position with overlap
            let next_start = if actual_end >= text.len() {
                break;
            } else {
                let overlap_start = actual_end.saturating_sub(self.chunk_overlap);
                let safe_overlap = self.find_char_boundary(text, overlap_start);
                std::cmp::max(start + 1, safe_overlap)
            };

            start = next_start;
        }

        Ok(chunks)
    }

    /// Find a safe character boundary at or before the given position
    fn find_char_boundary(&self, text: &str, mut pos: usize) -> usize {
        pos = pos.min(text.len());
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }

    /// Find a safe character boundary within a slice at or before the given position
    fn find_char_boundary_in_slice(&self, text: &str, mut pos: usize) -> usize {
        pos = pos.min(text.len());
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }
        pos
    }

    /// Find a good sentence boundary for chunking
    fn find_sentence_boundary(
        &self,
        text: &str,
        start: usize,
        preferred_end: usize,
    ) -> Option<usize> {
        // Ensure we're at character boundaries
        let safe_start = self.find_char_boundary(text, start);
        let safe_end = self.find_char_boundary(text, preferred_end);

        if safe_start >= safe_end {
            return None;
        }

        let search_window = &text[safe_start..safe_end];

        // Look for sentence boundaries in the last part of the chunk
        let search_start = search_window.len().saturating_sub(200);
        // Find character boundary within the search window
        let safe_search_start = self.find_char_boundary_in_slice(search_window, search_start);
        let search_text = &search_window[safe_search_start..];

        // Simple sentence boundary detection
        let sentence_endings = ['.', '!', '?'];
        let mut last_boundary = None;

        for (i, ch) in search_text.char_indices() {
            if sentence_endings.contains(&ch) {
                // Check if next character is whitespace or end of text
                let next_pos = i + ch.len_utf8();
                if next_pos >= search_text.len()
                    || search_text
                        .chars()
                        .nth(next_pos)
                        .is_none_or(|c| c.is_whitespace())
                {
                    last_boundary = Some(safe_start + safe_search_start + next_pos);
                }
            }
        }

        last_boundary.or_else(|| self.find_word_boundary(text, safe_start, safe_end))
    }

    /// Find a word boundary for chunking
    fn find_word_boundary(&self, text: &str, start: usize, preferred_end: usize) -> Option<usize> {
        // These should already be safe boundaries from the caller
        if start >= preferred_end {
            return None;
        }

        let search_window = &text[start..preferred_end];

        // Find the last whitespace in the last 50 characters
        let search_start = search_window.len().saturating_sub(50);
        let safe_search_start = self.find_char_boundary_in_slice(search_window, search_start);
        let search_text = &search_window[safe_search_start..];

        search_text
            .rfind(' ')
            .map(|pos| start + safe_search_start + pos)
    }

    /// Clean and normalize text
    pub fn clean_text(&self, text: &str) -> String {
        text
            // Normalize whitespace
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            // Remove excessive punctuation
            .chars()
            .collect::<String>()
    }

    /// Extract sentences from text
    pub fn extract_sentences(&self, text: &str) -> Vec<String> {
        let sentence_endings = ['.', '!', '?'];
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();

        for ch in text.chars() {
            if sentence_endings.contains(&ch) {
                let trimmed = current_sentence.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current_sentence.clear();
            } else {
                current_sentence.push(ch);
            }
        }

        // Add any remaining text as a sentence
        let trimmed = current_sentence.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Count words in text
    pub fn word_count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Process multiple documents in parallel
    pub fn batch_chunk_documents(&self, documents: Vec<Document>) -> Result<Vec<Vec<TextChunk>>> {
        self.performance_monitor.time_operation(|| {
            if let Some(ref processor) = self.parallel_processor {
                if processor.should_use_parallel(documents.len()) {
                    let results = processor.execute_parallel(documents, |doc| {
                        self.chunk_text(&doc).unwrap_or_default()
                    });
                    return Ok(results);
                }
            }

            // Sequential fallback
            documents
                .into_iter()
                .map(|doc| self.chunk_text(&doc))
                .collect()
        })
    }

    /// Parallel extraction of keywords from multiple texts
    pub fn batch_extract_keywords(&self, texts: &[&str], max_keywords: usize) -> Vec<Vec<String>> {
        self.performance_monitor.time_operation(|| {
            if let Some(ref processor) = self.parallel_processor {
                if processor.should_use_parallel(texts.len()) {
                    return processor.execute_parallel(texts, |&text| {
                        self.extract_keywords(text, max_keywords)
                    });
                }
            }

            // Sequential fallback
            texts
                .iter()
                .map(|&text| self.extract_keywords(text, max_keywords))
                .collect()
        })
    }

    /// Parallel sentence extraction from multiple texts
    pub fn batch_extract_sentences(&self, texts: &[&str]) -> Vec<Vec<String>> {
        self.performance_monitor.time_operation(|| {
            if let Some(ref processor) = self.parallel_processor {
                if processor.should_use_parallel(texts.len()) {
                    return processor.execute_parallel(texts, |&text| self.extract_sentences(text));
                }
            }

            // Sequential fallback
            texts
                .iter()
                .map(|&text| self.extract_sentences(text))
                .collect()
        })
    }

    /// Parallel text cleaning for multiple texts
    pub fn batch_clean_text(&self, texts: &[&str]) -> Vec<String> {
        self.performance_monitor.time_operation(|| {
            if let Some(ref processor) = self.parallel_processor {
                if processor.should_use_parallel(texts.len()) {
                    return processor.execute_parallel(texts, |&text| self.clean_text(text));
                }
            }

            // Sequential fallback
            texts.iter().map(|&text| self.clean_text(text)).collect()
        })
    }

    /// Extract keywords using simple frequency analysis
    pub fn extract_keywords(&self, text: &str, max_keywords: usize) -> Vec<String> {
        use std::collections::HashMap;

        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 3) // Filter out short words
            .filter(|w| !self.is_stop_word(w))
            .collect();

        let mut word_counts = HashMap::new();
        for word in words {
            *word_counts.entry(word).or_insert(0) += 1;
        }

        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        sorted_words
            .into_iter()
            .take(max_keywords)
            .map(|(word, _)| word)
            .collect()
    }

    /// Simple stop word detection (English)
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (usize, std::time::Duration) {
        self.performance_monitor.get_stats()
    }

    /// Get average processing time per operation
    pub fn average_processing_time(&self) -> std::time::Duration {
        self.performance_monitor.average_duration()
    }

    /// Reset performance monitoring statistics
    pub fn reset_performance_stats(&self) {
        self.performance_monitor.reset();
    }

    /// Get parallel processing statistics if available
    pub fn get_parallel_stats(&self) -> Option<crate::parallel::ParallelStatistics> {
        self.parallel_processor.as_ref().map(|p| p.get_statistics())
    }
}

/// Language detection utilities
pub struct LanguageDetector;

impl LanguageDetector {
    /// Simple language detection based on character patterns
    /// This is a very basic implementation - in practice you'd want a proper library
    pub fn detect_language(text: &str) -> String {
        // Very basic detection - in practice use a proper language detection library
        if text
            .chars()
            .any(|c| matches!(c, 'ñ' | 'ó' | 'é' | 'í' | 'á' | 'ú'))
        {
            "es".to_string()
        } else if text.chars().any(|c| matches!(c, 'ç' | 'ã' | 'õ')) {
            "pt".to_string()
        } else if text.chars().any(|c| matches!(c, 'à' | 'è' | 'ù' | 'ò')) {
            "fr".to_string()
        } else {
            "en".to_string() // Default to English
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DocumentId;

    #[test]
    fn test_text_chunking() {
        let processor = TextProcessor::new(100, 20).unwrap();
        let document = Document::new(
            DocumentId::new("test".to_string()),
            "Test Document".to_string(),
            "This is a test document. It has multiple sentences. Each sentence should be processed correctly.".to_string(),
        );

        let chunks = processor.chunk_text(&document).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].content.len() <= 100);
    }

    #[test]
    fn test_keyword_extraction() {
        let processor = TextProcessor::new(1000, 100).unwrap();
        let text = "machine learning artificial intelligence data science computer vision natural language processing";
        let keywords = processor.extract_keywords(text, 3);

        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 3);
    }

    #[test]
    fn test_sentence_extraction() {
        let processor = TextProcessor::new(1000, 100).unwrap();
        let text = "First sentence. Second sentence! Third sentence?";
        let sentences = processor.extract_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence");
        assert_eq!(sentences[1], "Second sentence");
        assert_eq!(sentences[2], "Third sentence");
    }
}
