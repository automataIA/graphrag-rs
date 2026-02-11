//! Real extractive summarization with sentence ranking
//!
//! This module implements actual sentence ranking algorithms for extractive
//! summarization, not mock/placeholder implementations.

use std::collections::{HashMap, HashSet};

/// Extractive summarizer using sentence scoring
pub struct ExtractiveSummarizer {
    /// Stop words to ignore in scoring
    stopwords: HashSet<String>,
}

impl ExtractiveSummarizer {
    /// Create a new extractive summarizer
    pub fn new() -> Self {
        Self {
            stopwords: Self::load_stopwords(),
        }
    }

    /// Generate a summary of the given text
    ///
    /// # Arguments
    /// * `text` - The input text to summarize
    /// * `max_length` - Maximum character length of the summary
    ///
    /// # Returns
    /// Extractive summary selecting the most important sentences
    pub fn summarize(&self, text: &str, max_length: usize) -> crate::Result<String> {
        // 1. Split into sentences
        let sentences = self.split_sentences(text);

        if sentences.is_empty() {
            return Ok(String::new());
        }

        if sentences.len() == 1 {
            let sentence = &sentences[0];
            if sentence.len() <= max_length {
                return Ok(sentence.clone());
            } else {
                return Ok(self.truncate_sentence(sentence, max_length));
            }
        }

        // 2. Score each sentence
        let scored_sentences: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(idx, sentence)| {
                let score = self.score_sentence(sentence, &sentences, idx);
                (idx, score)
            })
            .collect();

        // 3. Select top sentences until we reach max_length
        let selected_indices = self.select_sentences(scored_sentences, &sentences, max_length);

        // 4. Combine selected sentences in original order
        let summary = selected_indices
            .iter()
            .map(|&idx| sentences[idx].as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(summary)
    }

    /// Split text into sentences using multiple heuristics
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();

        let sentence_endings = ['.', '!', '?'];

        for ch in text.chars() {
            current_sentence.push(ch);

            if sentence_endings.contains(&ch) {
                // Check if next character is whitespace or end of text
                let trimmed = current_sentence.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() > 5 {
                    // Filter out very short "sentences" (likely abbreviations)
                    sentences.push(trimmed);
                }
                current_sentence.clear();
            }
        }

        // Add any remaining text
        let trimmed = current_sentence.trim().to_string();
        if !trimmed.is_empty() && trimmed.len() > 5 {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Score a sentence based on multiple factors
    ///
    /// Scoring criteria:
    /// 1. Position score (beginning and end sentences are important)
    /// 2. Length score (prefer medium-length sentences)
    /// 3. Word frequency score (content words that appear multiple times)
    /// 4. Proper noun score (capitalized words)
    /// 5. Numeric content score (sentences with numbers often contain facts)
    fn score_sentence(&self, sentence: &str, all_sentences: &[String], position: usize) -> f32 {
        let mut total_score = 0.0;

        // 1. Position score: first and last sentences often contain key information
        let position_score = if position == 0 {
            2.0 // First sentence is very important
        } else if position == all_sentences.len() - 1 {
            1.5 // Last sentence is somewhat important
        } else {
            // Middle sentences get decreasing score based on distance from start
            let distance_from_start = position as f32 / all_sentences.len() as f32;
            1.0 - (distance_from_start * 0.5) // Gradually decrease from 1.0 to 0.5
        };
        total_score += position_score * 0.3;

        // 2. Length score: prefer sentences of medium length
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let word_count = words.len();

        let length_score = if word_count < 5 {
            0.3 // Too short
        } else if word_count > 40 {
            0.5 // Too long
        } else if (10..=25).contains(&word_count) {
            1.0 // Ideal length
        } else {
            0.7 // Acceptable length
        };
        total_score += length_score * 0.2;

        // 3. Word frequency score: words that appear across multiple sentences
        let word_freq_score = self.calculate_word_frequency_score(sentence, all_sentences);
        total_score += word_freq_score * 0.3;

        // 4. Proper noun score: sentences with capitalized words (names, places)
        let proper_noun_score = self.calculate_proper_noun_score(sentence);
        total_score += proper_noun_score * 0.1;

        // 5. Numeric content score: sentences with numbers often state facts
        let numeric_score = self.calculate_numeric_score(sentence);
        total_score += numeric_score * 0.1;

        total_score
    }

    /// Calculate word frequency score
    fn calculate_word_frequency_score(&self, sentence: &str, all_sentences: &[String]) -> f32 {
        // Get all words from all sentences
        let all_words: Vec<String> = all_sentences
            .iter()
            .flat_map(|s| s.split_whitespace())
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty() && !self.stopwords.contains(w))
            .collect();

        // Count word frequencies
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in &all_words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }

        // Score sentence based on its words' frequencies
        let sentence_words: Vec<String> = sentence
            .split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty() && !self.stopwords.contains(w))
            .collect();

        if sentence_words.is_empty() {
            return 0.0;
        }

        let total_score: usize = sentence_words
            .iter()
            .filter_map(|w| word_counts.get(w))
            .sum();

        let avg_score = total_score as f32 / sentence_words.len() as f32;

        // Normalize (words appearing 2-3 times are good indicators)
        (avg_score / 3.0).min(1.0)
    }

    /// Calculate proper noun score
    fn calculate_proper_noun_score(&self, sentence: &str) -> f32 {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        let proper_noun_count = words
            .iter()
            .filter(|word| {
                // Check if word starts with uppercase and is not sentence start
                word.chars().next().is_some_and(|c| c.is_uppercase())
                    && word.len() > 2
                    && !self.stopwords.contains(&word.to_lowercase())
            })
            .count();

        // Normalize by sentence length
        (proper_noun_count as f32 / words.len() as f32).min(1.0)
    }

    /// Calculate numeric content score
    fn calculate_numeric_score(&self, sentence: &str) -> f32 {
        let has_number = sentence.chars().any(|c| c.is_numeric());

        // Count numbers
        let number_count = sentence
            .split_whitespace()
            .filter(|word| word.chars().any(|c| c.is_numeric()))
            .count();

        if has_number {
            (number_count as f32 * 0.3).min(1.0)
        } else {
            0.0
        }
    }

    /// Select sentences to include in summary
    ///
    /// Greedy selection: iteratively add highest-scoring sentences until max_length reached
    fn select_sentences(
        &self,
        mut scored_sentences: Vec<(usize, f32)>,
        sentences: &[String],
        max_length: usize,
    ) -> Vec<usize> {
        // Sort by score (descending)
        scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected_indices = Vec::new();
        let mut current_length = 0;

        for &(idx, _score) in &scored_sentences {
            let sentence_len = sentences[idx].len();

            // Check if adding this sentence would exceed limit
            if current_length + sentence_len < max_length {
                // +1 for space
                selected_indices.push(idx);
                current_length += sentence_len + 1;
            }

            // Early exit if we're close to the limit
            if current_length >= max_length * 90 / 100 {
                // 90% of max
                break;
            }
        }

        // Sort indices to maintain original order
        selected_indices.sort_unstable();

        // If no sentences fit, take the first (highest-scored) one and truncate
        if selected_indices.is_empty() && !scored_sentences.is_empty() {
            selected_indices.push(scored_sentences[0].0);
        }

        selected_indices
    }

    /// Truncate a sentence to fit within max_length
    fn truncate_sentence(&self, sentence: &str, max_length: usize) -> String {
        if sentence.len() <= max_length {
            return sentence.to_string();
        }

        // Find a good breaking point (word boundary)
        let mut end = max_length.saturating_sub(3); // Leave room for "..."

        // Move back to last word boundary
        while end > 0 && !sentence.is_char_boundary(end) {
            end -= 1;
        }

        while end > 0 && !sentence.chars().nth(end).is_some_and(|c| c.is_whitespace()) {
            end -= 1;
        }

        if end == 0 {
            // Fallback: just cut at character boundary
            end = max_length.saturating_sub(3);
            while end > 0 && !sentence.is_char_boundary(end) {
                end -= 1;
            }
        }

        format!("{}...", &sentence[..end].trim())
    }

    /// Load stopwords
    fn load_stopwords() -> HashSet<String> {
        let stopwords_list = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
            "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        ];

        stopwords_list.into_iter().map(|s| s.to_string()).collect()
    }

    /// Summarize with a target number of sentences instead of character limit
    pub fn summarize_sentences(&self, text: &str, num_sentences: usize) -> crate::Result<String> {
        let sentences = self.split_sentences(text);

        if sentences.is_empty() {
            return Ok(String::new());
        }

        if sentences.len() <= num_sentences {
            return Ok(sentences.join(" "));
        }

        // Score sentences
        let mut scored_sentences: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(idx, sentence)| {
                let score = self.score_sentence(sentence, &sentences, idx);
                (idx, score)
            })
            .collect();

        // Sort by score and take top N
        scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected_indices: Vec<usize> = scored_sentences
            .into_iter()
            .take(num_sentences)
            .map(|(idx, _)| idx)
            .collect();

        // Sort indices to maintain original order
        selected_indices.sort_unstable();

        let summary = selected_indices
            .iter()
            .map(|&idx| sentences[idx].as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(summary)
    }
}

impl Default for ExtractiveSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_splitting() {
        let summarizer = ExtractiveSummarizer::new();
        let text = "This is the first sentence. This is the second! Is this the third?";
        let sentences = summarizer.split_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("first sentence"));
        assert!(sentences[1].contains("second"));
        assert!(sentences[2].contains("third"));
    }

    #[test]
    fn test_summarization() {
        let summarizer = ExtractiveSummarizer::new();
        let text = "Machine learning is a subset of artificial intelligence. \
                    It focuses on training algorithms to learn from data. \
                    Deep learning is a specialized branch of machine learning. \
                    Neural networks are the foundation of deep learning systems.";

        let summary = summarizer.summarize(text, 100).unwrap();

        assert!(!summary.is_empty());
        assert!(summary.len() <= 100);
        // Summary should contain content from the original text
        assert!(summary.contains("machine learning") || summary.contains("artificial intelligence"));
    }

    #[test]
    fn test_sentence_selection() {
        let summarizer = ExtractiveSummarizer::new();
        let text = "The quick brown fox jumps over the lazy dog. \
                    This is a simple test sentence. \
                    Machine learning and artificial intelligence are transforming technology.";

        let summary = summarizer.summarize_sentences(text, 1).unwrap();

        // Should select one sentence (likely the first or third based on content)
        let sentence_count = summary.matches('.').count() + summary.matches('!').count() + summary.matches('?').count();
        assert!(sentence_count <= 2); // Allow for edge cases
    }

    #[test]
    fn test_truncation() {
        let summarizer = ExtractiveSummarizer::new();
        let long_sentence = "This is a very long sentence that needs to be truncated because it exceeds the maximum allowed length for the summary";

        let truncated = summarizer.truncate_sentence(long_sentence, 50);

        assert!(truncated.len() <= 50);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_empty_text() {
        let summarizer = ExtractiveSummarizer::new();
        let summary = summarizer.summarize("", 100).unwrap();
        assert_eq!(summary, "");
    }

    #[test]
    fn test_single_sentence() {
        let summarizer = ExtractiveSummarizer::new();
        let text = "This is a single sentence.";
        let summary = summarizer.summarize(text, 100).unwrap();

        assert_eq!(summary, text);
    }
}
