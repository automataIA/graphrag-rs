// Advanced text chunking with hierarchical boundary preservation (2024 best practices)
// Shared utility module implementing state-of-the-art chunking strategies

/// Hierarchical text chunking following LangChain RecursiveCharacterTextSplitter approach
/// with semantic boundary preservation for optimal RAG performance
pub struct HierarchicalChunker {
    /// Hierarchical separators in order of preference
    separators: Vec<String>,
    /// Minimum chunk size to maintain
    min_chunk_size: usize,
}

impl HierarchicalChunker {
    /// Create a new hierarchical chunker with default separators
    pub fn new() -> Self {
        Self {
            // Following 2024 research best practices - hierarchical separators
            separators: vec![
                "\n\n".to_string(),     // Paragraph breaks (highest priority)
                "\n".to_string(),       // Line breaks
                ". ".to_string(),       // Sentence endings with space
                "! ".to_string(),       // Exclamation sentences
                "? ".to_string(),       // Question sentences
                "; ".to_string(),       // Semicolon clauses
                ": ".to_string(),       // Colon clauses
                " ".to_string(),        // Word boundaries
                "".to_string(),         // Character level (fallback)
            ],
            min_chunk_size: 50,
        }
    }

    /// Create chunker with custom separators
    pub fn with_separators(separators: Vec<String>) -> Self {
        Self {
            separators,
            min_chunk_size: 50,
        }
    }

    /// Set minimum chunk size
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_chunk_size = min_size;
        self
    }

    /// Split text into semantically coherent chunks
    pub fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let mut end = (start + chunk_size).min(text.len());

            // Ensure we're on a UTF-8 character boundary first
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }

            // If we're at the exact end, no need to adjust
            if end >= text.len() {
                let chunk = &text[start..];
                if chunk.trim().len() >= self.min_chunk_size {
                    chunks.push(chunk.to_string());
                }
                break;
            }

            // Find the best boundary to avoid semantic truncation
            let optimal_end = self.find_optimal_boundary(text, start, end);

            // If we found a good boundary, use it
            if optimal_end > start {
                end = optimal_end;
            }

            let chunk = &text[start..end];

            if chunk.trim().len() >= self.min_chunk_size {
                chunks.push(chunk.to_string());
            }

            if end >= text.len() {
                break;
            }

            // Calculate next start with overlap, preserving semantic boundaries
            let mut next_start = end.saturating_sub(overlap);

            // Ensure next start is on a UTF-8 boundary
            while next_start > 0 && !text.is_char_boundary(next_start) {
                next_start -= 1;
            }

            // Try to align next start with word boundary
            next_start = self.find_word_boundary_backward(text, next_start);

            start = next_start;
        }

        chunks
    }

    /// Find optimal boundary using hierarchical separators
    fn find_optimal_boundary(&self, text: &str, start: usize, max_end: usize) -> usize {
        let search_text = &text[start..max_end];

        // Try each separator in order of preference
        for separator in &self.separators {
            if separator.is_empty() {
                continue;
            }

            // Find the last occurrence of this separator within our range
            if let Some(sep_pos) = search_text.rfind(separator) {
                let boundary = start + sep_pos + separator.len();

                // Make sure we're not too close to the start (maintain minimum chunk size)
                if boundary > start + (max_end - start) / 4 {
                    return boundary;
                }
            }
        }

        // If no good separator found, try to at least end at a word boundary
        self.find_word_boundary_backward(text, max_end)
    }

    /// Find the nearest word boundary going backward from the given position
    fn find_word_boundary_backward(&self, text: &str, mut pos: usize) -> usize {
        // Ensure we're on a UTF-8 boundary
        while pos > 0 && !text.is_char_boundary(pos) {
            pos -= 1;
        }

        // Look for whitespace (word boundary) going backward
        while pos > 0 {
            if let Some(ch) = text.chars().nth(pos.saturating_sub(1)) {
                if ch.is_whitespace() {
                    return pos;
                }
            }
            pos = pos.saturating_sub(1);

            // Ensure we stay on UTF-8 boundaries
            while pos > 0 && !text.is_char_boundary(pos) {
                pos -= 1;
            }
        }

        pos
    }

    /// Advanced sentence boundary detection
    pub fn find_sentence_boundary(&self, text: &str, start: usize, preferred_end: usize) -> Option<usize> {
        let safe_start = self.find_char_boundary(text, start);
        let safe_end = self.find_char_boundary(text, preferred_end);

        if safe_start >= safe_end {
            return None;
        }

        let search_window = &text[safe_start..safe_end];

        // Look for sentence boundaries in the last part of the chunk
        let search_start = search_window.len().saturating_sub(300); // Larger window for better context
        let safe_search_start = self.find_char_boundary_in_slice(search_window, search_start);
        let search_text = &search_window[safe_search_start..];

        // Enhanced sentence boundary detection
        let sentence_endings = ['.', '!', '?'];
        let mut last_boundary = None;

        for (i, ch) in search_text.char_indices() {
            if sentence_endings.contains(&ch) {
                // Check if next character is whitespace or end of text
                let next_pos = i + ch.len_utf8();
                if next_pos >= search_text.len() {
                    last_boundary = Some(safe_start + safe_search_start + next_pos);
                } else if let Some(next_char) = search_text.chars().nth(next_pos) {
                    // More sophisticated sentence boundary detection
                    if next_char.is_whitespace() &&
                       (next_char == '\n' || next_char == ' ') {
                        // Make sure this isn't an abbreviation or decimal
                        if !self.is_likely_abbreviation(search_text, i) {
                            last_boundary = Some(safe_start + safe_search_start + next_pos);
                        }
                    }
                }
            }
        }

        last_boundary
    }

    /// Check if a period is likely part of an abbreviation
    fn is_likely_abbreviation(&self, text: &str, period_pos: usize) -> bool {
        // Simple heuristics for common abbreviations
        if period_pos == 0 {
            return false;
        }

        // Check for common abbreviation patterns
        let before_period = &text[..period_pos];
        if let Some(word_start) = before_period.rfind(' ') {
            let potential_abbrev = &before_period[word_start + 1..];

            // Common abbreviations
            let abbreviations = [
                "Dr", "Mr", "Mrs", "Ms", "Prof", "Jr", "Sr", "Inc", "Corp",
                "Ltd", "Co", "etc", "vs", "e.g", "i.e", "cf", "pp"
            ];

            return abbreviations.iter().any(|&abbrev|
                potential_abbrev.eq_ignore_ascii_case(abbrev)
            );
        }

        // Single letter followed by period (likely initial)
        if period_pos == 1 && before_period.chars().next().unwrap_or(' ').is_ascii_uppercase() {
            return true;
        }

        false
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
}

impl Default for HierarchicalChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_chunking() {
        let chunker = HierarchicalChunker::new();
        let text = "This is a test document.\n\nIt has multiple paragraphs. Each paragraph should be preserved as much as possible. This helps maintain semantic coherence in the chunks.";

        let chunks = chunker.chunk_text(text, 100, 20);

        assert!(!chunks.is_empty());
        // Should preserve paragraph boundaries
        assert!(chunks[0].contains("This is a test document."));
    }

    #[test]
    fn test_sentence_boundary_detection() {
        let chunker = HierarchicalChunker::new();
        let text = "Dr. Smith went to the store. He bought some milk. Then he went home.";

        // Should not break on "Dr." abbreviation
        if let Some(boundary) = chunker.find_sentence_boundary(text, 0, 30) {
            let chunk = &text[0..boundary];
            assert!(!chunk.ends_with("Dr."));
        }
    }

    #[test]
    fn test_word_boundary_preservation() {
        let chunker = HierarchicalChunker::new();
        let text = "This is a very long sentence that should be split at word boundaries rather than in the middle of words.";

        let chunks = chunker.chunk_text(text, 50, 10);

        // No chunk should end with a partial word
        for chunk in &chunks {
            let trimmed = chunk.trim();
            if !trimmed.is_empty() {
                let last_char = trimmed.chars().last().unwrap();
                // Should end with whitespace, punctuation, or be the complete text
                assert!(last_char.is_whitespace() ||
                       last_char.is_ascii_punctuation() ||
                       chunk.trim() == text.trim());
            }
        }
    }
}