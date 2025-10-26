//! Real TF-IDF keyword extraction
//!
//! This module implements actual TF-IDF (Term Frequency-Inverse Document Frequency)
//! algorithm for keyword extraction, not mock/placeholder implementations.

use std::collections::HashMap;

/// TF-IDF based keyword extractor
pub struct TfIdfKeywordExtractor {
    /// Document frequencies: how many documents contain each term
    document_frequencies: HashMap<String, usize>,
    /// Total number of documents in the corpus
    total_documents: usize,
    /// Stop words to ignore
    stopwords: std::collections::HashSet<String>,
}

impl TfIdfKeywordExtractor {
    /// Create a new TF-IDF extractor
    pub fn new(document_frequencies: HashMap<String, usize>, total_documents: usize) -> Self {
        let stopwords = Self::load_stopwords();
        Self {
            document_frequencies,
            total_documents: total_documents.max(1),
            stopwords,
        }
    }

    /// Create with default stopwords and empty IDF data (for single-document use)
    pub fn new_default() -> Self {
        Self::new(HashMap::new(), 1)
    }

    /// Extract keywords using TF-IDF scoring
    ///
    /// Returns keywords sorted by TF-IDF score (highest first)
    pub fn extract_keywords(&self, text: &str, top_k: usize) -> Vec<(String, f32)> {
        // 1. Tokenize and calculate term frequencies
        let tokens = self.tokenize(text);
        let tf_scores = self.calculate_tf(&tokens);

        // 2. Calculate TF-IDF scores
        let mut tfidf_scores: Vec<(String, f32)> = tf_scores
            .into_iter()
            .map(|(term, tf)| {
                let idf = self.calculate_idf(&term);
                let tfidf = tf * idf;
                (term, tfidf)
            })
            .collect();

        // 3. Sort by score (descending) and take top-k
        tfidf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        tfidf_scores.truncate(top_k);

        tfidf_scores
    }

    /// Extract just keyword strings (without scores)
    pub fn extract_keyword_strings(&self, text: &str, top_k: usize) -> Vec<String> {
        self.extract_keywords(text, top_k)
            .into_iter()
            .map(|(word, _score)| word)
            .collect()
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                // Remove punctuation and convert to lowercase
                word.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| {
                // Filter: non-empty, length > 2, not stopword, not pure numbers
                !word.is_empty()
                    && word.len() > 2
                    && !self.stopwords.contains(word)
                    && !word.chars().all(|c| c.is_numeric())
            })
            .collect()
    }

    /// Calculate term frequency (TF) using normalized frequency
    ///
    /// TF = (count of term in document) / (total terms in document)
    fn calculate_tf(&self, tokens: &[String]) -> HashMap<String, f32> {
        let mut term_counts: HashMap<String, usize> = HashMap::new();

        // Count occurrences
        for token in tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        let total_terms = tokens.len().max(1) as f32;

        // Normalize by total terms
        term_counts
            .into_iter()
            .map(|(term, count)| (term, count as f32 / total_terms))
            .collect()
    }

    /// Calculate inverse document frequency (IDF)
    ///
    /// IDF = log(total_documents / documents_containing_term)
    ///
    /// If term is not in corpus, uses a default IDF (assumes rare term)
    fn calculate_idf(&self, term: &str) -> f32 {
        let doc_freq = self
            .document_frequencies
            .get(term)
            .copied()
            .unwrap_or(1); // Default to 1 if not seen (rare term)

        let idf = (self.total_documents as f32 / doc_freq as f32).ln();
        idf.max(0.0) // Ensure non-negative
    }

    /// Load English stopwords
    fn load_stopwords() -> std::collections::HashSet<String> {
        // Common English stopwords
        let stopwords_list = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
            "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
            "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
            "way", "even", "new", "want", "because", "any", "these", "give", "day", "most",
            "us", "is", "was", "are", "been", "has", "had", "were", "said", "did",
        ];

        stopwords_list.into_iter().map(|s| s.to_string()).collect()
    }

    /// Update document frequencies with a new document (for corpus-level IDF)
    pub fn add_document_to_corpus(&mut self, text: &str) {
        let tokens = self.tokenize(text);
        let unique_terms: std::collections::HashSet<String> = tokens.into_iter().collect();

        for term in unique_terms {
            *self.document_frequencies.entry(term).or_insert(0) += 1;
        }

        self.total_documents += 1;
    }

    /// Get corpus statistics
    pub fn corpus_stats(&self) -> (usize, usize) {
        (self.total_documents, self.document_frequencies.len())
    }
}

impl Default for TfIdfKeywordExtractor {
    fn default() -> Self {
        Self::new_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization() {
        let extractor = TfIdfKeywordExtractor::new_default();
        let text = "Machine learning and artificial intelligence are transforming technology.";
        let tokens = extractor.tokenize(text);

        assert!(tokens.contains(&"machine".to_string()));
        assert!(tokens.contains(&"learning".to_string()));
        assert!(tokens.contains(&"artificial".to_string()));
        // Stopwords should be filtered
        assert!(!tokens.contains(&"and".to_string()));
        assert!(!tokens.contains(&"are".to_string()));
    }

    #[test]
    fn test_tf_calculation() {
        let extractor = TfIdfKeywordExtractor::new_default();
        let tokens = vec![
            "machine".to_string(),
            "learning".to_string(),
            "machine".to_string(),
            "learning".to_string(),
            "data".to_string(),
        ];

        let tf_scores = extractor.calculate_tf(&tokens);

        // machine and learning appear 2 times out of 5 = 0.4
        assert!((tf_scores["machine"] - 0.4).abs() < 0.001);
        assert!((tf_scores["learning"] - 0.4).abs() < 0.001);
        // data appears 1 time out of 5 = 0.2
        assert!((tf_scores["data"] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_idf_calculation() {
        let mut doc_freqs = HashMap::new();
        doc_freqs.insert("common".to_string(), 50); // appears in 50 docs
        doc_freqs.insert("rare".to_string(), 2); // appears in 2 docs

        let extractor = TfIdfKeywordExtractor::new(doc_freqs, 100);

        let idf_common = extractor.calculate_idf("common");
        let idf_rare = extractor.calculate_idf("rare");

        // Rare terms should have higher IDF
        assert!(idf_rare > idf_common);
        // log(100/50) = log(2) ≈ 0.69
        assert!((idf_common - 0.69).abs() < 0.1);
        // log(100/2) = log(50) ≈ 3.91
        assert!((idf_rare - 3.91).abs() < 0.1);
    }

    #[test]
    fn test_keyword_extraction() {
        // Build a proper corpus for realistic TF-IDF scores
        let mut extractor = TfIdfKeywordExtractor::new_default();

        // Add background corpus documents to establish IDF scores
        extractor.add_document_to_corpus("artificial intelligence is the future");
        extractor.add_document_to_corpus("deep learning uses neural networks");
        extractor.add_document_to_corpus("natural language processing is important");

        let text = "machine learning and deep learning are important topics in artificial intelligence. \
                    neural networks and machine learning models are widely used.";

        let keywords = extractor.extract_keywords(text, 5);

        assert!(keywords.len() >= 3);
        // "learning" and "machine" should rank high due to frequency in the target text
        let keyword_terms: Vec<&str> = keywords.iter().map(|(w, _)| w.as_str()).collect();

        // At least one of these high-frequency terms should appear
        assert!(keyword_terms.contains(&"learning") ||
                keyword_terms.contains(&"machine") ||
                keyword_terms.contains(&"neural"),
                "Expected high-frequency terms not found. Got: {:?}", keyword_terms);
    }

    #[test]
    fn test_corpus_building() {
        let mut extractor = TfIdfKeywordExtractor::new_default();

        extractor.add_document_to_corpus("machine learning is amazing");
        extractor.add_document_to_corpus("deep learning is powerful");
        extractor.add_document_to_corpus("natural language processing");

        let (total_docs, unique_terms) = extractor.corpus_stats();
        assert_eq!(total_docs, 4); // 1 initial + 3 added
        assert!(unique_terms > 0);
    }

    #[test]
    fn test_stopword_filtering() {
        let extractor = TfIdfKeywordExtractor::new_default();
        let text = "The quick brown fox jumps over the lazy dog and the cat";
        let keywords = extractor.extract_keyword_strings(text, 10);

        // Stopwords like "the", "and", "over" should not appear
        assert!(!keywords.iter().any(|w| w == "the"));
        assert!(!keywords.iter().any(|w| w == "and"));
        assert!(!keywords.iter().any(|w| w == "over"));

        // Content words should appear
        assert!(keywords.iter().any(|w| w == "quick" || w == "brown" || w == "fox"));
    }
}
