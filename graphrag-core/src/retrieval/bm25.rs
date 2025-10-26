use crate::Result;
use std::collections::HashMap;

/// Document ID type for BM25 indexing
pub type DocumentId = String;

/// A document for BM25 indexing
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier for the document
    pub id: DocumentId,
    /// Text content of the document
    pub content: String,
    /// Key-value metadata associated with document
    pub metadata: HashMap<String, String>,
}

/// BM25 search result
#[derive(Debug, Clone)]
pub struct BM25Result {
    /// Document identifier for this result
    pub doc_id: DocumentId,
    /// BM25 relevance score for the result
    pub score: f32,
    /// Text content of the matched document
    pub content: String,
}

/// BM25 retrieval system for keyword-based search
pub struct BM25Retriever {
    /// BM25 parameter k1 (term frequency saturation)
    k1: f32,
    /// BM25 parameter b (length normalization)
    b: f32,
    /// Indexed documents
    documents: HashMap<DocumentId, Document>,
    /// Term frequencies per document: term -> document_id -> frequency
    term_frequencies: HashMap<String, HashMap<DocumentId, f32>>,
    /// Document frequencies: term -> number of documents containing term
    document_frequencies: HashMap<String, usize>,
    /// Document lengths (in tokens)
    document_lengths: HashMap<DocumentId, usize>,
    /// Average document length
    avg_doc_length: f32,
    /// Total number of documents
    total_docs: usize,
}

impl BM25Retriever {
    /// Create a new BM25 retriever with default parameters
    pub fn new() -> Self {
        Self::with_parameters(1.2, 0.75)
    }

    /// Create a new BM25 retriever with custom parameters
    pub fn with_parameters(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            documents: HashMap::new(),
            term_frequencies: HashMap::new(),
            document_frequencies: HashMap::new(),
            document_lengths: HashMap::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
        }
    }

    /// Index a single document
    pub fn index_document(&mut self, document: Document) -> Result<()> {
        let doc_id = document.id.clone();
        let tokens = self.tokenize(&document.content);
        let doc_length = tokens.len();

        // Calculate term frequencies for this document
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        // Update document frequencies
        for term in term_freq.keys() {
            *self.document_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        // Store normalized term frequencies
        for (term, freq) in term_freq {
            let normalized_freq = freq as f32 / doc_length as f32;
            self.term_frequencies
                .entry(term)
                .or_default()
                .insert(doc_id.clone(), normalized_freq);
        }

        // Store document and metadata
        self.document_lengths.insert(doc_id.clone(), doc_length);
        self.documents.insert(doc_id, document);
        self.total_docs += 1;

        // Update average document length
        self.update_avg_doc_length();

        Ok(())
    }

    /// Index multiple documents
    pub fn index_documents(&mut self, documents: &[Document]) -> Result<()> {
        for document in documents {
            self.index_document(document.clone())?;
        }
        Ok(())
    }

    /// Search for documents matching the query
    pub fn search(&self, query: &str, limit: usize) -> Vec<BM25Result> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = self.tokenize(query);
        let mut doc_scores: HashMap<DocumentId, f32> = HashMap::new();

        // Calculate BM25 score for each document
        for token in &query_tokens {
            if let Some(doc_freqs) = self.term_frequencies.get(token) {
                let idf = self.calculate_idf(token);

                for (doc_id, tf) in doc_freqs {
                    let doc_length = *self.document_lengths.get(doc_id).unwrap_or(&0);
                    let bm25_term_score = self.calculate_bm25_term_score(*tf, doc_length, idf);

                    *doc_scores.entry(doc_id.clone()).or_insert(0.0) += bm25_term_score;
                }
            }
        }

        // Convert to results and sort by score
        let mut results: Vec<BM25Result> = doc_scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                self.documents.get(&doc_id).map(|doc| BM25Result {
                    doc_id: doc_id.clone(),
                    score,
                    content: doc.content.clone(),
                })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);
        results
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: &DocumentId) -> Option<&Document> {
        self.documents.get(doc_id)
    }

    /// Get total number of indexed documents
    pub fn document_count(&self) -> usize {
        self.total_docs
    }

    /// Get number of unique terms in the index
    pub fn term_count(&self) -> usize {
        self.term_frequencies.len()
    }

    /// Calculate IDF (Inverse Document Frequency) for a term
    /// Uses Lucene-style IDF to avoid negative values for common terms
    fn calculate_idf(&self, term: &str) -> f32 {
        let doc_freq = self.document_frequencies.get(term).unwrap_or(&0);
        if *doc_freq == 0 {
            return 0.0;
        }

        // Lucene-style IDF: log(N/df) + 1, which ensures non-negative values
        (self.total_docs as f32 / *doc_freq as f32).ln() + 1.0
    }

    /// Calculate BM25 term score
    fn calculate_bm25_term_score(&self, tf: f32, doc_length: usize, idf: f32) -> f32 {
        let tf_component = (tf * (self.k1 + 1.0))
            / (tf + self.k1 * (1.0 - self.b + self.b * (doc_length as f32 / self.avg_doc_length)));

        idf * tf_component
    }

    /// Update average document length
    fn update_avg_doc_length(&mut self) {
        if self.total_docs > 0 {
            let total_length: usize = self.document_lengths.values().sum();
            self.avg_doc_length = total_length as f32 / self.total_docs as f32;
        }
    }

    /// Tokenize text into terms
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| {
                // Remove punctuation and clean up
                s.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|s| !s.is_empty() && s.len() > 2 && !self.is_stop_word(s))
            .collect()
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
            "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most", "us",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Clear all indexed data
    pub fn clear(&mut self) {
        self.documents.clear();
        self.term_frequencies.clear();
        self.document_frequencies.clear();
        self.document_lengths.clear();
        self.avg_doc_length = 0.0;
        self.total_docs = 0;
    }

    /// Get statistics about the index
    pub fn get_statistics(&self) -> BM25Statistics {
        BM25Statistics {
            total_documents: self.total_docs,
            total_terms: self.term_frequencies.len(),
            avg_doc_length: self.avg_doc_length,
            parameters: BM25Parameters {
                k1: self.k1,
                b: self.b,
            },
        }
    }
}

impl Default for BM25Retriever {
    fn default() -> Self {
        Self::new()
    }
}

/// BM25 algorithm parameters
#[derive(Debug, Clone)]
pub struct BM25Parameters {
    /// Term frequency saturation parameter
    pub k1: f32,
    /// Length normalization parameter
    pub b: f32,
}

/// Statistics about the BM25 index
#[derive(Debug, Clone)]
pub struct BM25Statistics {
    /// Total number of indexed documents
    pub total_documents: usize,
    /// Total number of unique terms
    pub total_terms: usize,
    /// Average document length in tokens
    pub avg_doc_length: f32,
    /// BM25 algorithm parameters used
    pub parameters: BM25Parameters,
}

impl BM25Statistics {
    /// Print statistics
    pub fn print(&self) {
        println!("BM25 Index Statistics:");
        println!("  Total documents: {}", self.total_documents);
        println!("  Total terms: {}", self.total_terms);
        println!(
            "  Average document length: {:.2} tokens",
            self.avg_doc_length
        );
        println!(
            "  Parameters: k1={:.2}, b={:.2}",
            self.parameters.k1, self.parameters.b
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_documents() -> Vec<Document> {
        vec![
            Document {
                id: "doc1".to_string(),
                content: "The quick brown fox jumps over the lazy dog".to_string(),
                metadata: HashMap::new(),
            },
            Document {
                id: "doc2".to_string(),
                content: "A fast brown animal leaps across a sleeping canine".to_string(),
                metadata: HashMap::new(),
            },
            Document {
                id: "doc3".to_string(),
                content: "The weather is nice today".to_string(),
                metadata: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_bm25_creation() {
        let retriever = BM25Retriever::new();
        assert_eq!(retriever.document_count(), 0);
        assert_eq!(retriever.term_count(), 0);
    }

    #[test]
    fn test_document_indexing() {
        let mut retriever = BM25Retriever::new();
        let docs = create_test_documents();

        retriever.index_documents(&docs).unwrap();

        assert_eq!(retriever.document_count(), 3);
        assert!(retriever.term_count() > 0);
    }

    #[test]
    fn test_search() {
        let mut retriever = BM25Retriever::new();
        let docs = create_test_documents();

        retriever.index_documents(&docs).unwrap();

        let results = retriever.search("brown fox", 10);
        assert!(!results.is_empty());

        // First result should be the most relevant document
        assert_eq!(results[0].doc_id, "doc1");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_tokenization() {
        let retriever = BM25Retriever::new();
        let tokens = retriever.tokenize("The quick, brown fox!");

        // Should filter out stop words and punctuation
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
        assert!(!tokens.contains(&"the".to_string())); // stop word
    }

    #[test]
    fn test_empty_search() {
        let retriever = BM25Retriever::new();
        let results = retriever.search("test", 10);
        assert!(results.is_empty());
    }
}
