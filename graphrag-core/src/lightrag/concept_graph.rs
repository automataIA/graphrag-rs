//! Concept Graph for LazyGraphRAG
//!
//! This module implements the concept graph construction approach from LazyGraphRAG
//! (Microsoft Research, 2025), which eliminates the need for prior summarization
//! by using lightweight noun phrase extraction instead of LLM-based entity extraction.
//!
//! ## Key Features
//!
//! - **No LLM required**: Uses NLP noun phrase extraction instead of LLM calls
//! - **Co-occurrence based**: Builds relationships from concept co-occurrence
//! - **Cost efficient**: 0.1% of full GraphRAG indexing cost
//! - **Fast indexing**: Comparable to vector RAG indexing speed
//!
//! ## Architecture
//!
//! ```text
//! Document → Noun Phrases → Concepts → Co-occurrence Graph → Community Structure
//! ```
//!
//! ## References
//!
//! - LazyGraphRAG (Microsoft Research, November 2024)
//! - E2GraphRAG (May 2025) - SpaCy-based entity extraction
//!
//! ## Example
//!
//! ```rust,no_run
//! use graphrag_core::lightrag::concept_graph::{ConceptExtractor, ConceptGraphBuilder};
//!
//! // Extract concepts from text
//! let extractor = ConceptExtractor::new();
//! let concepts = extractor.extract_concepts("Your document text");
//!
//! // Build concept graph
//! let mut builder = ConceptGraphBuilder::new();
//! builder.add_document_concepts("doc1", concepts);
//! let graph = builder.build();
//! ```

use indexmap::IndexMap;
use petgraph::graph::{DiGraph, NodeIndex};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A concept extracted from text (noun phrase or key term)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Concept {
    /// The concept text (normalized)
    pub text: String,

    /// Concept type (e.g., "noun_phrase", "entity", "keyword")
    pub concept_type: ConceptType,

    /// Frequency in document
    pub frequency: usize,

    /// Document IDs where this concept appears
    pub document_ids: HashSet<String>,

    /// Chunk IDs where this concept appears
    pub chunk_ids: HashSet<String>,
}

/// Type of concept
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConceptType {
    /// Noun phrase (e.g., "machine learning", "knowledge graph")
    NounPhrase,

    /// Named entity (proper noun)
    NamedEntity,

    /// Keyword (important term)
    Keyword,

    /// Technical term
    TechnicalTerm,
}

/// Concept co-occurrence relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelation {
    /// Source concept
    pub source: String,

    /// Target concept
    pub target: String,

    /// Co-occurrence count
    pub count: usize,

    /// Shared chunk IDs
    pub shared_chunks: Vec<String>,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Concept graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptGraph {
    /// All concepts in the graph
    pub concepts: IndexMap<String, Concept>,

    /// Concept relationships (co-occurrence based)
    pub relations: Vec<ConceptRelation>,

    /// Graph structure for traversal
    #[serde(skip)]
    pub graph: DiGraph<String, f32>,

    /// Concept to node index mapping
    #[serde(skip)]
    pub concept_to_node: HashMap<String, NodeIndex>,
}

/// Extracts concepts from text using NLP noun phrase extraction
pub struct ConceptExtractor {
    /// Minimum concept length (in characters)
    min_length: usize,

    /// Maximum concept length (in words)
    max_words: usize,

    /// Pattern for noun phrases
    noun_phrase_pattern: Regex,

    /// Pattern for capitalized terms (potential named entities)
    capitalized_pattern: Regex,

    /// Stopwords to filter out
    stopwords: HashSet<String>,
}

impl ConceptExtractor {
    /// Create a new concept extractor with default settings
    pub fn new() -> Self {
        Self::with_config(ConceptExtractorConfig::default())
    }

    /// Create a concept extractor with custom configuration
    pub fn with_config(config: ConceptExtractorConfig) -> Self {
        // Simple noun phrase pattern: sequences of words
        let noun_phrase_pattern = Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){1,4}\b")
            .expect("Invalid regex pattern");

        // Capitalized terms (potential named entities)
        let capitalized_pattern = Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
            .expect("Invalid regex pattern");

        Self {
            min_length: config.min_length,
            max_words: config.max_words,
            noun_phrase_pattern,
            capitalized_pattern,
            stopwords: Self::default_stopwords(),
        }
    }

    /// Extract concepts from text
    pub fn extract_concepts(&self, text: &str) -> Vec<String> {
        let mut concepts = Vec::new();

        // 1. Extract capitalized noun phrases (likely named entities)
        for cap in self.capitalized_pattern.captures_iter(text) {
            if let Some(phrase) = cap.get(0) {
                let phrase_text = phrase.as_str();
                if self.is_valid_concept(phrase_text) {
                    concepts.push(phrase_text.to_string());
                }
            }
        }

        // 2. Extract general noun phrases
        for cap in self.noun_phrase_pattern.captures_iter(text) {
            if let Some(phrase) = cap.get(0) {
                let phrase_text = phrase.as_str();
                if self.is_valid_concept(phrase_text) {
                    concepts.push(phrase_text.to_string());
                }
            }
        }

        // 3. Extract important keywords (simplified TF-IDF approach)
        let keywords = self.extract_keywords(text);
        concepts.extend(keywords);

        // Deduplicate and normalize
        concepts.sort();
        concepts.dedup();

        concepts
    }

    /// Check if a phrase is a valid concept
    fn is_valid_concept(&self, phrase: &str) -> bool {
        // Check length
        if phrase.len() < self.min_length {
            return false;
        }

        // Check word count
        let word_count = phrase.split_whitespace().count();
        if word_count > self.max_words {
            return false;
        }

        // Check if it's mostly stopwords
        let words: Vec<&str> = phrase.split_whitespace().collect();
        let stopword_count = words.iter()
            .filter(|w| self.stopwords.contains(&w.to_lowercase()))
            .count();

        if stopword_count > words.len() / 2 {
            return false;
        }

        true
    }

    /// Extract keywords using simple term frequency
    fn extract_keywords(&self, text: &str) -> Vec<String> {
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        // Count word frequencies
        for word in text.split_whitespace() {
            let normalized = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();

            if normalized.len() >= self.min_length && !self.stopwords.contains(&normalized) {
                *word_freq.entry(normalized).or_insert(0) += 1;
            }
        }

        // Get top keywords by frequency
        let mut keywords: Vec<_> = word_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));

        keywords.into_iter()
            .take(20) // Top 20 keywords
            .filter(|(_, freq)| *freq >= 2) // Must appear at least twice
            .map(|(word, _)| word)
            .collect()
    }

    /// Default stopwords (English)
    fn default_stopwords() -> HashSet<String> {
        vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "must", "can", "this",
            "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
            "they", "them", "their", "what", "which", "who", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "now",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

/// Configuration for concept extraction
#[derive(Debug, Clone)]
pub struct ConceptExtractorConfig {
    /// Minimum concept length in characters
    pub min_length: usize,

    /// Maximum concept length in words
    pub max_words: usize,
}

impl Default for ConceptExtractorConfig {
    fn default() -> Self {
        Self {
            min_length: 3,
            max_words: 5,
        }
    }
}

/// Builder for creating concept graphs from documents
pub struct ConceptGraphBuilder {
    /// Concepts collected from all documents
    concepts: IndexMap<String, Concept>,

    /// Document ID to concepts mapping
    document_concepts: HashMap<String, Vec<String>>,

    /// Chunk ID to concepts mapping
    chunk_concepts: HashMap<String, Vec<String>>,

    /// Co-occurrence threshold (min shared chunks to create relation)
    co_occurrence_threshold: usize,
}

impl ConceptGraphBuilder {
    /// Create a new concept graph builder
    pub fn new() -> Self {
        Self {
            concepts: IndexMap::new(),
            document_concepts: HashMap::new(),
            chunk_concepts: HashMap::new(),
            co_occurrence_threshold: 1,
        }
    }

    /// Set co-occurrence threshold
    pub fn with_co_occurrence_threshold(mut self, threshold: usize) -> Self {
        self.co_occurrence_threshold = threshold;
        self
    }

    /// Add concepts from a document
    pub fn add_document_concepts(&mut self, document_id: &str, extracted_concepts: Vec<String>) {
        self.document_concepts.insert(
            document_id.to_string(),
            extracted_concepts.clone(),
        );

        // Update concept frequencies
        for concept_text in extracted_concepts {
            let concept = self.concepts
                .entry(concept_text.clone())
                .or_insert_with(|| Concept {
                    text: concept_text.clone(),
                    concept_type: ConceptType::NounPhrase,
                    frequency: 0,
                    document_ids: HashSet::new(),
                    chunk_ids: HashSet::new(),
                });

            concept.frequency += 1;
            concept.document_ids.insert(document_id.to_string());
        }
    }

    /// Add concepts from a chunk
    pub fn add_chunk_concepts(&mut self, chunk_id: &str, extracted_concepts: Vec<String>) {
        self.chunk_concepts.insert(
            chunk_id.to_string(),
            extracted_concepts.clone(),
        );

        // Update concept chunk IDs
        for concept_text in extracted_concepts {
            if let Some(concept) = self.concepts.get_mut(&concept_text) {
                concept.chunk_ids.insert(chunk_id.to_string());
            }
        }
    }

    /// Build the concept graph
    pub fn build(self) -> ConceptGraph {
        let mut graph = DiGraph::new();
        let mut concept_to_node = HashMap::new();

        // Add all concepts as nodes
        for (concept_text, _) in &self.concepts {
            let node_idx = graph.add_node(concept_text.clone());
            concept_to_node.insert(concept_text.clone(), node_idx);
        }

        // Build co-occurrence relations
        let relations = self.build_co_occurrence_relations();

        // Add edges to graph
        for relation in &relations {
            if let (Some(&source_idx), Some(&target_idx)) = (
                concept_to_node.get(&relation.source),
                concept_to_node.get(&relation.target),
            ) {
                graph.add_edge(source_idx, target_idx, relation.confidence);
            }
        }

        ConceptGraph {
            concepts: self.concepts,
            relations,
            graph,
            concept_to_node,
        }
    }

    /// Build co-occurrence relationships between concepts
    fn build_co_occurrence_relations(&self) -> Vec<ConceptRelation> {
        let mut relations = Vec::new();
        let concept_list: Vec<_> = self.concepts.keys().collect();

        // For each pair of concepts
        for i in 0..concept_list.len() {
            for j in (i + 1)..concept_list.len() {
                let concept_a = concept_list[i];
                let concept_b = concept_list[j];

                // Find shared chunks
                if let (Some(concept_a_data), Some(concept_b_data)) = (
                    self.concepts.get(concept_a),
                    self.concepts.get(concept_b),
                ) {
                    let shared_chunks: Vec<String> = concept_a_data
                        .chunk_ids
                        .intersection(&concept_b_data.chunk_ids)
                        .cloned()
                        .collect();

                    if shared_chunks.len() >= self.co_occurrence_threshold {
                        let confidence = self.calculate_confidence(
                            &concept_a_data.chunk_ids,
                            &concept_b_data.chunk_ids,
                            &shared_chunks,
                        );

                        relations.push(ConceptRelation {
                            source: concept_a.clone(),
                            target: concept_b.clone(),
                            count: shared_chunks.len(),
                            shared_chunks,
                            confidence,
                        });
                    }
                }
            }
        }

        relations
    }

    /// Calculate confidence score for a relationship
    fn calculate_confidence(
        &self,
        chunks_a: &HashSet<String>,
        chunks_b: &HashSet<String>,
        shared: &[String],
    ) -> f32 {
        // Jaccard similarity: |intersection| / |union|
        let intersection = shared.len();
        let union = chunks_a.len() + chunks_b.len() - intersection;

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}

impl Default for ConceptExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConceptGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConceptGraph {
    /// Get concepts related to a given concept
    pub fn get_related_concepts(&self, concept: &str, max_results: usize) -> Vec<String> {
        if let Some(&node_idx) = self.concept_to_node.get(concept) {
            let mut related = Vec::new();

            // Get outgoing edges
            for edge in self.graph.edges(node_idx) {
                related.push((self.graph[edge.target()].clone(), *edge.weight()));
            }

            // Sort by confidence
            related.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Return top results
            related.into_iter()
                .take(max_results)
                .map(|(c, _)| c)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get total number of concepts
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Get total number of relations
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_extraction() {
        let extractor = ConceptExtractor::new();
        let text = "Machine Learning and Artificial Intelligence are transforming Natural Language Processing.";
        let concepts = extractor.extract_concepts(text);

        assert!(!concepts.is_empty());
        // Should extract "Machine Learning", "Artificial Intelligence", "Natural Language Processing"
        assert!(concepts.iter().any(|c| c.contains("Machine") || c.contains("Learning")));
    }

    #[test]
    fn test_concept_graph_building() {
        let mut builder = ConceptGraphBuilder::new();

        builder.add_document_concepts("doc1", vec!["concept_a".to_string(), "concept_b".to_string()]);
        builder.add_chunk_concepts("chunk1", vec!["concept_a".to_string(), "concept_b".to_string()]);

        let graph = builder.build();

        assert_eq!(graph.concept_count(), 2);
        // Should create a relation between concept_a and concept_b
        assert!(!graph.relations.is_empty());
    }

    #[test]
    fn test_stopword_filtering() {
        let extractor = ConceptExtractor::new();
        assert!(!extractor.is_valid_concept("the the the"));
        assert!(extractor.is_valid_concept("Machine Learning"));
    }
}
