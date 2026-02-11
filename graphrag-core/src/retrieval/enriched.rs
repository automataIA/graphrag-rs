//! Enriched metadata-aware retrieval
//!
//! This module provides retrieval strategies that leverage enriched chunk metadata
//! (chapters, sections, keywords, summaries) to improve search relevance and precision.

use crate::{
    core::{KnowledgeGraph, TextChunk},
    retrieval::{QueryAnalysis, SearchResult, ResultType},
    Result,
};
use std::collections::{HashMap, HashSet};

/// Configuration for enriched metadata retrieval
#[derive(Debug, Clone)]
pub struct EnrichedRetrievalConfig {
    /// Weight for keyword matching (0.0 to 1.0)
    pub keyword_match_weight: f32,
    /// Weight for chapter/section context matching (0.0 to 1.0)
    pub structure_match_weight: f32,
    /// Weight for summary relevance (0.0 to 1.0)
    pub summary_weight: f32,
    /// Minimum number of keywords to match for boosting
    pub min_keyword_matches: usize,
    /// Enable chapter/section filtering
    pub enable_structure_filtering: bool,
}

impl Default for EnrichedRetrievalConfig {
    fn default() -> Self {
        Self {
            keyword_match_weight: 0.3,
            structure_match_weight: 0.2,
            summary_weight: 0.15,
            min_keyword_matches: 1,
            enable_structure_filtering: true,
        }
    }
}

/// Metadata-enhanced retrieval strategies
pub struct EnrichedRetriever {
    config: EnrichedRetrievalConfig,
}

impl EnrichedRetriever {
    /// Create a new enriched retriever
    pub fn new() -> Self {
        Self {
            config: EnrichedRetrievalConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EnrichedRetrievalConfig) -> Self {
        Self { config }
    }

    /// Search chunks using enriched metadata
    ///
    /// This method boosts chunks that match:
    /// 1. Query keywords present in chunk keywords (TF-IDF extracted)
    /// 2. Chapter/Section mentioned in query
    /// 3. Summary content relevant to query
    pub fn metadata_search(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
        _analysis: &QueryAnalysis,
        base_results: &[SearchResult],
    ) -> Result<Vec<SearchResult>> {
        let mut enriched_results = Vec::new();

        // Extract query keywords and potential structure references
        let query_lower = query.to_lowercase();
        let query_words: HashSet<String> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_string())
            .collect();

        // Detect chapter/section references in query
        let structure_refs = self.extract_structure_references(&query_lower);

        // Process each chunk in the graph
        for chunk in graph.chunks() {
            if !chunk.entities.is_empty() || !chunk.metadata.keywords.is_empty() {
                let mut base_score = self.find_base_score(chunk, base_results);
                let mut metadata_boost = 0.0;

                // 1. KEYWORD MATCHING BOOST
                let keyword_matches = self.count_keyword_matches(&chunk.metadata.keywords, &query_words);
                if keyword_matches >= self.config.min_keyword_matches {
                    let keyword_boost = (keyword_matches as f32 / query_words.len().max(1) as f32)
                        * self.config.keyword_match_weight;
                    metadata_boost += keyword_boost;
                }

                // 2. STRUCTURE MATCHING BOOST (Chapter/Section)
                if self.config.enable_structure_filtering {
                    if let Some(structure_boost) = self.calculate_structure_boost(chunk, &structure_refs) {
                        metadata_boost += structure_boost * self.config.structure_match_weight;
                    }
                }

                // 3. SUMMARY RELEVANCE BOOST
                if let Some(summary) = &chunk.metadata.summary {
                    if self.matches_query(summary, &query_words) {
                        metadata_boost += self.config.summary_weight;
                    }
                }

                // 4. COMPLETENESS BONUS
                let completeness = chunk.metadata.completeness_score();
                if completeness > 0.7 {
                    metadata_boost += 0.05; // Small bonus for high-quality metadata
                }

                // Apply boost only if significant
                if metadata_boost > 0.05 {
                    base_score = (base_score + metadata_boost).min(1.0);

                    enriched_results.push(SearchResult {
                        id: chunk.id.to_string(),
                        content: chunk.content.clone(),
                        score: base_score,
                        result_type: ResultType::Chunk,
                        entities: chunk
                            .entities
                            .iter()
                            .filter_map(|eid| graph.get_entity(eid))
                            .map(|e| e.name.clone())
                            .collect(),
                        source_chunks: vec![chunk.id.to_string()],
                    });
                }
            }
        }

        Ok(enriched_results)
    }

    /// Filter chunks by chapter or section
    ///
    /// Example: "What does Socrates say in Chapter 1?" -> filter to Chapter 1 chunks
    pub fn filter_by_structure(
        &self,
        query: &str,
        results: Vec<SearchResult>,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        let structure_refs = self.extract_structure_references(&query.to_lowercase());

        if structure_refs.is_empty() {
            return Ok(results);
        }

        let filtered: Vec<SearchResult> = results
            .into_iter()
            .filter(|result| {
                // Get chunk metadata
                if let Some(chunk_id) = result.source_chunks.first() {
                    if let Some(chunk) = graph.chunks().find(|c| c.id.to_string() == *chunk_id) {
                        return self.matches_structure(&chunk.metadata, &structure_refs);
                    }
                }
                true // Keep results without structure metadata
            })
            .collect();

        Ok(filtered)
    }

    /// Boost results based on enriched metadata
    pub fn boost_with_metadata(
        &self,
        mut results: Vec<SearchResult>,
        query: &str,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        let query_words: HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_string())
            .collect();

        for result in &mut results {
            if let Some(chunk_id) = result.source_chunks.first() {
                if let Some(chunk) = graph.chunks().find(|c| c.id.to_string() == *chunk_id) {
                    // Boost based on keyword matches
                    let keyword_matches = self.count_keyword_matches(&chunk.metadata.keywords, &query_words);
                    if keyword_matches > 0 {
                        let boost = (keyword_matches as f32 / query_words.len().max(1) as f32) * 0.2;
                        result.score = (result.score + boost).min(1.0);
                    }

                    // Boost if chapter/section matches query context
                    if let Some(chapter) = &chunk.metadata.chapter {
                        if query.to_lowercase().contains(&chapter.to_lowercase()) {
                            result.score = (result.score + 0.15).min(1.0);
                        }
                    }

                    if let Some(section) = &chunk.metadata.section {
                        if query.to_lowercase().contains(&section.to_lowercase()) {
                            result.score = (result.score + 0.1).min(1.0);
                        }
                    }
                }
            }
        }

        // Re-sort after boosting
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    /// Get chunks from a specific chapter
    pub fn get_chapter_chunks<'a>(&self, chapter_name: &str, graph: &'a KnowledgeGraph) -> Vec<&'a TextChunk> {
        graph
            .chunks()
            .filter(|chunk| {
                if let Some(ch) = &chunk.metadata.chapter {
                    ch.to_lowercase().contains(&chapter_name.to_lowercase())
                } else {
                    false
                }
            })
            .collect()
    }

    /// Get chunks from a specific section
    pub fn get_section_chunks<'a>(&self, section_name: &str, graph: &'a KnowledgeGraph) -> Vec<&'a TextChunk> {
        graph
            .chunks()
            .filter(|chunk| {
                if let Some(sec) = &chunk.metadata.section {
                    sec.to_lowercase().contains(&section_name.to_lowercase())
                } else {
                    false
                }
            })
            .collect()
    }

    /// Search by keywords extracted from chunks
    pub fn search_by_keywords(
        &self,
        keywords: &[String],
        graph: &KnowledgeGraph,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut keyword_scores: HashMap<String, (f32, &TextChunk)> = HashMap::new();

        for chunk in graph.chunks() {
            let mut score = 0.0;
            for keyword in keywords {
                if chunk.metadata.keywords.iter().any(|k| k.eq_ignore_ascii_case(keyword)) {
                    score += 1.0 / keywords.len() as f32;
                }
            }

            if score > 0.0 {
                keyword_scores.insert(chunk.id.to_string(), (score, chunk));
            }
        }

        let mut sorted_results: Vec<_> = keyword_scores.into_iter().collect();
        sorted_results.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());

        sorted_results
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, (score, chunk))| SearchResult {
                id: chunk_id.clone(),
                content: chunk.content.clone(),
                score,
                result_type: ResultType::Chunk,
                entities: chunk
                    .entities
                    .iter()
                    .filter_map(|eid| graph.get_entity(eid))
                    .map(|e| e.name.clone())
                    .collect(),
                source_chunks: vec![chunk_id],
            })
            .collect()
    }

    // === HELPER METHODS ===

    /// Count matching keywords between chunk and query
    fn count_keyword_matches(&self, chunk_keywords: &[String], query_words: &HashSet<String>) -> usize {
        chunk_keywords
            .iter()
            .filter(|k| query_words.contains(&k.to_lowercase()))
            .count()
    }

    /// Find base score from existing results
    fn find_base_score(&self, chunk: &TextChunk, base_results: &[SearchResult]) -> f32 {
        base_results
            .iter()
            .find(|r| r.source_chunks.contains(&chunk.id.to_string()))
            .map(|r| r.score)
            .unwrap_or(0.5) // Default moderate score
    }

    /// Extract chapter/section references from query
    fn extract_structure_references(&self, query_lower: &str) -> Vec<String> {
        let mut refs = Vec::new();

        // Detect "chapter X" or "section Y" patterns
        let patterns = [
            r"chapter\s+(\d+|[ivxlcdm]+|\w+)",
            r"section\s+(\d+\.?\d*)",
            r"part\s+(\d+|[ivxlcdm]+)",
        ];

        for pattern in &patterns {
            if let Some(captures) = regex::Regex::new(pattern).ok().and_then(|re| re.captures(query_lower)) {
                if let Some(matched) = captures.get(0) {
                    refs.push(matched.as_str().to_string());
                }
            }
        }

        // Also check for direct mentions like "Introduction", "Conclusion"
        for word in query_lower.split_whitespace() {
            if word.chars().next().is_some_and(|c| c.is_uppercase()) && word.len() > 5 {
                refs.push(word.to_string());
            }
        }

        refs
    }

    /// Calculate structure boost for chunk
    fn calculate_structure_boost(
        &self,
        chunk: &TextChunk,
        structure_refs: &[String],
    ) -> Option<f32> {
        if structure_refs.is_empty() {
            return None;
        }

        let mut boost = 0.0;

        for reference in structure_refs {
            let ref_lower = reference.to_lowercase();

            if let Some(chapter) = &chunk.metadata.chapter {
                if chapter.to_lowercase().contains(&ref_lower) {
                    boost += 0.5;
                }
            }

            if let Some(section) = &chunk.metadata.section {
                if section.to_lowercase().contains(&ref_lower) {
                    boost += 0.3;
                }
            }

            if let Some(subsection) = &chunk.metadata.subsection {
                if subsection.to_lowercase().contains(&ref_lower) {
                    boost += 0.2;
                }
            }
        }

        if boost > 0.0 {
            Some(boost)
        } else {
            None
        }
    }

    /// Check if text matches query words
    fn matches_query(&self, text: &str, query_words: &HashSet<String>) -> bool {
        let text_lower = text.to_lowercase();
        query_words
            .iter()
            .filter(|word| text_lower.contains(word.as_str()))
            .count()
            >= (query_words.len() / 2).max(1)
    }

    /// Check if chunk metadata matches structure references
    fn matches_structure(
        &self,
        metadata: &crate::core::ChunkMetadata,
        structure_refs: &[String],
    ) -> bool {
        for reference in structure_refs {
            let ref_lower = reference.to_lowercase();

            if let Some(chapter) = &metadata.chapter {
                if chapter.to_lowercase().contains(&ref_lower) {
                    return true;
                }
            }

            if let Some(section) = &metadata.section {
                if section.to_lowercase().contains(&ref_lower) {
                    return true;
                }
            }

            if let Some(subsection) = &metadata.subsection {
                if subsection.to_lowercase().contains(&ref_lower) {
                    return true;
                }
            }
        }

        false
    }
}

impl Default for EnrichedRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, ChunkMetadata, DocumentId, KnowledgeGraph, TextChunk};

    fn create_test_chunk(id: &str, content: &str, keywords: Vec<String>, chapter: Option<String>) -> TextChunk {
        let mut chunk = TextChunk::new(
            ChunkId::new(id.to_string()),
            DocumentId::new("test_doc".to_string()),
            content.to_string(),
            0,
            content.len(),
        );

        let mut metadata = ChunkMetadata::new();
        metadata.keywords = keywords;
        metadata.chapter = chapter;
        chunk.metadata = metadata;

        chunk
    }

    #[test]
    fn test_keyword_matching() {
        let retriever = EnrichedRetriever::new();
        let chunk_keywords = vec!["machine".to_string(), "learning".to_string(), "neural".to_string()];
        let query_words: HashSet<String> = vec!["machine".to_string(), "learning".to_string()]
            .into_iter()
            .collect();

        let matches = retriever.count_keyword_matches(&chunk_keywords, &query_words);
        assert_eq!(matches, 2);
    }

    #[test]
    fn test_structure_extraction() {
        let retriever = EnrichedRetriever::new();
        let query = "What does Socrates say in chapter 1?";
        let refs = retriever.extract_structure_references(&query.to_lowercase());

        assert!(!refs.is_empty());
    }

    #[test]
    fn test_chapter_filtering() {
        let retriever = EnrichedRetriever::new();
        let mut graph = KnowledgeGraph::new();

        let chunk1 = create_test_chunk(
            "chunk1",
            "Content from chapter 1",
            vec!["content".to_string()],
            Some("Chapter 1: Introduction".to_string()),
        );

        let chunk2 = create_test_chunk(
            "chunk2",
            "Content from chapter 2",
            vec!["content".to_string()],
            Some("Chapter 2: Methods".to_string()),
        );

        let _ = graph.add_chunk(chunk1);
        let _ = graph.add_chunk(chunk2);

        let chapter1_chunks = retriever.get_chapter_chunks("Chapter 1", &graph);
        assert_eq!(chapter1_chunks.len(), 1);
    }
}
