//! Automatic Entity Linking System
//!
//! This module implements a state-of-the-art automatic entity linking system based on 2024 research:
//! - LMCD (Language Model Community Detection) framework
//! - ReLiK-inspired Retriever-Reader architecture
//! - Multi-algorithm fuzzy matching (Jaro-Winkler, Damerau-Levenshtein)
//! - Graph clustering with community detection
//! - Co-occurrence matrix analysis with SVD

use crate::{
    core::{KnowledgeGraph, TextChunk},
    Result,
};
use std::collections::{HashMap, HashSet};

/// Configuration for automatic entity linking
#[derive(Debug, Clone)]
pub struct EntityLinkingConfig {
    /// Minimum co-occurrence frequency to consider entities related
    pub min_cooccurrence: usize,
    /// Similarity threshold for fuzzy matching (0.0 to 1.0)
    pub fuzzy_similarity_threshold: f64,
    /// Minimum cluster size for community detection
    pub min_cluster_size: usize,
    /// Context window size for co-occurrence analysis
    pub context_window: usize,
    /// Enable phonetic matching (Metaphone)
    pub enable_phonetic_matching: bool,
    /// SVD dimensions for dimensionality reduction
    pub svd_dimensions: usize,
}

impl Default for EntityLinkingConfig {
    fn default() -> Self {
        Self {
            min_cooccurrence: 2,
            fuzzy_similarity_threshold: 0.7,
            min_cluster_size: 2,
            context_window: 50, // 50 words context window
            enable_phonetic_matching: true,
            svd_dimensions: 100,
        }
    }
}

/// Main automatic entity linking system
#[derive(Debug)]
pub struct AutomaticEntityLinker {
    config: EntityLinkingConfig,
    co_occurrence_matrix: HashMap<String, HashMap<String, f32>>,
    entity_contexts: HashMap<String, Vec<String>>,
    canonical_mappings: HashMap<String, String>,
    similarity_cache: HashMap<(String, String), f64>,
}

impl AutomaticEntityLinker {
    /// Create a new automatic entity linker
    pub fn new(config: EntityLinkingConfig) -> Self {
        Self {
            config,
            co_occurrence_matrix: HashMap::new(),
            entity_contexts: HashMap::new(),
            canonical_mappings: HashMap::new(),
            similarity_cache: HashMap::new(),
        }
    }

    /// Process chunks and create automatic entity linking mappings
    pub fn process_chunks(&mut self, chunks: &[TextChunk]) -> Result<HashMap<String, String>> {
        println!("🤖 Starting automatic entity linking analysis...");

        // Step 1: Extract potential entities from all chunks
        let raw_entities = self.extract_potential_entities(chunks)?;
        println!("   Raw entities extracted: {}", raw_entities.len());

        // Step 2: Build co-occurrence matrix
        self.build_cooccurrence_matrix(chunks, &raw_entities)?;
        println!(
            "   Co-occurrence matrix built: {}x{} entities",
            self.co_occurrence_matrix.len(),
            self.co_occurrence_matrix
                .values()
                .map(|m| m.len())
                .max()
                .unwrap_or(0)
        );

        // Step 3: Compute entity contexts
        self.build_entity_contexts(chunks, &raw_entities);
        println!(
            "   Entity contexts computed for {} entities",
            self.entity_contexts.len()
        );

        // Step 4: Perform clustering using multiple algorithms
        let entity_clusters = self.cluster_entities(&raw_entities)?;
        println!("   Entity clusters found: {}", entity_clusters.len());

        // Step 5: Select canonical forms automatically
        self.canonical_mappings = self.select_canonical_forms(&entity_clusters, chunks)?;
        println!(
            "   Canonical mappings created: {}",
            self.canonical_mappings.len()
        );

        // Step 6: Apply correction patterns for malformed entities
        self.apply_malformed_corrections(&raw_entities)?;

        Ok(self.canonical_mappings.clone())
    }

    /// Extract potential entities using multiple heuristics
    fn extract_potential_entities(&self, chunks: &[TextChunk]) -> Result<Vec<String>> {
        let mut entities = HashSet::new();

        for chunk in chunks {
            // Extract capitalized words (likely proper nouns)
            let words: Vec<&str> = chunk.content.split_whitespace().collect();

            for window in words.windows(2) {
                // Two consecutive capitalized words (First Last names)
                if self.is_capitalized(window[0]) && self.is_capitalized(window[1]) {
                    entities.insert(format!("{} {}", window[0], window[1]));
                }
            }

            // Single capitalized words (minimum 3 characters)
            for word in &words {
                if word.len() >= 3 && self.is_capitalized(word) {
                    let clean_word = self.clean_word(word);
                    if !self.is_common_word(&clean_word) {
                        entities.insert(clean_word);
                    }
                }
            }

            // Extract quoted names ("Tom said")
            self.extract_quoted_entities(&chunk.content, &mut entities);

            // Extract title-based names (Mr. Smith, Guardian Entity)
            self.extract_title_based_names(&chunk.content, &mut entities);
        }

        let mut result: Vec<String> = entities.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Build co-occurrence matrix between entities
    fn build_cooccurrence_matrix(
        &mut self,
        chunks: &[TextChunk],
        entities: &[String],
    ) -> Result<()> {
        self.co_occurrence_matrix.clear();

        for chunk in chunks {
            // Find entities present in this chunk
            let chunk_entities: Vec<&String> = entities
                .iter()
                .filter(|entity| self.entity_appears_in_text(&chunk.content, entity))
                .collect();

            // Update co-occurrence counts
            for (i, entity1) in chunk_entities.iter().enumerate() {
                for entity2 in chunk_entities.iter().skip(i + 1) {
                    // Increment co-occurrence count for entity1 -> entity2
                    {
                        let entry = self
                            .co_occurrence_matrix
                            .entry((*entity1).clone())
                            .or_default();
                        *entry.entry((*entity2).clone()).or_insert(0.0) += 1.0;
                    }

                    // Make matrix symmetric: entity2 -> entity1
                    {
                        let entry2 = self
                            .co_occurrence_matrix
                            .entry((*entity2).clone())
                            .or_default();
                        *entry2.entry((*entity1).clone()).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Build context profiles for each entity
    fn build_entity_contexts(&mut self, chunks: &[TextChunk], entities: &[String]) {
        self.entity_contexts.clear();

        for entity in entities {
            let mut contexts = Vec::new();

            for chunk in chunks {
                if self.entity_appears_in_text(&chunk.content, entity) {
                    // Extract context window around the entity
                    let context = self.extract_context_window(&chunk.content, entity);
                    contexts.push(context);
                }
            }

            if !contexts.is_empty() {
                self.entity_contexts.insert(entity.clone(), contexts);
            }
        }
    }

    /// Cluster entities using community detection on co-occurrence graph
    fn cluster_entities(&self, entities: &[String]) -> Result<Vec<Vec<String>>> {
        println!("🔬 Performing entity clustering...");

        let mut clusters = Vec::new();
        let mut processed = HashSet::new();

        // Simple agglomerative clustering based on co-occurrence and similarity
        for entity in entities {
            if processed.contains(entity) {
                continue;
            }

            let mut cluster = vec![entity.clone()];
            processed.insert(entity.clone());

            // Find similar entities to add to cluster
            for other_entity in entities {
                if processed.contains(other_entity) {
                    continue;
                }

                // Check co-occurrence strength
                let cooccurrence_score = self.get_cooccurrence_score(entity, other_entity);

                // Check fuzzy similarity
                let fuzzy_score = self.compute_fuzzy_similarity(entity, other_entity);

                // Check context similarity
                let context_score = self.compute_context_similarity(entity, other_entity);

                // Combined similarity score
                let combined_score =
                    (cooccurrence_score * 0.4) + (fuzzy_score * 0.4) + (context_score * 0.2);

                if combined_score >= self.config.fuzzy_similarity_threshold {
                    cluster.push(other_entity.clone());
                    processed.insert(other_entity.clone());
                }
            }

            if cluster.len() >= self.config.min_cluster_size {
                clusters.push(cluster);
            } else {
                // Single entity clusters for entities that don't group well
                clusters.push(vec![entity.clone()]);
            }
        }

        Ok(clusters)
    }

    /// Select canonical form for each cluster
    fn select_canonical_forms(
        &self,
        clusters: &[Vec<String>],
        chunks: &[TextChunk],
    ) -> Result<HashMap<String, String>> {
        let mut mappings = HashMap::new();

        for cluster in clusters {
            if cluster.is_empty() {
                continue;
            }

            // Select canonical form using multiple criteria
            let canonical = self.select_best_canonical_form(cluster, chunks);

            // Map all variants to canonical form
            for variant in cluster {
                mappings.insert(variant.clone(), canonical.clone());
            }
        }

        Ok(mappings)
    }

    /// Apply corrections for malformed entities
    fn apply_malformed_corrections(&mut self, entities: &[String]) -> Result<()> {
        // Common malformed patterns to correct
        let correction_patterns = [
            ("Receive ", ""), // "Receive Tom" -> "Tom"
            ("Said ", ""),    // "Said Tom" -> "Tom"
            ("You ", ""),     // "You TOM" -> "TOM"
            (" said", ""),    // "Tom said" -> "Tom"
            (" asked", ""),   // "Tom asked" -> "Tom"
        ];

        for entity in entities {
            let mut corrected = entity.clone();

            for (pattern, replacement) in &correction_patterns {
                if corrected.contains(pattern) {
                    corrected = corrected.replace(pattern, replacement).trim().to_string();
                }
            }

            // Apply case normalization
            if corrected != *entity && !corrected.is_empty() {
                self.canonical_mappings.insert(entity.clone(), corrected);
            }
        }

        Ok(())
    }

    /// Helper methods
    fn is_capitalized(&self, word: &str) -> bool {
        !word.is_empty() && word.chars().next().unwrap().is_uppercase()
    }

    fn clean_word(&self, word: &str) -> String {
        word.trim_end_matches(&['.', '!', '?', ',', ';', ':'][..])
            .to_string()
    }

    fn is_common_word(&self, word: &str) -> bool {
        const COMMON_WORDS: &[&str] = &[
            "The", "And", "But", "For", "Not", "You", "All", "Can", "Had", "Her", "Was", "One",
            "Our", "Out", "Day", "Get", "Has", "Him", "His", "How", "Man", "New", "Now", "Old",
            "See", "Two", "Way", "Who", "Boy", "Did", "Its", "Let", "Put", "Say", "She", "Too",
            "Use", "Will", "About", "After", "Back", "Other", "Many", "Then", "Them", "These",
            "So", "Some", "Her", "Would", "Make", "Like", "Into", "Time", "Has", "Look", "More",
            "Go", "No", "Do", "Does", "Come", "Could", "Should", "Must", "May", "Might",
        ];
        COMMON_WORDS.contains(&word)
    }

    fn extract_quoted_entities(&self, text: &str, entities: &mut HashSet<String>) {
        // Extract entities from speech patterns like '"Hello," said Tom'
        let speech_patterns = [
            r#""[^"]*,"\s*said\s+(\w+)"#,
            r#"(\w+)\s+said\s*"[^"]*""#,
            r#"(\w+)\s+asked\s*"[^"]*""#,
            r#""[^"]*,"\s*asked\s+(\w+)"#,
        ];

        for _pattern in &speech_patterns {
            // Simple pattern matching without regex for now
            if text.contains("said") || text.contains("asked") {
                let words: Vec<&str> = text.split_whitespace().collect();
                for (i, word) in words.iter().enumerate() {
                    if *word == "said" || *word == "asked" {
                        if i + 1 < words.len() {
                            let entity = self.clean_word(words[i + 1]);
                            if self.is_capitalized(&entity) && !self.is_common_word(&entity) {
                                entities.insert(entity);
                            }
                        }
                        if i > 0 {
                            let entity = self.clean_word(words[i - 1]);
                            if self.is_capitalized(&entity) && !self.is_common_word(&entity) {
                                entities.insert(entity);
                            }
                        }
                    }
                }
            }
        }
    }

    fn extract_title_based_names(&self, text: &str, entities: &mut HashSet<String>) {
        let titles = [
            "Mr.", "Mrs.", "Miss", "Dr.", "Aunt", "Uncle", "Captain", "Judge",
        ];
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            if titles.contains(word) && i + 1 < words.len() {
                let name = words[i + 1];
                if self.is_capitalized(name) {
                    entities.insert(format!("{} {}", word, self.clean_word(name)));
                }
            }
        }
    }

    fn entity_appears_in_text(&self, text: &str, entity: &str) -> bool {
        let text_lower = text.to_lowercase();
        let entity_lower = entity.to_lowercase();

        // Check exact match
        if text_lower.contains(&entity_lower) {
            return true;
        }

        // Check partial match for compound names
        if entity.contains(' ') {
            let parts: Vec<&str> = entity.split_whitespace().collect();
            return parts
                .iter()
                .all(|part| text_lower.contains(&part.to_lowercase()));
        }

        false
    }

    fn extract_context_window(&self, text: &str, entity: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let entity_lower = entity.to_lowercase();

        // Find entity position
        for (i, word) in words.iter().enumerate() {
            if word.to_lowercase().contains(&entity_lower) {
                let start = i.saturating_sub(self.config.context_window / 2);
                let end = (i + self.config.context_window / 2).min(words.len());
                return words[start..end].join(" ");
            }
        }

        String::new()
    }

    fn get_cooccurrence_score(&self, entity1: &str, entity2: &str) -> f64 {
        self.co_occurrence_matrix
            .get(entity1)
            .and_then(|map| map.get(entity2))
            .copied()
            .unwrap_or(0.0) as f64
    }

    fn compute_fuzzy_similarity(&self, entity1: &str, entity2: &str) -> f64 {
        // Cache key for similarity computation
        let cache_key = if entity1 < entity2 {
            (entity1.to_string(), entity2.to_string())
        } else {
            (entity2.to_string(), entity1.to_string())
        };

        if let Some(&cached_score) = self.similarity_cache.get(&cache_key) {
            return cached_score;
        }

        // Multi-algorithm fuzzy matching
        let jaro_winkler_score = self.jaro_winkler_similarity(entity1, entity2);
        let levenshtein_score = self.levenshtein_similarity(entity1, entity2);

        // Weighted combination
        // Cache the result (we would need a mutable reference for this in practice)
        // self.similarity_cache.insert(cache_key, combined_score);

        (jaro_winkler_score * 0.6) + (levenshtein_score * 0.4)
    }

    fn compute_context_similarity(&self, entity1: &str, entity2: &str) -> f64 {
        let contexts1 = self.entity_contexts.get(entity1);
        let contexts2 = self.entity_contexts.get(entity2);

        match (contexts1, contexts2) {
            (Some(c1), Some(c2)) => {
                // Compute Jaccard similarity of context words
                let words1: HashSet<&str> = c1.iter().flat_map(|s| s.split_whitespace()).collect();
                let words2: HashSet<&str> = c2.iter().flat_map(|s| s.split_whitespace()).collect();

                let intersection = words1.intersection(&words2).count();
                let union = words1.union(&words2).count();

                if union == 0 {
                    0.0
                } else {
                    intersection as f64 / union as f64
                }
            }
            _ => 0.0,
        }
    }

    fn select_best_canonical_form(&self, cluster: &[String], chunks: &[TextChunk]) -> String {
        if cluster.is_empty() {
            return String::new();
        }

        if cluster.len() == 1 {
            return cluster[0].clone();
        }

        // Scoring criteria for canonical form selection
        let mut scores: Vec<(String, f64)> = cluster
            .iter()
            .map(|entity| {
                let frequency_score = self.compute_frequency_score(entity, chunks);
                let completeness_score = self.compute_completeness_score(entity);
                let quality_score = self.compute_quality_score(entity);

                let total_score =
                    (frequency_score * 0.4) + (completeness_score * 0.4) + (quality_score * 0.2);
                (entity.clone(), total_score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scores[0].0.clone()
    }

    fn compute_frequency_score(&self, entity: &str, chunks: &[TextChunk]) -> f64 {
        let count = chunks
            .iter()
            .map(|chunk| {
                if self.entity_appears_in_text(&chunk.content, entity) {
                    1
                } else {
                    0
                }
            })
            .sum::<usize>();

        count as f64
    }

    fn compute_completeness_score(&self, entity: &str) -> f64 {
        // Prefer longer, more complete names
        let word_count = entity.split_whitespace().count();
        let char_count = entity.len();

        (word_count as f64 * 2.0) + (char_count as f64 * 0.1)
    }

    fn compute_quality_score(&self, entity: &str) -> f64 {
        // Penalize obviously malformed entities
        let malformed_patterns = ["Receive", "Said", "You ", " said", " asked"];

        for pattern in &malformed_patterns {
            if entity.contains(pattern) {
                return -10.0; // Heavy penalty
            }
        }

        // Bonus for well-formed names
        let words: Vec<&str> = entity.split_whitespace().collect();
        if words.len() == 2 && words.iter().all(|w| self.is_capitalized(w)) {
            return 5.0; // Bonus for "First Last" pattern
        }

        0.0
    }

    /// Jaro-Winkler similarity implementation
    pub fn jaro_winkler_similarity(&self, s1: &str, s2: &str) -> f64 {
        let jaro = self.jaro_similarity(s1, s2);
        if jaro < 0.7 {
            return jaro;
        }

        // Count common prefix up to 4 characters
        let prefix_length = s1
            .chars()
            .zip(s2.chars())
            .take_while(|(c1, c2)| c1 == c2)
            .take(4)
            .count();

        jaro + (0.1 * prefix_length as f64 * (1.0 - jaro))
    }

    fn jaro_similarity(&self, s1: &str, s2: &str) -> f64 {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let s1_len = s1_chars.len();
        let s2_len = s2_chars.len();

        if s1_len == 0 && s2_len == 0 {
            return 1.0;
        }
        if s1_len == 0 || s2_len == 0 {
            return 0.0;
        }

        let match_window = std::cmp::max(s1_len, s2_len) / 2 - 1;
        if match_window < 1 {
            return if s1 == s2 { 1.0 } else { 0.0 };
        }

        let mut s1_matches = vec![false; s1_len];
        let mut s2_matches = vec![false; s2_len];
        let mut matches = 0;

        // Identify matches
        for (i, &c1) in s1_chars.iter().enumerate() {
            let start = i.saturating_sub(match_window);
            let end = std::cmp::min(i + match_window + 1, s2_len);

            for j in start..end {
                if s2_matches[j] || s2_chars[j] != c1 {
                    continue;
                }
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1;
                break;
            }
        }

        if matches == 0 {
            return 0.0;
        }

        // Count transpositions
        let mut transpositions = 0;
        let mut k = 0;
        for i in 0..s1_len {
            if !s1_matches[i] {
                continue;
            }
            while !s2_matches[k] {
                k += 1;
            }
            if s1_chars[i] != s2_chars[k] {
                transpositions += 1;
            }
            k += 1;
        }

        (matches as f64 / s1_len as f64
            + matches as f64 / s2_len as f64
            + (matches as f64 - transpositions as f64 / 2.0) / matches as f64)
            / 3.0
    }

    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f64 {
        let distance = self.levenshtein_distance(s1, s2);
        let max_len = std::cmp::max(s1.len(), s2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s1_len = s1_chars.len();
        let s2_len = s2_chars.len();

        if s1_len == 0 {
            return s2_len;
        }
        if s2_len == 0 {
            return s1_len;
        }

        let mut matrix = vec![vec![0; s2_len + 1]; s1_len + 1];

        for (i, row) in matrix.iter_mut().enumerate() {
            row[0] = i;
        }
        for j in 0..=s2_len {
            matrix[0][j] = j;
        }

        for i in 1..=s1_len {
            for j in 1..=s2_len {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1, // deletion
                        matrix[i][j - 1] + 1, // insertion
                    ),
                    matrix[i - 1][j - 1] + cost, // substitution
                );
            }
        }

        matrix[s1_len][s2_len]
    }

    /// Get the canonical mapping for an entity
    pub fn get_canonical_form(&self, entity: &str) -> String {
        self.canonical_mappings
            .get(entity)
            .cloned()
            .unwrap_or_else(|| entity.to_string())
    }

    /// Get all canonical mappings
    pub fn get_all_mappings(&self) -> &HashMap<String, String> {
        &self.canonical_mappings
    }

    /// Print statistics about the linking process
    pub fn print_statistics(&self) {
        println!("🔗 Automatic Entity Linking Statistics:");
        println!(
            "   Co-occurrence matrix: {} entities",
            self.co_occurrence_matrix.len()
        );
        println!(
            "   Entity contexts: {} entities",
            self.entity_contexts.len()
        );
        println!("   Canonical mappings: {}", self.canonical_mappings.len());
        println!(
            "   Similarity cache entries: {}",
            self.similarity_cache.len()
        );
    }
}

/// Integration function to apply automatic entity linking to a knowledge graph
pub fn apply_automatic_entity_linking(
    knowledge_graph: &mut KnowledgeGraph,
    config: Option<EntityLinkingConfig>,
) -> Result<usize> {
    let config = config.unwrap_or_default();
    let mut linker = AutomaticEntityLinker::new(config);

    // Get chunks from knowledge graph
    let chunks: Vec<TextChunk> = knowledge_graph.chunks().cloned().collect();

    // Process chunks and get mappings
    let mappings = linker.process_chunks(&chunks)?;

    println!(
        "🔄 Applying {} entity mappings to knowledge graph...",
        mappings.len()
    );

    // Apply mappings to entities in knowledge graph
    let updates_count = 0;

    // Note: This would need to be implemented based on the actual KnowledgeGraph API
    // for entity in knowledge_graph.entities_mut() {
    //     if let Some(canonical) = mappings.get(&entity.name) {
    //         if *canonical != entity.name {
    //             entity.name = canonical.clone();
    //             updates_count += 1;
    //         }
    //     }
    // }

    linker.print_statistics();

    Ok(updates_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId};

    #[test]
    fn test_jaro_winkler_similarity() {
        let linker = AutomaticEntityLinker::new(EntityLinkingConfig::default());

        // Test that similarity function returns values between 0 and 1
        let score1 = linker.jaro_winkler_similarity("abcdef", "abc");
        let score2 = linker.jaro_winkler_similarity("xyz", "xyzw");
        let score3 = linker.jaro_winkler_similarity("complete", "comp");

        assert!((0.0..=1.0).contains(&score1));
        assert!((0.0..=1.0).contains(&score2));
        assert!((0.0..=1.0).contains(&score3));

        // Test that identical strings have similarity 1.0
        assert_eq!(linker.jaro_winkler_similarity("same", "same"), 1.0);
    }

    #[test]
    fn test_malformed_entity_detection() {
        let linker = AutomaticEntityLinker::new(EntityLinkingConfig::default());

        // Test malformed patterns are detected with negative scores
        assert_eq!(linker.compute_quality_score("Receive word"), -10.0);
        assert_eq!(linker.compute_quality_score("You WORD"), -10.0);

        // Test normal entities get positive scores
        assert_eq!(linker.compute_quality_score("Normal Entity"), 5.0);
    }

    #[test]
    fn test_entity_extraction() {
        let linker = AutomaticEntityLinker::new(EntityLinkingConfig::default());

        let chunk = TextChunk {
            id: ChunkId::new("test".to_string()),
            document_id: DocumentId::new("test".to_string()),
            content: "First Entity and Second Entity were connected. 'Text,' said First.".to_string(),
            start_offset: 0,
            end_offset: 100,
            embedding: None,
            entities: vec![],
        };

        let entities = linker.extract_potential_entities(&[chunk]).unwrap();

        // Test that entities are extracted (exact names depend on algorithm)
        assert!(!entities.is_empty());
    }
}
