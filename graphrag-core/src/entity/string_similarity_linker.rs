//! String Similarity-based Entity Linking
//!
//! Deterministic entity linking using string similarity metrics without ML.
//! Implements multiple algorithms:
//! - Levenshtein edit distance
//! - Jaro-Winkler similarity
//! - Jaccard similarity (token-based)
//! - Exact match with normalization
//! - Phonetic matching (Soundex, Metaphone)

use crate::core::{Entity, EntityId, KnowledgeGraph};
use crate::Result;
use std::collections::{HashMap, HashSet};

/// Configuration for string similarity-based entity linking
#[derive(Debug, Clone)]
pub struct EntityLinkingConfig {
    /// Minimum similarity threshold (0.0-1.0)
    pub min_similarity: f32,

    /// Use case normalization
    pub case_insensitive: bool,

    /// Remove punctuation before comparison
    pub remove_punctuation: bool,

    /// Enable phonetic matching (Soundex)
    pub use_phonetic: bool,

    /// Minimum token overlap for Jaccard (0.0-1.0)
    pub min_jaccard_overlap: f32,

    /// Maximum edit distance for Levenshtein
    pub max_edit_distance: usize,

    /// Enable fuzzy matching with typo tolerance
    pub fuzzy_matching: bool,
}

impl Default for EntityLinkingConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.85,
            case_insensitive: true,
            remove_punctuation: true,
            use_phonetic: false,
            min_jaccard_overlap: 0.6,
            max_edit_distance: 2,
            fuzzy_matching: true,
        }
    }
}

/// Entity linker using string similarity metrics
pub struct StringSimilarityLinker {
    config: EntityLinkingConfig,
}

impl StringSimilarityLinker {
    /// Create a new entity linker with configuration
    pub fn new(config: EntityLinkingConfig) -> Self {
        Self { config }
    }

    /// Link entities in a knowledge graph based on string similarity
    ///
    /// Returns a mapping from entity IDs to their canonical entity ID
    pub fn link_entities(
        &self,
        graph: &KnowledgeGraph,
    ) -> Result<HashMap<EntityId, EntityId>> {
        let mut links: HashMap<EntityId, EntityId> = HashMap::new();
        let entities: Vec<Entity> = graph.entities().cloned().collect();

        // Build clusters of similar entities
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut clustered: HashSet<usize> = HashSet::new();

        for i in 0..entities.len() {
            if clustered.contains(&i) {
                continue;
            }

            let mut cluster = vec![i];
            clustered.insert(i);

            for j in (i + 1)..entities.len() {
                if clustered.contains(&j) {
                    continue;
                }

                let similarity = self.compute_similarity(&entities[i], &entities[j]);

                if similarity >= self.config.min_similarity {
                    cluster.push(j);
                    clustered.insert(j);
                }
            }

            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }

        // For each cluster, select canonical entity (highest confidence)
        for cluster in clusters {
            let canonical_idx = cluster
                .iter()
                .max_by(|&&a, &&b| {
                    entities[a]
                        .confidence
                        .partial_cmp(&entities[b].confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            let canonical_id = &entities[*canonical_idx].id;

            for &entity_idx in &cluster {
                if entity_idx != *canonical_idx {
                    links.insert(entities[entity_idx].id.clone(), canonical_id.clone());
                }
            }
        }

        Ok(links)
    }

    /// Compute overall similarity between two entities
    fn compute_similarity(&self, e1: &Entity, e2: &Entity) -> f32 {
        // Different entity types should not be linked
        if e1.entity_type != e2.entity_type {
            return 0.0;
        }

        let name1 = self.normalize_string(&e1.name);
        let name2 = self.normalize_string(&e2.name);

        // Exact match after normalization
        if name1 == name2 {
            return 1.0;
        }

        let mut scores = Vec::new();

        // 1. Levenshtein-based similarity
        if self.config.fuzzy_matching {
            let lev_sim = self.levenshtein_similarity(&name1, &name2);
            scores.push(lev_sim);
        }

        // 2. Jaro-Winkler similarity
        let jaro_sim = self.jaro_winkler_similarity(&name1, &name2);
        scores.push(jaro_sim);

        // 3. Token-based Jaccard similarity
        let jaccard_sim = self.jaccard_similarity(&name1, &name2);
        scores.push(jaccard_sim);

        // 4. Phonetic matching (if enabled)
        if self.config.use_phonetic {
            let phonetic_sim = self.phonetic_similarity(&name1, &name2);
            scores.push(phonetic_sim);
        }

        // Return maximum similarity across all methods
        scores.into_iter().fold(0.0, f32::max)
    }

    /// Normalize string for comparison
    fn normalize_string(&self, s: &str) -> String {
        let mut normalized = s.to_string();

        if self.config.case_insensitive {
            normalized = normalized.to_lowercase();
        }

        if self.config.remove_punctuation {
            normalized = normalized
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        // Normalize whitespace
        normalized
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute Levenshtein edit distance-based similarity
    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f32 {
        let distance = self.levenshtein_distance(s1, s2);

        if distance > self.config.max_edit_distance {
            return 0.0;
        }

        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 1.0;
        }

        1.0 - (distance as f32 / max_len as f32)
    }

    /// Compute Levenshtein edit distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        #[allow(clippy::needless_range_loop)]
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        #[allow(clippy::needless_range_loop)]
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        // Fill matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };

                matrix[i][j] = (matrix[i - 1][j] + 1) // deletion
                    .min(matrix[i][j - 1] + 1) // insertion
                    .min(matrix[i - 1][j - 1] + cost); // substitution
            }
        }

        matrix[len1][len2]
    }

    /// Compute Jaro-Winkler similarity
    fn jaro_winkler_similarity(&self, s1: &str, s2: &str) -> f32 {
        let jaro = self.jaro_similarity(s1, s2);

        // Apply Winkler prefix bonus
        let prefix_len = s1
            .chars()
            .zip(s2.chars())
            .take(4)
            .take_while(|(c1, c2)| c1 == c2)
            .count();

        jaro + (prefix_len as f32 * 0.1 * (1.0 - jaro))
    }

    /// Compute Jaro similarity
    fn jaro_similarity(&self, s1: &str, s2: &str) -> f32 {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let len1 = s1_chars.len();
        let len2 = s2_chars.len();

        if len1 == 0 && len2 == 0 {
            return 1.0;
        }
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        let match_distance = (len1.max(len2) / 2).saturating_sub(1);

        let mut s1_matches = vec![false; len1];
        let mut s2_matches = vec![false; len2];

        let mut matches = 0;
        let mut transpositions = 0;

        // Find matches
        for i in 0..len1 {
            let start = i.saturating_sub(match_distance);
            let end = (i + match_distance + 1).min(len2);

            for j in start..end {
                if s2_matches[j] || s1_chars[i] != s2_chars[j] {
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
        let mut k = 0;
        for i in 0..len1 {
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

        let m = matches as f32;
        (m / len1 as f32 + m / len2 as f32 + (m - transpositions as f32 / 2.0) / m) / 3.0
    }

    /// Compute token-based Jaccard similarity
    fn jaccard_similarity(&self, s1: &str, s2: &str) -> f32 {
        let tokens1: HashSet<&str> = s1.split_whitespace().collect();
        let tokens2: HashSet<&str> = s2.split_whitespace().collect();

        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }

        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    /// Compute phonetic similarity using simplified Soundex
    fn phonetic_similarity(&self, s1: &str, s2: &str) -> f32 {
        let soundex1 = self.soundex(s1);
        let soundex2 = self.soundex(s2);

        if soundex1 == soundex2 {
            0.9 // High but not perfect score for phonetic match
        } else {
            0.0
        }
    }

    /// Simple Soundex implementation
    fn soundex(&self, s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }

        let chars: Vec<char> = s.to_uppercase().chars().collect();
        let mut result = String::new();

        // Keep first letter
        if let Some(&first) = chars.first() {
            if first.is_alphabetic() {
                result.push(first);
            }
        }

        let mut prev_code = self.soundex_code(chars[0]);

        for &c in chars.iter().skip(1) {
            let code = self.soundex_code(c);

            if code != '0' && code != prev_code {
                result.push(code);
                prev_code = code;
            }

            if result.len() >= 4 {
                break;
            }
        }

        // Pad with zeros
        while result.len() < 4 {
            result.push('0');
        }

        result
    }

    /// Get Soundex code for a character
    fn soundex_code(&self, c: char) -> char {
        match c.to_ascii_uppercase() {
            'B' | 'F' | 'P' | 'V' => '1',
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
            'D' | 'T' => '3',
            'L' => '4',
            'M' | 'N' => '5',
            'R' => '6',
            _ => '0',
        }
    }

    /// Find candidate entity for linking a new mention
    pub fn find_canonical_entity(
        &self,
        mention: &str,
        entity_type: &str,
        candidates: &[Entity],
    ) -> Option<EntityId> {
        let normalized_mention = self.normalize_string(mention);

        let mut best_match: Option<(EntityId, f32)> = None;

        for candidate in candidates {
            if candidate.entity_type != entity_type {
                continue;
            }

            let normalized_candidate = self.normalize_string(&candidate.name);

            // Quick exact match check
            if normalized_mention == normalized_candidate {
                return Some(candidate.id.clone());
            }

            // Compute similarity
            let mut scores = Vec::new();

            if self.config.fuzzy_matching {
                let lev_sim = self.levenshtein_similarity(&normalized_mention, &normalized_candidate);
                scores.push(lev_sim);
            }

            let jaro_sim = self.jaro_winkler_similarity(&normalized_mention, &normalized_candidate);
            scores.push(jaro_sim);

            let jaccard_sim = self.jaccard_similarity(&normalized_mention, &normalized_candidate);
            scores.push(jaccard_sim);

            if self.config.use_phonetic {
                let phonetic_sim =
                    self.phonetic_similarity(&normalized_mention, &normalized_candidate);
                scores.push(phonetic_sim);
            }

            let max_similarity = scores.into_iter().fold(0.0, f32::max);

            if max_similarity >= self.config.min_similarity {
                if let Some((_, current_best_score)) = &best_match {
                    if max_similarity > *current_best_score {
                        best_match = Some((candidate.id.clone(), max_similarity));
                    }
                } else {
                    best_match = Some((candidate.id.clone(), max_similarity));
                }
            }
        }

        best_match.map(|(id, _)| id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, EntityMention};

    #[test]
    fn test_levenshtein_distance() {
        let linker = StringSimilarityLinker::new(EntityLinkingConfig::default());

        assert_eq!(linker.levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(linker.levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(linker.levenshtein_distance("", ""), 0);
        assert_eq!(linker.levenshtein_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_jaro_winkler_similarity() {
        let linker = StringSimilarityLinker::new(EntityLinkingConfig::default());

        let sim = linker.jaro_winkler_similarity("martha", "marhta");
        assert!(sim > 0.9, "Expected high similarity for transposition");

        let sim2 = linker.jaro_winkler_similarity("dwayne", "duane");
        assert!(sim2 > 0.8, "Expected decent similarity");

        let sim3 = linker.jaro_winkler_similarity("abc", "xyz");
        assert!(sim3 < 0.3, "Expected low similarity");
    }

    #[test]
    fn test_jaccard_similarity() {
        let linker = StringSimilarityLinker::new(EntityLinkingConfig::default());

        let sim = linker.jaccard_similarity("the quick brown fox", "the lazy brown dog");
        assert!(sim > 0.3 && sim < 0.5, "Expected moderate similarity");

        let sim2 = linker.jaccard_similarity("apple orange banana", "apple orange banana");
        assert!((sim2 - 1.0).abs() < 0.001, "Expected perfect match");
    }

    #[test]
    fn test_soundex() {
        let linker = StringSimilarityLinker::new(EntityLinkingConfig::default());

        assert_eq!(linker.soundex("Robert"), "R163");
        assert_eq!(linker.soundex("Rupert"), "R163");
        assert_eq!(linker.soundex("Rubin"), "R150");
        assert_eq!(linker.soundex("Smith"), "S530");
        assert_eq!(linker.soundex("Smyth"), "S530");
    }

    #[test]
    fn test_entity_normalization() {
        let linker = StringSimilarityLinker::new(EntityLinkingConfig::default());

        assert_eq!(
            linker.normalize_string("John  Smith!"),
            "john smith"
        );
        assert_eq!(
            linker.normalize_string("ACME Corp."),
            "acme corp"
        );
    }

    #[test]
    fn test_find_canonical_entity() {
        let config = EntityLinkingConfig {
            min_similarity: 0.8,
            ..Default::default()
        };
        let linker = StringSimilarityLinker::new(config);

        let candidates = vec![
            Entity {
                id: EntityId::new("e1".to_string()),
                name: "John Smith".to_string(),
                entity_type: "PERSON".to_string(),
                confidence: 0.9,
                mentions: vec![],
                embedding: None,
            },
            Entity {
                id: EntityId::new("e2".to_string()),
                name: "Acme Corp".to_string(),
                entity_type: "ORG".to_string(),
                confidence: 0.85,
                mentions: vec![],
                embedding: None,
            },
        ];

        // Should match John Smith
        let result = linker.find_canonical_entity("Jon Smith", "PERSON", &candidates);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), EntityId::new("e1".to_string()));

        // Should not match wrong type
        let result = linker.find_canonical_entity("John Smith", "ORG", &candidates);
        assert!(result.is_none());

        // Should match with typo
        let result = linker.find_canonical_entity("Jhon Smith", "PERSON", &candidates);
        assert!(result.is_some());
    }

    #[test]
    fn test_link_similar_entities() {
        let config = EntityLinkingConfig {
            min_similarity: 0.85,
            ..Default::default()
        };
        let linker = StringSimilarityLinker::new(config);

        let mut graph = KnowledgeGraph::new();

        // Add similar entities
        let _ = graph.add_entity(Entity {
            id: EntityId::new("e1".to_string()),
            name: "New York".to_string(),
            entity_type: "LOCATION".to_string(),
            confidence: 0.9,
            mentions: vec![EntityMention {
                chunk_id: ChunkId::new("chunk1".to_string()),
                start_offset: 0,
                end_offset: 8,
                confidence: 0.9,
            }],
            embedding: None,
        });

        let _ = graph.add_entity(Entity {
            id: EntityId::new("e2".to_string()),
            name: "New York City".to_string(),
            entity_type: "LOCATION".to_string(),
            confidence: 0.85,
            mentions: vec![EntityMention {
                chunk_id: ChunkId::new("chunk2".to_string()),
                start_offset: 0,
                end_offset: 13,
                confidence: 0.85,
            }],
            embedding: None,
        });

        let links = linker.link_entities(&graph).unwrap();

        // Should link similar location names
        assert!(links.len() > 0, "Expected some entities to be linked");
    }
}
