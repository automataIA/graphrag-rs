use crate::{core::KnowledgeGraph, vector::EmbeddingGenerator};
use std::collections::{HashMap, HashSet};

/// Configuration for query expansion
#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    /// Maximum number of expanded queries to generate
    pub max_expansions: usize,
    /// Enable synonym-based expansion
    pub enable_synonyms: bool,
    /// Enable entity-based expansion
    pub enable_entities: bool,
    /// Enable context-based expansion
    pub enable_context: bool,
    /// Similarity threshold for semantic expansion
    pub similarity_threshold: f32,
    /// Weight for original query in expanded results
    pub original_weight: f32,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            enable_synonyms: true,
            enable_entities: true,
            enable_context: true,
            similarity_threshold: 0.7,
            original_weight: 1.0,
        }
    }
}

/// Strategy for query expansion
#[derive(Debug, Clone, PartialEq)]
pub enum ExpansionStrategy {
    /// Synonym-based expansion using predefined dictionaries
    Synonyms,
    /// Entity-based expansion using knowledge graph entities
    Entities,
    /// Context-based expansion using semantic similarity
    Context,
    /// Phrase-based expansion using common phrases
    Phrases,
}

/// An expanded query with metadata
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// The expanded query text
    pub text: String,
    /// Weight/importance of this expanded query
    pub weight: f32,
    /// Strategy used to generate this expansion
    pub strategy: ExpansionStrategy,
    /// Original terms that were expanded
    pub expanded_terms: Vec<String>,
}

/// Query expansion system
pub struct QueryExpander {
    /// Synonym dictionary for word expansion
    synonyms: HashMap<String, Vec<String>>,
    /// Embedding generator for semantic expansion
    #[allow(dead_code)]
    embedding_generator: EmbeddingGenerator,
    /// Configuration for expansion
    config: ExpansionConfig,
    /// Common phrases dictionary
    phrases: HashMap<String, Vec<String>>,
}

impl QueryExpander {
    /// Create a new query expander with default configuration
    pub fn new() -> Self {
        Self::with_config(ExpansionConfig::default())
    }

    /// Create a new query expander with custom configuration
    pub fn with_config(config: ExpansionConfig) -> Self {
        Self {
            synonyms: Self::build_synonym_dictionary(),
            embedding_generator: EmbeddingGenerator::new(128),
            config,
            phrases: Self::build_phrase_dictionary(),
        }
    }

    /// Expand a query using configured strategies
    pub fn expand_query(&self, original_query: &str) -> Vec<ExpandedQuery> {
        let mut expanded_queries = Vec::new();

        // Always include the original query
        expanded_queries.push(ExpandedQuery {
            text: original_query.to_string(),
            weight: self.config.original_weight,
            strategy: ExpansionStrategy::Context, // Original query is treated as context
            expanded_terms: Vec::new(),
        });

        // Apply different expansion strategies
        if self.config.enable_synonyms {
            expanded_queries.extend(self.expand_with_synonyms(original_query));
        }

        if self.config.enable_context {
            expanded_queries.extend(self.expand_with_context(original_query));
        }

        // Phrase-based expansion
        expanded_queries.extend(self.expand_with_phrases(original_query));

        // Limit to max expansions
        expanded_queries.truncate(self.config.max_expansions);

        expanded_queries
    }

    /// Expand query using knowledge graph entities
    pub fn expand_with_entities(&self, query: &str, graph: &KnowledgeGraph) -> Vec<ExpandedQuery> {
        if !self.config.enable_entities {
            return Vec::new();
        }

        let mut expansions = Vec::new();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        // Find entities mentioned in the query
        let mut mentioned_entities = Vec::new();
        for entity in graph.entities() {
            let entity_name_lower = entity.name.to_lowercase();

            // Check if entity name appears in query
            if query_words
                .iter()
                .any(|&word| entity_name_lower.contains(word) || word.contains(&entity_name_lower))
            {
                mentioned_entities.push(entity);
            }
        }

        // Generate expansions based on found entities
        for entity in mentioned_entities.iter().take(3) {
            // Expansion 1: Add entity type
            let type_expansion = format!("{} {}", query, entity.entity_type);
            expansions.push(ExpandedQuery {
                text: type_expansion,
                weight: 0.8,
                strategy: ExpansionStrategy::Entities,
                expanded_terms: vec![entity.entity_type.clone()],
            });

            // Expansion 2: Replace entity with related entities
            let neighbors = graph.get_neighbors(&entity.id);
            for (neighbor, _) in neighbors.iter().take(2) {
                let related_expansion = query.replace(&entity.name, &neighbor.name);
                if related_expansion != query {
                    expansions.push(ExpandedQuery {
                        text: related_expansion,
                        weight: 0.6,
                        strategy: ExpansionStrategy::Entities,
                        expanded_terms: vec![neighbor.name.clone()],
                    });
                }
            }
        }

        // Limit entity expansions
        expansions.truncate(3);
        expansions
    }

    /// Expand query using synonyms
    fn expand_with_synonyms(&self, query: &str) -> Vec<ExpandedQuery> {
        let mut expansions = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        // Find words that have synonyms
        for word in &words {
            let word_lower = word.to_lowercase();
            if let Some(synonyms) = self.synonyms.get(&word_lower) {
                for synonym in synonyms.iter().take(2) {
                    let expanded = query.replace(word, synonym);
                    if expanded != query {
                        expansions.push(ExpandedQuery {
                            text: expanded,
                            weight: 0.7,
                            strategy: ExpansionStrategy::Synonyms,
                            expanded_terms: vec![synonym.clone()],
                        });
                    }
                }
            }
        }

        // Limit synonym expansions
        expansions.truncate(2);
        expansions
    }

    /// Expand query using contextual/semantic expansion
    fn expand_with_context(&self, query: &str) -> Vec<ExpandedQuery> {
        let mut expansions = Vec::new();

        // Generate semantic variations
        let variations = self.generate_semantic_variations(query);
        for (variation, weight) in variations {
            expansions.push(ExpandedQuery {
                text: variation.clone(),
                weight,
                strategy: ExpansionStrategy::Context,
                expanded_terms: Self::extract_new_terms(query, &variation),
            });
        }

        expansions
    }

    /// Expand query using common phrases
    fn expand_with_phrases(&self, query: &str) -> Vec<ExpandedQuery> {
        let mut expansions = Vec::new();
        let query_lower = query.to_lowercase();

        // Look for key terms that can be expanded with phrases
        for (key_term, phrases) in &self.phrases {
            if query_lower.contains(key_term) {
                for phrase in phrases.iter().take(1) {
                    let expanded = format!("{query} {phrase}");
                    expansions.push(ExpandedQuery {
                        text: expanded,
                        weight: 0.6,
                        strategy: ExpansionStrategy::Phrases,
                        expanded_terms: vec![phrase.clone()],
                    });
                }
            }
        }

        expansions.truncate(2);
        expansions
    }

    /// Generate semantic variations of the query
    fn generate_semantic_variations(&self, query: &str) -> Vec<(String, f32)> {
        let mut variations = Vec::new();

        // Simple pattern-based variations
        let words: Vec<&str> = query.split_whitespace().collect();

        // Add question variations
        if !query.contains('?') && words.len() > 1 {
            variations.push((format!("What is {query}?"), 0.8));
            variations.push((format!("How does {query} work?"), 0.7));
        }

        // Add descriptive variations
        if words.len() <= 3 {
            variations.push((format!("{query} definition"), 0.7));
            variations.push((format!("{query} explanation"), 0.6));
        }

        // Rephrase using common patterns
        if words.len() >= 2 {
            let last_word = words[words.len() - 1];
            let rest = words[..words.len() - 1].join(" ");
            variations.push((format!("{last_word} related to {rest}"), 0.6));
        }

        variations.truncate(2);
        variations
    }

    /// Extract new terms added in the expanded query
    fn extract_new_terms(original: &str, expanded: &str) -> Vec<String> {
        let original_words: HashSet<&str> = original.split_whitespace().collect();
        let expanded_words: HashSet<&str> = expanded.split_whitespace().collect();

        expanded_words
            .difference(&original_words)
            .map(|&word| word.to_string())
            .collect()
    }

    /// Build a basic synonym dictionary
    fn build_synonym_dictionary() -> HashMap<String, Vec<String>> {
        let mut synonyms = HashMap::new();

        // Basic synonyms for common words
        synonyms.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "massive".to_string(),
            ],
        );
        synonyms.insert(
            "small".to_string(),
            vec![
                "tiny".to_string(),
                "little".to_string(),
                "miniature".to_string(),
            ],
        );
        synonyms.insert(
            "fast".to_string(),
            vec![
                "quick".to_string(),
                "rapid".to_string(),
                "swift".to_string(),
            ],
        );
        synonyms.insert(
            "slow".to_string(),
            vec![
                "sluggish".to_string(),
                "gradual".to_string(),
                "leisurely".to_string(),
            ],
        );
        synonyms.insert(
            "good".to_string(),
            vec![
                "excellent".to_string(),
                "great".to_string(),
                "wonderful".to_string(),
            ],
        );
        synonyms.insert(
            "bad".to_string(),
            vec![
                "poor".to_string(),
                "terrible".to_string(),
                "awful".to_string(),
            ],
        );
        synonyms.insert(
            "happy".to_string(),
            vec![
                "joyful".to_string(),
                "pleased".to_string(),
                "cheerful".to_string(),
            ],
        );
        synonyms.insert(
            "sad".to_string(),
            vec![
                "unhappy".to_string(),
                "sorrowful".to_string(),
                "melancholy".to_string(),
            ],
        );
        synonyms.insert(
            "important".to_string(),
            vec![
                "significant".to_string(),
                "crucial".to_string(),
                "vital".to_string(),
            ],
        );
        synonyms.insert(
            "difficult".to_string(),
            vec![
                "challenging".to_string(),
                "hard".to_string(),
                "tough".to_string(),
            ],
        );
        synonyms.insert(
            "easy".to_string(),
            vec![
                "simple".to_string(),
                "effortless".to_string(),
                "straightforward".to_string(),
            ],
        );
        synonyms.insert(
            "beautiful".to_string(),
            vec![
                "gorgeous".to_string(),
                "stunning".to_string(),
                "attractive".to_string(),
            ],
        );
        synonyms.insert(
            "ugly".to_string(),
            vec![
                "hideous".to_string(),
                "unattractive".to_string(),
                "unsightly".to_string(),
            ],
        );
        synonyms.insert(
            "smart".to_string(),
            vec![
                "intelligent".to_string(),
                "clever".to_string(),
                "brilliant".to_string(),
            ],
        );
        synonyms.insert(
            "stupid".to_string(),
            vec![
                "foolish".to_string(),
                "dumb".to_string(),
                "ignorant".to_string(),
            ],
        );

        // Domain-specific synonyms
        synonyms.insert(
            "character".to_string(),
            vec![
                "person".to_string(),
                "individual".to_string(),
                "figure".to_string(),
            ],
        );
        synonyms.insert(
            "story".to_string(),
            vec![
                "narrative".to_string(),
                "tale".to_string(),
                "plot".to_string(),
            ],
        );
        synonyms.insert(
            "book".to_string(),
            vec!["novel".to_string(), "text".to_string(), "work".to_string()],
        );
        synonyms.insert(
            "author".to_string(),
            vec![
                "writer".to_string(),
                "novelist".to_string(),
                "creator".to_string(),
            ],
        );
        synonyms.insert(
            "relationship".to_string(),
            vec![
                "connection".to_string(),
                "bond".to_string(),
                "association".to_string(),
            ],
        );
        synonyms.insert(
            "conflict".to_string(),
            vec![
                "struggle".to_string(),
                "tension".to_string(),
                "dispute".to_string(),
            ],
        );
        synonyms.insert(
            "theme".to_string(),
            vec![
                "topic".to_string(),
                "subject".to_string(),
                "motif".to_string(),
            ],
        );
        synonyms.insert(
            "setting".to_string(),
            vec![
                "location".to_string(),
                "place".to_string(),
                "environment".to_string(),
            ],
        );

        synonyms
    }

    /// Build a basic phrase dictionary for common expansions
    fn build_phrase_dictionary() -> HashMap<String, Vec<String>> {
        let mut phrases = HashMap::new();

        // Character-related phrases
        phrases.insert(
            "character".to_string(),
            vec![
                "personality traits".to_string(),
                "character development".to_string(),
                "motivations".to_string(),
            ],
        );

        // Plot-related phrases
        phrases.insert(
            "plot".to_string(),
            vec![
                "story structure".to_string(),
                "narrative arc".to_string(),
                "plot points".to_string(),
            ],
        );

        // Relationship phrases
        phrases.insert(
            "relationship".to_string(),
            vec![
                "dynamics".to_string(),
                "interactions".to_string(),
                "connections".to_string(),
            ],
        );

        // Analysis phrases
        phrases.insert(
            "analysis".to_string(),
            vec![
                "detailed examination".to_string(),
                "in-depth study".to_string(),
                "critical review".to_string(),
            ],
        );

        // Theme phrases
        phrases.insert(
            "theme".to_string(),
            vec![
                "central message".to_string(),
                "underlying meaning".to_string(),
                "thematic elements".to_string(),
            ],
        );

        phrases
    }

    /// Update synonym dictionary
    pub fn add_synonyms(&mut self, word: String, synonyms: Vec<String>) {
        self.synonyms.insert(word, synonyms);
    }

    /// Get current configuration
    pub fn get_config(&self) -> &ExpansionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ExpansionConfig) {
        self.config = config;
    }

    /// Get expansion statistics
    pub fn get_statistics(&self) -> ExpansionStatistics {
        ExpansionStatistics {
            synonym_count: self.synonyms.len(),
            phrase_count: self.phrases.len(),
            config: self.config.clone(),
        }
    }
}

impl Default for QueryExpander {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about query expansion
#[derive(Debug, Clone)]
pub struct ExpansionStatistics {
    pub synonym_count: usize,
    pub phrase_count: usize,
    pub config: ExpansionConfig,
}

impl ExpansionStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Query Expansion Statistics:");
        println!("  Synonym dictionary: {} entries", self.synonym_count);
        println!("  Phrase dictionary: {} entries", self.phrase_count);
        println!("  Max expansions: {}", self.config.max_expansions);
        println!("  Enabled strategies:");
        if self.config.enable_synonyms {
            println!("    - Synonyms");
        }
        if self.config.enable_entities {
            println!("    - Entities");
        }
        if self.config.enable_context {
            println!("    - Context");
        }
        println!(
            "  Similarity threshold: {:.2}",
            self.config.similarity_threshold
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_expander_creation() {
        let expander = QueryExpander::new();
        let stats = expander.get_statistics();
        assert!(stats.synonym_count > 0);
        assert!(stats.phrase_count > 0);
    }

    #[test]
    fn test_basic_query_expansion() {
        let expander = QueryExpander::new();
        let expansions = expander.expand_query("big house");

        assert!(!expansions.is_empty());
        assert_eq!(expansions[0].text, "big house"); // Original query should be first
        assert!(expansions.len() <= expander.config.max_expansions);
    }

    #[test]
    fn test_synonym_expansion() {
        let expander = QueryExpander::new();
        let expansions = expander.expand_with_synonyms("big car");

        // Should find synonyms for "big"
        assert!(expansions
            .iter()
            .any(|e| e.text.contains("large") || e.text.contains("huge")));
    }

    #[test]
    fn test_context_expansion() {
        let expander = QueryExpander::new();
        let expansions = expander.expand_with_context("character development");

        assert!(!expansions.is_empty());
        assert!(expansions
            .iter()
            .all(|e| e.strategy == ExpansionStrategy::Context));
    }

    #[test]
    fn test_phrase_expansion() {
        let expander = QueryExpander::new();
        let expansions = expander.expand_with_phrases("character analysis");

        // Should add phrases related to "character"
        assert!(expansions
            .iter()
            .any(|e| e.strategy == ExpansionStrategy::Phrases));
    }

    #[test]
    fn test_expanded_query_structure() {
        let expanded = ExpandedQuery {
            text: "test query".to_string(),
            weight: 0.8,
            strategy: ExpansionStrategy::Synonyms,
            expanded_terms: vec!["test".to_string()],
        };

        assert_eq!(expanded.text, "test query");
        assert_eq!(expanded.weight, 0.8);
        assert_eq!(expanded.strategy, ExpansionStrategy::Synonyms);
        assert_eq!(expanded.expanded_terms.len(), 1);
    }
}
