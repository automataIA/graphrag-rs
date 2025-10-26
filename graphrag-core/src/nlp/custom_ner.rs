//! Custom NER Training Pipeline
//!
//! This module provides a framework for training custom Named Entity Recognition models:
//! - Pattern-based entity extraction
//! - Dictionary/gazetteer matching
//! - Rule-based extraction
//! - Active learning support
//! - Model fine-tuning preparation
//!
//! ## Use Cases
//!
//! - Domain-specific entities (medical terms, legal concepts, etc.)
//! - Company-specific terminology
//! - Custom product names
//! - Technical jargon extraction

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use regex::Regex;

/// Entity type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    /// Type name (e.g., "PROTEIN", "DRUG", "DISEASE")
    pub name: String,
    /// Type description
    pub description: String,
    /// Example entities of this type
    pub examples: Vec<String>,
    /// Patterns for recognition
    pub patterns: Vec<String>,
    /// Dictionary/gazetteer entries
    pub dictionary: HashSet<String>,
}

impl EntityType {
    /// Create new entity type
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            examples: Vec::new(),
            patterns: Vec::new(),
            dictionary: HashSet::new(),
        }
    }

    /// Add example entity
    pub fn add_example(&mut self, example: String) {
        self.examples.push(example.clone());
        self.dictionary.insert(example.to_lowercase());
    }

    /// Add pattern (regex)
    pub fn add_pattern(&mut self, pattern: String) {
        self.patterns.push(pattern);
    }

    /// Add dictionary entries (bulk)
    pub fn add_dictionary_entries(&mut self, entries: Vec<String>) {
        for entry in entries {
            self.dictionary.insert(entry.to_lowercase());
        }
    }
}

/// Extraction rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionRule {
    /// Rule name
    pub name: String,
    /// Entity type this rule extracts
    pub entity_type: String,
    /// Rule type
    pub rule_type: RuleType,
    /// Rule pattern or configuration
    pub pattern: String,
    /// Minimum confidence for matches
    pub min_confidence: f32,
    /// Priority (higher = checked first)
    pub priority: i32,
}

/// Rule types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RuleType {
    /// Exact string match
    ExactMatch,
    /// Regex pattern
    Regex,
    /// Prefix match
    Prefix,
    /// Suffix match
    Suffix,
    /// Contains substring
    Contains,
    /// Dictionary lookup
    Dictionary,
    /// Context-based (requires surrounding words)
    Contextual,
}

/// Custom NER model
pub struct CustomNER {
    /// Entity types
    entity_types: HashMap<String, EntityType>,
    /// Extraction rules
    rules: Vec<ExtractionRule>,
    /// Compiled regex patterns
    compiled_patterns: HashMap<String, Regex>,
}

impl CustomNER {
    /// Create new custom NER model
    pub fn new() -> Self {
        Self {
            entity_types: HashMap::new(),
            rules: Vec::new(),
            compiled_patterns: HashMap::new(),
        }
    }

    /// Register entity type
    pub fn register_entity_type(&mut self, entity_type: EntityType) {
        self.entity_types.insert(entity_type.name.clone(), entity_type);
    }

    /// Add extraction rule
    pub fn add_rule(&mut self, rule: ExtractionRule) {
        // Compile regex if needed
        if rule.rule_type == RuleType::Regex {
            if let Ok(regex) = Regex::new(&rule.pattern) {
                self.compiled_patterns.insert(rule.name.clone(), regex);
            }
        }

        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Extract entities from text
    pub fn extract(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        // Apply rules in priority order
        for rule in &self.rules {
            let rule_entities = self.apply_rule(text, rule);
            entities.extend(rule_entities);
        }

        // Deduplicate and resolve conflicts
        self.resolve_overlaps(entities)
    }

    /// Apply a single extraction rule
    fn apply_rule(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        match rule.rule_type {
            RuleType::ExactMatch => self.extract_exact_match(text, rule),
            RuleType::Regex => self.extract_regex(text, rule),
            RuleType::Prefix => self.extract_prefix(text, rule),
            RuleType::Suffix => self.extract_suffix(text, rule),
            RuleType::Contains => self.extract_contains(text, rule),
            RuleType::Dictionary => self.extract_dictionary(text, rule),
            RuleType::Contextual => self.extract_contextual(text, rule),
        }
    }

    /// Exact match extraction
    fn extract_exact_match(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let pattern = &rule.pattern;
        let text_lower = text.to_lowercase();
        let pattern_lower = pattern.to_lowercase();

        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(&pattern_lower) {
            let absolute_pos = start + pos;
            entities.push(ExtractedEntity {
                text: text[absolute_pos..absolute_pos + pattern.len()].to_string(),
                entity_type: rule.entity_type.clone(),
                start: absolute_pos,
                end: absolute_pos + pattern.len(),
                confidence: 1.0,
                rule_name: rule.name.clone(),
            });

            start = absolute_pos + pattern.len();
        }

        entities
    }

    /// Regex extraction
    fn extract_regex(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        if let Some(regex) = self.compiled_patterns.get(&rule.name) {
            for capture in regex.captures_iter(text) {
                if let Some(matched) = capture.get(0) {
                    entities.push(ExtractedEntity {
                        text: matched.as_str().to_string(),
                        entity_type: rule.entity_type.clone(),
                        start: matched.start(),
                        end: matched.end(),
                        confidence: 0.9,
                        rule_name: rule.name.clone(),
                    });
                }
            }
        }

        entities
    }

    /// Prefix match extraction
    fn extract_prefix(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut pos = 0;

        for word in words {
            if word.to_lowercase().starts_with(&rule.pattern.to_lowercase()) {
                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type: rule.entity_type.clone(),
                    start: pos,
                    end: pos + word.len(),
                    confidence: 0.7,
                    rule_name: rule.name.clone(),
                });
            }
            pos += word.len() + 1; // +1 for space
        }

        entities
    }

    /// Suffix match extraction
    fn extract_suffix(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut pos = 0;

        for word in words {
            if word.to_lowercase().ends_with(&rule.pattern.to_lowercase()) {
                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type: rule.entity_type.clone(),
                    start: pos,
                    end: pos + word.len(),
                    confidence: 0.7,
                    rule_name: rule.name.clone(),
                });
            }
            pos += word.len() + 1;
        }

        entities
    }

    /// Contains substring extraction
    fn extract_contains(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut pos = 0;

        for word in words {
            if word.to_lowercase().contains(&rule.pattern.to_lowercase()) {
                entities.push(ExtractedEntity {
                    text: word.to_string(),
                    entity_type: rule.entity_type.clone(),
                    start: pos,
                    end: pos + word.len(),
                    confidence: 0.6,
                    rule_name: rule.name.clone(),
                });
            }
            pos += word.len() + 1;
        }

        entities
    }

    /// Dictionary-based extraction
    fn extract_dictionary(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        if let Some(entity_type) = self.entity_types.get(&rule.entity_type) {
            let text_lower = text.to_lowercase();

            for entry in &entity_type.dictionary {
                let mut start = 0;
                while let Some(pos) = text_lower[start..].find(entry) {
                    let absolute_pos = start + pos;
                    entities.push(ExtractedEntity {
                        text: text[absolute_pos..absolute_pos + entry.len()].to_string(),
                        entity_type: rule.entity_type.clone(),
                        start: absolute_pos,
                        end: absolute_pos + entry.len(),
                        confidence: 0.95,
                        rule_name: rule.name.clone(),
                    });

                    start = absolute_pos + entry.len();
                }
            }
        }

        entities
    }

    /// Contextual extraction (requires specific surrounding words)
    fn extract_contextual(&self, text: &str, rule: &ExtractionRule) -> Vec<ExtractedEntity> {
        // Simplified contextual extraction
        // Pattern format: "before_word|target|after_word"
        let parts: Vec<&str> = rule.pattern.split('|').collect();
        if parts.len() != 3 {
            return Vec::new();
        }

        let before = parts[0];
        let target = parts[1];
        let after = parts[2];

        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for window in words.windows(3) {
            if window[0].to_lowercase().contains(&before.to_lowercase())
                && window[1].to_lowercase().contains(&target.to_lowercase())
                && window[2].to_lowercase().contains(&after.to_lowercase())
            {
                // Find position in original text
                if let Some(pos) = text.find(window[1]) {
                    entities.push(ExtractedEntity {
                        text: window[1].to_string(),
                        entity_type: rule.entity_type.clone(),
                        start: pos,
                        end: pos + window[1].len(),
                        confidence: 0.85,
                        rule_name: rule.name.clone(),
                    });
                }
            }
        }

        entities
    }

    /// Resolve overlapping entities (keep higher confidence)
    fn resolve_overlaps(&self, mut entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        if entities.is_empty() {
            return entities;
        }

        // Sort by position, then by confidence (descending)
        entities.sort_by(|a, b| {
            a.start.cmp(&b.start)
                .then(b.confidence.partial_cmp(&a.confidence).unwrap())
        });

        let mut result = Vec::new();
        let mut last_end = 0;

        for entity in entities {
            // Skip if overlaps with previous entity
            if entity.start < last_end {
                continue;
            }

            last_end = entity.end;
            result.push(entity);
        }

        result
    }

    /// Get entity types
    pub fn entity_types(&self) -> &HashMap<String, EntityType> {
        &self.entity_types
    }

    /// Get rules
    pub fn rules(&self) -> &[ExtractionRule] {
        &self.rules
    }
}

impl Default for CustomNER {
    fn default() -> Self {
        Self::new()
    }
}

/// Extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: String,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Rule that extracted this entity
    pub rule_name: String,
}

/// Training dataset for custom NER
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Annotated examples
    pub examples: Vec<AnnotatedExample>,
}

impl TrainingDataset {
    /// Create new training dataset
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Add annotated example
    pub fn add_example(&mut self, example: AnnotatedExample) {
        self.examples.push(example);
    }

    /// Get statistics
    pub fn statistics(&self) -> DatasetStatistics {
        let total_examples = self.examples.len();
        let mut entity_counts: HashMap<String, usize> = HashMap::new();

        for example in &self.examples {
            for entity in &example.entities {
                *entity_counts.entry(entity.entity_type.clone()).or_insert(0) += 1;
            }
        }

        DatasetStatistics {
            total_examples,
            entity_counts,
        }
    }
}

impl Default for TrainingDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Annotated text example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedExample {
    /// Original text
    pub text: String,
    /// Annotated entities
    pub entities: Vec<ExtractedEntity>,
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Total examples
    pub total_examples: usize,
    /// Entity type counts
    pub entity_counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_creation() {
        let mut entity_type = EntityType::new(
            "PROTEIN".to_string(),
            "Protein names".to_string(),
        );

        entity_type.add_example("hemoglobin".to_string());
        entity_type.add_example("insulin".to_string());

        assert_eq!(entity_type.examples.len(), 2);
        assert_eq!(entity_type.dictionary.len(), 2);
    }

    #[test]
    fn test_exact_match_extraction() {
        let mut ner = CustomNER::new();

        let rule = ExtractionRule {
            name: "protein_exact".to_string(),
            entity_type: "PROTEIN".to_string(),
            rule_type: RuleType::ExactMatch,
            pattern: "hemoglobin".to_string(),
            min_confidence: 0.9,
            priority: 1,
        };

        ner.add_rule(rule);

        let text = "The protein hemoglobin is important. Hemoglobin carries oxygen.";
        let entities = ner.extract(text);

        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].entity_type, "PROTEIN");
        assert_eq!(entities[0].text.to_lowercase(), "hemoglobin");
    }

    #[test]
    fn test_regex_extraction() {
        let mut ner = CustomNER::new();

        let rule = ExtractionRule {
            name: "gene_pattern".to_string(),
            entity_type: "GENE".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"[A-Z]{2,4}\d+".to_string(),
            min_confidence: 0.8,
            priority: 1,
        };

        ner.add_rule(rule);

        let text = "The genes TP53 and BRCA1 are tumor suppressors.";
        let entities = ner.extract(text);

        assert!(entities.len() >= 2);
        assert!(entities.iter().any(|e| e.text == "TP53"));
        assert!(entities.iter().any(|e| e.text == "BRCA1"));
    }

    #[test]
    fn test_dictionary_extraction() {
        let mut ner = CustomNER::new();

        let mut protein_type = EntityType::new(
            "PROTEIN".to_string(),
            "Protein names".to_string(),
        );
        protein_type.add_dictionary_entries(vec![
            "insulin".to_string(),
            "hemoglobin".to_string(),
            "collagen".to_string(),
        ]);

        ner.register_entity_type(protein_type);

        let rule = ExtractionRule {
            name: "protein_dict".to_string(),
            entity_type: "PROTEIN".to_string(),
            rule_type: RuleType::Dictionary,
            pattern: "".to_string(),
            min_confidence: 0.9,
            priority: 2,
        };

        ner.add_rule(rule);

        let text = "Insulin regulates blood sugar. Hemoglobin transports oxygen.";
        let entities = ner.extract(text);

        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_prefix_extraction() {
        let mut ner = CustomNER::new();

        let rule = ExtractionRule {
            name: "bio_prefix".to_string(),
            entity_type: "BIO_TERM".to_string(),
            rule_type: RuleType::Prefix,
            pattern: "bio".to_string(),
            min_confidence: 0.7,
            priority: 1,
        };

        ner.add_rule(rule);

        let text = "Biology and biochemistry are fascinating subjects.";
        let entities = ner.extract(text);

        assert!(entities.len() >= 2);
    }

    #[test]
    fn test_overlap_resolution() {
        let mut ner = CustomNER::new();

        let rule1 = ExtractionRule {
            name: "rule1".to_string(),
            entity_type: "TYPE1".to_string(),
            rule_type: RuleType::ExactMatch,
            pattern: "test".to_string(),
            min_confidence: 0.9,
            priority: 1,
        };

        let rule2 = ExtractionRule {
            name: "rule2".to_string(),
            entity_type: "TYPE2".to_string(),
            rule_type: RuleType::ExactMatch,
            pattern: "testing".to_string(),
            min_confidence: 0.95,
            priority: 2,
        };

        ner.add_rule(rule1);
        ner.add_rule(rule2);

        let text = "We are testing this code.";
        let entities = ner.extract(text);

        // Should only extract one entity (higher confidence/priority wins)
        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn test_training_dataset() {
        let mut dataset = TrainingDataset::new();

        let example = AnnotatedExample {
            text: "Insulin regulates glucose.".to_string(),
            entities: vec![
                ExtractedEntity {
                    text: "Insulin".to_string(),
                    entity_type: "PROTEIN".to_string(),
                    start: 0,
                    end: 7,
                    confidence: 1.0,
                    rule_name: "manual".to_string(),
                },
            ],
        };

        dataset.add_example(example);

        let stats = dataset.statistics();
        assert_eq!(stats.total_examples, 1);
        assert_eq!(stats.entity_counts.get("PROTEIN"), Some(&1));
    }
}
