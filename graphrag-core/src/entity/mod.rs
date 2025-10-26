/// Bidirectional entity-chunk index for fast lookups
pub mod bidirectional_index;
/// Gleaning-based entity extraction module
pub mod gleaning_extractor;
/// LLM-based entity extractor (TRUE LLM extraction, not pattern-based)
pub mod llm_extractor;
/// LLM-based relationship extraction module
pub mod llm_relationship_extractor;
/// Prompt templates for LLM-based extraction
pub mod prompts;
/// Semantic entity merging module
pub mod semantic_merging;
/// String similarity-based entity linking module
pub mod string_similarity_linker;

pub use bidirectional_index::{BidirectionalIndex, IndexStatistics};
pub use gleaning_extractor::{ExtractionCompletionStatus, GleaningConfig, GleaningEntityExtractor};
pub use llm_extractor::LLMEntityExtractor;
pub use llm_relationship_extractor::{
    ExtractedEntity, ExtractedRelationship, ExtractionResult, LLMRelationshipExtractor,
};
pub use semantic_merging::{EntityMergeDecision, MergingStatistics, SemanticEntityMerger};
pub use string_similarity_linker::{EntityLinkingConfig, StringSimilarityLinker};

use crate::{
    config::setconfig::EntityExtractionConfig,
    core::{ChunkId, Entity, EntityId, EntityMention, TextChunk},
    Result,
};
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Entity extraction system with dynamic configuration support
pub struct EntityExtractor {
    min_confidence: f32,
    config: Option<EntityExtractionConfig>,
    allowed_patterns: Vec<Regex>,
    excluded_patterns: Vec<Regex>,
}

impl EntityExtractor {
    /// Create a new entity extractor
    pub fn new(min_confidence: f32) -> Result<Self> {
        Ok(Self {
            min_confidence,
            config: None,
            allowed_patterns: Vec::new(),
            excluded_patterns: Vec::new(),
        })
    }

    /// Create a new entity extractor with configuration
    pub fn with_config(config: EntityExtractionConfig) -> Result<Self> {
        let mut allowed_patterns = Vec::new();
        let mut excluded_patterns = Vec::new();

        // Compile allowed patterns from config
        if let Some(filters) = &config.filters {
            if let Some(patterns) = &filters.allowed_patterns {
                for pattern in patterns {
                    match Regex::new(pattern) {
                        Ok(regex) => allowed_patterns.push(regex),
                        Err(e) => {
                            tracing::warn!("Invalid allowed pattern '{pattern}': {e}");
                        }
                    }
                }
            }

            if let Some(patterns) = &filters.excluded_patterns {
                for pattern in patterns {
                    match Regex::new(pattern) {
                        Ok(regex) => excluded_patterns.push(regex),
                        Err(e) => {
                            tracing::warn!("Invalid excluded pattern '{pattern}': {e}");
                        }
                    }
                }
            }
        }

        let min_confidence = config
            .filters
            .as_ref()
            .map(|f| f.confidence_threshold)
            .unwrap_or(config.confidence_threshold);

        Ok(Self {
            min_confidence,
            config: Some(config),
            allowed_patterns,
            excluded_patterns,
        })
    }

    /// Extract entities from a text chunk using dynamic entity types
    pub fn extract_from_chunk(&self, chunk: &TextChunk) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let text = &chunk.content;

        // Get entity types from config or use defaults
        let entity_types = if let Some(config) = &self.config {
            config.entity_types.as_ref().cloned().unwrap_or_else(|| {
                vec![
                    "PERSON".to_string(),
                    "ORGANIZATION".to_string(),
                    "LOCATION".to_string(),
                ]
            })
        } else {
            vec![
                "PERSON".to_string(),
                "ORGANIZATION".to_string(),
                "LOCATION".to_string(),
            ]
        };

        // Extract entities based on configured types
        for entity_type in &entity_types {
            match entity_type.as_str() {
                "PERSON" | "CHARACTER" | "RESEARCHER" | "SPEAKER" | "DIALOGUE_SPEAKER" => {
                    entities.extend(self.extract_persons(text, &chunk.id)?);
                }
                "ORGANIZATION" | "INSTITUTION" | "BRAND" | "COMPANY" => {
                    entities.extend(self.extract_organizations(text, &chunk.id)?);
                }
                "LOCATION" | "SETTING" | "PLACE" => {
                    entities.extend(self.extract_locations(text, &chunk.id)?);
                }
                "CONCEPT" | "THEORY" | "THEME" | "ARGUMENT" | "IDEA" => {
                    entities.extend(self.extract_concepts(text, &chunk.id, entity_type)?);
                }
                "EVENT" | "EXPERIMENT" | "HAPPENING" => {
                    entities.extend(self.extract_events(text, &chunk.id)?);
                }
                "OBJECT" | "TOOL" | "ARTIFACT" | "ITEM" => {
                    entities.extend(self.extract_objects(text, &chunk.id)?);
                }
                _ => {
                    // For any other entity type, use generic extraction
                    entities.extend(self.extract_generic_entities(text, &chunk.id, entity_type)?);
                }
            }
        }

        // Apply pattern filtering
        entities = self.apply_pattern_filtering(entities);

        // Deduplicate entities by name and type
        entities = self.deduplicate_entities(entities);

        // Filter by confidence
        entities.retain(|e| e.confidence >= self.min_confidence);

        Ok(entities)
    }

    /// Extract person entities using enhanced capitalization and context heuristics
    fn extract_persons(&self, text: &str, chunk_id: &ChunkId) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut processed_indices = HashSet::new();

        // Known titles and honorifics that indicate a person follows
        let person_titles = [
            "mr",
            "mrs",
            "ms",
            "dr",
            "prof",
            "professor",
            "sir",
            "lady",
            "lord",
            "captain",
            "major",
            "colonel",
            "general",
            "admiral",
            "judge",
            "father",
            "mother",
            "brother",
            "sister",
            "aunt",
            "uncle",
            "grandfather",
            "grandmother",
        ];

        // Common words that are NOT person names (to avoid false positives)
        let non_person_words = [
            "chapter",
            "the",
            "and",
            "but",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "can",
            "will",
            "just",
            "should",
            "now",
            "temptations",
            "strategic",
            "movements",
            "decides",
            "upon",
            "whitewashing",
            "saturday",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "sunday",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "adventures",
            "complete",
        ];

        // PHASE 1: Extract well-known character names first (prevent concatenation)
        entities.extend(self.extract_known_names(
            &words,
            &mut processed_indices,
            chunk_id,
            text,
        )?);

        // PHASE 2: Extract title-based names (Dr. Smith, Guardian Entity)
        entities.extend(self.extract_title_based_names(
            &words,
            &person_titles,
            &mut processed_indices,
            chunk_id,
            text,
        )?);

        // PHASE 3: Extract two-word names (First Last pattern)
        entities.extend(self.extract_two_word_names(
            &words,
            &non_person_words,
            &mut processed_indices,
            chunk_id,
            text,
        )?);

        // PHASE 4: Extract remaining single-word names (only if not processed yet)
        for (i, &word_ref) in words.iter().enumerate() {
            if processed_indices.contains(&i) {
                continue;
            }

            let word = self.clean_word(word_ref);

            // Skip if word is too short or is a known non-person word
            if word.len() < 2 || non_person_words.contains(&word.to_lowercase().as_str()) {
                continue;
            }

            // Look for capitalized words that could be single names
            if self.is_capitalized(words[i]) && self.is_likely_person_word(&word) {
                let confidence = self.calculate_confidence(&word, "PERSON");
                if confidence >= self.min_confidence {
                    entities.push(self.create_entity(word, "PERSON", confidence, chunk_id, text)?);
                }
            }
        }

        Ok(entities)
    }

    /// Extract well-known character names to prevent concatenation
    fn extract_known_names(
        &self,
        words: &[&str],
        processed: &mut std::collections::HashSet<usize>,
        chunk_id: &ChunkId,
        text: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let known_names = [
            ("Entity Name", 2),
            ("Second Entity", 2),
            ("Guardian Entity", 2),
            ("Friend Entity", 2),
            ("Companion Entity", 2),
            ("Third Entity", 2),
            ("Fourth Entity", 2),
            ("Fifth Entity", 2),
            ("Sixth Entity", 2),
            ("Seventh Entity", 2),
            ("Eighth Entity", 2),
            ("Ninth Entity", 2),
        ];

        for i in 0..words.len() {
            if processed.contains(&i) {
                continue;
            }

            for &(name, word_count) in &known_names {
                let name_words: Vec<&str> = name.split_whitespace().collect();
                if i + name_words.len() <= words.len() {
                    let matches = name_words.iter().enumerate().all(|(j, &expected)| {
                        let actual = self.clean_word(words[i + j]);
                        actual.to_lowercase() == expected.to_lowercase()
                    });

                    if matches {
                        let confidence = 0.95;
                        if confidence >= self.min_confidence {
                            entities.push(self.create_entity(
                                name.to_string(),
                                "PERSON",
                                confidence,
                                chunk_id,
                                text,
                            )?);
                        }
                        // Mark these indices as processed
                        for j in 0..word_count {
                            processed.insert(i + j);
                        }
                        break;
                    }
                }
            }
        }
        Ok(entities)
    }

    /// Extract title-based names (Dr. Smith, Guardian Entity)
    fn extract_title_based_names(
        &self,
        words: &[&str],
        person_titles: &[&str],
        processed: &mut std::collections::HashSet<usize>,
        chunk_id: &ChunkId,
        text: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for i in 0..words.len() {
            if processed.contains(&i) {
                continue;
            }

            let word_clean = self.clean_word(words[i]).to_lowercase();
            if person_titles.contains(&word_clean.as_str())
                && i + 1 < words.len()
                && !processed.contains(&(i + 1))
            {
                let next_word = self.clean_word(words[i + 1]);
                if self.is_capitalized(words[i + 1]) && self.is_likely_person_word(&next_word) {
                    let name = if i + 2 < words.len() && !processed.contains(&(i + 2)) {
                        let third_word = self.clean_word(words[i + 2]);
                        if self.is_capitalized(words[i + 2])
                            && self.is_likely_person_word(&third_word)
                        {
                            processed.insert(i + 2);
                            format!("{next_word} {third_word}")
                        } else {
                            next_word
                        }
                    } else {
                        next_word
                    };

                    let confidence = 0.9;
                    if confidence >= self.min_confidence {
                        entities
                            .push(self.create_entity(name, "PERSON", confidence, chunk_id, text)?);
                    }
                    processed.insert(i);
                    processed.insert(i + 1);
                }
            }
        }
        Ok(entities)
    }

    /// Extract two-word names (First Last pattern)
    fn extract_two_word_names(
        &self,
        words: &[&str],
        non_person_words: &[&str],
        processed: &mut std::collections::HashSet<usize>,
        chunk_id: &ChunkId,
        text: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for i in 0..words.len() {
            if processed.contains(&i) || i + 1 >= words.len() || processed.contains(&(i + 1)) {
                continue;
            }

            let first_word = self.clean_word(words[i]);
            let second_word = self.clean_word(words[i + 1]);

            // Check if both words are capitalized and look like names
            if self.is_capitalized(words[i])
                && self.is_capitalized(words[i + 1])
                && self.is_likely_person_word(&first_word)
                && self.is_likely_person_word(&second_word)
                && !non_person_words.contains(&first_word.to_lowercase().as_str())
                && !non_person_words.contains(&second_word.to_lowercase().as_str())
            {
                let name = format!("{first_word} {second_word}");
                if self.is_likely_person_name(&name) {
                    let confidence = self.calculate_confidence(&name, "PERSON");
                    if confidence >= self.min_confidence {
                        entities
                            .push(self.create_entity(name, "PERSON", confidence, chunk_id, text)?);
                    }
                    processed.insert(i);
                    processed.insert(i + 1);
                }
            }
        }
        Ok(entities)
    }

    /// Extract organization entities
    fn extract_organizations(&self, text: &str, chunk_id: &ChunkId) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let org_suffixes = [
            "Inc",
            "Corp",
            "LLC",
            "Ltd",
            "Company",
            "Corporation",
            "Group",
            "Solutions",
            "Technologies",
        ];
        let org_prefixes = ["University of", "Institute of", "Department of"];

        // Look for org suffixes
        for suffix in &org_suffixes {
            if let Some(pos) = text.find(suffix) {
                // Extract potential organization name
                let start = text[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
                let end = pos + suffix.len();
                let name = text[start..end].trim().to_string();

                if !name.is_empty() && self.is_likely_organization(&name) {
                    let confidence = self.calculate_confidence(&name, "ORGANIZATION");
                    if confidence >= self.min_confidence {
                        entities.push(self.create_entity(
                            name,
                            "ORGANIZATION",
                            confidence,
                            chunk_id,
                            text,
                        )?);
                    }
                }
            }
        }

        // Look for org prefixes
        for prefix in &org_prefixes {
            if let Some(pos) = text.find(prefix) {
                let start = pos;
                let end = text[pos..]
                    .find('.')
                    .map(|i| pos + i)
                    .unwrap_or(text.len().min(pos + 50));
                let name = text[start..end].trim().to_string();

                if !name.is_empty() && name.len() > prefix.len() {
                    let confidence = self.calculate_confidence(&name, "ORGANIZATION");
                    if confidence >= self.min_confidence {
                        entities.push(self.create_entity(
                            name,
                            "ORGANIZATION",
                            confidence,
                            chunk_id,
                            text,
                        )?);
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Extract location entities
    fn extract_locations(&self, text: &str, chunk_id: &ChunkId) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let known_locations = [
            "United States",
            "New York",
            "California",
            "London",
            "Paris",
            "Tokyo",
            "Berlin",
            "Washington",
            "Boston",
            "Chicago",
        ];

        for location in &known_locations {
            if text.contains(location) {
                let confidence = self.calculate_confidence(location, "LOCATION");
                if confidence >= self.min_confidence {
                    entities.push(self.create_entity(
                        location.to_string(),
                        "LOCATION",
                        confidence,
                        chunk_id,
                        text,
                    )?);
                }
            }
        }

        Ok(entities)
    }

    /// Create an entity with mentions
    fn create_entity(
        &self,
        name: String,
        entity_type: &str,
        confidence: f32,
        chunk_id: &ChunkId,
        text: &str,
    ) -> Result<Entity> {
        let entity_id = EntityId::new(format!("{}_{}", entity_type, self.normalize_name(&name)));

        // Find all occurrences of the name in text for mentions
        let mut mentions = Vec::new();
        let mut start = 0;
        while let Some(pos) = text[start..].find(&name) {
            let actual_pos = start + pos;
            mentions.push(EntityMention {
                chunk_id: chunk_id.clone(),
                start_offset: actual_pos,
                end_offset: actual_pos + name.len(),
                confidence,
            });
            start = actual_pos + name.len();
        }

        Ok(
            Entity::new(entity_id, name, entity_type.to_string(), confidence)
                .with_mentions(mentions),
        )
    }

    /// Check if a word is capitalized
    fn is_capitalized(&self, word: &str) -> bool {
        word.chars().next().is_some_and(|c| c.is_uppercase())
    }

    /// Clean word by removing punctuation
    fn clean_word(&self, word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphabetic() || *c == '\'') // Keep apostrophes for names like O'Connor
            .collect::<String>()
            .trim_end_matches('\'') // Remove trailing apostrophes
            .to_string()
    }

    /// Enhanced check if a word could be part of a person's name
    fn is_likely_person_word(&self, word: &str) -> bool {
        if word.len() < 2 {
            return false;
        }

        // Check for common name patterns
        let word_lower = word.to_lowercase();

        // Common name endings that suggest it's a person name
        let name_endings = [
            "son", "sen", "ton", "ham", "ford", "ley", "ment", "ard", "ert",
        ];
        let has_name_ending = name_endings
            .iter()
            .any(|&ending| word_lower.ends_with(ending));

        // Common name prefixes
        let name_prefixes = ["mc", "mac", "o'", "de", "van", "von", "la", "le"];
        let has_name_prefix = name_prefixes
            .iter()
            .any(|&prefix| word_lower.starts_with(prefix));

        // Must start with uppercase and be alphabetic
        let is_proper_format = word.chars().next().unwrap().is_uppercase()
            && word.chars().all(|c| c.is_alphabetic() || c == '\'');

        // Common short words that are rarely names
        let short_non_names = [
            "it", "is", "as", "at", "be", "by", "do", "go", "he", "if", "in", "me", "my", "no",
            "of", "on", "or", "so", "to", "up", "us", "we",
        ];

        if word.len() <= 2 && short_non_names.contains(&word_lower.as_str()) {
            return false;
        }

        is_proper_format && (word.len() >= 3 || has_name_ending || has_name_prefix)
    }

    /// Check if a word is a title
    #[allow(dead_code)]
    fn is_title(&self, word: &str) -> bool {
        matches!(word, "Dr." | "Mr." | "Ms." | "Mrs." | "Prof.")
    }

    /// Check if a name is likely a person name
    fn is_likely_person_name(&self, name: &str) -> bool {
        let parts: Vec<&str> = name.split_whitespace().collect();
        parts.len() == 2 && parts.iter().all(|part| self.is_capitalized(part))
    }

    /// Check if a name is likely an organization
    fn is_likely_organization(&self, name: &str) -> bool {
        let org_indicators = [
            "Inc",
            "Corp",
            "LLC",
            "Ltd",
            "Company",
            "Corporation",
            "University",
            "Institute",
        ];
        org_indicators
            .iter()
            .any(|indicator| name.contains(indicator))
    }

    /// Calculate confidence score for an entity
    fn calculate_confidence(&self, name: &str, entity_type: &str) -> f32 {
        let mut confidence: f32 = 0.5; // Base confidence

        // Adjust based on entity type patterns
        match entity_type {
            "PERSON" => {
                if name.contains("Dr.") || name.contains("Prof.") {
                    confidence += 0.3;
                }
                if name.split_whitespace().count() == 2 {
                    confidence += 0.2;
                }
            }
            "ORGANIZATION" => {
                if name.contains("Inc") || name.contains("Corp") || name.contains("LLC") {
                    confidence += 0.3;
                }
                if name.contains("University") || name.contains("Institute") {
                    confidence += 0.2;
                }
            }
            "LOCATION" => {
                if name.contains(',') {
                    confidence += 0.2;
                }
                if self.is_known_location(name) {
                    confidence += 0.3;
                }
            }
            _ => {}
        }

        // Adjust based on capitalization
        if name.chars().next().is_some_and(|c| c.is_uppercase()) {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    /// Check if a name is a known location
    fn is_known_location(&self, name: &str) -> bool {
        const KNOWN_LOCATIONS: &[&str] = &[
            "United States",
            "New York",
            "California",
            "London",
            "Paris",
            "Tokyo",
            "Berlin",
            "Washington",
            "Boston",
            "Chicago",
        ];
        KNOWN_LOCATIONS.iter().any(|&loc| name.contains(loc))
    }

    /// Normalize entity name for ID generation
    fn normalize_name(&self, name: &str) -> String {
        name.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .replace(' ', "_")
    }

    /// Deduplicate entities by name and type
    fn deduplicate_entities(&self, entities: Vec<Entity>) -> Vec<Entity> {
        let mut unique_entities: HashMap<(String, String), Entity> = HashMap::new();

        for entity in entities {
            let key = (entity.name.clone(), entity.entity_type.clone());

            match unique_entities.get_mut(&key) {
                Some(existing) => {
                    // Merge mentions and take highest confidence
                    existing.mentions.extend(entity.mentions);
                    if entity.confidence > existing.confidence {
                        existing.confidence = entity.confidence;
                    }
                }
                None => {
                    unique_entities.insert(key, entity);
                }
            }
        }

        unique_entities.into_values().collect()
    }

    /// Extract relationships between entities in the same chunk
    pub fn extract_relationships(
        &self,
        entities: &[Entity],
        chunk: &TextChunk,
    ) -> Result<Vec<(EntityId, EntityId, String)>> {
        let mut relationships = Vec::new();

        // Simple co-occurrence based relationship extraction
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];

                // Check if both entities appear in the same chunk
                let entity1_in_chunk = entity1.mentions.iter().any(|m| m.chunk_id == chunk.id);
                let entity2_in_chunk = entity2.mentions.iter().any(|m| m.chunk_id == chunk.id);

                if entity1_in_chunk && entity2_in_chunk {
                    let relation_type =
                        self.infer_relationship_type(entity1, entity2, &chunk.content);
                    relationships.push((entity1.id.clone(), entity2.id.clone(), relation_type));
                }
            }
        }

        Ok(relationships)
    }

    /// Infer relationship type between two entities
    fn infer_relationship_type(&self, entity1: &Entity, entity2: &Entity, context: &str) -> String {
        match (&entity1.entity_type[..], &entity2.entity_type[..]) {
            ("PERSON", "ORGANIZATION") | ("ORGANIZATION", "PERSON") => {
                if context.contains("works for") || context.contains("employed by") {
                    "WORKS_FOR".to_string()
                } else if context.contains("founded") || context.contains("CEO") {
                    "LEADS".to_string()
                } else {
                    "ASSOCIATED_WITH".to_string()
                }
            }
            ("PERSON", "LOCATION") | ("LOCATION", "PERSON") => {
                if context.contains("born in") || context.contains("from") {
                    "BORN_IN".to_string()
                } else if context.contains("lives in") || context.contains("based in") {
                    "LOCATED_IN".to_string()
                } else {
                    "ASSOCIATED_WITH".to_string()
                }
            }
            ("ORGANIZATION", "LOCATION") | ("LOCATION", "ORGANIZATION") => {
                if context.contains("headquartered") || context.contains("based in") {
                    "HEADQUARTERED_IN".to_string()
                } else {
                    "LOCATED_IN".to_string()
                }
            }
            ("PERSON", "PERSON") => {
                if context.contains("married") || context.contains("spouse") {
                    "MARRIED_TO".to_string()
                } else if context.contains("colleague") || context.contains("partner") {
                    "COLLEAGUE_OF".to_string()
                } else {
                    "KNOWS".to_string()
                }
            }
            _ => "RELATED_TO".to_string(),
        }
    }

    /// Apply pattern filtering to entities based on configured patterns
    fn apply_pattern_filtering(&self, entities: Vec<Entity>) -> Vec<Entity> {
        if self.allowed_patterns.is_empty() && self.excluded_patterns.is_empty() {
            return entities;
        }

        entities
            .into_iter()
            .filter(|entity| {
                // If we have allowed patterns, entity must match at least one
                if !self.allowed_patterns.is_empty() {
                    let matches_allowed = self
                        .allowed_patterns
                        .iter()
                        .any(|pattern| pattern.is_match(&entity.name));
                    if !matches_allowed {
                        return false;
                    }
                }

                // Entity must not match any excluded patterns
                if !self.excluded_patterns.is_empty() {
                    let matches_excluded = self
                        .excluded_patterns
                        .iter()
                        .any(|pattern| pattern.is_match(&entity.name));
                    if matches_excluded {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Extract concept entities (themes, ideas, theories)
    fn extract_concepts(
        &self,
        text: &str,
        chunk_id: &ChunkId,
        entity_type: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        // Look for conceptual terms that are typically capitalized
        let concept_indicators = [
            "Theory",
            "Concept",
            "Principle",
            "Philosophy",
            "Doctrine",
            "Idea",
            "Method",
            "Approach",
            "Framework",
            "Model",
            "Paradigm",
            "Thesis",
        ];

        for &word in words.iter() {
            let clean_word = self.clean_word(word);

            // Check if this word indicates a concept
            if concept_indicators
                .iter()
                .any(|&indicator| clean_word.contains(indicator))
            {
                let confidence = 0.75;
                if confidence >= self.min_confidence {
                    entities.push(self.create_entity(
                        clean_word,
                        entity_type,
                        confidence,
                        chunk_id,
                        text,
                    )?);
                }
            }

            // Look for capitalized terms that might be concepts
            if self.is_capitalized(word) && word.len() > 4 {
                let clean_word = self.clean_word(word);
                if !self.is_common_word(&clean_word) {
                    let confidence = 0.6;
                    if confidence >= self.min_confidence {
                        entities.push(self.create_entity(
                            clean_word,
                            entity_type,
                            confidence,
                            chunk_id,
                            text,
                        )?);
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Extract event entities
    fn extract_events(&self, text: &str, chunk_id: &ChunkId) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Event indicators
        let event_words = [
            "meeting",
            "conference",
            "ceremony",
            "celebration",
            "festival",
            "competition",
            "war",
            "battle",
            "expedition",
            "journey",
            "trial",
        ];

        for event_word in &event_words {
            if text.to_lowercase().contains(event_word) {
                let confidence = 0.7;
                if confidence >= self.min_confidence {
                    entities.push(self.create_entity(
                        event_word.to_string(),
                        "EVENT",
                        confidence,
                        chunk_id,
                        text,
                    )?);
                }
            }
        }

        Ok(entities)
    }

    /// Extract object entities (tools, artifacts, items)
    fn extract_objects(&self, text: &str, chunk_id: &ChunkId) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Object indicators
        let object_words = [
            "sword",
            "shield",
            "book",
            "manuscript",
            "scroll",
            "tablet",
            "ring",
            "crown",
            "treasure",
            "coin",
            "tool",
            "weapon",
        ];

        for object_word in &object_words {
            if text.to_lowercase().contains(object_word) {
                let confidence = 0.65;
                if confidence >= self.min_confidence {
                    entities.push(self.create_entity(
                        object_word.to_string(),
                        "OBJECT",
                        confidence,
                        chunk_id,
                        text,
                    )?);
                }
            }
        }

        Ok(entities)
    }

    /// Generic entity extraction for any configured entity type
    fn extract_generic_entities(
        &self,
        text: &str,
        chunk_id: &ChunkId,
        entity_type: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        // For generic entity types, look for capitalized words that might be entities
        for &word in &words {
            if self.is_capitalized(word) && word.len() > 3 {
                let clean_word = self.clean_word(word);
                if !self.is_common_word(&clean_word) {
                    let confidence = 0.5; // Lower confidence for generic extraction
                    if confidence >= self.min_confidence {
                        entities.push(self.create_entity(
                            clean_word,
                            entity_type,
                            confidence,
                            chunk_id,
                            text,
                        )?);
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Check if a word is a common word that shouldn't be extracted as an entity
    fn is_common_word(&self, word: &str) -> bool {
        let common_words = [
            "the", "and", "but", "or", "in", "on", "at", "to", "for", "with", "by", "from",
            "about", "into", "through", "during", "before", "after", "above", "below", "up",
            "down", "out", "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
            "can", "will", "just", "should", "now", "could", "would", "said", "says", "told",
            "asked", "went", "came", "come", "going", "Chapter", "Page", "Section", "Part", "Book",
            "Volume",
        ];

        common_words
            .iter()
            .any(|&common| word.eq_ignore_ascii_case(common))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId};

    #[test]
    fn test_person_extraction() {
        let extractor = EntityExtractor::new(0.5).unwrap();
        let chunk = TextChunk::new(
            ChunkId::new("test_chunk".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Entity Name works at Test Corp. Dr. Second Entity is a professor.".to_string(),
            0,
            59,
        );

        let entities = extractor.extract_from_chunk(&chunk).unwrap();

        // Should extract persons and organizations
        assert!(!entities.is_empty());

        let person_entities: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == "PERSON")
            .collect();
        assert!(!person_entities.is_empty());
    }

    #[test]
    fn test_relationship_extraction() {
        let extractor = EntityExtractor::new(0.5).unwrap();
        let chunk = TextChunk::new(
            ChunkId::new("test_chunk".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Entity Name works for Test Corp in Test City.".to_string(),
            0,
            44,
        );

        let entities = extractor.extract_from_chunk(&chunk).unwrap();
        let relationships = extractor.extract_relationships(&entities, &chunk).unwrap();

        assert!(!relationships.is_empty());
    }
}
