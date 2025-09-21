use crate::{
    core::{Entity, Relationship, Result, TextChunk},
    entity::EntityExtractor,
    ollama::OllamaClient,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct GleaningConfig {
    pub max_gleaning_rounds: usize,
    pub completion_threshold: f64,
    pub entity_confidence_threshold: f64,
    pub use_llm_completion_check: bool,
}

impl Default for GleaningConfig {
    fn default() -> Self {
        Self {
            max_gleaning_rounds: 3,
            completion_threshold: 0.8,
            entity_confidence_threshold: 0.7,
            use_llm_completion_check: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionCompletionStatus {
    pub is_complete: bool,
    pub confidence: f64,
    pub missing_aspects: Vec<String>,
    pub suggestions: Vec<String>,
}

pub struct GleaningEntityExtractor {
    base_extractor: EntityExtractor,
    llm_client: Option<OllamaClient>,
    config: GleaningConfig,
}

impl GleaningEntityExtractor {
    pub fn new(base_extractor: EntityExtractor, config: GleaningConfig) -> Self {
        Self {
            base_extractor,
            llm_client: None,
            config,
        }
    }

    pub fn with_llm_client(mut self, client: OllamaClient) -> Self {
        self.llm_client = Some(client);
        self.config.use_llm_completion_check = true;
        self
    }

    /// Extract entities with iterative refinement (gleaning)
    pub async fn extract_with_gleaning(
        &self,
        chunk: &TextChunk,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        let mut all_entities = Vec::new();
        let mut all_relationships = Vec::new();

        // Initial extraction
        println!("🔍 Initial extraction for chunk: {}", chunk.id);
        let entities = self.base_extractor.extract_from_chunk(chunk)?;

        // For now, start with just entities, no relationships from base extractor
        all_entities.extend(entities.clone());

        // Extract relationships between entities
        let relationships = self
            .base_extractor
            .extract_relationships(&entities, chunk)?;
        for (source, target, rel_type) in relationships {
            all_relationships.push(crate::core::Relationship {
                source,
                target,
                relation_type: rel_type,
                confidence: 0.7,
                context: vec![chunk.id.clone()],
            });
        }

        // Iterative gleaning rounds
        for round in 1..=self.config.max_gleaning_rounds {
            println!("🤖 Gleaning round {} for chunk: {}", round, chunk.id);

            // Check if extraction is complete
            let status = self
                .check_extraction_completion(chunk, &all_entities, &all_relationships)
                .await?;

            if status.is_complete && status.confidence > self.config.completion_threshold {
                println!("✅ Extraction complete after {round} rounds");
                break;
            }

            // Perform additional extraction round
            let (additional_entities, additional_rels) = self
                .extract_additional(chunk, &all_entities, &status.missing_aspects)
                .await?;

            if additional_entities.is_empty() && additional_rels.is_empty() {
                println!("⚠️ No additional entities found in round {round}");
                break;
            }

            println!(
                "📈 Round {} added: {} entities, {} relationships",
                round,
                additional_entities.len(),
                additional_rels.len()
            );

            // Filter by confidence threshold
            let filtered_entities: Vec<Entity> = additional_entities
                .into_iter()
                .filter(|e| e.confidence >= self.config.entity_confidence_threshold as f32)
                .collect();

            all_entities.extend(filtered_entities);
            all_relationships.extend(additional_rels);
        }

        // Deduplicate and merge similar entities
        let (deduplicated_entities, deduplicated_relationships) =
            self.deduplicate_results(all_entities, all_relationships)?;

        println!(
            "📊 Final gleaning results: {} entities, {} relationships",
            deduplicated_entities.len(),
            deduplicated_relationships.len()
        );

        Ok((deduplicated_entities, deduplicated_relationships))
    }

    async fn check_extraction_completion(
        &self,
        chunk: &TextChunk,
        entities: &[Entity],
        relationships: &[Relationship],
    ) -> Result<ExtractionCompletionStatus> {
        if self.config.use_llm_completion_check {
            if let Some(ref llm_client) = self.llm_client {
                return self
                    .llm_completion_check(llm_client, chunk, entities, relationships)
                    .await;
            }
        }

        // Fallback to heuristic completion check
        Ok(self.heuristic_completion_check(chunk, entities, relationships))
    }

    async fn llm_completion_check(
        &self,
        _llm_client: &OllamaClient,
        chunk: &TextChunk,
        entities: &[Entity],
        relationships: &[Relationship],
    ) -> Result<ExtractionCompletionStatus> {
        let prompt = self.build_completion_check_prompt(chunk, entities, relationships);

        // Simulate LLM response for now
        // In a real implementation, this would call the actual LLM
        let _response = prompt; // Placeholder

        // Simple heuristic for now
        let is_complete = entities.len() > 2;

        Ok(ExtractionCompletionStatus {
            is_complete,
            confidence: if is_complete { 0.9 } else { 0.5 },
            missing_aspects: if is_complete {
                Vec::new()
            } else {
                vec!["Additional entities may be present".to_string()]
            },
            suggestions: vec!["Look for more specific entity mentions".to_string()],
        })
    }

    fn heuristic_completion_check(
        &self,
        chunk: &TextChunk,
        entities: &[Entity],
        relationships: &[Relationship],
    ) -> ExtractionCompletionStatus {
        let text_length = chunk.content.len();
        let entity_count = entities.len();
        let relationship_count = relationships.len();

        // Simple heuristics for completion
        let entity_density = entity_count as f64 / (text_length as f64 / 100.0);
        let has_relationships = relationship_count > 0;

        // Check for common entity types that might be missing
        let has_person = entities
            .iter()
            .any(|e| e.entity_type.to_uppercase().contains("PERSON"));
        let has_org = entities
            .iter()
            .any(|e| e.entity_type.to_uppercase().contains("ORG"));
        let _has_location = entities
            .iter()
            .any(|e| e.entity_type.to_uppercase().contains("LOC"));

        let mut missing_aspects = Vec::new();
        let mut suggestions = Vec::new();

        // Check for potential missing entity types
        if text_length > 200 && entity_count < 3 {
            missing_aspects.push("Low entity density".to_string());
            suggestions.push("Look for more named entities".to_string());
        }

        if entity_count > 2 && relationship_count == 0 {
            missing_aspects.push("No relationships found".to_string());
            suggestions.push("Look for relationships between entities".to_string());
        }

        // Check text patterns for missing entities
        if (chunk.content.to_lowercase().contains(" inc")
            || chunk.content.to_lowercase().contains(" corp")
            || chunk.content.to_lowercase().contains(" company"))
            && !has_org
        {
            missing_aspects.push("Potential organization entities".to_string());
            suggestions.push("Look for organization names".to_string());
        }

        if (chunk.content.to_lowercase().contains(" said")
            || chunk.content.to_lowercase().contains(" told")
            || chunk.content.to_lowercase().contains("according to"))
            && !has_person
        {
            missing_aspects.push("Potential person entities".to_string());
            suggestions.push("Look for person names in quotes or attributions".to_string());
        }

        // Determine completion status
        let is_complete = missing_aspects.is_empty()
            && entity_density > 1.0
            && (entity_count < 2 || has_relationships);

        let confidence = if is_complete {
            0.8
        } else {
            0.5 - (missing_aspects.len() as f64 * 0.1)
        };

        ExtractionCompletionStatus {
            is_complete,
            confidence: confidence.max(0.1),
            missing_aspects,
            suggestions,
        }
    }

    fn build_completion_check_prompt(
        &self,
        chunk: &TextChunk,
        entities: &[Entity],
        relationships: &[Relationship],
    ) -> String {
        let mut prompt =
            String::from("Analyze the completeness of entity extraction from this text:\n\n");

        prompt.push_str(&format!("TEXT:\n{}\n\n", chunk.content));

        prompt.push_str("EXTRACTED ENTITIES:\n");
        for entity in entities {
            prompt.push_str(&format!(
                "- {} ({}): confidence {:.2}\n",
                entity.name, entity.entity_type, entity.confidence
            ));
        }

        prompt.push_str("\nEXTRACTED RELATIONSHIPS:\n");
        for relationship in relationships {
            prompt.push_str(&format!(
                "- {} -> {} ({})\n",
                relationship.source, relationship.target, relationship.relation_type
            ));
        }

        prompt.push_str(
            "\nAre there any important entities, people, places, concepts, or relationships \
             that seem to be missing? Consider:\n\
             1. Named entities (people, organizations, locations)\n\
             2. Important concepts or objects\n\
             3. Temporal references\n\
             4. Relationships between entities\n\n\
             Respond with 'COMPLETE' if extraction seems sufficient, or 'INCOMPLETE' with \
             suggestions for what might be missing.",
        );

        prompt
    }

    async fn extract_additional(
        &self,
        chunk: &TextChunk,
        existing_entities: &[Entity],
        missing_aspects: &[String],
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        let focus_areas = if missing_aspects.is_empty() {
            "additional named entities and relationships".to_string()
        } else {
            missing_aspects.join(", ")
        };

        let existing_names: Vec<String> =
            existing_entities.iter().map(|e| e.name.clone()).collect();

        // Create a focused extraction prompt
        println!("🎯 Focusing on: {focus_areas}");

        // For now, use the base extractor with additional context
        // In a real implementation, this would use a more sophisticated approach
        let mut additional_entities = self.base_extractor.extract_from_chunk(chunk)?;
        let additional_relationships = Vec::new(); // No relationships for now in additional extraction

        // Filter out entities that are too similar to existing ones
        additional_entities.retain(|new_entity| {
            !existing_names
                .iter()
                .any(|existing_name| self.are_similar_names(&new_entity.name, existing_name))
        });

        // Apply additional pattern-based extraction based on missing aspects
        let pattern_entities = self.extract_with_patterns(chunk, missing_aspects)?;
        additional_entities.extend(pattern_entities);

        Ok((additional_entities, additional_relationships))
    }

    fn are_similar_names(&self, name1: &str, name2: &str) -> bool {
        let name1_lower = name1.to_lowercase();
        let name2_lower = name2.to_lowercase();

        // Exact match
        if name1_lower == name2_lower {
            return true;
        }

        // One contains the other (with length check to avoid false positives)
        if name1_lower.len() > 3
            && name2_lower.len() > 3
            && (name1_lower.contains(&name2_lower) || name2_lower.contains(&name1_lower))
        {
            return true;
        }

        // Check for abbreviation patterns (e.g., "John Smith" vs "J. Smith")
        if self.check_abbreviation_similarity(&name1_lower, &name2_lower) {
            return true;
        }

        // Jaccard similarity on words
        let words1: std::collections::HashSet<&str> = name1_lower.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = name2_lower.split_whitespace().collect();

        if words1.is_empty() || words2.is_empty() {
            return false;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        let jaccard = intersection as f64 / union as f64;
        jaccard > 0.7
    }

    fn check_abbreviation_similarity(&self, name1: &str, name2: &str) -> bool {
        let words1: Vec<&str> = name1.split_whitespace().collect();
        let words2: Vec<&str> = name2.split_whitespace().collect();

        // Check if one name could be an abbreviation of the other
        self.is_abbreviation_match(&words1, &words2) || self.is_abbreviation_match(&words2, &words1)
    }

    fn is_abbreviation_match(&self, full_words: &[&str], abbrev_words: &[&str]) -> bool {
        if full_words.len() != abbrev_words.len() {
            return false;
        }

        for (full_word, abbrev_word) in full_words.iter().zip(abbrev_words.iter()) {
            // Check if abbrev_word is an abbreviation of full_word
            if !self.is_word_abbreviation(full_word, abbrev_word) && full_word != abbrev_word {
                return false;
            }
        }

        true
    }

    fn is_word_abbreviation(&self, full_word: &str, abbrev_word: &str) -> bool {
        // Remove punctuation from abbreviation
        let clean_abbrev = abbrev_word.trim_end_matches('.');

        // Check if it's a single letter abbreviation
        if clean_abbrev.len() == 1 && full_word.starts_with(clean_abbrev) {
            return true;
        }

        // Check if abbreviation matches the start of the full word
        if full_word.starts_with(clean_abbrev) && clean_abbrev.len() < full_word.len() {
            return true;
        }

        false
    }

    fn extract_with_patterns(
        &self,
        chunk: &TextChunk,
        missing_aspects: &[String],
    ) -> Result<Vec<Entity>> {
        let mut pattern_entities = Vec::new();
        let content = &chunk.content;

        // Pattern-based extraction for organizations
        if missing_aspects
            .iter()
            .any(|aspect| aspect.to_lowercase().contains("org"))
        {
            let org_patterns = [
                r"([A-Z][a-zA-Z\s]+(?:Inc|Corp|Company|Ltd|LLC)\.?)",
                r"([A-Z][a-zA-Z\s]+ (?:Corporation|Incorporated|Limited))",
            ];

            for pattern in &org_patterns {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    for captures in regex.captures_iter(content) {
                        if let Some(org_match) = captures.get(1) {
                            let org_name = org_match.as_str().trim();
                            if org_name.len() > 2 {
                                pattern_entities.push(Entity::new(
                                    crate::core::EntityId::new(format!(
                                        "gleaning_org_{}",
                                        pattern_entities.len()
                                    )),
                                    org_name.to_string(),
                                    "ORGANIZATION".to_string(),
                                    0.7,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Pattern-based extraction for persons
        if missing_aspects
            .iter()
            .any(|aspect| aspect.to_lowercase().contains("person"))
        {
            let person_patterns = [
                r"(?:Mr\.|Mrs\.|Ms\.|Dr\.) ([A-Z][a-z]+ [A-Z][a-z]+)",
                r"([A-Z][a-z]+ [A-Z][a-z]+) said",
                r"according to ([A-Z][a-z]+ [A-Z][a-z]+)",
            ];

            for pattern in &person_patterns {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    for captures in regex.captures_iter(content) {
                        if let Some(person_match) = captures.get(1) {
                            let person_name = person_match.as_str().trim();
                            if person_name.len() > 3 {
                                pattern_entities.push(Entity::new(
                                    crate::core::EntityId::new(format!(
                                        "gleaning_person_{}",
                                        pattern_entities.len()
                                    )),
                                    person_name.to_string(),
                                    "PERSON".to_string(),
                                    0.6,
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(pattern_entities)
    }

    fn deduplicate_results(
        &self,
        entities: Vec<Entity>,
        relationships: Vec<Relationship>,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        // Simple deduplication by name similarity
        let mut deduplicated_entities = Vec::new();
        let mut seen_names = std::collections::HashSet::<String>::new();

        for entity in entities {
            let normalized_name = entity.name.to_lowercase().trim().to_string();

            // Check if we've seen a similar name
            let is_duplicate = seen_names
                .iter()
                .any(|seen_name| self.are_similar_names(&normalized_name, seen_name));

            if !is_duplicate {
                seen_names.insert(normalized_name);
                deduplicated_entities.push(entity);
            }
        }

        // For relationships, deduplicate by source-target-type combination
        let mut deduplicated_relationships = Vec::new();
        let mut seen_relations = std::collections::HashSet::new();

        for relationship in relationships {
            let relation_key = format!(
                "{}->{}:{}",
                relationship.source, relationship.target, relationship.relation_type
            );

            if !seen_relations.contains(&relation_key) {
                seen_relations.insert(relation_key);
                deduplicated_relationships.push(relationship);
            }
        }

        Ok((deduplicated_entities, deduplicated_relationships))
    }

    /// Get extraction statistics
    pub fn get_statistics(&self) -> GleaningStatistics {
        GleaningStatistics {
            config: self.config.clone(),
            llm_available: self.llm_client.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GleaningStatistics {
    pub config: GleaningConfig,
    pub llm_available: bool,
}

impl GleaningStatistics {
    pub fn print(&self) {
        println!("🌾 Gleaning Extraction Statistics");
        println!("  Max rounds: {}", self.config.max_gleaning_rounds);
        println!(
            "  Completion threshold: {:.2}",
            self.config.completion_threshold
        );
        println!(
            "  Entity confidence threshold: {:.2}",
            self.config.entity_confidence_threshold
        );
        println!(
            "  Uses LLM completion check: {}",
            self.config.use_llm_completion_check
        );
        println!("  LLM available: {}", self.llm_available);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId, TextChunk};

    fn create_test_chunk() -> TextChunk {
        TextChunk::new(
            ChunkId::new("test_chunk".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Dr. Sarah Johnson, the lead researcher at MIT's AI Lab, published a breakthrough paper on neural networks. The research was funded by a $2M grant from the NSF and conducted in collaboration with Stanford University.".to_string(),
            0,
            200,
        )
    }

    #[test]
    fn test_gleaning_extractor_creation() {
        let base_extractor = EntityExtractor::new(0.7).unwrap();
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(base_extractor, config);

        let stats = extractor.get_statistics();
        assert_eq!(stats.config.max_gleaning_rounds, 3);
        assert!(!stats.llm_available);
    }

    #[test]
    fn test_heuristic_completion_check() {
        let base_extractor = EntityExtractor::new(0.7).unwrap();
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(base_extractor, config);

        let chunk = create_test_chunk();
        let entities = vec![Entity::new(
            crate::core::EntityId::new("test1".to_string()),
            "Dr. Sarah Johnson".to_string(),
            "PERSON".to_string(),
            0.9,
        )];
        let relationships = vec![];

        let status = extractor.heuristic_completion_check(&chunk, &entities, &relationships);

        assert!(!status.missing_aspects.is_empty());
        assert!(status.confidence > 0.0);
    }

    #[test]
    fn test_name_similarity() {
        let base_extractor = EntityExtractor::new(0.7).unwrap();
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(base_extractor, config);

        assert!(extractor.are_similar_names("Apple Inc", "Apple Inc."));
        assert!(extractor.are_similar_names("John Smith", "J. Smith"));
        assert!(!extractor.are_similar_names("Apple", "Microsoft"));
    }

    #[tokio::test]
    async fn test_pattern_extraction() {
        let base_extractor = EntityExtractor::new(0.7).unwrap();
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(base_extractor, config);

        let chunk = TextChunk::new(
            ChunkId::new("test_chunk".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Apple Inc. is a technology company. Dr. John Smith works there.".to_string(),
            0,
            60,
        );

        let missing_aspects = vec![
            "organization entities".to_string(),
            "person entities".to_string(),
        ];
        let pattern_entities = extractor
            .extract_with_patterns(&chunk, &missing_aspects)
            .unwrap();

        assert!(!pattern_entities.is_empty());

        let has_org = pattern_entities
            .iter()
            .any(|e| e.entity_type == "ORGANIZATION");
        let has_person = pattern_entities.iter().any(|e| e.entity_type == "PERSON");

        assert!(has_org || has_person);
    }

    #[test]
    fn test_deduplication() {
        let base_extractor = EntityExtractor::new(0.7).unwrap();
        let config = GleaningConfig::default();
        let extractor = GleaningEntityExtractor::new(base_extractor, config);

        let entities = vec![
            Entity::new(
                crate::core::EntityId::new("test1".to_string()),
                "Apple Inc".to_string(),
                "ORGANIZATION".to_string(),
                0.9,
            ),
            Entity::new(
                crate::core::EntityId::new("test2".to_string()),
                "Apple Inc.".to_string(),
                "ORGANIZATION".to_string(),
                0.8,
            ),
            Entity::new(
                crate::core::EntityId::new("test3".to_string()),
                "Microsoft".to_string(),
                "ORGANIZATION".to_string(),
                0.9,
            ),
        ];

        let relationships = vec![];

        let (dedup_entities, dedup_rels) = extractor
            .deduplicate_results(entities, relationships)
            .unwrap();

        // Should deduplicate Apple entities
        assert_eq!(dedup_entities.len(), 2); // Apple (merged) and Microsoft
        assert_eq!(dedup_rels.len(), 0);
    }
}
