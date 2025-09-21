//! Dual-level retrieval system implementation for LightRAG

use std::collections::HashMap;
use crate::core::Result;
use crate::lightrag::kv_store::{KVStore, KVData};
use crate::lightrag::graph_indexer::{EntityReference, ConceptSummary};

/// Retrieval modes for dual-level system
#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalMode {
    /// Context-dependent information retrieval
    Local,
    /// Global knowledge retrieval
    Global,
    /// Combines local and global retrieval
    Hybrid,
    /// Integrates knowledge graph and vector retrieval
    Mix,
}

/// Fusion weights for combining retrieval results
#[derive(Debug, Clone)]
pub struct FusionWeights {
    pub local_weight: f32,
    pub global_weight: f32,
    pub entity_weight: f32,
    pub concept_weight: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            local_weight: 0.6,
            global_weight: 0.4,
            entity_weight: 0.7,
            concept_weight: 0.3,
        }
    }
}

/// Dual-level retrieval system
pub struct DualRetriever {
    low_level_index: HashMap<String, Vec<EntityReference>>,
    high_level_index: HashMap<String, Vec<ConceptSummary>>,
    retrieval_mode: RetrievalMode,
    fusion_weights: FusionWeights,
    kv_store: Option<Box<dyn KVStore>>,
}

impl DualRetriever {
    pub fn new(retrieval_mode: RetrievalMode, fusion_weights: FusionWeights) -> Result<Self> {
        Ok(Self {
            low_level_index: HashMap::new(),
            high_level_index: HashMap::new(),
            retrieval_mode,
            fusion_weights,
            kv_store: None,
        })
    }

    pub fn set_kv_store(&mut self, store: Box<dyn KVStore>) {
        self.kv_store = Some(store);
    }

    /// Build dual-level indexes from extraction results
    pub fn build_indexes(&mut self, extraction_result: &crate::lightrag::graph_indexer::ExtractionResult) -> Result<()> {
        // Build low-level index (entities and their attributes)
        for entity_ref in &extraction_result.entities {
            let key = self.generate_entity_key(&entity_ref.name, &entity_ref.entity_type);
            self.low_level_index
                .entry(key)
                .or_default()
                .push(entity_ref.clone());

            // Store in KV store if available
            if let Some(ref mut store) = self.kv_store {
                let kv_data = KVData::Entity(crate::lightrag::kv_store::EntityData {
                    name: entity_ref.name.clone(),
                    entity_type: entity_ref.entity_type.clone(),
                    attributes: entity_ref.attributes.clone(),
                    mentions: entity_ref.mentions.clone(),
                });
                store.store(&entity_ref.id, kv_data)?;
            }
        }

        // Build high-level index (concepts and themes)
        for concept in &extraction_result.concepts {
            let key = self.generate_concept_key(&concept.theme);
            self.high_level_index
                .entry(key)
                .or_default()
                .push(concept.clone());

            // Store in KV store if available
            if let Some(ref mut store) = self.kv_store {
                let kv_data = KVData::Concept(crate::lightrag::kv_store::ConceptData {
                    theme: concept.theme.clone(),
                    summary: concept.summary.clone(),
                    related_entities: concept.related_entities.clone(),
                    confidence: concept.confidence,
                });
                store.store(&concept.id, kv_data)?;
            }
        }

        Ok(())
    }

    /// Retrieve information using dual-level approach
    pub fn retrieve(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        match self.retrieval_mode {
            RetrievalMode::Local => self.retrieve_local(query),
            RetrievalMode::Global => self.retrieve_global(query),
            RetrievalMode::Hybrid => self.retrieve_hybrid(query),
            RetrievalMode::Mix => self.retrieve_mix(query),
        }
    }

    fn retrieve_local(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        let mut results = Vec::new();
        let query_tokens = self.tokenize_query(query);

        // Search low-level index for specific entities
        for token in &query_tokens {
            if let Some(entity_refs) = self.low_level_index.get(token) {
                for entity_ref in entity_refs {
                    let score = self.calculate_entity_score(query, entity_ref);
                    results.push(RetrievalResult {
                        content: self.format_entity_content(entity_ref),
                        score,
                        level: crate::lightrag::RetrievalLevel::LowLevel,
                        source_id: entity_ref.id.clone(),
                    });
                }
            }
        }

        // Sort by score and return top results
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(10); // Limit to top 10 results
        Ok(results)
    }

    fn retrieve_global(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        let mut results = Vec::new();
        let query_tokens = self.tokenize_query(query);

        // Search high-level index for concepts and themes
        for token in &query_tokens {
            if let Some(concepts) = self.high_level_index.get(token) {
                for concept in concepts {
                    let score = self.calculate_concept_score(query, concept);
                    results.push(RetrievalResult {
                        content: self.format_concept_content(concept),
                        score,
                        level: crate::lightrag::RetrievalLevel::HighLevel,
                        source_id: concept.id.clone(),
                    });
                }
            }
        }

        // Sort by score and return top results
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(10);
        Ok(results)
    }

    fn retrieve_hybrid(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        let local_results = self.retrieve_local(query)?;
        let global_results = self.retrieve_global(query)?;

        // Combine and reweight results
        let mut combined_results = Vec::new();

        for mut result in local_results {
            result.score *= self.fusion_weights.local_weight;
            combined_results.push(result);
        }

        for mut result in global_results {
            result.score *= self.fusion_weights.global_weight;
            combined_results.push(result);
        }

        // Sort by adjusted scores
        combined_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        combined_results.truncate(15); // Return top 15 for hybrid approach
        Ok(combined_results)
    }

    fn retrieve_mix(&self, query: &str) -> Result<Vec<RetrievalResult>> {
        // Mix mode integrates both KG and vector retrieval
        let mut results = self.retrieve_hybrid(query)?;

        // If KV store is available, perform additional retrieval
        if let Some(ref store) = self.kv_store {
            let kv_results = store.search(query)?;
            for kv_data in kv_results {
                let score = 0.5; // Base score for KV results
                let content = self.format_kv_content(&kv_data);
                results.push(RetrievalResult {
                    content,
                    score,
                    level: crate::lightrag::RetrievalLevel::LowLevel,
                    source_id: format!("kv_{}", kv_data.get_id()),
                });
            }
        }

        // Final sorting and limiting
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(20); // Return top 20 for mix mode
        Ok(results)
    }

    // Helper methods

    fn generate_entity_key(&self, name: &str, entity_type: &str) -> String {
        format!("{}:{}", entity_type.to_lowercase(), name.to_lowercase())
    }

    fn generate_concept_key(&self, theme: &str) -> String {
        theme.to_lowercase().replace(' ', "_")
    }

    fn tokenize_query(&self, query: &str) -> Vec<String> {
        query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    fn calculate_entity_score(&self, query: &str, entity_ref: &EntityReference) -> f32 {
        let query_lower = query.to_lowercase();
        let name_lower = entity_ref.name.to_lowercase();

        let mut score = 0.0;

        // Exact name match
        if name_lower.contains(&query_lower) || query_lower.contains(&name_lower) {
            score += 1.0;
        }

        // Attribute matches
        for attribute in &entity_ref.attributes {
            if attribute.to_lowercase().contains(&query_lower) {
                score += 0.5;
            }
        }

        // Mention count boost
        score += (entity_ref.mentions.len() as f32).ln() * 0.1;

        score
    }

    fn calculate_concept_score(&self, query: &str, concept: &ConceptSummary) -> f32 {
        let query_lower = query.to_lowercase();
        let theme_lower = concept.theme.to_lowercase();
        let summary_lower = concept.summary.to_lowercase();

        let mut score = 0.0;

        // Theme match
        if theme_lower.contains(&query_lower) || query_lower.contains(&theme_lower) {
            score += 1.0;
        }

        // Summary content match
        if summary_lower.contains(&query_lower) {
            score += 0.7;
        }

        // Confidence boost
        score *= concept.confidence;

        score
    }

    fn format_entity_content(&self, entity_ref: &EntityReference) -> String {
        format!(
            "Entity: {} ({})\nAttributes: {}\nMentions: {}",
            entity_ref.name,
            entity_ref.entity_type,
            entity_ref.attributes.join(", "),
            entity_ref.mentions.len()
        )
    }

    fn format_concept_content(&self, concept: &ConceptSummary) -> String {
        format!(
            "Concept: {}\nSummary: {}\nRelated Entities: {}\nConfidence: {:.2}",
            concept.theme,
            concept.summary,
            concept.related_entities.join(", "),
            concept.confidence
        )
    }

    fn format_kv_content(&self, kv_data: &KVData) -> String {
        match kv_data {
            KVData::Entity(entity_data) => {
                format!(
                    "KV Entity: {} ({})\nAttributes: {}",
                    entity_data.name,
                    entity_data.entity_type,
                    entity_data.attributes.join(", ")
                )
            }
            KVData::Relation(relation_data) => {
                format!(
                    "KV Relation: {} -> {} ({})",
                    relation_data.source_entity,
                    relation_data.target_entity,
                    relation_data.relation_type
                )
            }
            KVData::Concept(concept_data) => {
                format!(
                    "KV Concept: {}\nSummary: {}",
                    concept_data.theme,
                    concept_data.summary
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub content: String,
    pub score: f32,
    pub level: crate::lightrag::RetrievalLevel,
    pub source_id: String,
}

// ExtractionResult is now imported from graph_indexer module