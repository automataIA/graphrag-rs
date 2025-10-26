//! Cross-document entity linking with LMCD clustering

use crate::core::{Result, Entity};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCluster {
    pub cluster_id: String,
    pub canonical_name: String,
    pub entity_type: String,
    pub member_entities: Vec<CrossDocumentEntity>,
    pub confidence_score: f32,
    pub document_frequency: usize,
    pub aliases: Vec<String>,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDocumentEntity {
    pub document_id: String,
    pub local_entity_id: String,
    pub name: String,
    pub context: String,
    pub confidence: f32,
    pub mentions: Vec<String>, // Simplified to avoid serialization issues
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinkingStrategy {
    /// Exact string matching
    ExactMatch,
    /// Fuzzy string matching with edit distance
    FuzzyMatch { threshold: f32 },
    /// Semantic similarity using embeddings
    SemanticSimilarity { threshold: f32 },
    /// LMCD (Language Model Confident Deduplication) clustering
    #[allow(clippy::upper_case_acronyms)]
    LMCD { confidence_threshold: f32 },
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct LinkingStats {
    pub entities_processed: usize,
    pub clusters_created: usize,
    pub exact_matches: usize,
    pub fuzzy_matches: usize,
    pub semantic_matches: usize,
    pub lmcd_clusters: usize,
    pub disambiguation_conflicts: usize,
    pub linking_accuracy: f32,
}

impl Default for LinkingStats {
    fn default() -> Self {
        Self {
            entities_processed: 0,
            clusters_created: 0,
            exact_matches: 0,
            fuzzy_matches: 0,
            semantic_matches: 0,
            lmcd_clusters: 0,
            disambiguation_conflicts: 0,
            linking_accuracy: 0.0,
        }
    }
}

impl LinkingStats {
    pub fn print(&self) {
        tracing::info!(
            entities_processed = self.entities_processed,
            clusters_created = self.clusters_created,
            exact_matches = self.exact_matches,
            fuzzy_matches = self.fuzzy_matches,
            semantic_matches = self.semantic_matches,
            lmcd_clusters = self.lmcd_clusters,
            disambiguation_conflicts = self.disambiguation_conflicts,
            linking_accuracy = format!("{:.1}%", self.linking_accuracy * 100.0),
            "Entity linking statistics"
        );
    }
}

pub struct CrossDocumentEntityLinker {
    strategy: LinkingStrategy,
    clusters: HashMap<String, EntityCluster>,
    entity_index: HashMap<String, Vec<String>>, // entity_name -> cluster_ids
    stats: LinkingStats,
}

impl CrossDocumentEntityLinker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            strategy: LinkingStrategy::Hybrid,
            clusters: HashMap::new(),
            entity_index: HashMap::new(),
            stats: LinkingStats::default(),
        })
    }

    pub fn with_strategy(strategy: LinkingStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            clusters: HashMap::new(),
            entity_index: HashMap::new(),
            stats: LinkingStats::default(),
        })
    }

    /// Link entities across documents using the configured strategy
    pub async fn link_entities(&mut self, document_entities: HashMap<String, Vec<Entity>>) -> Result<Vec<EntityCluster>> {
        tracing::info!(
            document_count = document_entities.len(),
            strategy = ?self.strategy,
            "Starting cross-document entity linking"
        );

        // Flatten entities from all documents
        let mut all_entities = Vec::new();
        for (doc_id, entities) in document_entities {
            for entity in entities {
                all_entities.push(CrossDocumentEntity {
                    document_id: doc_id.clone(),
                    local_entity_id: entity.id.to_string(),
                    name: entity.name.clone(),
                    context: "entity context".to_string(), // Entity doesn't have description field
                    confidence: 1.0, // Default confidence
                    mentions: Vec::new(), // Could be populated from entity mentions
                });
            }
        }

        self.stats.entities_processed = all_entities.len();
        tracing::info!(entity_count = all_entities.len(), "Total entities to link");

        // Perform linking based on strategy
        match &self.strategy {
            LinkingStrategy::ExactMatch => self.exact_match_linking(all_entities).await?,
            LinkingStrategy::FuzzyMatch { threshold } => self.fuzzy_match_linking(all_entities, *threshold).await?,
            LinkingStrategy::SemanticSimilarity { threshold } => self.semantic_similarity_linking(all_entities, *threshold).await?,
            LinkingStrategy::LMCD { confidence_threshold } => self.lmcd_clustering(all_entities, *confidence_threshold).await?,
            LinkingStrategy::Hybrid => self.hybrid_linking(all_entities).await?,
        }

        // Calculate final statistics
        self.stats.clusters_created = self.clusters.len();
        self.stats.linking_accuracy = self.calculate_linking_accuracy();

        tracing::info!(cluster_count = self.clusters.len(), "Linking complete");

        Ok(self.clusters.values().cloned().collect())
    }

    /// Exact string matching for entity linking
    async fn exact_match_linking(&mut self, entities: Vec<CrossDocumentEntity>) -> Result<()> {
        let mut name_clusters: HashMap<String, Vec<CrossDocumentEntity>> = HashMap::new();

        // Group entities by exact name match
        for entity in entities {
            name_clusters.entry(entity.name.clone()).or_default().push(entity);
        }

        // Create clusters for each name group
        for (name, group_entities) in &name_clusters {
            if group_entities.len() > 1 {
                self.stats.exact_matches += group_entities.len() - 1;
            }

            let cluster_id = format!("cluster_{}", uuid::Uuid::new_v4());
            let entity_type = self.infer_entity_type(name);

            let cluster = EntityCluster {
                cluster_id: cluster_id.clone(),
                canonical_name: name.clone(),
                entity_type,
                member_entities: group_entities.clone(),
                confidence_score: 1.0, // High confidence for exact matches
                document_frequency: group_entities.len(),
                aliases: vec![name.clone()],
                properties: HashMap::new(),
            };

            self.clusters.insert(cluster_id.clone(), cluster);
            self.entity_index.entry(name.to_string()).or_default().push(cluster_id);
        }

        Ok(())
    }

    /// Fuzzy string matching with edit distance
    async fn fuzzy_match_linking(&mut self, entities: Vec<CrossDocumentEntity>, threshold: f32) -> Result<()> {
        tracing::debug!(threshold = format!("{:.2}", threshold), "Fuzzy matching");

        let mut unprocessed = entities;
        let mut cluster_counter = 0;

        while !unprocessed.is_empty() {
            let seed_entity = unprocessed.remove(0);
            let seed_name = seed_entity.name.clone();

            let mut cluster_entities = vec![seed_entity];
            let mut to_remove = Vec::new();

            // Find similar entities using edit distance
            for (i, entity) in unprocessed.iter().enumerate() {
                let similarity = self.calculate_string_similarity(&seed_name, &entity.name);
                if similarity >= threshold {
                    cluster_entities.push(entity.clone());
                    to_remove.push(i);
                    self.stats.fuzzy_matches += 1;
                }
            }

            // Remove matched entities from unprocessed (in reverse order to maintain indices)
            for &index in to_remove.iter().rev() {
                unprocessed.remove(index);
            }

            // Create cluster
            if !cluster_entities.is_empty() {
                let cluster_id = format!("cluster_{cluster_counter}");
                cluster_counter += 1;

                let canonical_name = self.select_canonical_name(&cluster_entities);
                let entity_type = self.infer_entity_type(&canonical_name);

                let cluster = EntityCluster {
                    cluster_id: cluster_id.clone(),
                    canonical_name: canonical_name.clone(),
                    entity_type,
                    member_entities: cluster_entities,
                    confidence_score: threshold,
                    document_frequency: 1,
                    aliases: Vec::new(),
                    properties: HashMap::new(),
                };

                self.clusters.insert(cluster_id.clone(), cluster);
                self.entity_index.entry(canonical_name).or_default().push(cluster_id);
            }
        }

        Ok(())
    }

    /// Semantic similarity using embeddings (placeholder implementation)
    async fn semantic_similarity_linking(&mut self, entities: Vec<CrossDocumentEntity>, threshold: f32) -> Result<()> {
        tracing::debug!(threshold = format!("{:.2}", threshold), "Semantic similarity linking");

        // For now, fall back to fuzzy matching
        // In a real implementation, this would use embedding similarity
        self.fuzzy_match_linking(entities, threshold * 0.8).await?;
        self.stats.semantic_matches = self.stats.fuzzy_matches;
        self.stats.fuzzy_matches = 0;

        Ok(())
    }

    /// LMCD (Language Model Confident Deduplication) clustering
    async fn lmcd_clustering(&mut self, entities: Vec<CrossDocumentEntity>, confidence_threshold: f32) -> Result<()> {
        tracing::debug!(confidence_threshold = format!("{:.2}", confidence_threshold), "LMCD clustering");

        // Placeholder implementation - would use LLM for confident deduplication
        // For now, use a hybrid approach with high confidence

        // First pass: exact matching for high confidence
        let mut exact_clusters: HashMap<String, Vec<CrossDocumentEntity>> = HashMap::new();
        let mut remaining_entities = Vec::new();

        for entity in entities {
            if exact_clusters.contains_key(&entity.name) {
                exact_clusters.get_mut(&entity.name).unwrap().push(entity);
            } else {
                let similar_found = exact_clusters.keys()
                    .any(|name| self.calculate_string_similarity(name, &entity.name) > 0.9);

                if !similar_found {
                    exact_clusters.insert(entity.name.clone(), vec![entity]);
                } else {
                    remaining_entities.push(entity);
                }
            }
        }

        // Create clusters from exact matches
        for (name, group_entities) in exact_clusters {
            let cluster_id = format!("lmcd_cluster_{}", uuid::Uuid::new_v4());
            let entity_type = self.infer_entity_type(&name);

            let cluster = EntityCluster {
                cluster_id: cluster_id.clone(),
                canonical_name: name.clone(),
                entity_type,
                member_entities: group_entities,
                confidence_score: confidence_threshold + 0.1, // High confidence
                document_frequency: 1,
                aliases: Vec::new(),
                properties: HashMap::new(),
            };

            self.clusters.insert(cluster_id.clone(), cluster);
            self.entity_index.entry(name.to_string()).or_default().push(cluster_id);
            self.stats.lmcd_clusters += 1;
        }

        // Handle remaining entities with lower confidence
        for entity in remaining_entities {
            let cluster_id = format!("lmcd_singleton_{}", uuid::Uuid::new_v4());
            let entity_type = self.infer_entity_type(&entity.name);

            let cluster = EntityCluster {
                cluster_id: cluster_id.clone(),
                canonical_name: entity.name.clone(),
                entity_type,
                member_entities: vec![entity],
                confidence_score: confidence_threshold - 0.1, // Lower confidence
                document_frequency: 1,
                aliases: Vec::new(),
                properties: HashMap::new(),
            };

            self.clusters.insert(cluster_id.clone(), cluster);
        }

        Ok(())
    }

    /// Hybrid linking combining multiple strategies
    async fn hybrid_linking(&mut self, entities: Vec<CrossDocumentEntity>) -> Result<()> {
        tracing::debug!("Hybrid linking strategy");

        // Start with exact matching for high confidence entities
        let mut entity_groups: HashMap<String, Vec<CrossDocumentEntity>> = HashMap::new();
        for entity in entities {
            entity_groups.entry(entity.name.clone()).or_default().push(entity);
        }

        // Exact matches first
        let mut all_entities = Vec::new();
        for (name, group) in entity_groups {
            if group.len() > 1 {
                // Multiple entities with same name - create cluster
                let cluster_id = format!("exact_cluster_{}", uuid::Uuid::new_v4());
                let entity_type = self.infer_entity_type(&name);
                let group_len = group.len();

                let cluster = EntityCluster {
                    cluster_id: cluster_id.clone(),
                    canonical_name: name.clone(),
                    entity_type,
                    member_entities: group,
                    confidence_score: 1.0,
                    document_frequency: group_len,
                    aliases: vec![name.clone()],
                    properties: HashMap::new(),
                };

                self.clusters.insert(cluster_id.clone(), cluster);
                self.entity_index.entry(name.to_string()).or_default().push(cluster_id);
                self.stats.exact_matches += group_len - 1;
            } else {
                // Single entity - add to fuzzy matching pool
                all_entities.extend(group);
            }
        }

        // Fuzzy matching for remaining entities
        if !all_entities.is_empty() {
            self.fuzzy_match_linking(all_entities, 0.85).await?;
        }

        Ok(())
    }

    /// Link entities from a newly added document
    pub async fn link_new_document_entities(&mut self, entities: Vec<Entity>) -> Result<()> {
        for entity in entities {
            let cross_doc_entity = CrossDocumentEntity {
                document_id: "new_document".to_string(), // Would be actual doc ID
                local_entity_id: entity.id.to_string(),
                name: entity.name.clone(),
                context: "entity context".to_string(), // Entity doesn't have description field
                confidence: 1.0,
                mentions: Vec::new(),
            };

            // Try to match with existing clusters
            if let Some(cluster_ids) = self.entity_index.get(&entity.name) {
                if let Some(cluster_id) = cluster_ids.first() {
                    if let Some(cluster) = self.clusters.get_mut(cluster_id) {
                        cluster.member_entities.push(cross_doc_entity);
                        cluster.document_frequency += 1;
                        continue;
                    }
                }
            }

            // No exact match found - create new cluster
            let cluster_id = format!("new_cluster_{}", uuid::Uuid::new_v4());
            let entity_type = self.infer_entity_type(&entity.name);

            let cluster = EntityCluster {
                cluster_id: cluster_id.clone(),
                canonical_name: entity.name.clone(),
                entity_type,
                member_entities: vec![cross_doc_entity],
                confidence_score: 0.8, // Medium confidence for new entities
                document_frequency: 1,
                aliases: Vec::new(),
                properties: HashMap::new(),
            };

            self.clusters.insert(cluster_id.clone(), cluster);
            self.entity_index.entry(entity.name).or_default().push(cluster_id);
        }

        Ok(())
    }

    /// Calculate string similarity using edit distance
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f32 {
        if s1 == s2 {
            return 1.0;
        }

        let distance = self.levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len());

        if max_len == 0 {
            return 1.0;
        }

        1.0 - (distance as f32 / max_len as f32)
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Select canonical name from cluster entities
    fn select_canonical_name(&self, entities: &[CrossDocumentEntity]) -> String {
        // Use the most frequent name, or the longest one if frequencies are equal
        let mut name_counts: HashMap<String, usize> = HashMap::new();
        for entity in entities {
            *name_counts.entry(entity.name.clone()).or_insert(0) += 1;
        }

        name_counts.into_iter()
            .max_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| a.0.len().cmp(&b.0.len()))
            })
            .map(|(name, _)| name)
            .unwrap_or_else(|| entities[0].name.clone())
    }

    /// Infer entity type from name (placeholder implementation)
    fn infer_entity_type(&self, name: &str) -> String {
        // Simple heuristics - could be enhanced with NER
        let name_lower = name.to_lowercase();

        if name_lower.contains("company") || name_lower.contains("corp") || name_lower.contains("inc") {
            "organization".to_string()
        } else if name.chars().next().unwrap_or('a').is_uppercase() && !name.contains(' ') {
            "person".to_string()
        } else if name_lower.contains("city") || name_lower.contains("country") {
            "location".to_string()
        } else {
            "other".to_string()
        }
    }

    /// Calculate linking accuracy (placeholder implementation)
    fn calculate_linking_accuracy(&self) -> f32 {
        if self.stats.entities_processed == 0 {
            return 0.0;
        }

        // Simple metric: ratio of entities successfully clustered
        let successfully_linked = self.stats.exact_matches + self.stats.fuzzy_matches + self.stats.semantic_matches;
        successfully_linked as f32 / self.stats.entities_processed as f32
    }

    /// Get linking statistics
    pub fn get_stats(&self) -> &LinkingStats {
        &self.stats
    }

    /// Get all clusters
    pub fn get_clusters(&self) -> &HashMap<String, EntityCluster> {
        &self.clusters
    }

    /// Find clusters by entity name
    pub fn find_clusters(&self, entity_name: &str) -> Vec<&EntityCluster> {
        self.entity_index.get(entity_name)
            .map(|cluster_ids| {
                cluster_ids.iter()
                    .filter_map(|id| self.clusters.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }
}