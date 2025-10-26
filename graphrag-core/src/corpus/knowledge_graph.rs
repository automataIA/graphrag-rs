//! Corpus-level knowledge graph construction and management

use crate::core::Result;
use crate::corpus::entity_linker::EntityCluster;
use crate::corpus::document_manager::DocumentCollection;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use serde::{Deserialize, Serialize};
use petgraph::{Graph, Directed};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalEntity {
    pub id: String,
    pub canonical_name: String,
    pub entity_type: String,
    pub document_frequency: usize,
    pub total_mentions: usize,
    pub aliases: Vec<String>,
    pub properties: HashMap<String, String>,
    pub source_documents: Vec<String>,
    pub confidence_score: f32,
    pub importance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRelation {
    pub id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub relation_type: String,
    pub confidence: f32,
    pub document_frequency: usize,
    pub source_documents: Vec<String>,
    pub properties: HashMap<String, String>,
    pub contexts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CorpusKnowledgeGraph {
    pub global_entities: HashMap<String, GlobalEntity>,
    pub global_relations: HashMap<String, GlobalRelation>,
    pub graph: Graph<String, String, Directed>, // entity_id -> relation_type
    pub entity_node_map: HashMap<String, NodeIndex>,
    pub document_entity_map: HashMap<String, Vec<String>>, // doc_id -> entity_ids
    pub stats: GraphStats,
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_entities: usize,
    pub total_relations: usize,
    pub cross_document_entities: usize,
    pub single_document_entities: usize,
    pub avg_entity_connections: f32,
    pub graph_density: f32,
    pub largest_component_size: usize,
}

impl Default for GraphStats {
    fn default() -> Self {
        Self {
            total_entities: 0,
            total_relations: 0,
            cross_document_entities: 0,
            single_document_entities: 0,
            avg_entity_connections: 0.0,
            graph_density: 0.0,
            largest_component_size: 0,
        }
    }
}

impl GraphStats {
    pub fn print(&self) {
        tracing::info!(
            total_entities = self.total_entities,
            total_relations = self.total_relations,
            cross_document_entities = self.cross_document_entities,
            single_document_entities = self.single_document_entities,
            avg_entity_connections = format!("{:.1}", self.avg_entity_connections),
            graph_density = format!("{:.3}", self.graph_density),
            largest_component_size = self.largest_component_size,
            "Corpus knowledge graph statistics"
        );
    }
}

impl CorpusKnowledgeGraph {
    pub fn new() -> Result<Self> {
        Ok(Self {
            global_entities: HashMap::new(),
            global_relations: HashMap::new(),
            graph: Graph::new(),
            entity_node_map: HashMap::new(),
            document_entity_map: HashMap::new(),
            stats: GraphStats::default(),
        })
    }

    /// Build corpus knowledge graph from entity clusters
    pub async fn build_from_clusters(
        &mut self,
        clusters: Vec<EntityCluster>,
        collection: &DocumentCollection,
    ) -> Result<Self> {
        tracing::info!(
            cluster_count = clusters.len(),
            document_count = collection.documents.len(),
            "Building corpus-level knowledge graph"
        );

        // Step 1: Create global entities from clusters
        for cluster in &clusters {
            let global_entity = self.create_global_entity(cluster)?;
            let node_index = self.graph.add_node(global_entity.id.clone());

            self.entity_node_map.insert(global_entity.id.clone(), node_index);
            self.global_entities.insert(global_entity.id.clone(), global_entity);

            // Map documents to entities
            for member in &cluster.member_entities {
                self.document_entity_map
                    .entry(member.document_id.clone())
                    .or_default()
                    .push(cluster.cluster_id.clone());
            }
        }

        // Step 2: Identify cross-document relations
        self.identify_cross_document_relations(&clusters, collection).await?;

        // Step 3: Calculate entity importance scores
        self.calculate_importance_scores();

        // Step 4: Update statistics
        self.update_statistics();

        tracing::info!(
            entities = self.global_entities.len(),
            relations = self.global_relations.len(),
            "Knowledge graph built"
        );

        Ok(self.clone())
    }

    /// Create a global entity from an entity cluster
    fn create_global_entity(&self, cluster: &EntityCluster) -> Result<GlobalEntity> {
        let total_mentions = cluster.member_entities.len();
        let source_documents: Vec<String> = cluster.member_entities
            .iter()
            .map(|e| e.document_id.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Collect all unique aliases
        let mut aliases = HashSet::new();
        for member in &cluster.member_entities {
            aliases.insert(member.name.clone());
        }
        let aliases: Vec<String> = aliases.into_iter().collect();

        // Calculate importance score based on document frequency and mentions
        let importance_score = self.calculate_entity_importance(
            cluster.document_frequency,
            total_mentions,
            &source_documents,
        );

        Ok(GlobalEntity {
            id: cluster.cluster_id.clone(),
            canonical_name: cluster.canonical_name.clone(),
            entity_type: cluster.entity_type.clone(),
            document_frequency: cluster.document_frequency,
            total_mentions,
            aliases,
            properties: cluster.properties.clone(),
            source_documents,
            confidence_score: cluster.confidence_score,
            importance_score,
        })
    }

    /// Identify relations that span across documents
    async fn identify_cross_document_relations(
        &mut self,
        clusters: &[EntityCluster],
        _collection: &DocumentCollection,
    ) -> Result<()> {
        tracing::debug!("Identifying cross-document relations");

        // For now, create relations between entities that appear in the same documents
        let mut document_cooccurrences: HashMap<String, Vec<String>> = HashMap::new();

        // Group entities by document
        for cluster in clusters {
            for member in &cluster.member_entities {
                document_cooccurrences
                    .entry(member.document_id.clone())
                    .or_default()
                    .push(cluster.cluster_id.clone());
            }
        }

        // Create co-occurrence relations
        let mut relation_counter = 0;
        for (doc_id, entity_ids) in document_cooccurrences {
            for i in 0..entity_ids.len() {
                for j in i + 1..entity_ids.len() {
                    let source_id = &entity_ids[i];
                    let target_id = &entity_ids[j];

                    // Check if relation already exists
                    let relation_key = format!("{source_id}_{target_id}");
                    if !self.global_relations.contains_key(&relation_key) {
                        let relation = GlobalRelation {
                            id: format!("rel_{relation_counter}"),
                            source_entity_id: source_id.clone(),
                            target_entity_id: target_id.clone(),
                            relation_type: "co_occurs".to_string(),
                            confidence: 0.7, // Medium confidence for co-occurrence
                            document_frequency: 1,
                            source_documents: vec![doc_id.clone()],
                            properties: HashMap::new(),
                            contexts: Vec::new(),
                        };

                        // Add edge to graph
                        if let (Some(&source_node), Some(&target_node)) = (
                            self.entity_node_map.get(source_id),
                            self.entity_node_map.get(target_id),
                        ) {
                            self.graph.add_edge(source_node, target_node, relation.relation_type.clone());
                        }

                        self.global_relations.insert(relation_key, relation);
                        relation_counter += 1;
                    } else {
                        // Increment frequency for existing relation
                        if let Some(existing_relation) = self.global_relations.get_mut(&relation_key) {
                            existing_relation.document_frequency += 1;
                            existing_relation.source_documents.push(doc_id.clone());
                            existing_relation.confidence = (existing_relation.confidence + 0.1).min(1.0);
                        }
                    }
                }
            }
        }

        tracing::debug!(relation_count = self.global_relations.len(), "Cross-document relations identified");
        Ok(())
    }

    /// Calculate importance scores for entities
    fn calculate_importance_scores(&mut self) {
        let entity_ids: Vec<String> = self.global_entities.keys().cloned().collect();

        for entity_id in entity_ids {
            if let Some(entity) = self.global_entities.get(&entity_id) {
                let importance_score = self.calculate_entity_importance(
                    entity.document_frequency,
                    entity.total_mentions,
                    &entity.source_documents,
                );

                if let Some(entity) = self.global_entities.get_mut(&entity_id) {
                    entity.importance_score = importance_score;
                }
            }
        }
    }

    /// Calculate entity importance based on frequency and distribution
    fn calculate_entity_importance(
        &self,
        doc_frequency: usize,
        total_mentions: usize,
        source_documents: &[String],
    ) -> f32 {
        // Combine document frequency, mention count, and document spread
        let doc_freq_score = (doc_frequency as f32).ln() + 1.0;
        let mention_score = (total_mentions as f32).ln() + 1.0;
        let spread_score = source_documents.len() as f32;

        // Weighted combination
        (doc_freq_score * 0.4 + mention_score * 0.3 + spread_score * 0.3) / 3.0
    }

    /// Update graph statistics
    fn update_statistics(&mut self) {
        self.stats.total_entities = self.global_entities.len();
        self.stats.total_relations = self.global_relations.len();

        // Count cross-document vs single-document entities
        for entity in self.global_entities.values() {
            if entity.document_frequency > 1 {
                self.stats.cross_document_entities += 1;
            } else {
                self.stats.single_document_entities += 1;
            }
        }

        // Calculate average connections
        if self.stats.total_entities > 0 {
            self.stats.avg_entity_connections = (self.stats.total_relations * 2) as f32 / self.stats.total_entities as f32;
        }

        // Calculate graph density
        if self.stats.total_entities > 1 {
            let max_possible_edges = self.stats.total_entities * (self.stats.total_entities - 1) / 2;
            if max_possible_edges > 0 {
                self.stats.graph_density = self.stats.total_relations as f32 / max_possible_edges as f32;
            }
        }

        // Calculate largest connected component (simplified)
        self.stats.largest_component_size = self.find_largest_connected_component();
    }

    /// Find largest connected component size (simplified implementation)
    fn find_largest_connected_component(&self) -> usize {
        if self.graph.node_count() == 0 {
            return 0;
        }

        // For simplicity, return the total number of nodes
        // A proper implementation would use DFS/BFS to find connected components
        self.graph.node_count()
    }

    /// Integrate a new document into the existing knowledge graph
    pub async fn integrate_new_document(&mut self, _document_metadata: &crate::corpus::document_manager::DocumentMetadata) -> Result<()> {
        // Placeholder implementation
        // In practice, this would:
        // 1. Extract entities from the new document
        // 2. Link them with existing global entities
        // 3. Update relations and statistics
        tracing::debug!("Integrating new document into knowledge graph");
        Ok(())
    }

    /// Query the knowledge graph
    pub async fn query(&self, query: &str) -> Result<Vec<GlobalEntity>> {
        tracing::debug!(query = %query, "Querying corpus knowledge graph");

        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        // Simple text matching against entity names and aliases
        for entity in self.global_entities.values() {
            let name_match = entity.canonical_name.to_lowercase().contains(&query_lower);
            let alias_match = entity.aliases.iter()
                .any(|alias| alias.to_lowercase().contains(&query_lower));

            if name_match || alias_match {
                results.push(entity.clone());
            }
        }

        // Sort by importance score
        results.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap_or(std::cmp::Ordering::Equal));

        tracing::debug!(match_count = results.len(), "Found matching entities");
        Ok(results)
    }

    /// Export knowledge graph to file
    pub async fn export(&self, output_path: &Path) -> Result<()> {
        tracing::info!(path = %output_path.display(), "Exporting knowledge graph");

        let export_data = serde_json::json!({
            "entities": self.global_entities,
            "relations": self.global_relations,
            "statistics": {
                "total_entities": self.stats.total_entities,
                "total_relations": self.stats.total_relations,
                "cross_document_entities": self.stats.cross_document_entities,
                "single_document_entities": self.stats.single_document_entities,
                "avg_entity_connections": self.stats.avg_entity_connections,
                "graph_density": self.stats.graph_density,
                "largest_component_size": self.stats.largest_component_size,
            }
        });

        std::fs::write(output_path, serde_json::to_string_pretty(&export_data)?)?;
        tracing::info!("Knowledge graph exported successfully");

        Ok(())
    }

    /// Get entities by type
    pub fn get_entities_by_type(&self, entity_type: &str) -> Vec<&GlobalEntity> {
        self.global_entities.values()
            .filter(|entity| entity.entity_type == entity_type)
            .collect()
    }

    /// Get most important entities
    pub fn get_top_entities(&self, limit: usize) -> Vec<&GlobalEntity> {
        let mut entities: Vec<&GlobalEntity> = self.global_entities.values().collect();
        entities.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap_or(std::cmp::Ordering::Equal));
        entities.into_iter().take(limit).collect()
    }

    /// Get entities that appear in multiple documents
    pub fn get_cross_document_entities(&self) -> Vec<&GlobalEntity> {
        self.global_entities.values()
            .filter(|entity| entity.document_frequency > 1)
            .collect()
    }

    /// Find related entities through graph traversal
    pub fn find_related_entities(&self, entity_id: &str, max_depth: usize) -> Vec<&GlobalEntity> {
        let mut related = Vec::new();

        if let Some(&start_node) = self.entity_node_map.get(entity_id) {
            // Simple BFS to find connected entities
            let mut visited = HashSet::new();
            let mut queue = vec![(start_node, 0)];
            visited.insert(start_node);

            while let Some((node, depth)) = queue.pop() {
                if depth >= max_depth {
                    continue;
                }

                // Add neighbors
                for edge in self.graph.edges(node) {
                    let neighbor = edge.target();
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push((neighbor, depth + 1));

                        // Find entity for this node
                        for (ent_id, &ent_node) in &self.entity_node_map {
                            if ent_node == neighbor {
                                if let Some(entity) = self.global_entities.get(ent_id) {
                                    related.push(entity);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        related
    }

    /// Get statistics
    pub fn get_stats(&self) -> &GraphStats {
        &self.stats
    }
}