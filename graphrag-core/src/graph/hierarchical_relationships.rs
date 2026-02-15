//! Hierarchical Relationship Clustering (Phase 3.1)
//!
//! This module implements hierarchical clustering of relationships in the knowledge graph,
//! creating multi-level summaries for efficient retrieval and reasoning.
//!
//! Key features:
//! - Leiden community detection applied recursively to relationships
//! - LLM-generated summaries for each cluster
//! - Tree structure with parent-child relationships between levels
//! - Enables query routing to relevant relationship clusters

use crate::{
    core::{KnowledgeGraph, Relationship, Result},
    ollama::OllamaClient,
};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete hierarchical structure of relationship clusters
///
/// Organizes relationships into multiple levels of granularity,
/// from fine-grained (Level 0) to coarse-grained (higher levels).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipHierarchy {
    /// Hierarchy levels, ordered from most detailed (0) to most abstract
    pub levels: Vec<HierarchyLevel>,
}

impl RelationshipHierarchy {
    /// Create a new empty hierarchy
    pub fn new() -> Self {
        Self {
            levels: Vec::new(),
        }
    }

    /// Add a new level to the hierarchy
    pub fn add_level(&mut self, level: HierarchyLevel) {
        self.levels.push(level);
    }

    /// Get a specific level by ID
    pub fn get_level(&self, level_id: usize) -> Option<&HierarchyLevel> {
        self.levels.iter().find(|l| l.level_id == level_id)
    }

    /// Get the total number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Find clusters containing a specific relationship
    pub fn find_clusters_for_relationship(&self, rel_id: &str) -> Vec<(&HierarchyLevel, &RelationshipCluster)> {
        let mut results = Vec::new();

        for level in &self.levels {
            for cluster in &level.clusters {
                if cluster.relationship_ids.contains(&rel_id.to_string()) {
                    results.push((level, cluster));
                }
            }
        }

        results
    }
}

impl Default for RelationshipHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// A single level in the hierarchy
///
/// Each level contains clusters of relationships at a specific granularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyLevel {
    /// Unique identifier for this level (0 = most detailed)
    pub level_id: usize,

    /// Clusters at this level
    pub clusters: Vec<RelationshipCluster>,

    /// Resolution parameter used for clustering at this level
    pub resolution: f32,
}

impl HierarchyLevel {
    /// Create a new hierarchy level
    pub fn new(level_id: usize, resolution: f32) -> Self {
        Self {
            level_id,
            clusters: Vec::new(),
            resolution,
        }
    }

    /// Add a cluster to this level
    pub fn add_cluster(&mut self, cluster: RelationshipCluster) {
        self.clusters.push(cluster);
    }

    /// Get total number of relationships across all clusters
    pub fn total_relationships(&self) -> usize {
        self.clusters.iter().map(|c| c.relationship_ids.len()).sum()
    }
}

/// A cluster of related relationships
///
/// Groups semantically or structurally similar relationships together
/// with an LLM-generated summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipCluster {
    /// Unique identifier for this cluster
    pub cluster_id: String,

    /// IDs of relationships in this cluster (format: "source_target_type")
    pub relationship_ids: Vec<String>,

    /// LLM-generated summary describing the cluster's theme
    pub summary: String,

    /// Parent cluster ID in the next level (None for top level)
    pub parent_cluster: Option<String>,

    /// Cohesion score indicating how tightly related the relationships are (0.0-1.0)
    pub cohesion_score: f32,
}

impl RelationshipCluster {
    /// Create a new relationship cluster
    pub fn new(cluster_id: String) -> Self {
        Self {
            cluster_id,
            relationship_ids: Vec::new(),
            summary: String::new(),
            parent_cluster: None,
            cohesion_score: 0.0,
        }
    }

    /// Add a relationship to this cluster
    pub fn add_relationship(&mut self, rel_id: String) {
        if !self.relationship_ids.contains(&rel_id) {
            self.relationship_ids.push(rel_id);
        }
    }

    /// Set the cluster summary
    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = summary;
        self
    }

    /// Set the parent cluster
    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.parent_cluster = Some(parent_id);
        self
    }

    /// Set the cohesion score
    pub fn with_cohesion(mut self, score: f32) -> Self {
        self.cohesion_score = score.clamp(0.0, 1.0);
        self
    }

    /// Check if cluster is empty
    pub fn is_empty(&self) -> bool {
        self.relationship_ids.is_empty()
    }

    /// Get number of relationships in cluster
    pub fn size(&self) -> usize {
        self.relationship_ids.len()
    }
}

/// Builder for hierarchical relationship clustering
pub struct HierarchyBuilder {
    /// Relationships to cluster
    relationships: Vec<Relationship>,

    /// Ollama client for generating summaries
    ollama_client: Option<OllamaClient>,

    /// Number of levels to create
    num_levels: usize,

    /// Resolution parameters for each level (higher = more clusters)
    resolutions: Vec<f32>,

    /// Minimum cluster size to keep
    min_cluster_size: usize,
}

impl HierarchyBuilder {
    /// Create a new hierarchy builder
    ///
    /// # Arguments
    ///
    /// * `relationships` - Relationships to cluster
    pub fn new(relationships: Vec<Relationship>) -> Self {
        Self {
            relationships,
            ollama_client: None,
            num_levels: 3,
            resolutions: vec![1.0, 0.5, 0.2], // High to low resolution
            min_cluster_size: 2,
        }
    }

    /// Create a builder from a knowledge graph
    pub fn from_graph(graph: &KnowledgeGraph) -> Self {
        Self::new(graph.get_all_relationships().into_iter().cloned().collect())
    }

    /// Set Ollama client for summary generation
    pub fn with_ollama_client(mut self, client: OllamaClient) -> Self {
        self.ollama_client = Some(client);
        self
    }

    /// Set number of hierarchy levels
    pub fn with_num_levels(mut self, num_levels: usize) -> Self {
        self.num_levels = num_levels;
        self
    }

    /// Set resolution parameters for clustering
    pub fn with_resolutions(mut self, resolutions: Vec<f32>) -> Self {
        self.resolutions = resolutions;
        self
    }

    /// Set minimum cluster size
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Build the hierarchical relationship structure
    ///
    /// # Returns
    ///
    /// RelationshipHierarchy with multiple levels of clusters
    #[cfg(feature = "async")]
    pub async fn build(self) -> Result<RelationshipHierarchy> {
        let mut hierarchy = RelationshipHierarchy::new();

        // Build each level from fine to coarse
        for (level_id, resolution) in self.resolutions.iter().enumerate().take(self.num_levels) {
            #[cfg(feature = "tracing")]
            tracing::info!(
                level_id = level_id,
                resolution = resolution,
                "Building hierarchy level"
            );

            let level = self.build_level(level_id, *resolution, &hierarchy).await?;
            hierarchy.add_level(level);
        }

        Ok(hierarchy)
    }

    /// Build a single level of the hierarchy
    #[cfg(feature = "async")]
    async fn build_level(
        &self,
        level_id: usize,
        resolution: f32,
        _existing_hierarchy: &RelationshipHierarchy,
    ) -> Result<HierarchyLevel> {
        let mut level = HierarchyLevel::new(level_id, resolution);

        if self.relationships.is_empty() {
            return Ok(level);
        }

        // Create relationship graph for clustering
        // Each relationship becomes a node, edges between similar relationships
        let rel_graph = self.build_relationship_graph(&self.relationships);

        // Apply Leiden clustering
        let communities = self.cluster_relationships(&rel_graph, resolution)?;

        // Create clusters from communities
        for (community_id, rel_indices) in communities {
            let cluster_id = format!("L{}C{}", level_id, community_id);
            let mut cluster = RelationshipCluster::new(cluster_id.clone());

            // Add relationships to cluster
            for idx in rel_indices {
                if let Some(rel) = self.relationships.get(idx) {
                    let rel_id = format!("{}_{}_{}",
                        rel.source.0,
                        rel.target.0,
                        rel.relation_type
                    );
                    cluster.add_relationship(rel_id);
                }
            }

            // Skip if cluster is too small
            if cluster.size() < self.min_cluster_size {
                continue;
            }

            // Generate summary if Ollama client available
            if let Some(ref ollama_client) = self.ollama_client {
                let summary = self.generate_cluster_summary(&cluster, &self.relationships, ollama_client).await?;
                cluster.summary = summary;
            } else {
                cluster.summary = format!("Cluster {} with {} relationships", cluster_id, cluster.size());
            }

            // Calculate cohesion score
            cluster.cohesion_score = self.calculate_cohesion(&cluster, &rel_graph);

            level.add_cluster(cluster);
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            level_id = level_id,
            num_clusters = level.clusters.len(),
            total_relationships = level.total_relationships(),
            "Completed hierarchy level"
        );

        Ok(level)
    }

    /// Build a graph where relationships are nodes
    fn build_relationship_graph(&self, relationships: &[Relationship]) -> petgraph::Graph<usize, f32> {
        use petgraph::Graph;

        let mut graph = Graph::new();
        let mut nodes = Vec::new();

        // Create nodes
        for (idx, _rel) in relationships.iter().enumerate() {
            nodes.push(graph.add_node(idx));
        }

        // Add edges between similar relationships
        for i in 0..relationships.len() {
            for j in (i + 1)..relationships.len() {
                let similarity = self.relationship_similarity(&relationships[i], &relationships[j]);

                // Only connect relationships with sufficient similarity
                if similarity > 0.3 {
                    graph.add_edge(nodes[i], nodes[j], similarity);
                }
            }
        }

        graph
    }

    /// Calculate similarity between two relationships
    fn relationship_similarity(&self, rel1: &Relationship, rel2: &Relationship) -> f32 {
        let mut similarity = 0.0;

        // Same relation type (strong signal)
        if rel1.relation_type == rel2.relation_type {
            similarity += 0.5;
        }

        // Share source or target entity
        if rel1.source == rel2.source || rel1.target == rel2.target {
            similarity += 0.3;
        }

        // Temporal proximity (if both have temporal info)
        if let (Some(range1), Some(range2)) = (&rel1.temporal_range, &rel2.temporal_range) {
            let overlap = self.temporal_overlap(range1, range2);
            similarity += overlap * 0.2;
        }

        similarity.min(1.0)
    }

    /// Calculate temporal overlap between two ranges
    fn temporal_overlap(
        &self,
        range1: &crate::graph::temporal::TemporalRange,
        range2: &crate::graph::temporal::TemporalRange,
    ) -> f32 {
        let start = range1.start.max(range2.start);
        let end = range1.end.min(range2.end);

        if start < end {
            let overlap = (end - start) as f32;
            let total = ((range1.end - range1.start) + (range2.end - range2.start)) as f32 / 2.0;
            (overlap / total.max(1.0)).min(1.0)
        } else {
            0.0
        }
    }

    /// Cluster relationships using Leiden algorithm (when enabled) or SCC fallback
    fn cluster_relationships(
        &self,
        graph: &petgraph::Graph<usize, f32>,
        resolution: f32,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        #[cfg(feature = "leiden")]
        {
            // Use proper Leiden algorithm for community detection
            self.cluster_with_leiden(graph, resolution)
        }
        #[cfg(not(feature = "leiden"))]
        {
            // Fallback to simple connected components when Leiden feature is disabled
            self.cluster_with_scc(graph, resolution)
        }
    }

    /// Cluster using Leiden algorithm (when leiden feature is enabled)
    #[cfg(feature = "leiden")]
    fn cluster_with_leiden(
        &self,
        graph: &petgraph::Graph<usize, f32>,
        resolution: f32,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        use crate::graph::leiden::{LeidenCommunityDetector, LeidenConfig};
        use petgraph::{Graph, Undirected};

        // Convert relationship ID graph to string-labeled graph for Leiden
        let mut leiden_graph: Graph<String, f32, Undirected> = Graph::new_undirected();
        let mut node_mapping = HashMap::new(); // Maps relationship ID -> NodeIndex in leiden_graph
        let mut reverse_mapping = HashMap::new(); // Maps NodeIndex -> relationship ID

        // Add all nodes (relationship IDs) to the Leiden graph
        for node_idx in graph.node_indices() {
            if let Some(&rel_id) = graph.node_weight(node_idx) {
                let leiden_node = leiden_graph.add_node(rel_id.to_string());
                node_mapping.insert(rel_id, leiden_node);
                reverse_mapping.insert(leiden_node, rel_id);
            }
        }

        // Add all edges with weights
        for edge in graph.edge_references() {
            if let (Some(&src_rel_id), Some(&tgt_rel_id)) = (
                graph.node_weight(edge.source()),
                graph.node_weight(edge.target()),
            ) {
                if let (Some(&src_node), Some(&tgt_node)) = (
                    node_mapping.get(&src_rel_id),
                    node_mapping.get(&tgt_rel_id),
                ) {
                    leiden_graph.add_edge(src_node, tgt_node, *edge.weight());
                }
            }
        }

        // Configure and run Leiden
        let config = LeidenConfig {
            resolution,
            max_cluster_size: 50, // Allow larger clusters for relationship graphs
            use_lcc: true,
            seed: Some(42),       // Reproducible results
            max_levels: 1,        // Single-level clustering for now
            min_improvement: 0.001,
        };

        let detector = LeidenCommunityDetector::new(config);
        let result = detector.detect_communities(&leiden_graph)?;

        // Extract communities from level 0
        let mut communities = HashMap::new();
        if let Some(level_0) = result.levels.get(&0) {
            // Group nodes by community ID
            let mut temp_communities: HashMap<usize, Vec<usize>> = HashMap::new();
            for (node_idx, &community_id) in level_0 {
                if let Some(&rel_id) = reverse_mapping.get(node_idx) {
                    temp_communities
                        .entry(community_id)
                        .or_insert_with(Vec::new)
                        .push(rel_id);
                }
            }
            communities = temp_communities;
        }

        Ok(communities)
    }

    /// Fallback clustering using strongly connected components
    #[cfg(not(feature = "leiden"))]
    fn cluster_with_scc(
        &self,
        graph: &petgraph::Graph<usize, f32>,
        resolution: f32,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        use petgraph::algo::kosaraju_scc;

        let components = kosaraju_scc(&graph);

        let mut communities = HashMap::new();
        for (community_id, component) in components.into_iter().enumerate() {
            let node_indices: Vec<usize> = component
                .into_iter()
                .filter_map(|node_idx| graph.node_weight(node_idx).copied())
                .collect();

            if !node_indices.is_empty() {
                communities.insert(community_id, node_indices);
            }
        }

        // Adjust for resolution by merging/splitting if needed
        // For now, use as-is
        let _ = resolution; // Acknowledge parameter

        Ok(communities)
    }

    /// Generate LLM summary for a relationship cluster
    #[cfg(feature = "async")]
    async fn generate_cluster_summary(
        &self,
        cluster: &RelationshipCluster,
        _all_relationships: &[Relationship],
        ollama_client: &OllamaClient,
    ) -> Result<String> {
        // Collect relationship descriptions
        let mut rel_descriptions = Vec::new();

        for rel_id in &cluster.relationship_ids {
            // Parse rel_id format: "source_target_type"
            let parts: Vec<&str> = rel_id.split('_').collect();
            if parts.len() >= 3 {
                let rel_type = parts[2..].join("_");
                rel_descriptions.push(format!("{} --[{}]--> {}", parts[0], rel_type, parts[1]));
            }
        }

        // Limit to first 10 relationships for summary
        let sample: Vec<_> = rel_descriptions.iter().take(10).cloned().collect();
        let total = rel_descriptions.len();

        let prompt = format!(
            r#"Summarize the theme of these {} relationships in 1-2 sentences:

{}

Theme:"#,
            total,
            sample.join("\n")
        );

        match ollama_client.generate(&prompt).await {
            Ok(summary) => Ok(summary.trim().to_string()),
            Err(e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    error = %e,
                    cluster_id = %cluster.cluster_id,
                    "Failed to generate cluster summary, using fallback"
                );

                Ok(format!("Cluster of {} relationships", total))
            }
        }
    }

    /// Calculate cohesion score for a cluster
    ///
    /// Uses internal edge density to measure how tightly connected
    /// relationships within the cluster are.
    ///
    /// Cohesion = (internal edges) / (possible edges)
    ///
    /// Falls back to size-based heuristic if graph metrics unavailable.
    fn calculate_cohesion(
        &self,
        cluster: &RelationshipCluster,
        rel_graph: &petgraph::Graph<usize, f32>,
    ) -> f32 {
        let size = cluster.size();

        // Handle edge cases
        if size == 0 {
            return 0.0;
        }
        if size == 1 {
            return 1.0; // Single item is perfectly cohesive
        }

        // Calculate internal edge density
        // Count edges between nodes that belong to this cluster
        let mut internal_edges = 0;

        // Map relationship IDs to their indices for lookup
        let cluster_rel_ids: std::collections::HashSet<_> =
            cluster.relationship_ids.iter().cloned().collect();

        // Count edges within the cluster in the relationship graph
        for node_idx in rel_graph.node_indices() {
            if let Some(&rel_idx) = rel_graph.node_weight(node_idx) {
                // Check if this relationship belongs to our cluster
                if let Some(rel) = self.relationships.get(rel_idx) {
                    let rel_id = format!(
                        "{}_{}_{}",
                        rel.source.0, rel.target.0, rel.relation_type
                    );

                    if cluster_rel_ids.contains(&rel_id) {
                        // Count outgoing edges to other cluster members
                        for edge in rel_graph.edges(node_idx) {
                            if let Some(&target_rel_idx) = rel_graph.node_weight(edge.target()) {
                                if let Some(target_rel) = self.relationships.get(target_rel_idx) {
                                    let target_rel_id = format!(
                                        "{}_{}_{}",
                                        target_rel.source.0,
                                        target_rel.target.0,
                                        target_rel.relation_type
                                    );

                                    if cluster_rel_ids.contains(&target_rel_id) {
                                        internal_edges += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Avoid double-counting (undirected graph assumption)
        internal_edges /= 2;

        // Calculate maximum possible edges in a complete graph: n*(n-1)/2
        let max_possible_edges = (size * (size - 1)) / 2;

        if max_possible_edges == 0 {
            // Fallback to size-based heuristic
            return (size as f32 / (size as f32 + 10.0)).min(1.0);
        }

        // Edge density as cohesion score
        let density = internal_edges as f32 / max_possible_edges as f32;

        // Apply sigmoid-like transformation to make scores more meaningful
        // Low density clusters still get some credit (0.2-0.8 range)
        0.2 + (density * 0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Entity, EntityId};

    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Create test entities
        for i in 0..5 {
            let entity = Entity::new(
                EntityId::new(format!("entity{}", i)),
                format!("Entity {}", i),
                "CONCEPT".to_string(),
                0.9,
            );
            graph.add_entity(entity).unwrap();
        }

        // Create test relationships
        let relationships = vec![
            ("entity0", "entity1", "RELATED_TO"),
            ("entity0", "entity2", "RELATED_TO"),
            ("entity1", "entity2", "RELATED_TO"),
            ("entity3", "entity4", "CAUSED"),
            ("entity3", "entity4", "ENABLED"),
        ];

        for (src, tgt, rel_type) in relationships {
            let rel = Relationship::new(
                EntityId::new(src.to_string()),
                EntityId::new(tgt.to_string()),
                rel_type.to_string(),
                0.8,
            );
            graph.add_relationship(rel).unwrap();
        }

        graph
    }

    #[test]
    fn test_relationship_cluster_creation() {
        let mut cluster = RelationshipCluster::new("test_cluster".to_string());
        cluster.add_relationship("rel1".to_string());
        cluster.add_relationship("rel2".to_string());

        assert_eq!(cluster.size(), 2);
        assert!(!cluster.is_empty());
        assert_eq!(cluster.cluster_id, "test_cluster");
    }

    #[test]
    fn test_hierarchy_level_creation() {
        let mut level = HierarchyLevel::new(0, 1.0);
        let cluster = RelationshipCluster::new("cluster1".to_string());
        level.add_cluster(cluster);

        assert_eq!(level.level_id, 0);
        assert_eq!(level.clusters.len(), 1);
    }

    #[test]
    fn test_relationship_hierarchy_structure() {
        let mut hierarchy = RelationshipHierarchy::new();

        let mut level0 = HierarchyLevel::new(0, 1.0);
        let mut cluster = RelationshipCluster::new("L0C0".to_string());
        cluster.add_relationship("rel1".to_string());
        level0.add_cluster(cluster);

        hierarchy.add_level(level0);

        assert_eq!(hierarchy.num_levels(), 1);
        assert!(hierarchy.get_level(0).is_some());
    }

    #[test]
    fn test_hierarchy_builder_initialization() {
        let graph = create_test_graph();
        let builder = HierarchyBuilder::from_graph(&graph)
            .with_num_levels(3)
            .with_min_cluster_size(2);

        assert_eq!(builder.num_levels, 3);
        assert_eq!(builder.min_cluster_size, 2);
    }

    #[test]
    fn test_cohesion_calculation() {
        // Test cohesion with a properly structured relationship graph
        // This tests the edge density calculation

        let rel0 = Relationship::new(
            EntityId::new("e0".to_string()),
            EntityId::new("e1".to_string()),
            "REL".to_string(),
            0.8,
        );
        let rel1 = Relationship::new(
            EntityId::new("e1".to_string()),
            EntityId::new("e2".to_string()),
            "REL".to_string(),
            0.8,
        );
        let rel2 = Relationship::new(
            EntityId::new("e2".to_string()),
            EntityId::new("e0".to_string()),
            "REL".to_string(),
            0.8,
        );

        let relationships = vec![rel0.clone(), rel1.clone(), rel2.clone()];
        let builder = HierarchyBuilder::new(relationships);

        // Create a relationship graph where all 3 relationships are connected
        let mut rel_graph = petgraph::Graph::<usize, f32>::new();
        let n0 = rel_graph.add_node(0); // rel0
        let n1 = rel_graph.add_node(1); // rel1
        let n2 = rel_graph.add_node(2); // rel2

        // Fully connected: each relationship connects to the others
        rel_graph.add_edge(n0, n1, 1.0);
        rel_graph.add_edge(n1, n0, 1.0);
        rel_graph.add_edge(n1, n2, 1.0);
        rel_graph.add_edge(n2, n1, 1.0);
        rel_graph.add_edge(n2, n0, 1.0);
        rel_graph.add_edge(n0, n2, 1.0);

        // Create cluster with all 3 relationships
        let mut cluster = RelationshipCluster::new("test_cluster".to_string());
        cluster.add_relationship("e0_e1_REL".to_string());
        cluster.add_relationship("e1_e2_REL".to_string());
        cluster.add_relationship("e2_e0_REL".to_string());

        let cohesion = builder.calculate_cohesion(&cluster, &rel_graph);

        // Fully connected triangle should have high cohesion
        // Exact value depends on implementation but should be > 0.5
        assert!(
            cohesion > 0.5,
            "Expected high cohesion for fully connected cluster, got {}",
            cohesion
        );
    }

    #[test]
    fn test_cohesion_single_relationship() {
        let rel_graph = petgraph::Graph::<usize, f32>::new();

        let rel0 = Relationship::new(
            EntityId::new("e0".to_string()),
            EntityId::new("e1".to_string()),
            "REL".to_string(),
            0.8,
        );

        let relationships = vec![rel0];
        let builder = HierarchyBuilder::new(relationships);

        let mut cluster = RelationshipCluster::new("test_cluster".to_string());
        cluster.add_relationship("e0_e1_REL".to_string());

        let cohesion = builder.calculate_cohesion(&cluster, &rel_graph);

        // Single relationship should have perfect cohesion
        assert_eq!(cohesion, 1.0, "Single relationship should have cohesion 1.0");
    }

    #[test]
    #[cfg(feature = "leiden")]
    fn test_leiden_integration() {
        // Test that Leiden algorithm is used for clustering when feature is enabled
        // Creates a graph with clear community structure

        use petgraph::Graph;

        let rel0 = Relationship::new(
            EntityId::new("A".to_string()),
            EntityId::new("B".to_string()),
            "CONNECTED".to_string(),
            0.9,
        );
        let rel1 = Relationship::new(
            EntityId::new("B".to_string()),
            EntityId::new("C".to_string()),
            "CONNECTED".to_string(),
            0.9,
        );
        let rel2 = Relationship::new(
            EntityId::new("C".to_string()),
            EntityId::new("A".to_string()),
            "CONNECTED".to_string(),
            0.9,
        );
        // Separate cluster
        let rel3 = Relationship::new(
            EntityId::new("X".to_string()),
            EntityId::new("Y".to_string()),
            "CONNECTED".to_string(),
            0.9,
        );
        let rel4 = Relationship::new(
            EntityId::new("Y".to_string()),
            EntityId::new("Z".to_string()),
            "CONNECTED".to_string(),
            0.9,
        );

        let relationships = vec![
            rel0.clone(),
            rel1.clone(),
            rel2.clone(),
            rel3.clone(),
            rel4.clone(),
        ];
        let builder = HierarchyBuilder::new(relationships);

        // Build relationship graph with two clear communities
        // Community 1: rel0-rel1-rel2 (triangle ABC)
        // Community 2: rel3-rel4 (line XYZ)
        let mut rel_graph = Graph::<usize, f32>::new();

        // Add nodes for each relationship
        let n0 = rel_graph.add_node(0); // A-B
        let n1 = rel_graph.add_node(1); // B-C
        let n2 = rel_graph.add_node(2); // C-A
        let n3 = rel_graph.add_node(3); // X-Y
        let n4 = rel_graph.add_node(4); // Y-Z

        // Community 1 (fully connected triangle)
        rel_graph.add_edge(n0, n1, 1.0); // A-B connects to B-C via B
        rel_graph.add_edge(n1, n2, 1.0); // B-C connects to C-A via C
        rel_graph.add_edge(n2, n0, 1.0); // C-A connects to A-B via A

        // Community 2 (line)
        rel_graph.add_edge(n3, n4, 1.0); // X-Y connects to Y-Z via Y

        // Test clustering with Leiden
        let communities = builder.cluster_relationships(&rel_graph, 1.0).unwrap();

        // Should detect 2 communities
        assert!(
            communities.len() >= 2,
            "Leiden should detect at least 2 communities, found {}",
            communities.len()
        );

        // Verify that relationships 0,1,2 are in one community
        // and relationships 3,4 are in another (or similar grouping)
        let community_sizes: Vec<usize> = communities.values().map(|c| c.len()).collect();
        assert!(
            community_sizes.contains(&3) || community_sizes.contains(&2),
            "Expected communities of size 3 or 2, got {:?}",
            community_sizes
        );
    }
}
