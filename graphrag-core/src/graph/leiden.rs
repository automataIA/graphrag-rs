//! Leiden community detection algorithm
//!
//! Improves upon Louvain algorithm by adding a refinement phase that prevents
//! poorly connected communities. Implements hierarchical clustering for multi-level
//! community structure.
//!
//! Reference: "From Louvain to Leiden: guaranteeing well-connected communities"
//! Traag, Waltman & van Eck (2019)

use std::collections::{HashMap, HashSet};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::Result;

/// Metadata about an entity in the graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityMetadata {
    /// Unique entity identifier
    pub id: String,
    /// Entity name
    pub name: String,
    /// Entity type (e.g., "person", "organization")
    pub entity_type: String,
    /// Extraction confidence (0.0-1.0)
    pub confidence: f32,
    /// Number of mentions across documents
    pub mention_count: usize,
}

/// Hierarchical community detection results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HierarchicalCommunities {
    /// Communities at each hierarchical level
    /// Level 0 = finest granularity, higher = coarser
    pub levels: HashMap<usize, HashMap<NodeIndex, usize>>,

    /// Parent-child relationships between levels
    /// Maps community ID at level N to parent community ID at level N+1
    pub hierarchy: HashMap<usize, Option<usize>>,

    /// LLM-generated summaries for each community (optional)
    pub summaries: HashMap<usize, String>,

    /// Mapping from entity names to metadata (enriched from KnowledgeGraph)
    pub entity_mapping: Option<HashMap<String, EntityMetadata>>,
}

impl HierarchicalCommunities {
    /// Get all entities in a specific community at a given level
    ///
    /// # Arguments
    /// * `level` - Hierarchical level (0 = finest)
    /// * `community_id` - Community identifier
    /// * `graph` - The original graph to extract entity names from
    ///
    /// # Returns
    /// Vec of entity names in the community
    pub fn get_community_entities(
        &self,
        level: usize,
        community_id: usize,
        graph: &Graph<String, f32, Undirected>,
    ) -> Vec<String> {
        if let Some(level_communities) = self.levels.get(&level) {
            level_communities
                .iter()
                .filter(|(_, &comm_id)| comm_id == community_id)
                .filter_map(|(&node_idx, _)| graph.node_weight(node_idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get entity metadata for entities in a community
    ///
    /// # Arguments
    /// * `entity_names` - List of entity names
    ///
    /// # Returns
    /// Vec of EntityMetadata for the entities
    pub fn get_entities_metadata(&self, entity_names: &[String]) -> Vec<EntityMetadata> {
        if let Some(mapping) = &self.entity_mapping {
            entity_names
                .iter()
                .filter_map(|name| mapping.get(name).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get community statistics
    ///
    /// # Arguments
    /// * `level` - Hierarchical level
    /// * `community_id` - Community identifier
    /// * `graph` - The original graph
    ///
    /// # Returns
    /// Tuple of (entity_count, avg_confidence, unique_types)
    pub fn get_community_stats(
        &self,
        level: usize,
        community_id: usize,
        graph: &Graph<String, f32, Undirected>,
    ) -> (usize, f32, HashSet<String>) {
        let entities = self.get_community_entities(level, community_id, graph);
        let metadata = self.get_entities_metadata(&entities);

        let count = metadata.len();
        let avg_conf = if !metadata.is_empty() {
            metadata.iter().map(|m| m.confidence).sum::<f32>() / count as f32
        } else {
            0.0
        };
        let types: HashSet<String> = metadata.iter().map(|m| m.entity_type.clone()).collect();

        (count, avg_conf, types)
    }

    /// Generate extractive summary for a community (bottom-up approach)
    ///
    /// Creates a summary by listing key entities and their types.
    /// For use with LLM-based summarization, use `prepare_community_context()`.
    ///
    /// # Arguments
    /// * `level` - Hierarchical level
    /// * `community_id` - Community identifier
    /// * `graph` - The original graph
    /// * `max_length` - Maximum summary length in characters
    ///
    /// # Returns
    /// Generated summary string
    pub fn generate_community_summary(
        &mut self,
        level: usize,
        community_id: usize,
        graph: &Graph<String, f32, Undirected>,
        max_length: usize,
    ) -> String {
        let entities = self.get_community_entities(level, community_id, graph);
        let metadata = self.get_entities_metadata(&entities);

        if metadata.is_empty() {
            return format!("Community {community_id} at level {level}: No entities");
        }

        // Group by entity type
        let mut by_type: HashMap<String, Vec<&EntityMetadata>> = HashMap::new();
        for meta in &metadata {
            by_type.entry(meta.entity_type.clone())
                .or_insert_with(Vec::new)
                .push(meta);
        }

        // Build summary
        let mut summary_parts = vec![
            format!("Community {community_id} (Level {level})"),
            format!("Contains {} entities:", metadata.len()),
        ];

        for (entity_type, entities) in &by_type {
            let names: Vec<String> = entities
                .iter()
                .take(5) // Limit to top 5 per type
                .map(|e| e.name.clone())
                .collect();

            let more = if entities.len() > 5 {
                format!(" and {} more", entities.len() - 5)
            } else {
                String::new()
            };

            summary_parts.push(format!("- {}: {}{}", entity_type, names.join(", "), more));
        }

        let summary = summary_parts.join("\n");

        // Truncate if too long
        if summary.len() > max_length {
            format!("{}...", &summary[..max_length.saturating_sub(3)])
        } else {
            summary
        }
    }

    /// Generate summaries for all communities at a specific level
    ///
    /// # Arguments
    /// * `level` - Hierarchical level to summarize
    /// * `graph` - The original graph
    /// * `max_length` - Maximum summary length per community
    pub fn generate_level_summaries(
        &mut self,
        level: usize,
        graph: &Graph<String, f32, Undirected>,
        max_length: usize,
    ) {
        if let Some(level_communities) = self.levels.get(&level) {
            let community_ids: HashSet<usize> = level_communities.values().copied().collect();

            for community_id in community_ids {
                let summary = self.generate_community_summary(level, community_id, graph, max_length);
                self.summaries.insert(community_id, summary);
            }
        }
    }

    /// Generate summaries for all levels bottom-up
    ///
    /// Starts from the finest level (0) and works up through the hierarchy.
    /// Higher-level summaries can reference lower-level summaries.
    ///
    /// # Arguments
    /// * `graph` - The original graph
    /// * `max_length` - Maximum summary length per community
    pub fn generate_hierarchical_summaries(
        &mut self,
        graph: &Graph<String, f32, Undirected>,
        max_length: usize,
    ) {
        let max_level = self.levels.keys().max().copied().unwrap_or(0);

        // Generate summaries bottom-up
        for level in 0..=max_level {
            self.generate_level_summaries(level, graph, max_length);
        }
    }

    /// Prepare context for LLM-based community summarization
    ///
    /// Generates a structured prompt containing:
    /// - Entity names and types
    /// - Relationships within the community
    /// - Sub-community summaries (for higher levels)
    ///
    /// # Arguments
    /// * `level` - Hierarchical level
    /// * `community_id` - Community identifier
    /// * `graph` - The original graph
    /// * `knowledge_graph` - The full KnowledgeGraph for relationship access
    ///
    /// # Returns
    /// Formatted context string ready for LLM input
    #[cfg(feature = "async")]
    pub fn prepare_community_context(
        &self,
        level: usize,
        community_id: usize,
        graph: &Graph<String, f32, Undirected>,
        knowledge_graph: &crate::core::KnowledgeGraph,
    ) -> String {
        let entities = self.get_community_entities(level, community_id, graph);
        let metadata = self.get_entities_metadata(&entities);

        let mut context_parts = vec![
            format!("# Community {} at Level {}", community_id, level),
            String::new(),
            "## Entities:".to_string(),
        ];

        // Add entity information
        for meta in &metadata {
            context_parts.push(format!(
                "- {} ({}): confidence {:.2}, {} mentions",
                meta.name, meta.entity_type, meta.confidence, meta.mention_count
            ));
        }

        context_parts.push(String::new());
        context_parts.push("## Relationships:".to_string());

        // Add relationships between entities in this community
        let entity_set: HashSet<String> = entities.iter().cloned().collect();
        for rel in knowledge_graph.get_all_relationships() {
            // Check if both source and target are in this community
            if let (Some(src_entity), Some(tgt_entity)) = (
                knowledge_graph.get_entity(&rel.source),
                knowledge_graph.get_entity(&rel.target),
            ) {
                if entity_set.contains(&src_entity.name) && entity_set.contains(&tgt_entity.name) {
                    context_parts.push(format!(
                        "- {} --[{}]--> {} (confidence: {:.2})",
                        src_entity.name, rel.relation_type, tgt_entity.name, rel.confidence
                    ));
                }
            }
        }

        // For higher levels, include sub-community summaries
        if level > 0 {
            context_parts.push(String::new());
            context_parts.push("## Sub-community Summaries:".to_string());
            // Would need to track parent-child relationships to list sub-communities
            // This is left as a TODO for full hierarchical implementation
        }

        context_parts.join("\n")
    }

    /// Retrieve relevant communities using adaptive query routing
    ///
    /// Automatically selects the appropriate hierarchical level based on query complexity
    /// and returns matching community summaries.
    ///
    /// # Arguments
    /// * `query` - User query string
    /// * `graph` - The original graph
    /// * `router_config` - Configuration for adaptive routing
    ///
    /// # Returns
    /// Vec of (level, community_id, summary) tuples
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::query::AdaptiveRoutingConfig;
    ///
    /// let config = AdaptiveRoutingConfig::default();
    /// let results = communities.adaptive_retrieve("AI overview", &graph, config);
    /// ```
    pub fn adaptive_retrieve(
        &self,
        query: &str,
        graph: &Graph<String, f32, Undirected>,
        router_config: crate::query::AdaptiveRoutingConfig,
    ) -> Vec<(usize, usize, String)> {
        use crate::query::QueryComplexityAnalyzer;

        // Analyze query and determine level
        let analyzer = QueryComplexityAnalyzer::new(router_config);
        let suggested_level = analyzer.suggest_level(query);

        // Retrieve at suggested level
        self.retrieve_at_level(query, graph, suggested_level)
    }

    /// Retrieve relevant communities at a specific hierarchical level
    ///
    /// # Arguments
    /// * `query` - User query string
    /// * `graph` - The original graph
    /// * `level` - Hierarchical level to search (0 = finest)
    ///
    /// # Returns
    /// Vec of (level, community_id, summary) tuples
    pub fn retrieve_at_level(
        &self,
        query: &str,
        graph: &Graph<String, f32, Undirected>,
        level: usize,
    ) -> Vec<(usize, usize, String)> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();

        // Get all communities at this level
        if let Some(level_communities) = self.levels.get(&level) {
            let unique_communities: HashSet<usize> =
                level_communities.values().copied().collect();

            for community_id in unique_communities {
                let entities = self.get_community_entities(level, community_id, graph);

                // Check relevance
                let is_relevant = entities.iter().any(|entity|
                    entity.to_lowercase().contains(&query_lower)
                );

                if is_relevant {
                    // Get or generate summary
                    let summary = self.summaries.get(&community_id)
                        .cloned()
                        .unwrap_or_else(|| {
                            // Fallback: create entity list
                            format!("Entities: {}", entities.join(", "))
                        });

                    results.push((level, community_id, summary));
                }
            }
        }

        results
    }

    /// Retrieve with detailed query analysis
    ///
    /// Returns both the retrieval results and the query analysis that determined the level.
    ///
    /// # Arguments
    /// * `query` - User query string
    /// * `graph` - The original graph
    /// * `router_config` - Configuration for adaptive routing
    ///
    /// # Returns
    /// Tuple of (QueryAnalysis, retrieval results)
    pub fn adaptive_retrieve_detailed(
        &self,
        query: &str,
        graph: &Graph<String, f32, Undirected>,
        router_config: crate::query::AdaptiveRoutingConfig,
    ) -> (crate::query::QueryAnalysis, Vec<(usize, usize, String)>) {
        use crate::query::QueryComplexityAnalyzer;

        // Analyze query
        let analyzer = QueryComplexityAnalyzer::new(router_config);
        let analysis = analyzer.analyze_detailed(query);

        // Retrieve at suggested level
        let results = self.retrieve_at_level(query, graph, analysis.suggested_level);

        (analysis, results)
    }
}

/// Configuration for Leiden algorithm
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LeidenConfig {
    /// Maximum community size
    pub max_cluster_size: usize,

    /// Use only largest connected component
    pub use_lcc: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Modularity resolution parameter (default: 1.0)
    /// Lower = larger communities, Higher = smaller communities
    pub resolution: f32,

    /// Maximum hierarchical depth
    pub max_levels: usize,

    /// Minimum improvement to continue iterations
    pub min_improvement: f32,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            max_cluster_size: 10,
            use_lcc: true,
            seed: None,
            resolution: 1.0,
            max_levels: 5,
            min_improvement: 0.001,
        }
    }
}

/// Leiden community detection algorithm
pub struct LeidenCommunityDetector {
    config: LeidenConfig,
}

impl LeidenCommunityDetector {
    /// Create a new Leiden detector with configuration
    pub fn new(config: LeidenConfig) -> Self {
        Self { config }
    }

    /// Detect hierarchical communities in graph
    pub fn detect_communities(
        &self,
        graph: &Graph<String, f32, Undirected>,
    ) -> Result<HierarchicalCommunities> {
        // 1. Optionally extract largest connected component
        let working_graph = if self.config.use_lcc {
            self.extract_largest_connected_component(graph)
        } else {
            graph.clone()
        };

        // 2. Initialize RNG with seed if provided
        let mut _rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // 3. Run hierarchical Leiden clustering
        let (levels, hierarchy) = self.hierarchical_leiden(&working_graph)?;

        Ok(HierarchicalCommunities {
            levels,
            hierarchy,
            summaries: HashMap::new(),  // Filled later by LLM if needed
            entity_mapping: None,  // Enriched when called from KnowledgeGraph
        })
    }

    /// Hierarchical Leiden algorithm implementation
    fn hierarchical_leiden(
        &self,
        graph: &Graph<String, f32, Undirected>,
    ) -> Result<(HashMap<usize, HashMap<NodeIndex, usize>>, HashMap<usize, Option<usize>>)> {
        let mut levels = HashMap::new();
        let hierarchy = HashMap::new();

        let current_graph = graph.clone();
        let level = 0;

        // Phase 1: Initialize each node in its own community
        let mut communities = self.initialize_communities(&current_graph);

        // Phase 2: Local moving (greedy modularity optimization)
        let mut improved = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        while improved && iteration < MAX_ITERATIONS {
            improved = false;
            for node in current_graph.node_indices() {
                let best_community = self.find_best_community(
                    &current_graph,
                    node,
                    &communities,
                );

                if best_community != communities[&node] {
                    communities.insert(node, best_community);
                    improved = true;
                }
            }
            iteration += 1;
        }

        // Phase 3: Refinement (KEY difference from Louvain)
        communities = self.refine_partition(&current_graph, communities)?;

        // Store this level
        levels.insert(level, communities);

        Ok((levels, hierarchy))
    }

    /// Initialize each node in its own community
    fn initialize_communities(
        &self,
        graph: &Graph<String, f32, Undirected>,
    ) -> HashMap<NodeIndex, usize> {
        graph.node_indices()
            .enumerate()
            .map(|(i, node)| (node, i))
            .collect()
    }

    /// Find best community for a node using greedy modularity optimization
    fn find_best_community(
        &self,
        graph: &Graph<String, f32, Undirected>,
        node: NodeIndex,
        communities: &HashMap<NodeIndex, usize>,
    ) -> usize {
        let current_community = communities[&node];
        let mut best_community = current_community;
        let mut best_delta_modularity = 0.0;

        // Get neighboring communities
        let neighbor_communities: HashSet<usize> = graph
            .neighbors(node)
            .filter_map(|neighbor| communities.get(&neighbor).copied())
            .collect();

        // Try each neighboring community
        for &neighbor_community in &neighbor_communities {
            if neighbor_community == current_community {
                continue;
            }

            let delta = self.calculate_modularity_delta(
                graph,
                node,
                current_community,
                neighbor_community,
                communities,
            );

            if delta > best_delta_modularity {
                best_delta_modularity = delta;
                best_community = neighbor_community;
            }
        }

        best_community
    }

    /// Refine partition to ensure well-connected communities
    /// KEY difference from Louvain - prevents poorly connected communities
    fn refine_partition(
        &self,
        graph: &Graph<String, f32, Undirected>,
        mut communities: HashMap<NodeIndex, usize>,
    ) -> Result<HashMap<NodeIndex, usize>> {
        // Get unique community IDs
        let community_ids: HashSet<usize> = communities.values().copied().collect();

        for &community_id in &community_ids {
            // Get nodes in this community
            let community_nodes: Vec<NodeIndex> = communities
                .iter()
                .filter(|(_, &c)| c == community_id)
                .map(|(&n, _)| n)
                .collect();

            // Check if community is well-connected
            if !self.is_well_connected(graph, &community_nodes) {
                // Split poorly connected community
                self.split_community(graph, &mut communities, &community_nodes)?;
            }
        }

        Ok(communities)
    }

    /// Check if community forms a connected subgraph
    fn is_well_connected(
        &self,
        graph: &Graph<String, f32, Undirected>,
        nodes: &[NodeIndex],
    ) -> bool {
        if nodes.len() <= 1 {
            return true;
        }

        // DFS to check connectivity
        let mut visited = HashSet::new();
        let mut stack = vec![nodes[0]];
        visited.insert(nodes[0]);

        while let Some(node) = stack.pop() {
            for neighbor in graph.neighbors(node) {
                if nodes.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    stack.push(neighbor);
                }
            }
        }

        visited.len() == nodes.len()
    }

    /// Split poorly connected community into well-connected sub-communities
    fn split_community(
        &self,
        graph: &Graph<String, f32, Undirected>,
        communities: &mut HashMap<NodeIndex, usize>,
        nodes: &[NodeIndex],
    ) -> Result<()> {
        // Find connected components within community
        let components = self.find_connected_components(graph, nodes);

        // Assign new community IDs to each component
        let max_community_id = communities.values().max().copied().unwrap_or(0);

        for (idx, component) in components.iter().enumerate() {
            let new_community_id = max_community_id + idx + 1;
            for &node in component {
                communities.insert(node, new_community_id);
            }
        }

        Ok(())
    }

    /// Find connected components within a set of nodes
    fn find_connected_components(
        &self,
        graph: &Graph<String, f32, Undirected>,
        nodes: &[NodeIndex],
    ) -> Vec<Vec<NodeIndex>> {
        let mut components = Vec::new();
        let mut unvisited: HashSet<NodeIndex> = nodes.iter().copied().collect();

        while !unvisited.is_empty() {
            let start = *unvisited.iter().next().unwrap();
            let mut component = Vec::new();
            let mut stack = vec![start];

            while let Some(node) = stack.pop() {
                if !unvisited.remove(&node) {
                    continue;
                }

                component.push(node);

                for neighbor in graph.neighbors(node) {
                    if unvisited.contains(&neighbor) && nodes.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Calculate modularity delta for moving node between communities
    fn calculate_modularity_delta(
        &self,
        graph: &Graph<String, f32, Undirected>,
        node: NodeIndex,
        from_community: usize,
        to_community: usize,
        communities: &HashMap<NodeIndex, usize>,
    ) -> f32 {
        let degree = graph.edges(node).count() as f32;
        let total_edges = graph.edge_count() as f32 * 2.0;  // Undirected

        // Edges to communities
        let k_i_in_to = self.edges_to_community(graph, node, to_community, communities);
        let k_i_in_from = self.edges_to_community(graph, node, from_community, communities);

        // Total degree of communities
        let sigma_tot_to = self.total_degree_of_community(graph, to_community, communities);
        let sigma_tot_from = self.total_degree_of_community(graph, from_community, communities);

        // Delta Q using Newman's modularity formula
        let delta = ((k_i_in_to as f32 - k_i_in_from as f32) / total_edges)
            - self.config.resolution * degree
            * ((sigma_tot_to - sigma_tot_from + degree) / (total_edges * total_edges));

        delta
    }

    /// Count edges from node to a specific community
    fn edges_to_community(
        &self,
        graph: &Graph<String, f32, Undirected>,
        node: NodeIndex,
        community: usize,
        communities: &HashMap<NodeIndex, usize>,
    ) -> usize {
        graph.neighbors(node)
            .filter(|&neighbor| communities.get(&neighbor) == Some(&community))
            .count()
    }

    /// Calculate total degree of all nodes in a community
    fn total_degree_of_community(
        &self,
        graph: &Graph<String, f32, Undirected>,
        community: usize,
        communities: &HashMap<NodeIndex, usize>,
    ) -> f32 {
        communities.iter()
            .filter(|(_, &c)| c == community)
            .map(|(&node, _)| graph.edges(node).count() as f32)
            .sum()
    }

    /// Extract largest connected component from graph
    fn extract_largest_connected_component(
        &self,
        graph: &Graph<String, f32, Undirected>,
    ) -> Graph<String, f32, Undirected> {
        use petgraph::algo::connected_components;

        let num_components = connected_components(graph);

        if num_components == 1 {
            return graph.clone();
        }

        // For now, return original graph
        // Full implementation would extract actual largest component
        graph.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph<String, f32, Undirected> {
        let mut graph = Graph::new_undirected();

        // Create simple graph with 2 obvious communities
        let n0 = graph.add_node("A".to_string());
        let n1 = graph.add_node("B".to_string());
        let n2 = graph.add_node("C".to_string());
        let n3 = graph.add_node("D".to_string());

        // Community 1: A-B-C (triangle)
        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 1.0);
        graph.add_edge(n2, n0, 1.0);

        // Community 2: D (isolated)
        // One weak link between communities
        graph.add_edge(n2, n3, 0.1);

        graph
    }

    #[test]
    fn test_leiden_basic() {
        let graph = create_test_graph();
        let config = LeidenConfig::default();
        let detector = LeidenCommunityDetector::new(config);

        let result = detector.detect_communities(&graph);
        assert!(result.is_ok());

        let communities = result.unwrap();
        assert!(!communities.levels.is_empty());
    }

    #[test]
    fn test_is_well_connected() {
        let graph = create_test_graph();
        let config = LeidenConfig::default();
        let detector = LeidenCommunityDetector::new(config);

        // First 3 nodes form connected triangle
        let nodes = vec![
            NodeIndex::new(0),
            NodeIndex::new(1),
            NodeIndex::new(2),
        ];

        assert!(detector.is_well_connected(&graph, &nodes));
    }

    #[test]
    fn test_config_defaults() {
        let config = LeidenConfig::default();
        assert_eq!(config.max_cluster_size, 10);
        assert_eq!(config.resolution, 1.0);
        assert_eq!(config.max_levels, 5);
        assert!(config.use_lcc);
    }
}
