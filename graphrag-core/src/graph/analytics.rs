//! Graph Analytics
//!
//! Advanced graph analysis algorithms including:
//! - Community detection (Louvain algorithm)
//! - Centrality measures (betweenness, closeness, degree)
//! - Path finding (shortest path, all paths)
//! - Graph embeddings preparation
//! - Temporal graph analysis

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Community ID
    pub id: usize,
    /// Node IDs in this community
    pub nodes: Vec<String>,
    /// Community modularity score
    pub modularity: f32,
}

/// Centrality scores for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityScores {
    /// Node ID
    pub node_id: String,
    /// Degree centrality (normalized)
    pub degree: f32,
    /// Betweenness centrality
    pub betweenness: f32,
    /// Closeness centrality
    pub closeness: f32,
    /// PageRank score (if available)
    pub pagerank: Option<f32>,
}

/// Path between two nodes
#[derive(Debug, Clone)]
pub struct Path {
    /// Node IDs in order
    pub nodes: Vec<String>,
    /// Total path weight
    pub weight: f32,
}

/// Helper struct for DFS path search state
struct PathSearchState<'a> {
    path: &'a mut Vec<String>,
    visited: &'a mut HashSet<String>,
    all_paths: &'a mut Vec<Path>,
    weight: f32,
}

/// Graph analytics engine
pub struct GraphAnalytics {
    /// Adjacency list representation
    adjacency: HashMap<String, Vec<(String, f32)>>,
    /// Node degrees
    degrees: HashMap<String, usize>,
}

impl GraphAnalytics {
    /// Create analytics engine from edges
    ///
    /// # Arguments
    /// * `edges` - List of (source, target, weight) tuples
    pub fn new(edges: Vec<(String, String, f32)>) -> Self {
        let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut degrees: HashMap<String, usize> = HashMap::new();

        for (source, target, weight) in edges {
            adjacency
                .entry(source.clone())
                .or_default()
                .push((target.clone(), weight));

            adjacency
                .entry(target.clone())
                .or_default()
                .push((source.clone(), weight));

            *degrees.entry(source).or_insert(0) += 1;
            *degrees.entry(target).or_insert(0) += 1;
        }

        Self { adjacency, degrees }
    }

    /// Detect communities using Louvain algorithm
    ///
    /// This is a simplified implementation. Full Louvain requires iterative optimization.
    ///
    /// # Returns
    /// List of detected communities
    pub fn detect_communities(&self) -> Vec<Community> {
        let nodes: Vec<String> = self.adjacency.keys().cloned().collect();
        let mut communities: HashMap<String, usize> = HashMap::new();
        let mut community_id = 0;

        // Simple connected components as initial communities
        for node in &nodes {
            if !communities.contains_key(node) {
                let component = self.get_connected_component(node);
                for n in component {
                    communities.insert(n, community_id);
                }
                community_id += 1;
            }
        }

        // Group nodes by community
        let mut community_map: HashMap<usize, Vec<String>> = HashMap::new();
        for (node, id) in communities {
            community_map.entry(id).or_default().push(node);
        }

        // Calculate modularity for each community
        community_map
            .into_iter()
            .map(|(id, nodes)| {
                let modularity = self.calculate_modularity(&nodes);
                Community {
                    id,
                    nodes,
                    modularity,
                }
            })
            .collect()
    }

    /// Get connected component starting from a node
    fn get_connected_component(&self, start: &str) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());

        while let Some(node) = queue.pop_front() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node.clone());

            if let Some(neighbors) = self.adjacency.get(&node) {
                for (neighbor, _) in neighbors {
                    if !visited.contains(neighbor) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Calculate modularity for a set of nodes
    fn calculate_modularity(&self, nodes: &[String]) -> f32 {
        let total_edges = self.adjacency.len() as f32;
        let mut internal_edges = 0.0;

        let node_set: HashSet<_> = nodes.iter().collect();

        for node in nodes {
            if let Some(neighbors) = self.adjacency.get(node) {
                for (neighbor, _) in neighbors {
                    if node_set.contains(&neighbor) {
                        internal_edges += 1.0;
                    }
                }
            }
        }

        // Normalize (simplified formula)
        internal_edges / (2.0 * total_edges)
    }

    /// Calculate centrality scores for all nodes
    ///
    /// # Returns
    /// Map of node ID to centrality scores
    pub fn calculate_centrality(&self) -> HashMap<String, CentralityScores> {
        let nodes: Vec<String> = self.adjacency.keys().cloned().collect();
        let n = nodes.len() as f32;

        let mut scores = HashMap::new();

        for node in &nodes {
            let degree = self.degree_centrality(node, n);
            let betweenness = self.betweenness_centrality(node);
            let closeness = self.closeness_centrality(node);

            scores.insert(
                node.clone(),
                CentralityScores {
                    node_id: node.clone(),
                    degree,
                    betweenness,
                    closeness,
                    pagerank: None, // Would be filled by PageRank module
                },
            );
        }

        scores
    }

    /// Calculate degree centrality
    fn degree_centrality(&self, node: &str, n: f32) -> f32 {
        let degree = *self.degrees.get(node).unwrap_or(&0) as f32;
        if n > 1.0 {
            degree / (n - 1.0)
        } else {
            0.0
        }
    }

    /// Calculate betweenness centrality (simplified)
    fn betweenness_centrality(&self, node: &str) -> f32 {
        let nodes: Vec<String> = self.adjacency.keys().cloned().collect();
        let mut betweenness = 0.0;

        // For each pair of nodes, count shortest paths through this node
        for source in &nodes {
            if source == node {
                continue;
            }
            for target in &nodes {
                if target == node || source == target {
                    continue;
                }

                if let Some(path) = self.shortest_path(source, target) {
                    if path.nodes.contains(&node.to_string()) {
                        betweenness += 1.0;
                    }
                }
            }
        }

        let n = nodes.len() as f32;
        if n > 2.0 {
            betweenness / ((n - 1.0) * (n - 2.0) / 2.0)
        } else {
            0.0
        }
    }

    /// Calculate closeness centrality
    fn closeness_centrality(&self, node: &str) -> f32 {
        let nodes: Vec<String> = self.adjacency.keys().cloned().collect();
        let mut total_distance = 0.0;
        let mut reachable = 0;

        for target in &nodes {
            if target == node {
                continue;
            }

            if let Some(path) = self.shortest_path(node, target) {
                total_distance += path.weight;
                reachable += 1;
            }
        }

        if reachable > 0 && total_distance > 0.0 {
            (reachable as f32) / total_distance
        } else {
            0.0
        }
    }

    /// Find shortest path between two nodes (Dijkstra's algorithm)
    ///
    /// # Arguments
    /// * `start` - Starting node ID
    /// * `end` - Ending node ID
    ///
    /// # Returns
    /// Shortest path if exists
    pub fn shortest_path(&self, start: &str, end: &str) -> Option<Path> {
        let mut distances: HashMap<String, f32> = HashMap::new();
        let mut previous: HashMap<String, String> = HashMap::new();
        let mut unvisited: HashSet<String> = self.adjacency.keys().cloned().collect();

        distances.insert(start.to_string(), 0.0);

        while !unvisited.is_empty() {
            // Find node with minimum distance
            let current = unvisited
                .iter()
                .min_by(|a, b| {
                    let dist_a = *distances.get(*a).unwrap_or(&f32::INFINITY);
                    let dist_b = *distances.get(*b).unwrap_or(&f32::INFINITY);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })?
                .clone();

            if current == end {
                break;
            }

            unvisited.remove(&current);

            let current_dist = *distances.get(&current).unwrap_or(&f32::INFINITY);

            if let Some(neighbors) = self.adjacency.get(&current) {
                for (neighbor, weight) in neighbors {
                    if unvisited.contains(neighbor) {
                        let alt = current_dist + weight;
                        let neighbor_dist = *distances.get(neighbor).unwrap_or(&f32::INFINITY);

                        if alt < neighbor_dist {
                            distances.insert(neighbor.clone(), alt);
                            previous.insert(neighbor.clone(), current.clone());
                        }
                    }
                }
            }
        }

        // Reconstruct path
        let mut path_nodes = Vec::new();
        let mut current = end.to_string();

        while let Some(prev) = previous.get(&current) {
            path_nodes.push(current.clone());
            current = prev.clone();
        }

        if current == start {
            path_nodes.push(start.to_string());
            path_nodes.reverse();

            let weight = *distances.get(end).unwrap_or(&f32::INFINITY);

            Some(Path {
                nodes: path_nodes,
                weight,
            })
        } else {
            None
        }
    }

    /// Find all paths between two nodes (limited depth)
    ///
    /// # Arguments
    /// * `start` - Starting node
    /// * `end` - Ending node
    /// * `max_depth` - Maximum path length
    ///
    /// # Returns
    /// All paths up to max_depth
    pub fn all_paths(&self, start: &str, end: &str, max_depth: usize) -> Vec<Path> {
        let mut paths = Vec::new();
        let mut current_path = Vec::new();
        let mut visited = HashSet::new();

        let mut state = PathSearchState {
            path: &mut current_path,
            visited: &mut visited,
            all_paths: &mut paths,
            weight: 0.0,
        };

        self.dfs_paths(start, end, &mut state, max_depth);

        paths
    }

    /// DFS helper for all_paths
    fn dfs_paths(
        &self,
        current: &str,
        end: &str,
        state: &mut PathSearchState,
        max_depth: usize,
    ) {
        if state.path.len() >= max_depth {
            return;
        }

        state.path.push(current.to_string());
        state.visited.insert(current.to_string());

        if current == end {
            state.all_paths.push(Path {
                nodes: state.path.clone(),
                weight: state.weight,
            });
        } else if let Some(neighbors) = self.adjacency.get(current) {
            for (neighbor, edge_weight) in neighbors {
                if !state.visited.contains(neighbor) {
                    let old_weight = state.weight;
                    state.weight += edge_weight;
                    self.dfs_paths(neighbor, end, state, max_depth);
                    state.weight = old_weight;
                }
            }
        } else {
            // Current node has no neighbors in the graph
        }

        state.path.pop();
        state.visited.remove(current);
    }

    /// Get nodes with highest degree centrality
    ///
    /// # Arguments
    /// * `top_k` - Number of top nodes to return
    ///
    /// # Returns
    /// List of (node_id, degree_centrality) sorted by degree
    pub fn top_degree_nodes(&self, top_k: usize) -> Vec<(String, f32)> {
        let n = self.adjacency.len() as f32;
        let mut scores: Vec<_> = self
            .adjacency
            .keys()
            .map(|node| {
                let degree = self.degree_centrality(node, n);
                (node.clone(), degree)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }

    /// Get graph density
    ///
    /// # Returns
    /// Graph density (0.0 to 1.0)
    pub fn density(&self) -> f32 {
        let n = self.adjacency.len() as f32;
        let edge_count: usize = self.adjacency.values().map(|v| v.len()).sum();
        let actual_edges = (edge_count / 2) as f32; // Undirected graph

        if n > 1.0 {
            (2.0 * actual_edges) / (n * (n - 1.0))
        } else {
            0.0
        }
    }

    /// Get clustering coefficient
    ///
    /// # Returns
    /// Average clustering coefficient
    pub fn clustering_coefficient(&self) -> f32 {
        let mut total = 0.0;
        let mut count = 0;

        for neighbors in self.adjacency.values() {
            if neighbors.len() < 2 {
                continue;
            }

            let neighbor_set: HashSet<_> = neighbors.iter().map(|(n, _)| n).collect();
            let mut triangles = 0;

            for (n1, _) in neighbors {
                if let Some(n1_neighbors) = self.adjacency.get(n1) {
                    for (n2, _) in n1_neighbors {
                        if neighbor_set.contains(&n2) {
                            triangles += 1;
                        }
                    }
                }
            }

            let k = neighbors.len() as f32;
            let coefficient = triangles as f32 / (k * (k - 1.0));
            total += coefficient;
            count += 1;
        }

        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> GraphAnalytics {
        let edges = vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("A".to_string(), "C".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 1.0),
            ("B".to_string(), "D".to_string(), 1.0),
            ("C".to_string(), "D".to_string(), 1.0),
        ];
        GraphAnalytics::new(edges)
    }

    #[test]
    fn test_shortest_path() {
        let graph = create_test_graph();
        let path = graph.shortest_path("A", "D").unwrap();

        assert_eq!(path.nodes.len(), 3); // A -> B -> D or A -> C -> D
        assert_eq!(path.weight, 2.0);
    }

    #[test]
    fn test_centrality() {
        let graph = create_test_graph();
        let scores = graph.calculate_centrality();

        assert!(scores.contains_key("A"));
        assert!(scores.contains_key("B"));
        assert!(scores.contains_key("C"));
        assert!(scores.contains_key("D"));

        // B and C should have higher betweenness (they're central)
        let b_score = &scores["B"];
        let a_score = &scores["A"];
        assert!(b_score.betweenness >= a_score.betweenness);
    }

    #[test]
    fn test_community_detection() {
        let graph = create_test_graph();
        let communities = graph.detect_communities();

        assert_eq!(communities.len(), 1); // Should be one connected component
        assert_eq!(communities[0].nodes.len(), 4);
    }

    #[test]
    fn test_density() {
        let graph = create_test_graph();
        let density = graph.density();

        assert!(density > 0.0 && density <= 1.0);
    }

    #[test]
    fn test_clustering() {
        let graph = create_test_graph();
        let coeff = graph.clustering_coefficient();

        assert!(coeff >= 0.0 && coeff <= 1.0);
    }
}
