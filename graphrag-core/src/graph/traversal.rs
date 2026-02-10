//! Graph Traversal Algorithms for GraphRAG
//!
//! This module implements deterministic graph traversal algorithms that don't require
//! machine learning, following NLP best practices for knowledge graph exploration:
//!
//! - **BFS (Breadth-First Search)**: Level-by-level exploration for shortest paths
//! - **DFS (Depth-First Search)**: Deep exploration for discovering all paths
//! - **Ego-Network Extraction**: K-hop neighborhoods around entities
//! - **Multi-Source Path Finding**: Simultaneous search from multiple entities
//! - **Query-Focused Subgraph Extraction**: Context-aware subgraph retrieval
//!
//! These algorithms are essential for the query phase of GraphRAG, enabling:
//! - Efficient entity-centric retrieval
//! - Relationship path discovery
//! - Context-aware information gathering
//! - Multi-hop reasoning without neural networks

use crate::core::{Entity, EntityId, KnowledgeGraph, Relationship, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for graph traversal algorithms
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum depth for BFS/DFS traversal
    pub max_depth: usize,
    /// Maximum number of paths to return
    pub max_paths: usize,
    /// Whether to include edge weights in path scoring
    pub use_edge_weights: bool,
    /// Minimum relationship strength to traverse
    pub min_relationship_strength: f32,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_paths: 100,
            use_edge_weights: true,
            min_relationship_strength: 0.5,
        }
    }
}

/// Result of a graph traversal operation
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Entities discovered during traversal
    pub entities: Vec<Entity>,
    /// Relationships traversed
    pub relationships: Vec<Relationship>,
    /// Paths found (for path-finding operations)
    pub paths: Vec<Vec<EntityId>>,
    /// Distance/depth of each entity from source
    pub distances: HashMap<EntityId, usize>,
}

/// Graph traversal system implementing various search algorithms
pub struct GraphTraversal {
    config: TraversalConfig,
}

impl Default for GraphTraversal {
    fn default() -> Self {
        Self::new(TraversalConfig::default())
    }
}

impl GraphTraversal {
    /// Create a new graph traversal system
    pub fn new(config: TraversalConfig) -> Self {
        Self { config }
    }

    /// Breadth-First Search (BFS) from a source entity
    ///
    /// BFS explores the graph level by level, guaranteeing shortest paths.
    /// Ideal for finding entities within a certain distance from the source.
    ///
    /// # Arguments
    /// * `graph` - The knowledge graph to traverse
    /// * `source` - Starting entity ID
    ///
    /// # Returns
    /// TraversalResult with entities, relationships, and distances
    pub fn bfs(
        &self,
        graph: &KnowledgeGraph,
        source: &EntityId,
    ) -> Result<TraversalResult> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut distances = HashMap::new();
        let mut discovered_entities = Vec::new();
        let mut discovered_relationships = Vec::new();

        // Initialize with source entity
        queue.push_back((source.clone(), 0));
        distances.insert(source.clone(), 0);

        while let Some((current_id, depth)) = queue.pop_front() {
            // Stop if we've reached max depth
            if depth >= self.config.max_depth {
                continue;
            }

            // Skip if already visited
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            // Add current entity to results
            if let Some(entity) = graph.get_entity(&current_id) {
                discovered_entities.push(entity.clone());
            }

            // Get all neighbors (entities connected by relationships)
            let neighbors = self.get_neighbors(graph, &current_id);

            for (neighbor_id, relationship) in neighbors {
                // Filter by relationship confidence
                if relationship.confidence < self.config.min_relationship_strength {
                    continue;
                }

                // Add to queue if not visited
                if !visited.contains(&neighbor_id) {
                    queue.push_back((neighbor_id.clone(), depth + 1));
                    distances.entry(neighbor_id.clone()).or_insert(depth + 1);
                    discovered_relationships.push(relationship);
                }
            }
        }

        Ok(TraversalResult {
            entities: discovered_entities,
            relationships: discovered_relationships,
            paths: Vec::new(), // BFS doesn't track individual paths
            distances,
        })
    }

    /// Depth-First Search (DFS) from a source entity
    ///
    /// DFS explores as far as possible along each branch before backtracking.
    /// Useful for finding all possible paths and deep exploration.
    ///
    /// # Arguments
    /// * `graph` - The knowledge graph to traverse
    /// * `source` - Starting entity ID
    ///
    /// # Returns
    /// TraversalResult with entities, relationships, and discovered paths
    pub fn dfs(
        &self,
        graph: &KnowledgeGraph,
        source: &EntityId,
    ) -> Result<TraversalResult> {
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();
        let mut discovered_entities = Vec::new();
        let mut discovered_relationships = Vec::new();

        self.dfs_recursive(
            graph,
            source,
            0,
            &mut visited,
            &mut distances,
            &mut discovered_entities,
            &mut discovered_relationships,
        )?;

        Ok(TraversalResult {
            entities: discovered_entities,
            relationships: discovered_relationships,
            paths: Vec::new(), // Basic DFS doesn't track paths
            distances,
        })
    }

    /// Recursive DFS helper
    #[allow(clippy::too_many_arguments)]
    fn dfs_recursive(
        &self,
        graph: &KnowledgeGraph,
        current_id: &EntityId,
        depth: usize,
        visited: &mut HashSet<EntityId>,
        distances: &mut HashMap<EntityId, usize>,
        discovered_entities: &mut Vec<Entity>,
        discovered_relationships: &mut Vec<Relationship>,
    ) -> Result<()> {
        // Stop if max depth reached
        if depth >= self.config.max_depth {
            return Ok(());
        }

        // Skip if already visited (avoid cycles)
        if visited.contains(current_id) {
            return Ok(());
        }

        visited.insert(current_id.clone());
        distances.insert(current_id.clone(), depth);

        // Add current entity
        if let Some(entity) = graph.get_entity(current_id) {
            discovered_entities.push(entity.clone());
        }

        // Recursively visit neighbors
        let neighbors = self.get_neighbors(graph, current_id);

        for (neighbor_id, relationship) in neighbors {
            if relationship.confidence < self.config.min_relationship_strength {
                continue;
            }

            if !visited.contains(&neighbor_id) {
                discovered_relationships.push(relationship);
                self.dfs_recursive(
                    graph,
                    &neighbor_id,
                    depth + 1,
                    visited,
                    distances,
                    discovered_entities,
                    discovered_relationships,
                )?;
            }
        }

        Ok(())
    }

    /// Extract K-hop ego-network around an entity
    ///
    /// An ego-network is a subgraph containing all entities within K hops
    /// of the source entity. This is useful for context-aware retrieval.
    ///
    /// # Arguments
    /// * `graph` - The knowledge graph
    /// * `entity_id` - Center entity for the ego-network
    /// * `k_hops` - Number of hops to include (defaults to config.max_depth)
    ///
    /// # Returns
    /// TraversalResult with the ego-network subgraph
    pub fn ego_network(
        &self,
        graph: &KnowledgeGraph,
        entity_id: &EntityId,
        k_hops: Option<usize>,
    ) -> Result<TraversalResult> {
        let hops = k_hops.unwrap_or(self.config.max_depth);

        let mut subgraph_entities = Vec::new();
        let mut subgraph_relationships = Vec::new();
        let mut visited = HashSet::new();
        let mut distances = HashMap::new();

        // Start with the ego entity
        visited.insert(entity_id.clone());
        distances.insert(entity_id.clone(), 0);

        if let Some(entity) = graph.get_entity(entity_id) {
            subgraph_entities.push(entity.clone());
        }

        // Use BFS to expand outward for k hops
        let mut current_layer = vec![entity_id.clone()];

        for hop in 1..=hops {
            let mut next_layer = Vec::new();

            for current_id in &current_layer {
                let neighbors = self.get_neighbors(graph, current_id);

                for (neighbor_id, relationship) in neighbors {
                    if relationship.confidence < self.config.min_relationship_strength {
                        continue;
                    }

                    // Add relationship
                    subgraph_relationships.push(relationship);

                    // Add neighbor if not visited
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id.clone());
                        distances.insert(neighbor_id.clone(), hop);

                        if let Some(entity) = graph.get_entity(&neighbor_id) {
                            subgraph_entities.push(entity.clone());
                        }

                        next_layer.push(neighbor_id);
                    }
                }
            }

            current_layer = next_layer;
        }

        Ok(TraversalResult {
            entities: subgraph_entities,
            relationships: subgraph_relationships,
            paths: Vec::new(),
            distances,
        })
    }

    /// Multi-source BFS pathfinding
    ///
    /// Performs simultaneous BFS from multiple source entities to find
    /// intersections and common neighbors efficiently.
    ///
    /// # Arguments
    /// * `graph` - The knowledge graph
    /// * `sources` - Multiple starting entity IDs
    ///
    /// # Returns
    /// TraversalResult with entities reachable from any source
    pub fn multi_source_bfs(
        &self,
        graph: &KnowledgeGraph,
        sources: &[EntityId],
    ) -> Result<TraversalResult> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut distances = HashMap::new();
        let mut discovered_entities = Vec::new();
        let mut discovered_relationships = Vec::new();

        // Initialize queue with all sources
        for source in sources {
            queue.push_back((source.clone(), 0));
            distances.insert(source.clone(), 0);
        }

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= self.config.max_depth {
                continue;
            }

            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(entity) = graph.get_entity(&current_id) {
                discovered_entities.push(entity.clone());
            }

            let neighbors = self.get_neighbors(graph, &current_id);

            for (neighbor_id, relationship) in neighbors {
                if relationship.confidence < self.config.min_relationship_strength {
                    continue;
                }

                if !visited.contains(&neighbor_id) {
                    queue.push_back((neighbor_id.clone(), depth + 1));
                    distances.entry(neighbor_id.clone()).or_insert(depth + 1);
                    discovered_relationships.push(relationship);
                }
            }
        }

        Ok(TraversalResult {
            entities: discovered_entities,
            relationships: discovered_relationships,
            paths: Vec::new(),
            distances,
        })
    }

    /// Find all paths between two entities
    ///
    /// Uses DFS to discover all possible paths from source to target
    /// within the maximum depth limit.
    ///
    /// # Arguments
    /// * `graph` - The knowledge graph
    /// * `source` - Starting entity
    /// * `target` - Target entity
    ///
    /// # Returns
    /// TraversalResult with all discovered paths
    pub fn find_all_paths(
        &self,
        graph: &KnowledgeGraph,
        source: &EntityId,
        target: &EntityId,
    ) -> Result<TraversalResult> {
        let mut all_paths = Vec::new();
        let mut current_path = vec![source.clone()];
        let mut visited = HashSet::new();
        let mut discovered_relationships = Vec::new();

        self.find_paths_recursive(
            graph,
            source,
            target,
            &mut current_path,
            &mut visited,
            &mut all_paths,
            &mut discovered_relationships,
            0,
        )?;

        // Collect all unique entities from paths
        let mut unique_entities = HashSet::new();
        for path in &all_paths {
            unique_entities.extend(path.iter().cloned());
        }

        let discovered_entities: Vec<Entity> = unique_entities
            .iter()
            .filter_map(|id| graph.get_entity(id).cloned())
            .collect();

        Ok(TraversalResult {
            entities: discovered_entities,
            relationships: discovered_relationships,
            paths: all_paths,
            distances: HashMap::new(),
        })
    }

    /// Recursive helper for find_all_paths
    #[allow(clippy::too_many_arguments)]
    fn find_paths_recursive(
        &self,
        graph: &KnowledgeGraph,
        current: &EntityId,
        target: &EntityId,
        current_path: &mut Vec<EntityId>,
        visited: &mut HashSet<EntityId>,
        all_paths: &mut Vec<Vec<EntityId>>,
        discovered_relationships: &mut Vec<Relationship>,
        depth: usize,
    ) -> Result<()> {
        // Stop if max depth or max paths reached
        if depth >= self.config.max_depth || all_paths.len() >= self.config.max_paths {
            return Ok(());
        }

        // Found target - save path
        if current == target {
            all_paths.push(current_path.clone());
            return Ok(());
        }

        visited.insert(current.clone());

        let neighbors = self.get_neighbors(graph, current);

        for (neighbor_id, relationship) in neighbors {
            if relationship.confidence < self.config.min_relationship_strength {
                continue;
            }

            if !visited.contains(&neighbor_id) {
                current_path.push(neighbor_id.clone());
                discovered_relationships.push(relationship);

                self.find_paths_recursive(
                    graph,
                    &neighbor_id,
                    target,
                    current_path,
                    visited,
                    all_paths,
                    discovered_relationships,
                    depth + 1,
                )?;

                current_path.pop();
            }
        }

        visited.remove(current);

        Ok(())
    }

    /// Get neighbors of an entity with their connecting relationships
    fn get_neighbors(
        &self,
        graph: &KnowledgeGraph,
        entity_id: &EntityId,
    ) -> Vec<(EntityId, Relationship)> {
        let mut neighbors = Vec::new();

        // Get all relationships where this entity is the source
        for relationship in graph.get_all_relationships() {
            if &relationship.source == entity_id {
                neighbors.push((relationship.target.clone(), relationship.clone()));
            }
            // Also consider bidirectional traversal
            if &relationship.target == entity_id {
                neighbors.push((relationship.source.clone(), relationship.clone()));
            }
        }

        neighbors
    }

    /// Extract query-focused subgraph
    ///
    /// Extracts a subgraph relevant to a specific query by:
    /// 1. Identifying seed entities from query
    /// 2. Expanding via ego-networks
    /// 3. Filtering by relevance
    ///
    /// # Arguments
    /// * `graph` - The knowledge graph
    /// * `seed_entities` - Starting entities identified in query
    /// * `expansion_hops` - How many hops to expand
    ///
    /// # Returns
    /// TraversalResult with query-relevant subgraph
    pub fn query_focused_subgraph(
        &self,
        graph: &KnowledgeGraph,
        seed_entities: &[EntityId],
        expansion_hops: usize,
    ) -> Result<TraversalResult> {
        let mut combined_entities = Vec::new();
        let mut combined_relationships = Vec::new();
        let mut combined_distances = HashMap::new();
        let mut seen_entities = HashSet::new();
        let mut seen_relationships = HashSet::new();

        // Extract ego-network for each seed entity
        for seed in seed_entities {
            let ego_result = self.ego_network(graph, seed, Some(expansion_hops))?;

            for entity in ego_result.entities {
                if !seen_entities.contains(&entity.id) {
                    seen_entities.insert(entity.id.clone());
                    combined_entities.push(entity);
                }
            }

            for rel in ego_result.relationships {
                let rel_key = (rel.source.clone(), rel.target.clone(), rel.relation_type.clone());
                if !seen_relationships.contains(&rel_key) {
                    seen_relationships.insert(rel_key);
                    combined_relationships.push(rel);
                }
            }

            for (entity_id, distance) in ego_result.distances {
                combined_distances
                    .entry(entity_id)
                    .and_modify(|d: &mut usize| *d = (*d).min(distance))
                    .or_insert(distance);
            }
        }

        Ok(TraversalResult {
            entities: combined_entities,
            relationships: combined_relationships,
            paths: Vec::new(),
            distances: combined_distances,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Entity;

    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Create entities: A -> B -> C
        //                  A -> D
        let entity_a = Entity::new(
            EntityId::new("A".to_string()),
            "Entity A".to_string(),
            "CONCEPT".to_string(),
            0.9,
        );
        let entity_b = Entity::new(
            EntityId::new("B".to_string()),
            "Entity B".to_string(),
            "CONCEPT".to_string(),
            0.9,
        );
        let entity_c = Entity::new(
            EntityId::new("C".to_string()),
            "Entity C".to_string(),
            "CONCEPT".to_string(),
            0.9,
        );
        let entity_d = Entity::new(
            EntityId::new("D".to_string()),
            "Entity D".to_string(),
            "CONCEPT".to_string(),
            0.9,
        );

        graph.add_entity(entity_a);
        graph.add_entity(entity_b);
        graph.add_entity(entity_c);
        graph.add_entity(entity_d);

        // Add relationships
        let _ = graph.add_relationship(Relationship {
            source: EntityId::new("A".to_string()),
            target: EntityId::new("B".to_string()),
            relation_type: "RELATED_TO".to_string(),
            confidence: 0.8,
            context: Vec::new(),
        });

        let _ = graph.add_relationship(Relationship {
            source: EntityId::new("B".to_string()),
            target: EntityId::new("C".to_string()),
            relation_type: "RELATED_TO".to_string(),
            confidence: 0.9,
            context: Vec::new(),
        });

        let _ = graph.add_relationship(Relationship {
            source: EntityId::new("A".to_string()),
            target: EntityId::new("D".to_string()),
            relation_type: "RELATED_TO".to_string(),
            confidence: 0.7,
            context: Vec::new(),
        });

        graph
    }

    #[test]
    fn test_bfs_traversal() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::default();
        let source = EntityId::new("A".to_string());

        let result = traversal.bfs(&graph, &source).unwrap();

        // Should discover all connected entities
        assert!(result.entities.len() >= 1);
        assert!(result.distances.contains_key(&source));
    }

    #[test]
    fn test_dfs_traversal() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::default();
        let source = EntityId::new("A".to_string());

        let result = traversal.dfs(&graph, &source).unwrap();

        // Should discover entities through DFS
        assert!(result.entities.len() >= 1);
        assert!(result.distances.contains_key(&source));
    }

    #[test]
    fn test_ego_network() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::default();
        let entity_id = EntityId::new("A".to_string());

        let result = traversal.ego_network(&graph, &entity_id, Some(1)).unwrap();

        // 1-hop ego network of A should include A, B, and D
        assert!(result.entities.len() >= 2); // At least A and one neighbor
        assert_eq!(*result.distances.get(&entity_id).unwrap(), 0);
    }

    #[test]
    fn test_multi_source_bfs() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::default();
        let sources = vec![
            EntityId::new("A".to_string()),
            EntityId::new("C".to_string()),
        ];

        let result = traversal.multi_source_bfs(&graph, &sources).unwrap();

        // Should discover entities from both sources
        assert!(result.entities.len() >= 2);
    }

    #[test]
    fn test_find_all_paths() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::default();
        let source = EntityId::new("A".to_string());
        let target = EntityId::new("C".to_string());

        let result = traversal.find_all_paths(&graph, &source, &target).unwrap();

        // Should find at least one path from A to C (A -> B -> C)
        assert!(!result.paths.is_empty());
        assert!(result.paths[0].contains(&source));
        assert!(result.paths[0].contains(&target));
    }

    #[test]
    fn test_query_focused_subgraph() {
        let graph = create_test_graph();
        let traversal = GraphTraversal::default();
        let seeds = vec![EntityId::new("A".to_string())];

        let result = traversal
            .query_focused_subgraph(&graph, &seeds, 2)
            .unwrap();

        // Should extract subgraph around seed entity
        assert!(!result.entities.is_empty());
        assert!(!result.relationships.is_empty());
    }
}
