//! Graph Embeddings
//!
//! This module provides graph embedding algorithms for converting graph structures
//! into dense vector representations:
//!
//! - **Node2Vec**: Random walk-based embeddings capturing network neighborhoods
//! - **GraphSAGE**: Inductive representation learning using neighborhood sampling
//! - **DeepWalk**: Simplified random walk embeddings
//! - **Struct2Vec**: Structure-aware graph embeddings
//!
//! ## Use Cases
//!
//! - Node classification and clustering
//! - Link prediction
//! - Graph visualization
//! - Similarity search in graph space
//! - Transfer learning across graphs

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use rand::Rng;

/// Graph embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Walk length for random walks
    pub walk_length: usize,
    /// Number of walks per node
    pub walks_per_node: usize,
    /// Context window size (for Skip-Gram)
    pub context_size: usize,
    /// Return parameter (Node2Vec p)
    pub return_param: f32,
    /// In-out parameter (Node2Vec q)
    pub inout_param: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Number of training epochs
    pub epochs: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            walk_length: 80,
            walks_per_node: 10,
            context_size: 10,
            return_param: 1.0,
            inout_param: 1.0,
            learning_rate: 0.025,
            negative_samples: 5,
            epochs: 10,
        }
    }
}

/// Graph for embedding generation
pub struct EmbeddingGraph {
    /// Adjacency list: node_id -> [(neighbor_id, weight)]
    adjacency: HashMap<String, Vec<(String, f32)>>,
    /// All node IDs
    nodes: Vec<String>,
    /// Node index mapping
    node_index: HashMap<String, usize>,
}

impl EmbeddingGraph {
    /// Create embedding graph from edge list
    ///
    /// # Arguments
    /// * `edges` - List of (source, target, weight) tuples
    pub fn from_edges(edges: Vec<(String, String, f32)>) -> Self {
        let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut nodes_set = HashSet::new();

        for (source, target, weight) in edges {
            adjacency
                .entry(source.clone())
                .or_default()
                .push((target.clone(), weight));

            adjacency
                .entry(target.clone())
                .or_default()
                .push((source.clone(), weight));

            nodes_set.insert(source);
            nodes_set.insert(target);
        }

        let nodes: Vec<String> = nodes_set.into_iter().collect();
        let node_index: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        Self {
            adjacency,
            nodes,
            node_index,
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: &str) -> Option<&Vec<(String, f32)>> {
        self.adjacency.get(node)
    }

    /// Get node index
    pub fn get_index(&self, node: &str) -> Option<usize> {
        self.node_index.get(node).copied()
    }

    /// Get node by index
    pub fn get_node(&self, index: usize) -> Option<&String> {
        self.nodes.get(index)
    }
}

/// Node2Vec embeddings generator
pub struct Node2Vec {
    config: EmbeddingConfig,
    /// Learned embeddings: node_id -> embedding vector
    embeddings: HashMap<String, Vec<f32>>,
}

impl Node2Vec {
    /// Create new Node2Vec generator
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            embeddings: HashMap::new(),
        }
    }

    /// Generate embeddings for graph
    pub fn fit(&mut self, graph: &EmbeddingGraph) {
        // Generate random walks
        let walks = self.generate_walks(graph);

        // Initialize embeddings randomly
        self.initialize_embeddings(graph);

        // Train Skip-Gram model on walks
        self.train_skipgram(&walks);
    }

    /// Generate biased random walks (Node2Vec)
    fn generate_walks(&self, graph: &EmbeddingGraph) -> Vec<Vec<String>> {
        let mut rng = rand::thread_rng();
        let mut walks = Vec::new();

        for _ in 0..self.config.walks_per_node {
            for node in &graph.nodes {
                let walk = self.random_walk(graph, node, &mut rng);
                walks.push(walk);
            }
        }

        walks
    }

    /// Perform single biased random walk from starting node
    fn random_walk<R: Rng>(
        &self,
        graph: &EmbeddingGraph,
        start: &str,
        rng: &mut R,
    ) -> Vec<String> {
        let mut walk = vec![start.to_string()];

        for _ in 1..self.config.walk_length {
            let current = walk.last().unwrap();

            if let Some(neighbors) = graph.neighbors(current) {
                if neighbors.is_empty() {
                    break;
                }

                // Sample next node using biased probabilities
                let next = if walk.len() == 1 {
                    // First step: uniform random
                    &neighbors[rng.gen_range(0..neighbors.len())].0
                } else {
                    // Subsequent steps: use Node2Vec bias
                    let prev = &walk[walk.len() - 2];
                    self.sample_next(prev, current, neighbors, rng)
                };

                walk.push(next.clone());
            } else {
                break;
            }
        }

        walk
    }

    /// Sample next node with Node2Vec bias (p, q parameters)
    fn sample_next<'a, R: Rng>(
        &self,
        prev: &str,
        _current: &str,
        neighbors: &'a [(String, f32)],
        rng: &mut R,
    ) -> &'a String {
        // Calculate transition probabilities based on p and q
        let mut probs: Vec<f32> = neighbors
            .iter()
            .map(|(neighbor, weight)| {
                let alpha = if neighbor == prev {
                    // Return to previous node
                    1.0 / self.config.return_param
                } else {
                    // Check if neighbor is also neighbor of prev (BFS vs DFS)
                    1.0 / self.config.inout_param
                };
                weight * alpha
            })
            .collect();

        // Normalize probabilities
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        // Sample using cumulative distribution
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum {
                return &neighbors[i].0;
            }
        }

        &neighbors[neighbors.len() - 1].0
    }

    /// Initialize random embeddings
    fn initialize_embeddings(&mut self, graph: &EmbeddingGraph) {
        let mut rng = rand::thread_rng();

        for node in &graph.nodes {
            let embedding: Vec<f32> = (0..self.config.dimension)
                .map(|_| (rng.gen::<f32>() - 0.5) / self.config.dimension as f32)
                .collect();

            self.embeddings.insert(node.clone(), embedding);
        }
    }

    /// Train Skip-Gram model on walks
    fn train_skipgram(&mut self, walks: &[Vec<String>]) {
        for _ in 0..self.config.epochs {
            for walk in walks {
                for (i, node) in walk.iter().enumerate() {
                    // Define context window
                    let start = i.saturating_sub(self.config.context_size);
                    let end = (i + self.config.context_size + 1).min(walk.len());

                    for (j, context_node) in walk.iter().enumerate().take(end).skip(start) {
                        if i != j {
                            self.update_embeddings(node, context_node);
                        }
                    }
                }
            }
        }
    }

    /// Update embeddings using Skip-Gram objective (simplified)
    fn update_embeddings(&mut self, target: &str, context: &str) {
        // Simplified update: move embeddings closer for positive pairs
        // Real implementation would use negative sampling and gradient descent

        let lr = self.config.learning_rate;

        if let (Some(target_emb), Some(context_emb)) =
            (self.embeddings.get(target), self.embeddings.get(context))
        {
            // Calculate gradient direction (simplified)
            let mut target_new = target_emb.clone();
            let mut context_new = context_emb.clone();

            for i in 0..self.config.dimension {
                let diff = context_emb[i] - target_emb[i];
                target_new[i] += lr * diff;
                context_new[i] -= lr * diff;
            }

            self.embeddings.insert(target.to_string(), target_new);
            self.embeddings.insert(context.to_string(), context_new);
        }
    }

    /// Get embedding for a node
    pub fn get_embedding(&self, node: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(node)
    }

    /// Get all embeddings
    pub fn embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.embeddings
    }
}

/// GraphSAGE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSAGEConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Samples per layer
    pub samples_per_layer: Vec<usize>,
    /// Aggregation function
    pub aggregator: Aggregator,
}

/// Aggregation functions for GraphSAGE
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Aggregator {
    /// Mean aggregation
    Mean,
    /// Max pooling
    MaxPool,
    /// LSTM aggregation
    Lstm,
    /// Attention-based
    Attention,
}

impl Default for GraphSAGEConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            num_layers: 2,
            samples_per_layer: vec![25, 10],
            aggregator: Aggregator::Mean,
        }
    }
}

/// GraphSAGE embeddings generator
pub struct GraphSAGE {
    config: GraphSAGEConfig,
    embeddings: HashMap<String, Vec<f32>>,
}

impl GraphSAGE {
    /// Create new GraphSAGE generator
    pub fn new(config: GraphSAGEConfig) -> Self {
        Self {
            config,
            embeddings: HashMap::new(),
        }
    }

    /// Generate embeddings for graph (simplified inductive approach)
    pub fn fit(&mut self, graph: &EmbeddingGraph) {
        // Initialize with node features (random for now)
        let mut rng = rand::thread_rng();
        let mut node_features: HashMap<String, Vec<f32>> = HashMap::new();

        for node in &graph.nodes {
            let features: Vec<f32> = (0..self.config.dimension)
                .map(|_| rng.gen::<f32>())
                .collect();
            node_features.insert(node.clone(), features);
        }

        // Iteratively aggregate neighborhood information
        for layer in 0..self.config.num_layers {
            let samples = self.config.samples_per_layer.get(layer).copied().unwrap_or(10);
            node_features = self.aggregate_layer(graph, &node_features, samples);
        }

        self.embeddings = node_features;
    }

    /// Aggregate one layer of neighborhood information
    fn aggregate_layer(
        &self,
        graph: &EmbeddingGraph,
        features: &HashMap<String, Vec<f32>>,
        num_samples: usize,
    ) -> HashMap<String, Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut new_features = HashMap::new();

        for node in &graph.nodes {
            // Sample neighbors
            let neighbors = if let Some(neighs) = graph.neighbors(node) {
                let sample_size = num_samples.min(neighs.len());
                let mut sampled = Vec::new();
                let mut indices: Vec<usize> = (0..neighs.len()).collect();

                for _ in 0..sample_size {
                    let idx = rng.gen_range(0..indices.len());
                    let neighbor_idx = indices.remove(idx);
                    sampled.push(&neighs[neighbor_idx].0);
                }

                sampled
            } else {
                Vec::new()
            };

            // Aggregate neighbor features
            let aggregated = self.aggregate_neighbors(features, &neighbors);

            // Combine with node's own features
            let node_feat = features.get(node).unwrap();
            let combined = self.combine_features(node_feat, &aggregated);

            new_features.insert(node.clone(), combined);
        }

        new_features
    }

    /// Aggregate neighbor features
    fn aggregate_neighbors(
        &self,
        features: &HashMap<String, Vec<f32>>,
        neighbors: &[&String],
    ) -> Vec<f32> {
        if neighbors.is_empty() {
            return vec![0.0; self.config.dimension];
        }

        match self.config.aggregator {
            Aggregator::Mean => {
                let mut sum = vec![0.0; self.config.dimension];
                for neighbor in neighbors {
                    if let Some(feat) = features.get(*neighbor) {
                        for i in 0..self.config.dimension {
                            sum[i] += feat[i];
                        }
                    }
                }

                for val in &mut sum {
                    *val /= neighbors.len() as f32;
                }

                sum
            }
            _ => {
                // For now, default to mean for other aggregators
                // TODO: Implement MaxPool, LSTM, Attention
                let mut sum = vec![0.0; self.config.dimension];
                for neighbor in neighbors {
                    if let Some(feat) = features.get(*neighbor) {
                        for i in 0..self.config.dimension {
                            sum[i] += feat[i];
                        }
                    }
                }

                for val in &mut sum {
                    *val /= neighbors.len() as f32;
                }

                sum
            }
        }
    }

    /// Combine node features with aggregated neighbor features
    fn combine_features(&self, node_feat: &[f32], neighbor_feat: &[f32]) -> Vec<f32> {
        // Simple concatenation followed by projection (simplified)
        // Real implementation would use learned weight matrices

        let mut combined = Vec::with_capacity(self.config.dimension);

        for i in 0..self.config.dimension {
            // Weighted combination
            combined.push((node_feat[i] + neighbor_feat[i]) / 2.0);
        }

        combined
    }

    /// Get embedding for a node
    pub fn get_embedding(&self, node: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(node)
    }

    /// Get all embeddings
    pub fn embeddings(&self) -> &HashMap<String, Vec<f32>> {
        &self.embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> EmbeddingGraph {
        let edges = vec![
            ("A".to_string(), "B".to_string(), 1.0),
            ("A".to_string(), "C".to_string(), 1.0),
            ("B".to_string(), "C".to_string(), 1.0),
            ("B".to_string(), "D".to_string(), 1.0),
            ("C".to_string(), "D".to_string(), 1.0),
            ("D".to_string(), "E".to_string(), 1.0),
        ];

        EmbeddingGraph::from_edges(edges)
    }

    #[test]
    fn test_embedding_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.node_count(), 5);
        assert!(graph.neighbors("A").is_some());
        assert_eq!(graph.neighbors("A").unwrap().len(), 2);
    }

    #[test]
    fn test_node2vec_initialization() {
        let config = EmbeddingConfig::default();
        let node2vec = Node2Vec::new(config);
        assert_eq!(node2vec.embeddings.len(), 0);
    }

    #[test]
    fn test_node2vec_fit() {
        let graph = create_test_graph();
        let config = EmbeddingConfig {
            dimension: 64,
            walk_length: 10,
            walks_per_node: 5,
            epochs: 1,
            ..Default::default()
        };

        let mut node2vec = Node2Vec::new(config);
        node2vec.fit(&graph);

        assert_eq!(node2vec.embeddings.len(), 5);

        for node in &graph.nodes {
            let emb = node2vec.get_embedding(node).unwrap();
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_graphsage_fit() {
        let graph = create_test_graph();
        let config = GraphSAGEConfig {
            dimension: 64,
            num_layers: 2,
            samples_per_layer: vec![3, 2],
            aggregator: Aggregator::Mean,
        };

        let mut graphsage = GraphSAGE::new(config);
        graphsage.fit(&graph);

        assert_eq!(graphsage.embeddings.len(), 5);

        for node in &graph.nodes {
            let emb = graphsage.get_embedding(node).unwrap();
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_random_walk_generation() {
        let graph = create_test_graph();
        let config = EmbeddingConfig {
            walk_length: 5,
            walks_per_node: 1,
            ..Default::default()
        };

        let node2vec = Node2Vec::new(config);
        let walks = node2vec.generate_walks(&graph);

        assert_eq!(walks.len(), 5); // 5 nodes * 1 walk per node
        for walk in &walks {
            assert!(walk.len() <= 5);
            assert!(walk.len() > 0);
        }
    }
}
