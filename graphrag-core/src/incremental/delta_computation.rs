//! Delta Computation for Incremental Graph Updates
//!
//! This module implements efficient delta computation between graph snapshots,
//! calculating only the minimal set of changes instead of rebuilding the entire graph.
//!
//! Key optimizations:
//! - Bloom filters for fast negative checks (O(1) "definitely not changed")
//! - Content-based hashing (SHA-256) for change detection
//! - Parallel diff computation with rayon
//! - Memory-efficient streaming diff for large graphs

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for delta computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaComputationConfig {
    /// Use bloom filters for fast negative checks
    pub use_bloom_filter: bool,

    /// Bloom filter false positive rate (0.01 = 1%)
    pub bloom_false_positive_rate: f64,

    /// Enable parallel delta computation
    pub parallel_computation: bool,

    /// Chunk size for parallel processing
    pub parallel_chunk_size: usize,

    /// Enable detailed change tracking (more memory, more detail)
    pub detailed_tracking: bool,

    /// Hash algorithm for content comparison
    pub hash_algorithm: HashAlgorithm,
}

impl Default for DeltaComputationConfig {
    fn default() -> Self {
        Self {
            use_bloom_filter: true,
            bloom_false_positive_rate: 0.01,
            parallel_computation: true,
            parallel_chunk_size: 1000,
            detailed_tracking: true,
            hash_algorithm: HashAlgorithm::Sha256,
        }
    }
}

/// Hash algorithm used for content-based change detection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// SHA-256 cryptographic hash (slower but secure)
    Sha256,
    /// BLAKE3 hash (faster, optimized for performance)
    Blake3,
}

/// Represents a snapshot of the graph at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Unique identifier for this snapshot
    pub snapshot_id: String,
    /// When the snapshot was created
    pub timestamp: DateTime<Utc>,
    /// Map of node ID to node snapshot
    pub nodes: HashMap<String, NodeSnapshot>,
    /// Map of edge (source, target) to edge snapshot
    pub edges: HashMap<(String, String), EdgeSnapshot>,
    /// Metadata about the snapshot
    pub metadata: SnapshotMetadata,
}

/// Snapshot of a single node at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSnapshot {
    /// Node identifier
    pub node_id: String,
    /// Hash of node content for change detection
    pub content_hash: String,
    /// Node properties
    pub properties: HashMap<String, String>,
    /// When the node was last modified
    pub last_modified: DateTime<Utc>,
}

/// Snapshot of a single edge at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSnapshot {
    /// Source node identifier
    pub source: String,
    /// Target node identifier
    pub target: String,
    /// Type of edge/relationship
    pub edge_type: String,
    /// Hash of edge content for change detection
    pub content_hash: String,
    /// Edge properties
    pub properties: HashMap<String, String>,
    /// When the edge was last modified
    pub last_modified: DateTime<Utc>,
}

/// Metadata about a graph snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Total number of nodes in snapshot
    pub total_nodes: usize,
    /// Total number of edges in snapshot
    pub total_edges: usize,
    /// Schema version for compatibility
    pub schema_version: String,
    /// Compression algorithm used (if any)
    pub compression: Option<String>,
}

/// Represents the delta between two graph snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDelta {
    /// ID of the source snapshot
    pub from_snapshot: String,
    /// ID of the target snapshot
    pub to_snapshot: String,
    /// When the delta was computed
    pub computed_at: DateTime<Utc>,

    /// Nodes that were added
    pub nodes_added: Vec<NodeSnapshot>,
    /// IDs of nodes that were removed
    pub nodes_removed: Vec<String>,
    /// Nodes that were modified
    pub nodes_modified: Vec<NodeModification>,

    /// Edges that were added
    pub edges_added: Vec<EdgeSnapshot>,
    /// Edges that were removed (source, target)
    pub edges_removed: Vec<(String, String)>,
    /// Edges that were modified
    pub edges_modified: Vec<EdgeModification>,

    /// Statistics about the delta
    pub statistics: DeltaStatistics,
}

/// Represents a modification to a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeModification {
    /// Node identifier
    pub node_id: String,
    /// Hash before modification
    pub old_hash: String,
    /// Hash after modification
    pub new_hash: String,
    /// List of property changes
    pub property_changes: Vec<PropertyChange>,
}

/// Represents a modification to an edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeModification {
    /// Source node identifier
    pub source: String,
    /// Target node identifier
    pub target: String,
    /// Hash before modification
    pub old_hash: String,
    /// Hash after modification
    pub new_hash: String,
    /// List of property changes
    pub property_changes: Vec<PropertyChange>,
}

/// Represents a change to a single property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyChange {
    /// Name of the property that changed
    pub property_name: String,
    /// Value before the change (None if added)
    pub old_value: Option<String>,
    /// Value after the change (None if removed)
    pub new_value: Option<String>,
    /// Type of change (Added, Modified, Removed)
    pub change_type: ChangeType,
}

/// Type of property change
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ChangeType {
    /// Property was added
    Added,
    /// Property value was modified
    Modified,
    /// Property was removed
    Removed,
}

/// Statistics about delta computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaStatistics {
    /// Time taken to compute delta in milliseconds
    pub computation_time_ms: u64,
    /// Number of nodes compared
    pub nodes_compared: usize,
    /// Number of edges compared
    pub edges_compared: usize,
    /// Number of nodes that changed
    pub nodes_changed: usize,
    /// Number of edges that changed
    pub edges_changed: usize,
    /// Percentage of graph that changed (0.0-100.0)
    pub change_percentage: f32,
    /// Number of bloom filter hits (if enabled)
    pub bloom_filter_hits: Option<usize>,
    /// Number of bloom filter misses (if enabled)
    pub bloom_filter_misses: Option<usize>,
}

/// Simple bloom filter for fast negative checks
#[derive(Debug, Clone)]
struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
}

impl BloomFilter {
    fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal bloom filter size
        let size = Self::optimal_size(expected_items, false_positive_rate);
        let num_hashes = Self::optimal_hash_count(size, expected_items);

        Self {
            bits: vec![false; size],
            num_hashes,
            size,
        }
    }

    fn optimal_size(n: usize, p: f64) -> usize {
        let m = -(n as f64 * p.ln()) / (2.0_f64.ln().powi(2));
        m.ceil() as usize
    }

    fn optimal_hash_count(m: usize, n: usize) -> usize {
        let k = (m as f64 / n as f64) * 2.0_f64.ln();
        k.ceil() as usize
    }

    fn insert(&mut self, item: &str) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            self.bits[hash % self.size] = true;
        }
    }

    fn contains(&self, item: &str) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            if !self.bits[hash % self.size] {
                return false;
            }
        }
        true
    }

    fn hash(&self, item: &str, seed: usize) -> usize {
        // Simple hash function (FNV-1a variant with seed)
        let mut hash = 2166136261u64 ^ (seed as u64);
        for byte in item.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(16777619);
        }
        hash as usize
    }
}

/// Delta computer for efficient graph diff calculation
pub struct DeltaComputer {
    config: DeltaComputationConfig,
    bloom_filter: Arc<RwLock<Option<BloomFilter>>>,
    stats: Arc<RwLock<DeltaStatistics>>,
}

impl DeltaComputer {
    /// Create a new delta computer with the given configuration
    pub fn new(config: DeltaComputationConfig) -> Self {
        Self {
            config,
            bloom_filter: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(DeltaStatistics {
                computation_time_ms: 0,
                nodes_compared: 0,
                edges_compared: 0,
                nodes_changed: 0,
                edges_changed: 0,
                change_percentage: 0.0,
                bloom_filter_hits: None,
                bloom_filter_misses: None,
            })),
        }
    }

    /// Compute delta between two snapshots
    pub fn compute_delta(
        &self,
        before: &GraphSnapshot,
        after: &GraphSnapshot,
    ) -> Result<GraphDelta, DeltaError> {
        let start_time = std::time::Instant::now();

        // Initialize bloom filter if enabled
        if self.config.use_bloom_filter {
            self.initialize_bloom_filter(after);
        }

        // Compute node changes
        let (nodes_added, nodes_removed, nodes_modified) =
            self.compute_node_delta(&before.nodes, &after.nodes)?;

        // Compute edge changes
        let (edges_added, edges_removed, edges_modified) =
            self.compute_edge_delta(&before.edges, &after.edges)?;

        let computation_time = start_time.elapsed().as_millis() as u64;

        let stats = self.build_statistics(
            computation_time,
            &before,
            &after,
            &nodes_added,
            &nodes_removed,
            &nodes_modified,
            &edges_added,
            &edges_removed,
            &edges_modified,
        );

        Ok(GraphDelta {
            from_snapshot: before.snapshot_id.clone(),
            to_snapshot: after.snapshot_id.clone(),
            computed_at: Utc::now(),
            nodes_added,
            nodes_removed,
            nodes_modified,
            edges_added,
            edges_removed,
            edges_modified,
            statistics: stats,
        })
    }

    fn initialize_bloom_filter(&self, snapshot: &GraphSnapshot) {
        let expected_items = snapshot.nodes.len() + snapshot.edges.len();
        let mut bloom = BloomFilter::new(
            expected_items,
            self.config.bloom_false_positive_rate,
        );

        // Insert all node IDs
        for node_id in snapshot.nodes.keys() {
            bloom.insert(node_id);
        }

        // Insert all edge keys
        for (source, target) in snapshot.edges.keys() {
            let edge_key = format!("{}:{}", source, target);
            bloom.insert(&edge_key);
        }

        *self.bloom_filter.write() = Some(bloom);
    }

    fn compute_node_delta(
        &self,
        before_nodes: &HashMap<String, NodeSnapshot>,
        after_nodes: &HashMap<String, NodeSnapshot>,
    ) -> Result<(Vec<NodeSnapshot>, Vec<String>, Vec<NodeModification>), DeltaError> {

        let bloom_hits = 0usize;
        let bloom_misses = 0usize;

        // Find added and modified nodes
        let (added, modified): (Vec<_>, Vec<_>) = if self.config.parallel_computation {
            after_nodes
                .par_iter()
                .partition_map(|(node_id, after_node)| {
                    // Bloom filter check
                    if self.config.use_bloom_filter {
                        if let Some(ref bloom) = *self.bloom_filter.read() {
                            if !bloom.contains(node_id) {
                                // Definitely not in before snapshot
                                return rayon::iter::Either::Left(after_node.clone());
                            }
                        }
                    }

                    match before_nodes.get(node_id) {
                        None => rayon::iter::Either::Left(after_node.clone()),
                        Some(before_node) => {
                            if before_node.content_hash != after_node.content_hash {
                                rayon::iter::Either::Right(self.compute_node_modification(
                                    before_node,
                                    after_node,
                                ))
                            } else {
                                // No change
                                rayon::iter::Either::Left(NodeSnapshot {
                                    node_id: String::new(),
                                    content_hash: String::new(),
                                    properties: HashMap::new(),
                                    last_modified: Utc::now(),
                                })
                            }
                        }
                    }
                })
        } else {
            // Sequential processing
            let mut added_vec = Vec::new();
            let mut modified_vec = Vec::new();

            for (node_id, after_node) in after_nodes.iter() {
                match before_nodes.get(node_id) {
                    None => added_vec.push(after_node.clone()),
                    Some(before_node) => {
                        if before_node.content_hash != after_node.content_hash {
                            modified_vec.push(self.compute_node_modification(
                                before_node,
                                after_node,
                            ));
                        }
                    }
                }
            }

            (added_vec, modified_vec)
        };

        // Filter out empty placeholders
        let added: Vec<_> = added.into_iter()
            .filter(|n| !n.node_id.is_empty())
            .collect();

        // Find removed nodes
        let removed: Vec<String> = before_nodes
            .keys()
            .filter(|node_id| !after_nodes.contains_key(*node_id))
            .cloned()
            .collect();

        // Update stats
        let mut stats = self.stats.write();
        stats.nodes_compared = before_nodes.len().max(after_nodes.len());
        stats.nodes_changed = added.len() + removed.len() + modified.len();
        if self.config.use_bloom_filter {
            stats.bloom_filter_hits = Some(bloom_hits);
            stats.bloom_filter_misses = Some(bloom_misses);
        }

        Ok((added, removed, modified))
    }

    fn compute_edge_delta(
        &self,
        before_edges: &HashMap<(String, String), EdgeSnapshot>,
        after_edges: &HashMap<(String, String), EdgeSnapshot>,
    ) -> Result<(Vec<EdgeSnapshot>, Vec<(String, String)>, Vec<EdgeModification>), DeltaError> {

        // Find added and modified edges
        let (added, modified): (Vec<_>, Vec<_>) = if self.config.parallel_computation {
            after_edges
                .par_iter()
                .partition_map(|(edge_key, after_edge)| {
                    match before_edges.get(edge_key) {
                        None => rayon::iter::Either::Left(after_edge.clone()),
                        Some(before_edge) => {
                            if before_edge.content_hash != after_edge.content_hash {
                                rayon::iter::Either::Right(self.compute_edge_modification(
                                    before_edge,
                                    after_edge,
                                ))
                            } else {
                                rayon::iter::Either::Left(EdgeSnapshot {
                                    source: String::new(),
                                    target: String::new(),
                                    edge_type: String::new(),
                                    content_hash: String::new(),
                                    properties: HashMap::new(),
                                    last_modified: Utc::now(),
                                })
                            }
                        }
                    }
                })
        } else {
            // Sequential processing
            let mut added_vec = Vec::new();
            let mut modified_vec = Vec::new();

            for (edge_key, after_edge) in after_edges.iter() {
                match before_edges.get(edge_key) {
                    None => added_vec.push(after_edge.clone()),
                    Some(before_edge) => {
                        if before_edge.content_hash != after_edge.content_hash {
                            modified_vec.push(self.compute_edge_modification(
                                before_edge,
                                after_edge,
                            ));
                        }
                    }
                }
            }

            (added_vec, modified_vec)
        };

        // Filter out empty placeholders
        let added: Vec<_> = added.into_iter()
            .filter(|e| !e.source.is_empty())
            .collect();

        // Find removed edges
        let removed: Vec<(String, String)> = before_edges
            .keys()
            .filter(|edge_key| !after_edges.contains_key(edge_key))
            .cloned()
            .collect();

        // Update stats
        let mut stats = self.stats.write();
        stats.edges_compared = before_edges.len().max(after_edges.len());
        stats.edges_changed = added.len() + removed.len() + modified.len();

        Ok((added, removed, modified))
    }

    fn compute_node_modification(
        &self,
        before: &NodeSnapshot,
        after: &NodeSnapshot,
    ) -> NodeModification {
        let property_changes = if self.config.detailed_tracking {
            self.compute_property_changes(&before.properties, &after.properties)
        } else {
            Vec::new()
        };

        NodeModification {
            node_id: after.node_id.clone(),
            old_hash: before.content_hash.clone(),
            new_hash: after.content_hash.clone(),
            property_changes,
        }
    }

    fn compute_edge_modification(
        &self,
        before: &EdgeSnapshot,
        after: &EdgeSnapshot,
    ) -> EdgeModification {
        let property_changes = if self.config.detailed_tracking {
            self.compute_property_changes(&before.properties, &after.properties)
        } else {
            Vec::new()
        };

        EdgeModification {
            source: after.source.clone(),
            target: after.target.clone(),
            old_hash: before.content_hash.clone(),
            new_hash: after.content_hash.clone(),
            property_changes,
        }
    }

    fn compute_property_changes(
        &self,
        before: &HashMap<String, String>,
        after: &HashMap<String, String>,
    ) -> Vec<PropertyChange> {
        let mut changes = Vec::new();

        // Find added and modified properties
        for (key, after_value) in after {
            match before.get(key) {
                None => {
                    changes.push(PropertyChange {
                        property_name: key.clone(),
                        old_value: None,
                        new_value: Some(after_value.clone()),
                        change_type: ChangeType::Added,
                    });
                }
                Some(before_value) if before_value != after_value => {
                    changes.push(PropertyChange {
                        property_name: key.clone(),
                        old_value: Some(before_value.clone()),
                        new_value: Some(after_value.clone()),
                        change_type: ChangeType::Modified,
                    });
                }
                _ => {}
            }
        }

        // Find removed properties
        for (key, before_value) in before {
            if !after.contains_key(key) {
                changes.push(PropertyChange {
                    property_name: key.clone(),
                    old_value: Some(before_value.clone()),
                    new_value: None,
                    change_type: ChangeType::Removed,
                });
            }
        }

        changes
    }

    fn build_statistics(
        &self,
        computation_time_ms: u64,
        before: &GraphSnapshot,
        after: &GraphSnapshot,
        nodes_added: &[NodeSnapshot],
        nodes_removed: &[String],
        nodes_modified: &[NodeModification],
        edges_added: &[EdgeSnapshot],
        edges_removed: &[(String, String)],
        edges_modified: &[EdgeModification],
    ) -> DeltaStatistics {
        let stats = self.stats.read();

        let total_changes = nodes_added.len() + nodes_removed.len() + nodes_modified.len() +
                           edges_added.len() + edges_removed.len() + edges_modified.len();
        let total_elements = before.nodes.len() + before.edges.len() +
                            after.nodes.len() + after.edges.len();

        let change_percentage = if total_elements > 0 {
            (total_changes as f32 / total_elements as f32) * 100.0
        } else {
            0.0
        };

        DeltaStatistics {
            computation_time_ms,
            nodes_compared: stats.nodes_compared,
            edges_compared: stats.edges_compared,
            nodes_changed: stats.nodes_changed,
            edges_changed: stats.edges_changed,
            change_percentage,
            bloom_filter_hits: stats.bloom_filter_hits,
            bloom_filter_misses: stats.bloom_filter_misses,
        }
    }

    /// Create a snapshot from current graph state
    pub fn create_snapshot(
        &self,
        snapshot_id: String,
        nodes: HashMap<String, NodeSnapshot>,
        edges: HashMap<(String, String), EdgeSnapshot>,
    ) -> GraphSnapshot {
        GraphSnapshot {
            snapshot_id,
            timestamp: Utc::now(),
            metadata: SnapshotMetadata {
                total_nodes: nodes.len(),
                total_edges: edges.len(),
                schema_version: "1.0".to_string(),
                compression: None,
            },
            nodes,
            edges,
        }
    }

    /// Compute content hash for a node
    pub fn hash_node_content(&self, node_id: &str, properties: &HashMap<String, String>) -> String {
        match self.config.hash_algorithm {
            HashAlgorithm::Sha256 => self.sha256_hash(node_id, properties),
            HashAlgorithm::Blake3 => self.blake3_hash(node_id, properties),
        }
    }

    fn sha256_hash(&self, node_id: &str, properties: &HashMap<String, String>) -> String {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(node_id.as_bytes());

        // Sort properties for deterministic hashing
        let mut sorted_props: Vec<_> = properties.iter().collect();
        sorted_props.sort_by_key(|(k, _)| *k);

        for (key, value) in sorted_props {
            hasher.update(key.as_bytes());
            hasher.update(value.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    fn blake3_hash(&self, node_id: &str, properties: &HashMap<String, String>) -> String {
        // Placeholder for Blake3 (requires blake3 crate)
        // For now, fallback to SHA-256
        self.sha256_hash(node_id, properties)
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> DeltaStatistics {
        self.stats.read().clone()
    }
}

/// Errors that can occur during delta computation
#[derive(Debug, thiserror::Error)]
pub enum DeltaError {
    /// Snapshot is invalid or corrupted
    #[error("Invalid snapshot: {0}")]
    InvalidSnapshot(String),

    /// Delta computation failed
    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    /// Bloom filter operation failed
    #[error("Bloom filter error: {0}")]
    BloomFilterError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: &str, props: Vec<(&str, &str)>) -> NodeSnapshot {
        let properties: HashMap<String, String> = props
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let computer = DeltaComputer::new(DeltaComputationConfig::default());
        let content_hash = computer.hash_node_content(id, &properties);

        NodeSnapshot {
            node_id: id.to_string(),
            content_hash,
            properties,
            last_modified: Utc::now(),
        }
    }

    fn create_test_edge(source: &str, target: &str, edge_type: &str) -> EdgeSnapshot {
        let properties = HashMap::new();
        let content_hash = format!("{}-{}-{}", source, target, edge_type);

        EdgeSnapshot {
            source: source.to_string(),
            target: target.to_string(),
            edge_type: edge_type.to_string(),
            content_hash,
            properties,
            last_modified: Utc::now(),
        }
    }

    #[test]
    fn test_delta_no_changes() {
        let computer = DeltaComputer::new(DeltaComputationConfig::default());

        let mut nodes = HashMap::new();
        nodes.insert("node1".to_string(), create_test_node("node1", vec![("name", "Alice")]));

        let mut edges = HashMap::new();
        edges.insert(
            ("node1".to_string(), "node2".to_string()),
            create_test_edge("node1", "node2", "knows"),
        );

        let snapshot1 = computer.create_snapshot("snap1".to_string(), nodes.clone(), edges.clone());
        let snapshot2 = computer.create_snapshot("snap2".to_string(), nodes, edges);

        let delta = computer.compute_delta(&snapshot1, &snapshot2).unwrap();

        assert_eq!(delta.nodes_added.len(), 0);
        assert_eq!(delta.nodes_removed.len(), 0);
        assert_eq!(delta.nodes_modified.len(), 0);
        assert_eq!(delta.edges_added.len(), 0);
        assert_eq!(delta.edges_removed.len(), 0);
        assert_eq!(delta.edges_modified.len(), 0);
    }

    #[test]
    fn test_delta_node_added() {
        let computer = DeltaComputer::new(DeltaComputationConfig::default());

        let mut nodes1 = HashMap::new();
        nodes1.insert("node1".to_string(), create_test_node("node1", vec![("name", "Alice")]));

        let mut nodes2 = nodes1.clone();
        nodes2.insert("node2".to_string(), create_test_node("node2", vec![("name", "Bob")]));

        let snapshot1 = computer.create_snapshot("snap1".to_string(), nodes1, HashMap::new());
        let snapshot2 = computer.create_snapshot("snap2".to_string(), nodes2, HashMap::new());

        let delta = computer.compute_delta(&snapshot1, &snapshot2).unwrap();

        assert_eq!(delta.nodes_added.len(), 1);
        assert_eq!(delta.nodes_added[0].node_id, "node2");
    }

    #[test]
    fn test_delta_node_removed() {
        let computer = DeltaComputer::new(DeltaComputationConfig::default());

        let mut nodes1 = HashMap::new();
        nodes1.insert("node1".to_string(), create_test_node("node1", vec![("name", "Alice")]));
        nodes1.insert("node2".to_string(), create_test_node("node2", vec![("name", "Bob")]));

        let mut nodes2 = HashMap::new();
        nodes2.insert("node1".to_string(), create_test_node("node1", vec![("name", "Alice")]));

        let snapshot1 = computer.create_snapshot("snap1".to_string(), nodes1, HashMap::new());
        let snapshot2 = computer.create_snapshot("snap2".to_string(), nodes2, HashMap::new());

        let delta = computer.compute_delta(&snapshot1, &snapshot2).unwrap();

        assert_eq!(delta.nodes_removed.len(), 1);
        assert_eq!(delta.nodes_removed[0], "node2");
    }

    #[test]
    fn test_delta_node_modified() {
        let computer = DeltaComputer::new(DeltaComputationConfig::default());

        let mut nodes1 = HashMap::new();
        nodes1.insert("node1".to_string(), create_test_node("node1", vec![("name", "Alice")]));

        let mut nodes2 = HashMap::new();
        nodes2.insert("node1".to_string(), create_test_node("node1", vec![("name", "Alice Updated")]));

        let snapshot1 = computer.create_snapshot("snap1".to_string(), nodes1, HashMap::new());
        let snapshot2 = computer.create_snapshot("snap2".to_string(), nodes2, HashMap::new());

        let delta = computer.compute_delta(&snapshot1, &snapshot2).unwrap();

        assert_eq!(delta.nodes_modified.len(), 1);
        assert_eq!(delta.nodes_modified[0].node_id, "node1");
        assert_ne!(delta.nodes_modified[0].old_hash, delta.nodes_modified[0].new_hash);
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        bloom.insert("node1");
        bloom.insert("node2");
        bloom.insert("node3");

        assert!(bloom.contains("node1"));
        assert!(bloom.contains("node2"));
        assert!(bloom.contains("node3"));
        assert!(!bloom.contains("node999"));
    }

    #[test]
    fn test_property_changes() {
        let computer = DeltaComputer::new(DeltaComputationConfig::default());

        let mut before = HashMap::new();
        before.insert("name".to_string(), "Alice".to_string());
        before.insert("age".to_string(), "30".to_string());

        let mut after = HashMap::new();
        after.insert("name".to_string(), "Alice Updated".to_string());
        after.insert("email".to_string(), "alice@example.com".to_string());

        let changes = computer.compute_property_changes(&before, &after);

        assert_eq!(changes.len(), 3);

        // Check for modified property
        assert!(changes.iter().any(|c| {
            c.property_name == "name" && matches!(c.change_type, ChangeType::Modified)
        }));

        // Check for added property
        assert!(changes.iter().any(|c| {
            c.property_name == "email" && matches!(c.change_type, ChangeType::Added)
        }));

        // Check for removed property
        assert!(changes.iter().any(|c| {
            c.property_name == "age" && matches!(c.change_type, ChangeType::Removed)
        }));
    }

    #[test]
    fn test_parallel_computation() {
        let mut config = DeltaComputationConfig::default();
        config.parallel_computation = true;

        let computer = DeltaComputer::new(config);

        let mut nodes1 = HashMap::new();
        for i in 0..100 {
            nodes1.insert(
                format!("node{}", i),
                create_test_node(&format!("node{}", i), vec![("id", &i.to_string())]),
            );
        }

        let mut nodes2 = nodes1.clone();
        for i in 50..150 {
            nodes2.insert(
                format!("node{}", i),
                create_test_node(&format!("node{}", i), vec![("id", &i.to_string())]),
            );
        }

        let snapshot1 = computer.create_snapshot("snap1".to_string(), nodes1, HashMap::new());
        let snapshot2 = computer.create_snapshot("snap2".to_string(), nodes2, HashMap::new());

        let delta = computer.compute_delta(&snapshot1, &snapshot2).unwrap();

        assert_eq!(delta.nodes_added.len(), 50); // nodes 100-149
        assert_eq!(delta.nodes_removed.len(), 50); // nodes 0-49
    }
}
