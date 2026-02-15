//! Temporal Graph Analysis
//!
//! This module provides analysis capabilities for time-evolving graphs:
//! - Temporal graph representation
//! - Snapshot-based analysis
//! - Evolution metrics
//! - Temporal community detection
//! - Time-aware path finding
//!
//! ## Use Cases
//!
//! - Social network evolution tracking
//! - Knowledge graph versioning
//! - Anomaly detection in dynamic networks
//! - Trend analysis and forecasting
//! - Event detection in temporal data

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};

/// Time range for temporal validity
///
/// Represents when an entity exists or a relationship is valid in the real world.
/// Uses Unix timestamps (seconds since epoch). Special values:
/// - i64::MIN represents unknown/unspecified start
/// - i64::MAX represents ongoing/current (no end)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalRange {
    /// Start of validity (Unix timestamp)
    pub start: i64,
    /// End of validity (Unix timestamp)
    pub end: i64,
}

impl TemporalRange {
    /// Create a new temporal range
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    /// Check if a timestamp falls within this range
    pub fn contains(&self, timestamp: i64) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &TemporalRange) -> bool {
        self.start <= other.end && self.end >= other.start
    }

    /// Get the duration of this range in seconds
    pub fn duration(&self) -> i64 {
        self.end.saturating_sub(self.start)
    }
}

/// Type of temporal relationship between entities
///
/// Defines how entities are related in time, including causal relationships.
/// Based on Allen's Interval Algebra and causal reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalRelationType {
    /// A occurred before B (temporal precedence)
    Before,
    /// A occurred during B (temporal containment)
    During,
    /// A occurred after B (temporal succession)
    After,
    /// A and B occurred simultaneously
    SimultaneousWith,
    /// A directly caused B (strong causal link)
    Caused,
    /// A enabled or facilitated B (weak causal link)
    Enabled,
    /// A prevented or inhibited B (negative causal link)
    Prevented,
    /// A and B mutually influenced each other
    Correlated,
}

impl TemporalRelationType {
    /// Check if this is a causal relationship type
    pub fn is_causal(&self) -> bool {
        matches!(
            self,
            TemporalRelationType::Caused
                | TemporalRelationType::Enabled
                | TemporalRelationType::Prevented
        )
    }

    /// Get the strength weight for this relationship type
    pub fn default_strength(&self) -> f32 {
        match self {
            TemporalRelationType::Caused => 0.9,
            TemporalRelationType::Enabled => 0.6,
            TemporalRelationType::Prevented => 0.7,
            TemporalRelationType::Correlated => 0.5,
            _ => 0.3, // Temporal-only relationships have lower default strength
        }
    }
}

/// Temporal edge with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    /// Source node
    pub source: String,
    /// Target node
    pub target: String,
    /// Edge type/label
    pub edge_type: String,
    /// Timestamp (Unix timestamp)
    pub timestamp: i64,
    /// Edge weight
    pub weight: f32,
    /// Start time (optional, for interval-based edges)
    pub start_time: Option<i64>,
    /// End time (optional, for interval-based edges)
    pub end_time: Option<i64>,
}

impl TemporalEdge {
    /// Check if edge is active at given timestamp
    pub fn is_active_at(&self, timestamp: i64) -> bool {
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            timestamp >= start && timestamp <= end
        } else {
            // Point-in-time edge
            self.timestamp == timestamp
        }
    }

    /// Check if edge is active in time range
    pub fn is_active_in_range(&self, start: i64, end: i64) -> bool {
        if let (Some(edge_start), Some(edge_end)) = (self.start_time, self.end_time) {
            // Interval overlap check
            edge_start <= end && edge_end >= start
        } else {
            // Point-in-time edge
            self.timestamp >= start && self.timestamp <= end
        }
    }
}

/// Graph snapshot at specific time
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Snapshot timestamp
    pub timestamp: i64,
    /// Nodes active in this snapshot
    pub nodes: HashSet<String>,
    /// Edges active in this snapshot
    pub edges: Vec<TemporalEdge>,
    /// Edge count
    pub edge_count: usize,
    /// Node count
    pub node_count: usize,
}

impl Snapshot {
    /// Create snapshot from temporal edges
    pub fn from_edges(timestamp: i64, edges: Vec<TemporalEdge>) -> Self {
        let mut nodes = HashSet::new();

        for edge in &edges {
            nodes.insert(edge.source.clone());
            nodes.insert(edge.target.clone());
        }

        let node_count = nodes.len();
        let edge_count = edges.len();

        Self {
            timestamp,
            nodes,
            edges,
            edge_count,
            node_count,
        }
    }

    /// Get node degree in snapshot
    pub fn node_degree(&self, node: &str) -> usize {
        self.edges
            .iter()
            .filter(|e| e.source == node || e.target == node)
            .count()
    }

    /// Get graph density
    pub fn density(&self) -> f32 {
        if self.node_count < 2 {
            return 0.0;
        }

        let max_edges = (self.node_count * (self.node_count - 1)) / 2;
        self.edge_count as f32 / max_edges as f32
    }
}

/// Temporal graph
pub struct TemporalGraph {
    /// All temporal edges
    edges: Vec<TemporalEdge>,
    /// Edge index by timestamp
    edge_index: BTreeMap<i64, Vec<usize>>,
    /// Node first appearance time
    node_first_seen: HashMap<String, i64>,
    /// Node last seen time
    node_last_seen: HashMap<String, i64>,
}

impl TemporalGraph {
    /// Create new temporal graph
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            edge_index: BTreeMap::new(),
            node_first_seen: HashMap::new(),
            node_last_seen: HashMap::new(),
        }
    }

    /// Add temporal edge
    pub fn add_edge(&mut self, edge: TemporalEdge) {
        let timestamp = edge.timestamp;
        let edge_idx = self.edges.len();

        // Update node timestamps
        self.update_node_timestamp(&edge.source, timestamp);
        self.update_node_timestamp(&edge.target, timestamp);

        // Add to edge index
        self.edge_index
            .entry(timestamp)
            .or_default()
            .push(edge_idx);

        self.edges.push(edge);
    }

    /// Update node first/last seen timestamps
    fn update_node_timestamp(&mut self, node: &str, timestamp: i64) {
        self.node_first_seen
            .entry(node.to_string())
            .and_modify(|t| *t = (*t).min(timestamp))
            .or_insert(timestamp);

        self.node_last_seen
            .entry(node.to_string())
            .and_modify(|t| *t = (*t).max(timestamp))
            .or_insert(timestamp);
    }

    /// Get snapshot at specific timestamp
    pub fn snapshot_at(&self, timestamp: i64) -> Snapshot {
        let edges: Vec<TemporalEdge> = self
            .edges
            .iter()
            .filter(|e| e.is_active_at(timestamp))
            .cloned()
            .collect();

        Snapshot::from_edges(timestamp, edges)
    }

    /// Get snapshot for time range
    pub fn snapshot_range(&self, start: i64, end: i64) -> Snapshot {
        let edges: Vec<TemporalEdge> = self
            .edges
            .iter()
            .filter(|e| e.is_active_in_range(start, end))
            .cloned()
            .collect();

        Snapshot::from_edges((start + end) / 2, edges)
    }

    /// Get all timestamps (discrete time points)
    pub fn timestamps(&self) -> Vec<i64> {
        self.edge_index.keys().copied().collect()
    }

    /// Get time range
    pub fn time_range(&self) -> Option<(i64, i64)> {
        if self.edges.is_empty() {
            return None;
        }

        let min = self.edges.iter().map(|e| e.timestamp).min().unwrap();
        let max = self.edges.iter().map(|e| e.timestamp).max().unwrap();

        Some((min, max))
    }

    /// Get node lifetime
    pub fn node_lifetime(&self, node: &str) -> Option<(i64, i64)> {
        let first = self.node_first_seen.get(node)?;
        let last = self.node_last_seen.get(node)?;

        Some((*first, *last))
    }

    /// Get all nodes
    pub fn nodes(&self) -> HashSet<String> {
        self.node_first_seen.keys().cloned().collect()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.node_first_seen.len()
    }
}

impl Default for TemporalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQuery {
    /// Start timestamp
    pub start_time: i64,
    /// End timestamp
    pub end_time: i64,
    /// Granularity (e.g., daily, weekly)
    pub granularity: i64,
    /// Node filter (optional)
    pub nodes: Option<Vec<String>>,
    /// Edge type filter (optional)
    pub edge_types: Option<Vec<String>>,
}

/// Temporal analytics engine
pub struct TemporalAnalytics {
    graph: TemporalGraph,
}

impl TemporalAnalytics {
    /// Create analytics engine
    pub fn new(graph: TemporalGraph) -> Self {
        Self { graph }
    }

    /// Calculate evolution metrics over time
    pub fn evolution_metrics(&self, query: &TemporalQuery) -> Vec<EvolutionMetrics> {
        let mut metrics = Vec::new();
        let mut current_time = query.start_time;

        while current_time <= query.end_time {
            let next_time = current_time + query.granularity;
            let snapshot = self.graph.snapshot_range(current_time, next_time);

            let metric = EvolutionMetrics {
                timestamp: current_time,
                node_count: snapshot.node_count,
                edge_count: snapshot.edge_count,
                density: snapshot.density(),
                avg_degree: self.calculate_avg_degree(&snapshot),
            };

            metrics.push(metric);
            current_time = next_time;
        }

        metrics
    }

    /// Calculate average node degree in snapshot
    fn calculate_avg_degree(&self, snapshot: &Snapshot) -> f32 {
        if snapshot.node_count == 0 {
            return 0.0;
        }

        let total_degree: usize = snapshot
            .nodes
            .iter()
            .map(|n| snapshot.node_degree(n))
            .sum();

        total_degree as f32 / snapshot.node_count as f32
    }

    /// Detect node churn (nodes appearing/disappearing)
    pub fn node_churn(&self, query: &TemporalQuery) -> NodeChurn {
        let start_snapshot = self.graph.snapshot_at(query.start_time);
        let end_snapshot = self.graph.snapshot_at(query.end_time);

        let added: HashSet<_> = end_snapshot
            .nodes
            .difference(&start_snapshot.nodes)
            .cloned()
            .collect();

        let removed: HashSet<_> = start_snapshot
            .nodes
            .difference(&end_snapshot.nodes)
            .cloned()
            .collect();

        let stable: HashSet<_> = start_snapshot
            .nodes
            .intersection(&end_snapshot.nodes)
            .cloned()
            .collect();

        let added_count = added.len();
        let removed_count = removed.len();
        let stable_count = stable.len();

        NodeChurn {
            added: added.into_iter().collect(),
            removed: removed.into_iter().collect(),
            stable: stable.into_iter().collect(),
            added_count,
            removed_count,
            stable_count,
        }
    }

    /// Find nodes with highest activity growth
    pub fn top_growing_nodes(&self, query: &TemporalQuery, top_k: usize) -> Vec<(String, f32)> {
        let start_snapshot = self.graph.snapshot_range(query.start_time, query.start_time + query.granularity);
        let end_snapshot = self.graph.snapshot_range(query.end_time - query.granularity, query.end_time);

        let mut growth_scores: Vec<(String, f32)> = Vec::new();

        for node in &end_snapshot.nodes {
            let start_degree = start_snapshot.node_degree(node) as f32;
            let end_degree = end_snapshot.node_degree(node) as f32;

            let growth = if start_degree > 0.0 {
                (end_degree - start_degree) / start_degree
            } else {
                end_degree
            };

            growth_scores.push((node.clone(), growth));
        }

        growth_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        growth_scores.truncate(top_k);

        growth_scores
    }

    /// Get temporal centrality (activity over time)
    pub fn temporal_centrality(&self, node: &str, query: &TemporalQuery) -> Vec<(i64, f32)> {
        let mut centrality = Vec::new();
        let mut current_time = query.start_time;

        while current_time <= query.end_time {
            let next_time = current_time + query.granularity;
            let snapshot = self.graph.snapshot_range(current_time, next_time);

            let degree = snapshot.node_degree(node) as f32;
            let centrality_score = if snapshot.node_count > 1 {
                degree / (snapshot.node_count - 1) as f32
            } else {
                0.0
            };

            centrality.push((current_time, centrality_score));
            current_time = next_time;
        }

        centrality
    }
}

/// Evolution metrics over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    /// Timestamp
    pub timestamp: i64,
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Graph density
    pub density: f32,
    /// Average degree
    pub avg_degree: f32,
}

/// Node churn analysis
#[derive(Debug, Clone)]
pub struct NodeChurn {
    /// Nodes added
    pub added: Vec<String>,
    /// Nodes removed
    pub removed: Vec<String>,
    /// Stable nodes
    pub stable: Vec<String>,
    /// Count of added nodes
    pub added_count: usize,
    /// Count of removed nodes
    pub removed_count: usize,
    /// Count of stable nodes
    pub stable_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_temporal_graph() -> TemporalGraph {
        let mut graph = TemporalGraph::new();

        // Add edges over time
        graph.add_edge(TemporalEdge {
            source: "A".to_string(),
            target: "B".to_string(),
            edge_type: "knows".to_string(),
            timestamp: 100,
            weight: 1.0,
            start_time: Some(100),
            end_time: Some(200),
        });

        graph.add_edge(TemporalEdge {
            source: "B".to_string(),
            target: "C".to_string(),
            edge_type: "knows".to_string(),
            timestamp: 150,
            weight: 1.0,
            start_time: Some(150),
            end_time: Some(250),
        });

        graph.add_edge(TemporalEdge {
            source: "A".to_string(),
            target: "C".to_string(),
            edge_type: "knows".to_string(),
            timestamp: 200,
            weight: 1.0,
            start_time: Some(200),
            end_time: Some(300),
        });

        graph
    }

    #[test]
    fn test_temporal_graph_creation() {
        let graph = create_test_temporal_graph();
        assert_eq!(graph.edge_count(), 3);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_snapshot_at_timestamp() {
        let graph = create_test_temporal_graph();
        let snapshot = graph.snapshot_at(150);

        assert!(snapshot.node_count > 0);
        assert!(snapshot.edge_count > 0);
    }

    #[test]
    fn test_snapshot_range() {
        let graph = create_test_temporal_graph();
        let snapshot = graph.snapshot_range(100, 200);

        assert_eq!(snapshot.node_count, 3);
        assert!(snapshot.edge_count >= 2);
    }

    #[test]
    fn test_time_range() {
        let graph = create_test_temporal_graph();
        let (min, max) = graph.time_range().unwrap();

        assert_eq!(min, 100);
        assert_eq!(max, 200);
    }

    #[test]
    fn test_node_lifetime() {
        let graph = create_test_temporal_graph();
        let (first, last) = graph.node_lifetime("A").unwrap();

        assert_eq!(first, 100);
        assert_eq!(last, 200);
    }

    #[test]
    fn test_evolution_metrics() {
        let graph = create_test_temporal_graph();
        let analytics = TemporalAnalytics::new(graph);

        let query = TemporalQuery {
            start_time: 100,
            end_time: 300,
            granularity: 50,
            nodes: None,
            edge_types: None,
        };

        let metrics = analytics.evolution_metrics(&query);
        assert!(!metrics.is_empty());

        for metric in &metrics {
            assert!(metric.timestamp >= 100);
            assert!(metric.timestamp <= 300);
        }
    }

    #[test]
    fn test_node_churn() {
        let mut graph = TemporalGraph::new();

        // Initial nodes: A, B
        graph.add_edge(TemporalEdge {
            source: "A".to_string(),
            target: "B".to_string(),
            edge_type: "knows".to_string(),
            timestamp: 100,
            weight: 1.0,
            start_time: None,
            end_time: None,
        });

        // Later: B, C (A removed, C added)
        graph.add_edge(TemporalEdge {
            source: "B".to_string(),
            target: "C".to_string(),
            edge_type: "knows".to_string(),
            timestamp: 200,
            weight: 1.0,
            start_time: None,
            end_time: None,
        });

        let analytics = TemporalAnalytics::new(graph);
        let query = TemporalQuery {
            start_time: 100,
            end_time: 200,
            granularity: 50,
            nodes: None,
            edge_types: None,
        };

        let churn = analytics.node_churn(&query);
        assert!(churn.added.contains(&"C".to_string()) || churn.stable_count > 0);
    }

    #[test]
    fn test_temporal_edge_is_active() {
        let edge = TemporalEdge {
            source: "A".to_string(),
            target: "B".to_string(),
            edge_type: "knows".to_string(),
            timestamp: 100,
            weight: 1.0,
            start_time: Some(100),
            end_time: Some(200),
        };

        assert!(edge.is_active_at(150));
        assert!(edge.is_active_at(100));
        assert!(edge.is_active_at(200));
        assert!(!edge.is_active_at(50));
        assert!(!edge.is_active_at(250));

        assert!(edge.is_active_in_range(90, 110));
        assert!(edge.is_active_in_range(150, 250));
        assert!(!edge.is_active_in_range(50, 90));
    }

    // Phase 1.2: Temporal Fields Tests
    #[test]
    fn test_temporal_range_creation() {
        let range = TemporalRange::new(100, 200);
        assert_eq!(range.start, 100);
        assert_eq!(range.end, 200);
    }

    #[test]
    fn test_temporal_range_contains() {
        let range = TemporalRange::new(100, 200);

        assert!(range.contains(100));
        assert!(range.contains(150));
        assert!(range.contains(200));
        assert!(!range.contains(50));
        assert!(!range.contains(250));
    }

    #[test]
    fn test_temporal_range_overlaps() {
        let range1 = TemporalRange::new(100, 200);
        let range2 = TemporalRange::new(150, 250);
        let range3 = TemporalRange::new(250, 300);

        assert!(range1.overlaps(&range2));
        assert!(range2.overlaps(&range1));
        assert!(!range1.overlaps(&range3));
        assert!(!range3.overlaps(&range1));
    }

    #[test]
    fn test_temporal_range_duration() {
        let range = TemporalRange::new(100, 200);
        assert_eq!(range.duration(), 100);

        let instant = TemporalRange::new(100, 100);
        assert_eq!(instant.duration(), 0);
    }

    #[test]
    fn test_temporal_range_serialization() {
        let range = TemporalRange::new(100, 200);
        let json = serde_json::to_string(&range).unwrap();
        assert!(json.contains("100"));
        assert!(json.contains("200"));

        let deserialized: TemporalRange = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.start, 100);
        assert_eq!(deserialized.end, 200);
    }

    #[test]
    fn test_temporal_relation_type_is_causal() {
        assert!(TemporalRelationType::Caused.is_causal());
        assert!(TemporalRelationType::Enabled.is_causal());
        assert!(TemporalRelationType::Prevented.is_causal());

        assert!(!TemporalRelationType::Before.is_causal());
        assert!(!TemporalRelationType::During.is_causal());
        assert!(!TemporalRelationType::After.is_causal());
        assert!(!TemporalRelationType::SimultaneousWith.is_causal());
    }

    #[test]
    fn test_temporal_relation_type_default_strength() {
        assert_eq!(TemporalRelationType::Caused.default_strength(), 0.9);
        assert_eq!(TemporalRelationType::Enabled.default_strength(), 0.6);
        assert_eq!(TemporalRelationType::Prevented.default_strength(), 0.7);
        assert_eq!(TemporalRelationType::Correlated.default_strength(), 0.5);

        // Non-causal should have lower strength
        assert!(TemporalRelationType::Before.default_strength() < 0.5);
    }

    #[test]
    fn test_temporal_relation_type_serialization() {
        let rel_type = TemporalRelationType::Caused;
        let json = serde_json::to_string(&rel_type).unwrap();
        assert!(json.contains("Caused"));

        let deserialized: TemporalRelationType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, TemporalRelationType::Caused);
    }
}
