//! Force-Directed Graph Layout Algorithm
//!
//! Implements a simple force-directed layout using:
//! - Repulsive forces between all nodes (Coulomb's law)
//! - Attractive forces along edges (Hooke's law)
//! - Velocity damping for stability
//!
//! Based on the Fruchterman-Reingold algorithm.

use std::collections::HashMap;

/// 2D vector for positions and forces
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

#[allow(dead_code)]
impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self {
                x: self.x / mag,
                y: self.y / mag,
            }
        } else {
            *self
        }
    }

    pub fn add(&self, other: &Vec2) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn sub(&self, other: &Vec2) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn mul(&self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

/// Node in the force-directed layout
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayoutNode {
    pub id: String,
    pub position: Vec2,
    pub velocity: Vec2,
    pub force: Vec2,
}

#[allow(dead_code)]
impl LayoutNode {
    pub fn new(id: String, x: f64, y: f64) -> Self {
        Self {
            id,
            position: Vec2::new(x, y),
            velocity: Vec2::default(),
            force: Vec2::default(),
        }
    }
}

/// Edge in the force-directed layout
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayoutEdge {
    pub source: String,
    pub target: String,
}

#[allow(dead_code)]
impl LayoutEdge {
    pub fn new(source: String, target: String) -> Self {
        Self { source, target }
    }
}

/// Configuration for force-directed layout
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayoutConfig {
    /// Width of the layout area
    pub width: f64,
    /// Height of the layout area
    pub height: f64,
    /// Repulsive force constant
    pub repulsion: f64,
    /// Attractive force constant
    pub attraction: f64,
    /// Damping factor (0.0 to 1.0)
    pub damping: f64,
    /// Time step for simulation
    pub dt: f64,
    /// Minimum movement threshold for convergence
    pub min_movement: f64,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            repulsion: 5000.0,
            attraction: 0.1,
            damping: 0.8,
            dt: 0.01,
            min_movement: 0.1,
        }
    }
}

/// Force-directed layout engine
#[allow(dead_code)]
pub struct ForceLayout {
    nodes: HashMap<String, LayoutNode>,
    edges: Vec<LayoutEdge>,
    config: LayoutConfig,
}

#[allow(dead_code)]
impl ForceLayout {
    /// Create a new force-directed layout
    pub fn new(config: LayoutConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            config,
        }
    }

    /// Add a node to the layout
    pub fn add_node(&mut self, id: String) {
        if !self.nodes.contains_key(&id) {
            // Random initial position
            let x = (js_sys::Math::random() - 0.5) * self.config.width;
            let y = (js_sys::Math::random() - 0.5) * self.config.height;
            self.nodes.insert(id.clone(), LayoutNode::new(id, x, y));
        }
    }

    /// Add an edge to the layout
    pub fn add_edge(&mut self, source: String, target: String) {
        self.edges.push(LayoutEdge::new(source, target));
    }

    /// Calculate repulsive forces between all nodes
    fn calculate_repulsion(&mut self) {
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let id_a = &node_ids[i];
                let id_b = &node_ids[j];

                let pos_a = self.nodes[id_a].position;
                let pos_b = self.nodes[id_b].position;

                let delta = pos_a.sub(&pos_b);
                let distance = delta.magnitude().max(10.0); // Avoid division by zero

                let force_magnitude = self.config.repulsion / (distance * distance);
                let force = delta.normalize().mul(force_magnitude);

                // Apply force to both nodes
                if let Some(node_a) = self.nodes.get_mut(id_a) {
                    node_a.force = node_a.force.add(&force);
                }
                if let Some(node_b) = self.nodes.get_mut(id_b) {
                    node_b.force = node_b.force.sub(&force);
                }
            }
        }
    }

    /// Calculate attractive forces along edges
    fn calculate_attraction(&mut self) {
        for edge in &self.edges {
            if let (Some(source_node), Some(target_node)) = (
                self.nodes.get(&edge.source),
                self.nodes.get(&edge.target),
            ) {
                let delta = target_node.position.sub(&source_node.position);
                let distance = delta.magnitude().max(1.0);

                let force_magnitude = self.config.attraction * distance;
                let force = delta.normalize().mul(force_magnitude);

                // Apply force to both nodes
                if let Some(node) = self.nodes.get_mut(&edge.source) {
                    node.force = node.force.add(&force);
                }
                if let Some(node) = self.nodes.get_mut(&edge.target) {
                    node.force = node.force.sub(&force);
                }
            }
        }
    }

    /// Update node positions based on forces
    fn update_positions(&mut self) {
        for node in self.nodes.values_mut() {
            // Update velocity with damping
            node.velocity = node.velocity.mul(self.config.damping).add(&node.force.mul(self.config.dt));

            // Update position
            node.position = node.position.add(&node.velocity.mul(self.config.dt));

            // Constrain to bounds
            let margin = 50.0;
            let half_width = self.config.width / 2.0;
            let half_height = self.config.height / 2.0;

            node.position.x = node.position.x.max(-half_width + margin).min(half_width - margin);
            node.position.y = node.position.y.max(-half_height + margin).min(half_height - margin);

            // Reset force
            node.force = Vec2::default();
        }
    }

    /// Check if layout has converged
    fn has_converged(&self) -> bool {
        let total_movement: f64 = self.nodes.values()
            .map(|node| node.velocity.magnitude())
            .sum();

        let avg_movement = total_movement / (self.nodes.len() as f64).max(1.0);
        avg_movement < self.config.min_movement
    }

    /// Run one iteration of the layout algorithm
    pub fn step(&mut self) {
        self.calculate_repulsion();
        self.calculate_attraction();
        self.update_positions();
    }

    /// Run the layout algorithm for a fixed number of iterations
    pub fn run(&mut self, max_iterations: usize) {
        for _ in 0..max_iterations {
            self.step();
            if self.has_converged() {
                break;
            }
        }
    }

    /// Get node positions
    pub fn get_positions(&self) -> HashMap<String, (f64, f64)> {
        self.nodes.iter()
            .map(|(id, node)| (id.clone(), (node.position.x, node.position.y)))
            .collect()
    }

    /// Get a specific node position
    pub fn get_position(&self, id: &str) -> Option<(f64, f64)> {
        self.nodes.get(id).map(|node| (node.position.x, node.position.y))
    }

    /// Clear all nodes and edges
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_operations() {
        let v1 = Vec2::new(3.0, 4.0);
        assert_eq!(v1.magnitude(), 5.0);

        let v2 = Vec2::new(1.0, 0.0);
        let normalized = v2.normalize();
        assert_eq!(normalized.magnitude(), 1.0);
    }

    #[test]
    fn test_force_layout_creation() {
        let config = LayoutConfig::default();
        let layout = ForceLayout::new(config);
        assert_eq!(layout.nodes.len(), 0);
        assert_eq!(layout.edges.len(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let config = LayoutConfig::default();
        let mut layout = ForceLayout::new(config);

        layout.add_node("node1".to_string());
        layout.add_node("node2".to_string());

        assert_eq!(layout.nodes.len(), 2);
        assert!(layout.get_position("node1").is_some());
        assert!(layout.get_position("node2").is_some());
    }

    #[test]
    fn test_add_edges() {
        let config = LayoutConfig::default();
        let mut layout = ForceLayout::new(config);

        layout.add_node("node1".to_string());
        layout.add_node("node2".to_string());
        layout.add_edge("node1".to_string(), "node2".to_string());

        assert_eq!(layout.edges.len(), 1);
    }
}
