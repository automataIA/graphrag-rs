//! Causal Chain Analysis (Phase 2.3)
//!
//! This module implements causal chain discovery and temporal validation.
//! It finds paths between cause and effect entities, validating temporal ordering.
//!
//! Example: Query "What caused the fall of Athens?" should find:
//! Plague → Weakened Athens → Sparta attacked → Athens fell
//!
//! Each step is validated for temporal consistency (t1 < t2 < t3).

use crate::{
    core::{EntityId, KnowledgeGraph, Relationship, Result},
    graph::temporal::{TemporalRange, TemporalRelationType},
};
use std::collections::VecDeque;
use std::sync::Arc;

/// A complete causal chain from cause to effect
///
/// Represents a series of causal steps connecting two entities.
/// All steps must be temporally ordered (earlier causes → later effects).
#[derive(Debug, Clone)]
pub struct CausalChain {
    /// Starting entity (the cause)
    pub cause: EntityId,

    /// Ending entity (the effect)
    pub effect: EntityId,

    /// Intermediate causal steps
    pub steps: Vec<CausalStep>,

    /// Overall confidence of the chain (product of step confidences)
    pub total_confidence: f32,

    /// Whether the chain is temporally consistent (all steps ordered correctly)
    pub temporal_consistency: bool,

    /// Total time span of the chain (if temporal data available)
    pub time_span: Option<i64>,
}

/// A single step in a causal chain
///
/// Represents one causal relationship in a chain.
#[derive(Debug, Clone)]
pub struct CausalStep {
    /// Source entity of this step
    pub source: EntityId,

    /// Target entity of this step
    pub target: EntityId,

    /// Type of causal relationship
    pub relation_type: String,

    /// Temporal type (Caused, Enabled, etc.)
    pub temporal_type: Option<TemporalRelationType>,

    /// When this step occurred
    pub temporal_range: Option<TemporalRange>,

    /// Confidence of this causal link
    pub confidence: f32,

    /// Strength of causality (0.0-1.0)
    pub causal_strength: Option<f32>,
}

impl CausalStep {
    /// Create a causal step from a relationship
    pub fn from_relationship(rel: &Relationship) -> Self {
        Self {
            source: rel.source.clone(),
            target: rel.target.clone(),
            relation_type: rel.relation_type.clone(),
            temporal_type: rel.temporal_type,
            temporal_range: rel.temporal_range,
            confidence: rel.confidence,
            causal_strength: rel.causal_strength,
        }
    }

    /// Check if this step has temporal information
    pub fn has_temporal_info(&self) -> bool {
        self.temporal_range.is_some()
    }

    /// Get the midpoint timestamp of this step (for ordering)
    pub fn get_timestamp(&self) -> Option<i64> {
        self.temporal_range.map(|tr| (tr.start + tr.end) / 2)
    }
}

impl CausalChain {
    /// Calculate the total confidence of the chain
    ///
    /// Uses product of step confidences, weighted by causal strengths
    pub fn calculate_confidence(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }

        let mut product = 1.0;
        for step in &self.steps {
            // Weight confidence by causal strength if available
            let weighted_confidence = if let Some(strength) = step.causal_strength {
                step.confidence * (0.5 + 0.5 * strength) // Range: 0.5*conf to 1.0*conf
            } else {
                step.confidence * 0.7 // Default weight for non-causal
            };
            product *= weighted_confidence;
        }

        product
    }

    /// Check temporal consistency of the chain
    ///
    /// Returns true if all steps are temporally ordered (t1 < t2 < t3...)
    pub fn check_temporal_consistency(&self) -> bool {
        let mut prev_timestamp: Option<i64> = None;

        for step in &self.steps {
            if let Some(current_ts) = step.get_timestamp() {
                if let Some(prev_ts) = prev_timestamp {
                    // Check if current step happened after previous
                    if current_ts < prev_ts {
                        return false; // Temporal violation
                    }
                }
                prev_timestamp = Some(current_ts);
            }
        }

        true
    }

    /// Calculate the time span of the chain
    pub fn calculate_time_span(&self) -> Option<i64> {
        let first_timestamp = self.steps.first()?.get_timestamp()?;
        let last_timestamp = self.steps.last()?.get_timestamp()?;

        Some(last_timestamp - first_timestamp)
    }

    /// Get a human-readable description of the chain
    pub fn describe(&self) -> String {
        let step_descriptions: Vec<String> = self.steps
            .iter()
            .map(|s| format!("{} --[{}]--> {}", s.source.0, s.relation_type, s.target.0))
            .collect();

        format!(
            "Causal chain (conf={:.2}, consistent={}): {}",
            self.total_confidence,
            self.temporal_consistency,
            step_descriptions.join(" → ")
        )
    }
}

/// Analyzer for finding causal chains in the knowledge graph
///
/// Uses depth-first search with temporal validation to find causal paths.
pub struct CausalAnalyzer {
    /// Reference to the knowledge graph
    graph: Arc<KnowledgeGraph>,

    /// Minimum confidence threshold for causal steps
    min_confidence: f32,

    /// Minimum causal strength to consider a relationship causal
    min_causal_strength: f32,

    /// Whether to require temporal consistency
    require_temporal_consistency: bool,
}

impl CausalAnalyzer {
    /// Create a new causal analyzer
    ///
    /// # Arguments
    ///
    /// * `graph` - Reference to the knowledge graph
    pub fn new(graph: Arc<KnowledgeGraph>) -> Self {
        Self {
            graph,
            min_confidence: 0.3,
            min_causal_strength: 0.0, // Accept all relationships by default
            require_temporal_consistency: false, // Lenient by default
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self
    }

    /// Set minimum causal strength threshold
    pub fn with_min_causal_strength(mut self, min_causal_strength: f32) -> Self {
        self.min_causal_strength = min_causal_strength.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable temporal consistency requirement
    pub fn with_temporal_consistency(mut self, required: bool) -> Self {
        self.require_temporal_consistency = required;
        self
    }

    /// Find all causal chains between cause and effect
    ///
    /// # Arguments
    ///
    /// * `cause` - Starting entity ID
    /// * `effect` - Target entity ID
    /// * `max_depth` - Maximum chain length (number of steps)
    ///
    /// # Returns
    ///
    /// Vector of causal chains, sorted by confidence (highest first)
    pub fn find_causal_chains(
        &self,
        cause: &EntityId,
        effect: &EntityId,
        max_depth: usize,
    ) -> Result<Vec<CausalChain>> {
        let mut chains = Vec::new();

        // Use BFS to find all paths
        let all_paths = self.find_all_paths(cause, effect, max_depth)?;

        #[cfg(feature = "tracing")]
        tracing::debug!(
            cause = %cause.0,
            effect = %effect.0,
            paths_found = all_paths.len(),
            "Found potential causal paths"
        );

        // Convert paths to causal chains
        for path in all_paths {
            let mut steps = Vec::new();

            for i in 0..path.len() - 1 {
                let source_id = &path[i];
                let target_id = &path[i + 1];

                // Find the relationship between these entities
                if let Some(rel) = self.find_relationship(source_id, target_id) {
                    // Check if this is a causal relationship
                    if self.is_causal_relationship(&rel) {
                        steps.push(CausalStep::from_relationship(&rel));
                    }
                }
            }

            // Only create chain if we have causal steps
            if !steps.is_empty() {
                let mut chain = CausalChain {
                    cause: cause.clone(),
                    effect: effect.clone(),
                    steps,
                    total_confidence: 0.0,
                    temporal_consistency: false,
                    time_span: None,
                };

                // Calculate properties
                chain.total_confidence = chain.calculate_confidence();
                chain.temporal_consistency = chain.check_temporal_consistency();
                chain.time_span = chain.calculate_time_span();

                // Filter by temporal consistency if required
                if self.require_temporal_consistency && !chain.temporal_consistency {
                    continue;
                }

                chains.push(chain);
            }
        }

        // Sort by confidence (highest first)
        chains.sort_by(|a, b| {
            b.total_confidence
                .partial_cmp(&a.total_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        #[cfg(feature = "tracing")]
        tracing::info!(
            causal_chains = chains.len(),
            "Found valid causal chains"
        );

        Ok(chains)
    }

    /// Find all paths between two entities using BFS
    fn find_all_paths(
        &self,
        start: &EntityId,
        end: &EntityId,
        max_depth: usize,
    ) -> Result<Vec<Vec<EntityId>>> {
        let mut paths = Vec::new();
        let mut queue: VecDeque<(EntityId, Vec<EntityId>)> = VecDeque::new();

        queue.push_back((start.clone(), vec![start.clone()]));

        while let Some((current, path)) = queue.pop_front() {
            // Check depth limit
            if path.len() > max_depth {
                continue;
            }

            // Found the target
            if current == *end {
                paths.push(path);
                continue;
            }

            // Explore neighbors
            for rel in self.graph.get_entity_relationships(&current.0) {
                let next = &rel.target;

                // Avoid cycles
                if path.contains(next) {
                    continue;
                }

                // Check if relationship meets minimum confidence
                if rel.confidence < self.min_confidence {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push(next.clone());
                queue.push_back((next.clone(), new_path));
            }
        }

        Ok(paths)
    }

    /// Find a relationship between two entities
    fn find_relationship(&self, source: &EntityId, target: &EntityId) -> Option<Relationship> {
        self.graph
            .get_entity_relationships(&source.0)
            .into_iter()
            .find(|rel| rel.target == *target)
            .cloned()
    }

    /// Check if a relationship is causal
    fn is_causal_relationship(&self, rel: &Relationship) -> bool {
        // Check temporal type
        if let Some(temporal_type) = rel.temporal_type {
            if temporal_type.is_causal() {
                // Check causal strength threshold
                if let Some(strength) = rel.causal_strength {
                    return strength >= self.min_causal_strength;
                }
                return true; // Has causal type but no strength specified
            }
        }

        // Check relation type contains causal keywords
        let relation_lower = rel.relation_type.to_lowercase();
        let causal_keywords = ["caused", "led_to", "resulted_in", "enabled", "triggered"];

        causal_keywords.iter().any(|kw| relation_lower.contains(kw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Entity;

    fn create_test_graph_with_causal_chain() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Create entities: A → B → C (causal chain)
        let entity_a = Entity::new(
            EntityId::new("a".to_string()),
            "Event A".to_string(),
            "EVENT".to_string(),
            0.9,
        );

        let entity_b = Entity::new(
            EntityId::new("b".to_string()),
            "Event B".to_string(),
            "EVENT".to_string(),
            0.9,
        );

        let entity_c = Entity::new(
            EntityId::new("c".to_string()),
            "Event C".to_string(),
            "EVENT".to_string(),
            0.9,
        );

        graph.add_entity(entity_a).unwrap();
        graph.add_entity(entity_b).unwrap();
        graph.add_entity(entity_c).unwrap();

        // A caused B (at time 100)
        let rel_ab = Relationship::new(
            EntityId::new("a".to_string()),
            EntityId::new("b".to_string()),
            "CAUSED".to_string(),
            0.8,
        )
        .with_temporal_type(TemporalRelationType::Caused)
        .with_temporal_range(100, 100)
        .with_causal_strength(0.9);

        // B caused C (at time 200)
        let rel_bc = Relationship::new(
            EntityId::new("b".to_string()),
            EntityId::new("c".to_string()),
            "CAUSED".to_string(),
            0.85,
        )
        .with_temporal_type(TemporalRelationType::Caused)
        .with_temporal_range(200, 200)
        .with_causal_strength(0.95);

        graph.add_relationship(rel_ab).unwrap();
        graph.add_relationship(rel_bc).unwrap();

        graph
    }

    #[test]
    fn test_causal_chain_creation() {
        let graph = Arc::new(create_test_graph_with_causal_chain());
        let analyzer = CausalAnalyzer::new(graph);

        let chains = analyzer
            .find_causal_chains(
                &EntityId::new("a".to_string()),
                &EntityId::new("c".to_string()),
                5,
            )
            .unwrap();

        assert_eq!(chains.len(), 1, "Should find exactly one causal chain");

        let chain = &chains[0];
        assert_eq!(chain.steps.len(), 2, "Chain should have 2 steps (A→B, B→C)");
        assert!(chain.temporal_consistency, "Chain should be temporally consistent");
        assert!(chain.total_confidence > 0.6, "Chain should have reasonable confidence");
    }

    #[test]
    fn test_temporal_consistency_validation() {
        let mut graph = KnowledgeGraph::new();

        // Create entities
        let a = Entity::new(EntityId::new("a".to_string()), "A".to_string(), "EVENT".to_string(), 0.9);
        let b = Entity::new(EntityId::new("b".to_string()), "B".to_string(), "EVENT".to_string(), 0.9);
        let c = Entity::new(EntityId::new("c".to_string()), "C".to_string(), "EVENT".to_string(), 0.9);

        graph.add_entity(a).unwrap();
        graph.add_entity(b).unwrap();
        graph.add_entity(c).unwrap();

        // A→B at time 100, B→C at time 50 (temporal violation!)
        let rel_ab = Relationship::new(
            EntityId::new("a".to_string()),
            EntityId::new("b".to_string()),
            "CAUSED".to_string(),
            0.8,
        )
        .with_temporal_range(100, 100)
        .with_causal_strength(0.9);

        let rel_bc = Relationship::new(
            EntityId::new("b".to_string()),
            EntityId::new("c".to_string()),
            "CAUSED".to_string(),
            0.8,
        )
        .with_temporal_range(50, 50) // Earlier than A→B!
        .with_causal_strength(0.9);

        graph.add_relationship(rel_ab).unwrap();
        graph.add_relationship(rel_bc).unwrap();

        let analyzer = CausalAnalyzer::new(Arc::new(graph))
            .with_temporal_consistency(true); // Require consistency

        let chains = analyzer
            .find_causal_chains(&EntityId::new("a".to_string()), &EntityId::new("c".to_string()), 5)
            .unwrap();

        assert_eq!(chains.len(), 0, "Should reject temporally inconsistent chain");
    }

    #[test]
    fn test_confidence_calculation() {
        let step1 = CausalStep {
            source: EntityId::new("a".to_string()),
            target: EntityId::new("b".to_string()),
            relation_type: "CAUSED".to_string(),
            temporal_type: Some(TemporalRelationType::Caused),
            temporal_range: None,
            confidence: 0.8,
            causal_strength: Some(0.9),
        };

        let step2 = CausalStep {
            source: EntityId::new("b".to_string()),
            target: EntityId::new("c".to_string()),
            relation_type: "CAUSED".to_string(),
            temporal_type: Some(TemporalRelationType::Caused),
            temporal_range: None,
            confidence: 0.9,
            causal_strength: Some(0.95),
        };

        let chain = CausalChain {
            cause: EntityId::new("a".to_string()),
            effect: EntityId::new("c".to_string()),
            steps: vec![step1, step2],
            total_confidence: 0.0,
            temporal_consistency: true,
            time_span: None,
        };

        let confidence = chain.calculate_confidence();

        // Confidence should be product of weighted confidences
        // step1: 0.8 * (0.5 + 0.5*0.9) = 0.8 * 0.95 = 0.76
        // step2: 0.9 * (0.5 + 0.5*0.95) = 0.9 * 0.975 = 0.8775
        // product: 0.76 * 0.8775 ≈ 0.667
        assert!(confidence > 0.65 && confidence < 0.7, "Confidence calculation incorrect: {}", confidence);
    }
}
