//! Query Optimizer for Knowledge Graph Queries
//!
//! Optimizes query execution plans using:
//! - Join ordering optimization
//! - Cost estimation based on graph statistics
//! - Selectivity estimation
//! - Query rewriting
//!
//! This is a rule-based optimizer without ML dependencies.

use crate::core::KnowledgeGraph;
use crate::Result;
use std::collections::HashMap;

/// Query operation types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOp {
    /// Scan all entities of a type
    EntityScan {
        /// The entity type to scan for
        entity_type: String,
    },
    /// Filter entities by property
    Filter {
        /// Property name to filter on
        property: String,
        /// Property value to match
        value: String,
    },
    /// Join two results on entity relationships
    Join {
        /// Left operand of the join
        left: Box<QueryOp>,
        /// Right operand of the join
        right: Box<QueryOp>,
        /// Type of join operation
        join_type: JoinType,
    },
    /// Get neighbors of entities
    Neighbors {
        /// Source entities to find neighbors for
        source: Box<QueryOp>,
        /// Optional relationship type filter
        relation_type: Option<String>,
        /// Maximum number of hops (graph traversal depth)
        max_hops: usize,
    },
    /// Union of two operations
    Union {
        /// Left operand of the union
        left: Box<QueryOp>,
        /// Right operand of the union
        right: Box<QueryOp>,
    },
    /// Limit results
    Limit {
        /// Source operation to limit
        source: Box<QueryOp>,
        /// Maximum number of results
        count: usize,
    },
}

/// Join type
#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    /// Inner join (intersection)
    Inner,
    /// Left outer join
    LeftOuter,
    /// Cross product
    Cross,
}

/// Cost statistics for an operation
#[derive(Debug, Clone)]
pub struct OperationCost {
    /// Estimated number of results
    pub cardinality: usize,
    /// Estimated cost (abstract units)
    pub cost: f64,
    /// Selectivity (0.0-1.0)
    pub selectivity: f64,
}

/// Graph statistics for cost estimation
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Total entities
    pub total_entities: usize,
    /// Entities per type
    pub entities_by_type: HashMap<String, usize>,
    /// Total relationships
    pub total_relationships: usize,
    /// Relationships per type
    pub relationships_by_type: HashMap<String, usize>,
    /// Average degree (edges per node)
    pub average_degree: f64,
}

impl GraphStatistics {
    /// Compute statistics from a knowledge graph
    pub fn from_graph(graph: &KnowledgeGraph) -> Self {
        let entities: Vec<_> = graph.entities().collect();
        let total_entities = entities.len();

        let mut entities_by_type: HashMap<String, usize> = HashMap::new();
        for entity in &entities {
            *entities_by_type
                .entry(entity.entity_type.clone())
                .or_insert(0) += 1;
        }

        let relationships = graph.get_all_relationships();
        let total_relationships = relationships.len();

        let mut relationships_by_type: HashMap<String, usize> = HashMap::new();
        for rel in &relationships {
            *relationships_by_type
                .entry(rel.relation_type.clone())
                .or_insert(0) += 1;
        }

        let average_degree = if total_entities > 0 {
            (total_relationships as f64 * 2.0) / total_entities as f64
        } else {
            0.0
        };

        Self {
            total_entities,
            entities_by_type,
            total_relationships,
            relationships_by_type,
            average_degree,
        }
    }
}

/// Query optimizer
pub struct QueryOptimizer {
    stats: GraphStatistics,
}

impl QueryOptimizer {
    /// Create a new optimizer with graph statistics
    pub fn new(stats: GraphStatistics) -> Self {
        Self { stats }
    }

    /// Optimize a query operation
    pub fn optimize(&self, query: QueryOp) -> Result<QueryOp> {
        // Apply optimization rules
        let rewritten = self.rewrite_query(query)?;
        let optimized = self.optimize_joins(rewritten)?;
        Ok(optimized)
    }

    /// Rewrite query using algebraic rules
    fn rewrite_query(&self, query: QueryOp) -> Result<QueryOp> {
        match query {
            // Push filters down through joins
            QueryOp::Filter { property, value } => {
                Ok(QueryOp::Filter { property, value })
            }

            // Reorder joins based on selectivity
            QueryOp::Join {
                left,
                right,
                join_type,
            } => {
                let left_opt = self.rewrite_query(*left)?;
                let right_opt = self.rewrite_query(*right)?;

                // Estimate costs
                let left_cost = self.estimate_cost(&left_opt)?;
                let right_cost = self.estimate_cost(&right_opt)?;

                // Put smaller (more selective) operand first for hash joins
                if left_cost.cardinality > right_cost.cardinality {
                    Ok(QueryOp::Join {
                        left: Box::new(right_opt),
                        right: Box::new(left_opt),
                        join_type,
                    })
                } else {
                    Ok(QueryOp::Join {
                        left: Box::new(left_opt),
                        right: Box::new(right_opt),
                        join_type,
                    })
                }
            }

            // Recursively optimize subqueries
            QueryOp::Neighbors {
                source,
                relation_type,
                max_hops,
            } => {
                let source_opt = self.rewrite_query(*source)?;
                Ok(QueryOp::Neighbors {
                    source: Box::new(source_opt),
                    relation_type,
                    max_hops,
                })
            }

            QueryOp::Union { left, right } => {
                let left_opt = self.rewrite_query(*left)?;
                let right_opt = self.rewrite_query(*right)?;
                Ok(QueryOp::Union {
                    left: Box::new(left_opt),
                    right: Box::new(right_opt),
                })
            }

            QueryOp::Limit { source, count } => {
                let source_opt = self.rewrite_query(*source)?;
                Ok(QueryOp::Limit {
                    source: Box::new(source_opt),
                    count,
                })
            }

            // Base case: entity scans
            QueryOp::EntityScan { entity_type } => Ok(QueryOp::EntityScan { entity_type }),
        }
    }

    /// Optimize join ordering using dynamic programming
    fn optimize_joins(&self, query: QueryOp) -> Result<QueryOp> {
        match query {
            QueryOp::Join {
                left,
                right,
                join_type,
            } => {
                // Recursively optimize sub-queries
                let left_opt = self.optimize_joins(*left)?;
                let right_opt = self.optimize_joins(*right)?;

                // For multi-way joins, collect all join operands
                let mut operands = Vec::new();
                Self::collect_join_operands(&left_opt, &mut operands);
                Self::collect_join_operands(&right_opt, &mut operands);

                if operands.len() > 2 {
                    // Multi-way join: find optimal order using greedy algorithm
                    self.find_optimal_join_order(operands, join_type)
                } else {
                    // Binary join: already optimized by rewrite_query
                    Ok(QueryOp::Join {
                        left: Box::new(left_opt),
                        right: Box::new(right_opt),
                        join_type,
                    })
                }
            }

            // Recursively process other operations
            QueryOp::Neighbors {
                source,
                relation_type,
                max_hops,
            } => {
                let source_opt = self.optimize_joins(*source)?;
                Ok(QueryOp::Neighbors {
                    source: Box::new(source_opt),
                    relation_type,
                    max_hops,
                })
            }

            QueryOp::Union { left, right } => {
                let left_opt = self.optimize_joins(*left)?;
                let right_opt = self.optimize_joins(*right)?;
                Ok(QueryOp::Union {
                    left: Box::new(left_opt),
                    right: Box::new(right_opt),
                })
            }

            QueryOp::Limit { source, count } => {
                let source_opt = self.optimize_joins(*source)?;
                Ok(QueryOp::Limit {
                    source: Box::new(source_opt),
                    count,
                })
            }

            // Leaf operations
            _ => Ok(query),
        }
    }

    /// Collect all join operands for multi-way join optimization
    fn collect_join_operands(op: &QueryOp, operands: &mut Vec<QueryOp>) {
        match op {
            QueryOp::Join { left, right, .. } => {
                Self::collect_join_operands(left, operands);
                Self::collect_join_operands(right, operands);
            }
            _ => {
                operands.push(op.clone());
            }
        }
    }

    /// Find optimal join order using greedy algorithm
    fn find_optimal_join_order(
        &self,
        mut operands: Vec<QueryOp>,
        join_type: JoinType,
    ) -> Result<QueryOp> {
        if operands.is_empty() {
            return Err(crate::core::GraphRAGError::Validation {
                message: "No operands for join".to_string(),
            });
        }

        if operands.len() == 1 {
            return Ok(operands.pop().unwrap());
        }

        // Greedy algorithm: repeatedly pick the two cheapest operands to join
        while operands.len() > 1 {
            let mut min_cost = f64::MAX;
            let mut best_i = 0;
            let mut best_j = 1;

            // Find pair with minimum join cost
            for i in 0..operands.len() {
                for j in (i + 1)..operands.len() {
                    let cost_i = self.estimate_cost(&operands[i])?;
                    let cost_j = self.estimate_cost(&operands[j])?;

                    // Estimate join cost as product of cardinalities (simplified)
                    let join_cost = (cost_i.cardinality as f64) * (cost_j.cardinality as f64);

                    if join_cost < min_cost {
                        min_cost = join_cost;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Create join of best pair
            let left = operands.remove(best_i);
            let right = operands.remove(if best_j > best_i {
                best_j - 1
            } else {
                best_j
            });

            let joined = QueryOp::Join {
                left: Box::new(left),
                right: Box::new(right),
                join_type: join_type.clone(),
            };

            operands.push(joined);
        }

        Ok(operands.pop().unwrap())
    }

    /// Estimate cost of an operation
    pub fn estimate_cost(&self, op: &QueryOp) -> Result<OperationCost> {
        match op {
            QueryOp::EntityScan { entity_type } => {
                let cardinality = self
                    .stats
                    .entities_by_type
                    .get(entity_type)
                    .copied()
                    .unwrap_or(0);

                Ok(OperationCost {
                    cardinality,
                    cost: cardinality as f64,
                    selectivity: if self.stats.total_entities > 0 {
                        cardinality as f64 / self.stats.total_entities as f64
                    } else {
                        0.0
                    },
                })
            }

            QueryOp::Filter { property: _, value: _ } => {
                // Assume filter has 10% selectivity (can be improved with histograms)
                let selectivity = 0.1;
                let cardinality = (self.stats.total_entities as f64 * selectivity) as usize;

                Ok(OperationCost {
                    cardinality,
                    cost: self.stats.total_entities as f64, // Must scan all
                    selectivity,
                })
            }

            QueryOp::Join { left, right, join_type } => {
                let left_cost = self.estimate_cost(left)?;
                let right_cost = self.estimate_cost(right)?;

                let cardinality = match join_type {
                    JoinType::Inner => {
                        // Estimate as geometric mean of inputs
                        ((left_cost.cardinality as f64) * (right_cost.cardinality as f64))
                            .sqrt() as usize
                    }
                    JoinType::LeftOuter => left_cost.cardinality,
                    JoinType::Cross => left_cost.cardinality * right_cost.cardinality,
                };

                let cost = left_cost.cost
                    + right_cost.cost
                    + (left_cost.cardinality as f64 * right_cost.cardinality as f64);

                Ok(OperationCost {
                    cardinality,
                    cost,
                    selectivity: left_cost.selectivity * right_cost.selectivity,
                })
            }

            QueryOp::Neighbors {
                source,
                relation_type: _,
                max_hops,
            } => {
                let source_cost = self.estimate_cost(source)?;

                // Estimate neighbors as source_cardinality * avg_degree^hops
                let expansion_factor = self.stats.average_degree.powi(*max_hops as i32);
                let cardinality =
                    (source_cost.cardinality as f64 * expansion_factor).min(self.stats.total_entities as f64) as usize;

                Ok(OperationCost {
                    cardinality,
                    cost: source_cost.cost + (cardinality as f64),
                    selectivity: cardinality as f64 / self.stats.total_entities as f64,
                })
            }

            QueryOp::Union { left, right } => {
                let left_cost = self.estimate_cost(left)?;
                let right_cost = self.estimate_cost(right)?;

                // Union cardinality (with some overlap assumed)
                let cardinality = (left_cost.cardinality + right_cost.cardinality) * 9 / 10;

                Ok(OperationCost {
                    cardinality,
                    cost: left_cost.cost + right_cost.cost,
                    selectivity: (left_cost.selectivity + right_cost.selectivity).min(1.0),
                })
            }

            QueryOp::Limit { source, count } => {
                let source_cost = self.estimate_cost(source)?;

                Ok(OperationCost {
                    cardinality: (*count).min(source_cost.cardinality),
                    cost: source_cost.cost,
                    selectivity: (*count as f64 / self.stats.total_entities as f64).min(1.0),
                })
            }
        }
    }

    /// Generate an execution plan with cost annotations
    pub fn explain(&self, op: &QueryOp) -> Result<String> {
        let cost = self.estimate_cost(op)?;
        let mut plan = String::new();

        self.explain_recursive(op, 0, &mut plan)?;

        plan.push_str(&format!(
            "\nEstimated Cost: {:.2}\nEstimated Cardinality: {}\nSelectivity: {:.2}%\n",
            cost.cost,
            cost.cardinality,
            cost.selectivity * 100.0
        ));

        Ok(plan)
    }

    /// Recursively build execution plan string
    fn explain_recursive(&self, op: &QueryOp, depth: usize, plan: &mut String) -> Result<()> {
        let indent = "  ".repeat(depth);
        let cost = self.estimate_cost(op)?;

        match op {
            QueryOp::EntityScan { entity_type } => {
                plan.push_str(&format!(
                    "{}EntityScan({}) [cost={:.0}, rows={}]\n",
                    indent, entity_type, cost.cost, cost.cardinality
                ));
            }
            QueryOp::Filter { property, value } => {
                plan.push_str(&format!(
                    "{}Filter({}={}) [cost={:.0}, rows={}]\n",
                    indent, property, value, cost.cost, cost.cardinality
                ));
            }
            QueryOp::Join {
                left,
                right,
                join_type,
            } => {
                plan.push_str(&format!(
                    "{}Join({:?}) [cost={:.0}, rows={}]\n",
                    indent, join_type, cost.cost, cost.cardinality
                ));
                self.explain_recursive(left, depth + 1, plan)?;
                self.explain_recursive(right, depth + 1, plan)?;
            }
            QueryOp::Neighbors {
                source,
                relation_type,
                max_hops,
            } => {
                let rel_str = relation_type.as_deref().unwrap_or("*");
                plan.push_str(&format!(
                    "{}Neighbors({}, hops={}) [cost={:.0}, rows={}]\n",
                    indent, rel_str, max_hops, cost.cost, cost.cardinality
                ));
                self.explain_recursive(source, depth + 1, plan)?;
            }
            QueryOp::Union { left, right } => {
                plan.push_str(&format!(
                    "{}Union [cost={:.0}, rows={}]\n",
                    indent, cost.cost, cost.cardinality
                ));
                self.explain_recursive(left, depth + 1, plan)?;
                self.explain_recursive(right, depth + 1, plan)?;
            }
            QueryOp::Limit { source, count } => {
                plan.push_str(&format!(
                    "{}Limit({}) [cost={:.0}, rows={}]\n",
                    indent, count, cost.cost, cost.cardinality
                ));
                self.explain_recursive(source, depth + 1, plan)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stats() -> GraphStatistics {
        let mut entities_by_type = HashMap::new();
        entities_by_type.insert("PERSON".to_string(), 100);
        entities_by_type.insert("ORGANIZATION".to_string(), 50);
        entities_by_type.insert("LOCATION".to_string(), 30);

        let mut relationships_by_type = HashMap::new();
        relationships_by_type.insert("WORKS_FOR".to_string(), 80);
        relationships_by_type.insert("LOCATED_IN".to_string(), 60);

        GraphStatistics {
            total_entities: 180,
            entities_by_type,
            total_relationships: 140,
            relationships_by_type,
            average_degree: 1.56,
        }
    }

    #[test]
    fn test_cost_estimation_scan() {
        let stats = create_test_stats();
        let optimizer = QueryOptimizer::new(stats);

        let query = QueryOp::EntityScan {
            entity_type: "PERSON".to_string(),
        };

        let cost = optimizer.estimate_cost(&query).unwrap();

        assert_eq!(cost.cardinality, 100);
        assert_eq!(cost.cost, 100.0);
    }

    #[test]
    fn test_cost_estimation_join() {
        let stats = create_test_stats();
        let optimizer = QueryOptimizer::new(stats);

        let query = QueryOp::Join {
            left: Box::new(QueryOp::EntityScan {
                entity_type: "PERSON".to_string(),
            }),
            right: Box::new(QueryOp::EntityScan {
                entity_type: "ORGANIZATION".to_string(),
            }),
            join_type: JoinType::Inner,
        };

        let cost = optimizer.estimate_cost(&query).unwrap();

        // Geometric mean: sqrt(100 * 50) = ~71
        assert!(cost.cardinality > 60 && cost.cardinality < 80);
    }

    #[test]
    fn test_join_reordering() {
        let stats = create_test_stats();
        let optimizer = QueryOptimizer::new(stats);

        // Join large table (PERSON=100) with small table (LOCATION=30)
        let query = QueryOp::Join {
            left: Box::new(QueryOp::EntityScan {
                entity_type: "PERSON".to_string(),
            }),
            right: Box::new(QueryOp::EntityScan {
                entity_type: "LOCATION".to_string(),
            }),
            join_type: JoinType::Inner,
        };

        let optimized = optimizer.optimize(query).unwrap();

        // Should reorder to put smaller table first
        if let QueryOp::Join { left, .. } = optimized {
            if let QueryOp::EntityScan { entity_type } = &*left {
                assert_eq!(entity_type, "LOCATION", "Smaller table should be first");
            }
        }
    }

    #[test]
    fn test_neighbors_cost() {
        let stats = create_test_stats();
        let optimizer = QueryOptimizer::new(stats);

        let query = QueryOp::Neighbors {
            source: Box::new(QueryOp::EntityScan {
                entity_type: "PERSON".to_string(),
            }),
            relation_type: Some("WORKS_FOR".to_string()),
            max_hops: 2,
        };

        let cost = optimizer.estimate_cost(&query).unwrap();

        // Should expand based on avg_degree^hops
        assert!(cost.cardinality > 100);
    }

    #[test]
    fn test_explain_plan() {
        let stats = create_test_stats();
        let optimizer = QueryOptimizer::new(stats);

        let query = QueryOp::Join {
            left: Box::new(QueryOp::EntityScan {
                entity_type: "PERSON".to_string(),
            }),
            right: Box::new(QueryOp::EntityScan {
                entity_type: "ORGANIZATION".to_string(),
            }),
            join_type: JoinType::Inner,
        };

        let plan = optimizer.explain(&query).unwrap();

        assert!(plan.contains("Join"));
        assert!(plan.contains("EntityScan"));
        assert!(plan.contains("Estimated Cost"));
    }
}
