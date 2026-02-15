//! Graph optimization module
//!
//! This module provides tools for optimizing knowledge graph structure and weights.

#[cfg(feature = "async")]
pub mod graph_weight_optimizer;

#[cfg(feature = "async")]
pub use graph_weight_optimizer::{
    GraphWeightOptimizer, ObjectiveWeights, OptimizationStep, OptimizerConfig, TestQuery,
};
