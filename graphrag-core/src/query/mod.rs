/// Adaptive query routing for hierarchical GraphRAG
pub mod adaptive_routing;
/// Advanced query pipeline implementations
pub mod advanced_pipeline;
/// Query analysis utilities
pub mod analysis;
/// Query expansion strategies
pub mod expansion;
/// Query intelligence and optimization
pub mod intelligence;
/// Multi-query processing
pub mod multi_query;
/// Query optimizer for join ordering and cost estimation
pub mod optimizer;
/// Query decomposition and planning
pub mod planner;
/// Ranking policy implementations
pub mod ranking_policies;

// Re-export only the types that actually exist
pub use intelligence::{QueryIntelligence, QueryTemplate, QueryType, RewrittenQuery};

// Re-export adaptive routing types
pub use adaptive_routing::{
    AdaptiveRoutingConfig, QueryAnalysis, QueryComplexity, QueryComplexityAnalyzer,
};

// Re-export optimizer types
pub use optimizer::{GraphStatistics, JoinType, OperationCost, QueryOp, QueryOptimizer};
