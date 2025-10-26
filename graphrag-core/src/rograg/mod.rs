//! ROGRAG (Robustly Optimized GraphRAG) module - Reasoning on Graphs for RAG
//!
//! This module implements a sophisticated query processing system that combines
//! structured reasoning with graph-based retrieval for enhanced accuracy and robustness.
//!
//! # Architecture
//!
//! ROGRAG introduces three primary retrieval strategies:
//!
//! 1. **Logic Form Retrieval** - Parses queries into structured logic forms for precise graph traversal
//! 2. **Fuzzy Matching** - Provides semantic similarity-based fallback when logic forms fail
//! 3. **Query Decomposition** - Breaks complex queries into manageable subqueries
//!
//! # Key Components
//!
//! - [`decomposer`] - Query decomposition using semantic/syntactic/hybrid strategies
//! - [`fuzzy_matcher`] - Semantic similarity matching for entities and content chunks
//! - [`intent_classifier`] - Query intent detection with refusal capabilities
//! - [`logic_form`] - Structured query representation and execution
//! - [`processor`] - Main ROGRAG processing pipeline orchestration
//! - [`validator`] - Query validation and quality assessment
//! - [`quality_metrics`] - Performance and quality tracking
//! - [`streaming`] - Streaming response generation
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use graphrag_core::rograg::{RogragProcessor, RogragConfig};
//!
//! // Initialize the ROGRAG processor
//! let processor = RogragProcessor::new(RogragConfig::default())?;
//!
//! // Process a query
//! let result = processor.process_query(
//!     "How are Entity A and Entity B related?",
//!     &knowledge_graph
//! ).await?;
//! ```
//!
//! # Quality Assurance
//!
//! The ROGRAG module includes built-in validation, quality metrics, and intent
//! classification to ensure reliable operation and prevent inappropriate responses.

#[cfg(feature = "rograg")]
pub mod decomposer;
#[cfg(feature = "rograg")]
pub mod fuzzy_matcher;
#[cfg(feature = "rograg")]
pub mod intent_classifier;
#[cfg(feature = "rograg")]
pub mod logic_form;
#[cfg(feature = "rograg")]
pub mod processor;
#[cfg(feature = "rograg")]
pub mod quality_metrics;
#[cfg(feature = "rograg")]
pub mod streaming;
#[cfg(feature = "rograg")]
pub mod validator;

// Re-export main types with specific naming to avoid conflicts
#[cfg(feature = "rograg")]
pub use decomposer::*;
#[cfg(feature = "rograg")]
pub use fuzzy_matcher::*;
#[cfg(feature = "rograg")]
pub use intent_classifier::*;
#[cfg(feature = "rograg")]
pub use logic_form::*;
#[cfg(feature = "rograg")]
pub use processor::*;
#[cfg(feature = "rograg")]
pub use quality_metrics::{
    QualityMetrics as QualityMetricsConfig, 
    QualityMetricsConfig as QualityMetricsOptions,
    QueryMetrics, 
    ResponseQuality, 
    PerformanceStatistics,
    ComparativeAnalysis
};
#[cfg(feature = "rograg")]
pub use streaming::*;
#[cfg(feature = "rograg")]
pub use validator::{
    QueryValidator, 
    ValidationResult, 
    ValidationIssue, 
    IssueType, 
    IssueSeverity,
    QualityMetrics as ValidatorQualityMetrics
};

#[cfg(feature = "rograg")]
use crate::Result;

/// Initialize the ROGRAG subsystem.
///
/// This function initializes all ROGRAG components and performs any necessary
/// startup configuration. Currently this is a no-op but serves as a future
/// extension point for:
///
/// - Loading pre-compiled pattern databases
/// - Initializing statistical models
/// - Warming up caches
/// - Validating system configuration
///
/// # Returns
///
/// Returns `Ok(())` on successful initialization, or an error if any subsystem
/// fails to initialize properly.
///
/// # Example
///
/// ```rust,ignore
/// use graphrag_core::rograg::initialize_rograg;
///
/// // Initialize before using ROGRAG features
/// initialize_rograg()?;
/// ```
#[cfg(feature = "rograg")]
pub fn initialize_rograg() -> Result<()> {
    // Initialize ROGRAG subsystems
    // Future: Load pattern databases, warm caches, etc.
    Ok(())
}
