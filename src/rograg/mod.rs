//! ROGRAG (Robustly Optimized GraphRAG) module
//! 
//! This module implements advanced query processing, validation, and quality metrics
//! for robust GraphRAG operations.

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

#[cfg(feature = "rograg")]
pub fn initialize_rograg() -> Result<()> {
    // Initialize ROGRAG subsystems
    Ok(())
}
