//! Pipeline Module
//!
//! This module provides data pipeline capabilities for GraphRAG:
//! - Data import from multiple formats
//! - ETL (Extract, Transform, Load) operations
//! - Batch and streaming processing
//!
//! ## Features
//!
//! ### Data Import
//! - CSV/TSV file support
//! - JSON/JSONL support
//! - RDF/Turtle semantic web formats
//! - GraphML graph exchange format
//! - Streaming ingestion
//!
//! ### Validation
//! - Schema validation
//! - Data quality checks
//! - Error handling and reporting

// Data import requires async feature
#[cfg(feature = "async")]
pub mod data_import;

// Re-export main types
#[cfg(feature = "async")]
pub use data_import::{
    DataFormat, ImportConfig, ColumnMappings,
    ImportedEntity, ImportedRelationship, ImportResult,
    ImportError, DataImporter, StreamingImporter, StreamingSource,
};
