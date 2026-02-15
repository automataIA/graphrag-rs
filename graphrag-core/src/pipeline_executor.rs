//! Composable pipeline executor for GraphRAG operations
//!
//! Provides a step-by-step interface for building knowledge graphs,
//! giving callers control over individual pipeline phases.
//!
//! # Example
//!
//! ```rust,no_run
//! use graphrag_core::pipeline_executor::PipelineExecutor;
//! use graphrag_core::GraphRAG;
//!
//! # async fn example() -> graphrag_core::Result<()> {
//! let mut graphrag = GraphRAG::builder()
//!     .with_output_dir("./output")
//!     .with_hash_embeddings()
//!     .build()?;
//!
//! let executor = PipelineExecutor::new(&mut graphrag);
//! let report = executor.run_full_pipeline().await?;
//!
//! println!("Pipeline complete: {} entities, {} relationships",
//!     report.entity_count, report.relationship_count);
//! # Ok(())
//! # }
//! ```

use crate::core::GraphRAGError;
use crate::{GraphRAG, Result};

/// Report produced after a pipeline run
#[derive(Debug, Clone)]
pub struct PipelineReport {
    /// Number of entities extracted
    pub entity_count: usize,
    /// Number of relationships built
    pub relationship_count: usize,
    /// Number of text chunks processed
    pub chunks_processed: usize,
    /// Number of documents loaded
    pub document_count: usize,
    /// Pipeline approach used ("semantic", "algorithmic", "hybrid")
    pub approach: String,
    /// Total elapsed time in milliseconds
    pub elapsed_ms: u128,
}

/// Composable pipeline executor
///
/// Wraps a `GraphRAG` instance and provides fine-grained control over
/// the build pipeline phases.
pub struct PipelineExecutor<'a> {
    graphrag: &'a mut GraphRAG,
}

impl<'a> PipelineExecutor<'a> {
    /// Create a new pipeline executor wrapping the given GraphRAG instance
    pub fn new(graphrag: &'a mut GraphRAG) -> Self {
        Self { graphrag }
    }

    /// Run the full pipeline (entity extraction → relationships → communities → summarization)
    ///
    /// This is equivalent to calling `GraphRAG::build_graph()` but returns a detailed report.
    #[cfg(feature = "async")]
    pub async fn run_full_pipeline(&mut self) -> Result<PipelineReport> {
        let start = std::time::Instant::now();

        // Verify we have documents
        if !self.graphrag.has_documents() {
            return Err(GraphRAGError::Config {
                message: "No documents loaded. Add documents before running the pipeline."
                    .to_string(),
            });
        }

        // Delegate to GraphRAG's build_graph which handles all phases
        self.graphrag.build_graph().await?;

        let elapsed = start.elapsed().as_millis();

        Ok(self.build_report(elapsed))
    }

    /// Run the full pipeline (synchronous version)
    #[cfg(not(feature = "async"))]
    pub fn run_full_pipeline(&mut self) -> Result<PipelineReport> {
        let start = std::time::Instant::now();

        if !self.graphrag.has_documents() {
            return Err(GraphRAGError::Config {
                message: "No documents loaded. Add documents before running the pipeline."
                    .to_string(),
            });
        }

        self.graphrag.build_graph()?;

        let elapsed = start.elapsed().as_millis();

        Ok(self.build_report(elapsed))
    }

    /// Add a document from text and run the full pipeline in one call
    #[cfg(feature = "async")]
    pub async fn ingest_and_build(&mut self, text: &str) -> Result<PipelineReport> {
        self.graphrag.add_document_from_text(text)?;
        self.run_full_pipeline().await
    }

    /// Add a document from text and run the full pipeline in one call (sync)
    #[cfg(not(feature = "async"))]
    pub fn ingest_and_build(&mut self, text: &str) -> Result<PipelineReport> {
        self.graphrag.add_document_from_text(text)?;
        self.run_full_pipeline()
    }

    /// Get a snapshot report of the current graph state without running any pipeline
    pub fn current_state(&self) -> PipelineReport {
        self.build_report(0)
    }

    /// Build report from current graph state
    fn build_report(&self, elapsed_ms: u128) -> PipelineReport {
        let kg = self.graphrag.knowledge_graph();
        let (entity_count, relationship_count, chunks_processed, document_count) = match kg {
            Some(kg) => (
                kg.entities().count(),
                kg.relationships().count(),
                kg.chunks().count(),
                kg.documents().count(),
            ),
            None => (0, 0, 0, 0),
        };

        PipelineReport {
            entity_count,
            relationship_count,
            chunks_processed,
            document_count,
            approach: self.graphrag.config().approach.clone(),
            elapsed_ms,
        }
    }
}

impl std::fmt::Display for PipelineReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pipeline Report ({}): {} entities, {} relationships, {} chunks from {} docs [{}ms]",
            self.approach,
            self.entity_count,
            self.relationship_count,
            self.chunks_processed,
            self.document_count,
            self.elapsed_ms,
        )
    }
}
