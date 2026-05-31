//! Main `GraphRAG` orchestrator type.
//!
//! Phase 4 follow-up split: the struct + impl bodies that previously lived in a
//! single 1753-LOC `graphrag.rs` are now distributed across per-concern files
//! in this directory. The struct definition + private `ensure_initialized`
//! helper + tests stay here; the impl methods live in:
//!
//! - [`lifecycle`] — constructors, `initialize`, workspace persistence, `clear_graph`
//! - [`documents`] — `add_document_from_text`, `add_document`
//! - [`build`] — async + sync `build_graph` pair
//! - [`query`] — `ask*` family, `query_internal*`, semantic-answer synthesis, `ask_with_pagerank`
//! - [`stats`] — getters (`config`, `knowledge_graph`, `get_entity`, etc.)
//! - [`factory`] — `from_json5_file`, `from_config_file`, `quick_start*`
//!
//! `lib.rs` re-exports `GraphRAG` so external paths remain unchanged.

#![allow(unused_imports)]

use crate::config::Config;
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use crate::{critic, ollama, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

mod ask;
mod build;
mod documents;
mod factory;
mod lifecycle;
mod stats;

/// This is the primary entry point for using GraphRAG. It orchestrates
/// all components: knowledge graph, retrieval, generation, and caching.
///
/// # Examples
///
/// ```ignore
/// use graphrag_core::{GraphRAG, Config};
///
/// # fn example() -> graphrag_core::Result<()> {
/// let config = Config::default();
/// let mut graphrag = GraphRAG::new(config)?;
/// graphrag.initialize()?;
///
/// // Add documents
/// graphrag.add_document_from_text("Your document text")?;
///
/// // Build knowledge graph
/// graphrag.build_graph()?;
///
/// // Query
/// let answer = graphrag.ask("Your question?")?;
/// println!("Answer: {}", answer);
/// # Ok(())
/// # }
/// ```
pub struct GraphRAG {
    config: Config,
    knowledge_graph: Option<KnowledgeGraph>,
    retrieval_system: Option<retrieval::RetrievalSystem>,
    query_planner: Option<query::planner::QueryPlanner>,
    #[cfg_attr(not(feature = "async"), allow(dead_code))]
    critic: Option<critic::Critic>,
    #[cfg(feature = "parallel-processing")]
    #[allow(dead_code)]
    parallel_processor: Option<parallel::ParallelProcessor>,
}

impl GraphRAG {
    /// Ensure system is initialized
    pub(super) fn ensure_initialized(&mut self) -> Result<()> {
        if !self.is_initialized() {
            self.initialize()
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let graphrag = GraphRAG::builder()
            .with_output_dir("./test_output")
            .with_chunk_size(512)
            .with_top_k(10)
            .build();
        assert!(graphrag.is_ok());
    }
}
