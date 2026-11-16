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
use crate::{critic, llm, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

mod ask;
mod build;
mod chunk_context;
mod documents;
mod factory;
mod lifecycle;
mod stats;
mod summaries;

#[cfg(feature = "async")]
pub use ask::{AnswerResponse, TokenUsage};

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
    /// Documents added since the last successful build (or all, after `clear_graph`).
    /// `build_graph` extracts only their chunks; reloaded workspace documents are not pending.
    pending_documents: indexmap::IndexSet<DocumentId>,
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

    /// Fake chat server that answers every request with HTTP 500.
    #[cfg(all(feature = "llm", feature = "async"))]
    fn failing_llm_server() -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom",
                );
            }
        });
        url
    }

    #[cfg(all(feature = "llm", feature = "async"))]
    #[tokio::test]
    async fn failed_chunk_extraction_fails_the_build_and_keeps_documents_pending() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        let mut config = Config::quick(&workspace).with_llm_base_url(failing_llm_server());
        config.entities.use_gleaning = false;
        config.llm.max_retries = 1;
        let mut graphrag = GraphRAG::new(config).unwrap();
        graphrag.initialize().unwrap();
        graphrag.add_document_from_text(DOC_A).unwrap();

        let error = graphrag.build_graph().await.unwrap_err().to_string();

        assert!(error.contains("failed for 1 of 1 chunks"), "{error}");
        assert!(!graphrag.pending_documents.is_empty());
        assert!(!workspace.exists(), "failed build must not save");
    }

    const DOC_A: &str = "Maria Rossi dirige il progetto Prisma insieme a Luca Bianchi.";
    const DOC_B: &str = "Giulia Verdi lavora al laboratorio Orione con Paolo Neri.";

    fn extracted_chunk_ids(graphrag: &GraphRAG) -> Vec<String> {
        GraphRAG::chunks_to_extract(
            graphrag.knowledge_graph().unwrap(),
            &graphrag.pending_documents,
        )
        .into_iter()
        .map(|c| c.id.to_string())
        .collect()
    }

    fn chunk_ids_of(graphrag: &GraphRAG, text: &str) -> Vec<String> {
        graphrag
            .knowledge_graph()
            .unwrap()
            .documents()
            .filter(|d| d.content == text)
            .flat_map(|d| d.chunks.iter().map(|c| c.id.to_string()))
            .collect()
    }

    fn mentions_of(graphrag: &GraphRAG, name: &str) -> usize {
        graphrag
            .knowledge_graph()
            .unwrap()
            .entities()
            .filter(|e| e.name == name)
            .map(|e| e.mentions.len())
            .sum()
    }

    #[test]
    fn same_text_added_twice_is_one_document() {
        let mut graphrag = GraphRAG::new(Config::default()).unwrap();
        graphrag.initialize().unwrap();
        graphrag.add_document_from_text(DOC_A).unwrap();
        graphrag.add_document_from_text(DOC_A).unwrap();
        graphrag.add_document_from_text(DOC_B).unwrap();
        let graph = graphrag.knowledge_graph().unwrap();
        assert_eq!(graph.documents().count(), 2);
        let chunk_total: usize = graph.documents().map(|d| d.chunks.len()).sum();
        assert_eq!(graph.chunks().count(), chunk_total);
    }

    #[test]
    fn build_extracts_only_documents_added_since_last_build() {
        let mut graphrag = GraphRAG::new(Config::default()).unwrap();
        graphrag.initialize().unwrap();
        graphrag.add_document_from_text(DOC_A).unwrap();
        assert_eq!(
            extracted_chunk_ids(&graphrag),
            chunk_ids_of(&graphrag, DOC_A)
        );
        graphrag.build_graph_sync().unwrap();
        assert!(extracted_chunk_ids(&graphrag).is_empty());
        let before = mentions_of(&graphrag, "Maria Rossi");
        assert!(before > 0);

        graphrag.add_document_from_text(DOC_B).unwrap();
        assert_eq!(
            extracted_chunk_ids(&graphrag),
            chunk_ids_of(&graphrag, DOC_B)
        );
        graphrag.build_graph_sync().unwrap();

        assert!(extracted_chunk_ids(&graphrag).is_empty());
        assert_eq!(mentions_of(&graphrag, "Maria Rossi"), before);
        assert!(mentions_of(&graphrag, "Giulia Verdi") > 0);
        assert_eq!(graphrag.knowledge_graph().unwrap().documents().count(), 2);
    }

    #[test]
    fn clear_graph_marks_every_document_pending_again() {
        let mut graphrag = GraphRAG::new(Config::default()).unwrap();
        graphrag.initialize().unwrap();
        graphrag.add_document_from_text(DOC_A).unwrap();
        graphrag.add_document_from_text(DOC_B).unwrap();
        graphrag.build_graph_sync().unwrap();
        assert!(extracted_chunk_ids(&graphrag).is_empty());

        graphrag.clear_graph().unwrap();
        let all = graphrag.knowledge_graph().unwrap().chunks().count();
        assert_eq!(extracted_chunk_ids(&graphrag).len(), all);
        graphrag.build_graph_sync().unwrap();
        assert!(mentions_of(&graphrag, "Maria Rossi") > 0);
        assert!(mentions_of(&graphrag, "Giulia Verdi") > 0);
    }

    #[test]
    fn reloaded_workspace_documents_are_not_extracted_again() {
        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path().join("ws");

        let mut first = GraphRAG::new(Config::quick(&workspace)).unwrap();
        first.initialize().unwrap();
        first.add_document_from_text(DOC_A).unwrap();
        first.build_graph_sync().unwrap();
        let before = mentions_of(&first, "Maria Rossi");
        assert!(before > 0);

        let mut second = GraphRAG::new(Config::quick(&workspace)).unwrap();
        second.initialize().unwrap();
        assert!(extracted_chunk_ids(&second).is_empty());
        // same text again: already indexed, nothing becomes pending
        second.add_document_from_text(DOC_A).unwrap();
        assert!(extracted_chunk_ids(&second).is_empty());

        second.add_document_from_text(DOC_B).unwrap();
        assert_eq!(extracted_chunk_ids(&second), chunk_ids_of(&second, DOC_B));
        second.build_graph_sync().unwrap();

        assert_eq!(mentions_of(&second, "Maria Rossi"), before);
        assert!(mentions_of(&second, "Giulia Verdi") > 0);
        assert_eq!(second.knowledge_graph().unwrap().documents().count(), 2);

        // and what was saved reloads with both documents
        let mut third = GraphRAG::new(Config::quick(&workspace)).unwrap();
        third.initialize().unwrap();
        assert_eq!(third.knowledge_graph().unwrap().documents().count(), 2);
        assert!(extracted_chunk_ids(&third).is_empty());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_build_extracts_only_new_documents() {
        let mut graphrag = GraphRAG::new(Config::default()).unwrap();
        graphrag.initialize().unwrap();
        graphrag.add_document_from_text(DOC_A).unwrap();
        graphrag.build_graph().await.unwrap();
        let before = mentions_of(&graphrag, "Maria Rossi");
        assert!(before > 0);
        graphrag.add_document_from_text(DOC_B).unwrap();
        assert_eq!(
            extracted_chunk_ids(&graphrag),
            chunk_ids_of(&graphrag, DOC_B)
        );
        graphrag.build_graph().await.unwrap();
        assert!(extracted_chunk_ids(&graphrag).is_empty());
        assert_eq!(mentions_of(&graphrag, "Maria Rossi"), before);
        assert!(mentions_of(&graphrag, "Giulia Verdi") > 0);
    }
}
