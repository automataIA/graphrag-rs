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

use super::GraphRAG;

impl GraphRAG {
    /// Add a document from text content
    pub fn add_document_from_text(&mut self, text: &str) -> Result<()> {
        use crate::text::TextProcessor;
        use indexmap::IndexMap;

        // Content-addressed id: identical text is the same document (and is skipped below).
        let hash = crate::entity::ExtractionCache::key(&["document", text]);
        let doc_id = DocumentId::new(format!("doc_{}", &hash[..16]));
        if let Some(graph) = &self.knowledge_graph {
            if graph.get_document(&doc_id).is_some() {
                #[cfg(feature = "tracing")]
                tracing::info!("Document {} already indexed, skipping", doc_id);
                return Ok(());
            }
        }

        let document = Document {
            id: doc_id,
            title: "Document".to_string(),
            content: text.to_string(),
            metadata: IndexMap::new(),
            chunks: Vec::new(),
        };

        let text_processor =
            TextProcessor::new(self.config.text.chunk_size, self.config.text.chunk_overlap)?;
        let mut chunks = text_processor.chunk_text(&document)?;
        if self.config.chapter_summaries.enabled {
            // chapter-wise summaries need the real chapters (only `metadata.chapter` is set)
            TextProcessor::assign_chapters(&document, &mut chunks);
        }

        let document_with_chunks = Document { chunks, ..document };

        self.add_document(document_with_chunks)
    }

    /// Add a document to the system
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let document_id = document.id.clone();
        graph.add_document(document)?;
        self.pending_documents.insert(document_id);
        Ok(())
    }
}
