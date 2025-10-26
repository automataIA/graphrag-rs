//! Chunk enrichment pipeline
//!
//! Orchestrates document structure parsing, keyword extraction, and summarization
//! to enrich text chunks with semantic metadata.

use crate::{
    core::{ChunkMetadata, Document, TextChunk},
    text::{
        DocumentStructure, ExtractiveSummarizer, LayoutParser, TfIdfKeywordExtractor,
    },
    Result,
};

/// Chunk enricher that adds semantic metadata to text chunks
pub struct ChunkEnricher {
    /// Document structure parser
    layout_parser: Box<dyn LayoutParser>,
    /// Keyword extractor
    keyword_extractor: TfIdfKeywordExtractor,
    /// Text summarizer
    summarizer: ExtractiveSummarizer,
}

impl ChunkEnricher {
    /// Create a new chunk enricher
    pub fn new(
        layout_parser: Box<dyn LayoutParser>,
        keyword_extractor: TfIdfKeywordExtractor,
        summarizer: ExtractiveSummarizer,
    ) -> Self {
        Self {
            layout_parser,
            keyword_extractor,
            summarizer,
        }
    }

    /// Create with default components
    pub fn new_default(layout_parser: Box<dyn LayoutParser>) -> Self {
        Self {
            layout_parser,
            keyword_extractor: TfIdfKeywordExtractor::new_default(),
            summarizer: ExtractiveSummarizer::new(),
        }
    }

    /// Enrich multiple chunks with metadata
    pub fn enrich_chunks(
        &mut self,
        chunks: &mut [TextChunk],
        document: &Document,
    ) -> Result<()> {
        tracing::debug!(
            "Enriching {} chunks for document: {}",
            chunks.len(),
            document.title
        );

        // 1. Parse document structure once
        let structure = self.layout_parser.parse(&document.content);

        tracing::debug!(
            "Detected {} headings in document structure",
            structure.headings.len()
        );

        // 2. Enrich each chunk
        let total_chunks = chunks.len();
        for (idx, chunk) in chunks.iter_mut().enumerate() {
            if idx % 10 == 0 {
                tracing::debug!("Processing chunk {}/{}", idx + 1, total_chunks);
            }

            self.enrich_single_chunk(chunk, &structure, document)?;
        }

        tracing::debug!("Enrichment complete!");
        Ok(())
    }

    /// Enrich a single chunk with metadata
    fn enrich_single_chunk(
        &mut self,
        chunk: &mut TextChunk,
        structure: &DocumentStructure,
        document: &Document,
    ) -> Result<()> {
        let mut metadata = ChunkMetadata::new();

        // 1. Find which section this chunk belongs to
        if let Some(section_idx) = structure.find_section_containing_offset(chunk.start_offset) {
            let section = &structure.sections[section_idx];

            // 2. Build heading path (Chapter > Section > Subsection)
            let heading_path = structure.get_heading_path(section_idx);
            metadata.heading_path = heading_path.clone();

            // 3. Set chapter/section/subsection based on path
            if !heading_path.is_empty() {
                metadata.chapter = Some(heading_path[0].clone());

                if heading_path.len() > 1 {
                    metadata.section = Some(heading_path[1].clone());
                }

                if heading_path.len() > 2 {
                    metadata.subsection = Some(heading_path[2].clone());
                }
            }

            // 4. Set structural level
            metadata.structural_level = Some(section.heading.level);
        }

        // 5. Extract keywords using TF-IDF
        let keywords = self
            .keyword_extractor
            .extract_keyword_strings(&chunk.content, 5);
        metadata.keywords = keywords;

        // 6. Generate summary if chunk is long enough
        if chunk.content.len() > 150 {
            if let Ok(summary) = self.summarizer.summarize(&chunk.content, 150) {
                if !summary.is_empty() {
                    metadata.summary = Some(summary);
                }
            }
        }

        // 7. Calculate position in document
        let position = chunk.start_offset as f32 / document.content.len().max(1) as f32;
        metadata.position_in_document = Some(position);

        // 8. Set confidence (simple heuristic: based on completeness)
        let confidence = metadata.completeness_score();
        metadata.confidence = Some(confidence);

        // 9. Assign metadata to chunk
        chunk.metadata = metadata;

        Ok(())
    }

    /// Get statistics about enrichment
    pub fn get_statistics(&self, chunks: &[TextChunk]) -> EnrichmentStatistics {
        let mut stats = EnrichmentStatistics::default();

        for chunk in chunks {
            stats.total_chunks += 1;

            if chunk.metadata.has_structure_info() {
                stats.chunks_with_structure += 1;
            }

            if chunk.metadata.has_semantic_info() {
                stats.chunks_with_semantics += 1;
            }

            if !chunk.metadata.keywords.is_empty() {
                stats.chunks_with_keywords += 1;
                stats.total_keywords += chunk.metadata.keywords.len();
            }

            if chunk.metadata.summary.is_some() {
                stats.chunks_with_summary += 1;
            }

            stats.avg_completeness += chunk.metadata.completeness_score();
        }

        if stats.total_chunks > 0 {
            stats.avg_completeness /= stats.total_chunks as f32;
        }

        stats
    }
}

/// Statistics about chunk enrichment
#[derive(Debug, Default, Clone)]
pub struct EnrichmentStatistics {
    /// Total number of chunks processed
    pub total_chunks: usize,
    /// Chunks with structure metadata (chapter/section)
    pub chunks_with_structure: usize,
    /// Chunks with semantic metadata (keywords/summary)
    pub chunks_with_semantics: usize,
    /// Chunks with keywords
    pub chunks_with_keywords: usize,
    /// Chunks with summary
    pub chunks_with_summary: usize,
    /// Total keywords extracted
    pub total_keywords: usize,
    /// Average metadata completeness score
    pub avg_completeness: f32,
}

impl EnrichmentStatistics {
    /// Print statistics summary
    #[allow(dead_code)]
    pub fn print_summary(&self) {
        tracing::info!("\nChunk Enrichment Statistics:");
        tracing::info!("  Total chunks: {}", self.total_chunks);
        tracing::info!(
            "  Chunks with structure: {} ({:.1}%)",
            self.chunks_with_structure,
            self.chunks_with_structure as f32 / self.total_chunks as f32 * 100.0
        );
        tracing::info!(
            "  Chunks with semantics: {} ({:.1}%)",
            self.chunks_with_semantics,
            self.chunks_with_semantics as f32 / self.total_chunks as f32 * 100.0
        );
        tracing::info!(
            "  Chunks with keywords: {} ({:.1}%)",
            self.chunks_with_keywords,
            self.chunks_with_keywords as f32 / self.total_chunks as f32 * 100.0
        );
        tracing::info!(
            "  Chunks with summary: {} ({:.1}%)",
            self.chunks_with_summary,
            self.chunks_with_summary as f32 / self.total_chunks as f32 * 100.0
        );
        tracing::info!("  Total keywords: {}", self.total_keywords);
        tracing::info!(
            "  Average completeness: {:.2}",
            self.avg_completeness
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChunkId, DocumentId};
    use crate::text::parsers::MarkdownLayoutParser;

    #[test]
    fn test_chunk_enrichment() {
        let document = Document::new(
            DocumentId::new("test".to_string()),
            "test.md".to_string(),
            "# Chapter 1\n\nThis is about machine learning and artificial intelligence.\n\n## Section 1.1\n\nDeep learning is a subset of machine learning.".to_string(),
        );

        let mut chunks = vec![
            TextChunk::new(
                ChunkId::new("chunk_0".to_string()),
                document.id.clone(),
                "This is about machine learning and artificial intelligence.".to_string(),
                15,
                72,
            ),
            TextChunk::new(
                ChunkId::new("chunk_1".to_string()),
                document.id.clone(),
                "Deep learning is a subset of machine learning.".to_string(),
                88,
                135,
            ),
        ];

        let parser = Box::new(MarkdownLayoutParser::new());
        let mut enricher = ChunkEnricher::new_default(parser);

        enricher.enrich_chunks(&mut chunks, &document).unwrap();

        // Verify enrichment
        assert!(chunks[0].metadata.chapter.is_some());
        assert!(!chunks[0].metadata.keywords.is_empty());
        assert!(chunks[0].metadata.position_in_document.is_some());
    }

    #[test]
    fn test_enrichment_statistics() {
        let document = Document::new(
            DocumentId::new("test".to_string()),
            "test.md".to_string(),
            "# Test\n\nContent here with keywords like machine learning.".to_string(),
        );

        let mut chunks = vec![TextChunk::new(
            ChunkId::new("chunk_0".to_string()),
            document.id.clone(),
            "Content here with keywords like machine learning.".to_string(),
            8,
            56,
        )];

        let parser = Box::new(MarkdownLayoutParser::new());
        let mut enricher = ChunkEnricher::new_default(parser);

        enricher.enrich_chunks(&mut chunks, &document).unwrap();

        let stats = enricher.get_statistics(&chunks);

        assert_eq!(stats.total_chunks, 1);
        assert!(stats.avg_completeness > 0.0);
    }
}
