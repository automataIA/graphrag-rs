//! Chunk metadata for semantic enrichment
//!
//! This module defines metadata structures that enhance text chunks with
//! semantic information including document structure (chapter, section),
//! keywords, summaries, and positional information.

use serde::{Deserialize, Serialize};

/// Metadata associated with a text chunk, providing semantic context
///
/// This structure enriches chunks with information about their position in the
/// document hierarchy, extracted keywords, automatically generated summaries,
/// and other contextual information useful for retrieval and understanding.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkMetadata {
    /// Chapter name/title this chunk belongs to
    ///
    /// Examples: "Chapter 1: Introduction", "Part I"
    pub chapter: Option<String>,

    /// Section name/title within the chapter
    ///
    /// Examples: "1.1 Background", "Introduction"
    pub section: Option<String>,

    /// Subsection name/title within the section
    ///
    /// Examples: "1.1.1 Historical Context", "Early Development"
    pub subsection: Option<String>,

    /// Automatically detected or manually assigned topic
    ///
    /// Examples: "Machine Learning", "Neural Networks", "Data Processing"
    pub topic: Option<String>,

    /// Extracted keywords from the chunk content (TF-IDF or similar)
    ///
    /// Ordered by relevance/importance, typically 3-10 keywords
    pub keywords: Vec<String>,

    /// Automatically generated summary of the chunk content
    ///
    /// Typically 1-3 sentences capturing the main points
    pub summary: Option<String>,

    /// Hierarchical level in document structure (0 = root/chapter, 1 = section, 2 = subsection, etc.)
    ///
    /// Used to understand the depth of this chunk in the document hierarchy
    pub structural_level: Option<u8>,

    /// Relative position in the document (0.0 to 1.0)
    ///
    /// 0.0 = beginning, 0.5 = middle, 1.0 = end
    /// Useful for positional weighting in retrieval
    pub position_in_document: Option<f32>,

    /// Full heading path from document root to this chunk
    ///
    /// Example: ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
    /// Provides complete context of the chunk's location in document hierarchy
    pub heading_path: Vec<String>,

    /// Confidence score for metadata extraction (0.0 to 1.0)
    ///
    /// Indicates how confident the system is about the assigned metadata
    pub confidence: Option<f32>,

    /// Custom metadata key-value pairs
    ///
    /// Allows for extensibility with domain-specific metadata
    #[serde(default)]
    pub custom: std::collections::HashMap<String, String>,
}

impl ChunkMetadata {
    /// Create a new empty metadata instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metadata with chapter information
    pub fn with_chapter(mut self, chapter: String) -> Self {
        self.chapter = Some(chapter);
        self
    }

    /// Create metadata with section information
    pub fn with_section(mut self, section: String) -> Self {
        self.section = Some(section);
        self
    }

    /// Create metadata with subsection information
    pub fn with_subsection(mut self, subsection: String) -> Self {
        self.subsection = Some(subsection);
        self
    }

    /// Create metadata with keywords
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Create metadata with summary
    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = Some(summary);
        self
    }

    /// Create metadata with structural level
    pub fn with_structural_level(mut self, level: u8) -> Self {
        self.structural_level = Some(level);
        self
    }

    /// Create metadata with position in document
    pub fn with_position(mut self, position: f32) -> Self {
        self.position_in_document = Some(position.clamp(0.0, 1.0));
        self
    }

    /// Create metadata with heading path
    pub fn with_heading_path(mut self, path: Vec<String>) -> Self {
        self.heading_path = path;
        self
    }

    /// Add a custom metadata field
    pub fn add_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Check if metadata has any structural information (chapter, section, or subsection)
    pub fn has_structure_info(&self) -> bool {
        self.chapter.is_some() || self.section.is_some() || self.subsection.is_some()
    }

    /// Check if metadata has semantic enrichment (keywords or summary)
    pub fn has_semantic_info(&self) -> bool {
        !self.keywords.is_empty() || self.summary.is_some()
    }

    /// Get the deepest level heading (subsection > section > chapter)
    pub fn get_deepest_heading(&self) -> Option<&String> {
        self.subsection
            .as_ref()
            .or(self.section.as_ref())
            .or(self.chapter.as_ref())
    }

    /// Get full hierarchical context as a formatted string
    ///
    /// Example: "Chapter 1 > Section 1.1 > Subsection 1.1.1"
    pub fn get_hierarchy_string(&self) -> Option<String> {
        if self.heading_path.is_empty() {
            return None;
        }
        Some(self.heading_path.join(" > "))
    }

    /// Calculate completeness score (0.0 to 1.0) based on populated fields
    ///
    /// Higher scores indicate more complete metadata
    pub fn completeness_score(&self) -> f32 {
        let mut score = 0.0;
        let total = 9.0;

        if self.chapter.is_some() {
            score += 1.0;
        }
        if self.section.is_some() {
            score += 1.0;
        }
        if self.subsection.is_some() {
            score += 1.0;
        }
        if self.topic.is_some() {
            score += 1.0;
        }
        if !self.keywords.is_empty() {
            score += 1.0;
        }
        if self.summary.is_some() {
            score += 1.0;
        }
        if self.structural_level.is_some() {
            score += 1.0;
        }
        if self.position_in_document.is_some() {
            score += 1.0;
        }
        if !self.heading_path.is_empty() {
            score += 1.0;
        }

        score / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = ChunkMetadata::new();
        assert!(metadata.chapter.is_none());
        assert!(metadata.keywords.is_empty());
        assert_eq!(metadata.completeness_score(), 0.0);
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = ChunkMetadata::new()
            .with_chapter("Chapter 1".to_string())
            .with_section("Section 1.1".to_string())
            .with_keywords(vec!["test".to_string(), "metadata".to_string()])
            .with_summary("This is a test summary.".to_string());

        assert_eq!(metadata.chapter, Some("Chapter 1".to_string()));
        assert_eq!(metadata.section, Some("Section 1.1".to_string()));
        assert_eq!(metadata.keywords.len(), 2);
        assert!(metadata.has_structure_info());
        assert!(metadata.has_semantic_info());
    }

    #[test]
    fn test_heading_hierarchy() {
        let metadata = ChunkMetadata::new().with_heading_path(vec![
            "Chapter 1".to_string(),
            "Section 1.1".to_string(),
            "Subsection 1.1.1".to_string(),
        ]);

        assert_eq!(
            metadata.get_hierarchy_string(),
            Some("Chapter 1 > Section 1.1 > Subsection 1.1.1".to_string())
        );
    }

    #[test]
    fn test_deepest_heading() {
        let mut metadata = ChunkMetadata::new();
        assert!(metadata.get_deepest_heading().is_none());

        metadata.chapter = Some("Chapter 1".to_string());
        assert_eq!(metadata.get_deepest_heading(), Some(&"Chapter 1".to_string()));

        metadata.section = Some("Section 1.1".to_string());
        assert_eq!(
            metadata.get_deepest_heading(),
            Some(&"Section 1.1".to_string())
        );

        metadata.subsection = Some("Subsection 1.1.1".to_string());
        assert_eq!(
            metadata.get_deepest_heading(),
            Some(&"Subsection 1.1.1".to_string())
        );
    }

    #[test]
    fn test_completeness_score() {
        let mut metadata = ChunkMetadata::new();
        assert_eq!(metadata.completeness_score(), 0.0);

        metadata.chapter = Some("Chapter 1".to_string());
        metadata.keywords = vec!["test".to_string()];
        metadata.summary = Some("Summary".to_string());

        let score = metadata.completeness_score();
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_position_clamping() {
        let metadata = ChunkMetadata::new().with_position(1.5);
        assert_eq!(metadata.position_in_document, Some(1.0));

        let metadata2 = ChunkMetadata::new().with_position(-0.5);
        assert_eq!(metadata2.position_in_document, Some(0.0));
    }

    #[test]
    fn test_custom_metadata() {
        let metadata = ChunkMetadata::new()
            .add_custom("author".to_string(), "John Doe".to_string())
            .add_custom("date".to_string(), "2024-01-01".to_string());

        assert_eq!(metadata.custom.len(), 2);
        assert_eq!(
            metadata.custom.get("author"),
            Some(&"John Doe".to_string())
        );
    }

    #[test]
    fn test_serialization() {
        let metadata = ChunkMetadata::new()
            .with_chapter("Chapter 1".to_string())
            .with_keywords(vec!["test".to_string()])
            .with_position(0.5);

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata, deserialized);
    }
}
