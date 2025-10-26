//! Document structure representation for hierarchical parsing
//!
//! This module provides data structures to represent the hierarchical structure
//! of documents, including headings, sections, and their relationships.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A heading in a document (e.g., chapter, section, subsection)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Heading {
    /// Hierarchical level (1 = chapter/h1, 2 = section/h2, 3 = subsection/h3, etc.)
    pub level: u8,

    /// Text content of the heading
    pub text: String,

    /// Starting character offset in the original document
    pub start_offset: usize,

    /// Ending character offset in the original document
    pub end_offset: usize,

    /// Line number in the document (if applicable)
    pub line_number: usize,

    /// Optional section number (e.g., "1.2.3", "Chapter 1")
    pub section_number: Option<String>,
}

impl Heading {
    /// Create a new heading
    pub fn new(level: u8, text: String, start_offset: usize, end_offset: usize) -> Self {
        Self {
            level,
            text,
            start_offset,
            end_offset,
            line_number: 0,
            section_number: None,
        }
    }

    /// Create heading with line number
    pub fn with_line_number(mut self, line_number: usize) -> Self {
        self.line_number = line_number;
        self
    }

    /// Create heading with section number
    pub fn with_section_number(mut self, section_number: String) -> Self {
        self.section_number = Some(section_number);
        self
    }

    /// Get a display string for the heading
    pub fn display_string(&self) -> String {
        if let Some(ref num) = self.section_number {
            format!("{} {}", num, self.text)
        } else {
            self.text.clone()
        }
    }
}

/// A section in a document, defined by a heading and its content range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// The heading that starts this section
    pub heading: Heading,

    /// Starting offset of the section content (after the heading)
    pub content_start: usize,

    /// Ending offset of the section content (before next heading or end of document)
    pub content_end: usize,

    /// Index of parent section in the sections array (None if root level)
    pub parent_section: Option<usize>,

    /// Indices of child sections in the sections array
    pub child_sections: Vec<usize>,
}

impl Section {
    /// Create a new section
    pub fn new(heading: Heading, content_start: usize, content_end: usize) -> Self {
        Self {
            heading,
            content_start,
            content_end,
            parent_section: None,
            child_sections: Vec::new(),
        }
    }

    /// Get the length of the section content in characters
    pub fn content_length(&self) -> usize {
        self.content_end.saturating_sub(self.content_start)
    }

    /// Check if this section contains the given offset
    pub fn contains_offset(&self, offset: usize) -> bool {
        offset >= self.heading.start_offset && offset < self.content_end
    }

    /// Check if this section is a root section (no parent)
    pub fn is_root(&self) -> bool {
        self.parent_section.is_none()
    }

    /// Check if this section has children
    pub fn has_children(&self) -> bool {
        !self.child_sections.is_empty()
    }
}

/// Hierarchical structure of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadingHierarchy {
    /// Indices of root-level sections
    pub root_sections: Vec<usize>,

    /// Mapping from section index to depth (0 = root, 1 = child of root, etc.)
    pub depth_map: HashMap<usize, usize>,
}

impl HeadingHierarchy {
    /// Create a new empty hierarchy
    pub fn new() -> Self {
        Self {
            root_sections: Vec::new(),
            depth_map: HashMap::new(),
        }
    }

    /// Get the depth of a section in the hierarchy
    pub fn get_depth(&self, section_idx: usize) -> Option<usize> {
        self.depth_map.get(&section_idx).copied()
    }

    /// Check if a section is at root level
    pub fn is_root(&self, section_idx: usize) -> bool {
        self.root_sections.contains(&section_idx)
    }
}

impl Default for HeadingHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete document structure with headings and sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStructure {
    /// All headings found in the document, in order of appearance
    pub headings: Vec<Heading>,

    /// All sections derived from headings
    pub sections: Vec<Section>,

    /// Hierarchical relationships between sections
    pub hierarchy: HeadingHierarchy,
}

impl DocumentStructure {
    /// Create a new empty document structure
    pub fn new() -> Self {
        Self {
            headings: Vec::new(),
            sections: Vec::new(),
            hierarchy: HeadingHierarchy::new(),
        }
    }

    /// Find the section index that contains the given offset
    pub fn find_section_containing_offset(&self, offset: usize) -> Option<usize> {
        self.sections
            .iter()
            .position(|section| section.contains_offset(offset))
    }

    /// Get the heading path from root to the given section
    ///
    /// Returns a vector of heading texts in hierarchical order
    /// Example: ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
    pub fn get_heading_path(&self, section_idx: usize) -> Vec<String> {
        let mut path = Vec::new();
        let mut current_idx = Some(section_idx);

        // Traverse up to root, collecting headings
        while let Some(idx) = current_idx {
            if idx < self.sections.len() {
                let section = &self.sections[idx];
                path.push(section.heading.display_string());
                current_idx = section.parent_section;
            } else {
                break;
            }
        }

        // Reverse to get root-to-leaf order
        path.reverse();
        path
    }

    /// Get all sections at a specific hierarchical level
    pub fn get_sections_at_level(&self, level: u8) -> Vec<&Section> {
        self.sections
            .iter()
            .filter(|s| s.heading.level == level)
            .collect()
    }

    /// Get the total number of sections
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Get the maximum depth of the hierarchy
    pub fn max_depth(&self) -> usize {
        self.hierarchy
            .depth_map
            .values()
            .max()
            .copied()
            .unwrap_or(0)
    }

    /// Check if the document has any structure
    pub fn has_structure(&self) -> bool {
        !self.headings.is_empty()
    }

    /// Get statistics about the document structure
    pub fn get_statistics(&self) -> StructureStatistics {
        let mut level_counts: HashMap<u8, usize> = HashMap::new();
        for heading in &self.headings {
            *level_counts.entry(heading.level).or_insert(0) += 1;
        }

        StructureStatistics {
            total_headings: self.headings.len(),
            total_sections: self.sections.len(),
            max_depth: self.max_depth(),
            level_counts,
            root_sections: self.hierarchy.root_sections.len(),
        }
    }
}

impl Default for DocumentStructure {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureStatistics {
    /// Total number of headings
    pub total_headings: usize,

    /// Total number of sections
    pub total_sections: usize,

    /// Maximum depth of the hierarchy
    pub max_depth: usize,

    /// Count of headings at each level
    pub level_counts: HashMap<u8, usize>,

    /// Number of root-level sections
    pub root_sections: usize,
}

impl StructureStatistics {
    /// Print a human-readable summary of the statistics
    pub fn print_summary(&self) {
        println!("Document Structure Statistics:");
        println!("  Total headings: {}", self.total_headings);
        println!("  Total sections: {}", self.total_sections);
        println!("  Max depth: {}", self.max_depth);
        println!("  Root sections: {}", self.root_sections);
        println!("  Headings by level:");
        let mut levels: Vec<_> = self.level_counts.iter().collect();
        levels.sort_by_key(|(level, _)| *level);
        for (level, count) in levels {
            println!("    Level {}: {} headings", level, count);
        }
    }
}

/// Section numbering format (e.g., "1.2.3", "Chapter 1", "I.A.1")
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SectionNumberFormat {
    /// Numeric: 1, 2, 3
    Numeric,
    /// Decimal: 1.1, 1.2, 2.1
    Decimal,
    /// Roman: I, II, III
    Roman,
    /// Alphabetic: A, B, C
    Alphabetic,
    /// Mixed: Chapter 1, Section A, etc.
    Mixed,
}

/// Parsed section number with format information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionNumber {
    /// Original string representation
    pub raw: String,

    /// Detected format
    pub format: SectionNumberFormat,

    /// Numeric components (e.g., [1, 2, 3] for "1.2.3")
    pub components: Vec<usize>,
}

impl SectionNumber {
    /// Get the depth of the section number (number of components)
    pub fn depth(&self) -> u8 {
        self.components.len().min(255) as u8
    }

    /// Check if this section number is deeper than another
    pub fn is_deeper_than(&self, other: &SectionNumber) -> bool {
        self.depth() > other.depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heading_creation() {
        let heading = Heading::new(1, "Chapter 1".to_string(), 0, 9)
            .with_line_number(5)
            .with_section_number("1".to_string());

        assert_eq!(heading.level, 1);
        assert_eq!(heading.text, "Chapter 1");
        assert_eq!(heading.line_number, 5);
        assert_eq!(heading.display_string(), "1 Chapter 1");
    }

    #[test]
    fn test_section_contains_offset() {
        let heading = Heading::new(1, "Test".to_string(), 0, 10);
        let section = Section::new(heading, 10, 100);

        assert!(section.contains_offset(0));
        assert!(section.contains_offset(50));
        assert!(!section.contains_offset(100));
        assert!(!section.contains_offset(150));
    }

    #[test]
    fn test_document_structure() {
        let mut structure = DocumentStructure::new();

        let h1 = Heading::new(1, "Chapter 1".to_string(), 0, 9);
        let h2 = Heading::new(2, "Section 1.1".to_string(), 50, 61);

        structure.headings.push(h1.clone());
        structure.headings.push(h2.clone());

        let s1 = Section::new(h1, 10, 50);
        let mut s2 = Section::new(h2, 62, 100);
        s2.parent_section = Some(0);

        structure.sections.push(s1);
        structure.sections.push(s2);
        structure.hierarchy.root_sections.push(0);

        assert_eq!(structure.section_count(), 2);
        assert!(structure.has_structure());
        assert_eq!(structure.find_section_containing_offset(25), Some(0));
        assert_eq!(structure.find_section_containing_offset(75), Some(1));
    }

    #[test]
    fn test_heading_path() {
        let mut structure = DocumentStructure::new();

        let h1 = Heading::new(1, "Chapter 1".to_string(), 0, 9);
        let h2 = Heading::new(2, "Section 1.1".to_string(), 50, 61);
        let h3 = Heading::new(3, "Subsection 1.1.1".to_string(), 100, 116);

        structure.headings.push(h1.clone());
        structure.headings.push(h2.clone());
        structure.headings.push(h3.clone());

        let s1 = Section::new(h1, 10, 50);
        let mut s2 = Section::new(h2, 62, 100);
        s2.parent_section = Some(0);
        let mut s3 = Section::new(h3, 117, 200);
        s3.parent_section = Some(1);

        structure.sections.push(s1);
        structure.sections.push(s2);
        structure.sections.push(s3);

        let path = structure.get_heading_path(2);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "Chapter 1");
        assert_eq!(path[1], "Section 1.1");
        assert_eq!(path[2], "Subsection 1.1.1");
    }

    #[test]
    fn test_structure_statistics() {
        let mut structure = DocumentStructure::new();

        structure.headings.push(Heading::new(1, "H1".to_string(), 0, 2));
        structure.headings.push(Heading::new(2, "H2".to_string(), 10, 12));
        structure.headings.push(Heading::new(2, "H2b".to_string(), 20, 23));

        let stats = structure.get_statistics();
        assert_eq!(stats.total_headings, 3);
        assert_eq!(stats.level_counts.get(&1), Some(&1));
        assert_eq!(stats.level_counts.get(&2), Some(&2));
    }

    #[test]
    fn test_section_number_depth() {
        let section_num = SectionNumber {
            raw: "1.2.3".to_string(),
            format: SectionNumberFormat::Decimal,
            components: vec![1, 2, 3],
        };

        assert_eq!(section_num.depth(), 3);
    }
}
