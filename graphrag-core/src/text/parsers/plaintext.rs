//! Plain text layout parser with heuristic-based detection

use crate::text::{
    document_structure::{DocumentStructure, Heading, HeadingHierarchy, Section},
    layout_parser::LayoutParser,
    TextAnalyzer,
};

/// Parser for plain text documents using heuristics
pub struct PlainTextLayoutParser;

impl PlainTextLayoutParser {
    /// Create new plain text parser
    pub fn new() -> Self {
        Self
    }

    /// Build sections from headings
    fn build_sections(&self, headings: &[Heading], content: &str) -> Vec<Section> {
        let mut sections = Vec::new();

        for (i, heading) in headings.iter().enumerate() {
            let content_start = heading.end_offset;
            let content_end = headings
                .get(i + 1)
                .map(|h| h.start_offset)
                .unwrap_or(content.len());

            sections.push(Section::new(heading.clone(), content_start, content_end));
        }

        sections
    }

    /// Build hierarchy from sections
    fn build_hierarchy(&self, sections: &mut [Section]) -> HeadingHierarchy {
        let mut hierarchy = HeadingHierarchy::new();
        let mut stack: Vec<usize> = Vec::new();

        for idx in 0..sections.len() {
            let section_level = sections[idx].heading.level;

            while let Some(&parent_idx) = stack.last() {
                if sections[parent_idx].heading.level < section_level {
                    break;
                }
                stack.pop();
            }

            if let Some(&parent_idx) = stack.last() {
                sections[parent_idx].child_sections.push(idx);
                sections[idx].parent_section = Some(parent_idx);
            } else {
                hierarchy.root_sections.push(idx);
            }

            stack.push(idx);
        }

        // Build depth map
        for (idx, section) in sections.iter().enumerate() {
            let mut depth = 0;
            let mut current = section.parent_section;
            while let Some(parent_idx) = current {
                depth += 1;
                current = sections[parent_idx].parent_section;
            }
            hierarchy.depth_map.insert(idx, depth);
        }

        hierarchy
    }
}

impl Default for PlainTextLayoutParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutParser for PlainTextLayoutParser {
    fn parse(&self, content: &str) -> DocumentStructure {
        let mut headings = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_offset = 0;

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                current_offset += line.len() + 1;
                i += 1;
                continue;
            }

            let mut detected_level: Option<u8> = None;
            let heading_text = trimmed.to_string();

            // Heuristic 1: Check if next line is underline
            if i + 1 < lines.len() {
                let next_line = lines[i + 1].trim();
                if let Some(level) = TextAnalyzer::is_underline(next_line) {
                    detected_level = Some(level);
                    // Skip the underline in next iteration
                    i += 1;
                    current_offset += line.len() + 1;
                    current_offset += next_line.len() + 1;
                }
            }

            // Heuristic 2: ALL CAPS detection
            if detected_level.is_none() {
                if let Some(level) = TextAnalyzer::detect_heading_level(line) {
                    detected_level = Some(level);
                }
            }

            // If heading detected, add it
            if let Some(level) = detected_level {
                let heading = Heading::new(level, heading_text, current_offset, current_offset + line.len())
                    .with_line_number(i);

                headings.push(heading);
            }

            if detected_level.is_none() {
                current_offset += line.len() + 1;
            }

            i += 1;
        }

        let mut sections = self.build_sections(&headings, content);
        let hierarchy = self.build_hierarchy(&mut sections);

        DocumentStructure {
            headings,
            sections,
            hierarchy,
        }
    }

    fn supports_format(&self, format: &str) -> bool {
        matches!(format.to_lowercase().as_str(), "text" | "txt" | "plain")
    }

    fn name(&self) -> &'static str {
        "PlainTextLayoutParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_underline_detection() {
        let parser = PlainTextLayoutParser::new();
        let content = "Chapter One\n===========\n\nSome text\n\nSection 1.1\n-----------\n\nMore text";

        let structure = parser.parse(content);

        assert!(structure.headings.len() >= 2);
        assert_eq!(structure.headings[0].level, 1);
        assert_eq!(structure.headings[0].text, "Chapter One");
    }

    #[test]
    fn test_all_caps_detection() {
        let parser = PlainTextLayoutParser::new();
        let content = "INTRODUCTION\n\nThis is the intro.\n\nBACKGROUND\n\nSome background.";

        let structure = parser.parse(content);

        assert!(structure.headings.len() >= 2);
        assert!(structure.headings[0].text.contains("INTRODUCTION"));
    }

    #[test]
    fn test_numbered_sections() {
        let parser = PlainTextLayoutParser::new();
        let content = "1. First Chapter\n\nText here.\n\n1.1 Subsection\n\nMore text.";

        let structure = parser.parse(content);

        // Should detect numbered headings
        assert!(!structure.headings.is_empty());
    }
}
