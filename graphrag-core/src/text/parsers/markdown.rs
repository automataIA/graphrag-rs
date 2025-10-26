//! Markdown layout parser

use crate::text::{
    document_structure::{DocumentStructure, Heading, HeadingHierarchy, Section},
    layout_parser::LayoutParser,
};

/// Parser for Markdown documents
pub struct MarkdownLayoutParser;

impl MarkdownLayoutParser {
    /// Create new Markdown parser
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

            // Pop stack until we find parent
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

impl Default for MarkdownLayoutParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutParser for MarkdownLayoutParser {
    fn parse(&self, content: &str) -> DocumentStructure {
        let mut headings = Vec::new();
        let mut current_offset = 0;

        for (line_num, line) in content.lines().enumerate() {
            // Detect markdown headings: # ## ### etc.
            if line.trim_start().starts_with('#') {
                let trimmed = line.trim();
                let level = trimmed.chars().take_while(|&c| c == '#').count();

                if level > 0 && level <= 6 {
                    // Verify proper markdown (space after hashes)
                    if trimmed.len() > level {
                        let after_hashes = trimmed.chars().nth(level);
                        if after_hashes == Some(' ') || after_hashes.is_none() {
                            let text = trimmed[level..].trim().to_string();
                            if !text.is_empty() {
                                headings.push(
                                    Heading::new(
                                        level.min(255) as u8,
                                        text,
                                        current_offset,
                                        current_offset + line.len(),
                                    )
                                    .with_line_number(line_num),
                                );
                            }
                        }
                    }
                }
            }

            current_offset += line.len() + 1; // +1 for newline
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
        matches!(format.to_lowercase().as_str(), "markdown" | "md")
    }

    fn name(&self) -> &'static str {
        "MarkdownLayoutParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_parsing() {
        let parser = MarkdownLayoutParser::new();
        let content = "# Chapter 1\n\nSome text\n\n## Section 1.1\n\nMore text\n\n### Subsection 1.1.1\n\nDetails";

        let structure = parser.parse(content);

        assert_eq!(structure.headings.len(), 3);
        assert_eq!(structure.headings[0].level, 1);
        assert_eq!(structure.headings[0].text, "Chapter 1");
        assert_eq!(structure.headings[1].level, 2);
        assert_eq!(structure.headings[1].text, "Section 1.1");
        assert_eq!(structure.headings[2].level, 3);
    }

    #[test]
    fn test_hierarchy_building() {
        let parser = MarkdownLayoutParser::new();
        let content = "# H1\n## H2\n### H3\n## H2b\n# H1b";

        let structure = parser.parse(content);

        assert_eq!(structure.hierarchy.root_sections.len(), 2); // Two H1s
        assert!(structure.sections[1].parent_section == Some(0)); // H2 parent is H1
    }
}
