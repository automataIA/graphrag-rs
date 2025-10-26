//! HTML layout parser
//!
//! Extracts document structure from HTML documents by parsing heading tags (h1-h6).
//! This is a simplified parser - for production use, consider using a proper HTML parsing library.

use crate::text::{
    document_structure::{DocumentStructure, Heading, HeadingHierarchy, Section},
    layout_parser::LayoutParser,
};

/// Parser for HTML documents
pub struct HtmlLayoutParser;

impl HtmlLayoutParser {
    /// Create new HTML parser
    pub fn new() -> Self {
        Self
    }

    /// Extract text content from an HTML tag
    fn extract_text_content(tag_content: &str) -> String {
        // Remove any nested HTML tags
        let mut text = String::new();
        let mut inside_tag = false;

        for ch in tag_content.chars() {
            match ch {
                '<' => inside_tag = true,
                '>' => inside_tag = false,
                _ if !inside_tag => text.push(ch),
                _ => {}
            }
        }

        text.trim().to_string()
    }

    /// Parse HTML headings from content
    fn parse_headings(&self, content: &str) -> Vec<Heading> {
        let mut headings = Vec::new();
        let mut current_offset = 0;

        // Simple regex-like pattern matching for heading tags
        let lines: Vec<&str> = content.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            // Look for heading tags h1-h6
            for level in 1..=6 {
                let open_tag = format!("<h{}", level);
                let close_tag = format!("</h{}>", level);

                if let Some(start_idx) = line.to_lowercase().find(&open_tag) {
                    if let Some(end_idx) = line.to_lowercase().find(&close_tag) {
                        // Extract the tag content (everything between > and </h)
                        if let Some(content_start) = line[start_idx..].find('>') {
                            let actual_start = start_idx + content_start + 1;
                            let tag_content = &line[actual_start..end_idx];
                            let text = Self::extract_text_content(tag_content);

                            if !text.is_empty() {
                                let heading = Heading::new(
                                    level as u8,
                                    text,
                                    current_offset + start_idx,
                                    current_offset + end_idx + close_tag.len(),
                                )
                                .with_line_number(line_num);

                                headings.push(heading);
                            }
                        }
                    }
                }
            }

            current_offset += line.len() + 1; // +1 for newline
        }

        headings
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

impl Default for HtmlLayoutParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutParser for HtmlLayoutParser {
    fn parse(&self, content: &str) -> DocumentStructure {
        let headings = self.parse_headings(content);
        let mut sections = self.build_sections(&headings, content);
        let hierarchy = self.build_hierarchy(&mut sections);

        DocumentStructure {
            headings,
            sections,
            hierarchy,
        }
    }

    fn supports_format(&self, format: &str) -> bool {
        matches!(format.to_lowercase().as_str(), "html" | "htm")
    }

    fn name(&self) -> &'static str {
        "HtmlLayoutParser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_heading_parsing() {
        let parser = HtmlLayoutParser::new();
        let content = r#"
<html>
<body>
<h1>Chapter 1: Introduction</h1>
<p>Some introductory text.</p>
<h2>Section 1.1: Background</h2>
<p>Background information.</p>
<h3>Subsection 1.1.1: Details</h3>
<p>Detailed information.</p>
<h2>Section 1.2: Methods</h2>
<p>Methodology.</p>
</body>
</html>
"#;

        let structure = parser.parse(content);

        assert_eq!(structure.headings.len(), 4);
        assert_eq!(structure.headings[0].level, 1);
        assert_eq!(structure.headings[0].text, "Chapter 1: Introduction");
        assert_eq!(structure.headings[1].level, 2);
        assert_eq!(structure.headings[1].text, "Section 1.1: Background");
        assert_eq!(structure.headings[2].level, 3);
        assert_eq!(structure.headings[2].text, "Subsection 1.1.1: Details");
        assert_eq!(structure.headings[3].level, 2);
        assert_eq!(structure.headings[3].text, "Section 1.2: Methods");
    }

    #[test]
    fn test_html_hierarchy() {
        let parser = HtmlLayoutParser::new();
        let content = r#"<h1>Main</h1>
<h2>Sub1</h2>
<h3>SubSub1</h3>
<h2>Sub2</h2>"#;

        let structure = parser.parse(content);

        assert_eq!(structure.hierarchy.root_sections.len(), 1); // One h1
        assert_eq!(structure.sections.len(), 4);

        // Check hierarchy relationships
        assert_eq!(structure.sections[1].parent_section, Some(0)); // h2 parent is h1
        assert_eq!(structure.sections[2].parent_section, Some(1)); // h3 parent is h2
        assert_eq!(structure.sections[3].parent_section, Some(0)); // h2 parent is h1
    }

    #[test]
    fn test_nested_tags_in_heading() {
        let parser = HtmlLayoutParser::new();
        let content = "<h1>Chapter <em>One</em></h1><p>Content</p>";

        let structure = parser.parse(content);

        assert_eq!(structure.headings.len(), 1);
        assert_eq!(structure.headings[0].text, "Chapter One"); // Nested tags removed
    }

    #[test]
    fn test_format_support() {
        let parser = HtmlLayoutParser::new();
        assert!(parser.supports_format("html"));
        assert!(parser.supports_format("HTML"));
        assert!(parser.supports_format("htm"));
        assert!(!parser.supports_format("md"));
    }
}
