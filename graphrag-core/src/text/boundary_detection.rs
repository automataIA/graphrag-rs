//! Semantic Boundary Detection for Boundary-Aware Chunking
//!
//! This module implements intelligent detection of semantic boundaries in text,
//! enabling chunking strategies that respect natural document structure.
//!
//! Key capabilities:
//! - Sentence boundary detection (NLTK-style rules)
//! - Paragraph detection (newline patterns)
//! - Heading detection (Markdown, RST, plaintext)
//! - List boundary detection
//! - Code block detection
//!
//! ## References
//!
//! - BAR-RAG Paper: "Boundary-Aware Retrieval-Augmented Generation"
//! - Target: +40% semantic coherence, -60% entity fragmentation

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for boundary detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDetectionConfig {
    /// Enable sentence boundary detection
    pub detect_sentences: bool,

    /// Enable paragraph boundary detection
    pub detect_paragraphs: bool,

    /// Enable heading boundary detection
    pub detect_headings: bool,

    /// Enable list boundary detection
    pub detect_lists: bool,

    /// Enable code block boundary detection
    pub detect_code_blocks: bool,

    /// Minimum sentence length (characters)
    pub min_sentence_length: usize,

    /// Heading markers (for plaintext detection)
    pub heading_markers: Vec<String>,
}

impl Default for BoundaryDetectionConfig {
    fn default() -> Self {
        Self {
            detect_sentences: true,
            detect_paragraphs: true,
            detect_headings: true,
            detect_lists: true,
            detect_code_blocks: true,
            min_sentence_length: 10,
            heading_markers: vec![
                "Chapter".to_string(),
                "Section".to_string(),
                "Introduction".to_string(),
                "Conclusion".to_string(),
            ],
        }
    }
}

/// Type of boundary detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Sentence boundary (. ! ?)
    Sentence,
    /// Paragraph boundary (double newline)
    Paragraph,
    /// Heading boundary (markdown #, RST underline)
    Heading,
    /// List boundary (bullet points, numbered lists)
    List,
    /// Code block boundary (```, indented blocks)
    CodeBlock,
}

/// Represents a detected boundary in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boundary {
    /// Position in text (byte offset)
    pub position: usize,

    /// Type of boundary
    pub boundary_type: BoundaryType,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,

    /// Optional context (e.g., heading text)
    pub context: Option<String>,
}

/// Boundary detector for semantic text segmentation
pub struct BoundaryDetector {
    config: BoundaryDetectionConfig,

    // Cached regex patterns
    sentence_endings: Regex,
    markdown_heading: Regex,
    numbered_list: Regex,
    bullet_list: Regex,
    code_block_fence: Regex,
    rst_heading_underline: Regex,
}

impl BoundaryDetector {
    /// Create a new boundary detector with default configuration
    pub fn new() -> Self {
        Self::with_config(BoundaryDetectionConfig::default())
    }

    /// Create a boundary detector with custom configuration
    pub fn with_config(config: BoundaryDetectionConfig) -> Self {
        Self {
            config,
            // Compile regex patterns once
            sentence_endings: Regex::new(r"[.!?]+[\s]+").unwrap(),
            markdown_heading: Regex::new(r"^#{1,6}\s+.+$").unwrap(),
            numbered_list: Regex::new(r"^\d+[.)]\s+").unwrap(),
            bullet_list: Regex::new(r"^[\-\*\+]\s+").unwrap(),
            code_block_fence: Regex::new(r"^```").unwrap(),
            rst_heading_underline: Regex::new("^[=\\-~^\"]+\\s*$").unwrap(),
        }
    }

    /// Detect all semantic boundaries in text
    pub fn detect_boundaries(&self, text: &str) -> Vec<Boundary> {
        let mut boundaries = Vec::new();

        if self.config.detect_sentences {
            boundaries.extend(self.detect_sentence_boundaries(text));
        }

        if self.config.detect_paragraphs {
            boundaries.extend(self.detect_paragraph_boundaries(text));
        }

        if self.config.detect_headings {
            boundaries.extend(self.detect_heading_boundaries(text));
        }

        if self.config.detect_lists {
            boundaries.extend(self.detect_list_boundaries(text));
        }

        if self.config.detect_code_blocks {
            boundaries.extend(self.detect_code_block_boundaries(text));
        }

        // Sort by position and deduplicate
        boundaries.sort_by_key(|b| b.position);
        boundaries.dedup_by_key(|b| b.position);

        boundaries
    }

    /// Detect sentence boundaries using NLTK-style rules
    fn detect_sentence_boundaries(&self, text: &str) -> Vec<Boundary> {
        let mut boundaries = Vec::new();

        // Common abbreviations that shouldn't end sentences
        let abbreviations: HashSet<&str> = [
            "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
            "etc.", "e.g.", "i.e.", "vs.", "cf.",
            "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
        ]
        .iter()
        .copied()
        .collect();

        // Find all potential sentence endings
        for mat in self.sentence_endings.find_iter(text) {
            let position = mat.start();

            // Check if this is a false positive (abbreviation)
            let before_text = &text[..position];
            let is_abbreviation = abbreviations.iter().any(|abbr| {
                before_text.ends_with(&abbr[..abbr.len() - 1])
            });

            if !is_abbreviation {
                // Check minimum sentence length
                let sentence_start = boundaries
                    .last()
                    .map(|b: &Boundary| b.position)
                    .unwrap_or(0);
                let sentence_length = position - sentence_start;

                if sentence_length >= self.config.min_sentence_length {
                    boundaries.push(Boundary {
                        position: mat.end(),
                        boundary_type: BoundaryType::Sentence,
                        confidence: 0.9,
                        context: None,
                    });
                }
            }
        }

        boundaries
    }

    /// Detect paragraph boundaries (double newlines)
    fn detect_paragraph_boundaries(&self, text: &str) -> Vec<Boundary> {
        let mut boundaries = Vec::new();

        // Look for double newlines (paragraph breaks)
        let paragraph_regex = Regex::new(r"\n\s*\n").unwrap();

        for mat in paragraph_regex.find_iter(text) {
            boundaries.push(Boundary {
                position: mat.end(),
                boundary_type: BoundaryType::Paragraph,
                confidence: 1.0,
                context: None,
            });
        }

        boundaries
    }

    /// Detect heading boundaries (Markdown, RST, plaintext)
    fn detect_heading_boundaries(&self, text: &str) -> Vec<Boundary> {
        let mut boundaries = Vec::new();

        let lines: Vec<&str> = text.lines().collect();
        let mut current_pos = 0;

        for (i, line) in lines.iter().enumerate() {
            let line_start = current_pos;
            let line_trimmed = line.trim();

            // Markdown headings (# ## ###)
            if self.markdown_heading.is_match(line) {
                let heading_text = line_trimmed.trim_start_matches('#').trim();
                boundaries.push(Boundary {
                    position: line_start,
                    boundary_type: BoundaryType::Heading,
                    confidence: 0.95,
                    context: Some(heading_text.to_string()),
                });
            }

            // RST-style underlined headings
            if i > 0 && self.rst_heading_underline.is_match(line_trimmed) {
                let prev_line = lines[i - 1].trim();
                if !prev_line.is_empty() && line_trimmed.len() >= prev_line.len() {
                    boundaries.push(Boundary {
                        position: line_start,
                        boundary_type: BoundaryType::Heading,
                        confidence: 0.9,
                        context: Some(prev_line.to_string()),
                    });
                }
            }

            // Plaintext heading detection (ALL CAPS, or starts with heading marker)
            if line_trimmed.len() > 3
                && line_trimmed.chars().all(|c| c.is_uppercase() || c.is_whitespace() || c.is_numeric())
                && line_trimmed.chars().any(|c| c.is_alphabetic())
            {
                boundaries.push(Boundary {
                    position: line_start,
                    boundary_type: BoundaryType::Heading,
                    confidence: 0.7,
                    context: Some(line_trimmed.to_string()),
                });
            }

            // Heading markers (Chapter, Section, etc.)
            for marker in &self.config.heading_markers {
                if line_trimmed.starts_with(marker) {
                    boundaries.push(Boundary {
                        position: line_start,
                        boundary_type: BoundaryType::Heading,
                        confidence: 0.85,
                        context: Some(line_trimmed.to_string()),
                    });
                    break;
                }
            }

            current_pos += line.len() + 1; // +1 for newline
        }

        boundaries
    }

    /// Detect list boundaries
    fn detect_list_boundaries(&self, text: &str) -> Vec<Boundary> {
        let mut boundaries = Vec::new();

        let lines: Vec<&str> = text.lines().collect();
        let mut current_pos = 0;
        let mut in_list = false;

        for line in lines {
            let line_trimmed = line.trim();

            // Check for list item
            let is_list_item = self.numbered_list.is_match(line_trimmed)
                || self.bullet_list.is_match(line_trimmed);

            // Transition into list
            if is_list_item && !in_list {
                boundaries.push(Boundary {
                    position: current_pos,
                    boundary_type: BoundaryType::List,
                    confidence: 0.9,
                    context: Some("list_start".to_string()),
                });
                in_list = true;
            }

            // Transition out of list
            if !is_list_item && in_list && !line_trimmed.is_empty() {
                boundaries.push(Boundary {
                    position: current_pos,
                    boundary_type: BoundaryType::List,
                    confidence: 0.9,
                    context: Some("list_end".to_string()),
                });
                in_list = false;
            }

            current_pos += line.len() + 1;
        }

        boundaries
    }

    /// Detect code block boundaries
    fn detect_code_block_boundaries(&self, text: &str) -> Vec<Boundary> {
        let mut boundaries = Vec::new();

        let lines: Vec<&str> = text.lines().collect();
        let mut current_pos = 0;
        let mut in_code_block = false;

        for line in lines {
            let line_trimmed = line.trim();

            // Fenced code blocks (```)
            if self.code_block_fence.is_match(line_trimmed) {
                boundaries.push(Boundary {
                    position: current_pos,
                    boundary_type: BoundaryType::CodeBlock,
                    confidence: 1.0,
                    context: if in_code_block {
                        Some("code_end".to_string())
                    } else {
                        Some("code_start".to_string())
                    },
                });
                in_code_block = !in_code_block;
            }

            // Indented code blocks (4+ spaces at start)
            if !in_code_block && line.starts_with("    ") && !line_trimmed.is_empty() {
                boundaries.push(Boundary {
                    position: current_pos,
                    boundary_type: BoundaryType::CodeBlock,
                    confidence: 0.7,
                    context: Some("indented_code".to_string()),
                });
            }

            current_pos += line.len() + 1;
        }

        boundaries
    }

    /// Get boundary positions of a specific type
    pub fn get_boundaries_by_type(
        &self,
        boundaries: &[Boundary],
        boundary_type: BoundaryType,
    ) -> Vec<usize> {
        boundaries
            .iter()
            .filter(|b| b.boundary_type == boundary_type)
            .map(|b| b.position)
            .collect()
    }

    /// Find the strongest boundary type at a given position
    pub fn get_strongest_boundary_at<'a>(
        &self,
        boundaries: &'a [Boundary],
        position: usize,
        tolerance: usize,
    ) -> Option<&'a Boundary> {
        boundaries
            .iter()
            .filter(|b| {
                let dist = if b.position > position {
                    b.position - position
                } else {
                    position - b.position
                };
                dist <= tolerance
            })
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for BoundaryDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_detection() {
        let detector = BoundaryDetector::new();
        let text = "This is a sentence. This is another! And a third?";

        let boundaries = detector.detect_sentence_boundaries(text);

        assert_eq!(boundaries.len(), 3);
        assert_eq!(boundaries[0].boundary_type, BoundaryType::Sentence);
    }

    #[test]
    fn test_abbreviation_handling() {
        let detector = BoundaryDetector::new();
        let text = "Dr. Smith went to the store. He bought milk.";

        let boundaries = detector.detect_sentence_boundaries(text);

        // Should detect only the second period, not "Dr."
        assert_eq!(boundaries.len(), 1);
    }

    #[test]
    fn test_paragraph_detection() {
        let detector = BoundaryDetector::new();
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";

        let boundaries = detector.detect_paragraph_boundaries(text);

        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0].boundary_type, BoundaryType::Paragraph);
    }

    #[test]
    fn test_markdown_heading_detection() {
        let detector = BoundaryDetector::new();
        let text = "# Main Heading\n\n## Subheading\n\n### Sub-subheading";

        let boundaries = detector.detect_heading_boundaries(text);

        assert!(boundaries.len() >= 3);
        assert!(boundaries.iter().all(|b| b.boundary_type == BoundaryType::Heading));
    }

    #[test]
    fn test_list_detection() {
        let detector = BoundaryDetector::new();
        let text = "Regular text\n- Item 1\n- Item 2\n* Item 3\nMore text";

        let boundaries = detector.detect_list_boundaries(text);

        assert!(boundaries.len() >= 2); // Start and end
        assert_eq!(boundaries[0].boundary_type, BoundaryType::List);
    }

    #[test]
    fn test_code_block_detection() {
        let detector = BoundaryDetector::new();
        let text = "Some text\n```python\ncode here\n```\nMore text";

        let boundaries = detector.detect_code_block_boundaries(text);

        assert_eq!(boundaries.len(), 2); // Start and end
        assert_eq!(boundaries[0].boundary_type, BoundaryType::CodeBlock);
    }

    #[test]
    fn test_combined_detection() {
        let detector = BoundaryDetector::new();
        let text = "# Heading\n\nFirst paragraph. Second sentence.\n\n- List item 1\n- List item 2\n\nLast paragraph.";

        let boundaries = detector.detect_boundaries(text);

        // Should detect headings, paragraphs, sentences, and lists
        assert!(boundaries.len() > 5);

        let types: HashSet<_> = boundaries.iter().map(|b| b.boundary_type).collect();
        assert!(types.contains(&BoundaryType::Heading));
        assert!(types.contains(&BoundaryType::Paragraph));
        assert!(types.contains(&BoundaryType::List));
    }

    #[test]
    fn test_get_strongest_boundary() {
        let detector = BoundaryDetector::new();
        let boundaries = vec![
            Boundary {
                position: 100,
                boundary_type: BoundaryType::Sentence,
                confidence: 0.7,
                context: None,
            },
            Boundary {
                position: 105,
                boundary_type: BoundaryType::Paragraph,
                confidence: 0.95,
                context: None,
            },
        ];

        let strongest = detector.get_strongest_boundary_at(&boundaries, 102, 10);
        assert!(strongest.is_some());
        assert_eq!(strongest.unwrap().boundary_type, BoundaryType::Paragraph);
        assert_eq!(strongest.unwrap().confidence, 0.95);
    }
}
