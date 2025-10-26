//! Text analysis utilities for document structure detection
//!
//! This module provides algorithms for analyzing text structure, including
//! heading detection, section numbering extraction, and statistical analysis.

use crate::text::document_structure::{SectionNumber, SectionNumberFormat};
use regex::Regex;
use std::sync::OnceLock;

/// Text analyzer for structural analysis
pub struct TextAnalyzer;

impl TextAnalyzer {
    /// Detect if a line is a heading and determine its level
    ///
    /// Supports multiple heading formats:
    /// - Markdown: #, ##, ###, etc.
    /// - Plain text: ALL CAPS, numeric prefixes
    /// - Underlined text (detected by caller)
    ///
    /// Returns Some(level) if detected, where level 1 is highest (chapter)
    pub fn detect_heading_level(line: &str) -> Option<u8> {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            return None;
        }

        // Markdown heading detection: # ## ### etc.
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            if level > 0 && level <= 6 {
                // Verify there's a space after the hashes (proper markdown)
                if trimmed.len() > level && trimmed.chars().nth(level) == Some(' ') {
                    return Some(level.min(255) as u8);
                }
            }
        }

        // ALL CAPS detection (likely chapter/section heading)
        if trimmed.len() >= 5 && Self::is_all_caps(trimmed) {
            // Shorter ALL CAPS lines are more likely to be high-level headings
            let level = if trimmed.len() < 20 {
                1 // Short ALL CAPS = chapter
            } else if trimmed.len() < 40 {
                2 // Medium ALL CAPS = section
            } else {
                3 // Long ALL CAPS = subsection
            };
            return Some(level);
        }

        // Numbered heading detection: "1.", "1.1", "Chapter 1", etc.
        if let Some(section_num) = Self::extract_section_number(trimmed) {
            let level = section_num.depth();
            if level > 0 && level <= 6 {
                return Some(level);
            }
        }

        None
    }

    /// Check if text is ALL CAPS (ignoring non-alphabetic characters)
    fn is_all_caps(text: &str) -> bool {
        let letters: String = text.chars().filter(|c| c.is_alphabetic()).collect();
        !letters.is_empty() && letters.chars().all(|c| c.is_uppercase())
    }

    /// Extract section number from heading text
    ///
    /// Recognizes patterns like:
    /// - "1.", "2.", "3."
    /// - "1.1", "1.2.3", "2.3.4.5"
    /// - "Chapter 1", "Section 2.1"
    /// - "I.", "II.", "III." (Roman numerals)
    /// - "A.", "B.", "C." (Alphabetic)
    pub fn extract_section_number(text: &str) -> Option<SectionNumber> {
        static DECIMAL_REGEX: OnceLock<Regex> = OnceLock::new();
        static ROMAN_REGEX: OnceLock<Regex> = OnceLock::new();
        static ALPHA_REGEX: OnceLock<Regex> = OnceLock::new();
        static CHAPTER_REGEX: OnceLock<Regex> = OnceLock::new();

        let decimal_re = DECIMAL_REGEX.get_or_init(|| {
            Regex::new(r"^(\d+(?:\.\d+)*)\s*[.:]?\s").unwrap()
        });

        let roman_re = ROMAN_REGEX.get_or_init(|| {
            Regex::new(r"^([IVXLCDM]+)[.:]?\s").unwrap()
        });

        let alpha_re = ALPHA_REGEX.get_or_init(|| {
            Regex::new(r"^([A-Z])[.:]?\s").unwrap()
        });

        let chapter_re = CHAPTER_REGEX.get_or_init(|| {
            Regex::new(r"(?i)^(chapter|section|part|appendix)\s+(\d+|[IVXLCDM]+|[A-Z])\b").unwrap()
        });

        // Try decimal numbering (most common)
        if let Some(caps) = decimal_re.captures(text) {
            if let Some(num_str) = caps.get(1) {
                let components: Vec<usize> = num_str
                    .as_str()
                    .split('.')
                    .filter_map(|s| s.parse().ok())
                    .collect();

                if !components.is_empty() {
                    return Some(SectionNumber {
                        raw: num_str.as_str().to_string(),
                        format: SectionNumberFormat::Decimal,
                        components,
                    });
                }
            }
        }

        // Try chapter/section keywords
        if let Some(caps) = chapter_re.captures(text) {
            if let Some(num_match) = caps.get(2) {
                let num_str = num_match.as_str();

                // Try parsing as decimal
                if let Ok(num) = num_str.parse::<usize>() {
                    return Some(SectionNumber {
                        raw: format!("{} {}", caps.get(1)?.as_str(), num_str),
                        format: SectionNumberFormat::Mixed,
                        components: vec![num],
                    });
                }

                // Try parsing as Roman numeral
                if let Some(num) = Self::parse_roman_numeral(num_str) {
                    return Some(SectionNumber {
                        raw: format!("{} {}", caps.get(1)?.as_str(), num_str),
                        format: SectionNumberFormat::Mixed,
                        components: vec![num],
                    });
                }

                // Try parsing as alphabetic
                if num_str.len() == 1 {
                    if let Some(ch) = num_str.chars().next() {
                        if ch.is_ascii_uppercase() {
                            let num = (ch as usize) - ('A' as usize) + 1;
                            return Some(SectionNumber {
                                raw: format!("{} {}", caps.get(1)?.as_str(), num_str),
                                format: SectionNumberFormat::Mixed,
                                components: vec![num],
                            });
                        }
                    }
                }
            }
        }

        // Try Roman numerals
        if let Some(caps) = roman_re.captures(text) {
            if let Some(roman_str) = caps.get(1) {
                if let Some(num) = Self::parse_roman_numeral(roman_str.as_str()) {
                    return Some(SectionNumber {
                        raw: roman_str.as_str().to_string(),
                        format: SectionNumberFormat::Roman,
                        components: vec![num],
                    });
                }
            }
        }

        // Try alphabetic
        if let Some(caps) = alpha_re.captures(text) {
            if let Some(letter) = caps.get(1) {
                let ch = letter.as_str().chars().next()?;
                let num = (ch as usize) - ('A' as usize) + 1;
                return Some(SectionNumber {
                    raw: letter.as_str().to_string(),
                    format: SectionNumberFormat::Alphabetic,
                    components: vec![num],
                });
            }
        }

        None
    }

    /// Parse Roman numeral to decimal
    fn parse_roman_numeral(roman: &str) -> Option<usize> {
        let mut result = 0;
        let mut prev_value = 0;

        for ch in roman.chars().rev() {
            let value = match ch {
                'I' => 1,
                'V' => 5,
                'X' => 10,
                'L' => 50,
                'C' => 100,
                'D' => 500,
                'M' => 1000,
                _ => return None,
            };

            if value < prev_value {
                result -= value;
            } else {
                result += value;
            }
            prev_value = value;
        }

        Some(result)
    }

    /// Find positions of blank lines (paragraph separators)
    ///
    /// Returns character offsets where blank lines occur
    pub fn find_blank_line_positions(text: &str) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut current_offset = 0;
        let mut prev_was_blank = false;

        for line in text.lines() {
            let is_blank = line.trim().is_empty();

            if is_blank && !prev_was_blank {
                positions.push(current_offset);
            }

            prev_was_blank = is_blank;
            current_offset += line.len() + 1; // +1 for newline
        }

        positions
    }

    /// Calculate statistics about text
    pub fn calculate_statistics(text: &str) -> TextStats {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        // Count sentences (simple heuristic)
        let sentence_endings = ['.', '!', '?'];
        let sentence_count = text
            .chars()
            .filter(|c| sentence_endings.contains(c))
            .count()
            .max(1); // At least 1 sentence

        let avg_sentence_length = if sentence_count > 0 {
            word_count as f32 / sentence_count as f32
        } else {
            0.0
        };

        // Count paragraphs (separated by blank lines)
        let paragraph_count = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .count()
            .max(1); // At least 1 paragraph

        let char_count = text.chars().count();

        TextStats {
            word_count,
            sentence_count,
            paragraph_count,
            char_count,
            avg_sentence_length,
            avg_word_length: if word_count > 0 {
                char_count as f32 / word_count as f32
            } else {
                0.0
            },
        }
    }

    /// Detect if a line is underlined (for plain text heading detection)
    ///
    /// Checks if next_line consists entirely of underline characters (=, -, _)
    pub fn is_underline(line: &str) -> Option<u8> {
        let trimmed = line.trim();

        if trimmed.len() < 3 {
            return None;
        }

        // Check if line is all underline characters
        if trimmed.chars().all(|c| c == '=') {
            Some(1) // === is level 1 (chapter)
        } else if trimmed.chars().all(|c| c == '-') {
            Some(2) // --- is level 2 (section)
        } else if trimmed.chars().all(|c| c == '_') {
            Some(3) // ___ is level 3 (subsection)
        } else {
            None
        }
    }

    /// Extract potential title from text (first non-empty line or ALL CAPS line)
    pub fn extract_title(text: &str) -> Option<String> {
        for line in text.lines().take(10) {
            // Check first 10 lines
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            // If it's ALL CAPS and reasonably short, it's likely the title
            if Self::is_all_caps(trimmed) && trimmed.len() < 100 {
                return Some(trimmed.to_string());
            }

            // If it looks like a heading, use it
            if Self::detect_heading_level(line).is_some() {
                // Strip heading markers
                let clean = trimmed
                    .trim_start_matches('#')
                    .trim_start_matches(|c: char| c.is_numeric() || c == '.')
                    .trim();
                if !clean.is_empty() {
                    return Some(clean.to_string());
                }
            }

            // Otherwise, first non-empty line
            if trimmed.len() > 5 {
                return Some(trimmed.to_string());
            }
        }

        None
    }
}

/// Text statistics
#[derive(Debug, Clone)]
pub struct TextStats {
    /// Total word count
    pub word_count: usize,
    /// Total sentence count
    pub sentence_count: usize,
    /// Total paragraph count
    pub paragraph_count: usize,
    /// Total character count
    pub char_count: usize,
    /// Average words per sentence
    pub avg_sentence_length: f32,
    /// Average characters per word
    pub avg_word_length: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_heading_detection() {
        assert_eq!(TextAnalyzer::detect_heading_level("# Chapter 1"), Some(1));
        assert_eq!(TextAnalyzer::detect_heading_level("## Section 1.1"), Some(2));
        assert_eq!(
            TextAnalyzer::detect_heading_level("### Subsection 1.1.1"),
            Some(3)
        );
        assert_eq!(TextAnalyzer::detect_heading_level("#### Level 4"), Some(4));
        assert_eq!(TextAnalyzer::detect_heading_level("#No space"), None);
    }

    #[test]
    fn test_all_caps_detection() {
        assert_eq!(TextAnalyzer::detect_heading_level("CHAPTER ONE"), Some(1));
        assert_eq!(
            TextAnalyzer::detect_heading_level("INTRODUCTION TO MACHINE LEARNING"),
            Some(2)
        );
        assert_eq!(TextAnalyzer::detect_heading_level("This is not ALL CAPS"), None);
    }

    #[test]
    fn test_section_number_extraction() {
        // Decimal numbering
        let sec1 = TextAnalyzer::extract_section_number("1. Introduction").unwrap();
        assert_eq!(sec1.components, vec![1]);
        assert_eq!(sec1.format, SectionNumberFormat::Decimal);

        let sec2 = TextAnalyzer::extract_section_number("1.2.3 Subsection").unwrap();
        assert_eq!(sec2.components, vec![1, 2, 3]);

        // Chapter/Section keywords
        let sec3 = TextAnalyzer::extract_section_number("Chapter 1 Introduction").unwrap();
        assert_eq!(sec3.components, vec![1]);
        assert_eq!(sec3.format, SectionNumberFormat::Mixed);

        // Roman numerals
        let sec4 = TextAnalyzer::extract_section_number("I. First Chapter").unwrap();
        assert_eq!(sec4.components, vec![1]);
        assert_eq!(sec4.format, SectionNumberFormat::Roman);

        let sec5 = TextAnalyzer::extract_section_number("IV. Fourth Chapter").unwrap();
        assert_eq!(sec5.components, vec![4]);
    }

    #[test]
    fn test_roman_numeral_parsing() {
        assert_eq!(TextAnalyzer::parse_roman_numeral("I"), Some(1));
        assert_eq!(TextAnalyzer::parse_roman_numeral("IV"), Some(4));
        assert_eq!(TextAnalyzer::parse_roman_numeral("IX"), Some(9));
        assert_eq!(TextAnalyzer::parse_roman_numeral("XL"), Some(40));
        assert_eq!(TextAnalyzer::parse_roman_numeral("MCMXCIV"), Some(1994));
        assert_eq!(TextAnalyzer::parse_roman_numeral("ABC"), None);
    }

    #[test]
    fn test_blank_line_detection() {
        let text = "Line 1\n\nLine 2\n\n\nLine 3";
        let positions = TextAnalyzer::find_blank_line_positions(text);
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_text_statistics() {
        let text = "This is a test. It has two sentences.";
        let stats = TextAnalyzer::calculate_statistics(text);

        assert_eq!(stats.sentence_count, 2);
        assert!(stats.word_count >= 7);
        assert!(stats.avg_sentence_length > 0.0);
    }

    #[test]
    fn test_underline_detection() {
        assert_eq!(TextAnalyzer::is_underline("====="), Some(1));
        assert_eq!(TextAnalyzer::is_underline("-----"), Some(2));
        assert_eq!(TextAnalyzer::is_underline("_____"), Some(3));
        assert_eq!(TextAnalyzer::is_underline("===---"), None);
    }

    #[test]
    fn test_title_extraction() {
        let text = "# Main Title\n\nSome content here.";
        let title = TextAnalyzer::extract_title(text);
        assert_eq!(title, Some("Main Title".to_string()));

        let text2 = "INTRODUCTION\n\nThis is the intro.";
        let title2 = TextAnalyzer::extract_title(text2);
        assert_eq!(title2, Some("INTRODUCTION".to_string()));
    }
}
