//! Layout parser trait and factory for document structure detection

use crate::text::document_structure::DocumentStructure;
use crate::core::Document;

/// Trait for document layout parsers
pub trait LayoutParser: Send + Sync {
    /// Parse document structure from content
    fn parse(&self, content: &str) -> DocumentStructure;

    /// Check if this parser supports a given format
    fn supports_format(&self, format: &str) -> bool;

    /// Get parser name
    fn name(&self) -> &'static str;
}

/// Factory for creating layout parsers based on document type
pub struct LayoutParserFactory;

impl LayoutParserFactory {
    /// Create appropriate parser for a document
    pub fn create_for_document(document: &Document) -> Box<dyn LayoutParser> {
        // Detect format from title/extension
        if document.title.ends_with(".md") || document.title.ends_with(".markdown") {
            return Box::new(crate::text::parsers::MarkdownLayoutParser::new());
        }

        if document.title.ends_with(".html") || document.title.ends_with(".htm") {
            return Box::new(crate::text::parsers::HtmlLayoutParser::new());
        }

        // Detect from content
        if document.content.contains("<h1") || document.content.contains("<h2")
            || document.content.contains("<html") || document.content.contains("<!DOCTYPE") {
            return Box::new(crate::text::parsers::HtmlLayoutParser::new());
        }

        // Check for markdown headings
        if document.content.lines().any(|line| line.trim_start().starts_with('#')) {
            return Box::new(crate::text::parsers::MarkdownLayoutParser::new());
        }

        // Default to plain text parser
        Box::new(crate::text::parsers::PlainTextLayoutParser::new())
    }

    /// Create parser for specific format
    pub fn create_for_format(format: &str) -> Box<dyn LayoutParser> {
        match format.to_lowercase().as_str() {
            "markdown" | "md" => Box::new(crate::text::parsers::MarkdownLayoutParser::new()),
            "html" | "htm" => Box::new(crate::text::parsers::HtmlLayoutParser::new()),
            "text" | "txt" | "plain" => Box::new(crate::text::parsers::PlainTextLayoutParser::new()),
            _ => Box::new(crate::text::parsers::PlainTextLayoutParser::new()),
        }
    }
}
