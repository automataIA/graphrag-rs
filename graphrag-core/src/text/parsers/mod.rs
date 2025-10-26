//! Document layout parsers

pub mod markdown;
pub mod plaintext;
pub mod html;

pub use markdown::MarkdownLayoutParser;
pub use plaintext::PlainTextLayoutParser;
pub use html::HtmlLayoutParser;
