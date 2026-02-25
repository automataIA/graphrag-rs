//! Document layout parsers

pub mod html;
pub mod markdown;
pub mod plaintext;

pub use html::HtmlLayoutParser;
pub use markdown::MarkdownLayoutParser;
pub use plaintext::PlainTextLayoutParser;
