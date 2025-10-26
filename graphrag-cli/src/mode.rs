//! Input utilities for the TUI
//!
//! The input box automatically detects whether input is a query or slash command.
//! - Regular text: Natural language questions
//! - Text starting with '/': Slash commands

/// Check if input text is a slash command
pub fn is_slash_command(input: &str) -> bool {
    input.trim().starts_with('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slash_command_detection() {
        assert!(is_slash_command("/config file.toml"));
        assert!(is_slash_command("  /load doc.txt"));
        assert!(!is_slash_command("What is GraphRAG?"));
        assert!(!is_slash_command(""));
    }
}
