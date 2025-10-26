//! Input validation middleware and utilities for GraphRAG Server
//!
//! Provides request validation, sanitization, and security checks.

/// Maximum request body size (10MB)
pub const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Maximum query length
pub const MAX_QUERY_LENGTH: usize = 10_000;

/// Maximum document title length
pub const MAX_TITLE_LENGTH: usize = 500;

/// Maximum document content length (5MB of text)
pub const MAX_CONTENT_LENGTH: usize = 5 * 1024 * 1024;

/// Maximum top_k value for queries
pub const MAX_TOP_K: usize = 100;

/// Validation error response
#[derive(serde::Serialize)]
pub struct ValidationError {
    pub error: String,
    pub field: Option<String>,
    pub max_length: Option<usize>,
}

/// Validate query string
pub fn validate_query(query: &str) -> Result<(), ValidationError> {
    if query.is_empty() {
        return Err(ValidationError {
            error: "Query cannot be empty".to_string(),
            field: Some("query".to_string()),
            max_length: None,
        });
    }

    if query.len() > MAX_QUERY_LENGTH {
        return Err(ValidationError {
            error: format!("Query exceeds maximum length of {} characters", MAX_QUERY_LENGTH),
            field: Some("query".to_string()),
            max_length: Some(MAX_QUERY_LENGTH),
        });
    }

    // Check for potentially malicious patterns
    if contains_sql_injection_patterns(query) {
        return Err(ValidationError {
            error: "Query contains potentially malicious patterns".to_string(),
            field: Some("query".to_string()),
            max_length: None,
        });
    }

    Ok(())
}

/// Validate document title
pub fn validate_title(title: &str) -> Result<(), ValidationError> {
    if title.is_empty() {
        return Err(ValidationError {
            error: "Title cannot be empty".to_string(),
            field: Some("title".to_string()),
            max_length: None,
        });
    }

    if title.len() > MAX_TITLE_LENGTH {
        return Err(ValidationError {
            error: format!("Title exceeds maximum length of {} characters", MAX_TITLE_LENGTH),
            field: Some("title".to_string()),
            max_length: Some(MAX_TITLE_LENGTH),
        });
    }

    Ok(())
}

/// Validate document content
pub fn validate_content(content: &str) -> Result<(), ValidationError> {
    if content.is_empty() {
        return Err(ValidationError {
            error: "Content cannot be empty".to_string(),
            field: Some("content".to_string()),
            max_length: None,
        });
    }

    if content.len() > MAX_CONTENT_LENGTH {
        return Err(ValidationError {
            error: format!("Content exceeds maximum length of {} characters", MAX_CONTENT_LENGTH),
            field: Some("content".to_string()),
            max_length: Some(MAX_CONTENT_LENGTH),
        });
    }

    Ok(())
}

/// Validate top_k parameter
pub fn validate_top_k(top_k: usize) -> Result<(), ValidationError> {
    if top_k == 0 {
        return Err(ValidationError {
            error: "top_k must be greater than 0".to_string(),
            field: Some("top_k".to_string()),
            max_length: None,
        });
    }

    if top_k > MAX_TOP_K {
        return Err(ValidationError {
            error: format!("top_k exceeds maximum value of {}", MAX_TOP_K),
            field: Some("top_k".to_string()),
            max_length: Some(MAX_TOP_K),
        });
    }

    Ok(())
}

/// Sanitize string by removing control characters
pub fn sanitize_string(input: &str) -> String {
    input
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .collect()
}

/// Check for common SQL injection patterns (basic check)
fn contains_sql_injection_patterns(input: &str) -> bool {
    let lower = input.to_lowercase();
    let dangerous_patterns = [
        "drop table",
        "drop database",
        "delete from",
        "insert into",
        "update set",
        "; --",
        "' or '1'='1",
        "\" or \"1\"=\"1",
        "union select",
        "exec(",
        "execute(",
    ];

    dangerous_patterns.iter().any(|pattern| lower.contains(pattern))
}

// Note: Request body size limits are now configured in main.rs using
// PayloadConfig and JsonConfig with MAX_BODY_SIZE constant

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_query() {
        // Valid queries
        assert!(validate_query("What is GraphRAG?").is_ok());
        assert!(validate_query("A".repeat(1000).as_str()).is_ok());

        // Invalid queries
        assert!(validate_query("").is_err());
        assert!(validate_query(&"A".repeat(MAX_QUERY_LENGTH + 1)).is_err());
        assert!(validate_query("'; DROP TABLE users; --").is_err());
    }

    #[test]
    fn test_validate_title() {
        assert!(validate_title("My Document").is_ok());
        assert!(validate_title("").is_err());
        assert!(validate_title(&"A".repeat(MAX_TITLE_LENGTH + 1)).is_err());
    }

    #[test]
    fn test_validate_content() {
        assert!(validate_content("Some content").is_ok());
        assert!(validate_content("").is_err());
        assert!(validate_content(&"A".repeat(MAX_CONTENT_LENGTH + 1)).is_err());
    }

    #[test]
    fn test_validate_top_k() {
        assert!(validate_top_k(5).is_ok());
        assert!(validate_top_k(100).is_ok());
        assert!(validate_top_k(0).is_err());
        assert!(validate_top_k(101).is_err());
    }

    #[test]
    fn test_sanitize_string() {
        assert_eq!(sanitize_string("Hello\x00World"), "HelloWorld");
        assert_eq!(sanitize_string("Hello\nWorld"), "Hello\nWorld");
        assert_eq!(sanitize_string("Normal text"), "Normal text");
    }

    #[test]
    fn test_sql_injection_detection() {
        assert!(contains_sql_injection_patterns("'; DROP TABLE users; --"));
        assert!(contains_sql_injection_patterns("' OR '1'='1"));
        assert!(contains_sql_injection_patterns("UNION SELECT * FROM passwords"));
        assert!(!contains_sql_injection_patterns("What is the drop in temperature?"));
        assert!(!contains_sql_injection_patterns("Normal query about tables"));
    }
}
