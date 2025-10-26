//! JSON Schema validation for configuration files
//!
//! This module provides JSON Schema validation to ensure configurations
//! are correct before they're used. This catches errors early with clear
//! messages about what's wrong and where.

use crate::core::error::{GraphRAGError, Result};
use serde_json::Value;
use std::path::Path;

/// Validate a configuration against a JSON Schema
///
/// # Arguments
///
/// * `config_value` - Configuration as serde_json::Value
/// * `schema_value` - JSON Schema as serde_json::Value
///
/// # Returns
///
/// Ok(()) if valid, Err with detailed validation errors if invalid
///
/// # Example
///
/// ```ignore
/// use graphrag_core::config::schema_validator::validate_config;
/// use serde_json::json;
///
/// let config = json!({"mode": {"approach": "semantic"}});
/// let schema = json!(/* schema here */);
///
/// validate_config(&config, &schema)?;
/// ```
#[cfg(feature = "json5-support")]
pub fn validate_config(config_value: &Value, schema_value: &Value) -> Result<()> {
    use jsonschema::JSONSchema;

    // Compile schema
    let schema = JSONSchema::compile(schema_value).map_err(|e| GraphRAGError::Config {
        message: format!("Invalid JSON Schema: {}", e),
    })?;

    // Validate
    if let Err(errors) = schema.validate(config_value) {
        let error_messages: Vec<String> = errors
            .map(|error| {
                format!(
                    "Validation error at '{}': {}",
                    error.instance_path, error
                )
            })
            .collect();

        return Err(GraphRAGError::Config {
            message: format!(
                "Configuration validation failed:\n{}",
                error_messages.join("\n")
            ),
        });
    }

    Ok(())
}

/// Load schema from file
///
/// # Arguments
///
/// * `schema_path` - Path to JSON Schema file
///
/// # Returns
///
/// Parsed schema as serde_json::Value
#[cfg(feature = "json5-support")]
pub fn load_schema<P: AsRef<Path>>(schema_path: P) -> Result<Value> {
    let path = schema_path.as_ref();

    let schema_str = std::fs::read_to_string(path).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to read schema file {:?}: {}", path, e),
    })?;

    serde_json::from_str(&schema_str).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to parse schema JSON: {}", e),
    })
}

/// Validate configuration file against schema file
///
/// # Arguments
///
/// * `config_path` - Path to configuration file (JSON5/JSON)
/// * `schema_path` - Path to JSON Schema file
///
/// # Returns
///
/// Ok(()) if valid, Err with validation errors if invalid
#[cfg(feature = "json5-support")]
pub fn validate_config_file<P1, P2>(config_path: P1, schema_path: P2) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    // Load config
    let config_str =
        std::fs::read_to_string(config_path.as_ref()).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to read config file: {}", e),
        })?;

    // Parse as JSON5 first, fallback to JSON
    let config_value: Value = if config_path
        .as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "json5")
        .unwrap_or(false)
    {
        // Parse as JSON5
        json5::from_str(&config_str).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse JSON5 config: {}", e),
        })?
    } else {
        // Parse as JSON
        serde_json::from_str(&config_str).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to parse JSON config: {}", e),
        })?
    };

    // Load schema
    let schema_value = load_schema(schema_path)?;

    // Validate
    validate_config(&config_value, &schema_value)
}

/// Get user-friendly error messages from validation errors
///
/// This function formats validation errors in a way that's easy to understand
/// and actionable for users.
#[cfg(feature = "json5-support")]
pub fn format_validation_error(error: &GraphRAGError) -> String {
    match error {
        GraphRAGError::Config { message } => {
            if message.contains("Validation error at") {
                // Parse and format validation errors nicely
                let lines: Vec<&str> = message.lines().collect();
                if lines.len() > 1 {
                    let mut formatted = String::from("❌ Configuration validation failed:\n\n");

                    for (i, line) in lines.iter().skip(1).enumerate() {
                        formatted.push_str(&format!("  {}. {}\n", i + 1, line));
                    }

                    formatted.push_str("\n💡 Tips:\n");
                    formatted.push_str("  - Check your config file for typos\n");
                    formatted.push_str("  - Verify required fields are present\n");
                    formatted.push_str("  - Ensure values are within valid ranges\n");
                    formatted.push_str("  - See examples in config/templates/\n");

                    return formatted;
                }
            }
            format!("❌ Configuration error: {}", message)
        }
        _ => format!("❌ Error: {:?}", error),
    }
}

/// Validation result with detailed error information
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation error messages (if any)
    pub errors: Vec<ValidationError>,
}

/// Detailed validation error information
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// JSON path where error occurred (e.g., "/mode/approach")
    pub path: String,
    /// Error message
    pub message: String,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
        }
    }

    /// Create a failed validation result
    pub fn failure(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get formatted error messages
    pub fn format_errors(&self) -> String {
        if self.valid {
            return String::from("✅ Configuration is valid");
        }

        let mut formatted = String::from("❌ Configuration validation failed:\n\n");

        for (i, error) in self.errors.iter().enumerate() {
            formatted.push_str(&format!("  {}. At '{}': {}\n", i + 1, error.path, error.message));

            if let Some(suggestion) = &error.suggestion {
                formatted.push_str(&format!("     💡 Suggestion: {}\n", suggestion));
            }
        }

        formatted
    }
}

#[cfg(all(test, feature = "json5-support"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_simple_config() {
        let schema = json!({
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0
                }
            }
        });

        // Valid config
        let valid_config = json!({
            "name": "Test",
            "age": 25
        });
        assert!(validate_config(&valid_config, &schema).is_ok());

        // Invalid: missing required field
        let invalid_config = json!({
            "age": 25
        });
        assert!(validate_config(&invalid_config, &schema).is_err());

        // Invalid: wrong type
        let invalid_config = json!({
            "name": "Test",
            "age": "not a number"
        });
        assert!(validate_config(&invalid_config, &schema).is_err());

        // Invalid: out of range
        let invalid_config = json!({
            "name": "Test",
            "age": -1
        });
        assert!(validate_config(&invalid_config, &schema).is_err());
    }

    #[test]
    fn test_validate_with_enum() {
        let schema = json!({
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "algorithmic", "hybrid"]
                }
            }
        });

        // Valid
        let valid = json!({"mode": "semantic"});
        assert!(validate_config(&valid, &schema).is_ok());

        // Invalid
        let invalid = json!({"mode": "invalid"});
        assert!(validate_config(&invalid, &schema).is_err());
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult::success();
        assert!(result.is_valid());
        assert!(result.format_errors().contains("✅"));

        let errors = vec![ValidationError {
            path: "/mode/approach".to_string(),
            message: "Invalid value".to_string(),
            suggestion: Some("Use 'semantic', 'algorithmic', or 'hybrid'".to_string()),
        }];

        let result = ValidationResult::failure(errors);
        assert!(!result.is_valid());
        assert!(result.format_errors().contains("❌"));
        assert!(result.format_errors().contains("💡"));
    }
}
