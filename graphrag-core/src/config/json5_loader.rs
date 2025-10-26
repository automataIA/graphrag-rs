//! JSON5 configuration loader
//!
//! This module provides JSON5 parsing support for GraphRAG configurations.
//! JSON5 extends JSON with:
//! - Comments (// and /* */)
//! - Trailing commas
//! - Unquoted object keys
//! - Single quotes for strings
//! - More JavaScript-like syntax

use crate::core::error::{GraphRAGError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Load and parse a JSON5 configuration file
///
/// # Arguments
///
/// * `path` - Path to the JSON5 file
///
/// # Returns
///
/// Deserialized configuration of type `T`
///
/// # Example
///
/// ```ignore
/// use graphrag_core::config::json5_loader::load_json5_config;
///
/// let config: MyConfig = load_json5_config("config.json5")?;
/// ```
#[cfg(feature = "json5-support")]
pub fn load_json5_config<T, P>(path: P) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
    P: AsRef<Path>,
{
    let path = path.as_ref();

    // Read file contents
    let contents = std::fs::read_to_string(path).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to read JSON5 file {:?}: {}", path, e),
    })?;

    // Parse JSON5
    parse_json5_str(&contents)
}

/// Parse a JSON5 string into a typed configuration
///
/// # Arguments
///
/// * `contents` - JSON5 string
///
/// # Returns
///
/// Deserialized configuration of type `T`
#[cfg(feature = "json5-support")]
pub fn parse_json5_str<T>(contents: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    json5::from_str(contents).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to parse JSON5: {}", e),
    })
}

/// Save a configuration to JSON5 format
///
/// # Arguments
///
/// * `config` - Configuration to serialize
/// * `path` - Destination path
///
/// # Example
///
/// ```ignore
/// use graphrag_core::config::json5_loader::save_json5_config;
///
/// save_json5_config(&my_config, "config.json5")?;
/// ```
#[cfg(feature = "json5-support")]
pub fn save_json5_config<T, P>(config: &T, path: P) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let path = path.as_ref();

    // Serialize to JSON5 (note: json5 crate doesn't have a pretty printer,
    // so we use serde_json for now with JSON format)
    let json_str = serde_json::to_string_pretty(config).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to serialize config: {}", e),
    })?;

    std::fs::write(path, json_str).map_err(|e| GraphRAGError::Config {
        message: format!("Failed to write JSON5 file {:?}: {}", path, e),
    })?;

    Ok(())
}

/// Detect configuration file format based on extension
pub fn detect_config_format(path: &Path) -> Option<ConfigFormat> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| match ext.to_lowercase().as_str() {
            "json5" => Some(ConfigFormat::Json5),
            "json" => Some(ConfigFormat::Json),
            "toml" => Some(ConfigFormat::Toml),
            "yaml" | "yml" => Some(ConfigFormat::Yaml),
            _ => None,
        })
}

/// Configuration file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// JSON5 (JSON with comments)
    Json5,
    /// Standard JSON
    Json,
    /// TOML format (legacy)
    Toml,
    /// YAML format
    Yaml,
}

impl ConfigFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ConfigFormat::Json5 => "json5",
            ConfigFormat::Json => "json",
            ConfigFormat::Toml => "toml",
            ConfigFormat::Yaml => "yaml",
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ConfigFormat::Json5 | ConfigFormat::Json => "application/json",
            ConfigFormat::Toml => "application/toml",
            ConfigFormat::Yaml => "application/x-yaml",
        }
    }
}

#[cfg(all(test, feature = "json5-support"))]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestConfig {
        name: String,
        value: i32,
        enabled: bool,
    }

    #[test]
    fn test_parse_json5_with_comments() {
        let json5_str = r#"{
            // This is a comment
            name: "test",  // Unquoted key, trailing comma
            value: 42,
            enabled: true,
        }"#;

        let config: TestConfig = parse_json5_str(json5_str).unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.value, 42);
        assert_eq!(config.enabled, true);
    }

    #[test]
    fn test_parse_json5_with_single_quotes() {
        let json5_str = r#"{
            name: 'test',
            value: 42,
            enabled: true
        }"#;

        let config: TestConfig = parse_json5_str(json5_str).unwrap();
        assert_eq!(config.name, "test");
    }

    #[test]
    fn test_detect_format() {
        assert_eq!(
            detect_config_format(Path::new("config.json5")),
            Some(ConfigFormat::Json5)
        );
        assert_eq!(
            detect_config_format(Path::new("config.json")),
            Some(ConfigFormat::Json)
        );
        assert_eq!(
            detect_config_format(Path::new("config.toml")),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            detect_config_format(Path::new("config.yaml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(detect_config_format(Path::new("config.txt")), None);
    }
}
