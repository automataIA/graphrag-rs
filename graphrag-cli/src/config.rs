//! Configuration loading and management

use color_eyre::eyre::{eyre, Result};
use graphrag_core::Config as GraphRAGConfig;
use graphrag_core::config::setconfig::SetConfig;
use graphrag_core::config::json5_loader::{detect_config_format, ConfigFormat};
use std::path::Path;
use crate::handlers::FileOperations;

/// Load GraphRAG configuration from file (supports JSON5, JSON, TOML)
pub async fn load_config(path: &Path) -> Result<GraphRAGConfig> {
    // Validate file
    FileOperations::validate_file(path).await?;

    // Read file
    let content = FileOperations::read_to_string(path).await?;

    // Detect format from file extension
    let format = detect_config_format(path)
        .ok_or_else(|| eyre!("Unsupported config file format: {:?}", path.extension()))?;

    // Parse based on detected format - always parse as SetConfig first, then convert
    let set_config: SetConfig = match format {
        ConfigFormat::Json5 => {
            #[cfg(feature = "json5-support")]
            {
                json5::from_str(&content)
                    .map_err(|e| eyre!("Failed to parse JSON5 config: {}", e))?
            }
            #[cfg(not(feature = "json5-support"))]
            {
                return Err(eyre!("JSON5 support not enabled. Recompile with json5-support feature."));
            }
        }
        ConfigFormat::Json => {
            serde_json::from_str(&content)
                .map_err(|e| eyre!("Failed to parse JSON config: {}", e))?
        }
        ConfigFormat::Toml => {
            toml::from_str(&content)
                .map_err(|e| eyre!("Failed to parse TOML config: {}", e))?
        }
        ConfigFormat::Yaml => {
            #[cfg(feature = "yaml-support")]
            {
                serde_yaml::from_str(&content)
                    .map_err(|e| eyre!("Failed to parse YAML config: {}", e))?
            }
            #[cfg(not(feature = "yaml-support"))]
            {
                return Err(eyre!("YAML support not enabled. Recompile with yaml-support feature."));
            }
        }
    };

    // Convert SetConfig to Config
    let config = set_config.to_graphrag_config();

    // Log configuration details for debugging
    tracing::info!("Loaded {:?} configuration from: {}", format, path.display());
    tracing::info!("Mode approach: {}", set_config.mode.approach);
    tracing::info!("Entity extraction use_gleaning: {}", config.entities.use_gleaning);
    tracing::info!("Entity extraction max_gleaning_rounds: {}", config.entities.max_gleaning_rounds);
    tracing::info!("Ollama enabled: {}", config.ollama.enabled);
    tracing::info!("Ollama chat_model: {}", config.ollama.chat_model);

    Ok(config)
}

/// Get default configuration
#[allow(dead_code)]
pub fn default_config() -> GraphRAGConfig {
    GraphRAGConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio;
    use std::io::Write;

    #[tokio::test]
    async fn test_load_valid_toml_config() {
        let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
        writeln!(temp_file, "[general]").unwrap();
        writeln!(temp_file, "log_level = \"info\"").unwrap();

        let result = load_config(temp_file.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[cfg(feature = "json5-support")]
    async fn test_load_valid_json5_config() {
        let mut temp_file = NamedTempFile::with_suffix(".json5").unwrap();
        writeln!(temp_file, "{{").unwrap();
        writeln!(temp_file, "  // Comment in JSON5").unwrap();
        writeln!(temp_file, "  general: {{").unwrap();
        writeln!(temp_file, "    log_level: \"info\",").unwrap();
        writeln!(temp_file, "  }},").unwrap();
        writeln!(temp_file, "}}").unwrap();

        let result = load_config(temp_file.path()).await;
        assert!(result.is_ok(), "Failed to load JSON5 config: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_load_valid_json_config() {
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(temp_file, "{{").unwrap();
        writeln!(temp_file, "  \"general\": {{").unwrap();
        writeln!(temp_file, "    \"log_level\": \"info\"").unwrap();
        writeln!(temp_file, "  }}").unwrap();
        writeln!(temp_file, "}}").unwrap();

        let result = load_config(temp_file.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_load_invalid_toml() {
        let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
        writeln!(temp_file, "invalid toml content {{{{").unwrap();

        let result = load_config(temp_file.path()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_unsupported_format() {
        let result = load_config(Path::new("/nonexistent/file.txt")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_nonexistent_file() {
        let result = load_config(Path::new("/nonexistent/file.toml")).await;
        assert!(result.is_err());
    }
}
