//! Configuration loading and management

use crate::handlers::FileOperations;
use color_eyre::eyre::{eyre, Result};
use graphrag_core::config::json5_loader::detect_config_format;
use graphrag_core::Config as GraphRAGConfig;
use std::path::Path;

/// Load GraphRAG configuration from file (supports JSON5, JSON, TOML)
pub async fn load_config(path: &Path) -> Result<GraphRAGConfig> {
    // Validate file
    FileOperations::validate_file(path).await?;

    // Keep extension errors at the CLI boundary, then delegate all parsing,
    // schema selection, legacy adaptation, and validation to the core loader.
    let format = detect_config_format(path)
        .ok_or_else(|| eyre!("Unsupported config file format: {:?}", path.extension()))?;
    let config = graphrag_core::config::loader::load_config(
        path.to_str()
            .ok_or_else(|| eyre!("Config path is not valid UTF-8"))?,
    )
    .map_err(|e| eyre!(e.to_string()))?;

    // Log configuration details for debugging
    tracing::info!("Loaded {:?} configuration from: {}", format, path.display());
    tracing::info!("Mode approach: {}", config.approach);
    tracing::info!(
        "Entity extraction use_gleaning: {}",
        config.entities.use_gleaning
    );
    tracing::info!(
        "Entity extraction max_gleaning_rounds: {}",
        config.entities.max_gleaning_rounds
    );
    tracing::info!("LLM enabled: {}", config.llm.enabled);
    tracing::info!("LLM chat_model: {}", config.llm.chat_model);

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
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tokio;

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
        assert!(
            result.is_ok(),
            "Failed to load JSON5 config: {:?}",
            result.err()
        );
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
