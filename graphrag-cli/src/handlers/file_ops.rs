//! File operations utilities
//!
//! Provides helpers for loading and validating files.

use color_eyre::eyre::{eyre, Result};
use std::path::{Path, PathBuf};
use tokio::fs;

/// File operations utility
pub struct FileOperations;

impl FileOperations {
    /// Check if a file exists
    pub async fn exists(path: &Path) -> bool {
        fs::metadata(path).await.is_ok()
    }

    /// Validate that a file exists and is readable
    pub async fn validate_file(path: &Path) -> Result<()> {
        // Debug log the exact path being checked
        tracing::debug!("Validating file path: {:?}", path);
        tracing::debug!("Path as string: {}", path.display());
        tracing::debug!("Path extension: {:?}", path.extension());

        if !Self::exists(path).await {
            return Err(eyre!("File not found: {}", path.display()));
        }

        if !path.is_file() {
            return Err(eyre!("Path is not a file: {}", path.display()));
        }

        // Try to read metadata to check permissions
        fs::metadata(path)
            .await
            .map_err(|e| eyre!("Cannot read file: {}", e))?;

        Ok(())
    }

    /// Read a file as string
    pub async fn read_to_string(path: &Path) -> Result<String> {
        Self::validate_file(path).await?;

        fs::read_to_string(path)
            .await
            .map_err(|e| eyre!("Failed to read file {}: {}", path.display(), e))
    }

    /// Write string to file
    #[allow(dead_code)]
    pub async fn write_string(path: &Path, content: &str) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| eyre!("Failed to create directory {}: {}", parent.display(), e))?;
        }

        fs::write(path, content)
            .await
            .map_err(|e| eyre!("Failed to write file {}: {}", path.display(), e))
    }

    /// Expand tilde (~) in path
    pub fn expand_tilde(path: &Path) -> PathBuf {
        if path.starts_with("~") {
            if let Some(home) = dirs::home_dir() {
                return home.join(path.strip_prefix("~").unwrap());
            }
        }
        path.to_path_buf()
    }

    /// Resolve relative path to absolute
    #[allow(dead_code)]
    pub fn canonicalize(path: &Path) -> Result<PathBuf> {
        let expanded = Self::expand_tilde(path);

        if expanded.is_absolute() {
            Ok(expanded)
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(expanded))
                .map_err(|e| eyre!("Failed to get current directory: {}", e))
        }
    }

    /// Get file extension
    #[allow(dead_code)]
    pub fn get_extension(path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
    }

    /// Check if file is a supported document format
    #[allow(dead_code)]
    pub fn is_supported_document(path: &Path) -> bool {
        if let Some(ext) = Self::get_extension(path) {
            matches!(ext.as_str(), "txt" | "md" | "rst" | "log")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde() {
        let path = Path::new("~/test.txt");
        let expanded = FileOperations::expand_tilde(path);

        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, home.join("test.txt"));
        }
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(
            FileOperations::get_extension(Path::new("test.txt")),
            Some("txt".to_string())
        );
        assert_eq!(
            FileOperations::get_extension(Path::new("test.TXT")),
            Some("txt".to_string())
        );
        assert_eq!(FileOperations::get_extension(Path::new("test")), None);
    }

    #[test]
    fn test_is_supported_document() {
        assert!(FileOperations::is_supported_document(Path::new("test.txt")));
        assert!(FileOperations::is_supported_document(Path::new("test.md")));
        assert!(!FileOperations::is_supported_document(Path::new("test.pdf")));
    }
}
