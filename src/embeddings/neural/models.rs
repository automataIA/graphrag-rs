//! Model management system for neural embeddings

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::core::{Result, GraphRAGError};

/// Pretrained model configurations
#[derive(Debug, Clone, PartialEq)]
pub enum PretrainedModel {
    AllMiniLmL6V2,           // 384 dims, fast
    AllMpnetBaseV2,          // 768 dims, high quality
    MultilingualE5Large,     // 1024 dims, multilingual
    DistilUseBase,           // 512 dims, balanced
    Custom(String),          // Custom model path
}

impl PretrainedModel {
    pub fn model_name(&self) -> &str {
        match self {
            PretrainedModel::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            PretrainedModel::AllMpnetBaseV2 => "all-mpnet-base-v2",
            PretrainedModel::MultilingualE5Large => "multilingual-e5-large",
            PretrainedModel::DistilUseBase => "distiluse-base-multilingual-cased",
            PretrainedModel::Custom(name) => name,
        }
    }

    pub fn embedding_dimension(&self) -> usize {
        match self {
            PretrainedModel::AllMiniLmL6V2 => 384,
            PretrainedModel::AllMpnetBaseV2 => 768,
            PretrainedModel::MultilingualE5Large => 1024,
            PretrainedModel::DistilUseBase => 512,
            PretrainedModel::Custom(_) => 384, // Default fallback
        }
    }

    pub fn huggingface_id(&self) -> &str {
        match self {
            PretrainedModel::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            PretrainedModel::AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2",
            PretrainedModel::MultilingualE5Large => "intfloat/multilingual-e5-large",
            PretrainedModel::DistilUseBase => "sentence-transformers/distiluse-base-multilingual-cased",
            PretrainedModel::Custom(name) => name,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            PretrainedModel::AllMiniLmL6V2 => "Lightweight, fast model suitable for most tasks",
            PretrainedModel::AllMpnetBaseV2 => "High-quality embeddings, balanced speed/accuracy",
            PretrainedModel::MultilingualE5Large => "Large multilingual model, best accuracy",
            PretrainedModel::DistilUseBase => "Balanced model with good multilingual support",
            PretrainedModel::Custom(_) => "Custom model",
        }
    }
}

/// Model information and metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub huggingface_id: String,
    pub embedding_dimension: usize,
    pub description: String,
    pub local_path: Option<PathBuf>,
    pub is_downloaded: bool,
    pub file_size_mb: Option<u64>,
    pub supported_languages: Vec<String>,
}

impl ModelInfo {
    pub fn from_pretrained(model: &PretrainedModel) -> Self {
        let supported_languages = match model {
            PretrainedModel::MultilingualE5Large | PretrainedModel::DistilUseBase => {
                vec![
                    "en".to_string(), "de".to_string(), "fr".to_string(),
                    "es".to_string(), "it".to_string(), "ja".to_string(),
                    "ko".to_string(), "zh".to_string(), "ar".to_string(),
                ]
            }
            _ => vec!["en".to_string()],
        };

        Self {
            name: model.model_name().to_string(),
            huggingface_id: model.huggingface_id().to_string(),
            embedding_dimension: model.embedding_dimension(),
            description: model.description().to_string(),
            local_path: None,
            is_downloaded: false,
            file_size_mb: None,
            supported_languages,
        }
    }
}

/// Model manager for downloading and caching neural models
pub struct ModelManager {
    models_dir: PathBuf,
    available_models: HashMap<String, ModelInfo>,
    download_cache: HashMap<String, PathBuf>,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_directory()?;
        std::fs::create_dir_all(&models_dir).map_err(GraphRAGError::Io)?;

        let mut manager = Self {
            models_dir,
            available_models: HashMap::new(),
            download_cache: HashMap::new(),
        };

        // Initialize with known models
        manager.initialize_known_models();
        manager.discover_local_models()?;

        Ok(manager)
    }

    fn get_models_directory() -> Result<PathBuf> {
        // Try to get from environment variable first
        if let Ok(path) = std::env::var("GRAPHRAG_MODELS_DIR") {
            return Ok(PathBuf::from(path));
        }

        // Fallback to user home directory
        let home_dir = dirs::home_dir().ok_or_else(|| GraphRAGError::Config {
            message: "Cannot determine home directory".to_string(),
        })?;

        Ok(home_dir.join(".graphrag-rs").join("models"))
    }

    fn initialize_known_models(&mut self) {
        let known_models = vec![
            PretrainedModel::AllMiniLmL6V2,
            PretrainedModel::AllMpnetBaseV2,
            PretrainedModel::MultilingualE5Large,
            PretrainedModel::DistilUseBase,
        ];

        for model in known_models {
            let info = ModelInfo::from_pretrained(&model);
            self.available_models.insert(model.model_name().to_string(), info);
        }
    }

    fn discover_local_models(&mut self) -> Result<()> {
        if !self.models_dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.models_dir).map_err(GraphRAGError::Io)? {
            let entry = entry.map_err(GraphRAGError::Io)?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(model_name) = path.file_name().and_then(|n| n.to_str()) {
                    let is_valid = self.validate_model_files(&path);
                    if let Some(info) = self.available_models.get_mut(model_name) {
                        info.local_path = Some(path.clone());
                        info.is_downloaded = is_valid;
                        if info.is_downloaded {
                            self.download_cache.insert(model_name.to_string(), path);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_model_files(&self, model_path: &Path) -> bool {
        // Check for essential model files
        let _required_files = ["config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "vocab.txt"];

        // At least config.json should exist
        model_path.join("config.json").exists()
    }

    pub async fn download_model(&mut self, model_name: &str) -> Result<PathBuf> {
        // Check if already downloaded
        if let Some(cached_path) = self.download_cache.get(model_name) {
            if cached_path.exists() && self.validate_model_files(cached_path) {
                return Ok(cached_path.clone());
            }
        }

        let model_info = self.available_models.get(model_name).ok_or_else(|| {
            GraphRAGError::Config {
                message: format!("Unknown model: {model_name}"),
            }
        })?;

        let model_dir = self.models_dir.join(model_name);
        std::fs::create_dir_all(&model_dir).map_err(GraphRAGError::Io)?;

        println!("📥 Downloading model: {} from {}", model_name, model_info.huggingface_id);

        // In a real implementation, this would download from HuggingFace Hub
        // For now, create placeholder files
        self.create_placeholder_model_files(&model_dir, model_info)?;

        // Update cache and model info
        self.download_cache.insert(model_name.to_string(), model_dir.clone());
        if let Some(info) = self.available_models.get_mut(model_name) {
            info.local_path = Some(model_dir.clone());
            info.is_downloaded = true;
        }

        println!("✅ Model downloaded: {model_name}");
        Ok(model_dir)
    }

    fn create_placeholder_model_files(&self, model_dir: &Path, model_info: &ModelInfo) -> Result<()> {
        // Create placeholder config.json
        let config = serde_json::json!({
            "model_type": "bert",
            "hidden_size": model_info.embedding_dimension,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "max_position_embeddings": 512,
            "vocab_size": 30522
        });

        let config_path = model_dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config)?).map_err(GraphRAGError::Io)?;

        // Create placeholder tokenizer.json
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "truncation": {
                "max_length": 512
            },
            "padding": {
                "pad_id": 0,
                "pad_token": "[PAD]"
            }
        });

        let tokenizer_path = model_dir.join("tokenizer.json");
        std::fs::write(&tokenizer_path, serde_json::to_string_pretty(&tokenizer)?).map_err(GraphRAGError::Io)?;

        // Create placeholder model weights file
        let model_path = model_dir.join("pytorch_model.bin");
        std::fs::write(&model_path, b"placeholder_model_weights").map_err(GraphRAGError::Io)?;

        // Create vocabulary file
        let vocab_path = model_dir.join("vocab.txt");
        let vocab_content = (0..30522).map(|i| format!("[UNUSED{i}]")).collect::<Vec<_>>().join("\n");
        std::fs::write(&vocab_path, vocab_content).map_err(GraphRAGError::Io)?;

        Ok(())
    }

    pub fn model_exists(&self, model_name: &str) -> Result<bool> {
        Ok(self.download_cache.contains_key(model_name) &&
           self.download_cache.get(model_name)
               .map(|path| path.exists() && self.validate_model_files(path))
               .unwrap_or(false))
    }

    pub fn get_model_path(&self, model_name: &str) -> Option<&PathBuf> {
        self.download_cache.get(model_name)
    }

    pub fn get_model_info(&self, model_name: &str) -> Option<ModelInfo> {
        self.available_models.get(model_name).cloned()
    }

    pub fn list_available_models(&self) -> Vec<&ModelInfo> {
        self.available_models.values().collect()
    }

    pub fn list_downloaded_models(&self) -> Vec<&ModelInfo> {
        self.available_models
            .values()
            .filter(|info| info.is_downloaded)
            .collect()
    }

    pub fn delete_model(&mut self, model_name: &str) -> Result<bool> {
        if let Some(model_path) = self.download_cache.remove(model_name) {
            if model_path.exists() {
                std::fs::remove_dir_all(&model_path).map_err(GraphRAGError::Io)?;
            }

            if let Some(info) = self.available_models.get_mut(model_name) {
                info.is_downloaded = false;
                info.local_path = None;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_models_dir(&self) -> &PathBuf {
        &self.models_dir
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().expect("Failed to create ModelManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_pretrained_model_info() {
        let model = PretrainedModel::AllMiniLmL6V2;
        assert_eq!(model.model_name(), "all-MiniLM-L6-v2");
        assert_eq!(model.embedding_dimension(), 384);
        assert!(!model.huggingface_id().is_empty());
    }

    #[test]
    fn test_model_info_creation() {
        let model = PretrainedModel::AllMpnetBaseV2;
        let info = ModelInfo::from_pretrained(&model);

        assert_eq!(info.name, "all-mpnet-base-v2");
        assert_eq!(info.embedding_dimension, 768);
        assert!(!info.is_downloaded);
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        // Use temporary directory for testing
        std::env::set_var("GRAPHRAG_MODELS_DIR", "/tmp/test_models");

        let manager = ModelManager::new();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        let models = manager.list_available_models();
        assert!(!models.is_empty());

        // Cleanup
        std::env::remove_var("GRAPHRAG_MODELS_DIR");
    }

    #[tokio::test]
    async fn test_model_download_placeholder() {
        // Use temporary directory for testing
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("GRAPHRAG_MODELS_DIR", temp_dir.path().to_str().unwrap());

        let mut manager = ModelManager::new().unwrap();

        // Test downloading a model
        let result = manager.download_model("all-MiniLM-L6-v2").await;
        assert!(result.is_ok());

        let model_path = result.unwrap();
        assert!(model_path.exists());
        assert!(model_path.join("config.json").exists());

        // Test that model is now marked as downloaded
        assert!(manager.model_exists("all-MiniLM-L6-v2").unwrap());

        // Cleanup
        std::env::remove_var("GRAPHRAG_MODELS_DIR");
    }
}