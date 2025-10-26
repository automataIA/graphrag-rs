//! Workspace management

use color_eyre::eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Workspace metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMetadata {
    /// Workspace ID
    pub id: String,
    /// Workspace name
    pub name: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last accessed timestamp
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Configuration file path (if any)
    pub config_path: Option<PathBuf>,
}

impl WorkspaceMetadata {
    /// Create a new workspace
    pub fn new(name: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            created_at: now,
            last_accessed: now,
            config_path: None,
        }
    }

    /// Update last accessed time
    pub fn touch(&mut self) {
        self.last_accessed = chrono::Utc::now();
    }
}

/// Workspace manager
pub struct WorkspaceManager {
    /// Base directory for workspaces
    base_dir: PathBuf,
}

impl WorkspaceManager {
    /// Create a new workspace manager
    pub fn new() -> Result<Self> {
        let base_dir = dirs::home_dir()
            .ok_or_else(|| eyre!("Could not determine home directory"))?
            .join(".graphrag")
            .join("workspaces");

        Ok(Self { base_dir })
    }

    /// Get workspace directory
    pub fn workspace_dir(&self, id: &str) -> PathBuf {
        self.base_dir.join(id)
    }

    /// Get workspace metadata file path
    pub fn metadata_path(&self, id: &str) -> PathBuf {
        self.workspace_dir(id).join("metadata.json")
    }

    /// Get query history file path
    pub fn query_history_path(&self, id: &str) -> PathBuf {
        self.workspace_dir(id).join("query_history.json")
    }

    /// Create a new workspace
    pub async fn create_workspace(&self, name: String) -> Result<WorkspaceMetadata> {
        let metadata = WorkspaceMetadata::new(name);

        // Create workspace directory
        let workspace_dir = self.workspace_dir(&metadata.id);
        tokio::fs::create_dir_all(&workspace_dir).await?;

        // Save metadata
        self.save_metadata(&metadata).await?;

        Ok(metadata)
    }

    /// Load workspace metadata
    pub async fn load_metadata(&self, id: &str) -> Result<WorkspaceMetadata> {
        let path = self.metadata_path(id);
        let content = tokio::fs::read_to_string(&path).await?;
        let mut metadata: WorkspaceMetadata = serde_json::from_str(&content)?;
        metadata.touch();
        self.save_metadata(&metadata).await?;
        Ok(metadata)
    }

    /// Save workspace metadata
    pub async fn save_metadata(&self, metadata: &WorkspaceMetadata) -> Result<()> {
        let path = self.metadata_path(&metadata.id);
        let json = serde_json::to_string_pretty(metadata)?;
        tokio::fs::write(&path, json).await?;
        Ok(())
    }

    /// List all workspaces
    pub async fn list_workspaces(&self) -> Result<Vec<WorkspaceMetadata>> {
        let mut workspaces = Vec::new();

        if !self.base_dir.exists() {
            return Ok(workspaces);
        }

        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                if let Some(id) = entry.file_name().to_str() {
                    if let Ok(metadata) = self.load_metadata(id).await {
                        workspaces.push(metadata);
                    }
                }
            }
        }

        // Sort by last accessed (most recent first)
        workspaces.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));

        Ok(workspaces)
    }

    /// Delete a workspace
    pub async fn delete_workspace(&self, id: &str) -> Result<()> {
        let workspace_dir = self.workspace_dir(id);
        if workspace_dir.exists() {
            tokio::fs::remove_dir_all(&workspace_dir).await?;
        }
        Ok(())
    }
}

impl Default for WorkspaceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create workspace manager")
    }
}
