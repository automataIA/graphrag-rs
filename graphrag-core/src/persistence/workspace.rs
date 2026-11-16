//! Workspace management for GraphRAG persistence
//!
//! Provides multi-workspace support with checkpointing and metadata tracking.

use crate::core::{GraphRAGError, KnowledgeGraph, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Workspace metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceMetadata {
    /// Workspace name
    pub name: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Number of entities
    pub entity_count: usize,
    /// Number of relationships
    pub relationship_count: usize,
    /// Number of documents
    pub document_count: usize,
    /// Number of chunks
    pub chunk_count: usize,
    /// Storage format version
    pub format_version: String,
    /// Description (optional)
    pub description: Option<String>,
    /// Id of the extraction ontology the graph was built with
    /// (`name@version#fingerprint`, see [`crate::ontology::Ontology::id`])
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology: Option<String>,
}

impl WorkspaceMetadata {
    /// Create new workspace metadata
    pub fn new(name: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            name,
            created_at: now,
            modified_at: now,
            entity_count: 0,
            relationship_count: 0,
            document_count: 0,
            chunk_count: 0,
            format_version: "1.0".to_string(),
            description: None,
            ontology: None,
        }
    }

    /// Update counts from knowledge graph
    pub fn update_from_graph(&mut self, graph: &KnowledgeGraph) {
        self.entity_count = graph.entity_count();
        self.relationship_count = graph.relationship_count();
        self.document_count = graph.document_count();
        self.chunk_count = graph.chunks().count();
        self.modified_at = chrono::Utc::now();
    }
}

/// Workspace information (lightweight version for listing)
#[derive(Debug, Clone)]
pub struct WorkspaceInfo {
    /// Workspace name
    pub name: String,
    /// Path to workspace directory
    pub path: PathBuf,
    /// Workspace metadata
    pub metadata: WorkspaceMetadata,
    /// Total size in bytes
    pub size_bytes: u64,
}

/// Workspace manager for multi-workspace support
#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    /// Base directory for all workspaces
    base_dir: PathBuf,
}

impl WorkspaceManager {
    /// Create a new workspace manager
    ///
    /// # Arguments
    /// * `base_dir` - Base directory path (e.g., "./workspace")
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::persistence::WorkspaceManager;
    ///
    /// let workspace = WorkspaceManager::new("./workspace").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir)?;
            #[cfg(feature = "tracing")]
            tracing::info!("Created workspace base directory: {:?}", base_dir);
        }

        Ok(Self { base_dir })
    }

    /// Get workspace directory path
    pub fn workspace_path(&self, workspace_name: &str) -> PathBuf {
        self.base_dir.join(workspace_name)
    }

    /// Check if workspace exists
    pub fn workspace_exists(&self, workspace_name: &str) -> bool {
        self.workspace_path(workspace_name).exists()
    }

    /// Create a new workspace
    pub fn create_workspace(&self, workspace_name: &str) -> Result<()> {
        let workspace_path = self.workspace_path(workspace_name);

        if workspace_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Workspace '{workspace_name}' already exists"),
            });
        }

        // Create workspace directory
        fs::create_dir_all(&workspace_path)?;

        // Create metadata
        let metadata = WorkspaceMetadata::new(workspace_name.to_string());
        self.save_metadata(&metadata, workspace_name)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Created workspace: {}", workspace_name);

        Ok(())
    }

    /// Delete a workspace
    pub fn delete_workspace(&self, workspace_name: &str) -> Result<()> {
        let workspace_path = self.workspace_path(workspace_name);

        if !workspace_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Workspace '{workspace_name}' does not exist"),
            });
        }

        fs::remove_dir_all(&workspace_path)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Deleted workspace: {}", workspace_name);

        Ok(())
    }

    /// List all workspaces
    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
        let mut workspaces = Vec::new();

        if !self.base_dir.exists() {
            return Ok(workspaces);
        }

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let workspace_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Load metadata
                let metadata = self
                    .load_metadata(&workspace_name)
                    .unwrap_or_else(|_| WorkspaceMetadata::new(workspace_name.clone()));

                // Calculate size
                let size_bytes = Self::calculate_dir_size(&path).unwrap_or(0);

                workspaces.push(WorkspaceInfo {
                    name: workspace_name,
                    path,
                    metadata,
                    size_bytes,
                });
            }
        }

        // Sort by modification time (newest first)
        workspaces.sort_by_key(|w| std::cmp::Reverse(w.metadata.modified_at));

        Ok(workspaces)
    }

    /// Save knowledge graph to workspace
    pub fn save_graph(&self, graph: &KnowledgeGraph, workspace_name: &str) -> Result<()> {
        // Create workspace if it doesn't exist
        if !self.workspace_exists(workspace_name) {
            self.create_workspace(workspace_name)?;
        }

        let workspace_path = self.workspace_path(workspace_name);

        // Save to JSON (always available as fallback)
        let json_path = workspace_path.join("graph.json");
        graph.save_to_json(json_path.to_str().expect("valid UTF-8 path"))?;

        // Save to Parquet (if feature enabled)
        #[cfg(feature = "persistent-storage")]
        {
            use super::parquet::ParquetPersistence;
            let parquet = ParquetPersistence::new(workspace_path.clone())?;
            parquet.save_graph(graph)?;
        }

        // Update metadata
        let mut metadata = self
            .load_metadata(workspace_name)
            .unwrap_or_else(|_| WorkspaceMetadata::new(workspace_name.to_string()));
        metadata.update_from_graph(graph);
        self.save_metadata(&metadata, workspace_name)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Saved graph to workspace: {}", workspace_name);

        Ok(())
    }

    /// Load knowledge graph from workspace
    pub fn load_graph(&self, workspace_name: &str) -> Result<KnowledgeGraph> {
        if !self.workspace_exists(workspace_name) {
            return Err(GraphRAGError::Config {
                message: format!("Workspace '{workspace_name}' does not exist"),
            });
        }

        let workspace_path = self.workspace_path(workspace_name);

        // Try loading from Parquet first (if feature enabled)
        #[cfg(feature = "persistent-storage")]
        {
            use super::parquet::ParquetPersistence;
            let parquet = ParquetPersistence::new(workspace_path.clone())?;
            if let Ok(graph) = parquet.load_graph() {
                #[cfg(feature = "tracing")]
                tracing::info!("Loaded graph from Parquet: {}", workspace_name);
                return Ok(graph);
            }
        }

        // Fallback to JSON
        let json_path = workspace_path.join("graph.json");
        if json_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::info!("Loading graph from JSON fallback: {}", workspace_name);
            return KnowledgeGraph::load_from_json(json_path.to_str().expect("valid UTF-8 path"));
        }

        Err(GraphRAGError::Config {
            message: format!("No graph data found in workspace '{workspace_name}'"),
        })
    }

    /// Ontology id recorded for a workspace (`None` if never recorded)
    pub fn workspace_ontology(&self, workspace_name: &str) -> Option<String> {
        self.load_metadata(workspace_name).ok()?.ontology
    }

    /// Record the ontology id the workspace graph was built with
    pub fn record_ontology(&self, workspace_name: &str, ontology_id: &str) -> Result<()> {
        let mut metadata = self
            .load_metadata(workspace_name)
            .unwrap_or_else(|_| WorkspaceMetadata::new(workspace_name.to_string()));
        metadata.ontology = Some(ontology_id.to_string());
        self.save_metadata(&metadata, workspace_name)
    }

    /// Save workspace metadata
    fn save_metadata(&self, metadata: &WorkspaceMetadata, workspace_name: &str) -> Result<()> {
        let workspace_path = self.workspace_path(workspace_name);
        let metadata_path = workspace_path.join("metadata.toml");

        let toml_string = toml::to_string_pretty(metadata).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to serialize metadata: {e}"),
        })?;

        fs::write(metadata_path, toml_string)?;

        Ok(())
    }

    /// Load workspace metadata
    fn load_metadata(&self, workspace_name: &str) -> Result<WorkspaceMetadata> {
        let workspace_path = self.workspace_path(workspace_name);
        let metadata_path = workspace_path.join("metadata.toml");

        if !metadata_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Metadata not found for workspace '{workspace_name}'"),
            });
        }

        let toml_string = fs::read_to_string(metadata_path)?;
        let metadata: WorkspaceMetadata =
            toml::from_str(&toml_string).map_err(|e| GraphRAGError::Config {
                message: format!("Failed to parse metadata: {e}"),
            })?;

        Ok(metadata)
    }

    /// Calculate directory size recursively
    fn calculate_dir_size(path: &Path) -> Result<u64> {
        let mut total_size = 0u64;

        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    total_size += Self::calculate_dir_size(&path)?;
                } else {
                    total_size += entry.metadata()?.len();
                }
            }
        } else {
            total_size = fs::metadata(path)?.len();
        }

        Ok(total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_workspace_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();
        assert!(workspace.base_dir.exists());
    }

    #[test]
    fn test_create_and_list_workspaces() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        workspace.create_workspace("test1").unwrap();
        workspace.create_workspace("test2").unwrap();

        let workspaces = workspace.list_workspaces().unwrap();
        assert_eq!(workspaces.len(), 2);
    }

    #[test]
    fn test_delete_workspace() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        workspace.create_workspace("test").unwrap();
        assert!(workspace.workspace_exists("test"));

        workspace.delete_workspace("test").unwrap();
        assert!(!workspace.workspace_exists("test"));
    }

    #[test]
    fn test_save_and_load_graph() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();

        let graph = KnowledgeGraph::new();
        workspace.save_graph(&graph, "test").unwrap();

        let loaded_graph = workspace.load_graph("test").unwrap();
        assert_eq!(loaded_graph.entity_count(), 0);
    }

    fn graph_with_described_relationship() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();
        for id in ["a", "b"] {
            graph
                .add_entity(crate::core::Entity::new(
                    crate::core::EntityId::new(id.to_string()),
                    id.to_uppercase(),
                    "PERSON".to_string(),
                    0.9,
                ))
                .unwrap();
        }
        let mut rel = crate::core::Relationship::new(
            crate::core::EntityId::new("a".to_string()),
            crate::core::EntityId::new("b".to_string()),
            "OTHER".to_string(),
            0.8,
        );
        rel.description = Some("questions: A questions B".to_string());
        graph.add_relationship(rel).unwrap();
        graph
    }

    #[test]
    fn relationship_description_and_ontology_survive_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let workspace = WorkspaceManager::new(temp_dir.path()).unwrap();
        workspace
            .save_graph(&graph_with_described_relationship(), "ws")
            .unwrap();
        assert_eq!(workspace.workspace_ontology("ws"), None);
        workspace
            .record_ontology("ws", "generic@1.0.0#abc")
            .unwrap();
        assert_eq!(
            workspace.workspace_ontology("ws").as_deref(),
            Some("generic@1.0.0#abc")
        );

        let loaded = workspace.load_graph("ws").unwrap();
        let rels: Vec<_> = loaded.relationships().collect();
        assert_eq!(rels.len(), 1);
        assert_eq!(
            rels[0].description.as_deref(),
            Some("questions: A questions B")
        );
    }
}
