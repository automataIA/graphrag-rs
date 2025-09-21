use crate::core::{GraphRAGError, KnowledgeGraph, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub checkpoints: Vec<Checkpoint>,
    pub active_checkpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub name: String,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub graph_snapshot: String, // Path to serialized graph
    pub metadata: CheckpointMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub document_count: usize,
    pub size_bytes: u64,
    pub hash: String, // Content hash for integrity
}

pub struct WorkspaceManager {
    workspace_dir: PathBuf,
    current_workspace: Option<Workspace>,
    auto_checkpoint_interval: Duration,
    last_checkpoint_time: Option<DateTime<Utc>>,
}

impl WorkspaceManager {
    pub fn new(workspace_dir: impl AsRef<Path>) -> Self {
        Self {
            workspace_dir: workspace_dir.as_ref().to_path_buf(),
            current_workspace: None,
            auto_checkpoint_interval: Duration::from_secs(3600), // 1 hour
            last_checkpoint_time: None,
        }
    }

    pub fn with_auto_checkpoint_interval(mut self, interval: Duration) -> Self {
        self.auto_checkpoint_interval = interval;
        self
    }

    /// Create a new workspace
    pub fn create_workspace(&mut self, name: &str) -> Result<Workspace> {
        let workspace_id = format!(
            "ws_{}_{}",
            Utc::now().timestamp(),
            Utc::now().timestamp_nanos_opt().unwrap_or(0) % 1000000
        );
        let workspace_path = self.workspace_dir.join(&workspace_id);

        // Create workspace directory
        fs::create_dir_all(&workspace_path)?;

        let workspace = Workspace {
            id: workspace_id.clone(),
            name: name.to_string(),
            created_at: Utc::now(),
            last_modified: Utc::now(),
            checkpoints: Vec::new(),
            active_checkpoint: None,
        };

        // Save workspace metadata
        self.save_workspace_metadata(&workspace)?;
        self.current_workspace = Some(workspace.clone());

        println!("✅ Created workspace '{name}' ({workspace_id})");
        Ok(workspace)
    }

    /// Load an existing workspace
    pub fn load_workspace(&mut self, workspace_id: &str) -> Result<Workspace> {
        let workspace_path = self.workspace_dir.join(workspace_id);
        let metadata_path = workspace_path.join("workspace.json");

        if !metadata_path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Workspace {workspace_id} not found"),
            });
        }

        let workspace_data = fs::read_to_string(&metadata_path)?;
        let workspace: Workspace = serde_json::from_str(&workspace_data)?;

        self.current_workspace = Some(workspace.clone());

        println!(
            "✅ Loaded workspace '{}' ({})",
            workspace.name, workspace_id
        );
        Ok(workspace)
    }

    /// List all available workspaces
    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
        if !self.workspace_dir.exists() {
            return Ok(Vec::new());
        }

        let mut workspaces = Vec::new();

        for entry in fs::read_dir(&self.workspace_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let workspace_path = entry.path();
                let metadata_path = workspace_path.join("workspace.json");

                if metadata_path.exists() {
                    if let Ok(workspace_data) = fs::read_to_string(&metadata_path) {
                        if let Ok(workspace) = serde_json::from_str::<Workspace>(&workspace_data) {
                            workspaces.push(WorkspaceInfo {
                                id: workspace.id,
                                name: workspace.name,
                                created_at: workspace.created_at,
                                last_modified: workspace.last_modified,
                                checkpoint_count: workspace.checkpoints.len(),
                            });
                        }
                    }
                }
            }
        }

        Ok(workspaces)
    }

    /// Create a checkpoint of the current graph state
    pub fn create_checkpoint(
        &mut self,
        name: &str,
        description: &str,
        graph: &KnowledgeGraph,
    ) -> Result<Checkpoint> {
        // Get workspace ID first
        let workspace_id = self
            .current_workspace
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "No active workspace".to_string(),
            })?
            .id
            .clone();

        let checkpoint_id = format!("cp_{}", Utc::now().timestamp());
        let checkpoint_path = self.get_workspace_path(&workspace_id)?.join("checkpoints");
        fs::create_dir_all(&checkpoint_path)?;

        let snapshot_file = checkpoint_path.join(format!("{checkpoint_id}.json"));

        // Serialize the graph
        let graph_data = self.serialize_graph(graph)?;
        fs::write(&snapshot_file, &graph_data)?;

        // Calculate metadata
        let metadata = CheckpointMetadata {
            entity_count: graph.entity_count(),
            relationship_count: graph.relationship_count(),
            document_count: graph.document_count(),
            size_bytes: graph_data.len() as u64,
            hash: self.calculate_hash(graph_data.as_bytes()),
        };

        let checkpoint = Checkpoint {
            id: checkpoint_id.clone(),
            name: name.to_string(),
            timestamp: Utc::now(),
            description: description.to_string(),
            graph_snapshot: snapshot_file.to_string_lossy().into_owned(),
            metadata,
        };

        // Add checkpoint to workspace
        if let Some(workspace) = self.current_workspace.as_mut() {
            workspace.checkpoints.push(checkpoint.clone());
            workspace.last_modified = Utc::now();
            workspace.active_checkpoint = Some(checkpoint_id.clone());
        }

        // Save workspace metadata in a separate scope
        if let Some(workspace) = self.current_workspace.as_ref() {
            self.save_workspace_metadata(workspace)?;
        }

        self.last_checkpoint_time = Some(Utc::now());

        println!("✅ Created checkpoint '{name}' ({checkpoint_id})");
        Ok(checkpoint)
    }

    /// Load a graph from a specific checkpoint
    pub fn load_checkpoint(&self, checkpoint_id: &str) -> Result<KnowledgeGraph> {
        let workspace = self
            .current_workspace
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "No active workspace".to_string(),
            })?;

        let checkpoint = workspace
            .checkpoints
            .iter()
            .find(|cp| cp.id == checkpoint_id)
            .ok_or_else(|| GraphRAGError::Config {
                message: format!("Checkpoint {checkpoint_id} not found"),
            })?;

        let graph_data = fs::read_to_string(&checkpoint.graph_snapshot)?;

        // Verify integrity
        let hash = self.calculate_hash(graph_data.as_bytes());
        if hash != checkpoint.metadata.hash {
            return Err(GraphRAGError::Config {
                message: "Checkpoint integrity check failed".to_string(),
            });
        }

        let graph = self.deserialize_graph(&graph_data)?;

        println!(
            "✅ Loaded checkpoint '{}' ({})",
            checkpoint.name, checkpoint_id
        );
        Ok(graph)
    }

    /// Check if auto checkpoint is needed and create one if so
    pub fn check_auto_checkpoint(&mut self, graph: &KnowledgeGraph) -> Result<Option<Checkpoint>> {
        if let Some(last_time) = self.last_checkpoint_time {
            let now = Utc::now();
            let elapsed = now.signed_duration_since(last_time);

            if elapsed.to_std().unwrap_or(Duration::ZERO) >= self.auto_checkpoint_interval {
                let checkpoint = self.create_checkpoint(
                    &format!("Auto checkpoint {}", now.format("%Y-%m-%d %H:%M:%S")),
                    "Automatic checkpoint",
                    graph,
                )?;
                return Ok(Some(checkpoint));
            }
        } else if self.current_workspace.is_some() {
            // First checkpoint
            let checkpoint =
                self.create_checkpoint("Initial checkpoint", "Initial graph state", graph)?;
            return Ok(Some(checkpoint));
        }

        Ok(None)
    }

    /// Delete a checkpoint
    pub fn delete_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        let checkpoint_to_delete = if let Some(workspace) = self.current_workspace.as_mut() {
            if let Some(pos) = workspace
                .checkpoints
                .iter()
                .position(|cp| cp.id == checkpoint_id)
            {
                let checkpoint = workspace.checkpoints.remove(pos);

                // Update active checkpoint if it was deleted
                if workspace.active_checkpoint.as_deref() == Some(checkpoint_id) {
                    workspace.active_checkpoint =
                        workspace.checkpoints.last().map(|cp| cp.id.clone());
                }

                workspace.last_modified = Utc::now();
                Some(checkpoint)
            } else {
                None
            }
        } else {
            return Err(GraphRAGError::Config {
                message: "No active workspace".to_string(),
            });
        };

        if let Some(checkpoint) = checkpoint_to_delete {
            // Remove snapshot file
            if fs::remove_file(&checkpoint.graph_snapshot).is_ok() {
                println!("🗑️ Deleted checkpoint file: {}", checkpoint.graph_snapshot);
            }

            // Save workspace metadata in a separate scope
            if let Some(workspace) = self.current_workspace.as_ref() {
                self.save_workspace_metadata(workspace)?;
            }

            println!(
                "✅ Deleted checkpoint '{}' ({})",
                checkpoint.name, checkpoint_id
            );
        }

        Ok(())
    }

    /// Get workspace statistics
    pub fn get_workspace_statistics(&self) -> Option<WorkspaceStatistics> {
        self.current_workspace
            .as_ref()
            .map(|workspace| WorkspaceStatistics {
                workspace_name: workspace.name.clone(),
                checkpoint_count: workspace.checkpoints.len(),
                created_at: workspace.created_at,
                last_modified: workspace.last_modified,
                total_size_bytes: workspace
                    .checkpoints
                    .iter()
                    .map(|cp| cp.metadata.size_bytes)
                    .sum(),
                auto_checkpoint_interval: self.auto_checkpoint_interval,
            })
    }

    fn get_workspace_path(&self, workspace_id: &str) -> Result<PathBuf> {
        let path = self.workspace_dir.join(workspace_id);
        if !path.exists() {
            return Err(GraphRAGError::Config {
                message: format!("Workspace path does not exist: {}", path.display()),
            });
        }
        Ok(path)
    }

    fn save_workspace_metadata(&self, workspace: &Workspace) -> Result<()> {
        let workspace_path = self.workspace_dir.join(&workspace.id);
        fs::create_dir_all(&workspace_path)?;

        let metadata_path = workspace_path.join("workspace.json");
        let workspace_data = serde_json::to_string_pretty(workspace)?;
        fs::write(metadata_path, workspace_data)?;

        Ok(())
    }

    fn serialize_graph(&self, graph: &KnowledgeGraph) -> Result<String> {
        // Create a serializable representation of the graph
        let graph_data = GraphSnapshot {
            entities: graph.entities().cloned().collect(),
            relationships: graph.relationships().cloned().collect(),
            documents: graph.documents().cloned().collect(),
            created_at: Utc::now(),
        };

        Ok(serde_json::to_string_pretty(&graph_data)?)
    }

    fn deserialize_graph(&self, data: &str) -> Result<KnowledgeGraph> {
        let snapshot: GraphSnapshot = serde_json::from_str(data)?;
        let mut graph = KnowledgeGraph::new();

        // Add entities
        for entity in snapshot.entities {
            graph.add_entity(entity)?;
        }

        // Add relationships
        for relationship in snapshot.relationships {
            graph.add_relationship(relationship)?;
        }

        // Add documents
        for document in snapshot.documents {
            graph.add_document(document)?;
        }

        Ok(graph)
    }

    fn calculate_hash(&self, data: &[u8]) -> String {
        // Simple hash calculation - in production, use a proper hash function
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphSnapshot {
    entities: Vec<crate::core::Entity>,
    relationships: Vec<crate::core::Relationship>,
    documents: Vec<crate::core::Document>,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceInfo {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub checkpoint_count: usize,
}

#[derive(Debug, Clone)]
pub struct WorkspaceStatistics {
    pub workspace_name: String,
    pub checkpoint_count: usize,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub total_size_bytes: u64,
    pub auto_checkpoint_interval: Duration,
}

impl WorkspaceStatistics {
    pub fn print(&self) {
        println!("📁 Workspace Statistics");
        println!("  Name: {}", self.workspace_name);
        println!("  Checkpoints: {}", self.checkpoint_count);
        println!("  Created: {}", self.created_at.format("%Y-%m-%d %H:%M:%S"));
        println!(
            "  Last modified: {}",
            self.last_modified.format("%Y-%m-%d %H:%M:%S")
        );
        println!("  Total size: {} bytes", self.total_size_bytes);
        println!(
            "  Auto checkpoint interval: {} minutes",
            self.auto_checkpoint_interval.as_secs() / 60
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Entity, EntityId, KnowledgeGraph};
    use tempfile::TempDir;

    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        let entity = Entity::new(
            EntityId::new("test_entity".to_string()),
            "Test Entity".to_string(),
            "ORGANIZATION".to_string(),
            0.9,
        );

        graph.add_entity(entity).unwrap();
        graph
    }

    #[test]
    fn test_workspace_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = WorkspaceManager::new(temp_dir.path());

        let workspace = manager.create_workspace("Test Workspace").unwrap();

        assert_eq!(workspace.name, "Test Workspace");
        assert!(workspace.id.starts_with("ws_"));
        assert!(workspace.checkpoints.is_empty());
    }

    #[test]
    fn test_checkpoint_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = WorkspaceManager::new(temp_dir.path());

        manager.create_workspace("Test Workspace").unwrap();
        let graph = create_test_graph();

        let checkpoint = manager
            .create_checkpoint("Test Checkpoint", "Test description", &graph)
            .unwrap();

        assert_eq!(checkpoint.name, "Test Checkpoint");
        assert_eq!(checkpoint.description, "Test description");
        assert_eq!(checkpoint.metadata.entity_count, 1);
    }

    #[test]
    fn test_checkpoint_loading() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = WorkspaceManager::new(temp_dir.path());

        manager.create_workspace("Test Workspace").unwrap();
        let original_graph = create_test_graph();

        let checkpoint = manager
            .create_checkpoint("Test Checkpoint", "Test description", &original_graph)
            .unwrap();
        let loaded_graph = manager.load_checkpoint(&checkpoint.id).unwrap();

        assert_eq!(loaded_graph.entity_count(), original_graph.entity_count());
    }

    #[test]
    fn test_workspace_listing() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = WorkspaceManager::new(temp_dir.path());

        manager.create_workspace("Workspace 1").unwrap();
        // Sleep briefly to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.create_workspace("Workspace 2").unwrap();

        let workspaces = manager.list_workspaces().unwrap();
        assert_eq!(workspaces.len(), 2);

        let names: Vec<String> = workspaces.iter().map(|w| w.name.clone()).collect();
        assert!(names.contains(&"Workspace 1".to_string()));
        assert!(names.contains(&"Workspace 2".to_string()));
    }

    #[test]
    fn test_auto_checkpoint_interval() {
        let temp_dir = TempDir::new().unwrap();
        let manager = WorkspaceManager::new(temp_dir.path())
            .with_auto_checkpoint_interval(Duration::from_secs(30));

        assert_eq!(manager.auto_checkpoint_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_workspace_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = WorkspaceManager::new(temp_dir.path());

        manager.create_workspace("Test Workspace").unwrap();
        let graph = create_test_graph();
        manager
            .create_checkpoint("Test Checkpoint", "Test description", &graph)
            .unwrap();

        let stats = manager.get_workspace_statistics().unwrap();
        assert_eq!(stats.workspace_name, "Test Workspace");
        assert_eq!(stats.checkpoint_count, 1);
        assert!(stats.total_size_bytes > 0);
    }
}
