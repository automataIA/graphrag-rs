//! GraphRAG operations handler
//!
//! Provides a thread-safe wrapper around GraphRAG instance with async operations.

use color_eyre::eyre::{eyre, Result};
use graphrag_core::{Config, Entity, GraphRAG, persistence::WorkspaceManager};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Statistics about the knowledge graph
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub entities: usize,
    pub relationships: usize,
    pub documents: usize,
    pub chunks: usize,
}

/// Thread-safe GraphRAG handler
#[derive(Clone)]
pub struct GraphRAGHandler {
    graphrag: Arc<Mutex<Option<GraphRAG>>>,
}

impl GraphRAGHandler {
    /// Create a new GraphRAG handler
    pub fn new() -> Self {
        Self {
            graphrag: Arc::new(Mutex::new(None)),
        }
    }

    /// Check if GraphRAG is initialized
    pub async fn is_initialized(&self) -> bool {
        let guard = self.graphrag.lock().await;
        guard.is_some()
    }

    /// Initialize GraphRAG with configuration
    pub async fn initialize(&self, config: Config) -> Result<()> {
        tracing::info!("Initializing GraphRAG with config");

        let mut graphrag = GraphRAG::new(config)?;
        graphrag.initialize()?;

        let mut guard = self.graphrag.lock().await;
        *guard = Some(graphrag);

        tracing::info!("GraphRAG initialized successfully");
        Ok(())
    }

    /// Load a document into the knowledge graph
    ///
    /// # Arguments
    /// * `path` - Path to the document to load
    /// * `rebuild` - If true, clears existing graph AND documents before loading (forces complete rebuild)
    pub async fn load_document_with_options(&self, path: &Path, rebuild: bool) -> Result<String> {
        tracing::info!("Loading document: {:?} (rebuild: {})", path, rebuild);

        // Read file asynchronously
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| eyre!("Failed to read file: {}", e))?;

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Add document and build graph with tokio::sync::Mutex
        let mut guard = self.graphrag.lock().await;
        if let Some(ref mut graphrag) = *guard {
            // Clear graph AND documents if rebuild is requested (BEFORE adding new document)
            if rebuild {
                tracing::info!("Clearing existing graph and documents for rebuild");
                // Re-initialize to clear everything including documents and chunks
                graphrag.initialize()?;
            }

            // Add the document
            graphrag.add_document_from_text(&content)?;

            // Build graph asynchronously (async feature is always enabled in CLI)
            graphrag.build_graph().await?;

            let message = if rebuild {
                format!("Document '{}' loaded successfully (complete rebuild from scratch)", filename)
            } else {
                format!("Document '{}' loaded successfully", filename)
            };

            Ok(message)
        } else {
            Err(eyre!("GraphRAG not initialized"))
        }
    }

    /// Load a document into the knowledge graph (backwards compatibility)
    pub async fn load_document(&self, path: &Path) -> Result<String> {
        self.load_document_with_options(path, false).await
    }

    /// Clear the knowledge graph (preserves documents and chunks)
    pub async fn clear_graph(&self) -> Result<String> {
        tracing::info!("Clearing knowledge graph");

        let mut guard = self.graphrag.lock().await;
        if let Some(ref mut graphrag) = *guard {
            graphrag.clear_graph()?;
            Ok("Knowledge graph cleared successfully. Entities and relationships removed, documents preserved.".to_string())
        } else {
            Err(eyre!("GraphRAG not initialized"))
        }
    }

    /// Rebuild the knowledge graph from existing documents
    ///
    /// This clears the graph and re-extracts entities and relationships from all loaded documents.
    /// Useful after changing configuration or to fix issues with the graph.
    pub async fn rebuild_graph(&self) -> Result<String> {
        tracing::info!("Rebuilding knowledge graph from existing documents");

        let mut guard = self.graphrag.lock().await;
        if let Some(ref mut graphrag) = *guard {
            // Clear the existing graph
            graphrag.clear_graph()?;

            // Check if there are documents to rebuild from
            if !graphrag.has_documents() {
                return Err(eyre!("No documents loaded. Use /load <file> to load a document first."));
            }

            // Rebuild the graph from existing documents
            graphrag.build_graph().await?;

            let stats = graphrag.knowledge_graph()
                .map(|kg| (kg.entities().count(), kg.relationships().count()))
                .unwrap_or((0, 0));

            Ok(format!(
                "Knowledge graph rebuilt successfully. Extracted {} entities and {} relationships.",
                stats.0, stats.1
            ))
        } else {
            Err(eyre!("GraphRAG not initialized"))
        }
    }

    /// Execute a query against the knowledge graph
    pub async fn query(&self, query_text: &str) -> Result<String> {
        tracing::info!("Executing query: {}", query_text);

        let mut guard = self.graphrag.lock().await;
        if let Some(ref mut graphrag) = *guard {
            // async feature is always enabled in CLI
            let answer = graphrag.ask(query_text).await?;
            Ok(answer)
        } else {
            Err(eyre!("GraphRAG not initialized. Use /config to load a configuration first."))
        }
    }

    /// Execute a query and return both LLM answer and raw search results
    ///
    /// Returns a tuple of (llm_answer, raw_results)
    pub async fn query_with_raw(&self, query_text: &str) -> Result<(String, Vec<String>)> {
        tracing::info!("Executing query with raw results: {}", query_text);

        let mut guard = self.graphrag.lock().await;
        if let Some(ref mut graphrag) = *guard {
            // Get raw search results first
            let raw_results = graphrag.query_internal(query_text)?;

            // Then get the LLM-processed answer
            let answer = graphrag.ask(query_text).await?;

            Ok((answer, raw_results))
        } else {
            Err(eyre!("GraphRAG not initialized. Use /config to load a configuration first."))
        }
    }

    /// Get knowledge graph statistics
    pub async fn get_stats(&self) -> Option<GraphStats> {
        let guard = self.graphrag.lock().await;
        guard.as_ref().and_then(|g| {
            g.knowledge_graph().map(|kg| GraphStats {
                entities: kg.entities().count(),
                relationships: kg.relationships().count(),
                documents: kg.documents().count(),
                chunks: kg.chunks().count(),
            })
        })
    }

    /// Get all entities, optionally filtered
    pub async fn get_entities(&self, filter: Option<&str>) -> Result<Vec<Entity>> {
        let guard = self.graphrag.lock().await;
        if let Some(ref graphrag) = *guard {
            if let Some(kg) = graphrag.knowledge_graph() {
                let entities: Vec<Entity> = match filter {
                    Some(f) => kg
                        .entities()
                        .filter(|e| {
                            e.name.to_lowercase().contains(&f.to_lowercase())
                                || e.entity_type.to_lowercase().contains(&f.to_lowercase())
                        })
                        .cloned()
                        .collect(),
                    None => kg.entities().cloned().collect(),
                };
                Ok(entities)
            } else {
                Err(eyre!("Knowledge graph not built yet"))
            }
        } else {
            Err(eyre!("GraphRAG not initialized"))
        }
    }

    /// Check if knowledge graph exists
    #[allow(dead_code)]
    pub async fn has_knowledge_graph(&self) -> bool {
        let guard = self.graphrag.lock().await;
        if let Some(ref graphrag) = *guard {
            graphrag.knowledge_graph().is_some()
        } else {
            false
        }
    }

    // ========= Workspace Operations =========

    /// List all available workspaces
    pub async fn list_workspaces(&self, workspace_dir: &str) -> Result<String> {
        let workspace_manager = WorkspaceManager::new(workspace_dir)?;
        let workspaces = workspace_manager.list_workspaces()?;

        if workspaces.is_empty() {
            return Ok("No workspaces found. Use /workspace save <name> to create one.".to_string());
        }

        let mut output = format!("📁 Available Workspaces ({} total):\n\n", workspaces.len());

        for (i, ws) in workspaces.iter().enumerate() {
            output.push_str(&format!(
                "{}. {} ({:.2} KB)\n",
                i + 1,
                ws.name,
                ws.size_bytes as f64 / 1024.0
            ));
            output.push_str(&format!(
                "   Entities: {}, Relationships: {}, Documents: {}, Chunks: {}\n",
                ws.metadata.entity_count,
                ws.metadata.relationship_count,
                ws.metadata.document_count,
                ws.metadata.chunk_count
            ));
            output.push_str(&format!(
                "   Created: {}\n",
                ws.metadata.created_at.format("%Y-%m-%d %H:%M:%S")
            ));
            if let Some(desc) = &ws.metadata.description {
                output.push_str(&format!("   Description: {}\n", desc));
            }
            output.push('\n');
        }

        Ok(output)
    }

    /// Save current knowledge graph to workspace
    pub async fn save_workspace(&self, workspace_dir: &str, name: &str) -> Result<String> {
        let guard = self.graphrag.lock().await;
        if let Some(ref graphrag) = *guard {
            if let Some(kg) = graphrag.knowledge_graph() {
                let workspace_manager = WorkspaceManager::new(workspace_dir)?;
                workspace_manager.save_graph(kg, name)?;

                let stats = (kg.entities().count(), kg.relationships().count(), kg.documents().count(), kg.chunks().count());

                Ok(format!(
                    "✅ Workspace '{}' saved successfully!\n\n\
                     Saved: {} entities, {} relationships, {} documents, {} chunks",
                    name, stats.0, stats.1, stats.2, stats.3
                ))
            } else {
                Err(eyre!("No knowledge graph to save. Build a graph first with /load <file>"))
            }
        } else {
            Err(eyre!("GraphRAG not initialized"))
        }
    }

    /// Load knowledge graph from workspace
    pub async fn load_workspace(&self, workspace_dir: &str, name: &str) -> Result<String> {
        let workspace_manager = WorkspaceManager::new(workspace_dir)?;
        let loaded_kg = workspace_manager.load_graph(name)?;

        let stats = (
            loaded_kg.entities().count(),
            loaded_kg.relationships().count(),
            loaded_kg.documents().count(),
            loaded_kg.chunks().count()
        );

        // Replace the current knowledge graph
        let mut guard = self.graphrag.lock().await;
        if let Some(ref mut graphrag) = *guard {
            // Replace the knowledge graph using the mutable accessor
            if let Some(kg_mut) = graphrag.knowledge_graph_mut() {
                *kg_mut = loaded_kg;
            } else {
                return Err(eyre!("Knowledge graph not initialized. Use /config first."));
            }

            Ok(format!(
                "✅ Workspace '{}' loaded successfully!\n\n\
                 Loaded: {} entities, {} relationships, {} documents, {} chunks",
                name, stats.0, stats.1, stats.2, stats.3
            ))
        } else {
            Err(eyre!("GraphRAG not initialized. Use /config to load configuration first."))
        }
    }

    /// Delete a workspace
    pub async fn delete_workspace(&self, workspace_dir: &str, name: &str) -> Result<String> {
        let workspace_manager = WorkspaceManager::new(workspace_dir)?;

        // Confirm deletion (in TUI this would be a confirmation dialog)
        workspace_manager.delete_workspace(name)?;

        Ok(format!("✅ Workspace '{}' deleted successfully.", name))
    }
}

impl Default for GraphRAGHandler {
    fn default() -> Self {
        Self::new()
    }
}
