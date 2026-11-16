#![allow(unused_imports)]

use crate::config::{Config, Validatable};
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use crate::{critic, llm, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

use super::GraphRAG;

impl GraphRAG {
    /// Reject providers that the synchronous, local pipeline cannot execute.
    pub(super) fn ensure_sync_supported(&self) -> Result<()> {
        if self.config.llm.enabled {
            return Err(GraphRAGError::Unsupported {
                operation: "synchronous GraphRAG query/build".to_string(),
                reason: "LLM generation and gleaning require the async API".to_string(),
            });
        }

        if !self.config.embeddings.backend.eq_ignore_ascii_case("hash") {
            return Err(GraphRAGError::Unsupported {
                operation: "synchronous GraphRAG query/build".to_string(),
                reason: format!(
                    "embedding provider '{}' (including llama.cpp) is unavailable in the local sync pipeline; use hash embeddings or the async API",
                    self.config.embeddings.backend
                ),
            });
        }

        Ok(())
    }

    /// Create a new GraphRAG instance with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        let validation = config.validate();
        if !validation.is_valid {
            return Err(GraphRAGError::Config {
                message: validation.errors.join("; "),
            });
        }
        Ok(Self {
            config,
            knowledge_graph: None,
            retrieval_system: None,
            query_planner: None,
            critic: None,
            pending_documents: indexmap::IndexSet::new(),
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
        })
    }

    /// Create a Zero-Config local GraphRAG instance
    /// Uses: Candle (MiniLM) for embeddings, Memory/LanceDB for storage, llama.cpp (OpenAI-compatible) for LLM
    pub fn default_local() -> Result<Self> {
        let mut config = Config::default();
        // Configure for local use
        config.llm.enabled = true;
        // config.storage.type = StorageType::LanceDB; // Future

        Self::new(config)
    }

    /// Create a builder for configuring GraphRAG
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::GraphRAG;
    ///
    /// # fn example() -> graphrag_core::Result<()> {
    /// let graphrag = GraphRAG::builder()
    ///     .with_output_dir("./workspace")
    ///     .with_chunk_size(512)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> crate::builder::GraphRAGBuilder {
        crate::builder::GraphRAGBuilder::new()
    }

    /// Initialize the GraphRAG system.
    ///
    /// When `auto_save.enabled = true` and a `base_dir` is configured, attempts to
    /// load an existing graph from the workspace on disk before starting fresh.
    /// This means a second run reuses the previously built graph automatically.
    pub fn initialize(&mut self) -> Result<()> {
        // Try to restore from workspace if persistent storage is configured
        let loaded = self.try_load_from_workspace()?;
        // reloaded documents were already extracted when the workspace was saved
        self.pending_documents.clear();

        if !loaded {
            self.knowledge_graph = Some(KnowledgeGraph::new());
        }

        self.retrieval_system = Some(retrieval::RetrievalSystem::new(&self.config)?);

        if self.config.llm.enabled {
            let client = llm::LlmClient::new(self.config.llm.clone());
            self.query_planner = Some(query::planner::QueryPlanner::new(client));
        }

        Ok(())
    }

    /// Attempt to load the knowledge graph from a workspace on disk.
    /// Returns `true` when a workspace was loaded. Missing optional workspaces are fine;
    /// malformed or inaccessible configured workspaces are returned to the caller.
    fn try_load_from_workspace(&mut self) -> Result<bool> {
        if !self.config.auto_save.enabled {
            return Ok(false);
        }
        let base_dir = match &self.config.auto_save.base_dir {
            Some(d) => d.clone(),
            None => return Ok(false),
        };
        let workspace_name = self
            .config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");

        let manager = persistence::WorkspaceManager::new(&base_dir)?;

        // A workspace holding only `extraction_cache/` (interrupted first index) has no graph yet.
        if !manager.workspace_exists(workspace_name)
            || !manager
                .workspace_path(workspace_name)
                .join("graph.json")
                .exists()
        {
            return Ok(false);
        }

        let graph = manager.load_graph(workspace_name)?;
        if let (Some(stored), Ok(current)) = (
            manager.workspace_ontology(workspace_name),
            self.config.ontology(),
        ) {
            if stored != current.id() {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    "Workspace '{}' was built with ontology {} but config uses {}; \
                     re-index to get consistent entity/relation types",
                    workspace_name,
                    stored,
                    current.id()
                );
            }
        }
        #[cfg(feature = "tracing")]
        tracing::info!(
            "Loaded graph from workspace '{}' ({} entities, {} relationships)",
            workspace_name,
            graph.entity_count(),
            graph.relationship_count(),
        );
        self.knowledge_graph = Some(graph);
        Ok(true)
    }

    /// Save the current knowledge graph to the configured workspace on disk.
    /// No-op when `auto_save.enabled = false` or `base_dir` is not set.
    pub fn save_to_workspace(&self) -> Result<()> {
        if !self.config.auto_save.enabled {
            return Ok(());
        }
        let base_dir = match &self.config.auto_save.base_dir {
            Some(d) => d,
            None => return Ok(()),
        };
        let workspace_name = self
            .config
            .auto_save
            .workspace_name
            .as_deref()
            .unwrap_or("default");

        let graph = self
            .knowledge_graph
            .as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let manager = persistence::WorkspaceManager::new(base_dir)?;
        manager.save_graph(graph, workspace_name)?;
        manager.record_ontology(workspace_name, &self.config.ontology()?.id())?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Saved graph to workspace '{}' in '{}' ({} entities, {} relationships)",
            workspace_name,
            base_dir,
            graph.entity_count(),
            graph.relationship_count(),
        );
        Ok(())
    }

    /// Clear all entities and relationships from the knowledge graph
    ///
    /// This method preserves documents and text chunks but removes all extracted entities and relationships.
    /// Useful for rebuilding the graph from scratch without reloading documents.
    pub fn clear_graph(&mut self) -> Result<()> {
        let graph = self
            .knowledge_graph
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Clearing knowledge graph (preserving documents and chunks)");

        graph.clear_entities_and_relationships();
        self.pending_documents = graph.documents().map(|d| d.id.clone()).collect();
        Ok(())
    }
}
