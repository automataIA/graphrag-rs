#![allow(unused_imports)]

use crate::config::Config;
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship,
    Result, TextChunk,
};
use crate::{critic, ollama, persistence, query, retrieval};

#[cfg(feature = "parallel-processing")]
#[allow(unused_imports)]
use crate::parallel;

use super::GraphRAG;

impl GraphRAG {

    // ================================
    // CONVENIENCE CONSTRUCTORS
    // ================================

    /// Create GraphRAG from a JSON5 config file
    ///
    /// This is a convenience method that loads a JSON5 config file and creates a GraphRAG instance.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "json5-support")]
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// let graphrag = GraphRAG::from_json5_file("config/templates/symposium_zero_cost.graphrag.json5")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "json5-support")]
    pub fn from_json5_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use crate::config::json5_loader::load_json5_config;
        use crate::config::setconfig::SetConfig;

        let set_config = load_json5_config::<SetConfig, _>(path)?;
        let config = set_config.to_graphrag_config();
        Self::new(config)
    }

    /// Create GraphRAG from a config file (auto-detect format: TOML, JSON5, YAML, JSON)
    ///
    /// This method automatically detects the config file format based on the file extension
    /// and loads it appropriately.
    ///
    /// Supported formats:
    /// - `.toml` - TOML format
    /// - `.json5` - JSON5 format (requires `json5-support` feature)
    /// - `.yaml`, `.yml` - YAML format
    /// - `.json` - JSON format
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// // Auto-detect format from extension
    /// let graphrag = GraphRAG::from_config_file("config/templates/symposium_zero_cost.graphrag.json5")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use crate::config::setconfig::SetConfig;

        let set_config = SetConfig::from_file(path)?;
        let config = set_config.to_graphrag_config();
        Self::new(config)
    }

    /// Complete workflow: load config + process document + build graph
    ///
    /// This is the most convenient method for getting started with GraphRAG. It:
    /// 1. Loads the config file (auto-detecting the format)
    /// 2. Initializes the GraphRAG system
    /// 3. Loads and processes the document
    /// 4. Builds the knowledge graph
    ///
    /// After this method completes, the GraphRAG instance is ready to answer queries.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # #[cfg(feature = "async")]
    /// # async fn example() -> graphrag_core::Result<()> {
    /// use graphrag_core::GraphRAG;
    ///
    /// // Complete workflow in one call
    /// let mut graphrag = GraphRAG::from_config_and_document(
    ///     "config/templates/symposium_zero_cost.graphrag.json5",
    ///     "docs-example/Symposium.txt"
    /// ).await?;
    ///
    /// // Ready to query
    /// let answer = graphrag.ask("What is Socrates' view on love?").await?;
    /// println!("Answer: {}", answer);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn from_config_and_document<P1, P2>(
        config_path: P1,
        document_path: P2,
    ) -> Result<Self>
    where
        P1: AsRef<std::path::Path>,
        P2: AsRef<std::path::Path>,
    {
        // Load config
        let mut graphrag = Self::from_config_file(config_path)?;

        // Initialize
        graphrag.initialize()?;

        // Load document
        let content = std::fs::read_to_string(document_path).map_err(GraphRAGError::Io)?;

        graphrag.add_document_from_text(&content)?;

        // Build graph
        graphrag.build_graph().await?;

        Ok(graphrag)
    }

    /// Quick start: Create a ready-to-query GraphRAG instance from text in one call
    ///
    /// This is the simplest way to get started with GraphRAG. It:
    /// 1. Creates a new instance with default or hierarchical configuration
    /// 2. Initializes all components
    /// 3. Processes your text document
    /// 4. Builds the knowledge graph
    ///
    /// After this call, you can immediately use `ask()` to query the system.
    ///
    /// # Example: Hello World in 5 lines
    /// ```rust,no_run
    /// use graphrag_core::prelude::*;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let mut graphrag = GraphRAG::quick_start("Your document text here").await?;
    /// let answer = graphrag.ask("What is this document about?").await?;
    /// println!("{}", answer);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Configuration
    /// - With `hierarchical-config` feature: Uses layered config (defaults → user → project → env)
    /// - Without: Uses sensible defaults optimized for local Ollama setup
    #[cfg(feature = "async")]
    pub async fn quick_start(text: &str) -> Result<Self> {
        // Load config (hierarchical if available, otherwise defaults)
        let config = Config::load()?;

        let mut graphrag = Self::new(config)?;
        graphrag.initialize()?;
        graphrag.add_document_from_text(text)?;
        graphrag.build_graph().await?;

        Ok(graphrag)
    }

    /// Quick start with custom configuration
    ///
    /// Like `quick_start()`, but allows you to customize the configuration
    /// using the builder pattern before processing the document.
    ///
    /// # Example
    /// ```rust,no_run
    /// use graphrag_core::prelude::*;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let mut graphrag = GraphRAG::quick_start_with_config(
    ///     "Your document text",
    ///     |builder| builder
    ///         .with_chunk_size(256)
    ///         .with_ollama_enabled(true)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub async fn quick_start_with_config<F>(text: &str, configure: F) -> Result<Self>
    where
        F: FnOnce(crate::builder::GraphRAGBuilder) -> crate::builder::GraphRAGBuilder,
    {
        let builder = configure(Self::builder());
        let mut graphrag = builder.build()?;
        graphrag.initialize()?;
        graphrag.add_document_from_text(text)?;
        graphrag.build_graph().await?;

        Ok(graphrag)
    }
}
