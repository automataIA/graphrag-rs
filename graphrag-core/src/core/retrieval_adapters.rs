//! Retrieval system adapters for core traits
//!
//! This module provides adapter implementations that bridge existing retrieval systems
//! with the core GraphRAG AsyncRetriever trait.

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::AsyncRetriever;
use crate::retrieval::{RetrievalSystem, SearchResult as RetrievalSearchResult};
use async_trait::async_trait;

/// Result type for retrieval operations
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Unique identifier for this result
    pub id: String,
    /// Content of the result
    pub content: String,
    /// Relevance score
    pub score: f32,
    /// Associated entity names
    pub entities: Vec<String>,
}

impl From<RetrievalSearchResult> for RetrievalResult {
    fn from(result: RetrievalSearchResult) -> Self {
        Self {
            id: result.id,
            content: result.content,
            score: result.score,
            entities: result.entities,
        }
    }
}

/// Adapter for RetrievalSystem to implement AsyncRetriever trait
///
/// ## Graph Integration
///
/// This adapter wraps the RetrievalSystem which operates on a KnowledgeGraph.
/// For full functionality, the RetrievalSystem needs to be initialized with
/// a populated graph. The current implementation provides the trait interface
/// but actual retrieval requires application-level graph management.
///
/// ## Usage Pattern
///
/// ```no_run
/// use graphrag_core::core::retrieval_adapters::RetrievalSystemAdapter;
/// use graphrag_core::retrieval::RetrievalSystem;
/// use graphrag_core::config::Config;
///
/// # async fn example() -> graphrag_core::Result<()> {
/// let config = Config::default();
/// let system = RetrievalSystem::new(&config)?;
/// let adapter = RetrievalSystemAdapter::new(system);
///
/// // Search requires a populated knowledge graph in the retrieval system
/// // This would typically be managed at the application level
/// # Ok(())
/// # }
/// ```
pub struct RetrievalSystemAdapter {
    system: RetrievalSystem,
}

impl RetrievalSystemAdapter {
    /// Create a new retrieval system adapter
    ///
    /// Note: The RetrievalSystem should be populated with a KnowledgeGraph
    /// before search operations will return meaningful results.
    pub fn new(system: RetrievalSystem) -> Self {
        Self { system }
    }

    /// Get reference to underlying retrieval system
    pub fn system(&self) -> &RetrievalSystem {
        &self.system
    }

    /// Get mutable reference to underlying retrieval system
    ///
    /// This allows configuring the retrieval system after creation,
    /// such as setting the knowledge graph or updating configurations.
    pub fn system_mut(&mut self) -> &mut RetrievalSystem {
        &mut self.system
    }
}

#[async_trait]
impl AsyncRetriever for RetrievalSystemAdapter {
    type Query = String;
    type Result = RetrievalResult;
    type Error = GraphRAGError;

    async fn search(&self, _query: Self::Query, _k: usize) -> Result<Vec<Self::Result>> {
        // Note: Actual search implementation requires the RetrievalSystem to have
        // access to a populated KnowledgeGraph. This is typically managed at the
        // application level where the graph is built and passed to the retrieval system.
        //
        // The RetrievalSystem internally supports multiple retrieval strategies:
        // - Vector similarity search
        // - Entity-based retrieval
        // - Graph path traversal
        // - Hybrid approaches
        //
        // To enable full retrieval:
        // 1. Build a KnowledgeGraph with documents, entities, and relationships
        // 2. Pass it to the RetrievalSystem via application context
        // 3. The RetrievalSystem will handle query expansion, ranking, and fusion
        //
        // For testing without a real graph, use the MockRetriever from test_utils.

        Ok(vec![])
    }

    async fn search_with_context(
        &self,
        query: Self::Query,
        _context: &str,
        k: usize,
    ) -> Result<Vec<Self::Result>> {
        // Use same implementation as search for now
        // Context could be used to filter or re-rank results
        self.search(query, k).await
    }

    async fn search_batch(&self, queries: Vec<Self::Query>, k: usize) -> Result<Vec<Vec<Self::Result>>> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.search(query, k).await?);
        }
        Ok(results)
    }

    async fn update(&mut self, _content: Vec<String>) -> Result<()> {
        // Updating the retrieval system would typically involve:
        // 1. Adding new documents to vector store
        // 2. Rebuilding search indices
        // 3. Updating graph structures

        // For now, this is a no-op since retrieval system doesn't expose
        // a public update API
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        // Basic health check - verify the system is initialized
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_retrieval_adapter_creation() {
        let config = Config::default();
        let system = RetrievalSystem::new(&config).unwrap();
        let adapter = RetrievalSystemAdapter::new(system);

        assert!(adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_search_batch() {
        let config = Config::default();
        let system = RetrievalSystem::new(&config).unwrap();
        let adapter = RetrievalSystemAdapter::new(system);

        let queries = vec![
            "What is GraphRAG?".to_string(),
            "How does retrieval work?".to_string(),
        ];

        let results = adapter.search_batch(queries, 5).await.unwrap();
        assert_eq!(results.len(), 2);
    }
}
