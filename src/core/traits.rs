//! Core traits for GraphRAG system components
//!
//! This module defines the fundamental abstractions that enable modularity,
//! testability, and flexibility throughout the GraphRAG system.
//!
//! ## Async Migration
//!
//! All core traits have been migrated to async/await patterns for:
//! - Non-blocking I/O operations (LLM calls, database access, network requests)
//! - Better resource utilization with concurrent processing
//! - Improved throughput for high-load scenarios
//! - Future-proof architecture for cloud deployments

use crate::core::Result;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

#[cfg(feature = "async-trait")]
use async_trait::async_trait;

/// Type alias for vector metadata
pub type VectorMetadata = Option<HashMap<String, String>>;

/// Type alias for vector batch operations
pub type VectorBatch = Vec<(String, Vec<f32>, VectorMetadata)>;

/// Core storage abstraction for persisting and retrieving entities, documents, and graph data
///
/// ## Synchronous Version
/// This trait provides synchronous operations for storage.
pub trait Storage {
    type Entity;
    type Document;
    type Chunk;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Store an entity and return its assigned ID
    fn store_entity(&mut self, entity: Self::Entity) -> Result<String>;

    /// Retrieve an entity by its ID
    fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>>;

    /// Store a document and return its assigned ID
    fn store_document(&mut self, document: Self::Document) -> Result<String>;

    /// Retrieve a document by its ID
    fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>>;

    /// Store a chunk and return its assigned ID
    fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String>;

    /// Retrieve a chunk by its ID
    fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>>;

    /// List all entity IDs
    fn list_entities(&self) -> Result<Vec<String>>;

    /// Batch operations for performance
    fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>>;
}

/// Async storage abstraction for non-blocking storage operations
///
/// ## Async Version
/// This trait provides async operations for storage with better concurrency and resource utilization.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncStorage: Send + Sync {
    type Entity: Send + Sync;
    type Document: Send + Sync;
    type Chunk: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Store an entity and return its assigned ID
    async fn store_entity(&mut self, entity: Self::Entity) -> Result<String>;

    /// Retrieve an entity by its ID
    async fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>>;

    /// Store a document and return its assigned ID
    async fn store_document(&mut self, document: Self::Document) -> Result<String>;

    /// Retrieve a document by its ID
    async fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>>;

    /// Store a chunk and return its assigned ID
    async fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String>;

    /// Retrieve a chunk by its ID
    async fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>>;

    /// List all entity IDs
    async fn list_entities(&self) -> Result<Vec<String>>;

    /// Batch operations for performance
    async fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>>;

    /// Health check for storage connection
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Flush any pending operations
    async fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Text embedding abstraction for converting text to vector representations
///
/// ## Synchronous Version
/// This trait provides synchronous operations for text embeddings.
pub trait Embedder {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Generate embeddings for a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts in batch
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the dimensionality of embeddings produced by this embedder
    fn dimension(&self) -> usize;

    /// Check if the embedder is ready for use
    fn is_ready(&self) -> bool;
}

/// Async text embedding abstraction for non-blocking embedding operations
///
/// ## Async Version
/// This trait provides async operations for text embeddings with better throughput for large batches.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncEmbedder: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Generate embeddings for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts in batch
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Generate embeddings for multiple texts with concurrency control
    async fn embed_batch_concurrent(&self, texts: &[&str], max_concurrent: usize) -> Result<Vec<Vec<f32>>> {
        if max_concurrent <= 1 {
            return self.embed_batch(texts).await;
        }

        let chunks: Vec<_> = texts.chunks(max_concurrent).collect();
        let mut results = Vec::with_capacity(texts.len());

        for chunk in chunks {
            let batch_results = self.embed_batch(chunk).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Get the dimensionality of embeddings produced by this embedder
    fn dimension(&self) -> usize;

    /// Check if the embedder is ready for use
    async fn is_ready(&self) -> bool;

    /// Health check for embedding service
    async fn health_check(&self) -> Result<bool> {
        self.is_ready().await.then_some(true).ok_or_else(|| {
            crate::core::GraphRAGError::Retrieval {
                message: "Embedding service health check failed".to_string(),
            }
        })
    }
}

/// Vector similarity search abstraction for finding similar embeddings
///
/// ## Synchronous Version
/// This trait provides synchronous operations for vector search.
pub trait VectorStore {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Add a vector with associated ID and metadata
    fn add_vector(&mut self, id: String, vector: Vec<f32>, metadata: VectorMetadata) -> Result<()>;

    /// Add multiple vectors in batch
    fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()>;

    /// Search for k most similar vectors
    fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search with distance threshold
    fn search_with_threshold(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>>;

    /// Remove a vector by ID
    fn remove_vector(&mut self, id: &str) -> Result<bool>;

    /// Get vector count
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool;
}

/// Async vector similarity search abstraction for non-blocking vector operations
///
/// ## Async Version
/// This trait provides async operations for vector search with better concurrency and scalability.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncVectorStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Add a vector with associated ID and metadata
    async fn add_vector(&mut self, id: String, vector: Vec<f32>, metadata: VectorMetadata) -> Result<()>;

    /// Add multiple vectors in batch
    async fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()>;

    /// Add vectors with concurrency control for large batches
    async fn add_vectors_batch_concurrent(&mut self, vectors: VectorBatch, max_concurrent: usize) -> Result<()> {
        if max_concurrent <= 1 {
            return self.add_vectors_batch(vectors).await;
        }

        for chunk in vectors.chunks(max_concurrent) {
            self.add_vectors_batch(chunk.to_vec()).await?;
        }

        Ok(())
    }

    /// Search for k most similar vectors
    async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search with distance threshold
    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>>;

    /// Search multiple queries concurrently
    async fn search_batch(&self, query_vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        let mut results = Vec::with_capacity(query_vectors.len());
        for query in query_vectors {
            let search_results = self.search(query, k).await?;
            results.push(search_results);
        }
        Ok(results)
    }

    /// Remove a vector by ID
    async fn remove_vector(&mut self, id: &str) -> Result<bool>;

    /// Remove multiple vectors in batch
    async fn remove_vectors_batch(&mut self, ids: &[&str]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let removed = self.remove_vector(id).await?;
            results.push(removed);
        }
        Ok(results)
    }

    /// Get vector count
    async fn len(&self) -> usize;

    /// Check if empty
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Health check for vector store
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Build index for better search performance (if applicable)
    async fn build_index(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Result from vector similarity search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<HashMap<String, String>>,
}

/// Entity extraction abstraction for identifying entities in text
///
/// ## Synchronous Version
/// This trait provides synchronous operations for entity extraction.
pub trait EntityExtractor {
    type Entity;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Extract entities from text
    fn extract(&self, text: &str) -> Result<Vec<Self::Entity>>;

    /// Extract entities with confidence scores
    fn extract_with_confidence(&self, text: &str) -> Result<Vec<(Self::Entity, f32)>>;

    /// Set minimum confidence threshold
    fn set_confidence_threshold(&mut self, threshold: f32);
}

/// Async entity extraction abstraction for non-blocking entity extraction
///
/// ## Async Version
/// This trait provides async operations for entity extraction with better throughput for large texts.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncEntityExtractor: Send + Sync {
    type Entity: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Extract entities from text
    async fn extract(&self, text: &str) -> Result<Vec<Self::Entity>>;

    /// Extract entities with confidence scores
    async fn extract_with_confidence(&self, text: &str) -> Result<Vec<(Self::Entity, f32)>>;

    /// Extract entities from multiple texts in batch
    async fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Vec<Self::Entity>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let entities = self.extract(text).await?;
            results.push(entities);
        }
        Ok(results)
    }

    /// Extract entities from multiple texts with concurrency control
    async fn extract_batch_concurrent(&self, texts: &[&str], max_concurrent: usize) -> Result<Vec<Vec<Self::Entity>>> {
        if max_concurrent <= 1 {
            return self.extract_batch(texts).await;
        }

        let chunks: Vec<_> = texts.chunks(max_concurrent).collect();
        let mut results = Vec::with_capacity(texts.len());

        for chunk in chunks {
            let batch_results = self.extract_batch(chunk).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Set minimum confidence threshold
    async fn set_confidence_threshold(&mut self, threshold: f32);

    /// Get current confidence threshold
    async fn get_confidence_threshold(&self) -> f32;

    /// Health check for entity extractor
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

/// Text retrieval abstraction for finding relevant content
///
/// ## Synchronous Version
/// This trait provides synchronous operations for content retrieval.
pub trait Retriever {
    type Query;
    type Result;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Perform a search query
    fn search(&self, query: Self::Query, k: usize) -> Result<Vec<Self::Result>>;

    /// Perform a search with additional context
    fn search_with_context(
        &self,
        query: Self::Query,
        context: &str,
        k: usize,
    ) -> Result<Vec<Self::Result>>;

    /// Update the retriever with new content
    fn update(&mut self, content: Vec<String>) -> Result<()>;
}

/// Async text retrieval abstraction for non-blocking content retrieval
///
/// ## Async Version
/// This trait provides async operations for content retrieval with better scalability and concurrency.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncRetriever: Send + Sync {
    type Query: Send + Sync;
    type Result: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Perform a search query
    async fn search(&self, query: Self::Query, k: usize) -> Result<Vec<Self::Result>>;

    /// Perform a search with additional context
    async fn search_with_context(
        &self,
        query: Self::Query,
        context: &str,
        k: usize,
    ) -> Result<Vec<Self::Result>>;

    /// Perform multiple search queries concurrently
    async fn search_batch(&self, queries: Vec<Self::Query>, k: usize) -> Result<Vec<Vec<Self::Result>>> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            let search_results = self.search(query, k).await?;
            results.push(search_results);
        }
        Ok(results)
    }

    /// Update the retriever with new content
    async fn update(&mut self, content: Vec<String>) -> Result<()>;

    /// Update the retriever with new content in batches
    async fn update_batch(&mut self, content_batches: Vec<Vec<String>>) -> Result<()> {
        for batch in content_batches {
            self.update(batch).await?;
        }
        Ok(())
    }

    /// Refresh/rebuild the retrieval index
    async fn refresh_index(&mut self) -> Result<()> {
        Ok(())
    }

    /// Health check for retrieval system
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Get retrieval statistics
    async fn get_stats(&self) -> Result<RetrievalStats> {
        Ok(RetrievalStats::default())
    }
}

/// Statistics for retrieval operations
#[derive(Debug, Clone, Default)]
pub struct RetrievalStats {
    pub total_queries: u64,
    pub average_response_time_ms: f64,
    pub index_size: usize,
    pub cache_hit_rate: f64,
}

/// Large Language Model abstraction for text generation
///
/// ## Synchronous Version
/// This trait provides synchronous operations for text generation.
pub trait LanguageModel {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Generate text completion
    fn complete(&self, prompt: &str) -> Result<String>;

    /// Generate text with custom parameters
    fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String>;

    /// Check if the model is available
    fn is_available(&self) -> bool;

    /// Get model information
    fn model_info(&self) -> ModelInfo;
}

/// Async Large Language Model abstraction for non-blocking text generation
///
/// ## Async Version
/// This trait provides async operations for text generation with better throughput and concurrency.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncLanguageModel: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Generate text completion
    async fn complete(&self, prompt: &str) -> Result<String>;

    /// Generate text with custom parameters
    async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String>;

    /// Generate multiple text completions concurrently
    async fn complete_batch(&self, prompts: &[&str]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let completion = self.complete(prompt).await?;
            results.push(completion);
        }
        Ok(results)
    }

    /// Generate multiple text completions with concurrency control
    async fn complete_batch_concurrent(&self, prompts: &[&str], max_concurrent: usize) -> Result<Vec<String>> {
        if max_concurrent <= 1 {
            return self.complete_batch(prompts).await;
        }

        let chunks: Vec<_> = prompts.chunks(max_concurrent).collect();
        let mut results = Vec::with_capacity(prompts.len());

        for chunk in chunks {
            let batch_results = self.complete_batch(chunk).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Generate streaming completion (if supported)
    async fn complete_streaming(&self, prompt: &str) -> Result<Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>> {
        // Default implementation converts regular completion to stream
        let result = self.complete(prompt).await?;
        let stream = futures::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }

    /// Check if the model is available
    async fn is_available(&self) -> bool;

    /// Get model information
    async fn model_info(&self) -> ModelInfo;

    /// Health check for language model service
    async fn health_check(&self) -> Result<bool> {
        self.is_available().await.then_some(true).ok_or_else(|| {
            crate::core::GraphRAGError::Generation {
                message: "Language model health check failed".to_string(),
            }
        })
    }

    /// Get model usage statistics
    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        Ok(ModelUsageStats::default())
    }

    /// Estimate tokens for prompt
    async fn estimate_tokens(&self, prompt: &str) -> Result<usize> {
        // Simple estimation: ~4 characters per token
        Ok(prompt.len() / 4)
    }
}

/// Usage statistics for language model
#[derive(Debug, Clone, Default)]
pub struct ModelUsageStats {
    pub total_requests: u64,
    pub total_tokens_processed: u64,
    pub average_response_time_ms: f64,
    pub error_rate: f64,
}

/// Parameters for text generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: None,
        }
    }
}

/// Information about a language model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: Option<String>,
    pub max_context_length: Option<usize>,
    pub supports_streaming: bool,
}

/// Graph operations abstraction for knowledge graph management
///
/// ## Synchronous Version
/// This trait provides synchronous operations for graph management.
pub trait GraphStore {
    type Node;
    type Edge;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Add a node to the graph
    fn add_node(&mut self, node: Self::Node) -> Result<String>;

    /// Add an edge between two nodes
    fn add_edge(&mut self, from_id: &str, to_id: &str, edge: Self::Edge) -> Result<String>;

    /// Find nodes by criteria
    fn find_nodes(&self, criteria: &str) -> Result<Vec<Self::Node>>;

    /// Get neighbors of a node
    fn get_neighbors(&self, node_id: &str) -> Result<Vec<Self::Node>>;

    /// Perform graph traversal
    fn traverse(&self, start_id: &str, max_depth: usize) -> Result<Vec<Self::Node>>;

    /// Get graph statistics
    fn stats(&self) -> GraphStats;
}

/// Async graph operations abstraction for non-blocking graph management
///
/// ## Async Version
/// This trait provides async operations for graph management with better scalability for large graphs.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncGraphStore: Send + Sync {
    type Node: Send + Sync;
    type Edge: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Add a node to the graph
    async fn add_node(&mut self, node: Self::Node) -> Result<String>;

    /// Add multiple nodes in batch
    async fn add_nodes_batch(&mut self, nodes: Vec<Self::Node>) -> Result<Vec<String>> {
        let mut ids = Vec::with_capacity(nodes.len());
        for node in nodes {
            let id = self.add_node(node).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Add an edge between two nodes
    async fn add_edge(&mut self, from_id: &str, to_id: &str, edge: Self::Edge) -> Result<String>;

    /// Add multiple edges in batch
    async fn add_edges_batch(&mut self, edges: Vec<(String, String, Self::Edge)>) -> Result<Vec<String>> {
        let mut ids = Vec::with_capacity(edges.len());
        for (from_id, to_id, edge) in edges {
            let id = self.add_edge(&from_id, &to_id, edge).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Find nodes by criteria
    async fn find_nodes(&self, criteria: &str) -> Result<Vec<Self::Node>>;

    /// Find nodes by multiple criteria concurrently
    async fn find_nodes_batch(&self, criteria_list: &[&str]) -> Result<Vec<Vec<Self::Node>>> {
        let mut results = Vec::with_capacity(criteria_list.len());
        for criteria in criteria_list {
            let nodes = self.find_nodes(criteria).await?;
            results.push(nodes);
        }
        Ok(results)
    }

    /// Get neighbors of a node
    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<Self::Node>>;

    /// Get neighbors of multiple nodes
    async fn get_neighbors_batch(&self, node_ids: &[&str]) -> Result<Vec<Vec<Self::Node>>> {
        let mut results = Vec::with_capacity(node_ids.len());
        for node_id in node_ids {
            let neighbors = self.get_neighbors(node_id).await?;
            results.push(neighbors);
        }
        Ok(results)
    }

    /// Perform graph traversal
    async fn traverse(&self, start_id: &str, max_depth: usize) -> Result<Vec<Self::Node>>;

    /// Perform multiple graph traversals concurrently
    async fn traverse_batch(&self, start_ids: &[&str], max_depth: usize) -> Result<Vec<Vec<Self::Node>>> {
        let mut results = Vec::with_capacity(start_ids.len());
        for start_id in start_ids {
            let traversal = self.traverse(start_id, max_depth).await?;
            results.push(traversal);
        }
        Ok(results)
    }

    /// Get graph statistics
    async fn stats(&self) -> GraphStats;

    /// Health check for graph store
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Optimize graph structure (rebuild indices, etc.)
    async fn optimize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Export graph data
    async fn export(&self) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    /// Import graph data
    async fn import(&mut self, data: &[u8]) -> Result<()> {
        let _ = data; // Unused parameter
        Ok(())
    }
}

/// Statistics about a graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub max_depth: usize,
}

/// Function calling abstraction for tool usage
///
/// ## Synchronous Version
/// This trait provides synchronous operations for function calling.
pub trait FunctionRegistry {
    type Function;
    type CallResult;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Register a new function
    fn register(&mut self, name: String, function: Self::Function) -> Result<()>;

    /// Call a function by name with arguments
    fn call(&self, name: &str, args: &str) -> Result<Self::CallResult>;

    /// List available functions
    fn list_functions(&self) -> Vec<String>;

    /// Check if a function exists
    fn has_function(&self, name: &str) -> bool;
}

/// Async function calling abstraction for non-blocking tool usage
///
/// ## Async Version
/// This trait provides async operations for function calling with better concurrency for tool usage.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncFunctionRegistry: Send + Sync {
    type Function: Send + Sync;
    type CallResult: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Register a new function
    async fn register(&mut self, name: String, function: Self::Function) -> Result<()>;

    /// Call a function by name with arguments
    async fn call(&self, name: &str, args: &str) -> Result<Self::CallResult>;

    /// Call multiple functions concurrently
    async fn call_batch(&self, calls: &[(&str, &str)]) -> Result<Vec<Self::CallResult>> {
        let mut results = Vec::with_capacity(calls.len());
        for (name, args) in calls {
            let result = self.call(name, args).await?;
            results.push(result);
        }
        Ok(results)
    }

    /// List available functions
    async fn list_functions(&self) -> Vec<String>;

    /// Check if a function exists
    async fn has_function(&self, name: &str) -> bool;

    /// Get function metadata
    async fn get_function_info(&self, name: &str) -> Result<Option<FunctionInfo>>;

    /// Health check for function registry
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Validate function arguments before calling
    async fn validate_args(&self, name: &str, args: &str) -> Result<bool> {
        let _ = (name, args); // Unused parameters
        Ok(true)
    }
}

/// Information about a registered function
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: Option<String>,
}

/// Information about a function parameter
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub param_type: String,
    pub description: Option<String>,
    pub required: bool,
}

/// Configuration management abstraction
///
/// ## Synchronous Version
/// This trait provides synchronous operations for configuration management.
pub trait ConfigProvider {
    type Config;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Load configuration from source
    fn load(&self) -> Result<Self::Config>;

    /// Save configuration to source
    fn save(&self, config: &Self::Config) -> Result<()>;

    /// Validate configuration
    fn validate(&self, config: &Self::Config) -> Result<()>;

    /// Get default configuration
    fn default_config(&self) -> Self::Config;
}

/// Async configuration management abstraction for non-blocking configuration operations
///
/// ## Async Version
/// This trait provides async operations for configuration management with better I/O handling.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncConfigProvider: Send + Sync {
    type Config: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Load configuration from source
    async fn load(&self) -> Result<Self::Config>;

    /// Save configuration to source
    async fn save(&self, config: &Self::Config) -> Result<()>;

    /// Validate configuration
    async fn validate(&self, config: &Self::Config) -> Result<()>;

    /// Get default configuration
    async fn default_config(&self) -> Self::Config;

    /// Watch for configuration changes
    async fn watch_changes(&self) -> Result<Pin<Box<dyn futures::Stream<Item = Result<Self::Config>> + Send + 'static>>>
    where
        Self::Config: 'static
    {
        // Default implementation - no change watching
        let stream = futures::stream::empty::<Result<Self::Config>>();
        Ok(Box::pin(stream))
    }

    /// Reload configuration from source
    async fn reload(&self) -> Result<Self::Config> {
        self.load().await
    }

    /// Health check for configuration source
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

/// Monitoring and metrics abstraction
///
/// ## Synchronous Version
/// This trait provides synchronous operations for metrics collection.
pub trait MetricsCollector {
    /// Record a counter metric
    fn counter(&self, name: &str, value: u64, tags: Option<&[(&str, &str)]>);

    /// Record a gauge metric
    fn gauge(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>);

    /// Record a histogram metric
    fn histogram(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>);

    /// Start a timer
    fn timer(&self, name: &str) -> Timer;
}

/// Async monitoring and metrics abstraction for non-blocking metrics collection
///
/// ## Async Version
/// This trait provides async operations for metrics collection with better throughput.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncMetricsCollector: Send + Sync {
    /// Record a counter metric
    async fn counter(&self, name: &str, value: u64, tags: Option<&[(&str, &str)]>);

    /// Record a gauge metric
    async fn gauge(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>);

    /// Record a histogram metric
    async fn histogram(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>);

    /// Record multiple metrics in batch
    async fn record_batch(&self, metrics: &[MetricRecord]) {
        for metric in metrics {
            match metric {
                MetricRecord::Counter { name, value, tags } => {
                    let tags_refs: Option<Vec<(&str, &str)>> = tags.as_ref().map(|t| {
                        t.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
                    });
                    self.counter(name, *value, tags_refs.as_deref()).await;
                }
                MetricRecord::Gauge { name, value, tags } => {
                    let tags_refs: Option<Vec<(&str, &str)>> = tags.as_ref().map(|t| {
                        t.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
                    });
                    self.gauge(name, *value, tags_refs.as_deref()).await;
                }
                MetricRecord::Histogram { name, value, tags } => {
                    let tags_refs: Option<Vec<(&str, &str)>> = tags.as_ref().map(|t| {
                        t.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
                    });
                    self.histogram(name, *value, tags_refs.as_deref()).await;
                }
            }
        }
    }

    /// Start an async timer
    async fn timer(&self, name: &str) -> AsyncTimer;

    /// Health check for metrics collection
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Flush pending metrics
    async fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Metric record for batch operations
#[derive(Debug, Clone)]
pub enum MetricRecord {
    Counter {
        name: String,
        value: u64,
        tags: Option<Vec<(String, String)>>,
    },
    Gauge {
        name: String,
        value: f64,
        tags: Option<Vec<(String, String)>>,
    },
    Histogram {
        name: String,
        value: f64,
        tags: Option<Vec<(String, String)>>,
    },
}

/// Async timer handle for measuring durations
pub struct AsyncTimer {
    name: String,
    start: std::time::Instant,
}

impl AsyncTimer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }

    pub async fn finish(self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Timer handle for measuring durations
pub struct Timer {
    #[allow(dead_code)]
    name: String,
    start: std::time::Instant,
}

impl Timer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }

    pub fn finish(self) -> std::time::Duration {
        self.start.elapsed()
    }
}

/// Serialization abstraction for different formats
///
/// ## Synchronous Version
/// This trait provides synchronous operations for serialization.
pub trait Serializer {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Serialize data to string
    fn serialize<T: serde::Serialize>(&self, data: &T) -> Result<String>;

    /// Deserialize data from string
    fn deserialize<T: serde::de::DeserializeOwned>(&self, data: &str) -> Result<T>;

    /// Get file extension for this format
    fn extension(&self) -> &'static str;
}

/// Async serialization abstraction for non-blocking serialization operations
///
/// ## Async Version
/// This trait provides async operations for serialization with better I/O handling.
#[cfg_attr(feature = "async-trait", async_trait)]
pub trait AsyncSerializer: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Serialize data to string
    async fn serialize<T: serde::Serialize + Send + Sync>(&self, data: &T) -> Result<String>;

    /// Deserialize data from string
    async fn deserialize<T: serde::de::DeserializeOwned + Send + Sync>(&self, data: &str) -> Result<T>;

    /// Serialize data to bytes
    async fn serialize_bytes<T: serde::Serialize + Send + Sync>(&self, data: &T) -> Result<Vec<u8>> {
        let string = self.serialize(data).await?;
        Ok(string.into_bytes())
    }

    /// Deserialize data from bytes
    async fn deserialize_bytes<T: serde::de::DeserializeOwned + Send + Sync>(&self, data: &[u8]) -> Result<T> {
        let string = String::from_utf8(data.to_vec()).map_err(|e| {
            crate::core::GraphRAGError::Serialization {
                message: format!("Invalid UTF-8 data: {e}"),
            }
        })?;
        self.deserialize(&string).await
    }

    /// Serialize multiple objects in batch
    async fn serialize_batch<T: serde::Serialize + Send + Sync>(&self, data: &[T]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(data.len());
        for item in data {
            let serialized = self.serialize(item).await?;
            results.push(serialized);
        }
        Ok(results)
    }

    /// Get file extension for this format
    fn extension(&self) -> &'static str;

    /// Health check for serializer
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

//
// COMPREHENSIVE ASYNC TRAIT EXPORTS AND ADAPTER UTILITIES
//

/// Adapter to convert sync traits to async
pub mod sync_to_async {
    use super::*;
    use std::sync::Arc;

    /// Adapter that wraps a sync Storage to implement AsyncStorage
    pub struct StorageAdapter<T>(pub Arc<tokio::sync::Mutex<T>>);

    #[cfg_attr(feature = "async-trait", async_trait)]
    impl<T> AsyncStorage for StorageAdapter<T>
    where
        T: Storage + Send + Sync + 'static,
        T::Entity: Send + Sync,
        T::Document: Send + Sync,
        T::Chunk: Send + Sync,
    {
        type Entity = T::Entity;
        type Document = T::Document;
        type Chunk = T::Chunk;
        type Error = T::Error;

        async fn store_entity(&mut self, entity: Self::Entity) -> Result<String> {
            let mut storage = self.0.lock().await;
            storage.store_entity(entity)
        }

        async fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
            let storage = self.0.lock().await;
            storage.retrieve_entity(id)
        }

        async fn store_document(&mut self, document: Self::Document) -> Result<String> {
            let mut storage = self.0.lock().await;
            storage.store_document(document)
        }

        async fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>> {
            let storage = self.0.lock().await;
            storage.retrieve_document(id)
        }

        async fn store_chunk(&mut self, chunk: Self::Chunk) -> Result<String> {
            let mut storage = self.0.lock().await;
            storage.store_chunk(chunk)
        }

        async fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>> {
            let storage = self.0.lock().await;
            storage.retrieve_chunk(id)
        }

        async fn list_entities(&self) -> Result<Vec<String>> {
            let storage = self.0.lock().await;
            storage.list_entities()
        }

        async fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>> {
            let mut storage = self.0.lock().await;
            storage.store_entities_batch(entities)
        }
    }

    /// Adapter that wraps a sync LanguageModel to implement AsyncLanguageModel
    pub struct LanguageModelAdapter<T>(pub Arc<T>);

    #[cfg_attr(feature = "async-trait", async_trait)]
    impl<T> AsyncLanguageModel for LanguageModelAdapter<T>
    where
        T: LanguageModel + Send + Sync + 'static,
    {
        type Error = T::Error;

        async fn complete(&self, prompt: &str) -> Result<String> {
            self.0.complete(prompt)
        }

        async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
            self.0.complete_with_params(prompt, params)
        }

        async fn is_available(&self) -> bool {
            self.0.is_available()
        }

        async fn model_info(&self) -> ModelInfo {
            self.0.model_info()
        }
    }
}

/// Comprehensive async trait utilities and helpers
pub mod async_utils {
    use super::*;
    use std::time::Duration;

    /// Timeout wrapper for any async operation
    pub async fn with_timeout<F, T>(
        future: F,
        timeout: Duration,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match tokio::time::timeout(timeout, future).await {
            Ok(result) => result,
            Err(_) => Err(crate::core::GraphRAGError::Timeout {
                operation: "async operation".to_string(),
                duration: timeout,
            }),
        }
    }

    /// Retry wrapper for async operations
    pub async fn with_retry<F, T, E>(
        mut operation: F,
        max_retries: usize,
        delay: Duration,
    ) -> std::result::Result<T, E>
    where
        F: FnMut() -> Pin<Box<dyn Future<Output = std::result::Result<T, E>> + Send>>,
        E: std::fmt::Debug,
    {
        let mut attempts = 0;
        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempts += 1;
                    if attempts >= max_retries {
                        return Err(err);
                    }
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Batch processor for concurrent operations with rate limiting
    pub async fn process_batch_with_rate_limit<T, F, R>(
        items: Vec<T>,
        processor: F,
        max_concurrent: usize,
        rate_limit: Option<Duration>,
    ) -> Vec<Result<R>>
    where
        T: Send + 'static,
        F: Fn(T) -> Pin<Box<dyn Future<Output = Result<R>> + Send>> + Send + Sync + 'static,
        R: Send + 'static,
    {
        use futures::stream::{FuturesUnordered, StreamExt};
        use std::sync::Arc;

        let processor = Arc::new(processor);
        let mut futures = FuturesUnordered::new();
        let mut results = Vec::with_capacity(items.len());
        let mut pending = 0;

        for item in items {
            if pending >= max_concurrent {
                if let Some(result) = futures.next().await {
                    results.push(result);
                    pending -= 1;
                }
            }

            let processor_clone = Arc::clone(&processor);
            futures.push(async move {
                if let Some(delay) = rate_limit {
                    tokio::time::sleep(delay).await;
                }
                processor_clone(item).await
            });
            pending += 1;
        }

        while let Some(result) = futures.next().await {
            results.push(result);
        }

        results
    }
}

/// Type aliases for common async trait objects
pub type BoxedAsyncLanguageModel = Box<dyn AsyncLanguageModel<Error = crate::core::GraphRAGError> + Send + Sync>;
pub type BoxedAsyncEmbedder = Box<dyn AsyncEmbedder<Error = crate::core::GraphRAGError> + Send + Sync>;
pub type BoxedAsyncVectorStore = Box<dyn AsyncVectorStore<Error = crate::core::GraphRAGError> + Send + Sync>;
pub type BoxedAsyncRetriever = Box<dyn AsyncRetriever<Query = String, Result = crate::retrieval::SearchResult, Error = crate::core::GraphRAGError> + Send + Sync>;
