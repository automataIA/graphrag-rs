//! LanceDB vector storage backend for GraphRAG embeddings
//!
//! This module provides vector storage using LanceDB, optimized for
//! similarity search and vector operations.
//!
//! ## Features
//!
//! - Efficient vector storage (Lance columnar format)
//! - Fast similarity search (ANN with IVF/HNSW)
//! - Append-only updates (no full rewrite)
//! - Zero-copy reads (memory-mapped)
//! - Cloud-native (S3/Azure/GCS support)
//!
//! ## Example
//!
//! ```no_run
//! use graphrag_core::persistence::{LanceVectorStore, LanceConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> graphrag_core::Result<()> {
//! let config = LanceConfig::default();
//! let store = LanceVectorStore::new(PathBuf::from("./vectors.lance"), config).await?;
//!
//! // Store embedding
//! let embedding = vec![0.1, 0.2, 0.3];
//! store.store_embedding("entity_id", embedding).await?;
//!
//! // Search similar
//! let query = vec![0.15, 0.25, 0.35];
//! let results = store.search_similar(&query, 10).await?;
//! # Ok(())
//! # }
//! ```

use crate::core::{GraphRAGError, Result};
use std::path::PathBuf;

#[cfg(feature = "lancedb")]
use std::sync::Arc;

#[cfg(feature = "lancedb")]
use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, StringArray, FixedSizeListArray};

#[cfg(feature = "lancedb")]
use arrow_array::types::Float32Type;

#[cfg(feature = "lancedb")]
use arrow_schema::{DataType, Field, Schema, SchemaRef};

#[cfg(feature = "lancedb")]
use lancedb::query::{QueryBase, ExecutableQuery};

/// Configuration for LanceDB vector store
#[derive(Debug, Clone)]
pub struct LanceConfig {
    /// Dimension of vectors
    pub dimension: usize,
    /// Index type (HNSW, IVF, etc.)
    pub index_type: IndexType,
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Vector index types
#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    /// Flat index (brute force, exact)
    Flat,
    /// HNSW index (fast, approximate)
    Hnsw,
    /// IVF index (inverted file)
    Ivf,
}

/// Distance metrics
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    L2,
    /// Cosine similarity
    Cosine,
    /// Dot product
    Dot,
}

impl Default for LanceConfig {
    fn default() -> Self {
        Self {
            dimension: 768, // Default BERT embedding size
            index_type: IndexType::Hnsw,
            distance_metric: DistanceMetric::Cosine,
        }
    }
}

/// LanceDB vector store
pub struct LanceVectorStore {
    /// Path to Lance database
    path: PathBuf,
    /// Configuration
    config: LanceConfig,
    /// LanceDB connection
    #[cfg(feature = "lancedb")]
    connection: lancedb::Connection,
    /// Table reference
    #[cfg(feature = "lancedb")]
    table: lancedb::Table,
}

#[cfg(feature = "lancedb")]
impl std::fmt::Debug for LanceVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceVectorStore")
            .field("path", &self.path)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(not(feature = "lancedb"))]
impl std::fmt::Debug for LanceVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceVectorStore")
            .field("path", &self.path)
            .field("config", &self.config)
            .finish()
    }
}

impl LanceVectorStore {
    /// Create a new LanceDB vector store
    ///
    /// # Arguments
    /// * `path` - Path to Lance database directory
    /// * `config` - Configuration for the vector store
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::persistence::{LanceVectorStore, LanceConfig};
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> graphrag_core::Result<()> {
    /// let config = LanceConfig::default();
    /// let store = LanceVectorStore::new(PathBuf::from("./vectors.lance"), config).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "lancedb")]
    pub async fn new(path: PathBuf, config: LanceConfig) -> Result<Self> {
        // Connect to LanceDB
        let db = lancedb::connect(path.to_str().ok_or_else(|| GraphRAGError::Config {
            message: "Invalid path encoding".to_string(),
        })?)
        .execute()
        .await
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to connect to LanceDB: {}", e),
        })?;

        // Create schema for embeddings table
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    config.dimension as i32,
                ),
                false,
            ),
        ]));

        // Try to open existing table or create new one
        let table = match db.open_table("embeddings").execute().await {
            Ok(table) => {
                #[cfg(feature = "tracing")]
                tracing::info!("Opened existing LanceDB table at: {:?}", path);
                table
            }
            Err(_) => {
                // Table doesn't exist, create it with empty data
                let empty_batches = create_empty_batch(schema.clone())?;

                db.create_table("embeddings", empty_batches)
                    .execute()
                    .await
                    .map_err(|e| GraphRAGError::Config {
                        message: format!("Failed to create LanceDB table: {}", e),
                    })?
            }
        };

        #[cfg(feature = "tracing")]
        tracing::info!("LanceDB vector store initialized at: {:?}", path);

        Ok(Self {
            path,
            config,
            connection: db,
            table,
        })
    }

    /// Store an embedding
    #[cfg(feature = "lancedb")]
    pub async fn store_embedding(&self, id: &str, embedding: Vec<f32>) -> Result<()> {
        // Validate embedding dimension
        if embedding.len() != self.config.dimension {
            return Err(GraphRAGError::Config {
                message: format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    embedding.len()
                ),
            });
        }

        // Create Arrow arrays for single embedding
        let id_array = Arc::new(StringArray::from(vec![id]));
        let vector_array = Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vec![Some(embedding.into_iter().map(Some).collect::<Vec<_>>())],
                self.config.dimension as i32,
            ),
        );

        // Create RecordBatch
        let schema = self.table.schema().await.map_err(|e| GraphRAGError::Config {
            message: format!("Failed to get table schema: {}", e),
        })?;

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, vector_array])
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create record batch: {}", e),
            })?;

        // Add to table
        let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

        self.table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to store embedding: {}", e),
            })?;

        #[cfg(feature = "tracing")]
        tracing::debug!("Stored embedding for id: {}", id);

        Ok(())
    }

    /// Search for similar embeddings
    #[cfg(feature = "lancedb")]
    pub async fn search_similar(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        use arrow_array::cast::AsArray;
        use futures::stream::TryStreamExt;

        // Validate query dimension
        if query.len() != self.config.dimension {
            return Err(GraphRAGError::Config {
                message: format!(
                    "Query dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    query.len()
                ),
            });
        }

        // Perform vector search
        // Note: limit() must come before nearest_to() in the query chain
        let results = self
            .table
            .query()
            .limit(k)
            .nearest_to(query)
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to create query: {}", e),
            })?
            .execute()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to execute search: {}", e),
            })?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to collect results: {}", e),
            })?;

        // Convert RecordBatch results to SearchResult
        let mut search_results = Vec::new();

        for batch in results {
            let id_array = batch
                .column(0)
                .as_string::<i32>()
                .iter()
                .map(|s| s.unwrap_or("").to_string())
                .collect::<Vec<_>>();

            let vector_array = batch.column(1).as_fixed_size_list();

            for (idx, id) in id_array.iter().enumerate() {
                // Extract embedding vector
                let embedding = if let Some(values) = vector_array.value(idx).as_primitive_opt::<Float32Type>() {
                    values.values().to_vec()
                } else {
                    vec![0.0; self.config.dimension]
                };

                // Calculate similarity score (distance from query)
                // Note: LanceDB returns results in order, closest first
                // For now, use inverse ranking as score (1.0 for closest, decreasing)
                let score = 1.0 / (search_results.len() as f32 + 1.0);

                search_results.push(SearchResult {
                    id: id.clone(),
                    score,
                    embedding,
                });
            }
        }

        Ok(search_results)
    }

    /// Batch store embeddings
    #[cfg(feature = "lancedb")]
    pub async fn store_embeddings_batch(
        &self,
        embeddings: Vec<(String, Vec<f32>)>,
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // Validate all embedding dimensions
        for (id, embedding) in &embeddings {
            if embedding.len() != self.config.dimension {
                return Err(GraphRAGError::Config {
                    message: format!(
                        "Embedding dimension mismatch for '{}': expected {}, got {}",
                        id,
                        self.config.dimension,
                        embedding.len()
                    ),
                });
            }
        }

        // Create Arrow arrays for batch
        let ids: Vec<&str> = embeddings.iter().map(|(id, _)| id.as_str()).collect();
        let id_array = Arc::new(StringArray::from(ids));

        let vectors: Vec<Option<Vec<Option<f32>>>> = embeddings
            .iter()
            .map(|(_, vec)| Some(vec.iter().map(|&v| Some(v)).collect()))
            .collect();

        let vector_array = Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vectors,
                self.config.dimension as i32,
            ),
        );

        // Create RecordBatch
        let schema = self.table.schema().await.map_err(|e| GraphRAGError::Config {
            message: format!("Failed to get table schema: {}", e),
        })?;

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, vector_array])
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create record batch: {}", e),
            })?;

        // Add to table
        let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

        self.table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to store embeddings batch: {}", e),
            })?;

        #[cfg(feature = "tracing")]
        tracing::debug!("Stored {} embeddings in batch", embeddings.len());

        Ok(())
    }

    /// Get embedding by ID
    #[cfg(feature = "lancedb")]
    pub async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>> {
        use arrow_array::cast::AsArray;
        use futures::stream::TryStreamExt;

        // Query table by ID using SQL filter
        let results = self
            .table
            .query()
            .only_if(format!("id = '{}'", id))
            .execute()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to query by ID: {}", e),
            })?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to collect results: {}", e),
            })?;

        // Extract embedding from first result
        for batch in results {
            if batch.num_rows() == 0 {
                continue;
            }

            let vector_array = batch.column(1).as_fixed_size_list();

            if let Some(values) = vector_array.value(0).as_primitive_opt::<Float32Type>() {
                return Ok(Some(values.values().to_vec()));
            }
        }

        Ok(None)
    }

    /// Count total embeddings
    #[cfg(feature = "lancedb")]
    pub async fn count(&self) -> Result<usize> {
        self.table
            .count_rows(None)
            .await
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to count rows: {}", e),
            })
    }

    /// Stub when feature is disabled
    #[cfg(not(feature = "lancedb"))]
    pub async fn new(_path: PathBuf, _config: LanceConfig) -> Result<Self> {
        Err(GraphRAGError::Config {
            message: "lancedb feature not enabled".to_string(),
        })
    }
}

/// Search result from vector store
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Entity or chunk ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// Create an empty RecordBatch for table initialization
#[cfg(feature = "lancedb")]
fn create_empty_batch(schema: SchemaRef) -> Result<Box<dyn arrow_array::RecordBatchReader + Send>> {
    // Extract size from FixedSizeList DataType
    let list_size = match schema.field(1).data_type() {
        DataType::FixedSizeList(_, size) => *size,
        _ => {
            return Err(GraphRAGError::Config {
                message: "Expected FixedSizeList data type for vector field".to_string(),
            })
        }
    };

    let empty_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(Vec::<String>::new())),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    std::iter::empty::<Option<Vec<Option<f32>>>>(),
                    list_size,
                ),
            ),
        ],
    )
    .map_err(|e| GraphRAGError::Config {
        message: format!("Failed to create empty batch: {}", e),
    })?;

    let reader = RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema);
    Ok(Box::new(reader))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lance_config_default() {
        let config = LanceConfig::default();
        assert_eq!(config.dimension, 768);
    }
}
