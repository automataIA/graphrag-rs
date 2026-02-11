use crate::core::{GraphRAGError, Result};
use crate::vector::store::{SearchResult, VectorStore};
use arrow_array::{FixedSizeListArray, Float32Array, StringArray};
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use lancedb::{connect, Connection, Table};
use std::collections::HashMap;
use std::sync::Arc;

pub struct LanceDBStore {
    uri: String,
    table_name: String,
    dimension: usize,
    connection: Option<Connection>,
    table: Option<Table>,
}

impl LanceDBStore {
    pub fn new(uri: &str, table_name: &str, dimension: usize) -> Self {
        Self {
            uri: uri.to_string(),
            table_name: table_name.to_string(),
            dimension,
            connection: None,
            table: None,
        }
    }

    async fn get_table(&self) -> Result<Table> {
        let conn = connect(&self.uri)
            .execute()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("LanceDB connect error: {}", e),
            })?;

        conn.open_table(&self.table_name)
            .execute()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("LanceDB open table error: {}", e),
            })
    }
}

#[async_trait]
impl VectorStore for LanceDBStore {
    async fn initialize(&self) -> Result<()> {
        // LanceDB lazy creation often happens at write time or explicit create_table
        // For now we just check connection
        let conn = connect(&self.uri)
            .execute()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to connect to LanceDB at {}: {}", self.uri, e),
            })?;

        // Define schema if creating new
        Ok(())
    }

    async fn add_vector(
        &self,
        id: &str,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        self.add_vectors_batch(vec![(id, embedding, metadata)])
            .await
    }

    async fn add_vectors_batch(
        &self,
        vectors: Vec<(&str, Vec<f32>, HashMap<String, String>)>,
    ) -> Result<()> {
        use arrow_array::types::Float32Type;

        if vectors.is_empty() {
            return Ok(());
        }

        let table = self.get_table().await?;

        // Extract IDs and embeddings
        let ids: Vec<&str> = vectors.iter().map(|(id, _, _)| *id).collect();
        let id_array = Arc::new(StringArray::from(ids));

        // Convert embeddings to FixedSizeListArray
        let embeddings: Vec<Option<Vec<Option<f32>>>> = vectors
            .iter()
            .map(|(_, vec, _)| Some(vec.iter().map(|&v| Some(v)).collect()))
            .collect();

        let vector_array = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings,
            self.dimension as i32,
        ));

        // Create schema and RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimension as i32,
                ),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, vector_array])
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to create record batch: {}", e),
            })?;

        // Add to table
        let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

        table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to add vectors: {}", e),
            })?;

        Ok(())
    }

    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float32Type;
        use futures::stream::TryStreamExt;
        use lancedb::query::{ExecutableQuery, QueryBase};

        let table = self.get_table().await?;

        // Perform vector search (limit before nearest_to)
        let results = table
            .query()
            .limit(top_k)
            .nearest_to(query_embedding)
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

            for id in id_array.iter() {
                // Use inverse ranking as score (1.0 for closest, decreasing)
                let score = 1.0 / (search_results.len() as f32 + 1.0);

                search_results.push(SearchResult {
                    id: id.clone(),
                    score,
                    metadata: HashMap::new(), // LanceDB doesn't return metadata by default
                });
            }
        }

        Ok(search_results)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let table = self.get_table().await?;
        table
            .delete(&format!("id = '{}'", id))
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: e.to_string(),
            })?;
        Ok(())
    }
}
