use crate::core::{GraphRAGError, Result};
use crate::vector::store::{SearchResult, VectorStore};
use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, VectorParamsBuilder,
};
use std::collections::HashMap;

/// Qdrant vector store implementation
///
/// Provides integration with Qdrant vector database for storing and searching embeddings.
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
    dimension: usize,
}

impl QdrantStore {
    /// Create a new Qdrant vector store
    ///
    /// # Arguments
    /// * `url` - Qdrant server URL
    /// * `collection_name` - Name of the collection to use
    /// * `dimension` - Dimensionality of the vectors
    /// * `api_key` - Optional API key for authentication
    pub fn new(
        url: &str,
        collection_name: &str,
        dimension: usize,
        api_key: Option<String>,
    ) -> Result<Self> {
        let mut builder = Qdrant::from_url(url);

        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }

        let client = builder.build().map_err(|e| GraphRAGError::VectorSearch {
            message: format!("Qdrant client init failed: {}", e),
        })?;

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
            dimension,
        })
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn initialize(&self) -> Result<()> {
        if !self
            .client
            .collection_exists(&self.collection_name)
            .await
            .unwrap_or(false)
        {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection_name)
                        .vectors_config(VectorParamsBuilder::new(
                            self.dimension as u64,
                            Distance::Cosine,
                        )),
                )
                .await
                .map_err(|e| GraphRAGError::VectorSearch {
                    message: format!("Failed to create Qdrant collection: {}", e),
                })?;
        }
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
        use qdrant_client::qdrant::Value;

        let points: Vec<PointStruct> = vectors
            .into_iter()
            .map(|(id, emb, meta)| {
                // Convert HashMap<String, String> to HashMap<String, Value>
                let payload: HashMap<String, Value> = meta
                    .into_iter()
                    .map(|(k, v)| (k, Value::from(v)))
                    .collect();

                PointStruct::new(id.to_string(), emb, payload)
            })
            .collect();

        self.client
            .upsert_points(
                qdrant_client::qdrant::UpsertPointsBuilder::new(&self.collection_name, points)
            )
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Qdrant upsert failed: {}", e),
            })?;

        Ok(())
    }

    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.collection_name, query_embedding.to_vec(), top_k as u64)
                    .with_payload(true)
            )
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Qdrant search failed: {}", e),
            })?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| {
                let metadata = point.payload.iter()
                .map(|(k, v)| (k.clone(), v.to_string())) // Simplified Payload conversion
                .collect();

                // Handle PointId conversion manually since it might not implement ToString directly or simply
                let id = match point.id {
                    Some(id) => match id.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => {
                            num.to_string()
                        },
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid,
                        None => "unknown".to_string(),
                    },
                    None => "unknown".to_string(),
                };

                SearchResult {
                    id,
                    score: point.score,
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        use qdrant_client::qdrant::DeletePointsBuilder;

        // Try to parse as UUID first, fallback to numeric ID
        let point_id: qdrant_client::qdrant::PointId = id
            .parse::<uuid::Uuid>()
            .map(|uuid| uuid.to_string().into())
            .unwrap_or_else(|_| id.to_string().into());

        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(vec![point_id])
            )
            .await
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Qdrant delete failed: {}", e),
            })?;
        Ok(())
    }
}
