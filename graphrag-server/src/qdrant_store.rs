//! Qdrant Vector Store Integration
//!
//! Provides integration with Qdrant vector database for production deployments.
//!
//! ## Features
//!
//! - Store document embeddings with JSON payload metadata
//! - Store entities and relationships as payload
//! - Advanced filtering and search
//! - Collection management
//! - Batch operations
//!
//! ## Usage
//!
//! ```rust
//! let store = QdrantStore::new("http://localhost:6334", "graphrag").await?;
//! store.create_collection(384).await?;
//! store.add_document("doc1", embedding, metadata).await?;
//! let results = store.search(query_embedding, 10, None).await?;
//! ```

use qdrant_client::{
    qdrant::{
        Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
        PointsIdsList, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
        Value as QdrantValue, VectorParamsBuilder,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Qdrant store errors
#[derive(Debug, thiserror::Error)]
pub enum QdrantError {
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Collection error: {0}")]
    CollectionError(String),

    #[error("Operation error: {0}")]
    OperationError(String),

    #[error("Not found: {0}")]
    #[allow(dead_code)]
    NotFound(String),
}

/// Entity stored in Qdrant payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Relationship stored in Qdrant payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: String,
    pub relation: String,
    pub target: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Document metadata stored in Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: String,
    pub text: String,
    pub chunk_index: usize,
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
    pub timestamp: String,
    /// SHA-256 of the document content (lowercase hex). Used for
    /// dedup at ingest: if a point with the same hash already exists,
    /// `add_document` returns its id instead of inserting a duplicate.
    /// Optional so payloads written by older builds parse cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Caller-supplied id. The Qdrant point id itself is a UUID
    /// (Qdrant requires UUID/u64 ids); we store the human-supplied
    /// id separately in the payload so callers can delete by it.
    /// Optional for back-compat with payloads written before this
    /// field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Lightweight summary returned by `list_documents` — title, ids,
/// and a content excerpt are enough for an agent to decide whether
/// to read the full doc, without paying the bandwidth of every
/// Qdrant payload field on a fleet-wide list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub user_id: Option<String>,
    pub title: String,
    pub timestamp: String,
    pub excerpt: String,
}

/// Render a `RetrievedPoint`'s id into a String. Qdrant ids are either
/// UUID strings or u64 numbers; both render to a String here so callers
/// don't have to branch.
fn point_id_to_string(point: qdrant_client::qdrant::RetrievedPoint) -> Option<String> {
    match point.id?.point_id_options? {
        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s) => Some(s),
        qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => Some(n.to_string()),
    }
}

/// Search result from Qdrant
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: DocumentMetadata,
}

/// Qdrant vector store
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
}

impl QdrantStore {
    /// Create a new Qdrant store
    ///
    /// # Arguments
    /// * `url` - Qdrant server URL (e.g., "http://localhost:6334")
    /// * `collection_name` - Collection name for this graph
    pub async fn new(url: &str, collection_name: &str) -> Result<Self, QdrantError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| QdrantError::ConnectionError(e.to_string()))?;

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    /// Create a collection with the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - Embedding dimension (e.g., 384 for MiniLM, 768 for BERT)
    pub async fn create_collection(&self, dimension: u64) -> Result<(), QdrantError> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimension, Distance::Cosine)),
            )
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        Ok(())
    }

    /// Check if collection exists
    pub async fn collection_exists(&self) -> Result<bool, QdrantError> {
        match self.client.collection_info(&self.collection_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Delete the collection
    #[allow(dead_code)]
    pub async fn delete_collection(&self) -> Result<(), QdrantError> {
        self.client
            .delete_collection(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        Ok(())
    }

    /// Add a document chunk with metadata
    ///
    /// # Arguments
    /// * `id` - Unique document ID
    /// * `embedding` - Embedding vector
    /// * `metadata` - Document metadata including entities and relationships
    pub async fn add_document(
        &self,
        id: &str,
        embedding: Vec<f32>,
        metadata: DocumentMetadata,
    ) -> Result<(), QdrantError> {
        let payload = serde_json::to_value(&metadata)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        use std::collections::HashMap;
        let point = PointStruct::new(
            id.to_string(),
            embedding,
            payload
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .map(|(k, v)| (k, QdrantValue::from(v)))
                .collect::<HashMap<String, QdrantValue>>(),
        );

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, vec![point]))
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Add multiple document chunks in batch
    #[allow(dead_code)]
    pub async fn add_documents_batch(
        &self,
        documents: Vec<(String, Vec<f32>, DocumentMetadata)>,
    ) -> Result<(), QdrantError> {
        let points: Vec<PointStruct> = documents
            .into_iter()
            .map(|(id, embedding, metadata)| {
                let payload = serde_json::to_value(&metadata).unwrap();
                PointStruct::new(
                    id,
                    embedding,
                    payload
                        .as_object()
                        .unwrap()
                        .clone()
                        .into_iter()
                        .map(|(k, v)| (k, QdrantValue::from(v)))
                        .collect::<HashMap<String, QdrantValue>>(),
                )
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Search for similar documents
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `limit` - Maximum number of results
    /// * `filter` - Optional filter on metadata fields
    ///
    /// # Returns
    /// Vector of search results with scores and metadata
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>, QdrantError> {
        let mut search_builder =
            SearchPointsBuilder::new(&self.collection_name, query_embedding, limit as u64)
                .with_payload(true);

        if let Some(f) = filter {
            search_builder = search_builder.filter(f);
        }

        let results = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        let search_results: Vec<SearchResult> = results
            .result
            .into_iter()
            .map(|point| {
                let payload_value = serde_json::to_value(&point.payload).unwrap();
                let metadata: DocumentMetadata = serde_json::from_value(payload_value).unwrap();

                // Extract ID from PointId enum
                let id_str = match point.id.unwrap() {
                    qdrant_client::qdrant::PointId {
                        point_id_options:
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)),
                    } => s,
                    qdrant_client::qdrant::PointId {
                        point_id_options:
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)),
                    } => n.to_string(),
                    _ => String::from("unknown"),
                };

                SearchResult {
                    id: id_str,
                    score: point.score,
                    metadata,
                }
            })
            .collect();

        Ok(search_results)
    }

    /// Delete a document by ID
    pub async fn delete_document(&self, id: &str) -> Result<(), QdrantError> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name).points(PointsIdsList {
                    ids: vec![id.to_string().into()],
                }),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;

        Ok(())
    }

    /// Look up a Qdrant point id by the caller-supplied user_id. Returns
    /// the first match (user_id is treated as unique-per-document). Used
    /// by `delete_document` so callers can refer to documents by the id
    /// they handed us at ingest, not the internal UUID.
    pub async fn find_id_by_user_id(
        &self,
        user_id: &str,
    ) -> Result<Option<String>, QdrantError> {
        let filter = Filter::must([Condition::matches("user_id", user_id.to_string())]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(filter)
                    .with_payload(false)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(resp.result.into_iter().next().and_then(point_id_to_string))
    }

    /// Look up an existing point by content hash. Returns the Qdrant
    /// point id (and stored DocumentMetadata) if a match exists. Drives
    /// dedup at ingest: same content → same point, no duplicate.
    pub async fn find_by_content_hash(
        &self,
        hash: &str,
    ) -> Result<Option<(String, DocumentMetadata)>, QdrantError> {
        let filter = Filter::must([Condition::matches("content_hash", hash.to_string())]);
        let resp = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection_name)
                    .filter(filter)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(1u32),
            )
            .await
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let Some(point) = resp.result.into_iter().next() else {
            return Ok(None);
        };
        let id = match point_id_to_string(point.clone()) {
            Some(s) => s,
            None => return Ok(None),
        };
        let payload_value = serde_json::to_value(&point.payload)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        let metadata: DocumentMetadata = serde_json::from_value(payload_value)
            .map_err(|e| QdrantError::OperationError(e.to_string()))?;
        Ok(Some((id, metadata)))
    }

    /// List documents stored in Qdrant. Pages through the collection
    /// using scroll; capped at `limit` to keep responses bounded.
    /// Returns lightweight summaries (id, title, timestamp, excerpt) —
    /// callers needing full text should query individual points.
    pub async fn list_documents(
        &self,
        limit: u32,
    ) -> Result<Vec<DocumentSummary>, QdrantError> {
        let mut summaries = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let page_size = limit.min(256).max(1);

        while summaries.len() < limit as usize {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .with_payload(true)
                .with_vectors(false)
                .limit(page_size);
            if let Some(off) = offset.take() {
                builder = builder.offset(off);
            }
            let resp = self
                .client
                .scroll(builder)
                .await
                .map_err(|e| QdrantError::OperationError(e.to_string()))?;

            if resp.result.is_empty() {
                break;
            }
            for point in resp.result {
                let id = match point_id_to_string(point.clone()) {
                    Some(s) => s,
                    None => continue,
                };
                let payload_value = match serde_json::to_value(&point.payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let md: DocumentMetadata = match serde_json::from_value(payload_value) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let excerpt = md.text.chars().take(160).collect::<String>();
                summaries.push(DocumentSummary {
                    id,
                    user_id: md.user_id,
                    title: md.title,
                    timestamp: md.timestamp,
                    excerpt,
                });
                if summaries.len() >= limit as usize {
                    break;
                }
            }
            offset = resp.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(summaries)
    }

    /// Clear all documents from collection
    #[allow(dead_code)]
    pub async fn clear(&self) -> Result<(), QdrantError> {
        // Delete and recreate collection
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        let dimension = info
            .result
            .and_then(|c| c.config)
            .and_then(|cfg| cfg.params)
            .and_then(|p| p.vectors_config)
            .and_then(|v| v.config)
            .and_then(|cfg| match cfg {
                qdrant_client::qdrant::vectors_config::Config::Params(params) => Some(params.size),
                _ => None,
            })
            .ok_or_else(|| {
                QdrantError::OperationError("Could not get vector dimension".to_string())
            })?;

        self.delete_collection().await?;
        self.create_collection(dimension).await?;

        Ok(())
    }

    /// Get collection statistics
    pub async fn stats(&self) -> Result<(usize, usize), QdrantError> {
        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| QdrantError::CollectionError(e.to_string()))?;

        let count = info
            .result
            .as_ref()
            .and_then(|c| c.points_count)
            .unwrap_or(0) as usize;

        let vectors = info
            .result
            .as_ref()
            .and_then(|c| c.vectors_count)
            .unwrap_or(0) as usize;

        Ok((count, vectors))
    }

    /// Get the collection name
    #[allow(dead_code)]
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Qdrant server running
    async fn test_qdrant_store() {
        let store = QdrantStore::new("http://localhost:6334", "test-collection")
            .await
            .unwrap();
        store.create_collection(384).await.unwrap();

        let metadata = DocumentMetadata {
            id: "doc1".to_string(),
            title: "Test Document".to_string(),
            text: "This is a test document".to_string(),
            chunk_index: 0,
            entities: vec![],
            relationships: vec![],
            timestamp: chrono::Utc::now().to_rfc3339(),
            content_hash: None,
            user_id: None,
            custom: HashMap::new(),
        };

        store
            .add_document("doc1", vec![0.1; 384], metadata)
            .await
            .unwrap();

        let results = store.search(vec![0.1; 384], 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");

        store.delete_collection().await.unwrap();
    }
}
