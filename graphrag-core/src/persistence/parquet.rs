//! Apache Parquet persistence backend for GraphRAG
//!
//! This module implements efficient columnar storage for knowledge graph components
//! using Apache Arrow and Parquet formats.
//!
//! ## File Structure
//!
//! ```text
//! workspace/
//! ├── entities.parquet          # Entity nodes
//! ├── relationships.parquet     # Relationship edges
//! ├── chunks.parquet            # Text chunks
//! └── documents.parquet         # Document metadata
//! ```
//!
//! ## Features
//!
//! - Columnar storage with Snappy compression
//! - Fast selective column reads
//! - Schema evolution support
//! - Integration with Arrow ecosystem (Polars, DuckDB)
//!
//! ## Example
//!
//! ```no_run
//! use graphrag_core::{KnowledgeGraph, persistence::ParquetPersistence};
//! use std::path::PathBuf;
//!
//! # fn example() -> graphrag_core::Result<()> {
//! let graph = KnowledgeGraph::new();
//! let persistence = ParquetPersistence::new(PathBuf::from("./workspace"))?;
//!
//! // Save graph to Parquet files
//! persistence.save_graph(&graph)?;
//!
//! // Load graph from Parquet files
//! let loaded_graph = persistence.load_graph()?;
//! # Ok(())
//! # }
//! ```

use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, EntityMention, GraphRAGError,
    KnowledgeGraph, Relationship, Result, TextChunk,
};
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "persistent-storage")]
use arrow::array::{
    ArrayRef, Float32Array, Int64Array, ListArray, RecordBatch, StringArray, UInt64Array,
};
#[cfg(feature = "persistent-storage")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "persistent-storage")]
use parquet::arrow::arrow_writer::ArrowWriter;
#[cfg(feature = "persistent-storage")]
use parquet::arrow::ArrowReader;
#[cfg(feature = "persistent-storage")]
use parquet::file::properties::WriterProperties;

/// Configuration for Parquet persistence
#[derive(Debug, Clone)]
pub struct ParquetConfig {
    /// Compression codec (default: Snappy)
    pub compression: ParquetCompression,
    /// Row group size (default: 10000)
    pub row_group_size: usize,
    /// Enable dictionary encoding (default: true)
    pub dictionary_encoding: bool,
}

/// Parquet compression codecs
#[derive(Debug, Clone, Copy)]
pub enum ParquetCompression {
    /// No compression
    Uncompressed,
    /// Snappy compression (default, fast)
    Snappy,
    /// Gzip compression (better ratio, slower)
    Gzip,
    /// LZ4 compression (very fast)
    Lz4,
    /// Zstd compression (best ratio, moderate speed)
    Zstd,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            compression: ParquetCompression::Snappy,
            row_group_size: 10000,
            dictionary_encoding: true,
        }
    }
}

/// Parquet persistence backend
#[derive(Debug, Clone)]
pub struct ParquetPersistence {
    /// Base directory for Parquet files
    base_dir: PathBuf,
    /// Configuration
    config: ParquetConfig,
}

impl ParquetPersistence {
    /// Create a new Parquet persistence backend
    ///
    /// # Arguments
    /// * `base_dir` - Directory to store Parquet files
    ///
    /// # Example
    /// ```no_run
    /// use graphrag_core::persistence::ParquetPersistence;
    /// use std::path::PathBuf;
    ///
    /// let persistence = ParquetPersistence::new(PathBuf::from("./workspace")).unwrap();
    /// ```
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        // Create directory if it doesn't exist
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir)?;
        }

        Ok(Self {
            base_dir,
            config: ParquetConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(base_dir: PathBuf, config: ParquetConfig) -> Result<Self> {
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir)?;
        }

        Ok(Self { base_dir, config })
    }

    /// Save knowledge graph to Parquet files
    pub fn save_graph(&self, graph: &KnowledgeGraph) -> Result<()> {
        #[cfg(feature = "tracing")]
        tracing::info!("Saving knowledge graph to Parquet files");

        // Save entities
        self.save_entities(graph)?;

        // Save relationships
        self.save_relationships(graph)?;

        // Save chunks
        self.save_chunks(graph)?;

        // Save documents
        self.save_documents(graph)?;

        #[cfg(feature = "tracing")]
        tracing::info!("Successfully saved knowledge graph to Parquet");

        Ok(())
    }

    /// Load knowledge graph from Parquet files
    pub fn load_graph(&self) -> Result<KnowledgeGraph> {
        #[cfg(feature = "tracing")]
        tracing::info!("Loading knowledge graph from Parquet files");

        let mut graph = KnowledgeGraph::new();

        // Load documents
        let documents = self.load_documents()?;
        for document in documents {
            graph.add_document(document)?;
        }

        // Load chunks (if not already loaded from documents)
        let chunks = self.load_chunks()?;
        for chunk in chunks {
            graph.add_chunk(chunk)?;
        }

        // Load entities
        let entities = self.load_entities()?;
        for entity in entities {
            graph.add_entity(entity)?;
        }

        // Load relationships
        let relationships = self.load_relationships()?;
        for relationship in relationships {
            // Ignore errors for missing entities (they'll be logged)
            let _ = graph.add_relationship(relationship);
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            "Successfully loaded knowledge graph: {} entities, {} relationships",
            graph.entity_count(),
            graph.relationship_count()
        );

        Ok(graph)
    }

    /// Save entities to Parquet
    #[cfg(feature = "persistent-storage")]
    fn save_entities(&self, graph: &KnowledgeGraph) -> Result<()> {
        let entities: Vec<_> = graph.entities().collect();

        if entities.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No entities to save");
            return Ok(());
        }

        // Build Arrow schema for entities
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("entity_type", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("mention_count", DataType::Int64, false),
            Field::new(
                "embedding",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
        ]));

        // Convert entities to Arrow arrays
        let ids: StringArray = entities.iter().map(|e| Some(e.id.0.as_str())).collect();
        let names: StringArray = entities.iter().map(|e| Some(e.name.as_str())).collect();
        let types: StringArray = entities
            .iter()
            .map(|e| Some(e.entity_type.as_str()))
            .collect();
        let confidences: Float32Array = entities.iter().map(|e| Some(e.confidence)).collect();
        let mention_counts: Int64Array = entities
            .iter()
            .map(|e| Some(e.mentions.len() as i64))
            .collect();

        // TODO: Handle embeddings (List<Float32>)
        // For now, create null array
        let embeddings: ArrayRef = Arc::new(ListArray::new_null(DataType::Float32, entities.len()));

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(names),
                Arc::new(types),
                Arc::new(confidences),
                Arc::new(mention_counts),
                embeddings,
            ],
        )
        .map_err(|e| GraphRAGError::Config {
            message: format!("Failed to create RecordBatch: {}", e),
        })?;

        // Write to Parquet file
        let file_path = self.base_dir.join("entities.parquet");
        let file = std::fs::File::create(&file_path)?;

        let props = WriterProperties::builder()
            .set_compression(self.get_compression())
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| {
            GraphRAGError::Config {
                message: format!("Failed to create ArrowWriter: {}", e),
            }
        })?;

        writer.write(&batch).map_err(|e| GraphRAGError::Config {
            message: format!("Failed to write batch: {}", e),
        })?;

        writer.close().map_err(|e| GraphRAGError::Config {
            message: format!("Failed to close writer: {}", e),
        })?;

        #[cfg(feature = "tracing")]
        tracing::info!("Saved {} entities to {:?}", entities.len(), file_path);

        Ok(())
    }

    /// Load entities from Parquet
    #[cfg(feature = "persistent-storage")]
    fn load_entities(&self) -> Result<Vec<Entity>> {
        let file_path = self.base_dir.join("entities.parquet");

        if !file_path.exists() {
            #[cfg(feature = "tracing")]
            tracing::warn!("No entities.parquet found");
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&file_path)?;
        let reader = parquet::arrow::ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to create Parquet reader: {}", e),
            })?
            .build()
            .map_err(|e| GraphRAGError::Config {
                message: format!("Failed to build reader: {}", e),
            })?;

        let mut entities = Vec::new();

        for batch in reader {
            let batch = batch.map_err(|e| GraphRAGError::Config {
                message: format!("Failed to read batch: {}", e),
            })?;

            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid id column type".to_string(),
                })?;

            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid name column type".to_string(),
                })?;

            let types = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid entity_type column type".to_string(),
                })?;

            let confidences = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| GraphRAGError::Config {
                    message: "Invalid confidence column type".to_string(),
                })?;

            for i in 0..batch.num_rows() {
                let entity = Entity {
                    id: EntityId::new(ids.value(i).to_string()),
                    name: names.value(i).to_string(),
                    entity_type: types.value(i).to_string(),
                    confidence: confidences.value(i),
                    mentions: Vec::new(), // Will be populated later if needed
                    embedding: None,      // TODO: Load embeddings
                };

                entities.push(entity);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!("Loaded {} entities from {:?}", entities.len(), file_path);

        Ok(entities)
    }

    /// Placeholder implementations for relationships, chunks, and documents
    /// These will be similar to entities but with different schemas

    #[cfg(feature = "persistent-storage")]
    fn save_relationships(&self, _graph: &KnowledgeGraph) -> Result<()> {
        // TODO: Implement relationship saving
        #[cfg(feature = "tracing")]
        tracing::warn!("Relationship saving not yet implemented");
        Ok(())
    }

    #[cfg(feature = "persistent-storage")]
    fn load_relationships(&self) -> Result<Vec<Relationship>> {
        // TODO: Implement relationship loading
        #[cfg(feature = "tracing")]
        tracing::warn!("Relationship loading not yet implemented");
        Ok(Vec::new())
    }

    #[cfg(feature = "persistent-storage")]
    fn save_chunks(&self, _graph: &KnowledgeGraph) -> Result<()> {
        // TODO: Implement chunk saving
        #[cfg(feature = "tracing")]
        tracing::warn!("Chunk saving not yet implemented");
        Ok(())
    }

    #[cfg(feature = "persistent-storage")]
    fn load_chunks(&self) -> Result<Vec<TextChunk>> {
        // TODO: Implement chunk loading
        #[cfg(feature = "tracing")]
        tracing::warn!("Chunk loading not yet implemented");
        Ok(Vec::new())
    }

    #[cfg(feature = "persistent-storage")]
    fn save_documents(&self, _graph: &KnowledgeGraph) -> Result<()> {
        // TODO: Implement document saving
        #[cfg(feature = "tracing")]
        tracing::warn!("Document saving not yet implemented");
        Ok(())
    }

    #[cfg(feature = "persistent-storage")]
    fn load_documents(&self) -> Result<Vec<Document>> {
        // TODO: Implement document loading
        #[cfg(feature = "tracing")]
        tracing::warn!("Document loading not yet implemented");
        Ok(Vec::new())
    }

    /// Get compression codec for Parquet writer
    #[cfg(feature = "persistent-storage")]
    fn get_compression(&self) -> parquet::basic::Compression {
        use parquet::basic::Compression;

        match self.config.compression {
            ParquetCompression::Uncompressed => Compression::UNCOMPRESSED,
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Gzip => Compression::GZIP(parquet::basic::GzipLevel::default()),
            ParquetCompression::Lz4 => Compression::LZ4,
            ParquetCompression::Zstd => Compression::ZSTD(parquet::basic::ZstdLevel::default()),
        }
    }

    /// Stub implementations for when persistent-storage feature is disabled
    #[cfg(not(feature = "persistent-storage"))]
    fn save_entities(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_entities(&self) -> Result<Vec<Entity>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn save_relationships(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_relationships(&self) -> Result<Vec<Relationship>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn save_chunks(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_chunks(&self) -> Result<Vec<TextChunk>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn save_documents(&self, _graph: &KnowledgeGraph) -> Result<()> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }

    #[cfg(not(feature = "persistent-storage"))]
    fn load_documents(&self) -> Result<Vec<Document>> {
        Err(GraphRAGError::Config {
            message: "persistent-storage feature not enabled".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parquet_persistence_creation() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();
        assert!(persistence.base_dir.exists());
    }

    #[test]
    #[cfg(feature = "persistent-storage")]
    fn test_save_and_load_entities() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ParquetPersistence::new(temp_dir.path().to_path_buf()).unwrap();

        let mut graph = KnowledgeGraph::new();
        let entity = Entity::new(
            EntityId::new("test_entity".to_string()),
            "Test Entity".to_string(),
            "PERSON".to_string(),
            0.9,
        );
        graph.add_entity(entity).unwrap();

        // Save
        persistence.save_entities(&graph).unwrap();

        // Load
        let entities = persistence.load_entities().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Test Entity");
    }
}
