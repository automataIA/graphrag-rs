//! Entity extraction adapters for core traits
//!
//! This module provides adapter implementations that bridge existing entity extractors
//! with the core GraphRAG AsyncEntityExtractor trait.

#[cfg(feature = "lightrag")]
use crate::core::error::{GraphRAGError, Result};
#[cfg(feature = "lightrag")]
use crate::core::traits::AsyncEntityExtractor;
#[cfg(feature = "lightrag")]
use crate::core::Entity;
#[cfg(feature = "lightrag")]
use async_trait::async_trait;

#[cfg(feature = "lightrag")]
use crate::lightrag::graph_indexer::{GraphIndexer, ExtractedEntity};

/// Adapter for GraphIndexer to implement AsyncEntityExtractor trait
#[cfg(feature = "lightrag")]
pub struct GraphIndexerAdapter {
    indexer: GraphIndexer,
    confidence_threshold: f32,
}

#[cfg(feature = "lightrag")]
impl GraphIndexerAdapter {
    /// Create a new GraphIndexer adapter
    pub fn new(entity_types: Vec<String>, max_depth: usize) -> Result<Self> {
        Ok(Self {
            indexer: GraphIndexer::new(entity_types, max_depth)?,
            confidence_threshold: 0.5,
        })
    }

    /// Create with custom confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Convert ExtractedEntity to core::Entity
    fn convert_entity(&self, extracted: &ExtractedEntity) -> Entity {
        use crate::core::EntityId;
        Entity {
            id: EntityId::new(extracted.id.clone()),
            name: extracted.name.clone(),
            entity_type: extracted.entity_type.clone(),
            confidence: extracted.confidence,
            mentions: vec![], // GraphIndexer doesn't track mentions
            embedding: None,  // No embedding in GraphIndexer
            first_mentioned: None,
            last_mentioned: None,
            temporal_validity: None,
        }
    }
}

#[cfg(feature = "lightrag")]
#[async_trait]
impl AsyncEntityExtractor for GraphIndexerAdapter {
    type Entity = Entity;
    type Error = GraphRAGError;

    async fn extract(&self, text: &str) -> Result<Vec<Self::Entity>> {
        let result = self.indexer.extract_from_text(text)?;

        // Filter by confidence threshold and convert
        Ok(result
            .entities
            .iter()
            .filter(|e| e.confidence >= self.confidence_threshold)
            .map(|e| self.convert_entity(e))
            .collect())
    }

    async fn extract_with_confidence(&self, text: &str) -> Result<Vec<(Self::Entity, f32)>> {
        let result = self.indexer.extract_from_text(text)?;

        // Filter by confidence threshold and convert
        Ok(result
            .entities
            .iter()
            .filter(|e| e.confidence >= self.confidence_threshold)
            .map(|e| (self.convert_entity(e), e.confidence))
            .collect())
    }

    async fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Vec<Self::Entity>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.extract(text).await?);
        }
        Ok(results)
    }

    async fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    async fn get_confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

#[cfg(all(test, feature = "lightrag"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graph_indexer_adapter() {
        let adapter = GraphIndexerAdapter::new(
            vec!["person".to_string(), "organization".to_string()],
            3,
        )
        .unwrap();

        let text = "John Smith works at Microsoft Corporation.";
        let entities = adapter.extract(text).await.unwrap();

        assert!(!entities.is_empty());
        for entity in &entities {
            assert!(entity.confidence >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_confidence_threshold_filtering() {
        let adapter = GraphIndexerAdapter::new(
            vec!["person".to_string()],
            3,
        )
        .unwrap()
        .with_confidence_threshold(0.6);

        let text = "John Smith works at Microsoft.";
        let entities = adapter.extract(text).await.unwrap();

        // All entities should have confidence >= 0.6
        for entity in &entities {
            assert!(entity.confidence >= 0.6);
        }
    }

    #[tokio::test]
    async fn test_batch_extraction() {
        let adapter = GraphIndexerAdapter::new(
            vec!["person".to_string(), "location".to_string()],
            3,
        )
        .unwrap();

        let texts = vec![
            "Alice lives in Paris.",
            "Bob works in London.",
        ];

        let results = adapter.extract_batch(&texts).await.unwrap();
        assert_eq!(results.len(), 2);
    }
}
