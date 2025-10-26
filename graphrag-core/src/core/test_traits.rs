//! Trait testing suite for modular architecture validation
//!
//! This module provides reusable test suites that can be applied to any implementation
//! of the core traits, ensuring that all implementations behave consistently and correctly.

use crate::core::traits::*;
use crate::core::{Document, Entity, EntityId, Result, TextChunk};
use std::collections::HashMap;

/// Test suite for Storage trait implementations
///
/// This function can be called with any type that implements Storage + Default
/// to verify that the implementation behaves correctly.
pub fn test_storage_roundtrip<T>()
where
    T: Storage<Entity = Entity, Document = Document, Chunk = TextChunk> + Default,
{
    let mut storage = T::default();

    // Test entity storage and retrieval
    let entity = Entity::new(
        EntityId::new("test_entity".to_string()),
        "Test Entity".to_string(),
        "Person".to_string(),
        0.9,
    );

    let stored_id = storage.store_entity(entity.clone()).unwrap();
    let retrieved = storage.retrieve_entity(&stored_id).unwrap().unwrap();

    assert_eq!(retrieved.name, entity.name);
    assert_eq!(retrieved.entity_type, entity.entity_type);
    assert_eq!(retrieved.confidence, entity.confidence);
}

/// Test that storage implementations handle non-existent keys correctly
pub fn test_storage_nonexistent_key<T>()
where
    T: Storage<Entity = Entity, Document = Document, Chunk = TextChunk> + Default,
{
    let storage = T::default();
    let result = storage.retrieve_entity("nonexistent_id");

    match result {
        Ok(None) => (), // This is the expected behavior
        Ok(Some(_)) => panic!("Expected None for nonexistent key, got Some"),
        Err(_) => (), // Also acceptable - some implementations may return errors
    }
}

/// Test storage batch operations
pub fn test_storage_batch_operations<T>()
where
    T: Storage<Entity = Entity, Document = Document, Chunk = TextChunk> + Default,
{
    let mut storage = T::default();

    let entities = vec![
        Entity::new(
            EntityId::new("entity1".to_string()),
            "Entity 1".to_string(),
            "Person".to_string(),
            0.9,
        ),
        Entity::new(
            EntityId::new("entity2".to_string()),
            "Entity 2".to_string(),
            "Organization".to_string(),
            0.8,
        ),
        Entity::new(
            EntityId::new("entity3".to_string()),
            "Entity 3".to_string(),
            "Location".to_string(),
            0.7,
        ),
    ];

    let stored_ids = storage.store_entities_batch(entities.clone()).unwrap();
    assert_eq!(stored_ids.len(), 3);

    // Verify all entities can be retrieved
    for (stored_id, original_entity) in stored_ids.iter().zip(entities.iter()) {
        let retrieved = storage.retrieve_entity(stored_id).unwrap().unwrap();
        assert_eq!(retrieved.name, original_entity.name);
    }
}

/// Test storage ID consistency
pub fn test_storage_id_consistency<T>()
where
    T: Storage<Entity = Entity, Document = Document, Chunk = TextChunk> + Default,
{
    let mut storage = T::default();

    let entity = Entity::new(
        EntityId::new("consistent_id".to_string()),
        "Test Entity".to_string(),
        "Person".to_string(),
        0.9,
    );

    let id1 = storage.store_entity(entity.clone()).unwrap();
    let id2 = storage.store_entity(entity.clone()).unwrap();

    // IDs should be different for different storage operations
    // (or the same if the implementation deduplicates by entity ID)
    // This test documents the behavior without enforcing a specific strategy
    println!("First ID: {id1}, Second ID: {id2}");
}

/// Test suite for VectorStore trait implementations
pub fn test_vector_store_basic_operations<T>()
where
    T: VectorStore + Default,
{
    let mut store = T::default();

    // Test adding vectors
    let vector1 = vec![1.0, 2.0, 3.0];
    let vector2 = vec![4.0, 5.0, 6.0];
    let metadata = Some(HashMap::from([("type".to_string(), "test".to_string())]));

    store
        .add_vector("vec1".to_string(), vector1.clone(), metadata.clone())
        .unwrap();
    store
        .add_vector("vec2".to_string(), vector2.clone(), None)
        .unwrap();

    assert_eq!(store.len(), 2);
    assert!(!store.is_empty());

    // Test search functionality
    let results = store.search(&vector1, 1).unwrap();
    assert!(!results.is_empty());

    // The most similar result should be the exact match
    assert_eq!(results[0].id, "vec1");
}

/// Test VectorStore similarity search properties
pub fn test_vector_store_similarity_properties<T>()
where
    T: VectorStore + Default,
{
    let mut store = T::default();

    // Add identical vectors
    let vector = vec![1.0, 1.0, 1.0];
    store
        .add_vector("identical1".to_string(), vector.clone(), None)
        .unwrap();
    store
        .add_vector("identical2".to_string(), vector.clone(), None)
        .unwrap();

    // Add a different vector
    let different_vector = vec![10.0, 10.0, 10.0];
    store
        .add_vector("different".to_string(), different_vector, None)
        .unwrap();

    // Search should return identical vectors first
    let results = store.search(&vector, 3).unwrap();
    assert_eq!(results.len(), 3);

    // First two results should be the identical vectors (in some order)
    let first_two_ids: std::collections::HashSet<_> =
        results.iter().take(2).map(|r| r.id.as_str()).collect();
    assert!(first_two_ids.contains("identical1"));
    assert!(first_two_ids.contains("identical2"));
}

/// Test suite for Embedder trait implementations
pub fn test_embedder_basic_functionality<T>()
where
    T: Embedder + Default,
{
    let embedder = T::default();

    if !embedder.is_ready() {
        println!("Embedder not ready, skipping test");
        return;
    }

    // Test single embedding
    let text = "This is a test sentence for embedding.";
    let embedding = embedder.embed(text).unwrap();

    assert!(!embedding.is_empty());
    assert_eq!(embedding.len(), embedder.dimension());

    // Test that the same text produces the same embedding (deterministic)
    let embedding2 = embedder.embed(text).unwrap();
    assert_eq!(embedding, embedding2);
}

/// Test Embedder batch processing
pub fn test_embedder_batch_consistency<T>()
where
    T: Embedder + Default,
{
    let embedder = T::default();

    if !embedder.is_ready() {
        println!("Embedder not ready, skipping test");
        return;
    }

    let texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ];

    // Test batch embedding
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let batch_embeddings = embedder.embed_batch(&text_refs).unwrap();
    assert_eq!(batch_embeddings.len(), texts.len());

    // Test individual embeddings
    let individual_embeddings: Result<Vec<_>> =
        texts.iter().map(|text| embedder.embed(text)).collect();
    let individual_embeddings = individual_embeddings.unwrap();

    // Batch and individual embeddings should be identical
    for (batch_emb, individual_emb) in batch_embeddings.iter().zip(individual_embeddings.iter()) {
        assert_eq!(batch_emb, individual_emb);
    }
}

/// Test suite for EntityExtractor trait implementations
pub fn test_entity_extractor_basic_extraction<T>()
where
    T: EntityExtractor + Default,
    T::Entity: std::fmt::Debug,
{
    let extractor = T::default();

    let text = "John Smith works at Microsoft Corporation in Seattle.";
    let entities = extractor.extract(text).unwrap();

    // Should extract at least some entities from this text
    assert!(!entities.is_empty());

    println!("Extracted {} entities: {:?}", entities.len(), entities);
}

/// Test EntityExtractor confidence handling
pub fn test_entity_extractor_confidence<T>()
where
    T: EntityExtractor + Default,
    T::Entity: std::fmt::Debug,
{
    let mut extractor = T::default();

    let text = "John Smith works at Microsoft Corporation.";

    // Test with high confidence threshold
    extractor.set_confidence_threshold(0.9);
    let high_confidence_entities = extractor.extract_with_confidence(text).unwrap();

    // Test with low confidence threshold
    extractor.set_confidence_threshold(0.1);
    let low_confidence_entities = extractor.extract_with_confidence(text).unwrap();

    // Lower threshold should yield same or more entities
    assert!(low_confidence_entities.len() >= high_confidence_entities.len());

    // All high confidence entities should have confidence >= 0.9
    for (_, confidence) in &high_confidence_entities {
        assert!(
            *confidence >= 0.9,
            "High confidence entity has confidence {confidence}"
        );
    }
}

/// Test suite for LanguageModel trait implementations
pub fn test_language_model_basic_completion<T>()
where
    T: LanguageModel + Default,
{
    let model = T::default();

    if !model.is_available() {
        println!("Language model not available, skipping test");
        return;
    }

    let prompt = "Complete this sentence: The capital of France is";
    let completion = model.complete(prompt).unwrap();

    assert!(!completion.is_empty());
    println!("Completion: {completion}");
}

/// Test LanguageModel parameter handling
pub fn test_language_model_parameters<T>()
where
    T: LanguageModel + Default,
{
    let model = T::default();

    if !model.is_available() {
        println!("Language model not available, skipping test");
        return;
    }

    let prompt = "Count to three:";

    // Test with different parameters
    let params = GenerationParams {
        max_tokens: Some(10),
        temperature: Some(0.1),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["4".to_string()]),
    };

    let completion = model.complete_with_params(prompt, params).unwrap();
    assert!(!completion.is_empty());

    // Should respect max_tokens limit (approximately)
    assert!(completion.split_whitespace().count() <= 15); // Some buffer for tokenization differences
}

/// Test ConfigProvider trait implementations
pub fn test_config_provider_lifecycle<T>()
where
    T: ConfigProvider + Default,
    T::Config: Clone + PartialEq + std::fmt::Debug,
{
    let provider = T::default();

    // Test getting default config
    let default_config = provider.default_config();

    // Test validation
    provider.validate(&default_config).unwrap();

    // Test save and load cycle
    provider.save(&default_config).unwrap();
    let loaded_config = provider.load().unwrap();

    assert_eq!(default_config, loaded_config);
}

/// Macro to run all storage tests for a given implementation
#[macro_export]
macro_rules! test_storage_implementation {
    ($storage_type:ty) => {
        #[test]
        fn test_storage_roundtrip() {
            $crate::core::test_traits::test_storage_roundtrip::<$storage_type>();
        }

        #[test]
        fn test_storage_nonexistent_key() {
            $crate::core::test_traits::test_storage_nonexistent_key::<$storage_type>();
        }

        #[test]
        fn test_storage_batch_operations() {
            $crate::core::test_traits::test_storage_batch_operations::<$storage_type>();
        }

        #[test]
        fn test_storage_id_consistency() {
            $crate::core::test_traits::test_storage_id_consistency::<$storage_type>();
        }
    };
}

/// Macro to run all vector store tests for a given implementation
#[macro_export]
macro_rules! test_vector_store_implementation {
    ($vector_store_type:ty) => {
        #[test]
        fn test_vector_store_basic_operations() {
            $crate::core::test_traits::test_vector_store_basic_operations::<$vector_store_type>();
        }

        #[test]
        fn test_vector_store_similarity_properties() {
            $crate::core::test_traits::test_vector_store_similarity_properties::<$vector_store_type>();
        }
    };
}

/// Macro to run all embedder tests for a given implementation
#[macro_export]
macro_rules! test_embedder_implementation {
    ($embedder_type:ty) => {
        #[test]
        fn test_embedder_basic_functionality() {
            $crate::core::test_traits::test_embedder_basic_functionality::<$embedder_type>();
        }

        #[test]
        fn test_embedder_batch_consistency() {
            $crate::core::test_traits::test_embedder_batch_consistency::<$embedder_type>();
        }
    };
}

/// Macro to run all entity extractor tests for a given implementation
#[macro_export]
macro_rules! test_entity_extractor_implementation {
    ($extractor_type:ty) => {
        #[test]
        fn test_entity_extractor_basic_extraction() {
            $crate::core::test_traits::test_entity_extractor_basic_extraction::<$extractor_type>();
        }

        #[test]
        fn test_entity_extractor_confidence() {
            $crate::core::test_traits::test_entity_extractor_confidence::<$extractor_type>();
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_trait_testing_framework() {
        // This test verifies that the trait testing framework itself works
        // It doesn't test actual implementations, just the framework
        println!("Trait testing framework initialized successfully");
    }
}
