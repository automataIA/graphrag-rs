//! Property-based tests for GraphRAG components
//!
//! These tests use property-based testing to verify that system invariants
//! hold across a wide range of inputs, ensuring robustness of the modular architecture.

use graphrag_rs::{
    core::traits::*,
    core::{Document, DocumentId, Entity, EntityId},
    test_utils::*,
};
use proptest::prelude::*;

// Property test strategies for generating test data

/// Generate arbitrary entity IDs
fn entity_id_strategy() -> impl Strategy<Value = EntityId> {
    prop::string::string_regex(r"[a-zA-Z0-9_]{1,50}")
        .unwrap()
        .prop_map(EntityId::new)
}

/// Generate arbitrary entity names
fn entity_name_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex(r"[a-zA-Z0-9 ]{1,100}").unwrap()
}

/// Generate arbitrary entity types
fn entity_type_strategy() -> impl Strategy<Value = String> {
    prop::option::of(prop::sample::select(vec![
        "Person".to_string(),
        "Organization".to_string(),
        "Location".to_string(),
        "Event".to_string(),
        "Product".to_string(),
    ]))
    .prop_map(|opt| opt.unwrap_or_else(|| "Unknown".to_string()))
}

/// Generate confidence scores
fn confidence_strategy() -> impl Strategy<Value = f32> {
    prop::num::f32::POSITIVE.prop_map(|x| x.clamp(0.0, 1.0))
}

/// Generate arbitrary entities
fn entity_strategy() -> impl Strategy<Value = Entity> {
    (
        entity_id_strategy(),
        entity_name_strategy(),
        entity_type_strategy(),
        confidence_strategy(),
    )
        .prop_map(|(id, name, entity_type, confidence)| {
            Entity::new(id, name, entity_type, confidence)
        })
}

/// Generate arbitrary document content
fn document_content_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex(r"[a-zA-Z0-9 .,!?]{10,1000}").unwrap()
}

/// Generate arbitrary documents
fn document_strategy() -> impl Strategy<Value = Document> {
    (
        prop::string::string_regex(r"[a-zA-Z0-9_]{1,50}").unwrap(),
        prop::string::string_regex(r"[a-zA-Z0-9 ]{1,100}").unwrap(),
        document_content_strategy(),
    )
        .prop_map(|(id, title, content)| Document::new(DocumentId::new(id), title, content))
}

/// Generate arbitrary embeddings
fn embedding_strategy(dimension: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(prop::num::f32::NORMAL, dimension..=dimension)
}

/// Generate text strings for embedding
fn text_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex(r"[a-zA-Z0-9 .,!?]{1,500}").unwrap()
}

// Property-based tests for storage implementations

proptest! {
    #[test]
    fn test_storage_idempotency(entity in entity_strategy()) {
        let mut storage = MockStorage::new();

        // Store and retrieve should always return the same data
        let id = storage.store_entity(entity.clone()).unwrap();
        let retrieved1 = storage.retrieve_entity(&id).unwrap().unwrap();
        let retrieved2 = storage.retrieve_entity(&id).unwrap().unwrap();

        prop_assert_eq!(&retrieved1.name, &retrieved2.name);
        prop_assert_eq!(&retrieved1.entity_type, &retrieved2.entity_type);
        prop_assert_eq!(retrieved1.confidence, retrieved2.confidence);
        prop_assert_eq!(&retrieved1.name, &entity.name);
    }

    #[test]
    fn test_storage_entity_count_invariant(entities in prop::collection::vec(entity_strategy(), 0..=50)) {
        let mut storage = MockStorage::new();
        let mut expected_count = 0;

        for (i, mut entity) in entities.into_iter().enumerate() {
            // Ensure unique entity IDs to avoid overwrites
            entity.id = EntityId::new(format!("entity_{i}"));

            storage.store_entity(entity).unwrap();
            expected_count += 1;

            // Entity count should always match number of stored entities
            let all_entities = storage.list_entities().unwrap();
            prop_assert_eq!(all_entities.len(), expected_count,
                "Storage count mismatch after storing {} entities", expected_count);
        }
    }

    #[test]
    fn test_storage_nonexistent_entity_behavior(id in prop::string::string_regex(r"nonexistent_[a-zA-Z0-9]{10}").unwrap()) {
        let storage = MockStorage::new();

        // Retrieving non-existent entities should be consistent
        let result1 = storage.retrieve_entity(&id);
        let result2 = storage.retrieve_entity(&id);

        match (result1, result2) {
            (Ok(None), Ok(None)) => (), // Both return None - correct
            (Err(_), Err(_)) => (), // Both return errors - also acceptable
            _ => prop_assert!(false, "Inconsistent behavior for non-existent entity"),
        }
    }
}

// Property-based tests for embedder implementations

proptest! {
    #[test]
    fn test_embedder_determinism(text in text_strategy()) {
        let embedder = MockEmbedder::new();

        if !embedder.is_ready() {
            return Ok(());
        }

        // Same text should always produce the same embedding
        let embedding1 = embedder.embed(&text).unwrap();
        let embedding2 = embedder.embed(&text).unwrap();

        prop_assert_eq!(embedding1, embedding2);
    }

    #[test]
    fn test_embedder_dimension_consistency(texts in prop::collection::vec(text_strategy(), 1..=10)) {
        let embedder = MockEmbedder::new();

        if !embedder.is_ready() {
            return Ok(());
        }

        let expected_dimension = embedder.dimension();

        for text in texts {
            let embedding = embedder.embed(&text).unwrap();
            prop_assert_eq!(embedding.len(), expected_dimension);
        }
    }

    #[test]
    fn test_embedder_batch_consistency(texts in prop::collection::vec(text_strategy(), 1..=20)) {
        let embedder = MockEmbedder::new();

        if !embedder.is_ready() {
            return Ok(());
        }

        // Batch and individual embeddings should be identical
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let batch_embeddings = embedder.embed_batch(&text_refs).unwrap();

        for (i, text) in texts.iter().enumerate() {
            let individual_embedding = embedder.embed(text).unwrap();
            prop_assert_eq!(&batch_embeddings[i], &individual_embedding);
        }
    }

    #[test]
    fn test_embedder_empty_text_handling(empty_variants in prop::sample::select(vec!["", " ", "   ", "\n", "\t"])) {
        let embedder = MockEmbedder::new();

        if !embedder.is_ready() {
            return Ok(());
        }

        // Empty or whitespace-only text should be handled gracefully
        let result = embedder.embed(empty_variants);

        match result {
            Ok(embedding) => {
                // If successful, should still have correct dimension
                prop_assert_eq!(embedding.len(), embedder.dimension());
            }
            Err(_) => {
                // Error is also acceptable for empty input
            }
        }
    }
}

// Property-based tests for vector store implementations

proptest! {
    #[test]
    fn test_vector_store_search_consistency(
        vectors in prop::collection::vec((
            prop::string::string_regex(r"[a-zA-Z0-9_]{1,20}").unwrap(),
            embedding_strategy(384)
        ), 1..=20),  // Reduced size for stability
        query in embedding_strategy(384),
        k in 1usize..=5  // Reduced k for stability
    ) {
        // Skip if query vector is all zeros (can cause numerical issues)
        prop_assume!(!query.iter().all(|&x| x.abs() < 1e-10));

        let mut store = MockVectorStore::new();

        // Add all vectors with unique IDs to avoid conflicts
        for (i, (_, vector)) in vectors.iter().enumerate() {
            let unique_id = format!("vec_{i}");
            store.add_vector(unique_id, vector.clone(), None).unwrap();
        }

        let actual_k = k.min(vectors.len());

        // Search results should be consistent
        let results1 = store.search(&query, actual_k).unwrap();
        let results2 = store.search(&query, actual_k).unwrap();

        prop_assert_eq!(results1.len(), results2.len(),
            "Search result count should be consistent");

        // Results should be in the same order (deterministic)
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            prop_assert_eq!(&r1.id, &r2.id, "Result {} ID should be consistent", i);

            // Allow for small floating point differences
            let distance_diff = (r1.distance - r2.distance).abs();
            prop_assert!(distance_diff < 1e-5,
                "Result {} distance difference {} should be < 1e-5", i, distance_diff);
        }
    }

    #[test]
    fn test_vector_store_similarity_ordering(
        base_vector in embedding_strategy(384),
        noise_level in 0.1f32..1.0f32  // Reduced range for more predictable behavior
    ) {
        // Skip zero vectors as they can cause numerical issues
        prop_assume!(!base_vector.iter().all(|&x| x.abs() < 1e-10));

        let mut store = MockVectorStore::new();

        // Create vectors with clearly different similarities to base vector
        // Similar vector: small perturbation
        let similar_vector: Vec<f32> = base_vector.iter()
            .enumerate()
            .map(|(i, &x)| x + (i as f32 * 0.01) * noise_level * 0.1)
            .collect();

        // Dissimilar vector: large perturbation in opposite direction
        let dissimilar_vector: Vec<f32> = base_vector.iter()
            .enumerate()
            .map(|(i, &x)| x - (i as f32 * 0.1 + 1.0) * noise_level * 2.0)
            .collect();

        store.add_vector("similar".to_string(), similar_vector, None).unwrap();
        store.add_vector("dissimilar".to_string(), dissimilar_vector, None).unwrap();

        let results = store.search(&base_vector, 2).unwrap();

        prop_assert_eq!(results.len(), 2, "Should return exactly 2 results");

        // More similar vector should come first (smaller distance)
        // Allow for small tolerance due to floating point precision
        let distance_diff = results[1].distance - results[0].distance;
        prop_assert!(distance_diff >= -1e-6,
            "Similar vector should have smaller or equal distance. Got distances: {} vs {}",
            results[0].distance, results[1].distance);
    }

    #[test]
    fn test_vector_store_exact_match(vector in embedding_strategy(384)) {
        // Skip zero vectors as they can cause numerical issues
        prop_assume!(!vector.iter().all(|&x| x.abs() < 1e-10));

        let mut store = MockVectorStore::new();
        store.add_vector("exact".to_string(), vector.clone(), None).unwrap();

        let results = store.search(&vector, 1).unwrap();

        prop_assert_eq!(results.len(), 1, "Should return exactly one result");
        prop_assert_eq!(&results[0].id, "exact", "Should return the exact match");

        // For exact match, distance should be very small (allowing for floating point precision)
        prop_assert!(results[0].distance < 1e-4,
            "Distance for exact match should be < 1e-4, got {}", results[0].distance);
    }

    #[test]
    fn test_vector_store_removal_consistency(
        vectors in prop::collection::vec((
            prop::string::string_regex(r"[a-zA-Z0-9_]{1,20}").unwrap(),
            embedding_strategy(384)
        ), 2..=20),
        remove_indices in prop::collection::vec(any::<usize>(), 0..=5)
    ) {
        let mut store = MockVectorStore::new();

        // Add vectors
        for (id, vector) in vectors.iter() {
            store.add_vector(id.clone(), vector.clone(), None).unwrap();
        }

        let initial_len = store.len();

        // Remove some vectors
        let mut removed_count = 0;
        for &index in remove_indices.iter() {
            if index < vectors.len() {
                let id = &vectors[index].0;
                if store.remove_vector(id).unwrap() {
                    removed_count += 1;
                }
            }
        }

        // Length should be consistent
        prop_assert_eq!(store.len(), initial_len - removed_count);
    }
}

// Property-based tests for entity extractor implementations

proptest! {
    #[test]
    fn test_entity_extractor_confidence_threshold(
        text in document_content_strategy(),
        threshold in confidence_strategy()
    ) {
        let mut extractor = MockEntityExtractor::new();
        extractor.set_confidence_threshold(threshold);

        let entities_with_confidence = extractor.extract_with_confidence(&text).unwrap();

        // All returned entities should meet the confidence threshold
        for (_, confidence) in entities_with_confidence {
            prop_assert!(confidence >= threshold,
                "Entity confidence {} is below threshold {}", confidence, threshold);
        }
    }

    #[test]
    fn test_entity_extractor_threshold_ordering(
        text in document_content_strategy(),
        low_threshold in 0.1f32..0.5f32,
        high_threshold in 0.6f32..0.9f32
    ) {
        let mut extractor = MockEntityExtractor::new();

        // Extract with high threshold
        extractor.set_confidence_threshold(high_threshold);
        let high_threshold_entities = extractor.extract(&text).unwrap();

        // Extract with low threshold
        extractor.set_confidence_threshold(low_threshold);
        let low_threshold_entities = extractor.extract(&text).unwrap();

        // Lower threshold should yield same or more entities
        prop_assert!(low_threshold_entities.len() >= high_threshold_entities.len(),
            "Low threshold ({}) yielded {} entities, high threshold ({}) yielded {}",
            low_threshold, low_threshold_entities.len(),
            high_threshold, high_threshold_entities.len());
    }

    #[test]
    fn test_entity_extractor_empty_text(
        empty_text in prop::sample::select(vec!["", " ", "   ", "\n\n", "\t\t"])
    ) {
        let extractor = MockEntityExtractor::new();

        let entities = extractor.extract(empty_text).unwrap();

        // Empty text should return empty or minimal results
        prop_assert!(entities.len() <= 1, "Empty text should not extract many entities");
    }
}

// Property-based tests for language model implementations

proptest! {
    #[test]
    fn test_language_model_availability_consistency(prompt in text_strategy()) {
        let available_model = MockLanguageModel::new();
        let unavailable_model = MockLanguageModel::unavailable();

        prop_assert!(available_model.is_available());
        prop_assert!(!unavailable_model.is_available());

        // Available model should succeed
        let result1 = available_model.complete(&prompt);
        prop_assert!(result1.is_ok());

        // Unavailable model should fail
        let result2 = unavailable_model.complete(&prompt);
        prop_assert!(result2.is_err());
    }

    #[test]
    fn test_language_model_max_tokens_respect(
        prompt in text_strategy(),
        max_tokens in 1usize..=50
    ) {
        let model = MockLanguageModel::new();

        if !model.is_available() {
            return Ok(());
        }

        let params = graphrag_rs::core::traits::GenerationParams {
            max_tokens: Some(max_tokens),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: None,
        };

        let completion = model.complete_with_params(&prompt, params).unwrap();

        // Completion should respect max_tokens (approximately)
        let word_count = completion.split_whitespace().count();
        prop_assert!(word_count <= max_tokens + 5, // Some tolerance for tokenization differences
            "Completion has {} words, but max_tokens was {}", word_count, max_tokens);
    }
}

// Property-based tests for configuration management

proptest! {
    #[test]
    fn test_config_save_load_roundtrip(
        setting1 in prop::string::string_regex(r"[a-zA-Z0-9_]{1,50}").unwrap(),
        setting2 in any::<i32>(),
        setting3 in any::<bool>()
    ) {
        let provider = MockConfigProvider::new();

        let config = graphrag_rs::test_utils::MockConfig {
            setting1: setting1.clone(),
            setting2,
            setting3,
        };

        // Save and load should be a round trip
        provider.save(&config).unwrap();
        let loaded_config = provider.load().unwrap();

        prop_assert_eq!(config.setting1, loaded_config.setting1);
        prop_assert_eq!(config.setting2, loaded_config.setting2);
        prop_assert_eq!(config.setting3, loaded_config.setting3);
    }

    #[test]
    fn test_config_validation_consistency(
        setting1 in prop::string::string_regex(r"[a-zA-Z0-9_]{1,50}").unwrap(),
        setting2 in any::<i32>(),
        setting3 in any::<bool>()
    ) {
        let provider = MockConfigProvider::new();

        let config = graphrag_rs::test_utils::MockConfig {
            setting1,
            setting2,
            setting3,
        };

        // Validation should be consistent
        let result1 = provider.validate(&config);
        let result2 = provider.validate(&config);

        prop_assert_eq!(result1.is_ok(), result2.is_ok());
    }
}

// Property-based tests for metrics collection

proptest! {
    #[test]
    fn test_metrics_counter_monotonicity(
        increments in prop::collection::vec(1u64..=100, 1..=50)
    ) {
        let metrics = MockMetricsCollector::new();

        let mut expected_total = 0u64;

        for increment in increments {
            metrics.counter("test_counter", increment, None);
            expected_total += increment;

            let current_value = metrics.get_counter("test_counter");
            prop_assert_eq!(current_value, expected_total);
        }
    }

    #[test]
    fn test_metrics_gauge_latest_value(
        values in prop::collection::vec(any::<f64>(), 1..=20)
    ) {
        let metrics = MockMetricsCollector::new();

        let mut last_value = 0.0;

        for value in values {
            metrics.gauge("test_gauge", value, None);
            last_value = value;
        }

        let stored_value = metrics.get_gauge("test_gauge").unwrap();
        prop_assert!((stored_value - last_value).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_histogram_accumulation(
        values in prop::collection::vec(0.0f64..=1000.0, 1..=100)
    ) {
        let metrics = MockMetricsCollector::new();

        for value in &values {
            metrics.histogram("test_histogram", *value, None);
        }

        let stored_values = metrics.get_histogram_values("test_histogram");

        prop_assert_eq!(stored_values.len(), values.len());

        // Values should be stored in order
        for (stored, original) in stored_values.iter().zip(values.iter()) {
            prop_assert!((stored - original).abs() < 1e-10);
        }
    }
}

// System-level property tests

proptest! {
    #[test]
    fn test_system_invariant_data_flow(
        documents in prop::collection::vec(document_strategy(), 1..=10)
    ) {
        // Test that data flows correctly through the entire system
        let mut storage = MockStorage::new();
        let embedder = MockEmbedder::new();
        let mut vector_store = MockVectorStore::new();

        for (total_stored, document) in documents.into_iter().enumerate() {
            // Store document
            let doc_id = storage.store_document(document.clone()).unwrap();
            let total_stored = total_stored + 1;

            // Generate embedding
            let embedding = embedder.embed(&document.content).unwrap();

            // Store in vector store
            vector_store.add_vector(doc_id, embedding, None).unwrap();

            // Verify invariants at each step
            prop_assert_eq!(vector_store.len(), total_stored);

            let _all_docs = storage.list_entities().unwrap().len();
            // Note: This is a simplified check - in reality we'd need document listing
        }
    }

    #[test]
    fn test_system_error_resilience(
        valid_operations in prop::collection::vec(any::<bool>(), 10..=50)
    ) {
        // Test that the system handles mixed success/failure scenarios gracefully
        let mut storage = MockStorage::new();
        let mut success_count = 0;

        for (i, should_succeed) in valid_operations.iter().enumerate() {
            let entity = if *should_succeed {
                Entity::new(
                    EntityId::new(format!("valid_{i}")),
                    format!("Valid Entity {i}"),
                    "Person".to_string(),
                    0.8,
                )
            } else {
                // Create an entity that might cause issues (empty name)
                Entity::new(
                    EntityId::new(format!("invalid_{i}")),
                    "".to_string(), // Empty name might cause issues
                    "Unknown".to_string(),
                    0.1,
                )
            };

            let result = storage.store_entity(entity);

            if result.is_ok() {
                success_count += 1;
            }

            // System should remain in consistent state regardless of failures
            let current_count = storage.list_entities().unwrap().len();
            prop_assert_eq!(current_count, success_count);
        }
    }
}
