//! Integration tests for the modular GraphRAG system
//!
//! These tests verify that different components work together correctly,
//! following the modular architecture principles.

#![cfg(feature = "test-utils")]

use graphrag_rs::{
    core::traits::*,
    core::{ChunkId, Document, DocumentId, Entity, EntityId, TextChunk},
    RegistryBuilder, ServiceRegistry,
};

use graphrag_rs::test_utils::*;

#[cfg(feature = "memory-storage")]
use graphrag_rs::MemoryStorage;

#[test]
fn test_service_registry_integration() {
    // Test that the service registry can manage different implementations
    let mut registry = ServiceRegistry::new();

    // Register mock services
    registry.register(MockStorage::new());
    registry.register(MockEmbedder::new());
    registry.register(MockVectorStore::new());
    registry.register(MockLanguageModel::new());

    // Verify services can be retrieved
    let storage = registry.get::<MockStorage>().unwrap();
    let embedder = registry.get::<MockEmbedder>().unwrap();
    let vector_store = registry.get::<MockVectorStore>().unwrap();
    let llm = registry.get::<MockLanguageModel>().unwrap();

    // Test that services are working
    assert!(embedder.is_ready());
    assert!(llm.is_available());
    assert_eq!(storage.call_count(), 0);
    assert_eq!(vector_store.len(), 0);
}

#[test]
fn test_registry_builder_patterns() {
    // Test the builder pattern for creating registries
    let registry = RegistryBuilder::new()
        .with_service(MockStorage::new())
        .with_service(MockEmbedder::with_dimension(512))
        .with_service(MockVectorStore::new())
        .build();

    let embedder = registry.get::<MockEmbedder>().unwrap();
    assert_eq!(embedder.dimension(), 512);
}

#[cfg(feature = "memory-storage")]
#[test]
fn test_memory_storage_with_test_defaults() {
    // Test creating a registry with test defaults
    let registry = RegistryBuilder::with_test_defaults().build();

    let _storage = registry.get::<MemoryStorage>().unwrap();
    // MemoryStorage should be available
    println!("Successfully retrieved MemoryStorage from test defaults");
}

#[test]
fn test_embedder_vector_store_integration() {
    // Test that embedder and vector store work together
    let embedder = MockEmbedder::new();
    let mut vector_store = MockVectorStore::new();

    let text_samples = [
        "Machine learning is transforming technology.",
        "Artificial intelligence helps solve complex problems.",
        "Data science combines statistics and programming.",
    ];

    // Generate embeddings and add to vector store
    for (i, text) in text_samples.iter().enumerate() {
        let embedding = embedder.embed(text).unwrap();
        let id = format!("doc_{i}");
        vector_store.add_vector(id, embedding, None).unwrap();
    }

    assert_eq!(vector_store.len(), 3);

    // Test search functionality
    let query_embedding = embedder.embed("AI and machine learning").unwrap();
    let results = vector_store.search(&query_embedding, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert!(!results[0].id.is_empty());
}

#[test]
fn test_entity_extraction_storage_pipeline() {
    // Test the complete pipeline: extract entities and store them
    let extractor = MockEntityExtractor::new();
    let mut storage = MockStorage::new();

    let text = "John Smith works at Microsoft Corporation in Seattle Washington.";
    let mock_entities = extractor.extract(text).unwrap();

    // Convert mock entities to actual entities for storage
    let mut stored_ids = Vec::new();
    for (i, mock_entity) in mock_entities.iter().enumerate() {
        let entity = Entity::new(
            EntityId::new(format!("extracted_{i}")),
            mock_entity.name.clone(),
            mock_entity.entity_type.clone(),
            0.8,
        );

        let id = storage.store_entity(entity).unwrap();
        stored_ids.push(id);
    }

    // Verify entities were stored
    assert!(!stored_ids.is_empty());
    for id in stored_ids {
        let retrieved = storage.retrieve_entity(&id).unwrap();
        assert!(retrieved.is_some());
    }
}

#[test]
fn test_language_model_metrics_integration() {
    // Test language model with metrics collection
    let llm = MockLanguageModel::new();
    let metrics = MockMetricsCollector::new();

    assert_eq!(metrics.call_count(), 0);

    // Simulate some operations with metrics
    metrics.counter("llm_requests", 1, None);
    let _completion = llm.complete("Test prompt").unwrap();
    metrics.histogram("llm_response_time", 0.25, None);

    assert!(metrics.call_count() > 0);
    assert_eq!(metrics.get_counter("llm_requests"), 1);
    assert!(!metrics.get_histogram_values("llm_response_time").is_empty());
}

#[test]
fn test_full_document_processing_pipeline() {
    // Test a complete document processing pipeline using mock components
    let mut storage = MockStorage::new();
    let embedder = MockEmbedder::new();
    let mut vector_store = MockVectorStore::new();
    let extractor = MockEntityExtractor::new();

    // Create a test document
    let document = Document::new(
        DocumentId::new("test_doc".to_string()),
        "Test Document".to_string(),
        "This is a test document about John Smith who works at Microsoft.".to_string(),
    );

    // Step 1: Store the document
    let doc_id = storage.store_document(document.clone()).unwrap();
    assert_eq!(doc_id, "test_doc");

    // Step 2: Create chunks from the document
    let chunk = TextChunk::new(
        ChunkId::new("chunk_1".to_string()),
        DocumentId::new(doc_id.clone()),
        document.content.clone(),
        0,
        document.content.len(),
    );

    let chunk_id = storage.store_chunk(chunk.clone()).unwrap();

    // Step 3: Extract entities from the chunk
    let entities = extractor.extract(&chunk.content).unwrap();
    assert!(!entities.is_empty());

    // Step 4: Generate embeddings for the chunk
    let embedding = embedder.embed(&chunk.content).unwrap();

    // Step 5: Store the embedding in vector store
    vector_store
        .add_vector(chunk_id.clone(), embedding, None)
        .unwrap();

    // Step 6: Verify the complete pipeline
    let retrieved_doc = storage.retrieve_document(&doc_id).unwrap().unwrap();
    assert_eq!(retrieved_doc.title, "Test Document");

    let retrieved_chunk = storage.retrieve_chunk(&chunk_id).unwrap().unwrap();
    assert_eq!(retrieved_chunk.content, document.content);

    assert_eq!(vector_store.len(), 1);

    println!("✅ Complete document processing pipeline test passed");
    println!("   - Document stored: {doc_id}");
    println!("   - Chunk stored: {chunk_id}");
    println!("   - Entities extracted: {}", entities.len());
    println!(
        "   - Embedding generated: {} dimensions",
        embedder.dimension()
    );
}

#[test]
fn test_service_swapping() {
    // Test that services can be swapped out easily (dependency injection principle)
    let mut registry = ServiceRegistry::new();

    // Start with mock embedder
    registry.register(MockEmbedder::with_dimension(256));
    let embedder1 = registry.get::<MockEmbedder>().unwrap();
    assert_eq!(embedder1.dimension(), 256);

    // Swap to different dimension embedder
    registry.register(MockEmbedder::with_dimension(512));
    let embedder2 = registry.get::<MockEmbedder>().unwrap();
    assert_eq!(embedder2.dimension(), 512);

    // Test with different language models
    registry.register(MockLanguageModel::new());
    assert!(registry.get::<MockLanguageModel>().unwrap().is_available());

    registry.register(MockLanguageModel::unavailable());
    assert!(!registry.get::<MockLanguageModel>().unwrap().is_available());
}

#[test]
fn test_error_handling_across_components() {
    // Test that errors are properly propagated across component boundaries
    let unavailable_llm = MockLanguageModel::unavailable();

    // This should return an error
    let result = unavailable_llm.complete("test prompt");
    assert!(result.is_err());

    match result {
        Err(graphrag_rs::GraphRAGError::LanguageModel { message }) => {
            assert!(message.contains("not available"));
        }
        _ => panic!("Expected LanguageModel error"),
    }
}

#[test]
fn test_batch_operations_integration() {
    // Test batch operations across multiple components
    let embedder = MockEmbedder::new();
    let mut vector_store = MockVectorStore::new();

    let texts = [
        "First document about machine learning",
        "Second document about artificial intelligence",
        "Third document about data science",
        "Fourth document about neural networks",
        "Fifth document about deep learning",
    ];

    // Batch generate embeddings
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let embeddings = embedder.embed_batch(&text_refs).unwrap();
    assert_eq!(embeddings.len(), texts.len());

    // Batch add to vector store
    let vectors_to_add: Vec<_> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, emb)| (format!("batch_doc_{i}"), emb, None))
        .collect();

    vector_store.add_vectors_batch(vectors_to_add).unwrap();
    assert_eq!(vector_store.len(), 5);

    // Test batch search
    let query_embedding = embedder.embed("machine learning and AI").unwrap();
    let results = vector_store.search(&query_embedding, 3).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_configuration_management_integration() {
    // Test configuration management across the system
    let config_provider = MockConfigProvider::new();

    // Test configuration lifecycle
    let default_config = config_provider.default_config();
    config_provider.validate(&default_config).unwrap();
    config_provider.save(&default_config).unwrap();

    let loaded_config = config_provider.load().unwrap();
    assert_eq!(default_config, loaded_config);

    // Test configuration modifications
    let mut modified_config = loaded_config;
    modified_config.setting1 = "modified_value".to_string();
    modified_config.setting2 = 100;

    config_provider.save(&modified_config).unwrap();
    let reloaded_config = config_provider.load().unwrap();
    assert_eq!(reloaded_config.setting1, "modified_value");
    assert_eq!(reloaded_config.setting2, 100);
}

#[test]
fn test_trait_isolation_principle() {
    // Test that traits can be used independently (Single Responsibility Principle)

    // Each trait should work independently
    let embedder = MockEmbedder::new();
    let embedding = embedder.embed("test").unwrap();
    assert_eq!(embedding.len(), embedder.dimension());

    let mut vector_store = MockVectorStore::new();
    vector_store
        .add_vector("test".to_string(), embedding, None)
        .unwrap();
    assert_eq!(vector_store.len(), 1);

    let mut storage = MockStorage::new();
    let entity = Entity::new(
        EntityId::new("test".to_string()),
        "Test".to_string(),
        "Person".to_string(),
        0.9,
    );
    storage.store_entity(entity).unwrap();
    assert_eq!(storage.call_count(), 1);

    // All components should work without dependencies on each other
}

#[test]
fn test_open_closed_principle() {
    // Test that the system is open for extension but closed for modification

    // We can easily add new implementations without modifying existing code
    let mut registry = ServiceRegistry::new();

    // Add different storage implementations
    registry.register(MockStorage::new());
    #[cfg(feature = "memory-storage")]
    registry.register(MemoryStorage::new());

    // Add different embedder implementations
    registry.register(MockEmbedder::with_dimension(384));
    registry.register(MockEmbedder::with_dimension(512));

    // The registry can handle all implementations through the same interface
    let mock_storage = registry.get::<MockStorage>().unwrap();
    #[cfg(feature = "memory-storage")]
    let _memory_storage = registry.get::<MemoryStorage>().unwrap();

    println!("Successfully demonstrated Open/Closed Principle");
    println!("Mock storage call count: {}", mock_storage.call_count());
    #[cfg(feature = "memory-storage")]
    println!("Memory storage created successfully");
}

#[test]
fn test_liskov_substitution_principle() {
    // Test that implementations can be substituted without breaking functionality

    fn test_storage_behavior<
        S: Storage<Entity = Entity, Document = Document, Chunk = TextChunk>,
    >(
        mut storage: S,
    ) -> Result<(), graphrag_rs::GraphRAGError> {
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        let id = storage.store_entity(entity)?;
        let retrieved = storage.retrieve_entity(&id)?;
        assert!(retrieved.is_some());

        Ok(())
    }

    // Both implementations should behave the same way
    test_storage_behavior(MockStorage::new()).unwrap();
    #[cfg(feature = "memory-storage")]
    test_storage_behavior(MemoryStorage::new()).unwrap();

    println!("✅ Liskov Substitution Principle verified");
}

#[test]
fn test_interface_segregation_principle() {
    // Test that clients don't depend on interfaces they don't use

    // A client that only needs embedding functionality
    fn embedding_only_client<E: Embedder<Error = graphrag_rs::GraphRAGError>>(
        embedder: &E,
        text: &str,
    ) -> Result<Vec<f32>, E::Error> {
        embedder.embed(text)
    }

    // A client that only needs storage functionality
    fn storage_only_client<
        S: Storage<
            Entity = Entity,
            Document = Document,
            Chunk = TextChunk,
            Error = graphrag_rs::GraphRAGError,
        >,
    >(
        storage: &mut S,
        entity: Entity,
    ) -> Result<String, S::Error> {
        storage.store_entity(entity)
    }

    let embedder = MockEmbedder::new();
    let mut storage = MockStorage::new();

    // Clients can use only the interfaces they need
    let _embedding = embedding_only_client(&embedder, "test").unwrap();

    let entity = Entity::new(
        EntityId::new("test".to_string()),
        "Test".to_string(),
        "Person".to_string(),
        0.9,
    );
    let _id = storage_only_client(&mut storage, entity).unwrap();

    println!("✅ Interface Segregation Principle verified");
}

#[test]
fn test_dependency_inversion_principle() {
    // Test that high-level modules don't depend on low-level modules
    // Both depend on abstractions (traits)

    // High-level module that depends on abstractions
    struct DocumentProcessor<S, E>
    where
        S: Storage<Entity = Entity, Document = Document, Chunk = TextChunk>,
        E: Embedder,
    {
        storage: S,
        embedder: E,
    }

    impl<S, E> DocumentProcessor<S, E>
    where
        S: Storage<Entity = Entity, Document = Document, Chunk = TextChunk>,
        E: Embedder,
    {
        fn new(storage: S, embedder: E) -> Self {
            Self { storage, embedder }
        }

        fn process_document(
            &mut self,
            document: Document,
        ) -> Result<String, graphrag_rs::GraphRAGError> {
            // Store document
            let doc_id = self.storage.store_document(document.clone())?;

            // Generate embedding for document content
            let _embedding = self.embedder.embed(&document.content).map_err(|_| {
                graphrag_rs::GraphRAGError::Embedding {
                    message: "Failed to generate embedding".to_string(),
                }
            })?;

            Ok(doc_id)
        }
    }

    // Can use any implementation that satisfies the trait constraints
    let mut processor = DocumentProcessor::new(MockStorage::new(), MockEmbedder::new());

    let document = Document::new(
        DocumentId::new("test".to_string()),
        "Test Document".to_string(),
        "Test content".to_string(),
    );

    let doc_id = processor.process_document(document).unwrap();
    assert_eq!(doc_id, "test");

    println!("✅ Dependency Inversion Principle verified");
}
