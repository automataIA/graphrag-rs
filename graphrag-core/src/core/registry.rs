//! Service registry for dependency injection
//!
//! This module provides a dependency injection system that allows
//! components to be swapped out for testing or different implementations.

use crate::core::traits::*;
use crate::core::{GraphRAGError, Result};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Type-erased service container
type ServiceBox = Box<dyn Any + Send + Sync>;

/// Service registry for dependency injection
pub struct ServiceRegistry {
    services: HashMap<TypeId, ServiceBox>,
}

impl ServiceRegistry {
    /// Create a new empty service registry
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
        }
    }

    /// Register a service implementation
    pub fn register<T: Any + Send + Sync>(&mut self, service: T) {
        let type_id = TypeId::of::<T>();
        self.services.insert(type_id, Box::new(service));
    }

    /// Get a service by type
    pub fn get<T: Any + Send + Sync>(&self) -> Result<&T> {
        let type_id = TypeId::of::<T>();

        self.services
            .get(&type_id)
            .and_then(|service| service.downcast_ref::<T>())
            .ok_or_else(|| GraphRAGError::Config {
                message: format!("Service not registered: {}", std::any::type_name::<T>()),
            })
    }

    /// Get a mutable service by type
    pub fn get_mut<T: Any + Send + Sync>(&mut self) -> Result<&mut T> {
        let type_id = TypeId::of::<T>();

        self.services
            .get_mut(&type_id)
            .and_then(|service| service.downcast_mut::<T>())
            .ok_or_else(|| GraphRAGError::Config {
                message: format!("Service not registered: {}", std::any::type_name::<T>()),
            })
    }

    /// Check if a service is registered
    pub fn has<T: Any + Send + Sync>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        self.services.contains_key(&type_id)
    }

    /// Remove a service
    pub fn remove<T: Any + Send + Sync>(&mut self) -> Option<T> {
        let type_id = TypeId::of::<T>();

        self.services
            .remove(&type_id)
            .and_then(|service| service.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Get the number of registered services
    pub fn len(&self) -> usize {
        self.services.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }

    /// Clear all services
    pub fn clear(&mut self) {
        self.services.clear();
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating and configuring service registries
pub struct RegistryBuilder {
    registry: ServiceRegistry,
}

impl RegistryBuilder {
    /// Create a new registry builder
    pub fn new() -> Self {
        Self {
            registry: ServiceRegistry::new(),
        }
    }

    /// Register a service and continue building
    pub fn with_service<T: Any + Send + Sync>(mut self, service: T) -> Self {
        self.registry.register(service);
        self
    }

    /// Register a storage implementation
    pub fn with_storage<S>(mut self, storage: S) -> Self
    where
        S: Storage + Any + Send + Sync,
    {
        self.registry.register(storage);
        self
    }

    /// Register an embedder implementation
    pub fn with_embedder<E>(mut self, embedder: E) -> Self
    where
        E: Embedder + Any + Send + Sync,
    {
        self.registry.register(embedder);
        self
    }

    /// Register a vector store implementation
    pub fn with_vector_store<V>(mut self, vector_store: V) -> Self
    where
        V: VectorStore + Any + Send + Sync,
    {
        self.registry.register(vector_store);
        self
    }

    /// Register an entity extractor implementation
    pub fn with_entity_extractor<E>(mut self, extractor: E) -> Self
    where
        E: EntityExtractor + Any + Send + Sync,
    {
        self.registry.register(extractor);
        self
    }

    /// Register a retriever implementation
    pub fn with_retriever<R>(mut self, retriever: R) -> Self
    where
        R: Retriever + Any + Send + Sync,
    {
        self.registry.register(retriever);
        self
    }

    /// Register a language model implementation
    pub fn with_language_model<L>(mut self, language_model: L) -> Self
    where
        L: LanguageModel + Any + Send + Sync,
    {
        self.registry.register(language_model);
        self
    }

    /// Register a graph store implementation
    pub fn with_graph_store<G>(mut self, graph_store: G) -> Self
    where
        G: GraphStore + Any + Send + Sync,
    {
        self.registry.register(graph_store);
        self
    }

    /// Register a function registry implementation
    pub fn with_function_registry<F>(mut self, function_registry: F) -> Self
    where
        F: FunctionRegistry + Any + Send + Sync,
    {
        self.registry.register(function_registry);
        self
    }

    /// Register a metrics collector implementation
    pub fn with_metrics_collector<M>(mut self, metrics: M) -> Self
    where
        M: MetricsCollector + Any + Send + Sync,
    {
        self.registry.register(metrics);
        self
    }

    /// Register a serializer implementation
    pub fn with_serializer<S>(mut self, serializer: S) -> Self
    where
        S: Serializer + Any + Send + Sync,
    {
        self.registry.register(serializer);
        self
    }

    /// Build the final registry
    pub fn build(self) -> ServiceRegistry {
        self.registry
    }

    /// Create a registry with default Ollama-based services
    #[cfg(feature = "ollama")]
    pub fn with_ollama_defaults() -> Self {
        #[cfg(feature = "memory-storage")]
        use crate::storage::MemoryStorage;

        let mut builder = Self::new();

        #[cfg(feature = "memory-storage")]
        {
            builder = builder.with_storage(MemoryStorage::new());
        }

        // Add other service implementations based on available features
        #[cfg(feature = "parallel-processing")]
        {
            use crate::parallel::ParallelProcessor;

            // Auto-detect number of threads (0 means use default)
            let num_threads = num_cpus::get();
            let parallel_processor = ParallelProcessor::new(num_threads);
            builder = builder.with_service(parallel_processor);
        }

        #[cfg(feature = "vector-hnsw")]
        {
            use crate::vector::VectorIndex;
            builder = builder.with_service(VectorIndex::new());
        }

        #[cfg(feature = "caching")]
        {
            // Add caching services when available
            // Note: Specific cache implementations would be added here
        }
        builder
    }

    /// Create a registry with memory-only services for testing
    #[cfg(feature = "memory-storage")]
    pub fn with_test_defaults() -> Self {
        use crate::storage::MemoryStorage;
        // TODO: Add mock implementations when test_utils module is created

        Self::new().with_storage(MemoryStorage::new())
    }
}

impl Default for RegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Context object that provides access to services
#[derive(Clone)]
pub struct ServiceContext {
    registry: Arc<ServiceRegistry>,
}

impl ServiceContext {
    /// Create a new service context
    pub fn new(registry: ServiceRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }

    /// Get a service by type
    pub fn get<T: Any + Send + Sync>(&self) -> Result<&T> {
        // Safety: This is safe because we're getting an immutable reference
        // from an Arc, which ensures the registry stays alive
        unsafe {
            let ptr = self.registry.as_ref() as *const ServiceRegistry;
            (*ptr).get::<T>()
        }
    }
}

/// Configuration for service creation
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Base URL for Ollama API server
    pub ollama_base_url: Option<String>,
    /// Model name for text embeddings
    pub embedding_model: Option<String>,
    /// Model name for text generation
    pub language_model: Option<String>,
    /// Dimensionality of embedding vectors
    pub vector_dimension: Option<usize>,
    /// Minimum confidence threshold for entity extraction
    pub entity_confidence_threshold: Option<f32>,
    /// Enable parallel processing for batch operations
    pub enable_parallel_processing: bool,
    /// Enable function calling capabilities
    pub enable_function_calling: bool,
    /// Enable monitoring and metrics collection
    pub enable_monitoring: bool,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            ollama_base_url: Some("http://localhost:11434".to_string()),
            embedding_model: Some("nomic-embed-text:latest".to_string()),
            language_model: Some("llama3.2:latest".to_string()),
            vector_dimension: Some(384),
            entity_confidence_threshold: Some(0.7),
            enable_parallel_processing: true,
            enable_function_calling: false,
            enable_monitoring: false,
        }
    }
}

impl ServiceConfig {
    /// Create a registry builder from this configuration
    ///
    /// This method creates service instances based on the configuration and available features.
    /// Services are registered in the following order:
    ///
    /// 1. Storage (MemoryStorage with memory-storage feature)
    /// 2. Vector Store (when vector storage implementations are available)
    /// 3. Embedder (when embedding providers are available)
    /// 4. Entity Extractor (when NER models are available)
    /// 5. Retriever (when retrieval systems are implemented)
    /// 6. Language Model (when LLM clients are available)
    /// 7. Metrics Collector (when monitoring is enabled)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use graphrag_core::core::registry::ServiceConfig;
    ///
    /// let config = ServiceConfig::default();
    /// let registry = config.build_registry().build();
    /// ```
    pub fn build_registry(&self) -> RegistryBuilder {
        let mut builder = RegistryBuilder::new();

        // 1. Storage Layer
        #[cfg(feature = "memory-storage")]
        {
            use crate::storage::MemoryStorage;
            builder = builder.with_storage(MemoryStorage::new());
        }

        // 2. Vector Store
        // Note: Core traits::VectorStore implementation needed
        // When available, create based on config (LanceDB, Qdrant, or Memory)
        #[cfg(feature = "vector-memory")]
        {
            // TODO: Register MemoryVectorStore when it implements core::traits::VectorStore
            // if let Some(dimension) = self.vector_dimension {
            //     use crate::vector::memory_store::MemoryVectorStore;
            //     builder = builder.with_vector_store(MemoryVectorStore::new());
            // }
        }

        // 3. Embedding Provider
        // Create embedder based on configuration and available features
        #[cfg(feature = "ollama")]
        {
            // TODO: Register Ollama embedder when EmbeddingProvider implements core::traits::Embedder
            // if let (Some(base_url), Some(model)) = (&self.ollama_base_url, &self.embedding_model) {
            //     use crate::embeddings::ollama::OllamaEmbedder;
            //     // builder = builder.with_embedder(OllamaEmbedder::new(base_url, model));
            // }
        }

        // 4. Entity Extractor
        // Register entity extraction service when NER models are available
        #[cfg(feature = "entity-extraction")]
        {
            // TODO: Register entity extractor
            // if let Some(threshold) = self.entity_confidence_threshold {
            //     // Create LLM-based or NER-based extractor
            //     // builder = builder.with_entity_extractor(extractor);
            // }
        }

        // 5. Retriever
        // Register retrieval system when implementation is complete
        #[cfg(feature = "retrieval")]
        {
            // TODO: Register retrieval system
            // use crate::retrieval::RetrievalSystem;
            // builder = builder.with_retriever(retrieval_system);
        }

        // 6. Language Model
        // Register LLM client for text generation
        #[cfg(feature = "ollama")]
        {
            // TODO: Register Ollama LLM client when it implements core::traits::LanguageModel
            // if let (Some(base_url), Some(model)) = (&self.ollama_base_url, &self.language_model) {
            //     use crate::ollama::OllamaClient;
            //     // builder = builder.with_language_model(OllamaClient::new(base_url, model));
            // }
        }

        // 7. Metrics Collector
        // Register metrics collector when monitoring is enabled
        #[cfg(feature = "monitoring")]
        {
            if self.enable_monitoring {
                // TODO: Register metrics collector
                // use crate::monitoring::MetricsCollector;
                // builder = builder.with_metrics_collector(MetricsCollector::new());
            }
        }

        // 8. Function Registry
        // Register function calling capabilities when enabled
        #[cfg(feature = "function-calling")]
        {
            if self.enable_function_calling {
                // TODO: Register function registry
                // use crate::functions::FunctionRegistry;
                // builder = builder.with_function_registry(FunctionRegistry::new());
            }
        }

        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestService {
        value: String,
    }

    impl TestService {
        fn new(value: String) -> Self {
            Self { value }
        }
    }

    #[test]
    fn test_registry_basic_operations() {
        let mut registry = ServiceRegistry::new();

        // Test registration
        registry.register(TestService::new("test".to_string()));
        assert!(registry.has::<TestService>());
        assert_eq!(registry.len(), 1);

        // Test retrieval
        let service = registry.get::<TestService>().unwrap();
        assert_eq!(service.value, "test");

        // Test removal
        let removed = registry.remove::<TestService>().unwrap();
        assert_eq!(removed.value, "test");
        assert!(!registry.has::<TestService>());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_builder() {
        let registry = RegistryBuilder::new()
            .with_service(TestService::new("builder".to_string()))
            .build();

        assert!(registry.has::<TestService>());
        let service = registry.get::<TestService>().unwrap();
        assert_eq!(service.value, "builder");
    }

    #[test]
    fn test_service_context() {
        let mut registry = ServiceRegistry::new();
        registry.register(TestService::new("context".to_string()));

        let context = ServiceContext::new(registry);
        let service = context.get::<TestService>().unwrap();
        assert_eq!(service.value, "context");

        // Test cloning
        let cloned_context = context.clone();
        let service2 = cloned_context.get::<TestService>().unwrap();
        assert_eq!(service2.value, "context");
    }

    #[test]
    fn test_service_config_default() {
        let config = ServiceConfig::default();
        assert!(config.ollama_base_url.is_some());
        assert!(config.embedding_model.is_some());
        assert!(config.language_model.is_some());
        assert!(config.vector_dimension.is_some());
        assert!(config.entity_confidence_threshold.is_some());
        assert!(config.enable_parallel_processing);
    }
}
