//! Test utilities and mock implementations
//!
//! This module provides mock implementations of all core traits for testing purposes.
//! These mocks are designed to be deterministic and controllable for reliable testing.

use crate::core::traits::*;
use crate::core::{
    ChunkId, Document, DocumentId, Entity, EntityId, GraphRAGError, Result, TextChunk,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock storage implementation for testing
#[derive(Debug, Clone, Default)]
pub struct MockStorage {
    entities: Arc<Mutex<HashMap<String, Entity>>>,
    documents: Arc<Mutex<HashMap<String, Document>>>,
    chunks: Arc<Mutex<HashMap<String, TextChunk>>>,
    next_id: Arc<Mutex<usize>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    pub fn reset_call_count(&self) {
        *self.call_count.lock().unwrap() = 0;
    }

    fn increment_call_count(&self) {
        *self.call_count.lock().unwrap() += 1;
    }

    fn next_id(&self) -> String {
        let mut counter = self.next_id.lock().unwrap();
        *counter += 1;
        format!("mock_{}", *counter)
    }
}

impl Storage for MockStorage {
    type Entity = Entity;
    type Document = Document;
    type Chunk = TextChunk;
    type Error = GraphRAGError;

    fn store_entity(&mut self, mut entity: Self::Entity) -> Result<String> {
        self.increment_call_count();

        let id = if entity.id.0.is_empty() {
            self.next_id()
        } else {
            entity.id.0.clone()
        };

        entity.id = EntityId::new(id.clone());
        self.entities.lock().unwrap().insert(id.clone(), entity);
        Ok(id)
    }

    fn retrieve_entity(&self, id: &str) -> Result<Option<Self::Entity>> {
        self.increment_call_count();
        Ok(self.entities.lock().unwrap().get(id).cloned())
    }

    fn store_document(&mut self, mut document: Self::Document) -> Result<String> {
        self.increment_call_count();

        let id = if document.id.0.is_empty() {
            self.next_id()
        } else {
            document.id.0.clone()
        };

        document.id = DocumentId::new(id.clone());
        self.documents.lock().unwrap().insert(id.clone(), document);
        Ok(id)
    }

    fn retrieve_document(&self, id: &str) -> Result<Option<Self::Document>> {
        self.increment_call_count();
        Ok(self.documents.lock().unwrap().get(id).cloned())
    }

    fn store_chunk(&mut self, mut chunk: Self::Chunk) -> Result<String> {
        self.increment_call_count();

        let id = if chunk.id.0.is_empty() {
            self.next_id()
        } else {
            chunk.id.0.clone()
        };

        chunk.id = ChunkId::new(id.clone());
        self.chunks.lock().unwrap().insert(id.clone(), chunk);
        Ok(id)
    }

    fn retrieve_chunk(&self, id: &str) -> Result<Option<Self::Chunk>> {
        self.increment_call_count();
        Ok(self.chunks.lock().unwrap().get(id).cloned())
    }

    fn list_entities(&self) -> Result<Vec<String>> {
        self.increment_call_count();
        Ok(self.entities.lock().unwrap().keys().cloned().collect())
    }

    fn store_entities_batch(&mut self, entities: Vec<Self::Entity>) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        for entity in entities {
            let id = self.store_entity(entity)?;
            ids.push(id);
        }
        Ok(ids)
    }
}

/// Mock embedder implementation for testing
#[derive(Debug, Clone)]
pub struct MockEmbedder {
    dimension: usize,
    deterministic: bool,
    call_count: Arc<Mutex<usize>>,
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEmbedder {
    pub fn new() -> Self {
        Self {
            dimension: 384, // Common embedding dimension
            deterministic: true,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            deterministic: true,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    /// Simple deterministic embedding based on text hash
    fn create_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimension];

        if self.deterministic {
            // Create deterministic embedding based on text hash
            let hash = self.simple_hash(text);
            for (i, val) in embedding.iter_mut().enumerate() {
                *val = ((hash.wrapping_add(i)) as f32 % 100.0) / 100.0;
            }
        } else {
            // Random embedding (not recommended for tests)
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let seed = hasher.finish();

            // Simple deterministic pseudo-random based on text hash
            for (i, val) in embedding.iter_mut().enumerate() {
                let x = (seed
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345)
                    .wrapping_add(i as u64)) as u32;
                *val = (x as f32) / (u32::MAX as f32);
            }
        }

        embedding
    }

    fn simple_hash(&self, text: &str) -> usize {
        text.bytes().fold(0usize, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as usize)
        })
    }
}

impl Embedder for MockEmbedder {
    type Error = GraphRAGError;

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        *self.call_count.lock().unwrap() += 1;
        Ok(self.create_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        *self.call_count.lock().unwrap() += 1;
        Ok(texts
            .iter()
            .map(|text| self.create_embedding(text))
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn is_ready(&self) -> bool {
        true
    }
}

/// Mock vector store implementation for testing
#[derive(Debug, Default)]
pub struct MockVectorStore {
    vectors: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, Option<HashMap<String, String>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockVectorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    fn increment_call_count(&self) {
        *self.call_count.lock().unwrap() += 1;
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Handle zero vectors and very small norms
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return if norm_a == norm_b { 1.0 } else { 0.0 };
        }

        let similarity = dot_product / (norm_a * norm_b);

        // Clamp to [-1, 1] to handle floating point precision issues
        similarity.clamp(-1.0, 1.0)
    }
}

impl VectorStore for MockVectorStore {
    type Error = GraphRAGError;

    fn add_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        self.increment_call_count();
        self.vectors.insert(id.clone(), vector);
        self.metadata.insert(id, metadata);
        Ok(())
    }

    fn add_vectors_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>,
    ) -> Result<()> {
        for (id, vector, metadata) in vectors {
            self.add_vector(id, vector, metadata)?;
        }
        Ok(())
    }

    fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.increment_call_count();

        let mut results = Vec::new();

        for (id, vector) in &self.vectors {
            let similarity = self.cosine_similarity(query_vector, vector);
            let distance = 1.0 - similarity; // Convert similarity to distance

            // Clamp distance to avoid negative values due to floating point precision
            let distance = distance.max(0.0);

            results.push(SearchResult {
                id: id.clone(),
                distance,
                metadata: self.metadata.get(id).unwrap().clone(),
            });
        }

        // Sort by distance (ascending), then by ID for deterministic ordering when distances are equal
        results.sort_by(|a, b| {
            match a.distance.partial_cmp(&b.distance) {
                Some(std::cmp::Ordering::Equal) => a.id.cmp(&b.id), // Stable sort by ID
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal, // Handle NaN case
            }
        });
        results.truncate(k);

        Ok(results)
    }

    fn search_with_threshold(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        let mut results = self.search(query_vector, k)?;
        results.retain(|r| r.distance <= threshold);
        Ok(results)
    }

    fn remove_vector(&mut self, id: &str) -> Result<bool> {
        self.increment_call_count();
        let removed = self.vectors.remove(id).is_some();
        self.metadata.remove(id);
        Ok(removed)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Mock entity extractor for testing
#[derive(Debug, Default)]
pub struct MockEntityExtractor {
    confidence_threshold: f32,
    call_count: Arc<Mutex<usize>>,
}

impl MockEntityExtractor {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    /// Extract mock entities based on simple patterns
    fn extract_mock_entities(&self, text: &str) -> Vec<(MockEntity, f32)> {
        let mut entities = Vec::new();

        // Mock pattern: capitalized words are potential entities
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            if word.chars().next().unwrap_or('a').is_uppercase() && word.len() > 2 {
                let confidence = 0.6 + (0.3 * (word.len() as f32 / 10.0).min(1.0));

                let entity_type = if word.ends_with("Corporation")
                    || word.ends_with("Corp")
                    || word.ends_with("Inc")
                {
                    "Organization"
                } else if i > 0 && words[i - 1].to_lowercase() == "at" {
                    "Location"
                } else {
                    "Person"
                };

                entities.push((
                    MockEntity {
                        name: word.to_string(),
                        entity_type: entity_type.to_string(),
                    },
                    confidence,
                ));
            }
        }

        entities
    }
}

#[derive(Debug, Clone)]
pub struct MockEntity {
    pub name: String,
    pub entity_type: String,
}

impl EntityExtractor for MockEntityExtractor {
    type Entity = MockEntity;
    type Error = GraphRAGError;

    fn extract(&self, text: &str) -> Result<Vec<Self::Entity>> {
        *self.call_count.lock().unwrap() += 1;

        let entities_with_confidence = self.extract_mock_entities(text);
        Ok(entities_with_confidence
            .into_iter()
            .filter(|(_, conf)| *conf >= self.confidence_threshold)
            .map(|(entity, _)| entity)
            .collect())
    }

    fn extract_with_confidence(&self, text: &str) -> Result<Vec<(Self::Entity, f32)>> {
        *self.call_count.lock().unwrap() += 1;

        let entities_with_confidence = self.extract_mock_entities(text);
        Ok(entities_with_confidence
            .into_iter()
            .filter(|(_, conf)| *conf >= self.confidence_threshold)
            .collect())
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }
}

/// Mock language model for testing
#[derive(Debug, Default)]
pub struct MockLanguageModel {
    available: bool,
    call_count: Arc<Mutex<usize>>,
}

impl MockLanguageModel {
    pub fn new() -> Self {
        Self {
            available: true,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn unavailable() -> Self {
        Self {
            available: false,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    fn generate_mock_completion(&self, prompt: &str, params: Option<&GenerationParams>) -> String {
        // Simple mock completion logic
        let max_tokens = params.and_then(|p| p.max_tokens).unwrap_or(50);

        let response = if prompt.to_lowercase().contains("capital")
            && prompt.to_lowercase().contains("france")
        {
            "Paris"
        } else if prompt.to_lowercase().contains("count") {
            "1, 2, 3"
        } else {
            "This is a mock completion for testing purposes."
        };

        // Respect max_tokens by truncating
        let words: Vec<&str> = response.split_whitespace().collect();
        if words.len() > max_tokens {
            words
                .into_iter()
                .take(max_tokens)
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            response.to_string()
        }
    }
}

impl LanguageModel for MockLanguageModel {
    type Error = GraphRAGError;

    fn complete(&self, prompt: &str) -> Result<String> {
        *self.call_count.lock().unwrap() += 1;

        if !self.available {
            return Err(GraphRAGError::LanguageModel {
                message: "Mock language model is not available".to_string(),
            });
        }

        Ok(self.generate_mock_completion(prompt, None))
    }

    fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        *self.call_count.lock().unwrap() += 1;

        if !self.available {
            return Err(GraphRAGError::LanguageModel {
                message: "Mock language model is not available".to_string(),
            });
        }

        Ok(self.generate_mock_completion(prompt, Some(&params)))
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "MockLLM".to_string(),
            version: Some("1.0.0".to_string()),
            max_context_length: Some(4096),
            supports_streaming: false,
        }
    }
}

/// Mock metrics collector for testing
#[derive(Debug, Default)]
pub struct MockMetricsCollector {
    counters: Arc<Mutex<HashMap<String, u64>>>,
    gauges: Arc<Mutex<HashMap<String, f64>>>,
    histograms: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockMetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .unwrap_or(0)
    }

    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.lock().unwrap().get(name).copied()
    }

    pub fn get_histogram_values(&self, name: &str) -> Vec<f64> {
        self.histograms
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl MetricsCollector for MockMetricsCollector {
    fn counter(&self, name: &str, value: u64, _tags: Option<&[(&str, &str)]>) {
        *self.call_count.lock().unwrap() += 1;
        *self
            .counters
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert(0) += value;
    }

    fn gauge(&self, name: &str, value: f64, _tags: Option<&[(&str, &str)]>) {
        *self.call_count.lock().unwrap() += 1;
        self.gauges.lock().unwrap().insert(name.to_string(), value);
    }

    fn histogram(&self, name: &str, value: f64, _tags: Option<&[(&str, &str)]>) {
        *self.call_count.lock().unwrap() += 1;
        self.histograms
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    fn timer(&self, name: &str) -> Timer {
        Timer::new(name.to_string())
    }
}

/// Mock configuration for testing
#[derive(Debug, Clone, PartialEq)]
pub struct MockConfig {
    pub setting1: String,
    pub setting2: i32,
    pub setting3: bool,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            setting1: "default_value".to_string(),
            setting2: 42,
            setting3: true,
        }
    }
}

/// Mock config provider for testing
#[derive(Debug, Default)]
pub struct MockConfigProvider {
    config: Arc<Mutex<Option<MockConfig>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockConfigProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl ConfigProvider for MockConfigProvider {
    type Config = MockConfig;
    type Error = GraphRAGError;

    fn load(&self) -> Result<Self::Config> {
        *self.call_count.lock().unwrap() += 1;
        Ok(self.config.lock().unwrap().clone().unwrap_or_default())
    }

    fn save(&self, config: &Self::Config) -> Result<()> {
        *self.call_count.lock().unwrap() += 1;
        *self.config.lock().unwrap() = Some(config.clone());
        Ok(())
    }

    fn validate(&self, _config: &Self::Config) -> Result<()> {
        *self.call_count.lock().unwrap() += 1;
        Ok(())
    }

    fn default_config(&self) -> Self::Config {
        MockConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_storage() {
        let mut storage = MockStorage::new();
        assert_eq!(storage.call_count(), 0);

        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test".to_string(),
            "Person".to_string(),
            0.9,
        );

        let id = storage.store_entity(entity).unwrap();
        assert!(storage.call_count() > 0);

        let retrieved = storage.retrieve_entity(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Test");
    }

    #[test]
    fn test_mock_embedder() {
        let embedder = MockEmbedder::new();
        assert_eq!(embedder.call_count(), 0);

        let embedding1 = embedder.embed("test").unwrap();
        let embedding2 = embedder.embed("test").unwrap();

        assert_eq!(embedding1, embedding2); // Deterministic
        assert_eq!(embedding1.len(), 384);
        assert!(embedder.call_count() > 0);
    }

    #[test]
    fn test_mock_vector_store() {
        let mut store = MockVectorStore::new();

        store
            .add_vector("vec1".to_string(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        store
            .add_vector("vec2".to_string(), vec![0.0, 1.0, 0.0], None)
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "vec1");
        assert!(store.call_count() > 0);
    }

    #[test]
    fn test_mock_entity_extractor() {
        let extractor = MockEntityExtractor::new();

        let entities = extractor
            .extract("John Smith works at Microsoft Corporation")
            .unwrap();
        assert!(!entities.is_empty());
        assert!(extractor.call_count() > 0);
    }

    #[test]
    fn test_mock_language_model() {
        let model = MockLanguageModel::new();
        assert!(model.is_available());

        let completion = model.complete("The capital of France is").unwrap();
        assert!(completion.contains("Paris"));
        assert!(model.call_count() > 0);
    }
}
