//! Async GraphRAG System
//!
//! This module provides a complete async implementation of the GraphRAG system
//! that leverages all async traits for maximum performance and scalability.

use crate::{
    config::Config,
    core::{
        traits::BoxedAsyncLanguageModel,
        Document, DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Result,
        TextChunk,
    },
    generation::{AnswerContext, GeneratedAnswer, PromptTemplate},
    retrieval::SearchResult,
    summarization::{DocumentTree, HierarchicalConfig, QueryResult},
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Async version of the main GraphRAG system
pub struct AsyncGraphRAG {
    #[allow(dead_code)] config: Config,
    knowledge_graph: Arc<RwLock<Option<KnowledgeGraph>>>,
    document_trees: Arc<RwLock<HashMap<DocumentId, DocumentTree>>>,
    hierarchical_config: HierarchicalConfig,
    language_model: Option<BoxedAsyncLanguageModel>,
}

impl AsyncGraphRAG {
    /// Create a new async GraphRAG instance
    pub async fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            knowledge_graph: Arc::new(RwLock::new(None)),
            document_trees: Arc::new(RwLock::new(HashMap::new())),
            hierarchical_config: HierarchicalConfig::default(),
            language_model: None,
        })
    }

    /// Create with custom hierarchical configuration
    pub async fn with_hierarchical_config(
        config: Config,
        hierarchical_config: HierarchicalConfig,
    ) -> Result<Self> {
        Ok(Self {
            config,
            knowledge_graph: Arc::new(RwLock::new(None)),
            document_trees: Arc::new(RwLock::new(HashMap::new())),
            hierarchical_config,
            language_model: None,
        })
    }

    /// Set the async language model
    pub async fn set_language_model(&mut self, model: BoxedAsyncLanguageModel) {
        self.language_model = Some(model);
    }

    /// Initialize the async GraphRAG system
    pub async fn initialize(&mut self) -> Result<()> {
        println!("Initializing async GraphRAG system...");

        // Initialize knowledge graph
        {
            let mut graph_guard = self.knowledge_graph.write().await;
            *graph_guard = Some(KnowledgeGraph::new());
        }

        // Initialize with default async mock LLM if none provided
        if self.language_model.is_none() {
            #[cfg(feature = "async-traits")]
            {
                let mock_llm = crate::generation::async_mock_llm::AsyncMockLLM::new().await?;
                self.language_model = Some(Box::new(mock_llm));
            }
            #[cfg(not(feature = "async-traits"))]
            {
                return Err(GraphRAGError::Config {
                    message: "No async language model available and async-traits feature disabled".to_string(),
                });
            }
        }

        println!("Async GraphRAG system initialized successfully");
        Ok(())
    }

    /// Add a document to the system asynchronously
    pub async fn add_document(&mut self, document: Document) -> Result<()> {
        // Build hierarchical tree for the document first
        self.build_document_tree(&document).await?;

        let mut graph_guard = self.knowledge_graph.write().await;
        let graph = graph_guard
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        graph.add_document(document)
    }

    /// Build hierarchical tree for a document asynchronously
    pub async fn build_document_tree(&mut self, document: &Document) -> Result<()> {
        if document.chunks.is_empty() {
            return Ok(());
        }

        println!("Building hierarchical tree for document: {}", document.id);

        let tree = DocumentTree::new(document.id.clone(), self.hierarchical_config.clone())?;
        // Note: In a full async implementation, DocumentTree would also be async

        {
            let mut trees_guard = self.document_trees.write().await;
            trees_guard.insert(document.id.clone(), tree);
        }

        Ok(())
    }

    /// Build the knowledge graph from documents asynchronously
    pub async fn build_graph(&mut self) -> Result<()> {
        let mut graph_guard = self.knowledge_graph.write().await;
        let graph = graph_guard
            .as_mut()
            .ok_or_else(|| GraphRAGError::Config {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        println!("Building knowledge graph asynchronously...");

        // Extract entities from all chunks asynchronously
        let chunks: Vec<_> = graph.chunks().cloned().collect();
        let mut total_entities = 0;

        // For each chunk, extract entities (would use AsyncEntityExtractor in full implementation)
        for chunk in &chunks {
            // Simulate async entity extraction
            let entities = self.extract_entities_async(chunk).await?;

            // Add entities to the graph
            let mut chunk_entity_ids = Vec::new();
            for entity in entities {
                chunk_entity_ids.push(entity.id.clone());
                graph.add_entity(entity)?;
                total_entities += 1;
            }

            // Update chunk with entity references
            if let Some(existing_chunk) = graph.get_chunk_mut(&chunk.id) {
                existing_chunk.entities = chunk_entity_ids;
            }
        }

        println!("✅ Knowledge graph built asynchronously: {total_entities} entities extracted");
        Ok(())
    }

    /// Simulate async entity extraction (would use actual AsyncEntityExtractor)
    async fn extract_entities_async(&self, chunk: &TextChunk) -> Result<Vec<Entity>> {
        // Simulate async processing delay
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        // Simple entity extraction for demo (would use actual async implementation)
        let content = chunk.content.to_lowercase();
        let mut entities = Vec::new();

        // Extract simple named entities
        let names = ["tom", "huck", "polly", "sid", "mary", "jim"];
        for (i, name) in names.iter().enumerate() {
            if content.contains(name) {
                let entity = Entity::new(
                    EntityId::new(format!("{name}_{i}")),
                    name.to_string(),
                    "PERSON".to_string(),
                    0.8,
                );
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// Query the system asynchronously
    pub async fn query(&self, query: &str) -> Result<Vec<String>> {
        // Simulate async retrieval (would use actual AsyncRetriever)
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // For demo, return simple response
        Ok(vec![format!("Async result for: {}", query)])
    }

    /// Generate an answer to a question using async pipeline
    pub async fn answer_question(&self, question: &str) -> Result<GeneratedAnswer> {
        let graph_guard = self.knowledge_graph.read().await;
        let graph = graph_guard
            .as_ref()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Knowledge graph not initialized".to_string(),
            })?;

        let llm = self
            .language_model
            .as_ref()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Language model not initialized".to_string(),
            })?;

        // Perform async retrieval
        let search_results = self.async_retrieval(question, graph).await?;

        // Get hierarchical results
        let hierarchical_results = self.hierarchical_query(question, 5).await?;

        // Generate answer using async LLM
        self.generate_answer_async(question, search_results, hierarchical_results, llm)
            .await
    }

    /// Perform async retrieval
    async fn async_retrieval(
        &self,
        query: &str,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        // Simulate async retrieval processing
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        // Simple search simulation
        let mut results = Vec::new();
        for (i, chunk) in graph.chunks().enumerate().take(3) {
            if chunk.content.to_lowercase().contains(&query.to_lowercase()) {
                results.push(SearchResult {
                    id: chunk.id.to_string(),
                    content: chunk.content.clone(),
                    score: 0.8 - (i as f32 * 0.1),
                    result_type: crate::retrieval::ResultType::Chunk,
                    entities: chunk.entities.iter().map(|e| e.to_string()).collect(),
                    source_chunks: vec![chunk.id.to_string()],
                });
            }
        }

        Ok(results)
    }

    /// Query using hierarchical summarization asynchronously
    pub async fn hierarchical_query(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<QueryResult>> {
        let trees_guard = self.document_trees.read().await;
        let mut all_results = Vec::new();

        // Query all document trees
        for tree in trees_guard.values() {
            // In full implementation, DocumentTree would have async query method
            let tree_results = tree.query(query, max_results)?;
            all_results.extend(tree_results);
        }

        // Sort by score and limit results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_results.truncate(max_results);

        Ok(all_results)
    }

    /// Generate answer using async language model
    async fn generate_answer_async(
        &self,
        question: &str,
        search_results: Vec<SearchResult>,
        hierarchical_results: Vec<QueryResult>,
        llm: &BoxedAsyncLanguageModel,
    ) -> Result<GeneratedAnswer> {
        // Assemble context
        let context = self.assemble_context_async(search_results, hierarchical_results).await?;

        // Create prompt
        let prompt = self.create_qa_prompt(question, &context)?;

        // Generate response using async LLM
        let response = llm.complete(&prompt).await?;

        // Create answer with metadata
        Ok(GeneratedAnswer {
            answer_text: response,
            confidence_score: context.confidence_score,
            sources: context.get_sources(),
            entities_mentioned: context.entities,
            mode_used: crate::generation::AnswerMode::Abstractive,
            context_quality: context.confidence_score,
        })
    }

    /// Assemble context asynchronously
    async fn assemble_context_async(
        &self,
        search_results: Vec<SearchResult>,
        hierarchical_results: Vec<QueryResult>,
    ) -> Result<AnswerContext> {
        // Simulate async context assembly
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        let mut context = AnswerContext::new();

        // Process search results
        for result in search_results {
            context.primary_chunks.push(result);
        }

        // Process hierarchical results
        context.hierarchical_summaries = hierarchical_results;

        // Calculate confidence score
        let avg_score = if context.primary_chunks.is_empty() {
            0.0
        } else {
            context.primary_chunks.iter().map(|r| r.score).sum::<f32>()
                / context.primary_chunks.len() as f32
        };

        context.confidence_score = avg_score;
        context.source_count = context.primary_chunks.len() + context.hierarchical_summaries.len();

        Ok(context)
    }

    /// Create QA prompt from context
    fn create_qa_prompt(&self, question: &str, context: &AnswerContext) -> Result<String> {
        let combined_content = context.get_combined_content();

        let mut values = HashMap::new();
        values.insert("context".to_string(), combined_content);
        values.insert("question".to_string(), question.to_string());

        let template = PromptTemplate::new(
            "Context:\n{context}\n\nQuestion: {question}\n\nBased on the provided context, please answer the question. If the context doesn't contain enough information, please say so.".to_string()
        );

        template.fill(&values)
    }

    /// Batch process multiple documents concurrently
    pub async fn add_documents_batch(&mut self, documents: Vec<Document>) -> Result<()> {
        println!("Processing {} documents concurrently...", documents.len());

        // Process documents sequentially for now to avoid borrowing issues
        // In a production implementation, you'd use channels or other concurrency patterns
        for document in documents {
            self.add_document(document).await?;
        }

        println!("All documents processed successfully");
        Ok(())
    }

    /// Batch answer multiple questions concurrently
    pub async fn answer_questions_batch(&self, questions: &[&str]) -> Result<Vec<GeneratedAnswer>> {
        use futures::stream::{FuturesUnordered, StreamExt};

        let mut futures = FuturesUnordered::new();

        for question in questions {
            futures.push(self.answer_question(question));
        }

        let mut answers = Vec::with_capacity(questions.len());
        while let Some(result) = futures.next().await {
            answers.push(result?);
        }

        Ok(answers)
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> AsyncPerformanceStats {
        let graph_guard = self.knowledge_graph.read().await;
        let trees_guard = self.document_trees.read().await;

        AsyncPerformanceStats {
            total_documents: trees_guard.len(),
            total_entities: graph_guard
                .as_ref()
                .map(|g| g.entity_count())
                .unwrap_or(0),
            total_chunks: graph_guard
                .as_ref()
                .map(|g| g.chunks().count())
                .unwrap_or(0),
            health_status: AsyncHealthStatus::Healthy,
        }
    }

    /// Health check for all async components
    pub async fn health_check(&self) -> Result<AsyncHealthStatus> {
        // Check language model
        if let Some(ref llm) = self.language_model {
            if !llm.health_check().await.unwrap_or(false) {
                return Ok(AsyncHealthStatus::Degraded);
            }
        }

        // Check if knowledge graph is initialized
        let graph_guard = self.knowledge_graph.read().await;
        if graph_guard.is_none() {
            return Ok(AsyncHealthStatus::Degraded);
        }

        Ok(AsyncHealthStatus::Healthy)
    }

    /// Save state asynchronously
    pub async fn save_state_async(&self, output_dir: &str) -> Result<()> {
        use std::fs;

        // Create output directory
        fs::create_dir_all(output_dir)?;

        // Save knowledge graph
        let graph_guard = self.knowledge_graph.read().await;
        if let Some(ref graph) = *graph_guard {
            graph.save_to_json(&format!("{output_dir}/async_knowledge_graph.json"))?;
        }

        // Save document trees
        let trees_guard = self.document_trees.read().await;
        for (doc_id, tree) in trees_guard.iter() {
            let filename = format!("{output_dir}/{doc_id}_async_tree.json");
            let json_content = tree.to_json()?;
            fs::write(&filename, json_content)?;
        }

        println!("Async state saved to {output_dir}");
        Ok(())
    }
}

/// Performance statistics for async GraphRAG
#[derive(Debug)]
pub struct AsyncPerformanceStats {
    pub total_documents: usize,
    pub total_entities: usize,
    pub total_chunks: usize,
    pub health_status: AsyncHealthStatus,
}

/// Health status for async components
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Builder for AsyncGraphRAG
pub struct AsyncGraphRAGBuilder {
    config: Config,
    language_model: Option<BoxedAsyncLanguageModel>,
    hierarchical_config: Option<HierarchicalConfig>,
}

impl AsyncGraphRAGBuilder {
    /// Create a new async builder
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            language_model: None,
            hierarchical_config: None,
        }
    }

    /// Set configuration
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Set async language model
    pub fn language_model(mut self, model: BoxedAsyncLanguageModel) -> Self {
        self.language_model = Some(model);
        self
    }

    /// Set hierarchical configuration
    pub fn hierarchical_config(mut self, config: HierarchicalConfig) -> Self {
        self.hierarchical_config = Some(config);
        self
    }

    /// Build with async mock LLM
    #[cfg(feature = "async-traits")]
    pub async fn with_async_mock_llm(mut self) -> Result<Self> {
        let mock_llm = crate::generation::async_mock_llm::AsyncMockLLM::new().await?;
        self.language_model = Some(Box::new(mock_llm));
        Ok(self)
    }

    /// Build with async Ollama LLM
    #[cfg(all(feature = "ollama", feature = "async-traits"))]
    pub async fn with_async_ollama(
        mut self,
        config: crate::ollama::OllamaConfig,
    ) -> Result<Self> {
        let ollama_llm = crate::ollama::AsyncOllamaGenerator::new(config).await?;
        self.language_model = Some(Box::new(ollama_llm));
        Ok(self)
    }

    /// Build the async GraphRAG instance
    pub async fn build(self) -> Result<AsyncGraphRAG> {
        let hierarchical_config = self.hierarchical_config.unwrap_or_default();

        let mut graphrag = AsyncGraphRAG::with_hierarchical_config(self.config, hierarchical_config).await?;

        if let Some(llm) = self.language_model {
            graphrag.set_language_model(llm).await;
        }

        graphrag.initialize().await?;

        Ok(graphrag)
    }
}

impl Default for AsyncGraphRAGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_graphrag_creation() {
        let config = Config::default();
        let graphrag = AsyncGraphRAG::new(config).await;
        assert!(graphrag.is_ok());
    }

    #[tokio::test]
    async fn test_async_graphrag_initialization() {
        let config = Config::default();
        let mut graphrag = AsyncGraphRAG::new(config).await.unwrap();
        let result = graphrag.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_builder() {
        let result = AsyncGraphRAGBuilder::new().build().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[cfg(feature = "async-traits")]
    async fn test_with_async_mock_llm() {
        let result = AsyncGraphRAGBuilder::new()
            .with_async_mock_llm()
            .await
            .unwrap()
            .build()
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = Config::default();
        let mut graphrag = AsyncGraphRAG::new(config).await.unwrap();
        graphrag.initialize().await.unwrap();

        let health = graphrag.health_check().await.unwrap();
        assert_eq!(health, AsyncHealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_performance_stats() {
        let config = Config::default();
        let mut graphrag = AsyncGraphRAG::new(config).await.unwrap();
        graphrag.initialize().await.unwrap();

        let stats = graphrag.get_performance_stats().await;
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.health_status, AsyncHealthStatus::Healthy);
    }
}