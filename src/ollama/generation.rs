//! Ollama-based answer generation for GraphRAG

use super::{OllamaClient, OllamaConfig, OllamaResult};
use crate::generation::{AnswerMode, GeneratedAnswer, GenerationConfig, LLMInterface, SourceAttribution};
use crate::retrieval::SearchResult;
use crate::summarization::QueryResult;
use crate::Result;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use std::time::Instant;

/// Answer generator using Ollama for local LLM inference
pub struct OllamaGenerator {
    client: OllamaClient,
    config: GenerationConfig,
    stats: GenerationStats,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub requests_made: usize,
    pub total_response_time: std::time::Duration,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub average_tokens_generated: f32,
}

impl GenerationStats {
    pub fn record_request(
        &mut self,
        duration: std::time::Duration,
        success: bool,
        tokens: Option<usize>,
    ) {
        self.requests_made += 1;
        self.total_response_time += duration;

        if success {
            self.successful_requests += 1;
            if let Some(token_count) = tokens {
                self.average_tokens_generated = (self.average_tokens_generated
                    * (self.successful_requests - 1) as f32
                    + token_count as f32)
                    / self.successful_requests as f32;
            }
        } else {
            self.failed_requests += 1;
        }
    }

    pub fn average_response_time(&self) -> std::time::Duration {
        if self.requests_made > 0 {
            self.total_response_time / self.requests_made as u32
        } else {
            std::time::Duration::ZERO
        }
    }

    pub fn success_rate(&self) -> f32 {
        if self.requests_made > 0 {
            self.successful_requests as f32 / self.requests_made as f32
        } else {
            0.0
        }
    }
}

impl OllamaGenerator {
    /// Create new Ollama generator from URL
    pub fn new(url: String) -> crate::Result<Self> {
        let config = OllamaConfig {
            enabled: true,
            host: url,
            port: 11434, // Default port, can be overridden
            ..Default::default()
        };

        let client = OllamaClient::new(config).map_err(|e| crate::GraphRAGError::Generation {
            message: format!("Failed to create Ollama client: {e}"),
        })?;
        let generation_config = GenerationConfig::default();

        Ok(Self {
            client,
            config: generation_config,
            stats: GenerationStats::default(),
        })
    }

    /// Create new Ollama generator with config
    pub fn from_config(config: OllamaConfig) -> OllamaResult<Self> {
        let client = OllamaClient::new(config)?;
        let generation_config = GenerationConfig::default();

        Ok(Self {
            client,
            config: generation_config,
            stats: GenerationStats::default(),
        })
    }

    /// Create generator with custom generation config
    pub fn with_generation_config(
        ollama_config: OllamaConfig,
        generation_config: GenerationConfig,
    ) -> OllamaResult<Self> {
        let client = OllamaClient::new(ollama_config)?;

        Ok(Self {
            client,
            config: generation_config,
            stats: GenerationStats::default(),
        })
    }

    /// Generate answer using Ollama chat completion
    pub async fn generate_answer_async(
        &mut self,
        question: &str,
        search_results: Vec<SearchResult>,
        hierarchical_results: Vec<QueryResult>,
    ) -> OllamaResult<GeneratedAnswer> {
        let start_time = Instant::now();

        // Build context from search results
        let context = self.build_context(&search_results, &hierarchical_results);

        // Create chat messages
        let messages = self.build_chat_messages(question, &context);

        // Prepare chat request
        let chat_request = ChatMessageRequest::new(self.client.chat_model().to_string(), messages);

        // Execute request with timeout
        let response = match self
            .client
            .with_timeout(self.client.inner().send_chat_messages(chat_request))
            .await
        {
            Ok(response) => {
                let duration = start_time.elapsed();
                self.stats.record_request(duration, true, None);
                response
            }
            Err(e) => {
                let duration = start_time.elapsed();
                self.stats.record_request(duration, false, None);
                return Err(e);
            }
        };

        // Extract answer from response
        let answer_text = response.message.content;

        // Build GeneratedAnswer
        let generated_answer = GeneratedAnswer {
            answer_text,
            confidence_score: 0.8, // Default confidence for Ollama responses
            sources: self.extract_sources(&search_results),
            entities_mentioned: self.extract_entities(&search_results),
            mode_used: AnswerMode::Abstractive, // Ollama generates new text
            context_quality: self.calculate_context_quality(&search_results, &hierarchical_results),
        };

        Ok(generated_answer)
    }

    /// Build context string from search results
    fn build_context(
        &self,
        search_results: &[SearchResult],
        hierarchical_results: &[QueryResult],
    ) -> String {
        let mut context = String::new();

        // Add search results
        if !search_results.is_empty() {
            context.push_str("## Search Results\n\n");
            for (i, result) in search_results.iter().enumerate() {
                context.push_str(&format!(
                    "{}. **{}** (Score: {:.2})\n{}\n\n",
                    i + 1,
                    result.id,
                    result.score,
                    result.content
                ));
            }
        }

        // Add hierarchical results
        if !hierarchical_results.is_empty() {
            context.push_str("## Document Summaries\n\n");
            for (i, result) in hierarchical_results.iter().enumerate() {
                context.push_str(&format!(
                    "{}. **Level {}** (Score: {:.2})\n{}\n\n",
                    i + 1,
                    result.level,
                    result.score,
                    result.summary
                ));
            }
        }

        if context.is_empty() {
            context = "No relevant context found.".to_string();
        }

        context
    }

    /// Build chat messages for Ollama
    fn build_chat_messages(&self, question: &str, context: &str) -> Vec<ChatMessage> {
        let system_prompt = format!(
            "You are an AI assistant that answers questions based on provided context. \
            Use the context below to answer the user's question accurately and comprehensively.\n\
            Be concise but informative. If the context doesn't contain enough information \
            to fully answer the question, say so explicitly.\n\n\
            Context:\n{context}"
        );

        vec![
            ChatMessage::new(MessageRole::System, system_prompt),
            ChatMessage::new(MessageRole::User, question.to_string()),
        ]
    }

    /// Extract source information from search results
    fn extract_sources(&self, search_results: &[SearchResult]) -> Vec<SourceAttribution> {
        search_results
            .iter()
            .enumerate()
            .map(|(i, result)| SourceAttribution {
                id: i + 1,
                content_type: format!("{:?}", result.result_type),
                source_id: result.id.clone(),
                confidence: result.score,
                snippet: if result.content.len() > 100 {
                    format!("{}...", &result.content[..100])
                } else {
                    result.content.clone()
                },
            })
            .collect()
    }

    /// Extract mentioned entities from search results
    fn extract_entities(&self, search_results: &[SearchResult]) -> Vec<String> {
        let mut entities = Vec::new();
        for result in search_results {
            entities.extend(result.entities.clone());
        }
        entities.sort();
        entities.dedup();
        entities
    }

    /// Calculate quality score for the available context
    fn calculate_context_quality(
        &self,
        search_results: &[SearchResult],
        hierarchical_results: &[QueryResult],
    ) -> f32 {
        if search_results.is_empty() && hierarchical_results.is_empty() {
            return 0.0;
        }

        let mut quality_score = 0.0;
        let mut total_weight = 0.0;

        // Score based on search results
        for result in search_results {
            quality_score += result.score * 0.6; // 60% weight for search results
            total_weight += 0.6;
        }

        // Score based on hierarchical results
        for result in hierarchical_results {
            quality_score += result.score * 0.4; // 40% weight for hierarchical results
            total_weight += 0.4;
        }

        if total_weight > 0.0 {
            quality_score / total_weight
        } else {
            0.0
        }
    }

    /// Get generation statistics
    pub fn get_stats(&self) -> &GenerationStats {
        &self.stats
    }

    /// Check if Ollama service is available
    pub async fn is_available(&self) -> bool {
        self.client.health_check().await.unwrap_or(false)
    }

    /// Validate that required models are available
    pub async fn validate_setup(&self) -> OllamaResult<()> {
        self.client.validate_models().await
    }
}

impl OllamaGenerator {
    /// Synchronous wrapper for generate_answer_async (for compatibility with existing code)
    pub fn generate_answer(
        &mut self,
        question: &str,
        search_results: Vec<SearchResult>,
        hierarchical_results: Vec<QueryResult>,
    ) -> crate::Result<GeneratedAnswer> {
        // Use tokio to run async code in sync context
        let rt = tokio::runtime::Runtime::new().map_err(|e| crate::GraphRAGError::Generation {
            message: format!("Failed to create async runtime: {e}"),
        })?;

        rt.block_on(async {
            self.generate_answer_async(question, search_results, hierarchical_results)
                .await
                .map_err(|e| crate::GraphRAGError::Generation {
                    message: e.to_string(),
                })
        })
    }

    /// Update generation configuration
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }
}

// Implement LLMInterface for compatibility with existing code
impl LLMInterface for OllamaGenerator {
    fn generate_response(&self, prompt: &str) -> Result<String> {
        // Create a simple async runtime for sync compatibility
        let rt = tokio::runtime::Runtime::new().map_err(|e| crate::GraphRAGError::Generation {
            message: format!("Failed to create async runtime: {e}"),
        })?;

        rt.block_on(async {
            let messages = vec![
                ChatMessage::new(MessageRole::System, "You are a helpful AI assistant. Provide clear, accurate, and concise responses.".to_string()),
                ChatMessage::new(MessageRole::User, prompt.to_string()),
            ];

            let chat_request = ChatMessageRequest::new(self.client.chat_model().to_string(), messages);

            match self.client.with_timeout(self.client.inner().send_chat_messages(chat_request)).await {
                Ok(response) => Ok(response.message.content),
                Err(e) => Err(crate::GraphRAGError::Generation {
                    message: format!("Ollama generation failed: {e}"),
                }),
            }
        })
    }

    fn generate_summary(&self, content: &str, max_length: usize) -> Result<String> {
        let prompt = format!(
            "Please provide a concise summary of the following content in no more than {max_length} characters:\n\n{content}"
        );
        self.generate_response(&prompt)
    }

    fn extract_key_points(&self, content: &str, num_points: usize) -> Result<Vec<String>> {
        let prompt = format!(
            "Extract exactly {num_points} key points from the following content. Present each point as a separate line:\n\n{content}"
        );

        let response = self.generate_response(&prompt)?;

        // Parse the response into separate points
        let points: Vec<String> = response
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .take(num_points)
            .collect();

        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::ResultType;

    #[test]
    fn test_generator_creation() {
        let config = OllamaConfig::default();
        let generator = OllamaGenerator::from_config(config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_context_building() {
        let config = OllamaConfig::default();
        let generator = OllamaGenerator::from_config(config).unwrap();

        let search_results = vec![SearchResult {
            id: "test1".to_string(),
            content: "Test content 1".to_string(),
            score: 0.9,
            result_type: ResultType::Chunk,
            entities: vec!["Entity1".to_string()],
            source_chunks: vec!["chunk1".to_string()],
        }];

        let hierarchical_results = vec![QueryResult {
            node_id: crate::summarization::NodeId::new("node1".to_string()),
            summary: "Test summary".to_string(),
            score: 0.8,
            level: 1,
            keywords: vec!["test".to_string()],
            chunk_ids: vec![crate::core::ChunkId::new("chunk1".to_string())],
        }];

        let context = generator.build_context(&search_results, &hierarchical_results);
        assert!(context.contains("Test content 1"));
        assert!(context.contains("Test summary"));
    }

    #[tokio::test]
    async fn test_availability_check() {
        let config = OllamaConfig::default();
        let generator = OllamaGenerator::from_config(config).unwrap();

        // This will only pass if Ollama is running
        let available = generator.is_available().await;
        println!("Ollama generator available: {available}");
    }
}
