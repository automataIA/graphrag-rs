//! Async implementation of MockLLM demonstrating async trait patterns
//!
//! This module provides an async version of MockLLM that implements the AsyncLanguageModel trait,
//! showcasing how to migrate synchronous implementations to async patterns.

use crate::core::traits::{AsyncLanguageModel, GenerationParams, ModelInfo, ModelUsageStats};
use crate::core::{GraphRAGError, Result};
use crate::generation::LLMInterface;
use crate::text::TextProcessor;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Async version of MockLLM that implements AsyncLanguageModel trait
#[derive(Debug)]
pub struct AsyncMockLLM {
    response_templates: Arc<RwLock<HashMap<String, String>>>,
    text_processor: Arc<TextProcessor>,
    stats: Arc<AsyncLLMStats>,
    simulate_delay: Option<Duration>,
}

/// Statistics tracking for the async LLM
#[derive(Debug, Default)]
struct AsyncLLMStats {
    total_requests: AtomicU64,
    total_tokens_processed: AtomicU64,
    total_response_time: Arc<RwLock<Duration>>,
    error_count: AtomicU64,
}

impl AsyncMockLLM {
    /// Create a new async mock LLM
    pub async fn new() -> Result<Self> {
        let mut templates = HashMap::new();

        // Default response templates
        templates.insert(
            "default".to_string(),
            "Based on the provided context, here is what I found: {context}".to_string(),
        );
        templates.insert(
            "not_found".to_string(),
            "I could not find specific information about this in the provided context.".to_string(),
        );
        templates.insert(
            "insufficient_context".to_string(),
            "The available context is insufficient to provide a complete answer.".to_string(),
        );

        let text_processor = TextProcessor::new(1000, 100)?;

        Ok(Self {
            response_templates: Arc::new(RwLock::new(templates)),
            text_processor: Arc::new(text_processor),
            stats: Arc::new(AsyncLLMStats::default()),
            simulate_delay: Some(Duration::from_millis(100)), // Simulate realistic delay
        })
    }

    /// Create with custom templates
    pub async fn with_templates(templates: HashMap<String, String>) -> Result<Self> {
        let text_processor = TextProcessor::new(1000, 100)?;

        Ok(Self {
            response_templates: Arc::new(RwLock::new(templates)),
            text_processor: Arc::new(text_processor),
            stats: Arc::new(AsyncLLMStats::default()),
            simulate_delay: Some(Duration::from_millis(100)),
        })
    }

    /// Set artificial delay to simulate network latency
    pub fn set_simulate_delay(&mut self, delay: Option<Duration>) {
        self.simulate_delay = delay;
    }

    /// Generate extractive answer from context with improved relevance scoring
    async fn generate_extractive_answer(&self, context: &str, query: &str) -> Result<String> {
        // Simulate processing delay
        if let Some(delay) = self.simulate_delay {
            tokio::time::sleep(delay).await;
        }

        let sentences = self.text_processor.extract_sentences(context);
        if sentences.is_empty() {
            return Ok("No relevant context found.".to_string());
        }

        // Enhanced scoring with partial word matching and named entity recognition
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2) // Filter out short words
            .collect();

        if query_words.is_empty() {
            return Ok("Query too short or contains no meaningful words.".to_string());
        }

        let mut sentence_scores: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(i, sentence)| {
                let sentence_lower = sentence.to_lowercase();
                let mut total_score = 0.0;
                let mut matches = 0;

                for word in &query_words {
                    // Exact word match (highest score)
                    if sentence_lower.contains(word) {
                        total_score += 2.0;
                        matches += 1;
                    }
                    // Partial match for longer words
                    else if word.len() > 4 {
                        for sentence_word in sentence_lower.split_whitespace() {
                            if sentence_word.contains(word) || word.contains(sentence_word) {
                                total_score += 1.0;
                                matches += 1;
                                break;
                            }
                        }
                    }
                }

                // Boost score for sentences with multiple matches
                let coverage_bonus = (matches as f32 / query_words.len() as f32) * 0.5;
                let final_score = total_score + coverage_bonus;

                (i, final_score)
            })
            .collect();

        // Sort by relevance
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top sentences with a minimum relevance threshold
        let mut answer_sentences = Vec::new();
        for (idx, score) in sentence_scores.iter().take(5) {
            if *score > 0.5 {
                // Higher threshold for better quality
                answer_sentences.push(format!(
                    "{} (relevance: {:.1})",
                    sentences[*idx].trim(),
                    score
                ));
            }
        }

        if answer_sentences.is_empty() {
            // If no high-quality matches, provide the best available with lower threshold
            for (idx, score) in sentence_scores.iter().take(2) {
                if *score > 0.0 {
                    answer_sentences.push(format!(
                        "{} (low confidence: {:.1})",
                        sentences[*idx].trim(),
                        score
                    ));
                }
            }
        }

        if answer_sentences.is_empty() {
            Ok("No directly relevant information found in the context.".to_string())
        } else {
            Ok(answer_sentences.join("\n\n"))
        }
    }

    /// Generate smart contextual answer
    async fn generate_smart_answer(&self, context: &str, question: &str) -> Result<String> {
        // First try extractive approach
        let extractive_result = self.generate_extractive_answer(context, question).await?;

        // If extractive failed, generate a contextual response
        if extractive_result.contains("No relevant") || extractive_result.contains("No directly") {
            return self.generate_contextual_response(context, question).await;
        }

        Ok(extractive_result)
    }

    /// Generate contextual response when direct extraction fails
    async fn generate_contextual_response(&self, context: &str, question: &str) -> Result<String> {
        let question_lower = question.to_lowercase();
        let context_lower = context.to_lowercase();

        // Pattern matching for common question types
        if question_lower.contains("who") && question_lower.contains("friend") {
            // Look for character names and relationships
            let names = self.extract_character_names(&context_lower).await;
            if !names.is_empty() {
                return Ok(format!("Based on the context, the main characters mentioned include: {}. These appear to be friends and companions in the story.", names.join(", ")));
            }
        }

        if question_lower.contains("what")
            && (question_lower.contains("adventure") || question_lower.contains("happen"))
        {
            let events = self.extract_key_events(&context_lower).await;
            if !events.is_empty() {
                return Ok(format!(
                    "The context describes several events: {}",
                    events.join(", ")
                ));
            }
        }

        if question_lower.contains("where") {
            let locations = self.extract_locations(&context_lower).await;
            if !locations.is_empty() {
                return Ok(format!(
                    "The story takes place in locations such as: {}",
                    locations.join(", ")
                ));
            }
        }

        // Fallback: provide a summary of the context
        let summary = self.generate_summary_async(context, 150).await?;
        Ok(format!("Based on the available context: {summary}"))
    }

    /// Generate response for direct questions
    async fn generate_question_response(&self, question: &str) -> Result<String> {
        let question_lower = question.to_lowercase();

        // Generic pattern-based responses for common query types
        if question_lower.contains("friend") || question_lower.contains("relationship") {
            return Ok("The text describes various character relationships and friendships throughout the narrative.".to_string());
        }

        if question_lower.contains("main character") || question_lower.contains("protagonist") {
            return Ok("The text features several important characters who drive the narrative forward.".to_string());
        }

        if question_lower.contains("event") || question_lower.contains("scene") {
            return Ok("The text contains various significant events and scenes that advance the story.".to_string());
        }

        Ok(
            "I need more specific context to provide a detailed answer to this question."
                .to_string(),
        )
    }

    /// Extract capitalized words that might be names from text
    async fn extract_character_names(&self, text: &str) -> Vec<String> {
        let mut found_names = Vec::new();

        // Extract capitalized words as potential names
        for word in text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
            if clean_word.len() > 2
                && clean_word.chars().next().unwrap().is_uppercase()
                && clean_word.chars().all(|c| c.is_alphabetic())
            {
                found_names.push(clean_word.to_lowercase());
            }
        }

        found_names
    }

    /// Extract key events/actions from text
    async fn extract_key_events(&self, text: &str) -> Vec<String> {
        let event_keywords = [
            "adventure",
            "treasure",
            "cave",
            "island",
            "painting",
            "school",
            "church",
            "graveyard",
            "river",
        ];
        let mut found_events = Vec::new();

        for event in &event_keywords {
            if text.contains(event) {
                found_events.push(format!("events involving {event}"));
            }
        }

        found_events
    }

    /// Extract locations from text
    async fn extract_locations(&self, text: &str) -> Vec<String> {
        let locations = [
            "village",
            "mississippi",
            "river",
            "cave",
            "island",
            "town",
            "church",
            "school",
            "house",
        ];
        let mut found_locations = Vec::new();

        for location in &locations {
            if text.contains(location) {
                found_locations.push(location.to_string());
            }
        }

        found_locations
    }

    /// Generate summary asynchronously
    async fn generate_summary_async(&self, content: &str, max_length: usize) -> Result<String> {
        let sentences = self.text_processor.extract_sentences(content);
        if sentences.is_empty() {
            return Ok(String::new());
        }

        let mut summary = String::new();
        for sentence in sentences.iter().take(3) {
            if summary.len() + sentence.len() > max_length {
                break;
            }
            if !summary.is_empty() {
                summary.push(' ');
            }
            summary.push_str(sentence);
        }

        Ok(summary)
    }

    /// Update statistics after a request
    async fn update_stats(&self, tokens: usize, response_time: Duration, is_error: bool) {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        if is_error {
            self.stats.error_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats
                .total_tokens_processed
                .fetch_add(tokens as u64, Ordering::Relaxed);
        }

        let mut total_time = self.stats.total_response_time.write().await;
        *total_time += response_time;
    }
}

#[async_trait]
impl AsyncLanguageModel for AsyncMockLLM {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        let start_time = Instant::now();

        // Simulate processing delay
        if let Some(delay) = self.simulate_delay {
            tokio::time::sleep(delay).await;
        }

        let result = self.generate_response_internal(prompt).await;
        let response_time = start_time.elapsed();

        // Estimate tokens (rough approximation)
        let tokens = prompt.len() / 4;
        self.update_stats(tokens, response_time, result.is_err())
            .await;

        result
    }

    async fn complete_with_params(
        &self,
        prompt: &str,
        _params: GenerationParams,
    ) -> Result<String> {
        // For mock LLM, we ignore parameters and just use the basic complete
        self.complete(prompt).await
    }

    async fn complete_batch(&self, prompts: &[&str]) -> Result<Vec<String>> {
        // Process prompts concurrently for better performance
        let mut handles = Vec::new();

        for prompt in prompts {
            let prompt_owned = prompt.to_string();
            let self_clone = self.clone();
            handles.push(tokio::spawn(async move {
                self_clone.complete(&prompt_owned).await
            }));
        }

        let mut results = Vec::with_capacity(prompts.len());
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result?),
                Err(e) => {
                    return Err(GraphRAGError::Generation {
                        message: format!("Task join error: {e}"),
                    })
                }
            }
        }

        Ok(results)
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "AsyncMockLLM".to_string(),
            version: Some("1.0.0".to_string()),
            max_context_length: Some(4096),
            supports_streaming: true,
        }
    }

    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        let total_requests = self.stats.total_requests.load(Ordering::Relaxed);
        let total_tokens = self.stats.total_tokens_processed.load(Ordering::Relaxed);
        let error_count = self.stats.error_count.load(Ordering::Relaxed);
        let total_time = *self.stats.total_response_time.read().await;

        let average_response_time_ms = if total_requests > 0 {
            total_time.as_millis() as f64 / total_requests as f64
        } else {
            0.0
        };

        let error_rate = if total_requests > 0 {
            error_count as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(ModelUsageStats {
            total_requests,
            total_tokens_processed: total_tokens,
            average_response_time_ms,
            error_rate,
        })
    }

    async fn estimate_tokens(&self, prompt: &str) -> Result<usize> {
        // Simple estimation: ~4 characters per token
        Ok(prompt.len() / 4)
    }
}

impl AsyncMockLLM {
    /// Internal response generation method
    async fn generate_response_internal(&self, prompt: &str) -> Result<String> {
        let prompt_lower = prompt.to_lowercase();

        // Handle Q&A format prompts
        if prompt_lower.contains("context:") && prompt_lower.contains("question:") {
            if let Some(context_start) = prompt.find("Context:") {
                let context_section = &prompt[context_start + 8..];
                if let Some(question_start) = context_section.find("Question:") {
                    let context = context_section[..question_start].trim();
                    let question_section = context_section[question_start + 9..].trim();

                    return self.generate_smart_answer(context, question_section).await;
                }
            }
        }

        // Handle direct questions about specific topics
        if prompt_lower.contains("who")
            || prompt_lower.contains("what")
            || prompt_lower.contains("where")
            || prompt_lower.contains("when")
            || prompt_lower.contains("how")
            || prompt_lower.contains("why")
        {
            return self.generate_question_response(prompt).await;
        }

        // Fallback to template
        let templates = self.response_templates.read().await;
        Ok(templates
            .get("default")
            .unwrap_or(&"I cannot provide a response based on the given prompt.".to_string())
            .replace("{context}", &prompt[..prompt.len().min(200)]))
    }
}

// Implement Clone for AsyncMockLLM
impl Clone for AsyncMockLLM {
    fn clone(&self) -> Self {
        Self {
            response_templates: Arc::clone(&self.response_templates),
            text_processor: Arc::clone(&self.text_processor),
            stats: Arc::clone(&self.stats),
            simulate_delay: self.simulate_delay,
        }
    }
}

/// Synchronous LLMInterface implementation for backward compatibility
#[async_trait]
impl LLMInterface for AsyncMockLLM {
    fn generate_response(&self, prompt: &str) -> Result<String> {
        // For sync compatibility, use tokio's block_in_place if we're in a tokio context
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(self.complete(prompt))
            })
        } else {
            // If not in async context, create a new runtime
            let rt = tokio::runtime::Runtime::new().map_err(|e| GraphRAGError::Generation {
                message: format!("Failed to create async runtime: {e}"),
            })?;
            rt.block_on(self.complete(prompt))
        }
    }

    fn generate_summary(&self, content: &str, max_length: usize) -> Result<String> {
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(self.generate_summary_async(content, max_length))
            })
        } else {
            let rt = tokio::runtime::Runtime::new().map_err(|e| GraphRAGError::Generation {
                message: format!("Failed to create async runtime: {e}"),
            })?;
            rt.block_on(self.generate_summary_async(content, max_length))
        }
    }

    fn extract_key_points(&self, content: &str, num_points: usize) -> Result<Vec<String>> {
        let keywords = self
            .text_processor
            .extract_keywords(content, num_points * 2);
        let sentences = self.text_processor.extract_sentences(content);

        let mut key_points = Vec::new();
        for keyword in keywords.iter().take(num_points) {
            // Find a sentence containing this keyword
            if let Some(sentence) = sentences
                .iter()
                .find(|s| s.to_lowercase().contains(&keyword.to_lowercase()))
            {
                key_points.push(sentence.clone());
            } else {
                key_points.push(format!("Key concept: {keyword}"));
            }
        }

        Ok(key_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_mock_llm_creation() {
        let llm = AsyncMockLLM::new().await;
        assert!(llm.is_ok());
    }

    #[tokio::test]
    async fn test_async_completion() {
        let llm = AsyncMockLLM::new().await.unwrap();
        let result = llm.complete("Hello, world!").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_batch_completion() {
        let llm = AsyncMockLLM::new().await.unwrap();
        let prompts = vec!["Hello", "World", "Test"];
        let results = llm.complete_batch(&prompts).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_async_usage_stats() {
        let llm = AsyncMockLLM::new().await.unwrap();

        // Make some requests
        let _ = llm.complete("Test prompt 1").await;
        let _ = llm.complete("Test prompt 2").await;

        let stats = llm.get_usage_stats().await.unwrap();
        assert_eq!(stats.total_requests, 2);
        assert!(stats.average_response_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_async_model_availability() {
        let llm = AsyncMockLLM::new().await.unwrap();
        let is_available = llm.is_available().await;
        assert!(is_available);
    }

    #[tokio::test]
    async fn test_async_model_info() {
        let llm = AsyncMockLLM::new().await.unwrap();
        let info = llm.model_info().await;
        assert_eq!(info.name, "AsyncMockLLM");
        assert_eq!(info.version, Some("1.0.0".to_string()));
        assert!(info.supports_streaming);
    }

    #[tokio::test]
    async fn test_token_estimation() {
        let llm = AsyncMockLLM::new().await.unwrap();
        let tokens = llm.estimate_tokens("This is a test prompt").await.unwrap();
        assert!(tokens > 0);
    }
}
