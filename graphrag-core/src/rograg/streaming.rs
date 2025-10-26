//! Streaming response generation for ROGRAG system
//!
//! Provides robust response generation with streaming capabilities
//! to improve user experience and system resilience.

#[cfg(feature = "rograg")]
use crate::rograg::{
    IntentResult, ProcessingStats, RogragResponse, SubqueryResult, SubqueryResultType,
};
#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use itertools::Itertools;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use std::collections::HashMap;
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Error types for streaming response generation
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum StreamingError {
    /// Response generation failed during template or content creation.
    #[error("Failed to generate response: {reason}")]
    GenerationFailed {
        /// Description of what failed during generation.
        reason: String
    },

    /// Not enough subquery results to synthesize a response.
    #[error("Insufficient subquery results: got {got}, needed {needed}")]
    InsufficientResults {
        /// Number of results available.
        got: usize,
        /// Minimum number of results required.
        needed: usize
    },

    /// Synthesis operation failed to combine subquery results.
    #[error("Response synthesis failed: {reason}")]
    SynthesisFailed {
        /// Description of the synthesis failure.
        reason: String
    },

    /// Generic streaming error occurred.
    #[error("Streaming error: {message}")]
    StreamingError {
        /// Error message describing the streaming issue.
        message: String
    },
}

/// Configuration for streaming response builder
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Whether to enable streaming mode for responses
    pub enable_streaming: bool,
    /// Size of each response chunk in words when streaming
    pub chunk_size: usize,
    /// Maximum length of the generated response in characters
    pub max_response_length: usize,
    /// Strategy to use for synthesizing multiple subquery results
    pub synthesis_strategy: SynthesisStrategy,
    /// Whether to weight results by confidence scores
    pub confidence_weighting: bool,
    /// Whether to include source attribution in responses
    pub source_attribution: bool,
    /// Whether to add citation markers to sources
    pub enable_citations: bool,
}

#[cfg(feature = "rograg")]
impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            chunk_size: 256,
            max_response_length: 2048,
            synthesis_strategy: SynthesisStrategy::Weighted,
            confidence_weighting: true,
            source_attribution: true,
            enable_citations: true,
        }
    }
}

/// Strategy for synthesizing multiple subquery results
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, Serialize, Deserialize)]
pub enum SynthesisStrategy {
    /// Concatenate results in order
    Sequential,
    /// Weight results by confidence
    Weighted,
    /// Select best result only
    BestOnly,
    /// Merge results intelligently
    SmartMerge,
    /// Hierarchical combination
    Hierarchical,
}

/// Streaming response chunk
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
    /// Sequential identifier for this chunk
    pub chunk_id: usize,
    /// Text content of this chunk
    pub content: String,
    /// Whether this is the final chunk in the stream
    pub is_final: bool,
    /// Confidence score for the information in this chunk
    pub confidence: f32,
    /// Source documents referenced in this chunk
    pub sources: Vec<String>,
    /// Additional metadata for this chunk
    pub metadata: HashMap<String, String>,
}

/// Template for response generation
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct ResponseTemplate {
    /// Type of response this template is designed for
    pub template_type: TemplateType,
    /// Template pattern with placeholder markers like {entity}
    pub pattern: String,
    /// List of placeholder names that can be substituted
    pub placeholders: Vec<String>,
    /// Minimum confidence required to use this template
    pub confidence_threshold: f32,
}

/// Type of response template for different query intents
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, StrumDisplay, EnumString, PartialEq, Eq, Hash)]
pub enum TemplateType {
    /// Template for factual information queries
    Factual,
    /// Template for definition queries
    Definitional,
    /// Template for relationship queries between entities
    Relational,
    /// Template for comparison queries
    Comparative,
    /// Template for summary or overview queries
    Summary,
    /// Template for cause-and-effect queries
    Causal,
    /// Template for time-based queries
    Temporal,
    /// Generic fallback template when no specific type matches
    Fallback,
}

/// Streaming response builder implementation
#[cfg(feature = "rograg")]
pub struct StreamingResponseBuilder {
    /// Configuration controlling streaming behavior
    config: StreamingConfig,
    /// Template registry organized by template type
    templates: HashMap<TemplateType, Vec<ResponseTemplate>>,
    /// Engine for synthesizing multiple subquery results
    synthesis_engine: SynthesisEngine,
}

#[cfg(feature = "rograg")]
impl Default for StreamingResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingResponseBuilder {
    /// Create a new streaming response builder
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }

    /// Create a new streaming response builder with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        let mut builder = Self {
            config,
            templates: HashMap::new(),
            synthesis_engine: SynthesisEngine::new(),
        };

        builder.initialize_templates();
        builder
    }

    /// Initialize response templates
    fn initialize_templates(&mut self) {
        // Factual templates
        self.add_template(
            TemplateType::Factual,
            ResponseTemplate {
                template_type: TemplateType::Factual,
                pattern: "Based on the available information, {content}. {confidence_indicator}"
                    .to_string(),
                placeholders: vec!["content".to_string(), "confidence_indicator".to_string()],
                confidence_threshold: 0.7,
            },
        );

        // Definitional templates
        self.add_template(
            TemplateType::Definitional,
            ResponseTemplate {
                template_type: TemplateType::Definitional,
                pattern: "{entity} is {definition}. {additional_context}".to_string(),
                placeholders: vec![
                    "entity".to_string(),
                    "definition".to_string(),
                    "additional_context".to_string(),
                ],
                confidence_threshold: 0.6,
            },
        );

        // Relational templates
        self.add_template(
            TemplateType::Relational,
            ResponseTemplate {
                template_type: TemplateType::Relational,
                pattern: "{entity1} and {entity2} are related through {relationship}. {details}"
                    .to_string(),
                placeholders: vec![
                    "entity1".to_string(),
                    "entity2".to_string(),
                    "relationship".to_string(),
                    "details".to_string(),
                ],
                confidence_threshold: 0.5,
            },
        );

        // Comparative templates
        self.add_template(
            TemplateType::Comparative,
            ResponseTemplate {
                template_type: TemplateType::Comparative,
                pattern: "Comparing {entity1} and {entity2}: {comparison}. {conclusion}"
                    .to_string(),
                placeholders: vec![
                    "entity1".to_string(),
                    "entity2".to_string(),
                    "comparison".to_string(),
                    "conclusion".to_string(),
                ],
                confidence_threshold: 0.6,
            },
        );

        // Summary templates
        self.add_template(
            TemplateType::Summary,
            ResponseTemplate {
                template_type: TemplateType::Summary,
                pattern: "Here's what I found about {topic}: {summary}. {key_points}".to_string(),
                placeholders: vec![
                    "topic".to_string(),
                    "summary".to_string(),
                    "key_points".to_string(),
                ],
                confidence_threshold: 0.5,
            },
        );

        // Fallback template
        self.add_template(
            TemplateType::Fallback,
            ResponseTemplate {
                template_type: TemplateType::Fallback,
                pattern: "Based on the available information: {content}".to_string(),
                placeholders: vec!["content".to_string()],
                confidence_threshold: 0.3,
            },
        );
    }

    /// Add a response template
    fn add_template(&mut self, template_type: TemplateType, template: ResponseTemplate) {
        self.templates
            .entry(template_type)
            .or_default()
            .push(template);
    }

    /// Build a streaming response from subquery results
    ///
    /// Synthesizes multiple subquery results into a coherent response using the configured
    /// synthesis strategy and appropriate response templates.
    ///
    /// # Arguments
    ///
    /// * `query` - The original user query
    /// * `subquery_results` - Results from decomposed subqueries
    /// * `intent_result` - Classified intent of the query
    ///
    /// # Returns
    ///
    /// A complete ROGRAG response with synthesized content, confidence scores, and sources
    pub async fn build_streaming_response(
        &self,
        query: String,
        subquery_results: Vec<SubqueryResult>,
        intent_result: IntentResult,
    ) -> Result<RogragResponse> {
        let start_time = std::time::Instant::now();

        // Synthesize subquery results
        let synthesis_result = self
            .synthesis_engine
            .synthesize(&subquery_results, &self.config.synthesis_strategy)?;

        // Select appropriate template
        let template_type = self.determine_template_type(&intent_result, &subquery_results);
        let template = self.select_template(&template_type, synthesis_result.confidence)?;

        // Generate response content
        let content = self.generate_content(template, &synthesis_result, &subquery_results)?;

        // Add source attribution if enabled
        let final_content = if self.config.source_attribution {
            self.add_source_attribution(content, &synthesis_result.sources)
        } else {
            content
        };

        // Calculate final confidence
        let confidence = self.calculate_final_confidence(&synthesis_result, &intent_result);

        // Extract sources
        let sources = self.extract_all_sources(&subquery_results);

        let processing_time = start_time.elapsed();

        Ok(RogragResponse {
            query,
            content: final_content,
            confidence,
            sources,
            subquery_results: subquery_results.clone(),
            intent_result,
            processing_stats: ProcessingStats {
                total_time_ms: processing_time.as_millis() as u64,
                decomposition_time_ms: 0, // Set by caller
                retrieval_time_ms: 0,     // Set by caller
                synthesis_time_ms: processing_time.as_millis() as u64,
                intent_classification_time_ms: 0,
                validation_time_ms: 0,
                subqueries_processed: subquery_results.len(),
                fallback_used: subquery_results
                    .iter()
                    .filter(|r| matches!(r.result_type, SubqueryResultType::Fallback))
                    .count()
                    > 0,
            },
            is_streaming: self.config.enable_streaming,
            is_refusal: false,
        })
    }

    /// Build a complete (non-streaming) response
    ///
    /// Generates a full response without streaming, using the same synthesis logic
    /// as streaming responses but delivering the complete result at once.
    ///
    /// # Arguments
    ///
    /// * `query` - The original user query
    /// * `subquery_results` - Results from decomposed subqueries
    /// * `intent_result` - Classified intent of the query
    ///
    /// # Returns
    ///
    /// A complete ROGRAG response marked as non-streaming
    pub async fn build_complete_response(
        &self,
        query: String,
        subquery_results: Vec<SubqueryResult>,
        intent_result: IntentResult,
    ) -> Result<RogragResponse> {
        // For now, use the same logic as streaming but mark as non-streaming
        let mut response = self
            .build_streaming_response(query, subquery_results, intent_result)
            .await?;
        response.is_streaming = false;
        Ok(response)
    }

    /// Determine the appropriate template type based on query intent
    ///
    /// Maps the classified query intent to the most suitable response template type,
    /// with fallback logic based on subquery result patterns.
    fn determine_template_type(
        &self,
        intent_result: &IntentResult,
        subquery_results: &[SubqueryResult],
    ) -> TemplateType {
        use crate::rograg::QueryIntent;

        // First, check intent classification
        match intent_result.primary_intent {
            QueryIntent::Factual => TemplateType::Factual,
            QueryIntent::Definitional => TemplateType::Definitional,
            QueryIntent::Relational => TemplateType::Relational,
            QueryIntent::Comparative => TemplateType::Comparative,
            QueryIntent::Summary | QueryIntent::Exploratory => TemplateType::Summary,
            QueryIntent::Causal => TemplateType::Causal,
            QueryIntent::Temporal => TemplateType::Temporal,
            _ => {
                // Fallback: determine from subquery results
                if subquery_results.len() > 1 {
                    TemplateType::Summary
                } else {
                    TemplateType::Fallback
                }
            }
        }
    }

    /// Select the best template for the given type and confidence level
    ///
    /// Finds a template matching the type with an appropriate confidence threshold.
    /// Falls back to generic templates if no exact match is found.
    fn select_template(
        &self,
        template_type: &TemplateType,
        confidence: f32,
    ) -> Result<&ResponseTemplate> {
        let templates = self
            .templates
            .get(template_type)
            .or_else(|| self.templates.get(&TemplateType::Fallback))
            .ok_or_else(|| StreamingError::GenerationFailed {
                reason: "No suitable template found".to_string(),
            })?;

        // Find template with appropriate confidence threshold
        templates
            .iter()
            .find(|t| confidence >= t.confidence_threshold)
            .or_else(|| templates.last()) // Fallback to last template
            .ok_or_else(|| crate::GraphRAGError::TextProcessing {
                message: "No template matches confidence level".to_string(),
            })
    }

    /// Generate content using template and synthesis result
    ///
    /// Replaces template placeholders with actual content from synthesis results
    /// and subquery data, then cleans up the final output.
    fn generate_content(
        &self,
        template: &ResponseTemplate,
        synthesis_result: &SynthesisResult,
        subquery_results: &[SubqueryResult],
    ) -> Result<String> {
        let mut content = template.pattern.clone();

        // Replace placeholders
        for placeholder in &template.placeholders {
            let replacement = match placeholder.as_str() {
                "content" => synthesis_result.content.clone(),
                "confidence_indicator" => {
                    self.generate_confidence_indicator(synthesis_result.confidence)
                }
                "entity" => self.extract_primary_entity(subquery_results),
                "entity1" => self.extract_entity_by_index(subquery_results, 0),
                "entity2" => self.extract_entity_by_index(subquery_results, 1),
                "definition" => synthesis_result.content.clone(),
                "relationship" => self.extract_relationship(subquery_results),
                "comparison" => synthesis_result.content.clone(),
                "topic" => self.extract_primary_entity(subquery_results),
                "summary" => synthesis_result.content.clone(),
                "additional_context" => self.generate_additional_context(subquery_results),
                "details" => self.generate_details(subquery_results),
                "conclusion" => self.generate_conclusion(synthesis_result),
                "key_points" => self.generate_key_points(subquery_results),
                _ => format!("[{placeholder}]"), // Placeholder not found
            };

            content = content.replace(&format!("{{{placeholder}}}"), &replacement);
        }

        // Clean up the content
        content = self.clean_content(content);

        Ok(content)
    }

    /// Generate confidence indicator text based on confidence score
    ///
    /// Returns human-readable text describing the confidence level of the response.
    fn generate_confidence_indicator(&self, confidence: f32) -> String {
        if confidence >= 0.9 {
            "I'm very confident in this information.".to_string()
        } else if confidence >= 0.7 {
            "This information appears to be reliable.".to_string()
        } else if confidence >= 0.5 {
            "This information has moderate confidence.".to_string()
        } else {
            "Please note that this information has limited confidence.".to_string()
        }
    }

    /// Extract primary entity from results
    ///
    /// Attempts to identify and extract the main entity being discussed
    /// from the first subquery result.
    fn extract_primary_entity(&self, results: &[SubqueryResult]) -> String {
        results
            .first()
            .map(|r| {
                // Try to extract entity name from content
                let words: Vec<&str> = r.content.split_whitespace().collect();
                words.first().unwrap_or(&"the subject").to_string()
            })
            .unwrap_or_else(|| "the subject".to_string())
    }

    /// Extract entity by index from subquery results
    ///
    /// Retrieves the entity at the specified index position, useful for
    /// comparative or relational queries with multiple entities.
    fn extract_entity_by_index(&self, results: &[SubqueryResult], index: usize) -> String {
        results
            .get(index)
            .map(|r| {
                let words: Vec<&str> = r.content.split_whitespace().collect();
                words.first().unwrap_or(&"entity").to_string()
            })
            .unwrap_or_else(|| format!("entity{}", index + 1))
    }

    /// Extract relationship information from subquery results
    ///
    /// Searches for and extracts relationship descriptions between entities
    /// from the subquery results.
    fn extract_relationship(&self, results: &[SubqueryResult]) -> String {
        results
            .iter()
            .find(|r| r.content.contains("related") || r.content.contains("relationship"))
            .map(|r| r.content.clone())
            .unwrap_or_else(|| "a connection".to_string())
    }

    /// Generate additional context from secondary subquery results
    ///
    /// Combines content from results beyond the first to provide supporting context.
    fn generate_additional_context(&self, results: &[SubqueryResult]) -> String {
        if results.len() > 1 {
            let additional: Vec<String> =
                results.iter().skip(1).map(|r| r.content.clone()).collect();

            if !additional.is_empty() {
                format!("Additionally, {}", additional.join(". "))
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    }

    /// Generate details from high-confidence results
    ///
    /// Filters and combines content from subquery results with confidence above 0.6
    /// to provide detailed supporting information.
    fn generate_details(&self, results: &[SubqueryResult]) -> String {
        let details: Vec<String> = results
            .iter()
            .filter(|r| r.confidence > 0.6)
            .map(|r| r.content.clone())
            .collect();

        if details.len() > 1 {
            details.join(". ")
        } else {
            String::new()
        }
    }

    /// Generate conclusion based on synthesis confidence
    ///
    /// Creates a closing statement that reflects the confidence level
    /// of the synthesized information.
    fn generate_conclusion(&self, synthesis_result: &SynthesisResult) -> String {
        if synthesis_result.confidence > 0.8 {
            "This appears to be well-supported by the available information.".to_string()
        } else if synthesis_result.confidence > 0.6 {
            "This conclusion is supported by the available evidence.".to_string()
        } else {
            "This is based on limited information.".to_string()
        }
    }

    /// Generate key points summary from subquery results
    ///
    /// Creates a numbered list of the top 3 key points from subquery results.
    fn generate_key_points(&self, results: &[SubqueryResult]) -> String {
        let points: Vec<String> = results
            .iter()
            .take(3) // Limit to top 3 points
            .enumerate()
            .map(|(i, r)| format!("{}. {}", i + 1, r.content))
            .collect();

        if points.is_empty() {
            String::new()
        } else {
            format!("Key points: {}", points.join("; "))
        }
    }

    /// Clean up content by removing placeholders and normalizing whitespace
    ///
    /// Removes unfilled placeholders, normalizes whitespace, and ensures
    /// proper sentence punctuation.
    fn clean_content(&self, mut content: String) -> String {
        // Remove empty placeholder brackets
        content = regex::Regex::new(r"\{\w+\}")
            .unwrap()
            .replace_all(&content, "")
            .to_string();

        // Clean up extra spaces
        content = regex::Regex::new(r"\s+")
            .unwrap()
            .replace_all(&content, " ")
            .to_string();

        // Remove trailing punctuation followed by spaces
        content = content.trim().to_string();

        // Ensure proper sentence ending
        if !content.is_empty() && !content.ends_with(['.', '!', '?']) {
            content.push('.');
        }

        content
    }

    /// Add source attribution to content
    ///
    /// Appends a formatted list of source citations to the response content
    /// if citation mode is enabled, limiting to the top 3 sources.
    fn add_source_attribution(&self, mut content: String, sources: &[String]) -> String {
        if self.config.enable_citations && !sources.is_empty() {
            let source_list = sources
                .iter()
                .take(3) // Limit to 3 sources
                .enumerate()
                .map(|(i, source)| format!("[{}] {}", i + 1, source))
                .join(", ");

            content = format!("{content}\n\nSources: {source_list}");
        }

        content
    }

    /// Calculate final confidence score
    ///
    /// Combines synthesis confidence with intent classification confidence
    /// using weighted averaging when confidence weighting is enabled.
    fn calculate_final_confidence(
        &self,
        synthesis_result: &SynthesisResult,
        intent_result: &IntentResult,
    ) -> f32 {
        if self.config.confidence_weighting {
            // Weight by both synthesis and intent confidence
            (synthesis_result.confidence * 0.7 + intent_result.confidence * 0.3).min(1.0)
        } else {
            synthesis_result.confidence
        }
    }

    /// Extract all unique sources from subquery results
    ///
    /// Collects and deduplicates all source references from the subquery results.
    fn extract_all_sources(&self, results: &[SubqueryResult]) -> Vec<String> {
        results
            .iter()
            .flat_map(|r| r.sources.iter())
            .cloned()
            .unique()
            .collect()
    }

    /// Generate streaming chunks from a complete response
    ///
    /// Splits a full response into smaller chunks for progressive streaming delivery.
    /// Each chunk is marked with its position and whether it's the final chunk.
    ///
    /// # Arguments
    ///
    /// * `response` - The complete response to split into chunks
    ///
    /// # Returns
    ///
    /// A vector of response chunks suitable for streaming
    pub async fn generate_streaming_chunks(
        &self,
        response: &RogragResponse,
    ) -> Result<Vec<ResponseChunk>> {
        let content = &response.content;
        let chunk_size = self.config.chunk_size;
        let mut chunks = Vec::new();

        // Split content into chunks
        let words: Vec<&str> = content.split_whitespace().collect();
        let total_words = words.len();

        for (chunk_idx, chunk_words) in words.chunks(chunk_size).enumerate() {
            let chunk_content = chunk_words.join(" ");
            let is_final = (chunk_idx + 1) * chunk_size >= total_words;

            chunks.push(ResponseChunk {
                chunk_id: chunk_idx,
                content: chunk_content,
                is_final,
                confidence: response.confidence,
                sources: response.sources.clone(),
                metadata: HashMap::new(),
            });
        }

        Ok(chunks)
    }

    /// Get the current streaming configuration
    ///
    /// Returns a reference to the active configuration settings.
    pub fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Update the streaming configuration
    ///
    /// Replaces the current configuration with new settings.
    pub fn update_config(&mut self, config: StreamingConfig) {
        self.config = config;
    }
}

/// Synthesis engine for combining subquery results
///
/// Provides multiple strategies for merging and synthesizing information
/// from multiple subquery results into coherent responses.
#[cfg(feature = "rograg")]
pub struct SynthesisEngine {
    // Configuration could be added here
}

/// Result from synthesis operation
#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// The synthesized content text
    pub content: String,
    /// Confidence score of the synthesis (0.0 to 1.0)
    pub confidence: f32,
    /// Unique sources used in the synthesis
    pub sources: Vec<String>,
    /// The synthesis strategy that was applied
    pub synthesis_method: SynthesisStrategy,
}

#[cfg(feature = "rograg")]
impl Default for SynthesisEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthesisEngine {
    /// Create a new synthesis engine
    pub fn new() -> Self {
        Self {}
    }

    /// Synthesize multiple subquery results using the specified strategy
    ///
    /// Combines information from multiple subquery results into a single coherent
    /// response using one of several synthesis strategies.
    ///
    /// # Arguments
    ///
    /// * `results` - The subquery results to synthesize
    /// * `strategy` - The synthesis strategy to apply
    ///
    /// # Returns
    ///
    /// A synthesis result containing the combined content and metadata
    ///
    /// # Errors
    ///
    /// Returns error if results are empty or synthesis fails
    pub fn synthesize(
        &self,
        results: &[SubqueryResult],
        strategy: &SynthesisStrategy,
    ) -> Result<SynthesisResult> {
        if results.is_empty() {
            return Err(StreamingError::InsufficientResults { got: 0, needed: 1 }.into());
        }

        match strategy {
            SynthesisStrategy::Sequential => self.synthesize_sequential(results),
            SynthesisStrategy::Weighted => self.synthesize_weighted(results),
            SynthesisStrategy::BestOnly => self.synthesize_best_only(results),
            SynthesisStrategy::SmartMerge => self.synthesize_smart_merge(results),
            SynthesisStrategy::Hierarchical => self.synthesize_hierarchical(results),
        }
    }

    /// Sequential synthesis strategy
    ///
    /// Concatenates results in order with equal weighting, computing average confidence.
    fn synthesize_sequential(&self, results: &[SubqueryResult]) -> Result<SynthesisResult> {
        let content = results.iter().map(|r| r.content.clone()).join(". ");

        let avg_confidence =
            results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;

        let sources = results
            .iter()
            .flat_map(|r| r.sources.iter())
            .cloned()
            .unique()
            .collect();

        Ok(SynthesisResult {
            content,
            confidence: avg_confidence,
            sources,
            synthesis_method: SynthesisStrategy::Sequential,
        })
    }

    /// Weighted synthesis strategy
    ///
    /// Prioritizes results by confidence score, taking the top 3 highest-confidence
    /// results and computing weighted confidence.
    fn synthesize_weighted(&self, results: &[SubqueryResult]) -> Result<SynthesisResult> {
        let total_weight: f32 = results.iter().map(|r| r.confidence).sum();

        if total_weight == 0.0 {
            return self.synthesize_sequential(results);
        }

        // Sort by confidence and combine
        let mut sorted_results = results.to_vec();
        sorted_results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let content = sorted_results
            .iter()
            .take(3) // Take top 3 results
            .map(|r| r.content.clone())
            .join(". ");

        let weighted_confidence = sorted_results
            .iter()
            .map(|r| r.confidence * r.confidence) // Square for weighting
            .sum::<f32>()
            / sorted_results.len() as f32;

        let sources = results
            .iter()
            .flat_map(|r| r.sources.iter())
            .cloned()
            .unique()
            .collect();

        Ok(SynthesisResult {
            content,
            confidence: weighted_confidence.sqrt(), // Take square root to normalize
            sources,
            synthesis_method: SynthesisStrategy::Weighted,
        })
    }

    /// Best only synthesis strategy
    ///
    /// Selects only the single result with the highest confidence score,
    /// discarding all other results.
    fn synthesize_best_only(&self, results: &[SubqueryResult]) -> Result<SynthesisResult> {
        let best_result = results
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or_else(|| StreamingError::SynthesisFailed {
                reason: "No best result found".to_string(),
            })?;

        Ok(SynthesisResult {
            content: best_result.content.clone(),
            confidence: best_result.confidence,
            sources: best_result.sources.clone(),
            synthesis_method: SynthesisStrategy::BestOnly,
        })
    }

    /// Smart merge synthesis strategy
    ///
    /// Intelligently combines results by grouping by result type (logic form vs fuzzy match)
    /// and preferring logic form results when available.
    fn synthesize_smart_merge(&self, results: &[SubqueryResult]) -> Result<SynthesisResult> {
        // Group by result type and merge intelligently
        let mut logic_results = Vec::new();
        let mut fuzzy_results = Vec::new();

        for result in results {
            match result.result_type {
                SubqueryResultType::LogicForm => logic_results.push(result),
                SubqueryResultType::FuzzyMatch => fuzzy_results.push(result),
                SubqueryResultType::Fallback => fuzzy_results.push(result),
            }
        }

        // Prefer logic form results
        let primary_results = if !logic_results.is_empty() {
            logic_results
        } else {
            fuzzy_results
        };

        if primary_results.is_empty() {
            return self.synthesize_sequential(results);
        }

        // Combine the best results
        let content = primary_results
            .iter()
            .take(2) // Take top 2
            .map(|r| r.content.clone())
            .join(". ");

        let confidence = primary_results.iter().map(|r| r.confidence).sum::<f32>()
            / primary_results.len() as f32;

        let sources = results
            .iter()
            .flat_map(|r| r.sources.iter())
            .cloned()
            .unique()
            .collect();

        Ok(SynthesisResult {
            content,
            confidence,
            sources,
            synthesis_method: SynthesisStrategy::SmartMerge,
        })
    }

    /// Hierarchical synthesis strategy
    ///
    /// Structures results by importance with the highest confidence result as primary
    /// and supporting results as secondary context.
    fn synthesize_hierarchical(&self, results: &[SubqueryResult]) -> Result<SynthesisResult> {
        // Sort by confidence and create hierarchical structure
        let mut sorted_results = results.to_vec();
        sorted_results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut content_parts = Vec::new();

        if let Some(primary) = sorted_results.first() {
            content_parts.push(primary.content.clone());
        }

        if sorted_results.len() > 1 {
            let supporting: Vec<String> = sorted_results
                .iter()
                .skip(1)
                .take(2)
                .map(|r| r.content.clone())
                .collect();

            if !supporting.is_empty() {
                content_parts.push(format!("Additionally: {}", supporting.join("; ")));
            }
        }

        let content = content_parts.join(". ");
        let confidence = sorted_results.first().map(|r| r.confidence).unwrap_or(0.0);

        let sources = results
            .iter()
            .flat_map(|r| r.sources.iter())
            .cloned()
            .unique()
            .collect();

        Ok(SynthesisResult {
            content,
            confidence,
            sources,
            synthesis_method: SynthesisStrategy::Hierarchical,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rograg::{IntentResult, QueryIntent};

    /// Create test subquery results for unit testing
    #[cfg(feature = "rograg")]
    fn create_test_subquery_results() -> Vec<SubqueryResult> {
        vec![
            SubqueryResult {
                subquery: "What is Entity Name?".to_string(),
                result_type: SubqueryResultType::LogicForm,
                confidence: 0.9,
                content: "Entity Name is a young boy character".to_string(),
                sources: vec!["source1".to_string()],
            },
            SubqueryResult {
                subquery: "Who is Second Entity?".to_string(),
                result_type: SubqueryResultType::FuzzyMatch,
                confidence: 0.8,
                content: "Second Entity is Tom's friend".to_string(),
                sources: vec!["source2".to_string()],
            },
        ]
    }

    /// Create test intent result for unit testing
    #[cfg(feature = "rograg")]
    fn create_test_intent_result() -> IntentResult {
        IntentResult {
            primary_intent: QueryIntent::Factual,
            secondary_intents: vec![],
            confidence: 0.8,
            should_refuse: false,
            refusal_reason: None,
            suggested_reformulation: None,
            complexity_score: 0.3,
        }
    }

    /// Test weighted synthesis strategy
    #[cfg(feature = "rograg")]
    #[test]
    fn test_synthesis_weighted() {
        let engine = SynthesisEngine::new();
        let results = create_test_subquery_results();

        let synthesis = engine
            .synthesize(&results, &SynthesisStrategy::Weighted)
            .unwrap();

        assert!(!synthesis.content.is_empty());
        assert!(synthesis.confidence > 0.0);
        assert_eq!(synthesis.sources.len(), 2);
    }

    /// Test best-only synthesis strategy
    #[cfg(feature = "rograg")]
    #[test]
    fn test_synthesis_best_only() {
        let engine = SynthesisEngine::new();
        let results = create_test_subquery_results();

        let synthesis = engine
            .synthesize(&results, &SynthesisStrategy::BestOnly)
            .unwrap();

        assert_eq!(synthesis.content, "Entity Name is a young boy character");
        assert_eq!(synthesis.confidence, 0.9);
    }

    /// Test complete response building pipeline
    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_response_building() {
        let builder = StreamingResponseBuilder::new();
        let results = create_test_subquery_results();
        let intent = create_test_intent_result();

        let response = builder
            .build_streaming_response("What is Entity Name?".to_string(), results, intent)
            .await
            .unwrap();

        assert!(!response.content.is_empty());
        assert!(response.confidence > 0.0);
        assert!(!response.sources.is_empty());
        assert!(response.is_streaming);
    }

    /// Test template selection based on type and confidence
    #[cfg(feature = "rograg")]
    #[test]
    fn test_template_selection() {
        let builder = StreamingResponseBuilder::new();

        let template = builder
            .select_template(&TemplateType::Factual, 0.8)
            .unwrap();
        assert_eq!(template.template_type, TemplateType::Factual);

        let template = builder
            .select_template(&TemplateType::Factual, 0.3)
            .unwrap();
        // Should fall back to a template with lower threshold or fallback
        assert!(
            template.confidence_threshold <= 0.3
                || template.template_type == TemplateType::Fallback
        );
    }

    /// Test generation of streaming chunks from a complete response
    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_streaming_chunks() {
        let builder = StreamingResponseBuilder::new();
        let results = create_test_subquery_results();
        let intent = create_test_intent_result();

        let response = builder
            .build_streaming_response("Test query".to_string(), results, intent)
            .await
            .unwrap();

        let chunks = builder.generate_streaming_chunks(&response).await.unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().is_final);
    }
}
