use crate::{
    retrieval::{ResultType, SearchResult},
    summarization::QueryResult,
    text::TextProcessor,
    core::traits::{LanguageModel, GenerationParams, ModelInfo},
    GraphRAGError, Result,
};
use std::collections::{HashMap, HashSet};

// Async implementation module
#[cfg(feature = "async-traits")]
pub mod async_mock_llm;

/// Mock LLM interface for testing without external dependencies
pub trait LLMInterface {
    fn generate_response(&self, prompt: &str) -> Result<String>;
    fn generate_summary(&self, content: &str, max_length: usize) -> Result<String>;
    fn extract_key_points(&self, content: &str, num_points: usize) -> Result<Vec<String>>;
}

/// Simple mock LLM implementation for testing
pub struct MockLLM {
    response_templates: HashMap<String, String>,
    text_processor: TextProcessor,
}

impl MockLLM {
    pub fn new() -> Result<Self> {
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
            response_templates: templates,
            text_processor,
        })
    }

    pub fn with_templates(templates: HashMap<String, String>) -> Result<Self> {
        let text_processor = TextProcessor::new(1000, 100)?;

        Ok(Self {
            response_templates: templates,
            text_processor,
        })
    }

    /// Generate extractive answer from context with improved relevance scoring
    fn generate_extractive_answer(&self, context: &str, query: &str) -> Result<String> {
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
    fn generate_smart_answer(&self, context: &str, question: &str) -> Result<String> {
        // First try extractive approach
        let extractive_result = self.generate_extractive_answer(context, question)?;

        // If extractive failed, generate a contextual response
        if extractive_result.contains("No relevant") || extractive_result.contains("No directly") {
            return self.generate_contextual_response(context, question);
        }

        Ok(extractive_result)
    }

    /// Generate contextual response when direct extraction fails
    fn generate_contextual_response(&self, context: &str, question: &str) -> Result<String> {
        let question_lower = question.to_lowercase();
        let context_lower = context.to_lowercase();

        // Pattern matching for common question types
        if question_lower.contains("who") && question_lower.contains("friend") {
            // Look for character names and relationships
            let names = self.extract_character_names(&context_lower);
            if !names.is_empty() {
                return Ok(format!("Based on the context, the main characters mentioned include: {}. These appear to be friends and companions in the story.", names.join(", ")));
            }
        }

        if question_lower.contains("what")
            && (question_lower.contains("adventure") || question_lower.contains("happen"))
        {
            let events = self.extract_key_events(&context_lower);
            if !events.is_empty() {
                return Ok(format!(
                    "The context describes several events: {}",
                    events.join(", ")
                ));
            }
        }

        if question_lower.contains("where") {
            let locations = self.extract_locations(&context_lower);
            if !locations.is_empty() {
                return Ok(format!(
                    "The story takes place in locations such as: {}",
                    locations.join(", ")
                ));
            }
        }

        // Fallback: provide a summary of the context
        let summary = self.generate_summary(context, 150)?;
        Ok(format!("Based on the available context: {summary}"))
    }

    /// Generate response for direct questions
    fn generate_question_response(&self, question: &str) -> Result<String> {
        let question_lower = question.to_lowercase();

        if question_lower.contains("entity") && question_lower.contains("friend") {
            return Ok("Entity Name's main friends include Second Entity, Friend Entity, and Companion Entity. These characters share many relationships throughout the story.".to_string());
        }

        if question_lower.contains("guardian") {
            return Ok("Guardian Entity is Entity Name's guardian who raised them. They are known for their caring but strict nature.".to_string());
        }

        if question_lower.contains("activity") && question_lower.contains("main") {
            return Ok("The main activity episode is one of the most famous events, where they cleverly convince other characters to participate in the main activity.".to_string());
        }

        Ok(
            "I need more specific context to provide a detailed answer to this question."
                .to_string(),
        )
    }

    /// Extract character names from text
    fn extract_character_names(&self, text: &str) -> Vec<String> {
        let common_names = [
            "entity", "second", "third", "fourth", "fifth", "sixth", "guardian", "companion", "friend", "character",
        ];
        let mut found_names = Vec::new();

        for name in &common_names {
            if text.contains(name) {
                found_names.push(name.to_string());
            }
        }

        found_names
    }

    /// Extract key events/actions from text
    fn extract_key_events(&self, text: &str) -> Vec<String> {
        let event_keywords = [
            "activity",
            "discovery",
            "location",
            "place",
            "action",
            "building",
            "structure",
            "area",
            "water",
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
    fn extract_locations(&self, text: &str) -> Vec<String> {
        let locations = [
            "settlement",
            "waterway",
            "river",
            "cavern",
            "landmass",
            "town",
            "building",
            "institution",
            "dwelling",
        ];
        let mut found_locations = Vec::new();

        for location in &locations {
            if text.contains(location) {
                found_locations.push(location.to_string());
            }
        }

        found_locations
    }
}

impl Default for MockLLM {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl LLMInterface for MockLLM {
    fn generate_response(&self, prompt: &str) -> Result<String> {
        // Debug: Log the prompt to understand what's being sent (uncomment for debugging)
        // println!("DEBUG MockLLM received prompt: {}", &prompt[..prompt.len().min(200)]);

        // Enhanced pattern matching for more intelligent mock responses
        let prompt_lower = prompt.to_lowercase();

        // Handle Q&A format prompts
        if prompt_lower.contains("context:") && prompt_lower.contains("question:") {
            if let Some(context_start) = prompt.find("Context:") {
                let context_section = &prompt[context_start + 8..];
                if let Some(question_start) = context_section.find("Question:") {
                    let context = context_section[..question_start].trim();
                    let question_section = context_section[question_start + 9..].trim();

                    return self.generate_smart_answer(context, question_section);
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
            return self.generate_question_response(prompt);
        }

        // Fallback to template
        Ok(self
            .response_templates
            .get("default")
            .unwrap_or(&"I cannot provide a response based on the given prompt.".to_string())
            .replace("{context}", &prompt[..prompt.len().min(200)]))
    }

    fn generate_summary(&self, content: &str, max_length: usize) -> Result<String> {
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

impl LanguageModel for MockLLM {
    type Error = GraphRAGError;

    fn complete(&self, prompt: &str) -> Result<String> {
        self.generate_response(prompt)
    }

    fn complete_with_params(&self, prompt: &str, _params: GenerationParams) -> Result<String> {
        // For mock LLM, we ignore parameters and just use the basic complete
        self.complete(prompt)
    }

    fn is_available(&self) -> bool {
        true
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

/// Template system for constructing context-aware prompts
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: HashSet<String>,
}

impl PromptTemplate {
    pub fn new(template: String) -> Self {
        let variables = Self::extract_variables(&template);
        Self {
            template,
            variables,
        }
    }

    /// Extract variable names from template (e.g., {context}, {question})
    fn extract_variables(template: &str) -> HashSet<String> {
        let mut variables = HashSet::new();
        let mut chars = template.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                let mut var_name = String::new();
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == '}' {
                        chars.next(); // consume '}'
                        break;
                    }
                    var_name.push(chars.next().unwrap());
                }
                if !var_name.is_empty() {
                    variables.insert(var_name);
                }
            }
        }

        variables
    }

    /// Fill template with provided values
    pub fn fill(&self, values: &HashMap<String, String>) -> Result<String> {
        let mut result = self.template.clone();

        for (key, value) in values {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }

        // Check for unfilled variables
        for var in &self.variables {
            let placeholder = format!("{{{var}}}");
            if result.contains(&placeholder) {
                return Err(GraphRAGError::Generation {
                    message: format!("Template variable '{var}' not provided"),
                });
            }
        }

        Ok(result)
    }

    pub fn required_variables(&self) -> &HashSet<String> {
        &self.variables
    }
}

/// Context information assembled from search results
#[derive(Debug, Clone)]
pub struct AnswerContext {
    pub primary_chunks: Vec<SearchResult>,
    pub supporting_chunks: Vec<SearchResult>,
    pub hierarchical_summaries: Vec<QueryResult>,
    pub entities: Vec<String>,
    pub confidence_score: f32,
    pub source_count: usize,
}

impl AnswerContext {
    pub fn new() -> Self {
        Self {
            primary_chunks: Vec::new(),
            supporting_chunks: Vec::new(),
            hierarchical_summaries: Vec::new(),
            entities: Vec::new(),
            confidence_score: 0.0,
            source_count: 0,
        }
    }

    /// Combine all content into a single text block
    pub fn get_combined_content(&self) -> String {
        let mut content = String::new();

        // Add primary chunks first
        for chunk in &self.primary_chunks {
            if !content.is_empty() {
                content.push_str("\n\n");
            }
            content.push_str(&chunk.content);
        }

        // Add supporting chunks
        for chunk in &self.supporting_chunks {
            if !content.is_empty() {
                content.push_str("\n\n");
            }
            content.push_str(&chunk.content);
        }

        // Add hierarchical summaries
        for summary in &self.hierarchical_summaries {
            if !content.is_empty() {
                content.push_str("\n\n");
            }
            content.push_str(&summary.summary);
        }

        content
    }

    /// Get source attribution information
    pub fn get_sources(&self) -> Vec<SourceAttribution> {
        let mut sources = Vec::new();
        let mut source_id = 1;

        for chunk in &self.primary_chunks {
            sources.push(SourceAttribution {
                id: source_id,
                content_type: "chunk".to_string(),
                source_id: chunk.id.clone(),
                confidence: chunk.score,
                snippet: Self::truncate_content(&chunk.content, 100),
            });
            source_id += 1;
        }

        for chunk in &self.supporting_chunks {
            sources.push(SourceAttribution {
                id: source_id,
                content_type: "supporting_chunk".to_string(),
                source_id: chunk.id.clone(),
                confidence: chunk.score,
                snippet: Self::truncate_content(&chunk.content, 100),
            });
            source_id += 1;
        }

        for summary in &self.hierarchical_summaries {
            sources.push(SourceAttribution {
                id: source_id,
                content_type: "summary".to_string(),
                source_id: summary.node_id.to_string(),
                confidence: summary.score,
                snippet: Self::truncate_content(&summary.summary, 100),
            });
            source_id += 1;
        }

        sources
    }

    fn truncate_content(content: &str, max_len: usize) -> String {
        if content.len() <= max_len {
            content.to_string()
        } else {
            format!("{}...", &content[..max_len])
        }
    }
}

impl Default for AnswerContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Source attribution for generated answers
#[derive(Debug, Clone)]
pub struct SourceAttribution {
    pub id: usize,
    pub content_type: String,
    pub source_id: String,
    pub confidence: f32,
    pub snippet: String,
}

/// Different modes for answer generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerMode {
    Extractive,  // Extract relevant sentences from context
    Abstractive, // Generate new text based on context
    Hybrid,      // Combine extraction and generation
}

/// Configuration for answer generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub mode: AnswerMode,
    pub max_answer_length: usize,
    pub min_confidence_threshold: f32,
    pub max_sources: usize,
    pub include_citations: bool,
    pub include_confidence_score: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            mode: AnswerMode::Hybrid,
            max_answer_length: 500,
            min_confidence_threshold: 0.3,
            max_sources: 10,
            include_citations: true,
            include_confidence_score: true,
        }
    }
}

/// Generated answer with metadata
#[derive(Debug, Clone)]
pub struct GeneratedAnswer {
    pub answer_text: String,
    pub confidence_score: f32,
    pub sources: Vec<SourceAttribution>,
    pub entities_mentioned: Vec<String>,
    pub mode_used: AnswerMode,
    pub context_quality: f32,
}

impl GeneratedAnswer {
    /// Format the answer with citations
    pub fn format_with_citations(&self) -> String {
        let mut formatted = self.answer_text.clone();

        if !self.sources.is_empty() {
            formatted.push_str("\n\nSources:");
            for source in &self.sources {
                formatted.push_str(&format!(
                    "\n[{}] {} (confidence: {:.2}) - {}",
                    source.id, source.content_type, source.confidence, source.snippet
                ));
            }
        }

        if self.confidence_score > 0.0 {
            formatted.push_str(&format!(
                "\n\nOverall confidence: {:.2}",
                self.confidence_score
            ));
        }

        formatted
    }

    /// Get a quality assessment of the answer
    pub fn get_quality_assessment(&self) -> String {
        let confidence_level = if self.confidence_score >= 0.8 {
            "High"
        } else if self.confidence_score >= 0.5 {
            "Medium"
        } else {
            "Low"
        };

        let source_quality = if self.sources.len() >= 3 {
            "Well-sourced"
        } else if !self.sources.is_empty() {
            "Moderately sourced"
        } else {
            "Poorly sourced"
        };

        format!(
            "Confidence: {} | Sources: {} | Context Quality: {:.2}",
            confidence_level, source_quality, self.context_quality
        )
    }
}

/// Main answer generator that orchestrates the response generation process
pub struct AnswerGenerator {
    llm: Box<dyn LLMInterface>,
    config: GenerationConfig,
    prompt_templates: HashMap<String, PromptTemplate>,
}

impl AnswerGenerator {
    pub fn new(llm: Box<dyn LLMInterface>, config: GenerationConfig) -> Result<Self> {
        let mut prompt_templates = HashMap::new();

        // Default prompt templates
        prompt_templates.insert("qa".to_string(), PromptTemplate::new(
            "Context:\n{context}\n\nQuestion: {question}\n\nBased on the provided context, please answer the question. If the context doesn't contain enough information, please say so.".to_string()
        ));

        prompt_templates.insert(
            "summary".to_string(),
            PromptTemplate::new(
                "Please provide a summary of the following content:\n\n{content}\n\nSummary:"
                    .to_string(),
            ),
        );

        prompt_templates.insert("extractive".to_string(), PromptTemplate::new(
            "Extract the most relevant information from the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nRelevant information:".to_string()
        ));

        Ok(Self {
            llm,
            config,
            prompt_templates,
        })
    }

    pub fn with_custom_templates(
        llm: Box<dyn LLMInterface>,
        config: GenerationConfig,
        templates: HashMap<String, PromptTemplate>,
    ) -> Result<Self> {
        Ok(Self {
            llm,
            config,
            prompt_templates: templates,
        })
    }

    /// Generate an answer from search results
    pub fn generate_answer(
        &self,
        query: &str,
        search_results: Vec<SearchResult>,
        hierarchical_results: Vec<QueryResult>,
    ) -> Result<GeneratedAnswer> {
        // Assemble context from results
        let context = self.assemble_context(search_results, hierarchical_results)?;

        // Check if we have sufficient context
        if context.confidence_score < self.config.min_confidence_threshold {
            return Ok(GeneratedAnswer {
                answer_text: "Insufficient information available to answer this question."
                    .to_string(),
                confidence_score: context.confidence_score,
                sources: context.get_sources(),
                entities_mentioned: context.entities.clone(),
                mode_used: self.config.mode.clone(),
                context_quality: context.confidence_score,
            });
        }

        // Generate answer based on mode
        let answer_text = match self.config.mode {
            AnswerMode::Extractive => self.generate_extractive_answer(query, &context)?,
            AnswerMode::Abstractive => self.generate_abstractive_answer(query, &context)?,
            AnswerMode::Hybrid => self.generate_hybrid_answer(query, &context)?,
        };

        // Calculate final confidence score
        let final_confidence = self.calculate_answer_confidence(&answer_text, &context);

        Ok(GeneratedAnswer {
            answer_text,
            confidence_score: final_confidence,
            sources: context.get_sources(),
            entities_mentioned: context.entities,
            mode_used: self.config.mode.clone(),
            context_quality: context.confidence_score,
        })
    }

    /// Assemble context from search results
    fn assemble_context(
        &self,
        search_results: Vec<SearchResult>,
        hierarchical_results: Vec<QueryResult>,
    ) -> Result<AnswerContext> {
        let mut context = AnswerContext::new();

        // Separate results by type and quality
        let mut primary_chunks = Vec::new();
        let mut supporting_chunks = Vec::new();
        let mut all_entities = HashSet::new();

        for result in search_results {
            // Collect entities
            all_entities.extend(result.entities.iter().cloned());

            // Categorize by score and type
            if result.score >= 0.7
                && matches!(result.result_type, ResultType::Chunk | ResultType::Entity)
            {
                primary_chunks.push(result);
            } else if result.score >= 0.3 {
                supporting_chunks.push(result);
            }
        }

        // Limit results
        primary_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        supporting_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        primary_chunks.truncate(self.config.max_sources / 2);
        supporting_chunks.truncate(self.config.max_sources / 2);

        let mut hierarchical_summaries = hierarchical_results;
        hierarchical_summaries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        hierarchical_summaries.truncate(3);

        // Calculate confidence based on result quality and quantity
        let avg_primary_score = if primary_chunks.is_empty() {
            0.0
        } else {
            primary_chunks.iter().map(|r| r.score).sum::<f32>() / primary_chunks.len() as f32
        };

        let avg_supporting_score = if supporting_chunks.is_empty() {
            0.0
        } else {
            supporting_chunks.iter().map(|r| r.score).sum::<f32>() / supporting_chunks.len() as f32
        };

        let avg_hierarchical_score = if hierarchical_summaries.is_empty() {
            0.0
        } else {
            hierarchical_summaries.iter().map(|r| r.score).sum::<f32>()
                / hierarchical_summaries.len() as f32
        };

        let confidence_score =
            (avg_primary_score * 0.5 + avg_supporting_score * 0.3 + avg_hierarchical_score * 0.2)
                .min(1.0);

        context.primary_chunks = primary_chunks;
        context.supporting_chunks = supporting_chunks;
        context.hierarchical_summaries = hierarchical_summaries;
        context.entities = all_entities.into_iter().collect();
        context.confidence_score = confidence_score;
        context.source_count = context.primary_chunks.len()
            + context.supporting_chunks.len()
            + context.hierarchical_summaries.len();

        Ok(context)
    }

    /// Generate extractive answer by selecting relevant sentences
    fn generate_extractive_answer(&self, query: &str, context: &AnswerContext) -> Result<String> {
        let combined_content = context.get_combined_content();

        if combined_content.is_empty() {
            return Ok("No relevant content found.".to_string());
        }

        // Use the LLM's extractive capabilities or fallback to simple extraction
        let template =
            self.prompt_templates
                .get("extractive")
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "Extractive template not found".to_string(),
                })?;

        let mut values = HashMap::new();
        values.insert("context".to_string(), combined_content);
        values.insert("question".to_string(), query.to_string());

        let prompt = template.fill(&values)?;
        let response = self.llm.generate_response(&prompt)?;

        // Truncate if too long
        if response.len() > self.config.max_answer_length {
            Ok(format!(
                "{}...",
                &response[..self.config.max_answer_length - 3]
            ))
        } else {
            Ok(response)
        }
    }

    /// Generate abstractive answer using LLM
    fn generate_abstractive_answer(&self, query: &str, context: &AnswerContext) -> Result<String> {
        let combined_content = context.get_combined_content();

        if combined_content.is_empty() {
            return Ok("No relevant content found.".to_string());
        }

        let template =
            self.prompt_templates
                .get("qa")
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "QA template not found".to_string(),
                })?;

        let mut values = HashMap::new();
        values.insert("context".to_string(), combined_content);
        values.insert("question".to_string(), query.to_string());

        let prompt = template.fill(&values)?;
        let response = self.llm.generate_response(&prompt)?;

        // Truncate if too long
        if response.len() > self.config.max_answer_length {
            Ok(format!(
                "{}...",
                &response[..self.config.max_answer_length - 3]
            ))
        } else {
            Ok(response)
        }
    }

    /// Generate hybrid answer combining extraction and generation
    fn generate_hybrid_answer(&self, query: &str, context: &AnswerContext) -> Result<String> {
        // First try extractive approach
        let extractive_answer = self.generate_extractive_answer(query, context)?;

        // If extractive answer is too short or generic, try abstractive
        if extractive_answer.len() < 50 || extractive_answer.contains("No relevant") {
            return self.generate_abstractive_answer(query, context);
        }

        // For hybrid, we return the extractive answer but could enhance it
        Ok(extractive_answer)
    }

    /// Calculate confidence score for the generated answer
    fn calculate_answer_confidence(&self, answer: &str, context: &AnswerContext) -> f32 {
        // Base confidence from context
        let mut confidence = context.confidence_score;

        // Adjust based on answer length and content
        if answer.len() < 20 {
            confidence *= 0.7; // Penalize very short answers
        }

        if answer.contains("No relevant") || answer.contains("insufficient") {
            confidence *= 0.5; // Penalize negative responses
        }

        // Boost confidence if answer mentions entities from context
        let answer_lower = answer.to_lowercase();
        let entity_mentions = context
            .entities
            .iter()
            .filter(|entity| answer_lower.contains(&entity.to_lowercase()))
            .count();

        if entity_mentions > 0 {
            confidence += (entity_mentions as f32 * 0.1).min(0.2);
        }

        confidence.min(1.0)
    }

    /// Add a custom prompt template
    pub fn add_template(&mut self, name: String, template: PromptTemplate) {
        self.prompt_templates.insert(name, template);
    }

    /// Update generation configuration
    pub fn update_config(&mut self, new_config: GenerationConfig) {
        self.config = new_config;
    }

    /// Get statistics about the generator
    pub fn get_statistics(&self) -> GeneratorStatistics {
        GeneratorStatistics {
            template_count: self.prompt_templates.len(),
            config: self.config.clone(),
            available_templates: self.prompt_templates.keys().cloned().collect(),
        }
    }
}

/// Statistics about the answer generator
#[derive(Debug)]
pub struct GeneratorStatistics {
    pub template_count: usize,
    pub config: GenerationConfig,
    pub available_templates: Vec<String>,
}

impl GeneratorStatistics {
    pub fn print(&self) {
        println!("Answer Generator Statistics:");
        println!("  Mode: {:?}", self.config.mode);
        println!("  Max answer length: {}", self.config.max_answer_length);
        println!(
            "  Min confidence threshold: {:.2}",
            self.config.min_confidence_threshold
        );
        println!("  Max sources: {}", self.config.max_sources);
        println!("  Include citations: {}", self.config.include_citations);
        println!(
            "  Include confidence: {}",
            self.config.include_confidence_score
        );
        println!("  Available templates: {}", self.available_templates.len());
        for template in &self.available_templates {
            println!("    - {template}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_llm_creation() {
        let llm = MockLLM::new();
        assert!(llm.is_ok());
    }

    #[test]
    fn test_prompt_template() {
        let template = PromptTemplate::new("Hello {name}, how are you?".to_string());
        assert!(template.variables.contains("name"));

        let mut values = HashMap::new();
        values.insert("name".to_string(), "World".to_string());

        let filled = template.fill(&values).unwrap();
        assert_eq!(filled, "Hello World, how are you?");
    }

    #[test]
    fn test_answer_context() {
        let context = AnswerContext::new();
        assert_eq!(context.confidence_score, 0.0);
        assert_eq!(context.source_count, 0);

        let content = context.get_combined_content();
        assert!(content.is_empty());
    }

    #[test]
    fn test_answer_generator_creation() {
        let llm = Box::new(MockLLM::new().unwrap());
        let config = GenerationConfig::default();
        let generator = AnswerGenerator::new(llm, config);
        assert!(generator.is_ok());
    }
}
