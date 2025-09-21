//! Ollama function calling integration for GraphRAG

use super::{OllamaClient, OllamaConfig, OllamaError, OllamaResult};
use crate::core::KnowledgeGraph;
use crate::function_calling::{
    tools::ToolRegistry, FunctionCall, FunctionCaller, FunctionContext, FunctionResult,
};
use json;
use ollama_rs::generation::chat::{request::ChatMessageRequest, ChatMessage};
use std::collections::HashMap;
use std::time::Instant;

/// Ollama-based function calling agent
pub struct OllamaFunctionAgent {
    client: OllamaClient,
    function_caller: FunctionCaller,
    max_iterations: usize,
    conversation_history: Vec<ChatMessage>,
}

/// Configuration for function calling behavior
#[derive(Debug, Clone)]
pub struct FunctionCallingConfig {
    /// Maximum number of function call iterations per query
    pub max_iterations: usize,
    /// Whether to include function call history in prompts
    pub include_history: bool,
    /// Temperature for chat generation
    pub temperature: f32,
    /// Maximum tokens for responses
    pub max_tokens: Option<u32>,
}

impl Default for FunctionCallingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            include_history: true,
            temperature: 0.1, // Low temperature for more deterministic function calls
            max_tokens: Some(1000),
        }
    }
}

/// Result of a complete function calling session
#[derive(Debug, Clone)]
pub struct FunctionCallingSession {
    /// Original user query
    pub query: String,
    /// Final answer from the LLM
    pub final_answer: String,
    /// All function calls made during the session
    pub function_calls: Vec<FunctionResult>,
    /// LLM conversation messages
    pub conversation: Vec<ChatMessage>,
    /// Total execution time
    pub execution_time_ms: u64,
    /// Number of iterations used
    pub iterations_used: usize,
    /// Whether the session completed successfully
    pub success: bool,
}

impl OllamaFunctionAgent {
    /// Create a new Ollama function calling agent
    pub fn new(ollama_config: OllamaConfig) -> OllamaResult<Self> {
        let client = OllamaClient::new(ollama_config)?;
        let mut function_caller = FunctionCaller::new();

        // Register default GraphRAG functions
        ToolRegistry::register_default_functions(&mut function_caller)
            .map_err(|e| OllamaError::GenerationError(e.to_string()))?;

        Ok(Self {
            client,
            function_caller,
            max_iterations: 3,
            conversation_history: Vec::new(),
        })
    }

    /// Process a query using function calling with Ollama
    pub async fn process_query(
        &mut self,
        query: &str,
        knowledge_graph: &KnowledgeGraph,
        config: FunctionCallingConfig,
    ) -> OllamaResult<FunctionCallingSession> {
        let start_time = Instant::now();
        self.max_iterations = config.max_iterations;

        // Reset conversation for new query
        self.conversation_history.clear();

        let mut all_function_calls = Vec::new();
        let mut iterations = 0;

        // Initial system prompt with function definitions
        let system_prompt = self.build_system_prompt();
        let initial_user_message = self.build_initial_user_prompt(query);

        self.conversation_history
            .push(ChatMessage::system(system_prompt));
        self.conversation_history
            .push(ChatMessage::user(initial_user_message));

        // Main function calling loop
        loop {
            if iterations >= self.max_iterations {
                break;
            }

            iterations += 1;

            // Get LLM response
            let llm_response = self.get_llm_response(&config).await?;
            self.conversation_history
                .push(ChatMessage::assistant(llm_response.clone()));

            // Check if LLM wants to make function calls
            if let Some(function_calls) = self.parse_function_calls(&llm_response)? {
                if function_calls.is_empty() {
                    // No function calls, this should be the final answer
                    break;
                }

                // Execute function calls
                let function_results = self
                    .execute_function_calls(
                        function_calls,
                        knowledge_graph,
                        query,
                        &all_function_calls,
                    )
                    .await?;

                all_function_calls.extend(function_results.clone());

                // Format function results for LLM
                let function_results_message = self.format_function_results(&function_results);
                self.conversation_history
                    .push(ChatMessage::user(function_results_message));
            } else {
                // No function calls detected, treat as final answer
                break;
            }
        }

        // Get final answer if we haven't already
        let final_answer = if iterations >= self.max_iterations {
            // Force a final answer with natural language instruction
            let final_prompt = "Based on all the information gathered above, please provide a comprehensive and natural answer to the original question. Speak directly to the user without mentioning function calls, JSON, or technical processes. Present the information as if you have direct knowledge of the data.";
            self.conversation_history
                .push(ChatMessage::user(final_prompt.to_string()));
            self.get_llm_response(&config).await?
        } else {
            // Check if last response contains function calls - if so, get a natural final answer
            let last_response = self
                .conversation_history
                .last()
                .map(|msg| msg.content.clone())
                .unwrap_or_else(|| "No answer generated".to_string());

            // If the last response contains technical language, request a natural response
            if last_response.contains("function_call")
                || last_response.contains("```json")
                || last_response.contains("Function:")
                || last_response.contains("Result:")
            {
                let natural_prompt = "Please provide a natural, conversational answer to the original question based on the information we found. Do not mention function calls or technical details.";
                self.conversation_history
                    .push(ChatMessage::user(natural_prompt.to_string()));
                self.get_llm_response(&config).await?
            } else {
                last_response
            }
        };

        Ok(FunctionCallingSession {
            query: query.to_string(),
            final_answer,
            function_calls: all_function_calls,
            conversation: self.conversation_history.clone(),
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            iterations_used: iterations,
            success: true,
        })
    }

    /// Build system prompt with function definitions
    fn build_system_prompt(&self) -> String {
        let mut prompt = String::from(
            "You are a GraphRAG assistant that can interact with a knowledge graph database through function calls.\n\n"
        );

        prompt.push_str("Available functions:\n");
        let function_definitions = self.function_caller.get_function_definitions();
        for def in function_definitions {
            prompt.push_str(&format!("- {}: {}\n", def.name, def.description));
        }

        prompt.push_str(
            "\nTo call a function, use this exact JSON format with EXACT parameter names:\n",
        );
        prompt.push_str("```json\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"function_call\": {\n");
        prompt.push_str("    \"name\": \"function_name\",\n");
        prompt.push_str("    \"arguments\": {\"exact_param_name\": \"value\"}\n");
        prompt.push_str("  }\n");
        prompt.push_str("}\n");
        prompt.push_str("```\n\n");

        prompt.push_str("FUNCTION CALL EXAMPLES - USE EXACT PARAMETER NAMES:\n\n");

        prompt.push_str("1. SEARCH FOR ENTITIES:\n");
        prompt.push_str("   {\"function_call\": {\"name\": \"graph_search\", \"arguments\": {\"entity_name\": \"<entity_name>\", \"limit\": 10}}}\n\n");

        prompt.push_str("2. EXPAND ENTITY RELATIONSHIPS:\n");
        prompt.push_str("   {\"function_call\": {\"name\": \"entity_expand\", \"arguments\": {\"entity_id\": \"<entity_id>\", \"depth\": 1, \"limit\": 20}}}\n\n");

        prompt.push_str("3. INFER IMPLICIT RELATIONSHIPS:\n");
        prompt.push_str("   {\"function_call\": {\"name\": \"infer_relationships\", \"arguments\": {\"entity_name\": \"<entity_name>\", \"relation_type\": \"<relation_type>\", \"min_confidence\": 0.3}}}\n\n");

        prompt.push_str("4. TRAVERSE RELATIONSHIP PATHS:\n");
        prompt.push_str("   {\"function_call\": {\"name\": \"relationship_traverse\", \"arguments\": {\"source_entity\": \"<source_entity>\", \"target_entity\": \"<target_entity>\", \"max_hops\": 3}}}\n\n");

        prompt.push_str("5. GET ENTITY CONTEXT:\n");
        prompt.push_str("   {\"function_call\": {\"name\": \"get_entity_context\", \"arguments\": {\"entity_id\": \"<entity_id>\", \"limit\": 5}}}\n\n");

        prompt.push_str("CRITICAL: Use EXACT parameter names as shown. Never use 'entity_id_1', 'entity1_id', 'relationship_type', etc.\n\n");

        prompt.push_str("You can make multiple function calls in sequence. After getting function results, analyze the information and either:\n");
        prompt.push_str("1. Make additional function calls if you need more information\n");
        prompt
            .push_str("2. Provide a final comprehensive answer based on the function results\n\n");

        prompt.push_str("IMPORTANT: In your final answer, speak naturally and directly to the user. Do not mention function calls, JSON, or technical details. Present the information as if you have direct knowledge of the graph data. Use specific information from function results but present it conversationally.");

        prompt
    }

    /// Build initial user prompt
    fn build_initial_user_prompt(&self, query: &str) -> String {
        format!(
            "User question: {query}\n\nPlease analyze this question and make appropriate function calls to gather information from the knowledge graph. Then provide a comprehensive answer based on the results."
        )
    }

    /// Get response from Ollama LLM
    async fn get_llm_response(&self, _config: &FunctionCallingConfig) -> OllamaResult<String> {
        let request = ChatMessageRequest::new(
            self.client.chat_model().to_string(),
            self.conversation_history.clone(),
        );

        let response = self.client.inner().send_chat_messages(request).await?;

        Ok(response.message.content)
    }

    /// Parse function calls from LLM response
    fn parse_function_calls(&self, response: &str) -> OllamaResult<Option<Vec<FunctionCall>>> {
        // Debug: Print the response only if DEBUG env var is set
        if std::env::var("GRAPHRAG_DEBUG").is_ok() {
            println!("🔍 DEBUG: LLM Response:");
            println!("{}", "-".repeat(40));
            println!("{response}");
            println!("{}", "-".repeat(40));
        }

        // Look for JSON blocks containing function calls
        let mut function_calls = Vec::new();

        // Find all JSON code blocks - more robust parsing
        for line in response.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('{')
                && (trimmed.contains("function_call")
                    || trimmed.contains("name")
                    || trimmed.contains("arguments"))
            {
                if std::env::var("GRAPHRAG_DEBUG").is_ok() {
                    println!("🧪 Trying to parse JSON line: {trimmed}");
                }

                match json::parse(trimmed) {
                    Ok(json_obj) => {
                        if std::env::var("GRAPHRAG_DEBUG").is_ok() {
                            println!("✅ JSON parsed successfully: {}", json_obj.pretty(2));
                        }

                        // Try different JSON structures
                        if json_obj["function_call"].is_object() {
                            let function_call = &json_obj["function_call"];
                            if let (Some(name), arguments) =
                                (function_call["name"].as_str(), &function_call["arguments"])
                            {
                                // println!("📞 Found function call: {} with args: {}", name, arguments.pretty(2));
                                function_calls.push(FunctionCall {
                                    name: name.to_string(),
                                    arguments: arguments.clone(),
                                });
                            }
                        } else if json_obj["name"].is_string() && json_obj["arguments"].is_object()
                        {
                            // Alternative format: {"name": "function_name", "arguments": {...}}
                            if let (Some(name), arguments) =
                                (json_obj["name"].as_str(), &json_obj["arguments"])
                            {
                                // println!("📞 Found alternative format function call: {} with args: {}", name, arguments.pretty(2));
                                function_calls.push(FunctionCall {
                                    name: name.to_string(),
                                    arguments: arguments.clone(),
                                });
                            }
                        }
                    }
                    Err(e) => {
                        if std::env::var("GRAPHRAG_DEBUG").is_ok() {
                            println!("❌ JSON parse error: {e}");
                        }
                        // Try to extract from markdown JSON blocks
                        continue;
                    }
                }
            }
        }

        // Also try to find ```json blocks
        if function_calls.is_empty() {
            // println!("🔍 Looking for ```json blocks...");
            let mut in_json_block = false;
            let mut json_content = String::new();

            for line in response.lines() {
                if line.trim() == "```json" {
                    // println!("📦 Found ```json start");
                    in_json_block = true;
                    json_content.clear();
                } else if line.trim() == "```" && in_json_block {
                    // println!("📦 Found ```json end, content: {}", json_content);
                    in_json_block = false;

                    // Try to parse the collected JSON
                    match json::parse(&json_content) {
                        Ok(json_obj) => {
                            // println!("✅ Markdown JSON parsed: {}", json_obj.pretty(2));

                            // Try different JSON structures
                            if json_obj["function_call"].is_object() {
                                let function_call = &json_obj["function_call"];
                                if let (Some(name), arguments) =
                                    (function_call["name"].as_str(), &function_call["arguments"])
                                {
                                    // println!("📞 Found markdown function call: {} with args: {}", name, arguments.pretty(2));
                                    function_calls.push(FunctionCall {
                                        name: name.to_string(),
                                        arguments: arguments.clone(),
                                    });
                                }
                            } else if json_obj["name"].is_string()
                                && json_obj["arguments"].is_object()
                            {
                                // Alternative format
                                if let (Some(name), arguments) =
                                    (json_obj["name"].as_str(), &json_obj["arguments"])
                                {
                                    // println!("📞 Found alternative markdown function call: {} with args: {}", name, arguments.pretty(2));
                                    function_calls.push(FunctionCall {
                                        name: name.to_string(),
                                        arguments: arguments.clone(),
                                    });
                                }
                            }
                        }
                        Err(_e) => {
                            // println!("❌ Failed to parse markdown JSON: {}", e);
                        }
                    }
                } else if in_json_block {
                    json_content.push_str(line);
                    json_content.push('\n');
                }
            }
        }

        Ok(if function_calls.is_empty() {
            None
        } else {
            Some(function_calls)
        })
    }

    /// Execute function calls with retry logic and fallback strategies
    async fn execute_function_calls(
        &mut self,
        function_calls: Vec<FunctionCall>,
        knowledge_graph: &KnowledgeGraph,
        query: &str,
        previous_results: &[FunctionResult],
    ) -> OllamaResult<Vec<FunctionResult>> {
        let context = FunctionContext {
            knowledge_graph,
            query,
            previous_results,
            metadata: HashMap::new(),
        };

        let mut results = Vec::new();
        let mut successful_calls = 0;

        for call in function_calls {
            // Try primary function call first
            match self.function_caller.call_function(call.clone(), &context) {
                Ok(result) => {
                    successful_calls += 1;
                    results.push(result);
                }
                Err(e) => {
                    // Implement fallback strategies based on function type
                    let fallback_result = self
                        .try_fallback_strategies(call.clone(), &context, &e)
                        .await;

                    match fallback_result {
                        Ok(result) => {
                            successful_calls += 1;
                            results.push(result);
                        }
                        Err(_fallback_err) => {
                            // Create a failed result instead of stopping execution
                            let failed_result = FunctionResult {
                                function_name: call.name.clone(),
                                arguments: call.arguments.clone(),
                                success: false,
                                result: json::object! {
                                    "error": e.to_string(),
                                    "message": "Function call failed but continuing with next functions",
                                    "attempted_fallbacks": true
                                },
                                error: Some(e.to_string()),
                                execution_time_ms: 0,
                            };
                            results.push(failed_result);
                        }
                    }
                }
            }
        }

        // If all functions failed, try global fallback strategy
        if successful_calls == 0 && !results.is_empty() {
            if let Some(fallback_result) = self.try_global_fallback(query, knowledge_graph).await {
                results.push(fallback_result);
                successful_calls = 1;
            }
        }

        // If still no success, return an error
        if successful_calls == 0 && !results.is_empty() {
            return Err(OllamaError::GenerationError(
                "All function calls failed. Check entity names and try different queries."
                    .to_string(),
            ));
        }

        Ok(results)
    }

    /// Try fallback strategies for failed function calls
    async fn try_fallback_strategies(
        &mut self,
        original_call: FunctionCall,
        context: &FunctionContext<'_>,
        original_error: &crate::GraphRAGError,
    ) -> Result<FunctionResult, crate::GraphRAGError> {
        match original_call.name.as_str() {
            "graph_search" => {
                // Fallback 1: Try with normalized entity name
                if let Some(entity_name) = original_call.arguments["entity_name"].as_str() {
                    let normalized_name = entity_name.to_lowercase().replace(" ", "_");
                    let fallback_call = FunctionCall {
                        name: "graph_search".to_string(),
                        arguments: json::object! {
                            "entity_name": normalized_name,
                            "limit": original_call.arguments["limit"].as_u32().unwrap_or(10)
                        },
                    };

                    if let Ok(result) = self.function_caller.call_function(fallback_call, context) {
                        return Ok(result);
                    }
                }

                // Fallback 2: Try partial name matching
                if let Some(entity_name) = original_call.arguments["entity_name"].as_str() {
                    let words: Vec<&str> = entity_name.split_whitespace().collect();
                    if words.len() > 1 {
                        // Try with first name only
                        let fallback_call = FunctionCall {
                            name: "graph_search".to_string(),
                            arguments: json::object! {
                                "entity_name": words[0],
                                "limit": original_call.arguments["limit"].as_u32().unwrap_or(10)
                            },
                        };

                        if let Ok(result) =
                            self.function_caller.call_function(fallback_call, context)
                        {
                            return Ok(result);
                        }
                    }
                }
            }

            "infer_relationships" => {
                // Fallback: Try with lower confidence threshold
                if let Some(entity_name) = original_call.arguments["entity_name"].as_str() {
                    let fallback_call = FunctionCall {
                        name: "infer_relationships".to_string(),
                        arguments: json::object! {
                            "entity_name": entity_name,
                            "relation_type": original_call.arguments["relation_type"].as_str().unwrap_or("RELATED"),
                            "min_confidence": 0.1  // Lower threshold
                        },
                    };

                    if let Ok(result) = self.function_caller.call_function(fallback_call, context) {
                        return Ok(result);
                    }
                }
            }

            "entity_expand" => {
                // Fallback: Try with smaller depth
                if let Some(entity_id) = original_call.arguments["entity_id"].as_str() {
                    let fallback_call = FunctionCall {
                        name: "entity_expand".to_string(),
                        arguments: json::object! {
                            "entity_id": entity_id,
                            "depth": 1,  // Reduced depth
                            "limit": original_call.arguments["limit"].as_u32().unwrap_or(10)
                        },
                    };

                    if let Ok(result) = self.function_caller.call_function(fallback_call, context) {
                        return Ok(result);
                    }
                }
            }

            _ => {
                // Generic fallback: retry with simplified parameters
                // (Implementation depends on specific function requirements)
            }
        }

        Err(crate::GraphRAGError::FunctionCall {
            message: format!("Fallback strategies failed: {original_error}"),
        })
    }

    /// Global fallback strategy when all function calls fail
    async fn try_global_fallback(
        &mut self,
        query: &str,
        knowledge_graph: &KnowledgeGraph,
    ) -> Option<FunctionResult> {
        // Extract potential entity names from query using simple heuristics
        let potential_entities = self.extract_entities_from_query(query);

        for entity_name in potential_entities {
            // Try basic graph search with extracted entity
            let fallback_call = FunctionCall {
                name: "graph_search".to_string(),
                arguments: json::object! {
                    "entity_name": entity_name,
                    "limit": 5
                },
            };

            let context = FunctionContext {
                knowledge_graph,
                query,
                previous_results: &[],
                metadata: HashMap::new(),
            };

            if let Ok(result) = self.function_caller.call_function(fallback_call, &context) {
                if result.success {
                    return Some(result);
                }
            }
        }

        None
    }

    /// Extract potential entity names from user query using simple heuristics
    fn extract_entities_from_query(&self, query: &str) -> Vec<String> {
        let mut entities = Vec::new();

        // Common patterns for entity extraction
        let words: Vec<&str> = query.split_whitespace().collect();

        // Look for capitalized words (likely proper nouns)
        for window in words.windows(2) {
            if window[0].chars().next().unwrap_or('a').is_uppercase()
                && window[1].chars().next().unwrap_or('a').is_uppercase()
            {
                entities.push(format!("{} {}", window[0], window[1]));
            }
        }

        // Single capitalized words
        for word in &words {
            if word.len() > 2 && word.chars().next().unwrap_or('a').is_uppercase() {
                let clean_word = word.trim_end_matches(&['.', '?', '!', ','][..]);
                entities.push(clean_word.to_string());
            }
        }

        // Remove duplicates and common words
        let common_words = [
            "Who", "What", "Where", "When", "Why", "How", "The", "Are", "Is",
        ];
        entities.retain(|e| !common_words.contains(&e.as_str()));
        entities.sort();
        entities.dedup();

        entities
    }

    /// Format function results for LLM consumption
    fn format_function_results(&self, results: &[FunctionResult]) -> String {
        let mut formatted = String::from("Data retrieved from knowledge graph:\n\n");

        for result in results.iter() {
            if result.success {
                // Format results more naturally based on function type
                match result.function_name.as_str() {
                    "graph_search" => {
                        formatted.push_str("Search results:\n");
                        if result.result["entities"].is_array() {
                            for entity in result.result["entities"].members() {
                                if let (Some(name), Some(entity_type)) =
                                    (entity["name"].as_str(), entity["type"].as_str())
                                {
                                    formatted.push_str(&format!("- {name} ({entity_type})\n"));
                                }
                            }
                        }
                    }
                    "infer_relationships" => {
                        formatted.push_str("Relationship analysis:\n");
                        if result.result["inferred_relationships"].is_array() {
                            for rel in result.result["inferred_relationships"].members() {
                                if let (Some(entity), Some(confidence)) =
                                    (rel["entity"].as_str(), rel["confidence"].as_f64())
                                {
                                    formatted.push_str(&format!("- {} appears to have this relationship (confidence: {:.1}%)\n", entity, confidence * 100.0));
                                }
                            }
                        }
                    }
                    "entity_expand" => {
                        formatted.push_str("Connected entities and relationships:\n");
                        formatted.push_str(&format!("   {}\n", result.result.pretty(2)));
                    }
                    _ => {
                        formatted.push_str(&format!(
                            "Information found:\n   {}\n",
                            result.result.pretty(2)
                        ));
                    }
                }
            } else {
                formatted.push_str(&format!(
                    "Could not retrieve information from function {}",
                    result.function_name
                ));
                if let Some(ref error) = result.error {
                    formatted.push_str(&format!(" ({error})"));
                }
                formatted.push('\n');
            }
            formatted.push('\n');
        }

        formatted.push_str("Based on this information from the knowledge graph, provide a natural response to the user's question without mentioning technical details.");

        formatted
    }

    /// Get function caller statistics
    pub fn get_statistics(&self) -> crate::function_calling::FunctionCallStatistics {
        self.function_caller.get_statistics()
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
        self.function_caller.clear_history();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_function_agent_creation() {
        let config = OllamaConfig::default();
        let agent = OllamaFunctionAgent::new(config);

        // Should succeed even if Ollama is not running
        assert!(agent.is_ok());
    }

    #[test]
    fn test_function_call_parsing() {
        let config = OllamaConfig::default();
        let agent = OllamaFunctionAgent::new(config).unwrap();

        let response = r#"
        I need to search for information.

        ```json
        {
          "function_call": {
            "name": "graph_search",
            "arguments": {"entity_name": "<entity_name>", "limit": 5}
          }
        }
        ```
        "#;

        let parsed = agent.parse_function_calls(response).unwrap();
        assert!(parsed.is_some());

        let calls = parsed.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "graph_search");
    }

    #[test]
    fn test_system_prompt_generation() {
        let config = OllamaConfig::default();
        let agent = OllamaFunctionAgent::new(config).unwrap();

        let prompt = agent.build_system_prompt();
        assert!(prompt.contains("GraphRAG assistant"));
        assert!(prompt.contains("function_call"));
        assert!(prompt.contains("graph_search"));
    }
}
