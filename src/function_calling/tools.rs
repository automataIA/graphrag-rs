//! Tool utilities for function calling

use super::functions::{
    EntityExpandFunction, GetEntityContextFunction, GraphSearchFunction,
    InferRelationshipsFunction, RelationshipTraverseFunction,
};
use super::FunctionCaller;
use crate::Result;

/// Tool registry for managing available functions
pub struct ToolRegistry;

impl ToolRegistry {
    /// Register all default GraphRAG functions
    pub fn register_default_functions(function_caller: &mut FunctionCaller) -> Result<()> {
        // Register graph search function
        function_caller.register_function(Box::new(GraphSearchFunction));

        // Register entity expand function
        function_caller.register_function(Box::new(EntityExpandFunction));

        // Register relationship traverse function
        function_caller.register_function(Box::new(RelationshipTraverseFunction));

        // Register entity context function
        function_caller.register_function(Box::new(GetEntityContextFunction));

        // Register inference function
        function_caller.register_function(Box::new(InferRelationshipsFunction::new()));

        Ok(())
    }

    /// Get function definitions in OpenAI function calling format
    pub fn get_openai_function_definitions(function_caller: &FunctionCaller) -> json::JsonValue {
        let definitions = function_caller.get_function_definitions();

        let function_objects: Vec<_> = definitions
            .into_iter()
            .map(|def| {
                json::object! {
                    "type": "function",
                    "function": {
                        "name": def.name,
                        "description": def.description,
                        "parameters": def.parameters
                    }
                }
            })
            .collect();

        json::JsonValue::Array(function_objects)
    }

    /// Parse function call from LLM response (OpenAI format)
    pub fn parse_openai_function_call(
        response: &json::JsonValue,
    ) -> Result<Vec<super::FunctionCall>> {
        let mut function_calls = Vec::new();

        // Handle both single function call and tool_calls array
        if response["function_call"].is_object() {
            let function_call = &response["function_call"];
            if let (Some(name), Some(arguments_str)) = (
                function_call["name"].as_str(),
                function_call["arguments"].as_str(),
            ) {
                let arguments = json::parse(arguments_str).map_err(crate::GraphRAGError::Json)?;

                function_calls.push(super::FunctionCall {
                    name: name.to_string(),
                    arguments,
                });
            }
        }

        // Handle tool_calls format (newer OpenAI format)
        if response["tool_calls"].is_array() {
            for tool_call in response["tool_calls"].members() {
                if tool_call["function"].is_object() {
                    let function = &tool_call["function"];
                    if let (Some(name), Some(arguments_str)) =
                        (function["name"].as_str(), function["arguments"].as_str())
                    {
                        let arguments =
                            json::parse(arguments_str).map_err(crate::GraphRAGError::Json)?;

                        function_calls.push(super::FunctionCall {
                            name: name.to_string(),
                            arguments,
                        });
                    }
                }
            }
        }

        Ok(function_calls)
    }

    /// Format function results for LLM context
    pub fn format_function_results_for_llm(results: &[super::FunctionResult]) -> String {
        if results.is_empty() {
            return "No function calls were made.".to_string();
        }

        let mut formatted = String::from("Function call results:\n\n");

        for (i, result) in results.iter().enumerate() {
            formatted.push_str(&format!(
                "{index}. Function: {name}\n",
                index = i + 1,
                name = result.function_name
            ));

            if result.success {
                formatted.push_str("   Status: Success\n");
                formatted.push_str(&format!(
                    "   Result: {result}\n",
                    result = result.result.pretty(2)
                ));
            } else {
                formatted.push_str("   Status: Failed\n");
                if let Some(ref error) = result.error {
                    formatted.push_str(&format!("   Error: {error}\n"));
                }
            }

            formatted.push_str(&format!(
                "   Execution time: {time}ms\n\n",
                time = result.execution_time_ms
            ));
        }

        formatted
    }

    /// Create a comprehensive system prompt for GraphRAG function calling
    pub fn create_system_prompt() -> String {
        r#"You are a GraphRAG assistant that can interact with a knowledge graph database through function calls.

Available functions:
1. graph_search: Search for entities by name or partial name match
2. entity_expand: Get all relationships and connected entities for a specific entity
3. relationship_traverse: Find paths between two entities in the graph
4. get_entity_context: Get text chunks where an entity appears for detailed context
5. infer_relationships: Infer implicit relationships based on context patterns and co-occurrence

Guidelines:
- Use function calls to gather information from the knowledge graph before answering
- Start with graph_search to find relevant entities
- Use entity_expand to understand explicit relationships around key entities
- Use infer_relationships to find implicit relationships (friends, enemies, etc.)
- Use relationship_traverse to find connections between entities
- Use get_entity_context to get detailed textual information
- Synthesize information from multiple function calls for comprehensive answers
- Distinguish between explicit (directly stated) and implicit (inferred) relationships
- If no relevant information is found, state that clearly

Remember to make strategic function calls based on the user's question type:
- For entity information: graph_search → entity_expand → get_entity_context
- For explicit relationships: graph_search → relationship_traverse
- For implicit relationships: graph_search → infer_relationships → get_entity_context
- For friendship queries: graph_search → infer_relationships (relation_type: "FRIEND")
- For detailed analysis: graph_search → entity_expand → infer_relationships → get_entity_context

Always explain your reasoning and cite the function call results in your answers."#.to_string()
    }
}

/// Utility for building prompts with function calling context
pub struct PromptBuilder;

impl PromptBuilder {
    /// Build a complete prompt with system message, user query, and function definitions
    pub fn build_function_calling_prompt(
        user_query: &str,
        function_caller: &FunctionCaller,
        previous_results: &[super::FunctionResult],
    ) -> String {
        let mut prompt = String::new();

        // System prompt
        prompt.push_str(&ToolRegistry::create_system_prompt());
        prompt.push_str("\n\n");

        // Available functions
        let function_defs = function_caller.get_function_definitions();
        if !function_defs.is_empty() {
            prompt.push_str("Available functions:\n");
            for def in function_defs {
                prompt.push_str(&format!("- {}: {}\n", def.name, def.description));
            }
            prompt.push('\n');
        }

        // Previous function results if any
        if !previous_results.is_empty() {
            prompt.push_str("Previous function call results:\n");
            prompt.push_str(&ToolRegistry::format_function_results_for_llm(
                previous_results,
            ));
            prompt.push('\n');
        }

        // User query
        prompt.push_str(&format!("User query: {user_query}\n\n"));

        prompt.push_str("Please analyze the query and make appropriate function calls to gather information from the knowledge graph, then provide a comprehensive answer based on the results.");

        prompt
    }

    /// Build a prompt for answer synthesis after function calls
    pub fn build_synthesis_prompt(
        user_query: &str,
        function_results: &[super::FunctionResult],
    ) -> String {
        let mut prompt = String::new();

        prompt.push_str("Based on the following function call results, provide a comprehensive answer to the user's query.\n\n");

        prompt.push_str(&format!("User query: {user_query}\n\n"));

        prompt.push_str(&ToolRegistry::format_function_results_for_llm(
            function_results,
        ));

        prompt.push_str("\nPlease synthesize this information into a clear, comprehensive answer that directly addresses the user's question. ");
        prompt.push_str("Include specific details from the function results and explain any relationships or connections found. ");
        prompt.push_str("If insufficient information was found, state that clearly and suggest what additional information might be helpful.");

        prompt
    }
}

/// Query analysis utilities
pub struct QueryAnalyzer;

impl QueryAnalyzer {
    /// Analyze query type and suggest appropriate function calling strategy
    pub fn analyze_query(query: &str) -> QueryAnalysis {
        let query_lower = query.to_lowercase();

        let is_entity_focused = query_lower.contains("what is")
            || query_lower.contains("who is")
            || query_lower.contains("tell me about");

        let is_relationship_focused = query_lower.contains("relationship")
            || query_lower.contains("connect")
            || query_lower.contains("relation")
            || query_lower.contains("between")
            || query_lower.contains("how are")
            || query_lower.contains("associated");

        let is_context_focused = query_lower.contains("context")
            || query_lower.contains("detail")
            || query_lower.contains("information")
            || query_lower.contains("describe")
            || query_lower.contains("explain");

        let requires_search =
            !query_lower.contains("list all") && !query_lower.contains("show all");

        QueryAnalysis {
            is_entity_focused,
            is_relationship_focused,
            is_context_focused,
            requires_search,
            complexity: if is_relationship_focused {
                QueryComplexity::High
            } else if is_context_focused {
                QueryComplexity::Medium
            } else {
                QueryComplexity::Low
            },
        }
    }

    /// Extract quoted entities from query
    pub fn extract_quoted_entities(query: &str) -> Vec<String> {
        let mut entities = Vec::new();
        let mut in_quotes = false;
        let mut current_entity = String::new();

        for ch in query.chars() {
            match ch {
                '"' | '\'' => {
                    if in_quotes && !current_entity.is_empty() {
                        entities.push(current_entity.trim().to_string());
                        current_entity.clear();
                    }
                    in_quotes = !in_quotes;
                }
                _ if in_quotes => {
                    current_entity.push(ch);
                }
                _ => {}
            }
        }

        entities
    }

    /// Extract potential entity names using capitalization heuristics
    pub fn extract_capitalized_terms(query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .filter(|word| {
                word.len() > 2
                    && word.chars().next().unwrap().is_uppercase()
                    && !word.chars().all(|c| c.is_uppercase()) // Skip all-caps words
            })
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }
}

/// Query analysis result
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub is_entity_focused: bool,
    pub is_relationship_focused: bool,
    pub is_context_focused: bool,
    pub requires_search: bool,
    pub complexity: QueryComplexity,
}

/// Query complexity levels
#[derive(Debug, Clone, PartialEq)]
pub enum QueryComplexity {
    Low,    // Simple entity lookup
    Medium, // Entity with context
    High,   // Multi-entity relationships
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_analysis() {
        let analysis =
            QueryAnalyzer::analyze_query("What is the relationship between John and Mary?");
        assert!(analysis.is_relationship_focused);
        assert_eq!(analysis.complexity, QueryComplexity::High);
    }

    #[test]
    fn test_extract_quoted_entities() {
        let entities =
            QueryAnalyzer::extract_quoted_entities("Tell me about \"John Smith\" and 'Mary Jones'");
        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&"John Smith".to_string()));
        assert!(entities.contains(&"Mary Jones".to_string()));
    }

    #[test]
    fn test_extract_capitalized_terms() {
        let terms =
            QueryAnalyzer::extract_capitalized_terms("John Smith works at Acme Corp in New York");
        assert!(terms.contains(&"John".to_string()));
        assert!(terms.contains(&"Smith".to_string()));
        assert!(terms.contains(&"Acme".to_string()));
    }

    #[test]
    fn test_system_prompt_creation() {
        let prompt = ToolRegistry::create_system_prompt();
        assert!(prompt.contains("GraphRAG assistant"));
        assert!(prompt.contains("function calls"));
    }
}
