//! Function Calling Framework for Dynamic GraphRAG Queries
//!
//! This module implements a function calling system that allows LLMs to interact
//! dynamically with the knowledge graph database through specific functions.

use crate::{core::KnowledgeGraph, GraphRAGError, Result};
use json::JsonValue;
use std::collections::HashMap;

pub mod agent;
pub mod enhanced_registry;
pub mod functions;
pub mod tools;

/// Function definition for the LLM to call
#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,
    /// Description for the LLM
    pub description: String,
    /// JSON schema for parameters
    pub parameters: JsonValue,
    /// Whether this function is required for the task
    pub required: bool,
}

/// Result of a function call
#[derive(Debug, Clone)]
pub struct FunctionResult {
    /// Name of the function that was called
    pub function_name: String,
    /// Arguments passed to the function
    pub arguments: JsonValue,
    /// Result returned by the function
    pub result: JsonValue,
    /// Success status
    pub success: bool,
    /// Error message if any
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Function call request from LLM
#[derive(Debug, Clone)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// Arguments for the function
    pub arguments: JsonValue,
}

/// Trait for implementing callable functions
pub trait CallableFunction: Send + Sync {
    /// Execute the function with given arguments
    fn call(&self, arguments: JsonValue, context: &FunctionContext) -> Result<JsonValue>;

    /// Get function definition for LLM
    fn definition(&self) -> FunctionDefinition;

    /// Validate arguments before execution
    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()>;
}

/// Context provided to functions during execution
#[derive(Debug)]
pub struct FunctionContext<'a> {
    /// Reference to the knowledge graph
    pub knowledge_graph: &'a KnowledgeGraph,
    /// User query that triggered this function call
    pub query: &'a str,
    /// Previous function results in this session
    pub previous_results: &'a [FunctionResult],
    /// Additional context metadata
    pub metadata: HashMap<String, JsonValue>,
}

/// Function calling orchestrator
pub struct FunctionCaller {
    /// Available functions mapped by name
    functions: HashMap<String, Box<dyn CallableFunction>>,
    /// Maximum number of function calls per query
    max_calls_per_query: usize,
    /// Function call history
    call_history: Vec<FunctionResult>,
}

impl FunctionCaller {
    /// Create a new function caller
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            max_calls_per_query: 10,
            call_history: Vec::new(),
        }
    }

    /// Register a new function
    pub fn register_function(&mut self, function: Box<dyn CallableFunction>) {
        let name = function.definition().name.clone();
        self.functions.insert(name, function);
    }

    /// Get all available function definitions
    pub fn get_function_definitions(&self) -> Vec<FunctionDefinition> {
        self.functions.values().map(|f| f.definition()).collect()
    }

    /// Execute a function call
    pub fn call_function(
        &mut self,
        function_call: FunctionCall,
        context: &FunctionContext,
    ) -> Result<FunctionResult> {
        let start_time = std::time::Instant::now();

        // Check if function exists
        let function =
            self.functions
                .get(&function_call.name)
                .ok_or_else(|| GraphRAGError::Generation {
                    message: format!("Function '{}' not found", function_call.name),
                })?;

        // Validate arguments
        if let Err(e) = function.validate_arguments(&function_call.arguments) {
            return Ok(FunctionResult {
                function_name: function_call.name,
                arguments: function_call.arguments,
                result: JsonValue::Null,
                success: false,
                error: Some(e.to_string()),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            });
        }

        // Execute function
        let result = match function.call(function_call.arguments.clone(), context) {
            Ok(result) => FunctionResult {
                function_name: function_call.name.clone(),
                arguments: function_call.arguments,
                result,
                success: true,
                error: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            },
            Err(e) => FunctionResult {
                function_name: function_call.name,
                arguments: function_call.arguments,
                result: JsonValue::Null,
                success: false,
                error: Some(e.to_string()),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
            },
        };

        // Store in history
        self.call_history.push(result.clone());

        Ok(result)
    }

    /// Execute multiple function calls in sequence
    pub fn call_functions(
        &mut self,
        function_calls: Vec<FunctionCall>,
        context: &FunctionContext,
    ) -> Result<Vec<FunctionResult>> {
        if function_calls.len() > self.max_calls_per_query {
            return Err(GraphRAGError::Generation {
                message: format!(
                    "Too many function calls requested: {} (max: {})",
                    function_calls.len(),
                    self.max_calls_per_query
                ),
            });
        }

        let mut results = Vec::new();
        for call in function_calls {
            let result = self.call_function(call, context)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get function call history
    pub fn get_call_history(&self) -> &[FunctionResult] {
        &self.call_history
    }

    /// Clear function call history
    pub fn clear_history(&mut self) {
        self.call_history.clear();
    }

    /// Get statistics about function usage
    pub fn get_statistics(&self) -> FunctionCallStatistics {
        let total_calls = self.call_history.len();
        let successful_calls = self.call_history.iter().filter(|r| r.success).count();
        let failed_calls = total_calls - successful_calls;

        let total_execution_time: u64 = self.call_history.iter().map(|r| r.execution_time_ms).sum();

        let average_execution_time = if total_calls > 0 {
            total_execution_time / total_calls as u64
        } else {
            0
        };

        let mut function_usage = HashMap::new();
        for result in &self.call_history {
            *function_usage
                .entry(result.function_name.clone())
                .or_insert(0) += 1;
        }

        FunctionCallStatistics {
            total_calls,
            successful_calls,
            failed_calls,
            total_execution_time_ms: total_execution_time,
            average_execution_time_ms: average_execution_time,
            function_usage,
        }
    }

    /// Set maximum calls per query
    pub fn set_max_calls_per_query(&mut self, max_calls: usize) {
        self.max_calls_per_query = max_calls;
    }
}

impl Default for FunctionCaller {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about function call usage
#[derive(Debug, Clone)]
pub struct FunctionCallStatistics {
    pub total_calls: usize,
    pub successful_calls: usize,
    pub failed_calls: usize,
    pub total_execution_time_ms: u64,
    pub average_execution_time_ms: u64,
    pub function_usage: HashMap<String, usize>,
}

impl FunctionCallStatistics {
    /// Print statistics in a readable format
    pub fn print(&self) {
        println!("🔧 Function Call Statistics");
        println!("{}", "=".repeat(30));
        println!("Total calls: {}", self.total_calls);
        println!("Successful: {}", self.successful_calls);
        println!("Failed: {}", self.failed_calls);

        if self.total_calls > 0 {
            let success_rate = (self.successful_calls as f64 / self.total_calls as f64) * 100.0;
            println!("Success rate: {success_rate:.1}%");
        }

        println!("Total execution time: {}ms", self.total_execution_time_ms);
        println!(
            "Average execution time: {}ms",
            self.average_execution_time_ms
        );

        if !self.function_usage.is_empty() {
            println!("\nFunction usage:");
            let mut usage_vec: Vec<_> = self.function_usage.iter().collect();
            usage_vec.sort_by(|a, b| b.1.cmp(a.1));
            for (function, count) in usage_vec {
                println!("  {function}: {count} calls");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::KnowledgeGraph;

    struct MockFunction {
        name: String,
    }

    impl CallableFunction for MockFunction {
        fn call(&self, arguments: JsonValue, _context: &FunctionContext) -> Result<JsonValue> {
            Ok(json::object! {
                "function": self.name.clone(),
                "arguments": arguments,
                "result": "mock_result"
            })
        }

        fn definition(&self) -> FunctionDefinition {
            FunctionDefinition {
                name: self.name.clone(),
                description: "Mock function for testing".to_string(),
                parameters: json::object! {
                    "type": "object",
                    "properties": {
                        "test_param": {
                            "type": "string",
                            "description": "Test parameter"
                        }
                    },
                    "required": ["test_param"]
                },
                required: false,
            }
        }

        fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
            if arguments["test_param"].is_null() {
                return Err(GraphRAGError::Generation {
                    message: "test_param is required".to_string(),
                });
            }
            Ok(())
        }
    }

    #[test]
    fn test_function_caller_creation() {
        let caller = FunctionCaller::new();
        assert_eq!(caller.get_function_definitions().len(), 0);
    }

    #[test]
    fn test_function_registration() {
        let mut caller = FunctionCaller::new();
        let mock_function = Box::new(MockFunction {
            name: "test_function".to_string(),
        });

        caller.register_function(mock_function);
        assert_eq!(caller.get_function_definitions().len(), 1);
    }

    #[test]
    fn test_function_call() {
        let mut caller = FunctionCaller::new();
        let mock_function = Box::new(MockFunction {
            name: "test_function".to_string(),
        });

        caller.register_function(mock_function);

        let graph = KnowledgeGraph::new();
        let context = FunctionContext {
            knowledge_graph: &graph,
            query: "test query",
            previous_results: &[],
            metadata: HashMap::new(),
        };

        let function_call = FunctionCall {
            name: "test_function".to_string(),
            arguments: json::object! {
                "test_param": "test_value"
            },
        };

        let result = caller.call_function(function_call, &context).unwrap();
        assert!(result.success);
        assert_eq!(result.function_name, "test_function");
    }
}
