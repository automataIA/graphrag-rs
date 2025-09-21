//! Enhanced tool registry with dynamic function management

use super::{CallableFunction, FunctionCaller, FunctionDefinition};
use crate::{GraphRAGError, Result};
use json::JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Enhanced tool registry with dynamic capabilities
pub struct EnhancedToolRegistry {
    function_caller: Arc<Mutex<FunctionCaller>>,
    _custom_functions: HashMap<String, Box<dyn CallableFunction>>,
    function_categories: HashMap<String, Vec<String>>,
    usage_statistics: HashMap<String, usize>,
}

impl EnhancedToolRegistry {
    /// Create a new enhanced tool registry
    pub fn new() -> Self {
        Self {
            function_caller: Arc::new(Mutex::new(FunctionCaller::new())),
            _custom_functions: HashMap::new(),
            function_categories: HashMap::new(),
            usage_statistics: HashMap::new(),
        }
    }

    /// Register all default GraphRAG functions
    pub fn register_default_functions(&mut self) -> Result<()> {
        // Register core search functions
        self.register_function_in_category(
            Box::new(EnhancedGraphSearchFunction),
            "search".to_string(),
        )?;

        self.register_function_in_category(
            Box::new(EnhancedEntityExpandFunction),
            "entity".to_string(),
        )?;

        self.register_function_in_category(
            Box::new(EnhancedRelationshipFunction),
            "relationship".to_string(),
        )?;

        // Register analysis functions
        self.register_function_in_category(
            Box::new(ContextAnalysisFunction),
            "analysis".to_string(),
        )?;

        self.register_function_in_category(
            Box::new(SummaryGenerationFunction),
            "generation".to_string(),
        )?;

        Ok(())
    }

    /// Register a function in a specific category
    pub fn register_function_in_category(
        &mut self,
        function: Box<dyn CallableFunction>,
        category: String,
    ) -> Result<()> {
        let function_name = function.definition().name.clone();

        // Register with function caller
        {
            let mut caller = self.function_caller.lock().unwrap();
            caller.register_function(function);
        }

        // Add to category
        self.function_categories
            .entry(category)
            .or_default()
            .push(function_name.clone());

        // Initialize usage statistics
        self.usage_statistics.insert(function_name, 0);

        Ok(())
    }

    /// Get functions by category
    pub fn get_functions_by_category(&self, category: &str) -> Vec<String> {
        self.function_categories
            .get(category)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all available categories
    pub fn get_categories(&self) -> Vec<String> {
        self.function_categories.keys().cloned().collect()
    }

    /// Record function usage
    pub fn record_function_usage(&mut self, function_name: &str) {
        *self
            .usage_statistics
            .entry(function_name.to_string())
            .or_insert(0) += 1;
    }

    /// Get usage statistics
    pub fn get_usage_statistics(&self) -> &HashMap<String, usize> {
        &self.usage_statistics
    }

    /// Get function definitions for a category
    pub fn get_category_definitions(&self, category: &str) -> Vec<FunctionDefinition> {
        let function_names = self.get_functions_by_category(category);
        let caller = self.function_caller.lock().unwrap();
        let all_definitions = caller.get_function_definitions();

        all_definitions
            .into_iter()
            .filter(|def| function_names.contains(&def.name))
            .collect()
    }

    /// Get function definitions in OpenAI format for a category
    pub fn get_openai_definitions_for_category(&self, category: &str) -> JsonValue {
        let definitions = self.get_category_definitions(category);

        let function_objects: Vec<_> = definitions
            .into_iter()
            .map(|def| {
                json::object! {
                    "name": def.name,
                    "description": def.description,
                    "parameters": def.parameters
                }
            })
            .collect();

        JsonValue::Array(function_objects)
    }

    /// Get recommended functions based on query type
    pub fn get_recommended_functions(&self, query_type: &str) -> Vec<String> {
        match query_type.to_lowercase().as_str() {
            "entity" => self.get_functions_by_category("entity"),
            "search" => self.get_functions_by_category("search"),
            "relationship" => self.get_functions_by_category("relationship"),
            "analysis" => self.get_functions_by_category("analysis"),
            _ => {
                // Return most used functions
                let mut sorted_functions: Vec<_> = self.usage_statistics.iter().collect();
                sorted_functions.sort_by(|a, b| b.1.cmp(a.1));
                sorted_functions
                    .into_iter()
                    .take(5)
                    .map(|(name, _)| name.clone())
                    .collect()
            }
        }
    }

    /// Get a mutable reference to the function caller
    pub fn get_function_caller(&self) -> Arc<Mutex<FunctionCaller>> {
        Arc::clone(&self.function_caller)
    }
}

impl Default for EnhancedToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced graph search function with better context handling
pub struct EnhancedGraphSearchFunction;

impl CallableFunction for EnhancedGraphSearchFunction {
    fn call(&self, arguments: JsonValue, context: &super::FunctionContext) -> Result<JsonValue> {
        let query = arguments["query"]
            .as_str()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Query parameter required".to_string(),
            })?;

        let limit = arguments["limit"].as_usize().unwrap_or(10);

        // Enhanced search with entity-aware filtering
        let entities = context.knowledge_graph.find_entities_by_name(query);
        let mut results = Vec::new();

        for entity in entities.take(limit) {
            results.push(json::object! {
                "entity_id": entity.id.to_string(),
                "entity_name": entity.name.clone(),
                "entity_type": entity.entity_type.clone(),
                "confidence": entity.confidence,
                "mentions": entity.mentions.len()
            });
        }

        let total_found = results.len();
        Ok(json::object! {
            "results": results,
            "total_found": total_found,
            "query": query
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "enhanced_graph_search".to_string(),
            description: "Search the knowledge graph for entities with enhanced context"
                .to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for entities"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            required: true,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["query"].is_null() {
            return Err(GraphRAGError::Generation {
                message: "Query parameter is required".to_string(),
            });
        }
        Ok(())
    }
}

/// Enhanced entity expansion function
pub struct EnhancedEntityExpandFunction;

impl CallableFunction for EnhancedEntityExpandFunction {
    fn call(&self, arguments: JsonValue, context: &super::FunctionContext) -> Result<JsonValue> {
        let entity_id =
            arguments["entity_id"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "Entity ID parameter required".to_string(),
                })?;

        let expand_depth = arguments["depth"].as_usize().unwrap_or(1);

        // Find entity and expand with relationships
        if let Some(entity) = context.knowledge_graph.get_entity_by_id(entity_id) {
            let relationships = context
                .knowledge_graph
                .get_entity_relationships(entity_id)
                .take(20)
                .collect::<Vec<_>>();

            let mut expanded_entities = Vec::new();

            // Add direct relationships
            for relationship in &relationships {
                if expand_depth > 1 {
                    // Recursively expand
                    let related_entity_id = if relationship.source == entity.id {
                        &relationship.target
                    } else {
                        &relationship.source
                    };

                    if let Some(related_entity) = context
                        .knowledge_graph
                        .get_entity_by_id(&related_entity_id.to_string())
                    {
                        expanded_entities.push(json::object! {
                            "entity_id": related_entity.id.to_string(),
                            "entity_name": related_entity.name.clone(),
                            "relationship": relationship.relation_type.clone(),
                            "confidence": related_entity.confidence
                        });
                    }
                }
            }

            Ok(json::object! {
                "entity": {
                    "id": entity.id.to_string(),
                    "name": entity.name.clone(),
                    "type": entity.entity_type.clone(),
                },
                "relationships": relationships.len(),
                "expanded_entities": expanded_entities,
                "expansion_depth": expand_depth
            })
        } else {
            Err(GraphRAGError::Generation {
                message: format!("Entity not found: {entity_id}"),
            })
        }
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "enhanced_entity_expand".to_string(),
            description: "Expand an entity with its relationships and connected entities"
                .to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "ID of the entity to expand"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Expansion depth (1-3)",
                        "default": 1
                    }
                },
                "required": ["entity_id"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["entity_id"].is_null() {
            return Err(GraphRAGError::Generation {
                message: "Entity ID parameter is required".to_string(),
            });
        }
        Ok(())
    }
}

/// Enhanced relationship analysis function
pub struct EnhancedRelationshipFunction;

impl CallableFunction for EnhancedRelationshipFunction {
    fn call(&self, arguments: JsonValue, context: &super::FunctionContext) -> Result<JsonValue> {
        let entity1 = arguments["entity1"]
            .as_str()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Entity1 parameter required".to_string(),
            })?;

        let entity2 = arguments["entity2"]
            .as_str()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Entity2 parameter required".to_string(),
            })?;

        // Find relationship path between entities
        let relationships = context
            .knowledge_graph
            .find_relationship_path(entity1, entity2, 3);
        let path_length = relationships.len();
        let has_direct_relationship = !relationships.is_empty();

        Ok(json::object! {
            "entity1": entity1,
            "entity2": entity2,
            "direct_relationship": has_direct_relationship,
            "relationship_path": relationships,
            "path_length": path_length
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "enhanced_relationship_analysis".to_string(),
            description: "Analyze relationships between two entities".to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "entity1": {
                        "type": "string",
                        "description": "First entity name or ID"
                    },
                    "entity2": {
                        "type": "string",
                        "description": "Second entity name or ID"
                    }
                },
                "required": ["entity1", "entity2"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["entity1"].is_null() || arguments["entity2"].is_null() {
            return Err(GraphRAGError::Generation {
                message: "Both entity1 and entity2 parameters are required".to_string(),
            });
        }
        Ok(())
    }
}

/// Context analysis function for understanding query context
pub struct ContextAnalysisFunction;

impl CallableFunction for ContextAnalysisFunction {
    fn call(&self, arguments: JsonValue, _context: &super::FunctionContext) -> Result<JsonValue> {
        let query = arguments["query"]
            .as_str()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "Query parameter required".to_string(),
            })?;

        // Analyze query context
        let word_count = query.split_whitespace().count();
        let has_question_words = query.to_lowercase().contains("who")
            || query.to_lowercase().contains("what")
            || query.to_lowercase().contains("where")
            || query.to_lowercase().contains("when")
            || query.to_lowercase().contains("how");

        let complexity = if word_count > 10 {
            "high"
        } else if word_count > 5 {
            "medium"
        } else {
            "low"
        };

        Ok(json::object! {
            "query": query,
            "word_count": word_count,
            "complexity": complexity,
            "has_question_words": has_question_words,
            "recommended_strategy": if has_question_words { "entity_search" } else { "vector_search" }
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "context_analysis".to_string(),
            description: "Analyze query context and recommend search strategy".to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to analyze"
                    }
                },
                "required": ["query"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["query"].is_null() {
            return Err(GraphRAGError::Generation {
                message: "Query parameter is required".to_string(),
            });
        }
        Ok(())
    }
}

/// Summary generation function
pub struct SummaryGenerationFunction;

impl CallableFunction for SummaryGenerationFunction {
    fn call(&self, arguments: JsonValue, context: &super::FunctionContext) -> Result<JsonValue> {
        let entity_ids = if arguments["entity_ids"].is_array() {
            &arguments["entity_ids"]
        } else {
            return Err(GraphRAGError::Generation {
                message: "Entity IDs array required".to_string(),
            });
        };

        let max_length = arguments["max_length"].as_usize().unwrap_or(200);

        let mut summary_parts = Vec::new();

        for i in 0..entity_ids.len() {
            if let Some(entity_id) = entity_ids[i].as_str() {
                if let Some(entity) = context.knowledge_graph.get_entity_by_id(entity_id) {
                    summary_parts.push(format!("{} ({})", entity.name, entity.entity_type));
                }
            }
        }

        let summary = summary_parts.join(", ");
        let truncated_summary = if summary.len() > max_length {
            format!("{}...", &summary[..max_length - 3])
        } else {
            summary
        };
        let summary_length = truncated_summary.len();

        Ok(json::object! {
            "summary": truncated_summary,
            "entity_count": entity_ids.len(),
            "length": summary_length
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "summary_generation".to_string(),
            description: "Generate a summary from a list of entities".to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity IDs to summarize"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum summary length",
                        "default": 200
                    }
                },
                "required": ["entity_ids"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if !arguments["entity_ids"].is_array() {
            return Err(GraphRAGError::Generation {
                message: "Entity IDs must be an array".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_registry_creation() {
        let registry = EnhancedToolRegistry::new();
        assert_eq!(registry.get_categories().len(), 0);
    }

    #[test]
    fn test_category_management() {
        let mut registry = EnhancedToolRegistry::new();

        let test_function = Box::new(ContextAnalysisFunction);
        registry
            .register_function_in_category(test_function, "test".to_string())
            .unwrap();

        assert_eq!(registry.get_categories().len(), 1);
        assert_eq!(registry.get_functions_by_category("test").len(), 1);
    }

    #[test]
    fn test_usage_statistics() {
        let mut registry = EnhancedToolRegistry::new();
        registry.record_function_usage("test_function");

        assert_eq!(
            registry.get_usage_statistics().get("test_function"),
            Some(&1)
        );
    }
}
