//! Concrete function implementations for GraphRAG

use super::{CallableFunction, FunctionContext, FunctionDefinition};
use crate::core::Entity;
use crate::inference::{InferenceConfig, InferenceEngine};
use crate::{GraphRAGError, Result};
use json::JsonValue;
use std::collections::HashSet;

/// Enhanced entity resolution utilities
struct EntityResolver;

impl EntityResolver {
    /// Find entity by name with fuzzy matching
    fn find_entity_by_name<'a>(
        knowledge_graph: &'a crate::core::KnowledgeGraph,
        name: &str,
    ) -> Option<&'a Entity> {
        let name_lower = name.to_lowercase().trim().to_string();

        // Direct exact match first
        if let Some(entity) = knowledge_graph
            .entities()
            .find(|e| e.name.to_lowercase().trim() == name_lower)
        {
            return Some(entity);
        }

        // Partial match
        if let Some(entity) = knowledge_graph.entities().find(|e| {
            e.name.to_lowercase().contains(&name_lower)
                || name_lower.contains(&e.name.to_lowercase())
        }) {
            return Some(entity);
        }

        // Fuzzy match for common variations
        knowledge_graph
            .entities()
            .find(|e| Self::fuzzy_name_match(&e.name, name))
    }

    /// Fuzzy matching for name variations
    fn fuzzy_name_match(entity_name: &str, query_name: &str) -> bool {
        let entity_lower = entity_name.to_lowercase();
        let query_lower = query_name.to_lowercase();

        // Handle common variations
        let entity_parts: Vec<&str> = entity_lower.split_whitespace().collect();
        let query_parts: Vec<&str> = query_lower.split_whitespace().collect();

        // Check if first names match (for "Tom" -> "Entity Name")
        if query_parts.len() == 1 && !entity_parts.is_empty() {
            return entity_parts[0].starts_with(query_parts[0])
                || query_parts[0].starts_with(entity_parts[0]);
        }

        // Check if last names match
        if let (Some(entity_last), Some(query_last)) = (entity_parts.last(), query_parts.last()) {
            if entity_last == query_last {
                return true;
            }
        }

        // Levenshtein distance check for typos (simplified)
        let distance = Self::simple_edit_distance(&entity_lower, &query_lower);
        let max_allowed = (query_lower.len().min(entity_lower.len()) / 4).max(1);
        distance <= max_allowed
    }

    /// Simple edit distance calculation
    fn simple_edit_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate() {
            row[0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = *[
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost,
                ]
                .iter()
                .min()
                .unwrap();
            }
        }

        matrix[len1][len2]
    }
}

/// Search for entities in the knowledge graph
pub struct GraphSearchFunction;

impl CallableFunction for GraphSearchFunction {
    fn call(&self, arguments: JsonValue, context: &FunctionContext) -> Result<JsonValue> {
        let entity_name =
            arguments["entity_name"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "entity_name parameter is required".to_string(),
                })?;

        let limit = arguments["limit"].as_usize().unwrap_or(10);

        // Enhanced search with multiple matching strategies
        let mut matching_entities = Vec::new();
        let mut seen_ids = HashSet::new();

        // 1. Exact name match (highest priority)
        for entity in context.knowledge_graph.entities() {
            if entity.name.to_lowercase().trim() == entity_name.to_lowercase().trim()
                && seen_ids.insert(entity.id.to_string())
            {
                matching_entities.push((entity, 1.0)); // Perfect match score
            }
        }

        // 2. Contains match (medium priority)
        if matching_entities.len() < limit {
            for entity in context.knowledge_graph.entities() {
                if (entity
                    .name
                    .to_lowercase()
                    .contains(&entity_name.to_lowercase())
                    || entity_name
                        .to_lowercase()
                        .contains(&entity.name.to_lowercase()))
                    && seen_ids.insert(entity.id.to_string())
                {
                    matching_entities.push((entity, 0.8)); // Good match score
                }
            }
        }

        // 3. Fuzzy match (lower priority)
        if matching_entities.len() < limit {
            for entity in context.knowledge_graph.entities() {
                if EntityResolver::fuzzy_name_match(&entity.name, entity_name)
                    && seen_ids.insert(entity.id.to_string())
                {
                    matching_entities.push((entity, 0.6)); // Fuzzy match score
                }
            }
        }

        // Sort by relevance score and limit results
        matching_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matching_entities.truncate(limit);

        let results: Vec<JsonValue> = matching_entities
            .into_iter()
            .map(|(entity, relevance)| {
                json::object! {
                    "id": entity.id.to_string(),
                    "name": entity.name.clone(),
                    "type": entity.entity_type.clone(),
                    "confidence": entity.confidence,
                    "relevance_score": relevance,
                    "mentions_count": entity.mentions.len()
                }
            })
            .collect();

        Ok(json::object! {
            "entities": results.clone(),
            "total_found": results.len(),
            "query": entity_name,
            "limit": limit
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "graph_search".to_string(),
            description: "Search for entities in the knowledge graph by name or partial name match"
                .to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name or partial name of the entity to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["entity_name"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["entity_name"].as_str().is_none() {
            return Err(GraphRAGError::Generation {
                message: "entity_name must be a string".to_string(),
            });
        }

        if let Some(limit) = arguments["limit"].as_number() {
            let limit_val = limit.as_parts().0 as i64;
            if limit_val <= 0 || limit_val > 100 {
                return Err(GraphRAGError::Generation {
                    message: "limit must be between 1 and 100".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Expand entity information by finding related entities
pub struct EntityExpandFunction;

impl CallableFunction for EntityExpandFunction {
    fn call(&self, arguments: JsonValue, context: &FunctionContext) -> Result<JsonValue> {
        let entity_id =
            arguments["entity_id"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "entity_id parameter is required".to_string(),
                })?;

        let depth = arguments["depth"].as_usize().unwrap_or(1);
        let limit = arguments["limit"].as_usize().unwrap_or(20);

        // Find the entity
        let entity = context
            .knowledge_graph
            .entities()
            .find(|e| e.id.to_string() == entity_id)
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Entity with id '{entity_id}' not found"),
            })?;

        // Get relationships for this entity
        let relationships: Vec<_> = context
            .knowledge_graph
            .get_all_relationships()
            .into_iter()
            .filter(|rel| {
                rel.source.to_string() == entity_id || rel.target.to_string() == entity_id
            })
            .take(limit)
            .map(|rel| {
                let is_source = rel.source.to_string() == entity_id;
                let related_entity_id = if is_source { &rel.target } else { &rel.source };

                // Find the related entity
                let related_entity = context
                    .knowledge_graph
                    .entities()
                    .find(|e| &e.id == related_entity_id);

                json::object! {
                    "relationship_type": rel.relation_type.clone(),
                    "direction": if is_source { "outgoing" } else { "incoming" },
                    "related_entity": if let Some(related) = related_entity {
                        json::object! {
                            "id": related.id.to_string(),
                            "name": related.name.clone(),
                            "type": related.entity_type.clone()
                        }
                    } else {
                        JsonValue::Null
                    },
                    "confidence": rel.confidence,
                    "context_chunks": rel.context.len()
                }
            })
            .collect();

        let relationships_len = relationships.len();

        Ok(json::object! {
            "entity": {
                "id": entity.id.to_string(),
                "name": entity.name.clone(),
                "type": entity.entity_type.clone(),
                "confidence": entity.confidence
            },
            "relationships": relationships,
            "total_relationships": relationships_len,
            "depth": depth,
            "limit": limit
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "entity_expand".to_string(),
            description: "Expand an entity by finding all its relationships and connected entities"
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
                        "description": "Depth of expansion (how many hops away from the entity, default: 1)",
                        "default": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of relationships to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["entity_id"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["entity_id"].as_str().is_none() {
            return Err(GraphRAGError::Generation {
                message: "entity_id must be a string".to_string(),
            });
        }

        if let Some(depth) = arguments["depth"].as_number() {
            let depth_val = depth.as_parts().0 as i64;
            if depth_val <= 0 || depth_val > 5 {
                return Err(GraphRAGError::Generation {
                    message: "depth must be between 1 and 5".to_string(),
                });
            }
        }

        if let Some(limit) = arguments["limit"].as_number() {
            let limit_val = limit.as_parts().0 as i64;
            if limit_val <= 0 || limit_val > 100 {
                return Err(GraphRAGError::Generation {
                    message: "limit must be between 1 and 100".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Traverse relationships between entities
pub struct RelationshipTraverseFunction;

impl CallableFunction for RelationshipTraverseFunction {
    fn call(&self, arguments: JsonValue, context: &FunctionContext) -> Result<JsonValue> {
        let source_entity =
            arguments["source_entity"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "source_entity parameter is required".to_string(),
                })?;

        let target_entity =
            arguments["target_entity"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "target_entity parameter is required".to_string(),
                })?;

        let max_hops = arguments["max_hops"].as_usize().unwrap_or(3);

        // Find paths between entities using simple BFS
        let paths = self.find_paths(context, source_entity, target_entity, max_hops)?;

        Ok(json::object! {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "paths_found": paths.len(),
            "max_hops": max_hops,
            "paths": paths
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "relationship_traverse".to_string(),
            description: "Find relationship paths between two entities in the knowledge graph"
                .to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "Name or ID of the source entity"
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "Name or ID of the target entity"
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of hops to traverse (default: 3)",
                        "default": 3
                    }
                },
                "required": ["source_entity", "target_entity"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        // Check for common parameter name mistakes
        if arguments["entity_id_1"].is_string()
            || arguments["entity1_id"].is_string()
            || arguments["entity_id_2"].is_string()
            || arguments["entity2_id"].is_string()
        {
            return Err(GraphRAGError::Generation {
                message: "PARAMETER ERROR: Use 'source_entity' and 'target_entity' (not 'entity_id_1', 'entity1_id', etc.)".to_string(),
            });
        }

        if arguments["source_entity"].as_str().is_none() {
            return Err(GraphRAGError::Generation {
                message: "REQUIRED PARAMETER: 'source_entity' must be a string".to_string(),
            });
        }

        if arguments["target_entity"].as_str().is_none() {
            return Err(GraphRAGError::Generation {
                message: "REQUIRED PARAMETER: 'target_entity' must be a string".to_string(),
            });
        }

        if let Some(max_hops) = arguments["max_hops"].as_number() {
            let hops_val = max_hops.as_parts().0 as i64;
            if hops_val <= 0 || hops_val > 10 {
                return Err(GraphRAGError::Generation {
                    message: "PARAMETER ERROR: 'max_hops' must be between 1 and 10".to_string(),
                });
            }
        }

        Ok(())
    }
}

impl RelationshipTraverseFunction {
    fn find_paths(
        &self,
        context: &FunctionContext,
        source: &str,
        target: &str,
        max_hops: usize,
    ) -> Result<Vec<JsonValue>> {
        // Find source and target entities
        let source_entity = self.find_entity_by_name_or_id(context, source)?;
        let target_entity = self.find_entity_by_name_or_id(context, target)?;

        if source_entity.id == target_entity.id {
            return Ok(vec![json::object! {
                "path": [source_entity.id.to_string()],
                "length": 0,
                "relationships": []
            }]);
        }

        // Simple BFS to find paths
        let mut queue = vec![(source_entity.id.clone(), vec![], vec![])];
        let mut visited = HashSet::new();
        let mut paths = Vec::new();

        while let Some((current_id, path, relationships)) = queue.pop() {
            if path.len() >= max_hops {
                continue;
            }

            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            // Check if we reached the target
            if current_id == target_entity.id {
                let mut full_path = path.clone();
                full_path.push(current_id.to_string());
                paths.push(json::object! {
                    "path": full_path,
                    "length": path.len(),
                    "relationships": relationships
                });
                continue;
            }

            // Find connected entities
            for relationship in context.knowledge_graph.get_all_relationships() {
                let next_entity_id = if relationship.source == current_id {
                    Some(&relationship.target)
                } else if relationship.target == current_id {
                    Some(&relationship.source)
                } else {
                    None
                };

                if let Some(next_id) = next_entity_id {
                    if !visited.contains(next_id) {
                        let mut new_path = path.clone();
                        new_path.push(current_id.to_string());

                        let mut new_relationships = relationships.clone();
                        new_relationships.push(json::object! {
                            "type": relationship.relation_type.clone(),
                            "confidence": relationship.confidence,
                            "from": current_id.to_string(),
                            "to": next_id.to_string()
                        });

                        queue.push((next_id.clone(), new_path, new_relationships));
                    }
                }
            }
        }

        // Sort paths by length (shortest first)
        paths.sort_by(|a, b| {
            a["length"]
                .as_usize()
                .unwrap_or(0)
                .cmp(&b["length"].as_usize().unwrap_or(0))
        });

        Ok(paths)
    }

    fn find_entity_by_name_or_id<'a>(
        &self,
        context: &'a FunctionContext,
        name_or_id: &str,
    ) -> Result<&'a crate::core::Entity> {
        // First try to find by exact ID match
        if let Some(entity) = context
            .knowledge_graph
            .entities()
            .find(|e| e.id.to_string() == name_or_id)
        {
            return Ok(entity);
        }

        // Then try to find by exact name match
        if let Some(entity) = context
            .knowledge_graph
            .entities()
            .find(|e| e.name == name_or_id)
        {
            return Ok(entity);
        }

        // Finally try partial name match
        context
            .knowledge_graph
            .entities()
            .find(|e| e.name.to_lowercase().contains(&name_or_id.to_lowercase()))
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Entity '{name_or_id}' not found"),
            })
    }
}

/// Get context chunks for entities
pub struct GetEntityContextFunction;

impl CallableFunction for GetEntityContextFunction {
    fn call(&self, arguments: JsonValue, context: &FunctionContext) -> Result<JsonValue> {
        let entity_id =
            arguments["entity_id"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "entity_id parameter is required".to_string(),
                })?;

        let limit = arguments["limit"].as_usize().unwrap_or(5);

        // Find the entity
        let entity = context
            .knowledge_graph
            .entities()
            .find(|e| e.id.to_string() == entity_id)
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Entity with id '{entity_id}' not found"),
            })?;

        // Get context chunks where this entity appears
        let chunks: Vec<_> = context
            .knowledge_graph
            .chunks()
            .filter(|chunk| chunk.entities.contains(&entity.id))
            .take(limit)
            .map(|chunk| {
                json::object! {
                    "id": chunk.id.to_string(),
                    "content": chunk.content.clone(),
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "document_id": chunk.document_id.to_string(),
                    "entities_count": chunk.entities.len()
                }
            })
            .collect();

        // Get entity mentions with their positions
        let mentions: Vec<_> = entity
            .mentions
            .iter()
            .map(|mention| {
                json::object! {
                    "chunk_id": mention.chunk_id.to_string(),
                    "start_offset": mention.start_offset,
                    "end_offset": mention.end_offset,
                    "confidence": mention.confidence
                }
            })
            .collect();

        let chunks_len = chunks.len();
        let mentions_len = mentions.len();

        Ok(json::object! {
            "entity": {
                "id": entity.id.to_string(),
                "name": entity.name.clone(),
                "type": entity.entity_type.clone()
            },
            "context_chunks": chunks,
            "mentions": mentions,
            "total_chunks": chunks_len,
            "total_mentions": mentions_len,
            "limit": limit
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "get_entity_context".to_string(),
            description:
                "Get text chunks and mentions where an entity appears for detailed context"
                    .to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "ID of the entity to get context for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of context chunks to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["entity_id"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        if arguments["entity_id"].as_str().is_none() {
            return Err(GraphRAGError::Generation {
                message: "entity_id must be a string".to_string(),
            });
        }

        if let Some(limit) = arguments["limit"].as_number() {
            let limit_val = limit.as_parts().0 as i64;
            if limit_val <= 0 || limit_val > 50 {
                return Err(GraphRAGError::Generation {
                    message: "limit must be between 1 and 50".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Function for inferring implicit relationships
pub struct InferRelationshipsFunction {
    inference_engine: InferenceEngine,
}

impl InferRelationshipsFunction {
    pub fn new() -> Self {
        let config = InferenceConfig::default();
        Self {
            inference_engine: InferenceEngine::new(config),
        }
    }
}

impl Default for InferRelationshipsFunction {
    fn default() -> Self {
        Self::new()
    }
}

impl CallableFunction for InferRelationshipsFunction {
    fn call(&self, arguments: JsonValue, context: &FunctionContext) -> Result<JsonValue> {
        let entity_name =
            arguments["entity_name"]
                .as_str()
                .ok_or_else(|| GraphRAGError::Generation {
                    message: "entity_name is required".to_string(),
                })?;

        let relation_type = arguments["relation_type"].as_str().unwrap_or("FRIEND");
        let min_confidence = arguments["min_confidence"]
            .as_f64()
            .map(|f| f as f32)
            .unwrap_or(0.3);

        // Enhanced entity finding with multiple strategies
        let entity = EntityResolver::find_entity_by_name(context.knowledge_graph, entity_name)
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!(
                    "Entity '{}' not found. Available entities: {}",
                    entity_name,
                    context
                        .knowledge_graph
                        .entities()
                        .take(5)
                        .map(|e| e.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })?;

        // Execute inference
        let inferred_relations = self.inference_engine.infer_relationships(
            &entity.id,
            relation_type,
            context.knowledge_graph,
        );

        // Filter by confidence
        let filtered_relations: Vec<_> = inferred_relations
            .into_iter()
            .filter(|r| r.confidence >= min_confidence)
            .collect();

        // Format results
        let results = filtered_relations
            .into_iter()
            .map(|relation| {
                let target_name = context
                    .knowledge_graph
                    .entities()
                    .find(|e| e.id == relation.target)
                    .map(|e| e.name.clone())
                    .unwrap_or_else(|| relation.target.to_string());

                json::object! {
                    "entity": target_name,
                    "relation_type": relation.relation_type,
                    "confidence": relation.confidence,
                    "evidence_count": relation.evidence_count
                }
            })
            .collect::<Vec<_>>();

        let total_found = results.len();

        Ok(json::object! {
            "entity": entity_name,
            "relation_type": relation_type,
            "inferred_relationships": results,
            "total_found": total_found,
            "min_confidence_threshold": min_confidence
        })
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: "infer_relationships".to_string(),
            description: "Infer implicit relationships between entities based on context patterns, co-occurrence, and interaction indicators".to_string(),
            parameters: json::object! {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity to find relationships for"
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Type of relationship to infer (FRIEND, ENEMY, ALLY, FAMILY, etc.)",
                        "default": "FRIEND"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold for relationships (0.0-1.0)",
                        "default": 0.3
                    }
                },
                "required": ["entity_name"]
            },
            required: false,
        }
    }

    fn validate_arguments(&self, arguments: &JsonValue) -> Result<()> {
        // Check for common parameter name mistakes
        if arguments["entity_id"].is_string()
            || arguments["entity_id_1"].is_string()
            || arguments["entity1_id"].is_string()
        {
            return Err(GraphRAGError::Generation {
                message: "PARAMETER ERROR: Use 'entity_name' (not 'entity_id', 'entity_id_1', or 'entity1_id')".to_string(),
            });
        }

        if arguments["relationship_type"].is_string() {
            return Err(GraphRAGError::Generation {
                message: "PARAMETER ERROR: Use 'relation_type' (not 'relationship_type')".to_string(),
            });
        }

        if arguments["entity_name"].as_str().is_none() {
            return Err(GraphRAGError::Generation {
                message: "REQUIRED PARAMETER: 'entity_name' must be a string".to_string(),
            });
        }

        if let Some(conf) = arguments["min_confidence"].as_number() {
            if let Some(conf_int) = conf.as_fixed_point_i64(2) {
                let conf_val = conf_int as f64 / 100.0;
                if !(0.0..=1.0).contains(&conf_val) {
                    return Err(GraphRAGError::Generation {
                        message: "PARAMETER ERROR: 'min_confidence' must be between 0.0 and 1.0".to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}
