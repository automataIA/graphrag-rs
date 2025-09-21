//! Logic form retrieval for ROGRAG system
//!
//! Implements structured reasoning using logic forms to represent queries
//! and perform precise retrieval based on semantic relationships.

#[cfg(feature = "rograg")]
use crate::core::{Entity, KnowledgeGraph};
#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use std::collections::HashSet;
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

/// Error types for logic form retrieval
#[cfg(feature = "rograg")]
#[derive(Error, Debug)]
pub enum LogicFormError {
    #[error("Cannot parse query into logic form: {query}")]
    ParseError { query: String },

    #[error("Invalid logic form structure: {reason}")]
    InvalidStructure { reason: String },

    #[error("Logic form execution failed: {reason}")]
    ExecutionFailed { reason: String },

    #[error("No results found for logic form query")]
    NoResults,
}

/// Logic form query representation
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicFormQuery {
    pub predicate: Predicate,
    pub arguments: Vec<Argument>,
    pub constraints: Vec<Constraint>,
    pub query_type: LogicQueryType,
    pub confidence: f32,
}

/// Predicate in a logic form
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, StrumDisplay, EnumString, PartialEq)]
pub enum Predicate {
    Is,       // is(X, Y) - X is Y
    Has,      // has(X, Y) - X has property Y
    Related,  // related(X, Y, R) - X and Y are related by R
    Located,  // located(X, Y) - X is located at Y
    Happened, // happened(X, T) - X happened at time T
    Caused,   // caused(X, Y) - X caused Y
    Compare,  // compare(X, Y, P) - compare X and Y on property P
    Exists,   // exists(X) - X exists
    Count,    // count(X) - count instances of X
    Similar,  // similar(X, Y) - X is similar to Y
}

/// Argument in a logic form
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    pub arg_type: ArgumentType,
    pub value: String,
    pub variable: Option<String>, // For variables like ?x, ?y
    pub constraints: Vec<String>,
}

/// Type of argument
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, StrumDisplay, EnumString)]
pub enum ArgumentType {
    Entity,   // Named entity
    Property, // Property or attribute
    Relation, // Relationship type
    Time,     // Temporal expression
    Location, // Spatial expression
    Variable, // Logic variable
    Literal,  // Literal value
}

/// Constraint on logic form execution
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub constraint_type: ConstraintType,
    pub target: String,
    pub condition: String,
    pub value: Option<String>,
}

/// Type of constraint
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, StrumDisplay, EnumString)]
pub enum ConstraintType {
    TypeConstraint,       // Variable must be of specific type
    ValueConstraint,      // Variable must have specific value
    RangeConstraint,      // Variable must be in range
    ExistenceConstraint,  // Variable must exist
    UniquenessConstraint, // Variable must be unique
}

/// Type of logic query
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, StrumDisplay, EnumString)]
pub enum LogicQueryType {
    Select,    // SELECT query - retrieve entities/facts
    Ask,       // ASK query - yes/no question
    Count,     // COUNT query - count results
    Aggregate, // AGGREGATE query - compute aggregations
}

/// Result from logic form retrieval
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicFormResult {
    pub query: String,
    pub logic_form: LogicFormQuery,
    pub bindings: Vec<VariableBinding>,
    pub answer: String,
    pub confidence: f32,
    pub sources: Vec<String>,
    pub execution_stats: LogicExecutionStats,
}

/// Variable binding from logic form execution
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBinding {
    pub variable: String,
    pub value: String,
    pub entity_id: Option<String>,
    pub confidence: f32,
}

/// Statistics from logic form execution
#[cfg(feature = "rograg")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LogicExecutionStats {
    pub parsing_time_ms: u64,
    pub execution_time_ms: u64,
    pub entities_examined: usize,
    pub relationships_examined: usize,
    pub bindings_found: usize,
}

/// Logic form retriever implementation
#[cfg(feature = "rograg")]
pub struct LogicFormRetriever {
    parsers: Vec<Box<dyn LogicFormParser>>,
    executor: LogicFormExecutor,
}

/// Trait for parsing queries into logic forms
#[cfg(feature = "rograg")]
pub trait LogicFormParser: Send + Sync {
    /// Parse a query into a logic form
    fn parse(&self, query: &str) -> Result<Option<LogicFormQuery>>;

    /// Check if this parser can handle the query
    fn can_parse(&self, query: &str) -> bool;

    /// Get parser name
    fn name(&self) -> &str;
}

/// Simple pattern-based logic form parser
#[cfg(feature = "rograg")]
pub struct PatternBasedParser {
    patterns: Vec<LogicPattern>,
}

#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
struct LogicPattern {
    regex: regex::Regex,
    predicate: Predicate,
    query_type: LogicQueryType,
    argument_extractors: Vec<ArgumentExtractor>,
}

#[cfg(feature = "rograg")]
#[derive(Debug, Clone)]
struct ArgumentExtractor {
    group_index: usize,
    arg_type: ArgumentType,
    variable_name: Option<String>,
}

#[cfg(feature = "rograg")]
impl PatternBasedParser {
    pub fn new() -> Result<Self> {
        let patterns = vec![
            // "What is X?" pattern
            LogicPattern {
                regex: regex::Regex::new(r"(?i)what (?:is|are) (?:the )?(.+)\??")?,
                predicate: Predicate::Is,
                query_type: LogicQueryType::Select,
                argument_extractors: vec![ArgumentExtractor {
                    group_index: 1,
                    arg_type: ArgumentType::Entity,
                    variable_name: Some("X".to_string()),
                }],
            },
            // "Who is X?" pattern
            LogicPattern {
                regex: regex::Regex::new(r"(?i)who (?:is|are) (?:the )?(.+)\??")?,
                predicate: Predicate::Is,
                query_type: LogicQueryType::Select,
                argument_extractors: vec![ArgumentExtractor {
                    group_index: 1,
                    arg_type: ArgumentType::Entity,
                    variable_name: Some("X".to_string()),
                }],
            },
            // "How are X and Y related?" pattern
            LogicPattern {
                regex: regex::Regex::new(
                    r"(?i)how (?:is|are) (.+?) (?:related to|connected to) (.+)\??",
                )?,
                predicate: Predicate::Related,
                query_type: LogicQueryType::Select,
                argument_extractors: vec![
                    ArgumentExtractor {
                        group_index: 1,
                        arg_type: ArgumentType::Entity,
                        variable_name: Some("X".to_string()),
                    },
                    ArgumentExtractor {
                        group_index: 2,
                        arg_type: ArgumentType::Entity,
                        variable_name: Some("Y".to_string()),
                    },
                ],
            },
            // "When did X happen?" pattern
            LogicPattern {
                regex: regex::Regex::new(r"(?i)when (?:did|does|will) (.+?) (?:happen|occur)\??")?,
                predicate: Predicate::Happened,
                query_type: LogicQueryType::Select,
                argument_extractors: vec![ArgumentExtractor {
                    group_index: 1,
                    arg_type: ArgumentType::Entity,
                    variable_name: Some("X".to_string()),
                }],
            },
            // "Why did X cause Y?" pattern
            LogicPattern {
                regex: regex::Regex::new(r"(?i)why (?:did|does) (.+?) (?:cause|lead to) (.+)\??")?,
                predicate: Predicate::Caused,
                query_type: LogicQueryType::Select,
                argument_extractors: vec![
                    ArgumentExtractor {
                        group_index: 1,
                        arg_type: ArgumentType::Entity,
                        variable_name: Some("X".to_string()),
                    },
                    ArgumentExtractor {
                        group_index: 2,
                        arg_type: ArgumentType::Entity,
                        variable_name: Some("Y".to_string()),
                    },
                ],
            },
            // "Compare X and Y" pattern
            LogicPattern {
                regex: regex::Regex::new(
                    r"(?i)compare (.+?) (?:and|with|to) (.+)(?:\s+(?:on|in terms of) (.+))?\??",
                )?,
                predicate: Predicate::Compare,
                query_type: LogicQueryType::Select,
                argument_extractors: vec![
                    ArgumentExtractor {
                        group_index: 1,
                        arg_type: ArgumentType::Entity,
                        variable_name: Some("X".to_string()),
                    },
                    ArgumentExtractor {
                        group_index: 2,
                        arg_type: ArgumentType::Entity,
                        variable_name: Some("Y".to_string()),
                    },
                ],
            },
        ];

        Ok(Self { patterns })
    }
}

#[cfg(feature = "rograg")]
impl LogicFormParser for PatternBasedParser {
    fn parse(&self, query: &str) -> Result<Option<LogicFormQuery>> {
        for pattern in &self.patterns {
            if let Some(captures) = pattern.regex.captures(query) {
                let mut arguments = Vec::new();

                for extractor in &pattern.argument_extractors {
                    if let Some(captured) = captures.get(extractor.group_index) {
                        let value = captured.as_str().trim().to_string();
                        if !value.is_empty() {
                            arguments.push(Argument {
                                arg_type: extractor.arg_type.clone(),
                                value,
                                variable: extractor.variable_name.clone(),
                                constraints: vec![],
                            });
                        }
                    }
                }

                // Add basic type constraints
                let constraints = arguments
                    .iter()
                    .filter_map(|arg| {
                        arg.variable.as_ref().map(|var| Constraint {
                            constraint_type: ConstraintType::TypeConstraint,
                            target: var.clone(),
                            condition: format!("type = {:?}", arg.arg_type),
                            value: None,
                        })
                    })
                    .collect();

                return Ok(Some(LogicFormQuery {
                    predicate: pattern.predicate.clone(),
                    arguments,
                    constraints,
                    query_type: pattern.query_type.clone(),
                    confidence: 0.8, // Default confidence for pattern matches
                }));
            }
        }

        Ok(None)
    }

    fn can_parse(&self, query: &str) -> bool {
        self.patterns
            .iter()
            .any(|pattern| pattern.regex.is_match(query))
    }

    fn name(&self) -> &str {
        "pattern_based"
    }
}

/// Logic form executor
#[cfg(feature = "rograg")]
pub struct LogicFormExecutor {
    // Configuration could be added here
}

#[cfg(feature = "rograg")]
impl Default for LogicFormExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl LogicFormExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a logic form query against the knowledge graph
    pub fn execute(
        &self,
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        match logic_form.predicate {
            Predicate::Is => self.execute_is_query(logic_form, graph),
            Predicate::Related => self.execute_related_query(logic_form, graph),
            Predicate::Has => self.execute_has_query(logic_form, graph),
            Predicate::Compare => self.execute_compare_query(logic_form, graph),
            Predicate::Happened => self.execute_happened_query(logic_form, graph),
            Predicate::Caused => self.execute_caused_query(logic_form, graph),
            _ => Ok(vec![]),
        }
    }

    /// Execute "is" queries (What is X?)
    fn execute_is_query(
        &self,
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        if let Some(entity_arg) = logic_form.arguments.first() {
            let entity_name = &entity_arg.value;

            // Find matching entities by name
            for entity in graph.entities() {
                if entity
                    .name
                    .to_lowercase()
                    .contains(&entity_name.to_lowercase())
                {
                    bindings.push(VariableBinding {
                        variable: entity_arg.variable.clone().unwrap_or("X".to_string()),
                        value: format!("{} ({})", entity.name, entity.entity_type),
                        entity_id: Some(entity.id.to_string()),
                        confidence: self.calculate_name_similarity(entity_name, &entity.name),
                    });
                }
            }
        }

        Ok(bindings)
    }

    /// Execute "related" queries (How are X and Y related?)
    fn execute_related_query(
        &self,
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        if logic_form.arguments.len() >= 2 {
            let entity1_name = &logic_form.arguments[0].value;
            let entity2_name = &logic_form.arguments[1].value;

            // Find entities by name
            let entity1 = self.find_entity_by_name(graph, entity1_name);
            let entity2 = self.find_entity_by_name(graph, entity2_name);

            if let (Some(e1), Some(e2)) = (entity1, entity2) {
                // Look for direct relationships
                let relationships = graph.get_entity_relationships(&e1.id.0);
                for rel in relationships {
                    if rel.target == e2.id || rel.source == e2.id {
                        bindings.push(VariableBinding {
                            variable: "R".to_string(),
                            value: format!("{} {} {}", e1.name, rel.relation_type, e2.name),
                            entity_id: None,
                            confidence: rel.confidence,
                        });
                    }
                }

                // If no direct relationship, look for indirect connections
                if bindings.is_empty() {
                    bindings.push(VariableBinding {
                        variable: "R".to_string(),
                        value: format!(
                            "No direct relationship found between {} and {}",
                            e1.name, e2.name
                        ),
                        entity_id: None,
                        confidence: 0.3,
                    });
                }
            }
        }

        Ok(bindings)
    }

    /// Execute "has" queries (What does X have?)
    fn execute_has_query(
        &self,
        _logic_form: &LogicFormQuery,
        _graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        // TODO: Implement property extraction
        Ok(vec![])
    }

    /// Execute "compare" queries (Compare X and Y)
    fn execute_compare_query(
        &self,
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        if logic_form.arguments.len() >= 2 {
            let entity1_name = &logic_form.arguments[0].value;
            let entity2_name = &logic_form.arguments[1].value;

            let entity1 = self.find_entity_by_name(graph, entity1_name);
            let entity2 = self.find_entity_by_name(graph, entity2_name);

            if let (Some(e1), Some(e2)) = (entity1, entity2) {
                bindings.push(VariableBinding {
                    variable: "comparison".to_string(),
                    value: format!(
                        "{} is a {} while {} is a {}",
                        e1.name, e1.entity_type, e2.name, e2.entity_type
                    ),
                    entity_id: None,
                    confidence: 0.7,
                });
            }
        }

        Ok(bindings)
    }

    /// Execute temporal queries (When did X happen?)
    fn execute_happened_query(
        &self,
        _logic_form: &LogicFormQuery,
        _graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        // TODO: Implement temporal reasoning
        Ok(vec![])
    }

    /// Execute causal queries (Why did X cause Y?)
    fn execute_caused_query(
        &self,
        _logic_form: &LogicFormQuery,
        _graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        // TODO: Implement causal reasoning
        Ok(vec![])
    }

    /// Find entity by name (fuzzy matching)
    fn find_entity_by_name<'a>(&self, graph: &'a KnowledgeGraph, name: &str) -> Option<&'a Entity> {
        let name_lower = name.to_lowercase();

        // Try exact match first
        for entity in graph.entities() {
            if entity.name.to_lowercase() == name_lower {
                return Some(entity);
            }
        }

        // Try partial match
        graph.entities().find(|&entity| entity.name.to_lowercase().contains(&name_lower)
                || name_lower.contains(&entity.name.to_lowercase()))
    }

    /// Calculate name similarity
    fn calculate_name_similarity(&self, query_name: &str, entity_name: &str) -> f32 {
        let query_lower = query_name.to_lowercase();
        let entity_lower = entity_name.to_lowercase();

        if query_lower == entity_lower {
            1.0
        } else if entity_lower.contains(&query_lower) || query_lower.contains(&entity_lower) {
            0.8
        } else {
            let query_words: HashSet<&str> = query_lower.split_whitespace().collect();
            let entity_words: HashSet<&str> = entity_lower.split_whitespace().collect();
            let intersection = query_words.intersection(&entity_words).count();
            let union = query_words.union(&entity_words).count();

            if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            }
        }
    }
}

#[cfg(feature = "rograg")]
impl Default for LogicFormRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl LogicFormRetriever {
    /// Create a new logic form retriever
    pub fn new() -> Self {
        let parsers: Vec<Box<dyn LogicFormParser>> = vec![Box::new(PatternBasedParser::new().unwrap())];

        Self {
            parsers,
            executor: LogicFormExecutor::new(),
        }
    }

    /// Retrieve using logic form
    pub async fn retrieve(&self, query: &str, graph: &KnowledgeGraph) -> Result<LogicFormResult> {
        let start_time = std::time::Instant::now();

        // Try to parse the query with each parser
        let mut logic_form = None;
        for parser in &self.parsers {
            if let Some(parsed) = parser.parse(query)? {
                logic_form = Some(parsed);
                break;
            }
        }

        let parsing_time = start_time.elapsed().as_millis() as u64;

        let logic_form = logic_form.ok_or_else(|| LogicFormError::ParseError {
            query: query.to_string(),
        })?;

        // Execute the logic form
        let execution_start = std::time::Instant::now();
        let bindings = self.executor.execute(&logic_form, graph)?;
        let execution_time = execution_start.elapsed().as_millis() as u64;

        if bindings.is_empty() {
            return Err(LogicFormError::NoResults.into());
        }

        // Generate answer from bindings
        let answer = self.generate_answer(&logic_form, &bindings);
        let confidence = self.calculate_overall_confidence(&bindings);
        let sources = self.extract_sources(&bindings);

        Ok(LogicFormResult {
            query: query.to_string(),
            logic_form,
            bindings: bindings.clone(),
            answer,
            confidence,
            sources,
            execution_stats: LogicExecutionStats {
                parsing_time_ms: parsing_time,
                execution_time_ms: execution_time,
                entities_examined: graph.entities().count(),
                relationships_examined: 0, // TODO: Track this
                bindings_found: bindings.len(),
            },
        })
    }

    /// Generate answer from variable bindings
    fn generate_answer(&self, logic_form: &LogicFormQuery, bindings: &[VariableBinding]) -> String {
        match logic_form.predicate {
            Predicate::Is => {
                if let Some(binding) = bindings.first() {
                    binding.value.clone()
                } else {
                    "No information found.".to_string()
                }
            }
            Predicate::Related => {
                if let Some(binding) = bindings.first() {
                    binding.value.clone()
                } else {
                    "No relationship found.".to_string()
                }
            }
            Predicate::Compare => {
                if let Some(binding) = bindings.first() {
                    binding.value.clone()
                } else {
                    "Cannot compare the specified entities.".to_string()
                }
            }
            _ => {
                let values: Vec<String> = bindings.iter().map(|b| b.value.clone()).collect();
                values.join("; ")
            }
        }
    }

    /// Calculate overall confidence from bindings
    fn calculate_overall_confidence(&self, bindings: &[VariableBinding]) -> f32 {
        if bindings.is_empty() {
            return 0.0;
        }

        let sum: f32 = bindings.iter().map(|b| b.confidence).sum();
        sum / bindings.len() as f32
    }

    /// Extract source IDs from bindings
    fn extract_sources(&self, bindings: &[VariableBinding]) -> Vec<String> {
        bindings
            .iter()
            .filter_map(|b| b.entity_id.clone())
            .collect()
    }

    /// Add a custom parser
    pub fn add_parser(&mut self, parser: Box<dyn LogicFormParser>) {
        self.parsers.push(parser);
    }

    /// Get supported predicates
    pub fn get_supported_predicates(&self) -> Vec<Predicate> {
        vec![
            Predicate::Is,
            Predicate::Related,
            Predicate::Has,
            Predicate::Compare,
            Predicate::Happened,
            Predicate::Caused,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Entity, EntityId, KnowledgeGraph};

    #[cfg(feature = "rograg")]
    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        let entity1 = Entity {
            id: EntityId::new("entity_1".to_string()),
            name: "Entity Name".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 1.0,
            mentions: vec![],
            embedding: None,
        };

        let entity2 = Entity {
            id: EntityId::new("entity_2".to_string()),
            name: "Second Entity".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 1.0,
            mentions: vec![],
            embedding: None,
        };

        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();

        graph
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_pattern_parser() {
        let parser = PatternBasedParser::new().unwrap();

        // Test "What is" pattern
        let result = parser.parse("What is Entity Name?").unwrap();
        assert!(result.is_some());

        let logic_form = result.unwrap();
        assert_eq!(logic_form.predicate, Predicate::Is);
        assert_eq!(logic_form.arguments.len(), 1);
        assert_eq!(logic_form.arguments[0].value, "Entity Name");

        // Test relationship pattern
        let result = parser
            .parse("How is Entity Name related to Second Entity?")
            .unwrap();
        assert!(result.is_some());

        let logic_form = result.unwrap();
        assert_eq!(logic_form.predicate, Predicate::Related);
        assert_eq!(logic_form.arguments.len(), 2);
    }

    #[cfg(feature = "rograg")]
    #[tokio::test]
    async fn test_logic_form_retrieval() {
        let retriever = LogicFormRetriever::new();
        let graph = create_test_graph();

        let result = retriever
            .retrieve("What is Entity Name?", &graph)
            .await
            .unwrap();

        assert!(!result.bindings.is_empty());
        assert!(result.confidence > 0.0);
        assert!(!result.answer.is_empty());
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_executor_is_query() {
        let executor = LogicFormExecutor::new();
        let graph = create_test_graph();

        let logic_form = LogicFormQuery {
            predicate: Predicate::Is,
            arguments: vec![Argument {
                arg_type: ArgumentType::Entity,
                value: "Entity Name".to_string(),
                variable: Some("X".to_string()),
                constraints: vec![],
            }],
            constraints: vec![],
            query_type: LogicQueryType::Select,
            confidence: 0.8,
        };

        let bindings = executor.execute(&logic_form, &graph).unwrap();
        assert!(!bindings.is_empty());
        assert!(bindings[0].confidence > 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_name_similarity() {
        let executor = LogicFormExecutor::new();

        assert_eq!(
            executor.calculate_name_similarity("Entity Name", "Entity Name"),
            1.0
        );
        assert!(executor.calculate_name_similarity("Entity", "Entity Name") > 0.5);
        assert!(executor.calculate_name_similarity("Completely Different", "Entity Name") < 0.5);
    }
}
