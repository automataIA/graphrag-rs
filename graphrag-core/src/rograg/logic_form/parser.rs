#![allow(unused_imports)]

#[cfg(feature = "rograg")]
use crate::core::{Entity, KnowledgeGraph};
#[cfg(feature = "rograg")]
use crate::retrieval::causal_analysis::CausalAnalyzer;
#[cfg(feature = "rograg")]
use crate::Result;
#[cfg(feature = "rograg")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rograg")]
use std::collections::HashSet;
#[cfg(feature = "rograg")]
use std::sync::Arc;
#[cfg(feature = "rograg")]
use strum::{Display as StrumDisplay, EnumString};
#[cfg(feature = "rograg")]
use thiserror::Error;

#[cfg(feature = "rograg")]
use super::*;

/// Trait for implementing logic form parsers.
///
/// Different parsers can implement different parsing strategies
/// (pattern-based, rule-based, learned, etc.).
#[cfg(feature = "rograg")]
pub trait LogicFormParser: Send + Sync {
    /// Parse a query into a logic form representation.
    ///
    /// Returns `None` if the query cannot be parsed by this parser.
    fn parse(&self, query: &str) -> Result<Option<LogicFormQuery>>;

    /// Check if this parser can handle the given query.
    ///
    /// Used for parser selection before attempting full parsing.
    fn can_parse(&self, query: &str) -> bool;

    /// Get the parser's identifier.
    fn name(&self) -> &str;
}

/// Pattern-based parser for converting queries to logic forms.
///
/// Uses regex patterns to recognize common query structures and map them
/// to formal logic representations. Supports "What is X?", "How are X and Y
/// related?", temporal, causal, and comparison queries.
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
    /// Create a new pattern-based parser with predefined logic patterns.
    ///
    /// Initializes patterns for common query structures including "What is X?",
    /// "How are X and Y related?", temporal, causal, and comparison queries.
    ///
    /// # Returns
    ///
    /// Returns a `PatternBasedParser` ready for query parsing, or an error
    /// if regex pattern compilation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if any regex pattern fails to compile during initialization.
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
