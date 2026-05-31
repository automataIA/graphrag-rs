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

/// Logic form retriever for structured query processing.
///
/// Parses natural language queries into logic forms and executes them
/// against the knowledge graph for precise retrieval.
#[cfg(feature = "rograg")]
pub struct LogicFormRetriever {
    parsers: Vec<Box<dyn LogicFormParser>>,
    executor: LogicFormExecutor,
}

#[cfg(feature = "rograg")]
impl Default for LogicFormRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "rograg")]
impl LogicFormRetriever {
    /// Create a new logic form retriever with default parsers.
    ///
    /// Initializes with a pattern-based parser for common query structures.
    ///
    /// # Returns
    ///
    /// Returns a `LogicFormRetriever` ready for query processing.
    pub fn new() -> Self {
        let parsers: Vec<Box<dyn LogicFormParser>> =
            vec![Box::new(PatternBasedParser::new().expect("static pattern set"))];

        Self {
            parsers,
            executor: LogicFormExecutor::new(),
        }
    }

    /// Retrieve information using logic form query processing.
    ///
    /// Parses the query into a logic form, executes it against the knowledge
    /// graph, and generates a natural language answer from the bindings.
    ///
    /// # Arguments
    ///
    /// * `query` - The natural language query to process
    /// * `graph` - The knowledge graph to query against
    ///
    /// # Returns
    ///
    /// Returns a `LogicFormResult` containing bindings, answer, and statistics.
    ///
    /// # Errors
    ///
    /// - `LogicFormError::ParseError` if the query cannot be parsed
    /// - `LogicFormError::NoResults` if no bindings are found
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let retriever = LogicFormRetriever::new();
    /// let result = retriever.retrieve("What is Tom?", &graph).await?;
    /// println!("Answer: {}", result.answer);
    /// ```
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

        // Count relationships examined based on query type
        let relationships_examined = match logic_form.predicate {
            // These predicates examine relationships
            Predicate::Related | Predicate::Caused | Predicate::Compare => {
                graph.relationships().count()
            },
            // These predicates don't examine relationships
            Predicate::Is
            | Predicate::Has
            | Predicate::Happened
            | Predicate::Exists
            | Predicate::Similar
            | Predicate::Located => 0,
            // For Count and aggregate queries, examine all if needed
            _ => 0,
        };

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
                relationships_examined,
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
            },
            Predicate::Related => {
                if let Some(binding) = bindings.first() {
                    binding.value.clone()
                } else {
                    "No relationship found.".to_string()
                }
            },
            Predicate::Compare => {
                if let Some(binding) = bindings.first() {
                    binding.value.clone()
                } else {
                    "Cannot compare the specified entities.".to_string()
                }
            },
            _ => {
                let values: Vec<String> = bindings.iter().map(|b| b.value.clone()).collect();
                values.join("; ")
            },
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

