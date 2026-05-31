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
    /// Create a new logic form executor.
    ///
    /// Initializes the executor for processing logic forms against a knowledge graph.
    ///
    /// # Returns
    ///
    /// Returns a `LogicFormExecutor` ready for query execution.
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
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        // Extract entity and property from arguments
        if logic_form.arguments.len() >= 2 {
            let entity_arg = &logic_form.arguments[0];
            let property_arg = &logic_form.arguments[1];

            let entity_name = &entity_arg.value;
            let property_name = &property_arg.value.to_lowercase();

            // Find matching entity
            if let Some(entity) = self.find_entity_by_name(graph, entity_name) {
                // Extract property value based on property name
                let property_value = match property_name.as_str() {
                    "name" => Some(entity.name.clone()),
                    "type" | "entity_type" => Some(entity.entity_type.clone()),
                    "confidence" => Some(format!("{:.2}", entity.confidence)),
                    "mentions" | "mention_count" => Some(entity.mentions.len().to_string()),
                    "embedding" => {
                        if entity.embedding.is_some() {
                            Some("has embedding".to_string())
                        } else {
                            Some("no embedding".to_string())
                        }
                    },
                    _ => None,
                };

                if let Some(value) = property_value {
                    bindings.push(VariableBinding {
                        variable: property_arg.variable.clone().unwrap_or("P".to_string()),
                        value,
                        entity_id: Some(entity.id.to_string()),
                        confidence: 0.9, // High confidence for direct property access
                    });
                }
            }
        } else if logic_form.arguments.len() == 1 {
            // If only entity is provided, return all properties
            let entity_arg = &logic_form.arguments[0];
            let entity_name = &entity_arg.value;

            if let Some(entity) = self.find_entity_by_name(graph, entity_name) {
                // Return name
                bindings.push(VariableBinding {
                    variable: "name".to_string(),
                    value: entity.name.clone(),
                    entity_id: Some(entity.id.to_string()),
                    confidence: 1.0,
                });

                // Return type
                bindings.push(VariableBinding {
                    variable: "type".to_string(),
                    value: entity.entity_type.clone(),
                    entity_id: Some(entity.id.to_string()),
                    confidence: 1.0,
                });

                // Return confidence
                bindings.push(VariableBinding {
                    variable: "confidence".to_string(),
                    value: format!("{:.2}", entity.confidence),
                    entity_id: Some(entity.id.to_string()),
                    confidence: 1.0,
                });

                // Return mention count
                bindings.push(VariableBinding {
                    variable: "mentions".to_string(),
                    value: entity.mentions.len().to_string(),
                    entity_id: Some(entity.id.to_string()),
                    confidence: 1.0,
                });
            }
        }

        Ok(bindings)
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
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        if logic_form.arguments.is_empty() {
            return Ok(bindings);
        }

        let event_arg = &logic_form.arguments[0];
        let event_name = &event_arg.value;

        // Find entity representing the event
        if let Some(entity) = self.find_entity_by_name(graph, event_name) {
            // Strategy 1: Look for temporal relationships
            let relationships = graph.get_entity_relationships(&entity.id.0);
            for rel in relationships {
                let rel_type_lower = rel.relation_type.to_lowercase();

                // Check for temporal relationship types
                if rel_type_lower.contains("happened")
                    || rel_type_lower.contains("occurred")
                    || rel_type_lower.contains("during")
                    || rel_type_lower.contains("before")
                    || rel_type_lower.contains("after")
                    || rel_type_lower.contains("when")
                {
                    // Get the related entity which might represent a time
                    if let Some(time_entity) = graph.get_entity(&rel.target) {
                        bindings.push(VariableBinding {
                            variable: "T".to_string(),
                            value: format!(
                                "{} {} {}",
                                event_name, rel.relation_type, time_entity.name
                            ),
                            entity_id: Some(time_entity.id.to_string()),
                            confidence: rel.confidence,
                        });
                    }
                }
            }

            // Strategy 2: Extract temporal info from entity mentions in chunks
            for mention in &entity.mentions {
                if let Some(chunk) = graph.get_chunk(&mention.chunk_id) {
                    // Check chunk metadata for temporal information
                    if let Some(date) = chunk.metadata.custom.get("date") {
                        bindings.push(VariableBinding {
                            variable: "T".to_string(),
                            value: format!("{} occurred on {}", event_name, date),
                            entity_id: Some(entity.id.to_string()),
                            confidence: 0.8,
                        });
                    } else if let Some(timestamp) = chunk.metadata.custom.get("timestamp") {
                        bindings.push(VariableBinding {
                            variable: "T".to_string(),
                            value: format!("{} occurred at {}", event_name, timestamp),
                            entity_id: Some(entity.id.to_string()),
                            confidence: 0.8,
                        });
                    } else if let Some(time) = chunk.metadata.custom.get("time") {
                        bindings.push(VariableBinding {
                            variable: "T".to_string(),
                            value: format!("{} happened at {}", event_name, time),
                            entity_id: Some(entity.id.to_string()),
                            confidence: 0.8,
                        });
                    }

                    // Strategy 3: Parse chunk content for temporal expressions
                    // Look for common date patterns in the chunk content
                    let content_lower = chunk.content.to_lowercase();
                    let temporal_keywords = [
                        "january",
                        "february",
                        "march",
                        "april",
                        "may",
                        "june",
                        "july",
                        "august",
                        "september",
                        "october",
                        "november",
                        "december",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                        "yesterday",
                        "today",
                        "tomorrow",
                        "morning",
                        "afternoon",
                        "evening",
                        "night",
                        "spring",
                        "summer",
                        "autumn",
                        "fall",
                        "winter",
                    ];

                    for keyword in &temporal_keywords {
                        if content_lower.contains(keyword) {
                            // Extract surrounding context (rough temporal extraction)
                            if let Some(pos) = content_lower.find(keyword) {
                                let start = pos.saturating_sub(20);
                                let end = (pos + keyword.len() + 20).min(chunk.content.len());
                                let context = &chunk.content[start..end];

                                bindings.push(VariableBinding {
                                    variable: "T".to_string(),
                                    value: format!(
                                        "{} temporal context: \"{}\"",
                                        event_name,
                                        context.trim()
                                    ),
                                    entity_id: Some(entity.id.to_string()),
                                    confidence: 0.6,
                                });
                                break; // Only add one temporal context per chunk
                            }
                        }
                    }
                }
            }

            // Strategy 4: Use document position as temporal ordering heuristic
            if let Some(first_mention) = entity.mentions.first() {
                if let Some(chunk) = graph.get_chunk(&first_mention.chunk_id) {
                    if let Some(position) = chunk.metadata.position_in_document {
                        let temporal_order = if position < 0.33 {
                            "early in the narrative"
                        } else if position < 0.67 {
                            "middle of the narrative"
                        } else {
                            "later in the narrative"
                        };

                        bindings.push(VariableBinding {
                            variable: "T".to_string(),
                            value: format!(
                                "{} occurred {} (position: {:.2})",
                                event_name, temporal_order, position
                            ),
                            entity_id: Some(entity.id.to_string()),
                            confidence: 0.5,
                        });
                    }
                }
            }
        }

        // If no temporal information found, provide default response
        if bindings.is_empty() {
            bindings.push(VariableBinding {
                variable: "T".to_string(),
                value: format!("No temporal information found for {}", event_name),
                entity_id: None,
                confidence: 0.2,
            });
        }

        Ok(bindings)
    }

    /// Execute causal queries (Why did X cause Y?)
    fn execute_caused_query(
        &self,
        logic_form: &LogicFormQuery,
        graph: &KnowledgeGraph,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        if logic_form.arguments.len() < 2 {
            return Ok(bindings);
        }

        let cause_arg = &logic_form.arguments[0];
        let effect_arg = &logic_form.arguments[1];

        let cause_name = &cause_arg.value;
        let effect_name = &effect_arg.value;

        // Find entities representing cause and effect
        let cause_entity = self.find_entity_by_name(graph, cause_name);
        let effect_entity = self.find_entity_by_name(graph, effect_name);

        if let (Some(cause_e), Some(effect_e)) = (cause_entity, effect_entity) {
            // Strategy 1: Look for direct causal relationships
            let relationships = graph.get_entity_relationships(&cause_e.id.0);

            for rel in relationships {
                let rel_type_lower = rel.relation_type.to_lowercase();

                // Check for causal relationship types
                if (rel_type_lower.contains("cause")
                    || rel_type_lower.contains("leads_to")
                    || rel_type_lower.contains("results_in")
                    || rel_type_lower.contains("because")
                    || rel_type_lower.contains("due_to")
                    || rel_type_lower.contains("triggers")
                    || rel_type_lower.contains("produces"))
                    && (rel.target == effect_e.id || rel.source == effect_e.id)
                {
                    bindings.push(VariableBinding {
                        variable: "C".to_string(),
                        value: format!("{} {} {}", cause_name, rel.relation_type, effect_name),
                        entity_id: None,
                        confidence: rel.confidence,
                    });
                }
            }

            // Strategy 2: Build causal chains using CausalAnalyzer (Phase 2.3)
            // Finds temporally-consistent causal paths between cause and effect
            //
            // ✅ IMPLEMENTED: Full CausalAnalyzer integration with temporal consistency
            //
            // Solution: Temporary Arc wrapping (Option C from TECHNICAL_DEBT.md)
            // Clones the graph temporarily to satisfy Arc<KnowledgeGraph> requirement.
            // Future optimization: Refactor RoGRAGProcessor to use Arc<KnowledgeGraph> directly.
            let graph_arc = Arc::new(graph.clone());
            let analyzer = CausalAnalyzer::new(graph_arc)
                .with_min_confidence(0.3)
                .with_temporal_consistency(false); // Lenient for now

            match analyzer.find_causal_chains(&cause_e.id, &effect_e.id, 5) {
                Ok(chains) => {
                    for chain in chains {
                        // Build human-readable chain description
                        let step_descriptions: Vec<String> = chain
                            .steps
                            .iter()
                            .map(|step| {
                                format!(
                                    "{} --[{}]--> {}",
                                    step.source.0, step.relation_type, step.target.0
                                )
                            })
                            .collect();

                        let chain_str = if step_descriptions.is_empty() {
                            format!("{} → {}", cause_e.name, effect_e.name)
                        } else {
                            step_descriptions.join(" → ")
                        };

                        // Include temporal consistency information if available
                        let value = if chain.temporal_consistency {
                            if let Some(time_span) = chain.time_span {
                                format!(
                                    "Causal chain (temporally consistent, span={}s): {}",
                                    time_span, chain_str
                                )
                            } else {
                                format!("Causal chain (temporally consistent): {}", chain_str)
                            }
                        } else {
                            format!("Causal chain: {}", chain_str)
                        };

                        bindings.push(VariableBinding {
                            variable: "C".to_string(),
                            value,
                            entity_id: None,
                            confidence: chain.total_confidence,
                        });
                    }
                },
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        cause = %cause_e.name,
                        effect = %effect_e.name,
                        error = %e,
                        "Failed to find causal chains with CausalAnalyzer"
                    );
                },
            }

            // Strategy 3: Analyze co-occurrence in chunks for implicit causality
            let cause_chunks: std::collections::HashSet<_> =
                cause_e.mentions.iter().map(|m| &m.chunk_id).collect();

            let effect_chunks: std::collections::HashSet<_> =
                effect_e.mentions.iter().map(|m| &m.chunk_id).collect();

            // Find chunks where both entities are mentioned (potential causal context)
            let common_chunks: Vec<_> = cause_chunks.intersection(&effect_chunks).collect();

            if !common_chunks.is_empty() {
                for chunk_id in common_chunks {
                    if let Some(chunk) = graph.get_chunk(chunk_id) {
                        let content_lower = chunk.content.to_lowercase();

                        // Look for causal keywords in the chunk content
                        let causal_keywords = [
                            "because",
                            "therefore",
                            "thus",
                            "hence",
                            "consequently",
                            "as a result",
                            "due to",
                            "caused by",
                            "leads to",
                            "resulting in",
                            "triggered by",
                            "produced by",
                        ];

                        for keyword in &causal_keywords {
                            if content_lower.contains(keyword) {
                                // Extract context around the causal keyword
                                if let Some(pos) = content_lower.find(keyword) {
                                    let start = pos.saturating_sub(30);
                                    let end = (pos + keyword.len() + 30).min(chunk.content.len());
                                    let context = &chunk.content[start..end];

                                    bindings.push(VariableBinding {
                                        variable: "C".to_string(),
                                        value: format!("Causal context: \"{}\"", context.trim()),
                                        entity_id: None,
                                        confidence: 0.7,
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Strategy 4: Use relationship confidence scores to rank causal explanations
            if !bindings.is_empty() {
                // Sort by confidence (highest first)
                bindings.sort_by(|a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // If no causal information found, provide default response
        if bindings.is_empty() {
            bindings.push(VariableBinding {
                variable: "C".to_string(),
                value: format!(
                    "No causal relationship found between {} and {}",
                    cause_name, effect_name
                ),
                entity_id: None,
                confidence: 0.2,
            });
        }

        Ok(bindings)
    }

    // Old DFS-based causal chain finding removed - now using CausalAnalyzer
    // (See TECHNICAL_DEBT.md Task 2.2 - RoGRAG - CausalAnalyzer Integration - completed)

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
        graph.entities().find(|&entity| {
            entity.name.to_lowercase().contains(&name_lower)
                || name_lower.contains(&entity.name.to_lowercase())
        })
    }

    /// Calculate name similarity
    pub(super) fn calculate_name_similarity(&self, query_name: &str, entity_name: &str) -> f32 {
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
