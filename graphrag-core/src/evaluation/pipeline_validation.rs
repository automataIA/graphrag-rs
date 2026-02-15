//! Pipeline validation framework
//!
//! This module provides tools to validate each phase of the GraphRAG pipeline,
//! ensuring that every step produces expected outputs before proceeding.

use crate::{Document, TextChunk, Entity, Relationship};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation result for a pipeline phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseValidation {
    /// Phase name
    pub phase_name: String,
    /// Whether the phase passed validation
    pub passed: bool,
    /// Validation checks performed
    pub checks: Vec<ValidationCheck>,
    /// Warnings (non-fatal issues)
    pub warnings: Vec<String>,
    /// Metrics collected during validation
    pub metrics: HashMap<String, f64>,
}

/// A single validation check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    /// Name of the check
    pub name: String,
    /// Whether the check passed
    pub passed: bool,
    /// Expected value or condition
    pub expected: String,
    /// Actual value observed
    pub actual: String,
    /// Detailed message
    pub message: String,
}

/// Validator for document processing phase
pub struct DocumentProcessingValidator;

impl DocumentProcessingValidator {
    /// Validate document processing results
    pub fn validate(document: &Document, chunks: &[TextChunk]) -> PhaseValidation {
        let mut checks = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Check 1: Document is not empty
        checks.push(ValidationCheck {
            name: "document_not_empty".to_string(),
            passed: !document.content.is_empty(),
            expected: "Non-empty content".to_string(),
            actual: format!("{} characters", document.content.len()),
            message: if document.content.is_empty() {
                "Document content is empty".to_string()
            } else {
                "Document contains content".to_string()
            },
        });

        // Check 2: Chunks were created
        checks.push(ValidationCheck {
            name: "chunks_created".to_string(),
            passed: !chunks.is_empty(),
            expected: "At least 1 chunk".to_string(),
            actual: format!("{} chunks", chunks.len()),
            message: if chunks.is_empty() {
                "No chunks were created from document".to_string()
            } else {
                format!("Successfully created {} chunks", chunks.len())
            },
        });

        // Check 3: Chunks cover document content
        if !chunks.is_empty() {
            let total_chunk_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
            let coverage_ratio = total_chunk_chars as f64 / document.content.len() as f64;

            checks.push(ValidationCheck {
                name: "content_coverage".to_string(),
                passed: coverage_ratio >= 0.9, // At least 90% coverage
                expected: "Coverage ratio >= 0.9".to_string(),
                actual: format!("{:.2}", coverage_ratio),
                message: format!("Chunks cover {:.1}% of original content", coverage_ratio * 100.0),
            });

            metrics.insert("coverage_ratio".to_string(), coverage_ratio);
        }

        // Check 4: No chunk is empty
        let empty_chunks = chunks.iter().filter(|c| c.content.trim().is_empty()).count();
        checks.push(ValidationCheck {
            name: "no_empty_chunks".to_string(),
            passed: empty_chunks == 0,
            expected: "0 empty chunks".to_string(),
            actual: format!("{} empty chunks", empty_chunks),
            message: if empty_chunks > 0 {
                format!("Found {} empty chunks", empty_chunks)
            } else {
                "All chunks have content".to_string()
            },
        });

        // Check 5: Chunk metadata is populated
        let chunks_with_metadata = chunks
            .iter()
            .filter(|c|
                c.metadata.chapter.is_some() ||
                !c.metadata.keywords.is_empty() ||
                c.metadata.summary.is_some()
            )
            .count();

        let metadata_ratio = if chunks.is_empty() {
            0.0
        } else {
            chunks_with_metadata as f64 / chunks.len() as f64
        };

        if metadata_ratio < 0.5 {
            warnings.push(format!(
                "Only {}/{} chunks have enriched metadata ({}%)",
                chunks_with_metadata, chunks.len(), (metadata_ratio * 100.0) as u32
            ));
        }

        checks.push(ValidationCheck {
            name: "metadata_enrichment".to_string(),
            passed: true, // Metadata enrichment is optional - always pass but collect metrics
            expected: "Metadata enrichment (optional)".to_string(),
            actual: format!("{}/{} chunks", chunks_with_metadata, chunks.len()),
            message: format!("{:.1}% of chunks have metadata", metadata_ratio * 100.0),
        });

        metrics.insert("metadata_ratio".to_string(), metadata_ratio);
        metrics.insert("chunks_count".to_string(), chunks.len() as f64);
        metrics.insert("avg_chunk_size".to_string(),
            chunks.iter().map(|c| c.content.len()).sum::<usize>() as f64 / chunks.len().max(1) as f64
        );

        let passed = checks.iter().all(|c| c.passed);

        PhaseValidation {
            phase_name: "Document Processing".to_string(),
            passed,
            checks,
            warnings,
            metrics,
        }
    }
}

/// Validator for entity extraction phase
pub struct EntityExtractionValidator;

impl EntityExtractionValidator {
    /// Validate entity extraction results
    pub fn validate(chunks: &[TextChunk], entities: &[Entity]) -> PhaseValidation {
        let mut checks = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Check 1: Entities were extracted
        checks.push(ValidationCheck {
            name: "entities_extracted".to_string(),
            passed: !entities.is_empty(),
            expected: "At least 1 entity".to_string(),
            actual: format!("{} entities", entities.len()),
            message: if entities.is_empty() {
                "No entities were extracted".to_string()
            } else {
                format!("Successfully extracted {} entities", entities.len())
            },
        });

        // Check 2: Entity confidence scores are valid
        let invalid_confidence = entities
            .iter()
            .filter(|e| e.confidence < 0.0 || e.confidence > 1.0)
            .count();

        checks.push(ValidationCheck {
            name: "confidence_scores_valid".to_string(),
            passed: invalid_confidence == 0,
            expected: "All confidences in [0.0, 1.0]".to_string(),
            actual: format!("{} invalid scores", invalid_confidence),
            message: if invalid_confidence > 0 {
                format!("{} entities have invalid confidence scores", invalid_confidence)
            } else {
                "All confidence scores are valid".to_string()
            },
        });

        // Check 3: Entity types are populated
        let missing_types = entities.iter().filter(|e| e.entity_type.is_empty()).count();
        checks.push(ValidationCheck {
            name: "entity_types_populated".to_string(),
            passed: missing_types == 0,
            expected: "All entities have types".to_string(),
            actual: format!("{} without types", missing_types),
            message: if missing_types > 0 {
                format!("{} entities missing entity_type", missing_types)
            } else {
                "All entities have types assigned".to_string()
            },
        });

        // Check 4: Entity names are not empty
        let empty_names = entities.iter().filter(|e| e.name.trim().is_empty()).count();
        checks.push(ValidationCheck {
            name: "entity_names_valid".to_string(),
            passed: empty_names == 0,
            expected: "All entities have names".to_string(),
            actual: format!("{} empty names", empty_names),
            message: if empty_names > 0 {
                format!("{} entities have empty names", empty_names)
            } else {
                "All entities have valid names".to_string()
            },
        });

        // Check 5: Entity mentions reference valid chunks
        if !entities.is_empty() {
            let chunk_ids: Vec<_> = chunks.iter().map(|c| &c.id).collect();
            let invalid_mentions = entities
                .iter()
                .flat_map(|e| &e.mentions)
                .filter(|m| !chunk_ids.contains(&&m.chunk_id))
                .count();

            checks.push(ValidationCheck {
                name: "entity_mentions_valid".to_string(),
                passed: invalid_mentions == 0,
                expected: "All mentions reference valid chunks".to_string(),
                actual: format!("{} invalid references", invalid_mentions),
                message: if invalid_mentions > 0 {
                    format!("{} entity mentions reference non-existent chunks", invalid_mentions)
                } else {
                    "All entity mentions are valid".to_string()
                },
            });

            if invalid_mentions > 0 {
                warnings.push("Some entity mentions reference non-existent chunks".to_string());
            }
        }

        // Metrics
        metrics.insert("entities_count".to_string(), entities.len() as f64);
        if !entities.is_empty() {
            metrics.insert("avg_confidence".to_string(),
                entities.iter().map(|e| e.confidence as f64).sum::<f64>() / entities.len() as f64
            );
            metrics.insert("avg_mentions_per_entity".to_string(),
                entities.iter().map(|e| e.mentions.len()).sum::<usize>() as f64 / entities.len() as f64
            );
        }

        // Warning: Low average confidence
        if let Some(&avg_conf) = metrics.get("avg_confidence") {
            if avg_conf < 0.5 {
                warnings.push(format!("Low average entity confidence: {:.2}", avg_conf));
            }
        }

        let passed = checks.iter().all(|c| c.passed);

        PhaseValidation {
            phase_name: "Entity Extraction".to_string(),
            passed,
            checks,
            warnings,
            metrics,
        }
    }
}

/// Validator for relationship extraction phase
pub struct RelationshipExtractionValidator;

impl RelationshipExtractionValidator {
    /// Validate relationship extraction results
    pub fn validate(entities: &[Entity], relationships: &[Relationship]) -> PhaseValidation {
        let mut checks = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Check 1: Relationships were extracted (if entities exist)
        if !entities.is_empty() {
            let has_relationships = !relationships.is_empty();
            checks.push(ValidationCheck {
                name: "relationships_extracted".to_string(),
                passed: has_relationships,
                expected: "At least 1 relationship".to_string(),
                actual: format!("{} relationships", relationships.len()),
                message: if !has_relationships {
                    "No relationships extracted despite entities present".to_string()
                } else {
                    format!("Extracted {} relationships", relationships.len())
                },
            });

            if !has_relationships {
                warnings.push("No relationships found between entities".to_string());
            }
        }

        // Check 2: Relationship confidence scores are valid
        let invalid_confidence = relationships
            .iter()
            .filter(|r| r.confidence < 0.0 || r.confidence > 1.0)
            .count();

        checks.push(ValidationCheck {
            name: "relationship_confidence_valid".to_string(),
            passed: invalid_confidence == 0,
            expected: "All confidences in [0.0, 1.0]".to_string(),
            actual: format!("{} invalid", invalid_confidence),
            message: if invalid_confidence > 0 {
                format!("{} relationships have invalid confidence", invalid_confidence)
            } else {
                "All relationship confidences valid".to_string()
            },
        });

        // Check 3: Relationship types are populated
        let missing_types = relationships.iter().filter(|r| r.relation_type.is_empty()).count();
        checks.push(ValidationCheck {
            name: "relationship_types_populated".to_string(),
            passed: missing_types == 0,
            expected: "All relationships typed".to_string(),
            actual: format!("{} untyped", missing_types),
            message: if missing_types > 0 {
                format!("{} relationships missing type", missing_types)
            } else {
                "All relationships have types".to_string()
            },
        });

        // Check 4: Source and target entities exist
        let entity_ids: Vec<_> = entities.iter().map(|e| &e.id).collect();
        let orphan_relationships = relationships
            .iter()
            .filter(|r| !entity_ids.contains(&&r.source) || !entity_ids.contains(&&r.target))
            .count();

        checks.push(ValidationCheck {
            name: "relationship_entities_exist".to_string(),
            passed: orphan_relationships == 0,
            expected: "All relationships reference valid entities".to_string(),
            actual: format!("{} orphaned", orphan_relationships),
            message: if orphan_relationships > 0 {
                format!("{} relationships reference non-existent entities", orphan_relationships)
            } else {
                "All relationships have valid entity references".to_string()
            },
        });

        if orphan_relationships > 0 {
            warnings.push("Some relationships reference entities that don't exist in the graph".to_string());
        }

        // Metrics
        metrics.insert("relationships_count".to_string(), relationships.len() as f64);
        if !entities.is_empty() {
            metrics.insert("relationships_per_entity".to_string(),
                relationships.len() as f64 / entities.len() as f64
            );
        }
        if !relationships.is_empty() {
            metrics.insert("avg_relationship_confidence".to_string(),
                relationships.iter().map(|r| r.confidence as f64).sum::<f64>() / relationships.len() as f64
            );
        }

        let passed = checks.iter().all(|c| c.passed);

        PhaseValidation {
            phase_name: "Relationship Extraction".to_string(),
            passed,
            checks,
            warnings,
            metrics,
        }
    }
}

/// Validator for graph construction phase
pub struct GraphConstructionValidator;

impl GraphConstructionValidator {
    /// Validate constructed knowledge graph
    pub fn validate(
        documents: usize,
        chunks: usize,
        entities: usize,
        relationships: usize,
    ) -> PhaseValidation {
        let mut checks = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Check 1: Graph has content
        checks.push(ValidationCheck {
            name: "graph_not_empty".to_string(),
            passed: entities > 0 || documents > 0,
            expected: "At least some nodes".to_string(),
            actual: format!("{} entities, {} docs", entities, documents),
            message: if entities == 0 && documents == 0 {
                "Graph is completely empty".to_string()
            } else {
                "Graph contains content".to_string()
            },
        });

        // Check 2: Reasonable entity-to-chunk ratio
        if chunks > 0 {
            let entities_per_chunk = entities as f64 / chunks as f64;
            let reasonable = (0.1..=10.0).contains(&entities_per_chunk);

            checks.push(ValidationCheck {
                name: "entity_chunk_ratio_reasonable".to_string(),
                passed: reasonable,
                expected: "0.1 to 10 entities per chunk".to_string(),
                actual: format!("{:.2} entities/chunk", entities_per_chunk),
                message: if !reasonable {
                    format!("Unusual entity-to-chunk ratio: {:.2}", entities_per_chunk)
                } else {
                    "Entity density looks reasonable".to_string()
                },
            });

            metrics.insert("entities_per_chunk".to_string(), entities_per_chunk);

            if entities_per_chunk < 0.5 {
                warnings.push("Low entity density - may need better entity extraction".to_string());
            }
            if entities_per_chunk > 5.0 {
                warnings.push("High entity density - may have duplicate extractions".to_string());
            }
        }

        // Check 3: Graph connectivity
        if entities > 1 {
            let connectivity = relationships as f64 / entities as f64;
            let is_connected = connectivity > 0.1; // At least 10% connectivity

            checks.push(ValidationCheck {
                name: "graph_connectivity".to_string(),
                passed: is_connected,
                expected: ">0.1 relationships per entity".to_string(),
                actual: format!("{:.2} rels/entity", connectivity),
                message: if !is_connected {
                    "Graph is sparsely connected".to_string()
                } else {
                    "Graph has reasonable connectivity".to_string()
                },
            });

            metrics.insert("connectivity".to_string(), connectivity);

            if connectivity < 0.5 {
                warnings.push("Graph is sparsely connected - entities may be isolated".to_string());
            }
        }

        // Metrics
        metrics.insert("documents".to_string(), documents as f64);
        metrics.insert("chunks".to_string(), chunks as f64);
        metrics.insert("entities".to_string(), entities as f64);
        metrics.insert("relationships".to_string(), relationships as f64);

        let passed = checks.iter().all(|c| c.passed);

        PhaseValidation {
            phase_name: "Graph Construction".to_string(),
            passed,
            checks,
            warnings,
            metrics,
        }
    }
}

/// Complete pipeline validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineValidationReport {
    /// Validation results for each phase
    pub phases: Vec<PhaseValidation>,
    /// Overall validation status
    pub overall_passed: bool,
    /// Total checks performed
    pub total_checks: usize,
    /// Number of passed checks
    pub passed_checks: usize,
    /// Summary message
    pub summary: String,
}

impl PipelineValidationReport {
    /// Create a report from phase validations
    pub fn from_phases(phases: Vec<PhaseValidation>) -> Self {
        let overall_passed = phases.iter().all(|p| p.passed);
        let total_checks = phases.iter().map(|p| p.checks.len()).sum();
        let passed_checks = phases
            .iter()
            .flat_map(|p| &p.checks)
            .filter(|c| c.passed)
            .count();

        let summary = if overall_passed {
            format!("✅ All pipeline phases validated successfully ({}/{} checks passed)",
                passed_checks, total_checks)
        } else {
            let failed_phases: Vec<_> = phases
                .iter()
                .filter(|p| !p.passed)
                .map(|p| p.phase_name.as_str())
                .collect();
            format!("❌ Pipeline validation failed in: {} ({}/{} checks passed)",
                failed_phases.join(", "), passed_checks, total_checks)
        };

        Self {
            phases,
            overall_passed,
            total_checks,
            passed_checks,
            summary,
        }
    }

    /// Generate a detailed report string
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Pipeline Validation Report\n\n");
        report.push_str(&format!("{}\n\n", self.summary));
        report.push_str(&format!("**Total Checks**: {}/{} passed\n\n",
            self.passed_checks, self.total_checks));

        for phase in &self.phases {
            report.push_str(&format!("## Phase: {}\n", phase.phase_name));
            report.push_str(&format!("**Status**: {}\n\n",
                if phase.passed { "✅ PASSED" } else { "❌ FAILED" }));

            // Checks
            report.push_str("### Checks\n");
            for check in &phase.checks {
                let icon = if check.passed { "✅" } else { "❌" };
                report.push_str(&format!("{} **{}**: {}\n", icon, check.name, check.message));
                report.push_str(&format!("   - Expected: {}\n", check.expected));
                report.push_str(&format!("   - Actual: {}\n\n", check.actual));
            }

            // Warnings
            if !phase.warnings.is_empty() {
                report.push_str("### Warnings\n");
                for warning in &phase.warnings {
                    report.push_str(&format!("⚠️  {}\n", warning));
                }
                report.push('\n');
            }

            // Metrics
            if !phase.metrics.is_empty() {
                report.push_str("### Metrics\n");
                for (key, value) in &phase.metrics {
                    report.push_str(&format!("- {}: {:.2}\n", key, value));
                }
                report.push('\n');
            }

            report.push_str("---\n\n");
        }

        report
    }

    /// Get all warnings across all phases
    pub fn all_warnings(&self) -> Vec<String> {
        self.phases
            .iter()
            .flat_map(|p| p.warnings.clone())
            .collect()
    }

    /// Get failed phases
    pub fn failed_phases(&self) -> Vec<&PhaseValidation> {
        self.phases.iter().filter(|p| !p.passed).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DocumentId, ChunkId, EntityId};

    #[test]
    fn test_document_processing_validation() {
        let doc = Document::new(
            DocumentId::new("test".to_string()),
            "Test".to_string(),
            "This is test content with multiple words.".to_string(),
        );

        let chunks = vec![
            TextChunk::new(
                ChunkId::new("c1".to_string()),
                doc.id.clone(),
                "This is test".to_string(),
                0,
                12,
            ),
            TextChunk::new(
                ChunkId::new("c2".to_string()),
                doc.id.clone(),
                "content with multiple words.".to_string(),
                13,
                41,
            ),
        ];

        let validation = DocumentProcessingValidator::validate(&doc, &chunks);
        assert!(validation.passed);
        assert!(validation.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn test_entity_extraction_validation() {
        let chunks = vec![
            TextChunk::new(
                ChunkId::new("c1".to_string()),
                DocumentId::new("test".to_string()),
                "Alice works at Stanford".to_string(),
                0,
                23,
            ),
        ];

        let entities = vec![
            Entity {
                id: EntityId::new("e1".to_string()),
                name: "Alice".to_string(),
                entity_type: "person".to_string(),
                confidence: 0.9,
                mentions: vec![],
                embedding: None,
                first_mentioned: None,
                last_mentioned: None,
                temporal_validity: None,
            },
        ];

        let validation = EntityExtractionValidator::validate(&chunks, &entities);
        assert!(validation.passed);
    }

    #[test]
    fn test_pipeline_report() {
        let doc_validation = PhaseValidation {
            phase_name: "Test Phase".to_string(),
            passed: true,
            checks: vec![
                ValidationCheck {
                    name: "test_check".to_string(),
                    passed: true,
                    expected: "pass".to_string(),
                    actual: "pass".to_string(),
                    message: "OK".to_string(),
                },
            ],
            warnings: vec![],
            metrics: HashMap::new(),
        };

        let report = PipelineValidationReport::from_phases(vec![doc_validation]);
        assert!(report.overall_passed);
        assert_eq!(report.total_checks, 1);
        assert_eq!(report.passed_checks, 1);
    }
}
