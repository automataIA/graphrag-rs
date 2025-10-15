//! LLM-Based Evaluation Demo
//!
//! This example demonstrates how to create evaluable query results from GraphRAG
//! and generate prompts for LLM-based evaluation.
//!
//! Run with:
//! ```bash
//! cargo run --package graphrag-core --example llm_evaluation_demo
//! ```

use graphrag_core::{
    Document, DocumentId, Entity, EntityId, Relationship, ChunkId,
    evaluation::{
        EvaluableQueryResultBuilder, LLMEvaluationPrompt, LLMEvaluation,
        DocumentProcessingValidator, EntityExtractionValidator,
        RelationshipExtractionValidator, GraphConstructionValidator,
        PipelineValidationReport,
    },
    text::TextProcessor,
    Result,
};

fn main() -> Result<()> {
    println!("\n🔬 LLM-Based Evaluation Framework Demo");
    println!("{}", "=".repeat(70));

    // PART 1: Pipeline Phase Validation
    println!("\n## PART 1: Pipeline Phase Validation\n");
    demo_pipeline_validation()?;

    // PART 2: Query Result Evaluation
    println!("\n## PART 2: Query Result Evaluation for LLM\n");
    demo_query_evaluation()?;

    println!("\n{}", "=".repeat(70));
    println!("✅ Evaluation demo completed successfully!\n");

    Ok(())
}

/// Demonstrate pipeline phase validation
fn demo_pipeline_validation() -> Result<()> {
    // Simulate a document processing pipeline
    let document = Document::new(
        DocumentId::new("doc1".to_string()),
        "Knowledge Graphs Overview".to_string(),
        r#"# Introduction to Knowledge Graphs

Knowledge graphs are structured representations of knowledge that capture entities,
their properties, and relationships. They are used in search engines, recommendation
systems, and question answering applications.

## Key Components

### Entities
Entities represent real-world objects like people, organizations, and locations.

### Relationships
Relationships define how entities are connected, such as "works_for" or "located_in".
"#.to_string(),
    );

    // Phase 1: Document Processing
    let processor = TextProcessor::new(200, 50)?;
    let chunks = processor.chunk_and_enrich(&document)?;

    let doc_validation = DocumentProcessingValidator::validate(&document, &chunks);
    println!("### Phase 1: Document Processing");
    println!("Status: {}", if doc_validation.passed { "✅ PASSED" } else { "❌ FAILED" });
    println!("Checks performed: {}", doc_validation.checks.len());
    for check in &doc_validation.checks {
        let icon = if check.passed { "  ✅" } else { "  ❌" };
        println!("{} {}: {}", icon, check.name, check.message);
    }
    if !doc_validation.warnings.is_empty() {
        println!("Warnings:");
        for warning in &doc_validation.warnings {
            println!("  ⚠️  {}", warning);
        }
    }
    println!();

    // Phase 2: Entity Extraction (simulated)
    let entities = vec![
        Entity {
            id: EntityId::new("e1".to_string()),
            name: "Knowledge Graphs".to_string(),
            entity_type: "concept".to_string(),
            confidence: 0.95,
            mentions: vec![],
            embedding: None,
        },
        Entity {
            id: EntityId::new("e2".to_string()),
            name: "Search Engines".to_string(),
            entity_type: "system".to_string(),
            confidence: 0.85,
            mentions: vec![],
            embedding: None,
        },
        Entity {
            id: EntityId::new("e3".to_string()),
            name: "Entities".to_string(),
            entity_type: "concept".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        },
        Entity {
            id: EntityId::new("e4".to_string()),
            name: "Relationships".to_string(),
            entity_type: "concept".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        },
    ];

    let entity_validation = EntityExtractionValidator::validate(&chunks, &entities);
    println!("### Phase 2: Entity Extraction");
    println!("Status: {}", if entity_validation.passed { "✅ PASSED" } else { "❌ FAILED" });
    for check in &entity_validation.checks {
        let icon = if check.passed { "  ✅" } else { "  ❌" };
        println!("{} {}: {}", icon, check.name, check.message);
    }
    println!();

    // Phase 3: Relationship Extraction (simulated)
    let relationships = vec![
        Relationship {
            source: EntityId::new("e1".to_string()),
            target: EntityId::new("e3".to_string()),
            relation_type: "composed_of".to_string(),
            confidence: 0.85,
            context: vec![],
        },
        Relationship {
            source: EntityId::new("e3".to_string()),
            target: EntityId::new("e4".to_string()),
            relation_type: "connected_by".to_string(),
            confidence: 0.8,
            context: vec![],
        },
    ];

    let rel_validation = RelationshipExtractionValidator::validate(&entities, &relationships);
    println!("### Phase 3: Relationship Extraction");
    println!("Status: {}", if rel_validation.passed { "✅ PASSED" } else { "❌ FAILED" });
    for check in &rel_validation.checks {
        let icon = if check.passed { "  ✅" } else { "  ❌" };
        println!("{} {}: {}", icon, check.name, check.message);
    }
    println!();

    // Phase 4: Graph Construction
    let graph_validation = GraphConstructionValidator::validate(
        1,                    // documents
        chunks.len(),         // chunks
        entities.len(),       // entities
        relationships.len(),  // relationships
    );
    println!("### Phase 4: Graph Construction");
    println!("Status: {}", if graph_validation.passed { "✅ PASSED" } else { "❌ FAILED" });
    for check in &graph_validation.checks {
        let icon = if check.passed { "  ✅" } else { "  ❌" };
        println!("{} {}: {}", icon, check.name, check.message);
    }
    println!();

    // Generate complete report
    let report = PipelineValidationReport::from_phases(vec![
        doc_validation,
        entity_validation,
        rel_validation,
        graph_validation,
    ]);

    println!("### Complete Pipeline Report");
    println!("{}", report.summary);
    println!("Total checks: {}/{} passed", report.passed_checks, report.total_checks);

    if !report.all_warnings().is_empty() {
        println!("\n⚠️  All Warnings:");
        for warning in report.all_warnings() {
            println!("   - {}", warning);
        }
    }

    Ok(())
}

/// Demonstrate query result evaluation
fn demo_query_evaluation() -> Result<()> {
    // Simulate a GraphRAG query result
    let query = "What are knowledge graphs and how are they used?";
    let answer = "Knowledge graphs are structured representations of knowledge that capture \
                  entities, properties, and relationships. They are widely used in search engines \
                  like Google for enhancing search results, in recommendation systems to find related \
                  items, and in question-answering systems to provide accurate responses. The key \
                  components include entities (real-world objects like people and organizations) and \
                  relationships (connections such as 'works_for' or 'located_in').";

    // Retrieved entities
    let entities = vec![
        Entity {
            id: EntityId::new("e1".to_string()),
            name: "Knowledge Graphs".to_string(),
            entity_type: "concept".to_string(),
            confidence: 0.95,
            mentions: vec![],
            embedding: None,
        },
        Entity {
            id: EntityId::new("e2".to_string()),
            name: "Google".to_string(),
            entity_type: "organization".to_string(),
            confidence: 0.9,
            mentions: vec![],
            embedding: None,
        },
        Entity {
            id: EntityId::new("e3".to_string()),
            name: "Search Engines".to_string(),
            entity_type: "system".to_string(),
            confidence: 0.85,
            mentions: vec![],
            embedding: None,
        },
    ];

    // Retrieved relationships
    let relationships = vec![
        Relationship {
            source: EntityId::new("e2".to_string()),
            target: EntityId::new("e3".to_string()),
            relation_type: "is_a".to_string(),
            confidence: 0.9,
            context: vec![],
        },
    ];

    // Context chunks
    let chunks = vec![
        "Knowledge graphs are structured representations of knowledge.".to_string(),
        "They are used in search engines, recommendation systems, and QA applications.".to_string(),
        "Google introduced the Google Knowledge Graph in 2012.".to_string(),
    ];

    // Build evaluable query result
    let result = EvaluableQueryResultBuilder::new()
        .query(query)
        .answer(answer)
        .entities(entities)
        .relationships(relationships)
        .chunks(chunks)
        .retrieval_strategy("hybrid")
        .processing_time_ms(150)
        .custom_metadata("model".to_string(), "gemma2:2b".to_string())
        .build()?;

    println!("### Query Result Summary");
    println!("Query: {}", result.query);
    println!("Answer length: {} chars", result.answer.len());
    println!("Retrieved: {} entities, {} relationships, {} chunks",
        result.metadata.entities_count,
        result.metadata.relationships_count,
        result.metadata.chunks_count);
    println!("Retrieval strategy: {}", result.metadata.retrieval_strategy);
    println!("Processing time: {}ms", result.metadata.processing_time_ms);
    println!();

    // Generate LLM evaluation prompt
    let prompt_generator = LLMEvaluationPrompt::default();
    let evaluation_prompt = prompt_generator.generate(&result);

    println!("### Generated LLM Evaluation Prompt");
    println!("(Prompt length: {} chars)", evaluation_prompt.len());
    println!("\n--- BEGIN PROMPT ---");
    println!("{}", evaluation_prompt);
    println!("--- END PROMPT ---\n");

    // Simulate LLM response parsing
    println!("### Simulating LLM Evaluation");
    let mock_llm_response = r#"{
  "relevance": {
    "score": 5,
    "reasoning": "The answer directly addresses what knowledge graphs are and provides specific use cases as requested"
  },
  "faithfulness": {
    "score": 5,
    "reasoning": "All information in the answer is supported by the retrieved context chunks. No hallucination detected."
  },
  "completeness": {
    "score": 4,
    "reasoning": "Covers definition and main use cases. Could include more detail on technical implementation."
  },
  "coherence": {
    "score": 5,
    "reasoning": "Well-structured answer with clear flow from definition to applications to components"
  },
  "groundedness": {
    "score": 5,
    "reasoning": "All entities (Knowledge Graphs, Google, Search Engines) and relationships correctly mentioned and used"
  },
  "overall_score": 4.8,
  "summary": "Excellent answer that accurately and comprehensively addresses the query with strong grounding in retrieved context."
}"#;

    let evaluation = LLMEvaluation::from_json(mock_llm_response)?;

    println!("\n{}", evaluation.report());

    // Check quality threshold
    println!("### Quality Checks");
    println!("Passes threshold 4.0: {}", evaluation.passes_threshold(4.0));
    println!("Passes threshold 4.5: {}", evaluation.passes_threshold(4.5));
    println!("Passes threshold 5.0: {}", evaluation.passes_threshold(5.0));

    let (weak_dim, weak_score) = evaluation.weakest_dimension();
    println!("\nWeakest dimension: {} (score: {})", weak_dim, weak_score.score);
    println!("Recommendation: {}", weak_score.reasoning);

    Ok(())
}
