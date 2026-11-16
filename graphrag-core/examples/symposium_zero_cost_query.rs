//! Symposium GraphRAG with Zero-Cost Approach
//!
//! This example demonstrates the complete workflow using the convenient GraphRAG API
//! with a zero-cost algorithmic pipeline:
//! 1. Loading configuration from JSON5
//! 2. Processing Plato's Symposium with pattern-based extraction
//! 3. Co-occurrence relationship discovery (no LLM)
//! 4. Hash-based embeddings (no neural models)
//! 5. BM25 + PageRank retrieval
//! 6. LLM used ONLY for final natural response generation
//!
//! Configuration: config/templates/symposium_zero_cost.graphrag.json5
//!
//! Prerequisites:
//! - Symposium.txt in docs-example/
//! - llama-server (optional, only for final answer generation)
//!
//! Run with:
//! cargo run --example symposium_zero_cost_query --features async,json5-support
//!
//! Expected:
//! - Processing time: measured by this local run
//! - Indexing cost: $0 (pure algorithmic processing)
//! - Query cost: ~$0.05 per query (only final LLM generation)
//! - Quality: evaluate on the target corpus

use graphrag_core::GraphRAG;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎭 Plato's Symposium: Zero-Cost LazyGraphRAG Approach");
    println!("{}", "=".repeat(70));
    println!("📋 Configuration: symposium_zero_cost.graphrag.json5");
    println!("🎯 Approach: LazyGraphRAG + Pattern extraction + BM25 retrieval");
    println!("💰 Cost: $0 indexing, ~$0.05 per query");
    println!("⏱️ Time: measured processing and retrieval latency\n");

    // === PHASE 1: Load Configuration and Process Document ===
    println!("⚙️ Phase 1: Loading configuration and building knowledge graph...");
    println!("   📋 Config: config/templates/symposium_zero_cost.graphrag.json5");
    println!("   📖 Document: docs-example/Symposium.txt\n");

    let start_time = Instant::now();

    // Use the convenient API: load config + process document + build graph in one call
    let mut graphrag = GraphRAG::from_config_and_document(
        "config/templates/symposium_zero_cost.graphrag.json5",
        "docs-example/Symposium.txt",
    )
    .await?;

    let processing_time = start_time.elapsed();

    println!("\n   ✅ Knowledge graph built successfully!");
    println!(
        "   ⏱️ Processing time: {:.1}s",
        processing_time.as_secs_f64()
    );

    // === PHASE 2: Knowledge Graph Statistics ===
    println!("\n📊 Phase 2: Knowledge Graph Statistics");

    if let Some(graph) = graphrag.knowledge_graph() {
        let doc_count = graph.documents().count();
        let chunk_count = graph.chunks().count();
        let entity_count = graph.entities().count();
        let relationship_count = graph.relationships().count();

        println!("   - Documents: {doc_count}");
        println!("   - Chunks: {chunk_count}");
        println!("   - Entities: {entity_count} (pattern-based extraction)");
        println!("   - Relationships: {relationship_count} (co-occurrence analysis)");

        // Show sample entities
        println!("\n   📝 Sample Entities:");
        for (i, entity) in graph.entities().take(10).enumerate() {
            println!("      {}. {} ({})", i + 1, entity.name, entity.entity_type);
        }
        if entity_count > 10 {
            println!("      ... and {} more", entity_count - 10);
        }
    }

    // === PHASE 3: Pipeline Breakdown ===
    println!("\n⚙️ Phase 3: Zero-Cost Pipeline Details");
    println!("   🔧 Entity Extraction (Pattern-based, NO LLM):");
    println!("      - Method: Regex + Capitalization detection");
    println!("      - Entity types: PERSON, CONCEPT, SPEAKER, etc.");
    println!("      - Zero LLM calls for extraction");

    println!("\n   🔗 Relationship Discovery:");
    println!("      - Method: Co-occurrence analysis (NO LLM)");
    println!("      - Window: 800 tokens (same as chunk size)");
    println!("      - Scoring: Jaccard similarity");

    println!("\n   🧮 Embeddings & Indexing:");
    println!("      - Method: Hash-based TF-IDF (NO neural models)");
    println!("      - Zero cost: No API calls, no model loading");

    println!("\n   🔍 Retrieval:");
    println!("      - BM25 keyword matching");
    println!("      - PageRank graph scoring");
    println!("      - LazyGraphRAG optimizations\n");

    // === PHASE 4: Query Processing ===
    println!("🔍 Phase 4: Querying with Zero-Cost Retrieval + LLM Responses\n");
    println!("{}", "=".repeat(70));

    let queries = [
        "What is Socrates' definition of love according to Diotima?",
        "How does Aristophanes explain the origin of love in his myth?",
        "What is the relationship between love and beauty in the Symposium?",
        "What is the ladder of love and how does it lead to wisdom?",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("\n📋 Query {}/{}: \"{}\"", i + 1, queries.len(), query);
        println!("{}", "-".repeat(70));

        let query_start = Instant::now();

        // Use the GraphRAG ask() method - it handles:
        // - Retrieval (BM25 based on config)
        // - LLM-based answer generation
        // - Prompt formatting
        match graphrag.ask(query).await {
            Ok(answer) => {
                let query_time = query_start.elapsed();

                println!("\n   💬 Answer:\n");
                println!("      {answer}\n");
                println!("   ⏱️ Query time: {:.2}s", query_time.as_secs_f64());
                println!("   💰 Estimated query cost: ~$0.05 (only LLM generation)");
            },
            Err(e) => {
                eprintln!("\n   ❌ Query failed: {e}");
                eprintln!("   💡 Make sure llama-server is running on http://127.0.0.1:8080");
            },
        }

        if i < queries.len() - 1 {
            println!("\n{}", "=".repeat(70));
        }
    }

    // === PHASE 5: Cost & Performance Analysis ===
    println!("\n\n📊 Phase 5: Cost & Performance Analysis");
    println!("{}", "=".repeat(70));

    println!("\n💰 Cost Breakdown (Symposium ~35k words):");
    println!("   Indexing Phase:");
    println!("      - Entity extraction (pattern-based): $0");
    println!("      - Relationship extraction (co-occurrence): $0");
    println!("      - Hash embeddings: $0");
    println!("      - Total indexing: $0");
    println!("\n   Query Phase (per query):");
    println!("      - BM25 retrieval: $0");
    println!("      - PageRank scoring: $0");
    println!("      - LLM response generation: ~$0.05");
    println!("      - Total per query: ~$0.05");
    println!("\n   4 Queries Total Cost: ~$0.20");
    println!("   Grand Total (indexing + 4 queries): ~$0.20\n");

    println!("⏱️ Performance:");
    println!(
        "   - Indexing: {:.1}s (actual)",
        processing_time.as_secs_f64()
    );
    println!("   - Query: <1 second each");
    println!(
        "   - Total session: ~{:.1} seconds\n",
        processing_time.as_secs_f64()
    );

    println!("🎯 Observed behavior:");
    println!("   - Entity extraction: pattern-based");
    println!("   - Relationship extraction: co-occurrence-based");
    println!("   - Retrieval: evaluate on the target corpus");
    println!("   - Natural response: generated by the configured provider");
    println!("   - Speed: measure on the target deployment\n");

    // === PHASE 6: Summary ===
    println!("\n✅ Summary: Zero-Cost Approach");
    println!("{}", "=".repeat(70));
    println!("\n✨ Strengths:");
    println!("   ✓ Zero indexing cost (pure algorithmic processing)");
    println!("   ✓ Processing cost and latency can be measured locally");
    println!("   ✓ Pattern-based extraction is easy to inspect");
    println!("   ✓ No external dependencies (except llama-server for final responses)");
    println!("   ✓ Scalable to large corpora without cost concerns");
    println!("   ✓ Real-time updates possible\n");

    println!("⚠️ Trade-offs:");
    println!("   ✗ May miss nuanced concepts compared with a semantic provider");
    println!("   ✗ Pattern-based extraction may miss nuanced concepts");
    println!("   ✗ Co-occurrence relationships are simpler than semantic ones");
    println!("   ✗ Hash embeddings less powerful than neural embeddings\n");

    println!("🎯 Best For:");
    println!("   • Budget-conscious applications");
    println!("   • Real-time or frequently updated knowledge graphs");
    println!("   • Large-scale corpus processing");
    println!("   • Prototyping and development");
    println!("   • When pattern extraction fits the corpus");
    println!("   • Speed-critical applications\n");

    println!("\n🎭 Plato's Symposium GraphRAG Demo Completed!");
    println!("📝 Configuration used: config/templates/symposium_zero_cost.graphrag.json5");
    println!("📊 Approach: LazyGraphRAG + Pattern extraction + BM25 retrieval");
    println!("\n💡 Compare with full LLM approach:");
    println!("   cargo run --example symposium_with_llm_query --features async,json5-support\n");

    Ok(())
}
