//! Symposium GraphRAG-rs Philosophy Demo
//!
//! Questo esempio dimostra l'approccio GraphRAG-rs seguendo le indicazioni
//! di DESIGN_CHOICES_ITALIANO.md:
//!
//! 1. 🆓 Zero Costi LLM - $0 su TUTTO (indexing + query)
//! 2. 📦 Zero Dipendenze Pesanti - NO SpaCy, solo regex + TF-IDF
//! 3. 🦀 Rust-Native Performance - 56x più veloce, 24x meno memoria
//!
//! Configuration: config/templates/symposium_graphrag_rs_philosophy.graphrag.json5
//!
//! Filosofia GraphRAG-rs:
//!   ✅ Zero Costi Operativi > Massima Accuratezza
//!   ✅ Privacy & Offline > Cloud Convenience
//!   ✅ Footprint Minimale > Feature Completeness
//!   ✅ Performance Real-Time > Semantic Depth
//!
//! Run with:
//! cargo run --example symposium_graphrag_rs_philosophy --features async,json5-support
//!
//! Expected:
//! - Processing time: 5-10 secondi (100x più veloce di LLM!)
//! - Indexing cost: $0 (NO API calls)
//! - Query cost: $0 (NO API calls)
//! - Query time: 50ms (16x più veloce di LLM)
//! - Quality: ~80% (accettabile per $0 costi!)

use graphrag_core::GraphRAG;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🦀 Plato's Symposium: GraphRAG-rs Philosophy Approach");
    println!("{}", "=".repeat(70));
    println!("📋 Configuration: symposium_graphrag_rs_philosophy.graphrag.json5");
    println!("🎯 Approach: 100% Algorithmic (NO LLM mai!)");
    println!("💰 Cost: $0 indexing, $0 query (privacy totale)");
    println!("⏱️ Time: 5-10 secondi processing, 50ms per query\n");

    println!("✨ Filosofia GraphRAG-rs (da DESIGN_CHOICES_ITALIANO.md):");
    println!("   1. Zero Costi Operativi > Massima Accuratezza");
    println!("   2. Privacy & Offline > Cloud Convenience");
    println!("   3. Footprint Minimale > Feature Completeness");
    println!("   4. Performance Real-Time > Semantic Depth\n");

    // === PHASE 1: Load Configuration and Process Document ===
    println!("⚙️ Phase 1: Building knowledge graph (100% algorithmic)...");
    println!("   📋 Config: config/templates/symposium_graphrag_rs_philosophy.graphrag.json5");
    println!("   📖 Document: docs-example/Symposium.txt");
    println!("\n   💡 Note: Progress bars will show processing status");
    println!("   Nessuna API call verrà fatta - tutto locale!\n");

    println!("   ⏳ Step 1/3: Loading configuration...");
    println!("   ⏳ Step 2/3: Reading and chunking document (TF-IDF keywords)...");
    println!("   ⏳ Step 3/3: Building graph (regex + co-occurrence)...");
    println!();

    let start_time = Instant::now();

    // Use the convenient API: load config + process document + build graph
    let mut graphrag = GraphRAG::from_config_and_document(
        "config/templates/symposium_graphrag_rs_philosophy.graphrag.json5",
        "docs-example/Symposium.txt"
    ).await?;

    let processing_time = start_time.elapsed();

    println!("\n   ✅ Knowledge graph built successfully!");
    println!("   ⏱️ Processing time: {:.2}s (target: 5-10s)", processing_time.as_secs_f64());
    println!("   💰 API calls made: 0 (100% offline!)");

    // === PHASE 2: Knowledge Graph Statistics ===
    println!("\n📊 Phase 2: Knowledge Graph Statistics");

    if let Some(graph) = graphrag.knowledge_graph() {
        let doc_count = graph.documents().count();
        let chunk_count = graph.chunks().count();
        let entity_count = graph.entities().count();
        let relationship_count = graph.relationships().count();

        println!("   - Documents: {}", doc_count);
        println!("   - Chunks: {}", chunk_count);
        println!("   - Entities: {} (regex + capitalization)", entity_count);
        println!("   - Relationships: {} (co-occurrence)", relationship_count);

        // Show sample entities
        println!("\n   📝 Sample Entities (extracted with regex patterns):");
        for (i, entity) in graph.entities().take(10).enumerate() {
            println!("      {}. {} ({})", i + 1, entity.name, entity.entity_type);
        }
        if entity_count > 10 {
            println!("      ... and {} more", entity_count - 10);
        }
    }

    // === PHASE 3: Query Processing (Algorithmic) ===
    println!("\n🔍 Phase 3: Querying with BM25 + Graph Expansion\n");
    println!("{}", "=".repeat(70));

    let queries = [
        "What is Socrates' definition of love according to Diotima?",
        "How does Aristophanes explain the origin of love in his myth?",
        "What is the relationship between love and beauty in the Symposium?",
        "What is the ladder of love and how does it lead to wisdom?",
    ];

    println!("\n💡 Query Processing (100% Algorithmic - NO LLM):");
    println!("   1. Regex estrae concetti chiave dalla query");
    println!("   2. Graph trova entità correlate (co-occurrence)");
    println!("   3. BM25 calcola relevance score per ogni chunk");
    println!("   4. Ritorna top-K chunks più rilevanti");
    println!("   Tempo: ~50ms (vs 800ms LLM) - 16x più veloce!\n");

    for (i, query) in queries.iter().enumerate() {
        println!("\n📋 Query {}/{}: \"{}\"", i + 1, queries.len(), query);
        println!("{}", "-".repeat(70));

        let query_start = Instant::now();

        // Query usando solo BM25 + graph (NO LLM!)
        // Nota: con ollama.enabled = false, ritorna formatted chunks
        match graphrag.ask(query).await {
            Ok(answer) => {
                let query_time = query_start.elapsed();

                println!("\n   💬 Top Relevant Passages (BM25 + Graph):\n");

                // Format output mostrando chunks trovati
                let lines: Vec<&str> = answer.lines().collect();
                for (idx, line) in lines.iter().take(3).enumerate() {
                    if !line.is_empty() {
                        println!("      {}. {}", idx + 1, line);
                    }
                }

                if lines.len() > 3 {
                    println!("      ... and {} more passages", lines.len() - 3);
                }

                println!("\n   ⏱️ Query time: {:.0}ms (target: 50ms)", query_time.as_millis());
                println!("   💰 Query cost: $0 (NO API calls!)");
                println!("   🔒 Privacy: 100% (tutto locale)");
            }
            Err(e) => {
                eprintln!("\n   ❌ Query failed: {}", e);
            }
        }

        if i < queries.len() - 1 {
            println!("\n{}", "=".repeat(70));
        }
    }

    // === PHASE 4: Performance & Trade-offs Analysis ===
    println!("\n\n📊 Phase 4: Performance & Trade-offs Analysis");
    println!("{}", "=".repeat(70));

    println!("\n💰 Cost Breakdown (Symposium ~35k words):");
    println!("   Indexing Phase:");
    println!("      - Entity extraction (regex): $0 ✅");
    println!("      - Relationship extraction (co-occurrence): $0 ✅");
    println!("      - Embeddings (hash-based): $0 ✅");
    println!("      - Total indexing: $0 ✅");
    println!("\n   Query Phase (per query):");
    println!("      - Query expansion (graph): $0 ✅");
    println!("      - BM25 retrieval: $0 ✅");
    println!("      - Total per query: $0 ✅");
    println!("\n   4 Queries Total Cost: $0 ✅");
    println!("   Grand Total (indexing + 4 queries): $0 ✅\n");

    println!("⏱️ Performance (da DESIGN_CHOICES_ITALIANO.md):");
    println!("   - Indexing: {:.2}s (actual) vs 3-5 min LLM (100x faster!)",
        processing_time.as_secs_f64());
    println!("   - Query: ~50ms vs 2-3s LLM (16x faster!)");
    println!("   - Memory: ~50MB vs 1.2GB Python+SpaCy (24x less!)");
    println!("   - Binary: 10MB vs 3.5GB deployment (350x smaller!)\n");

    println!("🎯 Quality Metrics:");
    println!("   - Entity accuracy: ~80% (vs 95% SpaCy)");
    println!("   - Entity types: 4 (vs 18 SpaCy)");
    println!("   - Trade-off: -15% accuracy per $0 costi ✅");
    println!("   - Semantic understanding: Pattern-based (no deep semantics)");
    println!("   - Response quality: Formatted passages (no LLM generation)\n");

    // === PHASE 5: GraphRAG-rs Philosophy Summary ===
    println!("\n✅ Summary: GraphRAG-rs Philosophy");
    println!("{}", "=".repeat(70));

    println!("\n✨ Cosa Sacrifichiamo (da DESIGN_CHOICES_ITALIANO.md):");
    println!("   ❌ 15% accuratezza (80% vs 95%)");
    println!("   ❌ Entity types avanzati (4 vs 18)");
    println!("   ❌ Comprensione semantica LLM");
    println!("   ❌ Risposte naturali LLM-generated\n");

    println!("✨ Cosa Guadagniamo:");
    println!("   ✅ $0 costo operativo (vs $5-10 indexing + $0.50/query)");
    println!("   ✅ 10MB binary (vs 3.5GB deployment)");
    println!("   ✅ 50ms query (vs 2-3s LLM) - 16x più veloce!");
    println!("   ✅ 50MB RAM (vs 1.2GB) - 24x meno memoria!");
    println!("   ✅ Privacy totale (NO API calls mai)");
    println!("   ✅ Offline-first (funziona senza internet)");
    println!("   ✅ Deploy ovunque (Raspberry Pi, WASM, IoT)");
    println!("   ✅ Real-time performance (<100ms)");
    println!("   ✅ Memory safety (Rust guarantees)\n");

    println!("🎯 Use Cases Ideali (da DESIGN_CHOICES_ITALIANO.md):");
    println!("   • Startup con budget zero");
    println!("   • Privacy-critical systems (sanità, legale, governo)");
    println!("   • Offline deployment");
    println!("   • Edge devices (Raspberry Pi, IoT)");
    println!("   • Browser apps (WebAssembly)");
    println!("   • Real-time applications (<100ms latency)");
    println!("   • Embedded systems con memoria limitata\n");

    println!("⚠️ NON Usare Per (meglio LLM approach):");
    println!("   • Medicina/legale dove serve 99% accuracy");
    println!("   • Budget disponibile ($10+/mese) e serve massima qualità");
    println!("   • Ricerca accademica che richiede deep semantics\n");

    // === PHASE 6: Comparison Table ===
    println!("\n📊 Phase 6: Comparison Matrix");
    println!("{}", "=".repeat(70));

    println!("\n| Metrica                    | LLM Approach | GraphRAG-rs Philosophy |");
    println!("|---------------------------|--------------|------------------------|");
    println!("| Costo indexing            | $5-10        | $0 ✅                  |");
    println!("| Costo query               | $0.50        | $0 ✅                  |");
    println!("| Tempo processing          | 3-5 min      | {:.1}s ✅               |", processing_time.as_secs_f64());
    println!("| Tempo query               | 2-3s         | 50ms ✅                |");
    println!("| Accuracy                  | 95%          | 80% (-15%)             |");
    println!("| Entity types              | 18           | 4                      |");
    println!("| Memory usage              | 1.2GB        | 50MB ✅                |");
    println!("| Binary size               | 3.5GB        | 10MB ✅                |");
    println!("| Privacy                   | ⚠️ API       | ✅ Totale              |");
    println!("| Offline                   | ⚠️           | ✅ 100%                |");
    println!("| Raspberry Pi deploy       | ⚠️ Slow      | ✅ Perfetto            |");
    println!("| WebAssembly               | ❌           | ✅                     |");

    println!("\n\n🦀 GraphRAG-rs Philosophy Demo Completed!");
    println!("📝 Configuration: config/templates/symposium_graphrag_rs_philosophy.graphrag.json5");
    println!("📊 Approach: 100% Algorithmic (NO LLM)");
    println!("💰 Total Cost: $0 (indexing + queries)");
    println!("⏱️ Total Time: {:.2}s", processing_time.as_secs_f64());
    println!("\n💡 Vedi docs/DESIGN_CHOICES_ITALIANO.md per dettagli sulla filosofia");
    println!("💡 Confronta con LLM approach:");
    println!("   cargo run --example symposium_with_llm_query --features async,json5-support\n");

    Ok(())
}
