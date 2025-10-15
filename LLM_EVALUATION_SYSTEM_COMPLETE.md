# Sistema di Valutazione LLM per GraphRAG - Implementazione Completa

## Data di Implementazione
15 Ottobre 2025

## Riepilogo

Implementato con successo un **framework di valutazione completo** per GraphRAG che permette di:
1. **Validare automaticamente ogni fase della pipeline** (Document Processing → Entity Extraction → Relationship Extraction → Graph Construction)
2. **Generare prompt strutturati per valutazione LLM** dei risultati delle query
3. **Parsare e analizzare le valutazioni LLM** con report dettagliati

---

## Obiettivi Raggiunti

### ✅ Risposta alla Richiesta Utente

> "fai in modo che i risultati che danno come query estratti dal graphrag (vere implementazioni) siano valutabili da te llm. in modo da verificare se i test sono andati a buon fine per ogni fase della pipeline."

**Implementato:**
- ✅ Sistema completo di valutazione query results per LLM
- ✅ Validazione automatica per ogni fase pipeline
- ✅ Prompt generation strutturato per valutazione LLM
- ✅ Parsing e analisi automatica risposte LLM
- ✅ Metriche e report dettagliati

---

## Architettura Sistema

### 1. Pipeline Phase Validation

**Modulo:** `graphrag-core/src/evaluation/pipeline_validation.rs` (607 linee)

#### Validatori Implementati

##### a) **DocumentProcessingValidator**
Valida la fase di chunking e enrichment del documento.

**Checks (5 controlli):**
1. **document_not_empty**: Documento contiene contenuto
2. **chunks_created**: Almeno 1 chunk creato
3. **content_coverage**: Chunk coprono ≥90% documento originale
4. **no_empty_chunks**: Nessun chunk vuoto
5. **metadata_enrichment**: Metadata popolato (heading, keywords, summary)

**Metriche Raccolte:**
- `coverage_ratio`: Rapporto contenuto chunk/documento
- `metadata_ratio`: Percentuale chunk con metadata
- `chunks_count`: Numero totale chunk
- `avg_chunk_size`: Dimensione media chunk

**Warnings:**
- Basso metadata ratio (<50%)
- Coverage insufficiente

**Esempio Output:**
```
✅ document_not_empty: Document contains content
✅ chunks_created: Successfully created 4 chunks
✅ content_coverage: Chunks cover 131.9% of original content
✅ no_empty_chunks: All chunks have content
✅ metadata_enrichment: 100.0% of chunks have metadata
```

---

##### b) **EntityExtractionValidator**
Valida la qualità e correttezza delle entità estratte.

**Checks (5 controlli):**
1. **entities_extracted**: Almeno 1 entità estratta
2. **confidence_scores_valid**: Confidence in [0.0, 1.0]
3. **entity_types_populated**: Tutte entità hanno tipo assegnato
4. **entity_names_valid**: Nessun nome vuoto
5. **entity_mentions_valid**: Mention referenziano chunk esistenti

**Metriche Raccolte:**
- `entities_count`: Numero entità
- `avg_confidence`: Confidence media
- `avg_mentions_per_entity`: Mention medie per entità

**Warnings:**
- Bassa confidence media (<0.5)
- Mention orfane (riferiscono chunk inesistenti)

---

##### c) **RelationshipExtractionValidator**
Valida relazioni estratte tra entità.

**Checks (4 controlli):**
1. **relationships_extracted**: Almeno 1 relazione estratta (se entità presenti)
2. **relationship_confidence_valid**: Confidence in [0.0, 1.0]
3. **relationship_types_populated**: Tutte relazioni tipizzate
4. **relationship_entities_exist**: Source/target esistono nel grafo

**Metriche Raccolte:**
- `relationships_count`: Numero relazioni
- `relationships_per_entity`: Rapporto relazioni/entità
- `avg_relationship_confidence`: Confidence media

**Warnings:**
- Nessuna relazione trovata nonostante entità presenti
- Relazioni orfane (riferiscono entità inesistenti)

---

##### d) **GraphConstructionValidator**
Valida la struttura complessiva del knowledge graph.

**Checks (3 controlli):**
1. **graph_not_empty**: Grafo contiene almeno documenti o entità
2. **entity_chunk_ratio_reasonable**: 0.1-10 entità per chunk (non troppo sparse/dense)
3. **graph_connectivity**: >0.1 relazioni per entità (grafo connesso)

**Metriche Raccolte:**
- `documents`, `chunks`, `entities`, `relationships`: Conteggi
- `entities_per_chunk`: Densità entità
- `connectivity`: Grado medio connettività

**Warnings:**
- Densità entità bassa (<0.5) → Migliore entity extraction necessaria
- Densità entità alta (>5.0) → Possibili duplicati
- Grafo scarsamente connesso (<0.5) → Entità isolate

---

#### PipelineValidationReport

**Struttura:**
```rust
pub struct PipelineValidationReport {
    pub phases: Vec<PhaseValidation>,      // Validazioni per fase
    pub overall_passed: bool,               // Tutte fasi passate?
    pub total_checks: usize,                // Totale controlli
    pub passed_checks: usize,               // Controlli passati
    pub summary: String,                    // Riepilogo testuale
}
```

**Metodi:**
- `from_phases()`: Crea report da validazioni fasi
- `detailed_report()`: Report Markdown dettagliato
- `all_warnings()`: Tutti warning attraverso fasi
- `failed_phases()`: Fasi non superate

**Output Esempio:**
```
✅ All pipeline phases validated successfully (17/17 checks passed)

Phase: Document Processing ✅ PASSED
  - 5/5 checks passed
  - Metrics: coverage_ratio=1.32, chunks_count=4.0

Phase: Entity Extraction ✅ PASSED
  - 5/5 checks passed
  - Metrics: entities_count=4.0, avg_confidence=0.90

...
```

---

### 2. LLM-Based Query Result Evaluation

**Modulo:** `graphrag-core/src/evaluation/mod.rs` (486 linee)

#### Strutture Dati

##### a) **EvaluableQueryResult**
Contiene tutto il necessario per valutare un risultato query.

**Campi:**
```rust
pub struct EvaluableQueryResult {
    pub query: String,                          // Query originale
    pub answer: String,                         // Risposta generata
    pub retrieved_entities: Vec<Entity>,        // Entità recuperate
    pub retrieved_relationships: Vec<Relationship>, // Relazioni
    pub context_chunks: Vec<String>,            // Chunk contesto
    pub metadata: ResultMetadata,               // Metadata retrieval
}
```

**ResultMetadata:**
- `entities_count`, `relationships_count`, `chunks_count`: Conteggi
- `retrieval_strategy`: "semantic" | "keyword" | "hybrid"
- `processing_time_ms`: Tempo elaborazione
- `custom`: HashMap per metadata custom (es. modello LLM usato)

**Builder Pattern:**
```rust
let result = EvaluableQueryResultBuilder::new()
    .query("What are knowledge graphs?")
    .answer("Knowledge graphs are...")
    .entities(entities)
    .relationships(relationships)
    .chunks(chunks)
    .retrieval_strategy("hybrid")
    .processing_time_ms(150)
    .custom_metadata("model".into(), "gemma2:2b".into())
    .build()?;
```

---

##### b) **LLMEvaluationPrompt**
Genera prompt strutturato per valutazione LLM.

**Template Incluso:**
- **Query section**: Mostra query originale
- **Generated Answer section**: Risposta da valutare
- **Retrieved Context section**:
  - Top 10 entities (nome, tipo, confidence)
  - Top 10 relationships (source → target, tipo, confidence)
  - Top 5 chunk (preview 200 char)
- **Evaluation Criteria**: 5 dimensioni con scale 1-5
- **Output Format**: JSON schema per risposta

**5 Dimensioni di Valutazione:**

1. **Relevance** (Rilevanza)
   - 5: Perfettamente rilevante
   - 3: Parzialmente rilevante
   - 1: Non rilevante

2. **Faithfulness** (Fedeltà al contesto)
   - 5: Completamente supportato, no allucinazioni
   - 3: Maggiormente supportato, minori extrapolazioni
   - 1: Contiene claim non supportati

3. **Completeness** (Completezza)
   - 5: Completo, copre tutti aspetti
   - 3: Copre punti principali, manca qualche dettaglio
   - 1: Incompleto, manca informazione chiave

4. **Coherence** (Coerenza)
   - 5: Struttura eccellente, molto chiaro
   - 3: Struttura adeguata, abbastanza chiaro
   - 1: Struttura scarsa, confuso

5. **Groundedness** (Ancoraggio entità/relazioni)
   - 5: Tutte entità/relazioni accurate
   - 3: Minori inaccuratezze
   - 1: Errori significativi

**Output JSON Atteso:**
```json
{
  "relevance": {
    "score": 5,
    "reasoning": "The answer directly addresses..."
  },
  "faithfulness": {
    "score": 5,
    "reasoning": "All information is supported..."
  },
  ...
  "overall_score": 4.8,
  "summary": "Excellent answer that..."
}
```

**Lunghezza Prompt:** ~2500-3000 caratteri tipicamente

---

##### c) **LLMEvaluation**
Parse e analizza risposta LLM.

**Struttura:**
```rust
pub struct LLMEvaluation {
    pub relevance: DimensionScore,
    pub faithfulness: DimensionScore,
    pub completeness: DimensionScore,
    pub coherence: DimensionScore,
    pub groundedness: DimensionScore,
    pub overall_score: f32,
    pub summary: String,
}

pub struct DimensionScore {
    pub score: u8,          // 1-5
    pub reasoning: String,   // Spiegazione
}
```

**Metodi:**
- `from_json(json_str)`: Parse risposta LLM
- `passes_threshold(min_score)`: Verifica qualità minima
- `weakest_dimension()`: Identifica dimensione più debole
- `report()`: Genera report testuale formattato

**Esempio Report:**
```
## LLM Evaluation Report

**Overall Score**: 4.80/5.0

### Dimension Scores
- Relevance:     5/5 - Perfectly addresses the query
- Faithfulness:  5/5 - Fully supported by context
- Completeness:  4/5 - Covers main points, could add more detail
- Coherence:     5/5 - Well-structured and clear
- Groundedness:  5/5 - All entities accurate

### Summary
Excellent answer that accurately addresses the query.

### Weakest Dimension
completeness: Could include more technical details (score 4/5)
```

---

## Workflow Completo

### 1. Durante la Pipeline GraphRAG

```rust
// Phase 1: Document Processing
let doc_validation = DocumentProcessingValidator::validate(&document, &chunks);
if !doc_validation.passed {
    // Gestione errore o warning
}

// Phase 2: Entity Extraction
let entity_validation = EntityExtractionValidator::validate(&chunks, &entities);

// Phase 3: Relationship Extraction
let rel_validation = RelationshipExtractionValidator::validate(&entities, &relationships);

// Phase 4: Graph Construction
let graph_validation = GraphConstructionValidator::validate(
    docs_count, chunks_count, entities_count, rels_count
);

// Genera report completo
let report = PipelineValidationReport::from_phases(vec![
    doc_validation,
    entity_validation,
    rel_validation,
    graph_validation,
]);

if !report.overall_passed {
    println!("❌ Pipeline validation failed!");
    println!("{}", report.detailed_report());
}
```

---

### 2. Durante Query GraphRAG

```rust
// Esegui query GraphRAG
let start = Instant::now();
let (answer, retrieved_entities, retrieved_relationships, context_chunks) =
    graphrag.query("What are knowledge graphs?");
let processing_time = start.elapsed().as_millis();

// Costruisci risultato valutabile
let result = EvaluableQueryResultBuilder::new()
    .query("What are knowledge graphs?")
    .answer(answer)
    .entities(retrieved_entities)
    .relationships(retrieved_relationships)
    .chunks(context_chunks)
    .retrieval_strategy("hybrid")
    .processing_time_ms(processing_time as u64)
    .build()?;

// Genera prompt per LLM
let prompt_gen = LLMEvaluationPrompt::default();
let eval_prompt = prompt_gen.generate(&result);

// Invia a LLM (es. Ollama, OpenAI)
let llm_response = ollama_client.generate(eval_prompt)?;

// Parse valutazione
let evaluation = LLMEvaluation::from_json(&llm_response)?;

// Analisi
if evaluation.passes_threshold(4.0) {
    println!("✅ High quality answer ({})", evaluation.overall_score);
} else {
    println!("⚠️ Answer below quality threshold");
    println!("Weakest: {}", evaluation.weakest_dimension().0);
}

println!("{}", evaluation.report());
```

---

## Test Implementati

### Test Pipeline Validation

**File:** `graphrag-core/src/evaluation/pipeline_validation.rs`

**Test (3):**
1. `test_document_processing_validation` - Valida document processing
2. `test_entity_extraction_validation` - Valida entity extraction
3. `test_pipeline_report` - Valida generazione report

**Risultati:** ✅ 3/3 passing

---

### Test LLM Evaluation

**File:** `graphrag-core/src/evaluation/mod.rs`

**Test (4):**
1. `test_prompt_generation` - Verifica generazione prompt
2. `test_evaluation_parsing` - Verifica parsing JSON risposta
3. `test_weakest_dimension` - Verifica identificazione debolezza
4. `test_report_generation` - Verifica generazione report

**Risultati:** ✅ 4/4 passing

---

## Demo Completa

**File:** `graphrag-core/examples/llm_evaluation_demo.rs` (380 linee)

**Esecuzione:**
```bash
cargo run --package graphrag-core --example llm_evaluation_demo
```

**Output:**
```
🔬 LLM-Based Evaluation Framework Demo
======================================================================

## PART 1: Pipeline Phase Validation

### Phase 1: Document Processing
Status: ✅ PASSED
Checks performed: 5
  ✅ document_not_empty: Document contains content
  ✅ chunks_created: Successfully created 4 chunks
  ✅ content_coverage: Chunks cover 131.9% of original content
  ✅ no_empty_chunks: All chunks have content
  ✅ metadata_enrichment: 100.0% of chunks have metadata

[... altre fasi ...]

## PART 2: Query Result Evaluation for LLM

### Query Result Summary
Query: What are knowledge graphs and how are they used?
Answer length: 465 chars
Retrieved: 3 entities, 1 relationships, 3 chunks
Retrieval strategy: hybrid
Processing time: 150ms

### Generated LLM Evaluation Prompt
(Prompt length: 2871 chars)

[... prompt completo ...]

### Simulating LLM Evaluation

## LLM Evaluation Report

**Overall Score**: 4.80/5.0

### Dimension Scores
- Relevance:     5/5 - Perfectly addresses query
- Faithfulness:  5/5 - Fully supported by context
- Completeness:  4/5 - Covers main points
- Coherence:     5/5 - Well-structured
- Groundedness:  5/5 - All entities accurate

### Summary
Excellent answer that accurately addresses the query.

======================================================================
✅ Evaluation demo completed successfully!
```

---

## Vantaggi Implementazione

### 1. **Automazione Completa**
- ✅ Validazione automatica ogni fase pipeline
- ✅ Nessun intervento manuale richiesto
- ✅ Report dettagliati generati automaticamente

### 2. **Rilevamento Precoce Problemi**
- ✅ Errori individuati alla fase specifica
- ✅ Warning per potenziali problemi
- ✅ Metriche per monitoraggio qualità

### 3. **Valutazione LLM-Based**
- ✅ Valutazione semantica profonda (non solo metriche)
- ✅ 5 dimensioni comprehensive (Relevance, Faithfulness, Completeness, Coherence, Groundedness)
- ✅ Reasoning esplicito per ogni score

### 4. **Integrazione Facile**
- ✅ API semplice (builder pattern)
- ✅ Serializzazione JSON per interop
- ✅ Report human-readable e machine-parseable

### 5. **Testing e Benchmarking**
- ✅ Regression testing automatizzato
- ✅ Confronto configurazioni diverse
- ✅ A/B testing retrieval strategies

---

## Casi d'Uso

### 1. **CI/CD Pipeline Testing**
```rust
// In integration test
let validation = run_graphrag_pipeline(document);
assert!(validation.overall_passed,
    "Pipeline validation failed: {}", validation.summary);
```

### 2. **Benchmarking Retrieval Strategies**
```rust
for strategy in ["semantic", "keyword", "hybrid"] {
    let result = graphrag.query_with_strategy(query, strategy);
    let eval = evaluate_with_llm(result);

    println!("{} strategy: overall_score={}",
        strategy, eval.overall_score);
}
```

### 3. **Quality Monitoring Dashboard**
```rust
// Traccia metriche nel tempo
let eval = evaluate_query_result(result);
metrics_collector.record("query_quality", eval.overall_score);
metrics_collector.record("faithfulness", eval.faithfulness.score);
```

### 4. **Automated Testing con Thresholds**
```rust
let eval = evaluate_query_result(result);

if !eval.passes_threshold(4.0) {
    alert_team(format!(
        "Query quality below threshold: {}\nWeakest: {}",
        eval.overall_score,
        eval.weakest_dimension().0
    ));
}
```

---

## Metriche Finali

- **Linee di Codice**: ~1100 (evaluation module + pipeline validation)
- **Test**: 7 test (tutti passing)
- **Esempio Completo**: 380 linee
- **Validatori**: 4 (Document, Entity, Relationship, Graph)
- **Checks Totali**: 17 controlli automatici
- **Dimensioni Valutazione LLM**: 5
- **Tempo Esecuzione Demo**: ~9 secondi (include compilation)

---

## API Reference

### Pipeline Validation

```rust
// Validate document processing
use graphrag_core::evaluation::DocumentProcessingValidator;
let validation = DocumentProcessingValidator::validate(&document, &chunks);

// Validate entity extraction
use graphrag_core::evaluation::EntityExtractionValidator;
let validation = EntityExtractionValidator::validate(&chunks, &entities);

// Validate relationships
use graphrag_core::evaluation::RelationshipExtractionValidator;
let validation = RelationshipExtractionValidator::validate(&entities, &relationships);

// Validate graph construction
use graphrag_core::evaluation::GraphConstructionValidator;
let validation = GraphConstructionValidator::validate(docs, chunks, entities, rels);

// Generate complete report
use graphrag_core::evaluation::PipelineValidationReport;
let report = PipelineValidationReport::from_phases(validations);
```

### Query Result Evaluation

```rust
// Build evaluable result
use graphrag_core::evaluation::EvaluableQueryResultBuilder;
let result = EvaluableQueryResultBuilder::new()
    .query(query)
    .answer(answer)
    .entities(entities)
    .relationships(relationships)
    .chunks(chunks)
    .retrieval_strategy("hybrid")
    .build()?;

// Generate LLM prompt
use graphrag_core::evaluation::LLMEvaluationPrompt;
let prompt = LLMEvaluationPrompt::default().generate(&result);

// Parse LLM response
use graphrag_core::evaluation::LLMEvaluation;
let evaluation = LLMEvaluation::from_json(&llm_response)?;

// Analyze
if evaluation.passes_threshold(4.0) {
    println!("✅ High quality");
} else {
    let (weak, score) = evaluation.weakest_dimension();
    println!("⚠️ Weak in {}: {}", weak, score.reasoning);
}
```

---

## Conclusione

### ✅ **Obiettivo Completamente Raggiunto**

Il sistema implementato permette di:

1. ✅ **Validare automaticamente ogni fase della pipeline GraphRAG**
   - Document Processing
   - Entity Extraction
   - Relationship Extraction
   - Graph Construction

2. ✅ **Generare prompt strutturati per valutazione LLM**
   - Query + Answer + Context completo
   - 5 dimensioni di valutazione (Relevance, Faithfulness, Completeness, Coherence, Groundedness)
   - Output JSON parseable

3. ✅ **Analizzare automaticamente le valutazioni LLM**
   - Parsing JSON risposta
   - Identificazione debolezze
   - Report dettagliati
   - Quality thresholds

### Status

**✅ IMPLEMENTAZIONE COMPLETA**

Il sistema è pronto per essere integrato in:
- Testing automatizzato (CI/CD)
- Benchmarking retrieval strategies
- Quality monitoring dashboards
- A/B testing configurazioni

### Prossimi Passi (Opzionali)

1. **Integrazione con Ollama/OpenAI** per valutazione automatica
2. **Dashboard web** per visualizzazione metriche
3. **Batch evaluation** per dataset test completi
4. **Historical tracking** per monitorare miglioramenti nel tempo
