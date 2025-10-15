# Testing con Dati Reali (Fixture-Based Testing) - Proposta per GraphRAG

## Domanda Iniziale

> "Non si potrebbe usare del testo vero, anche una sua porzione più piccola per usarlo come source di testo vero e verificare la pipeline? Questi sono integration tests o end-to-end tests o altro?"

## Risposta: Sì, e sarebbero **Fixture-Based Integration Tests**

---

## 📚 Definizioni dai Best Practices (Ricerca Web)

### **Integration Tests**
> "Integration testing involves evaluating multiple software components collectively to confirm they function correctly when combined. The goal is to identify issues stemming from the interaction between different modules."
>
> — LambdaTest, "End-to-End vs Integration Testing"

**Caratteristiche:**
- ✅ Testano **interazione tra componenti**
- ✅ **Non** testano l'intero sistema
- ✅ Possono usare **mock per servizi esterni**
- ✅ Focus su **interfacce tra moduli**

### **End-to-End Tests**
> "End-to-end testing evaluates the entire application's workflow, replicating actual user scenarios from start to finish, including databases, third-party services, and front-end interfaces."
>
> — Medium, "7 Differences Between Integration and E2E Testing"

**Caratteristiche:**
- ✅ Testano **intero flusso applicativo**
- ✅ Simulano **comportamento utente reale**
- ✅ Includono **tutti i sistemi** (DB, API esterne, UI)
- ✅ **Più lenti e costosi**

### **Fixture-Based Tests**
> "The annotated dataset should accurately mirror the real-world use cases your system will encounter in production. If you use data that is very differently distributed from the real dataset, your evaluation won't be representative."
>
> — deepset, "How to Test an NLP System"

**Caratteristiche:**
- ✅ Usano **dati reali o realistici**
- ✅ Memorizzati come **fixture files**
- ✅ Verificano **comportamento su casi reali**
- ✅ Migliorano **rappresentatività dei test**

---

## 🎯 Classificazione per GraphRAG Pipeline

### **Cosa Abbiamo Ora:**
```
Unit Tests con dati sintetici
├── Input: Stringhe hardcoded nel codice
├── Es: "This is a test document."
└── Verificano: Logica singole funzioni
```

### **Cosa Proponiamo:**
```
Fixture-Based Integration Tests
├── Input: File di testo reali (fixtures/)
├── Es: sample_article.txt, legal_document.pdf
└── Verificano: Pipeline completa su dati reali
```

---

## 📊 Tabella Comparativa

| Aspetto | Unit Tests Attuali | Fixture Integration | End-to-End |
|---------|-------------------|---------------------|------------|
| **Dati** | Sintetici (hardcoded) | Reali (file fixture) | Reali + User Input |
| **Scope** | Singola funzione | Pipeline interna | Sistema completo |
| **Componenti** | 1 modulo | 2-5 moduli | Tutti i moduli |
| **Servizi Esterni** | ❌ Nessuno | ❌ Nessuno | ✅ DB, API, LLM |
| **Tempo** | <1ms | 5-50ms | >1s |
| **Tipo** | Unit | **Integration** | E2E |
| **Per GraphRAG** | ✅ Già fatto | ⚠️ **Da fare** | ❌ Non necessario ora |

---

## 💡 Proposta Implementazione

### **Struttura Directory**
```
graphrag-core/
├── tests/
│   └── fixtures/              # ← NUOVO
│       ├── documents/
│       │   ├── sample_article.txt
│       │   ├── markdown_example.md
│       │   ├── html_sample.html
│       │   ├── technical_doc.txt
│       │   └── narrative_story.txt
│       ├── expected/
│       │   ├── sample_article_chunks.json
│       │   ├── markdown_example_headings.json
│       │   └── technical_doc_keywords.json
│       └── README.md
└── src/
    └── text/
        └── mod.rs
```

### **Esempio di Fixture Test**

#### 1. **File Fixture: `tests/fixtures/documents/sample_article.txt`**
```text
Machine Learning in 2024: A Comprehensive Overview

Machine learning has transformed the technology landscape.
This article explores recent advances in deep learning,
natural language processing, and computer vision.

## Introduction

Artificial intelligence and machine learning have become
ubiquitous in modern applications. From recommendation
systems to autonomous vehicles, these technologies power
countless innovations.

## Deep Learning Fundamentals

Neural networks form the backbone of modern AI systems...
```

#### 2. **Expected Output: `tests/fixtures/expected/sample_article_chunks.json`**
```json
{
  "expected_chunks": 3,
  "expected_headings": [
    {"level": 1, "text": "Machine Learning in 2024: A Comprehensive Overview"},
    {"level": 2, "text": "Introduction"},
    {"level": 2, "text": "Deep Learning Fundamentals"}
  ],
  "expected_keywords": [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural networks"
  ]
}
```

#### 3. **Test Code: `tests/text_pipeline_integration_tests.rs`**
```rust
//! Fixture-Based Integration Tests for Text Pipeline
//!
//! These tests use REAL text documents stored in fixtures/
//! to verify the ENTIRE text processing pipeline works correctly
//! on realistic data.

use graphrag_core::{
    Document, DocumentId, Result,
    text::{TextProcessor, ChunkEnricher, LayoutParserFactory},
};
use std::fs;
use serde_json::Value;

/// Helper: Load fixture file
fn load_fixture(filename: &str) -> String {
    let path = format!("tests/fixtures/documents/{}", filename);
    fs::read_to_string(&path)
        .expect(&format!("Failed to load fixture: {}", filename))
}

/// Helper: Load expected output
fn load_expected(filename: &str) -> Value {
    let path = format!("tests/fixtures/expected/{}", filename);
    let content = fs::read_to_string(&path)
        .expect(&format!("Failed to load expected: {}", filename));
    serde_json::from_str(&content)
        .expect("Failed to parse expected JSON")
}

#[test]
fn test_complete_pipeline_on_real_article() {
    // ========================================
    // QUESTO È UN FIXTURE-BASED INTEGRATION TEST
    // ========================================

    // 1. Load REAL text from fixture
    let content = load_fixture("sample_article.txt");
    let expected = load_expected("sample_article_chunks.json");

    // 2. Create document
    let document = Document::new(
        DocumentId::new("fixture_test".to_string()),
        "sample_article.txt".to_string(),
        content,
    );

    // 3. Run ENTIRE pipeline (multiple components)
    let processor = TextProcessor::new(500, 50).unwrap();
    let chunks = processor.chunk_and_enrich(&document).unwrap();

    // 4. Verify against expected output
    let expected_chunks = expected["expected_chunks"].as_u64().unwrap() as usize;
    assert!(
        chunks.len() >= expected_chunks,
        "Expected at least {} chunks, got {}",
        expected_chunks,
        chunks.len()
    );

    // 5. Verify headings were detected
    let chunks_with_headings: Vec<_> = chunks
        .iter()
        .filter(|c| c.metadata.chapter.is_some())
        .collect();

    assert!(
        !chunks_with_headings.is_empty(),
        "Should detect headings in real document"
    );

    // 6. Verify keywords were extracted
    let chunks_with_keywords: Vec<_> = chunks
        .iter()
        .filter(|c| !c.metadata.keywords.is_empty())
        .collect();

    assert!(
        !chunks_with_keywords.is_empty(),
        "Should extract keywords from real document"
    );

    // 7. Verify specific expected keywords
    let all_keywords: Vec<String> = chunks
        .iter()
        .flat_map(|c| c.metadata.keywords.clone())
        .collect();

    let expected_keywords = expected["expected_keywords"]
        .as_array()
        .unwrap();

    for expected_kw in expected_keywords {
        let kw = expected_kw.as_str().unwrap();
        assert!(
            all_keywords.iter().any(|k| k.to_lowercase().contains(kw)),
            "Expected keyword '{}' not found in extracted keywords",
            kw
        );
    }
}

#[test]
fn test_markdown_heading_extraction_real_doc() {
    // Load REAL markdown document
    let content = load_fixture("markdown_example.md");
    let expected = load_expected("markdown_example_headings.json");

    let document = Document::new(
        DocumentId::new("md_test".to_string()),
        "markdown_example.md".to_string(),
        content,
    );

    // Parse with Markdown parser
    let parser = LayoutParserFactory::create_for_document(&document);
    let structure = parser.parse(&document.content).unwrap();

    // Verify heading count
    let expected_headings = expected["expected_headings"].as_array().unwrap();
    assert_eq!(
        structure.headings.len(),
        expected_headings.len(),
        "Expected {} headings, found {}",
        expected_headings.len(),
        structure.headings.len()
    );

    // Verify heading levels and text
    for (i, expected_h) in expected_headings.iter().enumerate() {
        let expected_level = expected_h["level"].as_u64().unwrap() as usize;
        let expected_text = expected_h["text"].as_str().unwrap();

        let actual_heading = &structure.headings[i];
        assert_eq!(
            actual_heading.level, expected_level,
            "Heading {} has wrong level",
            i
        );
        assert_eq!(
            actual_heading.text.trim(),
            expected_text,
            "Heading {} has wrong text",
            i
        );
    }
}

#[test]
fn test_html_parsing_real_doc() {
    // Load REAL HTML document
    let content = load_fixture("html_sample.html");

    let document = Document::new(
        DocumentId::new("html_test".to_string()),
        "html_sample.html".to_string(),
        content,
    );

    // Parse with HTML parser
    let parser = LayoutParserFactory::create_for_document(&document);
    let structure = parser.parse(&document.content).unwrap();

    // Verify HTML tags are properly removed
    assert!(
        structure.headings.iter().all(|h| !h.text.contains('<')),
        "HTML tags should be removed from heading text"
    );

    // Verify heading hierarchy is preserved
    if structure.headings.len() >= 2 {
        assert!(
            structure.headings[0].level <= structure.headings[1].level,
            "Heading hierarchy should be preserved"
        );
    }
}

#[test]
fn test_technical_document_keyword_extraction() {
    // Load REAL technical document
    let content = load_fixture("technical_doc.txt");
    let expected = load_expected("technical_doc_keywords.json");

    let document = Document::new(
        DocumentId::new("tech_test".to_string()),
        "technical_doc.txt".to_string(),
        content,
    );

    // Extract keywords
    let processor = TextProcessor::new(1000, 100).unwrap();
    let chunks = processor.chunk_and_enrich(&document).unwrap();

    let all_keywords: Vec<String> = chunks
        .iter()
        .flat_map(|c| c.metadata.keywords.clone())
        .collect();

    // Verify technical terms are extracted
    let expected_terms = expected["technical_terms"].as_array().unwrap();

    for term in expected_terms {
        let term_str = term.as_str().unwrap();
        assert!(
            all_keywords.iter().any(|k|
                k.to_lowercase().contains(&term_str.to_lowercase())
            ),
            "Technical term '{}' should be extracted",
            term_str
        );
    }
}

#[test]
fn test_narrative_text_sentence_boundary() {
    // Load REAL narrative story
    let content = load_fixture("narrative_story.txt");

    let processor = TextProcessor::new(300, 50).unwrap();
    let sentences = processor.extract_sentences(&content);

    // Verify sentence boundaries are correct
    assert!(sentences.len() > 5, "Should extract multiple sentences");

    // Verify no sentence ends with abbreviation
    for sentence in &sentences {
        assert!(
            !sentence.ends_with("Dr.") &&
            !sentence.ends_with("Mr.") &&
            !sentence.ends_with("Mrs."),
            "Should not break on abbreviations"
        );
    }

    // Verify sentences are not empty
    for sentence in &sentences {
        assert!(!sentence.trim().is_empty(), "Sentences should not be empty");
    }
}

#[test]
fn test_multilingual_document_detection() {
    // Test with multiple language fixtures
    let test_cases = vec![
        ("english_doc.txt", "en"),
        ("spanish_doc.txt", "es"),
        ("french_doc.txt", "fr"),
    ];

    for (filename, expected_lang) in test_cases {
        let content = load_fixture(filename);
        let detected = graphrag_core::text::LanguageDetector::detect_language(&content);

        assert_eq!(
            detected, expected_lang,
            "Failed to detect language for {}",
            filename
        );
    }
}
```

---

## 🎯 Vantaggi di Questo Approccio

### **1. Rappresentatività**
✅ Test su **dati reali** → risultati più affidabili
✅ Copertura **edge cases** che non pensi a mano
✅ Verifica su **diversi formati** (MD, HTML, TXT)

### **2. Regressione**
✅ **Golden dataset** memorizzato
✅ Rileva quando pipeline **cambia comportamento**
✅ Documentazione **comportamento atteso**

### **3. Debugging**
✅ Facile **riprodurre problemi** (file fixture)
✅ Possibilità di **isolare casi specifici**
✅ **Commit fixture** insieme al fix

### **4. Documentazione**
✅ Fixture = **esempi reali** di input
✅ Expected = **specifica comportamento**
✅ Test = **documentazione eseguibile**

---

## 📈 Best Practices per NLP Testing (da deepset)

> "One of the trickiest tasks in NLP systems is evaluating how well a model functions. Developers need to test both quality and compare different models."

### **Raccomandazioni:**

1. **Annotation Guidelines**
   - Documenta cosa costituisce output corretto
   - Testa guidelines su dati reali

2. **Real User Feedback**
   - Niente batte valutazione utenti reali
   - Prototipi rapidi per A/B testing

3. **Qualitative + Quantitative**
   - Non solo metriche (F1, accuracy)
   - Anche valutazione qualitativa

4. **Representative Dataset**
   - Dataset deve rispecchiare produzione
   - Aggiorna fixture quando cambiano use case

---

## 🚀 Piano di Implementazione

### **Fase 1: Setup (1-2 ore)**
```bash
# 1. Crea struttura directory
mkdir -p tests/fixtures/documents
mkdir -p tests/fixtures/expected

# 2. Aggiungi fixture files
echo "Sample article..." > tests/fixtures/documents/sample_article.txt

# 3. Crea expected outputs
echo '{"expected_chunks": 3}' > tests/fixtures/expected/sample_article_chunks.json
```

### **Fase 2: Primi Test (2-3 ore)**
- Implementa 3-5 test base
- Usa documenti piccoli (<1KB)
- Verifica funzionamento pipeline

### **Fase 3: Espansione (ongoing)**
- Aggiungi fixture quando trovi bug
- Documenta edge cases scoperti
- Copri diversi formati/lingue

---

## 📝 Esempio Fixture Minimo

### **File: `tests/fixtures/documents/minimal_example.txt`**
```text
# Introduction to AI

Artificial Intelligence is transforming technology.
Machine learning enables computers to learn from data.

## Key Concepts

Neural networks mimic brain structure. Deep learning
uses multiple layers for complex pattern recognition.
```

### **File: `tests/fixtures/expected/minimal_example.json`**
```json
{
  "expected_chunks": 2,
  "expected_headings": [
    {"level": 1, "text": "Introduction to AI"},
    {"level": 2, "text": "Key Concepts"}
  ],
  "expected_keywords": ["artificial intelligence", "machine learning", "neural networks"],
  "min_keyword_count": 5
}
```

### **Test:**
```rust
#[test]
fn test_minimal_fixture() {
    let content = load_fixture("minimal_example.txt");
    let expected = load_expected("minimal_example.json");

    let document = Document::new(
        DocumentId::new("min".to_string()),
        "minimal_example.txt".to_string(),
        content,
    );

    let processor = TextProcessor::new(200, 20).unwrap();
    let chunks = processor.chunk_and_enrich(&document).unwrap();

    // Verifica base
    assert!(chunks.len() >= 2, "Should create at least 2 chunks");
    assert!(chunks.iter().any(|c| c.metadata.chapter.is_some()),
            "Should detect headings");
    assert!(chunks.iter().any(|c| !c.metadata.keywords.is_empty()),
            "Should extract keywords");
}
```

---

## ✅ Risposta alla Domanda

### **"Che tipo di test sono?"**

Usando testo vero da file fixture, questi sarebbero:

**✅ Fixture-Based Integration Tests**

**Non sono:**
- ❌ Unit Tests (testano più componenti insieme)
- ❌ End-to-End Tests (non testano sistema completo con DB/API)
- ❌ Property Tests (non generano input random)

**Sono:**
- ✅ **Integration Tests** perché testano pipeline completa
- ✅ **Fixture-Based** perché usano dati reali da file
- ✅ **Regression Tests** perché verificano output atteso
- ✅ **Golden Tests** perché confrontano con reference output

---

## 🎓 Conclusione

### **Raccomandazione:**

**SÌ**, dovresti assolutamente implementare fixture-based integration tests per la pipeline di testo di GraphRAG.

**Perché:**
1. ✅ Maggiore **affidabilità** (test su dati reali)
2. ✅ Migliore **documentazione** (fixture = esempi)
3. ✅ **Regressione** detection (golden dataset)
4. ✅ **Facile debugging** (file fixture riproducibili)
5. ✅ Standard **industry best practice** per NLP

**Effort:**
- Setup iniziale: 2-3 ore
- Manutenzione: Bassa (aggiungi fixture quando trovi bug)
- Valore: Alto (catch regression, migliore confidence)

**Priorità:**
🔥 **Alta** - Implementa subito per pipeline critica (chunking, enrichment)
