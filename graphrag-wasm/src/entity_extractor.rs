//! Entity Extraction using WebLLM with Rule-Based Fallback
//!
//! Extracts entities and relationships from text using WebLLM (Qwen) when available,
//! or falls back to simple rule-based extraction.

use crate::webllm::{WebLLM, ChatMessage};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub from: String,
    pub relation: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub entities: Vec<Entity>,
    pub relationships: Vec<Relationship>,
}

/// Extract entities and relationships from text using WebLLM
pub async fn extract_entities(llm: &WebLLM, text: &str) -> Result<ExtractionResult, String> {
    let prompt = format!(
        r#"Extract entities and relationships from the following text. Return ONLY valid JSON, no markdown, no explanations.

Format:
{{
  "entities": [
    {{"name": "entity name", "entity_type": "PERSON|ORGANIZATION|CONCEPT|TECHNOLOGY|LOCATION", "description": "brief description"}}
  ],
  "relationships": [
    {{"from": "entity1", "relation": "relationship type", "to": "entity2"}}
  ]
}}

Text:
{}

JSON:"#,
        text
    );

    web_sys::console::log_1(&format!("🤖 Extracting entities with WebLLM...").into());

    let messages = vec![
        ChatMessage::system("You are a knowledge graph entity extractor. Extract entities and relationships. Return ONLY valid JSON."),
        ChatMessage::user(&prompt),
    ];

    match llm.chat(messages, Some(0.3), Some(512)).await {
        Ok(response) => {
            web_sys::console::log_1(&format!("📝 LLM response: {}", response).into());

            // Clean response - remove markdown code blocks if present
            let json_str = response
                .trim()
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();

            // Try to parse JSON
            match serde_json::from_str::<ExtractionResult>(json_str) {
                Ok(result) => {
                    web_sys::console::log_1(&format!(
                        "✅ Extracted {} entities, {} relationships",
                        result.entities.len(),
                        result.relationships.len()
                    ).into());
                    Ok(result)
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!(
                        "⚠️  Failed to parse JSON: {}. Response: {}",
                        e, json_str
                    ).into());

                    // Return empty result instead of error
                    Ok(ExtractionResult {
                        entities: Vec::new(),
                        relationships: Vec::new(),
                    })
                }
            }
        }
        Err(e) => {
            web_sys::console::error_1(&format!("❌ LLM inference failed: {}", e).into());
            Err(format!("LLM inference failed: {}", e))
        }
    }
}

/// Batch extract entities from multiple text chunks
#[allow(dead_code)]
pub async fn batch_extract_entities(
    llm: &WebLLM,
    texts: Vec<String>,
) -> Result<ExtractionResult, String> {
    let mut all_entities = Vec::new();
    let mut all_relationships = Vec::new();

    for (idx, text) in texts.iter().enumerate() {
        web_sys::console::log_1(&format!("📄 Processing chunk {}/{}", idx + 1, texts.len()).into());

        match extract_entities(llm, text).await {
            Ok(result) => {
                all_entities.extend(result.entities);
                all_relationships.extend(result.relationships);
            }
            Err(e) => {
                web_sys::console::warn_1(&format!(
                    "⚠️  Failed to extract from chunk {}: {}",
                    idx + 1, e
                ).into());
            }
        }

        // Small delay to avoid rate limits
        gloo_timers::future::TimeoutFuture::new(100).await;
    }

    // Deduplicate entities by name
    all_entities.sort_by(|a, b| a.name.cmp(&b.name));
    all_entities.dedup_by(|a, b| a.name == b.name);

    Ok(ExtractionResult {
        entities: all_entities,
        relationships: all_relationships,
    })
}

/// Simple rule-based entity extraction (fallback when WebLLM is unavailable)
///
/// Extracts:
/// - Capitalized words as potential entities (PERSON/ORGANIZATION)
/// - Technical terms as TECHNOLOGY entities
/// - Common relationship patterns
pub fn extract_entities_simple(text: &str) -> ExtractionResult {
    web_sys::console::log_1(&"📝 Using simple rule-based entity extraction...".into());

    let mut entities = Vec::new();
    let mut seen_names = HashSet::new();

    // Common technical terms to identify as TECHNOLOGY entities
    let tech_terms = [
        "GraphRAG", "LLM", "language model", "knowledge graph", "neural network",
        "embedding", "retrieval", "generation", "WebGPU", "WASM", "Rust",
        "entity extraction", "relationship", "vector", "semantic search",
        "transformer", "attention", "tokenizer", "ONNX", "WebLLM",
    ];

    // Extract capitalized phrases (potential PERSON/ORGANIZATION/CONCEPT entities)
    let words: Vec<&str> = text.split_whitespace().collect();
    for window in words.windows(3) {
        let phrase = window.join(" ");

        // Check if starts with capital letter
        if let Some(first_char) = phrase.chars().next() {
            if first_char.is_uppercase() && phrase.len() > 3 {
                let clean = phrase.trim_end_matches(|c: char| !c.is_alphanumeric());
                if !seen_names.contains(clean) && clean.len() > 2 {
                    entities.push(Entity {
                        name: clean.to_string(),
                        entity_type: "CONCEPT".to_string(),
                        description: format!("Extracted from text: {}", clean),
                    });
                    seen_names.insert(clean.to_string());
                }
            }
        }
    }

    // Extract technical terms
    for term in tech_terms.iter() {
        if text.to_lowercase().contains(&term.to_lowercase()) {
            if !seen_names.contains(*term) {
                entities.push(Entity {
                    name: term.to_string(),
                    entity_type: "TECHNOLOGY".to_string(),
                    description: format!("Technology: {}", term),
                });
                seen_names.insert(term.to_string());
            }
        }
    }

    // Extract simple relationships based on common patterns
    let mut relationships = Vec::new();

    // "X uses Y" pattern
    if let Some(uses_idx) = text.find(" uses ") {
        let before = &text[uses_idx.saturating_sub(30)..uses_idx];
        let after = &text[uses_idx + 6..std::cmp::min(uses_idx + 40, text.len())];

        if let Some(subj) = before.split_whitespace().last() {
            if let Some(obj) = after.split_whitespace().next() {
                relationships.push(Relationship {
                    from: subj.to_string(),
                    relation: "USES".to_string(),
                    to: obj.trim_end_matches(|c: char| !c.is_alphanumeric()).to_string(),
                });
            }
        }
    }

    // "X combines Y" pattern
    if let Some(combines_idx) = text.find(" combines ") {
        let before = &text[combines_idx.saturating_sub(30)..combines_idx];
        let after = &text[combines_idx + 10..std::cmp::min(combines_idx + 40, text.len())];

        if let Some(subj) = before.split_whitespace().last() {
            if let Some(obj) = after.split_whitespace().next() {
                relationships.push(Relationship {
                    from: subj.to_string(),
                    relation: "COMBINES".to_string(),
                    to: obj.trim_end_matches(|c: char| !c.is_alphanumeric()).to_string(),
                });
            }
        }
    }

    web_sys::console::log_1(&format!(
        "✅ Simple extraction: {} entities, {} relationships",
        entities.len(),
        relationships.len()
    ).into());

    ExtractionResult {
        entities,
        relationships,
    }
}
