#![cfg_attr(not(feature = "async"), allow(unused_imports))]

//! LLM-based entity and relationship extraction
//!
//! This module provides TRUE LLM-based extraction through any OpenAI-compatible endpoint (llama.cpp, Groq, Cerebras, ...).
//! Unlike pattern-based extraction, this uses actual language model inference to extract
//! entities and relationships from text with deep semantic understanding.

use crate::{
    core::{ChunkId, Entity, EntityId, EntityMention, Relationship, TextChunk},
    entity::extraction_cache::ExtractionCache,
    entity::prompts::{
        said_description, EntityData, ExtractionOutput, PromptBuilder, RelationExtractionOutput,
        RelationshipData,
    },
    llm::LlmClient,
    ontology::Ontology,
    GraphRAGError, Result,
};
use serde_json;

/// Cache namespace of relation-only extraction (GLiNER to LLM cascade).
#[cfg(feature = "async")]
const RELATIONS_CACHE_TAG: &str = "relations-v1";

/// Replace every non-ASCII character with an ASCII placeholder (`zZu` + 6 hex digits of the
/// code point + `Zz`), so the text can be sliced at any byte; undone by [`unmask_non_ascii`].
#[cfg(feature = "async")]
fn mask_non_ascii(text: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let _ = write!(out, "zZu{:06x}Zz", c as u32);
        }
    }
    out
}

/// Put back the characters replaced by [`mask_non_ascii`].
#[cfg(feature = "async")]
fn unmask_non_ascii(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find("zZu") {
        out.push_str(&rest[..pos]);
        let token = rest.get(pos + 3..pos + 11);
        let decoded = token
            .filter(|t| t.ends_with("Zz"))
            .and_then(|t| u32::from_str_radix(&t[..6], 16).ok())
            .and_then(char::from_u32);
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &rest[pos + 11..];
            },
            None => {
                out.push_str("zZu");
                rest = &rest[pos + 3..];
            },
        }
    }
    out.push_str(rest);
    out
}

/// LLM-based entity extractor that uses actual language model calls
pub struct LLMEntityExtractor {
    #[cfg_attr(not(feature = "async"), allow(dead_code))]
    llm_client: LlmClient,
    #[cfg_attr(not(feature = "async"), allow(dead_code))]
    prompt_builder: PromptBuilder,
    temperature: f32,
    max_tokens: usize,
    cache: Option<ExtractionCache>,
}

impl LLMEntityExtractor {
    /// Create a new LLM-based entity extractor
    ///
    /// # Arguments
    /// * `llm_client` - LLM client for LLM inference
    /// * `entity_types` - List of entity types to extract (e.g., ["PERSON", "LOCATION", "ORGANIZATION"])
    pub fn new(llm_client: LlmClient, entity_types: Vec<String>) -> Self {
        Self {
            llm_client,
            prompt_builder: PromptBuilder::new(entity_types),
            temperature: 0.0, // Zero temperature for deterministic JSON extraction
            max_tokens: 1500,
            cache: None,
        }
    }

    /// Cache raw extraction responses on disk; identical requests are not re-sent.
    ///
    /// The key covers ontology id, chat model, temperature, max tokens and the
    /// full prompt (template + chunk text).
    pub fn with_cache(mut self, cache: ExtractionCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Use an explicit ontology (replaces the generic one built from `entity_types`).
    ///
    /// Prompts list its type definitions, the output is constrained by its JSON
    /// schema, and extracted labels are normalized onto it.
    pub fn with_ontology(mut self, ontology: Ontology) -> Self {
        let epistemic = self.prompt_builder.epistemic_status();
        self.prompt_builder =
            PromptBuilder::with_ontology(ontology).with_epistemic_status(epistemic);
        self
    }

    /// Ask the model for `status` / `said_by` on every relationship and mark `said`
    /// ones in their description (`entity_extraction.epistemic_status`, default off:
    /// prompts, schemas and cache keys are then exactly the ones without this option).
    pub fn with_epistemic_status(mut self, enabled: bool) -> Self {
        self.prompt_builder = self.prompt_builder.with_epistemic_status(enabled);
        self
    }

    /// Ontology used for prompts and normalization
    pub fn ontology(&self) -> &Ontology {
        self.prompt_builder.ontology()
    }

    /// Set temperature for LLM generation (default: 0.1)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set maximum tokens for LLM generation (default: 1500)
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Extract entities and relationships from a text chunk using LLM
    ///
    /// This is the REAL extraction that makes actual LLM API calls.
    /// Expected time: 15-30 seconds per chunk depending on chunk size and model.
    #[cfg(feature = "async")]
    pub async fn extract_from_chunk(
        &self,
        chunk: &TextChunk,
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "LLM extraction for chunk: {} (size: {} chars)",
            chunk.id,
            chunk.content.len()
        );

        // Build extraction prompt
        let prompt = self.prompt_builder.build_extraction_prompt(&chunk.content);

        // Call LLM for extraction (THIS IS THE REAL LLM CALL!)
        let llm_response = self.call_llm_with_retry(&prompt).await?;

        // Parse response into structured data
        let extraction_output = self.parse_extraction_response(&llm_response)?;

        // Convert to domain entities and relationships
        let entities =
            self.convert_to_entities(&extraction_output.entities, &chunk.id, &chunk.content)?;
        let relationships =
            self.convert_to_relationships(&extraction_output.relationships, &entities, &chunk.id)?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            "LLM extracted {} entities and {} relationships from chunk {}",
            entities.len(),
            relationships.len(),
            chunk.id
        );

        Ok((entities, relationships))
    }

    /// Extract additional entities in a gleaning round (continuation)
    ///
    /// This is used after the initial extraction to catch missed entities.
    #[cfg(feature = "async")]
    pub async fn extract_additional(
        &self,
        chunk: &TextChunk,
        previous_entities: &[EntityData],
        previous_relationships: &[RelationshipData],
    ) -> Result<(Vec<Entity>, Vec<Relationship>)> {
        #[cfg(feature = "tracing")]
        tracing::debug!("LLM gleaning round for chunk: {}", chunk.id);

        // Build continuation prompt with previous extraction
        let prompt = self.prompt_builder.build_continuation_prompt(
            &chunk.content,
            previous_entities,
            previous_relationships,
        );

        // Call LLM for additional extraction
        let llm_response = self.call_llm_with_retry(&prompt).await?;

        // Parse response
        let extraction_output = self.parse_extraction_response(&llm_response)?;

        // Convert to domain entities
        let entities =
            self.convert_to_entities(&extraction_output.entities, &chunk.id, &chunk.content)?;
        let relationships =
            self.convert_to_relationships(&extraction_output.relationships, &entities, &chunk.id)?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            "LLM gleaning extracted {} additional entities and {} relationships",
            entities.len(),
            relationships.len()
        );

        Ok((entities, relationships))
    }

    /// Check if extraction is complete using LLM judgment
    ///
    /// Uses the LLM to determine if all significant entities have been extracted.
    #[cfg(feature = "async")]
    pub async fn check_completion(
        &self,
        chunk: &TextChunk,
        entities: &[EntityData],
        relationships: &[RelationshipData],
    ) -> Result<bool> {
        #[cfg(feature = "tracing")]
        tracing::debug!("LLM completion check for chunk: {}", chunk.id);

        // Build completion check prompt
        let prompt =
            self.prompt_builder
                .build_completion_prompt(&chunk.content, entities, relationships);

        // Call LLM with logit bias for YES/NO response
        let llm_response = self.call_llm_completion_check(&prompt).await?;

        // Parse YES/NO response
        let response_trimmed = llm_response.trim().to_uppercase();
        let is_complete = response_trimmed.starts_with("YES") || response_trimmed.contains("YES");

        #[cfg(feature = "tracing")]
        tracing::debug!(
            "LLM completion check result: {} (response: {})",
            if is_complete {
                "COMPLETE"
            } else {
                "INCOMPLETE"
            },
            llm_response.trim()
        );

        Ok(is_complete)
    }

    /// Extract all chunks with up to `limit` LLM requests in flight.
    ///
    /// Results are returned in chunk order regardless of completion order, so
    /// merging them into the graph is deterministic. `progress(i)` is called as
    /// each chunk (in order) is delivered.
    #[cfg(feature = "async")]
    pub async fn extract_chunks_concurrent(
        &self,
        chunks: &[TextChunk],
        limit: usize,
        progress: impl FnMut(usize),
    ) -> Vec<Result<(Vec<Entity>, Vec<Relationship>)>> {
        run_ordered_concurrent(
            chunks,
            limit,
            |chunk| self.extract_from_chunk(chunk),
            progress,
        )
        .await
    }

    /// Extract only the relationships among `entities` (already found, e.g. by
    /// GLiNER) in `chunk`, with one LLM call.
    ///
    /// The model sees the entity list and may only link entities from it: the
    /// output schema enumerates their names, and any relationship whose source
    /// or target is not in the list, or that links an entity to itself, is
    /// dropped. Fewer than two distinct entity names -> no call, empty result.
    /// Goes through the same cache, retry and JSON-repair path as full extraction.
    #[cfg(feature = "async")]
    pub async fn extract_relationships_for_entities(
        &self,
        chunk: &TextChunk,
        entities: &[Entity],
    ) -> Result<Vec<Relationship>> {
        let names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let distinct: std::collections::HashSet<String> =
            names.iter().map(|n| n.to_lowercase()).collect();
        if distinct.len() < 2 {
            return Ok(Vec::new());
        }

        let prompt = self
            .prompt_builder
            .build_relation_prompt(&chunk.content, entities);
        let schema = self.relation_schema(&names);
        let response = self
            .call_llm_cached(RELATIONS_CACHE_TAG, &prompt, "graph_relations", schema)
            .await?;
        let output = self
            .parse_json::<RelationExtractionOutput>(&response)
            .unwrap_or_else(|| {
                #[cfg(feature = "tracing")]
                tracing::error!(
                    "Failed to parse LLM relation response as JSON. Response preview: {}",
                    &response.chars().take(200).collect::<String>()
                );
                RelationExtractionOutput {
                    relationships: vec![],
                }
            });

        let (kept, _dropped) = Self::filter_relationships(output.relationships, entities);
        #[cfg(feature = "tracing")]
        if _dropped > 0 {
            tracing::debug!(
                chunk_id = %chunk.id,
                dropped = _dropped,
                kept = kept.len(),
                "dropped LLM relationships with unknown entities or self-loops"
            );
        }
        self.convert_to_relationships(&kept, entities, &chunk.id)
    }

    /// [`Self::extract_relationships_for_entities`] for many chunks with up to
    /// `limit` requests in flight. Results come back in job order (a job is a
    /// chunk and its entities), so merging them into the graph is deterministic.
    /// `progress(i)` is called as each job (in order) is delivered.
    #[cfg(feature = "async")]
    pub async fn extract_relationships_concurrent(
        &self,
        jobs: &[(&TextChunk, &[Entity])],
        limit: usize,
        progress: impl FnMut(usize),
    ) -> Vec<Result<Vec<Relationship>>> {
        run_ordered_concurrent(
            jobs,
            limit,
            |(chunk, entities)| self.extract_relationships_for_entities(chunk, entities),
            progress,
        )
        .await
    }

    /// Keep relationships whose source and target both name a listed entity
    /// (case-insensitive) and differ from each other. Returns them with the
    /// number dropped.
    #[cfg(feature = "async")]
    fn filter_relationships(
        relationships: Vec<RelationshipData>,
        entities: &[Entity],
    ) -> (Vec<RelationshipData>, usize) {
        let known: std::collections::HashSet<String> =
            entities.iter().map(|e| e.name.to_lowercase()).collect();
        let total = relationships.len();
        let kept: Vec<RelationshipData> = relationships
            .into_iter()
            .filter(|r| {
                let (source, target) = (r.source.to_lowercase(), r.target.to_lowercase());
                source != target && known.contains(&source) && known.contains(&target)
            })
            .collect();
        let dropped = total - kept.len();
        (kept, dropped)
    }

    /// Cache-aware extraction call with the ontology extraction schema.
    #[cfg(feature = "async")]
    async fn call_llm_with_retry(&self, prompt: &str) -> Result<String> {
        self.call_llm_cached(
            "extraction-v1",
            prompt,
            "graph_extraction",
            self.extraction_schema(),
        )
        .await
    }

    /// Full extraction schema, with the epistemic fields when enabled.
    #[cfg(feature = "async")]
    fn extraction_schema(&self) -> serde_json::Value {
        let schema = self.ontology().extraction_json_schema();
        if self.prompt_builder.epistemic_status() {
            crate::ontology::with_epistemic_status(schema)
        } else {
            schema
        }
    }

    /// Relation-only schema, with the epistemic fields when enabled.
    #[cfg(feature = "async")]
    fn relation_schema(&self, names: &[String]) -> serde_json::Value {
        let schema = self.ontology().relation_extraction_json_schema(names);
        if self.prompt_builder.epistemic_status() {
            crate::ontology::with_epistemic_status(schema)
        } else {
            schema
        }
    }

    /// Cache key of one request. The epistemic option is not a part of it: with the option
    /// on, the prompt (hence the key) already differs; with it off the key is unchanged.
    #[cfg(feature = "async")]
    fn cache_key(&self, key_tag: &str, prompt: &str) -> String {
        ExtractionCache::key(&[
            key_tag,
            &self.ontology().id(),
            &self.llm_client.config().chat_model,
            &format!("{:?}", self.temperature),
            &self.max_tokens.to_string(),
            prompt,
        ])
    }

    /// Cache-aware wrapper around [`Self::call_llm_uncached`].
    ///
    /// `key_tag` separates cache namespaces; `schema` constrains the output.
    /// A response is cached only when the matching parser recovers it:
    /// relation-only responses (`key_tag == "relations-v1"`) are stored as an
    /// `ExtractionOutput` with no entities, which is what `ExtractionCache::get`
    /// validates.
    #[cfg(feature = "async")]
    async fn call_llm_cached(
        &self,
        key_tag: &str,
        prompt: &str,
        schema_name: &str,
        schema: serde_json::Value,
    ) -> Result<String> {
        let Some(cache) = &self.cache else {
            return self.call_llm_uncached(prompt, schema_name, schema).await;
        };
        let key = self.cache_key(key_tag, prompt);
        if let Some(hit) = cache.get(&key) {
            return Ok(hit);
        }
        let response = self.call_llm_uncached(prompt, schema_name, schema).await?;
        // Replay anything the parser can recover (markdown-wrapped / repaired JSON too),
        // stored as normalized JSON so `ExtractionCache::get` accepts it.
        let normalized = if key_tag == RELATIONS_CACHE_TAG {
            self.parse_json::<RelationExtractionOutput>(&response)
                .and_then(|o| {
                    serde_json::to_string(&ExtractionOutput {
                        entities: vec![],
                        relationships: o.relationships,
                    })
                    .ok()
                })
        } else {
            self.cacheable_response(&response)
        };
        if let Some(normalized) = normalized {
            cache.put(&key, &normalized);
        }
        Ok(response)
    }

    /// Call the LLM with one retry for extraction.
    ///
    /// The first attempt constrains the output with `schema` (OpenAI
    /// `json_schema` response format, grammar-enforced by llama.cpp).
    /// The retry drops the schema, so providers that reject it still work and
    /// the parser + ontology normalization handle free-form output.
    ///
    /// The server-side prompt cache (llama.cpp `cache_prompt`) reuses the KV
    /// cache of the shared prompt prefix across chunks automatically.
    #[cfg(feature = "async")]
    async fn call_llm_uncached(
        &self,
        prompt: &str,
        schema_name: &str,
        schema: serde_json::Value,
    ) -> Result<String> {
        use crate::llm::LlmGenerationParams;
        let params = LlmGenerationParams {
            max_tokens: Some(self.max_tokens as u32),
            temperature: Some(self.temperature),
            ..Default::default()
        };
        let constrained = params.clone().with_json_schema(schema_name, schema);
        match self
            .llm_client
            .generate_with_params(prompt, constrained)
            .await
        {
            Ok(response) => Ok(response),
            Err(_e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!("LLM call failed, retrying without JSON schema: {}", _e);
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                self.llm_client.generate_with_params(prompt, params).await
            },
        }
    }

    /// Call LLM for completion check (expects a short YES/NO answer).
    ///
    /// Uses a much smaller `max_tokens` (50) and a deterministic
    /// temperature of 0.0 to get a reliable YES/NO response.
    #[cfg(feature = "async")]
    async fn call_llm_completion_check(&self, prompt: &str) -> Result<String> {
        use crate::llm::LlmGenerationParams;
        let params = LlmGenerationParams {
            max_tokens: Some(50),
            temperature: Some(0.0),
            ..Default::default()
        };
        self.llm_client.generate_with_params(prompt, params).await
    }

    /// Parse LLM response into structured extraction output
    ///
    /// Handles multiple JSON formats and attempts repair if needed
    #[cfg(feature = "async")]
    fn parse_extraction_response(&self, response: &str) -> Result<ExtractionOutput> {
        if let Some(output) = self.try_parse_extraction(response) {
            return Ok(output);
        }

        // If all strategies fail, return empty extraction
        #[cfg(feature = "tracing")]
        tracing::error!(
            "Failed to parse LLM response as JSON. Response preview: {}",
            &response.chars().take(200).collect::<String>()
        );
        Ok(ExtractionOutput {
            entities: vec![],
            relationships: vec![],
        })
    }

    /// Normalized JSON of `response` when it is worth caching (the parser recovered it).
    #[cfg(feature = "async")]
    fn cacheable_response(&self, response: &str) -> Option<String> {
        let output = self.try_parse_extraction(response)?;
        serde_json::to_string(&output).ok()
    }

    /// Run the parsing strategies; `None` when none of them recovers an extraction.
    #[cfg(feature = "async")]
    fn try_parse_extraction(&self, response: &str) -> Option<ExtractionOutput> {
        self.parse_json(response)
    }

    /// Run the parsing strategies for any JSON output type.
    #[cfg(feature = "async")]
    fn parse_json<T: serde::de::DeserializeOwned>(&self, response: &str) -> Option<T> {
        // Strategy 1: Try direct JSON parsing
        if let Ok(output) = serde_json::from_str::<T>(response) {
            return Some(output);
        }

        // Strategy 2: Try to extract JSON from markdown code blocks
        if let Some(json_str) = Self::extract_json_from_markdown(response) {
            if let Ok(output) = serde_json::from_str::<T>(json_str) {
                return Some(output);
            }
        }

        // Strategy 3: Try JSON repair using jsonfixer
        match self.repair_and_parse_json::<T>(response) {
            Ok(output) => return Some(output),
            Err(_e) => {
                #[cfg(feature = "tracing")]
                tracing::warn!("JSON repair failed: {}", _e);
            },
        }

        // Strategy 4: Look for JSON anywhere in the response
        if let Some(json_str) = Self::find_json_in_text(response) {
            if let Ok(output) = serde_json::from_str::<T>(json_str) {
                return Some(output);
            }

            // Try repairing the extracted JSON
            if let Ok(output) = self.repair_and_parse_json::<T>(json_str) {
                return Some(output);
            }
        }

        None
    }

    /// Extract JSON from markdown code blocks
    #[cfg(feature = "async")]
    fn extract_json_from_markdown(text: &str) -> Option<&str> {
        // Look for ```json ... ``` or ``` ... ```
        if let Some(start) = text.find("```json") {
            let json_start = start + 7; // length of ```json
            if let Some(end) = text[json_start..].find("```") {
                return Some(text[json_start..json_start + end].trim());
            }
        }

        if let Some(start) = text.find("```") {
            let json_start = start + 3;
            if let Some(end) = text[json_start..].find("```") {
                let candidate = &text[json_start..json_start + end].trim();
                // Check if it looks like JSON
                if candidate.starts_with('{') || candidate.starts_with('[') {
                    return Some(candidate);
                }
            }
        }

        None
    }

    /// Find JSON object or array anywhere in text
    #[cfg(feature = "async")]
    fn find_json_in_text(text: &str) -> Option<&str> {
        // Find first { and last }
        if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                if end > start {
                    return Some(&text[start..=end]);
                }
            }
        }
        None
    }

    /// Attempt to repair malformed JSON using jsonfixer
    #[cfg(feature = "async")]
    fn repair_and_parse_json<T: serde::de::DeserializeOwned>(&self, json_str: &str) -> Result<T> {
        // jsonfixer::repair_json returns Result<String, Error>
        let options = jsonfixer::JsonRepairOptions::default();
        // jsonfixer 0.1.0 looks ahead by byte offset and panics inside a multi-byte character
        // (seen on truncated Italian output; it also mangles `\\u` escapes), so it only ever
        // sees ASCII: non-ASCII characters are masked before the repair and restored after.
        let fixed_json = jsonfixer::repair_json(&mask_non_ascii(json_str), options)
            .map(|fixed| unmask_non_ascii(&fixed))
            .map_err(|e| GraphRAGError::Generation {
                message: format!("JSON repair failed: {e:?}"),
            })?;

        serde_json::from_str::<T>(&fixed_json).map_err(|e| GraphRAGError::Generation {
            message: format!("Failed to parse repaired JSON: {e}"),
        })
    }

    /// Convert EntityData to domain Entity objects
    #[cfg(feature = "async")]
    fn convert_to_entities(
        &self,
        entity_data: &[EntityData],
        chunk_id: &ChunkId,
        chunk_text: &str,
    ) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        for entity_item in entity_data {
            let entity_type = self
                .ontology()
                .normalize_entity_type(&entity_item.entity_type);
            // Generate entity ID
            let entity_id = EntityId::from_type_and_name(&entity_type, &entity_item.name);

            // Find mentions in chunk
            let mentions = self.find_mentions(&entity_item.name, chunk_id, chunk_text);

            // Create entity with mentions
            // Note: Description is stored in the entity but not used in current Entity struct
            // We store it in the entity name or as a separate field if needed
            let entity = Entity::new(
                entity_id,
                entity_item.name.clone(),
                entity_type,
                0.9, // High confidence since it's LLM-extracted
            )
            .with_mentions(mentions);

            entities.push(entity);
        }

        Ok(entities)
    }

    /// Find all mentions of an entity name in the chunk text
    #[cfg(feature = "async")]
    fn find_mentions(&self, name: &str, chunk_id: &ChunkId, text: &str) -> Vec<EntityMention> {
        let mut mentions = Vec::new();
        let mut start = 0;

        while let Some(pos) = text[start..].find(name) {
            let actual_pos = start + pos;
            mentions.push(EntityMention {
                chunk_id: chunk_id.clone(),
                start_offset: actual_pos,
                end_offset: actual_pos + name.len(),
                confidence: 0.9,
            });
            start = actual_pos + name.len();
        }

        // If no exact matches, try case-insensitive
        if mentions.is_empty() {
            let name_lower = name.to_lowercase();
            let text_lower = text.to_lowercase();
            let mut start = 0;

            while let Some(pos) = text_lower[start..].find(&name_lower) {
                let actual_pos = start + pos;
                mentions.push(EntityMention {
                    chunk_id: chunk_id.clone(),
                    start_offset: actual_pos,
                    end_offset: actual_pos + name.len(),
                    confidence: 0.85, // Slightly lower confidence for case-insensitive match
                });
                start = actual_pos + name.len();
            }
        }

        mentions
    }

    /// Convert RelationshipData to domain Relationship objects
    #[cfg(feature = "async")]
    fn convert_to_relationships(
        &self,
        relationship_data: &[RelationshipData],
        entities: &[Entity],
        chunk_id: &ChunkId,
    ) -> Result<Vec<Relationship>> {
        let mut relationships = Vec::new();

        // Build entity name to ID mapping
        let mut name_to_entity: std::collections::HashMap<String, &Entity> =
            std::collections::HashMap::new();
        for entity in entities {
            name_to_entity.insert(entity.name.to_lowercase(), entity);
        }

        for rel_item in relationship_data {
            // Find source and target entities
            let source_entity = name_to_entity.get(&rel_item.source.to_lowercase());
            let target_entity = name_to_entity.get(&rel_item.target.to_lowercase());

            if let (Some(source), Some(target)) = (source_entity, target_entity) {
                let (relation_type, mut description) = self
                    .ontology()
                    .normalize_relation(&rel_item.relation_type, &rel_item.description);
                if self.prompt_builder.epistemic_status() && rel_item.is_said() {
                    let said_by = normalize_said_by(&rel_item.said_by, entities);
                    description = said_description(description, &said_by);
                }
                let relationship = Relationship {
                    source: source.id.clone(),
                    target: target.id.clone(),
                    relation_type,
                    description,
                    confidence: (rel_item.strength as f32).clamp(0.0, 1.0),
                    context: vec![chunk_id.clone()],
                    embedding: None,
                    temporal_type: None,
                    temporal_range: None,
                    causal_strength: None,
                };

                relationships.push(relationship);
            } else {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    "Skipping relationship: entity not found. Source: {}, Target: {}",
                    rel_item.source,
                    rel_item.target
                );
            }
        }

        Ok(relationships)
    }
}

/// Strip ONE leading article (Italian or English) from `s`; returns `s` unchanged if none.
#[cfg(feature = "async")]
fn strip_article(s: &str) -> &str {
    let lower = s.to_lowercase();
    for a in ["l'", "l\u{2019}", "un'", "un\u{2019}"] {
        if lower.starts_with(a) {
            return &s[a.len()..];
        }
    }
    for a in [
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "the", "a", "an",
    ] {
        if let Some(rest) = lower.strip_prefix(a) {
            if rest.starts_with(char::is_whitespace) {
                // lowercasing can change byte length only for exotic chars; article is ASCII
                return s[a.len()..].trim_start();
            }
        }
    }
    s
}

/// Map the model's raw `said_by` onto an entity name of the same chunk: case-insensitive
/// exact match, else match ignoring one leading article on both sides (first entity in
/// order wins). No match keeps the trimmed raw string (no coreference).
#[cfg(feature = "async")]
fn normalize_said_by(said_by: &str, entities: &[Entity]) -> String {
    let raw = said_by.trim();
    if raw.is_empty() {
        return String::new();
    }
    let raw_lower = raw.to_lowercase();
    if let Some(e) = entities.iter().find(|e| e.name.to_lowercase() == raw_lower) {
        return e.name.clone();
    }
    let bare = strip_article(raw).trim().to_lowercase();
    if bare.is_empty() {
        return raw.to_string();
    }
    entities
        .iter()
        .find(|e| strip_article(e.name.trim()).trim().to_lowercase() == bare)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| raw.to_string())
}

/// Run `f` over `items` with at most `limit` futures in flight (min 1).
///
/// Results come back in item order, whatever the completion order, so callers
/// can merge them deterministically. `progress(i)` fires as item `i` is delivered.
#[cfg(feature = "async")]
pub async fn run_ordered_concurrent<'a, T, R, F, Fut>(
    items: &'a [T],
    limit: usize,
    f: F,
    mut progress: impl FnMut(usize),
) -> Vec<R>
where
    F: Fn(&'a T) -> Fut,
    Fut: std::future::Future<Output = R>,
{
    use futures::stream::{self, StreamExt};
    let mut stream = stream::iter(items.iter().map(f)).buffered(limit.max(1));
    let mut out = Vec::with_capacity(items.len());
    while let Some(r) = stream.next().await {
        progress(out.len());
        out.push(r);
    }
    out
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::{core::DocumentId, llm::LlmConfig};

    fn create_test_chunk() -> TextChunk {
        TextChunk::new(
            ChunkId::new("chunk_001".to_string()),
            DocumentId::new("doc_001".to_string()),
            "Tom Sawyer is a young boy who lives in St. Petersburg with his Aunt Polly. \
             Tom is best friends with Huckleberry Finn. They often go on adventures together."
                .to_string(),
            0,
            150,
        )
    }

    #[test]
    fn filter_relationships_drops_unknown_entities_and_self_loops() {
        let entities = ["Geppetto", "Pinocchio"].map(|n| {
            Entity::new(
                EntityId::from_type_and_name("PERSON", n),
                n.to_string(),
                "PERSON".to_string(),
                0.9,
            )
        });
        let rel = |s: &str, t: &str| RelationshipData {
            source: s.to_string(),
            target: t.to_string(),
            description: String::new(),
            relation_type: "OTHER".to_string(),
            strength: 0.5,
            status: String::new(),
            said_by: String::new(),
        };
        let (kept, dropped) = LLMEntityExtractor::filter_relationships(
            vec![
                rel("Geppetto", "Pinocchio"),
                rel("Geppetto", "Mangiafuoco"), // unknown target
                rel("Ignoto", "Pinocchio"),     // unknown source
                rel("Pinocchio", "pinocchio"),  // self loop (case-insensitive)
                rel("pinocchio", "GEPPETTO"),   // case differences are fine
            ],
            &entities,
        );
        assert_eq!(dropped, 3);
        assert_eq!(kept.len(), 2);
        assert_eq!(
            (kept[0].source.as_str(), kept[0].target.as_str()),
            ("Geppetto", "Pinocchio")
        );
        assert_eq!(kept[1].source, "pinocchio");
    }

    const ACCENTED_JSON: &str = r#"{"entities":[{"name":"Fata","type":"PERSON","description":"è la più buona, perché aiutò il burattino 😀"}],"relationships":[{"source":"Fata","target":"Pinocchio","description":"lo salvò, così poté tornare"}]}"#;

    /// Malformed or truncated outputs with accented text.
    const BROKEN_ACCENTED: [&str; 4] = [
        r#"{"a":"vero, è così","b":"tronc"#,
        r#"{"a":"perché" "b":"così"}"#,
        r#"{"a":"città"è,"b":1}"#,
        r#"{"a":["è" "à" "ù"],"b":"x"è"}"#,
    ];

    #[test]
    fn mask_non_ascii_round_trips() {
        let masked = mask_non_ascii(ACCENTED_JSON);
        assert!(masked.is_ascii());
        assert_eq!(unmask_non_ascii(&masked), ACCENTED_JSON);
        // Text that only looks like a placeholder is left alone.
        assert_eq!(
            unmask_non_ascii("zZu zZu12 zZuGGGGGGZz"),
            "zZu zZu12 zZuGGGGGGZz"
        );
    }

    #[test]
    fn repair_does_not_panic_on_truncated_non_ascii_json() {
        let options = jsonfixer::JsonRepairOptions::default;
        // jsonfixer 0.1.0 panics on raw UTF-8 for inputs like the first BROKEN_ACCENTED case.
        for (cut, _) in ACCENTED_JSON.char_indices() {
            let _ = jsonfixer::repair_json(&mask_non_ascii(&ACCENTED_JSON[..cut]), options());
        }
        for text in BROKEN_ACCENTED {
            let _ = jsonfixer::repair_json(&mask_non_ascii(text), options());
        }
        let fixed = jsonfixer::repair_json(&mask_non_ascii(BROKEN_ACCENTED[0]), options()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&unmask_non_ascii(&fixed)).unwrap();
        assert_eq!(value["a"], "vero, è così");
    }

    #[test]
    fn test_extract_json_from_markdown() {
        let markdown = r#"
Here's the extraction:
```json
{
  "entities": [],
  "relationships": []
}
```
"#;
        let json = LLMEntityExtractor::extract_json_from_markdown(markdown);
        assert!(json.is_some());
        assert!(json.unwrap().contains("entities"));
    }

    #[test]
    fn test_find_json_in_text() {
        let text = "Some text before { \"entities\": [] } some text after";
        let json = LLMEntityExtractor::find_json_in_text(text);
        assert!(json.is_some());
        assert_eq!(json.unwrap(), "{ \"entities\": [] }");
    }

    #[test]
    fn cacheable_response_accepts_markdown_and_repaired_json_but_not_garbage() {
        let extractor = LLMEntityExtractor::new(
            LlmClient::new(LlmConfig::default()),
            vec!["PERSON".to_string()],
        );
        let strict = r#"{"entities":[{"name":"Tom","type":"PERSON","description":"boy"}],"relationships":[]}"#;
        let markdown = format!("Ecco:\n```json\n{strict}\n```\nFine.");
        let truncated = r#"{"entities":[{"name":"Tom","type":"PERSON","description":"boy"}],"relationships":[]"#;

        for response in [strict.to_string(), markdown, truncated.to_string()] {
            let normalized = extractor
                .cacheable_response(&response)
                .unwrap_or_else(|| panic!("should be cacheable: {response}"));
            // what we store must be accepted back by the cache
            let dir = tempfile::tempdir().unwrap();
            let cache = crate::entity::ExtractionCache::new(dir.path().to_path_buf());
            cache.put("k", &normalized);
            assert!(cache.get("k").is_some(), "cache rejected: {normalized}");
        }
        assert!(extractor.cacheable_response("nessun json qui").is_none());
    }

    #[test]
    fn test_parse_valid_json() {
        let llm_config = LlmConfig::default();
        let llm_client = LlmClient::new(llm_config);
        let extractor = LLMEntityExtractor::new(
            llm_client,
            vec!["PERSON".to_string(), "LOCATION".to_string()],
        );

        let response = r#"
{
  "entities": [
    {
      "name": "Tom Sawyer",
      "type": "PERSON",
      "description": "A young boy"
    }
  ],
  "relationships": []
}
"#;

        let result = extractor.parse_extraction_response(response);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.entities.len(), 1);
        assert_eq!(output.entities[0].name, "Tom Sawyer");
    }

    #[test]
    fn test_convert_to_entities() {
        let llm_config = LlmConfig::default();
        let llm_client = LlmClient::new(llm_config);
        let extractor = LLMEntityExtractor::new(llm_client, vec!["PERSON".to_string()]);

        let chunk = create_test_chunk();
        let entity_data = vec![EntityData {
            name: "Tom Sawyer".to_string(),
            entity_type: "PERSON".to_string(),
            description: "A young boy".to_string(),
        }];

        let entities = extractor
            .convert_to_entities(&entity_data, &chunk.id, &chunk.content)
            .unwrap();

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Tom Sawyer");
        assert_eq!(entities[0].entity_type, "PERSON");
        assert!(!entities[0].mentions.is_empty());
    }

    #[test]
    fn test_find_mentions() {
        let llm_config = LlmConfig::default();
        let llm_client = LlmClient::new(llm_config);
        let extractor = LLMEntityExtractor::new(llm_client, vec!["PERSON".to_string()]);

        let chunk = create_test_chunk();
        let mentions = extractor.find_mentions("Tom", &chunk.id, &chunk.content);

        assert!(!mentions.is_empty());
        assert!(mentions.len() >= 2); // "Tom Sawyer" and "Tom is best friends"
    }

    #[test]
    fn test_convert_to_relationships_sets_chunk_context() {
        let extractor = LLMEntityExtractor::new(
            LlmClient::new(LlmConfig::default()),
            vec!["PERSON".to_string()],
        );
        let chunk = create_test_chunk();
        let entity_data = vec![
            EntityData {
                name: "Tom".to_string(),
                entity_type: "PERSON".to_string(),
                description: String::new(),
            },
            EntityData {
                name: "Huckleberry Finn".to_string(),
                entity_type: "PERSON".to_string(),
                description: String::new(),
            },
        ];
        let entities = extractor
            .convert_to_entities(&entity_data, &chunk.id, &chunk.content)
            .unwrap();
        let rels = extractor
            .convert_to_relationships(
                &[RelationshipData {
                    source: "Tom".to_string(),
                    target: "Huckleberry Finn".to_string(),
                    description: "FRIENDS_WITH".to_string(),
                    relation_type: String::new(),
                    strength: 0.9,
                    status: String::new(),
                    said_by: String::new(),
                }],
                &entities,
                &chunk.id,
            )
            .unwrap();
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].context, vec![chunk.id.clone()]);
        // legacy output (label in description) still maps onto the ontology
        assert_eq!(rels[0].relation_type, "FRIEND_OF");
    }

    #[test]
    fn test_conversion_normalizes_onto_ontology() {
        let extractor = LLMEntityExtractor::new(
            LlmClient::new(LlmConfig::default()),
            vec!["PERSON".to_string(), "LOCATION".to_string()],
        );
        let chunk = create_test_chunk();
        let entities = extractor
            .convert_to_entities(
                &[
                    EntityData {
                        name: "Tom".to_string(),
                        entity_type: "character".to_string(),
                        description: String::new(),
                    },
                    EntityData {
                        name: "St. Petersburg".to_string(),
                        entity_type: "City".to_string(),
                        description: String::new(),
                    },
                ],
                &chunk.id,
                &chunk.content,
            )
            .unwrap();
        assert_eq!(entities[0].entity_type, "PERSON");
        assert_eq!(entities[1].entity_type, "LOCATION");
        let rel = |ty: &str, desc: &str| RelationshipData {
            source: "Tom".to_string(),
            target: "St. Petersburg".to_string(),
            description: desc.to_string(),
            relation_type: ty.to_string(),
            strength: 1.7,
            status: String::new(),
            said_by: String::new(),
        };
        let rels = extractor
            .convert_to_relationships(
                &[
                    rel("LIVES_IN", "Tom lives in St. Petersburg"),
                    rel("dreams of", "Tom dreams of the town"),
                ],
                &entities,
                &chunk.id,
            )
            .unwrap();
        assert_eq!(rels[0].relation_type, "LOCATED_IN");
        assert_eq!(
            rels[0].description.as_deref(),
            Some("Tom lives in St. Petersburg")
        );
        assert_eq!(rels[0].confidence, 1.0);
        assert_eq!(rels[1].relation_type, "OTHER");
        assert_eq!(
            rels[1].description.as_deref(),
            Some("dreams of: Tom dreams of the town")
        );
    }

    fn epistemic_rel(status: &str, said_by: &str) -> RelationshipData {
        RelationshipData {
            source: "Tom".to_string(),
            target: "Huckleberry Finn".to_string(),
            description: "Tom lies to Huck".to_string(),
            relation_type: "FRIEND_OF".to_string(),
            strength: 0.8,
            status: status.to_string(),
            said_by: said_by.to_string(),
        }
    }

    fn epistemic_extractor(on: bool) -> LLMEntityExtractor {
        LLMEntityExtractor::new(
            LlmClient::new(LlmConfig::default()),
            vec!["PERSON".to_string()],
        )
        .with_epistemic_status(on)
    }

    fn convert_one(extractor: &LLMEntityExtractor, rel: RelationshipData) -> Option<String> {
        let chunk = create_test_chunk();
        let entities = extractor
            .convert_to_entities(
                &["Tom", "Huckleberry Finn"].map(|n| EntityData {
                    name: n.to_string(),
                    entity_type: "PERSON".to_string(),
                    description: String::new(),
                }),
                &chunk.id,
                &chunk.content,
            )
            .unwrap();
        let rels = extractor
            .convert_to_relationships(&[rel], &entities, &chunk.id)
            .unwrap();
        assert_eq!(rels.len(), 1);
        rels[0].description.clone()
    }

    #[test]
    fn epistemic_off_schema_is_the_ontology_schema_and_ignores_status() {
        let off = epistemic_extractor(false);
        assert_eq!(
            serde_json::to_string(&off.extraction_schema()).unwrap(),
            serde_json::to_string(&off.ontology().extraction_json_schema()).unwrap()
        );
        let names = vec!["Tom".to_string(), "Huck".to_string()];
        assert_eq!(
            off.relation_schema(&names),
            off.ontology().relation_extraction_json_schema(&names)
        );
        // flag off: parsed relationships unchanged, even if the model sent `said`
        assert_eq!(
            convert_one(&off, epistemic_rel("said", "Tom")).as_deref(),
            Some("Tom lies to Huck")
        );
    }

    #[test]
    fn epistemic_on_schema_has_status_enum_and_said_by_last() {
        let on = epistemic_extractor(true);
        for schema in [
            on.extraction_schema(),
            on.relation_schema(&["Tom".to_string(), "Huck".to_string()]),
        ] {
            let props = schema["properties"]["relationships"]["items"]["properties"]
                .as_object()
                .unwrap();
            let keys: Vec<_> = props.keys().cloned().collect();
            assert_eq!(&keys[keys.len() - 2..], ["status", "said_by"]);
            assert_eq!(
                props["status"]["enum"],
                serde_json::json!(["narrated", "said"])
            );
        }
        assert!(on
            .prompt_builder
            .build_extraction_prompt("x")
            .contains("said_by"));
    }

    #[test]
    fn epistemic_flag_survives_with_ontology() {
        let on = epistemic_extractor(true).with_ontology(Ontology::generic());
        assert!(on.prompt_builder.epistemic_status());
    }

    #[test]
    fn epistemic_said_relationships_get_marker() {
        let on = epistemic_extractor(true);
        assert_eq!(
            convert_one(&on, epistemic_rel("said", "Pinocchio")).as_deref(),
            Some("[detto da Pinocchio] Tom lies to Huck")
        );
        assert_eq!(
            convert_one(&on, epistemic_rel("SAID", "")).as_deref(),
            Some("[detto da un personaggio] Tom lies to Huck")
        );
        for status in ["narrated", "", "hearsay"] {
            assert_eq!(
                convert_one(&on, epistemic_rel(status, "Pinocchio")).as_deref(),
                Some("Tom lies to Huck"),
                "status {status:?}"
            );
        }
    }

    #[test]
    fn epistemic_response_parsing_with_and_without_new_fields() {
        let on = epistemic_extractor(true);
        let old_shape = r#"{"entities":[],"relationships":[{"source":"A","target":"B","description":"d","type":"FRIEND_OF","strength":0.5}]}"#;
        let out = on.parse_extraction_response(old_shape).unwrap();
        assert_eq!(out.relationships.len(), 1);
        assert!(!out.relationships[0].is_said());
        // JSON repair path (trailing comma) keeps the new fields
        let broken = r#"{"entities":[],"relationships":[{"source":"A","target":"B","description":"d","type":"FRIEND_OF","strength":0.5,"status":"said","said_by":"Tom",},]}"#;
        let out = on.parse_extraction_response(broken).unwrap();
        assert_eq!(out.relationships.len(), 1);
        assert!(out.relationships[0].is_said());
        assert_eq!(out.relationships[0].said_by, "Tom");
    }

    #[test]
    fn cache_key_flag_off_is_unchanged_and_on_differs() {
        let off = epistemic_extractor(false);
        let on = epistemic_extractor(true);
        let text = "Tom and Huck.";
        let old_key = |prompt: &str| {
            ExtractionCache::key(&[
                "extraction-v1",
                &off.ontology().id(),
                &off.llm_client.config().chat_model,
                &format!("{:?}", off.temperature),
                &off.max_tokens.to_string(),
                prompt,
            ])
        };
        let off_prompt = off.prompt_builder.build_extraction_prompt(text);
        assert_eq!(
            off.cache_key("extraction-v1", &off_prompt),
            old_key(&off_prompt)
        );
        let on_prompt = on.prompt_builder.build_extraction_prompt(text);
        assert_ne!(on_prompt, off_prompt);
        assert_ne!(
            on.cache_key("extraction-v1", &on_prompt),
            off.cache_key("extraction-v1", &off_prompt)
        );
    }

    fn named(names: &[&str]) -> Vec<Entity> {
        names
            .iter()
            .map(|n| {
                Entity::new(
                    EntityId::from_type_and_name("PERSON", n),
                    n.to_string(),
                    "PERSON".to_string(),
                    0.9,
                )
            })
            .collect()
    }

    #[test]
    fn said_by_is_normalized_onto_chunk_entities() {
        let volpe = named(&["Volpe", "Gatto"]);
        assert_eq!(normalize_said_by("la Volpe", &volpe), "Volpe");
        assert_eq!(normalize_said_by("VOLPE", &volpe), "Volpe");
        assert_eq!(normalize_said_by("  il Gatto ", &volpe), "Gatto");
        assert_eq!(normalize_said_by("l'Omino", &named(&["Omino"])), "Omino");
        assert_eq!(
            normalize_said_by("l\u{2019}Omino", &named(&["Omino"])),
            "Omino"
        );
        assert_eq!(normalize_said_by("the Fox", &named(&["Fox"])), "Fox");
        assert_eq!(normalize_said_by("Fata", &named(&["La Fata"])), "La Fata");
        // exact match wins over article stripping
        assert_eq!(
            normalize_said_by("la Fata", &named(&["Fata", "La Fata"])),
            "La Fata"
        );
        // several article-stripped matches: first in entity order
        assert_eq!(
            normalize_said_by("Fata", &named(&["La Fata", "Una Fata"])),
            "La Fata"
        );
        // no match / no coreference: trimmed raw string
        assert_eq!(normalize_said_by(" il burattino ", &volpe), "il burattino");
        assert_eq!(normalize_said_by("   ", &volpe), "");
        assert_eq!(normalize_said_by("", &volpe), "");
    }

    #[test]
    fn said_by_normalization_feeds_the_marker() {
        let volpe = named(&["Volpe"]);
        assert_eq!(
            said_description(Some("x".into()), &normalize_said_by("la Volpe", &volpe)).as_deref(),
            Some("[detto da Volpe] x")
        );
        assert_eq!(
            said_description(Some("x".into()), &normalize_said_by("", &volpe)).as_deref(),
            Some("[detto da un personaggio] x")
        );
    }
}
