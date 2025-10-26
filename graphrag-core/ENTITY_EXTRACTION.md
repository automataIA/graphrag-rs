# Entity Extraction System

This document describes the TRUE LLM-based entity extraction system implemented in GraphRAG Core.

## Overview

GraphRAG Core implements **genuine LLM-based gleaning extraction**, following the Microsoft GraphRAG approach with multi-round iterative refinement. This is NOT pattern matching or heuristics - it makes real API calls to LLMs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  GleaningEntityExtractor                     │
│                  (Orchestrates N rounds)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLMEntityExtractor                          │
│           (Makes actual Ollama API calls)                    │
│                                                              │
│  • extract_from_chunk()    - Initial extraction             │
│  • extract_additional()    - Gleaning continuation          │
│  • check_completion()      - LLM-based completion check     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PromptBuilder                             │
│         (Microsoft GraphRAG-style prompts)                   │
│                                                              │
│  • ENTITY_EXTRACTION_PROMPT      - Initial extraction       │
│  • GLEANING_CONTINUATION_PROMPT  - Additional rounds        │
│  • COMPLETION_CHECK_PROMPT       - Completion judgment      │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. PromptBuilder (`src/entity/prompts.rs`)

Generates Microsoft GraphRAG-style prompts:

```rust
pub const ENTITY_EXTRACTION_PROMPT: &str = r#"
-Goal-
Given a text document and entity types, identify all entities and relationships.

-Steps-
1. Identify all entities with:
   - entity_name: Name (capitalized)
   - entity_type: One of [{entity_types}]
   - entity_description: Comprehensive description

2. Identify relationships between entities:
   - source_entity: Source entity name
   - target_entity: Target entity name
   - relationship_description: Relationship explanation
   - relationship_strength: Score from 1-10

3. Return JSON format:
{
  "entities": [...],
  "relationships": [...]
}
"#;
```

**Methods**:
- `build_extraction_prompt(text)` - Initial extraction
- `build_continuation_prompt(text, prev_entities, prev_rels)` - Gleaning rounds
- `build_completion_prompt(text, entities, rels)` - Check if complete

### 2. LLMEntityExtractor (`src/entity/llm_extractor.rs`)

Makes real LLM API calls:

```rust
pub struct LLMEntityExtractor {
    ollama_client: OllamaClient,
    prompt_builder: PromptBuilder,
    temperature: f32,
    max_tokens: usize,
}

impl LLMEntityExtractor {
    /// Extract entities using REAL LLM call
    /// Expected time: 15-30 seconds per chunk
    pub async fn extract_from_chunk(&self, chunk: &TextChunk)
        -> Result<(Vec<Entity>, Vec<Relationship>)>
    {
        // 1. Build prompt
        let prompt = self.prompt_builder.build_extraction_prompt(&chunk.content);

        // 2. Call LLM (THIS IS THE REAL API CALL!)
        let llm_response = self.call_llm_with_retry(&prompt).await?;

        // 3. Parse JSON response (multi-strategy)
        let extraction_output = self.parse_extraction_response(&llm_response)?;

        // 4. Convert to domain objects
        Ok((entities, relationships))
    }

    /// Extract additional entities in gleaning round
    pub async fn extract_additional(&self, ...) -> Result<...> { ... }

    /// Check if extraction is complete using LLM judgment
    pub async fn check_completion(&self, ...) -> Result<bool> { ... }
}
```

**Features**:
- Multi-strategy JSON parsing (direct → markdown → repair → regex)
- Automatic retry on API failures (3 attempts)
- jsonfixer integration for malformed JSON repair
- Confidence scoring and mention tracking

### 3. GleaningEntityExtractor (`src/entity/gleaning_extractor.rs`)

Orchestrates multi-round gleaning:

```rust
pub struct GleaningEntityExtractor {
    llm_extractor: LLMEntityExtractor,
    config: GleaningConfig,
}

pub struct GleaningConfig {
    pub max_gleaning_rounds: usize,           // Default: 4
    pub completion_threshold: f64,            // Default: 0.8
    pub entity_confidence_threshold: f64,     // Default: 0.7
    pub use_llm_completion_check: bool,       // Default: true
    pub entity_types: Vec<String>,            // Required
    pub temperature: f32,                     // Default: 0.1
    pub max_tokens: usize,                    // Default: 1500
}

impl GleaningEntityExtractor {
    /// Extract with iterative refinement
    /// Expected time: 60-120 seconds per chunk (4 rounds × 15-30s)
    pub async fn extract_with_gleaning(&self, chunk: &TextChunk)
        -> Result<(Vec<Entity>, Vec<Relationship>)>
    {
        // Round 1: Initial extraction
        let (entities, relationships) = self.llm_extractor
            .extract_from_chunk(chunk).await?;

        // Rounds 2-N: Gleaning continuation
        for round in 2..=self.config.max_gleaning_rounds {
            // Check if complete (LLM judgment)
            if self.llm_extractor.check_completion(...).await? {
                break;
            }

            // Extract additional entities
            let (new_entities, new_rels) = self.llm_extractor
                .extract_additional(chunk, &all_entities, &all_rels).await?;

            if new_entities.is_empty() && new_rels.is_empty() {
                break;
            }

            // Merge using length-based strategy (LightRAG approach)
            all_entities = self.merge_entity_data(all_entities, new_entities);
            all_relationships.extend(new_rels);
        }

        Ok((final_entities, deduplicated_relationships))
    }
}
```

## Usage Example

### Basic Usage

```rust
use graphrag_core::entity::gleaning_extractor::GleaningEntityExtractor;
use graphrag_core::entity::GleaningConfig;
use graphrag_core::ollama::OllamaClient;
use graphrag_core::text::TextChunk;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create Ollama client
    let ollama_config = OllamaConfig {
        host: "http://localhost".to_string(),
        port: 11434,
        chat_model: "llama3.1:8b".to_string(),
        embedding_model: "nomic-embed-text".to_string(),
        timeout_seconds: 120,
    };
    let ollama_client = OllamaClient::new(ollama_config);

    // 2. Configure gleaning
    let gleaning_config = GleaningConfig {
        max_gleaning_rounds: 4,
        completion_threshold: 0.8,
        entity_confidence_threshold: 0.7,
        use_llm_completion_check: true,
        entity_types: vec![
            "PERSON".to_string(),
            "ORGANIZATION".to_string(),
            "LOCATION".to_string(),
            "EVENT".to_string(),
        ],
        temperature: 0.1,
        max_tokens: 1500,
    };

    // 3. Create extractor
    let extractor = GleaningEntityExtractor::new(ollama_client, gleaning_config);

    // 4. Create text chunk
    let chunk = TextChunk {
        id: "chunk_001".to_string(),
        content: "Tom Sawyer and Huckleberry Finn lived in St. Petersburg, Missouri...".to_string(),
        start: 0,
        end: 100,
        metadata: Default::default(),
    };

    // 5. Extract (THIS TAKES 60-120 SECONDS!)
    println!("Starting extraction (this will take 1-2 minutes)...");
    let (entities, relationships) = extractor.extract_with_gleaning(&chunk).await?;

    println!("Extracted {} entities and {} relationships",
        entities.len(), relationships.len());

    for entity in entities {
        println!("  - {}: {} (confidence: {:.2})",
            entity.name, entity.entity_type, entity.confidence);
    }

    Ok(())
}
```

### Integration with GraphRAG

```rust
use graphrag_core::GraphRAG;

// Build GraphRAG with gleaning extractor
let graphrag = GraphRAG::builder()
    .with_config_file("config.toml")?
    .build_graph()  // Automatically uses GleaningEntityExtractor
    .await?;

// The extractor is used internally during build_graph()
```

## Performance Characteristics

### Processing Times

| Document Size | Chunks | Time per Chunk | Total Time (4 rounds) |
|---------------|--------|----------------|-----------------------|
| 5-10 pages    | ~20    | 60-120s        | 5-15 minutes          |
| 50-100 pages  | ~200   | 60-120s        | 30-60 minutes         |
| 500-1000 pages| ~2000  | 60-120s        | 2-4 hours             |

**Factors affecting time**:
- Chunk size (default: 500 chars)
- Number of gleaning rounds (default: 4)
- LLM model speed (llama3.1:8b is mid-range)
- Early termination (if LLM says "complete" before round 4)

### Memory Usage

- **LLM Client**: ~50MB
- **Conversation History**: ~1-5MB per chunk
- **Entity Storage**: ~100KB per 1000 entities
- **Total**: ~100-500MB for typical workloads

### Comparison: Old (Fake) vs New (Real)

| Metric | Old (Pattern Matching) | New (Real LLM) |
|--------|------------------------|----------------|
| Processing Time | <1 second | 2-4 hours |
| Quality | Low (misses context) | High (semantic understanding) |
| Entity Types | Limited patterns | Configurable, semantic |
| Relationships | Basic co-occurrence | True semantic relationships |
| Confidence | Heuristic | LLM-judged |
| API Calls | 0 | N × chunks × rounds |

## Configuration

### Via TOML

```toml
[entities]
# Gleaning configuration
max_gleaning_rounds = 4
min_confidence = 0.7
use_llm_completion_check = true

# Entity types to extract
entity_types = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "TECHNOLOGY",
    "CONCEPT"
]

[ollama]
enabled = true
host = "http://localhost"
port = 11434
chat_model = "llama3.1:8b"
embedding_model = "nomic-embed-text"
timeout_seconds = 120
```

### Programmatic

```rust
let config = GleaningConfig {
    max_gleaning_rounds: 4,        // More rounds = better quality, slower
    completion_threshold: 0.8,      // Lower = earlier termination
    entity_confidence_threshold: 0.7,  // Lower = more entities, less accurate
    use_llm_completion_check: true,    // true = LLM judges, false = heuristic
    entity_types: vec![
        "PERSON".to_string(),
        "ORGANIZATION".to_string(),
    ],
    temperature: 0.1,               // Lower = more consistent
    max_tokens: 1500,               // Higher = more detailed descriptions
};
```

## Advanced Features

### 1. Length-Based Entity Merging (LightRAG)

When the same entity appears in multiple rounds, we keep the description with more detail:

```rust
fn merge_entity_data(&self, existing: Vec<EntityData>, new: Vec<EntityData>)
    -> Vec<EntityData>
{
    for new_entity in new {
        if let Some(existing_entity) = existing.get(&new_entity.name) {
            // Keep the longer description (more information)
            if new_entity.description.len() > existing_entity.description.len() {
                existing.insert(new_entity.name, new_entity);
            }
        }
    }
}
```

### 2. Multi-Strategy JSON Parsing

LLMs often produce malformed JSON. We use 4-level fallback:

```rust
fn parse_extraction_response(&self, response: &str) -> Result<ExtractionOutput> {
    // Strategy 1: Direct JSON parsing
    if let Ok(output) = serde_json::from_str::<ExtractionOutput>(response) {
        return Ok(output);
    }

    // Strategy 2: Extract from markdown code blocks
    if let Some(json_str) = Self::extract_json_from_markdown(response) {
        if let Ok(output) = serde_json::from_str::<ExtractionOutput>(json_str) {
            return Ok(output);
        }
    }

    // Strategy 3: JSON repair using jsonfixer
    match self.repair_and_parse_json(response) {
        Ok(output) => return Ok(output),
        Err(e) => tracing::warn!("JSON repair failed: {}", e),
    }

    // Strategy 4: Find JSON anywhere in text
    if let Some(json_str) = Self::find_json_in_text(response) {
        if let Ok(output) = serde_json::from_str::<ExtractionOutput>(json_str) {
            return Ok(output);
        }
    }

    // Fallback: Empty extraction
    Ok(ExtractionOutput { entities: vec![], relationships: vec![] })
}
```

### 3. LLM-Based Completion Check

Instead of heuristics, we ask the LLM if extraction is complete:

```rust
pub async fn check_completion(&self, chunk: &TextChunk,
    entities: &[EntityData], relationships: &[RelationshipData])
    -> Result<bool>
{
    let prompt = self.prompt_builder.build_completion_prompt(
        &chunk.content, entities, relationships
    );

    let response = self.call_llm_with_retry(&prompt).await?;

    // Parse YES/NO response
    let is_complete = response.trim().to_uppercase().starts_with("YES");
    Ok(is_complete)
}
```

## Logging and Monitoring

The extractor provides detailed logging with emoji indicators:

```
🔍 Starting REAL LLM gleaning extraction for chunk: chunk_001 (1234 chars)
📝 Round 1: Initial LLM extraction...
   ✅ Extracted 15 entities, 23 relationships (18.5s)
📝 Round 2: Gleaning continuation...
   ✅ Extracted 8 additional entities, 12 relationships (16.2s)
📝 Round 3: Gleaning continuation...
   ✅ LLM determined extraction is COMPLETE after 3 rounds
🎉 REAL LLM gleaning complete: 23 entities, 35 relationships (52.1s total)
```

Enable debug logging for more detail:

```bash
RUST_LOG=graphrag_core=debug cargo run
```

## Troubleshooting

### Issue: Processing takes too long

**Solutions**:
1. Reduce `max_gleaning_rounds` (e.g., 2 instead of 4)
2. Increase `chunk_size` (fewer chunks to process)
3. Use faster LLM model (e.g., `llama3.1:3b` instead of `8b`)
4. Set `use_llm_completion_check = false` (use heuristic instead)

### Issue: Low quality entities

**Solutions**:
1. Increase `max_gleaning_rounds` (more refinement)
2. Lower `entity_confidence_threshold` (more permissive)
3. Use better LLM model (e.g., `llama3.1:70b`)
4. Adjust `temperature` (0.0 = deterministic, 0.5 = creative)

### Issue: Ollama connection errors

**Solutions**:
1. Verify Ollama is running: `ollama list`
2. Check model is installed: `ollama pull llama3.1:8b`
3. Increase timeout: `timeout_seconds = 300`
4. Check network: `curl http://localhost:11434/api/tags`

### Issue: JSON parsing failures

The multi-strategy parser handles most cases, but if you see warnings:

```
JSON repair failed: invalid JSON structure
```

**Solutions**:
1. Increase `max_tokens` (LLM might be truncating)
2. Lower `temperature` (more consistent formatting)
3. Update jsonfixer: `cargo update -p jsonfixer`

## Testing

### Unit Tests

```bash
# Test individual components
cargo test --lib entity::llm_extractor
cargo test --lib entity::gleaning_extractor
cargo test --lib entity::prompts
```

### Integration Tests

```bash
# Test with real Ollama (requires running Ollama server)
RUST_LOG=debug cargo test --test integration_tests
```

### Example Scripts

```bash
# Test with Tom Sawyer document
./test_tom_sawyer.sh

# Expected output:
# Processing time: 2-3 hours
# Entities: 200-300 (characters, locations, events)
# Relationships: 400-600
```

## Performance Tuning

### For Speed (Lower Quality)

```toml
[entities]
max_gleaning_rounds = 2          # Fewer rounds
use_llm_completion_check = false # Skip LLM check
min_confidence = 0.5             # More permissive

[pipeline.text_extraction]
chunk_size = 1000                # Larger chunks, fewer API calls
```

### For Quality (Slower)

```toml
[entities]
max_gleaning_rounds = 6          # More rounds
use_llm_completion_check = true  # LLM judges completion
min_confidence = 0.8             # Higher threshold

[pipeline.text_extraction]
chunk_size = 300                 # Smaller chunks, more detail

[ollama]
chat_model = "llama3.1:70b"      # Better model
```

### Balanced (Default)

```toml
[entities]
max_gleaning_rounds = 4
use_llm_completion_check = true
min_confidence = 0.7

[pipeline.text_extraction]
chunk_size = 500

[ollama]
chat_model = "llama3.1:8b"
```

## References

- **Microsoft GraphRAG**: [GitHub](https://github.com/microsoft/graphrag)
  - Multi-round gleaning with 4 default rounds
  - Structured prompts with JSON output
  - ~4 hours for 200k tokens

- **LightRAG**: [Paper](https://arxiv.org/abs/2410.05779)
  - Length-based entity merging
  - Dual-level retrieval system

- **RAGFlow**: [GitHub](https://github.com/infiniflow/ragflow)
  - Logit bias for YES/NO completion
  - Graph-based retrieval

## See Also

- [README.md](README.md) - Main documentation
- [EMBEDDINGS_CONFIG.md](EMBEDDINGS_CONFIG.md) - Embedding configuration
- [../REAL_LLM_GLEANING_IMPLEMENTATION.md](../REAL_LLM_GLEANING_IMPLEMENTATION.md) - Implementation details
- [../HOW_IT_WORKS.md](../HOW_IT_WORKS.md) - Pipeline overview
