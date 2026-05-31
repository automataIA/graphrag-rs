# JSON5 Configuration System for GraphRAG

**Type-safe, validated configuration for GraphRAG pipelines.**

## Table of Contents

- [Why JSON5?](#why-json5)
- [Quick Start](#quick-start)
- [VSCode Setup](#vscode-setup)
- [Creating Configurations](#creating-configurations)
- [Validation](#validation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Why JSON5?

### The Critical Advantage: **Comments!**

Unlike standard JSON, JSON5 allows comments to document your configuration choices:

**❌ Standard JSON:**
```json
{
  "temperature": 0.1,
  "chunk_size": 800
}
```
**No comments allowed** - JSON syntax forbids comments entirely!

**✅ JSON5:**
```json5
{
  // Low temperature for consistent character analysis
  "temperature": 0.1,  // 0.05-0.3 optimal for narrative (IBM 2024)

  // Larger chunks capture complete narrative scenes
  "chunk_size": 800,  // LlamaIndex research: 800-1024 for narratives
}
```
**Comments everywhere** - document choices, cite research, explain "why"!

### JSON5 Features

1. **Comments** (`//` and `/* */`) 
   - Document WHY you chose parameter values
   - Add research references inline
   - Explain domain-specific choices

2. **Trailing Commas** ✅
   ```json5
   {
     "a": 1,
     "b": 2,  // ← This trailing comma is valid!
   }
   ```

3. **Flexible Syntax** 
   - More forgiving than strict JSON
   - Numbers: `+123`, `0xFF`, `Infinity`, `NaN`
   - Multi-line strings
   - Unquoted keys (we use quoted for consistency)

4. **Schema Validation** 
   - Real-time autocomplete in VSCode
   - Catch errors before runtime
   - Range and enum validation
   - Hover documentation

### JSON5 vs JSON

| Feature | JSON | JSON5 |
|---------|------|-------|
| **Comments** | ❌ | ✅ `//` or `/* */` |
| **Trailing commas** | ❌ | ✅ |
| **Unquoted keys** | ❌ | ✅ |
| **Numbers** | Limited | `+123`, `0xFF`, `Infinity` |
| **Strings** | Single line | Multi-line |
| **Schema support** | ✅ | ✅ |
| **Autocomplete** | ✅ | ✅ |
| **Validation** | ✅ | ✅ |

**Winner**: JSON5 = Best of JSON (tooling) + Comments + Flexible syntax

---

## Quick Start

### 1. Use an Existing Template

GraphRAG provides 13 pre-configured templates for different use cases:

```bash
# List available templates
ls config/templates/*.graphrag.json5

# Copy a template
cp config/templates/narrative_fiction.graphrag.json5 my_config.graphrag.json5

# Edit with autocomplete in VSCode!
code my_config.graphrag.json5
```

**Available templates:**
- `semantic_pipeline.graphrag.json5` - LLM-based semantic analysis
- `algorithmic_pipeline.graphrag.json5` - Fast pattern-based extraction
- `hybrid_pipeline.graphrag.json5` - Combined semantic + algorithmic
- `narrative_fiction.graphrag.json5` - Novels, stories, literature
- `technical_documentation.graphrag.json5` - API docs, manuals
- `academic_research.graphrag.json5` - Research papers, theses
- `legal_documents.graphrag.json5` - Contracts, regulations
- `web_blog_content.graphrag.json5` - Blog posts, articles
- And more!

### 2. Template Structure

```json5
{
  // ==========================================================================
  // GraphRAG Configuration - YOUR PROJECT NAME
  // ==========================================================================
  // VSCode: This file has autocomplete! Press Ctrl+Space for suggestions.
  // ==========================================================================

  "$schema": "../schema/graphrag-config.schema.json",

  "mode": {
    "approach": "semantic"  // Options: semantic | algorithmic | hybrid
  },

  "general": {
    "input_document_path": "path/to/your/document.txt",
    "output_dir": "./output/analysis",
    "log_level": "info",
    "max_threads": 4
  },

  "pipeline": {
    "workflows": ["extract_text", "extract_entities", "build_graph"],
    "text_extraction": {
      "chunk_size": 800,
      "chunk_overlap": 300
    },
    "entity_extraction": {
      "model_name": "llama3.1:8b",
      "temperature": 0.1,
      "entity_types": ["PERSON", "LOCATION", "EVENT"]
    }
  },

  "ollama": {
    "enabled": true,
    "chat_model": "llama3.1:8b",
    "embedding_model": "nomic-embed-text"
  }
}
```

### 3. Load in Rust (Coming Soon)

```rust
use graphrag_core::config::json5_loader::load_json5_config;

fn main() -> Result<()> {
    let config: GraphRAGConfig = load_json5_config("my_config.graphrag.json5")?;
    println!("Approach: {:?}", config.mode.approach);
    Ok(())
}
```

---

## VSCode Setup

### Automatic Setup (Already Done!)

The repository includes:
- `.vscode/settings.json` - Schema mapping for `*.graphrag.json5` files
- `.vscode/graphrag.code-snippets` - Quick templates

### What You Get

**1. Autocomplete** (Press Ctrl+Space)
```json5
{
  "mode": {
    "approach": ""  // ← Press Ctrl+Space here: semantic | algorithmic | hybrid
  }
}
```

**2. Real-time Validation**
```json5
{
  "general": {
    "max_threads": 999  // ❌ Red underline: Maximum is 128
  }
}
```

**3. Hover Documentation**
- Hover over any field
- See description, valid range, default value
- Research-based recommendations

**4. Error Prevention**
```json5
{
  "mode": {
    "approach": "invalid"  // ❌ Error: must be semantic/algorithmic/hybrid
  },
  "text_processing": {
    "chunk_size": 99999  // ❌ Error: maximum is 4096
  }
}
```

### Manual Setup (If Needed)

If autocomplete doesn't work automatically:

1. Open VSCode Settings (Ctrl+,)
2. Search for "json.schemas"
3. Verify this mapping exists:
   ```json
   "json.schemas": [{
     "fileMatch": ["*.graphrag.json5", "*.graphrag.json"],
     "url": "./config/schema/graphrag-config.schema.json"
   }]
   ```
4. Reload VSCode: Ctrl+Shift+P → "Reload Window"

---

## Creating Configurations

### Option 1: Copy a Template

Start with a template matching your use case:

```bash
# For semantic pipeline (LLM-based, high quality)
cp config/templates/semantic_pipeline.graphrag.json5 my_config.graphrag.json5

# For narrative fiction (novels, stories)
cp config/templates/narrative_fiction.graphrag.json5 my_novel_config.graphrag.json5

# For technical docs (API documentation, manuals)
cp config/templates/technical_documentation.graphrag.json5 my_api_docs.graphrag.json5

# For hybrid approach (balanced quality and speed)
cp config/templates/hybrid_pipeline.graphrag.json5 my_hybrid_config.graphrag.json5
```

Then customize:
1. Update `input_document_path`
2. Adjust `output_dir`
3. Customize `entity_types` for your domain
4. Tune parameters based on your needs

### Option 2: Build from Scratch

In VSCode:
1. Create `my_config.graphrag.json5`
2. Add schema reference:
   ```json5
   {
     "$schema": "../config/schema/graphrag-config.schema.json"
   }
   ```
3. Press Ctrl+Space and follow autocomplete suggestions!

The schema will guide you through all required and optional fields.

### Option 3: Use Code Snippets

In VSCode:
1. Create new file: `my_config.graphrag.json5`
2. Type `graphrag-semantic` and press Tab
3. Full template inserted!

Available snippets:
- `graphrag-semantic` - Semantic pipeline template
- `graphrag-algorithmic` - Algorithmic pipeline template
- `graphrag-hybrid` - Hybrid pipeline template

---

## ✅ Validation

### Real-time (VSCode)

Errors show immediately as you type:

```json5
{
  "mode": {
    "approach": "semantic"
  },
  "general": {
    "max_threads": 999,  // ❌ Error: Maximum is 128
    "log_level": "invalid"  // ❌ Error: Must be trace/debug/info/warn/error
  },
  "ollama": {
    "temperature": 5.0  // ❌ Error: Maximum is 2.0
  }
}
```

### CLI Validation

Validate before running your application:

```bash
# Validate single config
uv run --with jsonschema --with json5 python scripts/validate_json5_configs.py \
  --config my_config.graphrag.json5

# Validate all configs in directory
uv run --with jsonschema --with json5 python scripts/validate_json5_configs.py \
  --dir config/templates

# Custom schema
uv run --with jsonschema --with json5 python scripts/validate_json5_configs.py \
  --config my_config.json5 \
  --schema path/to/schema.json
```

**Output:**
```
Validating 1 configuration file(s)...

✅ my_config.graphrag.json5

============================================================
Validation Complete: 1/1 valid
All configurations are valid!
```

**Error output example:**
```
❌ my_config.graphrag.json5
  • Path: general → max_threads
    Error: 999 is greater than the maximum of 128
    Allowed range: 1-128

  • Path: mode → approach
    Error: 'invalid' is not one of ['semantic', 'algorithmic', 'hybrid']
    Allowed values: "semantic", "algorithmic", "hybrid"
```

### Programmatic Validation (Rust - Coming Soon)

```rust
use graphrag_core::config::schema_validator::validate_config_file;

fn main() -> Result<()> {
    validate_config_file(
        "my_config.graphrag.json5",
        "config/schema/graphrag-config.schema.json"
    )?;

    println!("✅ Configuration is valid!");
    Ok(())
}
```

---

## Examples

### Example 1: Minimal Semantic Config

```json5
{
  "$schema": "../config/schema/graphrag-config.schema.json",

  "mode": { "approach": "semantic" },

  "general": {
    "input_document_path": "data/document.pdf",
    "output_dir": "./output"
  },

  "pipeline": {
    "workflows": ["extract_text", "extract_entities", "build_graph"]
  },

  "ollama": {
    "enabled": true,
    "host": "http://localhost",
    "port": 11434,
    "chat_model": "llama3.1:8b"
  }
}
```

### Example 2: Narrative Fiction

```json5
{
  "$schema": "../config/schema/graphrag-config.schema.json",

  "mode": { "approach": "semantic" },

  "general": {
    "input_document_path": "novels/tom_sawyer.txt",
    "output_dir": "./output/narrative",
    "log_level": "info"
  },

  // Narrative-optimized chunking (LlamaIndex 2024 research)
  "pipeline": {
    "text_extraction": {
      "chunk_size": 800,      // Captures complete scenes
      "chunk_overlap": 300,   // 37.5% overlap for character continuity
      "min_chunk_size": 200
    },
    "entity_extraction": {
      "model_name": "llama3.1:8b",
      "temperature": 0.1,     // Low for consistent character analysis
      "entity_types": [
        "PERSON",              // Characters
        "CHARACTER_TRAIT",     // Personality, appearance
        "LOCATION",            // Settings, places
        "EMOTION",             // Emotional states
        "THEME",               // Literary themes
        "RELATIONSHIP",        // Character relationships
        "EVENT"                // Plot events
      ],
      "confidence_threshold": 0.6  // Captures literary nuances
    }
  },

  "ollama": {
    "enabled": true,
    "chat_model": "llama3.1:8b",
    "generation": {
      "temperature": 0.3,    // Balanced for narrative analysis
      "max_tokens": 1500
    }
  }
}
```

### Example 3: Technical Documentation

```json5
{
  "$schema": "../config/schema/graphrag-config.schema.json",

  "mode": { "approach": "semantic" },

  "general": {
    "input_document_path": "docs/api_reference.md",
    "output_dir": "./output/tech_docs"
  },

  // Technical precision (Databricks 2024 research)
  "pipeline": {
    "text_extraction": {
      "chunk_size": 512,      // Smaller chunks for precision
      "chunk_overlap": 100,   // 20% minimal overlap
      "min_chunk_size": 128
    },
    "entity_extraction": {
      "model_name": "llama3.1:8b",
      "temperature": 0.05,    // Maximum precision
      "entity_types": [
        "API_ENDPOINT",        // REST endpoints
        "FUNCTION",            // Functions, methods
        "PARAMETER",           // Function parameters
        "ERROR_CODE",          // Error codes, exceptions
        "LIBRARY",             // External libraries
        "VERSION",             // Version numbers
        "DATA_TYPE"            // Data types
      ],
      "confidence_threshold": 0.8  // High accuracy for technical content
    }
  },

  "ollama": {
    "enabled": true,
    "generation": {
      "temperature": 0.1,    // Very low for technical precision
      "max_tokens": 1200
    }
  }
}
```

### Example 4: Hybrid Pipeline

```json5
{
  "$schema": "../config/schema/graphrag-config.schema.json",

  // Hybrid: Combines semantic (LLM) + algorithmic (patterns)
  "mode": { "approach": "hybrid" },

  "general": {
    "input_document_path": "data/mixed_content",
    "output_dir": "./output/hybrid"
  },

  "pipeline": {
    "workflows": ["extract_text", "extract_entities", "build_graph"],
    "text_extraction": {
      "chunk_size": 600,
      "chunk_overlap": 150
    },
    "entity_extraction": {
      "model_name": "llama3.1:8b",
      "temperature": 0.15,
      "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"],
      "confidence_threshold": 0.6
    }
  },

  "ollama": {
    "enabled": true,
    "chat_model": "llama3.1:8b",
    "fallback_to_hash": true  // Graceful degradation if LLM fails
  },

  "performance": {
    "batch_processing": true,
    "batch_size": 32,
    "worker_threads": 6,
    "cache_embeddings": true
  }
}
```

---

## Troubleshooting

### Autocomplete Not Working

**Problem**: No suggestions when typing

**Solutions**:
1. ✅ Verify `$schema` field points to correct path
2. ✅ Check file extension is `.graphrag.json5` or `.json5`
3. ✅ Reload VSCode: Ctrl+Shift+P → "Reload Window"
4. ✅ Check `.vscode/settings.json` has schema mapping
5. ✅ Ensure you're in VSCode (not other editors)

### Validation Errors

**Problem**: Red underlines everywhere

**Common Fixes**:

| Error | Fix |
|-------|-----|
| `Missing required field` | Add required fields: `mode`, `general` |
| `Invalid enum value` | Use Ctrl+Space to see valid options |
| `Number out of range` | Hover to see valid range (e.g., 0.0-1.0) |
| `Wrong type` | Ensure strings have quotes, numbers don't |
| `Additional properties not allowed` | Remove unsupported fields |

**Example fixes:**
```json5
// ❌ Wrong
{
  "mode": { "approach": "semantic" },
  "unsupported_field": "value"  // Error: additional property
}

// ✅ Correct
{
  "$schema": "../config/schema/graphrag-config.schema.json",
  "mode": { "approach": "semantic" },
  "general": {
    "input_document_path": "data/input.txt",
    "output_dir": "./output"
  }
}
```

### Schema Path Issues

**Problem**: VSCode can't find schema

**Solution**: Use relative path from config file location:
```json5
{
  // If config is in project root:
  "$schema": "./config/schema/graphrag-config.schema.json",

  // If config is in config/:
  "$schema": "./schema/graphrag-config.schema.json",

  // If config is in config/templates/:
  "$schema": "../schema/graphrag-config.schema.json"
}
```

### "Property keys must be doublequoted" Warning

**Problem**: VSCode shows warnings on unquoted keys (e.g., `mode: {...}`)

**Why This Happens**:
- VSCode treats `.json5` files as JSONC (JSON with Comments)
- JSONC requires quoted keys: `"mode": {...}`
- JSON5 allows unquoted keys: `mode: {...}` ✅ Valid!
- This is a **false positive** - your JSON5 syntax is correct

**Example Warning**:
```json5
{
  mode: {  // VSCode warning: "Property keys must be doublequoted"
    approach: "semantic"
  }
}
```

**Solutions**:

**Option 1: Ignore the Warnings (Recommended)**
- These are cosmetic warnings only
- Your JSON5 files are valid and will work correctly
- The warnings don't affect functionality

**Option 2: Install JSON5 Extension**
- Install "JSON5 syntax" extension from VSCode marketplace
- Provides true JSON5 language support
- Eliminates false positives

**Option 3: Use Quoted Keys**
```json5
{
  "mode": {  // ✅ No warning with quoted keys
    "approach": "semantic"
  }
}
```
**Trade-off**: Loses the readability advantage of unquoted keys

**Our Recommendation**: Ignore the warnings. They're false positives caused by VSCode's JSONC mode not fully supporting JSON5's unquoted key feature. Your configs are valid and will work correctly.

---

## Best Practices

### 1. Always Use `$schema` Reference

```json5
{
  // ✅ First line: enables autocomplete and validation
  "$schema": "../config/schema/graphrag-config.schema.json",

  // ... rest of config
}
```

This single line enables:
- ✅ Real-time autocomplete
- ✅ Instant error detection
- ✅ Hover documentation
- ✅ Type validation

### 2. Document with Comments

```json5
{
  "pipeline": {
    "text_extraction": {
      // Research-based: LlamaIndex 2024 study shows 800-1024 optimal
      // for narrative continuity and character relationship tracking.
      // See: https://www.llamaindex.ai/blog/evaluating-chunk-size
      "chunk_size": 800,

      // 37.5% overlap preserves scene boundaries and dialogue context.
      // Critical for maintaining character consistency across chunks.
      // Pinecone 2024: "Chunking Strategies for LLM Applications"
      "chunk_overlap": 300
    }
  }
}
```

### 3. Use Descriptive Filenames

```
✅ Good:
  - narrative_dickens_analysis.graphrag.json5
  - api_docs_v2_production.graphrag.json5
  - legal_contracts_compliance.graphrag.json5

❌ Bad:
  - config.json5
  - test.json5
  - c1.json5
```

### 4. Validate Before Running

```bash
# Always validate before deploying
uv run --with jsonschema --with json5 python scripts/validate_json5_configs.py \
  --config production.graphrag.json5
```

### 5. Version Control Your Configs

```bash
git add my_project.graphrag.json5
git commit -m "feat: add GraphRAG config for project XYZ"
```

Keep configs in version control to track changes over time.

### 6. Document Custom Parameters

```json5
{
  "entity_extraction": {
    // Custom threshold chosen after A/B testing:
    // - 0.7: 85% precision, 72% recall
    // - 0.6: 78% precision, 84% recall ← chosen
    // - 0.5: 65% precision, 91% recall
    // Decision: Prioritize recall for this corpus (historical texts)
    "confidence_threshold": 0.6
  }
}
```

---

## Advantages Summary

### Why JSON5 for GraphRAG?

✅ **Comments** - Document configuration choices inline
✅ **Autocomplete** - VSCode suggests all available fields
✅ **Validation** - Catch errors before runtime
✅ **Research Documentation** - Cite sources directly in config
✅ **Trailing Commas** - More forgiving, easier editing
✅ **Schema Support** - Full IDE integration
✅ **Better DX** - Faster development, fewer errors
✅ **Self-Documenting** - Configuration explains itself

### Available Templates (All Validated ✅)

All 13 templates pass JSON Schema validation:
- ✅ `semantic_pipeline.graphrag.json5` - General semantic
- ✅ `algorithmic_pipeline.graphrag.json5` - General algorithmic
- ✅ `hybrid_pipeline.graphrag.json5` - General hybrid
- ✅ `narrative_fiction.graphrag.json5` - Novels, stories
- ✅ `technical_documentation.graphrag.json5` - API docs, manuals
- ✅ `academic_research.graphrag.json5` - Research papers
- ✅ `legal_documents.graphrag.json5` - Contracts, regulations
- ✅ `web_blog_content.graphrag.json5` - Blog posts, articles
- ✅ `dynamic_universal.graphrag.json5` - Adaptive configuration
- ✅ `enrichment_example.graphrag.json5` - Text enrichment
- ✅ `semantic.graphrag.json5` - Basic semantic
- ✅ `algorithmic.graphrag.json5` - Basic algorithmic
- ✅ `hybrid.graphrag.json5` - Basic hybrid

**Status**: 13/13 pass JSON Schema validation

---

## Additional Resources

- **JSON Schema**: `config/schema/graphrag-config.schema.json`
- **Template Examples**: `config/templates/*.graphrag.json5`
- **Validation Scripts**: `scripts/README.md`
- **VSCode Settings**: `.vscode/settings.json`
- **Code Snippets**: `.vscode/graphrag.code-snippets`

---

## Common Questions

**Q: What file extension should I use?**
A: Use `.graphrag.json5` for automatic schema mapping, or `.json5` for general JSON5 files.

**Q: Can I use regular JSON instead of JSON5?**
A: Yes! JSON5 is a superset of JSON. Any valid JSON is valid JSON5. But you'll lose the ability to add comments.

**Q: How do I know which template to use?**
A: Match your content type:
- Novels/stories → `narrative_fiction`
- API docs → `technical_documentation`
- Research papers → `academic_research`
- Legal docs → `legal_documents`
- Mixed content → `hybrid_pipeline`

**Q: What if I need to customize entity types?**
A: Edit the `entity_types` array in your config:
```json5
"entity_types": [
  "CUSTOM_TYPE_1",
  "CUSTOM_TYPE_2",
  "PERSON",
  "LOCATION"
]
```

**Q: How do I tune for my specific domain?**
A: Start with the closest template, then adjust:
1. `chunk_size` - larger for better context, smaller for precision
2. `confidence_threshold` - higher for precision, lower for recall
3. `entity_types` - add domain-specific types
4. `temperature` - lower for consistency, higher for variety

---

**Ready to start?** 

```bash
cp config/templates/semantic_pipeline.graphrag.json5 my_config.graphrag.json5
code my_config.graphrag.json5
```

Press Ctrl+Space and let autocomplete guide you!
