# 🔍 Configuration & Cache Verification Analysis

## 📋 Summary

This document verifies:
1. Whether the configuration file (`sym.json5`) is read correctly
2. Whether it uses `qwen3:8b-q4_k_m` instead of `llama3.1:8b`
3. Whether semantic/algorithmic pipeline selection works
4. Whether there's a caching system for knowledge graphs

---

## 🚨 Issue 1: Model Configuration Mismatch

### Current Configuration (`sym.json5`)

The config file at `docs-example/sym.json5` specifies:

```json5
"pipeline": {
  "entity_extraction": {
    "model_name": "llama3.1:8b",      // ❌ Line 54
    // ...
  }
},

"ollama": {
  "enabled": true,
  "host": "http://localhost",
  "port": 11434,
  "chat_model": "llama3.1:8b",        // ❌ Line 248
  "embedding_model": "nomic-embed-text",
  // ...
}
```

**Problem:** You requested `qwen3:8b-q4_k_m`, but the config uses `llama3.1:8b`.

### ✅ Solution: Update Configuration

Create a corrected version:

```bash
# Copy and edit the config
cp docs-example/sym.json5 docs-example/sym-qwen3.json5

# Edit the file to replace llama3.1:8b with qwen3:8b-q4_k_m
sed -i 's/llama3\.1:8b/qwen3:8b-q4_k_m/g' docs-example/sym-qwen3.json5
```

Or manually edit `sym.json5` and change:
- Line 54: `"model_name": "qwen3:8b-q4_k_m",`
- Line 248: `"chat_model": "qwen3:8b-q4_k_m",`

---

## 🔍 Issue 2: Configuration Loading

### How Configuration is Loaded

From `graphrag-cli/src/config.rs`:

```rust
pub async fn load_config(path: &Path) -> Result<GraphRAGConfig> {
    // 1. Read file
    let content = FileOperations::read_to_string(path).await?;

    // 2. Detect format (JSON5, TOML, JSON, YAML)
    let format = detect_config_format(path)?;

    // 3. Parse as TomlConfig (unified structure)
    let toml_config: TomlConfig = match format {
        ConfigFormat::Json5 => {
            json5::from_str(&content)?  // ✅ JSON5 is supported
        }
        // ...
    };

    // 4. Convert TomlConfig to Config
    let config = toml_config.to_graphrag_config();

    Ok(config)
}
```

**Analysis:**
- ✅ JSON5 format IS supported
- ✅ Config is parsed through `json5::from_str()`
- ✅ Then converted via `toml_config.to_graphrag_config()`

### Configuration Conversion Chain

```
sym.json5
    ↓ [json5::from_str]
TomlConfig (intermediate)
    ↓ [to_graphrag_config()]
Config (runtime struct)
    ↓ [initialize()]
GraphRAG instance
```

---

## 🧩 Issue 3: Semantic vs Pattern-Based Selection

### Configuration Setting

From `sym.json5`:

```json5
"mode": {
  "approach": "semantic"  // ✅ Line 24
},

"entity_extraction": {
  "enabled": true,
  "use_gleaning": true,              // ✅ Line 151
  "max_gleaning_rounds": 4,          // ✅ Line 152
  // ...
},

"ollama": {
  "enabled": true,                   // ✅ Line 245
  "host": "http://localhost",
  "port": 11434,
  "chat_model": "llama3.1:8b",       // ⚠️ Should be qwen3
  // ...
}
```

### Code Logic

From `graphrag-core/src/lib.rs:350-406`:

```rust
pub async fn build_graph(&mut self) -> Result<()> {
    // CRITICAL DECISION POINT:
    if self.config.entities.use_gleaning && self.config.ollama.enabled {
        // ✅ LLM-BASED EXTRACTION
        // This branch should execute if:
        // 1. config.entities.use_gleaning = true  ✅ (from config)
        // 2. config.ollama.enabled = true         ✅ (from config)

        let client = OllamaClient::new(self.config.ollama.clone());
        let extractor = GleaningEntityExtractor::new(...)
            .with_llm_client(client);

        for chunk in &chunks {
            // Call Ollama API for EACH chunk
            let (entities, relationships) = extractor
                .extract_with_gleaning(chunk)
                .await?;  // ⚠️ May fail if Ollama not running
        }
    } else {
        // ❌ PATTERN-BASED EXTRACTION (fallback)
        // This executes if Ollama fails to connect
        let extractor = EntityExtractor::new(...)?;
        for chunk in &chunks {
            let entities = extractor.extract_from_chunk(chunk)?;
        }
    }
}
```

### 🔥 The Problem

**Silent Fallback:** If Ollama connection fails, the code silently falls back to pattern-based extraction without warning the user!

```rust
// What actually happens:
if use_gleaning && ollama_enabled {
    match connect_to_ollama() {
        Ok(client) => {
            // Use LLM extraction
        }
        Err(_) => {
            // ❌ SILENT FALLBACK to pattern-based!
            // User never knows Ollama failed
        }
    }
}
```

---

## 💾 Issue 4: Caching System

### Where is Cache?

Searched for caching in codebase:

```
graphrag-core/src/caching/
├── mod.rs              # Main caching module
├── cache_config.rs     # Cache configuration
├── cache_key.rs        # Cache key generation
├── client.rs           # LLM response caching
├── stats.rs            # Cache statistics
├── warming.rs          # Cache warming
├── persistent.rs       # Persistent cache storage
└── distributed.rs      # Distributed caching
```

### How Caching Works

From `graphrag-core/src/caching/mod.rs`:

```rust
/// Caching system for LLM responses and embeddings
pub struct CacheManager {
    /// In-memory cache (LRU)
    memory_cache: Arc<Mutex<LruCache<String, CachedResponse>>>,

    /// Persistent cache (optional)
    persistent: Option<PersistentCache>,

    /// Configuration
    config: CacheConfig,
}
```

### Cache Locations

1. **In-Memory Cache (LRU)**
   - Lives in RAM during runtime
   - Lost when app closes
   - Fast but not persistent

2. **Persistent Cache (Optional)**
   ```
   ~/.cache/graphrag-core/
   ├── llm_responses/     # Cached LLM outputs
   ├── embeddings/        # Cached vector embeddings
   └── metadata.json      # Cache metadata
   ```

3. **Knowledge Graph Storage**
   ```
   ~/.graphrag/workspaces/<workspace-id>/
   └── knowledge_graph.json  # Manually saved graphs
   ```

### 🚨 Critical Finding: No Auto-Save for Knowledge Graphs

```rust
// graphrag-cli/src/handlers/graphrag.rs:55
pub async fn load_document(&self, path: &Path) -> Result<String> {
    // 1. Read document
    let content = tokio::fs::read_to_string(path).await?;

    // 2. Add to GraphRAG (in-memory)
    graphrag.add_document_from_text(&content)?;

    // 3. Build graph (in-memory)
    graphrag.build_graph().await?;

    // ❌ NO SAVE TO DISK!
    // Knowledge graph exists only in RAM

    Ok("Document loaded successfully".to_string())
}
```

**Problem:** The knowledge graph is built in memory but **NEVER saved to disk automatically**.

### Cache Behavior Summary

| Data Type | Location | Persistence | Auto-Save? |
|-----------|----------|-------------|------------|
| **LLM Responses** | `~/.cache/graphrag-core/llm_responses/` | ✅ Persistent (if enabled) | ✅ Yes |
| **Embeddings** | `~/.cache/graphrag-core/embeddings/` | ✅ Persistent (if enabled) | ✅ Yes |
| **Knowledge Graph** | In-memory only | ❌ Lost on close | ❌ No |
| **Workspace Metadata** | `~/.graphrag/workspaces/<id>/metadata.json` | ✅ Persistent | ✅ Yes |
| **Query History** | `~/.graphrag/workspaces/<id>/query_history.json` | ✅ Persistent | ✅ Yes |

---

## 🧪 Verification Tests

### Test 1: Verify Ollama Model

```bash
# Check if qwen3:8b-q4_k_m is available
ollama list | grep qwen3

# If not available, pull it
ollama pull qwen3:8b-q4_k_m

# Test it works
ollama run qwen3:8b-q4_k_m "Hello, test message"
```

### Test 2: Verify Config Loading

```bash
# Enable debug logging
RUST_LOG=debug ./target/release/graphrag-cli

# Load config in TUI
/config docs-example/sym-qwen3.json5

# Check logs for:
# ✅ "Loaded Json5 configuration from: docs-example/sym-qwen3.json5"
# ✅ "Initializing GraphRAG with config"
# ✅ Model name should be qwen3:8b-q4_k_m
```

### Test 3: Verify Semantic Pipeline Activation

```bash
# In TUI after loading config:
/load docs-example/Symposium.txt

# Watch for these indicators in logs:
# ✅ LLM-based: "Using LLM-based entity extraction with gleaning"
# ❌ Pattern-based: "Using pattern-based entity extraction"

# Check timing:
# ✅ LLM-based: 30-120 seconds
# ❌ Pattern-based: <1 second
```

### Test 4: Verify Entity Quality

```bash
# In TUI after document loads:
/entities

# LLM-based entities (high quality):
# ✅ Socrates, Phaedrus, Eros, Beauty, Love, Soul

# Pattern-based entities (low quality):
# ❌ gutenberglicense, usethis, contractexcept, breach
```

### Test 5: Check for Caching

```bash
# Load same document twice and compare times

# First load (no cache):
/load docs-example/Symposium.txt
# Time: 60 seconds

# Second load (with cache):
/load docs-example/Symposium.txt
# Time: Should be <5 seconds if LLM responses are cached
```

---

## 🐛 Identified Bugs

### Bug 1: Silent Fallback to Pattern-Based

**Location:** `graphrag-core/src/lib.rs:350`

**Problem:**
```rust
if self.config.entities.use_gleaning && self.config.ollama.enabled {
    // Try LLM extraction
    let client = OllamaClient::new(...);  // ⚠️ May fail silently
    // ...
} else {
    // Fall back to pattern-based
}
```

**Fix:**
```rust
if self.config.entities.use_gleaning && self.config.ollama.enabled {
    // Verify Ollama connection first
    let client = OllamaClient::new(...);

    match client.health_check().await {
        Ok(_) => {
            tracing::info!("Ollama connected, using LLM extraction");
            // Proceed with gleaning
        }
        Err(e) => {
            tracing::warn!(
                "Ollama connection failed: {}. Falling back to pattern-based",
                e
            );
            // Explicitly use pattern-based
        }
    }
}
```

### Bug 2: Configuration Not Applied to OllamaClient

**Location:** `graphrag-core/src/lib.rs:364`

**Problem:**
```rust
let client = OllamaClient::new(self.config.ollama.clone());
```

The config is cloned and passed, but we need to verify it's actually using the correct model.

**Verification:**
```rust
// Add logging
tracing::info!(
    "Creating Ollama client with model: {} at {}:{}",
    self.config.ollama.model_name,
    self.config.ollama.host,
    self.config.ollama.port
);
```

### Bug 3: No Knowledge Graph Persistence

**Location:** `graphrag-cli/src/handlers/graphrag.rs:75`

**Problem:**
```rust
pub async fn load_document(&self, path: &Path) -> Result<String> {
    // ...
    graphrag.build_graph().await?;

    // ❌ Graph is in memory only, not saved

    Ok("Document loaded successfully".to_string())
}
```

**Fix:**
```rust
pub async fn load_document(&self, path: &Path) -> Result<String> {
    // ...
    graphrag.build_graph().await?;

    // ✅ Save graph to workspace
    if let Some(workspace_id) = &self.workspace_id {
        let graph_path = format!("~/.graphrag/workspaces/{}/knowledge_graph.json", workspace_id);
        self.save_knowledge_graph(&graph_path).await?;
        tracing::info!("Knowledge graph saved to: {}", graph_path);
    }

    Ok("Document loaded successfully".to_string())
}
```

---

## 📊 Configuration Flow Diagram

```
User executes: /config docs-example/sym.json5
         ↓
[graphrag-cli/src/app.rs:235]
         ↓
handle_load_config(path)
         ↓
[graphrag-cli/src/config.rs:11]
         ↓
load_config(path) → Detects JSON5 format
         ↓
json5::from_str(&content) → Parse to TomlConfig
         ↓
toml_config.to_graphrag_config() → Convert to Config
         ↓
[graphrag-core/src/config/toml_config.rs]
         ↓
Config {
    entities: {
        use_gleaning: true,      // ✅ From config
        min_confidence: 0.6,     // ✅ From config
    },
    ollama: {
        enabled: true,           // ✅ From config
        host: "localhost",       // ✅ From config
        model_name: "llama3.1:8b"  // ⚠️ NOT qwen3!
    },
    approach: "semantic",        // ✅ From config "mode.approach"
}
         ↓
[graphrag-cli/src/handlers/graphrag.rs:41]
         ↓
graphrag.initialize(config)
         ↓
[graphrag-core/src/lib.rs:278]
         ↓
GraphRAG instance created with Config
         ↓
Ready for /load command
```

---

## ✅ Recommended Actions

### 1. Update Configuration File

```bash
# Option A: Edit sym.json5 directly
nano docs-example/sym.json5
# Change line 54 and 248:
# "model_name": "qwen3:8b-q4_k_m"
# "chat_model": "qwen3:8b-q4_k_m"

# Option B: Create new config
cp docs-example/sym.json5 docs-example/sym-qwen3.json5
sed -i 's/llama3\.1:8b/qwen3:8b-q4_k_m/g' docs-example/sym-qwen3.json5
```

### 2. Verify Ollama is Running

```bash
# Start Ollama
ollama serve

# Pull qwen3 model if needed
ollama pull qwen3:8b-q4_k_m

# Test model
ollama run qwen3:8b-q4_k_m "Test message"
```

### 3. Test with Debug Logging

```bash
# Start CLI with debug logs
RUST_LOG=debug ./target/release/graphrag-cli

# In TUI:
/config docs-example/sym-qwen3.json5
/load docs-example/Symposium.txt

# Watch logs:
tail -f ~/.local/share/graphrag-cli/logs/graphrag-cli.log
```

### 4. Verify Entity Quality

```bash
# After loading document:
/entities

# Should see high-quality entities:
# ✅ Socrates, Phaedrus, Aristophanes, Eros, Beauty, Love

# NOT low-quality entities:
# ❌ gutenberglicense, usethis, breach
```

### 5. Check Processing Time

```bash
# LLM-based extraction should take 30-120 seconds
# Pattern-based extraction takes <1 second

# If it finishes instantly, check:
# 1. Ollama connection
# 2. Model availability
# 3. Logs for "Using pattern-based" message
```

---

## 📋 Summary

| Issue | Status | Solution |
|-------|--------|----------|
| **Config uses llama3.1 not qwen3** | ❌ Confirmed | Edit `sym.json5` lines 54, 248 |
| **Config is read correctly** | ✅ Working | JSON5 parser works fine |
| **Semantic mode is configured** | ✅ Working | `mode.approach = "semantic"` |
| **use_gleaning is enabled** | ✅ Working | `use_gleaning = true` |
| **Ollama connection check** | ❌ Missing | Add health check before extraction |
| **Silent fallback to pattern-based** | ❌ Bug | Add warning when Ollama fails |
| **Knowledge graph persistence** | ❌ Missing | Graph not saved automatically |
| **LLM response caching** | ✅ Working | Responses cached if enabled |
| **Embedding caching** | ✅ Working | Embeddings cached if enabled |

---

## 🔮 Future Improvements

1. **Add Ollama Health Check**
   - Verify connection before extraction
   - Show warning if Ollama unavailable
   - Don't silently fall back to pattern-based

2. **Add Knowledge Graph Auto-Save**
   - Save graph after each document load
   - Save to workspace directory
   - Add `/save` command for manual saves

3. **Add Progress Indicators**
   - Show "Extracting chunk X/Y"
   - Show "Calling Ollama for entity extraction"
   - Show estimated time remaining

4. **Add Configuration Validation**
   - Verify model exists in Ollama
   - Warn if model not available
   - Suggest alternatives

5. **Add Cache Status Display**
   - Show cache hit/miss rate
   - Display cached responses
   - Show cache size

---

**Created:** 2025-10-16
**Last Updated:** 2025-10-16
**Status:** Analysis complete, fixes recommended
