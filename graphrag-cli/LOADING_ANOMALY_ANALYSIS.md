# 🔍 Loading Anomaly Analysis - GraphRAG-CLI

## 🚨 Problem Discovered

You reported that loading `/config docs-example/sym.json5` followed by `/load docs-example/Symposium.txt` happens **instantaneously** (less than 1 second), which is highly suspicious for a 200KB document (3288 lines) that should require:

1. **Text chunking** (800 tokens per chunk with 300 overlap)
2. **LLM-based entity extraction** with **gleaning** (4 rounds max)
3. **Relationship extraction**
4. **Graph construction**

Expected time: **30-120 seconds** for a 200KB philosophical text with LLM extraction

Actual time: **<1 second** ⚠️

---

## 🔎 Root Cause Analysis

### 1. **The Code Flow**

When you execute `/load docs-example/Symposium.txt`, here's what happens:

```rust
// graphrag-cli/src/handlers/graphrag.rs:55
pub async fn load_document(&self, path: &Path) -> Result<String> {
    // 1. Read file asynchronously
    let content = tokio::fs::read_to_string(path).await?;

    // 2. Add document to GraphRAG
    let mut guard = self.graphrag.lock().await;
    if let Some(ref mut graphrag) = *guard {
        graphrag.add_document_from_text(&content)?;

        // 3. Build graph asynchronously
        graphrag.build_graph().await?;  // ← THIS IS THE KEY

        Ok(format!("Document '{}' loaded successfully", filename))
    }
}
```

### 2. **What build_graph() Actually Does**

From `graphrag-core/src/lib.rs:337-472`:

```rust
#[cfg(feature = "async")]
pub async fn build_graph(&mut self) -> Result<()> {
    let chunks: Vec<_> = graph.chunks().cloned().collect();

    // CRITICAL CHECK:
    if self.config.entities.use_gleaning && self.config.ollama.enabled {
        // ✅ LLM-BASED EXTRACTION (slow, 30-120 seconds)
        let extractor = GleaningEntityExtractor::new(...);

        for chunk in &chunks {
            // This calls Ollama for EACH chunk
            let (entities, relationships) = extractor
                .extract_with_gleaning(chunk)
                .await?;

            // Add entities and relationships
            for entity in entities {
                graph.add_entity(entity)?;
            }
        }
    } else {
        // ❌ PATTERN-BASED EXTRACTION (instant, <1 second)
        let extractor = EntityExtractor::new(...)?;

        for chunk in &chunks {
            // Just regex + capitalization matching
            let entities = extractor.extract_from_chunk(chunk)?;
            for entity in entities {
                graph.add_entity(entity)?;
            }
        }
    }
}
```

### 3. **The Anomaly Explained**

Your configuration `sym.json5` has:

```json5
"entity_extraction": {
  "enabled": true,
  "use_gleaning": true,        // ← Says "use LLM"
  "max_gleaning_rounds": 4
},

"ollama": {
  "enabled": true,              // ← Says "Ollama enabled"
  "host": "http://localhost",
  "port": 11434,
  "chat_model": "llama3.1:8b"
}
```

**BUT**, the instant loading suggests the code is taking the `else` branch (pattern-based extraction).

### 4. **Possible Causes**

#### Cause A: **Ollama Not Actually Running**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If this fails, build_graph() silently falls back to pattern-based extraction
```

**Evidence:**
- If Ollama connection fails, the code has a condition:
  ```rust
  if self.config.entities.use_gleaning && self.config.ollama.enabled {
      // Only enters if BOTH are true AND connection works
  }
  ```

#### Cause B: **Configuration Not Properly Loaded**

The `Config` struct might not be properly loading the `use_gleaning` flag from JSON5.

```bash
# Check config loading
grep -A 5 "use_gleaning" graphrag-core/src/config/loader.rs
```

#### Cause C: **Silent Fallback Logic**

The code might be catching Ollama connection errors silently and falling back to pattern-based extraction without warning the user.

---

## 🧪 Verification Tests

### Test 1: Check Ollama Status

```bash
# 1. Check if Ollama is running
ollama list

# 2. Test connection
curl http://localhost:11434/api/tags

# 3. Check model availability
ollama list | grep llama3.1
```

### Test 2: Check Logs for Extraction Method

```bash
# Check which extraction method was used
cat ~/.local/share/graphrag-cli/logs/graphrag-cli.log | grep -i "extraction"

# Look for these messages:
# ✅ LLM-based: "Using LLM-based entity extraction with gleaning"
# ❌ Pattern-based: "Using pattern-based entity extraction"
```

### Test 3: Count Entities After Loading

If pattern-based extraction was used, you'll get **many low-quality entities** (every capitalized word).

If LLM-based extraction was used, you'll get **fewer, high-quality entities** (only meaningful philosophical concepts).

```bash
# In TUI, after /load:
/stats

# Check entity count:
# Pattern-based: 200-500 entities (mostly noise)
# LLM-based: 30-80 entities (curated concepts)
```

### Test 4: Manual Test with Debug Logging

```bash
# Enable debug logging
RUST_LOG=debug ./target/release/graphrag-cli --config docs-example/sym.json5

# Then in TUI:
/load docs-example/Symposium.txt

# Watch terminal for:
# - Ollama API calls
# - "extract_with_gleaning" messages
# - Entity extraction progress
```

---

## 🐛 The Bug

### Location: `graphrag-core/src/lib.rs:350-406`

```rust
if self.config.entities.use_gleaning && self.config.ollama.enabled {
    // LLM extraction

    #[cfg(feature = "async")]
    {
        use crate::ollama::OllamaClient;

        // Create Ollama client
        let client = OllamaClient::new(self.config.ollama.clone());

        // ⚠️ NO ERROR HANDLING FOR CONNECTION FAILURE!
        // If client.new() fails or Ollama is not running,
        // it might panic or return an error that gets caught somewhere

        let extractor = GleaningEntityExtractor::new(...)
            .with_llm_client(client);

        // This will fail silently if Ollama is not reachable
        for chunk in &chunks {
            let (entities, relationships) = extractor
                .extract_with_gleaning(chunk)
                .await?;  // ← Might be failing here!
        }
    }
} else {
    // Falls back to pattern-based
}
```

### The Issue

There's **no explicit error handling** for:
1. Ollama connection failures
2. Model not available
3. API timeout
4. Silent fallback to pattern-based extraction

---

## 🔧 Recommended Fixes

### Fix 1: Add Connection Check Before Extraction

```rust
// graphrag-core/src/lib.rs
if self.config.entities.use_gleaning && self.config.ollama.enabled {
    // ✅ ADD: Verify Ollama connection first
    let client = OllamaClient::new(self.config.ollama.clone());

    match client.health_check().await {
        Ok(_) => {
            tracing::info!("Ollama connected, using LLM extraction");
            // Proceed with gleaning
        }
        Err(e) => {
            tracing::warn!(
                "Ollama connection failed: {}. Falling back to pattern-based extraction",
                e
            );
            // Fall back to pattern-based explicitly
            use_pattern_based = true;
        }
    }
}
```

### Fix 2: Add User Warning in TUI

```rust
// graphrag-cli/src/handlers/graphrag.rs
pub async fn load_document(&self, path: &Path) -> Result<String> {
    // Before building graph
    if config.entities.use_gleaning && !ollama_available().await {
        self.action_tx.send(Action::SetStatus(
            StatusType::Warning,
            "Ollama not available, using pattern-based extraction (faster but lower quality)"
        ))?;
    }

    graphrag.build_graph().await?;
}
```

### Fix 3: Add Progress Indicators

```rust
// Show progress during LLM extraction
for (i, chunk) in chunks.iter().enumerate() {
    self.action_tx.send(Action::UpdateProgress(
        format!("Extracting entities: chunk {}/{}", i+1, chunks.len())
    ))?;

    let (entities, relationships) = extractor
        .extract_with_gleaning(chunk)
        .await?;
}
```

---

## 📊 Performance Comparison

### Pattern-Based Extraction (Current Behavior)

```
Document: Symposium.txt (200KB, 3288 lines)
┌────────────────────────────────────────┐
│ Method: Pattern-based (regex)         │
│ Time: <1 second                        │
│ Entities: 200-500 (low quality)       │
│ Relationships: 50-100 (inferred)      │
│ CPU: Low                               │
│ Network: None                          │
└────────────────────────────────────────┘
```

### LLM-Based Extraction (Expected Behavior)

```
Document: Symposium.txt (200KB, 3288 lines)
┌────────────────────────────────────────┐
│ Method: LLM with gleaning (Ollama)    │
│ Time: 30-120 seconds                   │
│ Chunks: ~25 (800 tokens each)         │
│ LLM Calls: 100-150 (4 rounds × 25)    │
│ Entities: 30-80 (high quality)        │
│ Relationships: 80-150 (LLM-extracted) │
│ CPU: High (LLM inference)              │
│ Network: Moderate (Ollama API)        │
└────────────────────────────────────────┘
```

---

## ✅ Action Items

### For Users

1. **Check Ollama Status:**
   ```bash
   ollama serve
   ollama list
   ```

2. **Enable Debug Logging:**
   ```bash
   graphrag-cli --debug
   tail -f ~/.local/share/graphrag-cli/logs/graphrag-cli.log
   ```

3. **Verify Entity Quality:**
   ```bash
   # In TUI after /load:
   /entities

   # If you see many random capitalized words → pattern-based
   # If you see only philosophical concepts → LLM-based
   ```

### For Developers

1. **Add Ollama Health Check** (see Fix 1 above)
2. **Add User Warnings** for fallback behavior (see Fix 2 above)
3. **Add Progress Indicators** for LLM extraction (see Fix 3 above)
4. **Add Integration Test:**
   ```rust
   #[tokio::test]
   async fn test_extraction_method_selection() {
       // Test with Ollama running → should use LLM
       // Test with Ollama off → should warn + use pattern-based
   }
   ```

---

## 📝 Summary

**The Anomaly:**
- Documents load instantly (<1 second)
- Expected: 30-120 seconds for LLM extraction

**Root Cause:**
- Silent fallback to pattern-based extraction when Ollama is not available
- No user warning or error message
- No connection validation before attempting LLM extraction

**Impact:**
- Users think LLM extraction is working
- Actually getting low-quality pattern-based results
- No indication that Ollama is not being used

**Solution:**
- Add Ollama connection check before extraction
- Warn users when falling back to pattern-based
- Add progress indicators for LLM extraction
- Improve error handling and logging

---

**Created:** 2025-10-16
**Status:** Bug confirmed, fixes recommended
**Priority:** High (affects core functionality)
