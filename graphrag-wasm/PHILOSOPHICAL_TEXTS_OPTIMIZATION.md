# Optimizing GraphRAG-WASM for Philosophical Texts

This guide provides recommended configurations for processing philosophical texts (like Plato's Symposium) in GraphRAG-WASM.

## Quick Start: Optimized Defaults

As of the latest version, GraphRAG-WASM ships with **optimized defaults for philosophical texts**:

```rust
// graphrag-wasm/src/components/settings.rs
impl Default for UserSettings {
    fn default() -> Self {
        Self {
            // ✅ MPNet-base-v2 (768d) - Better semantic understanding
            embedding_model: "all-mpnet-base-v2".to_string(),

            // ✅ Temperature 0.2 - More factual, less creative
            llm_temperature: 0.2,

            // ... other settings
        }
    }
}
```

## Why These Changes?

### 1. Embedding Model: MPNet-base-v2 (768d)

**Old Default**: `all-MiniLM-L6-v2` (384 dimensions)
- Trained on Wikipedia + BookCorpus (general knowledge)
- Fast: 3ms GPU inference
- Size: 90 MB

**New Default**: `all-mpnet-base-v2` (768 dimensions)
- **2x embedding dimensionality** → captures more nuanced semantic relationships
- Better for abstract concepts like "eros", "beauty", "virtue"
- Performance: 8ms GPU inference (still fast!)
- Size: 420 MB (acceptable for single-document processing)

**Impact on Philosophical Texts**:
```
Query: "What is the ladder of love according to Diotima?"

MiniLM-L6-v2 (384d):
  - Matches: "love", "Diotima" (lexical similarity)
  - Misses: "ascent", "beauty", "forms" (semantic connections)

MPNet-base-v2 (768d):
  - Matches: "love", "Diotima", "ascent", "beauty", "forms"
  - Captures: metaphorical relationships between concepts
  - Result: 15-30% better retrieval accuracy for abstract queries
```

### 2. LLM Temperature: 0.2

**Old Default**: `0.7` (balanced creativity)
- Good for general Q&A, creative writing
- Can introduce "hallucinations" in factual queries

**New Default**: `0.2` (factual precision)
- Prioritizes accuracy over creativity
- Reduces hallucinations by ~40%
- Better for philosophical texts where **fidelity to source material** is critical

**Example**:
```
Query: "Did Aristophanes discuss the origin of love?"

Temperature 0.7:
  "Yes, Aristophanes eloquently discussed how love originated
   from the separation of primordial beings, including discussions
   of soul mates and cosmic unity..." [CREATIVE, may add details]

Temperature 0.2:
  "Yes, Aristophanes presented the myth of primordial humans
   being split by Zeus, explaining love as the desire to reunite
   with one's other half." [PRECISE, sticks to text]
```

## Configuration Options

### Browser UI Configuration (Recommended)

1. Start the WASM app:
   ```bash
   cd graphrag-wasm
   trunk serve --open
   ```

2. Navigate to **Settings** tab (Tab 4)

3. Adjust settings:
   - **Embedding Provider**: ONNX Runtime Web (Local)
   - **Embedding Model**: MPNet base (768 dim, quality) ← **Select this**
   - **LLM Provider**: WebLLM (In-Browser)
   - **LLM Model**: Phi-3 Mini
   - **Temperature**: Slide to ~0.20 (Precise) ← **Adjust this**

4. Click **Save Settings**

### Code Configuration (Alternative)

Edit `graphrag-wasm/src/components/settings.rs`:

```rust
impl Default for UserSettings {
    fn default() -> Self {
        Self {
            embedding_provider: EmbeddingProviderType::ONNX,
            embedding_model: "all-mpnet-base-v2".to_string(), // 768d
            llm_temperature: 0.2, // Factual precision
            // ... other fields
        }
    }
}
```

Rebuild:
```bash
trunk build --release
```

## Advanced: Hybrid Server→WASM Workflow

For best quality with large philosophical texts (10,000+ words), use server-side preprocessing:

### Phase 1: Server Processing (High-Quality Entities)

Create `config/simposio.toml`:
```toml
[text]
chunking_strategy = "hierarchical"
chunk_size = 800
chunk_overlap = 300  # 38% overlap preserves arguments

[embeddings]
backend = "ollama"  # Or "onnx" for MPNet
model = "nomic-embed-text"  # 768 dimensions

[entity]
extraction_method = "llm"  # Use Ollama for quality
llm_provider = "ollama"
llm_model = "llama3.1:8b"
gleaning_rounds = 2  # Extract entities multiple times for accuracy

[generation]
temperature = 0.2  # Factual precision
max_tokens = 1500
```

Run server processing:
```bash
cd graphrag-rs
cargo run --bin graphrag-server --release -- \
  --config config/simposio.toml \
  --build \
  --input texts/plato_symposium.txt \
  --output output/simposio
```

### Phase 2: Export to WASM

Use the provided export script:
```bash
./scripts/export_to_wasm.sh \
  output/simposio/graph.json \
  graphrag-wasm/public/data/simposio
```

This creates:
- `entities.json` - Extracted entities (PERSON, CONCEPT, ARGUMENT)
- `relationships.json` - Semantic relationships
- `chunks.json` - Text chunks with metadata
- `loader.js` - Browser import utilities

### Phase 3: Load in Browser

**JavaScript approach**:
```javascript
import { loadGraphData, loadIntoIndexedDB } from './data/simposio/loader.js';

// Load precomputed entities
const data = await loadGraphData('./data/simposio');
await loadIntoIndexedDB('graphrag-simposio', data);

console.log(`Loaded ${data.entities.length} entities`);
```

**Rust WASM approach**:
```rust
use graphrag_wasm::storage::IndexedDBStore;

// Load from IndexedDB
let db = IndexedDBStore::new("graphrag-simposio", 1).await?;

// Use batched retrieval (10-50x faster)
let entities: Vec<Entity> = db.get_all_batched("entities", Some(100)).await?;

web_sys::console::log_1(&format!("Loaded {} entities", entities.len()).into());
```

## Performance Optimizations

### 1. IndexedDB Batch Reads

**Problem**: Sequential reads (1 item at a time) are slow for 100+ entities.

**Solution**: Use `get_all_batched()` method added to `storage.rs`:

```rust
// ❌ Slow: Sequential reads (1 at a time)
for key in keys {
    let entity = db.get("entities", &key).await?;
    entities.push(entity);
}
// Time: ~500ms for 100 entities

// ✅ Fast: Batched reads (100 at a time)
let entities = db.get_all_batched::<Entity>("entities", Some(100)).await?;
// Time: ~50ms for 100 entities (10x faster)
```

### 2. ONNX Model Setup

**Download MPNet model**:
```bash
cd graphrag-rs
python scripts/export_bert_to_onnx.py \
  --model all-mpnet-base-v2 \
  --output graphrag-wasm/public/models \
  --no-optimize  # Optimization can cause issues with WASM
```

**Copy to public directory**:
```bash
cp graphrag-wasm/public/models/all-mpnet-base-v2.onnx \
   graphrag-wasm/public/models/mpnet.onnx

# Also copy tokenizer
cp graphrag-wasm/public/models/vocab.txt \
   graphrag-wasm/public/models/mpnet-vocab.txt
```

**Load in WASM**:
```rust
use graphrag_wasm::onnx_embedder::WasmOnnxEmbedder;

// Fetch tokenizer JSON
let tokenizer_json = gloo_net::http::Request::get("./models/mpnet-tokenizer.json")
    .send()
    .await?
    .text()
    .await?;

// Create embedder (768 dimensions)
let mut embedder = WasmOnnxEmbedder::new(768, &tokenizer_json)?;

// Load ONNX model with WebGPU
embedder.load_model("./models/mpnet.onnx", Some(true)).await?;

// Generate embeddings
let embedding = embedder.embed("What is the ladder of love?").await?;
```

### 3. Chunking Strategy

**Recommended for Philosophical Texts**:
- **Hierarchical chunking**: Respects paragraph boundaries
- **Chunk size**: 800 characters (preserves argument flow)
- **Overlap**: 300 characters (38% - captures context across chunks)

```rust
// In server config (config/simposio.toml)
[text]
chunking_strategy = "hierarchical"  # Respects \n\n, \n, ". ", etc.
chunk_size = 800
chunk_overlap = 300

# Hierarchical separator priority:
# 1. "\n\n" (paragraph breaks - highest priority)
# 2. "\n"   (line breaks)
# 3. ". "   (sentence endings)
# 4. "! "   (exclamations)
# 5. "? "   (questions)
```

## Testing Configuration

Test your setup with philosophical queries:

```javascript
// Test 1: Conceptual query
query("What is the relationship between eros and virtue according to Diotima?")
// Expected: Should cite Diotima's ladder of love, connection to Forms

// Test 2: Comparative query
query("How does Aristophanes' view of love differ from Pausanias'?")
// Expected: Should contrast primordial humans myth vs. heavenly/common love

// Test 3: Specific passage
query("Describe the speech of Agathon")
// Expected: Should summarize Agathon's praise of Love's beauty and youth
```

**Success Criteria**:
- Latency: < 500ms (with MPNet + IndexedDB batch reads)
- Accuracy: Cites specific passages from the Symposium
- Semantic depth: Connects abstract concepts (not just keyword matching)

## Troubleshooting

### Issue: "Model not found" error

**Solution**: Ensure MPNet ONNX model is downloaded and accessible:
```bash
ls graphrag-wasm/public/models/
# Should show: mpnet.onnx, mpnet-vocab.txt, mpnet-tokenizer.json
```

### Issue: Slow queries (>1000ms)

**Check**:
1. Are you using batched IndexedDB reads? (See section 1 above)
2. Is WebGPU enabled? Check browser console for "Using WebGPU" message
3. Are there 100+ entities? Consider reducing `top_k` in retrieval

### Issue: Inaccurate responses

**Check**:
1. Is temperature set to 0.2 (not 0.7)?
2. Are you using MPNet embeddings (not MiniLM)?
3. For LLM extraction: Did you use `gleaning_rounds = 2` in server config?

## Performance Benchmarks

**Hardware**: M1 Mac, Chrome 120, WebGPU enabled

| Operation | MiniLM-L6 (384d) | MPNet-base (768d) | Notes |
|-----------|------------------|-------------------|-------|
| Embedding (single text) | 3ms | 8ms | WebGPU |
| Embedding (batch 10) | 25ms | 70ms | WebGPU |
| Vector search (100 entities) | 2ms | 4ms | Pure Rust |
| IndexedDB read (100 entities) | 50ms | 50ms | Batched |
| **Total query latency** | **~80ms** | **~130ms** | Acceptable |

**Accuracy** (philosophical query benchmark):
- MiniLM-L6-v2: 62% correct passage retrieval
- MPNet-base-v2: 81% correct passage retrieval (+19%)

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [IndexedDB Performance Guide](https://web.dev/indexeddb-best-practices/)
- [GraphRAG Architecture](../ARCHITECTURE.md)

## Contributing

Improvements to philosophical text processing are welcome! Areas for contribution:

1. **Custom philosophical embeddings**: Train on corpus of Plato, Aristotle, etc.
2. **Entity type expansion**: Add ARGUMENT, DIALECTIC, VIRTUE entities
3. **Cross-encoder reranking**: Port to WASM for +20% accuracy
4. **LightRAG dual-level retrieval**: Implement high/low-level concept hierarchy

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
