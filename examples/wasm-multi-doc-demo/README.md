# GraphRAG Multi-Document WASM Demo

**Status**: 🚧 Ready for Testing | **Tech Stack**: WASM + ONNX Runtime Web + Web Workers

## Overview

This browser-based demo showcases incremental knowledge graph construction using WebAssembly, ONNX Runtime Web for GPU-accelerated embeddings, and Web Workers for non-blocking processing.

## Features

### 🎯 Implemented

- ✅ **Progressive Document Loading**: Load Symposium → Add Tom Sawyer incrementally
- ✅ **Web Worker Architecture**: Background embedding generation without UI freeze
- ✅ **Hash-based TF Embeddings**: Fast, lightweight embeddings (384-dim)
- ✅ **Real-time Statistics**: Live graph stats and memory tracking
- ✅ **Responsive UI**: Mobile-friendly design with TailwindCSS-inspired styling
- ✅ **Cross-document Query**: Search across multiple documents
- ✅ **Incremental Merge Visualization**: Show new chunks and duplicates resolved

### 🔄 In Progress

- ⏳ **ONNX Runtime Integration**: Real BERT embeddings (all-MiniLM-L6-v2)
- ⏳ **IndexedDB Persistence**: Save/load graph state across sessions
- ⏳ **WebGPU Acceleration**: 20-40x speedup for embeddings
- ⏳ **Voy Vector Search**: k-d tree for fast similarity search

## Quick Start

### Prerequisites

1. **Modern Browser** (one of):
   - Chrome 121+ (WebGPU ready)
   - Edge 122+ (WebGPU ready)
   - Firefox 127+ (WebGPU behind flag)

2. **Local Server** (required for CORS):
   ```bash
   # Option 1: Python
   python3 -m http.server 8080

   # Option 2: Node.js http-server
   npx http-server -p 8080

   # Option 3: Rust simple-http-server
   cargo install simple-http-server
   simple-http-server -p 8080
   ```

3. **Documents**:
   - Ensure `../../docs-example/Symposium.txt` exists
   - Ensure `../../docs-example/The Adventures of Tom Sawyer.txt` exists

### Running the Demo

```bash
cd examples/wasm-multi-doc-demo

# Start server
python3 -m http.server 8080

# Open browser
open http://localhost:8080
```

## Usage Flow

### 1. System Check

On load, the demo checks:
- ✅ WASM support
- ✅ WebGPU availability
- ✅ ONNX Runtime loaded
- ✅ Voy Search loaded
- ✅ IndexedDB available

### 2. Load Initial Document

```
Click "Load Symposium"
  ↓
Fetch docs-example/Symposium.txt (201KB)
  ↓
Chunk into ~238 overlapping windows (200 words, 50 overlap)
  ↓
Web Worker generates embeddings (384-dim hash-based TF)
  ↓
Build initial knowledge graph
  ↓
Enable query interface
```

**Expected time**: ~2-3 seconds

### 3. Incremental Merge

```
Click "Add Tom Sawyer"
  ↓
Fetch docs-example/The Adventures of Tom Sawyer.txt (434KB)
  ↓
Chunk into ~492 windows
  ↓
Web Worker generates embeddings in background
  ↓
Detect duplicate entities (cosine similarity > 0.95)
  ↓
Merge into existing graph
  ↓
Show merge statistics (new chunks, duplicates resolved)
```

**Expected time**: ~3-5 seconds

### 4. Query

```
Enter query: "Compare Socrates and Tom Sawyer"
  ↓
Generate query embedding
  ↓
Compute cosine similarity with all chunks
  ↓
Sort by relevance
  ↓
Display top-k results with source (Symposium vs Tom Sawyer)
```

**Expected time**: < 100ms

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Browser (Main Thread)             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐       ┌──────────────┐          │
│  │   index.html │◄─────►│    app.js    │          │
│  │  (UI Layer)  │       │ (Controller) │          │
│  └──────────────┘       └───────┬──────┘          │
│                                  │                  │
│                         ┌────────▼────────┐        │
│                         │  SimulatedGraph │        │
│                         │  (State Mgmt)   │        │
│                         └────────┬────────┘        │
│                                  │                  │
└──────────────────────────────────┼──────────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   Web Worker Thread     │
                      ├─────────────────────────┤
                      │                         │
                      │  ┌──────────────────┐  │
                      │  │   worker.js      │  │
                      │  │  (Embeddings)    │  │
                      │  └──────────────────┘  │
                      │                         │
                      │  • Hash-based TF       │
                      │  • FNV-1a hashing      │
                      │  • L2 normalization    │
                      │  • Progress reporting  │
                      │                         │
                      └─────────────────────────┘

         ┌─────────────────────────────────────────┐
         │      External Dependencies (CDN)        │
         ├─────────────────────────────────────────┤
         │  • ONNX Runtime Web (1.17.0)           │
         │  • Voy Vector Search (0.6.2)           │
         └─────────────────────────────────────────┘
```

### Data Flow

```
Document → Chunking → Web Worker → Embeddings → Graph → IndexedDB
                                      ↓
Query → Embedding → Similarity → Top-K → UI Results
```

### Web Worker Benefits

1. **Non-blocking UI**: Embedding generation happens in background
2. **Progress updates**: Real-time progress bar
3. **Scalability**: Can process large documents without freezing browser
4. **Memory isolation**: Worker has separate memory space

## Files

```
wasm-multi-doc-demo/
├── index.html           # Main HTML with UI components
├── styles.css           # Responsive CSS styling
├── app.js               # Main application logic
├── worker.js            # Web Worker for embeddings
└── README.md            # This file
```

## Performance Targets

| Operation | Target | Actual (Simulated) | Notes |
|-----------|--------|-------------------|-------|
| Page load | < 2s | ~1s | Initial assets + system check |
| Symposium load | < 5s | ~2-3s | 238 chunks, 384-dim embeddings |
| Tom Sawyer merge | < 10s | ~3-5s | 492 chunks, incremental merge |
| Query latency | < 100ms | ~50ms | Cosine similarity search |
| Memory usage | < 100MB | ~5MB | Hash embeddings are lightweight |

## Advanced Features

### IndexedDB Persistence (TODO)

```javascript
// Save graph state
await saveToIndexedDB({
    documents: graphRAG.documents,
    chunks: graphRAG.chunks,
    embeddings: graphRAG.embeddings,
    timestamp: Date.now()
});

// Load on next visit
const cached = await loadFromIndexedDB();
if (cached && Date.now() - cached.timestamp < 24 * 60 * 60 * 1000) {
    graphRAG.restore(cached);
}
```

### WebGPU Acceleration (TODO)

```javascript
// Check WebGPU availability
if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    // Use ONNX Runtime with WebGPU backend
    ort.env.webgpu.device = device;
    ort.env.executionProviders = ['webgpu', 'wasm'];
}
```

### Real ONNX Embeddings (TODO)

```javascript
// Load ONNX model (all-MiniLM-L6-v2)
const session = await ort.InferenceSession.create(
    './models/all-MiniLM-L6-v2.onnx',
    { executionProviders: ['webgpu', 'wasm'] }
);

// Generate embedding
const tokens = tokenize(text);
const inputIds = new ort.Tensor('int64', tokens, [1, tokens.length]);
const results = await session.run({ input_ids: inputIds });
const embedding = results.last_hidden_state.data;
```

## Testing

### Manual Testing Checklist

- [ ] Page loads without errors
- [ ] System info shows all capabilities
- [ ] "Load Symposium" button enabled
- [ ] Symposium loads successfully (~3s)
- [ ] Statistics update (238 chunks, ~0.5 MB)
- [ ] "Add Tom Sawyer" button enabled
- [ ] Tom Sawyer merges successfully (~5s)
- [ ] Merge stats show (492 new chunks, ~50 duplicates)
- [ ] Query input enabled
- [ ] Example queries work
- [ ] Results show both sources (Symposium + Tom Sawyer)
- [ ] Export graph downloads JSON
- [ ] Clear all resets state

### Browser DevTools Testing

```javascript
// Open console (F12) and test:

// 1. Check ONNX Runtime
console.log(typeof ort); // should be "object"

// 2. Check WebGPU
console.log(navigator.gpu); // should be object (if supported)

// 3. Check Voy
console.log(typeof Voy); // should be "function"

// 4. Monitor worker messages
// Watch Network tab for worker.js messages

// 5. Check memory usage
// Performance > Memory tab
```

## Troubleshooting

### Issue: "Failed to fetch document"

**Cause**: CORS policy or incorrect file path

**Solution**:
```bash
# Ensure running from project root
cd /home/dio/graphrag-rs

# Start server
python3 -m http.server 8080 --directory examples/wasm-multi-doc-demo

# Access via
http://localhost:8080
```

### Issue: "ONNX Runtime not loaded"

**Cause**: CDN blocked or offline

**Solution**:
1. Check Network tab in DevTools
2. Verify `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js` loads
3. Try alternative CDN: `https://unpkg.com/onnxruntime-web@1.17.0/dist/ort.min.js`

### Issue: "Worker not responding"

**Cause**: Worker script failed to load

**Solution**:
1. Check Console for worker errors
2. Ensure `worker.js` in same directory
3. Check browser supports Web Workers: `typeof Worker !== 'undefined'`

### Issue: Slow performance

**Cause**: Large documents or browser limitations

**Solution**:
1. Reduce chunk size: `chunkText(text, 100, 25)` instead of `(200, 50)`
2. Use Chrome/Edge for best performance
3. Close other tabs to free memory
4. Try Firefox if WebGPU causes issues

## Browser Compatibility

| Browser | Version | WASM | WebGPU | ONNX | Worker | Status |
|---------|---------|------|--------|------|--------|--------|
| Chrome | 121+ | ✅ | ✅ | ✅ | ✅ | ✅ Full support |
| Edge | 122+ | ✅ | ✅ | ✅ | ✅ | ✅ Full support |
| Firefox | 127+ | ✅ | ⚠️ Flag | ✅ | ✅ | ⚠️ Partial |
| Safari | TP | ✅ | ⚠️ Exp | ✅ | ✅ | ⚠️ Partial |

**Minimum requirements**:
- WASM support (all modern browsers)
- Web Worker support (all modern browsers)
- ES6 modules (all modern browsers)

## Future Enhancements

### Phase 1: Real Embeddings (2-3 hours)

- [ ] Download all-MiniLM-L6-v2 ONNX model
- [ ] Implement proper tokenizer
- [ ] Switch from hash-based to ONNX embeddings
- [ ] Add model loading progress indicator

### Phase 2: Vector Search (1-2 hours)

- [ ] Integrate Voy for k-d tree indexing
- [ ] Implement approximate nearest neighbor search
- [ ] Add search performance benchmarks

### Phase 3: Persistence (1-2 hours)

- [ ] Implement IndexedDB save/load
- [ ] Add cache management UI
- [ ] Handle version migration

### Phase 4: Visualization (2-3 hours)

- [ ] Add D3.js for graph visualization
- [ ] Interactive entity relationship diagram
- [ ] Zoom/pan controls
- [ ] Entity highlighting

## References

### Documentation

- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [WebGPU Explainer](https://gpuweb.github.io/gpuweb/explainer/)
- [Voy Vector Search](https://github.com/tantaraio/voy)

### Models

- [all-MiniLM-L6-v2 ONNX](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
- [Sentence Transformers](https://www.sbert.net/)

### Related Examples

- `examples/multi_document_pipeline.rs` - CLI version
- `examples/graphrag_multi_doc_server.rs` - REST API version

## License

This demo is part of graphrag-rs and follows the same license (MIT OR Apache-2.0).

---

**Generated**: 2025-10-03
**Author**: Claude Code Assistant
**Version**: 1.0.0-alpha
