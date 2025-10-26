# GraphRAG WASM Quick Start Guide

## Overview

This is a complete browser-based GraphRAG application with document ingestion, knowledge graph building, and intelligent querying—all running in WebAssembly!

## Features

### 1. Document Ingestion
- **File Upload**: Support for .txt, .md, and .pdf files
- **Text Paste**: Directly paste document content
- **Multi-Document**: Manage multiple documents simultaneously
- **Document Preview**: See content snippets and metadata

### 2. Knowledge Graph Building
- **Progress Visualization**: See real-time progress through 4 stages:
  1. Chunking Documents
  2. Extracting Entities
  3. Computing Embeddings
  4. Building Search Index
- **Status Indicators**: Clear feedback on graph build status
- **Performance**: Simulated pipeline (easily replaceable with real GraphRAG logic)

### 3. Graph Exploration
- **Statistics Dashboard**: 6 key metrics cards
  - Documents count
  - Text chunks
  - Entities
  - Relationships
  - Embeddings
  - Graph density
- **Health Indicators**: Visual progress bars for:
  - Coverage
  - Entity linking quality
  - Embedding quality
- **System Info**: Technology stack details

### 4. Intelligent Querying
- **Semantic Search**: Query the knowledge graph
- **Contextual Results**: Entity matches with relevance scores
- **Source Attribution**: Track answers back to documents
- **Visual Feedback**: Loading states and progress indicators

## Running the Application

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WebAssembly target
rustup target add wasm32-unknown-unknown

# Install Trunk (WASM build tool)
cargo install trunk
```

### Development Server

```bash
# Navigate to the graphrag-wasm directory
cd /home/dio/graphrag-rs/graphrag-wasm

# Start the development server (with hot reload)
trunk serve

# The app will be available at http://localhost:8080
```

### Production Build

```bash
# Build optimized WASM bundle
trunk build --release

# Output will be in the dist/ directory
# You can serve it with any static file server
```

## User Workflow

### Step 1: Add Documents

1. **Upload Files**:
   - Click the file input
   - Select one or more .txt, .md, or .pdf files
   - Files are read and added to the document library

2. **Paste Text**:
   - Optionally enter a document name
   - Paste content into the textarea
   - Click "Add Document"

3. **Manage Documents**:
   - View all documents in the library
   - See content previews and metadata
   - Remove documents with the trash icon

### Step 2: Build Knowledge Graph

1. Click "Build Knowledge Graph" button
2. Watch the progress visualization:
   - Purple gradient progress bar
   - Stage labels with percentages
   - Current/total item counts
3. Wait for completion (simulation takes ~5-10 seconds for 3 documents)
4. See "Knowledge Graph Ready!" success message

### Step 3: Explore Graph (Optional)

1. Switch to "Explore Graph" tab
2. View statistics:
   - Document count and chunking stats
   - Entity and relationship counts
   - Embedding and indexing metrics
3. Check health indicators
4. Review system configuration

### Step 4: Query Graph

1. Switch to "Query Graph" tab
2. Enter your question in the input field
3. Click "Search Graph"
4. View results:
   - Query summary
   - Entity matches with relevance scores
   - Context snippets
   - Source attribution

## Architecture Highlights

### Technology Stack

- **Frontend**: Leptos 0.8 (Reactive Rust UI)
- **Styling**: Tailwind CSS + DaisyUI (npm install)
- **Build**: Trunk (WASM bundler)
- **Tokenizer**: HuggingFace tokenizers (unstable_wasm)
- **Embeddings**: ONNX Runtime Web (WebGPU accelerated)
- **LLM**: WebLLM (Qwen3-1.7B)
- **Vector Search**: Pure Rust cosine similarity
- **Storage**: IndexedDB + Cache API

### Component Structure

```
App
├── Header
├── TabNavigation
├── BuildTab
│   ├── Document Input (upload + paste)
│   ├── Document Library
│   └── BuildProgress
├── ExploreTab
│   ├── StatCard × 6
│   ├── HealthIndicator × 3
│   └── System Configuration
├── QueryTab
│   ├── Query Input
│   └── Results Display
└── Footer
```

### State Management

All state is managed with Leptos signals:
- `documents`: Vec of Document structs
- `build_status`: Idle | Building | Ready | Error
- `graph_stats`: Documents, chunks, entities, relationships, embeddings counts
- `active_tab`: Build | Explore | Query
- `query` + `results` + `loading`: Query interface state

## Customization Guide

### Replace Mock Graph Building

The current implementation simulates graph building for demonstration. To integrate real GraphRAG:

```rust
// In BuildTab, replace the build_graph function:

let build_graph = move |_| {
    spawn_local(async move {
        // 1. Chunk documents
        set_build_status.set(BuildStatus::Building(BuildStage::Chunking { ... }));
        let chunks = chunk_documents(&docs).await;

        // 2. Extract entities
        set_build_status.set(BuildStatus::Building(BuildStage::Extracting { ... }));
        let entities = extract_entities(&chunks).await;

        // 3. Compute embeddings
        set_build_status.set(BuildStatus::Building(BuildStage::Embedding { ... }));
        let embeddings = compute_embeddings(&entities).await;

        // 4. Build index
        set_build_status.set(BuildStatus::Building(BuildStage::Indexing { ... }));
        build_voy_index(&embeddings).await;

        // 5. Complete
        set_build_status.set(BuildStatus::Ready);
        set_graph_stats.set(/* real stats */);
    });
};
```

### Add Persistent Storage

To save documents and graphs across sessions:

```rust
// Add to main.rs:

use web_sys::window;

// Save documents to IndexedDB
async fn save_documents_to_indexeddb(docs: &Vec<Document>) {
    let window = window().unwrap();
    let idb = window.indexed_db().unwrap().unwrap();
    // ... IndexedDB logic
}

// Load on app init
async fn load_documents_from_indexeddb() -> Vec<Document> {
    // ... IndexedDB logic
}
```

### Enhance Graph Visualization

Add a canvas-based graph visualization:

```rust
#[component]
fn GraphCanvas(graph_stats: ReadSignal<GraphStats>) -> impl IntoView {
    view! {
        <canvas
            id="graph-canvas"
            width="800"
            height="600"
            class="w-full h-96 bg-slate-900 rounded-lg"
        />
        // Use JS interop to draw with D3.js or raw Canvas API
    }
}
```

## Accessibility Features

### Keyboard Navigation

- **Tab**: Navigate between interactive elements
- **Enter/Space**: Activate buttons
- **Arrow Keys**: Switch tabs (when tab is focused)
- **Escape**: Clear inputs or cancel operations

### Screen Reader Support

All interactive elements have proper ARIA labels:
- Tab navigation: `role="tablist"`, `role="tab"`, `aria-selected`
- Progress bars: `role="progressbar"`, `aria-valuenow`
- Form inputs: Associated `<label>` elements
- Status updates: `aria-live="polite"` regions

### Visual Accessibility

- **Contrast**: WCAG 2.1 AA compliant (4.5:1 for text)
- **Focus**: Visible purple focus rings on all interactive elements
- **Colors**: Status shown with icons AND color
- **Text Size**: Responsive and scalable

## Performance Tips

### Development

```bash
# Fast rebuilds with trunk serve
trunk serve

# With specific port
trunk serve --port 3000

# With release optimizations (slower build, faster runtime)
trunk serve --release
```

### Production

```bash
# Full optimization
trunk build --release

# Additional WASM optimization
wasm-opt dist/*.wasm -O3 -o dist/optimized.wasm

# Compress with Brotli
brotli dist/*.wasm

# Serve with compression
# Ensure your server sends proper WASM MIME type:
# Content-Type: application/wasm
# Content-Encoding: br
```

### Bundle Size Tips

Current bundle sizes (approximate):
- WASM module: ~300KB (compressed)
- Leptos framework: ~50KB
- Dependencies: ~100KB
- Total first load: ~450KB

To reduce:
- Use `cargo bloat` to analyze
- Remove unused features
- Split code with dynamic imports
- Enable LTO in release profile

## Browser Compatibility

### Supported Browsers

- ✅ Chrome/Edge 87+ (best performance with WebGPU)
- ✅ Firefox 89+ (fallback to WebGL for embeddings)
- ✅ Safari 15.2+ (some WebGPU limitations)
- ✅ Mobile Safari (iOS 15.2+)
- ✅ Mobile Chrome (Android)

### Required Features

- WebAssembly (all modern browsers)
- ES2020 modules
- FileReader API
- Optional: WebGPU (for fast embeddings)

## Troubleshooting

### Build Errors

```bash
# Clear build cache
cargo clean

# Update dependencies
cargo update

# Check Trunk version
trunk --version  # Should be 0.17+
```

### Runtime Errors

**Error: "WASM module not found"**
- Ensure you're serving from the `dist/` directory
- Check that `trunk build` completed successfully

**Error: "Failed to load file"**
- Check browser console for CORS errors
- Use `trunk serve` for local development

**Error: "IndexedDB not available"**
- Check browser privacy settings
- Some browsers block IndexedDB in private mode

### Performance Issues

- Use Chrome DevTools → Performance tab
- Enable WebGPU in chrome://flags
- Check WASM execution time in profiler
- Consider code splitting for large apps

## Next Steps

### Immediate Enhancements

1. **Real GraphRAG Integration**:
   - Replace mock functions with actual processing
   - Integrate ONNX Runtime Web for embeddings
   - Connect to WebLLM for entity extraction

2. **Persistent Storage**:
   - IndexedDB for documents
   - Cache API for models
   - LocalStorage for preferences

3. **Advanced Features**:
   - Drag-and-drop file upload
   - Document preview modal
   - Export results (JSON/CSV)
   - Query history

### Long-term Roadmap

1. **Visualization**:
   - Interactive graph canvas
   - Entity relationship diagram
   - Timeline view

2. **Collaboration**:
   - Share graphs via URL
   - Export/import functionality
   - Cloud sync (optional)

3. **Advanced RAG**:
   - Multi-hop reasoning
   - Confidence scores
   - Hybrid search (BM25 + vector)

## Resources

- **Leptos Docs**: https://leptos.dev
- **Trunk Docs**: https://trunkrs.dev
- **WASM Docs**: https://rustwasm.github.io
- **Tailwind CSS**: https://tailwindcss.com

## License

See the main repository LICENSE file.

## Contributing

Contributions welcome! Please see the main repository CONTRIBUTING guide.

---

**Ready to build knowledge graphs in your browser? Run `trunk serve` and start exploring!** 🚀
