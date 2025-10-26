# GraphRAG Web App Example

A complete example web application demonstrating all GraphRAG Leptos components running 100% client-side with Rust + WASM.

## Features

- ✅ **Chat Interface**: Interactive chat with GraphRAG queries
- ✅ **Document Upload**: Drag & drop document management
- ✅ **Graph Visualization**: Interactive knowledge graph display
- ✅ **Real-time Stats**: Live graph statistics
- ✅ **WebGPU Detection**: Automatic GPU acceleration detection
- ✅ **IndexedDB Storage**: Persistent data in browser
- ✅ **TailwindCSS + daisyUI**: Modern, responsive design
- ✅ **100% Rust + WASM**: No JavaScript framework needed

## Quick Start

### Prerequisites

```bash
# Install Trunk (WASM bundler)
cargo install trunk wasm-bindgen-cli

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Development

```bash
# Run development server (with hot reload)
trunk serve --open

# Or navigate to:
cd examples/web-app
trunk serve --open
```

The app will be available at http://localhost:8080

### Production Build

```bash
# Build for production (optimized)
trunk build --release

# Output will be in ./dist/
# Deploy this folder to any static hosting (Netlify, Vercel, GitHub Pages)
```

## Architecture

```
Browser (100% Client-Side)
├─ Leptos UI Components
│  ├─ ChatWindow
│  ├─ QueryInterface
│  ├─ GraphStats
│  ├─ DocumentManager
│  └─ GraphVisualization
│
├─ GraphRAG WASM Bindings
│  ├─ Vector Search (Voy)
│  ├─ Embeddings (Candle or Burn)
│  └─ LLM (WebLLM or Candle)
│
└─ Browser Storage
   ├─ IndexedDB (graph data)
   └─ Cache API (ML models)
```

## Components Demonstrated

### 1. ChatWindow
```rust
<ChatWindow on_query=handle_query />
```
- Message history with timestamps
- Loading states
- Error handling
- Clear history

### 2. QueryInterface
```rust
<QueryInterface on_submit=handle_submit />
```
- Text input with auto-focus
- Enter to submit
- Character counter
- Disabled states

### 3. GraphStats
```rust
<GraphStats
    entity_count=entity_count
    relationship_count=relationship_count
    document_count=document_count
    vector_count=vector_count
/>
```
- Real-time statistics
- Animated counters
- Responsive layout

### 4. DocumentManager
```rust
<DocumentManager
    on_upload=handle_upload
    on_remove=handle_remove
/>
```
- File upload interface
- Document list
- Remove functionality
- Supported formats: TXT, MD, PDF

### 5. GraphVisualization
```rust
<GraphVisualization
    nodes=nodes
    edges=edges
    on_node_click=Some(handle_node_click)
/>
```
- Interactive graph rendering
- Zoom controls
- Node selection
- Edge visualization

## Browser Support

| Feature | Chrome/Edge | Firefox | Safari |
|---------|-------------|---------|--------|
| WASM | ✅ 100% | ✅ 100% | ✅ 100% |
| WebGPU | ✅ 113+ | ⚠️ 121+ (flag) | ⚠️ 18+ (partial) |
| IndexedDB | ✅ | ✅ | ✅ |
| Cache API | ✅ | ✅ | ✅ |

**Recommended**: Chrome/Edge 113+ for full WebGPU acceleration

## Customization

### Themes

Edit `index.html` to change daisyUI theme:

```html
<html lang="en" data-theme="dark">
```

Available themes: `light`, `dark`, `cupcake`, `cyberpunk`, `forest`, `luxury`, etc.

### Styling

Components use Tailwind utility classes. Customize in your components:

```rust
view! {
    <div class="bg-primary text-primary-content p-4 rounded-lg">
        "Custom styled element"
    </div>
}
```

## Deployment

### Static Hosting

```bash
# Build for production
trunk build --release

# Deploy ./dist/ to:
# - Netlify: netlify deploy --dir=dist --prod
# - Vercel: vercel --prod
# - GitHub Pages: Copy dist/ to gh-pages branch
# - Any CDN or static host
```

### Configuration

No server configuration needed! The app runs entirely in the browser.

For CDN deployment, ensure MIME types are correct:
- `.wasm` → `application/wasm`
- `.js` → `application/javascript`

## Troubleshooting

**WASM not loading:**
```bash
# Clear browser cache
# Rebuild with: trunk clean && trunk build --release
```

**WebGPU not available:**
```
Check chrome://gpu
Enable experimental features if needed
```

**Build errors:**
```bash
# Update dependencies
cargo update

# Clean and rebuild
trunk clean
cargo clean
trunk build --release
```

## Learn More

- [Leptos Documentation](https://leptos.dev)
- [Trunk Documentation](https://trunkrs.dev)
- [TailwindCSS](https://tailwindcss.com)
- [daisyUI Components](https://daisyui.com/components/)
- [GraphRAG Implementation Plan](../../IMPLEMENTATION_PLAN.md)
- [GraphRAG Architecture](../../ARCHITECTURE.md)

## License

MIT
