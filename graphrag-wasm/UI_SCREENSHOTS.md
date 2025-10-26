# GraphRAG WASM UI Visual Guide

## Layout Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         HEADER SECTION                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          🧠 GraphRAG WASM                                 │  │
│  │     Knowledge Graph RAG in Your Browser                   │  │
│  │                                                            │  │
│  │  [✅ rust_tokenizers] [✅ ONNX] [✅ WebLLM] [✅ Voy]     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│                       TAB NAVIGATION                            │
│  ┌──────────────┬──────────────┬──────────────┐               │
│  │ 📄 1. Build  │ 🔍 2. Explore│ 💬 3. Query   │               │
│  │ Graph        │ Graph        │ Graph         │               │
│  │ [3 docs]     │ [Ready]      │               │               │
│  └──────────────┴──────────────┴──────────────┘               │
│                                                                 │
│                     CONTENT AREA                                │
│  [Active tab content displayed here]                           │
│                                                                 │
│                       FOOTER                                    │
│  Built with Leptos + Rust + WebAssembly                        │
│  GraphRAG WASM • Complete Document Pipeline • v0.2.0           │
└─────────────────────────────────────────────────────────────────┘
```

## Tab 1: Build Graph

### Document Input Section
```
┌─────────────────────────────────────────────────────────────┐
│ 📝 Add Documents                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Upload Files (txt, md, pdf)                                 │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ [Choose Files] No file chosen          [Browse Button] │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ Document Name (optional)                                    │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ My Document                                            │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ Paste Text Content                                          │
│ ┌────────────────────────────────────────────────────────┐ │
│ │                                                        │ │
│ │ Paste your document content here...                   │ │
│ │                                                        │ │
│ │                                                        │ │
│ │                                                        │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │           ➕ Add Document                              │ │
│ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Document Library (Empty State)
```
┌─────────────────────────────────────────────────────────────┐
│ 📚 Document Library (0 documents)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                          📭                                  │
│                                                              │
│                    No documents yet                          │
│           Upload files or paste text to get started          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Document Library (With Documents)
```
┌─────────────────────────────────────────────────────────────┐
│ 📚 Document Library (3 documents)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ My Research Paper                              [🗑️] │   │
│ │ This is a comprehensive research paper about...      │   │
│ │ 15,432 bytes • Added 2 min ago                       │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Introduction to GraphRAG                       [🗑️] │   │
│ │ GraphRAG combines knowledge graphs with retrieval... │   │
│ │ 8,921 bytes • Added 5 min ago                        │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Technical Documentation                        [🗑️] │   │
│ │ Detailed technical specifications and architecture... │   │
│ │ 22,104 bytes • Added 10 min ago                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Build Progress (Idle)
```
┌─────────────────────────────────────────────────────────────┐
│ ⚡ Build Knowledge Graph                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Add documents above, then click the button below           │
│          to build your knowledge graph                       │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │           🚀 Build Knowledge Graph                     │ │
│ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Build Progress (Building - Chunking)
```
┌─────────────────────────────────────────────────────────────┐
│ ⚡ Build Knowledge Graph                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 📄 Chunking Documents                              67%      │
│ ┌────────────────────────────────────────────────────────┐ │
│ │████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░│ │
│ └────────────────────────────────────────────────────────┘ │
│              2 / 3 documents                                │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │               ⏳ Building...                           │ │
│ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Build Progress (Building - Entity Extraction)
```
┌─────────────────────────────────────────────────────────────┐
│ ⚡ Build Knowledge Graph                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 🔍 Extracting Entities                             45%      │
│ ┌────────────────────────────────────────────────────────┐ │
│ │████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│ └────────────────────────────────────────────────────────┘ │
│              7 / 15 chunks                                  │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │               ⏳ Building...                           │ │
│ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Build Progress (Ready)
```
┌─────────────────────────────────────────────────────────────┐
│ ⚡ Build Knowledge Graph                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │                      ✅                            │    │
│  │           Knowledge Graph Ready!                   │    │
│  │   You can now query your graph in the Query tab    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │           ✅ Graph Ready - Rebuild?                    │ │
│ └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Tab 2: Explore Graph

### Graph Statistics
```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Graph Statistics                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│ │ 📄      │ │ 🧩      │ │ 🏷️      │                        │
│ │   3     │ │   15    │ │   45    │                        │
│ │Documents│ │ Chunks  │ │Entities │                        │
│ └─────────┘ └─────────┘ └─────────┘                        │
│                                                              │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│ │ 🔗      │ │ 🧮      │ │ 📈      │                        │
│ │   90    │ │   45    │ │  50.0%  │                        │
│ │Relations│ │Embeddings│ │ Density │                        │
│ └─────────┘ └─────────┘ └─────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Graph Health
```
┌─────────────────────────────────────────────────────────────┐
│ 💚 Graph Health                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Coverage                                           100%     │
│ ┌────────────────────────────────────────────────────────┐ │
│ │████████████████████████████████████████████████████████│ │
│ └────────────────────────────────────────────────────────┘ │
│ All documents processed                                     │
│                                                              │
│ Entity Linking                                      85%     │
│ ┌────────────────────────────────────────────────────────┐ │
│ │████████████████████████████████████████████░░░░░░░░░░░│ │
│ └────────────────────────────────────────────────────────┘ │
│ Strong entity connections                                   │
│                                                              │
│ Embedding Quality                                   92%     │
│ ┌────────────────────────────────────────────────────────┐ │
│ │██████████████████████████████████████████████████░░░░░│ │
│ └────────────────────────────────────────────────────────┘ │
│ High-quality vector representations                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### System Configuration
```
┌─────────────────────────────────────────────────────────────┐
│ ⚙️ System Configuration                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Tokenizer:        rust_tokenizers                           │
│ Vocabulary:       30,522 tokens                             │
│ Embeddings:       ONNX Web                                  │
│ LLM:              WebLLM (Qwen3)                            │
│ Vector Search:    Voy k-d tree                              │
│ Storage:          IndexedDB                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Empty State (No Graph)
```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                          🏗️                                  │
│                                                              │
│                      No Graph Yet                            │
│                                                              │
│     Build a knowledge graph from your documents to           │
│          see statistics and insights                         │
│                                                              │
│         Go to [Build Graph] tab to get started              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Tab 3: Query Graph

### Query Input (Graph Not Ready)
```
┌─────────────────────────────────────────────────────────────┐
│ 💬 Query Interface                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ ⚠️ Please build a knowledge graph first before querying│ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ Enter your query                                            │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ What would you like to know?                 [DISABLED]│ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │             🔍 Search Graph            [DISABLED]      │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Query Input (Ready)
```
┌─────────────────────────────────────────────────────────────┐
│ 💬 Query Interface                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Enter your query                                            │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ What is GraphRAG and how does it work?                 │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐ │
│ │             🔍 Search Graph                            │ │
│ └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Loading State
```
┌─────────────────────────────────────────────────────────────┐
│ 📄 Results                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                                                              │
│                         ⏳                                   │
│                      [Spinner]                               │
│                         🔍                                   │
│                                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Results Display
```
┌─────────────────────────────────────────────────────────────┐
│ 📄 Results                                                   │
├─────────────────────────────────────────────────────────────┤
│ Query: "What is GraphRAG and how does it work?"             │
│                                                              │
│ ✅ Graph Search Complete                                    │
│ 📊 Searched 45 entities across 3 documents                  │
│ 🔍 Found 3 relevant entities and 5 relationships            │
│ 🧮 Used 45 embeddings for semantic matching                 │
│                                                              │
│ --- Results ---                                              │
│                                                              │
│ [Entity: "Rust Programming"]                                │
│ Relevance: 95%                                              │
│ Context: Rust is a systems programming language focused     │
│ on safety and performance...                                │
│ Source: Document 1, Chunk 3                                 │
│                                                              │
│ [Entity: "WebAssembly"]                                     │
│ Relevance: 87%                                              │
│ Context: WebAssembly enables high-performance applications  │
│ in browsers...                                              │
│ Source: Document 2, Chunk 1                                 │
│                                                              │
│ [Entity: "GraphRAG"]                                        │
│ Relevance: 82%                                              │
│ Context: GraphRAG combines knowledge graphs with            │
│ retrieval-augmented generation...                           │
│ Source: Document 1, Chunk 7                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Color Scheme

### Background
- **Primary**: Gradient from slate-900 → purple-900 → slate-900
- **Cards**: slate-800/50 with backdrop blur
- **Inputs**: slate-900 solid

### Accent Colors
- **Primary Action**: Purple-600 (#9333ea)
- **Hover**: Purple-700 (#7e22ce)
- **Success**: Green-500 (#22c55e)
- **Warning**: Yellow-500 (#eab308)
- **Error**: Red-500 (#ef4444)
- **Info**: Blue-500 (#3b82f6)

### Text
- **Primary**: White (#ffffff)
- **Secondary**: Slate-300 (#cbd5e1)
- **Muted**: Slate-400 (#94a3b8)
- **Disabled**: Slate-500 (#64748b)

### Borders
- **Default**: Slate-700 (#334155)
- **Hover**: Slate-600 (#475569)
- **Focus**: Purple-500 (#a855f7) ring

## Responsive Behavior

### Mobile (< 768px)
- Tabs stack vertically
- Stats in single column
- Reduced padding
- Full-width buttons
- Smaller text sizes

### Tablet (768px - 1023px)
- Horizontal tabs (may wrap)
- Stats in 2 columns
- Moderate padding
- Standard sizes

### Desktop (1024px+)
- Full horizontal tabs
- Stats in 3 columns
- Max width: 1280px
- Generous spacing
- Optimal readability

## Animation Details

### Progress Bar
- **Duration**: 300ms
- **Easing**: ease-out
- **Property**: width
- **Gradient**: Purple-600 → Pink-600

### Spinner
- **Duration**: 1s
- **Easing**: linear
- **Rotation**: 360°
- **Infinite**: Yes

### Hover Effects
- **Card Scale**: 1.05
- **Duration**: 200ms
- **Easing**: ease-out

### Tab Switching
- **Fade In**: 200ms
- **Content**: Instant replace
- **Tab Highlight**: 200ms slide

## Interactive States

### Button States
1. **Default**: Solid color, no interaction
2. **Hover**: Darker shade
3. **Active**: Pressed effect
4. **Disabled**: 50% opacity, no cursor
5. **Loading**: Spinner, disabled interaction

### Input States
1. **Default**: Subtle border
2. **Focus**: Purple ring, brighter border
3. **Error**: Red border, error message
4. **Disabled**: Muted colors, no interaction

### Card States
1. **Default**: Subtle border, standard opacity
2. **Hover**: Brighter border, slight scale
3. **Active**: Maintains hover state

---

This visual guide provides a comprehensive overview of the GraphRAG WASM user interface. All screens are implemented and functional in the current codebase at `/home/dio/graphrag-rs/graphrag-wasm/src/main.rs`.
