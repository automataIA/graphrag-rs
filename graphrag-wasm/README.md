# GraphRAG WASM — Browser-Native Knowledge Graph RAG

![Rust](https://img.shields.io/badge/Rust-WebAssembly-red?style=for-the-badge)
![Leptos](https://img.shields.io/badge/Leptos-reactive-orange?style=for-the-badge)

A complete GraphRAG pipeline — document ingestion, knowledge-graph build, retrieval,
and LLM synthesis — running **entirely in the browser** via WebAssembly. No server
required (an optional local llama.cpp `llama-server` backend is supported).

## Quick Start

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk

cd graphrag-wasm
trunk serve            # dev server on http://localhost:8080
trunk build --release  # release bundle in dist/
```

## The UI: a 3-column chat shell

The interface is a single **Nordic-Minimal chat shell** (no tabs, no DaisyUI — a flat
hand-written stylesheet). See [`Chat discussion.html`](../Chat%20discussion.html) for the
reference mockup the layout mirrors verbatim.

| Column | Contents |
|--------|----------|
| **LeftRail** | Brand, source documents, Flat/Hierarchy toggle, **Build** button |
| **Stage** | Active source header, the thread of question/answer turns, the composer input |
| **RightRail** | Per-query **subgraph SVG**, pipeline progress rows, mini-stats, **reference cards** |

Answers are streamed token-by-token; inline **citations** (`[1]`, `[2]`…) link to
reference cards in the RightRail. The per-query subgraph unions the entities from the
top-K retrieved chunks and lays them out with a built-in force-directed layout.

## How it works (end-to-end, in the browser)

1. **Document processing** — chunking with configurable size/overlap.
2. **Entity extraction** — rule-based / WebLLM-assisted extraction.
3. **Embeddings** — **ONNX Runtime Web** (MiniLM-L6), run **off the main thread**
   (`ort.env.wasm.proxy = true`) so the UI never blocks during inference.
4. **Knowledge graph** — in-memory entities, chunks, and relationships.
5. **Retrieval** — pure-Rust cosine similarity, top-K via `VectorIndex::search`.
6. **Synthesis** — **WebLLM** (in-browser) or **llama.cpp** (local `llama-server`); citations are
   post-processed and wired to reference cards.

Documents persist across reloads in **IndexedDB** (see `src/persist.rs`).

### What comes from `graphrag-core` vs. reimplemented here

This crate is **not a mock** — it links `graphrag-core` (path dependency, wasm-safe
feature subset) and drives a real `graphrag_core::GraphRAG` instance: document ingestion
(`add_document_from_text`), the knowledge-graph types (`Entity`, `Relationship`), Leiden
community detection, and adaptive query routing all come straight from core.

The ML hot-path stages are **reimplemented browser-side**, because core's native backends
(the native LLM client, candle, the LLM extractors) do not run inside a browser:

| Stage | Source |
|-------|--------|
| Document ingestion, graph types, Leiden, adaptive routing | **graphrag-core** |
| Embeddings | wasm-side `onnx_embedder.rs` (ONNX Runtime Web / WebGPU, hash fallback) |
| Entity extraction | wasm-side `entity_extractor.rs` (WebLLM-assisted or rule-based) |
| Vector search | wasm-side `vector_search.rs` (pure-Rust cosine) |

> Note: `src/lib.rs` also exposes a **separate** `wasm_bindgen` `GraphRAG` wrapper for
> direct JS use (`new GraphRAG(384)` + pure vector search) — distinct from
> `graphrag_core::GraphRAG` despite the shared name.

## LLM backends: WebLLM vs llama.cpp

### WebLLM (default) — in-browser via WebGPU
```js
import { UnifiedLlmClient } from './graphrag_wasm.js';
const llm = UnifiedLlmClient.withWebLLM("Phi-3-mini-4k-instruct-q4f16_1-MLC");
llm.setTemperature(0.7);
const answer = await llm.generate("What is GraphRAG?");
```
- ✅ Full privacy (no data leaves the browser), works offline after model download.
- First load downloads the model (~1–2 GB); needs a WebGPU-capable browser; small models only (1–3B).

WebLLM and ONNX inference both run in **dedicated web workers**
(`webllm-worker.js` + ORT's proxy worker), keeping main-thread blocking under ~50 ms.

### llama.cpp HTTP — local server, larger models
```js
const llm = UnifiedLlmClient.withLlamaCpp("http://127.0.0.1:8090/v1", "qwen3.5-9b");
const answer = await llm.generate("What is GraphRAG?");
```
- ✅ Support for larger models and GPU execution depends on the selected backend and device.
- Talks to the OpenAI-compatible endpoints `/v1/chat/completions` and `/v1/models`
  (`src/llama_http.rs`); no API key is sent from the browser.
- Requires a running `llama-server`. Trunk already serves the demo on port 8080, so
  start the server on another port for the browser demo:

```bash
llama-server -m Qwen3.5-9B-UD-Q5_K_XL.gguf --alias qwen3.5-9b --port 8090
```

`UnifiedLlmClient` exposes the same `generate` / `chat` / `checkAvailability` API for both
backends, so switching is a one-line change.

## Tech stack

| Component | Technology |
|-----------|-----------|
| UI | Leptos (reactive Rust) |
| Build | Trunk |
| Styling | flat Nordic-Minimal CSS (`tailwind.css`, no `@tailwind` directives) |
| Tokenizer | HuggingFace `tokenizers` (`unstable_wasm`) |
| Embeddings | ONNX Runtime Web (off-main-thread, optional WebGPU) |
| LLM | WebLLM (in-browser) or llama.cpp HTTP |
| Vector search | pure Rust (cosine similarity) |
| Storage | IndexedDB |

## Project layout

```
graphrag-wasm/
├── src/
│   ├── main.rs                 # chat-shell UI (LeftRail / Stage / RightRail)
│   ├── components/
│   │   ├── chat_shell.rs       # data types, citation parser, subgraph builder
│   │   └── force_layout.rs     # force-directed subgraph layout
│   ├── webllm.rs               # WebLLM client (+ web-worker engine)
│   ├── llama_http.rs           # llama.cpp (OpenAI-compatible) HTTP client
│   ├── llm_provider.rs         # UnifiedLlmClient abstraction
│   ├── onnx_embedder.rs        # ONNX Runtime Web embeddings
│   ├── vector_search.rs        # cosine similarity
│   └── persist.rs              # IndexedDB persistence
├── webllm-worker.js            # WebWorker MLC engine handler
├── index.html                  # entry point + ORT/WebLLM worker wiring
├── tailwind.css                # flat stylesheet
└── Trunk.toml                  # build config
```

## Browser support

Chrome/Edge 87+, Firefox 89+, Safari 15.2+ (incl. mobile). Requires WebAssembly +
ES2020 modules; WebGPU is optional (accelerates embeddings/WebLLM when present).

## Tests

A Playwright parity test (`tests/playwright/chat_layout.sh`) asserts the WASM SPA matches
the mockup on 19 shared selectors. Unit tests:

```bash
cargo test --target wasm32-unknown-unknown
```

## License

See the main repository LICENSE.
