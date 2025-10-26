# Runtime Testing Guide - GraphRAG Leptos Demo

> Guida completa per testare l'applicazione Leptos + ONNX + GraphRAG nel browser

## 📋 Indice

1. [Prerequisiti](#prerequisiti)
2. [Download Modello ONNX](#download-modello-onnx)
3. [Setup Browser](#setup-browser)
4. [Avvio Ambiente Sviluppo](#avvio-ambiente-sviluppo)
5. [Test Funzionalità](#test-funzionalità)
6. [Debugging](#debugging)
7. [Performance Testing](#performance-testing)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisiti

### Software Richiesto

```bash
# 1. Rust + wasm32 target
rustup target add wasm32-unknown-unknown

# 2. Trunk (build tool per WASM)
cargo install trunk

# 3. wasm-bindgen-cli (per debugging)
cargo install wasm-bindgen-cli
```

### Browser Compatibili

**Raccomandati per WebGPU:**
- ✅ **Chrome 121+** (Windows/Mac/Linux/ChromeOS)
- ✅ **Edge 122+** (Windows/Mac)
- ⚠️ **Firefox 127+** (dietro flag: `dom.webgpu.enabled`)
- ⚠️ **Safari Technology Preview** (sperimentale)

**Verifica WebGPU:**
```javascript
// Apri Console DevTools (F12) e testa:
navigator.gpu !== undefined
// true = WebGPU disponibile
```

---

## 📥 Download Modello ONNX

### Opzione 1: HuggingFace (Consigliato)

**Modello consigliato:** `Xenova/all-MiniLM-L6-v2`

```bash
cd examples/graphrag-leptos-demo

# Crea directory modelli
mkdir -p models

# Scarica modello ONNX da HuggingFace
# Opzione A: Git LFS (se installato)
git lfs install
git clone https://huggingface.co/Xenova/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2-temp

# Copia solo i file necessari
cp models/all-MiniLM-L6-v2-temp/onnx/model.onnx models/all-MiniLM-L6-v2.onnx
cp models/all-MiniLM-L6-v2-temp/tokenizer.json models/

# Rimuovi directory temporanea
rm -rf models/all-MiniLM-L6-v2-temp
```

**Opzione B: Download manuale**

1. Vai a https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main
2. Scarica `onnx/model.onnx` → salva come `models/all-MiniLM-L6-v2.onnx`
3. Scarica `tokenizer.json` → salva in `models/tokenizer.json`

**Opzione C: Modello ottimizzato (più veloce)**

```bash
# Download da repository ottimizzato per ONNX Runtime Web
wget -P models/ https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx
mv models/model.onnx models/all-MiniLM-L6-v2.onnx
```

### Verifica Download

```bash
# Controlla dimensione file (~90MB)
ls -lh models/all-MiniLM-L6-v2.onnx

# Output atteso:
# -rw-r--r-- 1 user user 90M Oct  3 10:00 all-MiniLM-L6-v2.onnx
```

---

## 🌐 Setup Browser

### Chrome/Edge (Consigliato)

**1. Verifica versione**
```
chrome://version/
# Richiesto: Chrome 121+ o Edge 122+
```

**2. Abilita WebGPU (se necessario)**
```
chrome://flags/#enable-unsafe-webgpu
# Imposta su "Enabled"
# Riavvia browser
```

**3. Verifica funzionamento**
```
chrome://gpu/
# Cerca "WebGPU: Hardware accelerated"
```

### Firefox (Sperimentale)

**1. Abilita WebGPU**
```
about:config
# Cerca: dom.webgpu.enabled
# Imposta: true
```

**2. Abilita SharedArrayBuffer (per WASM threads)**
```
about:config
# Cerca: javascript.options.shared_memory
# Imposta: true
```

### Safari Technology Preview

WebGPU è disponibile ma sperimentale. Non consigliato per testing primario.

---

## 🚀 Avvio Ambiente Sviluppo

### Build & Serve Veloce

```bash
cd examples/graphrag-leptos-demo

# Opzione 1: Serve con auto-reload (sviluppo)
trunk serve --open

# Opzione 2: Serve su porta specifica
trunk serve --port 8080

# Opzione 3: Build + serve manuale
trunk build
python -m http.server 8080 -d dist/
```

### Build di Produzione

```bash
# Build ottimizzata
trunk build --release

# Output in dist/ (pronto per deploy)
ls -la dist/
```

### Logs Utili

**Terminal output:**
```
Finished dev [unoptimized + debuginfo] target(s) in 2.34s
📦 success

📡 Server listening at http://127.0.0.1:8080
```

**Browser Console (F12):**
```
GraphRAG Leptos Demo starting...
GraphRAG WASM initialized
```

---

## ✅ Test Funzionalità

### Test Checklist Completa

#### 1. **Inizializzazione App** ⏱️ ~2-3 secondi

**Cosa testare:**
- [ ] Pagina carica senza errori 404
- [ ] Nessun errore nella Console (F12 → Console)
- [ ] UI rendering completo (header, chat, sidebar)
- [ ] CSS/TailwindCSS applicato correttamente

**Browser Console dovrebbe mostrare:**
```javascript
GraphRAG Leptos Demo starting...
GraphRAG WASM initialized
```

**❌ Errori comuni:**
```
Failed to load WASM module
→ Verifica compilazione WASM
→ Controlla Network tab per 404

Uncaught ReferenceError: ort is not defined
→ ONNX Runtime CDN non caricato
→ Controlla index.html script tag
```

---

#### 2. **ONNX Runtime CDN Loading** ⏱️ ~1-2 secondi

**Cosa testare:**
- [ ] Script CDN caricato (Network tab)
- [ ] Oggetto `window.ort` disponibile
- [ ] Nessun CORS error

**Test manuale in Console:**
```javascript
// Verifica ONNX Runtime
console.log(window.ort)
// Output: {InferenceSession: ƒ, Tensor: ƒ, ...}

// Verifica versione
console.log(ort.env.wasm.numThreads)
// Output: 4 (o numero threads CPU)
```

**❌ Errori comuni:**
```
CORS error loading ort.min.js
→ Usa trunk serve (non file://)
→ Controlla CDN URL in index.html
```

---

#### 3. **Caricamento Modello ONNX** ⏱️ ~3-5 secondi (GPU) / ~10-15 secondi (CPU)

**Cosa testare:**
- [ ] Modello .onnx caricato da `./models/`
- [ ] WebGPU o WASM backend attivato
- [ ] Status message: "Ready! Add documents or ask questions."

**Browser Console dovrebbe mostrare:**
```javascript
Loading ONNX model from: ./models/all-MiniLM-L6-v2.onnx
✅ ONNX model loaded with WebGPU
WasmOnnxEmbedder initialized
```

**Test manuale:**
1. Apri Network tab (F12)
2. Filtra per `all-MiniLM-L6-v2.onnx`
3. Verifica Status: 200 OK
4. Verifica Size: ~90MB

**Performance check:**
```javascript
// Console: controlla ExecutionProvider
// Output atteso: "webgpu" (veloce) o "wasm" (fallback)
```

**❌ Errori comuni:**
```
Failed to load model: 404
→ Modello non in ./models/all-MiniLM-L6-v2.onnx
→ Verifica path in src/main.rs:61

WebGPU not available
→ Browser non supporta WebGPU
→ Fallback su CPU (più lento ma funzionante)

Out of memory
→ Modello troppo grande per dispositivo
→ Usa modello quantizzato (INT8)
```

---

#### 4. **Document Upload & Embedding** ⏱️ ~100ms per documento (GPU) / ~1s (CPU)

**Cosa testare:**
- [ ] File upload funziona
- [ ] Embeddings generati per ogni documento
- [ ] Index building completato
- [ ] Document count aggiornato

**Procedura test:**
```
1. Click "Upload Documents"
2. Seleziona 3-5 file di testo (.txt)
3. Osserva progress in status message
4. Verifica "Indexed X documents" al completamento
```

**Browser Console dovrebbe mostrare:**
```javascript
Processing 5 documents...
✅ Added: document1.txt
✅ Added: document2.txt
...
✅ Voy index built successfully: 5 documents
```

**Performance benchmark:**
```javascript
// Per 10 documenti:
// WebGPU: ~1-2 secondi totali
// CPU:    ~10-15 secondi totali
```

**❌ Errori comuni:**
```
Embedding generation failed
→ Modello ONNX non caricato
→ Verifica step 3

Index build failed
→ Voy non disponibile
→ Controlla Voy CDN in index.html
```

---

#### 5. **Query Processing** ⏱️ ~50-100ms (GPU) / ~500ms (CPU)

**Cosa testare:**
- [ ] Query embedding generata
- [ ] Vector search funziona
- [ ] Top-k risultati ritornati
- [ ] Similarità calcolata correttamente

**Procedura test:**
```
1. Dopo upload documenti (step 4)
2. Digita query: "what is machine learning?"
3. Premi Enter o click "Send"
4. Osserva risultati in console
```

**Browser Console dovrebbe mostrare:**
```javascript
Processing: what is machine learning?
Using Voy k-d tree for fast search ⚡
✅ Query results: [{"id":2,"similarity":0.87,...}]
```

**Verifica qualità risultati:**
```javascript
// Similarità dovrebbe essere 0.0-1.0
// Risultati ordinati per similarity (descending)
// Top risultato dovrebbe essere rilevante
```

**❌ Errori comuni:**
```
Query failed: No embeddings to index
→ Prima carica documenti (step 4)

Search failed: Voy index not found
→ Rebuild index dopo upload documenti
```

---

#### 6. **Graph Visualization** ⏱️ ~100ms rendering

**Cosa testare:**
- [ ] SVG canvas renderizzato
- [ ] Nodi e edges visualizzati
- [ ] Force-directed layout applicato
- [ ] Interazioni (hover, click) funzionano

**Procedura test:**
```
1. Scroll in fondo alla pagina
2. Osserva grafo (se dati presenti)
3. Hover su nodi (highlight)
4. Click su nodi (selection)
5. Zoom in/out con scroll
```

**❌ Note:**
```
Graph potrebbe essere vuoto se:
- Nessun documento caricato
- Nessuna entità estratta
- Feature non ancora implementata
```

---

## 🐛 Debugging

### Chrome DevTools Setup

**1. Abilita WASM Debugging**
```
1. Apri DevTools (F12)
2. Settings (⚙️) → Experiments
3. Enable "WebAssembly Debugging: Enable DWARF support"
4. Riavvia DevTools
```

**2. Source Maps**

Trunk genera automaticamente source maps. Verifica in:
```
Sources tab → wasm://wasm/
→ Dovresti vedere file .rs originali
```

**3. Breakpoints**

```
1. Sources tab
2. Cerca file: graphrag_leptos_demo_bg.wasm
3. Espandi → trova funzione Rust
4. Click numero linea per breakpoint
```

### Console Logging

**Rust side:**
```rust
web_sys::console::log_1(&"Debug message".into());
web_sys::console::error_1(&format!("Error: {:?}", err).into());
```

**Performance timing:**
```rust
let window = web_sys::window().unwrap();
let performance = window.performance().unwrap();
let start = performance.now();

// ... operazione ...

let duration = performance.now() - start;
web_sys::console::log_1(&format!("Took {}ms", duration).into());
```

### Network Debugging

**ONNX Model Loading:**
```
1. Network tab → Filter: onnx
2. Click request → Headers
3. Verifica:
   - Status: 200
   - Content-Type: application/octet-stream
   - Size: ~90MB
4. Timing tab: download time
```

**WebGPU Shader Compilation:**
```
chrome://gpu-internals/
→ WebGPU internals
→ Shader compilation logs
```

### Memory Profiling

**1. Chrome Memory Profiler**
```
1. DevTools → Memory tab
2. Take heap snapshot
3. Filtra per "wasm"
4. Cerca memory leaks
```

**2. Performance Monitor**
```
DevTools → Performance Monitor
→ JS heap size (should be < 500MB)
→ WebAssembly memory (model ~90MB + buffers)
```

**Limiti attesi:**
```
- JS Heap: ~100-200 MB
- WASM Memory: ~150-250 MB (include modello)
- Total: ~300-450 MB (accettabile)
```

---

## ⚡ Performance Testing

### Embedding Benchmarks

**Test script in Console:**
```javascript
// Test embedding performance
async function benchmarkEmbedding() {
  const texts = ["test sentence " + i for i in range(100)];
  const start = performance.now();

  for (let text of texts) {
    // Trigger embedding via UI
  }

  const duration = performance.now() - start;
  console.log(`100 embeddings: ${duration}ms`);
  console.log(`Average: ${duration/100}ms per embedding`);
}
```

**Valori attesi:**

| Backend | Per Embedding | 100 Embeddings |
|---------|---------------|----------------|
| WebGPU  | 3-10ms        | 0.3-1s         |
| WASM    | 50-100ms      | 5-10s          |
| CPU     | 100-200ms     | 10-20s         |

### Vector Search Benchmarks

**Test k-d tree performance:**
```javascript
// Test vector search
async function benchmarkSearch() {
  // Assume 1000 documents indexed
  const start = performance.now();

  for (let i = 0; i < 100; i++) {
    // Trigger query via UI
  }

  const duration = performance.now() - start;
  console.log(`100 searches: ${duration}ms`);
}
```

**Valori attesi (Voy k-d tree):**

| Documents | Per Query | 100 Queries |
|-----------|-----------|-------------|
| 1K        | 1-5ms     | 100-500ms   |
| 10K       | 5-15ms    | 0.5-1.5s    |
| 100K      | 15-50ms   | 1.5-5s      |

### Page Load Performance

**Lighthouse Audit:**
```
1. DevTools → Lighthouse tab
2. Mode: Navigation
3. Categories: Performance
4. Click "Analyze page load"
```

**Target scores:**
- Performance: > 80
- First Contentful Paint: < 1.5s
- Time to Interactive: < 3s
- Total Bundle Size: < 2MB (excluding ONNX model)

---

## 🔧 Troubleshooting

### Errore: "ONNX Runtime not found"

**Causa:** CDN script non caricato

**Soluzione:**
```html
<!-- Verifica in index.html -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
```

**Test:**
```javascript
console.log(window.ort); // Should not be undefined
```

---

### Errore: "Model not found: 404"

**Causa:** File ONNX non in posizione corretta

**Soluzione:**
```bash
# Verifica path
ls examples/graphrag-leptos-demo/models/all-MiniLM-L6-v2.onnx

# Oppure copia in public/
mkdir -p public/models/
cp models/all-MiniLM-L6-v2.onnx public/models/
```

**Aggiorna path in src/main.rs:**
```rust
emb.load_model("./models/all-MiniLM-L6-v2.onnx", Some(true)).await
// oppure
emb.load_model("/models/all-MiniLM-L6-v2.onnx", Some(true)).await
```

---

### Errore: "WebGPU not available"

**Causa:** Browser non supporta WebGPU

**Soluzione 1: Upgrade browser**
```
Chrome 121+ o Edge 122+
```

**Soluzione 2: Fallback CPU**
```rust
// App continua a funzionare, ma più lento
// ONNX Runtime usa WASM backend automaticamente
```

**Test:**
```javascript
navigator.gpu !== undefined // false = usa CPU fallback
```

---

### Errore: "Out of memory"

**Causa:** Dispositivo con poca RAM

**Soluzione 1: Modello quantizzato**
```bash
# Usa modello INT8 (più piccolo)
# ~30MB invece di 90MB
wget https://huggingface.co/Xenova/all-MiniLM-L6-v2-quantized/resolve/main/model_quantized.onnx
```

**Soluzione 2: Ridurre batch size**
```rust
// Processa documenti uno alla volta invece di batch
for doc in documents {
    process_single(doc).await;
}
```

---

### Performance Lenta

**Diagnosi:**
```javascript
// Console
console.log(navigator.gpu ? "WebGPU" : "CPU fallback");
```

**Soluzioni:**

1. **Se CPU fallback:**
   - Abilita WebGPU in browser
   - Verifica `chrome://gpu/`

2. **Se WebGPU ma lento:**
   - Chiudi altre tab (libera GPU)
   - Disattiva estensioni browser
   - Verifica driver GPU aggiornati

3. **Network lento:**
   - Cache CDN: `trunk serve --release`
   - Hosting locale modello

---

### Voy Index Errors

**Errore:** "Voy not found"

**Soluzione:**
```html
<!-- Aggiungi a index.html -->
<script type="module">
  import { Voy } from "https://cdn.jsdelivr.net/npm/voy-search@0.6.3/dist/voy.js";
  window.Voy = Voy;
</script>
```

**Test:**
```javascript
console.log(window.Voy); // Should be defined
```

---

## 📊 Success Criteria

### Checklist Completo

- [ ] ✅ App carica in < 3 secondi
- [ ] ✅ Modello ONNX carica in < 10 secondi
- [ ] ✅ WebGPU attivato (o CPU fallback funzionante)
- [ ] ✅ Document upload funziona
- [ ] ✅ Embeddings generati correttamente
- [ ] ✅ Vector search ritorna risultati
- [ ] ✅ Similarità scores hanno senso (0.0-1.0)
- [ ] ✅ UI responsive e senza crash
- [ ] ✅ Console senza errori critici
- [ ] ✅ Memory usage < 500MB
- [ ] ✅ Lighthouse Performance > 80

---

## 🎯 Next Steps

Dopo test runtime completo:

1. **Ottimizzazione**
   - Quantizzare modello (INT8)
   - Code splitting con Trunk
   - Lazy loading componenti

2. **Testing Automatizzato**
   - wasm-bindgen-test per unit tests
   - Playwright per E2E tests
   - GitHub Actions CI/CD

3. **Production Deployment**
   - Build release: `trunk build --release`
   - Deploy su Netlify/Vercel/CloudFlare Pages
   - CDN per asset statici

---

## 📚 Risorse Utili

### Documentazione

- [Leptos Book](https://book.leptos.dev/)
- [ONNX Runtime Web Docs](https://onnxruntime.ai/docs/tutorials/web/)
- [Trunk Guide](https://trunkrs.dev/)
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)

### Modelli ONNX

- [HuggingFace ONNX Models](https://huggingface.co/models?library=onnx)
- [Xenova Transformers.js](https://huggingface.co/Xenova)

### Browser Tools

- [Chrome DevTools WASM](https://developer.chrome.com/docs/devtools/wasm)
- [WebGPU Samples](https://webgpu.github.io/webgpu-samples/)

---

**Creato:** 2025-10-03
**Versione:** 1.0
**Autore:** Claude Code Assistant
