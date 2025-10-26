# WebLLM Integration for GraphRAG WASM

This document describes the WebLLM integration for natural language answer synthesis in the GraphRAG WASM application.

## Overview

WebLLM provides browser-based Large Language Model inference using WebGPU acceleration. This enables the GraphRAG pipeline to synthesize natural language answers from retrieved context entirely client-side, with no server required.

## Architecture

```
User Query
    ↓
Vector Search (ONNX embeddings)
    ↓
Context Retrieval (chunks + entities + relationships)
    ↓
**WebLLM Synthesis** ← You are here
    ↓
Natural Language Answer
```

## Implementation

### 1. WebLLM Setup (index.html)

WebLLM is loaded via CDN as an ES module:

```html
<!-- WebLLM Integration for GPU-Accelerated LLM -->
<script type="module">
    import * as webllm from "https://esm.run/@mlc-ai/web-llm";
    window.webllm = webllm;
    console.log("✅ WebLLM loaded successfully");
</script>
```

### 2. Rust Bindings (src/webllm.rs)

The `webllm.rs` module provides comprehensive Rust bindings to the JavaScript WebLLM library:

#### Key Components

**WebLLM Struct**:
```rust
pub struct WebLLM {
    engine: JsValue,
    model_id: String,
}
```

**Chat Message**:
```rust
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self
    pub fn system(content: impl Into<String>) -> Self
    pub fn assistant(content: impl Into<String>) -> Self
}
```

**Main API**:
```rust
// Initialize with a model
WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await?

// Chat with messages
llm.chat(messages, temperature, max_tokens).await?

// Simple query
llm.ask("Your question").await?

// Streaming (real-time token generation)
llm.chat_stream(messages, |chunk| { ... }, temp, tokens).await?
```

### 3. Query Pipeline Integration (src/main.rs)

The synthesis happens in the query handler after vector search retrieval:

```rust
// 1. Perform vector search
let vector_results = /* ... vector search ... */;

// 2. Build context from retrieved chunks
let mut context_for_llm = String::new();
for (chunk_id, _similarity) in vector_results.iter() {
    if let Some(chunk) = graphrag.get_chunk(chunk_id) {
        context_for_llm.push_str(&chunk.content);

        // Add entity information
        for entity_id in chunk.entities.iter() {
            if let Some(entity) = graphrag.get_entity(&entity_id.0) {
                context_for_llm.push_str(&format!(
                    "[Entity: {} - {}] ",
                    entity.entity_type, entity.name
                ));
            }
        }
    }
}

// 3. Synthesize answer with WebLLM
match webllm::WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC").await {
    Ok(llm) => {
        let messages = vec![
            webllm::ChatMessage::system(
                "You are a helpful AI assistant that answers questions based on provided context..."
            ),
            webllm::ChatMessage::user(
                format!("Question: {}\\n\\nContext: {}...", query, context)
            ),
        ];

        match llm.chat(messages, Some(0.7), Some(512)).await {
            Ok(answer) => {
                // Display synthesized answer
                result_text.push_str(&format!("💬 Synthesized Answer:\\n{}\\n", answer));
            }
            Err(e) => {
                // Handle error
            }
        }
    }
    Err(e) => {
        // WebLLM not available (e.g., no WebGPU)
    }
}
```

## Supported Models

WebLLM supports various quantized models optimized for browser inference:

### Recommended Models

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| **Phi-3-mini-4k-instruct-q4f16_1-MLC** | 2.4GB | 40 tok/s | Balanced (default) |
| Llama-3.2-1B-Instruct-q4f16_1-MLC | 1.2GB | 62 tok/s | Fast responses |
| Qwen2-1.5B-Instruct-q4f16_1-MLC | 1.6GB | 50 tok/s | Multilingual |
| gemma-2b-it-q4f16_1-MLC | 2.0GB | 45 tok/s | General purpose |

### Changing the Model

To use a different model, update the model ID in `src/main.rs`:

```rust
// Change from:
webllm::WebLLM::new("Phi-3-mini-4k-instruct-q4f16_1-MLC")

// To:
webllm::WebLLM::new("Llama-3.2-1B-Instruct-q4f16_1-MLC")
```

## Performance

### WebGPU Acceleration

WebLLM uses WebGPU for GPU-accelerated inference:

- **With WebGPU**: 40-62 tokens/second (depending on model and GPU)
- **Without WebGPU**: Falls back to WebAssembly SIMD (~5-10 tokens/second)

### Browser Support

WebGPU is required for optimal performance:

| Browser | WebGPU Support | Performance |
|---------|----------------|-------------|
| Chrome 113+ | ✅ Stable | Excellent |
| Edge 113+ | ✅ Stable | Excellent |
| Firefox 118+ | ⚠️ Experimental | Good (enable flag) |
| Safari 17+ | ⚠️ Preview | Good (enable flag) |

### Enabling WebGPU in Firefox

```
1. Open about:config
2. Set dom.webgpu.enabled = true
3. Restart browser
```

### Enabling WebGPU in Safari

```
1. Safari → Preferences → Advanced
2. Check "Show Develop menu in menu bar"
3. Develop → Experimental Features → WebGPU
4. Restart browser
```

## Error Handling

The implementation includes comprehensive error handling:

### Common Errors

#### 1. WebLLM Not Loaded
```
Error: "WebLLM not loaded. Add <script> tag to index.html"
```
**Solution**: Ensure the WebLLM script is present in `index.html`

#### 2. WebGPU Not Available
```
Warning: "Failed to create WebGPU Context Provider"
```
**Solution**:
- Check browser WebGPU support
- Enable WebGPU flags (Firefox/Safari)
- Falls back to WASM SIMD (slower but functional)

#### 3. Model Loading Failed
```
Error: "Initialization failed: ..."
```
**Solution**:
- Check internet connection (models download on first use)
- Verify model ID is correct
- Check browser console for detailed error

### Graceful Degradation

If WebLLM fails to initialize, the system gracefully falls back:

```
1. Attempt WebLLM initialization
2. If fails → Show warning message
3. Display retrieved context instead
4. User still gets relevant information from knowledge graph
```

## Configuration

### Synthesis Parameters

You can customize the LLM synthesis behavior:

```rust
llm.chat(
    messages,
    Some(0.7),    // temperature: 0.0 = deterministic, 1.0 = creative
    Some(512)     // max_tokens: maximum length of generated answer
).await?
```

**Recommended Values**:
- **Factual Answers**: `temperature: 0.3, max_tokens: 256`
- **Balanced (default)**: `temperature: 0.7, max_tokens: 512`
- **Creative Responses**: `temperature: 0.9, max_tokens: 1024`

### System Prompt

The system prompt guides the LLM's behavior:

```rust
ChatMessage::system(
    "You are a helpful AI assistant that answers questions based on provided context from a knowledge graph. \
    Be concise, accurate, and cite specific entities and relationships when relevant. \
    If the context doesn't contain enough information, say so clearly."
)
```

**Customization Tips**:
- Add domain-specific instructions (e.g., "Focus on legal precedents")
- Specify answer format (e.g., "Answer in bullet points")
- Set tone (e.g., "Use simple language suitable for beginners")

## Progressive Enhancement

The WebLLM integration follows progressive enhancement principles:

### Level 1: Basic Retrieval (Always Available)
- Vector search finds relevant chunks
- Entities and relationships displayed
- User gets structured information

### Level 2: Context Enhancement (Always Available)
- Chunks combined with entity information
- Relationships between entities shown
- Rich contextual display

### Level 3: LLM Synthesis (When WebLLM Available)
- Natural language answer generated
- Context synthesized into coherent response
- Best user experience

## Testing

### Manual Testing

1. **Load the application**:
   ```bash
   cd graphrag-wasm
   trunk serve
   ```

2. **Open browser console** to see WebLLM initialization

3. **Load Symposium demo** to build knowledge graph

4. **Submit a query** (e.g., "What does Socrates say about love?")

5. **Verify synthesis**:
   - Check for "🤖 Synthesizing natural language answer..."
   - Verify "💬 Synthesized Answer:" section appears
   - Confirm answer is relevant and well-formed

### Browser Console Monitoring

Expected console logs:
```
✅ WebLLM loaded successfully
✅ ONNX Runtime Web loaded
🔍 Extracting entities with WebLLM (Qwen)...
⚠️  WebLLM not available, using simple rule-based extraction
// ... (entity extraction uses fallback)

// During query:
[WebLLM Rust] Initializing with model: Phi-3-mini-4k-instruct-q4f16_1-MLC
[WebLLM] Engine initialized successfully
[WebLLM Rust] Synthesizing answer for: What does Socrates say about love?
✅ LLM synthesis successful: 245 chars
```

### Error Scenarios

Test these scenarios to verify graceful degradation:

1. **No WebGPU**: Verify fallback to WASM SIMD
2. **Network offline**: Verify model caching works
3. **Model load failure**: Verify error message shown
4. **Synthesis timeout**: Verify graceful handling

## Future Improvements

### Planned Features

1. **Model Selection UI**
   - Allow users to choose model based on speed/quality tradeoff
   - Show download progress for first-time model loading

2. **Streaming Responses**
   - Display tokens as they're generated
   - Better UX for long answers

3. **Answer Caching**
   - Cache synthesized answers for repeated queries
   - Reduce redundant LLM calls

4. **Custom System Prompts**
   - User-configurable prompt templates
   - Domain-specific answer styles

5. **Citation Enhancement**
   - Automatic source attribution in answers
   - Link answers back to specific entities/chunks

## Troubleshooting

### Issue: WebLLM loads but initialization fails

**Symptoms**: Console shows "✅ WebLLM loaded" but synthesis doesn't work

**Solutions**:
1. Check browser DevTools → Network tab for model download
2. Verify WebGPU is enabled and working
3. Try a smaller model (e.g., Llama-3.2-1B)
4. Clear browser cache and reload

### Issue: Slow synthesis (>30 seconds)

**Symptoms**: Synthesis works but takes very long

**Solutions**:
1. Enable WebGPU (check browser flags)
2. Use a smaller model
3. Reduce `max_tokens` parameter
4. Check GPU is not being used by other applications

### Issue: Out of memory errors

**Symptoms**: Browser tab crashes during synthesis

**Solutions**:
1. Close other browser tabs
2. Use a smaller model
3. Reduce context size (limit chunks to top 3 instead of 5)
4. Increase browser memory limit (if available)

## References

- [WebLLM Documentation](https://webllm.mlc.ai/)
- [WebLLM GitHub](https://github.com/mlc-ai/web-llm)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [Browser WebGPU Support](https://caniuse.com/webgpu)
- [Model Performance Benchmarks](https://webllm.mlc.ai/#benchmarks)

## License

This integration uses WebLLM under Apache License 2.0.
Model licenses vary by model - check individual model licenses before use.
