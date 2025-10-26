# Ollama HTTP Integration for GraphRAG WASM

This document explains how to use Ollama as an alternative LLM backend for GraphRAG WASM.

## Overview

GraphRAG WASM now supports **two LLM providers**:

1. **WebLLM** - 100% in-browser inference via WebGPU (default)
2. **Ollama HTTP** - Local Ollama server via REST API (alternative)

## Why Ollama HTTP?

While WebLLM provides complete privacy and offline functionality, Ollama HTTP offers:

- **Larger models**: Use 7B, 13B, or even 70B parameter models
- **Better quality**: State-of-the-art models like Llama 3.1, Qwen 2.5, Mistral
- **Full GPU**: Utilize your full GPU via CUDA or Metal
- **Older browsers**: Works without WebGPU support

## Prerequisites

### Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download installer from https://ollama.com/download

### Pull a Model

```bash
# Recommended models for GraphRAG
ollama pull llama3.1:8b      # Best balance (4.7GB)
ollama pull qwen2.5:7b       # Excellent for reasoning (4.7GB)
ollama pull mistral:7b       # Fast and efficient (4.1GB)

# Larger models (requires more VRAM)
ollama pull llama3.1:70b     # Best quality (40GB)
ollama pull qwen2.5:32b      # Advanced reasoning (19GB)
```

### Start Ollama Server with CORS

**Important:** You must enable CORS for the browser to access Ollama:

```bash
# Enable CORS for localhost:8080 (default Trunk port)
OLLAMA_ORIGINS="http://localhost:8080" ollama serve
```

**For custom ports:**
```bash
# If your WASM app runs on a different port
OLLAMA_ORIGINS="http://localhost:3000,http://localhost:8080" ollama serve
```

## Usage

### JavaScript API

```javascript
import init, { UnifiedLlmClient } from './graphrag_wasm.js';

// Initialize WASM
await init();

// Create Ollama HTTP client
const llm = UnifiedLlmClient.withOllama(
  "http://localhost:11434",  // Ollama endpoint
  "llama3.1:8b"              // Model name
);

// Configure generation parameters
llm.setTemperature(0.7);      // Creativity (0.0-1.0)
llm.setSystemPrompt("You are a helpful assistant for GraphRAG queries.");

// Generate response
const answer = await llm.generate("What is knowledge graph RAG?");
console.log(answer);

// Chat-style interaction
const chatAnswer = await llm.chat("Explain it in simpler terms");
console.log(chatAnswer);

// Check if Ollama is available
const available = await llm.checkAvailability();
if (!available) {
  console.error("Ollama server is not running!");
}
```

### Rust API

```rust
use graphrag_wasm::{UnifiedLlmClient, LlmProviderType};

// Create Ollama HTTP client
let mut llm = UnifiedLlmClient::with_ollama(
    "http://localhost:11434".to_string(),
    "llama3.1:8b".to_string(),
);

// Configure
llm.set_temperature(0.7);
llm.set_system_prompt("You are a helpful assistant.".to_string());

// Generate
let answer = llm.generate("What is GraphRAG?".to_string()).await?;
```

## Model Recommendations

### For GraphRAG Answer Generation

| Model | Size | Best For | Performance |
|-------|------|----------|-------------|
| **llama3.1:8b** | 4.7GB | General queries, summaries | 40-50 tok/s |
| **qwen2.5:7b** | 4.7GB | Complex reasoning, analysis | 35-45 tok/s |
| **mistral:7b** | 4.1GB | Fast responses, simple tasks | 50-60 tok/s |
| **llama3.1:70b** | 40GB | Highest quality, research | 10-15 tok/s |

### For Entity Extraction

| Model | Size | Best For | Performance |
|-------|------|----------|-------------|
| **qwen2.5:7b** | 4.7GB | Structured extraction | 35-45 tok/s |
| **llama3.1:8b** | 4.7GB | Named entity recognition | 40-50 tok/s |

## Performance Comparison

### WebLLM vs Ollama HTTP

| Metric | WebLLM | Ollama HTTP |
|--------|--------|-------------|
| **Model Size** | 1-3B params | 7-70B params |
| **First Load** | 1-2GB download | Model already local |
| **Inference Speed** | 40-62 tok/s (WebGPU) | 30-60 tok/s (GPU) |
| **Quality** | Good (3B) | Excellent (7B+) |
| **Privacy** | 100% local | Local server |
| **Setup** | None | Install Ollama |
| **Browser Support** | Chrome 113+, Edge 113+ | All browsers |

## Switching Between Providers

The `UnifiedLlmClient` provides a unified interface:

```javascript
// Create both clients
const webllm = UnifiedLlmClient.withWebLLM("Phi-3-mini-4k-instruct-q4f16_1-MLC");
const ollama = UnifiedLlmClient.withOllama("http://localhost:11434", "llama3.1:8b");

// Both have identical API
async function generateAnswer(llm, prompt) {
  return await llm.generate(prompt);
}

// Use either provider
const answer1 = await generateAnswer(webllm, "What is GraphRAG?");
const answer2 = await generateAnswer(ollama, "What is GraphRAG?");
```

## Troubleshooting

### Error: Failed to fetch from Ollama

**Cause:** CORS not enabled or Ollama not running

**Solution:**
```bash
# Check if Ollama is running
ollama list

# Restart with CORS
OLLAMA_ORIGINS="http://localhost:8080" ollama serve
```

### Error: Model not found

**Cause:** Model not pulled

**Solution:**
```bash
# Pull the model
ollama pull llama3.1:8b

# Verify it's available
ollama list
```

### Error: Connection refused

**Cause:** Ollama server not running

**Solution:**
```bash
# Start Ollama server
ollama serve
```

## Advanced Configuration

### Custom Endpoints

```javascript
// Use custom Ollama endpoint
const llm = UnifiedLlmClient.withOllama(
  "http://192.168.1.100:11434",  // Remote Ollama server
  "llama3.1:8b"
);
```

### Temperature Control

```javascript
// Low temperature (0.0-0.3): More deterministic, factual
llm.setTemperature(0.2);

// Medium temperature (0.4-0.7): Balanced creativity
llm.setTemperature(0.7);

// High temperature (0.8-1.0): More creative, varied
llm.setTemperature(0.9);
```

### System Prompts

```javascript
// For factual answers
llm.setSystemPrompt(
  "You are a precise assistant that provides factual answers based on the provided context."
);

// For creative summaries
llm.setSystemPrompt(
  "You are a creative writer who synthesizes information into engaging summaries."
);

// For technical analysis
llm.setSystemPrompt(
  "You are a technical expert who analyzes complex systems and relationships."
);
```

## API Reference

### UnifiedLlmClient

```rust
// Constructor methods
UnifiedLlmClient::new() -> Self
UnifiedLlmClient::with_webllm(model: String) -> Self
UnifiedLlmClient::with_ollama(endpoint: String, model: String) -> Self

// Configuration
set_temperature(&mut self, temperature: f32)
set_system_prompt(&mut self, prompt: String)

// Generation
generate(&mut self, prompt: String) -> Result<String, JsValue>
chat(&mut self, message: String) -> Result<String, JsValue>

// Utility
get_provider(&self) -> String
get_model(&self) -> String
check_availability(&self) -> Result<bool, JsValue>
```

### OllamaHttpClient

```rust
// Constructor
OllamaHttpClient::new() -> Self
OllamaHttpClient::with_config(endpoint: String, model: String) -> Self

// Configuration
set_model(&mut self, model: String)
set_temperature(&mut self, temperature: f32)
set_system_prompt(&mut self, prompt: String)

// Generation
generate(&self, prompt: String) -> Result<String, JsValue>
chat(&self, message: String) -> Result<String, JsValue>

// Utility
check_availability(&self) -> Result<bool, JsValue>
list_models(&self) -> Result<JsValue, JsValue>
```

## Examples

### Example 1: Answer Generation

```javascript
const llm = UnifiedLlmClient.withOllama("http://localhost:11434", "llama3.1:8b");
llm.setTemperature(0.7);

const context = `
Entities: Socrates, Love, Beauty
Relationships: Socrates-discusses->Love, Love-relates_to->Beauty
Text: "Socrates says that love is the desire for beauty..."
`;

const prompt = `Based on this context, answer: What does Socrates say about love?

Context: ${context}`;

const answer = await llm.generate(prompt);
console.log(answer);
```

### Example 2: Multi-turn Conversation

```javascript
const llm = UnifiedLlmClient.withOllama("http://localhost:11434", "qwen2.5:7b");
llm.setSystemPrompt("You are a philosophy expert analyzing ancient texts.");

// First question
const answer1 = await llm.chat("What is Plato's theory of forms?");
console.log(answer1);

// Follow-up (note: stateless, need to include context)
const answer2 = await llm.chat("How does this relate to his view of knowledge?");
console.log(answer2);
```

### Example 3: Error Handling

```javascript
const llm = UnifiedLlmClient.withOllama("http://localhost:11434", "llama3.1:8b");

try {
  // Check availability first
  const available = await llm.check_availability();

  if (!available) {
    console.error("Ollama server is not available");
    // Fall back to WebLLM
    const fallbackLlm = UnifiedLlmClient.withWebLLM("Phi-3-mini-4k-instruct-q4f16_1-MLC");
    const answer = await fallbackLlm.generate(prompt);
    console.log(answer);
  } else {
    const answer = await llm.generate(prompt);
    console.log(answer);
  }
} catch (error) {
  console.error("Error:", error);
}
```

## Best Practices

1. **Always check availability** before making requests
2. **Use appropriate temperatures** based on task (0.2 for factual, 0.7 for creative)
3. **Set clear system prompts** to guide model behavior
4. **Handle errors gracefully** with fallback to WebLLM
5. **Choose models based on task** (7B for quality, 3B for speed)
6. **Enable CORS properly** to avoid security errors
7. **Monitor token usage** for cost optimization

## License

See main repository LICENSE file.
