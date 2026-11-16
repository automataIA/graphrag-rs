# Embeddings

The default native path uses local hash embeddings together with lexical and
graph retrieval. This keeps the quickstart self-contained and does not require
an API key or network service.

Provider-backed embeddings are optional. Ollama and other backends require the
matching Cargo feature, model configuration, and an available service. The WASM
package has separate browser-oriented integrations and target requirements.
Embedding quality, memory use, and latency depend on the selected model and
deployment, so evaluate them on the application's documents.
