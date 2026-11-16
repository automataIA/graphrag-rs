# Model selection

GraphRAG-rs does not prescribe one best model. The base local workflow uses no
LLM and is useful for testing the indexing and retrieval path. Provider-backed
extraction, embeddings, or generation can be enabled when an application has a
model and service that meet its language, latency, memory, and licensing needs.

Choose a model from its primary provider documentation, record the required
feature and configuration, and measure retrieval and answer quality on your own
documents. Browser models also depend on the user's device and WebGPU support.
