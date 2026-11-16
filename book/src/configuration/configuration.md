# Configuration guide

The 0.3 runtime has one canonical configuration. The effective value is merged
from built-in defaults, `~/.graphrag/config.toml`, `./graphrag.toml`, and then
`GRAPHRAG_*` environment variables. Later sources override earlier sources;
conflicting canonical and legacy names in one source are rejected.

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5
```

`GRAPHRAG_TEXT_CHUNK_SIZE` and `GRAPHRAG_TEXT__CHUNK_SIZE` address nested
settings. Legacy names such as `GRAPHRAG_CHUNK_SIZE` remain readable. Values
are parsed and validated after the final merge: chunk sizes must be positive,
overlap must be smaller than the chunk size, and `top_k` must be greater than
zero. TOML, JSON, and JSON5 adapters use the same semantic rules.

Optional `retrieval.hyde = true` (default `false`) enables HyDE fusion in `ask`:
one extra LLM call per question writes a hypothetical passage. Hybrid search and the
answer prompt keep the original question; the reranker pool is extended with the
candidates of the question plus that passage (up to twice `reranker.candidates`),
and each chunk is ranked with both queries; the two rankings are interleaved by rank
(original first), since their scores are not comparable. It needs
`llm.enabled`, the `llm` feature and `reranker.enabled` (without the reranker HyDE is
skipped with a warning and no LLM call), and falls back to the plain question if the
call fails.

Pipeline-specific sections are optional when their feature is not used. Provider
sections require their corresponding feature and external service.
