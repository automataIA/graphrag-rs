# GraphRAG-rs 0.3 local benchmark

This versioned fixture is a local benchmark, not a SOTA certification. It
contains eight source documents, stable chunk/source IDs, nine graph entities,
eight annotated relationships, and 24 bilingual questions (six factual, six
relational, six cross-document, and six unanswerable).

The smoke runner is a `graphrag-core` example, so the benchmark data does not
add a sixth Cargo package. It uses one chunk per source, the versioned
chunking/top-k/context settings in `config.json`, and the same Ornith generator
for every mode. The four retrieval paths are distinct:

* `lexical`: BM25 over the versioned chunk texts.
* `vector`: real `multilingual-e5-small` E5 vectors from the configured
  llama.cpp embedding service, cosine-ranked over the same chunks.
* `hybrid`: reciprocal-rank fusion of lexical and vector rankings, using the
  weights and `k` from `config.json` (currently equal weights and `k=60`).
* `graph`: query entity seeds expanded through explicit relationships; relation
  context chunk IDs and target mentions are ranked by graph confidence.

Document embeddings are computed once per runner process and reused by vector
and hybrid retrieval. Raw records call their supplied rows `context_source_ids`;
they are not provider citations unless the provider emits citations explicitly.

Configure `GRAPHRAG_E5_ENDPOINT` and `GRAPHRAG_ORNITH_ENDPOINT` when services
do not use `http://127.0.0.1:18081` and `http://127.0.0.1:18080`. The runner
fails if either real provider fails and never replaces an error with hash vectors
or a mock answer.

The checked-in configuration is part of the run manifest. It fixes the E5 model
and dimension, generation model, chunking fixture, context budget, maximum
generation tokens, temperature, fusion settings, provider revisions, and seed.
Endpoints remain environment overrides so the same fixture can run against
local service instances.

Run one real question per mode as a smoke check from the repository root:

```bash
cargo run -p graphrag-core --example benchmark_smoke --features async,llama-cpp --locked -- --smoke
```

It writes raw retrieval, answer, usage, latency, and the complete provenance
manifest to `benchmarks/graphrag-0.3/results/smoke.json`. The manifest includes
SHA-256 digests for corpus, questions, configuration, and locally available E5
and Ornith GGUF files, plus provider revisions and seed 42 (the provider/server
seed).

After the smoke passes, run exactly three repetitions over all 24 questions and
all four modes:

```bash
cargo run -p graphrag-core --example benchmark_smoke --features async,llama-cpp --locked -- --runs 3
```

This writes 12 per-mode/run reports with 24 retrieval rows and 24 answer rows
each to `results/evaluation.json`, and a retrieval/answer quality report to
`results/report.md`. Provider usage remains optional per answer; missing usage
is recorded separately and never estimated. The runner fails on provider
errors and never replaces them with hash vectors or mock answers.
