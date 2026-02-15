# GraphRAG E2E Pipeline Benchmarks

End-to-end benchmarks that test different pipeline configurations against real books.

## Quick Start

```bash
# List all available pipelines
./run_benchmarks.sh --list

# Dry-run: generate configs only (no LLM needed)
./run_benchmarks.sh --dry-run

# Run only algorithmic pipelines (fast, no LLM)
./run_benchmarks.sh --filter algo

# Run a single pipeline against one book
./run_benchmarks.sh --pipeline semantic_qwen3 --book symposium

# Run everything (requires Ollama with models)
./run_benchmarks.sh
```

## Pipeline Dimensions (7 Parameters)

| # | Dimension | Values |
|---|-----------|--------|
| 1 | **Approach** | `algorithmic`, `semantic`, `hybrid` |
| 2 | **Embeddings** | `hash` (fast), `ollama` (nomic-embed-text) |
| 3 | **Entity extraction** | gleaning on/off, rounds 0-4 |
| 4 | **Graph construction** | max_connections, similarity_threshold |
| 5 | **Text processing** | chunk_size/overlap: 256/50, 512/100, 800/300 |
| 6 | **Retrieval** | LightRAG, Leiden, CrossEncoder on/off |
| 7 | **LLM** | qwen3:8b, llama3.1:8b, mistral-nemo |

## Defined Pipelines

### Algorithmic (no LLM, fast)
- `algo_hash_small` — hash embeddings, 256/50 chunks
- `algo_hash_medium` — hash embeddings, 512/100 chunks
- `algo_hash_large` — hash embeddings, 800/300 chunks, Leiden
- `algo_ollama_embed` — Ollama embeddings, 512/100 chunks

### Semantic (require LLM)
- `semantic_qwen3` — Qwen3 8B, gleaning×2, LightRAG+Leiden
- `semantic_llama31` — Llama 3.1 8B, gleaning×2, LightRAG+Leiden
- `semantic_mistral` — Mistral Nemo, gleaning×4, LightRAG+Leiden
- `semantic_no_gleaning` — Qwen3 8B, no gleaning, no enhancements

### Hybrid
- `hybrid_qwen3` — Qwen3 8B, gleaning×2, LightRAG+Leiden
- `hybrid_small_chunks` — Llama 3.1, 256/50 chunks, minimal

## Output

```
results/
  <pipeline>__<book>.json    # Full result with params + timing + Q&A

reports/
  benchmark_report.json      # Combined JSON with all runs
  benchmark_report.md        # Human-readable comparison
```

### Result JSON Structure

```json
{
  "run_id": "semantic_qwen3__symposium",
  "pipeline": "semantic_qwen3",
  "book": "symposium",
  "parameters": { /* 7 pipeline dimensions */ },
  "timing": { "init_ms": 100, "build_ms": 35000, "total_query_ms": 8000 },
  "stats": { "entities": 142, "relationships": 89 },
  "questions_and_answers": [
    { "question": "...", "answer": "...", "sources": [], "query_time_ms": 1200 }
  ]
}
```

## Adding New Pipelines

Add a new function in `run_benchmarks.sh`:

```bash
pipeline_my_custom() {
    cat <<-PARAMS
name=my_custom
approach=semantic
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=600
chunk_overlap=200
use_gleaning=true
gleaning_rounds=3
llm_model=qwen3:8b-q4_k_m
llm_temp=0.15
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=true
PARAMS
}
```

Then add it to `ALL_PIPELINES` array.

## Requirements

- `jq` (for JSON processing)
- `graphrag-cli` (built via `cargo build --release -p graphrag-cli`)
- Ollama (for semantic/hybrid pipelines) with models listed in `ollama list`
