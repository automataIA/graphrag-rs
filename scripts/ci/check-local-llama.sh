#!/usr/bin/env bash
set -euo pipefail

: "${GRAPHRAG_E5_ENDPOINT:?set GRAPHRAG_E5_ENDPOINT to a running local E5 service}"
: "${GRAPHRAG_ORNITH_ENDPOINT:?set GRAPHRAG_ORNITH_ENDPOINT to a running local Ornith service}"
: "${GRAPHRAG_E5_GGUF:?set GRAPHRAG_E5_GGUF to the runner-local E5 GGUF path}"
: "${GRAPHRAG_ORNITH_GGUF:?set GRAPHRAG_ORNITH_GGUF to the runner-local Ornith GGUF path}"

for model in "$GRAPHRAG_E5_GGUF" "$GRAPHRAG_ORNITH_GGUF"; do
  test -f "$model" || { echo "model path is not a file: $model" >&2; exit 1; }
done

cargo run -p graphrag-core --example benchmark_smoke \
  --features async,llama-cpp --locked -- --smoke
