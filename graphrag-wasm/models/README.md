# ONNX embedding model (not tracked)

`index.html` copies this directory into `dist/` (`<link data-trunk rel="copy-dir" href="models" />`),
so it must exist for `trunk build` to succeed. The model itself (~87 MB) is git-ignored.

Optional download, enables in-browser semantic embeddings (`src/main.rs` loads `./models/minilm-l6.onnx`;
without it the app logs a `load_model` warning and keeps working without the ONNX embedder):

```bash
curl -L -o graphrag-wasm/models/minilm-l6.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
```
