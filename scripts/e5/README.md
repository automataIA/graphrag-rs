# multilingual-e5-small reproducibility record

The `llama-cpp` embedding adapter targets the pinned `multilingual-e5-small`
conversion used for GraphRAG 0.3. The source model is Hugging Face revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`; the llama.cpp server used for the
probe was revision `cc231cb0da565440cf6a3e5b55dfeba477972cb6`. The source weights
and converted probe are identified by the SHA-256 values in
`checksums.sha256`. The upstream model card is
[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).

Conversion must use the XLM-R tokenizer and BERT model weights from the pinned
source revision. The runtime contract is 384 dimensions, E5 `query:` and
`passage:` prefixes (each followed by one space), a 512-token limit with the
final token retained as EOS,
masked mean pooling on the server, and L2 normalization.

`e5-parity-probe.json` records five multilingual token and embedding comparisons
against the pinned source. The parity script also runs a long-input regression
whose raw IDs are reduced to the first 511 tokens plus the original EOS and sent
as integer IDs to `/v1/embeddings`; that generated result is not included in the
checked-in five-case record. All recorded token sequences matched; the minimum
recorded cosine similarity was `0.9999994039535522`. Reproduce or refresh this record
with the prepared conversion/parity tools from the phase-6 handoff:

```text
python scripts/e5/convert_e5_probe.py /path/to/multilingual-e5-small /path/to/probe.gguf --llama-cpp /path/to/llama.cpp
python scripts/e5/e5_parity_probe.py /path/to/multilingual-e5-small --probe /path/to/probe.gguf
```

The HTTP provider does not synthesize vectors when the service is unavailable,
returns malformed data, or reports the wrong dimension. Those conditions are
surfaced as errors to the caller.
