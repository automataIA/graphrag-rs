# GraphRAG-rs 0.3 release verification

This page records the 0.3 verification evidence available in this checkout.
The benchmark is a local comparison and is not a SOTA certification. Provider
services and model files used by the benchmark are local and are intentionally
outside the ordinary CI path.

## Package and compatibility decisions

The workspace contains five packages: `graphrag-core`, `graphrag`,
`graphrag-cli`, `graphrag-server`, and `graphrag-wasm`. All packages inherit
version `0.3.0` and Rust `1.88` from the workspace. The facade's default
features are `async`, `memory-storage`, and `basic-retrieval`; its minimal
feature set has no Tokio or CLI dependency. The WASM check is run separately
for `graphrag-wasm` with `wasm32-unknown-unknown`.

## Benchmark evidence

The reproducible fixture and raw output are under
[`benchmarks/graphrag-0.3`](../../../benchmarks/graphrag-0.3). The completed
evaluation contains 12 mode/run blocks and 288 question records (24 questions
× 4 modes × 3 runs). It records the provider revisions, configuration and
model checksums, raw retrieval and answers, optional measured usage, and
latency. The checked-in aggregate was produced with:

```text
lexical  Recall@3 0.8796  MRR 0.9167  source precision 0.7500
vector   Recall@3 0.9537  MRR 0.9722  source precision 0.4630
hybrid   Recall@3 0.9537  MRR 0.9722  source precision 0.4630
graph    Recall@3 0.5833  MRR 0.5463  source precision 0.3796
```

The report separates retrieval metrics from answer term coverage, abstention,
latency, and measured provider tokens. Missing provider usage stays `None` and
is counted separately. The fixture records E5 revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3` and llama.cpp revision
`cc231cb0da565440cf6a3e5b55dfeba477972cb6`; the exact SHA-256 values are in
the evaluation manifest and `scripts/e5/checksums.sha256` where available.
Across the completed runs, provider usage totals **38,238 measured tokens**.
These are measured prompt/completion/total fields; missing usage is never
estimated.

## Verification matrix

| Check | Status in this checkout |
|---|---|
| Workspace has exactly five packages and all versions are 0.3.0 | passed locally with `cargo metadata --no-deps --locked` |
| Rust 1.88 native package checks | passed by the package checks and feature matrix in this checkout |
| Minimal facade dependency tree excludes Tokio, CLI, and color-eyre | recorded by the feature-matrix check |
| `wasm32-unknown-unknown` check for `graphrag-wasm` | passed locally with Rust 1.88; the dedicated CI job runs the same check |
| Blocking rustfmt, unit/integration tests, and doctests | passed locally for the supported core, facade, CLI, and server profiles; full remote CI remains separate |
| Clippy with `-D warnings` on supported feature matrix | passed locally for core, facade, CLI, and server default/no-default matrices |
| `cargo-deny` 0.20.2 advisory check | locally passed with narrowly scoped, dated transitive exceptions in `deny.toml` |
| `cargo package` and metadata checks | passed locally for core, CLI, and facade; server/wasm are publish=false |
| Facade installation with the `cli` feature | passed locally; the installed `graphrag` binary is executable |
| Three-run, four-mode benchmark | completed; see `results/evaluation.json` |

The implementation agent's results were independently reproduced by the
supervisor in this checkout with Rust 1.88, `--locked`, and offline Cargo
resolution. Remote GitHub Actions remains separate and should run the same
commands on the release commit before publishing. The local llama workflow is
manual and requires operators to supply provider endpoints and model paths; it
downloads no multi-gigabyte artifacts.

## Known limits

The server is a trusted local service without multi-user authentication. Qdrant
and LanceDB do not by themselves construct a knowledge graph. Browser/WASM
integration and several advanced providers remain experimental. The base path
uses local retrieval and does not claim generated prose. Provider-backed runs
depend on the exact local service and model configuration, and the benchmark
does not certify production quality or SOTA performance.
