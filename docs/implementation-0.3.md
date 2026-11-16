# GraphRAG-rs 0.3 implementation record

Implementation requested on 2026-09-05. GPT-5.6 Luna (high reasoning) owns
implementation, one phase at a time. The supervising agent reviews each phase
and reproduces decisive acceptance checks before authorizing the next phase.

## Workspace preservation

The workspace started with user changes, an untracked golden evaluator, and a
staged deletion of `CHANGELOG.md`. These changes must be retained. The initial
binary diff against HEAD and a copy of the untracked evaluator were captured
outside the repository at `/tmp/graphrag-03-baseline/` for review. No release,
commit, push, or deployment is part of this task.

## Phase gates

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | Canonical validated configuration and meaningful CLI validation errors | Approved |
| 2 | Stable async APIs and functional persistent synchronous path | Approved |
| 3 | Optional CLI, lightweight facade, portable builds and verified MSRV | Approved |
| 4 | Executable quickstart and accurate user documentation | Approved |
| 5 | Local server defaults, Unicode safety and atomic configuration updates | Approved |
| 6 | Optional llama.cpp generation and embedding integration | Approved |
| 7 | Versioned bilingual corpus and three real runs per retrieval mode | In progress |
| 8 | Executed CI matrix, dependency review, migration guide and final report | Pending |

## Verification policy

Tests must exercise behavior, including failures, rather than only mirror the
implementation. Offline tests and tests requiring local models are separate.
Claims about passed checks are recorded only after commands complete. Measured
quality, retrieval quality, model usage and estimates must remain distinguishable.
The local benchmark is a reproducibility exercise, not a SOTA certification.

## Supervisor decisions and baseline checks

- `cargo +1.85.0 check -p graphrag-core --no-default-features --locked --offline`
  passed before implementation. This proves only the minimal native core target.
- The existing lockfile includes Leptos 0.8 dependencies requiring Rust 1.88.
  The user subsequently authorized newer stable dependency versions and a
  corresponding Rust minimum when required. The final supported toolchains
  will be declared from actual checks, without downgrading the UI framework
  solely to retain the original 1.85 target.
- Configuration acceptance includes rejecting wrong types and contradictory
  explicitly supplied values, even when one value equals the default. JSON,
  TOML and JSON5 must converge before runtime construction. A loader must not
  silently turn malformed configuration into default settings.

## Execution interruption

Luna's implementation turn stopped at an account usage limit during phase 1,
before its delivery or approval. Changes remain available for review. The
supervisor completed independent checks during the interruption. The user then
confirmed availability had returned, and implementation resumed with the same
Luna High agent. No phase has been marked complete on the basis of partial work.

Supervisor checks on the interrupted phase-1 tree:

- CLI build (`cargo build -p graphrag-cli --locked --offline`): passed.
- External CLI harness: 17/17 cases passed, covering JSON/TOML/JSON5,
  legacy aliases, conflicts, invalid types/ranges and exit codes.
- External library consumer: 4/4 tests passed for constructors/builders,
  nested limits, missing-document independence and complete JSON roundtrip.
- Still failing: the distributed algorithmic JSON5 template is rejected for a
  missing `graph_construction` field; direct runtime loaders accept zero chunk
  size; hierarchical loading also accepts zero and ignores
  `GRAPHRAG_OUTPUT_DIR`. These are blockers for phase 1, despite the CLI passes.
- These are intermediate results, not final release acceptance. The external
  harnesses and logs are in `/tmp/graphrag-03-baseline/`; passing behavioral
  cases must become durable repository regression coverage before approval.

## Local model setup discovered

The existing llama.cpp checkout is `/home/dio/llama.cpp`. The selected generation
model is `Ornith-1.5-9B-MTP-Q6_K.gguf` in `/home/dio/modelli`, served on the RTX 4070.
The user authorized downloading `intfloat/multilingual-e5-small` for CPU embedding
inference. Model revisions, hashes, runtime versions and benchmark settings will
be recorded with the measured results. Existing personal model scripts are not
owned by this implementation.

Read-only environment inventory (before model runs):

- llama.cpp checkout and server binary report commit
  `cc231cb0da565440cf6a3e5b55dfeba477972cb6` (build 305).
- Generation GGUF SHA-256:
  `79a9925bb7771dea3530d57b185e97b9713a73a7c06bdb9804f7d019aa42f480`.
- GPU: NVIDIA GeForce RTX 4070, 12282 MiB, driver 595.84.
- Installed Rust toolchains include 1.85.0 and 1.88; the stable toolchain has
  `wasm32-unknown-unknown` installed.

Resumed phase-1 supervision:

- The hierarchical/direct-loader harness reached 15/15 passing cases after
  corrections. Subsequent edits are still being reviewed for regressions.
- `cargo test -p graphrag-core --lib --locked --offline`: 371 passed.
- `cargo check -p graphrag-server --locked --offline`: passed.
- Phase 1 remains open pending template corrections and public-parser coverage.

Embedding environment preparation (not phase-6 implementation acceptance):

- Downloaded E5 source revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- The unmodified converter fails because this model combines the BERT encoder
  with the XLM-R tokenizer. A temporary conversion wrapper uses llama.cpp's
  existing XLM-R vocabulary exporter while preserving BERT positions/weights.
  Neither the personal llama.cpp checkout nor the downloaded source was edited.
- The experimental F16 GGUF was served on CPU, loopback port 18081. Five texts
  spanning Italian, English and Japanese had identical token IDs to Transformers;
  minimum embedding cosine was 0.9999994 and maximum absolute error 0.000248.
- This experiment must become a pinned, reproducible conversion procedure and
  durable verification before the real benchmark. Probe scripts and raw results
  currently live in `/tmp/graphrag-03-baseline/`.

## Phase 1 approval

Supervisor approved phase 1 after Luna delivery on 2026-09-05. Canonical
`text.chunk_size`, `text.chunk_overlap` and `retrieval.top_k` replace duplicate
runtime fields. All public parsers validate; file layers normalize/remove aliases
before merging, with conflicts rejected within a layer. Document existence is
checked separately. Runtime factories, CLI and server use shared validation.

Final phase evidence: 373 core unit tests, 33 CLI unit tests, 48 passing doctests
(17 ignored); 4 external consumer tests; 17 CLI validation cases; 15 loader/env
cases; 8 additional layer/parser cases; all 30 distributed TOML/JSON5 templates.
Core minimal and JSON5 builds, server check, workspace formatting and diff
whitespace checks passed. These checks cover configuration, not later phases.
Luna has been assigned phase 2; phase 3 remains pending.

## Phase 2 supervision in progress

The same external synchronous consumer builds without `async` and with `async`.
Separate-process indexing/querying passes for sync → sync, sync → async,
async → sync and async → async. Each reader rejects corrupted `graph.json`
without overwriting it. The current core unit suite remains at 373 passing tests.
Supervisor harness: `/tmp/graphrag-03-baseline/phase2_persistence_review.py`.

Phase 2 is not approved: two external regression tests demonstrate that
`ask_sync` silently ignores a configured LLM generation or embedding provider.
These must return explicit unsupported-provider errors. Formatting also needs
completion, followed by durable regression tests and final worker delivery.
Luna reported another usage limit; a resume was attempted after the user's
request to continue. All partial changes remain preserved.

Luna resumed successfully at approximately 18:29 local time. A centralized sync
provider guard fixes both supervisor regressions: the external consumer now
passes all six tests with `async` enabled. Review of factory/pipeline coverage and
remaining feature combinations is ongoing.

Generation environment preparation: Ornith was started on loopback port 18080,
GPU offload 99 layers, context 8192, one slot, six CPU threads, flash attention on,
reasoning off, temperature 0 and seed 42. A real HTTP chat completion returned
“Maria Rossi dirige il progetto Prisma.” with provider-reported usage 51 prompt,
9 completion and 60 total tokens. Raw evidence is
`/tmp/graphrag-03-baseline/ornith-real-probe.json`. This confirms environment
readiness only; the project adapter and benchmark are still pending phases 6–7.

## Phase 2 approval

Supervisor approved phase 2 after final delivery and verification. Sync methods,
factory methods and pipeline methods use stable `_sync` names regardless of the
`async` feature. Local retrieval sorts graph-derived collections and returns
source context; unsupported configured providers return explicit errors. Build
persists the graph, and initialization propagates corrupt-workspace errors.
Optional API wrappers were updated to compile against current methods.

Final verification: 373 core unit tests and six integration tests passed; the
external six-test consumer passed both feature configurations; the integration
suite passed with API and PageRank enabled. All four cross-process sync/async
save/load combinations and corrupt-workspace checks passed on rebuilt binaries.
Minimal core has no normal Tokio or CLI dependency. Formatting and diff checks
passed. Luna also reported successful CLI/server checks and all 30 templates.
Optional API warnings and broader documentation cleanup remain for later gates.
Phase 3 is now assigned; later phases remain pending.


## Phase 3 approval

Supervisor approved phase 3 after Luna delivery on 2026-09-05. All five crates
inherit workspace version 0.3.0 and Rust minimum 1.88. Core/facade defaults are
async, memory storage and basic retrieval. The facade CLI is optional and the
binary requires `cli`; advanced integrations are forwarded as opt-in features.
CPU-native compilation is no longer the workspace default.

Supervisor reproduced: an external facade consumer with no default features and
with async (identical synchronous source, persistent index/query), a normal
dependency tree without Tokio/Clap/CLI for the minimal facade, the installed
`graphrag 0.3.0` binary and separate-process indexing/querying, formatting, and
Rust 1.88 checks of the minimal facade, all five workspace packages, and the
actual wasm32-unknown-unknown target for graphrag-wasm. All passed. Luna also
reported 290 passing minimal core unit tests and the stable workspace check.
Existing warnings remain for the phase-8 lint review.

Phase 4 is now assigned. Verification logs and external consumer are under
`/tmp/graphrag-03-baseline/phase3-*`; the final release report will contain
reproducible commands rather than depend on these temporary paths.


## Phase 4 supervision in progress

After the user's continuation, Luna resumed work on documentation and the CLI
configuration path. Supervisor independently ran both standalone quickstart
examples, the separate-process CLI integration test, and parsed all eight TOML
blocks in the three public READMEs. These now pass after syntax corrections.

The supervisor reproduced that `index` and `ask` ignored `--config`, accepting
`retrieval.top_k = 0`. On the corrected binary, valid configuration succeeds and
both invalid cases return exit 1 with the actual Top-k validation error. Review
also identified a false-positive negative test (empty workspace), a broken facade
manual link, and manual includes referencing ignored local-only documentation.
These corrections and final worker delivery are still required before approval.
No phase after 4 is authorized yet. Logs are in `/tmp/graphrag-03-baseline/phase4-*`.


## Phase 4 approval

Supervisor approved phase 4 on 2026-09-06 (Europe/Rome). The public quickstart
installs the facade binary, indexes a UTF-8 document and queries it from another
process. Both complete standalone library examples run successfully. CLI index
and ask now honor explicit configuration files and reject invalid settings.
Repeated indexing preserves previously stored documents; output no longer
promises a fresh empty workspace. All manual includes reference files that can
be delivered from Git; twelve pages now contain their own maintained content.

Supervisor reproduced three CLI integration tests, both final library examples,
valid and invalid explicit configuration probes, eight README TOML blocks,
manual include availability, formatting and diff checks. All passed. The
installed release binary was verified in phase 3; phase-4 behavior was verified
on the current test binary. mdbook rendering was not run because the tool is
not installed; static include validation is not a rendering test.
Phase 5 is assigned to Luna, with actual HTTP supervision planned.


## Phase 5 supervision in progress

An isolated `cargo build -p graphrag-server --no-default-features --locked
--offline` fails because core's `pagerank` feature uses Rayon without enabling
it. Workspace feature unification had concealed this failure. Luna's ownership
was extended to the small core feature-dependency correction. The independent
HTTP harness is ready at `/tmp/graphrag-03-baseline/phase5_http_review.py`; it
checks actual listening sockets, CORS, multilingual excerpts and failed
configuration rollback. It has not yet passed on a corrected server.

Luna hit a usage limit at phase-5 startup. The environment clock subsequently
reported 2026-09-06 05:02 UTC, after the displayed reset time; the same agent
was resumed and has started modifying the server. No replacement model was
used. Model services on loopback ports 18080 and 18081 still report healthy.

The corrected isolated no-default server build passes. The first live HTTP
review passes actual loopback listening, default CORS denial, explicit allowed
origin and preflight checks, but `POST /api/config` returns 404 because of route
registration. Luna is correcting this before rollback tests can run. A separate
PageRank-only core build exposed an async-trait import behind the wrong feature
gate; its correction is also under review. These are intermediate findings,
not phase approval.

The subsequent live HTTP review passes after route corrections: configuration,
ingestion, query and build are reachable; Unicode excerpts work; malformed
settings are rejected at startup; semantic-invalid and corrupt-workspace
candidates preserve the active config/graph; deletion is explicitly rejected
when the configured graph cannot be updated consistently. Default server check
and PageRank-only core check also pass. Durable regression coverage and final
Luna delivery are still required for phase-5 approval.


## Phase 5 approval

Supervisor approved phase 5 on 2026-09-06. Production and in-process tests share
the route registration, preserving OpenAPI operations. Loopback is the default;
CORS requires explicit valid origins. Configuration candidates initialize before
an atomic swap. Failed candidates preserve a populated active graph and config.
UTF-8 excerpts are safe; unsupported graph deletion is rejected before mutation.

Supervisor rebuilt the final no-default server and reproduced its unit/integration
and documentation tests, then the actual HTTP/OpenAPI harness covering bind,
CORS, multilingual ingestion/query, graph build, invalid configuration and corrupt
workspace rollback. Formatting and diff checks passed. Luna additionally ran the
default server tests and isolated PageRank/core Unicode checks; the supervisor
had also reproduced the default server check and isolated PageRank check.
Existing warnings remain for phase 8. Phase 6 is now assigned to Luna High.


## Phase 6 approval

Supervisor approved phase 6 on 2026-09-06. The opt-in `llama-cpp` feature
provides OpenAI-compatible generation and multilingual-e5-small embeddings,
using explicit endpoints, model names and timeouts from the common configuration.
The runtime applies query/passage prefixes, preserves the original EOS within a
512-token bound, validates and normalizes 384-dimensional vectors, and propagates
provider failures without hash or simulated fallbacks. `ask_with_usage` exposes
provider-reported token counts while the existing `ask` result remains stable.

Controlled provider tests cover malformed and partial responses, HTTP errors,
UTF-8-safe bounded diagnostics, batch indices and ordering, missing usage and the
511+EOS boundary. The minimal core, minimal facade and full async provider
combination compile offline; the facade without defaults still excludes Tokio,
ureq and the CLI. Formatting and diff checks pass.

The supervisor also ran an external facade consumer against the two real local
llama.cpp services. It indexed, answered in Italian, reopened the persisted
workspace in another instance and answered in English. The provider reported
usage of 208/18/226 and 207/21/228 prompt/completion/total tokens. A pinned E5
parity procedure compares Hugging Face and llama.cpp tokenizer IDs and vectors;
five multilingual probes reached minimum cosine 0.9999994, and the final script
also passed a real 3005-token input reduced to 512 IDs. The generation and E5
revisions and SHA-256 values are recorded under `scripts/e5/`. Phase 7 is now
assigned to Luna High; these local checks are reproducibility evidence rather
than a SOTA or production certification.


## Phase 7 approval

Supervisor approved phase 7 on 2026-09-06. The versioned bilingual corpus has
24 unique questions split evenly across Italian and English and across factual,
relational, cross-document, and unanswerable categories. The runner compares
lexical, vector, hybrid, and graph retrieval over the same chunks, generator,
top-k, and context budget. It uses real llama.cpp generation and E5 embeddings,
shares fixed document/query embedding work, and never substitutes hash vectors
or mock answers.

The final run contains exactly three repetitions of four modes and 24 questions:
12 aligned reports and 288 unique run/mode/question answer rows. All provider
usage is present and totals 38,238 measured tokens. The supervisor independently
recomputed Recall@3, MRR with misses scored as zero, retrieved-source precision,
expected-term coverage, abstention, token totals, mode summaries, and fixture
checksums. The evaluator's six tests, minimal runner build, argument rejection,
formatting, and diff checks pass. Results remain a local benchmark and are not a
SOTA certification. Phase 8 is now assigned to Luna High.


## Phase 8 approval

Supervisor approved phase 8 on 2026-09-06. The blocking CI now runs rustfmt,
offline unit and integration tests, doctests, the cross-process CLI quickstart,
standalone sync and async examples, the sync/async feature matrix, a real
`wasm32-unknown-unknown` package check, Rust 1.88 checks, Clippy with warnings
denied, a pinned cargo-deny advisory audit, and packaging validation. Local
llama.cpp verification is isolated in a manual self-hosted workflow.

The supervisor independently reproduced the final matrix with Rust 1.88 and a
locked offline dependency graph. Core default/minimal tests and doctests,
facade default/minimal tests, CLI tests and doctests, the three-test quickstart,
both standalone examples, server default/no-default tests and doctests, and the
WASM target all passed. All eight Clippy configurations passed with
`-D warnings`; cargo-deny 0.20.2 passed with the documented, dated transitive
exceptions. Metadata contains exactly five 0.3.0 packages at MSRV 1.88, the
core, CLI, and facade packages were created, and installing the facade with the
`cli` feature produced an executable `graphrag` binary.

The 0.2 to 0.3 migration guide covers canonical configuration fields and legacy
conflicts, synchronous `_sync` names, feature/default/CLI changes, bind and CORS
defaults, llama.cpp opt-in behavior, and workspace persistence. The release
report records the independently reproduced checks, benchmark metrics, model
revisions, 38,238 measured provider tokens, and remaining limits. Remote CI,
publishing, and deployment remain separate release operations.
