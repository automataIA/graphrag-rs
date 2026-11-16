# GraphRAG local server

`graphrag-server` is an optional Actix Web service for a trusted local process. It
provides document ingestion, local retrieval, configuration, health, graph
statistics, and OpenAPI/Swagger endpoints. It has no enabled multi-user
authentication; keep the listener on loopback or protect it at a trusted network
boundary.

## Run it

From a checkout:

```bash
cargo run --locked -p graphrag-server --no-default-features
```

The default listener is `127.0.0.1:8080`. Set `GRAPHRAG_BIND_ADDR` explicitly
when another address is required; it must be an IP address and port.

```bash
GRAPHRAG_BIND_ADDR=127.0.0.1:18080 cargo run --locked -p graphrag-server --no-default-features
```

The `qdrant` feature enables the optional Qdrant document/vector backend. A
Qdrant store supports vector document search; it does not construct a
GraphRAG knowledge graph by itself. A configured GraphRAG pipeline is required
for `POST /api/graph/build` to perform graph construction. The in-memory backend
is useful for local experiments and is ephemeral.

The server's `auth` feature is incomplete and is not an access-control boundary.
LanceDB is present as an experimental module and is not wired as an operational
server backend. Do not expose this service to untrusted users.

OpenAPI is available at `/openapi.json` and Swagger UI at `/swagger` after the
server starts.

## CORS

Cross-origin access is disabled by default. For a browser client, provide an
exact comma-separated allowlist of `http://` or `https://` origins:

```bash
GRAPHRAG_CORS_ORIGINS=http://localhost:3000,https://app.example.test \
  cargo run --locked -p graphrag-server --no-default-features
```

Wildcard origins, blank entries, malformed origins, and implicit expansion are
rejected at startup. This setting controls browser CORS headers; it is not
authentication.

## Minimal API walkthrough

In another shell, add a UTF-8 document and query it:

```bash
curl -X POST http://127.0.0.1:8080/api/documents \
  -H 'Content-Type: application/json' \
  -d '{"title":"Prisma project","content":"Maria Rossi maintains the Prisma project."}'

curl -X POST http://127.0.0.1:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Prisma","topK":3}'
```

The base in-memory path performs local lexical retrieval and returns document
excerpts with scores. It does not call an LLM or claim generated prose. Optional
embedding/provider features can add remote services when configured. Provider
fallback behavior is backend-specific and should be treated as a legacy
compatibility path rather than as evidence of generated or remote results.

Documents remain in memory until the process exits. Qdrant persistence belongs
to the Qdrant deployment. A GraphRAG pipeline can persist its configured
workspace through its normal core configuration.

## Configuration API

Configure the pipeline before adding documents when graph construction is
needed. `POST /api/config` parses the same canonical and legacy configuration forms as
the core loader, validates the complete configuration, initializes a candidate
pipeline, and swaps it into service only after initialization succeeds. A
malformed or semantically invalid candidate leaves the active configuration and
graph unchanged. `GET /api/config` returns the active canonical configuration
and graph metadata from the same snapshot.

Example request (JSON):

```bash
curl -X POST http://127.0.0.1:8080/api/config \
  -H 'Content-Type: application/json' \
  -d '{"text":{"chunk_size":1000,"chunk_overlap":200},"retrieval":{"top_k":5},"auto_save":{"enabled":true,"base_dir":"./workspaces","workspace_name":"active"}}'
```

Validate without applying:

```bash
curl -X POST http://127.0.0.1:8080/api/config/validate \
  -H 'Content-Type: application/json' \
  -d '{"retrieval":{"top_k":5}}'
```

`POST /api/config/validate` reports parse and semantic errors without changing
state. The default precedence for file-based core configuration is documented
by the core crate: defaults, home config, project config, then environment
variables.

## Graph endpoints and capabilities

- `GET /health` reports process health, document count, active backend, query
  count, and whether the active GraphRAG pipeline has built a graph.
- `GET /api/documents` and `POST /api/documents` list/add searchable documents.
- `POST /api/query` performs retrieval and returns relevant excerpts.
- `POST /api/graph/build` runs the configured GraphRAG pipeline. It returns an
  error when only memory or Qdrant storage is available, because those stores
  do not create graph entities and relationships.
- `DELETE /api/documents/{id}` is available for storage-only mode. It returns an
  error while a GraphRAG pipeline is configured because the core cannot yet
  remove a document atomically from the active graph.
- `GET /api/graph/stats` reports graph entities, relationships, chunks, and the
  `graph_built` state for a configured pipeline. Storage-only backends report
  zero graph entities and `graph_built: false`.
- `GET /api/config/template` and `GET /api/config/default` expose starter
  configuration documents.

Build failures propagate as HTTP errors. There is no mock-success fallback for
graph construction.

## Features

```bash
# local memory retrieval (smallest server build)
cargo run --locked -p graphrag-server --no-default-features

# Qdrant vector document search
cargo run --locked -p graphrag-server --features qdrant

# Optional provider integrations; configure and operate them explicitly
cargo run --locked -p graphrag-server --features 'qdrant,llm'
```

The server is version `0.3.0` and follows the workspace Rust 1.88 MSRV. For
core pipeline APIs and sync/async examples, see the root and facade crate
README files.
