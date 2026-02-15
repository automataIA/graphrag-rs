# ============================================================================
# GraphRAG Server — Multi-stage Dockerfile
# ============================================================================
# Build:   docker build -t graphrag-server -f graphrag-server/Dockerfile .
# Run:     docker run -p 8080:8080 graphrag-server
# ============================================================================

# --- Stage 1: Build ---
FROM rust:1.82-slim-bookworm AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --release -p graphrag-server

# --- Stage 2: Runtime ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/graphrag-server /usr/local/bin/graphrag-server

# Non-root user
RUN useradd -r -s /bin/false graphrag
USER graphrag

# Default environment variables
ENV EMBEDDING_BACKEND=hash \
    EMBEDDING_DIM=384 \
    JWT_SECRET=change-me-in-production \
    QDRANT_URL=http://qdrant:6334 \
    COLLECTION_NAME=graphrag

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD ["/usr/local/bin/graphrag-server", "--version"] || exit 1

ENTRYPOINT ["graphrag-server"]
