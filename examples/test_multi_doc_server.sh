#!/bin/bash
#
# Test script for Multi-Document GraphRAG REST API Server
#
# This script:
# 1. Starts the server in background
# 2. Loads Symposium.txt
# 3. Incrementally merges Tom Sawyer.txt
# 4. Runs cross-document queries
# 5. Shows statistics

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVER_URL="http://localhost:3000"
COLLECTION="classics"
DOCS_DIR="docs-example"

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  Multi-Document GraphRAG Server Test                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

wait_for_server() {
    print_step "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
            print_success "Server is ready"
            return 0
        fi
        sleep 1
    done
    print_error "Server failed to start"
    exit 1
}

# Main
print_header

# Check if documents exist
if [ ! -f "$DOCS_DIR/Symposium.txt" ]; then
    print_error "Symposium.txt not found in $DOCS_DIR/"
    exit 1
fi

if [ ! -f "$DOCS_DIR/The Adventures of Tom Sawyer.txt" ]; then
    print_error "Tom Sawyer.txt not found in $DOCS_DIR/"
    exit 1
fi

print_success "Found both documents"

# Start server in background
print_step "Starting server..."
cargo run --example graphrag_multi_doc_server > /tmp/graphrag_server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
wait_for_server

# Trap to kill server on exit
trap "echo ''; print_step 'Stopping server...'; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; print_success 'Server stopped'" EXIT

echo ""
print_step "Testing API endpoints..."
echo ""

# ============================================================================
# PHASE 1: Batch Upload Symposium
# ============================================================================

echo -e "${BLUE}════ PHASE 1: Batch Upload Symposium ════${NC}"
echo ""

SYMPOSIUM_TEXT=$(cat "$DOCS_DIR/Symposium.txt" | jq -Rs .)

BATCH_REQUEST=$(cat <<EOF
{
  "documents": [
    {
      "id": "symposium",
      "text": $SYMPOSIUM_TEXT,
      "metadata": {
        "title": "Plato's Symposium",
        "author": "Plato",
        "genre": "Philosophy"
      }
    }
  ]
}
EOF
)

print_step "Uploading Symposium.txt..."
BATCH_RESPONSE=$(curl -s -X POST "$SERVER_URL/api/collections/$COLLECTION/documents/batch" \
    -H "Content-Type: application/json" \
    -d "$BATCH_REQUEST")

echo "$BATCH_RESPONSE" | jq .

PROCESSED=$(echo "$BATCH_RESPONSE" | jq -r '.processed')
CHUNKS=$(echo "$BATCH_RESPONSE" | jq -r '.total_chunks')
ENTITIES=$(echo "$BATCH_RESPONSE" | jq -r '.total_entities')
ELAPSED=$(echo "$BATCH_RESPONSE" | jq -r '.elapsed_ms')

print_success "Uploaded: $PROCESSED docs, $CHUNKS chunks, $ENTITIES entities (${ELAPSED}ms)"

echo ""

# ============================================================================
# PHASE 2: Incremental Merge Tom Sawyer
# ============================================================================

echo -e "${BLUE}════ PHASE 2: Incremental Merge Tom Sawyer ════${NC}"
echo ""

TOM_SAWYER_TEXT=$(cat "$DOCS_DIR/The Adventures of Tom Sawyer.txt" | jq -Rs .)

MERGE_REQUEST=$(cat <<EOF
{
  "document_id": "tom_sawyer",
  "text": $TOM_SAWYER_TEXT,
  "strategy": "incremental",
  "metadata": {
    "title": "The Adventures of Tom Sawyer",
    "author": "Mark Twain",
    "genre": "Fiction"
  }
}
EOF
)

print_step "Merging Tom Sawyer.txt..."
MERGE_RESPONSE=$(curl -s -X POST "$SERVER_URL/api/collections/$COLLECTION/merge" \
    -H "Content-Type: application/json" \
    -d "$MERGE_REQUEST")

echo "$MERGE_RESPONSE" | jq .

NEW_CHUNKS=$(echo "$MERGE_RESPONSE" | jq -r '.new_chunks')
NEW_ENTITIES=$(echo "$MERGE_RESPONSE" | jq -r '.new_entities')
MERGED_ENTITIES=$(echo "$MERGE_RESPONSE" | jq -r '.merged_entities')
TOTAL_ENTITIES=$(echo "$MERGE_RESPONSE" | jq -r '.total_entities')
ELAPSED=$(echo "$MERGE_RESPONSE" | jq -r '.elapsed_ms')

print_success "Merged: $NEW_CHUNKS chunks, $NEW_ENTITIES new entities, $MERGED_ENTITIES duplicates (${ELAPSED}ms)"

echo ""

# ============================================================================
# PHASE 3: Cross-Document Queries
# ============================================================================

echo -e "${BLUE}════ PHASE 3: Cross-Document Queries ════${NC}"
echo ""

QUERIES=(
    "What is Socrates' view on love?"
    "Compare Socrates and Tom Sawyer's approaches to life"
    "Find similarities between ancient philosophy and American literature"
    "What wisdom can we learn from both texts about human nature?"
)

for i in "${!QUERIES[@]}"; do
    QUERY="${QUERIES[$i]}"
    NUM=$((i+1))

    print_step "Query $NUM: \"$QUERY\""

    QUERY_REQUEST=$(cat <<EOF
{
  "query": "$QUERY",
  "collections": ["$COLLECTION"],
  "top_k": 3,
  "strategy": "rrf"
}
EOF
    )

    QUERY_RESPONSE=$(curl -s -X POST "$SERVER_URL/api/query/multi" \
        -H "Content-Type: application/json" \
        -d "$QUERY_REQUEST")

    # Extract results
    RESULTS=$(echo "$QUERY_RESPONSE" | jq -r '.results[] | "    \(.rank). [\(.source)] \(.doc_id) (sim: \(.similarity | tostring[:6]))\n       \(.text[:80])..."')

    SOURCE_DIST=$(echo "$QUERY_RESPONSE" | jq -r '.source_distribution')
    ELAPSED=$(echo "$QUERY_RESPONSE" | jq -r '.elapsed_ms')

    echo ""
    echo "  Top 3 Results (${ELAPSED}ms):"
    echo "$RESULTS"
    echo "  Source distribution: $SOURCE_DIST"
    echo ""
done

# ============================================================================
# PHASE 4: Collection Statistics
# ============================================================================

echo -e "${BLUE}════ PHASE 4: Collection Statistics ════${NC}"
echo ""

print_step "Fetching statistics..."
STATS_RESPONSE=$(curl -s "$SERVER_URL/api/collections/$COLLECTION/stats")

echo "$STATS_RESPONSE" | jq .

DOCS=$(echo "$STATS_RESPONSE" | jq -r '.documents')
CHUNKS=$(echo "$STATS_RESPONSE" | jq -r '.chunks')
ENTITIES=$(echo "$STATS_RESPONSE" | jq -r '.entities')
MEMORY=$(echo "$STATS_RESPONSE" | jq -r '.memory_mb')

print_success "Collection: $DOCS docs, $CHUNKS chunks, $ENTITIES entities, ${MEMORY}MB"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
print_success "All tests completed successfully!"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

print_step "Server logs available at: /tmp/graphrag_server.log"
print_step "Press Ctrl+C to stop server"
echo ""

# Keep script running to view server logs
tail -f /tmp/graphrag_server.log
