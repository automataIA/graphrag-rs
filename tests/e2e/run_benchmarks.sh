#!/usr/bin/env bash
# =============================================================================
# GraphRAG E2E Pipeline Benchmark Runner
# =============================================================================
# Usage:
#   ./run_benchmarks.sh                     # Run all pipelines
#   ./run_benchmarks.sh --filter algo       # Only pipelines matching "algo"
#   ./run_benchmarks.sh --dry-run           # Generate configs only, no execution
#   ./run_benchmarks.sh --pipeline P1       # Run a single pipeline by name
#   ./run_benchmarks.sh --book symposium    # Only test against one book
#   ./run_benchmarks.sh --list              # List all defined pipelines
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI_BIN="$PROJECT_ROOT/target/release/graphrag-cli"
CONFIGS_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"
REPORTS_DIR="$SCRIPT_DIR/reports"
BOOKS_DIR="$PROJECT_ROOT/docs-example"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# =============================================================================
# CONFIG: Books
# =============================================================================
declare -A BOOKS
BOOKS[symposium]="/home/dio/graphrag-rs/docs-example/Symposium.txt"
BOOKS[tom_sawyer]="/home/dio/graphrag-rs/docs-example/The_Adventures_of_Tom_Sawyer.txt"

# =============================================================================
# CONFIG: Questions per book (pipe-separated)
# =============================================================================
declare -A QUESTIONS
QUESTIONS[symposium]="What is the nature of love according to Socrates?|Who are the main speakers in the Symposium?|What is Aristophanes' myth about the origin of love?|How does Diotima describe the ladder of beauty?|What is the relationship between love and wisdom?"
QUESTIONS[tom_sawyer]="Who is Tom Sawyer and what are his main characteristics?|What happens during the fence painting scene?|Who is Huckleberry Finn and how does he relate to Tom?|What adventure do Tom and Huck have in the cave?|What role does Injun Joe play in the story?"

# =============================================================================
# PIPELINES: Each function defines one pipeline configuration
# =============================================================================
# Format: pipeline_<name>() prints the 7 dimensions as key=value pairs
# Dimensions:
#   approach:      algorithmic | semantic | hybrid
#   embedding:     hash | ollama
#   embed_model:   (model name, only if embedding=ollama)
#   chunk_size:    integer
#   chunk_overlap: integer
#   use_gleaning:  true | false
#   gleaning_rounds: integer
#   llm_model:     (ollama model name)
#   llm_temp:      float
#   lightrag:      true | false
#   leiden:        true | false
#   cross_encoder: true | false
# =============================================================================

# --- Algorithmic pipelines (no LLM needed, fast) ---

pipeline_algo_hash_small() {
    cat <<-PARAMS
name=algo_hash_small
approach=algorithmic
embedding=hash
embed_model=none
embed_dim=384
chunk_size=256
chunk_overlap=50
use_gleaning=false
gleaning_rounds=0
llm_model=none
llm_temp=0.0
ollama_enabled=false
lightrag=false
leiden=false
cross_encoder=false
PARAMS
}

pipeline_algo_hash_medium() {
    cat <<-PARAMS
name=algo_hash_medium
approach=algorithmic
embedding=hash
embed_model=none
embed_dim=384
chunk_size=512
chunk_overlap=100
use_gleaning=false
gleaning_rounds=0
llm_model=none
llm_temp=0.0
ollama_enabled=false
lightrag=false
leiden=false
cross_encoder=false
PARAMS
}

pipeline_algo_hash_large() {
    cat <<-PARAMS
name=algo_hash_large
approach=algorithmic
embedding=hash
embed_model=none
embed_dim=384
chunk_size=800
chunk_overlap=300
use_gleaning=false
gleaning_rounds=0
llm_model=none
llm_temp=0.0
ollama_enabled=false
lightrag=false
leiden=true
cross_encoder=false
PARAMS
}

pipeline_algo_ollama_embed() {
    cat <<-PARAMS
name=algo_ollama_embed
approach=algorithmic
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=512
chunk_overlap=100
use_gleaning=false
gleaning_rounds=0
llm_model=none
llm_temp=0.0
ollama_enabled=true
lightrag=false
leiden=false
cross_encoder=false
PARAMS
}

# --- Semantic pipelines (require LLM) ---

pipeline_semantic_qwen3() {
    cat <<-PARAMS
name=semantic_qwen3
approach=semantic
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=800
chunk_overlap=300
use_gleaning=true
gleaning_rounds=2
llm_model=qwen3:8b-q4_k_m
llm_temp=0.1
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=false
PARAMS
}

pipeline_semantic_llama31() {
    cat <<-PARAMS
name=semantic_llama31
approach=semantic
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=800
chunk_overlap=300
use_gleaning=true
gleaning_rounds=2
llm_model=llama3.1:8b
llm_temp=0.1
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=false
PARAMS
}

pipeline_semantic_mistral() {
    cat <<-PARAMS
name=semantic_mistral
approach=semantic
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=800
chunk_overlap=300
use_gleaning=true
gleaning_rounds=4
llm_model=mistral-nemo:latest
llm_temp=0.2
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=false
PARAMS
}

pipeline_semantic_no_gleaning() {
    cat <<-PARAMS
name=semantic_no_gleaning
approach=semantic
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=512
chunk_overlap=100
use_gleaning=false
gleaning_rounds=0
llm_model=qwen3:8b-q4_k_m
llm_temp=0.1
ollama_enabled=true
lightrag=false
leiden=false
cross_encoder=false
PARAMS
}

# --- Hybrid pipelines ---

pipeline_hybrid_qwen3() {
    cat <<-PARAMS
name=hybrid_qwen3
approach=hybrid
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=512
chunk_overlap=100
use_gleaning=true
gleaning_rounds=2
llm_model=qwen3:8b-q4_k_m
llm_temp=0.15
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=false
PARAMS
}

pipeline_hybrid_small_chunks() {
    cat <<-PARAMS
name=hybrid_small_chunks
approach=hybrid
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=256
chunk_overlap=50
use_gleaning=true
gleaning_rounds=1
llm_model=qwen3:8b-q4_k_m
llm_temp=0.1
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=false
PARAMS
}

# --- Future RAG 2026 (All features) ---
pipeline_future_rag_2026() {
    cat <<-PARAMS
name=future_rag_2026
approach=hybrid
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=512
chunk_overlap=100
use_gleaning=true
gleaning_rounds=2
llm_model=qwen3:8b-q4_k_m
llm_temp=0.1
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=true
PARAMS
}

# --- Mock Future RAG (Features without LLM) ---
pipeline_mock_future_rag() {
    cat <<-PARAMS
name=mock_future_rag
approach=algorithmic
embedding=hash
embed_model=none
embed_dim=384
chunk_size=256
chunk_overlap=50
use_gleaning=false
gleaning_rounds=0
llm_model=none
llm_temp=0.0
ollama_enabled=false
lightrag=true
leiden=true
cross_encoder=false
PARAMS
}

# --- Fast Future RAG (No Gleaning) ---
pipeline_fast_future_rag() {
    cat <<-PARAMS
name=fast_future_rag
approach=hybrid
embedding=ollama
embed_model=nomic-embed-text
embed_dim=768
chunk_size=2048
chunk_overlap=300
use_gleaning=false
gleaning_rounds=0
llm_model=qwen3:8b-q4_k_m
llm_temp=0.1
ollama_enabled=true
lightrag=true
leiden=true
cross_encoder=true
PARAMS
}


# All pipeline function names
ALL_PIPELINES=(
    pipeline_algo_hash_small
    pipeline_algo_hash_medium
    pipeline_algo_hash_large
    pipeline_algo_ollama_embed
    pipeline_semantic_qwen3
    pipeline_semantic_llama31
    pipeline_semantic_mistral
    pipeline_semantic_no_gleaning
    pipeline_hybrid_qwen3
    pipeline_hybrid_small_chunks
    pipeline_future_rag_2026
    pipeline_mock_future_rag
    pipeline_fast_future_rag
)

# =============================================================================
# Helper: parse pipeline params into associative array
# =============================================================================
parse_params() {
    local func="$1"
    declare -gA P=()
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        P["$key"]="$value"
    done < <("$func")
}

# =============================================================================
# Helper: extract JSON from CLI output (CLI mixes INFO logs with JSON on stdout)
# =============================================================================
extract_json() {
    grep '^{' | tail -1
}

# =============================================================================
# Helper: generate a JSON5 config file from pipeline parameters
# =============================================================================
generate_config() {
    local output_file="$1"
    local output_dir="$2"

    cat > "$output_file" <<EOF
{
  "mode": {
    "approach": "${P[approach]}"
  },
  "general": {
    "output_dir": "$output_dir",
    "log_level": "info",
    "max_threads": 4
  },
  "pipeline": {
    "workflows": [
      "extract_text",
      "extract_entities",
      "build_graph",
      "detect_communities"
    ],
    "parallel_execution": true,
    "text_extraction": {
      "chunk_size": ${P[chunk_size]},
      "chunk_overlap": ${P[chunk_overlap]},
      "min_chunk_size": 100,
      "clean_control_chars": true,
      "normalize_whitespace": true
    },
    "entity_extraction": {
      "model_name": "${P[llm_model]}",
      "temperature": ${P[llm_temp]},
      "max_tokens": 1200,
      "entity_types": ["PERSON", "CONCEPT", "LOCATION", "OBJECT", "EVENT", "RELATIONSHIP"],
      "confidence_threshold": 0.6
    },
    "graph_building": {
      "relation_scorer": "cosine_similarity",
      "min_relation_score": 0.4,
      "max_connections_per_node": 20,
      "bidirectional_relations": true
    },
    "community_detection": {
      "algorithm": "leiden",
      "resolution": 0.8,
      "min_community_size": 2,
      "max_community_size": 20
    }
  },
  "text_processing": {
    "enabled": true,
    "chunk_size": ${P[chunk_size]},
    "chunk_overlap": ${P[chunk_overlap]},
    "min_chunk_size": 100,
    "max_chunk_size": 1500,
    "normalize_whitespace": true,
    "remove_artifacts": true,
    "extract_keywords": true,
    "keyword_min_score": 0.15,
    "enrichment": {
      "enabled": true,
      "auto_detect_format": true,
      "parser_type": "auto",
      "extract_keywords": true,
      "max_keywords_per_chunk": 5,
      "use_tfidf": true,
      "generate_summaries": true,
      "min_chunk_length_for_summary": 150,
      "max_summary_length": 150,
      "extract_chapter": true,
      "extract_section": true,
      "extract_position": true,
      "calculate_confidence": true,
      "detect_headings": true,
      "detect_numbering": true,
      "detect_underlines": true,
      "detect_all_caps": true
    }
  },
  "entity_extraction": {
    "enabled": true,
    "min_confidence": 0.6,
    "use_gleaning": ${P[use_gleaning]},
    "max_gleaning_rounds": ${P[gleaning_rounds]},
    "gleaning_improvement_threshold": 0.08,
    "semantic_merging": true,
    "merge_similarity_threshold": 0.85,
    "automatic_linking": true,
    "linking_confidence_threshold": 0.7
  },
  "graph_construction": {
    "enabled": true,
    "incremental_updates": true,
    "use_pagerank": true,
    "pagerank_damping": 0.85,
    "pagerank_iterations": 50,
    "pagerank_convergence": 0.0001,
    "extract_relationships": true,
    "relationship_confidence_threshold": 0.5
  },
  "vector_processing": {
    "enabled": true,
    "embedding_model": "${P[embed_model]}",
    "embedding_dimensions": ${P[embed_dim]},
    "use_hnsw_index": true,
    "hnsw_ef_construction": 200,
    "hnsw_m": 16,
    "similarity_threshold": 0.7
  },
  "query_processing": {
    "enabled": true,
    "use_advanced_pipeline": true,
    "use_intent_classification": true,
    "use_concept_extraction": true,
    "use_temporal_parsing": false,
    "confidence_threshold": 0.5
  },
  "ollama": {
    "enabled": ${P[ollama_enabled]},
    "host": "http://localhost",
    "port": 11434,
    "chat_model": "${P[llm_model]}",
    "embedding_model": "${P[embed_model]}",
    "timeout_seconds": 120,
    "max_retries": 3,
    "fallback_to_hash": true,
    "max_tokens": 1200,
    "temperature": ${P[llm_temp]},
    "generation": {
      "temperature": ${P[llm_temp]},
      "top_p": 0.9,
      "max_tokens": 1500,
      "stream": false
    }
  },
  "performance": {
    "batch_processing": true,
    "batch_size": 50,
    "worker_threads": 4,
    "memory_limit_mb": 4096,
    "cache_embeddings": true
  },
  "enhancements": {
    "enabled": true,
    "query_analysis": {
      "enabled": true,
      "min_confidence": 0.6,
      "enable_strategy_suggestion": true,
      "enable_keyword_analysis": true,
      "enable_complexity_scoring": true
    },
    "adaptive_retrieval": {
      "enabled": true,
      "use_query_analysis": true,
      "enable_cross_strategy_fusion": true,
      "diversity_threshold": 0.8,
      "enable_diversity_selection": true,
      "enable_confidence_weighting": true
    },
    "lightrag": {
      "enabled": ${P[lightrag]},
      "max_keywords": 15,
      "high_level_weight": 0.6,
      "low_level_weight": 0.4,
      "merge_strategy": "weighted",
      "language": "English",
      "enable_cache": true
    },
    "leiden": {
      "enabled": ${P[leiden]},
      "max_cluster_size": 15,
      "use_lcc": true,
      "seed": 42,
      "resolution": 0.8,
      "max_levels": 4,
      "min_improvement": 0.001,
      "enable_hierarchical": true,
      "generate_summaries": true,
      "max_summary_length": 5,
      "use_extractive_summary": true,
      "adaptive_routing": {
        "enabled": true,
        "default_level": 1,
        "keyword_weight": 0.5,
        "length_weight": 0.3,
        "entity_weight": 0.2
      }
    },
    "cross_encoder": {
      "enabled": ${P[cross_encoder]},
      "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "max_length": 512,
      "batch_size": 32,
      "top_k": 10,
      "min_confidence": 0.0,
      "normalize_scores": true
    },
    "enhanced_function_registry": {
      "enabled": false,
      "categorization": false,
      "usage_statistics": false,
      "dynamic_registration": false,
      "performance_monitoring": false,
      "recommendation_system": false
    },
    "performance_benchmarking": {
      "enabled": false,
      "auto_recommendations": false,
      "comprehensive_testing": false,
      "iterations": 1,
      "include_parallel": false,
      "enable_memory_profiling": false
    }
  }
}
EOF
}

# =============================================================================
# Helper: run a single pipeline against a single book
# =============================================================================
run_pipeline_book() {
    local pipeline_func="$1"
    local book_key="$2"
    local book_path="${BOOKS[$book_key]}"

    parse_params "$pipeline_func"
    local pipeline_name="${P[name]}"
    local run_id="${pipeline_name}__${book_key}"
    local config_file="$CONFIGS_DIR/${run_id}.json5"
    local output_dir="$RESULTS_DIR/output_${run_id}"
    local result_file="$RESULTS_DIR/${run_id}.json"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Pipeline:${NC} ${pipeline_name}"
    echo -e "${BLUE}Book:${NC}     ${book_key}"
    echo -e "${BLUE}Approach:${NC} ${P[approach]} | Embed: ${P[embedding]}/${P[embed_model]}"
    echo -e "${BLUE}Chunks:${NC}   ${P[chunk_size]}/${P[chunk_overlap]} | Gleaning: ${P[use_gleaning]}×${P[gleaning_rounds]}"
    echo -e "${BLUE}LLM:${NC}     ${P[llm_model]} (T=${P[llm_temp]})"
    echo -e "${BLUE}Features:${NC} LightRAG=${P[lightrag]} Leiden=${P[leiden]} CrossEnc=${P[cross_encoder]}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Generate config
    mkdir -p "$output_dir"
    generate_config "$config_file" "$output_dir"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY-RUN] Config generated: $config_file${NC}"
        # Write a minimal result JSON for dry-run
        cat > "$result_file" <<EOF
{
  "run_id": "$run_id",
  "pipeline": "$pipeline_name",
  "book": "$book_key",
  "dry_run": true,
  "config_file": "$config_file",
  "parameters": {
    "approach": "${P[approach]}",
    "embedding_backend": "${P[embedding]}",
    "embedding_model": "${P[embed_model]}",
    "embedding_dim": ${P[embed_dim]},
    "chunk_size": ${P[chunk_size]},
    "chunk_overlap": ${P[chunk_overlap]},
    "use_gleaning": ${P[use_gleaning]},
    "gleaning_rounds": ${P[gleaning_rounds]},
    "llm_model": "${P[llm_model]}",
    "llm_temperature": ${P[llm_temp]},
    "ollama_enabled": ${P[ollama_enabled]},
    "lightrag": ${P[lightrag]},
    "leiden": ${P[leiden]},
    "cross_encoder": ${P[cross_encoder]}
  }
}
EOF
        return 0
    fi

    # ── Single Phase: Benchmark (Init + Load + Query in-memory) ──
    echo -e "  ${GREEN}[1/1]${NC} Running benchmark..."
    
    local questions_str="${QUESTIONS[$book_key]}"
    local bench_output
    
    # Run the benchmark command
    # extract_json ensures we only get the JSON line even if logs leak
    if ! bench_output=$(RUST_LOG=info $CLI_BIN --format json bench --config "$config_file" --book "$book_path" --questions "$questions_str" | extract_json); then
        echo -e "  ${RED}[FAIL] Benchmark failed${NC}"
        write_error_result "$result_file" "$run_id" "$pipeline_name" "$book_key" "bench_failed"
        return 1
    fi

    # ── Write full result JSON ──
    # We take the bench output and wrap it with run metadata
    # Bench output has: {config_file, book_file, timing, stats, questions_and_answers}
    
    jq -n \
        --arg run_id "$run_id" \
        --arg pipeline "$pipeline_name" \
        --arg book "$book_key" \
        --arg timestamp "$(date -Iseconds)" \
        --argjson bench "$bench_output" \
        --argjson params "$(jq -n \
            --arg approach "${P[approach]}" \
            --arg embed_backend "${P[embedding]}" \
            --arg embed_model "${P[embed_model]}" \
            --argjson embed_dim "${P[embed_dim]}" \
            --argjson chunk_size "${P[chunk_size]}" \
            --argjson chunk_overlap "${P[chunk_overlap]}" \
            --argjson use_gleaning "${P[use_gleaning]}" \
            --argjson gleaning_rounds "${P[gleaning_rounds]}" \
            --arg llm_model "${P[llm_model]}" \
            --argjson llm_temp "${P[llm_temp]}" \
            --argjson ollama_enabled "${P[ollama_enabled]}" \
            --argjson lightrag "${P[lightrag]}" \
            --argjson leiden "${P[leiden]}" \
            --argjson cross_encoder "${P[cross_encoder]}" \
            '{
              approach: $approach,
              embedding_backend: $embed_backend,
              embedding_model: $embed_model,
              embedding_dim: $embed_dim,
              chunk_size: $chunk_size,
              chunk_overlap: $chunk_overlap,
              use_gleaning: $use_gleaning,
              gleaning_rounds: $gleaning_rounds,
              llm_model: $llm_model,
              llm_temperature: $llm_temp,
              ollama_enabled: $ollama_enabled,
              lightrag: $lightrag,
              leiden: $leiden,
              cross_encoder: $cross_encoder
            }')" \
        '{
          run_id: $run_id,
          pipeline: $pipeline,
          book: $book,
          book_path: $bench.book_file,
          config_file: $bench.config_file,
          timestamp: $timestamp,
          parameters: $params,
          timing: $bench.timing,
          stats: $bench.stats,
          questions_and_answers: $bench.questions_and_answers
        }' > "$result_file"

    local total_ms=$(echo "$bench_output" | jq -r '.timing.total_ms')
    local build_ms=$(echo "$bench_output" | jq -r '.timing.build_ms')
    local query_ms=$(echo "$bench_output" | jq -r '.timing.total_query_ms')
    local entities=$(echo "$bench_output" | jq -r '.stats.entities')
    local rels=$(echo "$bench_output" | jq -r '.stats.relationships')

    echo -e "${GREEN}  ✅ Result saved: $result_file${NC}"
    echo -e "  📊 Entities: ${entities} | Relations: ${rels}"
    echo -e "  ⏱  Total: ${total_ms}ms (build: ${build_ms}ms, queries: ${query_ms}ms)"
    echo ""
}

# =============================================================================
# Helper: write error result
# =============================================================================
write_error_result() {
    local file="$1" run_id="$2" pipeline="$3" book="$4" error="$5"
    jq -n \
        --arg run_id "$run_id" \
        --arg pipeline "$pipeline" \
        --arg book "$book" \
        --arg error "$error" \
        --arg ts "$(date -Iseconds)" \
        '{run_id: $run_id, pipeline: $pipeline, book: $book, error: $error, timestamp: $ts}' > "$file"
}

# =============================================================================
# CLI argument parsing
# =============================================================================
FILTER=""
DRY_RUN="false"
SINGLE_PIPELINE=""
BOOK_FILTER=""
LIST_ONLY="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)   FILTER="$2"; shift 2 ;;
        --dry-run)  DRY_RUN="true"; shift ;;
        --pipeline) SINGLE_PIPELINE="$2"; shift 2 ;;
        --book)     BOOK_FILTER="$2"; shift 2 ;;
        --list)     LIST_ONLY="true"; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --filter <pattern>    Only run pipelines matching pattern"
            echo "  --pipeline <name>     Run a single pipeline by name"
            echo "  --book <key>          Only test against one book (symposium|tom_sawyer)"
            echo "  --dry-run             Generate configs only, no execution"
            echo "  --list                List all defined pipelines"
            echo "  --help                Show this help"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# List mode
# =============================================================================
if [[ "$LIST_ONLY" == "true" ]]; then
    echo -e "${CYAN}Available Pipelines:${NC}"
    echo ""
    for func in "${ALL_PIPELINES[@]}"; do
        parse_params "$func"
        printf "  %-25s  approach=%-12s  embed=%-8s  llm=%-22s  chunks=%s/%s\n" \
            "${P[name]}" "${P[approach]}" "${P[embedding]}" "${P[llm_model]}" "${P[chunk_size]}" "${P[chunk_overlap]}"
    done
    echo ""
    echo "Books: ${!BOOKS[*]}"
    exit 0
fi

# =============================================================================
# Pre-flight checks
# =============================================================================
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   GraphRAG E2E Pipeline Benchmark Runner        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check CLI binary
if [[ ! -x "$CLI_BIN" ]]; then
    echo -e "${YELLOW}CLI binary not found at $CLI_BIN${NC}"
    echo -e "${YELLOW}Building release binary...${NC}"
    (cd "$PROJECT_ROOT" && cargo build --release -p graphrag-cli)
fi

# Check jq
if ! command -v jq &>/dev/null; then
    echo -e "${RED}ERROR: 'jq' is required. Install with: sudo apt install jq${NC}"
    exit 1
fi

# Check books exist
for key in "${!BOOKS[@]}"; do
    if [[ ! -f "${BOOKS[$key]}" ]]; then
        echo -e "${RED}ERROR: Book not found: ${BOOKS[$key]}${NC}"
        exit 1
    fi
done

mkdir -p "$CONFIGS_DIR" "$RESULTS_DIR" "$REPORTS_DIR"

# =============================================================================
# Main execution loop
# =============================================================================
run_count=0
fail_count=0
skip_count=0
total_start=$(date +%s)

for func in "${ALL_PIPELINES[@]}"; do
    parse_params "$func"
    local_name="${P[name]}"

    # Apply filters
    if [[ -n "$SINGLE_PIPELINE" && "$local_name" != "$SINGLE_PIPELINE" ]]; then
        continue
    fi
    if [[ -n "$FILTER" && "$local_name" != *"$FILTER"* ]]; then
        skip_count=$((skip_count + 1))
        continue
    fi

    # Determine books to test
    local_books=("${!BOOKS[@]}")
    if [[ -n "$BOOK_FILTER" ]]; then
        local_books=("$BOOK_FILTER")
    fi

    for book_key in "${local_books[@]}"; do
        run_count=$((run_count + 1))
        if ! run_pipeline_book "$func" "$book_key"; then
            fail_count=$((fail_count + 1))
        fi
    done
done

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "  Runs:    ${run_count}"
echo -e "  Failed:  ${fail_count}"
echo -e "  Skipped: ${skip_count}"
echo -e "  Time:    ${total_elapsed}s"
echo -e "  Results: ${RESULTS_DIR}/"
echo ""

# Auto-generate report if runs completed
if [[ "$run_count" -gt 0 && "$DRY_RUN" != "true" ]]; then
    echo -e "${BLUE}Generating report...${NC}"
    "$SCRIPT_DIR/generate_report.sh" || true
fi
