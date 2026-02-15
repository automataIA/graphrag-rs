#!/usr/bin/env bash
# =============================================================================
# GraphRAG E2E Benchmark Report Generator
# =============================================================================
# Reads results/*.json and generates:
#   1. reports/benchmark_report.json  — full combined JSON
#   2. reports/benchmark_report.md    — human-readable markdown
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
REPORTS_DIR="$SCRIPT_DIR/reports"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

if ! command -v jq &>/dev/null; then
    echo -e "${RED}ERROR: 'jq' is required. Install with: sudo apt install jq${NC}"
    exit 1
fi

# Count result files
result_files=("$RESULTS_DIR"/*.json)
if [[ ! -f "${result_files[0]}" ]]; then
    echo -e "${RED}No result files found in $RESULTS_DIR${NC}"
    exit 1
fi

num_results=${#result_files[@]}
echo -e "${CYAN}Generating report from ${num_results} result files...${NC}"

# =============================================================================
# 1. Combined JSON report
# =============================================================================
COMBINED_JSON="$REPORTS_DIR/benchmark_report.json"

jq -n \
    --arg generated "$(date -Iseconds)" \
    --argjson results "$(jq -s '.' "${result_files[@]}")" \
    '{
      report_generated: $generated,
      total_runs: ($results | length),
      runs: $results,
      summary: {
        pipelines: ($results | [.[].pipeline] | unique),
        books: ($results | [.[].book] | unique),
        approaches: ($results | [.[].parameters.approach // "unknown"] | unique),
        llm_models: ($results | [.[].parameters.llm_model // "none"] | unique),
        total_questions_answered: ($results | [.[].questions_and_answers // [] | length] | add // 0)
      }
    }' > "$COMBINED_JSON"

echo -e "${GREEN}  ✅ JSON report: $COMBINED_JSON${NC}"

# =============================================================================
# 2. Markdown report
# =============================================================================
REPORT_MD="$REPORTS_DIR/benchmark_report.md"

{
    echo "# GraphRAG Pipeline Benchmark Report"
    echo ""
    echo "> Generated: $(date -Iseconds)"
    echo ""

    # Summary table
    echo "## Summary"
    echo ""
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Total Runs | ${num_results} |"

    local_approaches=$(jq -r '[.[].parameters.approach // "?"] | unique | join(", ")' < <(jq -s '.' "${result_files[@]}"))
    local_models=$(jq -r '[.[].parameters.llm_model // "none"] | unique | join(", ")' < <(jq -s '.' "${result_files[@]}"))
    echo "| Approaches | ${local_approaches} |"
    echo "| LLM Models | ${local_models} |"
    echo ""

    # Pipeline comparison table
    echo "## Pipeline Comparison"
    echo ""
    echo "| Pipeline | Book | Approach | LLM | Chunks | Entities | Relations | Build (ms) | Query (ms) |"
    echo "|----------|------|----------|-----|--------|----------|-----------|------------|------------|"

    for f in "${result_files[@]}"; do
        # skip errors / dry-runs
        if jq -e '.error // .dry_run' "$f" &>/dev/null; then
            local_pipe=$(jq -r '.pipeline' "$f")
            local_book=$(jq -r '.book' "$f")
            local_status=$(jq -r '.error // "dry-run"' "$f")
            echo "| ${local_pipe} | ${local_book} | — | — | — | — | — | ${local_status} | — |"
            continue
        fi

        local_pipe=$(jq -r '.pipeline' "$f")
        local_book=$(jq -r '.book' "$f")
        local_approach=$(jq -r '.parameters.approach' "$f")
        local_llm=$(jq -r '.parameters.llm_model' "$f")
        local_chunks=$(jq -r '"\(.parameters.chunk_size)/\(.parameters.chunk_overlap)"' "$f")
        local_entities=$(jq -r '.stats.entities // "?"' "$f")
        local_rels=$(jq -r '.stats.relationships // "?"' "$f")
        local_build=$(jq -r '.timing.build_ms // "?"' "$f")
        local_query=$(jq -r '.timing.total_query_ms // "?"' "$f")

        echo "| ${local_pipe} | ${local_book} | ${local_approach} | ${local_llm} | ${local_chunks} | ${local_entities} | ${local_rels} | ${local_build} | ${local_query} |"
    done
    echo ""

    # Detailed Q&A per pipeline
    echo "## Detailed Q&A Results"
    echo ""

    for f in "${result_files[@]}"; do
        if jq -e '.error // .dry_run' "$f" &>/dev/null; then
            continue
        fi

        local_pipe=$(jq -r '.pipeline' "$f")
        local_book=$(jq -r '.book' "$f")

        echo "### ${local_pipe} — ${local_book}"
        echo ""
        echo "**Parameters:**"
        echo "- Approach: $(jq -r '.parameters.approach' "$f")"
        echo "- LLM: $(jq -r '.parameters.llm_model' "$f") (T=$(jq -r '.parameters.llm_temperature' "$f"))"
        echo "- Embedding: $(jq -r '.parameters.embedding_backend' "$f")/$(jq -r '.parameters.embedding_model' "$f")"
        echo "- Chunks: $(jq -r '.parameters.chunk_size' "$f")/$(jq -r '.parameters.chunk_overlap' "$f")"
        echo "- Gleaning: $(jq -r '.parameters.use_gleaning' "$f") (rounds=$(jq -r '.parameters.gleaning_rounds' "$f"))"
        echo "- LightRAG: $(jq -r '.parameters.lightrag' "$f") | Leiden: $(jq -r '.parameters.leiden' "$f") | CrossEncoder: $(jq -r '.parameters.cross_encoder' "$f")"
        echo ""

        num_qa=$(jq '.questions_and_answers | length' "$f")
        if [[ "$num_qa" -gt 0 ]]; then
            echo "| # | Question | Answer (truncated) | Time (ms) |"
            echo "|---|----------|-------------------|-----------|"

            jq -r '.questions_and_answers[] | "| \(.index) | \(.question | .[0:80]) | \(.answer | .[0:120] | gsub("\n"; " ")) | \(.query_time_ms) |"' "$f"
        else
            echo "*No Q&A results*"
        fi

        echo ""
    done

    # Cross-pipeline answer comparison
    echo "## Cross-Pipeline Answer Comparison"
    echo ""
    echo "Compare how different pipelines answer the same questions:"
    echo ""

    # Use the already-generated combined JSON
    local_combined=$(jq -s '.' "${result_files[@]}")

    for book in $(echo "$local_combined" | jq -r '[.[].book] | unique[]'); do
        echo "### Book: ${book}"
        echo ""

        # Get valid (non-error, non-dry-run) results for this book
        local_book_results=$(echo "$local_combined" | jq --arg b "$book" '[.[] | select(.book == $b and .error == null and .dry_run == null)]')
        local_count=$(echo "$local_book_results" | jq 'length')

        if [[ "$local_count" -eq 0 ]]; then
            echo "*No results for this book*"
            echo ""
            continue
        fi

        # Get questions from first result
        local_num_q=$(echo "$local_book_results" | jq '.[0].questions_and_answers | length')

        for q_idx in $(seq 0 $((local_num_q - 1))); do
            local_q=$(echo "$local_book_results" | jq -r --argjson i "$q_idx" '.[0].questions_and_answers[$i].question')
            echo "**Q$((q_idx+1)): ${local_q}**"
            echo ""
            echo "| Pipeline | Approach | LLM | Answer (first 150 chars) | Time |"
            echo "|----------|----------|-----|--------------------------|------|"

            echo "$local_book_results" | jq -r --argjson i "$q_idx" '.[] | "| \(.pipeline) | \(.parameters.approach) | \(.parameters.llm_model) | \(.questions_and_answers[$i].answer // "N/A" | .[0:150] | gsub("\n"; " ")) | \(.questions_and_answers[$i].query_time_ms // "?")ms |"'
            echo ""
        done
    done

} > "$REPORT_MD"

echo -e "${GREEN}  ✅ Markdown report: $REPORT_MD${NC}"

# Print quick summary
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Report generated!${NC}"
echo -e "  📊 JSON:     $COMBINED_JSON"
echo -e "  📝 Markdown: $REPORT_MD"
echo ""

# Print quick comparison to terminal
echo -e "${CYAN}Quick Comparison:${NC}"
echo ""
jq -r '.runs[] | select(.error == null and .dry_run == null) | "  \(.pipeline) × \(.book): entities=\(.stats.entities // "?") rels=\(.stats.relationships // "?") build=\(.timing.build_ms // "?")ms"' "$COMBINED_JSON" 2>/dev/null || true
