#!/usr/bin/env bash
# Compare mockup and WASM chat shell side-by-side via playwright-cli.
# Required: trunk serve already running on http://127.0.0.1:8080 and mockup at
# /home/dio/graphrag-rs/Chat discussion.html.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="${HERE}/artifacts"
mkdir -p "${ART}"

MOCKUP_URL="http://127.0.0.1:8181/Chat%20discussion.html"
WASM_URL="http://127.0.0.1:8080/"
SESS_M="mockup"
SESS_W="wasm"

cleanup() {
  playwright-cli -s=${SESS_M} close >/dev/null 2>&1 || true
  playwright-cli -s=${SESS_W} close >/dev/null 2>&1 || true
}
trap cleanup EXIT

pass=0
fail=0
check() {
  local sess="$1"; local sel="$2"; local label="$3"
  local out
  out=$(playwright-cli -s=${sess} eval "() => !!document.querySelector('${sel}')")
  if [[ "${out}" == *"true"* ]]; then
    echo "  ✓ ${label}: ${sel}"
    pass=$((pass+1))
  else
    echo "  ✗ ${label}: ${sel}"
    fail=$((fail+1))
  fi
}

echo "── Mockup (file://) ──────────────────────────────"
playwright-cli -s=${SESS_M} open "${MOCKUP_URL}" >/dev/null
playwright-cli -s=${SESS_M} resize 1440 900 >/dev/null
sleep 1
playwright-cli -s=${SESS_M} screenshot --filename "${ART}/mockup.png" >/dev/null
check ${SESS_M} ".app"                      "shell"
check ${SESS_M} ".rail-left .doc-item"      "doc-item"
check ${SESS_M} ".stage .stage-head .stage-title" "stage title"
check ${SESS_M} ".stage .bubble-q"          "user bubble"
check ${SESS_M} ".stage .answer .cite"      "citation"
check ${SESS_M} ".rail-right .graph-frame svg" "subgraph svg"
check ${SESS_M} ".rail-right .stages .pls" "pipeline rows"
check ${SESS_M} ".rail-right .ref-block .ref-card" "ref-card"
check ${SESS_M} ".composer input"           "composer input"

echo ""
echo "── WASM SPA (http://127.0.0.1:8080) ──────────────"
playwright-cli -s=${SESS_W} open "${WASM_URL}" >/dev/null
playwright-cli -s=${SESS_W} resize 1440 900 >/dev/null
# Wait for Leptos mount
for i in $(seq 1 20); do
  out=$(playwright-cli -s=${SESS_W} eval "() => !!document.querySelector('.app')" 2>/dev/null || echo "")
  [[ "${out}" == *"true"* ]] && break
  sleep 1
done
playwright-cli -s=${SESS_W} screenshot --filename "${ART}/wasm.png" >/dev/null
check ${SESS_W} ".app"                      "shell"
check ${SESS_W} ".rail-left"                "left rail"
check ${SESS_W} ".rail-left .brand"         "brand mark"
check ${SESS_W} ".stage .stage-head"        "stage head"
check ${SESS_W} ".stage .stage-head .stage-title" "stage title"
check ${SESS_W} ".rail-right .stages .pls" "pipeline rows"
check ${SESS_W} ".rail-right .graph-frame" "graph frame"
check ${SESS_W} ".rail-right .ref-block"   "ref block"
check ${SESS_W} ".composer input"          "composer input"
check ${SESS_W} ".composer .prompt"        "prompt prefix"

echo ""
echo "── Result ────────────────────────────────────────"
echo "  pass: ${pass}    fail: ${fail}"
echo "  artifacts: ${ART}/mockup.png  ${ART}/wasm.png"

[[ "${fail}" -eq 0 ]]
