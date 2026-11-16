#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

has_rust_change=0
while IFS= read -r -d '' path; do
  case "$path" in
    *.rs|Cargo.toml|*/Cargo.toml|Cargo.lock|*/Cargo.lock|rustfmt.toml|*/rustfmt.toml|.rustfmt.toml|*/.rustfmt.toml)
      has_rust_change=1
      break
      ;;
  esac
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

if (( has_rust_change )); then
  cargo +1.89.0 fmt --all -- --check
fi
