#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

cargo +1.89.0 clippy -p graphrag-core --no-default-features --features memory-storage,basic-retrieval --all-targets --locked -- -D warnings
cargo +1.89.0 clippy -p graphrag-core --all-targets --locked -- -D warnings
cargo +1.89.0 clippy -p graphrag --no-default-features --features memory-storage,basic-retrieval --all-targets --locked -- -D warnings
cargo +1.89.0 clippy -p graphrag --all-targets --locked -- -D warnings
cargo +1.89.0 clippy -p graphrag-cli --all-targets --locked -- -D warnings
cargo +1.89.0 clippy -p graphrag-server --no-default-features --all-targets --locked -- -D warnings
cargo +1.89.0 clippy -p graphrag-server --all-targets --locked -- -D warnings

cargo +1.89.0 test -p graphrag-core --tests --locked
cargo +1.89.0 test -p graphrag-core --doc --locked
cargo +1.89.0 test -p graphrag --lib --no-default-features --features memory-storage,basic-retrieval --locked
cargo +1.89.0 test -p graphrag --lib --locked
cargo +1.89.0 test -p graphrag-cli --lib --locked
cargo +1.89.0 test -p graphrag-server --no-default-features --locked
cargo +1.89.0 test -p graphrag-server --locked
cargo +1.89.0 test -p graphrag --features cli --test quickstart_cli --locked
cargo +1.89.0 check --manifest-path examples/quickstart-sync/Cargo.toml --locked
cargo +1.89.0 check --manifest-path examples/quickstart-async/Cargo.toml --locked
cargo +1.89.0 test -p graphrag --doc --locked
cargo +1.89.0 test -p graphrag-cli --doc --locked
cargo +1.89.0 test -p graphrag-server --doc --no-default-features --locked
cargo +1.89.0 test -p graphrag-server --doc --locked
