# Local Git hooks

Install these hooks for this checkout with:

```sh
git config core.hooksPath .githooks
```

This is a local Git configuration change and is not committed. Setting
`core.hooksPath` changes where Git looks for all hooks, so existing hooks in
`.git/hooks` or another configured hooks directory will no longer run. To
retain them, copy or invoke their scripts from the corresponding `.githooks`
hooks, or run the existing checks separately. Git does not chain multiple
hooks directories.

The pre-commit hook checks staged path names. If any staged file is Rust source
or a Cargo/rustfmt manifest or configuration file, it runs:

```sh
cargo +1.89.0 fmt --all -- --check
```

The check reads the current working tree, as `cargo fmt` does; it does not
format or reconstruct the staged blob. Unstaged edits to Rust files can
therefore affect the result, and a clean result does not prove that the staged
snapshot alone is formatted if staged and working-tree content differ. Stage
the intended formatting together with the code, or inspect the index and
working tree separately when they differ.

The pre-push hook is a no-op unless enabled for that push:

```sh
GRAPHRAG_RUN_PRE_PUSH_CHECKS=1 git push
```

For the same heavier checks without pushing, run:

```sh
scripts/ci/check-pre-push.sh
```

That script runs the Clippy feature matrix and tests/checks from the repository
CI workflow. Both scripts pin Cargo to `+1.89.0`, matching CI; install that
toolchain and the `rustfmt` and `clippy` components to use them.
