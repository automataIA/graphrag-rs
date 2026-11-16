# Troubleshooting builds

The default native facade and core features do not require an Objective-C
compiler. If a build reports an `objc_exception` or `cc1obj` error, inspect the
enabled feature graph: a browser, Metal, or GPU-related dependency may have
been enabled for a native build.

Start with the smallest supported build and add features one at a time:

```bash
cargo check -p graphrag --no-default-features --locked
cargo check -p graphrag-core --no-default-features --locked
```

For WASM, install `wasm32-unknown-unknown` and use the WASM package's documented
target command. For a native GPU feature, install the compiler and system SDK
required by that feature on the target platform. The error is a toolchain
configuration problem; it is not fixed by changing the indexed document.
