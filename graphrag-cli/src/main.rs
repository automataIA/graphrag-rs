//! GraphRAG CLI binary entry point.
//!
//! All logic lives in the library crate (`graphrag_cli::run`).

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> color_eyre::eyre::Result<()> {
    // color_eyre is installed once inside `run()` (shared with the `graphrag`
    // meta-crate binary). Installing here too double-installs and aborts.
    graphrag_cli::run().await
}
