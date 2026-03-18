//! GraphRAG CLI binary entry point.
//!
//! All logic lives in the library crate (`graphrag_cli::run`).

#[tokio::main]
async fn main() -> color_eyre::eyre::Result<()> {
    graphrag_cli::run().await
}
