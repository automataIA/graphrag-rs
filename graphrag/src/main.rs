//! `graphrag` binary entry point — delegates to graphrag-cli.

#[tokio::main]
async fn main() -> color_eyre::eyre::Result<()> {
    graphrag_cli::run().await
}
