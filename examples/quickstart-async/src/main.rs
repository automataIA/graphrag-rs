use std::{error::Error, fs};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let path = format!(
        "{}/../quickstart/document.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = fs::read_to_string(path)?;
    let mut rag = graphrag::GraphRAG::quick_start(&text).await?;
    println!(
        "{}",
        rag.ask("Who maintains the Prisma project?").await?
    );
    Ok(())
}
