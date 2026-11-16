use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let path = format!(
        "{}/../quickstart/document.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = fs::read_to_string(path)?;
    let mut rag = graphrag::GraphRAG::quick_start_sync(&text)?;
    println!("{}", rag.ask_sync("Who maintains the Prisma project?")?);
    Ok(())
}
