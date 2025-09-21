use graphrag_rs::{config::loader::load_config, GraphRAG, Result};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <config_file>", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[1];
    println!("Loading configuration from: {config_path}");

    let config = load_config(config_path)?;
    let mut graphrag = GraphRAG::new(config)?;

    println!("Initializing GraphRAG system...");
    graphrag.initialize()?;

    println!("GraphRAG system initialized successfully");

    // Example usage - in a real CLI you'd have proper command parsing
    match args.get(2).map(|s| s.as_str()) {
        Some("build") => {
            println!("Building knowledge graph...");
            graphrag.build_graph()?;
            println!("Knowledge graph built successfully");
        }
        Some("query") => {
            if let Some(query) = args.get(3) {
                println!("Processing query: {query}");
                let results = graphrag.query(query)?;
                println!("Results: {results:#?}");
            } else {
                eprintln!("No query provided");
                std::process::exit(1);
            }
        }
        _ => {
            println!("Available commands: build, query <text>");
        }
    }

    Ok(())
}
