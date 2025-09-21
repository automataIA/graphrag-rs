//! Smart GraphRAG CLI
//! Usage:
//!   cargo run --bin simple_cli -- <config.toml> "Your question here"
//!
//! Automatically detects if GraphRAG needs to be built or if it can query directly

use graphrag_rs::config::TomlConfig;
use std::env;
use std::fs;
use std::process::Command;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Smart GraphRAG CLI");
        eprintln!("==================");
        eprintln!("Usage:");
        eprintln!("  {} <config.toml> [\"Your question here\"]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} config.toml", args[0]);
        eprintln!("  {} config.toml \"What are the main themes?\"", args[0]);
        eprintln!();
        eprintln!("How it works:");
        eprintln!("- If no question provided: builds GraphRAG if needed");
        eprintln!("- If question provided: builds GraphRAG if needed, then answers");
        std::process::exit(1);
    }

    let config_path = &args[1];
    let query = args.get(2);

    println!("🎯 Smart GraphRAG CLI");
    println!("=====================");
    println!("📝 Config: {config_path}");

    // Load and validate config
    let mut config = TomlConfig::from_file(config_path)?;

    // Auto-generate output directory based on document name
    if let Some(doc_path) = &config.general.input_document_path {
        if !std::path::Path::new(doc_path).exists() {
            eprintln!("❌ Document not found: {doc_path}");
            std::process::exit(1);
        }

        // Extract filename without extension and create clean directory name
        let filename = std::path::Path::new(doc_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        // Clean filename: replace spaces with underscores, remove special chars
        let clean_name = filename
            .replace([' ', '.'], "_")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect::<String>()
            .to_lowercase();

        // Override output directory with auto-generated name
        config.general.output_dir = format!("./output/{clean_name}");
    }

    println!("✅ Configuration loaded successfully");
    println!("  📊 Document: {:?}", config.general.input_document_path);
    println!(
        "  📁 Output: {} (auto-generated)",
        config.general.output_dir
    );

    let output_dir = &config.general.output_dir;
    let entities_file = format!("{output_dir}/entities.json");
    let graphrag_exists = std::path::Path::new(&entities_file).exists();

    // Determine what to do
    match (graphrag_exists, query) {
        (false, None) => {
            // Need to build, no query
            println!("🔨 GraphRAG not found. Building knowledge graph...");
            build_graphrag(config_path, &config).await?;
            println!("✅ GraphRAG ready! You can now ask questions:");
            println!("   {} {} \"Your question here\"", args[0], config_path);
        }
        (false, Some(question)) => {
            // Need to build, then query
            println!("🔨 GraphRAG not found. Building knowledge graph first...");
            build_graphrag(config_path, &config).await?;
            println!("🔍 Now answering your question...");
            query_graphrag(config_path, &config, question).await?;
        }
        (true, None) => {
            // GraphRAG exists, no query
            println!("✅ GraphRAG already exists in: {output_dir}");
            println!("💡 Ask a question:");
            println!("   {} {} \"Your question here\"", args[0], config_path);
        }
        (true, Some(question)) => {
            // GraphRAG exists, answer query
            println!("✅ GraphRAG found. Answering your question...");
            query_graphrag(config_path, &config, question).await?;
        }
    }

    Ok(())
}

async fn build_graphrag(
    _config_path: &str,
    config: &TomlConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("▶️ Running GraphRAG builder...");

    // Create a temporary config with the updated output directory
    let temp_config_path = "temp_config_auto.toml";
    let config_content =
        toml::to_string_pretty(config).map_err(|e| format!("Failed to serialize config: {e}"))?;
    fs::write(temp_config_path, config_content)?;

    // Use the graphrag_cli
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "graphrag_cli",
            "--",
            temp_config_path,
            "build",
        ])
        .output()?;

    if !output.status.success() {
        eprintln!("❌ Build failed!");
        eprintln!("Error: {}", String::from_utf8_lossy(&output.stderr));
        return Err("Build failed".into());
    }

    // Clean up temporary config
    fs::remove_file(temp_config_path)?;

    println!("✅ GraphRAG built successfully!");
    Ok(())
}

async fn query_graphrag(
    _config_path: &str,
    config: &TomlConfig,
    question: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("▶️ Querying: \"{question}\"");

    // Create a temporary config with the updated output directory
    let temp_config_path = "temp_config_query.toml";
    let config_content =
        toml::to_string_pretty(config).map_err(|e| format!("Failed to serialize config: {e}"))?;
    fs::write(temp_config_path, config_content)?;

    // Use the graphrag_cli instead of hardcoded query example
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "graphrag_cli",
            "--",
            temp_config_path,
            "query",
            question,
        ])
        .output()?;

    // Clean up temporary config
    fs::remove_file(temp_config_path)?;

    if !output.status.success() {
        eprintln!("❌ Query failed!");
        eprintln!("Error: {}", String::from_utf8_lossy(&output.stderr));
        return Err("Query failed".into());
    }

    // Print the output
    print!("{}", String::from_utf8_lossy(&output.stdout));

    Ok(())
}
