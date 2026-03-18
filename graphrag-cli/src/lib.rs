//! GraphRAG CLI library entry point.
//!
//! Exposes [`run`] so the `graphrag` meta-crate (and tests) can invoke the
//! full CLI without going through a subprocess.

pub mod action;
pub mod app;
pub mod commands;
pub mod config;
pub mod handlers;
pub mod mode;
pub mod query_history;
pub mod theme;
pub mod tui;
pub mod ui;
pub mod workspace;

use app::App;
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use std::path::PathBuf;

// ──────────────────────────────────────────────────────────────────────────────
// CLI types
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "graphrag")]
#[command(version, about = "Modern Terminal UI for GraphRAG operations", long_about = None)]
#[command(author = "GraphRAG Contributors")]
pub struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Workspace name
    #[arg(short, long)]
    pub workspace: Option<String>,

    /// Enable debug logging
    #[arg(short, long)]
    pub debug: bool,

    /// Output format: text (default) or json (for scripting/CI)
    #[arg(long, default_value = "text", value_parser = ["text", "json"])]
    pub format: String,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start interactive TUI (default)
    Tui,

    /// Interactive setup wizard - creates graphrag.toml with guided configuration
    Setup {
        /// Template to use: general, legal, medical, financial, technical
        #[arg(short, long)]
        template: Option<String>,

        /// Output path for configuration file
        #[arg(short, long, default_value = "./graphrag.toml")]
        output: PathBuf,
    },

    /// Validate a configuration file (TOML or JSON5)
    Validate {
        /// Path to the configuration file to validate
        config_file: PathBuf,
    },

    /// Initialize GraphRAG with configuration (deprecated: prefer TUI with /config)
    Init {
        /// Configuration file path
        config: PathBuf,
    },

    /// Load a document into the knowledge graph (deprecated: prefer TUI with /load)
    Load {
        /// Document file path
        document: PathBuf,

        /// Configuration file (required if not already initialized)
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Execute a query (deprecated: prefer TUI)
    Query {
        /// Query text
        query: String,

        /// Configuration file (required if not already initialized)
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// List entities in the knowledge graph (deprecated: prefer TUI with /entities)
    Entities {
        /// Filter by name or type
        filter: Option<String>,

        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Configuration file
    Stats {
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Run full E2E benchmark (Init -> Load -> Query) in memory
    Bench {
        /// Configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Book text file
        #[arg(short, long)]
        book: PathBuf,

        /// Pipe-separated list of questions e.g. "Q1?|Q2?"
        #[arg(short, long)]
        questions: String,
    },

    /// Workspace management commands
    Workspace {
        #[command(subcommand)]
        action: WorkspaceCommands,
    },
}

#[derive(Subcommand)]
pub enum WorkspaceCommands {
    /// List all workspaces
    List,

    /// Create a new workspace
    Create { name: String },

    /// Show workspace information
    Info { id: String },

    /// Delete a workspace
    Delete { id: String },
}

// ──────────────────────────────────────────────────────────────────────────────
// Public entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Run the full GraphRAG CLI. Called by both the `graphrag-cli` binary and
/// the `graphrag` meta-crate binary.
pub async fn run() -> Result<()> {
    install_panic_hook();

    let cli = Cli::parse();

    color_eyre::install()?;

    match cli.command {
        Some(Commands::Tui) | None => {
            run_tui(cli.config, cli.workspace).await?;
        },
        Some(Commands::Setup { template, output }) => {
            run_setup_wizard(template, output).await?;
        },
        Some(Commands::Validate { config_file }) => {
            setup_logging(cli.debug)?;
            run_validate(&config_file, &cli.format)?;
        },
        Some(Commands::Init { config }) => {
            setup_logging(cli.debug)?;
            eprintln!(
                "⚠️  `init` is deprecated. Prefer: graphrag tui --config {}",
                config.display()
            );

            let handler = handlers::graphrag::GraphRAGHandler::new();
            let cfg = load_config_from_file(&config).await?;
            handler.initialize(cfg).await?;

            if cli.format == "json" {
                println!(
                    "{}",
                    serde_json::json!({"status": "initialized", "config": config.display().to_string()})
                );
            } else {
                println!("✅ GraphRAG initialized with config: {}", config.display());
            }
        },
        Some(Commands::Load { document, config }) => {
            setup_logging(cli.debug)?;
            eprintln!(
                "⚠️  `load` is deprecated. Prefer: graphrag tui, then /load {}",
                document.display()
            );

            let handler = handlers::graphrag::GraphRAGHandler::new();
            let config_path = config.unwrap_or_else(|| PathBuf::from("./graphrag.toml"));
            let cfg = load_config_from_file(&config_path).await?;
            handler.initialize(cfg).await?;
            let result = handler.load_document_with_options(&document, false).await?;

            if cli.format == "json" {
                println!(
                    "{}",
                    serde_json::json!({"status": "loaded", "document": document.display().to_string(), "details": result})
                );
            } else {
                println!("✅ {}", result);
            }
        },
        Some(Commands::Query { query, config }) => {
            setup_logging(cli.debug)?;
            eprintln!(
                "⚠️  `query` is deprecated. Prefer: graphrag tui, then /query {}",
                query
            );

            let handler = handlers::graphrag::GraphRAGHandler::new();
            let config_path = config.unwrap_or_else(|| PathBuf::from("./graphrag.toml"));
            let cfg = load_config_from_file(&config_path).await?;
            handler.initialize(cfg).await?;

            let (answer, raw_results) = handler.query_with_raw(&query).await?;

            if cli.format == "json" {
                println!(
                    "{}",
                    serde_json::json!({"query": query, "answer": answer, "sources": raw_results})
                );
            } else {
                println!("📝 Query: {}\n", query);
                println!("💡 Answer:\n{}\n", answer);
                if !raw_results.is_empty() {
                    println!("📚 Sources:");
                    for (i, src) in raw_results.iter().enumerate() {
                        println!("   {}. {}", i + 1, src);
                    }
                }
            }
        },
        Some(Commands::Entities { filter, config }) => {
            setup_logging(cli.debug)?;
            eprintln!("⚠️  `entities` is deprecated. Prefer: graphrag tui, then /entities");

            let handler = handlers::graphrag::GraphRAGHandler::new();
            let config_path = config.unwrap_or_else(|| PathBuf::from("./graphrag.toml"));
            let cfg = load_config_from_file(&config_path).await?;
            handler.initialize(cfg).await?;
            let entities = handler.get_entities(filter.as_deref()).await?;

            if cli.format == "json" {
                let json_entities: Vec<serde_json::Value> = entities
                    .iter()
                    .map(|e| serde_json::json!({"name": e.name, "type": e.entity_type}))
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({"entities": json_entities, "count": entities.len()})
                );
            } else {
                println!("📊 Entities ({} found):\n", entities.len());
                for entity in &entities {
                    println!("   • {} [{}]", entity.name, entity.entity_type);
                }
            }
        },
        Some(Commands::Stats { config }) => {
            setup_logging(cli.debug)?;
            eprintln!("⚠️  `stats` is deprecated. Prefer: graphrag tui, then /stats");

            let handler = handlers::graphrag::GraphRAGHandler::new();
            let config_path = config.unwrap_or_else(|| PathBuf::from("./graphrag.toml"));
            let cfg = load_config_from_file(&config_path).await?;
            handler.initialize(cfg).await?;

            if let Some(stats) = handler.get_stats().await {
                if cli.format == "json" {
                    println!(
                        "{}",
                        serde_json::json!({
                            "entities": stats.entities,
                            "relationships": stats.relationships,
                            "documents": stats.documents,
                            "chunks": stats.chunks,
                        })
                    );
                } else {
                    println!("📊 Knowledge Graph Statistics:");
                    println!("   Entities:      {}", stats.entities);
                    println!("   Relationships: {}", stats.relationships);
                    println!("   Documents:     {}", stats.documents);
                    println!("   Chunks:        {}", stats.chunks);
                }
            } else if cli.format == "json" {
                println!(
                    "{}",
                    serde_json::json!({"error": "No knowledge graph built yet"})
                );
            } else {
                println!("⚠️  No knowledge graph built yet. Load documents first.");
            }
        },
        Some(Commands::Bench {
            config,
            book,
            questions,
        }) => {
            if !cli.debug {
                std::env::set_var("RUST_LOG", "error");
            }
            setup_logging(cli.debug)?;

            let q_vec: Vec<String> = questions.split('|').map(|s| s.to_string()).collect();
            handlers::bench::run_benchmark(&config, &book, q_vec).await?;
        },
        Some(Commands::Workspace { action }) => {
            setup_logging(cli.debug)?;
            handle_workspace_commands(action).await?;
        },
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

async fn load_config_from_file(path: &std::path::Path) -> Result<graphrag_core::Config> {
    config::load_config(path).await
}

fn run_validate(config_file: &std::path::Path, format: &str) -> Result<()> {
    use graphrag_core::config::json5_loader::{detect_config_format, ConfigFormat};
    use graphrag_core::config::setconfig::SetConfig;

    if !config_file.exists() {
        if format == "json" {
            println!(
                "{}",
                serde_json::json!({"valid": false, "error": format!("File not found: {}", config_file.display())})
            );
        } else {
            println!("❌ File not found: {}", config_file.display());
        }
        return Ok(());
    }

    let fmt = match detect_config_format(config_file) {
        Some(f) => f,
        None => {
            if format == "json" {
                println!(
                    "{}",
                    serde_json::json!({"valid": false, "error": "Unsupported file format. Use .toml, .json, or .json5"})
                );
            } else {
                println!("❌ Unsupported file format. Use .toml, .json, or .json5");
            }
            return Ok(());
        },
    };

    let content = std::fs::read_to_string(config_file)
        .map_err(|e| color_eyre::eyre::eyre!("Cannot read file: {}", e))?;

    let result: std::result::Result<SetConfig, String> = match fmt {
        ConfigFormat::Toml => toml::from_str(&content).map_err(|e| format!("{}", e)),
        ConfigFormat::Json => serde_json::from_str(&content).map_err(|e| format!("{}", e)),
        ConfigFormat::Json5 => {
            #[cfg(feature = "json5-support")]
            {
                json5::from_str(&content).map_err(|e| format!("{}", e))
            }
            #[cfg(not(feature = "json5-support"))]
            {
                Err("JSON5 support not enabled".to_string())
            }
        },
        ConfigFormat::Yaml => Err("YAML support not enabled".to_string()),
    };

    match result {
        Ok(set_config) => {
            let config = set_config.to_graphrag_config();
            if format == "json" {
                println!(
                    "{}",
                    serde_json::json!({
                        "valid": true,
                        "format": format!("{:?}", fmt),
                        "approach": set_config.mode.approach,
                        "ollama_enabled": config.ollama.enabled,
                        "chunk_size": config.chunk_size,
                    })
                );
            } else {
                println!("✅ Configuration is valid!");
                println!("   Format:    {:?}", fmt);
                println!("   Approach:  {}", set_config.mode.approach);
                println!(
                    "   Ollama:    {}",
                    if config.ollama.enabled { "enabled" } else { "disabled" }
                );
                println!("   Chunk size: {}", config.chunk_size);
            }
        },
        Err(err) => {
            if format == "json" {
                println!("{}", serde_json::json!({"valid": false, "error": err}));
            } else {
                println!("❌ Invalid configuration:\n   {}", err);
            }
        },
    }

    Ok(())
}

async fn run_tui(config_path: Option<PathBuf>, workspace: Option<String>) -> Result<()> {
    setup_tui_logging()?;
    let mut app = App::new(config_path, workspace)?;
    app.run().await?;
    Ok(())
}

async fn handle_workspace_commands(action: WorkspaceCommands) -> Result<()> {
    let workspace_manager = workspace::WorkspaceManager::new()?;

    match action {
        WorkspaceCommands::List => {
            let workspaces = workspace_manager.list_workspaces().await?;

            if workspaces.is_empty() {
                println!("No workspaces found.");
                println!("\nCreate a workspace with: graphrag workspace create <name>");
            } else {
                println!("Available workspaces:\n");
                for ws in workspaces {
                    println!("  📁 {} ({})", ws.name, ws.id);
                    println!("     Created: {}", ws.created_at.format("%Y-%m-%d %H:%M:%S"));
                    println!(
                        "     Last accessed: {}",
                        ws.last_accessed.format("%Y-%m-%d %H:%M:%S")
                    );
                    if let Some(ref cfg) = ws.config_path {
                        println!("     Config: {}", cfg.display());
                    }
                    println!();
                }
            }
        },
        WorkspaceCommands::Create { name } => {
            let workspace = workspace_manager.create_workspace(name.clone()).await?;
            println!("✅ Workspace created successfully!");
            println!("   Name: {}", workspace.name);
            println!("   ID:   {}", workspace.id);
            println!("\nUse it with: graphrag tui --workspace {}", workspace.id);
        },
        WorkspaceCommands::Info { id } => {
            match workspace_manager.load_metadata(&id).await {
                Ok(workspace) => {
                    println!("Workspace Information:\n");
                    println!("  Name: {}", workspace.name);
                    println!("  ID:   {}", workspace.id);
                    println!(
                        "  Created: {}",
                        workspace.created_at.format("%Y-%m-%d %H:%M:%S")
                    );
                    println!(
                        "  Last accessed: {}",
                        workspace.last_accessed.format("%Y-%m-%d %H:%M:%S")
                    );
                    if let Some(ref cfg) = workspace.config_path {
                        println!("  Config: {}", cfg.display());
                    }

                    let history_path = workspace_manager.query_history_path(&id);
                    if history_path.exists() {
                        if let Ok(history) =
                            query_history::QueryHistory::load(&history_path).await
                        {
                            println!("\n  Total queries: {}", history.total_queries());
                        }
                    }
                },
                Err(e) => {
                    eprintln!("❌ Error loading workspace: {}", e);
                    eprintln!("\nList available workspaces with: graphrag workspace list");
                },
            }
        },
        WorkspaceCommands::Delete { id } => {
            workspace_manager.delete_workspace(&id).await?;
            println!("✅ Workspace deleted: {}", id);
        },
    }

    Ok(())
}

async fn run_setup_wizard(template: Option<String>, output: PathBuf) -> Result<()> {
    use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
    use std::fs;

    let theme = ColorfulTheme::default();

    println!(
        "\n{}",
        "╔════════════════════════════════════════════════════════════╗\n\
         ║           GraphRAG Configuration Setup Wizard              ║\n\
         ╚════════════════════════════════════════════════════════════╝"
    );
    println!();

    let use_case = if let Some(ref t) = template {
        t.clone()
    } else {
        let options = vec![
            "General purpose - Mixed documents, articles (Recommended)",
            "Legal documents - Contracts, agreements, regulations",
            "Medical documents - Clinical notes, patient records",
            "Financial documents - Reports, SEC filings, analysis",
            "Technical documentation - API docs, code documentation",
        ];

        let selection = Select::with_theme(&theme)
            .with_prompt("Select your use case")
            .items(&options)
            .default(0)
            .interact()?;

        match selection {
            0 => "general",
            1 => "legal",
            2 => "medical",
            3 => "financial",
            4 => "technical",
            _ => "general",
        }
        .to_string()
    };

    println!("\n   Selected template: {}\n", use_case);

    let llm_options = vec![
        "Local Ollama (Recommended - free, private, runs locally)",
        "No LLM (Pattern-based extraction only, faster but less accurate)",
    ];

    let llm_selection = Select::with_theme(&theme)
        .with_prompt("Select LLM provider")
        .items(&llm_options)
        .default(0)
        .interact()?;

    let ollama_enabled = llm_selection == 0;

    let mut ollama_host = "localhost".to_string();
    let mut ollama_port: u16 = 11434;
    let mut chat_model = "llama3.2:3b".to_string();

    if ollama_enabled {
        println!("\n   Ollama Configuration:");

        ollama_host = Input::with_theme(&theme)
            .with_prompt("   Ollama host")
            .default("localhost".to_string())
            .interact_text()?;

        let port_str: String = Input::with_theme(&theme)
            .with_prompt("   Ollama port")
            .default("11434".to_string())
            .interact_text()?;

        ollama_port = port_str.parse().unwrap_or(11434);

        chat_model = Input::with_theme(&theme)
            .with_prompt("   Chat model")
            .default("llama3.2:3b".to_string())
            .interact_text()?;
    }

    let output_dir: String = Input::with_theme(&theme)
        .with_prompt("Output directory for graph data")
        .default("./graphrag-output".to_string())
        .interact_text()?;

    println!("\n   Generating configuration...\n");

    let config_content = generate_config(
        &use_case,
        ollama_enabled,
        &ollama_host,
        ollama_port,
        &chat_model,
        &output_dir,
    );

    if output.exists() {
        let overwrite = Confirm::with_theme(&theme)
            .with_prompt(format!(
                "File {} already exists. Overwrite?",
                output.display()
            ))
            .default(false)
            .interact()?;

        if !overwrite {
            println!("\n   Setup cancelled.");
            return Ok(());
        }
    }

    fs::write(&output, config_content)?;

    println!("   ✅ Configuration saved to: {}\n", output.display());
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                     Next Steps                             ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  1. Start the TUI:                                         ║");
    println!("║     graphrag tui --config {}                         ║", output.display());
    println!("║                                                            ║");
    println!("║  2. Load a document in the TUI:                            ║");
    println!("║     /load path/to/your/document.txt                        ║");
    println!("║                                                            ║");
    println!("║  3. Query your knowledge graph:                            ║");
    println!("║     Type your question and press Enter                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    if ollama_enabled {
        println!(
            "\n   💡 Tip: Make sure Ollama is running at {}:{}",
            ollama_host, ollama_port
        );
        println!("      Start it with: ollama serve");
        println!("      Pull model with: ollama pull {}", chat_model);
    }

    Ok(())
}

fn generate_config(
    use_case: &str,
    ollama_enabled: bool,
    ollama_host: &str,
    ollama_port: u16,
    chat_model: &str,
    output_dir: &str,
) -> String {
    let entity_types = match use_case {
        "legal" => {
            r#"["PARTY", "PERSON", "ORGANIZATION", "DATE", "MONETARY_VALUE", "JURISDICTION", "CLAUSE_TYPE", "OBLIGATION"]"#
        },
        "medical" => {
            r#"["PATIENT", "DIAGNOSIS", "MEDICATION", "PROCEDURE", "SYMPTOM", "LAB_VALUE", "PROVIDER", "DATE"]"#
        },
        "financial" => {
            r#"["COMPANY", "TICKER", "PERSON", "MONETARY_VALUE", "PERCENTAGE", "DATE", "METRIC", "INDUSTRY"]"#
        },
        "technical" => {
            r#"["FUNCTION", "CLASS", "MODULE", "API_ENDPOINT", "PARAMETER", "VERSION", "DEPENDENCY"]"#
        },
        _ => r#"["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT"]"#,
    };

    let approach = match use_case {
        "legal" | "medical" => "semantic",
        "technical" => "algorithmic",
        _ => "hybrid",
    };

    let chunk_size = match use_case {
        "legal" => 500,
        "medical" => 750,
        "technical" => 600,
        "financial" => 1200,
        _ => 1000,
    };

    let use_gleaning = ollama_enabled && matches!(use_case, "legal" | "medical" | "financial");

    format!(
        r#"# GraphRAG Configuration
# Generated by: graphrag setup
# Template: {use_case}
# ===================================================

output_dir = "{output_dir}"
approach = "{approach}"

# Text chunking settings
chunk_size = {chunk_size}
chunk_overlap = {overlap}

# Retrieval settings
top_k_results = 10
similarity_threshold = 0.7

[embeddings]
backend = "{embedding_backend}"
dimension = 384
fallback_to_hash = true
batch_size = 32

[entities]
min_confidence = 0.7
entity_types = {entity_types}
use_gleaning = {use_gleaning}
max_gleaning_rounds = 3

[graph]
max_connections = 10
similarity_threshold = 0.8
extract_relationships = true
relationship_confidence_threshold = 0.5

[graph.traversal]
max_depth = 3
max_paths = 10
use_edge_weights = true
min_relationship_strength = 0.3

[retrieval]
top_k = 10
search_algorithm = "cosine"

[parallel]
enabled = true
num_threads = 0
min_batch_size = 10

[ollama]
enabled = {ollama_enabled}
host = "{ollama_host}"
port = {ollama_port}
chat_model = "{chat_model}"
embedding_model = "nomic-embed-text"
timeout_seconds = 30
enable_caching = true

[auto_save]
enabled = false
interval_seconds = 300
max_versions = 5
"#,
        use_case = use_case,
        output_dir = output_dir,
        approach = approach,
        chunk_size = chunk_size,
        overlap = chunk_size / 5,
        embedding_backend = if ollama_enabled { "ollama" } else { "hash" },
        entity_types = entity_types,
        use_gleaning = use_gleaning,
        ollama_enabled = ollama_enabled,
        ollama_host = ollama_host,
        ollama_port = ollama_port,
        chat_model = chat_model,
    )
}

/// Restore the terminal on panic (called at the top of [`run`]).
pub fn install_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = crossterm::execute!(std::io::stderr(), crossterm::terminal::LeaveAlternateScreen);
        let _ = crossterm::terminal::disable_raw_mode();
        original_hook(panic_info);
    }));
}

fn setup_logging(debug: bool) -> Result<()> {
    use tracing_subscriber::EnvFilter;

    let filter = if debug {
        EnvFilter::new("graphrag_cli=debug,graphrag_core=debug")
    } else {
        EnvFilter::new("graphrag_cli=info,graphrag_core=info")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_file(true)
        .with_line_number(true)
        .init();

    Ok(())
}

fn setup_tui_logging() -> Result<()> {
    use std::fs::OpenOptions;
    use std::sync::Arc;
    use tracing_subscriber::EnvFilter;

    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("graphrag-cli")
        .join("logs");

    std::fs::create_dir_all(&log_dir)?;

    let log_file = log_dir.join("graphrag-cli.log");
    let file = OpenOptions::new().create(true).append(true).open(log_file)?;

    let filter = EnvFilter::new("graphrag_cli=warn,graphrag_core=warn");

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(Arc::new(file))
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_ansi(false)
        .init();

    Ok(())
}
