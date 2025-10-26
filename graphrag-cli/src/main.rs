//! GraphRAG CLI - Terminal User Interface for GraphRAG operations
//!
//! A modern TUI built with Ratatui for interactive GraphRAG queries,
//! document processing, and knowledge graph exploration.

use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use std::path::PathBuf;

mod action;
mod app;
mod commands;
mod config;
mod handlers;
mod mode;
mod query_history;
mod theme;
mod tui;
mod ui;
mod workspace;

use app::App;

#[derive(Parser)]
#[command(name = "graphrag-cli")]
#[command(version, about = "Modern Terminal UI for GraphRAG operations", long_about = None)]
#[command(author = "GraphRAG Contributors")]
struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Workspace name
    #[arg(short, long)]
    workspace: Option<String>,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive TUI (default)
    Tui,

    /// Initialize GraphRAG with configuration (deprecated: use TUI with /config)
    Init {
        /// Configuration file path
        config: PathBuf,
    },

    /// Load a document into the knowledge graph (deprecated: use TUI with /load)
    Load {
        /// Document file path
        document: PathBuf,

        /// Configuration file (required if not already initialized)
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Execute a query (deprecated: use TUI)
    Query {
        /// Query text
        query: String,

        /// Configuration file (required if not already initialized)
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// List entities in the knowledge graph (deprecated: use TUI with /entities)
    Entities {
        /// Filter by name or type
        filter: Option<String>,

        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Show knowledge graph statistics (deprecated: use TUI with /stats)
    Stats {
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Workspace management commands
    Workspace {
        #[command(subcommand)]
        action: WorkspaceCommands,
    },
}

#[derive(Subcommand)]
enum WorkspaceCommands {
    /// List all workspaces
    List,

    /// Create a new workspace
    Create { name: String },

    /// Show workspace information
    Info { id: String },

    /// Delete a workspace
    Delete { id: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install panic hook to restore terminal
    install_panic_hook();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Install color-eyre for beautiful error reports
    color_eyre::install()?;

    // Execute command
    match cli.command {
        Some(Commands::Tui) | None => {
            // Start interactive TUI (default) - logging setup is done inside run_tui
            run_tui(cli.config, cli.workspace).await?;
        }
        Some(Commands::Init { config }) => {
            // Setup logging for CLI commands
            setup_logging(cli.debug)?;

            println!("⚠️  The 'init' command is deprecated.");
            println!("    Please use the TUI and execute: /config {}", config.display());
            println!("\nStart TUI with: graphrag-cli tui --config {}", config.display());
        }
        Some(Commands::Load { document, config }) => {
            // Setup logging for CLI commands
            setup_logging(cli.debug)?;

            println!("⚠️  The 'load' command is deprecated.");
            println!("    Please use the TUI and execute: /load {}", document.display());
            if let Some(cfg) = config {
                println!("\nStart TUI with: graphrag-cli tui --config {}", cfg.display());
            } else {
                println!("\nStart TUI with: graphrag-cli tui");
            }
        }
        Some(Commands::Query { query, config }) => {
            // Setup logging for CLI commands
            setup_logging(cli.debug)?;

            println!("⚠️  The 'query' command is deprecated.");
            println!("    Please use the TUI and type your query: {}", query);
            if let Some(cfg) = config {
                println!("\nStart TUI with: graphrag-cli tui --config {}", cfg.display());
            } else {
                println!("\nStart TUI with: graphrag-cli tui");
            }
        }
        Some(Commands::Entities { filter, config }) => {
            // Setup logging for CLI commands
            setup_logging(cli.debug)?;

            println!("⚠️  The 'entities' command is deprecated.");
            if let Some(f) = filter {
                println!("    Please use the TUI and execute: /entities {}", f);
            } else {
                println!("    Please use the TUI and execute: /entities");
            }
            if let Some(cfg) = config {
                println!("\nStart TUI with: graphrag-cli tui --config {}", cfg.display());
            } else {
                println!("\nStart TUI with: graphrag-cli tui");
            }
        }
        Some(Commands::Stats { config }) => {
            // Setup logging for CLI commands
            setup_logging(cli.debug)?;

            println!("⚠️  The 'stats' command is deprecated.");
            println!("    Please use the TUI and execute: /stats");
            if let Some(cfg) = config {
                println!("\nStart TUI with: graphrag-cli tui --config {}", cfg.display());
            } else {
                println!("\nStart TUI with: graphrag-cli tui");
            }
        }
        Some(Commands::Workspace { action }) => {
            // Setup logging for CLI commands
            setup_logging(cli.debug)?;

            handle_workspace_commands(action).await?;
        }
    }

    Ok(())
}

/// Run the interactive TUI
async fn run_tui(config_path: Option<PathBuf>, workspace: Option<String>) -> Result<()> {
    // Disable logging to prevent interference with TUI
    // Logs will be written to file instead (see setup_tui_logging)
    setup_tui_logging()?;

    let mut app = App::new(config_path, workspace)?;
    app.run().await?;
    Ok(())
}

/// Handle workspace commands
async fn handle_workspace_commands(action: WorkspaceCommands) -> Result<()> {
    let workspace_manager = workspace::WorkspaceManager::new()?;

    match action {
        WorkspaceCommands::List => {
            let workspaces = workspace_manager.list_workspaces().await?;

            if workspaces.is_empty() {
                println!("No workspaces found.");
                println!("\nCreate a workspace with: graphrag-cli workspace create <name>");
            } else {
                println!("Available workspaces:\n");
                for ws in workspaces {
                    println!("  📁 {} ({})", ws.name, ws.id);
                    println!("     Created: {}", ws.created_at.format("%Y-%m-%d %H:%M:%S"));
                    println!("     Last accessed: {}", ws.last_accessed.format("%Y-%m-%d %H:%M:%S"));
                    if let Some(ref cfg) = ws.config_path {
                        println!("     Config: {}", cfg.display());
                    }
                    println!();
                }
            }
        }
        WorkspaceCommands::Create { name } => {
            let workspace = workspace_manager.create_workspace(name.clone()).await?;
            println!("✅ Workspace created successfully!");
            println!("   Name: {}", workspace.name);
            println!("   ID:   {}", workspace.id);
            println!("\nUse it with: graphrag-cli tui --workspace {}", workspace.id);
        }
        WorkspaceCommands::Info { id } => {
            match workspace_manager.load_metadata(&id).await {
                Ok(workspace) => {
                    println!("Workspace Information:\n");
                    println!("  Name: {}", workspace.name);
                    println!("  ID:   {}", workspace.id);
                    println!("  Created: {}", workspace.created_at.format("%Y-%m-%d %H:%M:%S"));
                    println!("  Last accessed: {}", workspace.last_accessed.format("%Y-%m-%d %H:%M:%S"));
                    if let Some(ref cfg) = workspace.config_path {
                        println!("  Config: {}", cfg.display());
                    }

                    // Show query history stats if available
                    let history_path = workspace_manager.query_history_path(&id);
                    if history_path.exists() {
                        if let Ok(history) = query_history::QueryHistory::load(&history_path).await {
                            println!("\n  Total queries: {}", history.total_queries());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Error loading workspace: {}", e);
                    eprintln!("\nList available workspaces with: graphrag-cli workspace list");
                }
            }
        }
        WorkspaceCommands::Delete { id } => {
            workspace_manager.delete_workspace(&id).await?;
            println!("✅ Workspace deleted: {}", id);
        }
    }

    Ok(())
}

/// Install panic hook to restore terminal on panic
fn install_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Try to restore terminal
        let _ = crossterm::execute!(
            std::io::stderr(),
            crossterm::terminal::LeaveAlternateScreen,
        );
        let _ = crossterm::terminal::disable_raw_mode();

        // Call original hook
        original_hook(panic_info);
    }));
}

/// Setup tracing/logging for CLI commands (non-TUI)
fn setup_logging(debug: bool) -> Result<()> {
    use tracing_subscriber::EnvFilter;

    let filter = if debug {
        EnvFilter::new("graphrag_cli=debug,graphrag_core=debug")
    } else {
        EnvFilter::new("graphrag_cli=info,graphrag_core=info")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_file(true)
        .with_line_number(true)
        .init();

    Ok(())
}

/// Setup tracing/logging for TUI mode (logs to file)
fn setup_tui_logging() -> Result<()> {
    use tracing_subscriber::EnvFilter;
    use std::fs::OpenOptions;
    use std::sync::Arc;

    // Create logs directory if it doesn't exist
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("graphrag-cli")
        .join("logs");

    std::fs::create_dir_all(&log_dir)?;

    // Create log file
    let log_file = log_dir.join("graphrag-cli.log");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)?;

    // Use a much stricter filter to reduce verbosity
    // Only log warnings and errors from graphrag_core to avoid TUI corruption
    let filter = EnvFilter::new("graphrag_cli=warn,graphrag_core=warn");

    // Log to file instead of stderr to avoid interfering with TUI
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(Arc::new(file))
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_ansi(false)  // Disable ANSI colors in file
        .init();

    Ok(())
}
