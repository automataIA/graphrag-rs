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
    /// Index a document into a workspace. No config file required.
    ///
    /// Example: `graphrag index ./book.txt`
    /// Example: `graphrag index ./book.txt --workspace ./my-graph --llm`
    Index {
        /// Path to a UTF-8 text document (for example .txt or Markdown)
        path: PathBuf,
        /// Workspace directory (created if missing)
        #[arg(long, default_value = "./graphrag-data")]
        workspace: PathBuf,
        /// Enable LLM extraction (requires a running llama-server or another OpenAI-compatible endpoint)
        #[arg(long)]
        llm: bool,
        /// Override chunk size (default 1000)
        #[arg(long)]
        chunk_size: Option<usize>,
    },

    /// Ask a question against an indexed workspace. No config file required.
    ///
    /// Example: `graphrag ask "Who is Diotima?"`
    Ask {
        /// Question to ask
        query: String,
        /// Workspace directory created by `graphrag index`
        #[arg(long, default_value = "./graphrag-data")]
        workspace: PathBuf,
        /// Enable the LLM for semantic answering
        #[arg(long)]
        llm: bool,
    },

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

    /// Explain the effective settings in a configuration file
    ExplainConfig {
        /// Path to the configuration file to explain
        config_file: PathBuf,
    },

    /// Check local setup, feature flags, and likely fallback modes
    Doctor {
        /// Optional configuration file to include in the diagnosis
        #[arg(short, long)]
        config: Option<PathBuf>,
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

        /// Pipe-separated list of questions e.g. "Q1?|Q2?" (ignored when --gold is given)
        #[arg(short, long, default_value = "")]
        questions: String,

        /// Gold file (JSON) with questions and expected keyword groups; enables scoring
        #[arg(long)]
        gold: Option<PathBuf>,
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
        Some(Commands::Index {
            path,
            workspace,
            llm,
            chunk_size,
        }) => {
            setup_logging(cli.debug)?;
            run_index(
                &path,
                &workspace,
                llm,
                chunk_size,
                cli.config.as_deref(),
                &cli.format,
            )
            .await?;
        },
        Some(Commands::Ask {
            query,
            workspace,
            llm,
        }) => {
            setup_logging(cli.debug)?;
            run_ask(&query, &workspace, llm, cli.config.as_deref(), &cli.format).await?;
        },
        Some(Commands::Setup { template, output }) => {
            run_setup_wizard(template, output).await?;
        },
        Some(Commands::Validate { config_file }) => {
            setup_logging(cli.debug)?;
            run_validate(&config_file, &cli.format)?;
        },
        Some(Commands::ExplainConfig { config_file }) => {
            setup_logging(cli.debug)?;
            run_explain_config(&config_file, &cli.format).await?;
        },
        Some(Commands::Doctor { config }) => {
            setup_logging(cli.debug)?;
            run_doctor(config.as_deref(), &cli.format).await?;
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
                println!("✅ {result}");
            }
        },
        Some(Commands::Query { query, config }) => {
            setup_logging(cli.debug)?;
            eprintln!("⚠️  `query` is deprecated. Prefer: graphrag tui, then /query {query}");

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
                println!("📝 Query: {query}\n");
                println!("💡 Answer:\n{answer}\n");
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
            gold,
        }) => {
            if !cli.debug {
                std::env::set_var("RUST_LOG", "error");
            }
            setup_logging(cli.debug)?;

            let q_vec: Vec<String> = questions.split('|').map(|s| s.to_string()).collect();
            handlers::bench::run_benchmark(&config, &book, q_vec, gold.as_deref()).await?;
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

/// Build a turnkey config for the given workspace, optionally enabling the LLM.
fn turnkey_config(
    workspace: &std::path::Path,
    llm: bool,
    chunk_size: Option<usize>,
) -> graphrag_core::Config {
    let mut cfg = graphrag_core::Config::quick(workspace);
    if llm {
        cfg = cfg.with_llm();
    }
    if let Some(n) = chunk_size {
        cfg = cfg.with_chunk_size(n);
    }
    cfg
}

/// Load a user configuration for the turnkey commands and bind it to the
/// workspace selected by the command line. The command line workspace is the
/// persistence boundary; configuration files provide the remaining settings.
async fn command_config(
    workspace: &std::path::Path,
    llm: bool,
    chunk_size: Option<usize>,
    config_file: Option<&std::path::Path>,
) -> Result<graphrag_core::Config> {
    let mut cfg = match config_file {
        Some(path) => load_config_from_file(path).await?,
        None => turnkey_config(workspace, llm, chunk_size),
    };

    if config_file.is_some() {
        let workspace_config = turnkey_config(workspace, false, None);
        cfg.output_dir = workspace_config.output_dir;
        cfg.auto_save.base_dir = workspace_config.auto_save.base_dir;
        cfg.auto_save.workspace_name = workspace_config.auto_save.workspace_name;
        cfg.auto_save.enabled = true;
        if llm {
            cfg = cfg.with_llm();
        }
        if let Some(n) = chunk_size {
            cfg = cfg.with_chunk_size(n);
        }
    }

    Ok(cfg)
}

/// Log the LLM usage accumulated since `start` (server-reported figures only).
fn log_llm_usage(start: &graphrag_core::llm::LlmUsageTotals) {
    let used = graphrag_core::llm::global_llm_usage().since(start);
    tracing::info!(
        "LLM usage: requests={} prompt_tokens={} completion_tokens={} cached_prompt_tokens={}",
        used.requests,
        used.prompt_tokens,
        used.completion_tokens,
        used.cached_prompt_tokens
    );
}

async fn run_index(
    path: &std::path::Path,
    workspace: &std::path::Path,
    llm: bool,
    chunk_size: Option<usize>,
    config_file: Option<&std::path::Path>,
    format: &str,
) -> Result<()> {
    if !path.exists() {
        return Err(color_eyre::eyre::eyre!(
            "Document not found: {}",
            path.display()
        ));
    }
    let cfg = command_config(workspace, llm, chunk_size, config_file).await?;
    let usage_start = graphrag_core::llm::global_llm_usage();
    let handler = handlers::graphrag::GraphRAGHandler::new();
    handler.initialize(cfg).await?;
    // `index` is the update path: initialization reopens an existing workspace
    // and this document is added to the loaded state. A fresh workspace gives a
    // clean index; explicit rebuild remains available through the TUI command.
    let summary = handler.load_document_with_options(path, false).await?;
    log_llm_usage(&usage_start);

    if format == "json" {
        println!(
            "{}",
            serde_json::json!({
                "status": "indexed",
                "document": path.display().to_string(),
                "workspace": workspace.display().to_string(),
                "details": summary,
            })
        );
    } else {
        println!(
            "✅ Indexed `{}` into `{}`",
            path.display(),
            workspace.display()
        );
        println!("   {summary}");
        println!("\nAsk a question:");
        println!(
            "   graphrag ask \"your question\" --workspace {}",
            workspace.display()
        );
    }
    Ok(())
}

async fn run_ask(
    query: &str,
    workspace: &std::path::Path,
    llm: bool,
    config_file: Option<&std::path::Path>,
    format: &str,
) -> Result<()> {
    if !workspace.exists() {
        return Err(color_eyre::eyre::eyre!(
            "Workspace not found: {}. Run `graphrag index <file>` first.",
            workspace.display()
        ));
    }
    let cfg = command_config(workspace, llm, None, config_file).await?;
    let usage_start = graphrag_core::llm::global_llm_usage();
    let handler = handlers::graphrag::GraphRAGHandler::new();
    handler.initialize(cfg).await?;
    let (answer, sources) = handler.query_with_raw(query).await?;
    log_llm_usage(&usage_start);

    if format == "json" {
        println!(
            "{}",
            serde_json::json!({"query": query, "answer": answer, "sources": sources})
        );
    } else {
        println!("📝 {query}\n");
        println!("💡 {answer}\n");
        if !sources.is_empty() {
            println!("📚 Sources:");
            for (i, src) in sources.iter().enumerate() {
                println!("   {}. {}", i + 1, src);
            }
        }
    }
    Ok(())
}

fn run_validate(config_file: &std::path::Path, format: &str) -> Result<()> {
    use graphrag_core::config::json5_loader::detect_config_format;

    if !config_file.exists() {
        let message = format!("File not found: {}", config_file.display());
        if format == "json" {
            println!("{}", serde_json::json!({"valid": false, "error": message}));
        } else {
            println!("❌ {message}");
        }
        return Err(color_eyre::eyre::eyre!(
            "Configuration file not found: {}",
            config_file.display()
        ));
    }
    let Some(config_format) = detect_config_format(config_file) else {
        let message = "Unsupported file format. Use .toml, .json, or .json5";
        if format == "json" {
            println!("{}", serde_json::json!({"valid": false, "error": message}));
        } else {
            println!("❌ {message}");
        }
        return Err(color_eyre::eyre::eyre!(
            "Unsupported configuration file format"
        ));
    };

    let path = config_file
        .to_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("Configuration path is not valid UTF-8"))?;
    match graphrag_core::config::loader::load_config(path) {
        Ok(config) => {
            let validation = graphrag_core::config::Validatable::validate(&config);
            if !validation.is_valid {
                let message = validation.errors.join("; ");
                if format == "json" {
                    println!(
                        "{}",
                        serde_json::json!({"valid": false, "errors": validation.errors})
                    );
                } else {
                    println!("❌ Invalid configuration:\n   {message}");
                }
                return Err(color_eyre::eyre::eyre!(
                    "Configuration validation failed: {message}"
                ));
            }
            if format == "json" {
                println!(
                    "{}",
                    serde_json::json!({
                        "valid": true,
                        "format": format!("{:?}", config_format),
                        "approach": config.approach,
                        "llm_enabled": config.llm.enabled,
                        "chunk_size": config.text.chunk_size,
                    })
                );
            } else {
                println!("✅ Configuration is valid!");
                println!("   Format:    {config_format:?}");
                println!("   Approach:  {}", config.approach);
                println!(
                    "   LLM:       {}",
                    if config.llm.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!("   Chunk size: {}", config.text.chunk_size);
            }
            Ok(())
        },
        Err(err) => {
            if format == "json" {
                println!(
                    "{}",
                    serde_json::json!({"valid": false, "error": err.to_string()})
                );
            } else {
                println!("❌ Invalid configuration:\n   {err}");
            }
            Err(color_eyre::eyre::eyre!(
                "Configuration parse or validation failed: {err}"
            ))
        },
    }
}

async fn run_explain_config(config_file: &std::path::Path, format: &str) -> Result<()> {
    if !config_file.exists() {
        return Err(color_eyre::eyre::eyre!(
            "Configuration file not found: {}",
            config_file.display()
        ));
    }

    let cfg = load_config_from_file(config_file).await?;
    let fallback_mode = if cfg.llm.enabled {
        "llm-enabled"
    } else {
        "hash-and-rules"
    };

    if format == "json" {
        println!(
            "{}",
            serde_json::json!({
                "config": config_file.display().to_string(),
                "approach": cfg.approach,
                "chunk_size": cfg.text.chunk_size,
                "chunk_overlap": cfg.text.chunk_overlap,
                "top_k": cfg.retrieval.top_k,
                "llm_enabled": cfg.llm.enabled,
                "fallback_mode": fallback_mode,
                "output_dir": cfg.output_dir,
                "entities": {
                    "use_gleaning": cfg.entities.use_gleaning,
                    "use_atomic_facts": cfg.entities.use_atomic_facts,
                    "triple_reflection": cfg.entities.enable_triple_reflection,
                }
            })
        );
    } else {
        println!("Configuration: {}", config_file.display());
        println!("  Approach:       {}", cfg.approach);
        println!(
            "  Chunking:       size {}, overlap {}",
            cfg.text.chunk_size, cfg.text.chunk_overlap
        );
        println!("  Retrieval:      top_k {}", cfg.retrieval.top_k);
        println!(
            "  LLM:            {}",
            if cfg.llm.enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!("  Fallback mode:  {fallback_mode}");
        println!("  Output dir:     {}", cfg.output_dir);
        println!(
            "  Entity options: gleaning={}, atomic_facts={}, triple_reflection={}",
            cfg.entities.use_gleaning,
            cfg.entities.use_atomic_facts,
            cfg.entities.enable_triple_reflection
        );
    }

    Ok(())
}

async fn run_doctor(config_file: Option<&std::path::Path>, format: &str) -> Result<()> {
    let cargo = command_version("cargo", &["--version"]);
    let rustc = command_version("rustc", &["--version"]);
    let wasm_target = command_status("rustup", &["target", "list", "--installed"])
        .map(|output| {
            output
                .lines()
                .any(|line| line.trim() == "wasm32-unknown-unknown")
        })
        .unwrap_or(false);
    // graphrag-cli always enables graphrag-core/llm in Cargo.toml.
    let llm_compiled = true;
    let json5_compiled = cfg!(feature = "json5-support");
    let config_exists = config_file.map(|path| path.exists());

    let config_summary = if let Some(path) = config_file {
        match load_config_from_file(path).await {
            Ok(cfg) => Some(serde_json::json!({
                "path": path.display().to_string(),
                "valid": true,
                "approach": cfg.approach,
                "llm_enabled": cfg.llm.enabled,
                "fallback_mode": if cfg.llm.enabled { "llm-enabled" } else { "hash-and-rules" },
            })),
            Err(err) => Some(serde_json::json!({
                "path": path.display().to_string(),
                "valid": false,
                "error": err.to_string(),
            })),
        }
    } else {
        None
    };

    let warnings = doctor_warnings(wasm_target, llm_compiled, config_exists, &config_summary);

    if format == "json" {
        println!(
            "{}",
            serde_json::json!({
                "cargo": cargo,
                "rustc": rustc,
                "wasm32_unknown_unknown_installed": wasm_target,
                "features": {
                    "llm": llm_compiled,
                    "json5": json5_compiled,
                },
                "config": config_summary,
                "warnings": warnings,
            })
        );
    } else {
        println!("GraphRAG Doctor");
        println!(
            "  cargo: {}",
            cargo.unwrap_or_else(|| "not found".to_string())
        );
        println!(
            "  rustc: {}",
            rustc.unwrap_or_else(|| "not found".to_string())
        );
        println!(
            "  wasm32 target: {}",
            if wasm_target { "installed" } else { "missing" }
        );
        println!("  compiled features: llm={llm_compiled}, json5={json5_compiled}");
        if let Some(summary) = config_summary {
            println!("  config: {summary}");
        }
        if warnings.is_empty() {
            println!("  status: ok");
        } else {
            println!("  warnings:");
            for warning in warnings {
                println!("    - {warning}");
            }
        }
    }

    Ok(())
}

fn command_version(program: &str, args: &[&str]) -> Option<String> {
    command_status(program, args).and_then(|output| output.lines().next().map(str::to_string))
}

fn command_status(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn doctor_warnings(
    wasm_target: bool,
    llm_compiled: bool,
    config_exists: Option<bool>,
    config_summary: &Option<serde_json::Value>,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if !wasm_target {
        warnings.push("Install wasm32-unknown-unknown before building the WASM crate.".to_string());
    }
    if !llm_compiled {
        warnings.push(
            "CLI was built without LLM support; semantic LLM extraction will use fallback paths."
                .to_string(),
        );
    }
    if let Some(false) = config_exists {
        warnings.push("Configured file path does not exist.".to_string());
    }
    if let Some(summary) = config_summary {
        if summary.get("llm_enabled").and_then(|v| v.as_bool()) == Some(false) {
            warnings.push("LLM is disabled in config; hash embeddings and rule-based extraction are demo/offline quality.".to_string());
        }
    }
    warnings
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
                    println!(
                        "     Created: {}",
                        ws.created_at.format("%Y-%m-%d %H:%M:%S")
                    );
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
        WorkspaceCommands::Info { id } => match workspace_manager.load_metadata(&id).await {
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
                    if let Ok(history) = query_history::QueryHistory::load(&history_path).await {
                        println!("\n  Total queries: {}", history.total_queries());
                    }
                }
            },
            Err(e) => {
                eprintln!("❌ Error loading workspace: {e}");
                eprintln!("\nList available workspaces with: graphrag workspace list");
            },
        },
        WorkspaceCommands::Delete { id } => {
            workspace_manager.delete_workspace(&id).await?;
            println!("✅ Workspace deleted: {id}");
        },
    }

    Ok(())
}

async fn run_setup_wizard(template: Option<String>, output: PathBuf) -> Result<()> {
    use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
    use std::fs;

    let theme = ColorfulTheme::default();

    println!(
        "\n╔════════════════════════════════════════════════════════════╗\n\
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

    println!("\n   Selected template: {use_case}\n");

    let llm_options = vec![
        "Local llama.cpp server (Recommended - free, private, OpenAI-compatible)",
        "No LLM (Pattern-based extraction only, faster but less accurate)",
    ];

    let llm_selection = Select::with_theme(&theme)
        .with_prompt("Select LLM provider")
        .items(&llm_options)
        .default(0)
        .interact()?;

    let llm_enabled = llm_selection == 0;

    let mut llm_base_url = "http://127.0.0.1:8080/v1".to_string();
    let mut chat_model = "qwen3.5-9b".to_string();

    if llm_enabled {
        println!("\n   LLM Configuration (OpenAI-compatible endpoint):");

        llm_base_url = Input::with_theme(&theme)
            .with_prompt("   LLM base URL")
            .default(llm_base_url)
            .interact_text()?;

        chat_model = Input::with_theme(&theme)
            .with_prompt("   Chat model (llama-server --alias)")
            .default(chat_model)
            .interact_text()?;
    }

    let output_dir: String = Input::with_theme(&theme)
        .with_prompt("Output directory for graph data")
        .default("./graphrag-output".to_string())
        .interact_text()?;

    println!("\n   Generating configuration...\n");

    let config_content = generate_config(
        &use_case,
        llm_enabled,
        &llm_base_url,
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
    println!(
        "║     graphrag tui --config {}                         ║",
        output.display()
    );
    println!("║                                                            ║");
    println!("║  2. Load a document in the TUI:                            ║");
    println!("║     /load path/to/your/document.txt                        ║");
    println!("║                                                            ║");
    println!("║  3. Query your knowledge graph:                            ║");
    println!("║     Type your question and press Enter                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    if llm_enabled {
        println!("\n   💡 Tip: Make sure llama-server is running at {llm_base_url}");
        println!("      Chat:       llama-server -m <model.gguf> --alias {chat_model} --port 8080");
        println!("      Embeddings: llama-server -m multilingual-e5-small.gguf --embeddings --pooling mean --port 8081");
    }

    Ok(())
}

fn generate_config(
    use_case: &str,
    llm_enabled: bool,
    llm_base_url: &str,
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

    let use_gleaning = llm_enabled && matches!(use_case, "legal" | "medical" | "financial");

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
fallback_to_hash = {fallback_to_hash}
batch_size = 32{embedding_endpoint}

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

[llm]
enabled = {llm_enabled}
base_url = "{llm_base_url}"
embedding_base_url = "http://127.0.0.1:8081/v1"
chat_model = "{chat_model}"
embedding_model = "multilingual-e5-small"
timeout_seconds = 120
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
        embedding_backend = if llm_enabled { "llama_cpp" } else { "hash" },
        fallback_to_hash = !llm_enabled,
        embedding_endpoint = if llm_enabled {
            "\napi_endpoint = \"http://127.0.0.1:8081/v1/embeddings\""
        } else {
            ""
        },
        entity_types = entity_types,
        use_gleaning = use_gleaning,
        llm_enabled = llm_enabled,
        llm_base_url = llm_base_url,
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
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)?;

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

#[cfg(test)]
mod tests {
    use super::generate_config;

    #[tokio::test]
    async fn setup_wizard_config_loads_with_llm_section() {
        let dir = tempfile::tempdir().unwrap();
        for llm_enabled in [true, false] {
            let path = dir.path().join(format!("graphrag-{llm_enabled}.toml"));
            let content = generate_config(
                "legal",
                llm_enabled,
                "http://127.0.0.1:8080/v1",
                "qwen3.5-9b",
                "./out",
            );
            std::fs::write(&path, content).unwrap();
            let cfg = crate::config::load_config(&path).await.unwrap();
            assert_eq!(cfg.llm.enabled, llm_enabled);
            assert_eq!(cfg.llm.base_url, "http://127.0.0.1:8080/v1");
            assert_eq!(cfg.llm.chat_model, "qwen3.5-9b");
            let backend = if llm_enabled { "llama_cpp" } else { "hash" };
            assert_eq!(cfg.embeddings.backend, backend);
        }
    }
}
