//! Slash command system for the TUI
//!
//! Provides parsing and execution of slash commands like:
//! - /config <file>
//! - /load <file>
//! - /stats
//! - /entities [filter]
//! - /workspace <name>

use color_eyre::eyre::{eyre, Result};
use std::path::PathBuf;

/// Slash command enum
#[derive(Debug, Clone, PartialEq)]
pub enum SlashCommand {
    /// Load a configuration file
    Config(PathBuf),
    /// Load a document (with optional rebuild flag)
    Load(PathBuf, bool), // (path, rebuild)
    /// Clear the knowledge graph
    Clear,
    /// Rebuild the knowledge graph from existing documents
    Rebuild,
    /// Show graph statistics
    Stats,
    /// List entities (with optional filter)
    Entities(Option<String>),
    /// Switch workspace (load)
    Workspace(String),
    /// List available workspaces
    WorkspaceList,
    /// Save current graph to workspace
    WorkspaceSave(String),
    /// Delete a workspace
    WorkspaceDelete(String),
    /// Show help
    Help,
}

impl SlashCommand {
    /// Parse a slash command from input string
    pub fn parse(input: &str) -> Result<Self> {
        let trimmed = input.trim();

        if !trimmed.starts_with('/') {
            return Err(eyre!("Not a slash command (must start with /)"));
        }

        let parts: Vec<&str> = trimmed[1..].split_whitespace().collect();

        if parts.is_empty() {
            return Err(eyre!("Empty command"));
        }

        let command = parts[0].to_lowercase();
        let args = &parts[1..];

        match command.as_str() {
            "config" => {
                // Get everything after "config" as the file path (join all args)
                // This handles paths with spaces or multiple parts
                let path_str = trimmed[1..].trim_start_matches("config").trim();

                if path_str.is_empty() {
                    return Err(eyre!("Missing argument: /config <file>"));
                }

                // Debug log to see what's being parsed
                tracing::debug!("Parsing config command - path_str: {:?}", path_str);
                Ok(SlashCommand::Config(PathBuf::from(path_str)))
            }
            "load" => {
                // Get everything after "load" command
                let rest = trimmed[1..].trim_start_matches("load").trim();

                if rest.is_empty() {
                    return Err(eyre!("Missing argument: /load <file> [--rebuild]"));
                }

                // Check for --rebuild flag
                let rebuild = rest.contains("--rebuild") || rest.contains("-r");

                // Remove flags to get the file path
                let path_str = rest
                    .replace("--rebuild", "")
                    .replace("-r", "")
                    .trim()
                    .to_string();

                if path_str.is_empty() {
                    return Err(eyre!("Missing file path argument"));
                }

                tracing::debug!("Parsing load command - path_str: {:?}, rebuild: {}", path_str, rebuild);
                Ok(SlashCommand::Load(PathBuf::from(path_str), rebuild))
            }
            "clear" => {
                if !args.is_empty() {
                    return Err(eyre!("/clear takes no arguments"));
                }
                Ok(SlashCommand::Clear)
            }
            "rebuild" => {
                if !args.is_empty() {
                    return Err(eyre!("/rebuild takes no arguments"));
                }
                Ok(SlashCommand::Rebuild)
            }
            "stats" => {
                if !args.is_empty() {
                    return Err(eyre!("/stats takes no arguments"));
                }
                Ok(SlashCommand::Stats)
            }
            "entities" => {
                let filter = if args.is_empty() {
                    None
                } else {
                    Some(args.join(" "))
                };
                Ok(SlashCommand::Entities(filter))
            }
            "workspace" | "ws" => {
                // /workspace <name> - load workspace
                // /workspace list - list workspaces
                // /workspace save <name> - save current graph
                // /workspace delete <name> - delete workspace

                if args.is_empty() {
                    return Err(eyre!("Missing argument. Usage: /workspace <name|list|save|delete>"));
                }

                match args[0].to_lowercase().as_str() {
                    "list" | "ls" => {
                        if args.len() > 1 {
                            return Err(eyre!("/workspace list takes no additional arguments"));
                        }
                        Ok(SlashCommand::WorkspaceList)
                    }
                    "save" => {
                        if args.len() < 2 {
                            return Err(eyre!("Missing workspace name: /workspace save <name>"));
                        }
                        Ok(SlashCommand::WorkspaceSave(args[1].to_string()))
                    }
                    "delete" | "del" | "rm" => {
                        if args.len() < 2 {
                            return Err(eyre!("Missing workspace name: /workspace delete <name>"));
                        }
                        Ok(SlashCommand::WorkspaceDelete(args[1].to_string()))
                    }
                    name => {
                        // Default: load workspace
                        Ok(SlashCommand::Workspace(name.to_string()))
                    }
                }
            }
            "help" => {
                if !args.is_empty() {
                    return Err(eyre!("/help takes no arguments"));
                }
                Ok(SlashCommand::Help)
            }
            _ => Err(eyre!(
                "Unknown command: /{}. Type /help for available commands.",
                command
            )),
        }
    }

    /// Get help text for all slash commands
    pub fn help_text() -> String {
        r#"
Available Slash Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/config <file>          Load GraphRAG configuration file
                        Supports: JSON5, JSON, TOML
                        Example: /config docs-example/sym.json5

/load <file> [--rebuild] Load and process a document into the knowledge graph
                        --rebuild: Clear existing graph before building
                        Example: /load info/Symposium.txt
                        Example: /load info/Symposium.txt --rebuild

/clear                  Clear the knowledge graph (preserves documents)
                        Removes all entities and relationships

/rebuild                Rebuild the knowledge graph from loaded documents
                        Clears graph and re-extracts entities/relationships
                        Useful after changing configuration or to fix issues

/stats                  Show knowledge graph statistics
                        Displays: entities, relationships, documents, chunks

/entities [filter]      List entities in the knowledge graph
                        Example: /entities socrates
                        Example: /entities PERSON

/workspace <command>    Workspace management commands:
  /ws list              List all available workspaces with statistics
  /ws save <name>       Save current graph to a workspace
  /ws <name>            Load graph from a workspace
  /ws delete <name>     Delete a workspace permanently

                        Examples:
                        /workspace list
                        /workspace save my_project
                        /workspace my_project
                        /workspace delete old_project

/help                   Show this help message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keyboard Shortcuts:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOCUS & NAVIGATION:
F1                      Focus Results Viewer (LLM answer)
F2                      Focus Raw Search Results
F3                      Focus Info Panel
Esc                     Return focus to Input (enable typing)

SCROLLING (when viewer is focused):
j / k                   Scroll down / up one line
Ctrl+D / Ctrl+U         Scroll down / up one page
Home / End              Scroll to top / bottom

OTHER:
Ctrl+C / Ctrl+Q         Quit application
?                       Toggle help

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Tip: Use F1/F2/F3 to switch focus between viewers
💡 Use --rebuild flag to force a fresh graph rebuild when loading documents
💡 Vim-style j/k scrolling works only when a viewer is focused
"#
        .trim()
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let cmd = SlashCommand::parse("/config test.toml").unwrap();
        assert_eq!(cmd, SlashCommand::Config(PathBuf::from("test.toml")));
    }

    #[test]
    fn test_parse_config_with_path() {
        let cmd = SlashCommand::parse("/config docs-example/sym.json5").unwrap();
        assert_eq!(cmd, SlashCommand::Config(PathBuf::from("docs-example/sym.json5")));
    }

    #[test]
    fn test_parse_config_with_spaces_in_dirname() {
        let cmd = SlashCommand::parse("/config my docs/config.toml").unwrap();
        assert_eq!(cmd, SlashCommand::Config(PathBuf::from("my docs/config.toml")));
    }

    #[test]
    fn test_parse_load() {
        let cmd = SlashCommand::parse("/load doc.txt").unwrap();
        assert_eq!(cmd, SlashCommand::Load(PathBuf::from("doc.txt"), false));
    }

    #[test]
    fn test_parse_load_with_rebuild() {
        let cmd = SlashCommand::parse("/load doc.txt --rebuild").unwrap();
        assert_eq!(cmd, SlashCommand::Load(PathBuf::from("doc.txt"), true));
    }

    #[test]
    fn test_parse_load_with_rebuild_short() {
        let cmd = SlashCommand::parse("/load doc.txt -r").unwrap();
        assert_eq!(cmd, SlashCommand::Load(PathBuf::from("doc.txt"), true));
    }

    #[test]
    fn test_parse_clear() {
        let cmd = SlashCommand::parse("/clear").unwrap();
        assert_eq!(cmd, SlashCommand::Clear);
    }

    #[test]
    fn test_parse_rebuild() {
        let cmd = SlashCommand::parse("/rebuild").unwrap();
        assert_eq!(cmd, SlashCommand::Rebuild);
    }

    #[test]
    fn test_parse_stats() {
        let cmd = SlashCommand::parse("/stats").unwrap();
        assert_eq!(cmd, SlashCommand::Stats);
    }

    #[test]
    fn test_parse_entities_no_filter() {
        let cmd = SlashCommand::parse("/entities").unwrap();
        assert_eq!(cmd, SlashCommand::Entities(None));
    }

    #[test]
    fn test_parse_entities_with_filter() {
        let cmd = SlashCommand::parse("/entities socrates").unwrap();
        assert_eq!(cmd, SlashCommand::Entities(Some("socrates".to_string())));
    }

    #[test]
    fn test_parse_workspace() {
        let cmd = SlashCommand::parse("/workspace test").unwrap();
        assert_eq!(cmd, SlashCommand::Workspace("test".to_string()));
    }

    #[test]
    fn test_parse_help() {
        let cmd = SlashCommand::parse("/help").unwrap();
        assert_eq!(cmd, SlashCommand::Help);
    }

    #[test]
    fn test_parse_unknown_command() {
        let result = SlashCommand::parse("/unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_not_slash_command() {
        let result = SlashCommand::parse("config test.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_arguments() {
        assert!(SlashCommand::parse("/config").is_err());
        assert!(SlashCommand::parse("/load").is_err());
        assert!(SlashCommand::parse("/workspace").is_err());
    }
}
