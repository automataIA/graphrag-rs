# 📂 GraphRAG-CLI Storage Locations

This document explains where GraphRAG-CLI saves knowledge graphs, workspaces, and other data.

## 🗂️ Directory Structure

GraphRAG-CLI stores all data in your home directory under `~/.graphrag/`:

```
~/.graphrag/
├── workspaces/              # All workspace data
│   └── <workspace-id>/      # Individual workspace folder
│       ├── metadata.json            # Workspace info (name, created_at, etc.)
│       ├── query_history.json       # Query history for this workspace
│       └── knowledge_graph.json     # THE KNOWLEDGE GRAPH (entities, relationships)
└── query_history.json       # Global query history (if not using workspaces)
```

---

## 📍 Main Data Locations

### 1. **Workspaces Directory**

**Path:** `~/.graphrag/workspaces/`

This is where ALL workspace data is stored.

```bash
# View all workspaces
ls -la ~/.graphrag/workspaces/

# Example output:
# drwxr-xr-x 2 dio dio 4096 Oct 10 17:54 symposium
# drwxr-xr-x 2 dio dio 4096 Oct 11 10:23 philosophy_research
# drwxr-xr-x 2 dio dio 4096 Oct 12 14:15 technical_docs
```

Each workspace has a unique ID (UUID) as its folder name, unless you created it with CLI commands which might use the name directly.

---

### 2. **Knowledge Graph File**

**Path:** `~/.graphrag/workspaces/<workspace-id>/knowledge_graph.json`

**This is the most important file** - it contains your entire knowledge graph:
- Entities (people, concepts, locations, etc.)
- Relationships between entities
- Document chunks
- Graph structure

```bash
# View knowledge graph for a workspace
cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq

# Check size
ls -lh ~/.graphrag/workspaces/symposium/knowledge_graph.json
# Example: -rw-r--r-- 1 dio dio 317K Oct 10 17:54 knowledge_graph.json
```

**Format:** JSON format with all graph data
**Size:** Varies (10KB - 10MB+) depending on documents loaded

---

### 3. **Workspace Metadata**

**Path:** `~/.graphrag/workspaces/<workspace-id>/metadata.json`

Contains workspace information:

```json
{
  "id": "symposium",
  "name": "Plato's Symposium Analysis",
  "created_at": "2025-10-10T17:54:23Z",
  "last_accessed": "2025-10-16T10:30:45Z",
  "config_path": "/home/dio/graphrag-rs/docs-example/symposium_config.toml"
}
```

**Fields:**
- `id`: Workspace identifier
- `name`: Human-readable name
- `created_at`: Creation timestamp (UTC)
- `last_accessed`: Last access timestamp (updated automatically)
- `config_path`: Path to configuration file used (optional)

---

### 4. **Query History**

**Path:** `~/.graphrag/workspaces/<workspace-id>/query_history.json`

Stores all queries executed in this workspace:

```json
{
  "queries": [
    {
      "query": "What does Socrates say about love?",
      "timestamp": "2025-10-16T10:35:12Z",
      "duration_ms": 1234,
      "result_length": 856
    },
    {
      "query": "Who are the main speakers?",
      "timestamp": "2025-10-16T10:36:45Z",
      "duration_ms": 892,
      "result_length": 423
    }
  ]
}
```

**Useful for:**
- Reviewing past queries
- Understanding query performance
- Navigating query history with ↑/↓ in TUI

---

### 5. **Logs**

**Path (Linux):** `~/.local/share/graphrag-cli/logs/graphrag-cli.log`

**Path (macOS):** `~/Library/Application Support/graphrag-cli/logs/graphrag-cli.log`

Contains application logs (warnings, errors, debug info):

```bash
# View logs
tail -f ~/.local/share/graphrag-cli/logs/graphrag-cli.log

# Search for errors
grep ERROR ~/.local/share/graphrag-cli/logs/graphrag-cli.log
```

---

## 🔍 Finding Your Data

### List All Workspaces

```bash
# Using CLI
./target/release/graphrag-cli workspace list

# Or directly
ls -la ~/.graphrag/workspaces/
```

### Find Specific Workspace

```bash
# By name (check metadata)
for ws in ~/.graphrag/workspaces/*/; do
  echo "=== $ws ==="
  cat "$ws/metadata.json" | jq -r '.name'
done

# Find workspace by name
find ~/.graphrag/workspaces/ -name metadata.json -exec grep -l "philosophy" {} \;
```

### Check Knowledge Graph Size

```bash
# All knowledge graphs
du -h ~/.graphrag/workspaces/*/knowledge_graph.json

# Largest knowledge graph
du -h ~/.graphrag/workspaces/*/knowledge_graph.json | sort -hr | head -5
```

### Total Storage Used

```bash
# Total size of all workspaces
du -sh ~/.graphrag/

# Per workspace
du -sh ~/.graphrag/workspaces/*
```

---

## 💾 Storage Type: In-Memory vs Persistent

### Current Implementation: **In-Memory Storage**

By default, GraphRAG-CLI uses **in-memory storage** (`MemoryStorage` in `graphrag-core/src/storage/mod.rs`):

```rust
pub struct MemoryStorage {
    documents: HashMap<String, Document>,
    entities: HashMap<String, Entity>,
    chunks: HashMap<String, TextChunk>,
    metadata: HashMap<String, String>,
}
```

**What this means:**
- Knowledge graph is built in RAM during runtime
- Data is **only saved when you explicitly save it** (TUI doesn't auto-save yet)
- If you close the TUI without saving, data is lost
- The `knowledge_graph.json` file must be explicitly written

### ⚠️ Important: Data Persistence

**Current Behavior (as of v0.1.0):**

1. When you `/load` a document → Knowledge graph is built in memory
2. When you `/stats` or `/entities` → Data is read from memory
3. When you **close the TUI** → Data is **NOT automatically saved**

**This means:**
- You need to manually save if you want persistence (future feature)
- OR keep the TUI running while working on a project
- OR reload documents each time you start the TUI

### 🔮 Future: Persistent Storage

Future versions will support:
- **Automatic saving** after document loading
- **LanceDB** integration for vector storage
- **SQLite** for structured data
- **Qdrant** for production vector search

Configuration example (future):
```toml
[storage]
type = "persistent"
backend = "lancedb"
path = "~/.graphrag/workspaces/{workspace_id}/vectors/"

[storage.lancedb]
uri = "~/.graphrag/workspaces/{workspace_id}/lancedb/"
table_name = "embeddings"
```

---

## 🛠️ Managing Storage

### Backup a Workspace

```bash
# Backup entire workspace
cp -r ~/.graphrag/workspaces/symposium ~/backups/symposium-$(date +%Y%m%d)

# Backup just the knowledge graph
cp ~/.graphrag/workspaces/symposium/knowledge_graph.json ~/backups/
```

### Restore a Workspace

```bash
# Restore from backup
cp -r ~/backups/symposium-20251016 ~/.graphrag/workspaces/symposium
```

### Delete a Workspace

```bash
# Using CLI (recommended)
./target/release/graphrag-cli workspace delete <workspace-id>

# Or manually
rm -rf ~/.graphrag/workspaces/<workspace-id>
```

### Export Knowledge Graph

```bash
# Pretty print knowledge graph
cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq '.' > exported_graph.json

# Extract just entities
cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq '.entities' > entities.json

# Extract just relationships
cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq '.relationships' > relationships.json
```

### Clean Up Old Workspaces

```bash
# Find workspaces not accessed in 30 days
find ~/.graphrag/workspaces/ -name metadata.json -mtime +30

# Script to clean old workspaces
for ws in ~/.graphrag/workspaces/*/; do
  last_access=$(cat "$ws/metadata.json" | jq -r '.last_accessed')
  echo "$ws last accessed: $last_access"
  # Add deletion logic if needed
done
```

---

## 📊 Storage Statistics

### Get Workspace Statistics

```bash
# Using CLI
./target/release/graphrag-cli workspace info <workspace-id>

# Manual calculation
echo "Workspace: symposium"
echo "Documents: $(cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq '.documents | length')"
echo "Entities: $(cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq '.entities | length')"
echo "Relationships: $(cat ~/.graphrag/workspaces/symposium/knowledge_graph.json | jq '.relationships | length')"
```

### Monitor Storage Growth

```bash
# Watch storage size in real-time
watch -n 5 'du -sh ~/.graphrag/'

# Track knowledge graph growth
ls -lh ~/.graphrag/workspaces/symposium/knowledge_graph.json
```

---

## 🔐 Security & Privacy

### Data Location

All data is stored locally in your home directory:
- **No cloud upload** by default
- **No external services** (except Ollama for LLM, which is also local)
- **Full control** over your data

### Sensitive Data

If your documents contain sensitive information:

```bash
# Encrypt workspace directory
tar czf - ~/.graphrag/workspaces/sensitive_project/ | \
  openssl enc -aes-256-cbc -out sensitive_project.tar.gz.enc

# Decrypt
openssl enc -d -aes-256-cbc -in sensitive_project.tar.gz.enc | \
  tar xz -C ~/.graphrag/workspaces/
```

### Permissions

Ensure proper file permissions:

```bash
# Set restrictive permissions
chmod 700 ~/.graphrag
chmod 600 ~/.graphrag/workspaces/*/knowledge_graph.json
chmod 600 ~/.graphrag/workspaces/*/metadata.json
```

---

## 🐛 Troubleshooting Storage Issues

### Problem: "Workspace not found"

```bash
# Check if workspace exists
ls ~/.graphrag/workspaces/

# Verify metadata is valid JSON
cat ~/.graphrag/workspaces/<id>/metadata.json | jq
```

### Problem: "Corrupted knowledge graph"

```bash
# Validate JSON
cat ~/.graphrag/workspaces/<id>/knowledge_graph.json | jq . > /dev/null
echo $?  # Should be 0 if valid

# Backup and recreate
cp ~/.graphrag/workspaces/<id>/knowledge_graph.json ~/backup.json
rm ~/.graphrag/workspaces/<id>/knowledge_graph.json
# Then reload documents in TUI
```

### Problem: "Out of disk space"

```bash
# Check disk usage
df -h ~

# Find largest workspaces
du -sh ~/.graphrag/workspaces/* | sort -hr

# Clean up old workspaces
./target/release/graphrag-cli workspace list
./target/release/graphrag-cli workspace delete <old-workspace-id>
```

---

## 📋 Summary Table

| Data Type | Path | Purpose | Auto-saved? |
|-----------|------|---------|-------------|
| Workspaces | `~/.graphrag/workspaces/` | All workspace data | ✓ |
| Knowledge Graph | `~/.graphrag/workspaces/<id>/knowledge_graph.json` | Entities, relationships, graph | ✗ (future) |
| Metadata | `~/.graphrag/workspaces/<id>/metadata.json` | Workspace info | ✓ |
| Query History | `~/.graphrag/workspaces/<id>/query_history.json` | Past queries | ✓ |
| Logs | `~/.local/share/graphrag-cli/logs/` | Application logs | ✓ |

---

## 🔮 Future Storage Features

Planned for future releases:

- ✨ **Auto-save** knowledge graph after document loading
- 📦 **LanceDB integration** for efficient vector storage
- 🗄️ **SQLite backend** for structured data
- ☁️ **Optional cloud sync** (user-controlled)
- 📤 **Export formats**: JSON, CSV, GraphML, Neo4j
- 🔍 **Full-text search** in stored documents
- 📊 **Storage analytics dashboard** in TUI
- 🗜️ **Compression** for large knowledge graphs

---

## 📚 Related Documentation

- [User Guide](USER_GUIDE.md) - Complete usage guide
- [README](README.md) - Technical documentation
- [Configuration Guide](../CONFIGURATION_GUIDE.md) - Config options

---

**Last Updated:** 2025-10-16
**Version:** 0.1.0
