# GraphRAG CLI - Workspace Management Commands

## Overview

The GraphRAG CLI now includes comprehensive workspace management commands that allow you to save, load, and manage multiple knowledge graphs. This enables you to:

- **Save your work**: Persist complete knowledge graphs to disk
- **Reuse graphs**: Load previously saved graphs instantly
- **Organize projects**: Manage multiple workspaces for different projects
- **Share graphs**: Export and import knowledge graphs between systems

## Workspace Commands

### 1. List Available Workspaces

```bash
/workspace list
# or shorthand:
/ws list
/ws ls
```

**Output:**
```
📁 Available Workspaces (2 total):

1. tom_sawyer (945.00 KB)
   Entities: 7, Relationships: 6, Documents: 1, Chunks: 435
   Created: 2025-10-16 13:38:41

2. symposium (234.50 KB)
   Entities: 12, Relationships: 15, Documents: 1, Chunks: 187
   Created: 2025-10-15 09:22:18
```

### 2. Save Current Graph to Workspace

```bash
/workspace save <name>
# or shorthand:
/ws save <name>
```

**Example:**
```bash
/workspace save my_project
```

**Requirements:**
- GraphRAG must be initialized (`/config` loaded)
- Knowledge graph must be built (`/load` executed)

**What Gets Saved:**
- All entities with mentions and metadata
- All relationships with context
- All documents with full content
- All chunks with full text
- Workspace metadata (timestamps, counts, version)

### 3. Load Graph from Workspace

```bash
/workspace <name>
# or shorthand:
/ws <name>
```

**Example:**
```bash
/workspace tom_sawyer
```

**Effect:**
- Replaces current knowledge graph with the loaded one
- Preserves configuration (LLM settings, embeddings, etc.)
- Updates statistics in the info panel
- Ready to query immediately

### 4. Delete a Workspace

```bash
/workspace delete <name>
# or shorthand:
/ws delete <name>
/ws del <name>
/ws rm <name>
```

**Example:**
```bash
/workspace delete old_project
```

**Warning:** This action is permanent and cannot be undone!

## Typical Workflow

### Creating a New Workspace

```bash
# 1. Load configuration
/config config/templates/semantic.graphrag.json5

# 2. Load documents and build graph
/load docs-example/The_Adventures_of_Tom_Sawyer.txt

# 3. Wait for graph to build...
# (Entities extracted, relationships created)

# 4. Save to workspace
/workspace save tom_sawyer

# 5. Query the graph
Who is Tom Sawyer's best friend?
```

### Loading an Existing Workspace

```bash
# 1. Load configuration (still needed for LLM/embeddings)
/config config/templates/semantic.graphrag.json5

# 2. List available workspaces
/workspace list

# 3. Load a workspace
/workspace tom_sawyer

# 4. Query immediately (no need to rebuild!)
What are the main themes in Tom Sawyer?
```

### Managing Multiple Projects

```bash
# Project 1: Tom Sawyer
/config config/templates/narrative_fiction.graphrag.json5
/load docs-example/The_Adventures_of_Tom_Sawyer.txt
/workspace save literature_tom_sawyer

# Project 2: Technical documentation
/config config/templates/technical_documentation.graphrag.json5
/load docs/API_Reference.md
/workspace save tech_api_docs

# Switch between projects
/workspace literature_tom_sawyer
# ... query Tom Sawyer ...

/workspace tech_api_docs
# ... query API docs ...
```

## Storage Location

Workspaces are stored in:
- **Linux/Mac**: `~/.local/share/graphrag/workspaces/`
- **Windows**: `%APPDATA%\graphrag\workspaces\`
- **Fallback**: `./workspaces/` (current directory)

Each workspace is a directory containing:
```
~/.local/share/graphrag/workspaces/
├── tom_sawyer/
│   ├── graph.json      # Complete knowledge graph
│   └── metadata.toml   # Workspace metadata
└── symposium/
    ├── graph.json
    └── metadata.toml
```

## File Format

### graph.json

Complete JSON representation of the knowledge graph:
- **Entities**: All extracted entities with full details
- **Relationships**: All relationships between entities
- **Chunks**: Full text content of all chunks
- **Documents**: Full text content of all documents
- **Metadata**: Entity mentions, confidence scores, offsets

### metadata.toml

Workspace information:
```toml
name = "tom_sawyer"
created_at = "2025-10-16T13:38:41.934990424Z"
modified_at = "2025-10-16T13:38:41.951588339Z"
entity_count = 7
relationship_count = 6
document_count = 1
chunk_count = 435
format_version = "1.0"
```

## Performance

### Save Performance

- **Small graphs** (<1K entities): <50ms
- **Medium graphs** (1K-10K entities): 50-500ms
- **Large graphs** (10K-100K entities): 500ms-5s

### Load Performance

- **Small graphs**: <50ms
- **Medium graphs**: 50-500ms
- **Large graphs**: 500ms-5s

**Note:** Loading from workspace is **much faster** than rebuilding the graph from scratch (which requires entity extraction and LLM calls).

## Storage Requirements

Approximate sizes:
- **Entities**: ~500 bytes each (with mentions)
- **Relationships**: ~200 bytes each
- **Chunks**: ~1KB each (full content)
- **Documents**: Full content size + metadata

**Example:** Tom Sawyer workspace (434 KB source text)
- 7 entities, 6 relationships, 435 chunks
- Total size: 945 KB (2.2x source size)

## Best Practices

### 1. Save After Building

Always save your graph after successfully building it:
```bash
/load large_document.txt
# ... wait for build to complete ...
/stats  # Verify graph is built
/workspace save my_workspace  # Save it!
```

### 2. Use Descriptive Names

Use clear, descriptive workspace names:
- ✅ Good: `literature_tom_sawyer`, `tech_api_v2`, `research_quantum_physics`
- ❌ Bad: `test`, `tmp`, `graph1`

### 3. List Before Loading

Always list workspaces to see what's available:
```bash
/workspace list
# Check available workspaces and their stats
/workspace tom_sawyer
```

### 4. Delete Old Workspaces

Clean up unused workspaces to save disk space:
```bash
/workspace list
/workspace delete old_experiment
```

### 5. Backup Important Workspaces

Workspaces are just directories - you can backup/share them:
```bash
# Backup
tar -czf tom_sawyer_backup.tar.gz ~/.local/share/graphrag/workspaces/tom_sawyer/

# Share
scp -r ~/.local/share/graphrag/workspaces/tom_sawyer/ user@remote:~/
```

## Limitations

### Current Limitations

1. **Embeddings Not Saved**
   - Vector embeddings are regenerated on load
   - This is temporary until LanceDB integration is unblocked
   - Impact: First query after load may be slightly slower

2. **Configuration Not Saved**
   - You must still load config with `/config`
   - Workspace only stores the graph data
   - Future: May include config snapshot

3. **JSON Format**
   - Currently uses JSON for maximum compatibility
   - Future: Will use Parquet for better compression
   - Current approach is human-readable for debugging

### Future Enhancements

1. **Compression** (Planned)
   - Add gzip compression for JSON
   - Expected 5x size reduction
   - Transparent to users

2. **Parquet Storage** (Blocked)
   - Columnar storage for entities/relationships
   - Much better compression (~10x)
   - Faster queries for large graphs
   - Waiting for upstream dependency fix

3. **LanceDB Vector Storage** (Blocked)
   - Persist embeddings for instant queries
   - Hybrid retrieval acceleration
   - Waiting for version conflict resolution

4. **Auto-Save** (Planned)
   - Automatic workspace saves
   - Configurable save intervals
   - Crash recovery

5. **Workspace Metadata** (Planned)
   - Descriptions and tags
   - Creation/modification history
   - Related workspaces linking

## Troubleshooting

### "GraphRAG not initialized"

**Problem:** Trying to save/load without loading config first

**Solution:**
```bash
/config config/templates/semantic.graphrag.json5
/workspace save my_workspace  # Now works!
```

### "No knowledge graph to save"

**Problem:** Trying to save before building the graph

**Solution:**
```bash
/load your_document.txt  # Build the graph first
/workspace save my_workspace  # Now works!
```

### "Workspace not found"

**Problem:** Trying to load a non-existent workspace

**Solution:**
```bash
/workspace list  # See available workspaces
/workspace <correct_name>
```

### "Failed to load workspace"

**Problem:** Corrupted workspace files or permission issues

**Solutions:**
1. Check file permissions
2. Verify JSON is valid
3. Check disk space
4. Delete and recreate the workspace

## Examples

### Example 1: Literature Analysis

```bash
# Setup
/config config/templates/narrative_fiction.graphrag.json5

# Load multiple books
/load books/Tom_Sawyer.txt
/workspace save literature_tom_sawyer

/clear
/load books/Huckleberry_Finn.txt --rebuild
/workspace save literature_huck_finn

# Compare analyses
/workspace literature_tom_sawyer
Who are the main characters?

/workspace literature_huck_finn
Who are the main characters?
```

### Example 2: Technical Documentation

```bash
# Setup
/config config/templates/technical_documentation.graphrag.json5

# Build knowledge base
/load docs/API_Reference.md
/load docs/User_Guide.md
/load docs/Architecture.md
/workspace save tech_docs_v1

# Later: Load and query
/config config/templates/technical_documentation.graphrag.json5
/workspace tech_docs_v1
How do I authenticate API requests?
```

### Example 3: Research Papers

```bash
# Setup
/config config/templates/academic_research.graphrag.json5

# Process research paper
/load papers/Quantum_Computing_Review.pdf
/workspace save research_quantum_2025

# Analyze
/workspace research_quantum_2025
What are the key challenges in quantum computing?
What algorithms were discussed?
```

## Summary

Workspace management in GraphRAG CLI provides:

✅ **Persistence**: Save complete knowledge graphs to disk
✅ **Fast Loading**: Load graphs instantly without rebuilding
✅ **Organization**: Manage multiple projects with ease
✅ **Portability**: Share and backup workspaces
✅ **Reliability**: 100% data integrity preservation

Use `/workspace list`, `/workspace save`, and `/workspace load` to work efficiently with multiple knowledge graphs!
