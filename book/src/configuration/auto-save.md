# Auto-save and persistence

The graph can be saved to a workspace and reopened by a later process. Configure
the workspace explicitly when using the library:

```toml
[auto_save]
enabled = true
base_dir = "/tmp/graphrag-workspaces"
workspace_name = "prisma"
```

After building, call `save_to_workspace()`. Create the next `GraphRAG` instance
with the same effective configuration and call `initialize()` to reopen the
saved graph. The workspace stores graph data and metadata; the original source
text remains the application's responsibility.

The CLI's `--workspace` option supplies the persistence location for `index` and
`ask`. Use a new workspace for an isolated run. Indexing an existing workspace
is an update path and can retain documents already stored there.
