# Incremental updates

Indexing an existing workspace is an update path. Initialization reopens the
stored graph, and the next document is added to that state. Use a separate
workspace when a clean graph is required.

Advanced incremental managers and batch processing are available through
explicit core features. They can track changed content, update graph items, or
defer propagation, but their cost and consistency characteristics depend on the
selected backend. Enable only the feature needed by the application and test
updates against representative workspaces.
