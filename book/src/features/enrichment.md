# Document enrichment

The text pipeline can attach structural metadata such as headings, positions,
keywords, and extractive summaries to chunks. Enrichment is useful when an
application needs to inspect or rank document structure, but it is separate from
the minimal local quickstart.

Use the public `graphrag_core::text` types to construct enrichment explicitly.
Treat generated metadata as a retrieval aid and validate it on the document
formats used by the application.
