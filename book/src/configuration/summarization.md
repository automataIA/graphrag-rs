# Summarization

Summarization is an optional processing stage. The base 0.3 quickstart uses
local chunking, pattern extraction, and retrieval without an LLM. Applications
that enable a supported provider may configure extractive or provider-backed
summaries in the relevant pipeline section.

Keep provider configuration explicit and validate it before indexing. A
provider-backed summary requires the matching Cargo feature, a reachable
service, and a configured model. If those requirements are absent, use the
local extraction path rather than assuming that a summary was generated.
