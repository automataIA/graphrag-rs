# Configuration templates

The files in this directory are starting points for different document styles. They are
examples of settings, not quality or performance guarantees. Check the selected file with
`graphrag validate` before using it and adjust the values for the corpus and providers in
your application.

## Available templates

| File | Intended starting point |
|---|---|
| `narrative_fiction.toml` | novels, stories, and character focused text |
| `technical_documentation.toml` | API references and software manuals |
| `academic_research.toml` | papers, theses, and literature reviews |
| `legal_documents.toml` | contracts, regulations, and case law |
| `web_blog_content.toml` | articles and other short web documents |
| `semantic_pipeline.toml` | provider backed semantic extraction |
| `algorithmic_pipeline.toml` | local pattern based extraction |
| `hybrid_pipeline.toml` | a mixed local and provider pipeline |

The JSON5 templates use the same runtime configuration model and allow comments. TOML,
JSON, and JSON5 loaders accept the canonical nested runtime fields. Existing files using
legacy top level names remain readable through the compatibility adapter; conflicting
canonical and legacy values are rejected.

## Use a template

Copy a file, edit the document independent settings, then validate it:

```bash
cp config/templates/technical_documentation.toml my-graphrag.toml
graphrag validate my-graphrag.toml
```

The most common local settings are:

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5
```

`chunk_size` must be positive, `chunk_overlap` must be smaller than `chunk_size`, and
`retrieval.top_k` must be greater than zero. The effective configuration is merged in this
order: built in defaults, the home file, the project file, and `GRAPHRAG_*` environment
variables. See the [configuration guide](../../book/src/configuration/configuration.md)
for the complete precedence and environment naming rules.

Templates that enable the `llm` section (llama.cpp) or another provider require that provider's feature, service,
credentials, and model configuration. The base local quickstart needs none of them and uses
hash embeddings, pattern extraction, and lexical retrieval.
