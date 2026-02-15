# GraphRAG Pipeline Architecture

This document details the 7-phase architecture of the GraphRAG pipeline, covering both Indexing (Graph Construction) and Query (Retrieval & Generation).

## Phase 1: Chunking (Indexing)
**Goal:** Split documents into manageable segments for processing.
> [!NOTE]
> This phase **only** handles text splitting. Vectorization (Embedding) happens in Phase 5, either immediately for indexing or later during query processing.

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Maximum number of tokens per chunk | 300 |
| `chunk_overlap` | Number of overlapping tokens between chunks | 30 |

### Guidelines
- **Small Chunks (100-300 tokens):** Best for granular retrieval and precise fact extraction.
- **Large Chunks (500-1000 tokens):** Better for capturing broader context and thematic relationships.
- **Overlap:** Ensure critical information isn't split across boundaries. 10-20% overlap is recommended.

## Phase 2: Entity Extraction (Indexing)
**Goal:** Identify key entities (People, Places, Organizations, Concepts) from text chunks.

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `approach` | Extraction method: `algorithmic` (NLP), `semantic` (LLM), or `hybrid` | `hybrid` |
| `entity_types` | List of entity types to extract | `["organization", "person", "geo", "event"]` |
| `use_gleaning` | Enable multi-pass extraction for missed entities | `true` |

### Methods
1.  **Algorithmic:** Uses NLP libraries (e.g., SpaCy) for fast, rule-based extraction. Good for standard entities.
2.  **Semantic:** Uses LLMs to identify entities based on context and meaning. Best for domain-specific or abstract concepts.
3.  **Hybrid:** Combines both for speed and accuracy.

## Phase 3: Relationship Extraction (Indexing)
**Goal:** Identify relationships between extracted entities.

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `extract_relationships` | Enable/disable relationship extraction | `true` |
| `use_gleaning` | Enable multi-pass extraction for missed relationships | `true` |
| `max_gleanings` | Maximum number of gleaning passes | `1` |

### Methods
1.  **Co-occurrence:** rapid identification based on proximity in text.
2.  **LLM:** Detailed relationship extraction with semantic understanding.
3.  **Gleaning:** Iterative refinement to catch subtle connections.

## Phase 4: Graph Construction (Indexing)
**Goal:** Build a knowledge graph from entities and relationships.

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_pagerank` | Calculate node importance using PageRank | `true` |
| `max_connections` | limit max degree for nodes to reduce noise | `50` |
| `graph.traversal` | Traversal depth for sub-graph queries | `2` |
| `clustering_algorithm` | Algorithm for community detection (e.g., `leiden`) | `leiden` |

### Features
- **PageRank:** Identifies central nodes in the graph.
- **Leiden:** Detects communities and hierarchical structures.
- **Clustering:** Groups related entities for better context retrieval.

## Phase 5: Embedding (Indexing & Query)
**Goal:** Generate vector embeddings for text chunks and graph nodes.
> [!IMPORTANT]
> This phase runs during **Indexing** (to vectorize chunks/nodes) AND during **Query** (to vectorize the user's question).

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend` | Embedding provider (OpenAI, Azure, Ollama, etc.) | `openai` |
| `dimension` | Vector dimension size | `1536` |
| `model` | Specific embedding model name | `text-embedding-3-small` |
| `batch_size` | Number of items to embed at once | `16` |

### Providers & Costs
| Provider | Model | Cost ($/1M tokens) | Dimensions |
|----------|-------|--------------------|------------|
| OpenAI | text-embedding-3-small | $0.02 | 1536 |
| OpenAI | text-embedding-3-large | $0.13 | 3072 |
| Azure | (varies) | (varies) | (varies) |
| Ollama | (local) | $0.00 | (varies) |

## Phase 6: Retrieval (Query)
**Goal:** Find relevant information based on user query.

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `strategy` | Retrieval strategy (`vector`, `graph`, `hybrid`, `local`, `global`) | `hybrid` |
| `top_k` | Number of results to return | `10` |
| `retrieval.score_threshold` | Minimum similarity score | `0.7` |

### Strategies
1.  **Vector:** Standard semantic search using embeddings.
2.  **Graph:** Traverses the knowledge graph to find related entities.
3.  **Hybrid:** Combines Vector and Graph search for best coverage.
4.  **Local:** Focuses on immediate neighborhood of query entities.
5.  **Global:** Scans entire graph/communities for broad queries.
6.  **Text:** Keyword-based search (BM25 or similar).

## Phase 7: Answer Generation (Query)
**Goal:** Synthesize a final answer from retrieved context.

### Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `chat_model` | LLM for generation | `gpt-4o` |
| `temperature` | Creativity control (0.0 - 1.0) | `0.0` |
| `max_tokens` | Maximum length of generated answer | `2000` |
| `ollama.model` | Local model name if using Ollama | `llama3` |

### Methods
1.  **LLM:** Direct generation using retrieved context.
2.  **Concatenation:** Simple joining of retrieved text segments.
3.  **Explained:** Generates answer with citations and reasoning steps.

---

## Quick Reference: All Configurable Parameters

| Section | Parameter | Type | Default | Description |
|---------|-----------|------|---------|-------------|
| **Chunking** | `chunk_size` | int | 300 | Max tokens per chunk |
| | `chunk_overlap` | int | 30 | Overlap tokens |
| **Extraction** | `entity_extraction.approach` | enum | hybrid | Extraction method |
| | `entity_extraction.entity_types` | list | [...] | Target entities |
| | `entity_extraction.use_gleaning` | bool | true | Multi-pass extraction |
| | `relationship_extraction.enabled` | bool | true | Extract relationships |
| **Graph** | `graph.enable_pagerank` | bool | true | Calculate PageRank |
| | `graph.max_connections` | int | 50 | Max node degree |
| | `graph.clustering_algorithm` | enum | leiden | Community detection |
| **Embedding** | `embeddings.backend` | string | openai | Provider name |
| | `embeddings.model` | string | ... | Model name |
| | `embeddings.dimension` | int | 1536 | Vector size |
| **Retrieval** | `retrieval.strategy` | enum | hybrid | Search strategy |
| | `retrieval.top_k` | int | 10 | 	Result count |
| **Generation** | `llm.model` | string | gpt-4o | Chat model |
| | `llm.temperature` | float | 0.0 | Creativity |

## Performance Table

| Configuration | Indexing Speed (100 chunks) | Query Latency | Cost Estimate |
|---------------|-----------------------------|---------------|---------------|
| **Light** (Algo Extract, Vector Only) | ~30s | ~500ms | Low |
| **Balanced** (Hybrid Extract, Hybrid Search) | ~2m | ~1.5s | Medium |
| **Deep** (All Semantic, Global Search) | ~10m | ~5s+ | High |

---

## Scientific Basis & Inspirations

GraphRAG-rs implements techniques from several cutting-edge research papers. Here is how they map to our 7-phase pipeline:

| Phase | Technique | Inspiration / Paper | Impact |
|-------|-----------|---------------------|--------|
| **1. Chunking** | **Semantic Chunking** | *LangChain / Gregory Kamradt (2024)* | Respects semantic boundaries better than fixed-size splitting. |
| **2. Extraction** | **Gleaning** | *Microsoft GraphRAG (2024)* | Iterative extraction to catch entities missed in the first pass. |
| **4. Graph** | **Leiden Algorithm** | *Traag et al. (Nature, 2019)* | Superior community detection (modularity) vs. Louvain. |
| **4. Graph** | **Hierarchical Communities** | *Microsoft GraphRAG (2024)* | Enables "Global Search" by summarizing communities at different levels. |
| **6. Retrieval** | **Fast-GraphRAG** | *Pang et al. (2024)* | Uses PageRank/PPR to traverse the graph efficiently. |
| **6. Retrieval** | **LightRAG** | *Zhang et al. (2024)* | Dual-level retrieval (Low-level keywords + High-level topics) for 6000x efficiency. |
| **6. Retrieval** | **HippORAG** | *He et al. (NeurIPS 2024)* | "In-context" learning on the graph via Personalized PageRank (PPR). |
| **6. Retrieval** | **Cross-Encoder** | *Reimers et al. (2019)* | Reranking step to drastically improve precision of retrieved results. |

## Time Estimates (per 100 chunks)
- **Chunking:** < 1s
- **Entity/Rel Extraction (LLM):** ~2-5 mins
- **Graph Construction:** ~10-30s
- **Embedding:** ~10-30s
- **Total Indexing:** ~3-6 mins (highly dependent on LLM speed)
