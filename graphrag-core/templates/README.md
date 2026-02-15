# GraphRAG Configuration Templates

Pre-configured templates for common use cases. Each template is optimized for specific document types with appropriate entity extraction, chunking, and retrieval settings.

## Available Templates

| Template | Best For | Approach | Ollama Required |
|----------|----------|----------|-----------------|
| `general.toml` | Mixed documents, articles | Hybrid | No |
| `legal.toml` | Contracts, legal agreements | Semantic | Yes |
| `medical.toml` | Clinical notes, patient records | Semantic | Yes |
| `financial.toml` | Financial reports, SEC filings | Hybrid | Yes |
| `technical.toml` | API docs, code documentation | Algorithmic | No |

## Quick Usage

### Option 1: CLI Setup Wizard (Recommended)

```bash
# Interactive setup with template selection
graphrag-cli setup --template legal --output ./graphrag.toml
```

### Option 2: Copy Template

```bash
# Copy template to your project
cp graphrag-core/templates/general.toml ./graphrag.toml

# Edit as needed, then use GraphRAG
```

### Option 3: Programmatic

```rust
use graphrag_core::{Config, GraphRAG};

// Load template directly
let config = Config::from_toml_file("templates/legal.toml")?;
let mut graphrag = GraphRAG::new(config)?;
```

### Option 4: With Environment Overrides

```bash
# Override template values with environment variables
export GRAPHRAG_OLLAMA_HOST=my-server
export GRAPHRAG_CHUNK_SIZE=2000

# Then load template (env vars take priority)
let config = Config::load()?;  // Requires hierarchical-config feature
```

## Template Details

### General Purpose (`general.toml`)

Best for mixed content like news articles, reports, and general documents.

**Entity types:**
- PERSON, ORGANIZATION, LOCATION, DATE, EVENT

**Settings:**
- Chunk size: 1000 (balanced for mixed content)
- Approach: Hybrid (algorithmic + semantic when available)
- Gleaning: Disabled (optional)

**Use cases:**
- News articles
- Blog posts
- Research reports
- General documents

---

### Legal Documents (`legal.toml`)

Optimized for contracts, agreements, and regulatory documents.

**Entity types:**
- PARTY, PERSON, ORGANIZATION, DATE
- MONETARY_VALUE, JURISDICTION
- CLAUSE_TYPE, OBLIGATION, RIGHT, TERM

**Settings:**
- Chunk size: 500 (smaller for precise clause extraction)
- Approach: Semantic (LLM-based for better clause understanding)
- Gleaning: Enabled (4 rounds for thorough extraction)
- Confidence threshold: 0.75

**Use cases:**
- Contracts and NDAs
- Court documents
- Regulations and compliance docs
- Legal memos

---

### Medical Documents (`medical.toml`)

Configured for clinical and healthcare documentation.

**Entity types:**
- PATIENT, DIAGNOSIS, MEDICATION, PROCEDURE
- SYMPTOM, LAB_VALUE, VITAL_SIGN
- ANATOMICAL_SITE, PROVIDER

**Settings:**
- Chunk size: 750 (medium for clinical notes)
- Approach: Semantic (LLM for medical terminology)
- Gleaning: Enabled (for complex medical terms)
- High relationship extraction

**Use cases:**
- Patient records
- Clinical notes
- Research papers
- Discharge summaries

**HIPAA Note:** Ensure compliance when processing patient data. Consider using local Ollama deployment for data privacy.

---

### Financial Documents (`financial.toml`)

Tailored for financial reports and analysis.

**Entity types:**
- COMPANY, TICKER, PERSON
- MONETARY_VALUE, PERCENTAGE
- FISCAL_PERIOD, METRIC, INDUSTRY, EVENT

**Settings:**
- Chunk size: 1200 (larger for financial narratives)
- Approach: Hybrid (algorithmic for numbers, semantic for context)
- Gleaning: Enabled (for financial terminology)

**Use cases:**
- 10-K and 10-Q filings
- Earnings call transcripts
- Analyst reports
- Financial statements

---

### Technical Documentation (`technical.toml`)

Optimized for API docs and code documentation.

**Entity types:**
- FUNCTION, CLASS, MODULE
- API_ENDPOINT, PARAMETER
- VERSION, DEPENDENCY, CONFIG_KEY

**Settings:**
- Chunk size: 600 (smaller to preserve code boundaries)
- Approach: Algorithmic (pattern-based for structured docs)
- Gleaning: Disabled (patterns work well for code)

**Use cases:**
- API documentation
- Code documentation
- Architecture specs
- Configuration guides

## Customization

Each template can be customized by editing the TOML file:

### Adjust Entity Extraction

```toml
[entities]
min_confidence = 0.8  # Higher = fewer but more accurate entities
entity_types = ["CUSTOM_TYPE_1", "CUSTOM_TYPE_2"]

# Enable LLM-based multi-round extraction
use_gleaning = true
max_gleaning_rounds = 4
```

### Modify Chunking

```toml
chunk_size = 1000     # Characters per chunk
chunk_overlap = 200   # Overlap between chunks
```

### Configure Ollama

```toml
[ollama]
enabled = true
host = "localhost"
port = 11434
chat_model = "llama3.2:3b"
embedding_model = "nomic-embed-text"
timeout_seconds = 60
```

### Enable Advanced Features

```toml
# PageRank for graph-based retrieval
[graph]
enable_pagerank = true

# Caching for reduced API costs
[generation]
enable_caching = true

# Parallel processing
[parallel]
enabled = true
num_threads = 4
```

## Environment Variable Overrides

With the `hierarchical-config` feature, environment variables override template values:

```bash
# Override Ollama host
export GRAPHRAG_OLLAMA_HOST=my-ollama-server

# Override chunk size
export GRAPHRAG_CHUNK_SIZE=2000

# Override approach
export GRAPHRAG_APPROACH=semantic
```

## Creating Custom Templates

1. Start with the closest existing template
2. Customize entity types for your domain
3. Adjust chunk size based on content structure
4. Enable/disable gleaning based on complexity
5. Save as `your-domain.toml`

**Example custom template:**

```toml
# academic.toml - For academic papers
output_dir = "./academic-output"
approach = "semantic"
chunk_size = 800

[entities]
min_confidence = 0.75
entity_types = [
    "AUTHOR",
    "INSTITUTION",
    "PUBLICATION",
    "METHODOLOGY",
    "FINDING",
    "CITATION",
    "DATE"
]
use_gleaning = true
max_gleaning_rounds = 3

[ollama]
enabled = true
chat_model = "llama3.2:3b"
```

## Comparison Matrix

| Feature | General | Legal | Medical | Financial | Technical |
|---------|---------|-------|---------|-----------|-----------|
| Chunk Size | 1000 | 500 | 750 | 1200 | 600 |
| Approach | Hybrid | Semantic | Semantic | Hybrid | Algorithmic |
| Gleaning | Off | 4 rounds | 3 rounds | 3 rounds | Off |
| Entity Types | 5 | 10 | 9 | 8 | 7 |
| Ollama Required | No | Yes | Yes | Yes | No |
| Processing Speed | Fast | Slow | Medium | Medium | Fast |
