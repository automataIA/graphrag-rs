# GraphRAG Pipeline Benchmark Report

> Generated: 2026-02-23T08:21:07+01:00

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | 3 |
| Approaches | algorithmic, semantic |
| LLM Models | mistral-nemo:latest, none |

## Pipeline Comparison

| Pipeline | Book | Approach | LLM | Chunks | Entities | Relations | Build (ms) | Query (ms) |
|----------|------|----------|-----|--------|----------|-----------|------------|------------|
| algo_hash_small | symposium | algorithmic | none | 256/50 | 1862 | 1669 | 146 | 92 |
| kv_semantic_mistral | symposium | — | — | — | — | — | dry-run | — |
| mock_future_rag | symposium | algorithmic | none | 256/50 | 1862 | 1669 | 68 | 35 |

## Detailed Q&A Results

### algo_hash_small — symposium

**Parameters:**
- Approach: algorithmic
- LLM: none (T=0.0)
- Embedding: hash/none
- Chunks: 256/50
- Gleaning: false (rounds=0)
- LightRAG: false | Leiden: false | CrossEncoder: false

| # | Question | Answer (truncated) | Time (ms) |
|---|----------|-------------------|-----------|
| 1 | What is the nature of love according to Socrates? | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise | 50 |
| 2 | Who are the main speakers in the Symposium? | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise | 12 |
| 3 | What is Aristophanes' myth about the origin of love? | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. ( | 10 |
| 4 | How does Diotima describe the ladder of beauty? | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise | 13 |
| 5 | What is the relationship between love and wisdom? | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. ( | 4 |

### mock_future_rag — symposium

**Parameters:**
- Approach: algorithmic
- LLM: none (T=0.0)
- Embedding: hash/none
- Chunks: 256/50
- Gleaning: false (rounds=0)
- LightRAG: true | Leiden: true | CrossEncoder: false

| # | Question | Answer (truncated) | Time (ms) |
|---|----------|-------------------|-----------|
| 1 | What is the nature of love according to Socrates? | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise | 24 |
| 2 | Who are the main speakers in the Symposium? | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise | 2 |
| 3 | What is Aristophanes' myth about the origin of love? | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. ( | 3 |
| 4 | How does Diotima describe the ladder of beauty? | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise | 2 |
| 5 | What is the relationship between love and wisdom? | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. ( | 2 |

## Cross-Pipeline Answer Comparison

Compare how different pipelines answer the same questions:

### Book: symposium

**Q1: What is the nature of love according to Socrates?**

| Pipeline | Approach | LLM | Answer (first 150 chars) | Time |
|----------|----------|-----|--------------------------|------|
| algo_hash_small | algorithmic | none | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise of love spoken by Socrates a | 50ms |
| mock_future_rag | algorithmic | none | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise of love spoken by Socrates a | 24ms |

**Q2: Who are the main speakers in the Symposium?**

| Pipeline | Approach | LLM | Answer (first 150 chars) | Time |
|----------|----------|-----|--------------------------|------|
| algo_hash_small | algorithmic | none | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise of love spoken by Socrates a | 12ms |
| mock_future_rag | algorithmic | none | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise of love spoken by Socrates a | 2ms |

**Q3: What is Aristophanes' myth about the origin of love?**

| Pipeline | Approach | LLM | Answer (first 150 chars) | Time |
|----------|----------|-----|--------------------------|------|
| algo_hash_small | algorithmic | none | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who | 10ms |
| mock_future_rag | algorithmic | none | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who | 3ms |

**Q4: How does Diotima describe the ladder of beauty?**

| Pipeline | Approach | LLM | Answer (first 150 chars) | Time |
|----------|----------|-----|--------------------------|------|
| algo_hash_small | algorithmic | none | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise of love spoken by Socrates a | 13ms |
| mock_future_rag | algorithmic | none | osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who had heard of the discourses in praise of love spoken by Socrates a | 2ms |

**Q5: What is the relationship between love and wisdom?**

| Pipeline | Approach | LLM | Answer (first 150 chars) | Time |
|----------|----------|-----|--------------------------|------|
| algo_hash_small | algorithmic | none | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who | 4ms |
| mock_future_rag | algorithmic | none | Love (PERSON) (score: 1338.43) Aristophanes (PERSON) (score: 24.44) osophy' has at least a superficial reconcilement. (Rep.)  An unknown person who | 2ms |

