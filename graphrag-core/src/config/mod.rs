//! Configuration types for GraphRAG.
//!
//! Holds the runtime [`Config`] struct used by `GraphRAG`, plus the serde-friendly
//! `SetConfig` schema (TOML / JSON5) that maps onto it. Both must stay in sync when
//! adding fields — see the crate `CLAUDE.md` for the update checklist.

use crate::Result;
use std::fs;

/// Enhanced configuration options for GraphRAG
pub mod enhancements;
/// Hand-rolled JSON loader/writer for `Config` (extracted Phase 4 split).
mod json_parser;
/// JSON5 configuration support
#[cfg(feature = "json5-support")]
pub mod json5_loader;
/// Configuration file loading utilities
pub mod loader;
/// JSON Schema validation
#[cfg(feature = "json5-support")]
pub mod schema_validator;
/// SetConfig configuration support (TOML, JSON5, YAML, JSON)
pub mod setconfig;
/// Configuration validation utilities
pub mod validation;

pub use setconfig::{
    AlgorithmicEmbeddingsConfig,
    AlgorithmicEntityConfig,
    AlgorithmicGraphConfig,
    // Algorithmic/Classic NLP pipeline
    AlgorithmicPipelineConfig,
    AlgorithmicRetrievalConfig,
    HybridEmbeddingsConfig,
    HybridEntityConfig,
    HybridGraphConfig,
    // Hybrid pipeline
    HybridPipelineConfig,
    HybridRetrievalConfig,
    HybridWeightsConfig,
    // Pipeline approach configuration
    ModeConfig,
    SemanticEmbeddingsConfig,
    SemanticEntityConfig,
    SemanticGraphConfig,
    // Semantic/Neural pipeline
    SemanticPipelineConfig,
    SemanticRetrievalConfig,
    SetConfig,
};
pub use validation::{validate_config_file, Validatable, ValidationResult};

/// Configuration for the GraphRAG system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    /// Output directory for storing graphs and data
    pub output_dir: String,

    /// Chunk size for text processing
    pub chunk_size: usize,

    /// Overlap between chunks
    pub chunk_overlap: usize,

    /// Maximum entities per chunk
    pub max_entities_per_chunk: Option<usize>,

    /// Top-k results for retrieval
    pub top_k_results: Option<usize>,

    /// Similarity threshold for retrieval
    pub similarity_threshold: Option<f32>,

    /// Pipeline approach: "semantic", "algorithmic", or "hybrid"
    /// Determines which implementation strategy to use for entity extraction and retrieval
    #[serde(default = "default_approach")]
    pub approach: String,

    /// Vector embedding configuration
    pub embeddings: EmbeddingConfig,

    /// Graph construction parameters
    pub graph: GraphConfig,

    /// Text processing settings
    pub text: TextConfig,

    /// Entity extraction settings
    pub entities: EntityConfig,

    /// Retrieval system configuration
    pub retrieval: RetrievalConfig,

    /// Parallel processing configuration
    pub parallel: ParallelConfig,

    /// Ollama integration configuration
    pub ollama: crate::ollama::OllamaConfig,

    /// GLiNER-Relex extractor configuration
    pub gliner: GlinerConfig,

    /// Latest enhancements configuration
    pub enhancements: enhancements::EnhancementsConfig,

    /// Auto-save configuration for workspace persistence
    pub auto_save: AutoSaveConfig,

    /// Hierarchical summarization configuration
    pub summarization: crate::summarization::HierarchicalConfig,

    /// Zero-cost approach configuration
    pub zero_cost_approach: ZeroCostApproachConfig,

    /// Advanced features configuration (Phases 2-3)
    #[serde(default)]
    pub advanced_features: AdvancedFeaturesConfig,

    /// Suppress indicatif progress bars (use hidden draw target).
    /// Set to `true` when running inside a TUI to avoid corrupting the terminal.
    #[serde(default)]
    pub suppress_progress_bars: bool,
}

/// GLiNER-Relex extractor configuration (joint NER + RE via ONNX Runtime)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlinerConfig {
    /// Enable GLiNER-Relex extraction
    pub enabled: bool,
    /// Path to the ONNX model file (e.g. "models/gliner-relex-large-v0.5.onnx")
    pub model_path: String,
    /// Path to tokenizer.json — defaults to same directory as model_path if empty
    pub tokenizer_path: String,
    /// Span-based ("span", default) or token-based ("token") NER pipeline
    pub mode: String,
    /// Entity types to extract
    pub entity_labels: Vec<String>,
    /// Relation types to extract (empty list disables RE stage)
    pub relation_labels: Vec<String>,
    /// Minimum entity confidence threshold (0.0–1.0)
    pub entity_threshold: f32,
    /// Minimum relation confidence threshold (0.0–1.0)
    pub relation_threshold: f32,
    /// Use GPU (CUDA) for inference
    pub use_gpu: bool,
    /// Max concurrent chunk inferences. `None` → default 4. Set to 1 to force
    /// sequential. Cap matches CPU cores or GPU stream count for best throughput.
    #[serde(default)]
    pub max_concurrent_chunks: Option<usize>,
}

impl Default for GlinerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: String::new(),
            tokenizer_path: String::new(),
            mode: "span".to_string(),
            entity_labels: vec![
                "person".into(),
                "organization".into(),
                "location".into(),
                "concept".into(),
            ],
            relation_labels: vec!["related to".into(), "part of".into(), "causes".into()],
            entity_threshold: 0.4,
            relation_threshold: 0.5,
            use_gpu: false,
            max_concurrent_chunks: None,
        }
    }
}

/// Configuration for automatic workspace saving
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoSaveConfig {
    /// Enable persistent storage. When false (default), the graph lives in memory only.
    /// When true, the graph is saved to disk after `build_graph()` and loaded from disk
    /// on the next `initialize()` call (if the workspace already exists).
    #[serde(default)]
    pub enabled: bool,

    /// Base directory where workspace folders are stored.
    /// Required when `enabled = true`. Example: `"./output"` or `"/data/graphrag"`.
    #[serde(default)]
    pub base_dir: Option<String>,

    /// Auto-save interval in seconds (0 = save after every graph build)
    #[serde(default = "default_auto_save_interval")]
    pub interval_seconds: u64,

    /// Workspace name — the sub-folder inside `base_dir` (default: "default").
    #[serde(default)]
    pub workspace_name: Option<String>,

    /// Maximum number of auto-save versions to keep (0 = unlimited)
    #[serde(default = "default_max_versions")]
    pub max_versions: usize,
}

/// Configuration for zero-cost GraphRAG approaches
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroCostApproachConfig {
    /// Which zero-cost approach to use
    #[serde(default = "default_zero_cost_approach")]
    pub approach: String,

    /// LazyGraphRAG-style configuration
    #[serde(default)]
    pub lazy_graphrag: LazyGraphRAGConfig,

    /// E2GraphRAG-style configuration
    #[serde(default)]
    pub e2_graphrag: E2GraphRAGConfig,

    /// Pure algorithmic configuration
    #[serde(default)]
    pub pure_algorithmic: PureAlgorithmicConfig,

    /// Hybrid strategy configuration
    #[serde(default)]
    pub hybrid_strategy: HybridStrategyConfig,
}

/// Configuration for LazyGraphRAG, an efficient approach for large-scale knowledge graphs.
/// This configuration enables lazy loading and processing of graph components.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct LazyGraphRAGConfig {
    /// Whether LazyGraphRAG is enabled
    pub enabled: bool,
    /// Configuration for concept extraction from text
    pub concept_extraction: ConceptExtractionConfig,
    /// Configuration for co-occurrence analysis of concepts
    pub co_occurrence: CoOccurrenceConfig,
    /// Configuration for lazy indexing of graph components
    pub indexing: LazyIndexingConfig,
    /// Configuration for query expansion strategies
    pub query_expansion: LazyQueryExpansionConfig,
    /// Configuration for relevance scoring of results
    pub relevance_scoring: LazyRelevanceScoringConfig,
}

/// Configuration for extracting concepts from text documents.
/// This configuration controls how key concepts are identified and extracted from text.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConceptExtractionConfig {
    /// Minimum length of a concept in characters
    pub min_concept_length: usize,
    /// Maximum number of words in a multi-word concept
    pub max_concept_words: usize,
    /// Whether to extract noun phrases as concepts
    pub use_noun_phrases: bool,
    /// Whether to consider capitalized words as potential concepts
    pub use_capitalization: bool,
    /// Whether to consider title-cased phrases as potential concepts
    pub use_title_case: bool,
    /// Whether to use TF-IDF scoring for concept importance
    pub use_tf_idf_scoring: bool,
    /// Minimum term frequency for a term to be considered a concept
    pub min_term_frequency: usize,
    /// Maximum number of concepts to extract per document chunk
    pub max_concepts_per_chunk: usize,
    /// Minimum score threshold for a term to be considered a concept
    pub min_concept_score: f32,
    /// Whether to exclude common stopwords from concept extraction
    pub exclude_stopwords: bool,
    /// Custom list of stopwords to exclude from concept extraction
    pub custom_stopwords: Vec<String>,
}

/// Configuration for co-occurrence analysis of concepts in documents.
/// This determines how relationships between concepts are identified based on their co-occurrence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoOccurrenceConfig {
    /// Size of the sliding window (in words) to consider for co-occurrence
    pub window_size: usize,
    /// Minimum number of co-occurrences required to create an edge between concepts
    pub min_co_occurrence: usize,
    /// Jaccard similarity threshold for merging similar concepts
    pub jaccard_threshold: f32,
    /// Maximum number of edges allowed per node in the co-occurrence graph
    pub max_edges_per_node: usize,
}

/// Configuration for lazy indexing of graph components.
/// Controls how graph components are indexed for efficient retrieval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LazyIndexingConfig {
    /// Whether to use bidirectional indexing for faster lookups
    pub use_bidirectional_index: bool,
    /// Whether to enable HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search
    pub enable_hnsw_index: bool,
    /// Maximum number of items to keep in the index cache
    pub cache_size: usize,
}

/// Configuration for lazy query expansion in the retrieval process.
/// Controls how queries are expanded to improve search results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LazyQueryExpansionConfig {
    /// Whether query expansion is enabled
    pub enabled: bool,
    /// Maximum number of query expansions to generate
    pub max_expansions: usize,
    /// Name of the model to use for query expansion
    pub expansion_model: String,
    /// Temperature parameter for controlling randomness in expansion generation
    pub expansion_temperature: f32,
    /// Maximum number of tokens to generate per expansion
    pub max_tokens_per_expansion: usize,
}

/// Configuration for lazy relevance scoring of search results.
/// Controls how search results are scored for relevance to the query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LazyRelevanceScoringConfig {
    /// Whether relevance scoring is enabled
    pub enabled: bool,
    /// Name of the model to use for relevance scoring
    pub scoring_model: String,
    /// Number of items to score in a single batch
    pub batch_size: usize,
    /// Temperature parameter for controlling randomness in scoring
    pub temperature: f32,
    /// Maximum number of tokens to consider for each score calculation
    pub max_tokens_per_score: usize,
}

/// End-to-End GraphRAG configuration for comprehensive knowledge graph construction.
/// This configuration enables fine-grained control over the entire pipeline from text to knowledge graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct E2GraphRAGConfig {
    /// Whether the E2E GraphRAG pipeline is enabled
    pub enabled: bool,

    /// Configuration for Named Entity Recognition (NER) extraction
    pub ner_extraction: NERExtractionConfig,

    /// Configuration for keyword extraction from text
    pub keyword_extraction: KeywordExtractionConfig,

    /// Configuration for graph construction parameters
    pub graph_construction: E2GraphConstructionConfig,

    /// Configuration for indexing strategies
    pub indexing: E2IndexingConfig,
}

/// Configuration for Named Entity Recognition (NER) extraction from text.
/// Controls how named entities are identified and extracted from documents.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NERExtractionConfig {
    /// List of entity types to recognize (e.g., ["PERSON", "ORG", "LOCATION"])
    pub entity_types: Vec<String>,

    /// Whether to recognize capitalized words as potential named entities
    pub use_capitalized_patterns: bool,

    /// Whether to recognize title-cased phrases as potential named entities
    pub use_title_case_patterns: bool,

    /// Whether to recognize quoted phrases as potential named entities
    pub use_quoted_patterns: bool,

    /// Whether to recognize common abbreviations as entities
    pub use_abbreviations: bool,

    /// Whether to use contextual disambiguation to resolve entity ambiguity
    pub use_contextual_disambiguation: bool,

    /// Minimum number of context words to consider for disambiguation
    pub min_context_words: usize,

    /// Minimum confidence score (0.0-1.0) required for an entity to be included
    pub min_confidence: f32,

    /// Whether to apply positional boost to entities based on their position in the text
    pub use_positional_boost: bool,

    /// Whether to apply frequency boost to entities based on their frequency in the text
    pub use_frequency_boost: bool,
}

impl Default for NERExtractionConfig {
    fn default() -> Self {
        Self {
            entity_types: vec![
                "PERSON".to_string(),
                "ORG".to_string(),
                "LOCATION".to_string(),
            ],
            use_capitalized_patterns: true,
            use_title_case_patterns: true,
            use_quoted_patterns: true,
            use_abbreviations: true,
            use_contextual_disambiguation: true,
            min_context_words: 5,
            min_confidence: 0.7,
            use_positional_boost: true,
            use_frequency_boost: true,
        }
    }
}

/// Configuration for keyword extraction from text documents.
/// Controls how keywords are identified and extracted from text content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KeywordExtractionConfig {
    /// List of algorithms to use for keyword extraction (e.g., ["tfidf", "yake", "textrank"])
    pub algorithms: Vec<String>,

    /// Maximum number of keywords to extract per document chunk
    pub max_keywords_per_chunk: usize,

    /// Minimum length of a keyword in characters
    pub min_keyword_length: usize,

    /// Whether to combine results from multiple algorithms
    pub combine_algorithms: bool,
}

impl Default for KeywordExtractionConfig {
    fn default() -> Self {
        Self {
            algorithms: vec!["tfidf".to_string(), "yake".to_string()],
            max_keywords_per_chunk: 10,
            min_keyword_length: 3,
            combine_algorithms: true,
        }
    }
}

/// Configuration for graph construction in the E2E GraphRAG pipeline.
/// Controls how entities and their relationships are organized into a knowledge graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct E2GraphConstructionConfig {
    /// Types of relationships to extract between entities (e.g., ["CO_OCCURS_WITH", "RELATED_TO"])
    pub relationship_types: Vec<String>,

    /// Minimum score required to establish a relationship between entities (0.0-1.0)
    pub min_relationship_score: f32,

    /// Maximum number of relationships to maintain per entity
    pub max_relationships_per_entity: usize,

    /// Whether to use mutual information for relationship scoring
    pub use_mutual_information: bool,
}

impl Default for E2GraphConstructionConfig {
    fn default() -> Self {
        Self {
            relationship_types: vec!["CO_OCCURS_WITH".to_string(), "RELATED_TO".to_string()],
            min_relationship_score: 0.5,
            max_relationships_per_entity: 20,
            use_mutual_information: true,
        }
    }
}

/// Configuration for indexing in the E2E GraphRAG pipeline.
/// Controls how entities, relationships, and their embeddings are indexed for efficient retrieval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct E2IndexingConfig {
    /// Number of items to process in a single batch during indexing
    pub batch_size: usize,

    /// Whether to enable parallel processing during indexing
    pub enable_parallel_processing: bool,

    /// Whether to cache concept vectors for faster retrieval
    pub cache_concept_vectors: bool,

    /// Whether to use hash embeddings for more efficient storage
    pub use_hash_embeddings: bool,
}

impl Default for E2IndexingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            enable_parallel_processing: true,
            cache_concept_vectors: true,
            use_hash_embeddings: false,
        }
    }
}

/// Configuration for pure algorithmic GraphRAG approach without LLM dependencies.
///
/// This configuration enables cost-effective graph construction and analysis
/// using only algorithmic methods for pattern extraction, keyword analysis,
/// and relationship discovery.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PureAlgorithmicConfig {
    /// Whether the pure algorithmic approach is enabled
    pub enabled: bool,
    /// Configuration for extracting linguistic patterns from text
    pub pattern_extraction: PatternExtractionConfig,
    /// Configuration for keyword extraction using statistical methods
    pub keyword_extraction: PureKeywordExtractionConfig,
    /// Configuration for discovering relationships between entities
    pub relationship_discovery: RelationshipDiscoveryConfig,
    /// Configuration for search result ranking algorithms
    pub search_ranking: SearchRankingConfig,
}

/// Configuration for pattern extraction from text using regex and linguistic rules.
///
/// Pattern extraction identifies consistent linguistic structures that can indicate
/// entities, relationships, and semantic patterns without requiring LLM processing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternExtractionConfig {
    /// Regex patterns for identifying capitalized entities (proper nouns, acronyms)
    pub capitalized_patterns: Vec<String>,
    /// Regex patterns for technical terms, jargon, and specialized language
    pub technical_patterns: Vec<String>,
    /// Regex patterns for contextual relationships and semantic structures
    pub context_patterns: Vec<String>,
}

/// Configuration for keyword extraction using statistical algorithms.
///
/// This configuration enables extraction of important terms from text using
/// algorithms like TF-IDF, RAKE, or YAKE without requiring LLM processing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PureKeywordExtractionConfig {
    /// Algorithm to use for keyword extraction (e.g., "tfidf", "rake", "yake")
    pub algorithm: String,
    /// Maximum number of keywords to extract per document
    pub max_keywords: usize,
    /// Minimum word length to consider for keywords
    pub min_word_length: usize,
    /// Whether to boost keywords based on their position in text
    pub use_positional_boost: bool,
    /// Whether to filter keywords based on frequency thresholds
    pub use_frequency_filter: bool,
    /// Minimum term frequency for a word to be considered a keyword
    pub min_term_frequency: usize,
    /// Maximum term frequency ratio to filter out overly common terms
    pub max_term_frequency_ratio: f32,
}

/// Configuration for discovering relationships between entities using co-occurrence analysis.
///
/// This configuration enables algorithmic relationship discovery by analyzing
/// word co-occurrence patterns and statistical measures without LLM inference.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RelationshipDiscoveryConfig {
    /// Window size for co-occurrence analysis (number of words to check around entities)
    pub window_size: usize,
    /// Minimum co-occurrence count to establish a relationship
    pub min_co_occurrence: usize,
    /// Whether to use mutual information scoring for relationship strength
    pub use_mutual_information: bool,
    /// Types of relationships to identify (e.g., "causal", "hierarchical", "temporal")
    pub relationship_types: Vec<String>,
    /// Scoring method for relationship ranking (e.g., "frequency", "mi", "pmi")
    pub scoring_method: String,
    /// Minimum similarity score threshold for valid relationships
    pub min_similarity_score: f32,
}

/// Configuration for search result ranking across multiple retrieval strategies.
///
/// This configuration enables combining different search approaches (vector, keyword,
/// graph traversal) and fusing their results for optimal relevance ranking.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchRankingConfig {
    /// Configuration for vector-based similarity search
    pub vector_search: VectorSearchConfig,
    /// Configuration for keyword-based search algorithms (e.g., BM25)
    pub keyword_search: KeywordSearchConfig,
    /// Configuration for graph-based traversal and ranking
    pub graph_traversal: GraphTraversalConfig,
    /// Configuration for hybrid fusion of multiple search strategies
    pub hybrid_fusion: HybridFusionConfig,
}

/// Configuration for vector-based similarity search.
///
/// Enables semantic search using embeddings and similarity scoring
/// for finding conceptually related content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct VectorSearchConfig {
    /// Whether vector similarity search is enabled
    pub enabled: bool,
}

/// Configuration for keyword-based search algorithms.
///
/// Enables traditional information retrieval algorithms like BM25
/// for keyword matching and scoring.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KeywordSearchConfig {
    /// Whether keyword-based search is enabled
    pub enabled: bool,
    /// Search algorithm to use (e.g., "bm25", "tfidf", "dirichlet")
    pub algorithm: String,
    /// BM25 parameter k1: controls term frequency saturation (typically 1.2-2.0)
    pub k1: f32,
    /// BM25 parameter b: controls document length normalization (typically 0.0-1.0)
    pub b: f32,
}

/// Configuration for graph-based traversal and ranking algorithms.
///
/// Enables graph algorithms like PageRank and personalized search
/// for navigating and ranking content in the knowledge graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphTraversalConfig {
    /// Whether graph traversal algorithms are enabled
    pub enabled: bool,
    /// Algorithm to use for graph traversal (e.g., "pagerank", "hits", "random_walk")
    pub algorithm: String,
    /// Damping factor for PageRank algorithm (typically 0.85)
    pub damping_factor: f32,
    /// Maximum iterations for graph algorithms
    pub max_iterations: usize,
    /// Whether to use personalized graph traversal
    pub personalized: bool,
}

/// Configuration for hybrid fusion of multiple search strategies.
///
/// Enables combining results from different search approaches (vector, keyword,
/// graph) using weighted scoring for improved relevance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HybridFusionConfig {
    /// Whether hybrid fusion of search results is enabled
    pub enabled: bool,
    /// Weight configuration for different search strategies
    pub weights: FusionWeights,
}

/// Weight configuration for combining different search strategies.
///
/// Defines the relative importance of each search approach in the
/// hybrid fusion algorithm. Weights should typically sum to 1.0.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionWeights {
    /// Weight for keyword-based search results
    pub keywords: f32,
    /// Weight for graph traversal-based search results
    pub graph: f32,
    /// Weight for BM25/TF-IDF statistical search results
    pub bm25: f32,
}

/// Configuration for hybrid GraphRAG strategies combining algorithmic and LLM approaches.
///
/// This configuration enables different hybrid strategies for balancing cost,
/// performance, and quality through intelligent LLM usage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HybridStrategyConfig {
    /// Configuration for lazy algorithmic approach with selective LLM enhancement
    pub lazy_algorithmic: LazyAlgorithmicConfig,
    /// Configuration for progressive multi-level LLM usage
    pub progressive: ProgressiveConfig,
    /// Configuration for budget-aware LLM optimization
    pub budget_aware: BudgetAwareConfig,
}

/// Configuration for lazy algorithmic approach with selective LLM enhancement.
///
/// This strategy primarily uses algorithmic methods and only invokes LLMs
/// when necessary to improve quality or handle complex cases.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LazyAlgorithmicConfig {
    /// Indexing strategy (e.g., "algorithmic_first", "llm_assisted", "hybrid")
    pub indexing_approach: String,
    /// Query processing strategy (e.g., "algorithmic_only", "selective_llm", "adaptive")
    pub query_approach: String,
    /// Cost optimization strategy (e.g., "aggressive", "balanced", "quality_first")
    pub cost_optimization: String,
}

/// Configuration for progressive multi-level LLM usage strategy.
///
/// This strategy uses different levels of LLM involvement based on
/// query complexity, budget, and quality requirements.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressiveConfig {
    /// Level 0: Pure algorithmic processing (no LLM usage)
    pub level_0: String,
    /// Level 1: Minimal LLM usage (entity extraction only)
    pub level_1: String,
    /// Level 2: Moderate LLM usage (entity + relationship extraction)
    pub level_2: String,
    /// Level 3: Heavy LLM usage (full semantic analysis)
    pub level_3: String,
    /// Level 4+: Maximum LLM usage (comprehensive processing)
    pub level_4_plus: String,
}

/// Configuration for budget-aware LLM optimization strategy.
///
/// This strategy dynamically adjusts LLM usage based on budget constraints,
/// query costs, and daily spending limits to ensure cost control.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BudgetAwareConfig {
    /// Daily budget limit in USD for LLM operations
    pub daily_budget_usd: f64,
    /// Maximum number of queries allowed per day
    pub queries_per_day: usize,
    /// Maximum LLM cost allowed per individual query
    pub max_llm_cost_per_query: f64,
    /// Budget management strategy (e.g., "throttle", "degrade", "stop")
    pub strategy: String,
    /// Whether to fall back to pure algorithmic processing when budget is exceeded
    pub fallback_to_algorithmic: bool,
}

// Default functions for zero-cost approach
fn default_zero_cost_approach() -> String {
    "pure_algorithmic".to_string()
}

impl Default for ZeroCostApproachConfig {
    fn default() -> Self {
        Self {
            approach: default_zero_cost_approach(),
            lazy_graphrag: LazyGraphRAGConfig::default(),
            e2_graphrag: E2GraphRAGConfig::default(),
            pure_algorithmic: PureAlgorithmicConfig::default(),
            hybrid_strategy: HybridStrategyConfig::default(),
        }
    }
}

// Default implementations for sub-configs (simplified for now)
impl Default for ConceptExtractionConfig {
    fn default() -> Self {
        Self {
            min_concept_length: 3,
            max_concept_words: 5,
            use_noun_phrases: true,
            use_capitalization: true,
            use_title_case: true,
            use_tf_idf_scoring: true,
            min_term_frequency: 2,
            max_concepts_per_chunk: 10,
            min_concept_score: 0.1,
            exclude_stopwords: true,
            custom_stopwords: vec!["the".to_string(), "and".to_string(), "or".to_string()],
        }
    }
}
impl Default for CoOccurrenceConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            min_co_occurrence: 2,
            jaccard_threshold: 0.2,
            max_edges_per_node: 25,
        }
    }
}
impl Default for LazyIndexingConfig {
    fn default() -> Self {
        Self {
            use_bidirectional_index: true,
            enable_hnsw_index: false,
            cache_size: 10000,
        }
    }
}
impl Default for LazyQueryExpansionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_expansions: 3,
            expansion_model: "llama3.1:8b".to_string(),
            expansion_temperature: 0.1,
            max_tokens_per_expansion: 50,
        }
    }
}
impl Default for LazyRelevanceScoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scoring_model: "llama3.1:8b".to_string(),
            batch_size: 10,
            temperature: 0.2,
            max_tokens_per_score: 30,
        }
    }
}
impl Default for PureAlgorithmicConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pattern_extraction: Default::default(),
            keyword_extraction: Default::default(),
            relationship_discovery: Default::default(),
            search_ranking: Default::default(),
        }
    }
}
impl Default for PatternExtractionConfig {
    fn default() -> Self {
        Self {
            capitalized_patterns: vec![r"[A-Z][a-z]+".to_string()],
            technical_patterns: vec![r"[a-z]+-[a-z]+".to_string()],
            context_patterns: vec![r"\b(the|this)\s+(\w+)".to_string()],
        }
    }
}
impl Default for PureKeywordExtractionConfig {
    fn default() -> Self {
        Self {
            algorithm: "tf_idf".to_string(),
            max_keywords: 20,
            min_word_length: 4,
            use_positional_boost: true,
            use_frequency_filter: true,
            min_term_frequency: 2,
            max_term_frequency_ratio: 0.8,
        }
    }
}
impl Default for RelationshipDiscoveryConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            min_co_occurrence: 2,
            use_mutual_information: true,
            relationship_types: vec!["co_occurs_with".to_string()],
            scoring_method: "jaccard_similarity".to_string(),
            min_similarity_score: 0.1,
        }
    }
}
impl Default for SearchRankingConfig {
    fn default() -> Self {
        Self {
            vector_search: VectorSearchConfig { enabled: false },
            keyword_search: KeywordSearchConfig {
                enabled: true,
                algorithm: "bm25".to_string(),
                k1: 1.2,
                b: 0.75,
            },
            graph_traversal: GraphTraversalConfig {
                enabled: true,
                algorithm: "pagerank".to_string(),
                damping_factor: 0.85,
                max_iterations: 20,
                personalized: true,
            },
            hybrid_fusion: HybridFusionConfig {
                enabled: true,
                weights: FusionWeights {
                    keywords: 0.4,
                    graph: 0.4,
                    bm25: 0.2,
                },
            },
        }
    }
}
impl Default for HybridStrategyConfig {
    fn default() -> Self {
        Self {
            lazy_algorithmic: LazyAlgorithmicConfig {
                indexing_approach: "e2_graphrag".to_string(),
                query_approach: "lazy_graphrag".to_string(),
                cost_optimization: "indexing".to_string(),
            },
            progressive: ProgressiveConfig {
                level_0: "pure_algorithmic".to_string(),
                level_1: "pure_algorithmic".to_string(),
                level_2: "e2_graphrag".to_string(),
                level_3: "lazy_graphrag".to_string(),
                level_4_plus: "lazy_graphrag".to_string(),
            },
            budget_aware: BudgetAwareConfig {
                daily_budget_usd: 1.0,
                queries_per_day: 1000,
                max_llm_cost_per_query: 0.002,
                strategy: "lazy_graphrag".to_string(),
                fallback_to_algorithmic: true,
            },
        }
    }
}
impl Default for KeywordSearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "bm25".to_string(),
            k1: 1.2,
            b: 0.75,
        }
    }
}
impl Default for GraphTraversalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "pagerank".to_string(),
            damping_factor: 0.85,
            max_iterations: 20,
            personalized: true,
        }
    }
}
impl Default for HybridFusionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            weights: FusionWeights {
                keywords: 0.4,
                graph: 0.4,
                bm25: 0.2,
            },
        }
    }
}
impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            keywords: 0.4,
            graph: 0.4,
            bm25: 0.2,
        }
    }
}
impl Default for LazyAlgorithmicConfig {
    fn default() -> Self {
        Self {
            indexing_approach: "e2_graphrag".to_string(),
            query_approach: "lazy_graphrag".to_string(),
            cost_optimization: "indexing".to_string(),
        }
    }
}
impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            level_0: "pure_algorithmic".to_string(),
            level_1: "pure_algorithmic".to_string(),
            level_2: "e2_graphrag".to_string(),
            level_3: "lazy_graphrag".to_string(),
            level_4_plus: "lazy_graphrag".to_string(),
        }
    }
}
impl Default for BudgetAwareConfig {
    fn default() -> Self {
        Self {
            daily_budget_usd: 1.0,
            queries_per_day: 1000,
            max_llm_cost_per_query: 0.002,
            strategy: "lazy_graphrag".to_string(),
            fallback_to_algorithmic: true,
        }
    }
}

/// Configuration for embedding generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingConfig {
    /// Dimension of the embedding vectors
    pub dimension: usize,

    /// Embedding backend: "hash", "ollama", "huggingface", "openai", "voyage", "cohere", "jina", "mistral", "together", "onnx", "candle"
    pub backend: String,

    /// Model identifier (provider-specific)
    /// - HuggingFace: "sentence-transformers/all-MiniLM-L6-v2"
    /// - OpenAI: "text-embedding-3-small"
    /// - Voyage: "voyage-3-large"
    /// - Cohere: "embed-english-v3.0"
    /// - Jina: "jina-embeddings-v3"
    /// - Mistral: "mistral-embed"
    /// - Together: "BAAI/bge-large-en-v1.5"
    /// - Ollama: "nomic-embed-text"
    #[serde(default)]
    pub model: Option<String>,

    /// Whether to fallback to hash-based embeddings if primary backend fails
    pub fallback_to_hash: bool,

    /// API endpoint for embeddings (if using external service)
    pub api_endpoint: Option<String>,

    /// API key for external embedding service
    /// Can also be set via environment variables (OPENAI_API_KEY, VOYAGE_API_KEY, etc.)
    pub api_key: Option<String>,

    /// Cache directory for downloaded models (HuggingFace)
    #[serde(default)]
    pub cache_dir: Option<String>,

    /// Batch size for processing multiple texts
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_batch_size() -> usize {
    32
}

/// Configuration for graph construction
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphConfig {
    /// Maximum number of connections per node
    pub max_connections: usize,

    /// Similarity threshold for creating edges
    pub similarity_threshold: f32,

    /// Whether to extract relationships between entities
    #[serde(default = "default_true")]
    pub extract_relationships: bool,

    /// Confidence threshold for relationships
    #[serde(default = "default_relationship_confidence")]
    pub relationship_confidence_threshold: f32,

    /// Graph traversal configuration
    #[serde(default)]
    pub traversal: TraversalConfigParams,
}

/// Configuration for graph traversal algorithms
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraversalConfigParams {
    /// Maximum depth for traversal algorithms (BFS, DFS)
    #[serde(default = "default_max_traversal_depth")]
    pub max_depth: usize,

    /// Maximum number of paths to find (for pathfinding algorithms)
    #[serde(default = "default_max_paths")]
    pub max_paths: usize,

    /// Whether to use edge weights in traversal
    #[serde(default = "default_true")]
    pub use_edge_weights: bool,

    /// Minimum relationship strength to consider in traversal
    #[serde(default = "default_min_relationship_strength")]
    pub min_relationship_strength: f32,
}

impl Default for TraversalConfigParams {
    fn default() -> Self {
        Self {
            max_depth: default_max_traversal_depth(),
            max_paths: default_max_paths(),
            use_edge_weights: true,
            min_relationship_strength: default_min_relationship_strength(),
        }
    }
}

/// Configuration for text processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextConfig {
    /// Maximum chunk size for text processing
    pub chunk_size: usize,

    /// Overlap between chunks
    pub chunk_overlap: usize,

    /// Languages to support for text processing
    pub languages: Vec<String>,
}

/// Configuration for entity extraction
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityConfig {
    /// Minimum confidence score for entity extraction
    pub min_confidence: f32,

    /// Types of entities to extract
    pub entity_types: Vec<String>,

    /// Whether to use LLM-based gleaning for entity extraction
    #[serde(default)]
    pub use_gleaning: bool,

    /// Maximum number of gleaning rounds for refinement
    #[serde(default = "default_max_gleaning_rounds")]
    pub max_gleaning_rounds: usize,

    /// Enable triple reflection validation (DEG-RAG methodology)
    /// Validates extracted relationships against source text using LLM
    #[serde(default)]
    pub enable_triple_reflection: bool,

    /// Minimum confidence score for relationship validation
    /// Relationships below this threshold will be filtered out
    #[serde(default = "default_validation_confidence")]
    pub validation_min_confidence: f32,

    /// Enable ATOM atomic fact extraction (Phase 1.3)
    /// Extracts self-contained facts as 5-tuples for better granularity
    #[serde(default)]
    pub use_atomic_facts: bool,

    /// Maximum tokens per atomic fact
    /// Facts longer than this will be rejected
    #[serde(default = "default_max_fact_tokens")]
    pub max_fact_tokens: usize,
}

/// Configuration for advanced GraphRAG features (Phases 2-3)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[derive(Default)]
pub struct AdvancedFeaturesConfig {
    /// Phase 2.1: Symbolic Anchoring (CatRAG)
    /// Automatically applied for conceptual queries - no config needed
    #[serde(default)]
    pub symbolic_anchoring: SymbolicAnchoringConfig,

    /// Phase 2.2: Dynamic Edge Weighting
    /// Query-aware relationship weight adjustment
    #[serde(default)]
    pub dynamic_weighting: DynamicWeightingConfig,

    /// Phase 2.3: Causal Chain Analysis
    /// Multi-step causal reasoning
    #[serde(default)]
    pub causal_analysis: CausalAnalysisConfig,

    /// Phase 3.1: Hierarchical Relationship Clustering
    /// Multi-level relationship organization
    #[serde(default)]
    pub hierarchical_clustering: HierarchicalClusteringConfig,

    /// Phase 3.2: Graph Weight Optimization (DW-GRPO)
    /// Heuristic optimization of relationship weights
    #[serde(default)]
    pub weight_optimization: WeightOptimizationConfig,
}

/// Configuration for Symbolic Anchoring (Phase 2.1)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SymbolicAnchoringConfig {
    /// Minimum relevance score to keep an anchor (0.0-1.0)
    #[serde(default = "default_anchor_min_relevance")]
    pub min_relevance: f32,

    /// Maximum number of anchors to extract per query
    #[serde(default = "default_max_anchors")]
    pub max_anchors: usize,

    /// Maximum entities per anchor
    #[serde(default = "default_max_entities_per_anchor")]
    pub max_entities_per_anchor: usize,
}

/// Configuration for Dynamic Edge Weighting (Phase 2.2)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DynamicWeightingConfig {
    /// Enable semantic boost using embeddings
    #[serde(default = "default_true")]
    pub enable_semantic_boost: bool,

    /// Enable temporal boost for recent relationships
    #[serde(default = "default_true")]
    pub enable_temporal_boost: bool,

    /// Enable conceptual boost for matching concepts
    #[serde(default = "default_true")]
    pub enable_concept_boost: bool,

    /// Enable causal boost for strong causal relationships
    #[serde(default = "default_true")]
    pub enable_causal_boost: bool,
}

/// Configuration for Causal Chain Analysis (Phase 2.3)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalAnalysisConfig {
    /// Minimum confidence for causal chains (0.0-1.0)
    #[serde(default = "default_causal_min_confidence")]
    pub min_confidence: f32,

    /// Minimum causal strength to consider (0.0-1.0)
    #[serde(default = "default_causal_min_strength")]
    pub min_causal_strength: f32,

    /// Maximum chain depth to search
    #[serde(default = "default_max_chain_depth")]
    pub max_chain_depth: usize,

    /// Require temporal consistency in chains
    #[serde(default = "default_true")]
    pub require_temporal_consistency: bool,
}

/// Configuration for Hierarchical Relationship Clustering (Phase 3.1)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HierarchicalClusteringConfig {
    /// Number of hierarchy levels (2-5)
    #[serde(default = "default_num_levels")]
    pub num_levels: usize,

    /// Resolution parameters for each level (higher = more clusters)
    /// Length should match num_levels
    #[serde(default = "default_resolutions")]
    pub resolutions: Vec<f32>,

    /// Minimum relationships per cluster
    #[serde(default = "default_min_cluster_size")]
    pub min_cluster_size: usize,

    /// Generate LLM summaries for clusters (requires Ollama)
    #[serde(default = "default_true")]
    pub generate_summaries: bool,
}

/// Configuration for Graph Weight Optimization (Phase 3.2)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WeightOptimizationConfig {
    /// Learning rate for weight adjustments (0.01-0.5)
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Maximum optimization iterations
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Window size for slope calculation
    #[serde(default = "default_slope_window")]
    pub slope_window: usize,

    /// Minimum slope to avoid stagnation
    #[serde(default = "default_stagnation_threshold")]
    pub stagnation_threshold: f32,

    /// Use LLM for quality evaluation
    #[serde(default = "default_true")]
    pub use_llm_eval: bool,

    /// Objective weights (relevance, faithfulness, conciseness)
    #[serde(default)]
    pub objective_weights: ObjectiveWeightsConfig,
}

/// Configuration for optimization objective weights
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObjectiveWeightsConfig {
    /// Weight for relevance objective (0.0-1.0)
    #[serde(default = "default_relevance_weight")]
    pub relevance: f32,

    /// Weight for faithfulness objective (0.0-1.0)
    #[serde(default = "default_faithfulness_weight")]
    pub faithfulness: f32,

    /// Weight for conciseness objective (0.0-1.0)
    #[serde(default = "default_conciseness_weight")]
    pub conciseness: f32,
}

/// Configuration for retrieval operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetrievalConfig {
    /// Number of top results to return
    pub top_k: usize,

    /// Search algorithm to use
    pub search_algorithm: String,
}

/// Configuration for parallel processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParallelConfig {
    /// Number of threads to use for parallel processing (0 = auto-detect)
    pub num_threads: usize,

    /// Enable parallel processing
    pub enabled: bool,

    /// Minimum batch size for parallel processing
    pub min_batch_size: usize,

    /// Chunk size for parallel text processing
    pub chunk_batch_size: usize,

    /// Parallel processing for embeddings
    pub parallel_embeddings: bool,

    /// Parallel graph construction
    pub parallel_graph_ops: bool,

    /// Parallel vector operations
    pub parallel_vector_ops: bool,
}

// Default value functions
fn default_embedding_dim() -> usize {
    384
}
fn default_embedding_backend() -> String {
    "hash".to_string()
}
fn default_max_connections() -> usize {
    10
}
fn default_similarity_threshold() -> f32 {
    0.8
}
fn default_chunk_size() -> usize {
    1000
}
fn default_chunk_overlap() -> usize {
    200
}
fn default_languages() -> Vec<String> {
    vec!["en".to_string()]
}
fn default_min_confidence() -> f32 {
    0.7
}
fn default_entity_types() -> Vec<String> {
    vec![
        "PERSON".to_string(),
        "ORG".to_string(),
        "LOCATION".to_string(),
    ]
}
fn default_top_k() -> usize {
    10
}
fn default_search_algorithm() -> String {
    "cosine".to_string()
}
fn default_num_threads() -> usize {
    0
} // Auto-detect
fn default_min_batch_size() -> usize {
    10
}
fn default_chunk_batch_size() -> usize {
    100
}
fn default_true() -> bool {
    true
}
fn default_relationship_confidence() -> f32 {
    0.5
}
fn default_max_gleaning_rounds() -> usize {
    3
}

fn default_validation_confidence() -> f32 {
    0.7
}

// Advanced features defaults (Phases 2-3)

// Phase 2.1: Symbolic Anchoring
fn default_anchor_min_relevance() -> f32 {
    0.3
}

fn default_max_anchors() -> usize {
    5
}

fn default_max_entities_per_anchor() -> usize {
    10
}

// Phase 2.3: Causal Analysis
fn default_causal_min_confidence() -> f32 {
    0.3
}

fn default_causal_min_strength() -> f32 {
    0.5
}

fn default_max_chain_depth() -> usize {
    5
}

// Phase 3.1: Hierarchical Clustering
fn default_num_levels() -> usize {
    3
}

fn default_resolutions() -> Vec<f32> {
    vec![1.0, 0.5, 0.2]
}

fn default_min_cluster_size() -> usize {
    2
}

// Phase 3.2: Weight Optimization
fn default_learning_rate() -> f32 {
    0.1
}

fn default_max_iterations() -> usize {
    20
}

fn default_slope_window() -> usize {
    3
}

fn default_stagnation_threshold() -> f32 {
    0.01
}

fn default_relevance_weight() -> f32 {
    0.4
}

fn default_faithfulness_weight() -> f32 {
    0.4
}

fn default_conciseness_weight() -> f32 {
    0.2
}

fn default_max_fact_tokens() -> usize {
    400
}

fn default_approach() -> String {
    "semantic".to_string()
}
fn default_max_traversal_depth() -> usize {
    3
}
fn default_max_paths() -> usize {
    10
}
fn default_min_relationship_strength() -> f32 {
    0.3
}
fn default_auto_save_interval() -> u64 {
    300 // 5 minutes
}
fn default_max_versions() -> usize {
    5 // Keep 5 versions by default
}

impl Default for Config {
    fn default() -> Self {
        Self {
            output_dir: "./output".to_string(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            max_entities_per_chunk: Some(10),
            top_k_results: Some(default_top_k()),
            similarity_threshold: Some(default_similarity_threshold()),
            approach: default_approach(),
            embeddings: EmbeddingConfig {
                dimension: default_embedding_dim(),
                backend: default_embedding_backend(),
                model: Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
                fallback_to_hash: true,
                api_endpoint: None,
                api_key: None,
                cache_dir: None,
                batch_size: default_batch_size(),
            },
            graph: GraphConfig {
                max_connections: default_max_connections(),
                similarity_threshold: default_similarity_threshold(),
                extract_relationships: default_true(),
                relationship_confidence_threshold: default_relationship_confidence(),
                traversal: TraversalConfigParams::default(),
            },
            text: TextConfig {
                chunk_size: default_chunk_size(),
                chunk_overlap: default_chunk_overlap(),
                languages: default_languages(),
            },
            entities: EntityConfig {
                min_confidence: default_min_confidence(),
                entity_types: default_entity_types(),
                use_gleaning: false,
                max_gleaning_rounds: default_max_gleaning_rounds(),
                enable_triple_reflection: false,
                validation_min_confidence: default_validation_confidence(),
                use_atomic_facts: false,
                max_fact_tokens: default_max_fact_tokens(),
            },
            retrieval: RetrievalConfig {
                top_k: default_top_k(),
                search_algorithm: default_search_algorithm(),
            },
            parallel: ParallelConfig {
                num_threads: default_num_threads(),
                enabled: true,
                min_batch_size: default_min_batch_size(),
                chunk_batch_size: default_chunk_batch_size(),
                parallel_embeddings: true,
                parallel_graph_ops: true,
                parallel_vector_ops: true,
            },
            ollama: crate::ollama::OllamaConfig::default(),
            gliner: GlinerConfig::default(),
            enhancements: enhancements::EnhancementsConfig::default(),
            auto_save: AutoSaveConfig {
                enabled: false,
                base_dir: None,
                interval_seconds: default_auto_save_interval(),
                workspace_name: None,
                max_versions: default_max_versions(),
            },
            summarization: crate::summarization::HierarchicalConfig::default(),
            zero_cost_approach: ZeroCostApproachConfig::default(),
            advanced_features: AdvancedFeaturesConfig::default(),
            suppress_progress_bars: false,
        }
    }
}

impl Default for AutoSaveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_dir: None,
            interval_seconds: default_auto_save_interval(),
            workspace_name: None,
            max_versions: default_max_versions(),
        }
    }
}

impl Config {
    /// Turnkey config for a workspace directory. Persists graph + index to
    /// `workspace`. Uses hash-fallback embeddings and pattern-based entity
    /// extraction by default — works offline, no Ollama required. Chain
    /// `.with_ollama(...)` etc. to enable LLM features.
    pub fn quick(workspace: impl AsRef<std::path::Path>) -> Self {
        let ws = workspace.as_ref();
        let ws_str = ws.to_string_lossy().into_owned();
        let (base, name) = match (ws.parent(), ws.file_name()) {
            (Some(p), Some(f)) if !p.as_os_str().is_empty() => (
                p.to_string_lossy().into_owned(),
                f.to_string_lossy().into_owned(),
            ),
            _ => (".".to_string(), ws_str.clone()),
        };
        Self {
            output_dir: ws_str,
            auto_save: AutoSaveConfig {
                enabled: true,
                base_dir: Some(base),
                workspace_name: Some(name),
                ..AutoSaveConfig::default()
            },
            ..Self::default()
        }
    }

    /// Enable Ollama with sensible defaults (localhost:11434, llama3.2:3b).
    pub fn with_ollama(mut self) -> Self {
        self.ollama.enabled = true;
        self.embeddings.backend = "ollama".to_string();
        self
    }

    /// Override Ollama host (e.g. `http://gpu-box:11434`).
    pub fn with_ollama_host(mut self, host: impl Into<String>) -> Self {
        self.ollama.host = host.into();
        self.ollama.enabled = true;
        self
    }

    /// Override chunk size and overlap (overlap defaults to 20 % of size).
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self.chunk_overlap = size / 5;
        self.text.chunk_size = size;
        self.text.chunk_overlap = size / 5;
        self
    }
}


impl Default for SymbolicAnchoringConfig {
    fn default() -> Self {
        Self {
            min_relevance: default_anchor_min_relevance(),
            max_anchors: default_max_anchors(),
            max_entities_per_anchor: default_max_entities_per_anchor(),
        }
    }
}

impl Default for DynamicWeightingConfig {
    fn default() -> Self {
        Self {
            enable_semantic_boost: default_true(),
            enable_temporal_boost: default_true(),
            enable_concept_boost: default_true(),
            enable_causal_boost: default_true(),
        }
    }
}

impl Default for CausalAnalysisConfig {
    fn default() -> Self {
        Self {
            min_confidence: default_causal_min_confidence(),
            min_causal_strength: default_causal_min_strength(),
            max_chain_depth: default_max_chain_depth(),
            require_temporal_consistency: default_true(),
        }
    }
}

impl Default for HierarchicalClusteringConfig {
    fn default() -> Self {
        Self {
            num_levels: default_num_levels(),
            resolutions: default_resolutions(),
            min_cluster_size: default_min_cluster_size(),
            generate_summaries: default_true(),
        }
    }
}

impl Default for WeightOptimizationConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            max_iterations: default_max_iterations(),
            slope_window: default_slope_window(),
            stagnation_threshold: default_stagnation_threshold(),
            use_llm_eval: default_true(),
            objective_weights: ObjectiveWeightsConfig::default(),
        }
    }
}

impl Default for ObjectiveWeightsConfig {
    fn default() -> Self {
        Self {
            relevance: default_relevance_weight(),
            faithfulness: default_faithfulness_weight(),
            conciseness: default_conciseness_weight(),
        }
    }
}

impl Config {
    /// Load configuration with hierarchical merging (requires `hierarchical-config` feature)
    ///
    /// Configuration sources are merged in order of priority (lowest to highest):
    /// 1. Built-in defaults
    /// 2. User config: `~/.graphrag/config.toml`
    /// 3. Project config: `./graphrag.toml`
    /// 4. Environment variables: `GRAPHRAG_*` (e.g., `GRAPHRAG_OLLAMA_HOST`)
    ///
    /// # Example
    /// ```rust,no_run
    /// use graphrag_core::Config;
    ///
    /// // Auto-loads from all sources
    /// let config = Config::load()?;
    /// # Ok::<(), graphrag_core::GraphRAGError>(())
    /// ```
    #[cfg(feature = "hierarchical-config")]
    pub fn load() -> Result<Self> {
        use figment::{
            providers::{Env, Format, Serialized, Toml},
            Figment,
        };

        // Build the configuration chain
        let mut figment = Figment::new()
            // 1. Start with defaults
            .merge(Serialized::defaults(Config::default()));

        // 2. User-level config (~/.graphrag/config.toml)
        if let Some(home) = dirs::home_dir() {
            let user_config = home.join(".graphrag").join("config.toml");
            if user_config.exists() {
                figment = figment.merge(Toml::file(user_config));
            }
        }

        // 3. Project-level config (./graphrag.toml)
        let project_config = std::path::Path::new("graphrag.toml");
        if project_config.exists() {
            figment = figment.merge(Toml::file(project_config));
        }

        // 4. Environment variables (GRAPHRAG_*)
        // Maps GRAPHRAG_OLLAMA_HOST -> ollama.host
        figment = figment.merge(Env::prefixed("GRAPHRAG_").split("_"));

        figment
            .extract()
            .map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to load hierarchical configuration: {}", e),
            })
    }

    /// Load configuration with hierarchical merging (stub for when feature is disabled)
    ///
    /// When the `hierarchical-config` feature is not enabled, this falls back to `Config::default()`.
    #[cfg(not(feature = "hierarchical-config"))]
    pub fn load() -> Result<Self> {
        Ok(Config::default())
    }

    /// Load configuration from a TOML file with environment variable overrides
    ///
    /// This is the preferred method for loading configuration from a specific file
    /// while still allowing environment variable overrides.
    ///
    /// # Example
    /// ```rust,no_run
    /// use graphrag_core::Config;
    ///
    /// let config = Config::from_toml_file("./my-config.toml")?;
    /// # Ok::<(), graphrag_core::GraphRAGError>(())
    /// ```
    pub fn from_toml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let config: Config =
            toml::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to parse TOML config: {}", e),
            })?;
        Ok(config)
    }

}
