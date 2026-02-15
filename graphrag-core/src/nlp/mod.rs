//! Advanced NLP Module
//!
//! This module provides advanced natural language processing capabilities:
//! - Semantic chunking algorithms
//! - Custom NER training pipeline
//! - Syntax analysis
//!
//! ## Features
//!
//! ### Semantic Chunking
//! - Multiple chunking strategies (sentence, paragraph, topic, semantic, hybrid)
//! - Intelligent boundary detection
//! - Coherence scoring
//! - Configurable chunk sizes and overlap
//!
//! ### Custom NER
//! - Pattern-based entity extraction
//! - Dictionary/gazetteer matching
//! - Rule-based extraction with priorities
//! - Training dataset management
//! - Active learning support
//!
//! ### Syntax Analysis
//! - Part-of-speech tagging
//! - Dependency parsing
//! - Noun phrase extraction
pub mod semantic_chunking;
pub mod custom_ner;
pub mod syntax_analyzer;

// Re-export main types
pub use semantic_chunking::{
    ChunkingStrategy, ChunkingConfig, SemanticChunk,
    SemanticChunker, ChunkingStats,
};

pub use custom_ner::{
    EntityType, ExtractionRule, RuleType, CustomNER,
    ExtractedEntity, TrainingDataset, AnnotatedExample,
    DatasetStatistics,
};

pub use syntax_analyzer::{
    POSTag, DependencyRelation, Token, Dependency, NounPhrase,
    SyntaxAnalyzer, SyntaxAnalyzerConfig,
};
