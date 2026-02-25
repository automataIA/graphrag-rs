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
pub mod custom_ner;
pub mod semantic_chunking;
pub mod syntax_analyzer;

// Re-export main types
pub use semantic_chunking::{
    ChunkingConfig, ChunkingStats, ChunkingStrategy, SemanticChunk, SemanticChunker,
};

pub use custom_ner::{
    AnnotatedExample, CustomNER, DatasetStatistics, EntityType, ExtractedEntity, ExtractionRule,
    RuleType, TrainingDataset,
};

pub use syntax_analyzer::{
    Dependency, DependencyRelation, NounPhrase, POSTag, SyntaxAnalyzer, SyntaxAnalyzerConfig, Token,
};
