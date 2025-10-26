//! Configuration for latest enhancements and atomic component control

use serde::{Deserialize, Serialize};

/// Configuration for latest enhancements with atomic control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementsConfig {
    /// Master switch for all enhancements
    pub enabled: bool,
    /// Query analysis configuration
    pub query_analysis: QueryAnalysisConfig,
    /// Adaptive retrieval configuration
    pub adaptive_retrieval: AdaptiveRetrievalConfig,
    /// Performance benchmarking configuration
    pub performance_benchmarking: BenchmarkingConfig,
    /// Enhanced function registry configuration
    pub enhanced_function_registry: FunctionRegistryConfig,
    /// LightRAG dual-level retrieval configuration
    #[cfg(feature = "lightrag")]
    pub lightrag: LightRAGConfig,
    /// Leiden community detection configuration
    #[cfg(feature = "leiden")]
    pub leiden: LeidenCommunitiesConfig,
    /// Cross-encoder reranking configuration
    #[cfg(feature = "cross-encoder")]
    pub cross_encoder: CrossEncoderConfig,
}

impl Default for EnhancementsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            query_analysis: QueryAnalysisConfig::default(),
            adaptive_retrieval: AdaptiveRetrievalConfig::default(),
            performance_benchmarking: BenchmarkingConfig::default(),
            enhanced_function_registry: FunctionRegistryConfig::default(),
            #[cfg(feature = "lightrag")]
            lightrag: LightRAGConfig::default(),
            #[cfg(feature = "leiden")]
            leiden: LeidenCommunitiesConfig::default(),
            #[cfg(feature = "cross-encoder")]
            cross_encoder: CrossEncoderConfig::default(),
        }
    }
}

/// Query analysis enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysisConfig {
    /// Enable query type analysis
    pub enabled: bool,
    /// Minimum confidence for type classification
    pub min_confidence: f32,
    /// Enable automatic strategy suggestion
    pub enable_strategy_suggestion: bool,
    /// Enable keyword-based analysis
    pub enable_keyword_analysis: bool,
    /// Enable complexity scoring
    pub enable_complexity_scoring: bool,
}

impl Default for QueryAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.6,
            enable_strategy_suggestion: true,
            enable_keyword_analysis: true,
            enable_complexity_scoring: true,
        }
    }
}

/// Adaptive retrieval enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRetrievalConfig {
    /// Enable adaptive strategy selection
    pub enabled: bool,
    /// Use query analysis for strategy selection
    pub use_query_analysis: bool,
    /// Enable cross-strategy result fusion
    pub enable_cross_strategy_fusion: bool,
    /// Diversity threshold for result selection
    pub diversity_threshold: f32,
    /// Enable diversity-aware selection
    pub enable_diversity_selection: bool,
    /// Enable confidence-based weighting
    pub enable_confidence_weighting: bool,
}

impl Default for AdaptiveRetrievalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_query_analysis: true,
            enable_cross_strategy_fusion: true,
            diversity_threshold: 0.8,
            enable_diversity_selection: true,
            enable_confidence_weighting: true,
        }
    }
}

/// Performance benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    /// Enable performance benchmarking
    pub enabled: bool,
    /// Generate automatic recommendations
    pub auto_recommendations: bool,
    /// Run comprehensive testing suite
    pub comprehensive_testing: bool,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Include parallel performance testing
    pub include_parallel: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
}

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default (dev/test only)
            auto_recommendations: true,
            comprehensive_testing: false,
            iterations: 3,
            include_parallel: true,
            enable_memory_profiling: false,
        }
    }
}

/// Enhanced function registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRegistryConfig {
    /// Enable enhanced function registry
    pub enabled: bool,
    /// Enable function categorization
    pub categorization: bool,
    /// Track function usage statistics
    pub usage_statistics: bool,
    /// Allow runtime function registration
    pub dynamic_registration: bool,
    /// Enable function performance monitoring
    pub performance_monitoring: bool,
    /// Enable function recommendation system
    pub recommendation_system: bool,
}

impl Default for FunctionRegistryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            categorization: true,
            usage_statistics: true,
            dynamic_registration: true,
            performance_monitoring: false,
            recommendation_system: true,
        }
    }
}

/// LightRAG dual-level retrieval configuration
#[cfg(feature = "lightrag")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightRAGConfig {
    /// Enable LightRAG dual-level retrieval
    pub enabled: bool,
    /// Maximum keywords for query extraction (LightRAG: <20)
    pub max_keywords: usize,
    /// Weight for high-level (topic) results
    pub high_level_weight: f32,
    /// Weight for low-level (entity) results
    pub low_level_weight: f32,
    /// Merge strategy: "interleave", "high_first", "low_first", "weighted"
    pub merge_strategy: String,
    /// Language for keyword extraction
    pub language: String,
    /// Enable caching for keyword extraction
    pub enable_cache: bool,
}

#[cfg(feature = "lightrag")]
impl Default for LightRAGConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_keywords: 20,  // LightRAG optimization: <20 keywords
            high_level_weight: 0.6,
            low_level_weight: 0.4,
            merge_strategy: "weighted".to_string(),
            language: "English".to_string(),
            enable_cache: true,
        }
    }
}

/// Leiden community detection configuration
#[cfg(feature = "leiden")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeidenCommunitiesConfig {
    /// Enable Leiden community detection
    pub enabled: bool,
    /// Maximum community size
    pub max_cluster_size: usize,
    /// Use only largest connected component
    pub use_lcc: bool,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Modularity resolution (lower = larger communities)
    pub resolution: f32,
    /// Maximum hierarchical depth
    pub max_levels: usize,
    /// Minimum improvement threshold
    pub min_improvement: f32,
    /// Enable hierarchical clustering on entity graph
    pub enable_hierarchical: bool,
    /// Auto-generate summaries for each community
    pub generate_summaries: bool,
    /// Maximum summary length (number of entities/sentences)
    pub max_summary_length: usize,
    /// Use extractive summarization (vs LLM-based)
    pub use_extractive_summary: bool,
    /// Adaptive query routing configuration
    pub adaptive_routing: AdaptiveRoutingConfig,
}

/// Adaptive query routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRoutingConfig {
    /// Enable adaptive query routing
    pub enabled: bool,
    /// Default level when complexity is unclear
    pub default_level: usize,
    /// Weight for keyword-based selection (0.0-1.0)
    pub keyword_weight: f32,
    /// Weight for query length-based selection (0.0-1.0)
    pub length_weight: f32,
    /// Weight for entity mention-based selection (0.0-1.0)
    pub entity_weight: f32,
}

impl Default for AdaptiveRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_level: 1,
            keyword_weight: 0.5,
            length_weight: 0.3,
            entity_weight: 0.2,
        }
    }
}

#[cfg(feature = "leiden")]
impl Default for LeidenCommunitiesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cluster_size: 10,
            use_lcc: true,
            seed: None,
            resolution: 1.0,
            max_levels: 5,
            min_improvement: 0.001,
            enable_hierarchical: true,
            generate_summaries: true,
            max_summary_length: 5,
            use_extractive_summary: true,
            adaptive_routing: AdaptiveRoutingConfig::default(),
        }
    }
}

/// Cross-encoder reranking configuration
#[cfg(feature = "cross-encoder")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderConfig {
    /// Enable cross-encoder reranking
    pub enabled: bool,
    /// Model name/path for cross-encoder
    pub model_name: String,
    /// Maximum sequence length
    pub max_length: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Top-k results to return after reranking
    pub top_k: usize,
    /// Minimum confidence threshold (0.0-1.0)
    pub min_confidence: f32,
    /// Enable score normalization
    pub normalize_scores: bool,
}

#[cfg(feature = "cross-encoder")]
impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 512,
            batch_size: 32,
            top_k: 10,
            min_confidence: 0.0,
            normalize_scores: true,
        }
    }
}

impl EnhancementsConfig {
    /// Check if any enhancement is enabled
    pub fn has_any_enabled(&self) -> bool {
        self.enabled
            && (self.query_analysis.enabled
                || self.adaptive_retrieval.enabled
                || self.performance_benchmarking.enabled
                || self.enhanced_function_registry.enabled
                || {
                    #[cfg(feature = "lightrag")]
                    {
                        self.lightrag.enabled
                    }
                    #[cfg(not(feature = "lightrag"))]
                    {
                        false
                    }
                }
                || {
                    #[cfg(feature = "leiden")]
                    {
                        self.leiden.enabled
                    }
                    #[cfg(not(feature = "leiden"))]
                    {
                        false
                    }
                }
                || {
                    #[cfg(feature = "cross-encoder")]
                    {
                        self.cross_encoder.enabled
                    }
                    #[cfg(not(feature = "cross-encoder"))]
                    {
                        false
                    }
                })
    }

    /// Get enabled enhancements as a list
    pub fn get_enabled_enhancements(&self) -> Vec<String> {
        let mut enabled = Vec::new();

        if !self.enabled {
            return enabled;
        }

        if self.query_analysis.enabled {
            enabled.push("Query Analysis".to_string());
        }
        if self.adaptive_retrieval.enabled {
            enabled.push("Adaptive Retrieval".to_string());
        }
        if self.performance_benchmarking.enabled {
            enabled.push("Performance Benchmarking".to_string());
        }
        if self.enhanced_function_registry.enabled {
            enabled.push("Enhanced Function Registry".to_string());
        }
        #[cfg(feature = "lightrag")]
        if self.lightrag.enabled {
            enabled.push("LightRAG Dual-Level Retrieval".to_string());
        }
        #[cfg(feature = "leiden")]
        if self.leiden.enabled {
            enabled.push("Leiden Community Detection".to_string());
        }
        #[cfg(feature = "cross-encoder")]
        if self.cross_encoder.enabled {
            enabled.push("Cross-Encoder Reranking".to_string());
        }

        enabled
    }

    /// Disable all enhancements
    pub fn disable_all(&mut self) {
        self.enabled = false;
    }

    /// Enable only specific enhancements
    pub fn enable_only(&mut self, components: &[&str]) {
        // First disable all
        self.query_analysis.enabled = false;
        self.adaptive_retrieval.enabled = false;
        self.performance_benchmarking.enabled = false;
        self.enhanced_function_registry.enabled = false;
        #[cfg(feature = "lightrag")]
        {
            self.lightrag.enabled = false;
        }
        #[cfg(feature = "leiden")]
        {
            self.leiden.enabled = false;
        }
        #[cfg(feature = "cross-encoder")]
        {
            self.cross_encoder.enabled = false;
        }

        // Then enable specified ones
        for component in components {
            match component.to_lowercase().as_str() {
                "query_analysis" | "query" => self.query_analysis.enabled = true,
                "adaptive_retrieval" | "adaptive" => self.adaptive_retrieval.enabled = true,
                "performance_benchmarking" | "benchmark" => {
                    self.performance_benchmarking.enabled = true
                }
                "enhanced_function_registry" | "registry" => {
                    self.enhanced_function_registry.enabled = true
                }
                #[cfg(feature = "lightrag")]
                "lightrag" | "dual_level" => self.lightrag.enabled = true,
                #[cfg(feature = "leiden")]
                "leiden" | "communities" => self.leiden.enabled = true,
                #[cfg(feature = "cross-encoder")]
                "cross_encoder" | "reranking" => self.cross_encoder.enabled = true,
                _ => eprintln!("Unknown enhancement component: {component}"),
            }
        }

        self.enabled = true;
    }

    /// Get configuration summary
    pub fn get_summary(&self) -> EnhancementsSummary {
        let total = {
            #[allow(unused_mut)] // mut is needed for cfg blocks
            let mut count = 4; // Base components
            #[cfg(feature = "lightrag")]
            {
                count += 1;
            }
            #[cfg(feature = "leiden")]
            {
                count += 1;
            }
            #[cfg(feature = "cross-encoder")]
            {
                count += 1;
            }
            count
        };

        EnhancementsSummary {
            master_enabled: self.enabled,
            total_components: total,
            enabled_components: self.get_enabled_enhancements().len(),
            components: vec![
                ComponentStatus {
                    name: "Query Analysis".to_string(),
                    enabled: self.query_analysis.enabled,
                    features: vec![
                        (
                            "Strategy Suggestion".to_string(),
                            self.query_analysis.enable_strategy_suggestion,
                        ),
                        (
                            "Keyword Analysis".to_string(),
                            self.query_analysis.enable_keyword_analysis,
                        ),
                        (
                            "Complexity Scoring".to_string(),
                            self.query_analysis.enable_complexity_scoring,
                        ),
                    ],
                },
                ComponentStatus {
                    name: "Adaptive Retrieval".to_string(),
                    enabled: self.adaptive_retrieval.enabled,
                    features: vec![
                        (
                            "Cross-Strategy Fusion".to_string(),
                            self.adaptive_retrieval.enable_cross_strategy_fusion,
                        ),
                        (
                            "Diversity Selection".to_string(),
                            self.adaptive_retrieval.enable_diversity_selection,
                        ),
                        (
                            "Confidence Weighting".to_string(),
                            self.adaptive_retrieval.enable_confidence_weighting,
                        ),
                    ],
                },
                ComponentStatus {
                    name: "Performance Benchmarking".to_string(),
                    enabled: self.performance_benchmarking.enabled,
                    features: vec![
                        (
                            "Auto Recommendations".to_string(),
                            self.performance_benchmarking.auto_recommendations,
                        ),
                        (
                            "Comprehensive Testing".to_string(),
                            self.performance_benchmarking.comprehensive_testing,
                        ),
                        (
                            "Memory Profiling".to_string(),
                            self.performance_benchmarking.enable_memory_profiling,
                        ),
                    ],
                },
                ComponentStatus {
                    name: "Enhanced Function Registry".to_string(),
                    enabled: self.enhanced_function_registry.enabled,
                    features: vec![
                        (
                            "Categorization".to_string(),
                            self.enhanced_function_registry.categorization,
                        ),
                        (
                            "Usage Statistics".to_string(),
                            self.enhanced_function_registry.usage_statistics,
                        ),
                        (
                            "Dynamic Registration".to_string(),
                            self.enhanced_function_registry.dynamic_registration,
                        ),
                    ],
                },
                #[cfg(feature = "lightrag")]
                ComponentStatus {
                    name: "LightRAG Dual-Level Retrieval".to_string(),
                    enabled: self.lightrag.enabled,
                    features: vec![
                        (
                            "Dual-Level Keywords".to_string(),
                            true, // Always enabled when LightRAG is enabled
                        ),
                        (
                            format!("Max Keywords: {}", self.lightrag.max_keywords),
                            true,
                        ),
                        (
                            format!("Merge: {}", self.lightrag.merge_strategy),
                            true,
                        ),
                    ],
                },
                #[cfg(feature = "leiden")]
                ComponentStatus {
                    name: "Leiden Community Detection".to_string(),
                    enabled: self.leiden.enabled,
                    features: vec![
                        (
                            format!("Max Cluster Size: {}", self.leiden.max_cluster_size),
                            true,
                        ),
                        (
                            format!("Resolution: {:.2}", self.leiden.resolution),
                            true,
                        ),
                        (
                            "Refinement Phase".to_string(),
                            true, // Always enabled (KEY difference from Louvain)
                        ),
                        (
                            "Hierarchical Clustering".to_string(),
                            self.leiden.enable_hierarchical,
                        ),
                        (
                            "Auto-Generate Summaries".to_string(),
                            self.leiden.generate_summaries,
                        ),
                        (
                            format!("Max Levels: {}", self.leiden.max_levels),
                            self.leiden.enable_hierarchical,
                        ),
                        (
                            "Adaptive Query Routing".to_string(),
                            self.leiden.adaptive_routing.enabled,
                        ),
                    ],
                },
                #[cfg(feature = "cross-encoder")]
                ComponentStatus {
                    name: "Cross-Encoder Reranking".to_string(),
                    enabled: self.cross_encoder.enabled,
                    features: vec![
                        (
                            format!("Model: {}", self.cross_encoder.model_name),
                            true,
                        ),
                        (
                            format!("Top-K: {}", self.cross_encoder.top_k),
                            true,
                        ),
                        (
                            "Score Normalization".to_string(),
                            self.cross_encoder.normalize_scores,
                        ),
                    ],
                },
            ],
        }
    }
}

/// Summary of enhancements configuration
#[derive(Debug)]
pub struct EnhancementsSummary {
    /// Whether enhancements are enabled at master level
    pub master_enabled: bool,
    /// Total number of enhancement components
    pub total_components: usize,
    /// Number of enabled enhancement components
    pub enabled_components: usize,
    /// Detailed status of each enhancement component
    pub components: Vec<ComponentStatus>,
}

/// Status of individual enhancement component
#[derive(Debug)]
pub struct ComponentStatus {
    /// Name of the component
    pub name: String,
    /// Whether the component is enabled
    pub enabled: bool,
    /// List of features and their enabled status
    pub features: Vec<(String, bool)>,
}

impl EnhancementsSummary {
    /// Print configuration summary
    pub fn print(&self) {
        println!("🚀 GraphRAG Enhancements Configuration");
        println!("{}", "=".repeat(50));
        println!(
            "Master Switch: {}",
            if self.master_enabled {
                "✅ Enabled"
            } else {
                "❌ Disabled"
            }
        );
        println!(
            "Components: {}/{} enabled",
            self.enabled_components, self.total_components
        );

        for component in &self.components {
            let status = if component.enabled && self.master_enabled {
                "✅"
            } else {
                "❌"
            };
            println!("\n{} {}", status, component.name);

            if component.enabled && self.master_enabled {
                for (feature, enabled) in &component.features {
                    let feature_status = if *enabled { "  ✓" } else { "  ✗" };
                    println!("  {feature_status} {feature}");
                }
            }
        }
    }

    /// Get enabled percentage
    pub fn get_enabled_percentage(&self) -> f32 {
        if !self.master_enabled {
            return 0.0;
        }
        (self.enabled_components as f32 / self.total_components as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EnhancementsConfig::default();
        assert!(config.enabled);
        assert!(config.query_analysis.enabled);
        assert!(config.adaptive_retrieval.enabled);
        assert!(!config.performance_benchmarking.enabled); // Disabled by default
        assert!(config.enhanced_function_registry.enabled);
    }

    #[test]
    fn test_enable_only() {
        let mut config = EnhancementsConfig::default();
        config.enable_only(&["query_analysis", "adaptive_retrieval"]);

        assert!(config.query_analysis.enabled);
        assert!(config.adaptive_retrieval.enabled);
        assert!(!config.performance_benchmarking.enabled);
        assert!(!config.enhanced_function_registry.enabled);
    }

    #[test]
    fn test_disable_all() {
        let mut config = EnhancementsConfig::default();
        config.disable_all();

        assert!(!config.enabled);
        assert!(!config.has_any_enabled());
    }

    #[test]
    fn test_summary() {
        let config = EnhancementsConfig::default();
        let summary = config.get_summary();

        let expected = {
            let mut count = 4;
            #[cfg(feature = "lightrag")]
            {
                count += 1;
            }
            #[cfg(feature = "leiden")]
            {
                count += 1;
            }
            #[cfg(feature = "cross-encoder")]
            {
                count += 1;
            }
            count
        };
        assert_eq!(summary.total_components, expected);
        assert!(summary.get_enabled_percentage() > 0.0);
    }
}
