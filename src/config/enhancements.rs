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
}

impl Default for EnhancementsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            query_analysis: QueryAnalysisConfig::default(),
            adaptive_retrieval: AdaptiveRetrievalConfig::default(),
            performance_benchmarking: BenchmarkingConfig::default(),
            enhanced_function_registry: FunctionRegistryConfig::default(),
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

impl EnhancementsConfig {
    /// Check if any enhancement is enabled
    pub fn has_any_enabled(&self) -> bool {
        self.enabled
            && (self.query_analysis.enabled
                || self.adaptive_retrieval.enabled
                || self.performance_benchmarking.enabled
                || self.enhanced_function_registry.enabled)
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
                _ => eprintln!("Unknown enhancement component: {component}"),
            }
        }

        self.enabled = true;
    }

    /// Get configuration summary
    pub fn get_summary(&self) -> EnhancementsSummary {
        EnhancementsSummary {
            master_enabled: self.enabled,
            total_components: 4,
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
            ],
        }
    }
}

/// Summary of enhancements configuration
#[derive(Debug)]
pub struct EnhancementsSummary {
    pub master_enabled: bool,
    pub total_components: usize,
    pub enabled_components: usize,
    pub components: Vec<ComponentStatus>,
}

/// Status of individual enhancement component
#[derive(Debug)]
pub struct ComponentStatus {
    pub name: String,
    pub enabled: bool,
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

        assert_eq!(summary.total_components, 4);
        assert!(summary.get_enabled_percentage() > 0.0);
    }
}
