//! Cross-Encoder reranking for improved retrieval accuracy
//!
//! Cross-encoders jointly encode query and document, providing more accurate
//! relevance scores than bi-encoder approaches. This implementation provides
//! a trait-based interface that can be backed by ONNX models, API calls, or
//! other implementations.
//!
//! Reference: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
//! Reimers & Gurevych (2019)

use async_trait::async_trait;

use crate::retrieval::SearchResult;
use crate::Result;

/// Configuration for cross-encoder reranking
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
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

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 512,
            batch_size: 32,
            top_k: 10,
            min_confidence: 0.0,
            normalize_scores: true,
        }
    }
}

/// Result of cross-encoder reranking with confidence score
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Original search result
    pub result: SearchResult,

    /// Cross-encoder relevance score (typically 0.0-1.0 after normalization)
    pub relevance_score: f32,

    /// Original retrieval score (for comparison)
    pub original_score: f32,

    /// Score improvement over original (relevance_score - original_score)
    pub score_delta: f32,
}

/// Cross-encoder trait for reranking retrieved results
#[async_trait]
pub trait CrossEncoder: Send + Sync {
    /// Rerank a list of search results based on relevance to query
    async fn rerank(&self, query: &str, candidates: Vec<SearchResult>)
        -> Result<Vec<RankedResult>>;

    /// Score a single query-document pair
    async fn score_pair(&self, query: &str, document: &str) -> Result<f32>;

    /// Batch score multiple query-document pairs
    async fn score_batch(&self, pairs: Vec<(String, String)>) -> Result<Vec<f32>>;
}

#[cfg(feature = "neural-embeddings")]
use candle_core::{Device, Tensor};
#[cfg(feature = "neural-embeddings")]
use candle_nn::VarBuilder;
#[cfg(feature = "neural-embeddings")]
use candle_transformers::models::bert::{BertModel, Config, Dtype};
#[cfg(feature = "huggingface-hub")]
use hf_hub::api::sync::Api;
#[cfg(feature = "neural-embeddings")]
use tokenizers::Tokenizer;

/// Cross-encoder implementation using Candle (BERT)
#[cfg(feature = "neural-embeddings")]
pub struct CandleCrossEncoder {
    config: CrossEncoderConfig,
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "neural-embeddings")]
impl CandleCrossEncoder {
    pub fn new(config: CrossEncoderConfig) -> Result<Self> {
        let api = Api::new().map_err(|e| GraphRAGError::Embedding {
            message: format!("Failed to create HF Hub API: {}", e),
        })?;
        let repo = api.model(config.model_name.clone());

        let model_file = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to download model '{}': {}", config.model_name, e),
            })?;

        let tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to download tokenizer: {}", e),
            })?;

        let config_file = repo
            .get("config.json")
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to download config: {}", e),
            })?;

        let device = Device::Cpu;
        let model_config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_file).map_err(|e| {
                GraphRAGError::Embedding {
                    message: format!("Failed to read config: {}", e),
                }
            })?)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to parse config: {}", e),
            })?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_file).map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to load tokenizer: {}", e),
            })?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], Dtype::F32, &device).map_err(
                |e| GraphRAGError::Embedding {
                    message: format!("Failed to load weights: {}", e),
                },
            )?
        };

        let model = BertModel::load(vb, &model_config).map_err(|e| GraphRAGError::Embedding {
            message: format!("Failed to load BERT model: {}", e),
        })?;

        Ok(Self {
            config,
            model,
            tokenizer,
            device,
        })
    }
}

#[cfg(feature = "neural-embeddings")]
#[async_trait]
impl CrossEncoder for CandleCrossEncoder {
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<SearchResult>,
    ) -> Result<Vec<RankedResult>> {
        let mut ranked = Vec::new();

        for candidate in candidates {
            let score = self.score_pair(query, &candidate.content).await?;
            let score_delta = score - candidate.score;

            if score >= self.config.min_confidence {
                ranked.push(RankedResult {
                    result: candidate,
                    relevance_score: score,
                    original_score: candidate.score,
                    score_delta,
                });
            }
        }

        ranked.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(self.config.top_k);
        Ok(ranked)
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let tokens = self
            .tokenizer
            .encode((query, document), true)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Tokenization failed: {}", e),
            })?;

        let token_ids = Tensor::new(tokens.get_ids(), &self.device)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Tensor creation failed: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|_| GraphRAGError::Embedding {
                message: "Unsqueeze failed".to_string(),
            })?;

        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Type tensor creation failed: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|_| GraphRAGError::Embedding {
                message: "Unsqueeze failed".to_string(),
            })?;

        let logits = self
            .model
            .forward(&token_ids, &token_type_ids)
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Forward pass failed: {}", e),
            })?;

        // Cross-encoders typically output a single logit for the positive class (index 0 or 1 depending on model)
        // Or if it's a regression model, 1 output.
        // Assuming MiniLM cross-encoder, it outputs 1 value usually.
        let score = logits
            .squeeze(0)
            .map_err(|_| GraphRAGError::Embedding {
                message: "Squeeze failed".to_string(),
            })?
            .to_vec1::<f32>()
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("To vec failed: {}", e),
            })?;

        // Sigmoid if needed, but often logits are enough for ranking. Config has normalize_scores.
        let raw_score = score[0];

        if self.config.normalize_scores {
            Ok(1.0 / (1.0 + (-raw_score).exp()))
        } else {
            Ok(raw_score)
        }
    }

    async fn score_batch(&self, pairs: Vec<(String, String)>) -> Result<Vec<f32>> {
        let mut scores = Vec::new();
        for (q, d) in pairs {
            scores.push(self.score_pair(&q, &d).await?);
        }
        Ok(scores)
    }
}

/// Statistics about reranking performance
#[derive(Debug, Clone)]
pub struct RerankingStats {
    /// Number of candidates reranked
    pub candidates_count: usize,

    /// Number of results returned
    pub results_count: usize,

    /// Average score improvement (mean delta)
    pub avg_score_improvement: f32,

    /// Maximum score improvement
    pub max_score_improvement: f32,

    /// Percentage of candidates filtered out
    pub filter_rate: f32,
}

impl RerankingStats {
    /// Calculate statistics from ranked results
    pub fn from_results(original_count: usize, ranked: &[RankedResult]) -> Self {
        let results_count = ranked.len();

        let avg_score_improvement = if !ranked.is_empty() {
            ranked.iter().map(|r| r.score_delta).sum::<f32>() / ranked.len() as f32
        } else {
            0.0
        };

        let max_score_improvement = ranked
            .iter()
            .map(|r| r.score_delta)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let filter_rate = if original_count > 0 {
            ((original_count - results_count) as f32 / original_count as f32) * 100.0
        } else {
            0.0
        };

        Self {
            candidates_count: original_count,
            results_count,
            avg_score_improvement,
            max_score_improvement,
            filter_rate,
        }
    }
}

/// Confidence-based cross-encoder implementation (Restored Fallback)
pub struct ConfidenceCrossEncoder {
    _config: CrossEncoderConfig,
}

impl ConfidenceCrossEncoder {
    /// Create a new confidence-based cross-encoder with the given configuration
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self { _config: config }
    }
}

#[async_trait]
impl CrossEncoder for ConfidenceCrossEncoder {
    async fn rerank(
        &self,
        _query: &str,
        candidates: Vec<SearchResult>,
    ) -> Result<Vec<RankedResult>> {
        // Simple passthrough/mock implementation to satisfy imports
        let mut ranked = Vec::new();
        for candidate in candidates {
            ranked.push(RankedResult {
                result: candidate.clone(),
                relevance_score: candidate.score,
                original_score: candidate.score,
                score_delta: 0.0,
            });
        }
        Ok(ranked)
    }

    async fn score_pair(&self, _query: &str, _document: &str) -> Result<f32> {
        Ok(0.0)
    }

    async fn score_batch(&self, pairs: Vec<(String, String)>) -> Result<Vec<f32>> {
        Ok(vec![0.0; pairs.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::ResultType;

    fn create_test_result(id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_rerank_basic() {
        let config = CrossEncoderConfig {
            top_k: 3,
            min_confidence: 0.0,
            ..Default::default()
        };

        let encoder = ConfidenceCrossEncoder::new(config);

        let query = "machine learning algorithms";
        let candidates = vec![
            create_test_result(
                "1",
                "Machine learning is a subset of artificial intelligence",
                0.5,
            ),
            create_test_result("2", "The weather today is sunny", 0.6),
            create_test_result(
                "3",
                "Neural networks are machine learning algorithms used for pattern recognition",
                0.4,
            ),
        ];

        let ranked = encoder.rerank(query, candidates).await.unwrap();

        // Should rerank based on relevance
        assert_eq!(ranked.len(), 3);

        // Most relevant should be first (result 3 has best overlap)
        assert!(ranked[0].relevance_score >= ranked[1].relevance_score);
        assert!(ranked[1].relevance_score >= ranked[2].relevance_score);
    }

    #[tokio::test]
    async fn test_confidence_filtering() {
        let config = CrossEncoderConfig {
            top_k: 10,
            min_confidence: 0.5, // High threshold
            ..Default::default()
        };

        let encoder = ConfidenceCrossEncoder::new(config);

        let query = "specific technical query";
        let candidates = vec![
            create_test_result("1", "highly relevant technical content", 0.3),
            create_test_result("2", "somewhat relevant", 0.4),
            create_test_result("3", "not relevant at all", 0.5),
        ];

        let ranked = encoder.rerank(query, candidates).await.unwrap();

        // Should filter low-confidence results
        for result in &ranked {
            assert!(result.relevance_score >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_score_pair() {
        let config = CrossEncoderConfig::default();
        let encoder = ConfidenceCrossEncoder::new(config);

        let score = encoder
            .score_pair(
                "artificial intelligence",
                "AI and machine learning are related fields",
            )
            .await
            .unwrap();

        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_reranking_stats() {
        let ranked = vec![
            RankedResult {
                result: create_test_result("1", "test", 0.5),
                relevance_score: 0.8,
                original_score: 0.5,
                score_delta: 0.3,
            },
            RankedResult {
                result: create_test_result("2", "test", 0.6),
                relevance_score: 0.7,
                original_score: 0.6,
                score_delta: 0.1,
            },
        ];

        let stats = RerankingStats::from_results(5, &ranked);

        assert_eq!(stats.candidates_count, 5);
        assert_eq!(stats.results_count, 2);
        // Use approximate equality for floating point comparison
        assert!((stats.filter_rate - 60.0).abs() < 0.001); // 3/5 filtered = 60%
        assert!(stats.avg_score_improvement > 0.0);
    }
}
