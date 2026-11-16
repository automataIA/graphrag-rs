//! Cross-encoder reranker client for llama.cpp (`llama-server --reranking`).
//!
//! Calls `POST {endpoint}/v1/rerank` and returns raw relevance logits
//! (unbounded, higher = more relevant) mapped back to the input positions.
//! The API key, when needed, is read from the env var named by
//! `reranker.api_key_env` — never stored in configuration.

use crate::config::RerankerConfig;
use crate::core::{GraphRAGError, Result};

fn rerank_error(message: impl Into<String>) -> GraphRAGError {
    GraphRAGError::Retrieval {
        message: message.into(),
    }
}

fn rerank_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1/rerank") {
        base.to_string()
    } else {
        format!("{base}/v1/rerank")
    }
}

/// Parse a `/v1/rerank` response into `(input_index, score)` pairs sorted by
/// descending score. The server may return results in any order and may omit
/// documents (`top_n`); entries with an out-of-range, duplicate or non-finite
/// value are skipped.
pub fn parse_rerank_response(
    value: &serde_json::Value,
    n_docs: usize,
) -> Result<Vec<(usize, f32)>> {
    let results = value
        .get("results")
        .and_then(|r| r.as_array())
        .ok_or_else(|| rerank_error("rerank response has no `results` array"))?;
    let mut seen = vec![false; n_docs];
    let mut scored = Vec::with_capacity(results.len());
    for item in results {
        let index = item
            .get("index")
            .and_then(|i| i.as_u64())
            .map(|i| i as usize);
        let score = item.get("relevance_score").and_then(|s| s.as_f64());
        let (Some(index), Some(score)) = (index, score) else {
            continue;
        };
        if index >= n_docs || seen[index] || !score.is_finite() {
            continue;
        }
        seen[index] = true;
        scored.push((index, score as f32));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

/// Maximum documents per `/v1/rerank` request. Smaller requests lower the load on
/// the llama.cpp server (a single request with ~300 documents crashed it in CUDA).
const RERANK_BATCH_SIZE: usize = 64;

/// Index ranges that split `n_docs` documents into consecutive batches.
fn batch_ranges(n_docs: usize) -> Vec<std::ops::Range<usize>> {
    (0..n_docs)
        .step_by(RERANK_BATCH_SIZE)
        .map(|start| start..(start + RERANK_BATCH_SIZE).min(n_docs))
        .collect()
}

/// Merge per-batch `(local_index, score)` lists (each with its global offset) into one
/// list of global indices sorted by descending score (ties by ascending index), then
/// keep the best `top_n`.
fn merge_batches(
    batches: Vec<(usize, Vec<(usize, f32)>)>,
    top_n: Option<usize>,
) -> Vec<(usize, f32)> {
    let mut merged: Vec<(usize, f32)> = batches
        .into_iter()
        .flat_map(|(offset, scored)| scored.into_iter().map(move |(i, s)| (offset + i, s)))
        .collect();
    merged.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if let Some(top_n) = top_n {
        merged.truncate(top_n);
    }
    merged
}

/// Blocking/async client for the llama.cpp rerank endpoint.
#[derive(Clone)]
pub struct LlamaCppReranker {
    url: String,
    model: String,
    api_key_env: Option<String>,
    timeout_seconds: u64,
    agent: ureq::Agent,
}

impl std::fmt::Debug for LlamaCppReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppReranker")
            .field("url", &self.url)
            .field("model", &self.model)
            .field("api_key_env", &self.api_key_env)
            .finish()
    }
}

impl LlamaCppReranker {
    /// Build a client from the runtime config. No network I/O happens here.
    pub fn from_config(config: &RerankerConfig) -> Self {
        let timeout_seconds = config.timeout_seconds.max(1);
        Self {
            url: rerank_url(&config.api_endpoint),
            model: config.model.clone(),
            api_key_env: config.api_key_env.clone(),
            timeout_seconds,
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(timeout_seconds))
                .build(),
        }
    }

    fn api_key(&self) -> Result<Option<String>> {
        let Some(var) = self.api_key_env.as_deref().filter(|v| !v.trim().is_empty()) else {
            return Ok(None);
        };
        match std::env::var(var) {
            Ok(key) if !key.trim().is_empty() => Ok(Some(key)),
            _ => Err(GraphRAGError::Config {
                message: format!(
                    "reranker.api_key_env names {var}, but that variable is unset or empty"
                ),
            }),
        }
    }

    /// Score `documents` against `query`. Returns `(document_index, score)` sorted by
    /// descending score. Documents are sent in sequential batches of at most
    /// `RERANK_BATCH_SIZE`; `top_n` is applied after the batches are merged.
    pub fn rerank_blocking(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.len() <= RERANK_BATCH_SIZE {
            // Single request: identical to the pre-batching behaviour.
            return self.rerank_request(query, documents, top_n);
        }
        let mut batches = Vec::new();
        for range in batch_ranges(documents.len()) {
            let offset = range.start;
            // Ask for every document of the batch: no candidate is cut early.
            batches.push((offset, self.rerank_request(query, &documents[range], None)?));
        }
        Ok(merge_batches(batches, top_n))
    }

    /// One `/v1/rerank` request; returned indices are local to `documents`.
    fn rerank_request(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        });
        if let Some(top_n) = top_n {
            body["top_n"] = top_n.into();
        }
        let mut request = self
            .agent
            .post(&self.url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.api_key()? {
            request = request.set("Authorization", &format!("Bearer {key}"));
        }
        let value: serde_json::Value = match request.send_json(&body) {
            Ok(response) => response.into_json().map_err(|e| {
                rerank_error(format!(
                    "rerank response from {} was not JSON: {e}",
                    self.url
                ))
            })?,
            Err(ureq::Error::Status(status, response)) => {
                let detail = crate::llama_cpp::bounded_response_body(response);
                return Err(rerank_error(format!(
                    "rerank HTTP {status} from {}: {detail}",
                    self.url
                )));
            },
            Err(ureq::Error::Transport(transport)) => {
                return Err(rerank_error(format!(
                    "rerank request to {} failed (timeout {}s): {transport}",
                    self.url, self.timeout_seconds
                )));
            },
        };
        parse_rerank_response(&value, documents.len())
    }

    /// Async wrapper around [`Self::rerank_blocking`] (runs on the blocking pool).
    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_n: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let client = self.clone();
        let query = query.to_string();
        tokio::task::spawn_blocking(move || client.rerank_blocking(&query, &documents, top_n))
            .await
            .map_err(|e| rerank_error(format!("rerank worker failed: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_out_of_order_results_by_index() {
        let v = json!({"results": [
            {"index": 2, "relevance_score": -8.3},
            {"index": 0, "relevance_score": 3.04},
            {"index": 1, "relevance_score": 0.5},
        ]});
        let r = parse_rerank_response(&v, 3).unwrap();
        assert_eq!(r, vec![(0, 3.04), (1, 0.5), (2, -8.3)]);
    }

    #[test]
    fn skips_missing_extra_duplicate_and_malformed_entries() {
        let v = json!({"results": [
            {"index": 5, "relevance_score": 9.0},      // out of range
            {"index": 1, "relevance_score": 1.0},
            {"index": 1, "relevance_score": 7.0},      // duplicate
            {"index": 0},                              // no score
            {"relevance_score": 2.0},                  // no index
        ]});
        // index 0 and 2 never scored: simply absent.
        assert_eq!(parse_rerank_response(&v, 3).unwrap(), vec![(1, 1.0)]);
    }

    #[test]
    fn missing_results_array_is_an_error() {
        assert!(parse_rerank_response(&json!({"error": "x"}), 2).is_err());
    }

    #[test]
    fn url_gets_rerank_path_once() {
        assert_eq!(rerank_url("http://h:1"), "http://h:1/v1/rerank");
        assert_eq!(rerank_url("http://h:1/"), "http://h:1/v1/rerank");
        assert_eq!(rerank_url("http://h:1/v1/rerank"), "http://h:1/v1/rerank");
    }

    #[test]
    fn batch_ranges_boundaries() {
        assert!(batch_ranges(0).is_empty());
        assert_eq!(batch_ranges(1), vec![0..1]);
        assert_eq!(batch_ranges(64), vec![0..64]);
        assert_eq!(batch_ranges(65), vec![0..64, 64..65]);
        assert_eq!(batch_ranges(130), vec![0..64, 64..128, 128..130]);
    }

    #[test]
    fn merge_maps_local_indices_to_global() {
        let merged = merge_batches(
            vec![
                (0, vec![(0, 1.0), (1, 3.0)]),
                (64, vec![(0, 2.0), (5, 4.0)]),
            ],
            None,
        );
        assert_eq!(merged, vec![(69, 4.0), (1, 3.0), (64, 2.0), (0, 1.0)]);
    }

    #[test]
    fn merge_breaks_ties_by_ascending_index() {
        let merged = merge_batches(
            vec![(64, vec![(0, 1.0)]), (0, vec![(3, 1.0), (1, 1.0)])],
            None,
        );
        assert_eq!(merged, vec![(1, 1.0), (3, 1.0), (64, 1.0)]);
    }

    #[test]
    fn merge_applies_top_n_after_merge() {
        let batches = vec![
            (0, vec![(0, 1.0), (1, 2.0)]),
            (64, vec![(0, 5.0), (1, 4.0)]),
        ];
        // Best two overall both come from the second batch.
        assert_eq!(
            merge_batches(batches.clone(), Some(2)),
            vec![(64, 5.0), (65, 4.0)]
        );
        assert_eq!(merge_batches(batches.clone(), Some(10)).len(), 4);
        assert_eq!(merge_batches(batches, None).len(), 4);
    }

    #[test]
    fn merge_of_nothing_is_empty() {
        assert!(merge_batches(Vec::new(), Some(3)).is_empty());
    }
}
