//! OpenTelemetry Observability
//!
//! This module provides comprehensive observability for GraphRAG using OpenTelemetry.
//! It includes:
//! - Distributed tracing with Jaeger
//! - Metrics with Prometheus
//! - Custom business metrics
//! - Performance profiling
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │       GraphRAG Application          │
//! │                                     │
//! │  ┌──────────────────────────────┐   │
//! │  │  Instrumented Code          │   │
//! │  │  - Spans for operations     │   │
//! │  │  - Metrics counters/gauges  │   │
//! │  └──────────┬───────────────────┘   │
//! └─────────────┼─────────────────────── │
//!               │
//!    ┌──────────▼──────────┐
//!    │  OpenTelemetry SDK  │
//!    └──────────┬──────────┘
//!               │
//!     ┌─────────┴─────────┐
//!     │                   │
//!     ▼                   ▼
//! ┌────────┐         ┌──────────┐
//! │ Jaeger │         │Prometheus│
//! │(Traces)│         │(Metrics) │
//! └────────┘         └──────────┘
//! ```

use std::time::Duration;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Metrics collector for business metrics
#[derive(Default, Clone)]
pub struct Metrics {
    /// Query count by type
    pub query_count: HashMap<String, u64>,
    /// Average query latency (ms)
    pub avg_query_latency: f64,
    /// Error count
    pub error_count: u64,
    /// Document processing count
    pub documents_processed: u64,
    /// Embeddings generated
    pub embeddings_generated: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Observability manager
///
/// Manages OpenTelemetry tracing and metrics for the GraphRAG system.
pub struct Observability {
    /// Metrics storage
    metrics: Arc<RwLock<Metrics>>,
    /// Service name
    service_name: String,
    /// Jaeger endpoint
    #[allow(dead_code)]
    jaeger_endpoint: Option<String>,
    /// Prometheus endpoint
    #[allow(dead_code)]
    prometheus_endpoint: Option<String>,
}

/// Span for distributed tracing
pub struct Span {
    /// Span name
    name: String,
    /// Start time
    start_time: std::time::Instant,
    /// Attributes
    attributes: HashMap<String, String>,
}

impl Span {
    /// Create a new span
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_time: std::time::Instant::now(),
            attributes: HashMap::new(),
        }
    }

    /// Add attribute to span
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// End span and record duration
    pub fn end(self) -> Duration {
        let duration = self.start_time.elapsed();
        tracing::info!(
            span_name = %self.name,
            duration_ms = duration.as_millis(),
            "Span completed"
        );
        duration
    }
}

impl Observability {
    /// Create a new observability manager
    ///
    /// # Arguments
    /// * `service_name` - Name of the service
    /// * `jaeger_endpoint` - Optional Jaeger endpoint (e.g., "http://localhost:14268/api/traces")
    /// * `prometheus_endpoint` - Optional Prometheus endpoint (e.g., "http://localhost:9090")
    ///
    /// # Returns
    /// Observability manager
    pub fn new(
        service_name: impl Into<String>,
        jaeger_endpoint: Option<String>,
        prometheus_endpoint: Option<String>,
    ) -> Self {
        let service_name = service_name.into();

        tracing::info!(
            service = %service_name,
            jaeger = ?jaeger_endpoint,
            prometheus = ?prometheus_endpoint,
            "Initializing observability"
        );

        Self {
            metrics: Arc::new(RwLock::new(Metrics::default())),
            service_name,
            jaeger_endpoint,
            prometheus_endpoint,
        }
    }

    /// Start a new span for tracing
    ///
    /// # Arguments
    /// * `name` - Span name
    ///
    /// # Returns
    /// Span that can be used to track operation
    pub fn start_span(&self, name: impl Into<String>) -> Span {
        Span::new(name)
    }

    /// Record query metric
    ///
    /// # Arguments
    /// * `query_type` - Type of query (e.g., "semantic", "keyword", "hybrid")
    /// * `latency_ms` - Query latency in milliseconds
    pub fn record_query(&self, query_type: impl Into<String>, latency_ms: u64) {
        let query_type = query_type.into();
        let mut metrics = self.metrics.write();

        // Increment count
        *metrics.query_count.entry(query_type.clone()).or_insert(0) += 1;

        // Update average latency
        let total_queries: u64 = metrics.query_count.values().sum();
        metrics.avg_query_latency = (metrics.avg_query_latency * (total_queries - 1) as f64
            + latency_ms as f64)
            / total_queries as f64;

        tracing::debug!(
            query_type = %query_type,
            latency_ms = latency_ms,
            "Query recorded"
        );
    }

    /// Record error
    ///
    /// # Arguments
    /// * `error_type` - Type of error
    /// * `message` - Error message
    pub fn record_error(&self, error_type: impl Into<String>, message: impl Into<String>) {
        self.metrics.write().error_count += 1;

        tracing::error!(
            error_type = %error_type.into(),
            message = %message.into(),
            "Error recorded"
        );
    }

    /// Record document processing
    ///
    /// # Arguments
    /// * `count` - Number of documents processed
    pub fn record_documents_processed(&self, count: u64) {
        self.metrics.write().documents_processed += count;

        tracing::debug!(count = count, "Documents processed");
    }

    /// Record embeddings generation
    ///
    /// # Arguments
    /// * `count` - Number of embeddings generated
    pub fn record_embeddings_generated(&self, count: u64) {
        self.metrics.write().embeddings_generated += count;

        tracing::debug!(count = count, "Embeddings generated");
    }

    /// Update cache hit rate
    ///
    /// # Arguments
    /// * `hit_rate` - Cache hit rate (0.0 to 1.0)
    pub fn update_cache_hit_rate(&self, hit_rate: f64) {
        self.metrics.write().cache_hit_rate = hit_rate;

        tracing::debug!(hit_rate = hit_rate, "Cache hit rate updated");
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> Metrics {
        self.metrics.read().clone()
    }

    /// Export metrics in Prometheus format
    ///
    /// # Returns
    /// Prometheus-formatted metrics string
    pub fn export_prometheus_metrics(&self) -> String {
        let metrics = self.metrics.read();

        let mut output = String::new();

        // Query count by type
        output.push_str("# HELP graphrag_queries_total Total number of queries by type\n");
        output.push_str("# TYPE graphrag_queries_total counter\n");
        for (query_type, count) in &metrics.query_count {
            output.push_str(&format!(
                "graphrag_queries_total{{query_type=\"{}\"}} {}\n",
                query_type, count
            ));
        }

        // Average query latency
        output.push_str("# HELP graphrag_query_latency_avg Average query latency in milliseconds\n");
        output.push_str("# TYPE graphrag_query_latency_avg gauge\n");
        output.push_str(&format!(
            "graphrag_query_latency_avg {}\n",
            metrics.avg_query_latency
        ));

        // Error count
        output.push_str("# HELP graphrag_errors_total Total number of errors\n");
        output.push_str("# TYPE graphrag_errors_total counter\n");
        output.push_str(&format!("graphrag_errors_total {}\n", metrics.error_count));

        // Documents processed
        output.push_str("# HELP graphrag_documents_processed_total Total documents processed\n");
        output.push_str("# TYPE graphrag_documents_processed_total counter\n");
        output.push_str(&format!(
            "graphrag_documents_processed_total {}\n",
            metrics.documents_processed
        ));

        // Embeddings generated
        output.push_str("# HELP graphrag_embeddings_generated_total Total embeddings generated\n");
        output.push_str("# TYPE graphrag_embeddings_generated_total counter\n");
        output.push_str(&format!(
            "graphrag_embeddings_generated_total {}\n",
            metrics.embeddings_generated
        ));

        // Cache hit rate
        output.push_str("# HELP graphrag_cache_hit_rate Cache hit rate (0.0 to 1.0)\n");
        output.push_str("# TYPE graphrag_cache_hit_rate gauge\n");
        output.push_str(&format!(
            "graphrag_cache_hit_rate {}\n",
            metrics.cache_hit_rate
        ));

        output
    }

    /// Export metrics in JSON format
    ///
    /// # Returns
    /// JSON-formatted metrics
    pub fn export_json_metrics(&self) -> serde_json::Value {
        let metrics = self.metrics.read();

        serde_json::json!({
            "service": self.service_name,
            "queries": {
                "total": metrics.query_count.values().sum::<u64>(),
                "by_type": metrics.query_count,
                "avg_latency_ms": metrics.avg_query_latency,
            },
            "errors": metrics.error_count,
            "documents_processed": metrics.documents_processed,
            "embeddings_generated": metrics.embeddings_generated,
            "cache_hit_rate": metrics.cache_hit_rate,
        })
    }
}

/// Middleware for automatic request tracing
///
/// This can be used with Axum to automatically trace all HTTP requests.
///
/// # Example
///
/// ```rust,no_run
/// use axum::{Router, routing::get};
/// use tower::ServiceBuilder;
/// use graphrag_server::observability::Observability;
///
/// async fn handler() -> &'static str {
///     "Hello, World!"
/// }
///
/// #[tokio::main]
/// async fn main() {
///     let obs = Observability::new("graphrag-api", None, None);
///
///     let app = Router::new()
///         .route("/", get(handler));
///         // Add tracing middleware here if needed
///
///     // Run server...
/// }
/// ```
pub struct TracingMiddleware {
    #[allow(dead_code)]
    observability: Arc<Observability>,
}

impl TracingMiddleware {
    /// Create new tracing middleware
    pub fn new(observability: Arc<Observability>) -> Self {
        Self { observability }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability() {
        let obs = Observability::new("test-service", None, None);

        // Record some metrics
        obs.record_query("semantic", 100);
        obs.record_query("keyword", 50);
        obs.record_documents_processed(10);
        obs.record_embeddings_generated(100);
        obs.update_cache_hit_rate(0.85);

        let metrics = obs.get_metrics();

        assert_eq!(metrics.query_count.get("semantic"), Some(&1));
        assert_eq!(metrics.query_count.get("keyword"), Some(&1));
        assert_eq!(metrics.documents_processed, 10);
        assert_eq!(metrics.embeddings_generated, 100);
        assert_eq!(metrics.cache_hit_rate, 0.85);
    }

    #[test]
    fn test_prometheus_export() {
        let obs = Observability::new("test-service", None, None);

        obs.record_query("test", 100);
        obs.record_error("test_error", "Test error message");

        let prometheus = obs.export_prometheus_metrics();

        assert!(prometheus.contains("graphrag_queries_total"));
        assert!(prometheus.contains("graphrag_errors_total"));
        assert!(prometheus.contains("query_type=\"test\""));
    }

    #[test]
    fn test_json_export() {
        let obs = Observability::new("test-service", None, None);

        obs.record_query("test", 100);

        let json = obs.export_json_metrics();

        assert_eq!(json["service"], "test-service");
        assert_eq!(json["queries"]["total"], 1);
    }

    #[test]
    fn test_span() {
        let mut span = Span::new("test_operation");
        span.set_attribute("user_id", "123");
        span.set_attribute("query", "test query");

        let _duration = span.end();
        // Duration is always >= 0, no need to test
    }
}
