//! Metrics collection implementation
//!
//! This module provides a comprehensive metrics collector for monitoring GraphRAG operations.

use crate::core::error::Result;
use crate::core::traits::{AsyncMetricsCollector, AsyncTimer};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Comprehensive metrics collector for GraphRAG operations
#[derive(Clone)]
pub struct MetricsCollector {
    /// Counter metrics (cumulative values)
    counters: Arc<dashmap::DashMap<String, Arc<AtomicU64>>>,
    /// Gauge metrics (current values)
    gauges: Arc<dashmap::DashMap<String, Arc<std::sync::RwLock<f64>>>>,
    /// Histogram metrics (value distributions)
    histograms: Arc<dashmap::DashMap<String, Arc<std::sync::RwLock<Vec<f64>>>>>,
    /// Enable metrics collection
    enabled: bool,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            counters: Arc::new(dashmap::DashMap::new()),
            gauges: Arc::new(dashmap::DashMap::new()),
            histograms: Arc::new(dashmap::DashMap::new()),
            enabled: true,
        }
    }

    /// Create a disabled metrics collector (no-op)
    pub fn disabled() -> Self {
        Self {
            counters: Arc::new(dashmap::DashMap::new()),
            gauges: Arc::new(dashmap::DashMap::new()),
            histograms: Arc::new(dashmap::DashMap::new()),
            enabled: false,
        }
    }

    /// Get the current value of a counter
    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.get(name).map(|counter| {
            counter.load(Ordering::Relaxed)
        })
    }

    /// Get the current value of a gauge
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.get(name).map(|gauge| {
            *gauge.read().unwrap()
        })
    }

    /// Get histogram statistics
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.histograms.get(name).map(|hist| {
            let values = hist.read().unwrap();
            if values.is_empty() {
                return HistogramStats::default();
            }

            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let count = sorted.len();
            let sum: f64 = sorted.iter().sum();
            let mean = sum / count as f64;

            let p50_idx = count / 2;
            let p95_idx = (count * 95) / 100;
            let p99_idx = (count * 99) / 100;

            HistogramStats {
                count,
                sum,
                mean,
                min: sorted[0],
                max: sorted[count - 1],
                p50: sorted[p50_idx],
                p95: sorted[p95_idx],
                p99: sorted[p99_idx],
            }
        })
    }

    /// Get all counter metrics
    pub fn get_all_counters(&self) -> HashMap<String, u64> {
        self.counters
            .iter()
            .map(|entry| {
                (entry.key().clone(), entry.value().load(Ordering::Relaxed))
            })
            .collect()
    }

    /// Get all gauge metrics
    pub fn get_all_gauges(&self) -> HashMap<String, f64> {
        self.gauges
            .iter()
            .map(|entry| {
                (entry.key().clone(), *entry.value().read().unwrap())
            })
            .collect()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.counters.clear();
        self.gauges.clear();
        self.histograms.clear();
    }

    /// Get total number of tracked metrics
    pub fn metric_count(&self) -> usize {
        self.counters.len() + self.gauges.len() + self.histograms.len()
    }

    /// Generate a metric key with tags
    fn metric_key(name: &str, tags: Option<&[(&str, &str)]>) -> String {
        if let Some(tags) = tags {
            if tags.is_empty() {
                return name.to_string();
            }
            let tag_str: Vec<String> = tags
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!("{}:{}", name, tag_str.join(","))
        } else {
            name.to_string()
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsyncMetricsCollector for MetricsCollector {
    async fn counter(&self, name: &str, value: u64, tags: Option<&[(&str, &str)]>) {
        if !self.enabled {
            return;
        }

        let key = Self::metric_key(name, tags);
        self.counters
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .fetch_add(value, Ordering::Relaxed);
    }

    async fn gauge(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>) {
        if !self.enabled {
            return;
        }

        let key = Self::metric_key(name, tags);
        let gauge = self.gauges
            .entry(key)
            .or_insert_with(|| Arc::new(std::sync::RwLock::new(0.0)));

        *gauge.write().unwrap() = value;
    }

    async fn histogram(&self, name: &str, value: f64, tags: Option<&[(&str, &str)]>) {
        if !self.enabled {
            return;
        }

        let key = Self::metric_key(name, tags);
        let hist = self.histograms
            .entry(key)
            .or_insert_with(|| Arc::new(std::sync::RwLock::new(Vec::new())));

        hist.write().unwrap().push(value);
    }

    async fn timer(&self, name: &str) -> AsyncTimer {
        AsyncTimer::new(name.to_string())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(self.enabled)
    }

    async fn flush(&self) -> Result<()> {
        // In a real implementation, this would flush metrics to an external system
        // (e.g., Prometheus, StatsD, CloudWatch, etc.)
        Ok(())
    }
}

/// Statistics for histogram metrics
#[derive(Debug, Clone, Default)]
pub struct HistogramStats {
    /// Number of observations
    pub count: usize,
    /// Sum of all values
    pub sum: f64,
    /// Mean value
    pub mean: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_counter_metrics() {
        let collector = MetricsCollector::new();

        collector.counter("requests", 1, None).await;
        collector.counter("requests", 5, None).await;
        collector.counter("requests", 3, None).await;

        assert_eq!(collector.get_counter("requests"), Some(9));
    }

    #[tokio::test]
    async fn test_gauge_metrics() {
        let collector = MetricsCollector::new();

        collector.gauge("cpu_usage", 45.5, None).await;
        assert_eq!(collector.get_gauge("cpu_usage"), Some(45.5));

        collector.gauge("cpu_usage", 78.2, None).await;
        assert_eq!(collector.get_gauge("cpu_usage"), Some(78.2));
    }

    #[tokio::test]
    async fn test_histogram_metrics() {
        let collector = MetricsCollector::new();

        collector.histogram("latency", 100.0, None).await;
        collector.histogram("latency", 200.0, None).await;
        collector.histogram("latency", 150.0, None).await;
        collector.histogram("latency", 300.0, None).await;

        let stats = collector.get_histogram_stats("latency").unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.mean, 187.5);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 300.0);
    }

    #[tokio::test]
    async fn test_metrics_with_tags() {
        let collector = MetricsCollector::new();

        let tags = vec![("method", "POST"), ("endpoint", "/api/query")];
        collector.counter("requests", 1, Some(&tags)).await;
        collector.counter("requests", 2, Some(&tags)).await;

        let key = format!("requests:method=POST,endpoint=/api/query");
        assert_eq!(collector.get_counter(&key), Some(3));
    }

    #[tokio::test]
    async fn test_disabled_collector() {
        let collector = MetricsCollector::disabled();

        collector.counter("requests", 10, None).await;
        assert_eq!(collector.get_counter("requests"), None);
        assert!(!collector.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_timer() {
        let collector = MetricsCollector::new();
        let timer = collector.timer("operation").await;

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let elapsed = timer.finish().await;
        assert!(elapsed.as_millis() >= 50);
    }

    #[tokio::test]
    async fn test_reset_metrics() {
        let collector = MetricsCollector::new();

        collector.counter("requests", 10, None).await;
        collector.gauge("cpu", 50.0, None).await;

        assert_eq!(collector.metric_count(), 2);

        collector.reset();
        assert_eq!(collector.metric_count(), 0);
    }
}
