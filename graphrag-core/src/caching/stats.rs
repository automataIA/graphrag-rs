//! Cache statistics and monitoring capabilities

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Comprehensive cache statistics
#[derive(Debug)]
pub struct CacheStatistics {
    /// Total number of cache requests (hits + misses)
    pub total_requests: AtomicU64,
    /// Number of cache hits (successful retrievals)
    pub cache_hits: AtomicU64,
    /// Number of cache misses (unsuccessful retrievals)
    pub cache_misses: AtomicU64,
    /// Total time saved by cache hits in microseconds
    pub time_saved_us: AtomicU64,
    /// Number of new entries inserted into cache
    pub insertions: AtomicU64,
    /// Number of entries evicted from cache due to capacity or TTL
    pub evictions: AtomicU64,
    /// Number of cache updates (refresh of existing entries)
    pub updates: AtomicU64,
    /// Current number of entries in the cache
    pub current_size: AtomicUsize,
    /// Total bytes stored in cache (approximate)
    pub total_bytes: AtomicUsize,
    /// Number of failed cache operations
    pub errors: AtomicU64,
    /// Cache start time for uptime calculation
    start_time: Instant,
}

impl CacheStatistics {
    /// Create new statistics instance
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            time_saved_us: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            updates: AtomicU64::new(0),
            current_size: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            errors: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a cache hit with time saved
    pub fn record_hit(&self, time_saved: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.time_saved_us.fetch_add(time_saved.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_miss(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache insertion
    pub fn record_insertion(&self, entry_size: usize) {
        self.insertions.fetch_add(1, Ordering::Relaxed);
        self.current_size.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(entry_size, Ordering::Relaxed);
    }

    /// Record a cache eviction
    pub fn record_eviction(&self, entry_size: usize) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.current_size.fetch_sub(1, Ordering::Relaxed);
        self.total_bytes.fetch_sub(entry_size, Ordering::Relaxed);
    }

    /// Record a cache update
    pub fn record_update(&self, old_size: usize, new_size: usize) {
        self.updates.fetch_add(1, Ordering::Relaxed);
        if new_size > old_size {
            self.total_bytes.fetch_add(new_size - old_size, Ordering::Relaxed);
        } else {
            self.total_bytes.fetch_sub(old_size - new_size, Ordering::Relaxed);
        }
    }

    /// Record an error
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get cache hit rate as a percentage (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let hits = self.cache_hits.load(Ordering::Relaxed);
        hits as f64 / total as f64
    }

    /// Get cache miss rate as a percentage (0.0 to 1.0)
    pub fn miss_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        1.0 - self.hit_rate()
    }

    /// Get total time saved by cache in seconds
    pub fn total_time_saved(&self) -> Duration {
        Duration::from_micros(self.time_saved_us.load(Ordering::Relaxed))
    }

    /// Get average time saved per hit
    pub fn avg_time_saved_per_hit(&self) -> Duration {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        if hits == 0 {
            return Duration::ZERO;
        }
        let total_us = self.time_saved_us.load(Ordering::Relaxed);
        Duration::from_micros(total_us / hits)
    }

    /// Get cache uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Get memory usage in human-readable format
    pub fn memory_usage_human(&self) -> String {
        let bytes = self.memory_usage_bytes();
        if bytes < 1024 {
            format!("{bytes} B")
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Get current cache utilization (0.0 to 1.0)
    pub fn utilization(&self, max_capacity: u64) -> f64 {
        let current = self.current_size.load(Ordering::Relaxed) as u64;
        if max_capacity == 0 {
            return 0.0;
        }
        (current as f64 / max_capacity as f64).min(1.0)
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.time_saved_us.store(0, Ordering::Relaxed);
        self.insertions.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.updates.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
        // Note: We don't reset current_size and total_bytes as they represent current state
    }

    /// Export statistics as a snapshot
    pub fn snapshot(&self) -> CacheMetrics {
        CacheMetrics {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            hit_rate: self.hit_rate(),
            miss_rate: self.miss_rate(),
            total_time_saved: self.total_time_saved(),
            avg_time_saved_per_hit: self.avg_time_saved_per_hit(),
            insertions: self.insertions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            updates: self.updates.load(Ordering::Relaxed),
            current_size: self.current_size.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_usage_bytes(),
            memory_usage_human: self.memory_usage_human(),
            errors: self.errors.load(Ordering::Relaxed),
            uptime: self.uptime(),
        }
    }

    /// Print detailed statistics
    pub fn print(&self) {
        let metrics = self.snapshot();
        tracing::info!(
            total_requests = metrics.total_requests,
            cache_hits = metrics.cache_hits,
            cache_misses = metrics.cache_misses,
            hit_rate = format!("{:.2}%", metrics.hit_rate * 100.0),
            miss_rate = format!("{:.2}%", metrics.miss_rate * 100.0),
            total_time_saved = format!("{:.2}s", metrics.total_time_saved.as_secs_f64()),
            avg_time_saved_per_hit = format!("{:.2}ms", metrics.avg_time_saved_per_hit.as_secs_f64() * 1000.0),
            insertions = metrics.insertions,
            evictions = metrics.evictions,
            updates = metrics.updates,
            current_size = metrics.current_size,
            memory_usage = %metrics.memory_usage_human,
            errors = metrics.errors,
            uptime = format!("{:.1}s", metrics.uptime.as_secs_f64()),
            "Cache statistics"
        );
    }
}

impl Default for CacheStatistics {
    /// Returns a new cache statistics instance with all counters at zero
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of cache metrics at a point in time
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheMetrics {
    /// Total number of cache requests (hits + misses)
    pub total_requests: u64,
    /// Number of cache hits (successful retrievals)
    pub cache_hits: u64,
    /// Number of cache misses (unsuccessful retrievals)
    pub cache_misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Cache miss rate (0.0 to 1.0)
    pub miss_rate: f64,
    /// Total time saved by cache hits
    pub total_time_saved: Duration,
    /// Average time saved per cache hit
    pub avg_time_saved_per_hit: Duration,
    /// Number of new entries inserted into cache
    pub insertions: u64,
    /// Number of entries evicted from cache
    pub evictions: u64,
    /// Number of cache updates (refresh of existing entries)
    pub updates: u64,
    /// Current number of entries in the cache
    pub current_size: usize,
    /// Total memory used by cache in bytes
    pub memory_usage_bytes: usize,
    /// Human-readable memory usage (e.g., "1.5 MB")
    pub memory_usage_human: String,
    /// Number of failed cache operations
    pub errors: u64,
    /// Duration since cache was created
    pub uptime: Duration,
}

impl CacheMetrics {
    /// Calculate cost savings based on average LLM call cost
    pub fn cost_savings(&self, cost_per_call: f64) -> f64 {
        self.cache_hits as f64 * cost_per_call
    }

    /// Calculate performance improvement factor
    pub fn performance_improvement(&self, avg_llm_latency: Duration) -> f64 {
        if self.avg_time_saved_per_hit.is_zero() {
            return 1.0;
        }
        avg_llm_latency.as_secs_f64() / (avg_llm_latency.as_secs_f64() - self.avg_time_saved_per_hit.as_secs_f64()).max(0.001)
    }

    /// Get efficiency score (0.0 to 1.0)
    pub fn efficiency_score(&self) -> f64 {
        let hit_rate_weight = 0.4;
        let memory_efficiency_weight = 0.3;
        let error_rate_weight = 0.3;

        let hit_score = self.hit_rate;

        let memory_score = if self.current_size == 0 {
            1.0
        } else {
            // Lower memory per entry is better
            let avg_memory_per_entry = self.memory_usage_bytes as f64 / self.current_size as f64;
            (1.0 / (1.0 + avg_memory_per_entry / 1024.0)).min(1.0) // Normalize around 1KB
        };

        let error_rate = if self.total_requests == 0 {
            0.0
        } else {
            self.errors as f64 / self.total_requests as f64
        };
        let error_score = (1.0 - error_rate).max(0.0);

        hit_score * hit_rate_weight + memory_score * memory_efficiency_weight + error_score * error_rate_weight
    }
}

/// Cache health monitoring
#[derive(Debug, Clone)]
pub struct CacheHealth {
    /// Overall health status of the cache
    pub status: HealthStatus,
    /// Current cache metrics snapshot
    pub metrics: CacheMetrics,
    /// List of health alerts based on thresholds
    pub alerts: Vec<HealthAlert>,
    /// Recommendations for improving cache performance
    pub recommendations: Vec<String>,
}

/// Cache health status
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    /// Cache is operating within normal parameters
    Healthy,
    /// Cache metrics exceed warning thresholds
    Warning,
    /// Cache metrics exceed critical thresholds
    Critical,
    /// Cache health cannot be determined
    Unknown,
}

/// Health alert types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthAlert {
    /// Severity level of the alert
    pub level: AlertLevel,
    /// Human-readable alert message
    pub message: String,
    /// Name of the metric that triggered the alert
    pub metric: String,
    /// Threshold value that was exceeded
    pub threshold: f64,
    /// Current value of the metric
    pub current_value: f64,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AlertLevel {
    /// Informational alert, no action required
    Info,
    /// Warning alert, attention recommended
    Warning,
    /// Critical alert, immediate action required
    Critical,
}

impl CacheHealth {
    /// Evaluate cache health based on metrics
    pub fn evaluate(metrics: CacheMetrics, max_capacity: u64) -> Self {
        let mut alerts = Vec::new();
        let mut recommendations = Vec::new();
        let mut status = HealthStatus::Healthy;

        // Check hit rate
        if metrics.hit_rate < 0.5 {
            alerts.push(HealthAlert {
                level: if metrics.hit_rate < 0.2 { AlertLevel::Critical } else { AlertLevel::Warning },
                message: "Low cache hit rate".to_string(),
                metric: "hit_rate".to_string(),
                threshold: 0.5,
                current_value: metrics.hit_rate,
            });
            recommendations.push("Consider adjusting cache key generation strategy or increasing cache size".to_string());
            if metrics.hit_rate < 0.2 {
                status = HealthStatus::Critical;
            } else if status == HealthStatus::Healthy {
                status = HealthStatus::Warning;
            } else {
                // Status is already Warning or Critical, keep it
            }
        }

        // Check error rate
        let error_rate = if metrics.total_requests == 0 {
            0.0
        } else {
            metrics.errors as f64 / metrics.total_requests as f64
        };

        if error_rate > 0.05 {
            alerts.push(HealthAlert {
                level: if error_rate > 0.2 { AlertLevel::Critical } else { AlertLevel::Warning },
                message: "High error rate".to_string(),
                metric: "error_rate".to_string(),
                threshold: 0.05,
                current_value: error_rate,
            });
            recommendations.push("Investigate cache errors and consider reducing cache size or entry limits".to_string());
            if error_rate > 0.2 {
                status = HealthStatus::Critical;
            } else if status == HealthStatus::Healthy {
                status = HealthStatus::Warning;
            } else {
                // Status is already Warning or Critical, keep it
            }
        }

        // Check memory usage
        let utilization = metrics.current_size as f64 / max_capacity as f64;
        if utilization > 0.9 {
            alerts.push(HealthAlert {
                level: if utilization > 0.95 { AlertLevel::Critical } else { AlertLevel::Warning },
                message: "High cache utilization".to_string(),
                metric: "utilization".to_string(),
                threshold: 0.9,
                current_value: utilization,
            });
            recommendations.push("Consider increasing cache capacity or reducing TTL".to_string());
            if utilization > 0.95 {
                status = HealthStatus::Critical;
            } else if status == HealthStatus::Healthy {
                status = HealthStatus::Warning;
            } else {
                // Status is already Warning or Critical, keep it
            }
        }

        // Check eviction rate
        if metrics.total_requests > 0 {
            let eviction_rate = metrics.evictions as f64 / metrics.total_requests as f64;
            if eviction_rate > 0.3 {
                alerts.push(HealthAlert {
                    level: AlertLevel::Warning,
                    message: "High eviction rate".to_string(),
                    metric: "eviction_rate".to_string(),
                    threshold: 0.3,
                    current_value: eviction_rate,
                });
                recommendations.push("Cache is evicting entries frequently; consider increasing capacity".to_string());
                if status == HealthStatus::Healthy {
                    status = HealthStatus::Warning;
                }
            }
        }

        // Add positive recommendations
        if metrics.hit_rate > 0.8 && error_rate < 0.01 {
            recommendations.push("Cache is performing well! Consider increasing capacity for even better performance".to_string());
        }

        if alerts.is_empty() && recommendations.is_empty() {
            recommendations.push("Cache is operating optimally".to_string());
        }

        Self {
            status,
            metrics,
            alerts,
            recommendations,
        }
    }

    /// Print health report
    pub fn print(&self) {
        tracing::info!(
            status = ?self.status,
            efficiency_score = format!("{:.2}", self.metrics.efficiency_score()),
            "Cache health report"
        );

        if !self.alerts.is_empty() {
            for alert in &self.alerts {
                tracing::warn!(
                    level = ?alert.level,
                    message = %alert.message,
                    current_value = format!("{:.3}", alert.current_value),
                    threshold = format!("{:.3}", alert.threshold),
                    "Cache alert"
                );
            }
        }

        if !self.recommendations.is_empty() {
            for rec in &self.recommendations {
                tracing::info!(recommendation = %rec, "Cache recommendation");
            }
        }
    }
}

/// Thread-safe wrapper for cache statistics
pub type SharedCacheStatistics = Arc<CacheStatistics>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_statistics() {
        let stats = CacheStatistics::new();

        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.miss_rate(), 0.0);

        stats.record_hit(Duration::from_millis(100));
        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.hit_rate(), 1.0);

        stats.record_miss();
        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 2);
        assert_eq!(stats.hit_rate(), 0.5);
        assert_eq!(stats.miss_rate(), 0.5);
    }

    #[test]
    fn test_memory_usage_human() {
        let stats = CacheStatistics::new();
        stats.total_bytes.store(512, Ordering::Relaxed);
        assert_eq!(stats.memory_usage_human(), "512 B");

        stats.total_bytes.store(1536, Ordering::Relaxed); // 1.5 KB
        assert_eq!(stats.memory_usage_human(), "1.5 KB");

        stats.total_bytes.store(2 * 1024 * 1024, Ordering::Relaxed); // 2 MB
        assert_eq!(stats.memory_usage_human(), "2.0 MB");
    }

    #[test]
    fn test_cache_metrics() {
        let stats = CacheStatistics::new();
        stats.record_hit(Duration::from_millis(50));
        stats.record_hit(Duration::from_millis(100));
        stats.record_miss();

        let metrics = stats.snapshot();
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.cache_hits, 2);
        assert_eq!(metrics.cache_misses, 1);
        assert!((metrics.hit_rate - 0.666).abs() < 0.01);

        // Test cost savings calculation
        let savings = metrics.cost_savings(0.01); // $0.01 per call
        assert_eq!(savings, 0.02); // 2 hits * $0.01

        // Test efficiency score
        let score = metrics.efficiency_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_cache_health() {
        let metrics = CacheMetrics {
            total_requests: 100,
            cache_hits: 15, // Very low hit rate (15%) - should be Critical
            cache_misses: 85,
            hit_rate: 0.15,
            miss_rate: 0.85,
            total_time_saved: Duration::from_secs(10),
            avg_time_saved_per_hit: Duration::from_millis(500),
            insertions: 50,
            evictions: 30,
            updates: 5,
            current_size: 950, // High utilization
            memory_usage_bytes: 1024 * 1024,
            memory_usage_human: "1.0 MB".to_string(),
            errors: 25, // High error rate (25%) - should be Critical
            uptime: Duration::from_secs(3600),
        };

        let health = CacheHealth::evaluate(metrics, 1000);
        assert_eq!(health.status, HealthStatus::Critical, "Expected Critical with hit_rate=0.15 and error_rate=0.25");
        assert!(!health.alerts.is_empty());
        assert!(!health.recommendations.is_empty());
    }
}