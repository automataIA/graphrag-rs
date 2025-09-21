use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct ProcessingMetrics {
    // Query metrics
    queries_started: AtomicUsize,
    queries_succeeded: AtomicUsize,
    queries_failed: AtomicUsize,
    query_durations: Arc<RwLock<Vec<Duration>>>,

    // Document processing metrics
    documents_started: AtomicUsize,
    documents_succeeded: AtomicUsize,
    documents_failed: AtomicUsize,
    document_durations: Arc<RwLock<Vec<Duration>>>,

    // Batch processing metrics
    batches_started: AtomicUsize,
    batch_durations: Arc<RwLock<Vec<Duration>>>,

    // Rate limiting metrics
    rate_limit_errors: AtomicUsize,

    // System metrics
    peak_memory_usage: AtomicU64,
    creation_time: Instant,
}

impl ProcessingMetrics {
    pub fn new() -> Self {
        Self {
            queries_started: AtomicUsize::new(0),
            queries_succeeded: AtomicUsize::new(0),
            queries_failed: AtomicUsize::new(0),
            query_durations: Arc::new(RwLock::new(Vec::new())),

            documents_started: AtomicUsize::new(0),
            documents_succeeded: AtomicUsize::new(0),
            documents_failed: AtomicUsize::new(0),
            document_durations: Arc::new(RwLock::new(Vec::new())),

            batches_started: AtomicUsize::new(0),
            batch_durations: Arc::new(RwLock::new(Vec::new())),

            rate_limit_errors: AtomicUsize::new(0),

            peak_memory_usage: AtomicU64::new(0),
            creation_time: Instant::now(),
        }
    }

    // Query metrics
    pub fn increment_query_started(&self) {
        self.queries_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_query_success(&self) {
        self.queries_succeeded.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_query_error(&self) {
        self.queries_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_query_duration(&self, duration: Duration) {
        let mut durations = self.query_durations.write();
        durations.push(duration);
        // Keep only last 1000 measurements to prevent memory growth
        if durations.len() > 1000 {
            durations.remove(0);
        }
    }

    // Document processing metrics
    pub fn increment_document_processing_started(&self) {
        self.documents_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_document_processing_success(&self) {
        self.documents_succeeded.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_document_processing_error(&self) {
        self.documents_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_document_processing_duration(&self, duration: Duration) {
        let mut durations = self.document_durations.write();
        durations.push(duration);
        // Keep only last 1000 measurements to prevent memory growth
        if durations.len() > 1000 {
            durations.remove(0);
        }
    }

    // Batch processing metrics
    pub fn increment_batch_processing_started(&self) {
        self.batches_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_batch_processing_duration(&self, duration: Duration) {
        let mut durations = self.batch_durations.write();
        durations.push(duration);
        // Keep only last 100 measurements to prevent memory growth
        if durations.len() > 100 {
            durations.remove(0);
        }
    }

    // Rate limiting metrics
    pub fn increment_rate_limit_errors(&self) {
        self.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
    }

    // System metrics
    pub fn update_peak_memory_usage(&self, memory_bytes: u64) {
        let current = self.peak_memory_usage.load(Ordering::Relaxed);
        if memory_bytes > current {
            self.peak_memory_usage
                .store(memory_bytes, Ordering::Relaxed);
        }
    }

    // Getters for current values
    pub fn get_queries_started(&self) -> usize {
        self.queries_started.load(Ordering::Relaxed)
    }

    pub fn get_queries_succeeded(&self) -> usize {
        self.queries_succeeded.load(Ordering::Relaxed)
    }

    pub fn get_queries_failed(&self) -> usize {
        self.queries_failed.load(Ordering::Relaxed)
    }

    pub fn get_documents_started(&self) -> usize {
        self.documents_started.load(Ordering::Relaxed)
    }

    pub fn get_documents_succeeded(&self) -> usize {
        self.documents_succeeded.load(Ordering::Relaxed)
    }

    pub fn get_documents_failed(&self) -> usize {
        self.documents_failed.load(Ordering::Relaxed)
    }

    pub fn get_batches_started(&self) -> usize {
        self.batches_started.load(Ordering::Relaxed)
    }

    pub fn get_rate_limit_errors(&self) -> usize {
        self.rate_limit_errors.load(Ordering::Relaxed)
    }

    pub fn get_peak_memory_usage(&self) -> u64 {
        self.peak_memory_usage.load(Ordering::Relaxed)
    }

    pub fn get_uptime(&self) -> Duration {
        self.creation_time.elapsed()
    }

    // Statistical methods
    pub fn get_average_query_duration(&self) -> Option<Duration> {
        let durations = self.query_durations.read();
        if durations.is_empty() {
            None
        } else {
            let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
            Some(Duration::from_nanos(total_nanos / durations.len() as u64))
        }
    }

    pub fn get_average_document_duration(&self) -> Option<Duration> {
        let durations = self.document_durations.read();
        if durations.is_empty() {
            None
        } else {
            let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
            Some(Duration::from_nanos(total_nanos / durations.len() as u64))
        }
    }

    pub fn get_query_success_rate(&self) -> f64 {
        let total = self.get_queries_started();
        if total == 0 {
            1.0
        } else {
            self.get_queries_succeeded() as f64 / total as f64
        }
    }

    pub fn get_document_success_rate(&self) -> f64 {
        let total = self.get_documents_started();
        if total == 0 {
            1.0
        } else {
            self.get_documents_succeeded() as f64 / total as f64
        }
    }

    // Summary report
    pub fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            queries: QueryMetrics {
                started: self.get_queries_started(),
                succeeded: self.get_queries_succeeded(),
                failed: self.get_queries_failed(),
                success_rate: self.get_query_success_rate(),
                average_duration: self.get_average_query_duration(),
            },
            documents: DocumentMetrics {
                started: self.get_documents_started(),
                succeeded: self.get_documents_succeeded(),
                failed: self.get_documents_failed(),
                success_rate: self.get_document_success_rate(),
                average_duration: self.get_average_document_duration(),
            },
            system: SystemMetrics {
                batches_processed: self.get_batches_started(),
                rate_limit_errors: self.get_rate_limit_errors(),
                peak_memory_usage: self.get_peak_memory_usage(),
                uptime: self.get_uptime(),
            },
        }
    }

    pub fn print_summary(&self) {
        let summary = self.get_summary();
        println!("\n📊 Processing Metrics Summary");
        println!("================================");

        println!("🔍 Queries:");
        println!("  Started: {}", summary.queries.started);
        println!("  Succeeded: {}", summary.queries.succeeded);
        println!("  Failed: {}", summary.queries.failed);
        println!(
            "  Success Rate: {:.1}%",
            summary.queries.success_rate * 100.0
        );
        if let Some(avg_duration) = summary.queries.average_duration {
            println!("  Average Duration: {avg_duration:?}");
        }

        println!("\n📄 Documents:");
        println!("  Started: {}", summary.documents.started);
        println!("  Succeeded: {}", summary.documents.succeeded);
        println!("  Failed: {}", summary.documents.failed);
        println!(
            "  Success Rate: {:.1}%",
            summary.documents.success_rate * 100.0
        );
        if let Some(avg_duration) = summary.documents.average_duration {
            println!("  Average Duration: {avg_duration:?}");
        }

        println!("\n⚙️ System:");
        println!("  Batches Processed: {}", summary.system.batches_processed);
        println!("  Rate Limit Errors: {}", summary.system.rate_limit_errors);
        if summary.system.peak_memory_usage > 0 {
            println!(
                "  Peak Memory: {} MB",
                summary.system.peak_memory_usage / 1024 / 1024
            );
        }
        println!("  Uptime: {:?}", summary.system.uptime);
        println!("================================\n");
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub queries: QueryMetrics,
    pub documents: DocumentMetrics,
    pub system: SystemMetrics,
}

#[derive(Debug, Clone)]
pub struct QueryMetrics {
    pub started: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub success_rate: f64,
    pub average_duration: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct DocumentMetrics {
    pub started: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub success_rate: f64,
    pub average_duration: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub batches_processed: usize,
    pub rate_limit_errors: usize,
    pub peak_memory_usage: u64,
    pub uptime: Duration,
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self::new()
    }
}
