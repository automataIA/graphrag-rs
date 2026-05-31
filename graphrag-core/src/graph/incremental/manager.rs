#![allow(unused_imports)]

use crate::core::{
    DocumentId, Entity, EntityId, GraphRAGError, KnowledgeGraph, Relationship, Result, TextChunk,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[cfg(feature = "incremental")]
use std::sync::Arc;

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::{Mutex, RwLock},
    tokio::sync::{broadcast, Semaphore},
    uuid::Uuid,
};

use super::*;

// ============================================================================
// Main Incremental Graph Manager
// ============================================================================

/// Comprehensive incremental graph manager with production features
#[cfg(feature = "incremental")]
pub struct IncrementalGraphManager {
    graph: Arc<RwLock<KnowledgeGraph>>,
    change_log: DashMap<UpdateId, ChangeRecord>,
    deltas: DashMap<UpdateId, GraphDelta>,
    cache_invalidation: Arc<SelectiveInvalidation>,
    conflict_resolver: Arc<ConflictResolver>,
    monitor: Arc<UpdateMonitor>,
    config: IncrementalConfig,
}

#[cfg(not(feature = "incremental"))]
/// Incremental graph manager (simplified version without incremental feature)
pub struct IncrementalGraphManager {
    graph: KnowledgeGraph,
    change_log: Vec<ChangeRecord>,
    config: IncrementalConfig,
}

/// Configuration for incremental operations
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Maximum number of changes to keep in the log
    pub max_change_log_size: usize,
    /// Maximum number of changes in a single delta
    pub max_delta_size: usize,
    /// Default conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,
    /// Cache invalidation strategy name
    pub cache_invalidation_strategy: String,
    /// Default batch size for batch operations
    pub batch_size: usize,
    /// Maximum number of concurrent operations
    pub max_concurrent_operations: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            max_change_log_size: 10000,
            max_delta_size: 1000,
            conflict_strategy: ConflictStrategy::Merge,
            enable_monitoring: true,
            cache_invalidation_strategy: "selective".to_string(),
            batch_size: 100,
            max_concurrent_operations: 10,
        }
    }
}

#[cfg(feature = "incremental")]
impl IncrementalGraphManager {
    /// Creates a new incremental graph manager with feature-gated capabilities
    pub fn new(graph: KnowledgeGraph, config: IncrementalConfig) -> Self {
        Self {
            graph: Arc::new(RwLock::new(graph)),
            change_log: DashMap::new(),
            deltas: DashMap::new(),
            cache_invalidation: Arc::new(SelectiveInvalidation::new()),
            conflict_resolver: Arc::new(ConflictResolver::new(config.conflict_strategy.clone())),
            monitor: Arc::new(UpdateMonitor::new()),
            config,
        }
    }

    /// Sets a custom conflict resolver for the manager
    pub fn with_conflict_resolver(mut self, resolver: ConflictResolver) -> Self {
        self.conflict_resolver = Arc::new(resolver);
        self
    }

    /// Get a read-only reference to the knowledge graph
    pub fn graph(&self) -> Arc<RwLock<KnowledgeGraph>> {
        Arc::clone(&self.graph)
    }

    /// Get the conflict resolver
    pub fn conflict_resolver(&self) -> Arc<ConflictResolver> {
        Arc::clone(&self.conflict_resolver)
    }

    /// Get the update monitor
    pub fn monitor(&self) -> Arc<UpdateMonitor> {
        Arc::clone(&self.monitor)
    }
}

#[cfg(not(feature = "incremental"))]
impl IncrementalGraphManager {
    /// Creates a new incremental graph manager without advanced features
    pub fn new(graph: KnowledgeGraph, config: IncrementalConfig) -> Self {
        Self {
            graph,
            change_log: Vec::new(),
            config,
        }
    }

    /// Gets a reference to the knowledge graph
    pub fn graph(&self) -> &KnowledgeGraph {
        &self.graph
    }

    /// Gets a mutable reference to the knowledge graph
    pub fn graph_mut(&mut self) -> &mut KnowledgeGraph {
        &mut self.graph
    }
}

// Common implementation for both feature-gated and non-feature-gated versions
impl IncrementalGraphManager {
    /// Create a new change record
    pub fn create_change_record(
        &self,
        change_type: ChangeType,
        operation: Operation,
        change_data: ChangeData,
        entity_id: Option<EntityId>,
        document_id: Option<DocumentId>,
    ) -> ChangeRecord {
        ChangeRecord {
            change_id: UpdateId::new(),
            timestamp: Utc::now(),
            change_type,
            entity_id,
            document_id,
            operation,
            data: change_data,
            metadata: HashMap::new(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &IncrementalConfig {
        &self.config
    }

    /// Basic entity upsert (works without incremental feature)
    pub fn basic_upsert_entity(&mut self, entity: Entity) -> Result<UpdateId> {
        let update_id = UpdateId::new();

        #[cfg(feature = "incremental")]
        {
            let operation_id = self.monitor.start_operation("upsert_entity");
            let mut graph = self.graph.write();

            match graph.add_entity(entity.clone()) {
                Ok(_) => {
                    let ent_id = entity.id.clone();
                    let change = self.create_change_record(
                        ChangeType::EntityAdded,
                        Operation::Upsert,
                        ChangeData::Entity(entity),
                        Some(ent_id),
                        None,
                    );
                    self.change_log.insert(change.change_id.clone(), change);
                    self.monitor
                        .complete_operation(&operation_id, true, None, 1, 0);
                    Ok(update_id)
                },
                Err(e) => {
                    self.monitor.complete_operation(
                        &operation_id,
                        false,
                        Some(e.to_string()),
                        0,
                        0,
                    );
                    Err(e)
                },
            }
        }

        #[cfg(not(feature = "incremental"))]
        {
            self.graph.add_entity(entity.clone())?;
            // Capture ID before moving `entity` into ChangeData
            let ent_id = entity.id.clone();
            let change = self.create_change_record(
                ChangeType::EntityAdded,
                Operation::Upsert,
                ChangeData::Entity(entity),
                Some(ent_id),
                None,
            );
            self.change_log.push(change);
            Ok(update_id)
        }
    }
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

/// Comprehensive statistics for incremental operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalStatistics {
    /// Total number of update operations
    pub total_updates: usize,
    /// Number of successful updates
    pub successful_updates: usize,
    /// Number of failed updates
    pub failed_updates: usize,
    /// Number of entities added
    pub entities_added: usize,
    /// Number of entities updated
    pub entities_updated: usize,
    /// Number of entities removed
    pub entities_removed: usize,
    /// Number of relationships added
    pub relationships_added: usize,
    /// Number of relationships updated
    pub relationships_updated: usize,
    /// Number of relationships removed
    pub relationships_removed: usize,
    /// Number of conflicts resolved
    pub conflicts_resolved: usize,
    /// Number of cache invalidations performed
    pub cache_invalidations: usize,
    /// Average update time in milliseconds
    pub average_update_time_ms: f64,
    /// Peak updates per second achieved
    pub peak_updates_per_second: f64,
    /// Current size of the change log
    pub current_change_log_size: usize,
    /// Current number of active deltas
    pub current_delta_count: usize,
}

impl IncrementalStatistics {
    /// Creates an empty statistics instance
    pub fn empty() -> Self {
        Self {
            total_updates: 0,
            successful_updates: 0,
            failed_updates: 0,
            entities_added: 0,
            entities_updated: 0,
            entities_removed: 0,
            relationships_added: 0,
            relationships_updated: 0,
            relationships_removed: 0,
            conflicts_resolved: 0,
            cache_invalidations: 0,
            average_update_time_ms: 0.0,
            peak_updates_per_second: 0.0,
            current_change_log_size: 0,
            current_delta_count: 0,
        }
    }

    /// Prints statistics to stdout in a formatted way
    pub fn print(&self) {
        println!("🔄 Incremental Updates Statistics");
        println!("  Total updates: {}", self.total_updates);
        println!(
            "  Successful: {} ({:.1}%)",
            self.successful_updates,
            if self.total_updates > 0 {
                (self.successful_updates as f64 / self.total_updates as f64) * 100.0
            } else {
                0.0
            }
        );
        println!("  Failed: {}", self.failed_updates);
        println!(
            "  Entities: +{} ~{} -{}",
            self.entities_added, self.entities_updated, self.entities_removed
        );
        println!(
            "  Relationships: +{} ~{} -{}",
            self.relationships_added, self.relationships_updated, self.relationships_removed
        );
        println!("  Conflicts resolved: {}", self.conflicts_resolved);
        println!("  Cache invalidations: {}", self.cache_invalidations);
        println!("  Avg update time: {:.2}ms", self.average_update_time_ms);
        println!("  Peak updates/sec: {:.1}", self.peak_updates_per_second);
        println!("  Change log size: {}", self.current_change_log_size);
        println!("  Active deltas: {}", self.current_delta_count);
    }
}

#[cfg(feature = "incremental")]
impl IncrementalGraphManager {
    /// Gets comprehensive statistics about incremental operations
    pub fn get_statistics(&self) -> IncrementalStatistics {
        let perf_stats = self.monitor.get_performance_stats();
        let invalidation_stats = self.cache_invalidation.get_invalidation_stats();

        // Calculate entity/relationship statistics from change log
        let mut entity_stats = (0, 0, 0); // added, updated, removed
        let mut relationship_stats = (0, 0, 0);
        let conflicts_resolved = 0;

        for change in self.change_log.iter() {
            match change.value().change_type {
                ChangeType::EntityAdded => entity_stats.0 += 1,
                ChangeType::EntityUpdated => entity_stats.1 += 1,
                ChangeType::EntityRemoved => entity_stats.2 += 1,
                ChangeType::RelationshipAdded => relationship_stats.0 += 1,
                ChangeType::RelationshipUpdated => relationship_stats.1 += 1,
                ChangeType::RelationshipRemoved => relationship_stats.2 += 1,
                _ => {},
            }
        }

        IncrementalStatistics {
            total_updates: perf_stats.total_operations as usize,
            successful_updates: perf_stats.successful_operations as usize,
            failed_updates: perf_stats.failed_operations as usize,
            entities_added: entity_stats.0,
            entities_updated: entity_stats.1,
            entities_removed: entity_stats.2,
            relationships_added: relationship_stats.0,
            relationships_updated: relationship_stats.1,
            relationships_removed: relationship_stats.2,
            conflicts_resolved,
            cache_invalidations: invalidation_stats.total_invalidations,
            average_update_time_ms: perf_stats.average_operation_time.as_millis() as f64,
            peak_updates_per_second: perf_stats.peak_operations_per_second,
            current_change_log_size: self.change_log.len(),
            current_delta_count: self.deltas.len(),
        }
    }
}

#[cfg(not(feature = "incremental"))]
impl IncrementalGraphManager {
    /// Gets basic statistics about incremental operations (non-feature version)
    pub fn get_statistics(&self) -> IncrementalStatistics {
        let mut stats = IncrementalStatistics::empty();
        stats.current_change_log_size = self.change_log.len();

        for change in &self.change_log {
            match change.change_type {
                ChangeType::EntityAdded => stats.entities_added += 1,
                ChangeType::EntityUpdated => stats.entities_updated += 1,
                ChangeType::EntityRemoved => stats.entities_removed += 1,
                ChangeType::RelationshipAdded => stats.relationships_added += 1,
                ChangeType::RelationshipUpdated => stats.relationships_updated += 1,
                ChangeType::RelationshipRemoved => stats.relationships_removed += 1,
                _ => {},
            }
        }

        stats.total_updates = self.change_log.len();
        stats.successful_updates = self.change_log.len(); // Assume all succeeded in basic mode
        stats
    }
}

// ============================================================================
// Incremental PageRank Implementation
// ============================================================================

/// Incremental PageRank calculator for efficient updates
#[cfg(feature = "incremental")]
#[allow(dead_code)]
pub struct IncrementalPageRank {
    pub(super) scores: DashMap<EntityId, f64>,
    adjacency_changes: DashMap<EntityId, Vec<(EntityId, f64)>>, // Node -> [(neighbor, weight)]
    damping_factor: f64,
    tolerance: f64,
    max_iterations: usize,
    last_full_computation: DateTime<Utc>,
    incremental_threshold: usize, // Number of changes before full recomputation
    pending_changes: RwLock<usize>,
}

#[cfg(feature = "incremental")]
impl IncrementalPageRank {
    /// Creates a new incremental PageRank calculator
    pub fn new(damping_factor: f64, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            scores: DashMap::new(),
            adjacency_changes: DashMap::new(),
            damping_factor,
            tolerance,
            max_iterations,
            last_full_computation: Utc::now(),
            incremental_threshold: 1000,
            pending_changes: RwLock::new(0),
        }
    }

    /// Update PageRank incrementally for a specific subgraph
    pub async fn update_incremental(
        &self,
        changed_entities: &[EntityId],
        graph: &KnowledgeGraph,
    ) -> Result<()> {
        let start = Instant::now();

        // If too many changes accumulated, do full recomputation
        {
            let pending = *self.pending_changes.read();
            if pending > self.incremental_threshold {
                return self.full_recomputation(graph).await;
            }
        }

        // Incremental update for changed entities and their neighborhoods
        let mut affected_entities = HashSet::new();

        // Add changed entities and their neighbors (2-hop neighborhood)
        for entity_id in changed_entities {
            affected_entities.insert(entity_id.clone());

            // Add direct neighbors
            for (neighbor, _) in graph.get_neighbors(entity_id) {
                affected_entities.insert(neighbor.id.clone());

                // Add second-hop neighbors
                for (second_hop, _) in graph.get_neighbors(&neighbor.id) {
                    affected_entities.insert(second_hop.id.clone());
                }
            }
        }

        // Perform localized PageRank computation
        self.localized_pagerank(&affected_entities, graph).await?;

        // Reset pending changes counter
        *self.pending_changes.write() = 0;

        let duration = start.elapsed();
        println!(
            "🔄 Incremental PageRank update completed in {:?} for {} entities",
            duration,
            affected_entities.len()
        );

        Ok(())
    }

    /// Perform full PageRank recomputation
    async fn full_recomputation(&self, graph: &KnowledgeGraph) -> Result<()> {
        let start = Instant::now();

        // Build adjacency matrix
        let entities: Vec<EntityId> = graph.entities().map(|e| e.id.clone()).collect();
        let n = entities.len();

        if n == 0 {
            return Ok(());
        }

        // Initialize scores
        let initial_score = 1.0 / n as f64;
        for entity_id in &entities {
            self.scores.insert(entity_id.clone(), initial_score);
        }

        // Power iteration
        for iteration in 0..self.max_iterations {
            let mut new_scores = HashMap::new();
            let mut max_diff: f64 = 0.0;

            for entity_id in &entities {
                let mut score = (1.0 - self.damping_factor) / n as f64;

                // Sum contributions from incoming links
                for other_entity in &entities {
                    if let Some(weight) = self.get_edge_weight(other_entity, entity_id, graph) {
                        let other_score = self
                            .scores
                            .get(other_entity)
                            .map(|s| *s.value())
                            .unwrap_or(initial_score);
                        let out_degree = self.get_out_degree(other_entity, graph);

                        if out_degree > 0.0 {
                            score += self.damping_factor * other_score * weight / out_degree;
                        }
                    }
                }

                let old_score = self
                    .scores
                    .get(entity_id)
                    .map(|s| *s.value())
                    .unwrap_or(initial_score);
                let diff = (score - old_score).abs();
                max_diff = max_diff.max(diff);

                new_scores.insert(entity_id.clone(), score);
            }

            // Update scores
            for (entity_id, score) in new_scores {
                self.scores.insert(entity_id, score);
            }

            // Check convergence
            if max_diff < self.tolerance {
                println!(
                    "🎯 PageRank converged after {} iterations (diff: {:.6})",
                    iteration + 1,
                    max_diff
                );
                break;
            }
        }

        let duration = start.elapsed();
        println!("🔄 Full PageRank recomputation completed in {duration:?} for {n} entities");

        Ok(())
    }

    /// Perform localized PageRank computation for a subset of entities
    async fn localized_pagerank(
        &self,
        entities: &HashSet<EntityId>,
        graph: &KnowledgeGraph,
    ) -> Result<()> {
        let entity_vec: Vec<EntityId> = entities.iter().cloned().collect();
        let n = entity_vec.len();

        if n == 0 {
            return Ok(());
        }

        // Localized power iteration
        for _iteration in 0..self.max_iterations {
            let mut max_diff: f64 = 0.0;

            for entity_id in &entity_vec {
                let mut score = (1.0 - self.damping_factor) / n as f64;

                // Only consider links within the subset for localized computation
                for other_entity in &entity_vec {
                    if let Some(weight) = self.get_edge_weight(other_entity, entity_id, graph) {
                        let other_score = self
                            .scores
                            .get(other_entity)
                            .map(|s| *s.value())
                            .unwrap_or(1.0 / n as f64);
                        let out_degree =
                            self.get_localized_out_degree(other_entity, entities, graph);

                        if out_degree > 0.0 {
                            score += self.damping_factor * other_score * weight / out_degree;
                        }
                    }
                }

                let old_score = self
                    .scores
                    .get(entity_id)
                    .map(|s| *s.value())
                    .unwrap_or(1.0 / n as f64);
                let diff = (score - old_score).abs();
                max_diff = max_diff.max(diff);

                self.scores.insert(entity_id.clone(), score);
            }

            // Check convergence
            if max_diff < self.tolerance {
                break;
            }
        }

        Ok(())
    }

    fn get_edge_weight(
        &self,
        from: &EntityId,
        to: &EntityId,
        graph: &KnowledgeGraph,
    ) -> Option<f64> {
        // Check if there's a relationship between entities
        for (neighbor, relationship) in graph.get_neighbors(from) {
            if neighbor.id == *to {
                return Some(relationship.confidence as f64);
            }
        }
        None
    }

    fn get_out_degree(&self, entity_id: &EntityId, graph: &KnowledgeGraph) -> f64 {
        graph
            .get_neighbors(entity_id)
            .iter()
            .map(|(_, rel)| rel.confidence as f64)
            .sum()
    }

    fn get_localized_out_degree(
        &self,
        entity_id: &EntityId,
        subset: &HashSet<EntityId>,
        graph: &KnowledgeGraph,
    ) -> f64 {
        graph
            .get_neighbors(entity_id)
            .iter()
            .filter(|(neighbor, _)| subset.contains(&neighbor.id))
            .map(|(_, rel)| rel.confidence as f64)
            .sum()
    }

    /// Get PageRank score for an entity
    pub fn get_score(&self, entity_id: &EntityId) -> Option<f64> {
        self.scores.get(entity_id).map(|s| *s.value())
    }

    /// Get top-k entities by PageRank score
    pub fn get_top_entities(&self, k: usize) -> Vec<(EntityId, f64)> {
        let mut entities: Vec<(EntityId, f64)> = self
            .scores
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entities.truncate(k);
        entities
    }

    /// Record a graph change for incremental updates
    pub fn record_change(&self, _entity_id: EntityId) {
        *self.pending_changes.write() += 1;
    }
}

// ============================================================================
// Batch Processing System
// ============================================================================

/// High-throughput batch processor for incremental updates
#[cfg(feature = "incremental")]
pub struct BatchProcessor {
    batch_size: usize,
    max_wait_time: Duration,
    pending_batches: DashMap<String, PendingBatch>,
    processing_semaphore: Semaphore,
    metrics: RwLock<BatchMetrics>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PendingBatch {
    changes: Vec<ChangeRecord>,
    created_at: Instant,
    batch_id: String,
}

/// Batch metrics for monitoring
#[derive(Debug, Clone)]
pub struct BatchMetrics {
    /// Total number of batches processed
    pub total_batches_processed: u64,
    /// Total number of changes processed across all batches
    pub total_changes_processed: u64,
    /// Average size of batches
    pub average_batch_size: f64,
    /// Average time to process a batch
    pub average_processing_time: Duration,
    /// Throughput in changes per second
    pub throughput_per_second: f64,
    /// Timestamp of last batch processed
    pub last_batch_processed: Option<DateTime<Utc>>,
}

#[cfg(feature = "incremental")]
impl BatchProcessor {
    /// Creates a new batch processor with specified configuration
    pub fn new(batch_size: usize, max_wait_time: Duration, max_concurrent_batches: usize) -> Self {
        Self {
            batch_size,
            max_wait_time,
            pending_batches: DashMap::new(),
            processing_semaphore: Semaphore::new(max_concurrent_batches),
            metrics: RwLock::new(BatchMetrics {
                total_batches_processed: 0,
                total_changes_processed: 0,
                average_batch_size: 0.0,
                average_processing_time: Duration::from_millis(0),
                throughput_per_second: 0.0,
                last_batch_processed: None,
            }),
        }
    }

    /// Adds a change to be processed in batches
    pub async fn add_change(&self, change: ChangeRecord) -> Result<String> {
        let batch_key = self.get_batch_key(&change);

        let batch_id = {
            let mut entry = self
                .pending_batches
                .entry(batch_key.clone())
                .or_insert_with(|| PendingBatch {
                    changes: Vec::new(),
                    created_at: Instant::now(),
                    batch_id: format!("batch_{}", Uuid::new_v4()),
                });

            entry.changes.push(change);
            let should_process = entry.changes.len() >= self.batch_size
                || entry.created_at.elapsed() > self.max_wait_time;

            let batch_id = entry.batch_id.clone();

            if should_process {
                // Move batch out for processing
                let batch = entry.clone();
                self.pending_batches.remove(&batch_key);

                // Process batch asynchronously
                let processor = Arc::new(self.clone());
                tokio::spawn(async move {
                    if let Err(e) = processor.process_batch(batch).await {
                        eprintln!("Batch processing error: {e}");
                    }
                });
            }

            batch_id
        };

        Ok(batch_id)
    }

    async fn process_batch(&self, batch: PendingBatch) -> Result<()> {
        let _permit = self.processing_semaphore.acquire().await.map_err(|_| {
            GraphRAGError::IncrementalUpdate {
                message: "Failed to acquire processing permit".to_string(),
            }
        })?;

        let start = Instant::now();

        // Group changes by type for optimized processing
        let mut entity_changes = Vec::new();
        let mut relationship_changes = Vec::new();
        let mut embedding_changes = Vec::new();

        for change in &batch.changes {
            match &change.change_type {
                ChangeType::EntityAdded | ChangeType::EntityUpdated | ChangeType::EntityRemoved => {
                    entity_changes.push(change);
                },
                ChangeType::RelationshipAdded
                | ChangeType::RelationshipUpdated
                | ChangeType::RelationshipRemoved => {
                    relationship_changes.push(change);
                },
                ChangeType::EmbeddingAdded
                | ChangeType::EmbeddingUpdated
                | ChangeType::EmbeddingRemoved => {
                    embedding_changes.push(change);
                },
                _ => {},
            }
        }

        // Process each type of change optimally
        self.process_entity_changes(&entity_changes).await?;
        self.process_relationship_changes(&relationship_changes)
            .await?;
        self.process_embedding_changes(&embedding_changes).await?;

        let processing_time = start.elapsed();

        // Update metrics
        self.update_metrics(&batch, processing_time).await;

        println!(
            "🚀 Processed batch {} with {} changes in {:?}",
            batch.batch_id,
            batch.changes.len(),
            processing_time
        );

        Ok(())
    }

    async fn process_entity_changes(&self, _changes: &[&ChangeRecord]) -> Result<()> {
        // Implementation would go here - process entity changes efficiently
        Ok(())
    }

    async fn process_relationship_changes(&self, _changes: &[&ChangeRecord]) -> Result<()> {
        // Implementation would go here - process relationship changes efficiently
        Ok(())
    }

    async fn process_embedding_changes(&self, _changes: &[&ChangeRecord]) -> Result<()> {
        // Implementation would go here - process embedding changes efficiently
        Ok(())
    }

    fn get_batch_key(&self, change: &ChangeRecord) -> String {
        // Group changes by entity or document for batching efficiency
        match (&change.entity_id, &change.document_id) {
            (Some(entity_id), _) => format!("entity:{entity_id}"),
            (None, Some(doc_id)) => format!("document:{doc_id}"),
            _ => "global".to_string(),
        }
    }

    async fn update_metrics(&self, batch: &PendingBatch, processing_time: Duration) {
        let mut metrics = self.metrics.write();

        metrics.total_batches_processed += 1;
        metrics.total_changes_processed += batch.changes.len() as u64;

        // Update running averages
        let total_batches = metrics.total_batches_processed as f64;
        metrics.average_batch_size = (metrics.average_batch_size * (total_batches - 1.0)
            + batch.changes.len() as f64)
            / total_batches;

        let prev_avg_ms = metrics.average_processing_time.as_millis() as f64;
        let new_avg_ms = (prev_avg_ms * (total_batches - 1.0) + processing_time.as_millis() as f64)
            / total_batches;
        metrics.average_processing_time = Duration::from_millis(new_avg_ms as u64);

        // Calculate throughput
        if processing_time.as_secs_f64() > 0.0 {
            metrics.throughput_per_second =
                batch.changes.len() as f64 / processing_time.as_secs_f64();
        }

        metrics.last_batch_processed = Some(Utc::now());
    }

    /// Gets the current batch processing metrics
    pub fn get_metrics(&self) -> BatchMetrics {
        self.metrics.read().clone()
    }
}

// Clone impl for BatchProcessor (required for Arc usage)
#[cfg(feature = "incremental")]
impl Clone for BatchProcessor {
    fn clone(&self) -> Self {
        Self {
            batch_size: self.batch_size,
            max_wait_time: self.max_wait_time,
            pending_batches: DashMap::new(), // New instance starts empty
            processing_semaphore: Semaphore::new(self.processing_semaphore.available_permits()),
            metrics: RwLock::new(self.get_metrics()),
        }
    }
}

// ============================================================================
// Error Extensions
// ============================================================================

impl GraphRAGError {
    /// Creates a conflict resolution error
    pub fn conflict_resolution(message: String) -> Self {
        GraphRAGError::GraphConstruction { message }
    }

    /// Creates an incremental update error
    pub fn incremental_update(message: String) -> Self {
        GraphRAGError::GraphConstruction { message }
    }
}
