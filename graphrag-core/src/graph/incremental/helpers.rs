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

/// Selective cache invalidation manager
#[cfg(feature = "incremental")]
pub struct SelectiveInvalidation {
    cache_regions: DashMap<String, CacheRegion>,
    entity_to_regions: DashMap<EntityId, HashSet<String>>,
    invalidation_log: Mutex<Vec<(DateTime<Utc>, InvalidationStrategy)>>,
}

#[cfg(feature = "incremental")]
impl Default for SelectiveInvalidation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "incremental")]
impl SelectiveInvalidation {
    /// Creates a new selective invalidation manager
    pub fn new() -> Self {
        Self {
            cache_regions: DashMap::new(),
            entity_to_regions: DashMap::new(),
            invalidation_log: Mutex::new(Vec::new()),
        }
    }

    /// Registers a cache region for invalidation tracking
    pub fn register_cache_region(&self, region: CacheRegion) {
        let region_id = region.region_id.clone();

        // Update entity mappings
        for entity_id in &region.entity_ids {
            self.entity_to_regions
                .entry(entity_id.clone())
                .or_default()
                .insert(region_id.clone());
        }

        self.cache_regions.insert(region_id, region);
    }

    /// Determines invalidation strategies for a set of changes
    pub fn invalidate_for_changes(&self, changes: &[ChangeRecord]) -> Vec<InvalidationStrategy> {
        let mut strategies = Vec::new();
        let mut affected_regions = HashSet::new();

        for change in changes {
            match &change.change_type {
                ChangeType::EntityAdded | ChangeType::EntityUpdated | ChangeType::EntityRemoved => {
                    if let Some(entity_id) = &change.entity_id {
                        if let Some(regions) = self.entity_to_regions.get(entity_id) {
                            affected_regions.extend(regions.clone());
                        }
                        strategies.push(InvalidationStrategy::Relational(entity_id.clone(), 2));
                    }
                },
                ChangeType::RelationshipAdded
                | ChangeType::RelationshipUpdated
                | ChangeType::RelationshipRemoved => {
                    // Invalidate based on relationship endpoints
                    if let ChangeData::Relationship(rel) = &change.data {
                        strategies.push(InvalidationStrategy::Relational(rel.source.clone(), 1));
                        strategies.push(InvalidationStrategy::Relational(rel.target.clone(), 1));
                    }
                },
                _ => {
                    // For other changes, use selective invalidation
                    let cache_keys = self.generate_cache_keys_for_change(change);
                    if !cache_keys.is_empty() {
                        strategies.push(InvalidationStrategy::Selective(cache_keys));
                    }
                },
            }
        }

        // Add regional invalidation for affected regions
        for region_id in affected_regions {
            strategies.push(InvalidationStrategy::Regional(region_id));
        }

        // Log invalidation
        let mut log = self.invalidation_log.lock();
        for strategy in &strategies {
            log.push((Utc::now(), strategy.clone()));
        }

        strategies
    }

    fn generate_cache_keys_for_change(&self, change: &ChangeRecord) -> Vec<String> {
        let mut keys = Vec::new();

        // Generate cache keys based on change type and data
        match &change.change_type {
            ChangeType::EntityAdded | ChangeType::EntityUpdated => {
                if let Some(entity_id) = &change.entity_id {
                    keys.push(format!("entity:{entity_id}"));
                    keys.push(format!("entity_neighbors:{entity_id}"));
                }
            },
            ChangeType::DocumentAdded | ChangeType::DocumentUpdated => {
                if let Some(doc_id) = &change.document_id {
                    keys.push(format!("document:{doc_id}"));
                    keys.push(format!("document_chunks:{doc_id}"));
                }
            },
            ChangeType::EmbeddingAdded | ChangeType::EmbeddingUpdated => {
                if let Some(entity_id) = &change.entity_id {
                    keys.push(format!("embedding:{entity_id}"));
                    keys.push(format!("similarity:{entity_id}"));
                }
            },
            _ => {},
        }

        keys
    }

    /// Gets statistics about cache invalidations
    pub fn get_invalidation_stats(&self) -> InvalidationStats {
        let log = self.invalidation_log.lock();

        InvalidationStats {
            total_invalidations: log.len(),
            cache_regions: self.cache_regions.len(),
            entity_mappings: self.entity_to_regions.len(),
            last_invalidation: log.last().map(|(time, _)| *time),
        }
    }
}

/// Statistics about cache invalidations
#[derive(Debug, Clone)]
pub struct InvalidationStats {
    /// Total number of invalidations performed
    pub total_invalidations: usize,
    /// Number of cache regions registered
    pub cache_regions: usize,
    /// Number of entity-to-region mappings
    pub entity_mappings: usize,
    /// Timestamp of last invalidation
    pub last_invalidation: Option<DateTime<Utc>>,
}

// ============================================================================
// Conflict Resolution
// ============================================================================

/// Conflict resolver with multiple strategies
pub struct ConflictResolver {
    pub(super) strategy: ConflictStrategy,
    custom_resolvers: HashMap<String, ConflictResolverFn>,
}

// Reduce type complexity for custom resolver function type
type ConflictResolverFn = Box<dyn Fn(&Conflict) -> Result<ConflictResolution> + Send + Sync>;

impl ConflictResolver {
    /// Creates a new conflict resolver with the given strategy
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self {
            strategy,
            custom_resolvers: HashMap::new(),
        }
    }

    /// Adds a custom resolver function by name
    pub fn with_custom_resolver<F>(mut self, name: String, resolver: F) -> Self
    where
        F: Fn(&Conflict) -> Result<ConflictResolution> + Send + Sync + 'static,
    {
        self.custom_resolvers.insert(name, Box::new(resolver));
        self
    }

    /// Resolves a conflict using the configured strategy
    pub async fn resolve_conflict(&self, conflict: &Conflict) -> Result<ConflictResolution> {
        match &self.strategy {
            ConflictStrategy::KeepExisting => Ok(ConflictResolution {
                strategy: ConflictStrategy::KeepExisting,
                resolved_data: conflict.existing_data.clone(),
                metadata: HashMap::new(),
            }),
            ConflictStrategy::KeepNew => Ok(ConflictResolution {
                strategy: ConflictStrategy::KeepNew,
                resolved_data: conflict.new_data.clone(),
                metadata: HashMap::new(),
            }),
            ConflictStrategy::Merge => self.merge_conflict_data(conflict).await,
            ConflictStrategy::Custom(resolver_name) => {
                if let Some(resolver) = self.custom_resolvers.get(resolver_name) {
                    resolver(conflict)
                } else {
                    Err(GraphRAGError::ConflictResolution {
                        message: format!("Custom resolver '{resolver_name}' not found"),
                    })
                }
            },
            _ => Err(GraphRAGError::ConflictResolution {
                message: "Conflict resolution strategy not implemented".to_string(),
            }),
        }
    }

    async fn merge_conflict_data(&self, conflict: &Conflict) -> Result<ConflictResolution> {
        match (&conflict.existing_data, &conflict.new_data) {
            (ChangeData::Entity(existing), ChangeData::Entity(new)) => {
                let merged = self.merge_entities(existing, new)?;
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Merge,
                    resolved_data: ChangeData::Entity(merged),
                    metadata: [("merge_strategy".to_string(), "entity_merge".to_string())]
                        .into_iter()
                        .collect(),
                })
            },
            (ChangeData::Relationship(existing), ChangeData::Relationship(new)) => {
                let merged = self.merge_relationships(existing, new)?;
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Merge,
                    resolved_data: ChangeData::Relationship(merged),
                    metadata: [(
                        "merge_strategy".to_string(),
                        "relationship_merge".to_string(),
                    )]
                    .into_iter()
                    .collect(),
                })
            },
            _ => Err(GraphRAGError::ConflictResolution {
                message: "Cannot merge incompatible data types".to_string(),
            }),
        }
    }

    pub(super) fn merge_entities(&self, existing: &Entity, new: &Entity) -> Result<Entity> {
        let mut merged = existing.clone();

        // Use higher confidence
        if new.confidence > existing.confidence {
            merged.confidence = new.confidence;
            merged.name = new.name.clone();
            merged.entity_type = new.entity_type.clone();
        }

        // Merge mentions
        let mut all_mentions = existing.mentions.clone();
        for new_mention in &new.mentions {
            if !all_mentions.iter().any(|m| {
                m.chunk_id == new_mention.chunk_id && m.start_offset == new_mention.start_offset
            }) {
                all_mentions.push(new_mention.clone());
            }
        }
        merged.mentions = all_mentions;

        // Prefer new embedding if available
        if new.embedding.is_some() {
            merged.embedding = new.embedding.clone();
        }

        Ok(merged)
    }

    fn merge_relationships(
        &self,
        existing: &Relationship,
        new: &Relationship,
    ) -> Result<Relationship> {
        let mut merged = existing.clone();

        // Use higher confidence
        if new.confidence > existing.confidence {
            merged.confidence = new.confidence;
            merged.relation_type = new.relation_type.clone();
        }

        // Merge contexts
        let mut all_contexts = existing.context.clone();
        for new_context in &new.context {
            if !all_contexts.contains(new_context) {
                all_contexts.push(new_context.clone());
            }
        }
        merged.context = all_contexts;

        Ok(merged)
    }
}

// ============================================================================
// Update Monitor and Metrics
// ============================================================================

/// Monitor for tracking update operations and performance
#[cfg(feature = "incremental")]
pub struct UpdateMonitor {
    metrics: DashMap<String, UpdateMetric>,
    operations_log: Mutex<Vec<OperationLog>>,
    performance_stats: RwLock<PerformanceStats>,
}

#[cfg(feature = "incremental")]
impl Default for UpdateMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Metric for tracking update operations
#[derive(Debug, Clone)]
pub struct UpdateMetric {
    /// Name of the metric
    pub name: String,
    /// Metric value
    pub value: f64,
    /// When the metric was recorded
    pub timestamp: DateTime<Utc>,
    /// Tags for categorizing the metric
    pub tags: HashMap<String, String>,
}

/// Log entry for an operation
#[derive(Debug, Clone)]
pub struct OperationLog {
    /// Unique operation identifier
    pub operation_id: UpdateId,
    /// Type of operation performed
    pub operation_type: String,
    /// When the operation started
    pub start_time: Instant,
    /// When the operation ended
    pub end_time: Option<Instant>,
    /// Whether the operation succeeded
    pub success: Option<bool>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Number of entities affected
    pub affected_entities: usize,
    /// Number of relationships affected
    pub affected_relationships: usize,
}

/// Performance statistics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of operations performed
    pub total_operations: u64,
    /// Number of successful operations
    pub successful_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Average time per operation
    pub average_operation_time: Duration,
    /// Peak throughput in operations per second
    pub peak_operations_per_second: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Conflict resolution rate (0.0 to 1.0)
    pub conflict_resolution_rate: f64,
}

#[cfg(feature = "incremental")]
impl UpdateMonitor {
    /// Creates a new update monitor
    pub fn new() -> Self {
        Self {
            metrics: DashMap::new(),
            operations_log: Mutex::new(Vec::new()),
            performance_stats: RwLock::new(PerformanceStats {
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                average_operation_time: Duration::from_millis(0),
                peak_operations_per_second: 0.0,
                cache_hit_rate: 0.0,
                conflict_resolution_rate: 0.0,
            }),
        }
    }

    /// Starts tracking a new operation and returns its ID
    pub fn start_operation(&self, operation_type: &str) -> UpdateId {
        let operation_id = UpdateId::new();
        let log_entry = OperationLog {
            operation_id: operation_id.clone(),
            operation_type: operation_type.to_string(),
            start_time: Instant::now(),
            end_time: None,
            success: None,
            error_message: None,
            affected_entities: 0,
            affected_relationships: 0,
        };

        self.operations_log.lock().push(log_entry);
        operation_id
    }

    /// Marks an operation as complete with results
    pub fn complete_operation(
        &self,
        operation_id: &UpdateId,
        success: bool,
        error: Option<String>,
        affected_entities: usize,
        affected_relationships: usize,
    ) {
        let mut log = self.operations_log.lock();
        if let Some(entry) = log.iter_mut().find(|e| &e.operation_id == operation_id) {
            entry.end_time = Some(Instant::now());
            entry.success = Some(success);
            entry.error_message = error;
            entry.affected_entities = affected_entities;
            entry.affected_relationships = affected_relationships;
        }

        // Update performance stats
        self.update_performance_stats();
    }

    fn update_performance_stats(&self) {
        let log = self.operations_log.lock();
        let completed_ops: Vec<_> = log
            .iter()
            .filter(|op| op.end_time.is_some() && op.success.is_some())
            .collect();

        if completed_ops.is_empty() {
            return;
        }

        let mut stats = self.performance_stats.write();
        stats.total_operations = completed_ops.len() as u64;
        stats.successful_operations = completed_ops
            .iter()
            .filter(|op| op.success == Some(true))
            .count() as u64;
        stats.failed_operations = stats.total_operations - stats.successful_operations;

        // Calculate average operation time
        let total_time: Duration = completed_ops
            .iter()
            .filter_map(|op| op.end_time.map(|end| end.duration_since(op.start_time)))
            .sum();

        if !completed_ops.is_empty() {
            stats.average_operation_time = total_time / completed_ops.len() as u32;
        }
    }

    /// Records a metric with tags
    pub fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) {
        let metric = UpdateMetric {
            name: name.to_string(),
            value,
            timestamp: Utc::now(),
            tags,
        };
        self.metrics.insert(name.to_string(), metric);
    }

    /// Gets the current performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_stats.read().clone()
    }

    /// Gets the most recent operations up to the specified limit
    pub fn get_recent_operations(&self, limit: usize) -> Vec<OperationLog> {
        let log = self.operations_log.lock();
        log.iter().rev().take(limit).cloned().collect()
    }
}
