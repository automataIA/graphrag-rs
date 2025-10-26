//! Query history management

use chrono::{DateTime, Utc};
use color_eyre::eyre::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Single query history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEntry {
    /// Query text
    pub query: String,
    /// Timestamp when query was executed
    pub timestamp: DateTime<Utc>,
    /// Execution duration in milliseconds
    pub duration_ms: u128,
    /// Number of results returned
    pub results_count: usize,
    /// Preview of first 3 results (truncated)
    pub results_preview: Vec<String>,
}

/// Query history manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryHistory {
    /// List of query entries
    entries: Vec<QueryEntry>,
    /// Maximum number of entries to keep
    #[serde(default = "default_max_entries")]
    max_entries: usize,
}

fn default_max_entries() -> usize {
    1000
}

impl QueryHistory {
    /// Create a new query history
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: default_max_entries(),
        }
    }

    /// Add a query entry
    pub fn add_entry(&mut self, entry: QueryEntry) {
        self.entries.insert(0, entry);

        // Trim if exceeds max entries
        if self.entries.len() > self.max_entries {
            self.entries.truncate(self.max_entries);
        }
    }

    /// Get all entries
    #[allow(dead_code)]
    pub fn entries(&self) -> &[QueryEntry] {
        &self.entries
    }

    /// Get last N entries
    #[allow(dead_code)]
    pub fn last_n(&self, n: usize) -> &[QueryEntry] {
        let end = n.min(self.entries.len());
        &self.entries[..end]
    }

    /// Clear all entries
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear()
    }

    /// Get total query count
    pub fn total_queries(&self) -> usize {
        self.entries.len()
    }

    /// Save to file
    #[allow(dead_code)]
    pub async fn save(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Load from file
    pub async fn load(path: &PathBuf) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        let history: Self = serde_json::from_str(&content)?;
        Ok(history)
    }
}

impl Default for QueryHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entry() {
        let mut history = QueryHistory::new();
        let entry = QueryEntry {
            query: "test".to_string(),
            timestamp: Utc::now(),
            duration_ms: 100,
            results_count: 5,
            results_preview: vec![],
        };

        history.add_entry(entry.clone());
        assert_eq!(history.total_queries(), 1);
        assert_eq!(history.entries()[0].query, "test");
    }

    #[test]
    fn test_max_entries() {
        let mut history = QueryHistory::new();
        history.max_entries = 5;

        for i in 0..10 {
            history.add_entry(QueryEntry {
                query: format!("query {}", i),
                timestamp: Utc::now(),
                duration_ms: 100,
                results_count: 1,
                results_preview: vec![],
            });
        }

        assert_eq!(history.total_queries(), 5);
    }
}
