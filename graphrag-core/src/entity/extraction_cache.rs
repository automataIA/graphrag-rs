//! On-disk cache of raw LLM extraction responses.
//!
//! One file per key (`<sha256>.json`) holding the model's raw JSON answer.
//! Any read problem (missing, unreadable, not a valid extraction) is a cache
//! miss and any write problem is ignored: the cache never makes extraction fail.

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Directory-backed cache of raw extraction responses.
#[derive(Debug, Clone)]
pub struct ExtractionCache {
    dir: PathBuf,
}

impl ExtractionCache {
    /// Cache rooted at `dir` (created lazily on the first write).
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Hex SHA-256 over the parts, each length-prefixed so boundaries are unambiguous.
    pub fn key(parts: &[&str]) -> String {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update((part.len() as u64).to_le_bytes());
            hasher.update(part.as_bytes());
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    /// Cached response for `key`, only if it is still a valid extraction JSON.
    pub fn get(&self, key: &str) -> Option<String> {
        let text = std::fs::read_to_string(self.path(key)).ok()?;
        serde_json::from_str::<super::prompts::ExtractionOutput>(&text)
            .ok()
            .map(|_| text)
    }

    /// Store `response` under `key` (atomic rename, errors ignored).
    pub fn put(&self, key: &str, response: &str) {
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let staging = self.dir.join(format!(
            "{key}.{}.{}.staging",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::write(&staging, response).is_err()
            || std::fs::rename(&staging, self.path(key)).is_err()
        {
            let _ = std::fs::remove_file(&staging);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_depends_on_every_part_and_boundaries() {
        let a = ExtractionCache::key(&["ab", "c"]);
        assert_eq!(a, ExtractionCache::key(&["ab", "c"]));
        assert_ne!(a, ExtractionCache::key(&["a", "bc"]));
        assert_ne!(a, ExtractionCache::key(&["ab", "d"]));
    }

    #[test]
    fn roundtrip_and_corrupt_is_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ExtractionCache::new(dir.path().join("nested"));
        let key = ExtractionCache::key(&["x"]);
        assert!(cache.get(&key).is_none());
        let body = r#"{"entities":[],"relationships":[]}"#;
        cache.put(&key, body);
        assert_eq!(cache.get(&key).as_deref(), Some(body));
        std::fs::write(cache.path(&key), "garbage").unwrap();
        assert!(cache.get(&key).is_none());
    }
}
