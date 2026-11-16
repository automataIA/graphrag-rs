//! Canonical serde-based loader/writer for `Config`.
//!
//! Extracted from `config/mod.rs` (Phase 4 file split). Provides
//! `Config::from_file` (read) and `Config::to_file` (write) via the lightweight
//! `json` crate. Distinct from `config::json5_loader` (serde-based typed JSON5)
//! and `config::loader` (multi-format dispatcher).

#![allow(clippy::needless_borrow, clippy::manual_clamp)]

use super::*;
use crate::Result;
use std::fs;

pub(crate) fn normalize_legacy_runtime_keys(value: &mut serde_json::Value) -> Result<()> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| crate::core::GraphRAGError::Config {
            message: "Configuration root must be a JSON object".to_string(),
        })?;
    for (legacy, section, canonical) in [
        ("chunk_size", "text", "chunk_size"),
        ("chunk_overlap", "text", "chunk_overlap"),
        ("top_k_results", "retrieval", "top_k"),
    ] {
        let legacy_value = object.get(legacy).cloned();
        let canonical_value = object
            .get(section)
            .and_then(serde_json::Value::as_object)
            .and_then(|section| section.get(canonical))
            .cloned();
        if let (Some(legacy_value), Some(canonical_value)) =
            (legacy_value.as_ref(), canonical_value.as_ref())
        {
            if legacy_value != canonical_value {
                return Err(crate::core::GraphRAGError::Config {
                    message: format!(
                        "Conflicting configuration values: {legacy} and {section}.{canonical}"
                    ),
                });
            }
        }
        if let Some(legacy_value) = legacy_value {
            if !legacy_value.is_u64() {
                return Err(crate::core::GraphRAGError::Config {
                    message: format!("{legacy} must be a non-negative integer"),
                });
            }
            let section_value = object
                .entry(section.to_string())
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            let section_object = section_value.as_object_mut().ok_or_else(|| {
                crate::core::GraphRAGError::Config {
                    message: format!("{section} must be a JSON object"),
                }
            })?;
            section_object
                .entry(canonical.to_string())
                .or_insert(legacy_value);
            // Consume the adapter key after projecting it into the canonical
            // section. Keeping it at the root would make a later merge
            // re-interpret a lower-priority alias and could overwrite a
            // higher-priority nested value.
            object.remove(legacy);
        }
    }
    Ok(())
}

pub(crate) fn merge_json(defaults: &mut serde_json::Value, overrides: serde_json::Value) {
    match (defaults, overrides) {
        (serde_json::Value::Object(defaults), serde_json::Value::Object(overrides)) => {
            for (key, value) in overrides {
                merge_json(
                    defaults.entry(key).or_insert(serde_json::Value::Null),
                    value,
                );
            }
        },
        (defaults, overrides) => *defaults = overrides,
    }
}

impl Config {
    /// Parse a JSON runtime configuration, applying canonical defaults and
    /// accepting historical top-level aliases with conflict checks.
    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to parse JSON config: {e}"),
            })?;
        from_json_value(value)
    }

    /// Parse a TOML runtime configuration through the canonical JSON adapter.
    pub fn from_toml_str(content: &str) -> Result<Self> {
        let value: toml::Value =
            toml::from_str(content).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to parse TOML config: {e}"),
            })?;
        let value =
            serde_json::to_value(value).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to convert TOML config: {e}"),
            })?;
        from_json_value(value)
    }

    /// Parse a JSON5 runtime configuration through the canonical JSON adapter.
    #[cfg(feature = "json5-support")]
    pub fn from_json5_str(content: &str) -> Result<Self> {
        let value: serde_json::Value =
            json5::from_str(content).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to parse JSON5 config: {e}"),
            })?;
        from_json_value(value)
    }

    /// Load configuration from a JSON file.
    ///
    /// The input is merged over the canonical defaults.  The historical
    /// top-level chunk/retrieval keys are accepted as read-only adapters, but
    /// conflicting legacy and canonical values are rejected.
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let value: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to parse JSON config: {e}"),
            })?;
        from_json_value(value)
    }

    /// Save the complete runtime configuration to a JSON file.
    pub fn to_file(&self, path: &str) -> Result<()> {
        let content =
            serde_json::to_string_pretty(self).map_err(|e| crate::core::GraphRAGError::Config {
                message: format!("Failed to serialize JSON config: {e}"),
            })?;
        fs::write(path, content)?;
        Ok(())
    }
}

pub(crate) fn from_json_value(mut value: serde_json::Value) -> Result<Config> {
    normalize_legacy_runtime_keys(&mut value)?;
    let mut defaults = serde_json::to_value(Config::default()).map_err(|e| {
        crate::core::GraphRAGError::Config {
            message: format!("Failed to serialize default config: {e}"),
        }
    })?;
    merge_json(&mut defaults, value);
    let config: Config =
        serde_json::from_value(defaults).map_err(|e| crate::core::GraphRAGError::Config {
            message: format!("Invalid JSON config: {e}"),
        })?;
    let validation = config.validate();
    if validation.is_valid {
        Ok(config)
    } else {
        Err(crate::core::GraphRAGError::Config {
            message: validation.errors.join("; "),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_runtime_aliases_are_adapted_to_nested_fields() {
        let config = Config::from_json_str(
            r#"{"chunk_size": 800, "chunk_overlap": 120, "top_k_results": 7}"#,
        )
        .expect("legacy runtime config should parse");
        assert_eq!(config.text.chunk_size, 800);
        assert_eq!(config.text.chunk_overlap, 120);
        assert_eq!(config.retrieval.top_k, 7);
    }

    #[test]
    fn legacy_and_canonical_conflict_is_rejected() {
        let error = Config::from_json_str(r#"{"chunk_size": 800, "text": {"chunk_size": 801}}"#)
            .expect_err("conflicting aliases must fail");
        assert!(error
            .to_string()
            .contains("Conflicting configuration values"));
    }

    #[test]
    fn invalid_legacy_type_is_rejected_instead_of_defaulted() {
        let error = Config::from_json_str(r#"{"chunk_size": "oops"}"#)
            .expect_err("invalid legacy type must fail");
        assert!(error.to_string().contains("chunk_size"));
    }

    #[test]
    fn serde_roundtrip_preserves_complete_runtime_config() {
        let mut config = Config {
            output_dir: "./custom-output".to_string(),
            ..Config::default()
        };
        config.auto_save.enabled = true;
        config.auto_save.base_dir = Some("./workspace".to_string());
        config.llm.enabled = true;
        config.text.chunk_size = 768;

        let file = tempfile::NamedTempFile::new().expect("temporary config path");
        config
            .to_file(file.path().to_str().expect("utf8 path"))
            .unwrap();
        let loaded = Config::from_file(file.path().to_str().expect("utf8 path")).unwrap();
        assert_eq!(loaded.output_dir, config.output_dir);
        assert_eq!(loaded.auto_save.enabled, config.auto_save.enabled);
        assert_eq!(loaded.auto_save.base_dir, config.auto_save.base_dir);
        assert_eq!(loaded.llm.enabled, config.llm.enabled);
        assert_eq!(loaded.text.chunk_size, config.text.chunk_size);
    }
}
