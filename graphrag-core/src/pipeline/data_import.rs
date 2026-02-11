//! Data Import Pipeline
//!
//! This module provides data import capabilities from multiple formats:
//! - CSV/TSV files
//! - JSON/JSONL (newline-delimited JSON)
//! - RDF/Turtle (semantic web formats)
//! - GraphML (graph exchange format)
//! - Streaming ingestion from various sources
//!
//! ## Architecture
//!
//! ```text
//! Data Source → Parser → Validator → Transformer → Graph Builder
//!      │           │          │           │              │
//!      ▼           ▼          ▼           ▼              ▼
//!   CSV/JSON   Schema    Required    Normalize      KnowledgeGraph
//!   Files      Check     Fields      Format
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader};
use csv::ReaderBuilder;

/// Supported data formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum DataFormat {
    /// CSV (comma-separated values)
    CSV,
    /// TSV (tab-separated values)
    TSV,
    /// JSON (JavaScript Object Notation)
    JSON,
    /// JSONL (newline-delimited JSON)
    JSONL,
    /// RDF/Turtle (Resource Description Framework)
    RDF,
    /// GraphML (graph markup language)
    GraphML,
}

/// Import configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Data format
    pub format: DataFormat,
    /// Skip validation
    pub skip_validation: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum errors before aborting
    pub max_errors: usize,
    /// Column mappings (for CSV/TSV)
    pub column_mappings: Option<ColumnMappings>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            format: DataFormat::JSON,
            skip_validation: false,
            batch_size: 1000,
            max_errors: 10,
            column_mappings: None,
        }
    }
}

/// Column mappings for CSV/TSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMappings {
    /// Entity ID column
    pub entity_id: String,
    /// Entity name column
    pub entity_name: String,
    /// Entity type column
    pub entity_type: String,
    /// Optional source column for relationships
    pub relationship_source: Option<String>,
    /// Optional target column for relationships
    pub relationship_target: Option<String>,
    /// Optional relationship type column
    pub relationship_type: Option<String>,
}

/// Imported entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedEntity {
    /// Unique identifier for the entity
    pub id: String,
    /// Display name of the entity
    pub name: String,
    /// Type/category of the entity
    pub entity_type: String,
    /// Additional attributes as key-value pairs
    pub attributes: std::collections::HashMap<String, String>,
}

/// Imported relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedRelationship {
    /// Source entity identifier
    pub source: String,
    /// Target entity identifier
    pub target: String,
    /// Type of the relationship
    pub relation_type: String,
    /// Additional attributes as key-value pairs
    pub attributes: std::collections::HashMap<String, String>,
}

/// Import result
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Number of entities imported
    pub entities_imported: usize,
    /// Number of relationships imported
    pub relationships_imported: usize,
    /// Number of errors
    pub errors: Vec<ImportError>,
    /// Processing time (milliseconds)
    pub processing_time_ms: u64,
}

/// Import errors
#[derive(Debug, Clone)]
pub enum ImportError {
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String, usize), // message, line number
    /// Validation error
    ValidationError(String),
    /// Missing required field
    MissingField(String),
    /// Invalid format
    InvalidFormat(String),
    /// Unsupported format (feature not enabled)
    UnsupportedFormat(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ImportError::FileNotFound(path) => write!(f, "File not found: {}", path),
            ImportError::ParseError(msg, line) => write!(f, "Parse error at line {}: {}", line, msg),
            ImportError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            ImportError::MissingField(field) => write!(f, "Missing required field: {}", field),
            ImportError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ImportError::UnsupportedFormat(msg) => write!(f, "Unsupported format: {}", msg),
        }
    }
}

impl std::error::Error for ImportError {}

/// Data importer
pub struct DataImporter {
    config: ImportConfig,
}

impl DataImporter {
    /// Create new importer
    pub fn new(config: ImportConfig) -> Self {
        Self { config }
    }

    /// Import from file
    pub fn import_file(&self, path: impl AsRef<Path>) -> Result<ImportResult, ImportError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ImportError::FileNotFound(path.display().to_string()));
        }

        let start_time = std::time::Instant::now();

        let result = match self.config.format {
            DataFormat::CSV => self.import_csv(path)?,
            DataFormat::TSV => self.import_tsv(path)?,
            DataFormat::JSON => self.import_json(path)?,
            DataFormat::JSONL => self.import_jsonl(path)?,
            DataFormat::RDF => self.import_rdf(path)?,
            DataFormat::GraphML => self.import_graphml(path)?,
        };

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ImportResult {
            entities_imported: result.entities_imported,
            relationships_imported: result.relationships_imported,
            errors: result.errors,
            processing_time_ms,
        })
    }

    /// Import CSV file
    fn import_csv(&self, path: &Path) -> Result<ImportResult, ImportError> {
        self.import_csv_with_delimiter(path, b',')
    }

    /// Import CSV/TSV with custom delimiter
    fn import_csv_with_delimiter(&self, path: &Path, delimiter: u8) -> Result<ImportResult, ImportError> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let mut errors = Vec::new();

        let file = File::open(path)
            .map_err(|e| ImportError::ParseError(format!("Failed to open file: {}", e), 0))?;

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .from_reader(file);

        // Get headers
        let headers = reader.headers()
            .map_err(|e| ImportError::ParseError(format!("Failed to read headers: {}", e), 0))?
            .clone();

        let mappings = self.config.column_mappings.as_ref()
            .ok_or_else(|| ImportError::ValidationError("Column mappings required for CSV import".to_string()))?;

        // Find column indices
        let entity_id_idx = headers.iter().position(|h| h == mappings.entity_id)
            .ok_or_else(|| ImportError::MissingField(mappings.entity_id.clone()))?;
        let entity_name_idx = headers.iter().position(|h| h == mappings.entity_name)
            .ok_or_else(|| ImportError::MissingField(mappings.entity_name.clone()))?;
        let entity_type_idx = headers.iter().position(|h| h == mappings.entity_type)
            .ok_or_else(|| ImportError::MissingField(mappings.entity_type.clone()))?;

        // Optional relationship columns
        let rel_source_idx = mappings.relationship_source.as_ref()
            .and_then(|col| headers.iter().position(|h| h == col));
        let rel_target_idx = mappings.relationship_target.as_ref()
            .and_then(|col| headers.iter().position(|h| h == col));
        let rel_type_idx = mappings.relationship_type.as_ref()
            .and_then(|col| headers.iter().position(|h| h == col));

        // Process records
        for (line_num, result) in reader.records().enumerate() {
            let record = match result {
                Ok(r) => r,
                Err(e) => {
                    errors.push(ImportError::ParseError(
                        format!("CSV parse error: {}", e),
                        line_num + 2, // +2 for header and 0-indexing
                    ));
                    if errors.len() >= self.config.max_errors {
                        break;
                    }
                    continue;
                }
            };

            // Extract entity
            let entity_id = record.get(entity_id_idx)
                .unwrap_or("")
                .to_string();
            let entity_name = record.get(entity_name_idx)
                .unwrap_or("")
                .to_string();
            let entity_type = record.get(entity_type_idx)
                .unwrap_or("")
                .to_string();

            if !entity_id.is_empty() && !entity_name.is_empty() && !entity_type.is_empty() {
                // Collect additional attributes
                let mut attributes = std::collections::HashMap::new();
                for (idx, header) in headers.iter().enumerate() {
                    if idx != entity_id_idx && idx != entity_name_idx && idx != entity_type_idx {
                        if let Some(value) = record.get(idx) {
                            if !value.is_empty() {
                                attributes.insert(header.to_string(), value.to_string());
                            }
                        }
                    }
                }

                let entity = ImportedEntity {
                    id: entity_id,
                    name: entity_name,
                    entity_type,
                    attributes,
                };

                // Validate if not skipped
                if !self.config.skip_validation {
                    if let Err(e) = self.validate_entity(&entity) {
                        errors.push(e);
                        if errors.len() >= self.config.max_errors {
                            break;
                        }
                        continue;
                    }
                }

                entities.push(entity);
            }

            // Extract relationship if columns present
            if let (Some(src_idx), Some(tgt_idx), Some(type_idx)) =
                (rel_source_idx, rel_target_idx, rel_type_idx) {

                if let (Some(source), Some(target), Some(rel_type)) =
                    (record.get(src_idx), record.get(tgt_idx), record.get(type_idx)) {

                    if !source.is_empty() && !target.is_empty() && !rel_type.is_empty() {
                        let relationship = ImportedRelationship {
                            source: source.to_string(),
                            target: target.to_string(),
                            relation_type: rel_type.to_string(),
                            attributes: std::collections::HashMap::new(),
                        };

                        if !self.config.skip_validation {
                            if let Err(e) = self.validate_relationship(&relationship) {
                                errors.push(e);
                                if errors.len() >= self.config.max_errors {
                                    break;
                                }
                                continue;
                            }
                        }

                        relationships.push(relationship);
                    }
                }
            }
        }

        Ok(ImportResult {
            entities_imported: entities.len(),
            relationships_imported: relationships.len(),
            errors,
            processing_time_ms: 0, // Will be filled by import_file
        })
    }

    /// Import TSV file
    fn import_tsv(&self, path: &Path) -> Result<ImportResult, ImportError> {
        // TSV is just CSV with tab delimiter
        self.import_csv_with_delimiter(path, b'\t')
    }

    /// Import JSON file
    fn import_json(&self, path: &Path) -> Result<ImportResult, ImportError> {
        let file = File::open(path)
            .map_err(|e| ImportError::ParseError(format!("Failed to open file: {}", e), 0))?;

        let reader = BufReader::new(file);

        // Expected JSON structure:
        // {
        //   "entities": [...],
        //   "relationships": [...]
        // }
        #[derive(Deserialize)]
        struct JsonData {
            entities: Option<Vec<ImportedEntity>>,
            relationships: Option<Vec<ImportedRelationship>>,
        }

        let json_data: JsonData = serde_json::from_reader(reader)
            .map_err(|e| ImportError::ParseError(format!("JSON parse error: {}", e), 0))?;

        let mut errors = Vec::new();
        let mut valid_entities = Vec::new();
        let mut valid_relationships = Vec::new();

        // Validate entities
        if let Some(entities) = json_data.entities {
            for entity in entities {
                if !self.config.skip_validation {
                    if let Err(e) = self.validate_entity(&entity) {
                        errors.push(e);
                        if errors.len() >= self.config.max_errors {
                            break;
                        }
                        continue;
                    }
                }
                valid_entities.push(entity);
            }
        }

        // Validate relationships
        if let Some(relationships) = json_data.relationships {
            for rel in relationships {
                if !self.config.skip_validation {
                    if let Err(e) = self.validate_relationship(&rel) {
                        errors.push(e);
                        if errors.len() >= self.config.max_errors {
                            break;
                        }
                        continue;
                    }
                }
                valid_relationships.push(rel);
            }
        }

        Ok(ImportResult {
            entities_imported: valid_entities.len(),
            relationships_imported: valid_relationships.len(),
            errors,
            processing_time_ms: 0,
        })
    }

    /// Import JSONL file
    fn import_jsonl(&self, path: &Path) -> Result<ImportResult, ImportError> {
        let file = File::open(path)
            .map_err(|e| ImportError::ParseError(format!("Failed to open file: {}", e), 0))?;

        let reader = BufReader::new(file);
        let mut errors = Vec::new();
        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        // Each line is either an entity or relationship JSON object
        // Expected format:
        // {"type": "entity", "id": "...", "name": "...", "entity_type": "...", "attributes": {...}}
        // {"type": "relationship", "source": "...", "target": "...", "relation_type": "...", "attributes": {...}}

        #[derive(Deserialize)]
        #[serde(tag = "type")]
        enum JsonLine {
            #[serde(rename = "entity")]
            Entity {
                id: String,
                name: String,
                entity_type: String,
                #[serde(default)]
                attributes: std::collections::HashMap<String, String>,
            },
            #[serde(rename = "relationship")]
            Relationship {
                source: String,
                target: String,
                relation_type: String,
                #[serde(default)]
                attributes: std::collections::HashMap<String, String>,
            },
        }

        for (line_num, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    errors.push(ImportError::ParseError(
                        format!("Failed to read line: {}", e),
                        line_num + 1,
                    ));
                    if errors.len() >= self.config.max_errors {
                        break;
                    }
                    continue;
                }
            };

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let parsed: JsonLine = match serde_json::from_str(&line) {
                Ok(p) => p,
                Err(e) => {
                    errors.push(ImportError::ParseError(
                        format!("JSON parse error: {}", e),
                        line_num + 1,
                    ));
                    if errors.len() >= self.config.max_errors {
                        break;
                    }
                    continue;
                }
            };

            match parsed {
                JsonLine::Entity { id, name, entity_type, attributes } => {
                    let entity = ImportedEntity {
                        id,
                        name,
                        entity_type,
                        attributes,
                    };

                    if !self.config.skip_validation {
                        if let Err(e) = self.validate_entity(&entity) {
                            errors.push(e);
                            if errors.len() >= self.config.max_errors {
                                break;
                            }
                            continue;
                        }
                    }

                    entities.push(entity);
                }
                JsonLine::Relationship { source, target, relation_type, attributes } => {
                    let rel = ImportedRelationship {
                        source,
                        target,
                        relation_type,
                        attributes,
                    };

                    if !self.config.skip_validation {
                        if let Err(e) = self.validate_relationship(&rel) {
                            errors.push(e);
                            if errors.len() >= self.config.max_errors {
                                break;
                            }
                            continue;
                        }
                    }

                    relationships.push(rel);
                }
            }
        }

        Ok(ImportResult {
            entities_imported: entities.len(),
            relationships_imported: relationships.len(),
            errors,
            processing_time_ms: 0,
        })
    }

    /// Import RDF/Turtle file
    fn import_rdf(&self, path: &Path) -> Result<ImportResult, ImportError> {
        #[cfg(not(feature = "rdf-import"))]
        {
            return Err(ImportError::UnsupportedFormat(
                "RDF import requires 'rdf-import' feature".to_string(),
            ));
        }

        #[cfg(feature = "rdf-import")]
        {
            use oxrdf::NamedNode;
            use oxttl::TurtleParser;
            use std::collections::HashMap;

            let start_time = std::time::Instant::now();
            let mut entities = Vec::new();
            let mut relationships = Vec::new();
            let mut errors = Vec::new();

            // Read file content
            let content = std::fs::read(path)
                .map_err(|e| ImportError::FileNotFound(format!("Failed to read file: {}", e)))?;

            // Parse Turtle/RDF
            let parser = TurtleParser::new().parse_read(&content[..]);

            // Track unique entities (subjects and objects)
            let mut entity_map: HashMap<String, HashMap<String, String>> = HashMap::new();

            for result in parser {
                match result {
                    Ok(triple) => {
                        let subject = triple.subject.to_string();
                        let predicate = triple.predicate.to_string();
                        let object = triple.object.to_string();

                        // Extract entity from subject
                        entity_map.entry(subject.clone()).or_insert_with(HashMap::new);

                        // Check if object is a URI (entity) or literal (property)
                        if object.starts_with('<') && object.ends_with('>') {
                            // Object is an entity - create relationship
                            let object_id = object.trim_start_matches('<').trim_end_matches('>').to_string();
                            entity_map.entry(object_id.clone()).or_insert_with(HashMap::new);

                            // Extract relation type from predicate URI
                            let relation_type = Self::extract_local_name(&predicate);

                            relationships.push(ImportedRelationship {
                                source: Self::extract_local_name(&subject),
                                target: Self::extract_local_name(&object_id),
                                relation_type,
                                attributes: HashMap::new(),
                            });
                        } else {
                            // Object is a literal - add as entity attribute
                            let prop_name = Self::extract_local_name(&predicate);
                            let value = object
                                .trim_start_matches('"')
                                .trim_end_matches('"')
                                .to_string();

                            if let Some(attrs) = entity_map.get_mut(&subject) {
                                attrs.insert(prop_name, value);
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(format!("RDF parse error: {}", e));
                    }
                }
            }

            // Convert entity_map to ImportedEntity list
            for (uri, attributes) in entity_map {
                let id = Self::extract_local_name(&uri);
                let name = attributes.get("label")
                    .or_else(|| attributes.get("name"))
                    .cloned()
                    .unwrap_or_else(|| id.clone());

                let entity_type = attributes.get("type")
                    .or_else(|| attributes.get("rdf:type"))
                    .cloned()
                    .unwrap_or_else(|| "resource".to_string());

                entities.push(ImportedEntity {
                    id,
                    name,
                    entity_type,
                    attributes,
                });
            }

            let processing_time_ms = start_time.elapsed().as_millis() as u64;

            Ok(ImportResult {
                entities_imported: entities.len(),
                relationships_imported: relationships.len(),
                errors,
                processing_time_ms,
            })
        }
    }

    /// Extract local name from RDF URI (after last # or /)
    #[cfg(feature = "rdf-import")]
    fn extract_local_name(uri: &str) -> String {
        let cleaned = uri.trim_start_matches('<').trim_end_matches('>');
        cleaned
            .split(&['#', '/'][..])
            .last()
            .unwrap_or(cleaned)
            .to_string()
    }

    /// Import GraphML file
    fn import_graphml(&self, path: &Path) -> Result<ImportResult, ImportError> {
        #[cfg(not(feature = "graphml-import"))]
        {
            return Err(ImportError::UnsupportedFormat(
                "GraphML import requires 'graphml-import' feature".to_string(),
            ));
        }

        #[cfg(feature = "graphml-import")]
        {
            use crate::core::{Entity, EntityId, EntityMention, Relationship, ChunkId};
            use quick_xml::events::Event;
            use quick_xml::Reader;
            use std::collections::HashMap;

            let start_time = std::time::Instant::now();
            let mut entities = Vec::new();
            let mut relationships = Vec::new();
            let mut errors = Vec::new();

            // Read file content
            let content = std::fs::read_to_string(path)
                .map_err(|e| ImportError::FileNotFound(format!("Failed to read file: {}", e)))?;

            let mut reader = Reader::from_str(&content);
            reader.config_mut().trim_text(true);

            let mut current_node_id = String::new();
            let mut current_node_attributes: HashMap<String, String> = HashMap::new();
            let mut in_node = false;
            let mut in_edge = false;
            let mut current_edge_source = String::new();
            let mut current_edge_target = String::new();
            let mut current_edge_attributes: HashMap<String, String> = HashMap::new();
            let mut current_data_key = String::new();

            let mut buf = Vec::new();
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::Start(e)) => {
                        match e.name().as_ref() {
                            b"node" => {
                                in_node = true;
                                current_node_id.clear();
                                current_node_attributes.clear();

                                // Read node attributes
                                for attr in e.attributes() {
                                    if let Ok(attr) = attr {
                                        if attr.key.as_ref() == b"id" {
                                            current_node_id = String::from_utf8_lossy(&attr.value).to_string();
                                        }
                                    }
                                }
                            }
                            b"edge" => {
                                in_edge = true;
                                current_edge_source.clear();
                                current_edge_target.clear();
                                current_edge_attributes.clear();

                                // Read edge attributes
                                for attr in e.attributes() {
                                    if let Ok(attr) = attr {
                                        let key = attr.key.as_ref();
                                        let value = String::from_utf8_lossy(&attr.value).to_string();
                                        match key {
                                            b"source" => current_edge_source = value,
                                            b"target" => current_edge_target = value,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            b"data" => {
                                // Read data key attribute
                                for attr in e.attributes() {
                                    if let Ok(attr) = attr {
                                        if attr.key.as_ref() == b"key" {
                                            current_data_key = String::from_utf8_lossy(&attr.value).to_string();
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(Event::Text(e)) => {
                        if !current_data_key.is_empty() {
                            let value = e.unescape().unwrap_or_default().to_string();
                            if in_node {
                                current_node_attributes.insert(current_data_key.clone(), value);
                            } else if in_edge {
                                current_edge_attributes.insert(current_data_key.clone(), value);
                            }
                        }
                    }
                    Ok(Event::End(e)) => {
                        match e.name().as_ref() {
                            b"node" => {
                                if in_node && !current_node_id.is_empty() {
                                    // Extract name and type from attributes
                                    let name = current_node_attributes.get("name")
                                        .or_else(|| current_node_attributes.get("label"))
                                        .cloned()
                                        .unwrap_or_else(|| current_node_id.clone());

                                    let entity_type = current_node_attributes.get("type")
                                        .or_else(|| current_node_attributes.get("category"))
                                        .cloned()
                                        .unwrap_or_else(|| "node".to_string());

                                    entities.push(ImportedEntity {
                                        id: current_node_id.clone(),
                                        name,
                                        entity_type,
                                        attributes: current_node_attributes.clone(),
                                    });
                                }
                                in_node = false;
                            }
                            b"edge" => {
                                if in_edge && !current_edge_source.is_empty() && !current_edge_target.is_empty() {
                                    let relation_type = current_edge_attributes.get("type")
                                        .or_else(|| current_edge_attributes.get("label"))
                                        .cloned()
                                        .unwrap_or_else(|| "related".to_string());

                                    relationships.push(ImportedRelationship {
                                        source: current_edge_source.clone(),
                                        target: current_edge_target.clone(),
                                        relation_type,
                                        attributes: current_edge_attributes.clone(),
                                    });
                                }
                                in_edge = false;
                            }
                            b"data" => {
                                current_data_key.clear();
                            }
                            _ => {}
                        }
                    }
                    Ok(Event::Eof) => break,
                    Err(e) => {
                        errors.push(format!("XML parse error: {}", e));
                    }
                    _ => {}
                }
                buf.clear();
            }

            let processing_time_ms = start_time.elapsed().as_millis() as u64;

            Ok(ImportResult {
                entities_imported: entities.len(),
                relationships_imported: relationships.len(),
                errors,
                processing_time_ms,
            })
        }
    }

    /// Validate imported data
    fn validate_entity(&self, entity: &ImportedEntity) -> Result<(), ImportError> {
        if entity.id.is_empty() {
            return Err(ImportError::MissingField("entity_id".to_string()));
        }

        if entity.name.is_empty() {
            return Err(ImportError::MissingField("entity_name".to_string()));
        }

        if entity.entity_type.is_empty() {
            return Err(ImportError::MissingField("entity_type".to_string()));
        }

        Ok(())
    }

    /// Validate relationship
    fn validate_relationship(&self, rel: &ImportedRelationship) -> Result<(), ImportError> {
        if rel.source.is_empty() {
            return Err(ImportError::MissingField("source".to_string()));
        }

        if rel.target.is_empty() {
            return Err(ImportError::MissingField("target".to_string()));
        }

        if rel.relation_type.is_empty() {
            return Err(ImportError::MissingField("relation_type".to_string()));
        }

        Ok(())
    }
}

/// Streaming data source
#[async_trait::async_trait]
pub trait StreamingSource: Send + Sync {
    /// Get next batch of entities
    async fn next_batch(&mut self) -> Result<Vec<ImportedEntity>, ImportError>;

    /// Check if more data available
    async fn has_more(&self) -> bool;
}

/// Streaming importer for continuous data ingestion
pub struct StreamingImporter {
    config: ImportConfig,
}

impl StreamingImporter {
    /// Create new streaming importer
    pub fn new(config: ImportConfig) -> Self {
        Self { config }
    }

    /// Import from streaming source
    pub async fn import_stream<S: StreamingSource>(
        &self,
        mut source: S,
    ) -> Result<ImportResult, ImportError> {
        let mut total_entities = 0;
        let mut errors = Vec::new();

        while source.has_more().await {
            match source.next_batch().await {
                Ok(entities) => {
                    total_entities += entities.len();

                    // Validate if not skipped
                    if !self.config.skip_validation {
                        for entity in &entities {
                            if let Err(e) = self.validate_entity(entity) {
                                errors.push(e);
                            }
                        }
                    }
                }
                Err(e) => {
                    errors.push(e);
                    if errors.len() >= self.config.max_errors {
                        break;
                    }
                }
            }
        }

        Ok(ImportResult {
            entities_imported: total_entities,
            relationships_imported: 0,
            errors,
            processing_time_ms: 0,
        })
    }

    /// Validate entity
    fn validate_entity(&self, entity: &ImportedEntity) -> Result<(), ImportError> {
        if entity.id.is_empty() {
            return Err(ImportError::MissingField("entity_id".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_config_default() {
        let config = ImportConfig::default();
        assert_eq!(config.format, DataFormat::JSON);
        assert_eq!(config.batch_size, 1000);
    }

    #[test]
    fn test_validation() {
        let importer = DataImporter::new(ImportConfig::default());

        let valid_entity = ImportedEntity {
            id: "1".to_string(),
            name: "Test".to_string(),
            entity_type: "Person".to_string(),
            attributes: std::collections::HashMap::new(),
        };

        assert!(importer.validate_entity(&valid_entity).is_ok());

        let invalid_entity = ImportedEntity {
            id: "".to_string(), // Missing ID
            name: "Test".to_string(),
            entity_type: "Person".to_string(),
            attributes: std::collections::HashMap::new(),
        };

        assert!(importer.validate_entity(&invalid_entity).is_err());
    }
}
