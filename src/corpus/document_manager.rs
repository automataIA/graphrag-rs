//! Document collection management and indexing

use crate::core::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: String,
    pub path: PathBuf,
    pub content: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub content_hash: String,
    pub document_type: DocumentType,
    pub language: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DocumentType {
    Text,
    Pdf,
    Html,
    Markdown,
    Json,
    Xml,
    Unknown,
}

impl DocumentType {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "txt" => Self::Text,
            "pdf" => Self::Pdf,
            "html" | "htm" => Self::Html,
            "md" | "markdown" => Self::Markdown,
            "json" => Self::Json,
            "xml" => Self::Xml,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DocumentCollection {
    pub id: String,
    pub name: String,
    pub documents: HashMap<String, DocumentMetadata>,
    pub created_at: DateTime<Utc>,
    pub total_size_bytes: u64,
    pub index: DocumentIndex,
}

#[derive(Debug, Clone)]
pub struct DocumentIndex {
    pub by_type: HashMap<DocumentType, Vec<String>>,
    pub by_date: Vec<(DateTime<Utc>, String)>,
    pub by_size: Vec<(u64, String)>,
    pub content_hashes: HashMap<String, String>,
}

impl Default for DocumentIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentIndex {
    pub fn new() -> Self {
        Self {
            by_type: HashMap::new(),
            by_date: Vec::new(),
            by_size: Vec::new(),
            content_hashes: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, doc: &DocumentMetadata) {
        // Index by type
        self.by_type
            .entry(doc.document_type.clone())
            .or_default()
            .push(doc.id.clone());

        // Index by date
        self.by_date.push((doc.created_at, doc.id.clone()));
        self.by_date.sort_by(|a, b| b.0.cmp(&a.0)); // Most recent first

        // Index by size
        self.by_size.push((doc.size_bytes, doc.id.clone()));
        self.by_size.sort_by(|a, b| b.0.cmp(&a.0)); // Largest first

        // Index by content hash for deduplication
        self.content_hashes.insert(doc.content_hash.clone(), doc.id.clone());
    }

    pub fn find_duplicates(&self, content_hash: &str) -> Option<&String> {
        self.content_hashes.get(content_hash)
    }

    pub fn documents_by_type(&self, doc_type: &DocumentType) -> Vec<&String> {
        self.by_type.get(doc_type).map(|docs| docs.iter().collect()).unwrap_or_default()
    }
}

pub struct DocumentManager {
    collections: HashMap<String, DocumentCollection>,
    current_collection: Option<String>,
}

impl DocumentManager {
    pub fn new() -> Result<Self> {
        Ok(Self {
            collections: HashMap::new(),
            current_collection: None,
        })
    }

    /// Load a complete document collection from a directory
    pub async fn load_collection(&mut self, collection_path: &Path) -> Result<&DocumentCollection> {
        let collection_name = collection_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed_collection")
            .to_string();

        let collection_id = format!("collection_{}", uuid::Uuid::new_v4());

        println!("📂 Loading document collection: {collection_name}");
        println!("   Path: {}", collection_path.display());

        let mut documents = HashMap::new();
        let mut total_size = 0u64;
        let mut index = DocumentIndex::new();

        // Walk directory and process files
        if collection_path.is_dir() {
            for entry in std::fs::read_dir(collection_path)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    if let Ok(metadata) = self.process_file(&path).await {
                        total_size += metadata.size_bytes;
                        index.add_document(&metadata);

                        // Check for duplicates
                        if let Some(existing_id) = index.find_duplicates(&metadata.content_hash) {
                            println!("   ⚠️  Duplicate content found: {} (same as {})",
                                metadata.path.display(), existing_id);
                            continue;
                        }

                        println!("   ✅ Loaded: {} ({} bytes)",
                            metadata.path.display(), metadata.size_bytes);
                        documents.insert(metadata.id.clone(), metadata);
                    }
                }
            }
        }

        let collection = DocumentCollection {
            id: collection_id.clone(),
            name: collection_name,
            documents,
            created_at: Utc::now(),
            total_size_bytes: total_size,
            index,
        };

        println!("   📊 Collection loaded: {} documents, {:.1} KB total",
            collection.documents.len(),
            total_size as f32 / 1024.0
        );

        self.collections.insert(collection_id.clone(), collection);
        self.current_collection = Some(collection_id.clone());

        Ok(self.collections.get(&collection_id).unwrap())
    }

    /// Add a single document to the current collection
    pub async fn add_document(&mut self, document_path: &Path) -> Result<DocumentMetadata> {
        let metadata = self.process_file(document_path).await?;

        if let Some(collection_id) = &self.current_collection {
            if let Some(collection) = self.collections.get_mut(collection_id) {
                // Check for duplicates
                if let Some(existing_id) = collection.index.find_duplicates(&metadata.content_hash) {
                    return Err(crate::core::GraphRAGError::Config {
                        message: format!("Document already exists with ID: {existing_id}")
                    });
                }

                collection.index.add_document(&metadata);
                collection.total_size_bytes += metadata.size_bytes;
                collection.documents.insert(metadata.id.clone(), metadata.clone());

                println!("   ✅ Added document: {} to collection",
                    metadata.path.display());
            }
        }

        Ok(metadata)
    }

    /// Process a single file into document metadata
    async fn process_file(&self, path: &Path) -> Result<DocumentMetadata> {
        let file_metadata = std::fs::metadata(path)?;
        let content = std::fs::read_to_string(path)?;

        let content_hash = self.calculate_content_hash(&content);
        let language = self.detect_language(&content);
        let document_type = path.extension()
            .and_then(|ext| ext.to_str())
            .map(DocumentType::from_extension)
            .unwrap_or(DocumentType::Unknown);

        let title = path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("untitled")
            .to_string();

        Ok(DocumentMetadata {
            id: format!("doc_{}", uuid::Uuid::new_v4()),
            title,
            path: path.to_path_buf(),
            content,
            size_bytes: file_metadata.len(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
            content_hash,
            document_type,
            language,
            tags: Vec::new(),
        })
    }

    /// Calculate content hash for deduplication
    fn calculate_content_hash(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("hash_{:x}", hasher.finish())
    }

    /// Basic language detection (placeholder implementation)
    fn detect_language(&self, content: &str) -> Option<String> {
        // Simple heuristic - could be enhanced with proper language detection
        if content.len() < 10 {
            return None;
        }

        // Look for common English words
        let english_indicators = ["the", "and", "or", "but", "in", "on", "at", "to"];
        let word_count = content.split_whitespace().count();
        let english_word_count = english_indicators.iter()
            .map(|word| content.matches(word).count())
            .sum::<usize>();

        if word_count > 0 && (english_word_count as f32 / word_count as f32) > 0.05 {
            Some("en".to_string())
        } else {
            Some("unknown".to_string())
        }
    }

    /// Get current collection
    pub fn get_current_collection(&self) -> Option<&DocumentCollection> {
        self.current_collection.as_ref()
            .and_then(|id| self.collections.get(id))
    }

    /// List all collections
    pub fn list_collections(&self) -> Vec<&DocumentCollection> {
        self.collections.values().collect()
    }

    /// Get collection by ID
    pub fn get_collection(&self, collection_id: &str) -> Option<&DocumentCollection> {
        self.collections.get(collection_id)
    }

    /// Switch to a different collection
    pub fn switch_collection(&mut self, collection_id: &str) -> Result<()> {
        if self.collections.contains_key(collection_id) {
            self.current_collection = Some(collection_id.to_string());
            Ok(())
        } else {
            Err(crate::core::GraphRAGError::Config {
                message: format!("Collection not found: {collection_id}")
            })
        }
    }
}