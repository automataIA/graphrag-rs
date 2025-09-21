use crate::{
    core::{ChunkId, DocumentId, TextChunk},
    parallel::ParallelProcessor,
    text::TextProcessor,
    Result,
};
use indexmap::IndexMap;
use std::collections::{HashMap, VecDeque};

/// Unique identifier for tree nodes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(id: String) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<NodeId> for String {
    fn from(id: NodeId) -> Self {
        id.0
    }
}

/// Configuration for hierarchical summarization
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    pub merge_size: usize,
    pub max_summary_length: usize,
    pub min_node_size: usize,
    pub overlap_sentences: usize,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            merge_size: 5,
            max_summary_length: 200,
            min_node_size: 50,
            overlap_sentences: 2,
        }
    }
}

/// A node in the hierarchical document tree
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: NodeId,
    pub content: String,
    pub summary: String,
    pub level: usize,
    pub children: Vec<NodeId>,
    pub parent: Option<NodeId>,
    pub chunk_ids: Vec<ChunkId>,
    pub keywords: Vec<String>,
    pub start_offset: usize,
    pub end_offset: usize,
}

/// Hierarchical document tree for multi-level summarization
#[derive(Debug)]
pub struct DocumentTree {
    nodes: IndexMap<NodeId, TreeNode>,
    root_nodes: Vec<NodeId>,
    levels: HashMap<usize, Vec<NodeId>>,
    document_id: DocumentId,
    config: HierarchicalConfig,
    text_processor: TextProcessor,
}

impl DocumentTree {
    /// Create a new document tree
    pub fn new(document_id: DocumentId, config: HierarchicalConfig) -> Result<Self> {
        let text_processor = TextProcessor::new(1000, 100)?;

        Ok(Self {
            nodes: IndexMap::new(),
            root_nodes: Vec::new(),
            levels: HashMap::new(),
            document_id,
            config,
            text_processor,
        })
    }

    /// Create a new document tree with parallel processing support
    pub fn with_parallel_processing(
        document_id: DocumentId,
        config: HierarchicalConfig,
        _parallel_processor: ParallelProcessor,
    ) -> Result<Self> {
        let text_processor = TextProcessor::new(1000, 100)?;

        Ok(Self {
            nodes: IndexMap::new(),
            root_nodes: Vec::new(),
            levels: HashMap::new(),
            document_id,
            config,
            text_processor,
        })
    }

    /// Build the hierarchical tree from text chunks
    pub fn build_from_chunks(&mut self, chunks: Vec<TextChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        println!("Building hierarchical tree from {} chunks", chunks.len());

        // Create leaf nodes from chunks
        let leaf_nodes = self.create_leaf_nodes(chunks)?;

        // Build the tree bottom-up
        self.build_bottom_up(leaf_nodes)?;

        println!(
            "Tree built with {} total nodes across {} levels",
            self.nodes.len(),
            self.levels.len()
        );

        Ok(())
    }

    /// Create leaf nodes from text chunks with parallel processing
    fn create_leaf_nodes(&mut self, chunks: Vec<TextChunk>) -> Result<Vec<NodeId>> {
        if chunks.len() < 10 {
            // Use sequential processing for small numbers of chunks
            return self.create_leaf_nodes_sequential(chunks);
        }

        #[cfg(feature = "parallel-processing")]
        {
            use rayon::prelude::*;

            // Parallel node creation with proper error handling
            let node_results: std::result::Result<Vec<_>, crate::GraphRAGError> = chunks
                .par_iter()
                .map(|chunk| {
                    let node_id = NodeId::new(format!("leaf_{}", chunk.id));

                    // Create a temporary text processor for each thread to avoid borrowing issues
                    let temp_processor = crate::text::TextProcessor::new(1000, 100)
                        .map_err(|e| crate::GraphRAGError::Config {
                            message: format!("Failed to create text processor: {e}"),
                        })?;

                    // Extract keywords for the chunk
                    let keywords = temp_processor.extract_keywords(&chunk.content, 5);

                    // Generate summary using simplified approach suitable for parallel execution
                    let summary = self.generate_parallel_summary(&chunk.content, &temp_processor)?;

                    let node = TreeNode {
                        id: node_id.clone(),
                        content: chunk.content.clone(),
                        summary,
                        level: 0,
                        children: Vec::new(),
                        parent: None,
                        chunk_ids: vec![chunk.id.clone()],
                        keywords,
                        start_offset: chunk.start_offset,
                        end_offset: chunk.end_offset,
                    };

                    Ok((node_id, node))
                })
                .collect();

            match node_results {
                Ok(nodes) => {
                    let mut leaf_node_ids = Vec::new();

                    // Insert nodes sequentially to avoid concurrent modification
                    for (node_id, node) in nodes {
                        leaf_node_ids.push(node_id.clone());
                        self.nodes.insert(node_id, node);
                    }

                    println!("Created {} leaf nodes in parallel", leaf_node_ids.len());

                    // Update levels tracking
                    self.levels.insert(0, leaf_node_ids.clone());

                    Ok(leaf_node_ids)
                }
                Err(e) => {
                    eprintln!("Error in parallel node creation: {e}");
                    // Fall back to sequential processing
                    self.create_leaf_nodes_sequential(chunks)
                }
            }
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            self.create_leaf_nodes_sequential(chunks)
        }
    }

    /// Sequential leaf node creation (fallback)
    fn create_leaf_nodes_sequential(&mut self, chunks: Vec<TextChunk>) -> Result<Vec<NodeId>> {
        let mut leaf_node_ids = Vec::new();

        for chunk in chunks {
            let node_id = NodeId::new(format!("leaf_{}", chunk.id));

            // Extract keywords for the chunk
            let keywords = self.text_processor.extract_keywords(&chunk.content, 5);

            let node = TreeNode {
                id: node_id.clone(),
                content: chunk.content.clone(),
                summary: self.generate_extractive_summary(&chunk.content)?,
                level: 0,
                children: Vec::new(),
                parent: None,
                chunk_ids: vec![chunk.id],
                keywords,
                start_offset: chunk.start_offset,
                end_offset: chunk.end_offset,
            };

            self.nodes.insert(node_id.clone(), node);
            leaf_node_ids.push(node_id);
        }

        // Update levels tracking
        self.levels.insert(0, leaf_node_ids.clone());

        Ok(leaf_node_ids)
    }

    /// Generate summary suitable for parallel processing
    fn generate_parallel_summary(
        &self,
        text: &str,
        processor: &crate::text::TextProcessor,
    ) -> Result<String> {
        let sentences = processor.extract_sentences(text);

        if sentences.is_empty() {
            return Ok(String::new());
        }

        if sentences.len() == 1 {
            return Ok(sentences[0].clone());
        }

        // Simplified scoring for parallel execution
        let mut best_sentence = &sentences[0];
        let mut best_score = 0.0;

        for sentence in &sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            // Simple scoring based on length and word density
            let length_score = if words.len() < 5 {
                0.1
            } else if words.len() > 30 {
                0.3
            } else {
                1.0
            };

            let word_score = words.len() as f32 * 0.1;
            let score = length_score + word_score;

            if score > best_score {
                best_score = score;
                best_sentence = sentence;
            }
        }

        // Truncate if necessary
        if best_sentence.len() > self.config.max_summary_length {
            Ok(best_sentence
                .chars()
                .take(self.config.max_summary_length - 3)
                .collect::<String>()
                + "...")
        } else {
            Ok(best_sentence.clone())
        }
    }

    /// Build the tree bottom-up by merging nodes at each level
    fn build_bottom_up(&mut self, leaf_nodes: Vec<NodeId>) -> Result<()> {
        let mut current_level_nodes = leaf_nodes;
        let mut current_level = 0;

        while current_level_nodes.len() > 1 {
            let next_level_nodes = self.merge_level(&current_level_nodes, current_level + 1)?;

            current_level_nodes = next_level_nodes;
            current_level += 1;
        }

        // Set root nodes
        self.root_nodes = current_level_nodes;

        Ok(())
    }

    /// Merge nodes at a level to create the next level up
    fn merge_level(&mut self, level_nodes: &[NodeId], new_level: usize) -> Result<Vec<NodeId>> {
        let mut new_level_nodes = Vec::new();
        // Group nodes by merge_size
        for (node_counter, chunk) in level_nodes.chunks(self.config.merge_size).enumerate() {
            let merged_node_id = NodeId::new(format!("level_{new_level}_{node_counter}"));
            let merged_node = self.merge_nodes(chunk, merged_node_id.clone(), new_level)?;

            // Update parent references for children
            for child_id in chunk {
                if let Some(child_node) = self.nodes.get_mut(child_id) {
                    child_node.parent = Some(merged_node_id.clone());
                }
            }

            self.nodes.insert(merged_node_id.clone(), merged_node);
            new_level_nodes.push(merged_node_id);
        }

        // Update levels tracking
        self.levels.insert(new_level, new_level_nodes.clone());

        Ok(new_level_nodes)
    }

    /// Merge multiple nodes into a single parent node
    fn merge_nodes(
        &self,
        node_ids: &[NodeId],
        merged_id: NodeId,
        level: usize,
    ) -> Result<TreeNode> {
        let mut combined_content = String::new();
        let mut all_chunk_ids = Vec::new();
        let mut all_keywords = Vec::new();
        let mut min_offset = usize::MAX;
        let mut max_offset = 0;

        for node_id in node_ids {
            if let Some(node) = self.nodes.get(node_id) {
                if !combined_content.is_empty() {
                    combined_content.push_str("\n\n");
                }
                combined_content.push_str(&node.content);
                all_chunk_ids.extend(node.chunk_ids.clone());
                all_keywords.extend(node.keywords.clone());
                min_offset = min_offset.min(node.start_offset);
                max_offset = max_offset.max(node.end_offset);
            }
        }

        // Deduplicate and limit keywords
        all_keywords.sort();
        all_keywords.dedup();
        all_keywords.truncate(10);

        // Generate summary for the merged content
        let summary = self.generate_extractive_summary(&combined_content)?;

        Ok(TreeNode {
            id: merged_id,
            content: combined_content,
            summary,
            level,
            children: node_ids.to_vec(),
            parent: None,
            chunk_ids: all_chunk_ids,
            keywords: all_keywords,
            start_offset: min_offset,
            end_offset: max_offset,
        })
    }

    /// Generate extractive summary using simple sentence ranking
    fn generate_extractive_summary(&self, text: &str) -> Result<String> {
        let sentences = self.text_processor.extract_sentences(text);

        if sentences.is_empty() {
            return Ok(String::new());
        }

        if sentences.len() == 1 {
            return Ok(sentences[0].clone());
        }

        // Score sentences based on length and keyword density
        let mut sentence_scores: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(i, sentence)| {
                let score = self.score_sentence(sentence, &sentences);
                (i, score)
            })
            .collect();

        // Sort by score descending
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top sentences up to max_summary_length
        let mut summary = String::new();
        let mut selected_indices = Vec::new();

        for (sentence_idx, _score) in sentence_scores {
            let sentence = &sentences[sentence_idx];
            if summary.len() + sentence.len() <= self.config.max_summary_length {
                selected_indices.push(sentence_idx);
                if !summary.is_empty() {
                    summary.push(' ');
                }
                summary.push_str(sentence);
            }
        }

        // If no sentences fit, take the first one truncated
        if summary.is_empty() && !sentences.is_empty() {
            let first_sentence = &sentences[0];
            if first_sentence.len() <= self.config.max_summary_length {
                summary = first_sentence.clone();
            } else {
                summary = first_sentence
                    .chars()
                    .take(self.config.max_summary_length - 3)
                    .collect::<String>()
                    + "...";
            }
        }

        Ok(summary)
    }

    /// Score a sentence for extractive summarization
    fn score_sentence(&self, sentence: &str, all_sentences: &[String]) -> f32 {
        let words: Vec<&str> = sentence.split_whitespace().collect();

        // Base score from length (prefer medium-length sentences)
        let length_score = if words.len() < 5 {
            0.1
        } else if words.len() > 30 {
            0.3
        } else {
            1.0
        };

        // Position score (prefer sentences from beginning and end)
        let position_score = 0.5; // Simplified for now

        // Word frequency score
        let mut word_freq_score = 0.0;
        let total_words: Vec<&str> = all_sentences
            .iter()
            .flat_map(|s| s.split_whitespace())
            .collect();

        for word in &words {
            let word_lower = word.to_lowercase();
            if word_lower.len() > 3 && !self.is_stop_word(&word_lower) {
                let freq = total_words
                    .iter()
                    .filter(|&&w| w.to_lowercase() == word_lower)
                    .count();
                if freq > 1 {
                    word_freq_score += freq as f32 / total_words.len() as f32;
                }
            }
        }

        length_score * 0.4 + position_score * 0.2 + word_freq_score * 0.4
    }

    /// Simple stop word detection (English)
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Query the tree for relevant nodes at different levels
    pub fn query(&self, query: &str, max_results: usize) -> Result<Vec<QueryResult>> {
        let query_keywords = self.text_processor.extract_keywords(query, 5);
        let mut results = Vec::new();

        // Search all nodes for keyword matches
        for (node_id, node) in &self.nodes {
            let score = self.calculate_relevance_score(node, &query_keywords, query);

            if score > 0.1 {
                results.push(QueryResult {
                    node_id: node_id.clone(),
                    score,
                    level: node.level,
                    summary: node.summary.clone(),
                    keywords: node.keywords.clone(),
                    chunk_ids: node.chunk_ids.clone(),
                });
            }
        }

        // Sort by score and return top results
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(max_results);

        Ok(results)
    }

    /// Calculate relevance score for a node given query keywords
    fn calculate_relevance_score(
        &self,
        node: &TreeNode,
        query_keywords: &[String],
        query: &str,
    ) -> f32 {
        let mut score = 0.0;

        // Keyword overlap score
        let node_text = format!("{} {}", node.summary, node.keywords.join(" ")).to_lowercase();
        for keyword in query_keywords {
            if node_text.contains(&keyword.to_lowercase()) {
                score += 1.0;
            }
        }

        // Direct text similarity (simple word overlap)
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let node_words: Vec<&str> = node_text.split_whitespace().collect();

        let mut overlap_count = 0;
        for query_word in &query_words {
            if node_words.contains(&query_word.to_lowercase().as_str()) {
                overlap_count += 1;
            }
        }

        if !query_words.is_empty() {
            score += (overlap_count as f32 / query_words.len() as f32) * 2.0;
        }

        // Level bonus (prefer higher levels for overview, lower levels for details)
        let level_score = 1.0 / (node.level + 1) as f32;
        score += level_score * 0.5;

        score
    }

    /// Get ancestors of a node (path to root)
    pub fn get_ancestors(&self, node_id: &NodeId) -> Vec<&TreeNode> {
        let mut ancestors = Vec::new();
        let mut current_id = Some(node_id.clone());

        while let Some(id) = current_id {
            if let Some(node) = self.nodes.get(&id) {
                ancestors.push(node);
                current_id = node.parent.clone();
            } else {
                break;
            }
        }

        ancestors
    }

    /// Get descendants of a node (all children recursively)
    pub fn get_descendants(&self, node_id: &NodeId) -> Vec<&TreeNode> {
        let mut descendants = Vec::new();
        let mut queue = VecDeque::new();

        if let Some(node) = self.nodes.get(node_id) {
            queue.extend(node.children.iter());
        }

        while let Some(child_id) = queue.pop_front() {
            if let Some(child_node) = self.nodes.get(child_id) {
                descendants.push(child_node);
                queue.extend(child_node.children.iter());
            }
        }

        descendants
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: &NodeId) -> Option<&TreeNode> {
        self.nodes.get(node_id)
    }

    /// Get all nodes at a specific level
    pub fn get_level_nodes(&self, level: usize) -> Vec<&TreeNode> {
        if let Some(node_ids) = self.levels.get(&level) {
            node_ids
                .iter()
                .filter_map(|id| self.nodes.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the root nodes of the tree
    pub fn get_root_nodes(&self) -> Vec<&TreeNode> {
        self.root_nodes
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    /// Get the document ID
    pub fn document_id(&self) -> &DocumentId {
        &self.document_id
    }

    /// Get tree statistics
    pub fn get_statistics(&self) -> TreeStatistics {
        let max_level = self.levels.keys().max().copied().unwrap_or(0);
        let total_nodes = self.nodes.len();
        let nodes_per_level: HashMap<usize, usize> = self
            .levels
            .iter()
            .map(|(level, nodes)| (*level, nodes.len()))
            .collect();

        TreeStatistics {
            total_nodes,
            max_level,
            nodes_per_level,
            root_count: self.root_nodes.len(),
            document_id: self.document_id.clone(),
        }
    }

    /// Serialize the tree to JSON format
    pub fn to_json(&self) -> Result<String> {
        use json::JsonValue;

        let mut tree_json = json::object! {
            "document_id": self.document_id.to_string(),
            "config": {
                "merge_size": self.config.merge_size,
                "max_summary_length": self.config.max_summary_length,
                "min_node_size": self.config.min_node_size,
                "overlap_sentences": self.config.overlap_sentences
            },
            "nodes": {},
            "root_nodes": [],
            "levels": {}
        };

        // Serialize nodes
        for (node_id, node) in &self.nodes {
            let node_json = json::object! {
                "id": node_id.to_string(),
                "content": node.content.clone(),
                "summary": node.summary.clone(),
                "level": node.level,
                "children": node.children.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                "parent": node.parent.as_ref().map(|id| id.to_string()),
                "chunk_ids": node.chunk_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                "keywords": node.keywords.clone(),
                "start_offset": node.start_offset,
                "end_offset": node.end_offset
            };
            tree_json["nodes"][node_id.to_string()] = node_json;
        }

        // Serialize root nodes
        tree_json["root_nodes"] = self
            .root_nodes
            .iter()
            .map(|id| JsonValue::String(id.to_string()))
            .collect::<Vec<_>>()
            .into();

        // Serialize levels
        for (level, node_ids) in &self.levels {
            tree_json["levels"][level.to_string()] = node_ids
                .iter()
                .map(|id| JsonValue::String(id.to_string()))
                .collect::<Vec<_>>()
                .into();
        }

        Ok(tree_json.dump())
    }

    /// Load tree from JSON format
    pub fn from_json(json_str: &str) -> Result<Self> {
        let json_data = json::parse(json_str).map_err(crate::GraphRAGError::Json)?;

        let document_id = DocumentId::new(
            json_data["document_id"]
                .as_str()
                .ok_or_else(|| {
                    crate::GraphRAGError::Json(json::Error::WrongType(
                        "document_id must be string".to_string(),
                    ))
                })?
                .to_string(),
        );

        let config_json = &json_data["config"];
        let config = HierarchicalConfig {
            merge_size: config_json["merge_size"].as_usize().unwrap_or(5),
            max_summary_length: config_json["max_summary_length"].as_usize().unwrap_or(200),
            min_node_size: config_json["min_node_size"].as_usize().unwrap_or(50),
            overlap_sentences: config_json["overlap_sentences"].as_usize().unwrap_or(2),
        };

        let mut tree = Self::new(document_id, config)?;

        // Load nodes
        if let json::JsonValue::Object(nodes_obj) = &json_data["nodes"] {
            for (node_id_str, node_json) in nodes_obj.iter() {
                let node_id = NodeId::new(node_id_str.to_string());

                let children: Vec<NodeId> = node_json["children"]
                    .members()
                    .filter_map(|v| v.as_str())
                    .map(|s| NodeId::new(s.to_string()))
                    .collect();

                let parent = node_json["parent"]
                    .as_str()
                    .map(|s| NodeId::new(s.to_string()));

                let chunk_ids: Vec<ChunkId> = node_json["chunk_ids"]
                    .members()
                    .filter_map(|v| v.as_str())
                    .map(|s| ChunkId::new(s.to_string()))
                    .collect();

                let keywords: Vec<String> = node_json["keywords"]
                    .members()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect();

                let node = TreeNode {
                    id: node_id.clone(),
                    content: node_json["content"].as_str().unwrap_or("").to_string(),
                    summary: node_json["summary"].as_str().unwrap_or("").to_string(),
                    level: node_json["level"].as_usize().unwrap_or(0),
                    children,
                    parent,
                    chunk_ids,
                    keywords,
                    start_offset: node_json["start_offset"].as_usize().unwrap_or(0),
                    end_offset: node_json["end_offset"].as_usize().unwrap_or(0),
                };

                tree.nodes.insert(node_id, node);
            }
        }

        // Load root nodes
        tree.root_nodes = json_data["root_nodes"]
            .members()
            .filter_map(|v| v.as_str())
            .map(|s| NodeId::new(s.to_string()))
            .collect();

        // Load levels
        if let json::JsonValue::Object(levels_obj) = &json_data["levels"] {
            for (level_str, level_json) in levels_obj.iter() {
                if let Ok(level) = level_str.parse::<usize>() {
                    let node_ids: Vec<NodeId> = level_json
                        .members()
                        .filter_map(|v| v.as_str())
                        .map(|s| NodeId::new(s.to_string()))
                        .collect();
                    tree.levels.insert(level, node_ids);
                }
            }
        }

        Ok(tree)
    }
}

/// Result from querying the hierarchical tree
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub node_id: NodeId,
    pub score: f32,
    pub level: usize,
    pub summary: String,
    pub keywords: Vec<String>,
    pub chunk_ids: Vec<ChunkId>,
}

/// Statistics about the hierarchical tree
#[derive(Debug)]
pub struct TreeStatistics {
    pub total_nodes: usize,
    pub max_level: usize,
    pub nodes_per_level: HashMap<usize, usize>,
    pub root_count: usize,
    pub document_id: DocumentId,
}

impl TreeStatistics {
    /// Print tree statistics
    pub fn print(&self) {
        println!("Hierarchical Tree Statistics:");
        println!("  Document ID: {}", self.document_id);
        println!("  Total nodes: {}", self.total_nodes);
        println!("  Max level: {}", self.max_level);
        println!("  Root nodes: {}", self.root_count);
        println!("  Nodes per level:");

        let mut levels: Vec<_> = self.nodes_per_level.iter().collect();
        levels.sort_by_key(|(level, _)| *level);

        for (level, count) in levels {
            println!("    Level {level}: {count} nodes");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DocumentId;

    #[test]
    fn test_tree_creation() {
        let config = HierarchicalConfig::default();
        let doc_id = DocumentId::new("test_doc".to_string());
        let tree = DocumentTree::new(doc_id, config);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_extractive_summarization() {
        let config = HierarchicalConfig::default();
        let doc_id = DocumentId::new("test_doc".to_string());
        let tree = DocumentTree::new(doc_id, config).unwrap();

        let text = "This is the first sentence. This is a second sentence with more details. This is the final sentence.";
        let summary = tree.generate_extractive_summary(text).unwrap();

        assert!(!summary.is_empty());
        assert!(summary.len() <= tree.config.max_summary_length);
    }

    #[test]
    fn test_json_serialization() {
        let config = HierarchicalConfig::default();
        let doc_id = DocumentId::new("test_doc".to_string());
        let tree = DocumentTree::new(doc_id, config).unwrap();

        let json = tree.to_json().unwrap();
        assert!(json.contains("test_doc"));

        let loaded_tree = DocumentTree::from_json(&json).unwrap();
        assert_eq!(loaded_tree.document_id.to_string(), "test_doc");
    }
}
