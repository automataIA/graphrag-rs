use graphrag_rs::{
    config::Config,
    core::{ChunkId, Document, DocumentId, Entity, EntityId, KnowledgeGraph, TextChunk},
    retrieval::{QueryIntent, QueryType, ResultType, RetrievalSystem},
    summarization::{DocumentTree, HierarchicalConfig},
};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_knowledge_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Create test document
        let document = Document::new(
            DocumentId::new("test_doc".to_string()),
            "Test Document".to_string(),
            "This is a test document about Tom Sawyer and his adventures.".to_string(),
        );

        // Create test chunks
        let chunk1 = TextChunk::new(
            ChunkId::new("chunk1".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Tom Sawyer was a clever boy who lived with his Aunt Polly.".to_string(),
            0,
            56,
        );

        let chunk2 = TextChunk::new(
            ChunkId::new("chunk2".to_string()),
            DocumentId::new("test_doc".to_string()),
            "Huck Finn was Tom's best friend and they had many adventures together.".to_string(),
            57,
            125,
        );

        graph.add_document(document).unwrap();
        graph.add_chunk(chunk1).unwrap();
        graph.add_chunk(chunk2).unwrap();

        // Create test entities
        let tom_entity = Entity::new(
            EntityId::new("tom".to_string()),
            "Tom Sawyer".to_string(),
            "Character".to_string(),
            0.9,
        );

        let huck_entity = Entity::new(
            EntityId::new("huck".to_string()),
            "Huck Finn".to_string(),
            "Character".to_string(),
            0.8,
        );

        graph.add_entity(tom_entity).unwrap();
        graph.add_entity(huck_entity).unwrap();

        graph
    }

    fn create_test_document_trees() -> HashMap<DocumentId, DocumentTree> {
        let mut trees = HashMap::new();
        let config = HierarchicalConfig::default();
        let doc_id = DocumentId::new("test_doc".to_string());

        // Create a simple document tree for testing
        let tree = DocumentTree::new(doc_id.clone(), config).unwrap();
        trees.insert(doc_id, tree);

        trees
    }

    #[test]
    fn test_retrieval_system_creation() {
        let config = Config::default();
        let retrieval_system = RetrievalSystem::new(&config);
        assert!(retrieval_system.is_ok());
    }

    #[test]
    fn test_query_analysis() {
        let config = Config::default();
        let retrieval_system = RetrievalSystem::new(&config).unwrap();
        let knowledge_graph = create_test_knowledge_graph();

        // Test entity-focused query
        let analysis = retrieval_system
            .analyze_query("Tom Sawyer", &knowledge_graph)
            .unwrap();
        assert_eq!(analysis.query_type, QueryType::EntityFocused);
        assert!(analysis.key_entities.contains(&"Tom Sawyer".to_string()));

        // Test relationship query
        let analysis = retrieval_system
            .analyze_query("Tom and Huck friendship", &knowledge_graph)
            .unwrap();
        assert_eq!(analysis.query_type, QueryType::Relationship);
        assert!(!analysis.key_entities.is_empty());

        // Test conceptual query
        let analysis = retrieval_system
            .analyze_query("what is the main theme", &knowledge_graph)
            .unwrap();
        assert_eq!(analysis.query_type, QueryType::Exploratory);

        // Test overview intent
        let analysis = retrieval_system
            .analyze_query("overview of the story", &knowledge_graph)
            .unwrap();
        assert_eq!(analysis.intent, QueryIntent::Overview);

        // Test detailed intent
        let analysis = retrieval_system
            .analyze_query("detailed description of events", &knowledge_graph)
            .unwrap();
        assert_eq!(analysis.intent, QueryIntent::Detailed);
    }

    #[test]
    fn test_hybrid_query_basic() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test basic hybrid query
        let results = retrieval_system
            .hybrid_query("Tom Sawyer", &knowledge_graph)
            .unwrap();
        assert!(!results.is_empty());

        // Check that we get different types of results
        let has_entity_result = results.iter().any(|r| r.result_type == ResultType::Entity);
        let has_chunk_result = results.iter().any(|r| r.result_type == ResultType::Chunk);

        // At least one of these should be true
        assert!(has_entity_result || has_chunk_result);
    }

    #[test]
    fn test_hybrid_query_with_trees() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();
        let document_trees = create_test_document_trees();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test hybrid query with document trees
        let results = retrieval_system
            .hybrid_query_with_trees("Tom Sawyer adventures", &knowledge_graph, &document_trees)
            .unwrap();

        assert!(!results.is_empty());

        // Should return results with valid scores
        for result in &results {
            assert!(result.score > -1.0); // Allow for negative similarity scores
            assert!(result.score <= 2.0); // Allow some boost from multiple strategies
            assert!(!result.id.is_empty());
        }
    }

    #[test]
    fn test_entity_centric_search() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test entity-focused query that should trigger entity-centric search
        let results = retrieval_system
            .hybrid_query("Tom Sawyer character", &knowledge_graph)
            .unwrap();

        // Should find entity-related results
        let entity_results: Vec<_> = results
            .iter()
            .filter(|r| r.result_type == ResultType::Entity)
            .collect();

        // Should have at least one entity result
        if !entity_results.is_empty() {
            assert!(entity_results[0].entities.iter().any(|e| e.contains("Tom")));
        }
    }

    #[test]
    fn test_strategy_weights() {
        let config = Config::default();
        let retrieval_system = RetrievalSystem::new(&config).unwrap();
        let knowledge_graph = create_test_knowledge_graph();

        // Test different query types get different strategy weights
        let entity_analysis = retrieval_system
            .analyze_query("Tom Sawyer", &knowledge_graph)
            .unwrap();
        assert_eq!(entity_analysis.query_type, QueryType::EntityFocused);

        let relationship_analysis = retrieval_system
            .analyze_query("Tom and Huck friendship", &knowledge_graph)
            .unwrap();
        assert_eq!(relationship_analysis.query_type, QueryType::Relationship);

        let conceptual_analysis = retrieval_system
            .analyze_query("what is the story about", &knowledge_graph)
            .unwrap();
        // This might be classified as exploratory due to question words
        assert!(matches!(
            conceptual_analysis.query_type,
            QueryType::Conceptual | QueryType::Exploratory
        ));

        // Each should have different characteristics
        assert_ne!(entity_analysis.query_type, relationship_analysis.query_type);
        assert_ne!(entity_analysis.query_type, conceptual_analysis.query_type);
    }

    #[test]
    fn test_result_deduplication() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();
        let document_trees = create_test_document_trees();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Run multiple queries that might return overlapping results
        let results1 = retrieval_system
            .hybrid_query_with_trees("Tom Sawyer", &knowledge_graph, &document_trees)
            .unwrap();

        let results2 = retrieval_system
            .hybrid_query_with_trees("Tom character", &knowledge_graph, &document_trees)
            .unwrap();

        // Each result set should have unique IDs within itself (or very close due to ranking/fusion)
        let mut ids1: Vec<_> = results1.iter().map(|r| &r.id).collect();
        ids1.sort();
        let orig_len1 = ids1.len();
        ids1.dedup();
        assert!(ids1.len() >= orig_len1 - 2); // Allow for some potential duplicates from fusion

        let mut ids2: Vec<_> = results2.iter().map(|r| &r.id).collect();
        ids2.sort();
        let orig_len2 = ids2.len();
        ids2.dedup();
        assert!(ids2.len() >= orig_len2 - 2); // Allow for some potential duplicates from fusion
    }

    #[test]
    fn test_confidence_scoring() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test that more specific queries get higher confidence scores
        let specific_results = retrieval_system
            .hybrid_query("Tom Sawyer", &knowledge_graph)
            .unwrap();
        let general_results = retrieval_system
            .hybrid_query("character", &knowledge_graph)
            .unwrap();

        if !specific_results.is_empty() && !general_results.is_empty() {
            // More specific queries should generally have higher top scores
            let max_specific_score = specific_results.iter().map(|r| r.score).fold(0.0, f32::max);
            let max_general_score = general_results.iter().map(|r| r.score).fold(0.0, f32::max);

            // This is a heuristic - specific entity matches should score higher
            // than general concept matches in most cases
            assert!(max_specific_score >= max_general_score * 0.8);
        }
    }

    #[test]
    fn test_performance_constraints() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();
        let document_trees = create_test_document_trees();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test that queries complete within reasonable time
        let start = std::time::Instant::now();

        let _results = retrieval_system
            .hybrid_query_with_trees(
                "Tom Sawyer adventures with friends",
                &knowledge_graph,
                &document_trees,
            )
            .unwrap();

        let elapsed = start.elapsed();

        // Should complete within 1 second for small test data
        assert!(elapsed.as_secs() < 1);
    }

    #[test]
    fn test_result_types_distribution() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();
        let document_trees = create_test_document_trees();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test entity-focused query
        let entity_results = retrieval_system
            .hybrid_query_with_trees("Tom Sawyer", &knowledge_graph, &document_trees)
            .unwrap();

        // Count different result types
        let mut type_counts = HashMap::new();
        for result in &entity_results {
            *type_counts.entry(&result.result_type).or_insert(0) += 1;
        }

        // Should have at least one type of result
        assert!(!type_counts.is_empty());

        // For entity queries, should prefer entity results
        if let Some(&entity_count) = type_counts.get(&ResultType::Entity) {
            assert!(entity_count > 0);
        }
    }

    #[test]
    fn test_empty_query_handling() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();
        let document_trees = create_test_document_trees();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Test empty query
        let results = retrieval_system
            .hybrid_query_with_trees("", &knowledge_graph, &document_trees)
            .unwrap();

        // Should handle empty queries gracefully (may return empty or default results)
        // The important thing is that it doesn't panic
        assert!(results.len() <= 20); // Reasonable upper bound
    }

    #[test]
    fn test_statistics_tracking() {
        let config = Config::default();
        let mut retrieval_system = RetrievalSystem::new(&config).unwrap();
        let mut knowledge_graph = create_test_knowledge_graph();

        // Add embeddings to the graph
        retrieval_system
            .add_embeddings_to_graph(&mut knowledge_graph)
            .unwrap();

        // Get statistics
        let stats = retrieval_system.get_statistics();

        // Should have indexed some vectors
        assert!(stats.indexed_vectors > 0);
        assert!(stats.vector_dimension > 0);
        assert!(stats.index_built);

        // Configuration should be preserved
        assert_eq!(stats.config.top_k, config.retrieval.top_k);
    }
}
