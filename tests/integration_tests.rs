use graphrag_rs::{
    config::Config,
    core::{Document, DocumentId, KnowledgeGraph},
    graph::GraphBuilder,
    GraphRAG, Result,
};

#[test]
fn test_end_to_end_workflow() -> Result<()> {
    // Create configuration
    let config = Config::default();

    // Create GraphRAG instance
    let mut graphrag = GraphRAG::new(config)?;
    graphrag.initialize()?;

    // Create test documents
    let documents = vec![
        Document::new(
            DocumentId::new("doc1".to_string()),
            "Machine Learning Basics".to_string(),
            "Machine learning is a subset of artificial intelligence. \
             John Smith is a researcher at MIT who works on neural networks. \
             The university is located in Cambridge, Massachusetts."
                .to_string(),
        ),
        Document::new(
            DocumentId::new("doc2".to_string()),
            "Deep Learning Applications".to_string(),
            "Deep learning has applications in computer vision and natural language processing. \
             Jane Doe leads the AI team at Google. The company is headquartered in Mountain View."
                .to_string(),
        ),
    ];

    // Add documents to the system
    for document in documents {
        graphrag.add_document(document)?;
    }

    // Build the knowledge graph
    graphrag.build_graph()?;

    // Query the system
    let results = graphrag.query("machine learning research")?;
    assert!(!results.is_empty());

    Ok(())
}

#[test]
fn test_graph_construction() -> Result<()> {
    let mut builder = GraphBuilder::new(500, 100, 0.7, 0.8, 5)?;

    let documents = vec![
        Document::new(
            DocumentId::new("doc1".to_string()),
            "Research Paper".to_string(),
            "Dr. Alice Johnson conducted research at Stanford University. \
             Her work focuses on natural language processing and machine learning."
                .to_string(),
        ),
        Document::new(
            DocumentId::new("doc2".to_string()),
            "Company Profile".to_string(),
            "OpenAI is an artificial intelligence company founded by Sam Altman. \
             The organization is based in San Francisco, California."
                .to_string(),
        ),
    ];

    let graph = builder.build_graph(documents)?;
    let stats = builder.analyze_graph(&graph);

    // Verify that entities were extracted
    assert!(stats.entity_count > 0);
    assert_eq!(stats.document_count, 2);
    assert!(stats.chunk_count >= 2);

    // Verify that different entity types were found
    assert!(!stats.entity_types.is_empty());

    Ok(())
}

#[test]
fn test_config_serialization() -> Result<()> {
    let config = Config::default();

    // Test serialization to file and back
    config.to_file("test_config.json")?;
    let _deserialized = Config::from_file("test_config.json")?;

    // Clean up test file
    std::fs::remove_file("test_config.json").ok();

    Ok(())
}

#[test]
fn test_empty_knowledge_graph() {
    let graph = KnowledgeGraph::new();

    assert_eq!(graph.entities().count(), 0);
    assert_eq!(graph.documents().count(), 0);
    assert_eq!(graph.chunks().count(), 0);
}

#[test]
fn test_document_processing_pipeline() -> Result<()> {
    use graphrag_rs::text::TextProcessor;

    let processor = TextProcessor::new(200, 50)?;

    let document = Document::new(
        DocumentId::new("test_doc".to_string()),
        "Test Document".to_string(),
        "This is the first sentence. This is the second sentence. \
         This is a longer sentence that should demonstrate the chunking behavior. \
         Finally, this is the last sentence in our test document."
            .to_string(),
    );

    let chunks = processor.chunk_text(&document)?;

    assert!(!chunks.is_empty());

    // Each chunk should be reasonably sized
    for chunk in &chunks {
        assert!(chunk.content.len() <= 250); // Some tolerance for sentence boundaries
        assert!(!chunk.content.trim().is_empty());
    }

    // Test text cleaning
    let cleaned = processor.clean_text("  This   has    extra    spaces  ");
    assert_eq!(cleaned, "This has extra spaces");

    // Test keyword extraction
    let keywords = processor.extract_keywords(
        "machine learning artificial intelligence data science neural networks",
        3,
    );
    assert!(!keywords.is_empty());
    assert!(keywords.len() <= 3);

    Ok(())
}

#[test]
fn test_entity_extraction() -> Result<()> {
    use graphrag_rs::{
        core::{ChunkId, DocumentId, TextChunk},
        entity::EntityExtractor,
    };

    let extractor = EntityExtractor::new(0.6)?;

    let chunk = TextChunk::new(
        ChunkId::new("test_chunk".to_string()),
        DocumentId::new("test_doc".to_string()),
        "Dr. John Smith works at Microsoft Corporation in Seattle, Washington. \
         The company was founded by Bill Gates and Paul Allen."
            .to_string(),
        0,
        97,
    );

    let entities = extractor.extract_from_chunk(&chunk)?;

    assert!(!entities.is_empty());

    // Should extract various types of entities
    let has_person = entities.iter().any(|e| e.entity_type == "PERSON");
    let has_organization = entities.iter().any(|e| e.entity_type == "ORGANIZATION");
    let has_location = entities.iter().any(|e| e.entity_type == "LOCATION");

    assert!(has_person || has_organization || has_location);

    // Test relationship extraction
    let relationships = extractor.extract_relationships(&entities, &chunk)?;

    // Should find some relationships between co-occurring entities
    if entities.len() >= 2 {
        assert!(!relationships.is_empty());
    }

    Ok(())
}

#[test]
fn test_vector_operations() -> Result<()> {
    use graphrag_rs::vector::{VectorIndex, VectorUtils};

    let mut index = VectorIndex::new();

    // Add test vectors
    index.add_vector("vec1".to_string(), vec![1.0, 0.0, 0.0])?;
    index.add_vector("vec2".to_string(), vec![0.0, 1.0, 0.0])?;
    index.add_vector("vec3".to_string(), vec![0.8, 0.6, 0.0])?;

    index.build_index()?;

    // Test search
    let query = vec![1.0, 0.0, 0.0];
    let results = index.search(&query, 2)?;

    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    // Most similar should be vec1
    assert_eq!(results[0].0, "vec1");

    // Test utility functions
    let vec_a = vec![1.0, 0.0];
    let vec_b = vec![0.0, 1.0];
    let similarity = VectorUtils::cosine_similarity(&vec_a, &vec_b);
    assert!((similarity - 0.0).abs() < 0.001);

    let distance = VectorUtils::euclidean_distance(&vec_a, &vec_b);
    assert!((distance - 2.0_f32.sqrt()).abs() < 0.001);

    // Test normalization
    let mut vector = vec![3.0, 4.0];
    VectorUtils::normalize(&mut vector);
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001);

    Ok(())
}
