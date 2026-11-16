use graphrag_core::{pipeline_executor::PipelineExecutor, Config, GraphRAG};

#[test]
fn sync_build_saves_and_reopens_without_reindexing() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    let mut config = Config::quick(&workspace);
    // Leave room for both chunk and graph facts; this also exercises stable ordering
    // across the in-process graph and the persisted/reopened graph.
    config.retrieval.top_k = 20;

    let mut first = GraphRAG::new(config.clone()).unwrap();
    first.initialize().unwrap();
    first
        .add_document_from_text(
            "Maria Rossi directs the Prisma project at Aurora Laboratory in Turin.",
        )
        .unwrap();
    first.build_graph_sync().unwrap();
    assert!(first.has_documents());
    assert!(first.has_graph());
    let first_answer = first.ask_sync("Who directs the Prisma project?").unwrap();

    let mut reopened = GraphRAG::new(config).unwrap();
    reopened.initialize().unwrap();
    assert!(reopened.has_documents());
    assert!(reopened.has_graph());
    let answer = reopened
        .ask_sync("Who directs the Prisma project?")
        .unwrap();
    assert!(answer.contains("Maria Rossi"), "answer was: {answer}");
    assert_eq!(
        answer, first_answer,
        "retrieval ordering changed after reopen"
    );
}

#[test]
fn sync_reports_missing_graph() {
    let mut graph = GraphRAG::new(Config::default()).unwrap();
    let error = graph.query_internal_sync("anything").unwrap_err();
    assert!(error.to_string().contains("Retrieval system"));
}

#[test]
fn sync_rejects_non_local_providers() {
    let mut config = Config::default();
    config.llm.enabled = true;
    let mut graph = GraphRAG::new(config).unwrap();
    graph.initialize().unwrap();
    graph
        .add_document_from_text("Maria Rossi dirige il progetto Prisma.")
        .unwrap();
    assert!(graph.query_internal_sync("anything").is_err());

    #[cfg(feature = "ureq")]
    {
        let mut config = Config::default();
        config.embeddings.backend = "llama_cpp".to_string();
        config.embeddings.dimension = 384;
        config.embeddings.fallback_to_hash = false;
        config.embeddings.api_endpoint = Some("http://127.0.0.1:9/v1/embeddings".to_string());
        let mut graph = GraphRAG::new(config).unwrap();
        graph.initialize().unwrap();
        graph
            .add_document_from_text("Maria Rossi dirige il progetto Prisma.")
            .unwrap();
        assert!(graph.ask_sync("anything").is_err());
    }
}

#[test]
fn sync_factory_loads_config_and_document() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.json");
    let document_path = temp.path().join("document.txt");
    let config = Config::default();
    config
        .to_file(config_path.to_str().unwrap())
        .expect("write canonical config");
    std::fs::write(&document_path, "Maria Rossi dirige il progetto Prisma.").unwrap();

    let mut graph = GraphRAG::from_config_and_document_sync(&config_path, &document_path).unwrap();
    assert!(graph.has_graph());
    assert!(graph
        .ask_sync("Chi dirige Prisma?")
        .unwrap()
        .contains("Maria Rossi"));
}

#[test]
fn sync_pipeline_reports_built_graph() {
    let mut graph = GraphRAG::new(Config::default()).unwrap();
    graph.initialize().unwrap();
    let mut pipeline = PipelineExecutor::new(&mut graph);
    let report = pipeline
        .ingest_and_build_sync("Maria Rossi dirige il progetto Prisma.")
        .unwrap();
    assert!(report.chunks_processed > 0);
    assert!(report.entity_count > 0);
}

#[test]
fn corrupted_workspace_is_returned_as_an_error() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(workspace.join("graph.json"), "{ definitely not graph }").unwrap();

    let mut graph = GraphRAG::new(Config::quick(&workspace)).unwrap();
    let error = graph.initialize().expect_err("corrupt workspace must fail");
    assert!(error.to_string().contains("graph") || error.to_string().contains("JSON"));
}
