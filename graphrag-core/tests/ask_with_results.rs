//! `ask_with_results` returns the answer and the retrieval results from a single retrieval pass.
#![cfg(feature = "async")]

use graphrag_core::{Config, GraphRAG};

#[tokio::test]
async fn ask_with_results_returns_answer_and_matching_sources() {
    let mut graphrag = GraphRAG::new(Config::default()).unwrap();
    graphrag.initialize().unwrap();
    graphrag
        .add_document_from_text(
            "Maria Rossi directs the Prisma project at Aurora Laboratory in Turin.",
        )
        .unwrap();

    let (answer, results) = graphrag
        .ask_with_results("Who directs the Prisma project?")
        .await
        .unwrap();

    assert!(answer.contains("Maria Rossi"), "answer was: {answer}");
    assert!(!results.is_empty());
    assert_eq!(
        answer,
        graphrag
            .ask("Who directs the Prisma project?")
            .await
            .unwrap()
    );
    assert_eq!(
        results.len(),
        graphrag
            .search("Who directs the Prisma project?")
            .await
            .unwrap()
            .len()
    );
}
