//! Logic form retrieval for ROGRAG system
//!
//! Implements structured reasoning using logic forms to represent queries
//! and perform precise retrieval based on semantic relationships.
//!
//! Decomposed (Phase 4 file split) from the previous monolithic `logic_form.rs`
//! (1517 LOC) into focused sub-modules. Public items are re-exported here so
//! external paths (`crate::rograg::logic_form::LogicFormRetriever`, etc.) are
//! unchanged.

mod executor;
mod parser;
mod retriever;
mod types;

#[cfg(feature = "rograg")]
pub use executor::*;
#[cfg(feature = "rograg")]
pub use parser::*;
#[cfg(feature = "rograg")]
pub use retriever::*;
#[cfg(feature = "rograg")]
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Entity, EntityId, KnowledgeGraph};

    #[cfg(feature = "rograg")]
    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        let entity1 = Entity {
            id: EntityId::new("entity_1".to_string()),
            name: "Entity Name".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 1.0,
            mentions: vec![],
            embedding: None,
            first_mentioned: None,
            last_mentioned: None,
            temporal_validity: None,
        };

        let entity2 = Entity {
            id: EntityId::new("entity_2".to_string()),
            name: "Second Entity".to_string(),
            entity_type: "ENTITY".to_string(),
            confidence: 1.0,
            mentions: vec![],
            embedding: None,
            first_mentioned: None,
            last_mentioned: None,
            temporal_validity: None,
        };

        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();

        graph
    }


    #[cfg(feature = "rograg")]
    #[test]
    fn test_executor_is_query() {
        let executor = LogicFormExecutor::new();
        let graph = create_test_graph();

        let logic_form = LogicFormQuery {
            predicate: Predicate::Is,
            arguments: vec![Argument {
                arg_type: ArgumentType::Entity,
                value: "Entity Name".to_string(),
                variable: Some("X".to_string()),
                constraints: vec![],
            }],
            constraints: vec![],
            query_type: LogicQueryType::Select,
            confidence: 0.8,
        };

        let bindings = executor.execute(&logic_form, &graph).unwrap();
        assert!(!bindings.is_empty());
        assert!(bindings[0].confidence > 0.0);
    }

    #[cfg(feature = "rograg")]
    #[test]
    fn test_name_similarity() {
        let executor = LogicFormExecutor::new();

        assert_eq!(
            executor.calculate_name_similarity("Entity Name", "Entity Name"),
            1.0
        );
        assert!(executor.calculate_name_similarity("Entity", "Entity Name") > 0.5);
        assert!(executor.calculate_name_similarity("Completely Different", "Entity Name") < 0.5);
    }
}
