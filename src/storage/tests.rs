//! Tests for storage implementations using the trait testing framework

#[cfg(test)]
mod storage_tests {
    use crate::storage::MemoryStorage;
    use crate::test_storage_implementation;

    // This will generate all the storage tests for MemoryStorage
    test_storage_implementation!(MemoryStorage);

    // Additional MemoryStorage-specific tests
    #[test]
    fn test_memory_storage_clear() {
        use crate::core::traits::Storage;
        use crate::core::{Entity, EntityId};

        let mut storage = MemoryStorage::new();

        // Add some data
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        storage.store_entity(entity).unwrap();
        assert_eq!(storage.entity_count(), 1);

        // Clear and verify
        storage.clear();
        assert_eq!(storage.entity_count(), 0);
        assert_eq!(storage.document_count(), 0);
        assert_eq!(storage.chunk_count(), 0);
    }

    #[test]
    fn test_memory_storage_has_methods() {
        use crate::core::traits::Storage;
        use crate::core::{Entity, EntityId};

        let mut storage = MemoryStorage::new();

        let entity = Entity::new(
            EntityId::new("test_entity".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        let id = storage.store_entity(entity).unwrap();
        assert!(storage.has_entity(&id));
        assert!(!storage.has_entity("nonexistent"));
    }

    #[test]
    fn test_memory_storage_auto_id_generation() {
        use crate::core::traits::Storage;
        use crate::core::{Entity, EntityId};

        let mut storage = MemoryStorage::new();

        // Entity with empty ID should get auto-generated ID
        let entity = Entity::new(
            EntityId::new("".to_string()),
            "Auto ID Entity".to_string(),
            "Person".to_string(),
            0.8,
        );

        let id = storage.store_entity(entity).unwrap();
        assert!(id.starts_with("entity_"));

        let retrieved = storage.retrieve_entity(&id).unwrap().unwrap();
        assert_eq!(retrieved.name, "Auto ID Entity");
    }
}
