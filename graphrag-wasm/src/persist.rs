//! Lightweight IndexedDB wrapper around [`crate::storage::IndexedDBStore`].
//!
//! Persists the `Document` set the user has loaded so it survives page reload.
//! Best-effort: any failure is logged to the console and the function returns
//! gracefully rather than propagating.

use crate::storage::IndexedDBStore;
use crate::Document;

const DB_NAME: &str = "graphrag-wasm";
const DB_VERSION: u32 = 1;
const STORE: &str = "documents";

/// Open the IndexedDB database. Returns `None` if the browser does not support it.
pub async fn open_store() -> Option<IndexedDBStore> {
    match IndexedDBStore::new(DB_NAME, DB_VERSION).await {
        Ok(store) => Some(store),
        Err(e) => {
            web_sys::console::warn_1(
                &format!("IndexedDB unavailable, persistence disabled: {e}").into(),
            );
            None
        },
    }
}

/// Persist a single document (overwrites if the id already exists).
pub async fn save_document(store: &IndexedDBStore, doc: &Document) {
    if let Err(e) = store.put(STORE, &doc.id, doc).await {
        web_sys::console::warn_1(&format!("failed to persist doc {}: {e}", doc.id).into());
    }
}

/// Delete a single document by id.
pub async fn delete_document(store: &IndexedDBStore, id: &str) {
    if let Err(e) = store.delete(STORE, id).await {
        web_sys::console::warn_1(&format!("failed to delete doc {id}: {e}").into());
    }
}

/// Load every document from the store. Returns an empty vector on error.
pub async fn load_all_documents(store: &IndexedDBStore) -> Vec<Document> {
    match store.get_all_batched::<Document>(STORE, None).await {
        Ok(docs) => docs,
        Err(e) => {
            web_sys::console::warn_1(&format!("failed to load documents: {e}").into());
            Vec::new()
        },
    }
}
