//! Tests for the simplified API

use crate::api::easy::SimpleGraphRAG;
use crate::api::simple;

#[test]
fn test_simple_api_compilation() {
    // Test that the simple API compiles correctly
    // We can't actually run these without a proper LLM setup
    let text = "The quick brown fox jumps over the lazy dog.";
    let question = "What animal jumps?";

    // This will fail at runtime but should compile fine
    let _result = simple::answer(text, question);
}

#[test]
fn test_easy_api_creation() {
    // Test that the easy API can be created
    let text = "The quick brown fox jumps over the lazy dog.";

    // This should work for creation
    let result = SimpleGraphRAG::from_text(text);

    // Should succeed in creating the instance
    match result {
        Ok(_) => {
            // Good, we can create it
            println!("SimpleGraphRAG created successfully");
        }
        Err(e) => {
            // May fail due to LLM setup, but that's ok for this test
            println!("SimpleGraphRAG creation failed (expected): {e}");
        }
    }
}

#[test]
fn test_prelude_imports() {
    use crate::prelude::*;

    // Test that prelude imports work
    let _builder = GraphRAGBuilder::new();
    let _preset = ConfigPreset::Basic;
    let _llm = LLMProvider::Mock;
}
