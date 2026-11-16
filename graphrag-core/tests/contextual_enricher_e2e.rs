//! End-to-end tests for ContextualEnricher with a real llama.cpp server
//!
//! These tests require a running `llama-server` and are **skipped by default**.
//!
//! ## Running the tests
//!
//! ```bash
//! llama-server -m Qwen3.5-9B-UD-Q5_K_XL.gguf --alias qwen3.5-9b \
//!     --port 8080 -c 32768 -np 4 -fa on
//!
//! ENABLE_LLM_TESTS=1 cargo test -p graphrag-core --features llm \
//!     --test contextual_enricher_e2e -- --ignored --nocapture
//! ```
//!
//! `GRAPHRAG_LLM_BASE_URL` / `GRAPHRAG_LLM_MODEL` override the defaults.
//!
//! ## What these tests verify
//!
//! 1. **Correctness**: enriched chunk contains the original text + LLM-generated context
//! 2. **Prefix cache**: a second request sharing the document prefix reuses the
//!    server KV cache (`timings.cache_n` > 0)
//! 3. **Context sizing**: `calculate_num_ctx` stays within sensible bounds (offline)

#[cfg(all(feature = "llm", feature = "async"))]
mod e2e {
    use graphrag_core::{
        core::{ChunkId, Document, DocumentId, TextChunk},
        llm::{LlmClient, LlmConfig, LlmGenerationParams},
        text::contextual_enricher::{ContextualEnricher, ContextualEnricherConfig},
    };
    use std::time::{Duration, Instant};

    /// Return true only if the user explicitly opted into live LLM tests.
    ///
    /// We use an env var instead of relying solely on `#[ignore]` so CI
    /// can never accidentally run these even with `-- --include-ignored`.
    fn llm_tests_enabled() -> bool {
        std::env::var("ENABLE_LLM_TESTS").is_ok()
    }

    /// LlmConfig for the local llama-server; env vars override URL and model.
    fn make_llm_config() -> LlmConfig {
        let defaults = LlmConfig::default();
        LlmConfig {
            enabled: true,
            base_url: std::env::var("GRAPHRAG_LLM_BASE_URL").unwrap_or(defaults.base_url.clone()),
            chat_model: std::env::var("GRAPHRAG_LLM_MODEL").unwrap_or(defaults.chat_model.clone()),
            timeout_seconds: 300,
            max_retries: 1,
            // Measure the server, not the in-process response cache.
            enable_caching: false,
            ..defaults
        }
    }

    /// Build a small Document with N chunks for testing.
    ///
    /// Content is the Plato Symposium excerpt (first ~500 words) repeated across chunks.
    fn make_test_document_and_chunks(n_chunks: usize) -> (Document, Vec<TextChunk>) {
        let content = "\
Socrates, Apollodorus, and their companions were walking to Athens when they \
encountered a friend who asked about the speeches in praise of love. The narrative \
begins with Apollodorus recounting what Aristodemus told him about a dinner party \
hosted by Agathon. Socrates arrived late, having fallen into a trance of thought \
along the way, as was his habit. The guests agreed that instead of indulging in \
excessive drinking, they would take turns giving speeches in praise of Eros, the \
god of love. Phaedrus spoke first, arguing that Eros is the most ancient of gods \
and the greatest source of virtue and happiness. He cited examples from mythology \
to show that love inspires courage and self-sacrifice, as seen in Alcestis and \
Achilles. Pausanias distinguished between Common Eros, which seeks physical \
gratification, and Heavenly Eros, which seeks virtue and wisdom in the beloved. \
He argued that true love is only possible between a man and a young man who has \
developed his intellect. Eryximachus, the physician, extended the concept of Eros \
to nature and medicine, arguing that love governs the harmony of all things, \
including music, astronomy, and the seasons."
            .to_string();

        let doc_id = DocumentId::new("test-symposium".to_string());
        let document = Document::new(
            doc_id.clone(),
            "Symposium excerpt".to_string(),
            content.clone(),
        );

        // Split content into N roughly equal chunks
        let total = content.len();
        let chunk_size = total / n_chunks;
        let mut chunks = Vec::with_capacity(n_chunks);

        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = if i + 1 == n_chunks {
                total
            } else {
                (i + 1) * chunk_size
            };
            // Snap to char boundary
            let end = content
                .char_indices()
                .map(|(pos, _)| pos)
                .rfind(|&p| p <= end)
                .unwrap_or(end);

            let chunk_text = content[start..end].to_string();
            if chunk_text.trim().is_empty() {
                continue;
            }

            chunks.push(TextChunk::new(
                ChunkId::new(format!("chunk_{i}")),
                doc_id.clone(),
                chunk_text,
                start,
                end,
            ));
        }

        (document, chunks)
    }

    // =========================================================================
    // TEST 1 — Correctness
    // =========================================================================

    /// Verify that `ContextualEnricher` produces valid enriched content.
    ///
    /// An enriched chunk must:
    /// - Contain the original chunk text verbatim
    /// - Have more content than the original (LLM added a context preamble)
    /// - Not be empty
    #[tokio::test]
    #[ignore] // Run with: ENABLE_LLM_TESTS=1 cargo test -- --ignored --nocapture
    async fn test_enriched_chunk_contains_original_and_context() {
        if !llm_tests_enabled() {
            println!("Skipping: set ENABLE_LLM_TESTS=1 to run");
            return;
        }

        let (document, chunks) = make_test_document_and_chunks(3);
        let enricher = ContextualEnricher::with_defaults(make_llm_config());

        println!("\n=== Contextual Enricher E2E — Correctness ===");
        println!(
            "Document: '{}' ({} chars)",
            document.title,
            document.content.len()
        );
        println!("Chunks: {}", chunks.len());
        println!(
            "Required context: {}",
            enricher.calculate_num_ctx(&document.content, &chunks)
        );

        let enriched = enricher
            .enrich_document_chunks(&document, &chunks)
            .await
            .expect("Enrichment failed");

        assert_eq!(
            enriched.len(),
            chunks.len(),
            "Number of chunks should be preserved"
        );

        for (original, enriched_chunk) in chunks.iter().zip(enriched.iter()) {
            println!(
                "\n--- Chunk {} ---\nOriginal ({} chars):\n{}\n\nEnriched ({} chars):\n{}",
                original.id,
                original.content.len(),
                &original.content[..original.content.len().min(120)],
                enriched_chunk.content.len(),
                &enriched_chunk.content[..enriched_chunk.content.len().min(200)],
            );

            // The original text must still be present in the enriched chunk
            assert!(
                enriched_chunk.content.contains(original.content.trim()),
                "Enriched chunk {} does not contain original text",
                original.id
            );

            // The LLM should have added at least some context (>= 20 chars extra)
            assert!(
                enriched_chunk.content.len() > original.content.len() + 20,
                "Chunk {} was not meaningfully enriched (added only {} chars)",
                original.id,
                enriched_chunk.content.len() as i64 - original.content.len() as i64,
            );
        }

        println!("\n✓ All chunks correctly enriched with LLM context");
    }

    // =========================================================================
    // TEST 2 — Server prefix cache
    // =========================================================================

    /// Two enrichment prompts for the same document share the `<document>` prefix.
    /// llama-server (`cache_prompt`, default on) must reuse that prefix from the
    /// KV cache on the second request, reported as `timings.cache_n`.
    ///
    /// With `-np > 1` the second request may land on another slot; llama-server
    /// prefers the slot with the longest common prefix, so reuse is still expected.
    #[tokio::test]
    #[ignore] // Run with: ENABLE_LLM_TESTS=1 cargo test -- --ignored --nocapture
    async fn test_server_prefix_cache_reuse() {
        if !llm_tests_enabled() {
            println!("Skipping: set ENABLE_LLM_TESTS=1 to run");
            return;
        }

        let (document, chunks) = make_test_document_and_chunks(3);
        let client = LlmClient::new(make_llm_config());
        let params = LlmGenerationParams {
            max_tokens: Some(60),
            temperature: Some(0.0),
            ..Default::default()
        };

        let mut timings = Vec::new();
        let mut cached = Vec::new();
        for chunk in &chunks {
            let prompt = ContextualEnricher::build_prompt(&document.content, &chunk.content);
            let start = Instant::now();
            let response = client
                .generate_with_full_response(&prompt, params.clone())
                .await
                .expect("generation failed");
            timings.push(start.elapsed());
            cached.push(response.cached_prompt_tokens.unwrap_or(0));
        }

        println!("\n=== Prefix cache ===");
        for (i, (elapsed, hit)) in timings.iter().zip(&cached).enumerate() {
            println!("chunk {i}: {:?}  cache_n={hit}", elapsed);
        }
        let doc_tokens = ContextualEnricher::estimate_tokens(&document.content) as u64;
        assert!(
            cached[1..].iter().all(|&hit| hit >= doc_tokens / 2),
            "warm requests should reuse >= half of the document prefix ({doc_tokens} est. tokens): {cached:?}"
        );
        assert!(timings.iter().all(|t| *t < Duration::from_secs(300)));
    }

    // =========================================================================
    // TEST 3 — num_ctx calculation sanity
    // =========================================================================

    /// Verify that `calculate_num_ctx` produces values within sensible bounds
    /// and that it grows proportionally with document length.
    ///
    /// This test does NOT call the LLM — it only exercises the calculation logic.
    /// Listed here (not in unit tests) because it uses document-size fixtures
    /// that are representative of real use.
    #[test]
    fn test_num_ctx_calculation_sanity() {
        let enricher = ContextualEnricher::with_defaults(LlmConfig::default());

        // Tiny document: should floor to minimum (4096)
        let tiny_doc = "A".repeat(100);
        let tiny_chunk = TextChunk::new(
            ChunkId::new("c0".to_string()),
            DocumentId::new("d".to_string()),
            "small chunk".to_string(),
            0,
            11,
        );
        let ctx_tiny = enricher.calculate_num_ctx(&tiny_doc, &[tiny_chunk]);
        assert_eq!(ctx_tiny, 4096, "Tiny docs should floor to 4096");

        // Plato Symposium (~45k tokens = ~180k chars): should exceed 45k
        let large_doc = "word ".repeat(36_000); // ~45k tokens
        let large_chunk = TextChunk::new(
            ChunkId::new("c0".to_string()),
            DocumentId::new("d".to_string()),
            "word ".repeat(500), // ~625 tokens
            0,
            2500,
        );
        let ctx_large = enricher.calculate_num_ctx(&large_doc, &[large_chunk]);
        assert!(
            ctx_large > 45_000,
            "Large doc: num_ctx={ctx_large} should exceed 45000 tokens"
        );
        assert!(
            ctx_large <= 131_072,
            "num_ctx should not exceed 131072 (128k)"
        );
        assert_eq!(ctx_large % 1024, 0, "num_ctx must be a multiple of 1024");

        // Multiple chunks: the largest chunk determines the num_ctx
        let doc = "x".repeat(40_000);
        let small_chunk = TextChunk::new(
            ChunkId::new("c0".to_string()),
            DocumentId::new("d".to_string()),
            "x".repeat(100),
            0,
            100,
        );
        let large_chunk = TextChunk::new(
            ChunkId::new("c1".to_string()),
            DocumentId::new("d".to_string()),
            "x".repeat(2000),
            100,
            2100,
        );
        let ctx_multi_small = enricher.calculate_num_ctx(&doc, std::slice::from_ref(&small_chunk));
        let ctx_multi_large = enricher.calculate_num_ctx(&doc, &[large_chunk]);
        assert!(
            ctx_multi_large > ctx_multi_small,
            "Larger chunk should produce larger num_ctx ({ctx_multi_large} vs {ctx_multi_small})"
        );

        println!(
            "num_ctx: tiny={}  large={}  multi_small={}  multi_large={}",
            ctx_tiny, ctx_large, ctx_multi_small, ctx_multi_large
        );
    }

    // =========================================================================
    // TEST 4 — Disabled enricher is a no-op
    // =========================================================================

    #[tokio::test]
    async fn test_disabled_enricher_returns_chunks_unchanged() {
        let config = ContextualEnricherConfig {
            enabled: false,
            ..Default::default()
        };
        let enricher = ContextualEnricher::new(LlmConfig::default(), config);
        let (document, chunks) = make_test_document_and_chunks(3);

        let result = enricher
            .enrich_document_chunks(&document, &chunks)
            .await
            .expect("Should not fail when disabled");

        assert_eq!(result.len(), chunks.len());
        for (orig, res) in chunks.iter().zip(result.iter()) {
            assert_eq!(
                orig.content, res.content,
                "Disabled enricher must not modify chunks"
            );
        }
    }
}
