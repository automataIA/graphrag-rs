//! End-to-end tests for ContextualEnricher with real Ollama + KV Cache
//!
//! These tests require a running Ollama instance and are **skipped by default**.
//!
//! ## Running the tests
//!
//! ```bash
//! # Start Ollama (if not already running)
//! ollama serve
//!
//! # Pull a model (Mistral-NeMo 12B if you have the VRAM, otherwise smaller)
//! ollama pull mistral-nemo:12b   # recommended (RTX 4070 12GB)
//! ollama pull llama3.2:3b        # fallback for smaller GPUs
//!
//! # Run the ignored tests
//! ENABLE_OLLAMA_TESTS=1 cargo test -p graphrag-core --test contextual_enricher_e2e -- --ignored --nocapture
//! ```
//!
//! ## What these tests verify
//!
//! 1. **Correctness**: enriched chunk contains the original text + LLM-generated context
//! 2. **KV Cache speedup**: first chunk is slow (document loaded into cache),
//!    subsequent chunks are fast (only the chunk is re-evaluated)
//! 3. **num_ctx calculation**: the value sent to Ollama is >= actual token count
//! 4. **keep_alive wiring**: the model stays loaded between requests (measured via timing)

#[cfg(all(feature = "ureq", feature = "async"))]
mod e2e {
    use graphrag_core::{
        core::{ChunkId, Document, DocumentId, TextChunk},
        ollama::OllamaConfig,
        text::contextual_enricher::{ContextualEnricher, ContextualEnricherConfig},
    };
    use std::time::{Duration, Instant};

    /// Return true only if the user explicitly opted into Ollama tests.
    ///
    /// We use an env var instead of relying solely on `#[ignore]` so CI
    /// can never accidentally run these even with `-- --include-ignored`.
    fn ollama_tests_enabled() -> bool {
        std::env::var("ENABLE_OLLAMA_TESTS").is_ok()
    }

    /// Build an OllamaConfig pointing at localhost with KV-cache settings.
    ///
    /// `OLLAMA_MODEL` env var overrides the default model — useful when you
    /// want to test a specific model without editing the source.
    fn make_ollama_config() -> OllamaConfig {
        let model =
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "mistral-nemo:12b".to_string());

        OllamaConfig {
            enabled: true,
            chat_model: model,
            keep_alive: Some("1h".to_string()), // keep model in VRAM between requests
            // num_ctx is calculated per-document by ContextualEnricher
            timeout_seconds: 300, // allow up to 5 min for first chunk (context load)
            max_retries: 1,
            ..OllamaConfig::default()
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
                .filter(|&p| p <= end)
                .last()
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
    #[ignore] // Run with: ENABLE_OLLAMA_TESTS=1 cargo test -- --ignored --nocapture
    async fn test_enriched_chunk_contains_original_and_context() {
        if !ollama_tests_enabled() {
            println!("Skipping: set ENABLE_OLLAMA_TESTS=1 to run");
            return;
        }

        let (document, chunks) = make_test_document_and_chunks(3);
        let enricher = ContextualEnricher::with_defaults(make_ollama_config());

        println!("\n=== Contextual Enricher E2E — Correctness ===");
        println!(
            "Document: '{}' ({} chars)",
            document.title,
            document.content.len()
        );
        println!("Chunks: {}", chunks.len());
        println!(
            "Estimated num_ctx: {}",
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
    // TEST 2 — KV Cache speedup
    // =========================================================================

    /// Measure the per-chunk timing to verify KV cache is active.
    ///
    /// With KV cache enabled:
    /// - Chunk 1 (cold): loads the document into Ollama's KV cache (~1-2 min for 12B)
    /// - Chunks 2..N (warm): only the chunk tokens are evaluated (~3-10 sec each)
    ///
    /// We do NOT assert on specific timing values (too hardware-dependent) but
    /// we log the speedup ratio and warn if the cache does not appear to be working.
    #[tokio::test]
    #[ignore] // Run with: ENABLE_OLLAMA_TESTS=1 cargo test -- --ignored --nocapture
    async fn test_kv_cache_speedup() {
        if !ollama_tests_enabled() {
            println!("Skipping: set ENABLE_OLLAMA_TESTS=1 to run");
            return;
        }

        // Use more chunks to get a meaningful speedup measurement
        let (document, chunks) = make_test_document_and_chunks(5);

        let ollama_config = make_ollama_config();
        let enricher_config = ContextualEnricherConfig {
            keep_alive: "1h".to_string(),
            max_output_tokens: 80, // smaller output → faster per chunk, cleaner timing
            ..ContextualEnricherConfig::default()
        };
        let enricher = ContextualEnricher::new(ollama_config.clone(), enricher_config);

        let num_ctx = enricher.calculate_num_ctx(&document.content, &chunks);

        println!("\n=== Contextual Enricher E2E — KV Cache Timing ===");
        println!("Model: {}", ollama_config.chat_model);
        println!(
            "Document: {} chars (~{} tokens)",
            document.content.len(),
            document.content.len() / 4
        );
        println!("Chunks: {}", chunks.len());
        println!("num_ctx: {} (calculated)", num_ctx);
        println!("keep_alive: 1h");
        println!("\nStarting enrichment — watch for KV cache speedup after chunk 1...\n");

        // Enrich chunk-by-chunk and measure each individually
        let mut timings: Vec<Duration> = Vec::with_capacity(chunks.len());

        // We use enrich_document_chunks which processes all chunks together,
        // but we instrument it by processing one-at-a-time with the same num_ctx.
        // This gives per-chunk timing while still benefitting from keep_alive.
        use graphrag_core::ollama::{OllamaClient, OllamaGenerationParams};

        let client = OllamaClient::new(make_ollama_config());

        for (i, chunk) in chunks.iter().enumerate() {
            let prompt = format!(
                "<document>\n{}\n</document>\n\n\
                 Here is the chunk we want to situate within the whole document:\n\
                 <chunk>\n{}\n</chunk>\n\n\
                 Please give a short succinct context to situate this chunk within \
                 the overall document for the purposes of improving search retrieval \
                 of the chunk. Answer only with the succinct context and nothing else.",
                document.content, chunk.content
            );

            let params = OllamaGenerationParams {
                num_predict: Some(80),
                temperature: Some(0.1),
                num_ctx: Some(num_ctx),
                keep_alive: Some("1h".to_string()),
                ..Default::default()
            };

            let t0 = Instant::now();
            let result = client.generate_with_params(&prompt, params).await;
            let elapsed = t0.elapsed();

            match result {
                Ok(context) => {
                    timings.push(elapsed);
                    println!(
                        "Chunk {}/{}: {:>7.1}s | {:>3} chars context | preview: \"{}\"",
                        i + 1,
                        chunks.len(),
                        elapsed.as_secs_f64(),
                        context.trim().len(),
                        &context.trim()[..context.trim().len().min(60)],
                    );
                },
                Err(e) => {
                    println!("Chunk {}/{}: ERROR — {e}", i + 1, chunks.len());
                    panic!("Ollama request failed: {e}");
                },
            }
        }

        // ── Analysis ─────────────────────────────────────────────────────────
        println!("\n=== Timing Analysis ===");

        let t_first = timings[0];
        let t_rest: Vec<_> = timings[1..].to_vec();

        if !t_rest.is_empty() {
            let avg_rest = t_rest.iter().sum::<Duration>() / t_rest.len() as u32;
            let min_rest = t_rest.iter().min().unwrap();
            let max_rest = t_rest.iter().max().unwrap();
            let speedup = t_first.as_secs_f64() / avg_rest.as_secs_f64();

            println!(
                "First chunk (cold, loads KV cache): {:.1}s",
                t_first.as_secs_f64()
            );
            println!(
                "Chunks 2..{} (warm): avg={:.1}s  min={:.1}s  max={:.1}s",
                timings.len(),
                avg_rest.as_secs_f64(),
                min_rest.as_secs_f64(),
                max_rest.as_secs_f64(),
            );
            println!("Speedup ratio (cold/warm avg): {:.1}x", speedup);

            if speedup < 3.0 {
                println!(
                    "\n⚠ WARNING: speedup ratio {:.1}x is lower than expected (>10x with KV cache).",
                    speedup
                );
                println!("  Possible causes:");
                println!("  - Model was already loaded (previous test warmed the cache)");
                println!("  - keep_alive not honoured by this Ollama version");
                println!("  - Document too short to show significant KV cache benefit");
                println!("  - Model already had num_ctx set higher than needed");
            } else {
                println!("\n✓ KV cache appears to be working (speedup ≥ 3x)");
            }
        } else {
            println!("Only 1 chunk — cannot measure speedup. Increase n_chunks.");
        }
    }

    // =========================================================================
    // TEST 2b — KV Cache speedup on the real Symposium.txt
    // =========================================================================

    /// Same as test_kv_cache_speedup but uses the full Symposium.txt as the document.
    ///
    /// With ~50k tokens as the static prefix, the KV cache effect is clearly visible:
    /// - Chunk 1 (cold): ~2-3 min to load the entire book into Ollama's KV cache
    /// - Chunks 2..N (warm): ~3-10 sec each (only the chunk tokens are re-evaluated)
    ///
    /// Set `SYMPOSIUM_CHUNKS=N` (default 10) to control how many chunks to test.
    /// Set `SYMPOSIUM_PATH` to override the book path.
    ///
    /// Run with:
    /// ```bash
    /// ENABLE_OLLAMA_TESTS=1 \
    /// SYMPOSIUM_PATH=/home/dio/graphrag-rs/docs-example/Symposium.txt \
    /// SYMPOSIUM_CHUNKS=10 \
    /// cargo test -p graphrag-core --test contextual_enricher_e2e \
    ///   -- test_kv_cache_speedup_symposium --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn test_kv_cache_speedup_symposium() {
        if !ollama_tests_enabled() {
            println!("Skipping: set ENABLE_OLLAMA_TESTS=1 to run");
            return;
        }

        // ── Load document ────────────────────────────────────────────────────
        let book_path = std::env::var("SYMPOSIUM_PATH")
            .unwrap_or_else(|_| "/home/dio/graphrag-rs/docs-example/Symposium.txt".to_string());
        let n_chunks: usize = std::env::var("SYMPOSIUM_CHUNKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        let content = std::fs::read_to_string(&book_path)
            .unwrap_or_else(|e| panic!("Cannot read {book_path}: {e}"));

        // ── Build chunks (character-based, 512 chars / 100 overlap) ─────────
        let chunk_size = 512usize;
        let overlap = 100usize;
        let step = chunk_size - overlap;

        let doc_id = graphrag_core::core::DocumentId::new("symposium".to_string());
        let document = graphrag_core::core::Document::new(
            doc_id.clone(),
            "Symposium".to_string(),
            content.clone(),
        );

        let mut all_chunks: Vec<graphrag_core::core::TextChunk> = Vec::new();
        let mut start = 0usize;
        let mut idx = 0usize;
        while start < content.len() {
            // Snap end to char boundary
            let raw_end = (start + chunk_size).min(content.len());
            let end = content
                .char_indices()
                .map(|(p, _)| p)
                .filter(|&p| p <= raw_end)
                .last()
                .unwrap_or(raw_end);

            let text = content[start..end].to_string();
            if !text.trim().is_empty() {
                all_chunks.push(graphrag_core::core::TextChunk::new(
                    graphrag_core::core::ChunkId::new(format!("c{idx}")),
                    doc_id.clone(),
                    text,
                    start,
                    end,
                ));
                idx += 1;
            }
            if end >= content.len() {
                break;
            }
            start += step;
        }

        // Take only the first N chunks for the timing test
        let chunks: Vec<_> = all_chunks.into_iter().take(n_chunks).collect();

        // ── Setup ────────────────────────────────────────────────────────────
        // Cap num_ctx to avoid OOM on consumer GPUs (RTX 4070 12GB: ~5GB free after model).
        // Mistral-NeMo 12B weights ≈ 7GB → ~5GB for KV cache → max ~32k tokens safely.
        // Override with OLLAMA_NUM_CTX env var (e.g. 65536 for 24GB GPUs).
        let num_ctx_cap: u32 = std::env::var("OLLAMA_NUM_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768);

        let ollama_config = make_ollama_config();
        let num_ctx = num_ctx_cap;

        // ── Critical: doc must fit inside num_ctx with room for instructions + chunk ──
        //
        // For KV cache reuse via `context`, the condition is:
        //   prime_context_tokens + chunk_tokens <= num_ctx
        //
        // prime_context_tokens ≈ doc_tokens + instruction_tokens + response_token
        //                      ≈ doc_tokens + 300  (generous estimate for overhead)
        // chunk_tokens ≈ 130 tokens (512 chars / 4)
        //
        // So: doc_tokens <= num_ctx - 300 - 130 = num_ctx - 430
        //     doc_chars  <= (num_ctx - 430) * 3   (using 3 chars/token conservatively)
        //
        // We use 3 chars/token (vs 4) because philosophical text often tokenizes poorly.
        let doc_token_budget = num_ctx.saturating_sub(512) as usize; // 512 tok overhead budget
        let max_doc_chars = doc_token_budget * 3; // conservative: 3 chars per token

        let doc_content_for_prompt = if document.content.len() > max_doc_chars {
            let truncated = &document.content[..max_doc_chars];
            // Snap to char boundary
            let safe_end = truncated
                .char_indices()
                .last()
                .map(|(i, _)| i + 1)
                .unwrap_or(max_doc_chars);
            &document.content[..safe_end]
        } else {
            &document.content
        };
        let truncated = doc_content_for_prompt.len() < document.content.len();

        println!("\n=== KV Cache Timing — Real Symposium.txt ===");
        println!("Model      : {}", ollama_config.chat_model);
        println!("Book       : {book_path}");
        println!(
            "Book size  : {} chars (~{} tokens)",
            content.len(),
            content.len() / 4
        );
        if truncated {
            println!(
                "Doc in prompt: {} chars (~{} tokens) [TRUNCATED to fit num_ctx]",
                doc_content_for_prompt.len(),
                doc_content_for_prompt.len() / 4
            );
        }
        println!("Total chunks available: {}", idx);
        println!("Testing first {n_chunks} chunks (chunk_size={chunk_size}, overlap={overlap})");
        println!("num_ctx    : {num_ctx} (cap={num_ctx_cap})");
        println!("keep_alive : 1h");
        println!(
            "\n⏱  Starting — chunk 1 loads the full book into KV cache, watch for speedup...\n"
        );

        use graphrag_core::ollama::{OllamaClient, OllamaGenerationParams};
        let client = OllamaClient::new(make_ollama_config());
        let mut timings: Vec<Duration> = Vec::with_capacity(n_chunks);

        // ── Step 1: Prime the KV cache ────────────────────────────────────
        // Send just the document (static prefix) to load it into Ollama's KV cache.
        // We request num_predict=1 so the response is instant; we only want the context.
        // The prime prompt must end mid-sentence so each chunk prompt continues naturally.
        let prime_prompt = format!(
            "<document>\n{doc_content_for_prompt}\n</document>\n\n\
             Here is a chunk from this document. \
             Provide a short succinct context (1-2 sentences) to situate it within the \
             overall document for improving search retrieval. \
             Answer only with the context, nothing else.\n\n<chunk>\n"
        );

        print!(
            "PRIME : [loading {:.0}k tokens into KV cache...]\r",
            doc_content_for_prompt.len() as f64 / 4000.0
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let t_prime = Instant::now();
        let prime_resp = client
            .generate_with_full_response(
                &prime_prompt,
                OllamaGenerationParams {
                    num_predict: Some(1), // generate minimal; we just want the context array
                    temperature: Some(0.1),
                    num_ctx: Some(num_ctx),
                    keep_alive: Some("1h".to_string()),
                    ..Default::default()
                },
            )
            .await
            .expect("Priming request failed");
        let t_prime_elapsed = t_prime.elapsed();

        println!(
            "PRIME : {:>7.1}s | {:.0}k tokens evaluated → KV cache ready",
            t_prime_elapsed.as_secs_f64(),
            prime_resp.prompt_eval_count as f64 / 1000.0,
        );
        println!(
            "        context array size: {} tokens",
            prime_resp.context.len()
        );
        println!();

        let priming_context = prime_resp.context;

        // ── Step 2: Enrich each chunk using the priming context ────────────
        // Each request starts from the priming context (document already in KV cache).
        // Only the chunk tokens are evaluated — should be ~3-5s vs ~60s without cache.
        for (i, chunk) in chunks.iter().enumerate() {
            // The chunk prompt CONTINUES from where the prime prompt ended.
            // The prime ended with "<chunk>\n", so we provide the chunk content + instruction.
            let chunk_prompt = format!("{}\n</chunk>", chunk.content);

            let params = OllamaGenerationParams {
                num_predict: Some(80),
                temperature: Some(0.1),
                num_ctx: Some(num_ctx),
                keep_alive: Some("1h".to_string()),
                context: Some(priming_context.clone()), // ← KV cache reuse
                ..Default::default()
            };

            print!(
                "Chunk {:>3}/{n_chunks}: [evaluating ~{} tokens...]\r",
                i + 1,
                chunk_prompt.len() / 4
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());

            let t0 = Instant::now();
            let result = client
                .generate_with_full_response(&chunk_prompt, params)
                .await;
            let elapsed = t0.elapsed();

            match result {
                Ok(resp) => {
                    timings.push(elapsed);
                    let kv_label = if i == 0 { " ← first chunk" } else { "" };
                    println!(
                        "Chunk {:>3}/{n_chunks}: {:>6.1}s | eval={:>4} tok | {:>3} chars | \"{}\"{kv_label}",
                        i + 1,
                        elapsed.as_secs_f64(),
                        resp.prompt_eval_count,
                        resp.text.trim().len(),
                        &resp.text.trim()[..resp.text.trim().len().min(50)],
                    );
                },
                Err(e) => {
                    println!("Chunk {:>3}/{n_chunks}: ERROR — {e}", i + 1);
                    panic!("Ollama request failed: {e}");
                },
            }
        }

        // ── Analysis ─────────────────────────────────────────────────────────
        println!("\n=== Timing Analysis ===");

        let avg_chunks = timings.iter().sum::<Duration>() / timings.len() as u32;
        let min_chunk = *timings.iter().min().unwrap();
        let max_chunk = *timings.iter().max().unwrap();

        println!(
            "Prime (document loaded): {:.1}s  ({:.0}k tokens)",
            t_prime_elapsed.as_secs_f64(),
            prime_resp.prompt_eval_count as f64 / 1000.0
        );
        println!(
            "Chunks 1..{n_chunks} (context reuse): avg={:.1}s  min={:.1}s  max={:.1}s",
            avg_chunks.as_secs_f64(),
            min_chunk.as_secs_f64(),
            max_chunk.as_secs_f64()
        );

        // Speedup vs naive (each chunk re-evaluating the full document)
        // Naive: each chunk ≈ prime_time (re-evaluating 32k tokens each time)
        let naive_per_chunk = t_prime_elapsed.as_secs_f64();
        let speedup = naive_per_chunk / avg_chunks.as_secs_f64();
        println!("Speedup vs naive (per-chunk re-eval): {:.1}x", speedup);

        if speedup < 3.0 {
            println!(
                "\n⚠  WARNING: speedup {:.1}x is low. The `context` parameter may not",
                speedup
            );
            println!("   be reusing the KV cache. Possible causes:");
            println!("   - Ollama version does not support context-based KV cache reuse");
            println!("   - context array is too large and was truncated");
            println!("   - num_ctx={num_ctx} forces a different KV cache allocation");
        } else {
            println!(
                "\n✓ KV cache working via `context` parameter! {:.1}x speedup",
                speedup
            );
            let total_est = t_prime_elapsed.as_secs_f64() + avg_chunks.as_secs_f64() * idx as f64;
            println!(
                "  Estimated time for all {} chunks: {:.0} min",
                idx,
                total_est / 60.0
            );
        }
    }

    // =========================================================================
    // TEST 3 — num_ctx calculation sanity
    // =========================================================================

    /// Verify that `calculate_num_ctx` produces values within sensible bounds
    /// and that it grows proportionally with document length.
    ///
    /// This test does NOT call Ollama — it only exercises the calculation logic.
    /// Listed here (not in unit tests) because it uses document-size fixtures
    /// that are representative of real use.
    #[test]
    fn test_num_ctx_calculation_sanity() {
        let enricher = ContextualEnricher::with_defaults(OllamaConfig::default());

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
        let ctx_multi_small = enricher.calculate_num_ctx(&doc, &[small_chunk.clone()]);
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
        let enricher = ContextualEnricher::new(OllamaConfig::default(), config);
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
