/**
 * Web Worker for Background Embedding Generation
 *
 * Handles CPU-intensive embedding generation without blocking the main thread.
 * Simulates hash-based TF embeddings for the demo.
 */

// ============================================================================
// Worker Message Handler
// ============================================================================

self.onmessage = async function(e) {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'generate-embeddings':
                await generateEmbeddings(data);
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            data: { error: error.message }
        });
    }
};

// ============================================================================
// Embedding Generation
// ============================================================================

async function generateEmbeddings(data) {
    const { chunks, docId } = data;
    const dimension = 384;
    const embeddings = [];

    for (let i = 0; i < chunks.length; i++) {
        // Generate hash-based TF embedding
        const embedding = hashEmbedding(chunks[i], dimension);
        embeddings.push(embedding);

        // Report progress every 10 chunks
        if (i % 10 === 0) {
            const progress = (i / chunks.length) * 100;
            self.postMessage({
                type: 'progress',
                data: { progress, current: i, total: chunks.length }
            });
        }

        // Yield to prevent blocking (simulate async work)
        if (i % 50 === 0) {
            await sleep(10);
        }
    }

    // Send completed embeddings
    self.postMessage({
        type: 'embeddings-complete',
        data: { embeddings, docId }
    });
}

// ============================================================================
// Hash-based TF Embedding (same as CLI example)
// ============================================================================

function hashEmbedding(text, dimension) {
    const embedding = new Float32Array(dimension);

    // Tokenize
    const tokens = text
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter(s => s.length > 2);

    if (tokens.length === 0) {
        return Array.from(embedding);
    }

    // Build term frequencies using hash-based indexing (FNV-1a)
    for (const token of tokens) {
        const hash = hashToken(token);
        const idx = hash % dimension;
        embedding[idx] += 1.0;
    }

    // Apply sublinear TF scaling: log(1 + tf)
    for (let i = 0; i < dimension; i++) {
        if (embedding[i] > 0.0) {
            embedding[i] = Math.log(1.0 + embedding[i]);
        }
    }

    // L2 normalization
    let norm = 0.0;
    for (let i = 0; i < dimension; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm);

    if (norm > 0.0) {
        for (let i = 0; i < dimension; i++) {
            embedding[i] /= norm;
        }
    }

    return Array.from(embedding);
}

/**
 * FNV-1a hash function
 * @param {string} token
 * @returns {number}
 */
function hashToken(token) {
    let hash = 0xcbf29ce484222325n; // FNV offset basis (64-bit)

    for (let i = 0; i < token.length; i++) {
        hash ^= BigInt(token.charCodeAt(i));
        hash = (hash * 0x100000001b3n) & 0xffffffffffffffffn; // FNV prime, keep 64-bit
    }

    // Convert to regular number (safe for modulo operation)
    return Number(hash & 0xffffffffn);
}

/**
 * Sleep utility for yielding control
 * @param {number} ms
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// Alternative: ONNX Runtime in Worker (if needed)
// ============================================================================

/**
 * Initialize ONNX Runtime in worker context
 * This would be used if we want to run ONNX inference in the worker
 */
async function initONNXInWorker() {
    // Import ONNX Runtime
    // importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js');

    // Configure ONNX
    // self.ort.env.wasm.numThreads = 1;
    // self.ort.env.wasm.simd = true;

    // Load model
    // const session = await self.ort.InferenceSession.create('./models/all-MiniLM-L6-v2.onnx');

    // return session;
}

/**
 * Generate embedding using ONNX model
 * This would replace hashEmbedding if using real ONNX inference
 */
async function onnxEmbedding(session, text) {
    // Tokenize text (simplified)
    const tokens = tokenize(text);

    // Create input tensor
    const inputIds = new self.ort.Tensor('int64', tokens, [1, tokens.length]);

    // Run inference
    const results = await session.run({ input_ids: inputIds });

    // Get embedding from output
    const embedding = results.last_hidden_state.data;

    return Array.from(embedding);
}

/**
 * Simple tokenizer
 * Real implementation would use proper BERT tokenizer
 */
function tokenize(text) {
    // This is a placeholder - real tokenization is more complex
    return text.toLowerCase().split(/\s+/).map((word, i) => i);
}

console.log('Worker initialized');
