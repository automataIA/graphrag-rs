// WebLLM Integration Module for GraphRAG WASM
// This module provides browser-based LLM inference using WebGPU

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

let engine = null;
let isInitializing = false;
let initializationPromise = null;

/**
 * Initialize WebLLM engine with a specific model
 * @param {string} modelId - Model identifier (e.g., "Llama-3.1-8B-Instruct-q4f32_1-MLC")
 * @param {function} progressCallback - Optional callback for initialization progress
 * @returns {Promise<void>}
 */
export async function initializeEngine(modelId = "Phi-3.5-mini-instruct-q4f16_1-MLC", progressCallback) {
    // Prevent multiple initializations
    if (engine) {
        console.log("[WebLLM] Engine already initialized");
        return;
    }

    if (isInitializing) {
        console.log("[WebLLM] Already initializing, waiting...");
        await initializationPromise;
        return;
    }

    isInitializing = true;

    const progressHandler = (progress) => {
        console.log("[WebLLM] Initialization progress:", progress);
        if (progressCallback) {
            progressCallback(progress);
        }
    };

    try {
        console.log(`[WebLLM] Initializing engine with model: ${modelId}`);
        initializationPromise = webllm.CreateMLCEngine(modelId, {
            initProgressCallback: progressHandler
        });

        engine = await initializationPromise;
        console.log("[WebLLM] Engine initialized successfully");
    } catch (error) {
        console.error("[WebLLM] Failed to initialize engine:", error);
        throw error;
    } finally {
        isInitializing = false;
    }
}

/**
 * Synthesize a natural language answer from context using the LLM
 * @param {string} query - User's original query
 * @param {string} context - Retrieved context from knowledge graph
 * @returns {Promise<string>} - Synthesized natural language answer
 */
export async function synthesizeAnswer(query, context) {
    if (!engine) {
        throw new Error("[WebLLM] Engine not initialized. Call initializeEngine() first.");
    }

    console.log("[WebLLM] Synthesizing answer for query:", query);
    console.log("[WebLLM] Context length:", context.length);

    const messages = [
        {
            role: "system",
            content: `You are a helpful AI assistant that answers questions based on provided context from a knowledge graph.
Your task is to synthesize natural language answers based on the retrieved information.
Be concise, accurate, and cite specific entities and relationships when relevant.
If the context doesn't contain enough information to answer the question, say so clearly.`
        },
        {
            role: "user",
            content: `Question: ${query}

Context from Knowledge Graph:
${context}

Please provide a natural language answer to the question based on the context above.`
        }
    ];

    try {
        const startTime = performance.now();

        const reply = await engine.chat.completions.create({
            messages: messages,
            temperature: 0.7,
            max_tokens: 512,
            stream: false
        });

        const endTime = performance.now();
        const synthesisTime = ((endTime - startTime) / 1000).toFixed(2);

        const answer = reply.choices[0].message.content;
        console.log(`[WebLLM] Answer synthesized in ${synthesisTime}s`);
        console.log("[WebLLM] Answer:", answer);

        return answer;
    } catch (error) {
        console.error("[WebLLM] Failed to synthesize answer:", error);
        throw error;
    }
}

/**
 * Check if the engine is initialized
 * @returns {boolean}
 */
export function isEngineReady() {
    return engine !== null && !isInitializing;
}

/**
 * Get engine status
 * @returns {object}
 */
export function getEngineStatus() {
    return {
        initialized: engine !== null,
        initializing: isInitializing,
        ready: isEngineReady()
    };
}

/**
 * Unload the engine to free resources
 */
export async function unloadEngine() {
    if (engine) {
        console.log("[WebLLM] Unloading engine...");
        await engine.unload();
        engine = null;
        console.log("[WebLLM] Engine unloaded");
    }
}

// Export the module to window for WASM access
if (typeof window !== 'undefined') {
    window.webllmModule = {
        initializeEngine,
        synthesizeAnswer,
        isEngineReady,
        getEngineStatus,
        unloadEngine
    };
}
