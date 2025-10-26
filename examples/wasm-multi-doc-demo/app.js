/**
 * GraphRAG Multi-Document Demo - Main Application
 *
 * Coordinates WASM, ONNX Runtime, Web Workers, and IndexedDB for
 * progressive loading and incremental knowledge graph construction.
 */

// ============================================================================
// State Management
// ============================================================================

const state = {
    wasmModule: null,
    graphRAG: null,
    onnxSession: null,
    worker: null,
    documents: [],
    stats: {
        documents: 0,
        chunks: 0,
        embeddings: 0,
        memory: 0
    },
    isProcessing: false
};

// ============================================================================
// Initialization
// ============================================================================

async function init() {
    updateStatus('Checking system capabilities...');

    // Check system capabilities
    await checkSystemInfo();

    // Initialize ONNX Runtime
    updateStatus('Initializing ONNX Runtime...');
    await initONNXRuntime();

    // Load WASM module
    updateStatus('Loading WASM module...');
    await loadWASM();

    // Initialize Web Worker
    updateStatus('Initializing Web Worker...');
    initWorker();

    // Check for cached data
    await checkIndexedDB();

    updateStatus('✅ Ready! Load a document to begin.');
    enableControls();
}

async function checkSystemInfo() {
    // WASM Support
    const wasmSupported = typeof WebAssembly !== 'undefined';
    updateInfo('info-wasm', wasmSupported ? '✅ Supported' : '❌ Not supported', wasmSupported);

    // WebGPU Support
    const webgpuSupported = navigator.gpu !== undefined;
    updateInfo('info-webgpu', webgpuSupported ? '✅ Available' : '⚠️ Not available', webgpuSupported);

    // ONNX Runtime
    const onnxSupported = typeof ort !== 'undefined';
    updateInfo('info-onnx', onnxSupported ? '✅ Loaded' : '❌ Not loaded', onnxSupported);

    // Voy Search
    const voySupported = typeof Voy !== 'undefined';
    updateInfo('info-voy', voySupported ? '✅ Loaded' : '❌ Not loaded', voySupported);

    // IndexedDB
    const idbSupported = 'indexedDB' in window;
    updateInfo('info-indexeddb', idbSupported ? '✅ Available' : '❌ Not available', idbSupported);

    if (!wasmSupported || !onnxSupported) {
        showError('Your browser does not support required features (WASM or ONNX Runtime)');
        return false;
    }

    return true;
}

async function initONNXRuntime() {
    if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime not loaded');
    }

    // Configure ONNX Runtime
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    // Try WebGPU if available
    if (document.getElementById('use-webgpu').checked && navigator.gpu) {
        try {
            ort.env.webgpu.powerPreference = 'high-performance';
            console.log('WebGPU enabled for ONNX Runtime');
        } catch (e) {
            console.warn('WebGPU not available, falling back to WASM:', e);
        }
    }

    console.log('ONNX Runtime initialized');
}

async function loadWASM() {
    try {
        // In a real implementation, this would load the compiled WASM module
        // For this demo, we'll simulate the GraphRAG instance
        state.graphRAG = new SimulatedGraphRAG();
        console.log('WASM module loaded (simulated)');
    } catch (error) {
        console.error('Failed to load WASM:', error);
        throw error;
    }
}

function initWorker() {
    // Initialize Web Worker for background processing
    state.worker = new Worker('worker.js');

    state.worker.onmessage = (e) => {
        const { type, data } = e.data;

        switch (type) {
            case 'embeddings-complete':
                handleEmbeddingsComplete(data);
                break;
            case 'progress':
                updateProgress(data.progress);
                break;
            case 'error':
                handleError(data.error);
                break;
        }
    };

    state.worker.onerror = (error) => {
        console.error('Worker error:', error);
        handleError(error.message);
    };

    console.log('Web Worker initialized');
}

// ============================================================================
// Document Loading
// ============================================================================

document.getElementById('load-symposium').addEventListener('click', async () => {
    await loadDocument('symposium', '../../docs-example/Symposium.txt');
});

document.getElementById('add-tom-sawyer').addEventListener('click', async () => {
    await addDocument('tom_sawyer', '../../docs-example/The Adventures of Tom Sawyer.txt');
});

document.getElementById('clear-all').addEventListener('click', async () => {
    if (confirm('Clear all documents and reset the knowledge graph?')) {
        await clearAll();
    }
});

async function loadDocument(id, path) {
    if (state.isProcessing) {
        showError('Already processing a document');
        return;
    }

    state.isProcessing = true;
    disableControls();
    showLoading(`Loading ${id}...`);

    try {
        // Fetch document
        updateStatus(`Fetching ${id}...`);
        const response = await fetch(path);
        const text = await response.text();

        console.log(`Loaded ${id}: ${text.length} characters`);

        // Process document
        await processDocument(id, text, 'load');

        // Update UI
        document.getElementById(`${id.replace('_', '-')}-status`).textContent = '✅ Loaded';
        document.getElementById(`${id.replace('_', '-')}-status`).classList.add('success');

        // Enable next step
        if (id === 'symposium') {
            document.getElementById('add-tom-sawyer').disabled = false;
        }

        // Enable query
        document.getElementById('query-input').disabled = false;
        document.getElementById('query-btn').disabled = false;

        updateStatus(`✅ ${id} loaded successfully`);
    } catch (error) {
        console.error(`Error loading ${id}:`, error);
        showError(`Failed to load ${id}: ${error.message}`);
    } finally {
        state.isProcessing = false;
        hideLoading();
        enableControls();
    }
}

async function addDocument(id, path) {
    if (state.isProcessing) {
        showError('Already processing a document');
        return;
    }

    state.isProcessing = true;
    disableControls();
    showLoading(`Adding ${id}...`);

    try {
        const startTime = performance.now();

        // Fetch document
        updateStatus(`Fetching ${id}...`);
        const response = await fetch(path);
        const text = await response.text();

        console.log(`Loaded ${id}: ${text.length} characters`);

        // Process with incremental merge
        await processDocument(id, text, 'merge');

        const elapsed = Math.round(performance.now() - startTime);

        // Show merge stats
        showMergeStats({
            newChunks: state.graphRAG.lastMergeStats.newChunks,
            duplicates: state.graphRAG.lastMergeStats.duplicates,
            time: elapsed
        });

        // Update UI
        document.getElementById(`${id.replace('_', '-')}-status`).textContent = '✅ Merged';
        document.getElementById(`${id.replace('_', '-')}-status`).classList.add('success');

        updateStatus(`✅ ${id} merged successfully (${elapsed}ms)`);
    } catch (error) {
        console.error(`Error adding ${id}:`, error);
        showError(`Failed to add ${id}: ${error.message}`);
    } finally {
        state.isProcessing = false;
        hideLoading();
        enableControls();
    }
}

async function processDocument(id, text, mode) {
    // Chunk the document
    updateStatus('Chunking document...');
    const chunks = chunkText(text, 200, 50);
    console.log(`Created ${chunks.length} chunks`);

    // Send to worker for embedding generation
    updateStatus('Generating embeddings (this may take a minute)...');
    showProgress(0);

    return new Promise((resolve, reject) => {
        const handleMessage = (e) => {
            const { type, data } = e.data;

            if (type === 'embeddings-complete') {
                state.worker.removeEventListener('message', handleMessage);

                if (mode === 'load') {
                    state.graphRAG.loadDocument(id, chunks, data.embeddings);
                } else {
                    state.graphRAG.mergeDocument(id, chunks, data.embeddings);
                }

                updateStats();

                // Save to IndexedDB if enabled
                if (document.getElementById('use-indexeddb').checked) {
                    saveToIndexedDB();
                }

                hideProgress();
                resolve();
            } else if (type === 'error') {
                state.worker.removeEventListener('message', handleMessage);
                reject(new Error(data.error));
            }
        };

        state.worker.addEventListener('message', handleMessage);
        state.worker.postMessage({
            type: 'generate-embeddings',
            data: { chunks, docId: id }
        });
    });
}

// ============================================================================
// Query Handling
// ============================================================================

document.getElementById('query-btn').addEventListener('click', async () => {
    await executeQuery();
});

document.getElementById('query-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        executeQuery();
    }
});

// Example queries
document.querySelectorAll('.example-query').forEach(btn => {
    btn.addEventListener('click', () => {
        const query = btn.dataset.query;
        document.getElementById('query-input').value = query;
        executeQuery();
    });
});

async function executeQuery() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;

    if (state.documents.length === 0) {
        showError('Load at least one document first');
        return;
    }

    updateStatus(`Searching: "${query}"`);
    showLoading('Searching...');

    try {
        // Generate query embedding (simulated)
        const topK = parseInt(document.getElementById('top-k').value);
        const results = await state.graphRAG.query(query, topK);

        // Display results
        displayResults(results);

        updateStatus(`✅ Found ${results.length} results`);
    } catch (error) {
        console.error('Query error:', error);
        showError(`Query failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayResults(results) {
    const container = document.getElementById('results-container');
    const resultsSection = document.getElementById('query-results');

    container.innerHTML = '';

    if (results.length === 0) {
        container.innerHTML = '<p>No results found.</p>';
        resultsSection.classList.remove('hidden');
        return;
    }

    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'result-item';

        item.innerHTML = `
            <div class="result-header">
                <span class="result-rank">#${index + 1}</span>
                <span class="result-source">${result.source}</span>
                <span class="result-similarity">Similarity: ${result.similarity.toFixed(4)}</span>
            </div>
            <div class="result-text">${escapeHtml(result.text)}</div>
        `;

        container.appendChild(item);
    });

    resultsSection.classList.remove('hidden');
}

// ============================================================================
// Utilities
// ============================================================================

function chunkText(text, chunkSize, overlap) {
    const words = text.split(/\s+/);
    const chunks = [];

    for (let i = 0; i < words.length; i += chunkSize - overlap) {
        const chunk = words.slice(i, i + chunkSize).join(' ');
        if (chunk.split(/\s+/).length > 50) {
            chunks.push(chunk);
        }
    }

    return chunks;
}

function updateStats() {
    const stats = state.graphRAG.getStats();

    document.getElementById('stat-documents').textContent = stats.documents;
    document.getElementById('stat-chunks').textContent = stats.chunks;
    document.getElementById('stat-embeddings').textContent = stats.embeddings;
    document.getElementById('stat-memory').textContent = stats.memory.toFixed(1);

    state.stats = stats;
}

function showMergeStats(stats) {
    document.getElementById('merge-new-chunks').textContent = stats.newChunks;
    document.getElementById('merge-duplicates').textContent = stats.duplicates;
    document.getElementById('merge-time').textContent = `${stats.time}ms`;
    document.getElementById('merge-stats').classList.remove('hidden');
}

function updateStatus(message) {
    document.getElementById('status-text').textContent = message;
}

function updateInfo(id, value, success) {
    const element = document.getElementById(id);
    element.textContent = value;
    element.className = 'info-value ' + (success ? 'success' : 'error');
}

function showProgress(percent) {
    const container = document.getElementById('progress-container');
    const bar = document.getElementById('progress-bar');
    const text = document.getElementById('progress-text');

    container.classList.remove('hidden');
    bar.style.width = `${percent}%`;
    text.textContent = `${Math.round(percent)}%`;
}

function hideProgress() {
    document.getElementById('progress-container').classList.add('hidden');
}

function showLoading(message) {
    document.getElementById('loading-message').textContent = message;
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

function showError(message) {
    alert(`Error: ${message}`);
}

function handleError(error) {
    showError(error);
    hideLoading();
    state.isProcessing = false;
    enableControls();
}

function enableControls() {
    if (!state.isProcessing) {
        document.getElementById('load-symposium').disabled = state.documents.some(d => d.id === 'symposium');
        document.getElementById('add-tom-sawyer').disabled = !state.documents.some(d => d.id === 'symposium') || state.documents.some(d => d.id === 'tom_sawyer');
        document.getElementById('clear-all').disabled = state.documents.length === 0;
        document.getElementById('export-graph').disabled = state.documents.length === 0;
        document.getElementById('load-from-cache').disabled = false;
    }
}

function disableControls() {
    document.getElementById('load-symposium').disabled = true;
    document.getElementById('add-tom-sawyer').disabled = true;
    document.getElementById('clear-all').disabled = true;
    document.getElementById('query-btn').disabled = true;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Advanced Features
// ============================================================================

document.getElementById('export-graph').addEventListener('click', () => {
    const data = state.graphRAG.export();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graphrag-export.json';
    a.click();
    URL.revokeObjectURL(url);
});

async function saveToIndexedDB() {
    // TODO: Implement IndexedDB persistence
    console.log('Saving to IndexedDB (not implemented)');
}

async function checkIndexedDB() {
    // TODO: Check for cached data
    console.log('Checking IndexedDB (not implemented)');
}

async function clearAll() {
    state.graphRAG.clear();
    state.documents = [];
    updateStats();

    document.getElementById('symposium-status').textContent = 'Not loaded';
    document.getElementById('symposium-status').classList.remove('success');
    document.getElementById('tom-sawyer-status').textContent = 'Not loaded';
    document.getElementById('tom-sawyer-status').classList.remove('success');

    document.getElementById('query-results').classList.add('hidden');
    document.getElementById('merge-stats').classList.add('hidden');

    enableControls();
    updateStatus('Ready to load documents');
}

// Toggle collapsible sections
window.toggleSection = function(element) {
    element.parentElement.classList.toggle('collapsed');
};

// ============================================================================
// Simulated GraphRAG (for demo without actual WASM)
// ============================================================================

class SimulatedGraphRAG {
    constructor() {
        this.documents = [];
        this.chunks = [];
        this.embeddings = [];
        this.lastMergeStats = { newChunks: 0, duplicates: 0 };
    }

    loadDocument(id, chunks, embeddings) {
        this.documents.push({ id });
        const startIdx = this.chunks.length;
        this.chunks.push(...chunks.map((text, i) => ({ id: id, chunkId: startIdx + i, text })));
        this.embeddings.push(...embeddings);
        state.documents.push({ id });
    }

    mergeDocument(id, chunks, embeddings) {
        const beforeChunks = this.chunks.length;

        this.documents.push({ id });
        const startIdx = this.chunks.length;
        this.chunks.push(...chunks.map((text, i) => ({ id: id, chunkId: startIdx + i, text })));
        this.embeddings.push(...embeddings);
        state.documents.push({ id });

        // Simulate duplicate detection
        const duplicates = Math.floor(Math.random() * 50) + 10;

        this.lastMergeStats = {
            newChunks: this.chunks.length - beforeChunks,
            duplicates
        };
    }

    query(query, topK) {
        // Simulate search results
        const results = [];
        for (let i = 0; i < Math.min(topK, this.chunks.length); i++) {
            const chunk = this.chunks[Math.floor(Math.random() * this.chunks.length)];
            results.push({
                source: chunk.id,
                text: chunk.text.substring(0, 200) + '...',
                similarity: 0.8 - i * 0.1
            });
        }
        return Promise.resolve(results);
    }

    getStats() {
        return {
            documents: this.documents.length,
            chunks: this.chunks.length,
            embeddings: this.embeddings.length,
            memory: (this.chunks.length * 0.002) + (this.embeddings.length * 0.0015)
        };
    }

    export() {
        return {
            documents: this.documents,
            chunks: this.chunks.length,
            embeddings: this.embeddings.length
        };
    }

    clear() {
        this.documents = [];
        this.chunks = [];
        this.embeddings = [];
        this.lastMergeStats = { newChunks: 0, duplicates: 0 };
    }
}

// ============================================================================
// Start Application
// ============================================================================

init().catch(error => {
    console.error('Initialization error:', error);
    showError(`Failed to initialize: ${error.message}`);
});
