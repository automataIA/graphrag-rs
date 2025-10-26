/**
 * GraphRAG Leptos Demo - Browser Testing Helpers
 *
 * Copia e incolla queste funzioni nella Console del browser (F12)
 * per testare manualmente le funzionalità dell'applicazione.
 */

// ============================================================================
// VERIFICA INIZIALE
// ============================================================================

/**
 * Verifica che tutti i prerequisiti siano soddisfatti
 */
async function checkPrerequisites() {
  console.group("🔍 Checking Prerequisites");

  // WebGPU
  const hasWebGPU = navigator.gpu !== undefined;
  console.log(`${hasWebGPU ? '✅' : '⚠️'} WebGPU: ${hasWebGPU ? 'Available' : 'Not available (will use CPU fallback)'}`);

  // ONNX Runtime
  const hasONNX = typeof window.ort !== 'undefined';
  console.log(`${hasONNX ? '✅' : '❌'} ONNX Runtime: ${hasONNX ? 'Loaded' : 'NOT FOUND'}`);

  if (hasONNX) {
    console.log(`   Version: ONNX Runtime Web ${ort.env.wasm.numThreads} threads`);
  }

  // Voy
  const hasVoy = typeof window.Voy !== 'undefined';
  console.log(`${hasVoy ? '✅' : '⚠️'} Voy Search: ${hasVoy ? 'Available' : 'Not available'}`);

  // WASM
  const hasWASM = typeof WebAssembly !== 'undefined';
  console.log(`${hasWASM ? '✅' : '❌'} WebAssembly: ${hasWASM ? 'Supported' : 'NOT SUPPORTED'}`);

  // SharedArrayBuffer (for multithreading)
  const hasSAB = typeof SharedArrayBuffer !== 'undefined';
  console.log(`${hasSAB ? '✅' : '⚠️'} SharedArrayBuffer: ${hasSAB ? 'Available' : 'Not available (single-threaded)'}`);

  console.groupEnd();

  return {
    webgpu: hasWebGPU,
    onnx: hasONNX,
    voy: hasVoy,
    wasm: hasWASM,
    sharedArrayBuffer: hasSAB,
    ready: hasONNX && hasWASM
  };
}

// ============================================================================
// PERFORMANCE TESTING
// ============================================================================

/**
 * Misura performance di caricamento modello ONNX
 */
async function benchmarkModelLoading() {
  console.group("⏱️ Model Loading Benchmark");

  const start = performance.now();

  try {
    // Nota: questo richiede che il modello sia già stato scaricato
    const session = await ort.InferenceSession.create(
      './models/all-MiniLM-L6-v2.onnx',
      {
        executionProviders: navigator.gpu ? ['webgpu', 'wasm'] : ['wasm']
      }
    );

    const duration = performance.now() - start;
    const backend = navigator.gpu ? 'WebGPU' : 'WASM CPU';

    console.log(`✅ Model loaded in ${duration.toFixed(0)}ms using ${backend}`);
    console.log(`   Input names:`, session.inputNames);
    console.log(`   Output names:`, session.outputNames);

    return { duration, backend, session };
  } catch (error) {
    const duration = performance.now() - start;
    console.error(`❌ Failed after ${duration.toFixed(0)}ms:`, error.message);
    return { duration, error: error.message };
  } finally {
    console.groupEnd();
  }
}

/**
 * Simula performance di embedding generation
 */
async function benchmarkEmbedding(session, numIterations = 10) {
  console.group(`⏱️ Embedding Benchmark (${numIterations} iterations)`);

  if (!session) {
    console.error("❌ No session provided. Run benchmarkModelLoading() first.");
    console.groupEnd();
    return;
  }

  // Dummy input (example tokenized sentence)
  const inputIds = new BigInt64Array(128).fill(1n);
  const attentionMask = new BigInt64Array(128).fill(1n);

  const feeds = {
    input_ids: new ort.Tensor('int64', inputIds, [1, 128]),
    attention_mask: new ort.Tensor('int64', attentionMask, [1, 128])
  };

  const times = [];

  for (let i = 0; i < numIterations; i++) {
    const start = performance.now();

    try {
      const results = await session.run(feeds);
      const duration = performance.now() - start;
      times.push(duration);

      if (i === 0) {
        console.log(`   First run: ${duration.toFixed(2)}ms (includes warmup)`);
      }
    } catch (error) {
      console.error(`❌ Iteration ${i + 1} failed:`, error.message);
      break;
    }
  }

  if (times.length > 1) {
    const avg = times.slice(1).reduce((a, b) => a + b, 0) / (times.length - 1);
    const min = Math.min(...times.slice(1));
    const max = Math.max(...times.slice(1));

    console.log(`✅ Results (excluding warmup):`);
    console.log(`   Average: ${avg.toFixed(2)}ms`);
    console.log(`   Min: ${min.toFixed(2)}ms`);
    console.log(`   Max: ${max.toFixed(2)}ms`);
    console.log(`   Throughput: ${(1000 / avg).toFixed(1)} embeddings/sec`);
  }

  console.groupEnd();
  return times;
}

/**
 * Benchmark memory usage
 */
function benchmarkMemory() {
  console.group("💾 Memory Benchmark");

  if (performance.memory) {
    const mb = (bytes) => (bytes / 1024 / 1024).toFixed(2);

    console.log(`Total JS Heap: ${mb(performance.memory.totalJSHeapSize)} MB`);
    console.log(`Used JS Heap: ${mb(performance.memory.usedJSHeapSize)} MB`);
    console.log(`Heap Limit: ${mb(performance.memory.jsHeapSizeLimit)} MB`);

    const usage = performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit;
    const status = usage < 0.7 ? '✅' : usage < 0.9 ? '⚠️' : '❌';
    console.log(`${status} Heap usage: ${(usage * 100).toFixed(1)}%`);
  } else {
    console.log("⚠️ performance.memory not available (use Chrome with --enable-precise-memory-info)");
  }

  console.groupEnd();
}

// ============================================================================
// NETWORK TESTING
// ============================================================================

/**
 * Verifica caricamento risorse
 */
function checkNetworkResources() {
  console.group("🌐 Network Resources");

  const resources = performance.getEntriesByType('resource');

  // ONNX Runtime
  const ortScript = resources.find(r => r.name.includes('ort.min.js') || r.name.includes('onnxruntime'));
  if (ortScript) {
    console.log(`✅ ONNX Runtime: ${(ortScript.transferSize / 1024).toFixed(0)} KB in ${ortScript.duration.toFixed(0)}ms`);
  } else {
    console.log("❌ ONNX Runtime script not found");
  }

  // ONNX Model
  const model = resources.find(r => r.name.includes('.onnx'));
  if (model) {
    const mb = (model.transferSize / 1024 / 1024).toFixed(2);
    const sec = (model.duration / 1000).toFixed(2);
    console.log(`✅ ONNX Model: ${mb} MB in ${sec}s`);
    console.log(`   Speed: ${(model.transferSize / 1024 / model.duration).toFixed(0)} KB/s`);
  } else {
    console.log("⚠️ ONNX Model not loaded yet");
  }

  // WASM
  const wasm = resources.filter(r => r.name.endsWith('.wasm'));
  if (wasm.length > 0) {
    const totalSize = wasm.reduce((sum, r) => sum + r.transferSize, 0);
    console.log(`✅ WASM modules: ${wasm.length} files, ${(totalSize / 1024).toFixed(0)} KB total`);
  }

  console.groupEnd();
}

// ============================================================================
// FUNCTIONAL TESTING
// ============================================================================

/**
 * Test vector similarity calculation
 */
function testCosineSimilarity() {
  console.group("🧮 Cosine Similarity Test");

  function cosineSim(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  }

  // Test cases
  const tests = [
    {
      name: "Identical vectors",
      a: [1, 2, 3],
      b: [1, 2, 3],
      expected: 1.0
    },
    {
      name: "Opposite vectors",
      a: [1, 0, 0],
      b: [-1, 0, 0],
      expected: -1.0
    },
    {
      name: "Orthogonal vectors",
      a: [1, 0],
      b: [0, 1],
      expected: 0.0
    },
    {
      name: "Similar vectors",
      a: [1, 2, 3],
      b: [1, 2, 4],
      expected: 0.99 // approximately
    }
  ];

  tests.forEach(test => {
    const result = cosineSim(test.a, test.b);
    const diff = Math.abs(result - test.expected);
    const status = diff < 0.01 ? '✅' : '❌';
    console.log(`${status} ${test.name}: ${result.toFixed(4)} (expected ~${test.expected})`);
  });

  console.groupEnd();
}

/**
 * Test Voy k-d tree performance
 */
async function testVoySearch(numVectors = 1000, dimension = 384) {
  console.group(`🔍 Voy Search Test (${numVectors} vectors, dim=${dimension})`);

  if (typeof window.Voy === 'undefined') {
    console.error("❌ Voy not available");
    console.groupEnd();
    return;
  }

  // Generate random embeddings
  const embeddings = [];
  for (let i = 0; i < numVectors; i++) {
    const vec = new Float32Array(dimension);
    for (let j = 0; j < dimension; j++) {
      vec[j] = Math.random();
    }

    embeddings.push({
      id: i.toString(),
      title: `doc_${i}`,
      url: `/${i}`,
      embeddings: vec
    });
  }

  console.log(`Generated ${numVectors} random ${dimension}-dim vectors`);

  // Build index
  const buildStart = performance.now();
  const index = new Voy({ embeddings });
  const buildTime = performance.now() - buildStart;
  console.log(`✅ Index built in ${buildTime.toFixed(2)}ms`);

  // Test search
  const query = new Float32Array(dimension).map(() => Math.random());
  const searchStart = performance.now();
  const results = index.search(query, 10);
  const searchTime = performance.now() - searchStart;

  console.log(`✅ Search completed in ${searchTime.toFixed(2)}ms`);
  console.log(`   Top 3 results:`, results.slice(0, 3).map(r => ({
    id: r.id,
    similarity: r.neighbors?.[0]?.similarity?.toFixed(4)
  })));

  console.groupEnd();

  return { buildTime, searchTime, results };
}

// ============================================================================
// MONITORING
// ============================================================================

/**
 * Monitor app performance in real-time
 */
function startPerformanceMonitor(intervalMs = 5000) {
  console.log(`📊 Starting performance monitor (every ${intervalMs}ms)`);
  console.log("   Run stopPerformanceMonitor() to stop");

  window.__perfMonitor = setInterval(() => {
    console.group(`📊 Performance Snapshot @ ${new Date().toLocaleTimeString()}`);

    // Memory
    if (performance.memory) {
      const used = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
      const total = (performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2);
      console.log(`Memory: ${used} / ${total} MB`);
    }

    // FPS (approximate)
    let lastTime = performance.now();
    requestAnimationFrame(() => {
      const fps = 1000 / (performance.now() - lastTime);
      console.log(`FPS: ${fps.toFixed(1)}`);
    });

    console.groupEnd();
  }, intervalMs);
}

function stopPerformanceMonitor() {
  if (window.__perfMonitor) {
    clearInterval(window.__perfMonitor);
    delete window.__perfMonitor;
    console.log("✅ Performance monitor stopped");
  }
}

// ============================================================================
// QUICK TEST SUITE
// ============================================================================

/**
 * Run all tests in sequence
 */
async function runAllTests() {
  console.clear();
  console.log("🚀 Running GraphRAG Leptos Demo Test Suite\n");

  // 1. Prerequisites
  const prereqs = await checkPrerequisites();
  console.log("");

  if (!prereqs.ready) {
    console.error("❌ Prerequisites not met. Fix errors above and try again.");
    return;
  }

  // 2. Network
  checkNetworkResources();
  console.log("");

  // 3. Memory
  benchmarkMemory();
  console.log("");

  // 4. Math functions
  testCosineSimilarity();
  console.log("");

  // 5. Model loading (optional - takes time)
  const skipModelTest = true;
  if (!skipModelTest && prereqs.onnx) {
    console.log("Loading ONNX model (this may take 10-15 seconds)...");
    const modelResult = await benchmarkModelLoading();
    console.log("");

    if (modelResult.session) {
      await benchmarkEmbedding(modelResult.session, 5);
      console.log("");
    }
  }

  // 6. Voy search (optional)
  const skipVoyTest = false;
  if (!skipVoyTest && prereqs.voy) {
    await testVoySearch(1000, 384);
    console.log("");
  }

  console.log("✅ Test suite completed!");
  console.log("\nTip: Run individual functions for detailed testing:");
  console.log("  - benchmarkModelLoading()");
  console.log("  - testVoySearch(1000, 384)");
  console.log("  - startPerformanceMonitor()");
}

// ============================================================================
// EXPORT
// ============================================================================

console.log("📚 GraphRAG Test Helpers Loaded!");
console.log("   Run: runAllTests() to start");
console.log("   Or use individual functions - see guide for details");

// Auto-export to window
window.graphragTests = {
  checkPrerequisites,
  benchmarkModelLoading,
  benchmarkEmbedding,
  benchmarkMemory,
  checkNetworkResources,
  testCosineSimilarity,
  testVoySearch,
  startPerformanceMonitor,
  stopPerformanceMonitor,
  runAllTests
};
