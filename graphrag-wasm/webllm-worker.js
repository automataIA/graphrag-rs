// WebLLM worker entry — hosts the MLC engine off the main thread.
//
// Pairs with `CreateWebWorkerMLCEngine` on the main thread. The worker
// receives postMessage calls from the engine proxy and forwards them to
// `WebWorkerMLCEngineHandler`, which owns the actual WebGPU/inference state.
//
// See: https://github.com/mlc-ai/web-llm/tree/main/examples/web-worker

import { WebWorkerMLCEngineHandler } from "https://esm.run/@mlc-ai/web-llm";

const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (msg) => {
    handler.onmessage(msg);
};
