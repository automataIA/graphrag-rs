let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_export_2.set(idx, obj);
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedBigInt64ArrayMemory0 = null;

function getBigInt64ArrayMemory0() {
    if (cachedBigInt64ArrayMemory0 === null || cachedBigInt64ArrayMemory0.byteLength === 0) {
        cachedBigInt64ArrayMemory0 = new BigInt64Array(wasm.memory.buffer);
    }
    return cachedBigInt64ArrayMemory0;
}

function getArrayI64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getBigInt64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(
state => {
    wasm.__wbindgen_export_6.get(state.dtor)(state.a, state.b);
}
);

function makeMutClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            if (--state.cnt === 0) {
                wasm.__wbindgen_export_6.get(state.dtor)(a, state.b);
                CLOSURE_DTORS.unregister(state);
            } else {
                state.a = a;
            }
        }
    };
    real.original = state;
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function makeClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        try {
            return f(state.a, state.b, ...args);
        } finally {
            if (--state.cnt === 0) {
                wasm.__wbindgen_export_6.get(state.dtor)(state.a, state.b); state.a = 0;
                CLOSURE_DTORS.unregister(state);
            }
        }
    };
    real.original = state;
    CLOSURE_DTORS.register(real, state, state);
    return real;
}
/**
 * List of recommended WebLLM models
 * @returns {any}
 */
export function get_recommended_models() {
    const ret = wasm.get_recommended_models();
    return ret;
}

/**
 * Check if WebLLM is available in the browser
 * @returns {boolean}
 */
export function is_webllm_available() {
    const ret = wasm.is_webllm_available();
    return ret !== 0;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}
/**
 * Initialize WASM module with panic hook
 */
export function init() {
    wasm.init();
}

/**
 * Check if WebGPU is available in the browser
 * @returns {Promise<boolean>}
 */
export function check_webgpu_support() {
    const ret = wasm.check_webgpu_support();
    return ret;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_2.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}
/**
 * Check if Voy is available in the browser
 *
 * Voy must be loaded via:
 * ```html
 * <script type="module">
 *   import { Voy } from "https://cdn.jsdelivr.net/npm/voy-search@0.6.3/dist/voy.js";
 *   window.Voy = Voy;
 * </script>
 * ```
 * @returns {boolean}
 */
export function checkVoyAvailable() {
    const ret = wasm.checkVoyAvailable();
    return ret !== 0;
}

/**
 * Check WebGPU availability with detailed information
 * @returns {Promise<WebGPUInfo>}
 */
export function checkWebGPUSupport() {
    const ret = wasm.checkWebGPUSupport();
    return ret;
}

/**
 * Quick WebGPU availability check (returns bool)
 * @returns {boolean}
 */
export function isWebGPUAvailable() {
    const ret = wasm.isWebGPUAvailable();
    return ret !== 0;
}

/**
 * Get recommended ML backend based on WebGPU support
 * @returns {Promise<string>}
 */
export function getRecommendedBackend() {
    const ret = wasm.getRecommendedBackend();
    return ret;
}

/**
 * Check if ONNX Runtime Web is available
 * @returns {boolean}
 */
export function check_onnx_runtime() {
    const ret = wasm.check_onnx_runtime();
    return ret !== 0;
}

function __wbg_adapter_6(arg0, arg1, arg2) {
    wasm.closure440_externref_shim(arg0, arg1, arg2);
}

function __wbg_adapter_11(arg0, arg1, arg2) {
    wasm.closure587_externref_shim(arg0, arg1, arg2);
}

function __wbg_adapter_20(arg0, arg1, arg2) {
    wasm.closure438_externref_shim(arg0, arg1, arg2);
}

function __wbg_adapter_23(arg0, arg1, arg2) {
    wasm.closure536_externref_shim(arg0, arg1, arg2);
}

function __wbg_adapter_26(arg0, arg1) {
    wasm.wasm_bindgen__convert__closures_____invoke__h2954a777834e13c8(arg0, arg1);
}

function __wbg_adapter_29(arg0, arg1) {
    wasm.wasm_bindgen__convert__closures_____invoke__h9b7b8bf718e022e3(arg0, arg1);
}

function __wbg_adapter_342(arg0, arg1, arg2, arg3) {
    wasm.closure602_externref_shim(arg0, arg1, arg2, arg3);
}

const __wbindgen_enum_IdbTransactionMode = ["readonly", "readwrite", "versionchange", "readwriteflush", "cleanup"];

const __wbindgen_enum_ReadableStreamType = ["bytes"];

const GraphRAGFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_graphrag_free(ptr >>> 0, 1));
/**
 * GraphRAG instance for WASM
 *
 * This provides a complete client-side knowledge graph implementation
 * using Voy for vector search and IndexedDB for persistence.
 */
export class GraphRAG {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GraphRAGFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_graphrag_free(ptr, 0);
    }
    /**
     * Create a new GraphRAG instance
     *
     * # Arguments
     * * `dimension` - Embedding dimension (384 for MiniLM, 768 for BERT)
     * @param {number} dimension
     */
    constructor(dimension) {
        const ret = wasm.graphrag_new(dimension);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        GraphRAGFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Add a document to the knowledge graph
     *
     * # Arguments
     * * `id` - Unique document identifier
     * * `text` - Document text content
     * * `embedding` - Pre-computed embedding vector
     * @param {string} id
     * @param {string} text
     * @param {Float32Array} embedding
     * @returns {Promise<void>}
     */
    add_document(id, text, embedding) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(embedding, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.graphrag_add_document(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        return ret;
    }
    /**
     * Build the vector index
     *
     * Must be called after adding documents and before querying.
     * Building the k-d tree is fast (typically <100ms for 10k docs).
     * @returns {Promise<void>}
     */
    build_index() {
        const ret = wasm.graphrag_build_index(this.__wbg_ptr);
        return ret;
    }
    /**
     * Query the knowledge graph
     *
     * # Arguments
     * * `query_embedding` - Pre-computed query embedding
     * * `top_k` - Number of results to return
     *
     * # Returns
     * JSON string with array of {id, similarity, text} objects
     * @param {Float32Array} query_embedding
     * @param {number} top_k
     * @returns {Promise<string>}
     */
    query(query_embedding, top_k) {
        const ptr0 = passArrayF32ToWasm0(query_embedding, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.graphrag_query(this.__wbg_ptr, ptr0, len0, top_k);
        return ret;
    }
    /**
     * Get the number of documents in the graph
     * @returns {number}
     */
    document_count() {
        const ret = wasm.graphrag_document_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the embedding dimension
     * @returns {number}
     */
    get_dimension() {
        const ret = wasm.graphrag_get_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Check if the index has been built
     * @returns {boolean}
     */
    is_index_built() {
        const ret = wasm.graphrag_is_index_built(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get information about the vector index
     * @returns {string}
     */
    index_info() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.graphrag_index_info(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Clear all documents and reset the index
     */
    clear() {
        wasm.graphrag_clear(this.__wbg_ptr);
    }
    /**
     * Save the index to IndexedDB for persistence
     * @param {string} db_name
     * @returns {Promise<void>}
     */
    save_to_storage(db_name) {
        const ptr0 = passStringToWasm0(db_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.graphrag_save_to_storage(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Load the index from IndexedDB
     * @param {string} db_name
     * @returns {Promise<void>}
     */
    load_from_storage(db_name) {
        const ptr0 = passStringToWasm0(db_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.graphrag_load_from_storage(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
}
if (Symbol.dispose) GraphRAG.prototype[Symbol.dispose] = GraphRAG.prototype.free;

const IntoUnderlyingByteSourceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingbytesource_free(ptr >>> 0, 1));

export class IntoUnderlyingByteSource {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingByteSourceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingbytesource_free(ptr, 0);
    }
    /**
     * @returns {ReadableStreamType}
     */
    get type() {
        const ret = wasm.intounderlyingbytesource_type(this.__wbg_ptr);
        return __wbindgen_enum_ReadableStreamType[ret];
    }
    /**
     * @returns {number}
     */
    get autoAllocateChunkSize() {
        const ret = wasm.intounderlyingbytesource_autoAllocateChunkSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {ReadableByteStreamController} controller
     */
    start(controller) {
        wasm.intounderlyingbytesource_start(this.__wbg_ptr, controller);
    }
    /**
     * @param {ReadableByteStreamController} controller
     * @returns {Promise<any>}
     */
    pull(controller) {
        const ret = wasm.intounderlyingbytesource_pull(this.__wbg_ptr, controller);
        return ret;
    }
    cancel() {
        const ptr = this.__destroy_into_raw();
        wasm.intounderlyingbytesource_cancel(ptr);
    }
}
if (Symbol.dispose) IntoUnderlyingByteSource.prototype[Symbol.dispose] = IntoUnderlyingByteSource.prototype.free;

const IntoUnderlyingSinkFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingsink_free(ptr >>> 0, 1));

export class IntoUnderlyingSink {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingSinkFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingsink_free(ptr, 0);
    }
    /**
     * @param {any} chunk
     * @returns {Promise<any>}
     */
    write(chunk) {
        const ret = wasm.intounderlyingsink_write(this.__wbg_ptr, chunk);
        return ret;
    }
    /**
     * @returns {Promise<any>}
     */
    close() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.intounderlyingsink_close(ptr);
        return ret;
    }
    /**
     * @param {any} reason
     * @returns {Promise<any>}
     */
    abort(reason) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.intounderlyingsink_abort(ptr, reason);
        return ret;
    }
}
if (Symbol.dispose) IntoUnderlyingSink.prototype[Symbol.dispose] = IntoUnderlyingSink.prototype.free;

const IntoUnderlyingSourceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingsource_free(ptr >>> 0, 1));

export class IntoUnderlyingSource {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingSourceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingsource_free(ptr, 0);
    }
    /**
     * @param {ReadableStreamDefaultController} controller
     * @returns {Promise<any>}
     */
    pull(controller) {
        const ret = wasm.intounderlyingsource_pull(this.__wbg_ptr, controller);
        return ret;
    }
    cancel() {
        const ptr = this.__destroy_into_raw();
        wasm.intounderlyingsource_cancel(ptr);
    }
}
if (Symbol.dispose) IntoUnderlyingSource.prototype[Symbol.dispose] = IntoUnderlyingSource.prototype.free;

const VoyConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_voyconfig_free(ptr >>> 0, 1));
/**
 * Configuration for creating a Voy index
 */
export class VoyConfig {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VoyConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_voyconfig_free(ptr, 0);
    }
    /**
     * Embedding dimension
     * @returns {number}
     */
    get dimension() {
        const ret = wasm.__wbg_get_voyconfig_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Embedding dimension
     * @param {number} arg0
     */
    set dimension(arg0) {
        wasm.__wbg_set_voyconfig_dimension(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} dimension
     */
    constructor(dimension) {
        const ret = wasm.voyconfig_new(dimension);
        this.__wbg_ptr = ret >>> 0;
        VoyConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) VoyConfig.prototype[Symbol.dispose] = VoyConfig.prototype.free;

const VoyIndexFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_voyindex_free(ptr >>> 0, 1));
/**
 * Voy vector search index
 *
 * This wraps the JavaScript Voy class which uses k-d trees
 * for efficient nearest neighbor search in the browser.
 *
 * Modern API (v0.6+):
 * - 75KB bundle size
 * - k-d tree algorithm
 * - Cosine similarity search
 * - Serialization support
 */
export class VoyIndex {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(VoyIndex.prototype);
        obj.__wbg_ptr = ptr;
        VoyIndexFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VoyIndexFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_voyindex_free(ptr, 0);
    }
    /**
     * Create a new Voy index from embeddings
     *
     * # Arguments
     * * `embeddings` - Array of embeddings (Float32Array or nested arrays)
     * * `dimension` - Embedding dimension
     *
     * # Example
     * ```javascript
     * const embeddings = [
     *   { id: "0", title: "doc1", url: "/0", embeddings: [0.1, 0.2, 0.3] },
     *   { id: "1", title: "doc2", url: "/1", embeddings: [0.4, 0.5, 0.6] }
     * ];
     * const index = VoyIndex.from_embeddings(embeddings, 3);
     * ```
     * @param {any} embeddings
     * @param {number} dimension
     * @returns {VoyIndex}
     */
    static fromEmbeddings(embeddings, dimension) {
        const ret = wasm.voyindex_fromEmbeddings(embeddings, dimension);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return VoyIndex.__wrap(ret[0]);
    }
    /**
     * Create an empty Voy index
     * @param {number} _dimension
     * @returns {VoyIndex}
     */
    static createEmpty(_dimension) {
        const ret = wasm.voyindex_createEmpty(_dimension);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return VoyIndex.__wrap(ret[0]);
    }
    /**
     * Add a single embedding to the index
     *
     * # Arguments
     * * `embedding` - Vector embedding as Float32Array
     * * `id` - Document ID
     * * `title` - Document title
     * * `url` - Document URL
     * @param {any} embedding
     * @param {string} id
     * @param {string} title
     * @param {string} url
     */
    add_embedding(embedding, id, title, url) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(title, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.voyindex_add_embedding(this.__wbg_ptr, embedding, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Search for k nearest neighbors
     *
     * # Arguments
     * * `query` - Query embedding as Float32Array
     * * `k` - Number of nearest neighbors to return
     *
     * # Returns
     * Array of search results with indices and distances
     * @param {any} query
     * @param {number} k
     * @returns {any}
     */
    search_neighbors(query, k) {
        const ret = wasm.voyindex_search_neighbors(this.__wbg_ptr, query, k);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Search and return parsed results with structured format
     *
     * # Arguments
     * * `query` - Query embedding as Float32Array
     * * `k` - Number of results to return
     *
     * # Returns
     * Array of {id, distance, title, url} objects
     * @param {any} query
     * @param {number} k
     * @returns {any}
     */
    searchParsed(query, k) {
        const ret = wasm.voyindex_searchParsed(this.__wbg_ptr, query, k);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Serialize the index for storage (returns JSON)
     * @returns {string}
     */
    serialize() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.voyindex_serialize(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Clear all embeddings from index
     */
    clear() {
        wasm.voyindex_clear(this.__wbg_ptr);
    }
    /**
     * Get the number of indexed embeddings (if available)
     * @returns {number}
     */
    size() {
        const ret = wasm.voyindex_size(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
}
if (Symbol.dispose) VoyIndex.prototype[Symbol.dispose] = VoyIndex.prototype.free;

const VoySearchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_voysearchresult_free(ptr >>> 0, 1));
/**
 * Result from a Voy search query
 */
export class VoySearchResult {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VoySearchResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_voysearchresult_free(ptr, 0);
    }
    /**
     * Index of the matching item
     * @returns {number}
     */
    get id() {
        const ret = wasm.__wbg_get_voyconfig_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Index of the matching item
     * @param {number} arg0
     */
    set id(arg0) {
        wasm.__wbg_set_voyconfig_dimension(this.__wbg_ptr, arg0);
    }
    /**
     * Distance to the query vector
     * @returns {number}
     */
    get distance() {
        const ret = wasm.__wbg_get_voysearchresult_distance(this.__wbg_ptr);
        return ret;
    }
    /**
     * Distance to the query vector
     * @param {number} arg0
     */
    set distance(arg0) {
        wasm.__wbg_set_voysearchresult_distance(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) VoySearchResult.prototype[Symbol.dispose] = VoySearchResult.prototype.free;

const WasmEmbedderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmembedder_free(ptr >>> 0, 1));
/**
 * WASM bindings for embedder
 */
export class WasmEmbedder {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmEmbedder.prototype);
        obj.__wbg_ptr = ptr;
        WasmEmbedderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEmbedderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmembedder_free(ptr, 0);
    }
    /**
     * Create a new embedder
     *
     * Automatically detects and uses the best available backend.
     *
     * # Arguments
     * * `model_name` - Model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
     * * `dimension` - Embedding dimension (384 for MiniLM, 768 for BERT)
     * @param {string} model_name
     * @param {number} dimension
     * @returns {Promise<WasmEmbedder>}
     */
    static new(model_name, dimension) {
        const ptr0 = passStringToWasm0(model_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_new(ptr0, len0, dimension);
        return ret;
    }
    /**
     * Embed a single text
     *
     * # Arguments
     * * `text` - Text to embed
     *
     * # Returns
     * Float32Array with embedding vector
     * @param {string} text
     * @returns {Promise<Float32Array>}
     */
    embed(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_embed(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Embed multiple texts (batched)
     *
     * # Arguments
     * * `texts` - Array of texts to embed
     *
     * # Returns
     * Array of Float32Array embeddings
     * @param {string[]} texts
     * @returns {Promise<any>}
     */
    embed_batch(texts) {
        const ptr0 = passArrayJsValueToWasm0(texts, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_embed_batch(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Get embedding dimension
     * @returns {number}
     */
    dimension() {
        const ret = wasm.wasmembedder_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Check if GPU acceleration is active
     * @returns {boolean}
     */
    is_gpu_accelerated() {
        const ret = wasm.wasmembedder_is_gpu_accelerated(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Load model from Cache API
     *
     * Downloads model files from HuggingFace if not cached.
     * This method should be called before calling embed() or embed_batch().
     *
     * # Example
     * ```javascript
     * const embedder = await WasmEmbedder.new("sentence-transformers/all-MiniLM-L6-v2", 384);
     * await embedder.load_model(); // Download and cache model
     * const embedding = await embedder.embed("Hello world");
     * ```
     * @returns {Promise<void>}
     */
    load_model() {
        const ret = wasm.wasmembedder_load_model(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmEmbedder.prototype[Symbol.dispose] = WasmEmbedder.prototype.free;

const WasmOnnxEmbedderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmonnxembedder_free(ptr >>> 0, 1));
/**
 * WASM bindings for ONNX embedder
 */
export class WasmOnnxEmbedder {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmOnnxEmbedderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmonnxembedder_free(ptr, 0);
    }
    /**
     * Create a new ONNX embedder
     *
     * # Arguments
     * * `dimension` - Embedding dimension (384 for MiniLM, 768 for BERT)
     * @param {number} dimension
     */
    constructor(dimension) {
        const ret = wasm.wasmonnxembedder_new(dimension);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmOnnxEmbedderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Load ONNX model
     *
     * # Arguments
     * * `model_url` - URL to ONNX model file (e.g., "./models/minilm-l6.onnx")
     * * `use_webgpu` - Use WebGPU acceleration (default: true)
     * @param {string} model_url
     * @param {boolean | null} [use_webgpu]
     * @returns {Promise<void>}
     */
    load_model(model_url, use_webgpu) {
        const ptr0 = passStringToWasm0(model_url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmonnxembedder_load_model(this.__wbg_ptr, ptr0, len0, isLikeNone(use_webgpu) ? 0xFFFFFF : use_webgpu ? 1 : 0);
        return ret;
    }
    /**
     * Generate embedding for text
     * @param {string} text
     * @returns {Promise<Float32Array>}
     */
    embed(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmonnxembedder_embed(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Generate embeddings for multiple texts
     * @param {string[]} texts
     * @returns {Promise<Array<any>>}
     */
    embed_batch(texts) {
        const ptr0 = passArrayJsValueToWasm0(texts, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmonnxembedder_embed_batch(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Get embedding dimension
     * @returns {number}
     */
    dimension() {
        const ret = wasm.wasmonnxembedder_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Check if model is loaded
     * @returns {boolean}
     */
    is_loaded() {
        const ret = wasm.wasmonnxembedder_is_loaded(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get model name
     * @returns {string | undefined}
     */
    model_name() {
        const ret = wasm.wasmonnxembedder_model_name(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
}
if (Symbol.dispose) WasmOnnxEmbedder.prototype[Symbol.dispose] = WasmOnnxEmbedder.prototype.free;

const WasmWebLLMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmwebllm_free(ptr >>> 0, 1));
/**
 * WASM bindings for WebLLM
 */
export class WasmWebLLM {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmWebLLM.prototype);
        obj.__wbg_ptr = ptr;
        WasmWebLLMFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmWebLLMFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmwebllm_free(ptr, 0);
    }
    /**
     * Initialize WebLLM
     *
     * # Arguments
     * * `model_id` - Model identifier (e.g., "Phi-3-mini-4k-instruct-q4f16_1-MLC")
     * @param {string} model_id
     * @returns {Promise<WasmWebLLM>}
     */
    static new(model_id) {
        const ptr0 = passStringToWasm0(model_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmwebllm_new(ptr0, len0);
        return ret;
    }
    /**
     * Initialize with progress callback
     *
     * # Arguments
     * * `model_id` - Model identifier
     * * `on_progress` - JavaScript callback function(progress: number, text: string)
     * @param {string} model_id
     * @param {Function} on_progress
     * @returns {Promise<WasmWebLLM>}
     */
    static new_with_progress(model_id, on_progress) {
        const ptr0 = passStringToWasm0(model_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmwebllm_new_with_progress(ptr0, len0, on_progress);
        return ret;
    }
    /**
     * Send a simple message and get response
     *
     * # Arguments
     * * `message` - User message
     *
     * # Returns
     * Assistant's response
     * @param {string} message
     * @returns {Promise<string>}
     */
    ask(message) {
        const ptr0 = passStringToWasm0(message, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmwebllm_ask(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Send chat messages and get response
     *
     * # Arguments
     * * `messages` - JSON array of {role: string, content: string}
     * * `temperature` - Sampling temperature (optional)
     * * `max_tokens` - Maximum tokens (optional)
     * @param {any} messages
     * @param {number | null} [temperature]
     * @param {number | null} [max_tokens]
     * @returns {Promise<string>}
     */
    chat(messages, temperature, max_tokens) {
        const ret = wasm.wasmwebllm_chat(this.__wbg_ptr, messages, !isLikeNone(temperature), isLikeNone(temperature) ? 0 : temperature, isLikeNone(max_tokens) ? 0x100000001 : (max_tokens) >>> 0);
        return ret;
    }
    /**
     * Get the model ID
     * @returns {string}
     */
    model_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmwebllm_model_id(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Stream chat response with real-time token generation
     *
     * # Arguments
     * * `messages` - JSON array of {role: string, content: string}
     * * `on_chunk` - JavaScript callback function(chunk: string)
     * * `temperature` - Sampling temperature (optional)
     * * `max_tokens` - Maximum tokens (optional)
     *
     * # Returns
     * Complete response text
     * @param {any} messages
     * @param {Function} on_chunk
     * @param {number | null} [temperature]
     * @param {number | null} [max_tokens]
     * @returns {Promise<string>}
     */
    chat_stream(messages, on_chunk, temperature, max_tokens) {
        const ret = wasm.wasmwebllm_chat_stream(this.__wbg_ptr, messages, on_chunk, !isLikeNone(temperature), isLikeNone(temperature) ? 0 : temperature, isLikeNone(max_tokens) ? 0x100000001 : (max_tokens) >>> 0);
        return ret;
    }
}
if (Symbol.dispose) WasmWebLLM.prototype[Symbol.dispose] = WasmWebLLM.prototype.free;

const WebGPUInfoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_webgpuinfo_free(ptr >>> 0, 1));
/**
 * WebGPU capability information
 */
export class WebGPUInfo {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WebGPUInfo.prototype);
        obj.__wbg_ptr = ptr;
        WebGPUInfoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WebGPUInfoFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_webgpuinfo_free(ptr, 0);
    }
    /**
     * Whether WebGPU is available
     * @returns {boolean}
     */
    get available() {
        const ret = wasm.__wbg_get_webgpuinfo_available(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether WebGPU is available
     * @param {boolean} arg0
     */
    set available(arg0) {
        wasm.__wbg_set_webgpuinfo_available(this.__wbg_ptr, arg0);
    }
    /**
     * Maximum buffer size in bytes
     * @returns {bigint}
     */
    get max_buffer_size() {
        const ret = wasm.__wbg_get_webgpuinfo_max_buffer_size(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Maximum buffer size in bytes
     * @param {bigint} arg0
     */
    set max_buffer_size(arg0) {
        wasm.__wbg_set_webgpuinfo_max_buffer_size(this.__wbg_ptr, arg0);
    }
    /**
     * Maximum texture dimension
     * @returns {number}
     */
    get max_texture_dimension() {
        const ret = wasm.__wbg_get_webgpuinfo_max_texture_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Maximum texture dimension
     * @param {number} arg0
     */
    set max_texture_dimension(arg0) {
        wasm.__wbg_set_webgpuinfo_max_texture_dimension(this.__wbg_ptr, arg0);
    }
    /**
     * Get GPU vendor name
     * @returns {string}
     */
    get vendor() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webgpuinfo_vendor(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get GPU architecture
     * @returns {string}
     */
    get architecture() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webgpuinfo_architecture(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get browser support level
     * @returns {string}
     */
    get browserSupport() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webgpuinfo_browserSupport(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get a human-readable summary
     * @returns {string}
     */
    getSummary() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webgpuinfo_getSummary(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) WebGPUInfo.prototype[Symbol.dispose] = WebGPUInfo.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_Error_e17e777aac105295 = function(arg0, arg1) {
        const ret = Error(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_InferenceSession_0547dff319075f63 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = ort.InferenceSession(getStringFromWasm0(arg0, arg1), arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_Number_998bea33bd87c3e0 = function(arg0) {
        const ret = Number(arg0);
        return ret;
    };
    imports.wbg.__wbg_String_8f0eb39a4a4c2f66 = function(arg0, arg1) {
        const ret = String(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_addEventListener_775911544ac9d643 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        arg0.addEventListener(getStringFromWasm0(arg1, arg2), arg3);
    }, arguments) };
    imports.wbg.__wbg_add_ec8f561cdbc98b04 = function(arg0, arg1) {
        const ret = arg0.add(arg1);
        return ret;
    };
    imports.wbg.__wbg_apply_55d63d092a912d6f = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = Reflect.apply(arg0, arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_arrayBuffer_9c99b8e2809e8cbb = function() { return handleError(function (arg0) {
        const ret = arg0.arrayBuffer();
        return ret;
    }, arguments) };
    imports.wbg.__wbg_body_8822ca55cb3730d2 = function(arg0) {
        const ret = arg0.body;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_buffer_8d40b1d762fb3c66 = function(arg0) {
        const ret = arg0.buffer;
        return ret;
    };
    imports.wbg.__wbg_byobRequest_2c036bceca1e6037 = function(arg0) {
        const ret = arg0.byobRequest;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_byteLength_331a6b5545834024 = function(arg0) {
        const ret = arg0.byteLength;
        return ret;
    };
    imports.wbg.__wbg_byteOffset_49a5b5608000358b = function(arg0) {
        const ret = arg0.byteOffset;
        return ret;
    };
    imports.wbg.__wbg_caches_12adc7af691f9083 = function() { return handleError(function (arg0) {
        const ret = arg0.caches;
        return ret;
    }, arguments) };
    imports.wbg.__wbg_call_13410aac570ffff7 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_call_641db1bb5db5a579 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        const ret = arg0.call(arg1, arg2, arg3);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_call_a5400b25a865cfd8 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.call(arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_cancelBubble_a4c48803e199b5e8 = function(arg0) {
        const ret = arg0.cancelBubble;
        return ret;
    };
    imports.wbg.__wbg_clear_4af4b3fdf796585a = function(arg0) {
        arg0.clear();
    };
    imports.wbg.__wbg_cloneNode_79d46b18d5619863 = function() { return handleError(function (arg0) {
        const ret = arg0.cloneNode();
        return ret;
    }, arguments) };
    imports.wbg.__wbg_cloneNode_82bea7899ad17d25 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.cloneNode(arg1 !== 0);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_close_cccada6053ee3a65 = function() { return handleError(function (arg0) {
        arg0.close();
    }, arguments) };
    imports.wbg.__wbg_close_d71a78219dc23e91 = function() { return handleError(function (arg0) {
        arg0.close();
    }, arguments) };
    imports.wbg.__wbg_composedPath_e5b3f0b3e8415bb5 = function(arg0) {
        const ret = arg0.composedPath();
        return ret;
    };
    imports.wbg.__wbg_content_a26016a510c10d06 = function(arg0) {
        const ret = arg0.content;
        return ret;
    };
    imports.wbg.__wbg_createComment_08abf524559fd4d7 = function(arg0, arg1, arg2) {
        const ret = arg0.createComment(getStringFromWasm0(arg1, arg2));
        return ret;
    };
    imports.wbg.__wbg_createElementNS_ffbb8bb20b2a7e4c = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        const ret = arg0.createElementNS(arg1 === 0 ? undefined : getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_createElement_4909dfa2011f2abe = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.createElement(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_createObjectStore_2bc52da689ca2130 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.createObjectStore(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_createTextNode_c71a51271fadf515 = function(arg0, arg1, arg2) {
        const ret = arg0.createTextNode(getStringFromWasm0(arg1, arg2));
        return ret;
    };
    imports.wbg.__wbg_debug_7f3000e7358ea482 = function(arg0, arg1, arg2, arg3) {
        console.debug(arg0, arg1, arg2, arg3);
    };
    imports.wbg.__wbg_deleteProperty_5fe99f4fd0f66ebe = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.deleteProperty(arg0, arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_document_7d29d139bd619045 = function(arg0) {
        const ret = arg0.document;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_done_75ed0ee6dd243d9d = function(arg0) {
        const ret = arg0.done;
        return ret;
    };
    imports.wbg.__wbg_enqueue_452bc2343d1c2ff9 = function() { return handleError(function (arg0, arg1) {
        arg0.enqueue(arg1);
    }, arguments) };
    imports.wbg.__wbg_error_0889f151acea569e = function(arg0, arg1, arg2, arg3) {
        console.error(arg0, arg1, arg2, arg3);
    };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_error_99981e16d476aa5c = function(arg0) {
        console.error(arg0);
    };
    imports.wbg.__wbg_fetch_44b6058021aef5e3 = function(arg0, arg1) {
        const ret = arg0.fetch(arg1);
        return ret;
    };
    imports.wbg.__wbg_firstElementChild_27076cbfeed86254 = function(arg0) {
        const ret = arg0.firstElementChild;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_from_88bc52ce20ba6318 = function(arg0) {
        const ret = Array.from(arg0);
        return ret;
    };
    imports.wbg.__wbg_get_0da715ceaecea5c8 = function(arg0, arg1) {
        const ret = arg0[arg1 >>> 0];
        return ret;
    };
    imports.wbg.__wbg_get_1b2c33a63c4be73f = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.get(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_get_458e874b43b18b25 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(arg0, arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_getwithrefkey_1dc361bd10053bfe = function(arg0, arg1) {
        const ret = arg0[arg1];
        return ret;
    };
    imports.wbg.__wbg_host_484d55073e076054 = function(arg0) {
        const ret = arg0.host;
        return ret;
    };
    imports.wbg.__wbg_indexedDB_1956995e4297311c = function() { return handleError(function (arg0) {
        const ret = arg0.indexedDB;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    }, arguments) };
    imports.wbg.__wbg_info_15c3631232fceddb = function(arg0, arg1, arg2, arg3) {
        console.info(arg0, arg1, arg2, arg3);
    };
    imports.wbg.__wbg_insertBefore_30228206e8f1d3fb = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.insertBefore(arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_instanceof_ArrayBuffer_67f3012529f6a2dd = function(arg0) {
        let result;
        try {
            result = arg0 instanceof ArrayBuffer;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Cache_5728ea8b04ac8a14 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Cache;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Element_162e4334c7d6f450 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Element;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_IdbDatabase_6e6efef94c4a355d = function(arg0) {
        let result;
        try {
            result = arg0 instanceof IDBDatabase;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_IdbOpenDbRequest_2be27facb05c6739 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof IDBOpenDBRequest;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Object_fbf5fef4952ff29b = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Object;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Response_50fde2cd696850bf = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Response;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_ShadowRoot_f3723967133597a3 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof ShadowRoot;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Uint8Array_9a8378d955933db7 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Uint8Array;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Window_12d20d558ef92592 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Window;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isArray_030cce220591fb41 = function(arg0) {
        const ret = Array.isArray(arg0);
        return ret;
    };
    imports.wbg.__wbg_isSafeInteger_1c0d1af5542e102a = function(arg0) {
        const ret = Number.isSafeInteger(arg0);
        return ret;
    };
    imports.wbg.__wbg_iterator_f370b34483c71a1c = function() {
        const ret = Symbol.iterator;
        return ret;
    };
    imports.wbg.__wbg_key_caac8fafdd6d5317 = function(arg0, arg1) {
        const ret = arg1.key;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_length_186546c51cd61acd = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_6bb7e81f9d7713e4 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_a8cca01d07ea9653 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_log_20ff6e9a1a3518e4 = function(arg0, arg1) {
        console.log(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_log_433daef57fab356f = function(arg0, arg1) {
        console.log(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_log_6c7b5f4f00b8ce3f = function(arg0) {
        console.log(arg0);
    };
    imports.wbg.__wbg_log_ddbf5bc3d4dae44c = function(arg0, arg1, arg2, arg3) {
        console.log(arg0, arg1, arg2, arg3);
    };
    imports.wbg.__wbg_match_bd877700647a2a0c = function(arg0, arg1, arg2) {
        const ret = arg0.match(getStringFromWasm0(arg1, arg2));
        return ret;
    };
    imports.wbg.__wbg_navigator_65d5ad763926b868 = function(arg0) {
        const ret = arg0.navigator;
        return ret;
    };
    imports.wbg.__wbg_new0_b0a0a38c201e6df5 = function() {
        const ret = new Date();
        return ret;
    };
    imports.wbg.__wbg_new_19c25a3f2fa63a02 = function() {
        const ret = new Object();
        return ret;
    };
    imports.wbg.__wbg_new_1f3a344cf3123716 = function() {
        const ret = new Array();
        return ret;
    };
    imports.wbg.__wbg_new_2e3c58a15f39f5f9 = function(arg0, arg1) {
        try {
            var state0 = {a: arg0, b: arg1};
            var cb0 = (arg0, arg1) => {
                const a = state0.a;
                state0.a = 0;
                try {
                    return __wbg_adapter_342(a, state0.b, arg0, arg1);
                } finally {
                    state0.a = a;
                }
            };
            const ret = new Promise(cb0);
            return ret;
        } finally {
            state0.a = state0.b = 0;
        }
    };
    imports.wbg.__wbg_new_2ff1f68f3676ea53 = function() {
        const ret = new Map();
        return ret;
    };
    imports.wbg.__wbg_new_638ebfaedbf32a5e = function(arg0) {
        const ret = new Uint8Array(arg0);
        return ret;
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return ret;
    };
    imports.wbg.__wbg_new_da9dc54c5db29dfa = function(arg0, arg1) {
        const ret = new Error(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_new_f0e1134532e689ac = function(arg0) {
        const ret = new window.Voy(arg0);
        return ret;
    };
    imports.wbg.__wbg_newfromslice_074c56947bd43469 = function(arg0, arg1) {
        const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_newfromslice_cda906eb14b58470 = function(arg0, arg1) {
        const ret = new BigInt64Array(getArrayI64FromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_newfromslice_eb3df67955925a7c = function(arg0, arg1) {
        const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_newnoargs_254190557c45b4ec = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_newwithbyteoffsetandlength_e8f53910b4d42b45 = function(arg0, arg1, arg2) {
        const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_newwithoptbuffersource_aaea28bc72faf640 = function() { return handleError(function (arg0) {
        const ret = new Response(arg0);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_newwithstr_1bc70be98f2e7425 = function() { return handleError(function (arg0, arg1) {
        const ret = new Request(getStringFromWasm0(arg0, arg1));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_next_5b3530e612fde77d = function(arg0) {
        const ret = arg0.next;
        return ret;
    };
    imports.wbg.__wbg_next_692e82279131b03c = function() { return handleError(function (arg0) {
        const ret = arg0.next();
        return ret;
    }, arguments) };
    imports.wbg.__wbg_now_1e80617bcee43265 = function() {
        const ret = Date.now();
        return ret;
    };
    imports.wbg.__wbg_objectStore_b2a5b80b2e5c5f8b = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.objectStore(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_of_30e97a7ad6e3518b = function(arg0) {
        const ret = Array.of(arg0);
        return ret;
    };
    imports.wbg.__wbg_of_d0e190785e1ebbb6 = function(arg0, arg1) {
        const ret = Array.of(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbg_open_7281831ed8ff7bd2 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        const ret = arg0.open(getStringFromWasm0(arg1, arg2), arg3 >>> 0);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_open_863b0b16d3c525d4 = function(arg0, arg1, arg2) {
        const ret = arg0.open(getStringFromWasm0(arg1, arg2));
        return ret;
    };
    imports.wbg.__wbg_parentNode_cc820baee7401ca3 = function(arg0) {
        const ret = arg0.parentNode;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_preventDefault_fab9a085b3006058 = function(arg0) {
        arg0.preventDefault();
    };
    imports.wbg.__wbg_prototypesetcall_3d4a26c1ed734349 = function(arg0, arg1, arg2) {
        Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
    };
    imports.wbg.__wbg_prototypesetcall_5521f1dd01df76fd = function(arg0, arg1, arg2) {
        Float32Array.prototype.set.call(getArrayF32FromWasm0(arg0, arg1), arg2);
    };
    imports.wbg.__wbg_push_330b2eb93e4e1212 = function(arg0, arg1) {
        const ret = arg0.push(arg1);
        return ret;
    };
    imports.wbg.__wbg_put_531712f439412bc0 = function(arg0, arg1, arg2) {
        const ret = arg0.put(arg1, arg2);
        return ret;
    };
    imports.wbg.__wbg_put_cdfadd5d7f714201 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.put(arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_queueMicrotask_25d0739ac89e8c88 = function(arg0) {
        queueMicrotask(arg0);
    };
    imports.wbg.__wbg_queueMicrotask_4488407636f5bf24 = function(arg0) {
        const ret = arg0.queueMicrotask;
        return ret;
    };
    imports.wbg.__wbg_random_7ed63a0b38ee3b75 = function() {
        const ret = Math.random();
        return ret;
    };
    imports.wbg.__wbg_removeAttribute_cf35412842be6ae4 = function() { return handleError(function (arg0, arg1, arg2) {
        arg0.removeAttribute(getStringFromWasm0(arg1, arg2));
    }, arguments) };
    imports.wbg.__wbg_removeEventListener_6d5be9c2821a511e = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        arg0.removeEventListener(getStringFromWasm0(arg1, arg2), arg3);
    }, arguments) };
    imports.wbg.__wbg_remove_0cd5aff2ca3eb753 = function(arg0) {
        arg0.remove();
    };
    imports.wbg.__wbg_remove_fec7bce376b31b32 = function(arg0) {
        arg0.remove();
    };
    imports.wbg.__wbg_resolve_4055c623acdd6a1b = function(arg0) {
        const ret = Promise.resolve(arg0);
        return ret;
    };
    imports.wbg.__wbg_respond_6c2c4e20ef85138e = function() { return handleError(function (arg0, arg1) {
        arg0.respond(arg1 >>> 0);
    }, arguments) };
    imports.wbg.__wbg_result_825a6aeeb31189d2 = function() { return handleError(function (arg0) {
        const ret = arg0.result;
        return ret;
    }, arguments) };
    imports.wbg.__wbg_run_247071e8ab5c554b = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.run(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_search_7397b40965630a19 = function(arg0, arg1, arg2) {
        const ret = arg0.search(arg1, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_serialize_dc8dc31207dd6151 = function(arg0) {
        const ret = arg0.serialize();
        return ret;
    };
    imports.wbg.__wbg_setAttribute_d1baf9023ad5696f = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        arg0.setAttribute(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
    }, arguments) };
    imports.wbg.__wbg_setProperty_a4431938dd3e6945 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        arg0.setProperty(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
    }, arguments) };
    imports.wbg.__wbg_set_1353b2a5e96bc48c = function(arg0, arg1, arg2) {
        arg0.set(getArrayU8FromWasm0(arg1, arg2));
    };
    imports.wbg.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
        arg0[arg1] = arg2;
    };
    imports.wbg.__wbg_set_453345bcda80b89a = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = Reflect.set(arg0, arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_set_90f6c0f7bd8c0415 = function(arg0, arg1, arg2) {
        arg0[arg1 >>> 0] = arg2;
    };
    imports.wbg.__wbg_set_b7f1cf4fae26fe2a = function(arg0, arg1, arg2) {
        const ret = arg0.set(arg1, arg2);
        return ret;
    };
    imports.wbg.__wbg_setinnerHTML_34e240d6b8e8260c = function(arg0, arg1, arg2) {
        arg0.innerHTML = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_setnodeValue_629799145cb84fd8 = function(arg0, arg1, arg2) {
        arg0.nodeValue = arg1 === 0 ? undefined : getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_setonerror_bcdbd7f3921ffb1f = function(arg0, arg1) {
        arg0.onerror = arg1;
    };
    imports.wbg.__wbg_setonsuccess_ffb2ddb27ce681d8 = function(arg0, arg1) {
        arg0.onsuccess = arg1;
    };
    imports.wbg.__wbg_setonupgradeneeded_4e32d1c6a08c4257 = function(arg0, arg1) {
        arg0.onupgradeneeded = arg1;
    };
    imports.wbg.__wbg_shiftKey_7793232603bd5f81 = function(arg0) {
        const ret = arg0.shiftKey;
        return ret;
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = arg1.stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_8921f820c2ce3f12 = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_f0a4409105898184 = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_995b214ae681ff99 = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_cde3890479c675ea = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_stringify_b98c93d0a190446a = function() { return handleError(function (arg0) {
        const ret = JSON.stringify(arg0);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_style_32a3c8393b46a115 = function(arg0) {
        const ret = arg0.style;
        return ret;
    };
    imports.wbg.__wbg_target_f2c963b447be6283 = function(arg0) {
        const ret = arg0.target;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_then_b33a773d723afa3e = function(arg0, arg1, arg2) {
        const ret = arg0.then(arg1, arg2);
        return ret;
    };
    imports.wbg.__wbg_then_e22500defe16819f = function(arg0, arg1) {
        const ret = arg0.then(arg1);
        return ret;
    };
    imports.wbg.__wbg_toISOString_f5382b37d44a0082 = function(arg0) {
        const ret = arg0.toISOString();
        return ret;
    };
    imports.wbg.__wbg_transaction_553a104dd139f032 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        const ret = arg0.transaction(getStringFromWasm0(arg1, arg2), __wbindgen_enum_IdbTransactionMode[arg3]);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_transaction_fc84f03ee76124ed = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.transaction(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_userAgent_2e89808dc5dc17d7 = function() { return handleError(function (arg0, arg1) {
        const ret = arg1.userAgent;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    }, arguments) };
    imports.wbg.__wbg_value_dd9372230531eade = function(arg0) {
        const ret = arg0.value;
        return ret;
    };
    imports.wbg.__wbg_value_fdf54c7557edc2e8 = function(arg0, arg1) {
        const ret = arg1.value;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_view_91cc97d57ab30530 = function(arg0) {
        const ret = arg0.view;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_warn_90eb15d986910fe9 = function(arg0, arg1, arg2, arg3) {
        console.warn(arg0, arg1, arg2, arg3);
    };
    imports.wbg.__wbg_warn_e2ada06313f92f09 = function(arg0) {
        console.warn(arg0);
    };
    imports.wbg.__wbg_wasmembedder_new = function(arg0) {
        const ret = WasmEmbedder.__wrap(arg0);
        return ret;
    };
    imports.wbg.__wbg_wasmwebllm_new = function(arg0) {
        const ret = WasmWebLLM.__wrap(arg0);
        return ret;
    };
    imports.wbg.__wbg_wbindgenbigintgetasi64_ac743ece6ab9bba1 = function(arg0, arg1) {
        const v = arg1;
        const ret = typeof(v) === 'bigint' ? v : undefined;
        getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbg_wbindgenbooleanget_3fe6f642c7d97746 = function(arg0) {
        const v = arg0;
        const ret = typeof(v) === 'boolean' ? v : undefined;
        return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
    };
    imports.wbg.__wbg_wbindgencbdrop_eb10308566512b88 = function(arg0) {
        const obj = arg0.original;
        if (obj.cnt-- == 1) {
            obj.a = 0;
            return true;
        }
        const ret = false;
        return ret;
    };
    imports.wbg.__wbg_wbindgendebugstring_99ef257a3ddda34d = function(arg0, arg1) {
        const ret = debugString(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_wbindgenin_d7a1ee10933d2d55 = function(arg0, arg1) {
        const ret = arg0 in arg1;
        return ret;
    };
    imports.wbg.__wbg_wbindgenisbigint_ecb90cc08a5a9154 = function(arg0) {
        const ret = typeof(arg0) === 'bigint';
        return ret;
    };
    imports.wbg.__wbg_wbindgenisfalsy_03f4059e2ea4ee87 = function(arg0) {
        const ret = !arg0;
        return ret;
    };
    imports.wbg.__wbg_wbindgenisfunction_8cee7dce3725ae74 = function(arg0) {
        const ret = typeof(arg0) === 'function';
        return ret;
    };
    imports.wbg.__wbg_wbindgenisnull_f3037694abe4d97a = function(arg0) {
        const ret = arg0 === null;
        return ret;
    };
    imports.wbg.__wbg_wbindgenisobject_307a53c6bd97fbf8 = function(arg0) {
        const val = arg0;
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbg_wbindgenisstring_d4fa939789f003b0 = function(arg0) {
        const ret = typeof(arg0) === 'string';
        return ret;
    };
    imports.wbg.__wbg_wbindgenisundefined_c4b71d073b92f3c5 = function(arg0) {
        const ret = arg0 === undefined;
        return ret;
    };
    imports.wbg.__wbg_wbindgenjsvaleq_e6f2ad59ccae1b58 = function(arg0, arg1) {
        const ret = arg0 === arg1;
        return ret;
    };
    imports.wbg.__wbg_wbindgenjsvallooseeq_9bec8c9be826bed1 = function(arg0, arg1) {
        const ret = arg0 == arg1;
        return ret;
    };
    imports.wbg.__wbg_wbindgennumberget_f74b4c7525ac05cb = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbg_wbindgenstringget_0f16a6ddddef376f = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'string' ? obj : undefined;
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_wbindgenthrow_451ec1a8469d7eb6 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_webgpuinfo_new = function(arg0) {
        const ret = WebGPUInfo.__wrap(arg0);
        return ret;
    };
    imports.wbg.__wbindgen_cast_030792f2cd23b2d3 = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 535, function: Function { arguments: [], shim_idx: 538, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, 535, __wbg_adapter_29);
        return ret;
    };
    imports.wbg.__wbindgen_cast_06b050ab0e7bdf14 = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 437, function: Function { arguments: [], shim_idx: 442, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, 437, __wbg_adapter_26);
        return ret;
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_cast_34eb5bcae3c4d8be = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 535, function: Function { arguments: [NamedExternref("Event")], shim_idx: 536, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, 535, __wbg_adapter_23);
        return ret;
    };
    imports.wbg.__wbindgen_cast_38bab93e5e373874 = function(arg0, arg1) {
        var v0 = getArrayF32FromWasm0(arg0, arg1).slice();
        wasm.__wbindgen_free(arg0, arg1 * 4, 4);
        // Cast intrinsic for `Vector(F32) -> Externref`.
        const ret = v0;
        return ret;
    };
    imports.wbg.__wbindgen_cast_4625c577ab2ec9ee = function(arg0) {
        // Cast intrinsic for `U64 -> Externref`.
        const ret = BigInt.asUintN(64, arg0);
        return ret;
    };
    imports.wbg.__wbindgen_cast_89c149fd90f02159 = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 437, function: Function { arguments: [NamedExternref("IDBVersionChangeEvent")], shim_idx: 440, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, 437, __wbg_adapter_6);
        return ret;
    };
    imports.wbg.__wbindgen_cast_9872e2fd90893faf = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 437, function: Function { arguments: [Externref], shim_idx: 438, ret: Unit, inner_ret: Some(Unit) }, mutable: false }) -> Externref`.
        const ret = makeClosure(arg0, arg1, 437, __wbg_adapter_20);
        return ret;
    };
    imports.wbg.__wbindgen_cast_9ae0607507abb057 = function(arg0) {
        // Cast intrinsic for `I64 -> Externref`.
        const ret = arg0;
        return ret;
    };
    imports.wbg.__wbindgen_cast_a71ab05db4ece344 = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 586, function: Function { arguments: [Externref], shim_idx: 587, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, 586, __wbg_adapter_11);
        return ret;
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return ret;
    };
    imports.wbg.__wbindgen_cast_e6fcbc8561f28563 = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 437, function: Function { arguments: [NamedExternref("Event")], shim_idx: 440, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, 437, __wbg_adapter_6);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_2;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedBigInt64ArrayMemory0 = null;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('graphrag-web-app_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
