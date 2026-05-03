//! API Models with Apistos OpenAPI support
//!
//! All request/response models with automatic OpenAPI schema generation

use actix_web::{error::ResponseError, http::StatusCode, HttpResponse};
use apistos::ApiComponent;
use apistos_gen::ApiErrorComponent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Query Models
// ============================================================================

/// Query mode — how the server interprets the question.
///
/// - `search` (default): pure vector similarity over Qdrant. Fast (~350ms),
///   returns ranked excerpts. No LLM call. Back-compatible default.
/// - `ask`: graph-aware retrieval + LLM-generated answer. Calls
///   `GraphRAG::ask`. Requires a configured chat backend.
/// - `explain`: like `ask`, but also returns confidence, source attribution
///   (text-chunk / entity / relationship), reasoning steps, and key entities.
///   Calls `GraphRAG::ask_explained`.
/// - `reason`: query decomposition for multi-hop questions; sub-queries are
///   answered and composed into a final answer. Calls
///   `GraphRAG::ask_with_reasoning`. Slower than `ask`.
/// - `local`: Microsoft GraphRAG-style `local_search` and equivalent of
///   LightRAG's `local` mode. Embeds the query, vector-searches the
///   entity sidecar (top-K seed entities), expands to 1-hop neighbors,
///   gathers their mentioning chunks, and feeds the assembled context
///   to the chat backend.
/// - `global`: LightRAG-paper `global` mode. Extracts dual-level
///   keywords from the query (one LLM call), embeds the **high-level**
///   set, vector-searches the *relationship* sidecar for top-K seed
///   relations, resolves their endpoint entities, expands neighborhoods,
///   gathers chunks, sends to chat backend. Themes-and-concepts shape;
///   answers thematic / cross-cutting questions better than `local`.
/// - `hybrid`: LightRAG-paper `hybrid` mode. Runs both retrieval streams
///   — low-level keywords → entity vector search → entity seeds; and
///   high-level keywords → relationship vector search → relation seeds.
///   Merges the result sets and feeds the union to the chat backend.
///   Most graph-aware retrieval; best default for entity-centric
///   questions where you also want thematic context.
/// - `mix`: LightRAG-paper `mix` mode. Hybrid plus a chunk-vector
///   search using the original query — the chunk results are added
///   directly as seed chunks alongside the entity/relation expansion.
///   Strongest recall, slightly slower (one extra Qdrant call).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    Search,
    Ask,
    Explain,
    Reason,
    Local,
    Global,
    Hybrid,
    Mix,
}

impl Default for QueryMode {
    fn default() -> Self {
        QueryMode::Search
    }
}

impl QueryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            QueryMode::Search => "search",
            QueryMode::Ask => "ask",
            QueryMode::Explain => "explain",
            QueryMode::Reason => "reason",
            QueryMode::Local => "local",
            QueryMode::Global => "global",
            QueryMode::Hybrid => "hybrid",
            QueryMode::Mix => "mix",
        }
    }
}

/// Query request
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct QueryRequest {
    /// The search query string
    #[schemars(example = "example_query")]
    pub query: String,

    /// Number of results to return
    #[serde(default = "default_top_k")]
    #[schemars(example = "example_top_k")]
    pub top_k: usize,

    /// Retrieval mode. Defaults to `search` for back-compat.
    /// See [QueryMode] for the full menu and their tradeoffs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<QueryMode>,
}

fn default_top_k() -> usize {
    5
}

fn example_query() -> &'static str {
    "What is GraphRAG?"
}

fn example_top_k() -> usize {
    5
}

/// Single query result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct QueryResult {
    /// Document identifier
    pub document_id: String,

    /// Document title
    pub title: String,

    /// Similarity score (0.0-1.0)
    #[schemars(example = "example_similarity")]
    pub similarity: f32,

    /// Text excerpt from the document
    pub excerpt: String,
}

fn example_similarity() -> f32 {
    0.85
}

/// Type of source reference returned in `explain` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    TextChunk,
    Entity,
    Relationship,
    Summary,
}

/// A source the answer relied on. Returned in `explain` mode alongside the
/// vector-search excerpts so callers can audit which chunks/entities/edges
/// supported the answer.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct SourceReferenceDto {
    /// Identifier of the source (chunk/entity/relationship id).
    pub id: String,
    /// What kind of source this is.
    pub kind: SourceKind,
    /// Excerpt or rendered summary of the source.
    pub excerpt: String,
    /// Relevance score to the query (0.0-1.0; higher = more relevant).
    pub relevance: f32,
}

/// One step in a reasoning trace. Returned in `explain` and `reason` modes.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningStepDto {
    /// 1-indexed step number.
    pub step: u8,
    /// Human-readable description of what the engine did at this step.
    pub description: String,
    /// Entity ids touched at this step.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities_used: Vec<String>,
    /// Snippet of evidence that backed this step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    /// Per-step confidence (0.0-1.0).
    pub confidence: f32,
}

/// Query response.
///
/// `results` is always populated (vector-search hits, ranked). Modes other
/// than `search` additionally populate `answer` (LLM-composed answer) and,
/// for `explain`, the full reasoning trace + source attribution.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct QueryResponse {
    /// Original query string
    pub query: String,

    /// Mode the server actually used (echoes the request, or `search` if
    /// the request omitted `mode`).
    pub mode: String,

    /// List of matching text-chunk results (vector search hits). Always
    /// populated.
    pub results: Vec<QueryResult>,

    /// LLM-composed answer. Populated for `ask`, `explain`, `reason` modes.
    /// `None` for `search` mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    /// Overall answer confidence (0.0-1.0). Populated for `explain` mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,

    /// Entities that were key to producing the answer. `explain` mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_entities: Option<Vec<String>>,

    /// Reasoning trace (one entry per retrieval/synthesis step). `explain` mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_steps: Option<Vec<ReasoningStepDto>>,

    /// Full source attribution (text chunks + entities + relationships
    /// the answer relied on). `explain` mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<SourceReferenceDto>>,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Backend used (`qdrant`, `memory`, or `graphrag` for graph-aware modes).
    pub backend: String,
}

// ============================================================================
// Document Models
// ============================================================================

/// Add document request
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct AddDocumentRequest {
    /// Optional caller-supplied id. Stored alongside the Qdrant point
    /// so callers can later delete by this same id (instead of having
    /// to remember the UUID the server assigned). When omitted, the
    /// server still returns a UUID.
    #[serde(default)]
    pub id: Option<String>,

    /// Document title
    #[schemars(example = "example_title")]
    pub title: String,

    /// Document content/text
    #[schemars(example = "example_content")]
    pub content: String,
}

fn example_title() -> &'static str {
    "Introduction to GraphRAG"
}

fn example_content() -> &'static str {
    "GraphRAG is a retrieval-augmented generation system that combines knowledge graphs with large language models..."
}

/// Document metadata (for listing)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    /// Document identifier
    pub id: String,

    /// Document title
    pub title: String,

    /// Full document content
    pub content: String,

    /// Timestamp when document was added (ISO 8601)
    pub added_at: String,
}

/// List documents response
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct ListDocumentsResponse {
    /// List of documents
    pub documents: Vec<DocumentSummary>,

    /// Total number of documents
    pub total: usize,

    /// Backend used
    pub backend: String,

    /// Optional note/message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Document summary (for lists)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSummary {
    /// Document identifier (Qdrant point UUID).
    pub id: String,

    /// Caller-supplied id, if any (the id passed to POST /api/documents).
    /// Useful for callers that want to re-issue delete/update by their
    /// own id without remembering the server-assigned UUID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,

    /// Document title
    pub title: String,

    /// Content length in characters (memory backend only — Qdrant
    /// returns an excerpt instead, see `excerpt`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_length: Option<usize>,

    /// First ~160 chars of content. Populated when listing from Qdrant;
    /// the memory backend leaves it None and uses content_length instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,

    /// Timestamp when added
    pub added_at: String,
}

// ============================================================================
// Graph Models
// ============================================================================

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct GraphStatsResponse {
    /// Number of documents
    pub document_count: usize,

    /// Number of entities
    pub entity_count: usize,

    /// Number of relationships
    pub relationship_count: usize,

    /// Number of vectors
    pub vector_count: usize,

    /// Whether graph has been built
    pub graph_built: bool,

    /// RFC 3339 timestamp of the last successful /api/graph/build, or
    /// null if the server hasn't built the graph since startup. Lets
    /// agents/cron decide whether the graph is fresh enough relative
    /// to recent ingests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_built_at: Option<String>,

    /// Backend used
    pub backend: String,
}

// ============================================================================
// Config Models
// ============================================================================

// Note: Config endpoints use direct JSON responses, no request models needed

// ============================================================================
// Health/Info Models
// ============================================================================

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    /// Service status
    pub status: String,

    /// Timestamp (ISO 8601)
    pub timestamp: String,

    /// Number of documents
    pub document_count: usize,

    /// Whether graph is built
    pub graph_built: bool,

    /// Total queries processed
    pub total_queries: usize,

    /// Backend in use (qdrant / memory)
    pub backend: String,

    /// Active embedding backend snapshot. Sourced from
    /// `state.config.embeddings` so it can never disagree with
    /// `GET /config` or `GET /embeddings/stats`.
    pub embeddings: HealthEmbeddings,
}

/// Embedding subsystem snapshot embedded in `/health` responses.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct HealthEmbeddings {
    pub backend: String,
    pub model: String,
    pub dimension: usize,
    pub endpoint: String,
    /// `true` if the configured upstream is reachable; `false` means
    /// the service is silently using the hash-based fallback.
    pub live: bool,
}

// ============================================================================
// Success Response Models
// ============================================================================

// Note: Using specific response types (DocumentOperationResponse, BuildGraphResponse, etc.)
// instead of generic success responses for better type safety and clearer API contracts

/// Document operation success
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct DocumentOperationResponse {
    /// Operation success flag
    pub success: bool,

    /// Document identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,

    /// Success/info message
    pub message: String,

    /// Backend used
    pub backend: String,
}

/// Graph build response
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct BuildGraphResponse {
    /// Operation success flag
    pub success: bool,

    /// Number of documents processed
    pub document_count: usize,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Success message
    pub message: String,

    /// Backend used
    pub backend: String,
}

// ============================================================================
// Authentication Models
// ============================================================================

#[cfg(feature = "auth")]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct LoginRequest {
    /// Username
    pub username: String,

    /// Password
    pub password: String,
}

#[cfg(feature = "auth")]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct LoginResponse {
    /// Success flag
    pub success: bool,

    /// JWT token
    pub token: String,

    /// User identifier
    pub user_id: String,

    /// User role
    pub role: String,

    /// Token expiration (hours)
    pub expires_in_hours: u32,

    /// Usage instructions
    pub usage: String,
}

#[cfg(feature = "auth")]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeyRequest {
    /// User identifier
    pub user_id: String,

    /// Optional role
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

// ============================================================================
// Error Models
// ============================================================================

/// API Error types with OpenAPI documentation
#[derive(Debug, Clone, Serialize, Deserialize, ApiErrorComponent)]
#[openapi_error(
    status(code = 400, description = "Bad Request - Invalid input or parameters"),
    status(
        code = 401,
        description = "Unauthorized - Authentication required or failed"
    ),
    status(code = 404, description = "Not Found - Resource does not exist"),
    status(
        code = 500,
        description = "Internal Server Error - Server encountered an error"
    )
)]
pub enum ApiError {
    /// Bad request error
    BadRequest(String),

    /// Unauthorized error
    Unauthorized(String),

    /// Not found error
    NotFound(String),

    /// Internal server error
    InternalError(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::BadRequest(msg) => write!(f, "Bad Request: {}", msg),
            ApiError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            ApiError::NotFound(msg) => write!(f, "Not Found: {}", msg),
            ApiError::InternalError(msg) => write!(f, "Internal Server Error: {}", msg),
        }
    }
}

impl ResponseError for ApiError {
    fn status_code(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_response(&self) -> HttpResponse {
        let status = self.status_code();
        let error_message = self.to_string();

        HttpResponse::build(status).json(serde_json::json!({
            "error": match self {
                ApiError::BadRequest(_) => "Bad Request",
                ApiError::Unauthorized(_) => "Unauthorized",
                ApiError::NotFound(_) => "Not Found",
                ApiError::InternalError(_) => "Internal Server Error",
            },
            "message": error_message,
            "status": status.as_u16(),
        }))
    }
}

// Helper to convert (StatusCode, String) errors to ApiError
impl From<(StatusCode, String)> for ApiError {
    fn from((status, message): (StatusCode, String)) -> Self {
        match status {
            StatusCode::BAD_REQUEST => ApiError::BadRequest(message),
            StatusCode::UNAUTHORIZED => ApiError::Unauthorized(message),
            StatusCode::NOT_FOUND => ApiError::NotFound(message),
            _ => ApiError::InternalError(message),
        }
    }
}
