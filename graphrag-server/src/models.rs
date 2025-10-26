//! API Models with Apistos OpenAPI support
//!
//! All request/response models with automatic OpenAPI schema generation

use apistos::ApiComponent;
use apistos_gen::ApiErrorComponent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;
use actix_web::{error::ResponseError, HttpResponse, http::StatusCode};

// ============================================================================
// Query Models
// ============================================================================

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

/// Query response
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct QueryResponse {
    /// Original query string
    pub query: String,

    /// List of matching results
    pub results: Vec<QueryResult>,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Backend used ("qdrant" or "memory")
    pub backend: String,
}

// ============================================================================
// Document Models
// ============================================================================

/// Add document request
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, ApiComponent)]
#[serde(rename_all = "camelCase")]
pub struct AddDocumentRequest {
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
    /// Document identifier
    pub id: String,

    /// Document title
    pub title: String,

    /// Content length in characters
    pub content_length: usize,

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

    /// Backend in use
    pub backend: String,
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
    status(code = 401, description = "Unauthorized - Authentication required or failed"),
    status(code = 404, description = "Not Found - Resource does not exist"),
    status(code = 500, description = "Internal Server Error - Server encountered an error")
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
