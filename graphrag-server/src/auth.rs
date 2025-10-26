//! Authentication and Authorization Middleware
//!
//! Provides JWT token-based authentication and API key authentication for the REST API.
//!
//! ## Features
//!
//! - JWT token generation and validation
//! - API key authentication
//! - Role-based access control (RBAC)
//! - Rate limiting per user/IP
//! - Request audit logging
//!
//! ## Usage
//!
//! ```rust
//! // Add authentication middleware to your routes
//! Router::new()
//!     .route("/api/protected", get(protected_handler))
//!     .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
//! ```

use axum::{
    extract::{Extension, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Issued at (timestamp)
    pub iat: u64,
    /// Expiration time (timestamp)
    pub exp: u64,
    /// User role
    pub role: UserRole,
    /// Custom claims
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// User roles for RBAC
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    /// Administrator with full access
    Admin,
    /// Regular user with read/write access
    User,
    /// Read-only user
    Readonly,
    /// Guest with limited access
    Guest,
}

/// API key structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key: String,
    pub user_id: String,
    pub role: UserRole,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub rate_limit: RateLimit,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum requests per window
    pub max_requests: usize,
    /// Window duration in seconds
    pub window_seconds: u64,
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            window_seconds: 3600, // 1 hour
        }
    }
}

/// Authentication state
#[derive(Clone)]
pub struct AuthState {
    /// JWT secret key
    jwt_secret: String,
    /// API keys storage
    api_keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    /// Rate limiting state: (user_id, (count, window_start))
    rate_limits: Arc<RwLock<HashMap<String, (usize, u64)>>>,
}

impl AuthState {
    /// Create a new authentication state
    ///
    /// # Arguments
    /// * `jwt_secret` - Secret key for JWT signing (should be 32+ characters)
    pub fn new(jwt_secret: String) -> Self {
        Self {
            jwt_secret,
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate a JWT token
    ///
    /// # Arguments
    /// * `user_id` - User identifier
    /// * `role` - User role
    /// * `duration_hours` - Token validity duration in hours
    pub fn generate_token(
        &self,
        user_id: &str,
        role: UserRole,
        duration_hours: u64,
    ) -> Result<String, AuthError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let claims = Claims {
            sub: user_id.to_string(),
            iat: now,
            exp: now + (duration_hours * 3600),
            role,
            custom: HashMap::new(),
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_bytes()),
        )
        .map_err(|e| AuthError::TokenGenerationFailed(e.to_string()))
    }

    /// Validate a JWT token
    pub fn validate_token(&self, token: &str) -> Result<Claims, AuthError> {
        decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &Validation::default(),
        )
        .map(|data| data.claims)
        .map_err(|e| AuthError::InvalidToken(e.to_string()))
    }

    /// Create an API key
    pub async fn create_api_key(
        &self,
        user_id: &str,
        role: UserRole,
        rate_limit: Option<RateLimit>,
    ) -> Result<String, AuthError> {
        let key = format!("grag_{}", uuid::Uuid::new_v4());

        let api_key = ApiKey {
            key: key.clone(),
            user_id: user_id.to_string(),
            role,
            created_at: chrono::Utc::now().to_rfc3339(),
            expires_at: None,
            rate_limit: rate_limit.unwrap_or_default(),
        };

        self.api_keys.write().await.insert(key.clone(), api_key);

        Ok(key)
    }

    /// Validate an API key
    pub async fn validate_api_key(&self, key: &str) -> Result<ApiKey, AuthError> {
        let keys = self.api_keys.read().await;
        keys.get(key)
            .cloned()
            .ok_or(AuthError::InvalidApiKey)
    }

    /// Revoke an API key
    #[allow(dead_code)]
    pub async fn revoke_api_key(&self, key: &str) -> Result<(), AuthError> {
        let mut keys = self.api_keys.write().await;
        keys.remove(key).ok_or(AuthError::InvalidApiKey)?;
        Ok(())
    }

    /// Check rate limit for a user
    pub async fn check_rate_limit(&self, user_id: &str, limit: &RateLimit) -> Result<(), AuthError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut rate_limits = self.rate_limits.write().await;

        let (count, window_start) = rate_limits
            .entry(user_id.to_string())
            .or_insert((0, now));

        // Reset if window expired
        if now - *window_start >= limit.window_seconds {
            *count = 0;
            *window_start = now;
        }

        // Check limit
        if *count >= limit.max_requests {
            return Err(AuthError::RateLimitExceeded {
                max: limit.max_requests,
                window: limit.window_seconds,
            });
        }

        // Increment count
        *count += 1;

        Ok(())
    }
}

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Missing authorization header")]
    MissingAuthHeader,

    #[error("Invalid authorization format")]
    InvalidAuthFormat,

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Token generation failed: {0}")]
    TokenGenerationFailed(String),

    #[error("Insufficient permissions")]
    #[allow(dead_code)]
    InsufficientPermissions,

    #[error("Rate limit exceeded: {max} requests per {window} seconds")]
    RateLimitExceeded { max: usize, window: u64 },
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingAuthHeader => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::InvalidAuthFormat => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::InvalidToken(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::InvalidApiKey => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::TokenGenerationFailed(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            AuthError::InsufficientPermissions => (StatusCode::FORBIDDEN, self.to_string()),
            AuthError::RateLimitExceeded { .. } => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
        };

        (status, message).into_response()
    }
}

/// Authenticated user information extracted from request
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AuthUser {
    pub user_id: String,
    pub role: UserRole,
}

/// Extract authenticated user from request headers
pub async fn extract_auth_user(
    auth_state: &AuthState,
    headers: &HeaderMap,
) -> Result<AuthUser, AuthError> {
    let auth_header = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::MissingAuthHeader)?;

    // Check for Bearer token (JWT)
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        let claims = auth_state.validate_token(token)?;
        return Ok(AuthUser {
            user_id: claims.sub,
            role: claims.role,
        });
    }

    // Check for API key
    if let Some(key) = auth_header.strip_prefix("ApiKey ") {
        let api_key = auth_state.validate_api_key(key).await?;

        // Check rate limit
        auth_state.check_rate_limit(&api_key.user_id, &api_key.rate_limit).await?;

        return Ok(AuthUser {
            user_id: api_key.user_id,
            role: api_key.role,
        });
    }

    Err(AuthError::InvalidAuthFormat)
}

/// Authentication middleware for Axum
///
/// Extracts and validates authentication from request headers.
/// Supports both JWT tokens and API keys.
pub async fn auth_middleware(
    State(auth_state): State<Arc<AuthState>>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let user = extract_auth_user(&auth_state, &headers).await?;

    // Store user in request extensions
    request.extensions_mut().insert(user);

    Ok(next.run(request).await)
}

/// Require minimum role for a route
///
/// Use this middleware after auth_middleware to enforce role requirements.
#[allow(dead_code)]
pub async fn require_role(
    minimum_role: UserRole,
) -> impl Fn(axum::extract::Extension<AuthUser>, Request, Next) -> futures::future::BoxFuture<'static, Result<Response, AuthError>> {
    move |Extension(user): axum::extract::Extension<AuthUser>, request: Request, next: Next| {
        let minimum_role = minimum_role.clone();
        Box::pin(async move {
            // Check role hierarchy: Admin > User > Readonly > Guest
            let has_permission = match (&user.role, &minimum_role) {
                (UserRole::Admin, _) => true,
                (UserRole::User, UserRole::User) => true,
                (UserRole::User, UserRole::Readonly) => true,
                (UserRole::User, UserRole::Guest) => true,
                (UserRole::Readonly, UserRole::Readonly) => true,
                (UserRole::Readonly, UserRole::Guest) => true,
                (UserRole::Guest, UserRole::Guest) => true,
                _ => false,
            };

            if !has_permission {
                return Err(AuthError::InsufficientPermissions);
            }

            Ok(next.run(request).await)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jwt_token() {
        let auth_state = AuthState::new("test_secret_key_32_characters_long".to_string());

        let token = auth_state.generate_token("user123", UserRole::User, 24).unwrap();
        let claims = auth_state.validate_token(&token).unwrap();

        assert_eq!(claims.sub, "user123");
        assert_eq!(claims.role, UserRole::User);
    }

    #[tokio::test]
    async fn test_api_key() {
        let auth_state = AuthState::new("test_secret".to_string());

        let key = auth_state.create_api_key("user123", UserRole::User, None).await.unwrap();
        let api_key = auth_state.validate_api_key(&key).await.unwrap();

        assert_eq!(api_key.user_id, "user123");
        assert_eq!(api_key.role, UserRole::User);
    }

    #[tokio::test]
    async fn test_rate_limit() {
        let auth_state = AuthState::new("test_secret".to_string());

        let limit = RateLimit {
            max_requests: 2,
            window_seconds: 60,
        };

        // First two requests should succeed
        auth_state.check_rate_limit("user123", &limit).await.unwrap();
        auth_state.check_rate_limit("user123", &limit).await.unwrap();

        // Third should fail
        let result = auth_state.check_rate_limit("user123", &limit).await;
        assert!(matches!(result, Err(AuthError::RateLimitExceeded { .. })));
    }
}
