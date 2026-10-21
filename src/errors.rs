//! Enterprise-grade error handling with structured error types and codes
//! Provides detailed error information for debugging and client error handling

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Structured error response for API clients
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Machine-readable error code
    pub code: String,

    /// Human-readable error message
    pub message: String,

    /// Additional error context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,

    /// Request ID for tracing (enterprise feature)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Sanitize an ID string for safe inclusion in error messages and logs.
///
/// Strips control characters and characters that could enable log injection.
/// Keeps alphanumeric characters, dashes, and underscores only.
/// Truncates to 64 characters to prevent log bloat.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect()
}

/// What a caller is told about a latched store. Reaches the client, because the
/// status is 503 rather than 500 — so it names the condition and no paths.
const WRITE_LATCHED_CLIENT_MESSAGE: &str =
    "Storage is not accepting writes (no space left on device); retry later";

const SHUTTING_DOWN_CLIENT_MESSAGE: &str =
    "The server is shutting down and wrote nothing for this request; retry once it is back";

/// Application error types with proper categorization
#[derive(Debug)]
pub enum AppError {
    // Validation Errors (400)
    InvalidInput {
        field: String,
        reason: String,
    },
    InvalidUserId(String),
    InvalidMemoryId(String),
    InvalidEmbeddings(String),
    ContentTooLarge {
        size: usize,
        max: usize,
    },

    // Resource Limit Errors (429)
    ResourceLimit {
        resource: String,
        current: usize,
        limit: usize,
    },

    // Ambiguity Errors (400)
    AmbiguousMemoryId {
        prefix: String,
        count: usize,
    },

    // Not Found Errors (404)
    MemoryNotFound(String),
    UserNotFound(String),
    TodoNotFound(String),
    ProjectNotFound(String),
    ContextBlockNotFound(String),

    // Conflict Errors (409)
    MemoryAlreadyExists(String),

    // Internal Errors (500)
    StorageError(String),
    DatabaseError(String),
    SerializationError(String),
    ConcurrencyError(String),

    // Lock failures (500) - non-panicking lock handling
    LockPoisoned {
        resource: String,
        details: String,
    },
    LockAcquisitionFailed {
        resource: String,
        reason: String,
    },

    // Service Errors (503)
    ServiceUnavailable(String),

    /// RocksDB has stopped accepting writes for want of disk space. Retryable
    /// from the caller's side, which `INTERNAL_ERROR` never said.
    StorageWriteLatched(anyhow::Error),

    // Generic wrapper for external errors
    Internal(anyhow::Error),
}

impl AppError {
    /// Create a lock poisoned error from a PoisonError
    pub fn from_lock_poison<T>(resource: &str, _err: std::sync::PoisonError<T>) -> Self {
        Self::LockPoisoned {
            resource: resource.to_string(),
            details: "Thread panicked while holding lock".to_string(),
        }
    }

    /// Create a lock acquisition failure
    pub fn lock_failed(resource: &str, reason: &str) -> Self {
        Self::LockAcquisitionFailed {
            resource: resource.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Whether this error is a RocksDB write latch, whichever variant carried
    /// it. 386 call sites build `Internal` directly rather than going through
    /// `From<anyhow::Error>`, so classifying only in the conversion would leave
    /// a latched upsert still answering `INTERNAL_ERROR`.
    fn is_write_latched(&self) -> bool {
        match self {
            Self::StorageWriteLatched(_) => true,
            Self::Internal(err) => crate::storage::write_latch::chain_is_enospc(err),
            _ => false,
        }
    }

    /// Whether a tenant's write gate refused this request at shutdown. Classified over
    /// the chain for the same reason as [`Self::is_write_latched`].
    fn is_shutting_down(&self) -> bool {
        match self {
            Self::Internal(err) => crate::memory::write_gate::chain_is_writes_closed(err),
            _ => false,
        }
    }

    /// Get error code for client identification
    pub fn code(&self) -> &'static str {
        if self.is_write_latched() {
            return "STORAGE_WRITE_LATCHED";
        }
        if self.is_shutting_down() {
            return "SHUTTING_DOWN";
        }
        match self {
            Self::InvalidInput { .. } => "INVALID_INPUT",
            Self::InvalidUserId(_) => "INVALID_USER_ID",
            Self::InvalidMemoryId(_) => "INVALID_MEMORY_ID",
            Self::InvalidEmbeddings(_) => "INVALID_EMBEDDINGS",
            Self::ContentTooLarge { .. } => "CONTENT_TOO_LARGE",
            Self::AmbiguousMemoryId { .. } => "AMBIGUOUS_MEMORY_ID",
            Self::ResourceLimit { .. } => "RESOURCE_LIMIT",
            Self::MemoryNotFound(_) => "MEMORY_NOT_FOUND",
            Self::UserNotFound(_) => "USER_NOT_FOUND",
            Self::TodoNotFound(_) => "TODO_NOT_FOUND",
            Self::ProjectNotFound(_) => "PROJECT_NOT_FOUND",
            Self::ContextBlockNotFound(_) => "CONTEXT_BLOCK_NOT_FOUND",
            Self::MemoryAlreadyExists(_) => "MEMORY_ALREADY_EXISTS",
            Self::StorageError(_) => "STORAGE_ERROR",
            Self::DatabaseError(_) => "DATABASE_ERROR",
            Self::SerializationError(_) => "SERIALIZATION_ERROR",
            Self::ConcurrencyError(_) => "CONCURRENCY_ERROR",
            Self::LockPoisoned { .. } => "LOCK_POISONED",
            Self::LockAcquisitionFailed { .. } => "LOCK_ACQUISITION_FAILED",
            Self::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            Self::StorageWriteLatched(_) => "STORAGE_WRITE_LATCHED",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }

    /// Get HTTP status code
    pub fn status_code(&self) -> StatusCode {
        if self.is_write_latched() || self.is_shutting_down() {
            return StatusCode::SERVICE_UNAVAILABLE;
        }
        match self {
            Self::InvalidInput { .. }
            | Self::InvalidUserId(_)
            | Self::InvalidMemoryId(_)
            | Self::InvalidEmbeddings(_)
            | Self::ContentTooLarge { .. }
            | Self::AmbiguousMemoryId { .. } => StatusCode::BAD_REQUEST,

            Self::ResourceLimit { .. } => StatusCode::TOO_MANY_REQUESTS,

            Self::MemoryNotFound(_)
            | Self::UserNotFound(_)
            | Self::TodoNotFound(_)
            | Self::ProjectNotFound(_)
            | Self::ContextBlockNotFound(_) => StatusCode::NOT_FOUND,

            Self::MemoryAlreadyExists(_) => StatusCode::CONFLICT,

            Self::ServiceUnavailable(_) | Self::StorageWriteLatched(_) => {
                StatusCode::SERVICE_UNAVAILABLE
            }

            Self::StorageError(_)
            | Self::DatabaseError(_)
            | Self::SerializationError(_)
            | Self::ConcurrencyError(_)
            | Self::LockPoisoned { .. }
            | Self::LockAcquisitionFailed { .. }
            | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Get detailed error message
    pub fn message(&self) -> String {
        if self.is_write_latched() {
            // The 503 body reaches the client, so the chain goes to the log and
            // the caller gets the one fact it can act on.
            let chain = match self {
                Self::StorageWriteLatched(err) | Self::Internal(err) => format!("{err:#}"),
                _ => String::new(),
            };
            tracing::error!(error = %chain, "Storage write latched");
            return WRITE_LATCHED_CLIENT_MESSAGE.to_string();
        }
        if self.is_shutting_down() {
            return SHUTTING_DOWN_CLIENT_MESSAGE.to_string();
        }
        match self {
            Self::InvalidInput { field, reason } => {
                format!("Invalid input for field '{field}': {reason}")
            }
            Self::InvalidUserId(msg) => format!("Invalid user ID: {msg}"),
            Self::InvalidMemoryId(msg) => format!("Invalid memory ID: {msg}"),
            Self::InvalidEmbeddings(msg) => format!("Invalid embeddings: {msg}"),
            Self::ContentTooLarge { size, max } => {
                format!("Content too large: {size} bytes (max: {max} bytes)")
            }
            Self::AmbiguousMemoryId { prefix, count } => {
                format!("Ambiguous memory ID prefix '{}': matches {count} memories. Use a longer prefix or full UUID.", sanitize_id(prefix))
            }
            Self::ResourceLimit {
                resource,
                current,
                limit,
            } => {
                format!("Resource limit exceeded for {resource}: current={current} MB, limit={limit} MB")
            }
            Self::MemoryNotFound(id) => format!("Memory not found: {}", sanitize_id(id)),
            Self::UserNotFound(id) => format!("User not found: {}", sanitize_id(id)),
            Self::TodoNotFound(id) => format!("Todo not found: {}", sanitize_id(id)),
            Self::ProjectNotFound(id) => format!("Project not found: {}", sanitize_id(id)),
            Self::ContextBlockNotFound(key) => {
                format!("Context block not found: {}", sanitize_id(key))
            }
            Self::MemoryAlreadyExists(id) => format!("Memory already exists: {}", sanitize_id(id)),
            Self::StorageError(msg) => {
                tracing::error!(error = %msg, "Storage error");
                "Internal server error".to_string()
            }
            Self::DatabaseError(msg) => {
                tracing::error!(error = %msg, "Database error");
                "Internal server error".to_string()
            }
            Self::SerializationError(msg) => {
                tracing::error!(error = %msg, "Serialization error");
                "Internal server error".to_string()
            }
            Self::ConcurrencyError(msg) => {
                tracing::error!(error = %msg, "Concurrency error");
                "Internal server error".to_string()
            }
            Self::LockPoisoned { resource, details } => {
                tracing::error!(resource = %resource, details = %details, "Lock poisoned");
                "Internal server error".to_string()
            }
            Self::LockAcquisitionFailed { resource, reason } => {
                tracing::error!(resource = %resource, reason = %reason, "Lock acquisition failed");
                "Internal server error".to_string()
            }
            Self::ServiceUnavailable(msg) => format!("Service unavailable: {msg}"),
            // Unreachable: `is_write_latched()` above answers for this variant.
            // The arm exists so the match stays exhaustive and so removing that
            // early return cannot silently change the answer.
            Self::StorageWriteLatched(_) => WRITE_LATCHED_CLIENT_MESSAGE.to_string(),
            Self::Internal(err) => {
                // `{:#}` prints the whole anyhow context chain; `%err` alone
                // prints only the outermost context, which is how the
                // 2026-07/08 tenant-reopen 500s stayed undiagnosable — every
                // incident logged the context string and swallowed the cause.
                tracing::error!(error = format!("{:#}", err), "Internal error");
                "Internal server error".to_string()
            }
        }
    }

    /// Convert to structured error response
    pub fn to_response(&self) -> ErrorResponse {
        ErrorResponse {
            code: self.code().to_string(),
            message: self.message(),
            details: None,
            request_id: None,
        }
    }

    /// Convert to structured error response with request ID
    pub fn to_response_with_request_id(&self, request_id: Option<String>) -> ErrorResponse {
        ErrorResponse {
            code: self.code().to_string(),
            message: self.message(),
            details: None,
            request_id,
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for AppError {}

/// Convert from anyhow::Error to AppError
impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        // Classified over the whole chain: the production message arrives inside
        // `Failed to put memory <uuid> in RocksDB: …`, so the outer string alone
        // sees only the wrapper.
        if crate::storage::write_latch::chain_is_enospc(&err) {
            return Self::StorageWriteLatched(err);
        }
        Self::Internal(err)
    }
}

/// Axum IntoResponse implementation for proper HTTP responses
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let code = self.code().to_string();
        let full_message = self.message();

        // For 500 errors: log full internal details server-side only.
        // Return a generic message to the client to avoid leaking filesystem
        // paths, RocksDB internals, or lock details in HTTP responses.
        let client_message = if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(
                error_code = %code,
                error_detail = %full_message,
                "Internal server error"
            );
            "Internal server error".to_string()
        } else {
            full_message
        };

        let body = ErrorResponse {
            code,
            message: client_message,
            details: None,
            request_id: None,
        };

        (status, Json(body)).into_response()
    }
}

/// Helper trait to convert validation errors
pub trait ValidationErrorExt<T> {
    fn map_validation_err(self, field: &str) -> Result<T>;
}

impl<T> ValidationErrorExt<T> for anyhow::Result<T> {
    fn map_validation_err(self, field: &str) -> Result<T> {
        self.map_err(|e| AppError::InvalidInput {
            field: field.to_string(),
            reason: e.to_string(),
        })
    }
}

/// Type alias for Results using AppError
pub type Result<T> = std::result::Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(
            AppError::InvalidUserId("test".to_string()).code(),
            "INVALID_USER_ID"
        );
        assert_eq!(
            AppError::MemoryNotFound("123".to_string()).code(),
            "MEMORY_NOT_FOUND"
        );
    }

    #[test]
    fn test_status_codes() {
        assert_eq!(
            AppError::InvalidUserId("test".to_string()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            AppError::MemoryNotFound("123".to_string()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            AppError::StorageError("failed".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn an_enospc_chain_becomes_a_503_write_latch_not_a_500() {
        let err = anyhow::anyhow!(
            "IO error: No space left on device: While open a file for appending: /store/000585.log"
        )
        .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB");

        let app_error = AppError::from(err);
        assert_eq!(app_error.code(), "STORAGE_WRITE_LATCHED");
        assert_eq!(
            app_error.status_code(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a latched store is retryable, which a 500 never said"
        );
    }

    /// The handlers build `AppError::Internal` directly far more often than they
    /// let `?` convert an `anyhow::Error`, so this is the path a real upsert
    /// against a latched store takes.
    #[test]
    fn a_directly_constructed_internal_also_answers_503_when_latched() {
        let err = AppError::Internal(
            anyhow::anyhow!("IO error: No space left on device: While open a file for appending: /store/000585.log")
                .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB"),
        );

        assert_eq!(err.code(), "STORAGE_WRITE_LATCHED");
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            !err.message().contains("000585.log"),
            "a 503 body reaches the client and must not carry store paths"
        );
    }

    /// The upsert handler wraps the refusal as `upsert_memory` returns it, inside
    /// `AppError::Internal`.
    #[test]
    fn a_closed_write_gate_answers_a_retryable_503() {
        let err = AppError::Internal(
            anyhow::Error::new(crate::memory::write_gate::WritesClosed)
                .context("upsert claude-memory/veld/x"),
        );

        assert_eq!(err.code(), "SHUTTING_DOWN");
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.message(), SHUTTING_DOWN_CLIENT_MESSAGE);
    }

    #[test]
    fn an_unrelated_chain_stays_an_internal_error() {
        let err = anyhow::anyhow!("Corruption: block checksum mismatch")
            .context("Failed to put memory 00000000-0000-0000-0000-000000000000 in RocksDB");

        let app_error = AppError::from(err);
        assert_eq!(app_error.code(), "INTERNAL_ERROR");
        assert_eq!(app_error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_error_response_serialization() {
        let err = AppError::InvalidUserId("test123".to_string());
        let response = err.to_response();

        assert_eq!(response.code, "INVALID_USER_ID");
        assert!(response.message.contains("test123"));
    }
}
