//! Shodh Cortex - Transparent Memory Layer for Claude
//!
//! Cortex is the "sunglasses" - a transparent proxy that gives the brain
//! full visibility into everything Claude sees, hears, and says.
//!
//! Like the human cortex, it:
//! - Perceives all sensory input (requests)
//! - Activates relevant memories (proactive context)
//! - Encodes new experiences (interactions)
//! - Reinforces learning (feedback loop)
//!
//! Usage:
//!   ANTHROPIC_BASE_URL=http://127.0.0.1:3031 claude
//!
//! Environment:
//!   CORTEX_PORT          - Port to listen on (default: 3031)
//!   SHODH_API_URL        - Shodh Memory API URL (default: http://127.0.0.1:3030)
//!   SHODH_API_KEY        - Shodh Memory API key

mod activation;
mod encoding;
mod feedback;
mod injection;
mod perception;
mod types;

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, Method, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use futures::stream::StreamExt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info, warn};

use types::{ClaudeRequest, CortexConfig, Session, SESSION_TTL_SECS};

// ============================================================================
// State
// ============================================================================

/// Session store - DashMap is already thread-safe, no RwLock needed
pub type Sessions = Arc<dashmap::DashMap<String, Session>>;

/// Circuit breaker for brain availability
pub struct CircuitBreaker {
    /// Whether brain is available
    available: AtomicBool,
    /// Consecutive failure count
    failures: AtomicU64,
    /// Last failure timestamp (unix secs)
    last_failure: AtomicU64,
}

impl CircuitBreaker {
    const MAX_FAILURES: u64 = 3;
    const RESET_AFTER_SECS: u64 = 30;

    pub fn new() -> Self {
        Self {
            available: AtomicBool::new(true),
            failures: AtomicU64::new(0),
            last_failure: AtomicU64::new(0),
        }
    }

    /// Check if brain is available
    pub fn is_available(&self) -> bool {
        // Check if we should reset after timeout
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last = self.last_failure.load(Ordering::Relaxed);

        if now - last > Self::RESET_AFTER_SECS {
            self.available.store(true, Ordering::Relaxed);
            self.failures.store(0, Ordering::Relaxed);
        }

        self.available.load(Ordering::Relaxed)
    }

    /// Record a failure
    pub fn record_failure(&self) {
        let count = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_failure.store(now, Ordering::Relaxed);

        if count >= Self::MAX_FAILURES {
            warn!("Brain circuit breaker OPEN after {} failures", count);
            self.available.store(false, Ordering::Relaxed);
        }
    }

    /// Record a success
    pub fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
        self.available.store(true, Ordering::Relaxed);
    }
}

/// Cortex application state
pub struct CortexState {
    /// Per-user sessions for tracking interactions
    pub sessions: Sessions,

    /// Optional fallback Anthropic API key
    pub anthropic_api_key: Option<String>,

    /// Configuration
    pub config: CortexConfig,

    /// HTTP client for all external calls (reused)
    pub http_client: reqwest::Client,

    /// Circuit breaker for brain
    pub brain_circuit: CircuitBreaker,
}

impl CortexState {
    pub fn new(config: CortexConfig) -> Self {
        // Build HTTP client with timeouts
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.brain_timeout_secs))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            sessions: Arc::new(dashmap::DashMap::new()),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            config,
            http_client,
            brain_circuit: CircuitBreaker::new(),
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

async fn health() -> &'static str {
    "OK"
}

/// Main proxy handler - the core of Cortex
///
/// This is where the magic happens:
/// 1. PERCEIVE - Extract full context from request
/// 2. FEEDBACK - Process feedback from previous interaction (background)
/// 3. ACTIVATE - Get relevant memories from brain
/// 4. INJECT - Add memories to system prompt
/// 5. FORWARD - Send to Anthropic (streaming passthrough)
/// 6. ENCODE - Store interaction to brain (background, skip on errors)
async fn proxy_messages(
    State(state): State<Arc<CortexState>>,
    headers: HeaderMap,
    body: String,
) -> Result<Response, (StatusCode, String)> {
    // Parse request
    let mut request: ClaudeRequest = serde_json::from_str(&body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid request: {e}")))?;

    let is_streaming = request.stream;

    // =========================================================================
    // STEP 1: PERCEIVE - Extract full context
    // =========================================================================
    let full_context = perception::extract_full_context(&request, &headers);
    let user_id = full_context.user_id.clone();

    debug!(
        user_id = %user_id,
        messages = full_context.messages.len(),
        tools = full_context.tool_uses.len(),
        "Perceived context"
    );

    // Get or create session (touch to update last_accessed)
    let session = {
        let mut entry = state.sessions.entry(user_id.clone()).or_default();
        entry.touch();
        entry.clone()
    };

    // Check circuit breaker
    let brain_available = state.brain_circuit.is_available();

    // =========================================================================
    // STEP 2: FEEDBACK - Process feedback (BACKGROUND - don't block request)
    // =========================================================================
    if brain_available {
        if let Some(user_msg) = full_context.last_user_message() {
            let http_client = state.http_client.clone();
            let brain_url = state.config.brain_url.clone();
            let brain_api_key = state.config.brain_api_key.clone();
            let user_id_clone = user_id.clone();
            let user_msg = user_msg.to_string();
            let session_clone = session.clone();

            // Spawn feedback processing in background
            tokio::spawn(async move {
                feedback::process_feedback(
                    &http_client,
                    &brain_url,
                    &brain_api_key,
                    &user_id_clone,
                    &user_msg,
                    &session_clone,
                )
                .await;
            });
        }
    }

    // =========================================================================
    // STEP 3: ACTIVATE - Get relevant memories from brain
    // =========================================================================
    let activation_result = if brain_available {
        let result = activation::activate_memories(
            &state.http_client,
            &state.config.brain_url,
            &state.config.brain_api_key,
            &full_context,
            &session,
            state.config.max_memories,
        )
        .await;

        // Update circuit breaker
        if result.memories.is_empty() && result.brain_error {
            state.brain_circuit.record_failure();
        } else {
            state.brain_circuit.record_success();
        }

        result
    } else {
        debug!("Brain circuit breaker open, skipping activation");
        activation::ActivationResult::empty()
    };

    // Log activation
    if !activation_result.memories.is_empty() {
        info!(
            "Activated {} memories for {}",
            activation_result.memories.len(),
            user_id
        );
    }

    // =========================================================================
    // STEP 4: INJECT - Add memories to system prompt
    // =========================================================================
    injection::inject_memories(&mut request, &activation_result.memories);

    // =========================================================================
    // STEP 5: FORWARD - True transparent proxy to Anthropic
    // =========================================================================

    // Build request body first so we know the new content length
    let request_body = serde_json::to_string(&request).unwrap();

    // Build upstream request - use configurable URL
    let upstream_url = format!("{}/v1/messages", state.config.upstream_url);
    let upstream_client = reqwest::Client::new();
    let mut upstream_request = upstream_client.post(&upstream_url);

    // Forward ALL headers except hop-by-hop headers and content-length (body changed)
    let skip_headers = [
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
    ];

    for (name, value) in headers.iter() {
        let name_str = name.as_str().to_lowercase();
        if !skip_headers.contains(&name_str.as_str()) {
            if let Ok(val) = value.to_str() {
                upstream_request = upstream_request.header(name.as_str(), val);
            }
        }
    }

    // If no auth was provided in headers and we have a fallback key, add it
    let has_auth = headers.get("authorization").is_some()
        || headers.get("x-api-key").is_some();
    if !has_auth {
        if let Some(key) = &state.anthropic_api_key {
            upstream_request = upstream_request.header("x-api-key", key);
        }
    }

    let upstream_response = upstream_request
        .body(request_body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Upstream LLM error: {e}")))?;

    let status = upstream_response.status();
    let response_headers = upstream_response.headers().clone();

    // Check if this is an error response - don't encode errors
    let is_error_response = !status.is_success();

    if is_streaming && !is_error_response {
        // =====================================================================
        // TRUE STREAMING - Forward chunks in real-time
        // =====================================================================
        let content_type = response_headers
            .get(header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| header::HeaderValue::from_static("text/event-stream"));

        // Create channel for collecting response data
        let (collector_tx, mut collector_rx) = mpsc::channel::<Bytes>(100);

        // Clone what we need for the collector task
        let sessions = state.sessions.clone();
        let user_id_for_encoding = user_id.clone();
        let http_client = state.http_client.clone();
        let brain_url = state.config.brain_url.clone();
        let brain_api_key = state.config.brain_api_key.clone();
        let context_for_encoding = full_context.clone();
        let memory_ids = activation_result.memory_ids.clone();
        let brain_available_for_encoding = brain_available;

        // Spawn collector task that assembles response for encoding
        tokio::spawn(async move {
            let mut collected = Vec::new();

            while let Some(chunk) = collector_rx.recv().await {
                collected.extend_from_slice(&chunk);
            }

            // Parse collected stream
            let (response_text, tool_uses) = extract_from_stream(&collected);

            // Encode to brain (background, non-blocking)
            if brain_available_for_encoding && !response_text.is_empty() {
                encoding::encode_interaction(
                    &http_client,
                    &brain_url,
                    &brain_api_key,
                    &context_for_encoding,
                    &response_text,
                    &tool_uses,
                )
                .await;
            }

            // Update session
            if let Some(mut session) = sessions.get_mut(&user_id_for_encoding) {
                session.last_response = Some(response_text);
                session.last_memory_ids = memory_ids;
                session.last_tool_uses = tool_uses;
                session.last_user_message = context_for_encoding
                    .last_user_message()
                    .map(|s| s.to_string());
                session.interaction_count += 1;
                session.touch();
            }
        });

        // Create streaming response
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(100);

        // Spawn forwarder task
        let mut stream = upstream_response.bytes_stream();
        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        // Send to collector for encoding
                        let _ = collector_tx.send(chunk.clone()).await;
                        // Forward to client
                        if tx.send(Ok(chunk)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Stream error: {}", e);
                        break;
                    }
                }
            }
            // Collector channel closes when collector_tx drops
        });

        let body = Body::from_stream(ReceiverStream::new(rx));

        Ok(Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .body(body)
            .unwrap())
    } else {
        // =====================================================================
        // NON-STREAMING or ERROR - Buffer response
        // =====================================================================
        let body_text = upstream_response
            .text()
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Read error: {e}")))?;

        // Only encode successful responses
        if !is_error_response && brain_available {
            let (response_text, tool_uses) = extract_from_response(&body_text);

            // Encode in background
            let encode_context = full_context.clone();
            let encode_client = state.http_client.clone();
            let encode_url = state.config.brain_url.clone();
            let encode_key = state.config.brain_api_key.clone();

            let memory_ids = activation_result.memory_ids.clone();
            let sessions = state.sessions.clone();
            let user_id_clone = user_id.clone();

            tokio::spawn(async move {
                encoding::encode_interaction(
                    &encode_client,
                    &encode_url,
                    &encode_key,
                    &encode_context,
                    &response_text,
                    &tool_uses,
                )
                .await;

                // Update session
                if let Some(mut session) = sessions.get_mut(&user_id_clone) {
                    session.last_response = Some(response_text);
                    session.last_memory_ids = memory_ids;
                    session.last_tool_uses = tool_uses;
                    session.last_user_message =
                        encode_context.last_user_message().map(|s| s.to_string());
                    session.interaction_count += 1;
                    session.touch();
                }
            });
        }

        let mut response = Response::builder().status(status);
        if let Some(ct) = response_headers.get(header::CONTENT_TYPE) {
            response = response.header(header::CONTENT_TYPE, ct);
        }

        Ok(response.body(Body::from(body_text)).unwrap())
    }
}

/// Extract text and tool uses from SSE stream
fn extract_from_stream(bytes: &[u8]) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut tools = Vec::new();

    if let Ok(stream_text) = std::str::from_utf8(bytes) {
        for line in stream_text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                    // Extract text deltas
                    if event.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                        if let Some(delta_text) = event
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            text.push_str(delta_text);
                        }
                    }

                    // Extract tool uses
                    if event.get("type").and_then(|t| t.as_str()) == Some("content_block_start") {
                        if let Some(name) = event
                            .get("content_block")
                            .and_then(|b| b.get("name"))
                            .and_then(|n| n.as_str())
                        {
                            tools.push(name.to_string());
                        }
                    }
                }
            }
        }
    }

    (text, tools)
}

/// Extract text and tool uses from non-streaming response
fn extract_from_response(body: &str) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut tools = Vec::new();

    if let Ok(resp) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(content) = resp.get("content").and_then(|c| c.as_array()) {
            for block in content {
                // Extract text
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }

                // Extract tool uses
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    if let Some(name) = block.get("name").and_then(|n| n.as_str()) {
                        tools.push(name.to_string());
                    }
                }
            }
        }
    }

    (text, tools)
}

/// Proxy models endpoint - true transparent proxy
async fn proxy_models(
    State(state): State<Arc<CortexState>>,
    headers: HeaderMap,
) -> Result<Response, (StatusCode, String)> {
    let skip_headers = [
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
    ];

    let models_url = format!("{}/v1/models", state.config.upstream_url);
    let mut request = state.http_client.get(&models_url);

    // Forward ALL headers except hop-by-hop headers
    for (name, value) in headers.iter() {
        let name_str = name.as_str().to_lowercase();
        if !skip_headers.contains(&name_str.as_str()) {
            if let Ok(val) = value.to_str() {
                request = request.header(name.as_str(), val);
            }
        }
    }

    // Fallback to configured API key if no auth in headers
    let has_auth = headers.get("authorization").is_some()
        || headers.get("x-api-key").is_some();
    if !has_auth {
        if let Some(key) = &state.anthropic_api_key {
            request = request.header("x-api-key", key);
        }
    }

    let response = request
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("API error: {e}")))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Read error: {e}")))?;

    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap())
}

/// Session cleanup task - runs periodically to evict expired sessions
async fn session_cleanup_task(sessions: Sessions) {
    let mut interval = tokio::time::interval(Duration::from_secs(SESSION_TTL_SECS / 4));

    loop {
        interval.tick().await;

        let before = sessions.len();
        sessions.retain(|_, session| !session.is_expired());
        let after = sessions.len();

        if before != after {
            info!("Session cleanup: evicted {} expired sessions", before - after);
        }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,cortex=debug".into()),
        )
        .init();

    // Load config
    let config = CortexConfig::from_env();
    let port = config.port;
    let brain_url = config.brain_url.clone();
    let upstream_url = config.upstream_url.clone();
    let upstream_format = config.upstream_format.clone();

    // Create state
    let state = Arc::new(CortexState::new(config));

    // Start session cleanup background task
    let sessions_for_cleanup = state.sessions.clone();
    tokio::spawn(session_cleanup_task(sessions_for_cleanup));

    // CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    // Routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/messages", post(proxy_messages))
        .route("/v1/models", get(proxy_models))
        .layer(cors)
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    info!("Cortex starting on {}", addr);
    info!("  Brain: {}", brain_url);
    info!("  Upstream: {} ({:?})", upstream_url, upstream_format);
    info!("  Set ANTHROPIC_BASE_URL=http://127.0.0.1:{} to use", port);
    info!("");
    info!("  Cortex is your transparent memory layer.");
    info!("  Works with Claude, GPT, Mistral, Ollama, and any LLM.");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
