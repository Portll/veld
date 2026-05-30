//! Server bootstrap module — starts the Veld HTTP API server.
//!
//! Extracted from `main.rs` so that both `veld` (standalone)
//! and `veld server` (unified CLI) can start the server with identical behavior.

use anyhow::{Context, Result};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::timeout::TimeoutLayer;
use tracing::{error, info, warn};

use crate::{
    auth,
    config::{ServerConfig, StorageBackend},
    embeddings::minilm::pre_init_ort_runtime,
    rate_limit_governance::{RateLimitGovernanceLayer, ResetHandle},
    roots::{AppState, RootsRuntime},
    metrics, middleware,
};

#[cfg(feature = "telemetry")]
use crate::tracing_setup;

use crate::constants::{
    DATABASE_FLUSH_TIMEOUT_SECS, GRACEFUL_SHUTDOWN_TIMEOUT_SECS, VECTOR_INDEX_SAVE_TIMEOUT_SECS,
};

// Timeout for draining in-flight requests (server-specific, not in constants.rs)
const SERVER_DRAIN_TIMEOUT_SECS: u64 = 5;

// =============================================================================
// PUBLIC API
// =============================================================================

/// Configuration for starting the server via [`run`].
pub struct ServerRunConfig {
    pub host: String,
    pub port: u16,
    pub storage_path: PathBuf,
    pub storage_backend: StorageBackend,
    pub production: bool,
    pub rate_limit: u64,
    pub max_concurrent: usize,
}

/// Start the veld HTTP server.
///
/// This is a **blocking** call that runs until a shutdown signal (Ctrl-C / SIGTERM).
/// It sets environment variables, pre-initialises the ONNX runtime, builds a tokio
/// runtime, and then enters the async server loop.
///
/// # Safety
/// Environment variables are set **before** the tokio runtime is created, so no
/// threads exist yet. This avoids the `set_var` unsoundness on multi-threaded runtimes.
pub fn run(config: ServerRunConfig) -> Result<()> {
    let local_dev_rate_limit_default =
        !config.production
            && crate::config::is_local_bind_host(&config.host)
            && std::env::var("VELD_RATE_LIMIT").is_err()
            && config.rate_limit == 4000;
    let effective_rate_limit = if local_dev_rate_limit_default {
        0
    } else {
        config.rate_limit
    };

    // SAFETY: These set_var calls run before any threads are spawned — the tokio
    // runtime is not yet built, and pre_init_ort_runtime (below) is also single-threaded.
    // `std::env::set_var` is marked unsafe starting in Rust 2024 edition because it is
    // unsound to call concurrently with `std::env::var` in other threads. Here, this
    // process is single-threaded, so the invariant holds.
    unsafe {
        std::env::set_var("VELD_HOST", &config.host);
        std::env::set_var("VELD_PORT", config.port.to_string());
        std::env::set_var(
            "VELD_MEMORY_PATH",
            config.storage_path.to_string_lossy().to_string(),
        );
        std::env::set_var("VELD_STORAGE_BACKEND", config.storage_backend.as_str());
        if config.production {
            std::env::set_var("VELD_ENV", "production");
        }
        std::env::set_var("VELD_RATE_LIMIT", effective_rate_limit.to_string());
        std::env::set_var("VELD_MAX_CONCURRENT", config.max_concurrent.to_string());
    }

    // Pre-initialize ORT_DYLIB_PATH before any threads are spawned.
    pre_init_ort_runtime(false);

    // SAFETY: Still single-threaded — setting default log level before runtime construction.
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "veld=info,tower_http=warn");
        }
    }

    // Load .env file if present (won't override CLI-set vars)
    let _ = dotenvy::dotenv();

    // Build and enter the tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime")
        .block_on(async_main())
}

/// Build the optional relational backend selected by
/// `VELD_RELATIONAL_BACKEND` (see [`crate::config::RelationalBackendChoice`]).
///
/// Returns `Ok(None)` when unset — the default rusqlite-only path. When set,
/// connects (and, for SQLite, creates) the database and returns it
/// type-erased behind `Arc<dyn RelationalStore<Error = BoxError>>`; the
/// caller runs schema setup and wires it onto the manager. A misconfigured
/// backend (missing env vars, unknown value, Postgres requested without the
/// `postgres` feature) is a hard startup error rather than a silent fallback.
async fn build_relational_backend() -> Result<
    Option<
        Arc<
            dyn crate::storage::relational::RelationalStore<
                Error = crate::storage::relational::BoxError,
            >,
        >,
    >,
> {
    use crate::config::RelationalBackendChoice;
    use crate::storage::relational::{
        BoxError, ErasedRelationalStore, RelationalStore, SqliteRelationalStore,
    };

    let choice = match RelationalBackendChoice::from_env() {
        Ok(c) => c,
        Err(msg) => {
            error!("relational backend misconfigured: {msg}");
            anyhow::bail!("relational backend misconfigured: {msg}");
        }
    };

    let store: Option<Arc<dyn RelationalStore<Error = BoxError>>> = match choice {
        None => None,
        Some(RelationalBackendChoice::Sqlite { path }) => {
            let s = SqliteRelationalStore::open(&path)
                .await
                .with_context(|| format!("open relational sqlite at {path}"))?;
            info!(backend = "sqlite", path = %path, "relational backend connected");
            Some(Arc::new(ErasedRelationalStore::new(s)))
        }
        #[cfg(feature = "postgres")]
        Some(RelationalBackendChoice::Postgres { url }) => {
            let s = crate::storage::relational::PostgresRelationalStore::connect(&url)
                .await
                .context("connect relational postgres")?;
            info!(backend = "postgres", "relational backend connected");
            Some(Arc::new(ErasedRelationalStore::new(s)))
        }
        #[cfg(feature = "postgres")]
        Some(RelationalBackendChoice::Supabase {
            project_ref,
            db_password,
        }) => {
            let s = crate::storage::relational::SupabaseRelationalStore::connect(
                &project_ref,
                &db_password,
            )
            .await
            .context("connect relational supabase")?;
            info!(backend = "supabase", project_ref = %project_ref, "relational backend connected");
            Some(Arc::new(ErasedRelationalStore::new(s)))
        }
    };
    Ok(store)
}

// =============================================================================
// ASYNC MAIN
// =============================================================================

async fn async_main() -> Result<()> {
    // Initialize tracing
    #[cfg(feature = "telemetry")]
    {
        tracing_setup::init_tracing().expect("Failed to initialize tracing");
    }
    #[cfg(not(feature = "telemetry"))]
    {
        tracing_subscriber::fmt::init();
    }

    // Print startup banner
    print_banner();

    // Log security/authentication status
    auth::log_security_status();

    // Register Prometheus metrics
    metrics::register_metrics().expect("Failed to register metrics");

    // Load configuration
    let server_config = ServerConfig::from_env();
    if server_config.requested_storage_backend != server_config.effective_storage_backend {
        warn!(
            requested = %server_config.requested_storage_backend,
            effective = %server_config.effective_storage_backend,
            "Requested storage backend is not active yet; running compatibility backend"
        );
    }
    log_production_security_warnings(&server_config);
    print_config(&server_config);

    // Initialize runtime-configurable decay scales from config
    crate::decay::init_runtime_scales(&server_config.log_periodic_scales);

    // Create orchestration runtime. Optionally wire a relational backend
    // (W4 cutover + W7 datasets + W6 query planner) when
    // VELD_RELATIONAL_BACKEND is set; otherwise the manager runs
    // rusqlite-only, exactly as before.
    let relational = build_relational_backend().await?;
    let mut manager_inner = crate::handlers::MultiUserMemoryManager::new(
        server_config.storage_path.clone(),
        server_config.clone(),
    )?;
    if let Some(store) = relational {
        // Ensure the slow-store `memories` schema exists in the backend so
        // the projection's first write doesn't hit a missing table.
        crate::memory::slow_store::RelationalSlowStoreAdapter::new(store.clone())
            .init_schema()
            .await
            .context("initialise relational slow-store schema")?;
        let dataset_store = Arc::new(
            crate::datasets::RelationalDatasetStore::new(store.clone())
                .await
                .context("initialise relational dataset catalog")?,
        );
        let link_store = Arc::new(
            crate::datasets::RelationalLinkStore::new(store.clone())
                .await
                .context("initialise relational link table")?,
        );
        manager_inner = manager_inner.with_dataset_stores(dataset_store, link_store, store);
        info!(
            "relational backend wired: memories projection cutover + datasets + query planner active"
        );
    }
    let runtime = RootsRuntime::from_manager(Arc::new(manager_inner));
    let manager: AppState = runtime.state();

    // Print storage stats
    print_storage_stats(&server_config.storage_path);

    // Keep reference for shutdown cleanup
    let manager_for_shutdown = Arc::clone(&manager);

    // Start background maintenance scheduler
    start_maintenance_scheduler(
        Arc::clone(&manager),
        server_config.maintenance_interval_secs,
    );

    // Start active reminder scheduler (checks every 60s for due reminders)
    start_reminder_scheduler(Arc::clone(&manager));

    // Start backup scheduler if enabled
    if server_config.backup_enabled && server_config.backup_interval_secs > 0 {
        start_backup_scheduler(
            Arc::clone(&manager),
            server_config.backup_interval_secs,
            server_config.backup_max_count,
        );
    }

    // Start sleep-time / observational memory orchestrator (V1).
    // No-op when `SleepTimeConfig::enabled = false` (the default). On enable,
    // resolves the Anthropic API key from the configured env var, runs the
    // cold-start queue purge, and spawns the worker pool.
    start_sleep_time_orchestrator(Arc::clone(&manager));

    // Start Zenoh transport if feature-enabled and configured
    #[cfg(feature = "zenoh")]
    let zenoh_handle = {
        let zenoh_config = crate::zenoh_transport::ZenohConfig::from_env();
        if zenoh_config.enabled {
            // Create a shared MiniLM embedder for serving peer embedding requests
            let zenoh_embedder: Option<Arc<dyn crate::embeddings::Embedder>> = {
                match crate::embeddings::minilm::MiniLMEmbedder::new(
                    crate::embeddings::minilm::EmbeddingConfig::default(),
                ) {
                    Ok(e) => Some(Arc::new(e)),
                    Err(e) => {
                        warn!("Could not create embedder for Zenoh serving: {e}");
                        None
                    }
                }
            };
            match crate::zenoh_transport::start(Arc::clone(&manager), zenoh_config, zenoh_embedder)
                .await
            {
                Ok(handle) => {
                    info!("Zenoh transport started successfully");
                    // Share the transport session so the ZenohEmbedder cache can reuse it
                    crate::memory::set_shared_zenoh_session(handle.session().clone());
                    Some(handle)
                }
                Err(e) => {
                    error!("Failed to start Zenoh transport: {}. HTTP server will continue without Zenoh.", e);
                    None
                }
            }
        } else {
            info!("Zenoh transport: disabled (set VELD_ZENOH_ENABLED=true to enable)");
            None
        }
    };

    // Configure rate limiting (0 = disabled, for localhost/embedded use).
    //
    // We use the in-tree `rate_limit_governance` layer (which delegates the GCRA
    // decision to the `governor` crate) instead of `tower_governor::GovernorLayer`
    // so we can:
    //   1. cap the reported `Wait for {N}s` value at `burst_size * cell_interval`,
    //      preventing the runaway-wait_time class of bug;
    //   2. atomically swap the inner `Arc<RateLimiter>` from an admin endpoint
    //      to recover from a stuck bucket without a process restart.
    let rate_limit_enabled = server_config.rate_limit_per_second > 0;
    let (rate_limit_layer, reset_handle) = if rate_limit_enabled {
        let rps = server_config.rate_limit_per_second;
        let burst = server_config.rate_limit_burst;
        let cell_interval = std::time::Duration::from_nanos(1_000_000_000 / rps.max(1));
        let handle = ResetHandle::new(rps, burst);
        // Sweep the keyed-store every 5 minutes to drop per-IP entries that
        // have replenished to full burst. Without this the DashMap inside
        // governor grows one entry per distinct peer IP that has ever made a
        // request — unbounded at public scale.
        let _sweeper = handle.spawn_keyed_store_sweeper(std::time::Duration::from_secs(300));
        info!(
            "Rate limiting: enabled rps={} burst={} (cell interval: {:?}, wait_time cap: {}s, keyed-store sweep: 300s)",
            rps,
            burst,
            cell_interval,
            handle.cap_secs()
        );
        (
            Some(RateLimitGovernanceLayer::new(handle.clone())),
            Some(handle),
        )
    } else {
        info!("Rate limiting: disabled (VELD_RATE_LIMIT=0)");
        (None, None)
    };

    // Build CORS layer
    let cors = server_config.cors.to_layer();

    let requested_storage_backend = server_config.requested_storage_backend.to_string();
    let effective_storage_backend = server_config.effective_storage_backend.to_string();

    let metrics_public = server_config.metrics_public;
    let public_rate_limit = server_config.public_rate_limit;

    // Build routes using handlers module
    let probe_routes = runtime.probe_routes();

    let public_routes = runtime.public_routes(metrics_public).route(
        "/",
        axum::routing::get(move || {
            let requested_storage_backend = requested_storage_backend.clone();
            let effective_storage_backend = effective_storage_backend.clone();
            async move {
            axum::Json(serde_json::json!({
                "name": "veld",
                "version": env!("VELD_VERSION_FULL"),
                "description": "Cognitive Memory for AI Agents",
                "requested_storage_backend": requested_storage_backend,
                "effective_storage_backend": effective_storage_backend,
                "health": "/health",
                "api": {
                    "remember": "POST /api/remember",
                    "recall": "POST /api/recall",
                    "forget": "POST /api/forget",
                    "todos": "GET /api/todos",
                    "graph": "GET /api/graph/stats"
                },
                "docs": "https://github.com/Portll/veld"
            }))
            }
        }),
    );

    // Protected routes: always rate-limited when rate limiting is enabled
    let protected_routes = if let Some(ref rate_limit) = rate_limit_layer {
        runtime
            .protected_routes(metrics_public)
            .layer(axum::middleware::from_fn(auth::auth_middleware))
            .layer(rate_limit.clone())
    } else {
        runtime
            .protected_routes(metrics_public)
            .layer(axum::middleware::from_fn(auth::auth_middleware))
    };

    // Public routes: rate-limited when rate limiting is enabled AND public_rate_limit=true.
    // The admin reset endpoint is always mounted on public routes so a stuck limiter can
    // never block its own recovery. If a ResetHandle exists, inject it via Extension.
    let public_routes = if let Some(ref handle) = reset_handle {
        public_routes.layer(axum::Extension(handle.clone()))
    } else {
        public_routes
    };
    let public_routes = if rate_limit_enabled && public_rate_limit {
        if let Some(rate_limit) = rate_limit_layer {
            public_routes.layer(rate_limit)
        } else {
            public_routes
        }
    } else {
        public_routes
    };

    // Probe routes: never rate-limited regardless of config
    // (Kubernetes must always be able to determine liveness/readiness)

    // Combine routes with global middleware
    let request_timeout = std::time::Duration::from_secs(server_config.request_timeout_secs);
    let app = axum::Router::new()
        .merge(probe_routes)
        .merge(public_routes)
        .merge(protected_routes)
        .layer(
            ServiceBuilder::new()
                .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024)) // 2 MB
                .layer(axum::middleware::from_fn(middleware::security_headers))
                .layer(axum::middleware::from_fn(middleware::track_metrics))
                .layer(TimeoutLayer::with_status_code(
                    axum::http::StatusCode::REQUEST_TIMEOUT,
                    request_timeout,
                ))
                .layer(tower::limit::ConcurrencyLimitLayer::new(
                    server_config.max_concurrent_requests,
                ))
                .layer(cors),
        );

    // Conditionally add trace propagation
    #[cfg(feature = "telemetry")]
    let app = app.layer(axum::middleware::from_fn(
        tracing_setup::trace_propagation::propagate_trace_context,
    ));

    // Start server
    let host = &server_config.host;
    let port = server_config.port;
    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap_or_else(|_| {
        tracing::warn!("Invalid VELD_HOST '{}', falling back to 127.0.0.1", host);
        SocketAddr::from(([127, 0, 0, 1], port))
    });

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("Failed to bind TCP listener on {}", addr))?;

    // Small delay for log flush
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    print_ready_message(addr);

    // Use a notify to signal the server to stop accepting new connections
    let shutdown_notify = Arc::new(tokio::sync::Notify::new());
    let shutdown_listener = shutdown_notify.clone();

    let server = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown_listener.notified().await;
    });

    let mut server_handle = tokio::spawn(async move { server.await });

    // Wait for shutdown signal (Ctrl+C / SIGTERM)
    shutdown_signal_with_drain().await;

    // Tell the server to stop accepting new connections
    shutdown_notify.notify_one();

    // Give the server a brief moment to finish in-flight requests
    info!(
        "Waiting up to {}s for in-flight requests...",
        SERVER_DRAIN_TIMEOUT_SECS
    );
    match tokio::time::timeout(
        std::time::Duration::from_secs(SERVER_DRAIN_TIMEOUT_SECS),
        &mut server_handle,
    )
    .await
    {
        Ok(Ok(Ok(()))) => info!("Server stopped gracefully"),
        Ok(Ok(Err(e))) => error!("Server error: {}", e),
        Ok(Err(e)) => error!("Server task panicked: {}", e),
        Err(_) => {
            info!(
                "Server drain timed out after {}s, aborting server task",
                SERVER_DRAIN_TIMEOUT_SECS
            );
            server_handle.abort();
        }
    }

    // Shut down Zenoh transport before flushing databases
    #[cfg(feature = "zenoh")]
    if let Some(handle) = zenoh_handle {
        handle.shutdown().await;
    }

    // Graceful shutdown with cleanup (flush databases, save indices)
    run_shutdown_cleanup(manager_for_shutdown).await;

    Ok(())
}

// =============================================================================
// Background Schedulers
// =============================================================================

fn start_maintenance_scheduler(manager: AppState, interval_secs: u64) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

        // Skip first immediate tick — let server warm up before running maintenance
        interval.tick().await;

        loop {
            interval.tick().await;

            // Cleanup stale streaming sessions
            let extractor = manager.streaming_extractor().clone();
            let cleaned = extractor.cleanup_stale_sessions().await;
            if cleaned > 0 {
                tracing::debug!("Cleaned {} stale streaming sessions", cleaned);
            }

            // Cleanup stale user sessions
            let session_cleaned = manager.session_store().cleanup_stale_sessions();
            if session_cleaned > 0 {
                tracing::debug!("Ended {} stale user sessions", session_cleaned);
            }

            // Run maintenance in blocking thread pool
            let manager_clone = Arc::clone(&manager);
            tokio::task::spawn_blocking(move || {
                manager_clone.run_maintenance_all_users();
            });
        }
    });

    info!(
        "Background maintenance scheduler started (interval: {}s)",
        interval_secs
    );
}

fn start_backup_scheduler(manager: AppState, interval_secs: u64, max_backups: usize) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

        // Skip first immediate tick
        interval.tick().await;

        loop {
            interval.tick().await;

            info!("Starting scheduled backup run...");
            let manager_clone = Arc::clone(&manager);
            let backed_up = tokio::task::spawn_blocking(move || {
                manager_clone.run_backup_all_users(max_backups)
            })
            .await
            .unwrap_or(0);

            if backed_up > 0 {
                info!("Scheduled backup completed: {} users backed up", backed_up);
            }
        }
    });

    info!(
        "Automatic backup scheduler started (interval: {}h, keep: {} backups)",
        interval_secs / 3600,
        max_backups
    );
}

fn start_reminder_scheduler(manager: AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));

        // Skip first immediate tick — let server warm up
        interval.tick().await;

        loop {
            interval.tick().await;

            let manager_clone = Arc::clone(&manager);
            match tokio::task::spawn_blocking(move || manager_clone.check_and_emit_due_reminders())
                .await
            {
                Ok(triggered) => {
                    if triggered > 0 {
                        info!("Active reminder check: {} reminder(s) triggered", triggered);
                    }
                }
                Err(e) => {
                    error!("Reminder scheduler task panicked: {}", e);
                }
            }
        }
    });

    info!("Active reminder scheduler started (interval: 60s)");
}

/// Start the sleep-time / observational memory orchestrator (V1).
///
/// Reads [`SleepTimeConfig`] from env. When `enabled=false` (default) this
/// is a no-op. When enabled:
///   1. Resolves the Anthropic API key from the configured env var. Bails
///      with a warn-level log if missing — server continues without
///      sleep-time, the `/api/sleep_time/*` surface returns 503.
///   2. Constructs the [`AnthropicRewriter`] + [`SleepTimeOrchestrator`].
///   3. Runs the cold-start queue purge (drops items older than the
///      configured TTL — R31 + R67).
///   4. Spawns the worker pool.
///   5. Installs the orchestrator onto the shared `MultiUserMemoryManager`
///      so handlers can find it.
fn start_sleep_time_orchestrator(manager: AppState) {
    use crate::config::SleepTimeProfile;
    use crate::memory::sleep_time::rewriter::{AnthropicRewriter, Rewriter};
    use crate::memory::sleep_time::SleepTimeOrchestrator;

    // Env-driven preset selection (V1 minimum surface). Future: deserialize
    // the full SleepTimeConfig from a config file.
    let profile = match std::env::var("VELD_SLEEP_TIME_PROFILE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "" | "disabled" | "off" | "false" => SleepTimeProfile::Disabled,
        "conservative" => SleepTimeProfile::Conservative,
        "balanced" => SleepTimeProfile::Balanced,
        "aggressive" => SleepTimeProfile::Aggressive,
        other => {
            warn!(
                profile = %other,
                "Unknown VELD_SLEEP_TIME_PROFILE; defaulting to disabled"
            );
            SleepTimeProfile::Disabled
        }
    };

    let cfg = profile.to_config();
    if !cfg.enabled {
        info!("Sleep-time orchestrator: disabled (set VELD_SLEEP_TIME_PROFILE=conservative|balanced|aggressive to enable)");
        return;
    }

    let Ok(api_key) = std::env::var(&cfg.anthropic_api_key_env) else {
        warn!(
            env_var = %cfg.anthropic_api_key_env,
            "Sleep-time enabled but {} is not set; orchestrator NOT started", cfg.anthropic_api_key_env,
        );
        return;
    };

    let rewriter = match AnthropicRewriter::new(api_key, cfg.model.clone()) {
        Ok(r) => Rewriter::Anthropic(r),
        Err(e) => {
            error!(error = %e, "Failed to build AnthropicRewriter; sleep-time NOT started");
            return;
        }
    };

    let orch = match SleepTimeOrchestrator::new(
        cfg,
        Arc::clone(&manager.shared_db),
        Arc::downgrade(&manager),
        Arc::clone(&manager.context_block_store),
        rewriter,
    ) {
        Ok(o) => Arc::new(o),
        Err(e) => {
            error!(error = %e, "Failed to construct SleepTimeOrchestrator; sleep-time NOT started");
            return;
        }
    };

    match orch.cold_start_purge() {
        Ok(purged) if purged > 0 => {
            info!(purged, "Sleep-time queue cold-start purge complete");
        }
        Ok(_) => {}
        Err(e) => {
            warn!(error = %e, "Sleep-time cold-start purge failed (continuing)");
        }
    }

    if let Err(e) = orch.start_workers() {
        error!(error = %e, "Failed to spawn sleep-time workers; sleep-time NOT started");
        return;
    }

    *manager.sleep_time_orchestrator.write() = Some(Arc::clone(&orch));
    info!(
        workers = orch.config().num_workers,
        model = %orch.config().model,
        "Sleep-time orchestrator started"
    );
}

// =============================================================================
// Shutdown Handling
// =============================================================================

/// Wait for shutdown signal (Ctrl+C or SIGTERM on Unix).
async fn shutdown_signal_with_drain() {
    let ctrl_c = async {
        match signal::ctrl_c().await {
            Ok(()) => {}
            Err(err) => {
                warn!(
                    "Ctrl+C shutdown handler unavailable ({}). Continuing without Ctrl+C shutdown support.",
                    err
                );
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(err) => {
                warn!(
                    "SIGTERM shutdown handler unavailable ({}). Continuing without SIGTERM shutdown support.",
                    err
                );
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received, starting graceful shutdown");
}

async fn run_shutdown_cleanup(manager: AppState) {
    info!("Proceeding with database flush...");

    let cleanup_future = async {
        // Flush databases (blocking operation, must use spawn_blocking)
        let manager_for_flush = Arc::clone(&manager);
        let flush_handle =
            tokio::task::spawn_blocking(move || manager_for_flush.flush_all_databases());

        match tokio::time::timeout(
            std::time::Duration::from_secs(DATABASE_FLUSH_TIMEOUT_SECS),
            flush_handle,
        )
        .await
        {
            Ok(Ok(Ok(()))) => info!("Databases flushed successfully"),
            Ok(Ok(Err(e))) => error!("Failed to flush databases: {}", e),
            Ok(Err(e)) => error!("Flush task panicked: {}", e),
            Err(_) => error!(
                "Database flush timed out after {}s",
                DATABASE_FLUSH_TIMEOUT_SECS
            ),
        }

        // Save vector indices (blocking operation, must use spawn_blocking)
        info!("Persisting vector indices...");
        let manager_for_save = Arc::clone(&manager);
        let save_handle =
            tokio::task::spawn_blocking(move || manager_for_save.save_all_vector_indices());

        match tokio::time::timeout(
            std::time::Duration::from_secs(VECTOR_INDEX_SAVE_TIMEOUT_SECS),
            save_handle,
        )
        .await
        {
            Ok(Ok(Ok(()))) => info!("Vector indices saved successfully"),
            Ok(Ok(Err(e))) => error!("Failed to save vector indices: {}", e),
            Ok(Err(e)) => error!("Save task panicked: {}", e),
            Err(_) => error!(
                "Vector index save timed out after {}s",
                VECTOR_INDEX_SAVE_TIMEOUT_SECS
            ),
        }

        #[cfg(feature = "telemetry")]
        tracing_setup::shutdown_tracing();
    };

    match tokio::time::timeout(
        std::time::Duration::from_secs(GRACEFUL_SHUTDOWN_TIMEOUT_SECS),
        cleanup_future,
    )
    .await
    {
        Ok(()) => info!("Server shutdown complete"),
        Err(_) => {
            error!(
                "Graceful shutdown timed out after {}s; aborting process. \
                 Cleanup tasks did not complete within the deadline, and continuing \
                 risks an indefinite hang. RocksDB WAL may not have been fully flushed.",
                GRACEFUL_SHUTDOWN_TIMEOUT_SECS
            );
            std::process::abort();
        }
    }
}

// =============================================================================
// Startup Output
// =============================================================================

fn print_banner() {
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════════════╗");
    eprintln!(
        "  ║            🧠 Veld Server v{}",
        env!("VELD_VERSION_FULL")
    );
    eprintln!("  ║       Cognitive Memory for AI Agents              ║");
    eprintln!("  ╚═══════════════════════════════════════════════════╝");
    eprintln!();
}

fn print_config(config: &ServerConfig) {
    let encryption_enabled = std::env::var("VELD_ENCRYPTION_KEY")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);

    eprintln!("  Configuration:");
    eprintln!(
        "     Mode:    {}",
        if config.is_production {
            "PRODUCTION"
        } else {
            "Development"
        }
    );
    eprintln!("     Host:    {}", config.host);
    eprintln!("     Port:    {}", config.port);
    eprintln!("     Storage: {}", config.storage_path.display());
    if config.requested_storage_backend == config.effective_storage_backend {
        eprintln!("     Backend: {}", config.effective_storage_backend);
    } else {
        eprintln!(
            "     Backend: {} requested, {} active",
            config.requested_storage_backend, config.effective_storage_backend
        );
    }
    eprintln!(
        "     Encryption: {}",
        if encryption_enabled { "enabled" } else { "disabled" }
    );
    if config.cors.deny_all {
        eprintln!("     CORS:    deny-all (set VELD_CORS_ORIGINS)");
    } else if config.cors.is_restricted() {
        eprintln!("     CORS:    restricted");
    } else {
        eprintln!("     CORS:    permissive");
    }
    if config.rate_limit_per_second > 0 {
        eprintln!(
            "     Rate:    {} req/sec (burst {})",
            config.rate_limit_per_second, config.rate_limit_burst
        );
    } else {
        eprintln!("     Rate:    disabled");
    }
    eprintln!();
}

fn log_production_security_warnings(config: &ServerConfig) {
    if !config.is_production {
        return;
    }

    let encryption_enabled = std::env::var("VELD_ENCRYPTION_KEY")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);

    if !encryption_enabled {
        warn!(
            "PRODUCTION WARNING: VELD_ENCRYPTION_KEY is not set. Memory content will be stored in plaintext at rest."
        );
    }

    if config.rate_limit_per_second == 0 {
        warn!(
            "PRODUCTION WARNING: rate limiting is disabled (VELD_RATE_LIMIT=0)."
        );
    }

    // TLS posture: Veld does not terminate TLS itself. Binding to a non-local
    // interface in production means the operator must have a TLS-terminating
    // reverse proxy in front (nginx/Caddy/ALB/Cloudflare) — otherwise API
    // keys, memory content, and webhook payloads cross the wire in plaintext.
    //
    // Operators who have such a proxy can set `VELD_TLS_ACK=true` to silence
    // this warning. Localhost binds never warn because the loopback interface
    // is not a network attacker's vantage point.
    if !crate::config::is_local_bind_host(&config.host) {
        let tls_ack = std::env::var("VELD_TLS_ACK")
            .map(|v| {
                let lower = v.trim().to_ascii_lowercase();
                lower == "true" || lower == "1" || lower == "yes"
            })
            .unwrap_or(false);
        if !tls_ack {
            warn!(
                bind = %config.host,
                "PRODUCTION WARNING: Veld does not terminate TLS. Bind host '{}' is not \
                 a loopback interface — API keys and memory content will cross the wire \
                 in plaintext unless a TLS-terminating reverse proxy (nginx/Caddy/ALB/\
                 Cloudflare) is in front of this process. Set VELD_TLS_ACK=true to \
                 acknowledge and silence this warning.",
                config.host
            );
        }
    }
}

fn print_storage_stats(storage_path: &std::path::Path) {
    if storage_path.exists() {
        let disk_usage = calculate_dir_size(storage_path);
        let user_count = count_user_directories(storage_path);
        eprintln!("  💾 Storage Statistics:");
        eprintln!(
            "     Location:  {}",
            storage_path
                .canonicalize()
                .unwrap_or_else(|_| storage_path.to_path_buf())
                .display()
        );
        eprintln!("     Disk used: {}", format_bytes(disk_usage));
        eprintln!("     Users:     {}", user_count);
        eprintln!();
    } else {
        eprintln!("  💾 Storage: New database (no existing data)");
        eprintln!();
    }
}

fn print_ready_message(addr: SocketAddr) {
    use std::io::Write;
    let _ = std::io::stderr().flush();
    eprintln!();
    eprintln!("  🚀 Server ready!");
    eprintln!("     API:       http://{}", addr);
    eprintln!("     Health:    http://{}/health", addr);
    eprintln!("     Stream:    ws://{}/api/stream", addr);
    #[cfg(feature = "zenoh")]
    {
        let zenoh_enabled = std::env::var("VELD_ZENOH_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        if zenoh_enabled {
            let prefix =
                std::env::var("VELD_ZENOH_PREFIX").unwrap_or_else(|_| "veld".to_string());
            eprintln!(
                "     Zenoh:     {}/*/{{remember,recall,forget,stream,mission}}",
                prefix
            );
            eprintln!("     Fleet:     {}/fleet/*", prefix);
        }
    }
    eprintln!();
    eprintln!("  Press Ctrl+C to stop");
    eprintln!();
    let _ = std::io::stderr().flush();
}

// =============================================================================
// Helper Functions
// =============================================================================

fn calculate_dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += calculate_dir_size(&path);
            } else if let Ok(metadata) = entry.metadata() {
                total += metadata.len();
            }
        }
    }
    total
}

fn count_user_directories(path: &std::path::Path) -> usize {
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| {
                    let name = e.file_name();
                    let name_str = name.to_string_lossy();
                    e.path().is_dir()
                        && name_str != "audit_logs"
                        && name_str != "backups"
                        && name_str != "feedback"
                        && name_str != "semantic_facts"
                        && name_str != "files"
                        && name_str != "prospective"
                        && name_str != "todos"
                })
                .count()
        })
        .unwrap_or(0)
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
