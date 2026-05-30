//! Configuration management for Veld
//!
//! All configurable parameters in one place with environment variable overrides.
//! Follows the principle: sensible defaults, configurable in production.

use std::env;
use std::path::PathBuf;
use tracing::info;

/// Legacy storage directory name used in versions <= 0.1.80.
const LEGACY_STORAGE_DIR: &str = "veld_data";

/// Requested storage backend.
///
/// `redb` is the default target for the storage migration, but the current
/// build still resolves to the legacy RocksDB runtime until the abstraction
/// layer is landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    Redb,
    RocksDb,
}

impl StorageBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Redb => "redb",
            Self::RocksDb => "rocksdb",
        }
    }

    pub const fn is_legacy(self) -> bool {
        matches!(self, Self::RocksDb)
    }
}

impl std::fmt::Display for StorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for StorageBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "redb" => Ok(Self::Redb),
            "rocksdb" | "rocks" | "legacy-rocksdb" => Ok(Self::RocksDb),
            other => Err(format!(
                "unsupported storage backend '{other}' (expected 'redb' or 'rocksdb')"
            )),
        }
    }
}

pub const fn default_requested_storage_backend() -> StorageBackend {
    StorageBackend::Redb
}

pub const fn effective_storage_backend_for_current_build(
    requested: StorageBackend,
) -> StorageBackend {
    match requested {
        StorageBackend::Redb => StorageBackend::RocksDb,
        StorageBackend::RocksDb => StorageBackend::RocksDb,
    }
}

/// Returns the platform-appropriate default storage path.
///
/// Resolution order:
/// 1. If `./veld_data` exists in the cwd (legacy location), use it and warn.
///    This preserves data for users upgrading from <= 0.1.80.
/// 2. Otherwise, use the platform data directory:
///    - Linux: `~/.local/share/veld/`
///    - macOS: `~/Library/Application Support/veld/`
///    - Windows: `C:\Users\<user>\AppData\Roaming\veld\`
/// 3. Falls back to `./veld_data` only if the home directory cannot be determined.
pub fn default_storage_path() -> PathBuf {
    let legacy_path = PathBuf::from(LEGACY_STORAGE_DIR);
    if legacy_path.exists() && legacy_path.is_dir() {
        eprintln!(
            "[veld] Found legacy data at ./{LEGACY_STORAGE_DIR}/ in the current directory. \
             Using it for backward compatibility. To migrate, move it to the platform default \
             and unset VELD_MEMORY_PATH. See: https://github.com/Portll/veld/issues/89"
        );
        return legacy_path;
    }

    dirs::data_dir()
        .map(|p| p.join("veld"))
        .unwrap_or_else(|| PathBuf::from(LEGACY_STORAGE_DIR))
}

/// Returns true when the server is bound only to the local machine.
pub fn is_local_bind_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

/// Path to the Veld `config.toml`, the file written by `veld init`.
///
/// Always `<platform config dir>/veld/config.toml` (e.g. `~/.config/veld/config.toml`
/// on Linux, `%APPDATA%\veld\config.toml` on Windows). Falls back to `./.veld/config.toml`
/// when the platform config directory cannot be determined. This must stay in lockstep
/// with the path `veld init` writes to.
pub fn config_file_path() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join("veld"))
        .unwrap_or_else(|| PathBuf::from(".veld"))
        .join("config.toml")
}

/// Bridges `config.toml` into the process environment.
///
/// `veld init` writes a `config.toml` with the generated API key, host, and port,
/// but every other part of Veld is configured from environment variables. This reads
/// that file once at startup and exports each recognized setting as its `VELD_*`
/// variable — but only when the variable is not already set, so an explicit
/// environment variable always wins.
///
/// Precedence (highest first): environment variable → `config.toml` → built-in default.
///
/// Call once at process start, before `clap` parsing and `ServerConfig::from_env()`.
/// A missing file is silently ignored; a malformed file is reported and ignored —
/// neither aborts startup. The API key value is never logged.
pub fn load_config_file_into_env() {
    let path = config_file_path();
    let raw = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => {
            eprintln!(
                "[veld] Could not read config file {}: {err}. Using environment and defaults.",
                path.display()
            );
            return;
        }
    };

    let table = match toml::from_str::<toml::Table>(&raw) {
        Ok(table) => table,
        Err(err) => {
            eprintln!(
                "[veld] Config file {} is not valid TOML: {err}. Using environment and defaults.",
                path.display()
            );
            return;
        }
    };

    // config.toml key → environment variable. Only these keys are bridged; any other
    // key in the file is ignored so the file format stays forward-compatible.
    const MAPPING: &[(&str, &str)] = &[
        ("api_key", "VELD_API_KEY"),
        ("host", "VELD_HOST"),
        ("port", "VELD_PORT"),
        ("storage", "VELD_MEMORY_PATH"),
        ("mcp", "VELD_MCP_ENABLED"),
    ];

    let mut applied: Vec<&str> = Vec::new();
    for (file_key, env_key) in MAPPING {
        // An explicit environment variable always overrides the config file.
        if env::var_os(env_key).is_some() {
            continue;
        }
        let Some(value) = table.get(*file_key) else {
            continue;
        };
        let rendered = match value {
            toml::Value::String(s) => s.clone(),
            toml::Value::Integer(n) => n.to_string(),
            toml::Value::Boolean(b) => b.to_string(),
            other => {
                eprintln!(
                    "[veld] Ignoring config key '{file_key}': unsupported value type {}.",
                    other.type_str()
                );
                continue;
            }
        };
        if rendered.trim().is_empty() {
            continue;
        }
        env::set_var(env_key, rendered);
        applied.push(*file_key);
    }

    if !applied.is_empty() {
        // Key names only — the API key value is never written to the log.
        eprintln!("[veld] Loaded {} from {}", applied.join(", "), path.display());
    }
}

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins (empty = allow all in development)
    pub allowed_origins: Vec<String>,
    /// Reject all cross-origin requests, used as a production safety fallback.
    pub deny_all: bool,
    /// Allowed HTTP methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    /// Whether to allow credentials
    pub allow_credentials: bool,
    /// Max age for preflight cache (seconds)
    pub max_age_seconds: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: Vec::new(), // Development default: permissive unless production safety checks override it.
            deny_all: false,
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            allow_credentials: false,
            max_age_seconds: 86400, // 24 hours
        }
    }
}

impl CorsConfig {
    /// Load from environment variables with production safety checks
    ///
    /// In production mode (VELD_ENV=production), deny all cross-origin requests
    /// unless VELD_CORS_ORIGINS is explicitly configured.
    pub fn from_env() -> Self {
        let mut config = Self::default();
        let mut cors_origins_configured = false;

        if let Ok(origins) = env::var("VELD_CORS_ORIGINS") {
            config.allowed_origins = origins
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            cors_origins_configured = !config.allowed_origins.is_empty();
        }

        if let Ok(methods) = env::var("VELD_CORS_METHODS") {
            config.allowed_methods = methods
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(headers) = env::var("VELD_CORS_HEADERS") {
            config.allowed_headers = headers
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(val) = env::var("VELD_CORS_CREDENTIALS") {
            config.allow_credentials = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("VELD_CORS_MAX_AGE") {
            if let Ok(n) = val.parse() {
                config.max_age_seconds = n;
            }
        }

        // Safety check: deny all in production if CORS origins are not configured.
        // Warn in development unless suppressed with VELD_CORS_WARN=false.
        let is_production = env::var("VELD_ENV")
            .map(|v| {
                let v = v.to_lowercase();
                v == "production" || v == "prod"
            })
            .unwrap_or(false);

        let cors_warn_suppressed = env::var("VELD_CORS_WARN")
            .map(|v| v.to_lowercase() == "false" || v == "0")
            .unwrap_or(false);

        if !cors_origins_configured {
            if is_production {
                config.deny_all = true;
                if !cors_warn_suppressed {
                    tracing::error!(
                        "PRODUCTION SAFETY: VELD_CORS_ORIGINS is not set. Rejecting all cross-origin requests until it is configured."
                    );
                }
            } else if !cors_warn_suppressed {
                tracing::warn!(
                    "CORS allows all origins in development (no VELD_CORS_ORIGINS set). \
                     Set VELD_CORS_WARN=false to suppress this warning."
                );
            }
        }

        config
    }

    /// Check if any origin restrictions are configured
    pub fn is_restricted(&self) -> bool {
        self.deny_all || !self.allowed_origins.is_empty()
    }

    /// Convert to tower-http CorsLayer
    pub fn to_layer(&self) -> tower_http::cors::CorsLayer {
        use tower_http::cors::{AllowOrigin, Any, CorsLayer};

        let mut layer = CorsLayer::new();

        // Configure allowed origins
        if self.deny_all {
            layer = layer.allow_origin(AllowOrigin::list(Vec::<axum::http::HeaderValue>::new()));
        } else if self.allowed_origins.is_empty() {
            // Intentionally permissive in development when no origins are configured.
            layer = layer.allow_origin(Any);
        } else {
            // Parse configured origins, tracking failures
            let mut valid_origins = Vec::new();
            let mut invalid_origins = Vec::new();

            for origin_str in &self.allowed_origins {
                match origin_str.parse::<axum::http::HeaderValue>() {
                    Ok(origin) => valid_origins.push(origin),
                    Err(_) => invalid_origins.push(origin_str.clone()),
                }
            }

            // Log any invalid origins
            for invalid in &invalid_origins {
                tracing::warn!("CORS: Invalid origin '{}' - skipping", invalid);
            }

            if valid_origins.is_empty() {
                // All configured origins failed to parse - this is a config error
                // Do NOT fall back to permissive - that would be a security hole
                tracing::error!(
                    "CORS: All {} configured origin(s) failed to parse. \
                     Rejecting all cross-origin requests. Fix VELD_CORS_ORIGINS.",
                    self.allowed_origins.len()
                );
                // Use an impossible origin to effectively deny all CORS
                layer =
                    layer.allow_origin(AllowOrigin::list(Vec::<axum::http::HeaderValue>::new()));
            } else {
                if !invalid_origins.is_empty() {
                    tracing::info!(
                        "CORS: Using {} valid origin(s), {} invalid skipped",
                        valid_origins.len(),
                        invalid_origins.len()
                    );
                }
                layer = layer.allow_origin(AllowOrigin::list(valid_origins));
            }
        }

        // Configure allowed methods
        let methods: Vec<axum::http::Method> = self
            .allowed_methods
            .iter()
            .filter_map(|m| m.parse().ok())
            .collect();
        if methods.is_empty() {
            layer = layer.allow_methods(Any);
        } else {
            layer = layer.allow_methods(methods);
        }

        // Configure allowed headers
        let headers: Vec<axum::http::HeaderName> = self
            .allowed_headers
            .iter()
            .filter_map(|h| h.parse().ok())
            .collect();
        if headers.is_empty() {
            layer = layer.allow_headers(Any);
        } else {
            layer = layer.allow_headers(headers);
        }

        // Configure credentials
        if self.allow_credentials {
            layer = layer.allow_credentials(true);
        }

        // Configure max age
        layer = layer.max_age(std::time::Duration::from_secs(self.max_age_seconds));

        layer
    }
}

/// Server configuration loaded from environment with defaults
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server host address (default: 127.0.0.1)
    /// Set to 0.0.0.0 for Docker or network-accessible deployments
    pub host: String,

    /// Server port (default: 3030)
    pub port: u16,

    /// Storage path for the selected storage backend.
    pub storage_path: PathBuf,

    /// Backend requested by config/CLI. `redb` is the default target.
    pub requested_storage_backend: StorageBackend,

    /// Backend actually used by the current build/runtime.
    pub effective_storage_backend: StorageBackend,

    /// Maximum users to keep in memory LRU cache (default: 1000)
    pub max_users_in_memory: usize,

    /// Maximum audit log entries per user (default: 10000)
    pub audit_max_entries_per_user: usize,

    /// Audit log rotation check interval (default: 100)
    pub audit_rotation_check_interval: usize,

    /// Audit log retention days (default: 30)
    pub audit_retention_days: u64,

    /// Rate limit: requests per second (default: 4000 - LLM-friendly)
    pub rate_limit_per_second: u64,

    /// Rate limit: burst size (default: 8000 - allows rapid agent bursts)
    pub rate_limit_burst: u32,

    /// Maximum concurrent requests (default: 200)
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds (default: 60)
    /// Requests exceeding this duration are terminated with 408 status
    pub request_timeout_secs: u64,

    /// Whether running in production mode
    pub is_production: bool,

    /// Apply rate limiting to public (non-probe) routes.
    /// Default: true. Set VELD_PUBLIC_RATE_LIMIT=false to exempt public routes.
    /// Health probe routes (/health*) are never rate-limited regardless.
    pub public_rate_limit: bool,

    /// Expose /metrics without authentication.
    /// Default: false (protected by API key). Set VELD_METRICS_PUBLIC=true for
    /// unauthenticated Prometheus scraping (e.g. in a private network scraper).
    pub metrics_public: bool,

    /// Enable multi-tenant collective learning and auth binding integration.
    pub multi_tenant_mode: bool,

    /// Directory for shared collective state when multi-tenant mode is enabled.
    pub collective_store_dir: PathBuf,

    /// CORS configuration
    pub cors: CorsConfig,

    /// Memory maintenance interval in seconds (default: 300 = 5 minutes)
    /// Controls how often consolidation and activation decay run
    pub maintenance_interval_secs: u64,

    /// Activation decay factor per maintenance cycle (default: 0.95)
    /// Memories lose 5% activation each cycle: A_new = A_old * 0.95
    pub activation_decay_factor: f32,

    /// Backup configuration
    /// Automatic backup interval in seconds (default: 86400 = 24 hours)
    /// Set to 0 to disable automatic backups
    pub backup_interval_secs: u64,

    /// Maximum backups to keep per user (default: 7)
    /// Older backups are automatically purged
    pub backup_max_count: usize,

    /// Whether backups are enabled (default: true in production, false in dev)
    pub backup_enabled: bool,

    /// Maximum entities extracted per memory for graph insertion (default: 10)
    /// Caps the number of NER/tag/regex entities to prevent O(n²) edge explosion
    /// in the knowledge graph. 10 entities → max 45 co-occurrence edges.
    pub max_entities_per_memory: usize,

    /// Log-periodic fractal decay scales in days (default: [7.0, 30.0, 365.0])
    /// Controls resonance frequencies in the power-law decay function.
    /// Weekly/monthly/yearly rhythms by default; override via VELD_LOG_PERIODIC_SCALES
    /// env var (comma-separated, e.g. "14.0,60.0,365.0" for biweekly/bimonthly/yearly)
    pub log_periodic_scales: Vec<f64>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3030,
            storage_path: default_storage_path(),
            requested_storage_backend: default_requested_storage_backend(),
            effective_storage_backend: effective_storage_backend_for_current_build(
                default_requested_storage_backend(),
            ),
            max_users_in_memory: 1000,
            audit_max_entries_per_user: 10_000,
            audit_rotation_check_interval: 100,
            audit_retention_days: 30,
            rate_limit_per_second: 4000,
            rate_limit_burst: 8000,
            max_concurrent_requests: 200,
            request_timeout_secs: 60,
            is_production: false,
            public_rate_limit: true,
            metrics_public: false,
            multi_tenant_mode: false,
            collective_store_dir: default_storage_path().join("collective"),
            cors: CorsConfig::default(),
            maintenance_interval_secs: 3600, // 1 hour (aligns with biological consolidation timescales)
            activation_decay_factor: 0.98, // 2% decay per cycle → 62% retained after 24hr, near-zero at 30 days
            backup_interval_secs: 86400,   // 24 hours
            backup_max_count: 7,           // Keep 7 backups (1 week of daily backups)
            backup_enabled: false,         // Disabled by default, auto-enabled in production
            max_entities_per_memory: 10,   // Cap entities per memory (10 → max 45 edges)
            log_periodic_scales: vec![7.0, 30.0, 365.0], // Weekly, monthly, yearly resonance
        }
    }
}

impl ServerConfig {
    /// Load configuration from environment variables with defaults
    #[allow(clippy::field_reassign_with_default)] // Environment overrides require mutable config
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Check production mode first
        config.is_production = env::var("VELD_ENV")
            .map(|v| {
                let v = v.to_lowercase();
                v == "production" || v == "prod"
            })
            .unwrap_or(false);

        // Host (bind address)
        if let Ok(val) = env::var("VELD_HOST") {
            config.host = val;
        }

        // Port
        if let Ok(val) = env::var("VELD_PORT") {
            if let Ok(port) = val.parse() {
                config.port = port;
            }
        }

        // Storage path
        if let Ok(val) = env::var("VELD_MEMORY_PATH") {
            config.storage_path = PathBuf::from(val);
        }
        config.collective_store_dir = config.storage_path.join("collective");

        if let Ok(val) = env::var("VELD_COLLECTIVE_STORE_DIR") {
            config.collective_store_dir = PathBuf::from(val);
        }

        if let Ok(val) = env::var("VELD_MULTI_TENANT") {
            config.multi_tenant_mode = val.eq_ignore_ascii_case("true") || val == "1";
        }

        // Requested storage backend
        if let Ok(val) = env::var("VELD_STORAGE_BACKEND") {
            match val.parse::<StorageBackend>() {
                Ok(backend) => {
                    config.requested_storage_backend = backend;
                }
                Err(err) => {
                    tracing::warn!(
                        "VELD_STORAGE_BACKEND='{}' ignored: {}. Using default target {}.",
                        val,
                        err,
                        default_requested_storage_backend()
                    );
                }
            }
        }

        config.effective_storage_backend =
            effective_storage_backend_for_current_build(config.requested_storage_backend);

        // Max users in memory
        if let Ok(val) = env::var("VELD_MAX_USERS") {
            if let Ok(n) = val.parse::<usize>() {
                config.max_users_in_memory = n.max(1);
                if n == 0 {
                    tracing::warn!(
                        "VELD_MAX_USERS=0 is invalid (would evict every user), clamped to 1"
                    );
                }
            }
        }

        // Audit settings
        if let Ok(val) = env::var("VELD_AUDIT_MAX_ENTRIES") {
            if let Ok(n) = val.parse::<usize>() {
                config.audit_max_entries_per_user = n.max(100);
                if n < 100 {
                    tracing::warn!(
                        "VELD_AUDIT_MAX_ENTRIES={} is below minimum, clamped to 100",
                        n
                    );
                }
            }
        }

        if let Ok(val) = env::var("VELD_AUDIT_RETENTION_DAYS") {
            if let Ok(n) = val.parse() {
                config.audit_retention_days = n;
            }
        }

        let rate_limit_explicit = env::var("VELD_RATE_LIMIT").is_ok();

        // Rate limiting
        if let Ok(val) = env::var("VELD_RATE_LIMIT") {
            if let Ok(n) = val.parse() {
                config.rate_limit_per_second = n;
            }
        }

        if let Ok(val) = env::var("VELD_RATE_BURST") {
            if let Ok(n) = val.parse() {
                config.rate_limit_burst = n;
            }
        }

        // Rate limiting is disabled on localhost in dev mode for convenience.
        // Risk: a compromised local process or SSRF can exhaust memory without limit.
        // Set VELD_RATE_LIMIT explicitly (or use production mode) to enable limiting.
        if !config.is_production && is_local_bind_host(&config.host) && !rate_limit_explicit {
            config.rate_limit_per_second = 0;
            tracing::info!(
                bind = %config.host,
                production = config.is_production,
                "Rate limiting auto-disabled: localhost+dev detected"
            );
        }

        // Public route rate limiting (default: on)
        if env::var("VELD_PUBLIC_RATE_LIMIT")
            .map(|v| v.eq_ignore_ascii_case("false") || v == "0")
            .unwrap_or(false)
        {
            config.public_rate_limit = false;
        }

        // Metrics exposure (default: protected)
        if env::var("VELD_METRICS_PUBLIC")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false)
        {
            config.metrics_public = true;
        }

        // Concurrency
        if let Ok(val) = env::var("VELD_MAX_CONCURRENT") {
            if let Ok(n) = val.parse::<usize>() {
                config.max_concurrent_requests = n.max(1);
                if n == 0 {
                    tracing::warn!(
                        "VELD_MAX_CONCURRENT=0 is invalid (would reject all requests), clamped to 1"
                    );
                }
            }
        }

        // Request timeout
        if let Ok(val) = env::var("VELD_REQUEST_TIMEOUT") {
            if let Ok(n) = val.parse::<u64>() {
                config.request_timeout_secs = n.max(1);
                if n == 0 {
                    tracing::warn!(
                        "VELD_REQUEST_TIMEOUT=0 is invalid (would instant-timeout all requests), clamped to 1"
                    );
                }
            }
        }

        // CORS configuration
        config.cors = CorsConfig::from_env();

        // Memory maintenance settings
        if let Ok(val) = env::var("VELD_MAINTENANCE_INTERVAL") {
            if let Ok(n) = val.parse::<u64>() {
                config.maintenance_interval_secs = n.max(10);
                if n < 10 {
                    tracing::warn!(
                        "VELD_MAINTENANCE_INTERVAL={} is below minimum (would cause CPU spin-loop), clamped to 10",
                        n
                    );
                }
            }
        }

        if let Ok(val) = env::var("VELD_ACTIVATION_DECAY") {
            if let Ok(n) = val.parse::<f32>() {
                if !n.is_finite() {
                    tracing::warn!(
                        "VELD_ACTIVATION_DECAY={} is not finite, using default {}",
                        val,
                        config.activation_decay_factor
                    );
                } else {
                    let clamped = n.clamp(0.5, 0.99);
                    if (clamped - n).abs() > f32::EPSILON {
                        tracing::warn!(
                            "VELD_ACTIVATION_DECAY={} clamped to {} (valid range: 0.5–0.99)",
                            n,
                            clamped
                        );
                    }
                    config.activation_decay_factor = clamped;
                }
            }
        }

        // Backup configuration
        if let Ok(val) = env::var("VELD_BACKUP_INTERVAL") {
            if let Ok(n) = val.parse::<u64>() {
                if n == 0 {
                    tracing::warn!(
                        "VELD_BACKUP_INTERVAL=0 — backups will run every maintenance cycle"
                    );
                }
                config.backup_interval_secs = n;
            }
        }

        if let Ok(val) = env::var("VELD_BACKUP_MAX_COUNT") {
            if let Ok(n) = val.parse::<usize>() {
                config.backup_max_count = n.max(1);
                if n == 0 {
                    tracing::warn!(
                        "VELD_BACKUP_MAX_COUNT=0 is invalid (would keep no backups), clamped to 1"
                    );
                }
            }
        }

        // Auto-enable backups in production mode unless explicitly disabled
        if let Ok(val) = env::var("VELD_BACKUP_ENABLED") {
            config.backup_enabled = val.to_lowercase() == "true" || val == "1";
        } else if config.is_production {
            // Auto-enable in production
            config.backup_enabled = true;
        }

        // Entity extraction cap
        if let Ok(val) = env::var("VELD_MAX_ENTITIES") {
            if let Ok(n) = val.parse::<usize>() {
                let clamped = n.clamp(1, 50);
                if clamped != n {
                    tracing::warn!(
                        "VELD_MAX_ENTITIES={} clamped to {} (valid range: 1–50)",
                        n,
                        clamped
                    );
                }
                config.max_entities_per_memory = clamped;
            }
        }

        // Log-periodic decay scales (comma-separated floats, e.g. "14.0,60.0,365.0")
        if let Ok(val) = env::var("VELD_LOG_PERIODIC_SCALES") {
            let parsed: Vec<f64> = val
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .filter(|&v| (1.0..=730.0).contains(&v))
                .collect();
            if parsed.is_empty() {
                tracing::warn!(
                    "VELD_LOG_PERIODIC_SCALES='{}' — no valid scales (need ≥1 float in 1.0–730.0), using defaults",
                    val
                );
            } else {
                tracing::info!(
                    "Log-periodic scales overridden: {:?} (default: [7.0, 30.0, 365.0])",
                    parsed
                );
                config.log_periodic_scales = parsed;
            }
        }

        config
    }

    /// Log the current configuration
    pub fn log(&self) {
        info!("📋 Configuration:");
        info!(
            "   Mode: {}",
            if self.is_production {
                "PRODUCTION"
            } else {
                "Development"
            }
        );
        info!("   Port: {}", self.port);
        info!("   Storage: {:?}", self.storage_path);
        if self.requested_storage_backend == self.effective_storage_backend {
            info!("   Backend: {}", self.effective_storage_backend);
        } else {
            info!(
                "   Backend: requested {}, running {} (compatibility path)",
                self.requested_storage_backend,
                self.effective_storage_backend
            );
        }
        info!("   Max users in memory: {}", self.max_users_in_memory);
        if self.rate_limit_per_second > 0 {
            info!(
                "   Rate limit: {} req/sec (burst: {})",
                self.rate_limit_per_second, self.rate_limit_burst
            );
        } else {
            info!("   Rate limit: disabled");
        }
        info!("   Max concurrent: {}", self.max_concurrent_requests);
        info!("   Request timeout: {}s", self.request_timeout_secs);
        info!("   Audit retention: {} days", self.audit_retention_days);
        if self.cors.is_restricted() {
            if self.cors.deny_all {
                info!("   CORS: deny-all (set VELD_CORS_ORIGINS to allow browsers)");
            } else {
                info!("   CORS origins: {:?}", self.cors.allowed_origins);
            }
        } else {
            info!("   CORS: Permissive (development default)");
        }
        info!(
            "   Maintenance interval: {}s (decay factor: {:.2})",
            self.maintenance_interval_secs, self.activation_decay_factor
        );
        if self.backup_enabled {
            let interval_hours = self.backup_interval_secs / 3600;
            info!(
                "   Backup: enabled (every {}h, keep {})",
                interval_hours, self.backup_max_count
            );
        } else {
            info!("   Backup: disabled");
        }
    }
}

/// Environment variable documentation
#[allow(unused)] // Public API - available for CLI help output
pub fn print_env_help() {
    println!("Veld Configuration Environment Variables:");
    println!();
    println!("  VELD_ENV              - Set to 'production' or 'prod' for production mode");
    println!(
        "  VELD_HOST             - Bind address (default: 127.0.0.1, use 0.0.0.0 for Docker)"
    );
    println!("  VELD_PORT             - Server port (default: 3030)");
    println!("  VELD_MEMORY_PATH      - Storage directory (default: platform data dir, e.g. ~/.local/share/veld/)");
    println!("  VELD_STORAGE_BACKEND  - Requested backend: redb (target default) or rocksdb (legacy compatibility)");
    println!("  VELD_API_KEYS         - Comma-separated API keys (required in production)");
    println!("  VELD_DEV_API_KEY      - Development API key (required in dev if VELD_API_KEYS not set)");
    println!("  VELD_ENCRYPTION_KEY   - 32-byte field-encryption key (required for encrypted-at-rest production)");
    println!("  VELD_MAX_USERS        - Max users in memory LRU (default: 1000)");
    println!("  VELD_RATE_LIMIT       - Requests per second (default: 0 on localhost/dev, otherwise 4000)");
    println!("  VELD_RATE_BURST       - Burst size (default: 8000)");
    println!("  VELD_MAX_CONCURRENT   - Max concurrent requests (default: 200)");
    println!("  VELD_REQUEST_TIMEOUT  - Request timeout in seconds (default: 60)");
    println!("  VELD_AUDIT_MAX_ENTRIES    - Max audit entries per user (default: 10000)");
    println!("  VELD_AUDIT_RETENTION_DAYS - Audit log retention days (default: 30)");
    println!();
    println!("Security — secure-by-default overrides:");
    println!("  VELD_ALLOW_UNSIGNED_WEBHOOKS - Allow webhooks when no *_WEBHOOK_SECRET is set (default: false).");
    println!("                                 False = reject unsigned webhooks with 503. True = warn and process.");
    println!("  VELD_PUBLIC_RATE_LIMIT       - Apply rate limiting to non-probe public routes (default: true).");
    println!("                                 Set to false to exempt public routes (probe routes are never limited).");
    println!("  VELD_METRICS_PUBLIC          - Expose /metrics without authentication (default: false).");
    println!("                                 True = /metrics is public (for unauthenticated Prometheus scrapers).");
    println!("  VELD_ADMIN_API_KEY           - API key for /api/admin/* endpoints (separate from VELD_API_KEYS).");
    println!("                                 Required to use admin operations such as rate-limit reset.");
    println!("  VELD_ENFORCE_HTTPS           - Reject insecure HTTP overrides for integration API URLs (default: false).");
    println!("                                 True = non-localhost http:// overrides fall back to the secure default.");
    println!();
    println!("Integration APIs:");
    println!("  LINEAR_API_URL         - Linear GraphQL API URL (default: https://api.linear.app/graphql)");
    println!("  LINEAR_WEBHOOK_SECRET  - Linear webhook signing secret for HMAC verification");
    println!("  GITHUB_API_URL         - GitHub REST API URL (default: https://api.github.com)");
    println!("  GITHUB_WEBHOOK_SECRET  - GitHub webhook secret for HMAC verification");
    println!();
    println!("CORS Configuration:");
    println!("  VELD_CORS_ORIGINS     - Comma-separated allowed origins (required in production for browser access)");
    println!("  VELD_CORS_METHODS     - Comma-separated allowed methods (default: GET,POST,PUT,DELETE,OPTIONS)");
    println!("  VELD_CORS_HEADERS     - Comma-separated allowed headers (default: Content-Type,Authorization,X-Request-ID)");
    println!("  VELD_CORS_CREDENTIALS - Allow credentials true/false (default: false)");
    println!("  VELD_CORS_MAX_AGE     - Preflight cache seconds (default: 86400)");
    println!();
    println!("Backup Configuration:");
    println!("  VELD_BACKUP_ENABLED   - Enable automatic backups true/false (default: auto in production)");
    println!("  VELD_BACKUP_INTERVAL  - Backup interval in seconds (default: 86400 = 24 hours)");
    println!("  VELD_BACKUP_MAX_COUNT - Max backups to keep per user (default: 7)");
    println!();
    println!("  RUST_LOG               - Log level (e.g., info, debug, trace)");
    println!();
}

// =============================================================================
// Sleep-time / observational memory config
// =============================================================================
//
// `SleepTimeConfig` is the full knob surface; `SleepTimeProfile` (E1) is a
// preset enum that collapses common combinations to a single user choice and
// expands to a full config via `to_config()`. The orchestrator consumes the
// expanded `SleepTimeConfig`.

use serde::{Deserialize, Serialize};

use crate::constants::{
    SLEEP_TIME_CALLS_PER_DAY, SLEEP_TIME_CLAIM_LEASE_SECS, SLEEP_TIME_DEBOUNCE_SECS,
    SLEEP_TIME_GLOBAL_CALLS_PER_DAY, SLEEP_TIME_GLOBAL_TOKENS_PER_DAY,
    SLEEP_TIME_IDLE_THRESHOLD_SECS, SLEEP_TIME_QUEUE_COLD_START_TTL_HOURS,
    SLEEP_TIME_TOKENS_PER_HOUR, SLEEP_TIME_WORKERS,
};

/// Full sleep-time configuration surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepTimeConfig {
    /// Master kill switch. When false the orchestrator is created in a
    /// shadow state — triggers may be accepted but no LLM call is made.
    /// V1 ships defaulting to `false` so deployments opt in explicitly.
    pub enabled: bool,

    /// LLM model id, e.g. `"claude-sonnet-4-6"`. Stamped into the
    /// `SleepTimeBlockRewritten` event for audit. Must match an Anthropic
    /// Messages-API model when `enabled = true`.
    pub model: String,

    /// Environment-variable name holding the Anthropic API key. The
    /// orchestrator looks it up via `std::env::var` at startup; the key
    /// itself is never persisted in config files or logged.
    pub anthropic_api_key_env: String,

    /// Worker pool size (R3). 0 means single-worker mode (no concurrency).
    pub num_workers: usize,

    /// Seconds of foreground inactivity before idle triggers fire (R13).
    pub idle_threshold_secs: i64,

    /// Repeated `(user, mode, trigger)` enqueues within this window
    /// collapse to a single queue item.
    pub debounce_secs: i64,

    /// Per-user hourly token cap (R12). Conservative default in
    /// `SLEEP_TIME_TOKENS_PER_HOUR`.
    pub tokens_per_hour: u32,

    /// Per-user daily LLM-call cap (R12).
    pub calls_per_day: u32,

    /// Global (cross-user) daily token cap (R33).
    pub global_tokens_per_day: u64,

    /// Global (cross-user) daily call cap (R33).
    pub global_calls_per_day: u64,

    /// Cold-start queue TTL in hours (R31).
    pub queue_cold_start_ttl_hours: i64,

    /// Worker claim lease in seconds (R3 — claims expire so a dead worker
    /// doesn't lock items forever).
    pub claim_lease_secs: i64,
}

impl Default for SleepTimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "claude-sonnet-4-6".into(),
            anthropic_api_key_env: "ANTHROPIC_API_KEY".into(),
            num_workers: SLEEP_TIME_WORKERS,
            idle_threshold_secs: SLEEP_TIME_IDLE_THRESHOLD_SECS,
            debounce_secs: SLEEP_TIME_DEBOUNCE_SECS,
            tokens_per_hour: SLEEP_TIME_TOKENS_PER_HOUR,
            calls_per_day: SLEEP_TIME_CALLS_PER_DAY,
            global_tokens_per_day: SLEEP_TIME_GLOBAL_TOKENS_PER_DAY,
            global_calls_per_day: SLEEP_TIME_GLOBAL_CALLS_PER_DAY,
            queue_cold_start_ttl_hours: SLEEP_TIME_QUEUE_COLD_START_TTL_HOURS,
            claim_lease_secs: SLEEP_TIME_CLAIM_LEASE_SECS,
        }
    }
}

/// Preset that materialises into a full [`SleepTimeConfig`] (E1).
///
/// Collapses the ~12-knob surface into one user-facing decision. Custom
/// users still have full access via [`SleepTimeProfile::Custom`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SleepTimeProfile {
    /// Off. `enabled=false`, conservative caps preserved.
    Disabled,
    /// Conservative: enabled but with very tight budgets and only one worker;
    /// suitable for first-week opt-in to observe behaviour at low cost.
    Conservative,
    /// Balanced (recommended): defaults.
    Balanced,
    /// Aggressive: higher budgets, more workers, looser debounce; for
    /// power users with established usage patterns.
    Aggressive,
    /// Operator-supplied custom config.
    Custom(Box<SleepTimeConfig>),
}

impl SleepTimeProfile {
    /// Expand the preset into a concrete config.
    pub fn to_config(&self) -> SleepTimeConfig {
        match self {
            Self::Disabled => SleepTimeConfig {
                enabled: false,
                ..SleepTimeConfig::default()
            },
            Self::Conservative => SleepTimeConfig {
                enabled: true,
                num_workers: 1,
                tokens_per_hour: SLEEP_TIME_TOKENS_PER_HOUR / 4, // 2500 tok/hr
                calls_per_day: SLEEP_TIME_CALLS_PER_DAY / 5,     // 10 calls/day
                debounce_secs: SLEEP_TIME_DEBOUNCE_SECS * 2,     // 10 min
                ..SleepTimeConfig::default()
            },
            Self::Balanced => SleepTimeConfig {
                enabled: true,
                ..SleepTimeConfig::default()
            },
            Self::Aggressive => SleepTimeConfig {
                enabled: true,
                num_workers: SLEEP_TIME_WORKERS * 2,
                tokens_per_hour: SLEEP_TIME_TOKENS_PER_HOUR * 3, // 30k tok/hr
                calls_per_day: SLEEP_TIME_CALLS_PER_DAY * 4,     // 200 calls/day
                debounce_secs: SLEEP_TIME_DEBOUNCE_SECS / 3,     // 100s
                ..SleepTimeConfig::default()
            },
            Self::Custom(cfg) => (**cfg).clone(),
        }
    }
}

impl Default for SleepTimeProfile {
    fn default() -> Self {
        Self::Disabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{LazyLock, Mutex};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 3030);
        assert_eq!(config.max_users_in_memory, 1000);
        assert!(!config.is_production);
        assert_eq!(config.requested_storage_backend, StorageBackend::Redb);
        assert_eq!(config.effective_storage_backend, StorageBackend::RocksDb);
    }

    #[test]
    fn test_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("VELD_PORT", "8080");
        env::set_var("VELD_MAX_USERS", "500");
        env::set_var("VELD_STORAGE_BACKEND", "rocksdb");

        let config = ServerConfig::from_env();
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_users_in_memory, 500);
        assert_eq!(config.requested_storage_backend, StorageBackend::RocksDb);
        assert_eq!(config.effective_storage_backend, StorageBackend::RocksDb);

        env::remove_var("VELD_PORT");
        env::remove_var("VELD_MAX_USERS");
        env::remove_var("VELD_STORAGE_BACKEND");
    }

    #[test]
    fn test_local_dev_disables_rate_limit_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::remove_var("VELD_RATE_LIMIT");
        env::remove_var("VELD_HOST");
        env::remove_var("VELD_ENV");

        let config = ServerConfig::from_env();
        assert_eq!(config.rate_limit_per_second, 0);
    }

    #[test]
    fn test_explicit_rate_limit_override_is_preserved_locally() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("VELD_RATE_LIMIT", "4000");
        env::remove_var("VELD_HOST");
        env::remove_var("VELD_ENV");

        let config = ServerConfig::from_env();
        assert_eq!(config.rate_limit_per_second, 4000);

        env::remove_var("VELD_RATE_LIMIT");
    }

    #[test]
    fn test_cors_default_is_permissive() {
        let cors = CorsConfig::default();
        assert!(!cors.is_restricted());
        assert!(cors.allowed_origins.is_empty());
        assert!(!cors.deny_all);
        assert!(!cors.allowed_methods.is_empty());
        assert!(!cors.allowed_headers.is_empty());
    }

    #[test]
    fn test_production_without_cors_origins_denies_all() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("VELD_ENV", "production");
        env::remove_var("VELD_CORS_ORIGINS");

        let cors = CorsConfig::from_env();
        assert!(cors.deny_all);
        assert!(cors.is_restricted());

        env::remove_var("VELD_ENV");
    }

    #[test]
    fn test_production_with_cors_origins_is_restricted_but_not_deny_all() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("VELD_ENV", "production");
        env::set_var("VELD_CORS_ORIGINS", "https://app.example.com");

        let cors = CorsConfig::from_env();
        assert!(!cors.deny_all);
        assert!(cors.is_restricted());
        assert_eq!(cors.allowed_origins, vec!["https://app.example.com".to_string()]);

        env::remove_var("VELD_ENV");
        env::remove_var("VELD_CORS_ORIGINS");
    }

    #[test]
    fn test_cors_with_origins_is_restricted() {
        let cors = CorsConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            ..Default::default()
        };
        assert!(cors.is_restricted());
    }

    #[test]
    fn test_cors_to_layer_permissive() {
        let cors = CorsConfig::default();
        let _layer = cors.to_layer(); // Should not panic
    }

    #[test]
    fn test_cors_to_layer_restricted() {
        let cors = CorsConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            ..Default::default()
        };
        let _layer = cors.to_layer(); // Should not panic
    }
}

/// Selected relational backend for the W4 cutover (the `memories`
/// projection writes through it), the W6 query planner (reads), and the W7
/// dataset surface — all share one store.
///
/// Parsed from `VELD_RELATIONAL_BACKEND`. Unset (the default) keeps the
/// rusqlite-only behaviour: the projection writes to the local slow store
/// and the dataset / `/api/query/*` surfaces return 503.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationalBackendChoice {
    /// A SQLite database file (sqlx-backed, distinct from the rusqlite slow
    /// store). `VELD_RELATIONAL_SQLITE_PATH` gives the path.
    Sqlite { path: String },
    /// A Postgres server. `VELD_POSTGRES_URL` is the connection string.
    /// Requires building with `--features postgres`.
    #[cfg(feature = "postgres")]
    Postgres { url: String },
    /// A Supabase project. `VELD_SUPABASE_PROJECT_REF` +
    /// `VELD_SUPABASE_DB_PASSWORD` (the database password — not the API
    /// JWTs). Requires `--features postgres`.
    #[cfg(feature = "postgres")]
    Supabase {
        project_ref: String,
        db_password: String,
    },
    /// A Microsoft SQL Server. `VELD_MSSQL_ADO_STRING` is the ADO-style
    /// connection string. Requires building with `--features mssql`.
    #[cfg(feature = "mssql")]
    Mssql { ado_string: String },
}

impl RelationalBackendChoice {
    /// Parse the relational backend from the environment.
    ///
    /// Returns `Ok(None)` when `VELD_RELATIONAL_BACKEND` is unset/empty.
    /// Returns `Err` with an actionable message when a backend is named but
    /// its required env vars are missing, the value is unknown, or a
    /// Postgres-family backend is requested in a build without the
    /// `postgres` feature.
    pub fn from_env() -> Result<Option<Self>, String> {
        let backend = match std::env::var("VELD_RELATIONAL_BACKEND") {
            Ok(v) if !v.trim().is_empty() => v.trim().to_ascii_lowercase(),
            _ => return Ok(None),
        };
        match backend.as_str() {
            "sqlite" => {
                let path = std::env::var("VELD_RELATIONAL_SQLITE_PATH").map_err(|_| {
                    "VELD_RELATIONAL_BACKEND=sqlite requires VELD_RELATIONAL_SQLITE_PATH".to_string()
                })?;
                Ok(Some(Self::Sqlite { path }))
            }
            #[cfg(feature = "postgres")]
            "postgres" => {
                let url = std::env::var("VELD_POSTGRES_URL").map_err(|_| {
                    "VELD_RELATIONAL_BACKEND=postgres requires VELD_POSTGRES_URL".to_string()
                })?;
                Ok(Some(Self::Postgres { url }))
            }
            #[cfg(feature = "postgres")]
            "supabase" => {
                let project_ref = std::env::var("VELD_SUPABASE_PROJECT_REF").map_err(|_| {
                    "VELD_RELATIONAL_BACKEND=supabase requires VELD_SUPABASE_PROJECT_REF".to_string()
                })?;
                let db_password = std::env::var("VELD_SUPABASE_DB_PASSWORD").map_err(|_| {
                    "VELD_RELATIONAL_BACKEND=supabase requires VELD_SUPABASE_DB_PASSWORD".to_string()
                })?;
                Ok(Some(Self::Supabase {
                    project_ref,
                    db_password,
                }))
            }
            #[cfg(feature = "mssql")]
            "mssql" => {
                let ado_string = std::env::var("VELD_MSSQL_ADO_STRING").map_err(|_| {
                    "VELD_RELATIONAL_BACKEND=mssql requires VELD_MSSQL_ADO_STRING".to_string()
                })?;
                Ok(Some(Self::Mssql { ado_string }))
            }
            #[cfg(not(feature = "postgres"))]
            "postgres" | "supabase" => Err(format!(
                "VELD_RELATIONAL_BACKEND={backend} requires building veld with --features postgres"
            )),
            #[cfg(not(feature = "mssql"))]
            "mssql" => Err(
                "VELD_RELATIONAL_BACKEND=mssql requires building veld with --features mssql"
                    .to_string(),
            ),
            other => Err(format!(
                "unknown VELD_RELATIONAL_BACKEND {other:?} (expected: sqlite, postgres, supabase, mssql)"
            )),
        }
    }
}
