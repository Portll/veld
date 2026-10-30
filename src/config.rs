//! Configuration management for Veld
//!
//! All configurable parameters in one place with environment variable overrides.
//! Follows the principle: sensible defaults, configurable in production.

use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing::info;

use crate::memory::gap_topology::GapWorkerConfig;

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

/// RocksDB whenever this build has it (ADR 0007), otherwise redb; also the CLI's
/// `--storage-backend` default. An existing redb store with no explicit request then meets the
/// boot guard, which refuses and names `VELD_STORAGE_BACKEND=redb` or `veld migrate`.
pub const fn default_requested_storage_backend() -> StorageBackend {
    #[cfg(any(feature = "rocksdb", not(feature = "storage-redb")))]
    {
        StorageBackend::RocksDb
    }
    #[cfg(all(not(feature = "rocksdb"), feature = "storage-redb"))]
    {
        StorageBackend::Redb
    }
}

// Every construction path is wired for redb (bundle, backup engine, MemorySystem,
// GraphMemory). A rocksdb request in a build without the `rocksdb` feature stays rocksdb here so
// bootstrap refuses it by name rather than serving another engine.
pub const fn effective_storage_backend_for_current_build(
    requested: StorageBackend,
) -> StorageBackend {
    match requested {
        #[cfg(feature = "storage-redb")]
        StorageBackend::Redb => StorageBackend::Redb,
        // Without the feature there is no redb code to run, so serving the
        // request would be a lie the binary cannot honour.
        #[cfg(not(feature = "storage-redb"))]
        StorageBackend::Redb => StorageBackend::RocksDb,
        StorageBackend::RocksDb => StorageBackend::RocksDb,
    }
}

/// What a storage commit guarantees by the time it returns. One class governs both
/// engines, so a store survives the same failures whichever engine holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDurability {
    /// On stable storage: survives a power loss. redb `Durability::Immediate`; RocksDB
    /// syncs its write-ahead log on every write.
    Immediate,
    /// Handed to the operating system: survives a process crash, and a power loss can take
    /// the most recent commits. redb `Durability::Eventual` (redb 2.6 still syncs the file on
    /// Linux and orders writes with `F_BARRIERFSYNC` on macOS); RocksDB writes its
    /// write-ahead log without a sync.
    Eventual,
}

impl StorageDurability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::Eventual => "eventual",
        }
    }
}

impl std::fmt::Display for StorageDurability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for StorageDurability {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "immediate" => Ok(Self::Immediate),
            "eventual" => Ok(Self::Eventual),
            other => Err(format!(
                "unsupported storage durability '{other}' (expected 'immediate' or 'eventual')"
            )),
        }
    }
}

// Eventual because RocksDB, the default engine (ADR 0007), has never synced a write unless asked.
pub const DEFAULT_STORAGE_DURABILITY: StorageDurability = StorageDurability::Eventual;

/// redb's own page-cache default, which every redb database opened with before
/// `VELD_REDB_CACHE_BYTES` existed. It is a ceiling per database file, filled as pages are read.
pub const DEFAULT_REDB_CACHE_BYTES: usize = 1024 * 1024 * 1024;

/// Block cache a RocksDB memory store opens with when no shared cache is supplied
/// (standalone and CLI use; the server supplies one shared cache for every store).
pub const DEFAULT_ROCKSDB_STANDALONE_BLOCK_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Storage-engine durability, cache and write-buffer settings, read from `VELD_*`.
///
/// Process-wide: every store either engine opens without an explicit tuning (the
/// `*_with_tuning` constructors, for tests and tools) takes [`StorageTuning::process`], so
/// two stores in one process never disagree about what a commit means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageTuning {
    /// `VELD_STORAGE_DURABILITY`.
    pub durability: StorageDurability,
    /// `VELD_REDB_CACHE_BYTES`: page cache per redb database file.
    pub redb_cache_bytes: usize,
    /// `VELD_ROCKSDB_WRITE_BUFFER_BYTES`: memtable size of each column family of a RocksDB
    /// memory store.
    pub rocksdb_write_buffer_bytes: usize,
    /// `VELD_ROCKSDB_BLOCK_CACHE_BYTES`: the RocksDB block cache, which is the server's one
    /// shared cache or a standalone store's own. `None` keeps each one's built-in size.
    pub rocksdb_block_cache_bytes: Option<usize>,
}

impl Default for StorageTuning {
    fn default() -> Self {
        Self {
            durability: DEFAULT_STORAGE_DURABILITY,
            redb_cache_bytes: DEFAULT_REDB_CACHE_BYTES,
            rocksdb_write_buffer_bytes: crate::constants::ROCKSDB_MEMORY_WRITE_BUFFER_BYTES,
            rocksdb_block_cache_bytes: None,
        }
    }
}

impl StorageTuning {
    /// The tuning this process opens its stores with: the environment, read once.
    pub fn process() -> Self {
        static PROCESS_TUNING: OnceLock<StorageTuning> = OnceLock::new();
        *PROCESS_TUNING.get_or_init(Self::from_env)
    }

    /// Parse the environment now. An unparsable value keeps its default and logs a warning.
    pub fn from_env() -> Self {
        Self::from_lookup(|name| env::var(name).ok())
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let defaults = Self::default();
        Self {
            durability: durability_from_lookup(&lookup),
            redb_cache_bytes: byte_count_from_lookup(
                &lookup,
                "VELD_REDB_CACHE_BYTES",
                defaults.redb_cache_bytes,
            ),
            rocksdb_write_buffer_bytes: byte_count_from_lookup(
                &lookup,
                "VELD_ROCKSDB_WRITE_BUFFER_BYTES",
                defaults.rocksdb_write_buffer_bytes,
            ),
            rocksdb_block_cache_bytes: lookup("VELD_ROCKSDB_BLOCK_CACHE_BYTES").and_then(|raw| {
                parse_byte_count("VELD_ROCKSDB_BLOCK_CACHE_BYTES", &raw, "the built-in size")
            }),
        }
    }

    /// Whether RocksDB writes sync the write-ahead log before returning.
    pub const fn rocksdb_sync_writes(&self) -> bool {
        matches!(self.durability, StorageDurability::Immediate)
    }

    /// Block cache for a RocksDB store opened without a shared cache.
    pub fn rocksdb_standalone_block_cache_bytes(&self) -> usize {
        self.rocksdb_block_cache_bytes
            .unwrap_or(DEFAULT_ROCKSDB_STANDALONE_BLOCK_CACHE_BYTES)
    }

    /// The server's one block cache, shared by every RocksDB store it opens.
    pub fn rocksdb_shared_block_cache_bytes(&self) -> usize {
        self.rocksdb_block_cache_bytes
            .unwrap_or(crate::constants::ROCKSDB_SHARED_CACHE_BYTES)
    }
}

/// `VELD_STORAGE_DURABILITY`, else the legacy `VELD_WRITE_MODE` (`sync` was RocksDB's opt-in
/// to a synced write), else the default.
fn durability_from_lookup(lookup: &impl Fn(&str) -> Option<String>) -> StorageDurability {
    if let Some(raw) = lookup("VELD_STORAGE_DURABILITY") {
        return match raw.parse::<StorageDurability>() {
            Ok(durability) => durability,
            Err(err) => {
                tracing::warn!(
                    "VELD_STORAGE_DURABILITY='{}' ignored: {}. Using default {}.",
                    raw,
                    err,
                    DEFAULT_STORAGE_DURABILITY
                );
                DEFAULT_STORAGE_DURABILITY
            }
        };
    }
    match lookup("VELD_WRITE_MODE") {
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "sync" => StorageDurability::Immediate,
            "async" => StorageDurability::Eventual,
            _ => {
                tracing::warn!(
                    "VELD_WRITE_MODE='{}' ignored (expected 'sync' or 'async'; \
                     VELD_STORAGE_DURABILITY supersedes it). Using default {}.",
                    raw,
                    DEFAULT_STORAGE_DURABILITY
                );
                DEFAULT_STORAGE_DURABILITY
            }
        },
        None => DEFAULT_STORAGE_DURABILITY,
    }
}

fn byte_count_from_lookup(
    lookup: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: usize,
) -> usize {
    lookup(name)
        .and_then(|raw| parse_byte_count(name, &raw, &default.to_string()))
        .unwrap_or(default)
}

/// A positive whole number of bytes, or `None` after a warning naming the default kept.
fn parse_byte_count(name: &str, raw: &str, default_label: &str) -> Option<usize> {
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            tracing::warn!(
                "{}=0 is invalid (a zero-byte cache or buffer), using default {}",
                name,
                default_label
            );
            None
        }
        Ok(bytes) => Some(bytes),
        Err(_) => {
            tracing::warn!(
                "{}='{}' ignored: not a whole number of bytes. Using default {}.",
                name,
                raw,
                default_label
            );
            None
        }
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

/// Bind address of the TLS listener when `VELD_TLS_HOST` is unset.
pub const TLS_HOST_DEFAULT: &str = "0.0.0.0";

/// The native TLS listener's settings as given: `VELD_TLS_*`, config.toml `tls_*`, or
/// `veld server --tls-*`. [`TlsSettings::validate`] turns them into a
/// [`TlsListenerConfig`], or into the refusal that names what is wrong.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TlsSettings {
    pub cert: Option<PathBuf>,
    pub key: Option<PathBuf>,
    pub port: Option<String>,
    pub host: Option<String>,
    pub client_ca: Option<PathBuf>,
}

/// A TLS listener whose settings are complete: where it binds and the files it serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsListenerConfig {
    pub addr: std::net::SocketAddr,
    pub cert: PathBuf,
    pub key: PathBuf,
    /// When set, a client must present a certificate that chains to this CA.
    pub client_ca: Option<PathBuf>,
}

impl TlsSettings {
    /// Read the `VELD_TLS_*` variables through `get`; a blank value counts as unset.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let set = |name: &str| {
            get(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        Self {
            cert: set("VELD_TLS_CERT").map(PathBuf::from),
            key: set("VELD_TLS_KEY").map(PathBuf::from),
            port: set("VELD_TLS_PORT"),
            host: set("VELD_TLS_HOST"),
            client_ca: set("VELD_TLS_CLIENT_CA").map(PathBuf::from),
        }
    }

    /// Each variable with its value, `None` where unset.
    pub fn vars(&self) -> [(&'static str, Option<String>); 5] {
        let path = |p: &Option<PathBuf>| p.as_ref().map(|p| p.to_string_lossy().into_owned());
        [
            ("VELD_TLS_CERT", path(&self.cert)),
            ("VELD_TLS_KEY", path(&self.key)),
            ("VELD_TLS_PORT", self.port.clone()),
            ("VELD_TLS_HOST", self.host.clone()),
            ("VELD_TLS_CLIENT_CA", path(&self.client_ca)),
        ]
    }

    /// These settings as [`Self::from_lookup`] reads them: each value trimmed, a blank one
    /// unset. `veld server --tls-*` values arrive as given.
    fn normalized(&self) -> Self {
        let vars = self.vars();
        Self::from_lookup(|name| {
            vars.iter()
                .find(|(n, _)| *n == name)
                .and_then(|(_, v)| v.clone())
        })
    }

    /// Whether the TLS listener is asked for on an interface other machines can reach.
    pub fn exposes_network(&self) -> bool {
        let tls = self.normalized();
        tls.port.is_some() && !is_local_bind_host(tls.host.as_deref().unwrap_or(TLS_HOST_DEFAULT))
    }

    /// `Ok(None)` when no `VELD_TLS_*` variable is set. Setting any of them asks for the
    /// listener, which then needs all of cert, key and port; the error names the missing
    /// ones. The port may not be the plain listener's on an overlapping host. Values are
    /// trimmed and a blank one is unset, wherever they came from.
    pub fn validate(
        &self,
        plain_host: &str,
        plain_port: u16,
    ) -> anyhow::Result<Option<TlsListenerConfig>> {
        let tls = self.normalized();
        let vars = tls.vars();
        let given: Vec<&str> = vars
            .iter()
            .filter(|(_, v)| v.is_some())
            .map(|(n, _)| *n)
            .collect();
        if given.is_empty() {
            return Ok(None);
        }
        let (cert, key, port) = match (&tls.cert, &tls.key, &tls.port) {
            (Some(cert), Some(key), Some(port)) => (cert, key, port),
            _ => {
                let missing: Vec<&str> = vars[..3]
                    .iter()
                    .filter(|(_, v)| v.is_none())
                    .map(|(n, _)| *n)
                    .collect();
                anyhow::bail!(
                    "the TLS listener is half configured: {} set, {} missing. Set all of \
                     VELD_TLS_CERT, VELD_TLS_KEY and VELD_TLS_PORT, or none of the VELD_TLS_* \
                     variables",
                    given.join(", "),
                    missing.join(", ")
                );
            }
        };
        let port: u16 = port
            .parse()
            .map_err(|_| anyhow::anyhow!("VELD_TLS_PORT='{port}' is not a port number"))?;
        let host = tls.host.as_deref().unwrap_or(TLS_HOST_DEFAULT);
        let ip = bind_ip(host).ok_or_else(|| {
            anyhow::anyhow!("VELD_TLS_HOST='{host}' is not an IP address or localhost")
        })?;
        let plain_ip = bind_ip(plain_host).unwrap_or(std::net::Ipv4Addr::LOCALHOST.into());
        let overlapping = ip == plain_ip || ip.is_unspecified() || plain_ip.is_unspecified();
        if port != 0 && port == plain_port && overlapping {
            anyhow::bail!(
                "VELD_TLS_PORT={port} is the plain listener's port on an overlapping host \
                 (VELD_HOST={plain_host}, VELD_TLS_HOST={host}); give the TLS listener its own port"
            );
        }
        Ok(Some(TlsListenerConfig {
            addr: std::net::SocketAddr::new(ip, port),
            cert: cert.clone(),
            key: key.clone(),
            client_ca: tls.client_ca.clone(),
        }))
    }
}

/// The address a bind host names: an IP literal (IPv6 with or without brackets) or
/// `localhost`.
fn bind_ip(host: &str) -> Option<std::net::IpAddr> {
    if host.eq_ignore_ascii_case("localhost") {
        return Some(std::net::Ipv4Addr::LOCALHOST.into());
    }
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .ok()
}

/// Requests per second per client when nothing sets `VELD_RATE_LIMIT` and the
/// local-development default does not switch limiting off.
pub const RATE_LIMIT_DEFAULT: u64 = 4000;

/// Whether the local-development default switches rate limiting off: outside
/// production, with the plain listener on loopback, no TLS listener asked for on a
/// reachable interface, and no limit asked for (`VELD_RATE_LIMIT` unset and the limit
/// at [`RATE_LIMIT_DEFAULT`]).
pub fn local_dev_disables_rate_limit(
    production: bool,
    host: &str,
    tls: &TlsSettings,
    rate_limit_env_set: bool,
    rate_limit: u64,
) -> bool {
    !production
        && is_local_bind_host(host)
        && !tls.exposes_network()
        && !rate_limit_env_set
        && rate_limit == RATE_LIMIT_DEFAULT
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
    let mut names: Vec<String> = env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| k.starts_with("VELD_"))
        .collect();
    names.sort();
    let _ = VELD_ENV_NAMES_AT_START.set(names);

    load_config_path_into_env(&config_file_path());
}

fn load_config_path_into_env(path: &std::path::Path) {
    let raw = match std::fs::read_to_string(path) {
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
        ("tls_cert", "VELD_TLS_CERT"),
        ("tls_key", "VELD_TLS_KEY"),
        ("tls_port", "VELD_TLS_PORT"),
        ("tls_host", "VELD_TLS_HOST"),
        ("tls_client_ca", "VELD_TLS_CLIENT_CA"),
        ("storage", "VELD_MEMORY_PATH"),
        ("storage_backend", "VELD_STORAGE_BACKEND"),
        ("mcp", "VELD_MCP_ENABLED"),
        ("auto_compress", "VELD_AUTO_COMPRESS"),
        ("graph_mirror_sync_secs", "VELD_GRAPH_MIRROR_SYNC_SECS"),
        ("bm25_commit_interval_ms", "VELD_BM25_COMMIT_INTERVAL_MS"),
        ("gap_worker_nice", "VELD_GAP_WORKER_NICE"),
        ("gap_worker_chunk_rows", "VELD_GAP_WORKER_CHUNK_ROWS"),
        ("gap_worker_cpu_pct", "VELD_GAP_WORKER_CPU_PCT"),
        ("intent_log_segment_mb", "VELD_INTENT_LOG_SEGMENT_MB"),
        (
            "journal_forget_purge_days",
            "VELD_JOURNAL_FORGET_PURGE_DAYS",
        ),
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
        eprintln!(
            "[veld] Loaded {} from {}",
            applied.join(", "),
            path.display()
        );
    }
}

static VELD_ENV_NAMES_AT_START: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// The `VELD_*` variables this process was started with, by name, taken before the
/// `config.toml` bridge and before anything else set a variable on the process.
/// None when [`load_config_file_into_env`] has not run (library use, tests). A
/// relauncher carries these; variables the process set on itself it sets again.
pub fn veld_env_names_at_start() -> Option<&'static [String]> {
    VELD_ENV_NAMES_AT_START.get().map(Vec::as_slice)
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

    /// The optional TLS listener beside the plain one, as configured;
    /// [`ServerConfig::tls_listener`] validates it.
    pub tls: TlsSettings,

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

    /// Request timeout for `/api/gap/*` in seconds (default: 300). Structural gap
    /// detection on a large graph outlasts the general timeout.
    pub gap_request_timeout_secs: u64,

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

    /// Derived tags (NER, YAKE, ingest hints, cold start) a stored record keeps beside the
    /// writer's own; default `constants::DERIVED_TAG_BUDGET` (50, which cuts nothing), env
    /// `VELD_DERIVED_TAG_BUDGET` (1–50) to lower it on a store that measures the effect.
    pub derived_tag_budget: usize,

    /// Log-periodic fractal decay scales in days (default: [7.0, 30.0, 365.0])
    /// Controls resonance frequencies in the power-law decay function.
    /// Weekly/monthly/yearly rhythms by default; override via VELD_LOG_PERIODIC_SCALES
    /// env var (comma-separated, e.g. "14.0,60.0,365.0" for biweekly/bimonthly/yearly)
    pub log_periodic_scales: Vec<f64>,

    /// Whether the maintenance cycle compresses old memories (default: true).
    /// Env `VELD_AUTO_COMPRESS`, config.toml key `auto_compress`.
    pub auto_compress: bool,

    /// Seconds a rebuilt graph mirror (SQLite) is served before a gap request
    /// rebuilds it again (default: 600, minimum: 30).
    /// Env `VELD_GRAPH_MIRROR_SYNC_SECS`, config.toml key `graph_mirror_sync_secs`.
    pub graph_mirror_sync_secs: u64,

    /// Milliseconds between background commits of a tenant's BM25 index
    /// (default: 5000, maximum: 60000). 0 commits on the write path instead, on
    /// the first write 200 ms after the last commit or every 100th write.
    /// Env `VELD_BM25_COMMIT_INTERVAL_MS`, config.toml key `bm25_commit_interval_ms`.
    pub bm25_commit_interval_ms: u64,

    /// Nice value of the gap analysis worker thread, 0 to 19 (default: 10).
    /// Env `VELD_GAP_WORKER_NICE`, config.toml key `gap_worker_nice`.
    pub gap_worker_nice: i32,

    /// Rows of work the gap analysis worker does between pauses, and rows a
    /// mirror sync reads per hold of the graph lock (default: 4096, minimum: 64).
    /// Env `VELD_GAP_WORKER_CHUNK_ROWS`, config.toml key `gap_worker_chunk_rows`.
    pub gap_worker_chunk_rows: usize,

    /// Share of one core the gap analysis worker may take, 1 to 100 (default: 50).
    /// Env `VELD_GAP_WORKER_CPU_PCT`, config.toml key `gap_worker_cpu_pct`.
    pub gap_worker_cpu_pct: u8,

    /// Size, in MiB, at which a tenant's intent log seals its head file into a
    /// segment and starts a new one (default: 256; 0 never rotates). A log that
    /// has rotated needs a build that reads segments.
    /// Env `VELD_INTENT_LOG_SEGMENT_MB`, config.toml key `intent_log_segment_mb`.
    pub intent_log_segment_mb: u64,

    /// Days after which a forgotten memory's frames, its `Forget` included, are
    /// erased from the tenant's sealed journal segments (default: 0, never).
    /// Must outlast any replication follower's time behind this server.
    /// Env `VELD_JOURNAL_FORGET_PURGE_DAYS`, config.toml key `journal_forget_purge_days`.
    pub journal_forget_purge_days: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3030,
            tls: TlsSettings::default(),
            storage_path: default_storage_path(),
            requested_storage_backend: default_requested_storage_backend(),
            effective_storage_backend: effective_storage_backend_for_current_build(
                default_requested_storage_backend(),
            ),
            max_users_in_memory: 1000,
            audit_max_entries_per_user: 10_000,
            audit_rotation_check_interval: 100,
            audit_retention_days: 30,
            rate_limit_per_second: RATE_LIMIT_DEFAULT,
            rate_limit_burst: 8000,
            max_concurrent_requests: 200,
            request_timeout_secs: 60,
            gap_request_timeout_secs: 300,
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
            derived_tag_budget: crate::constants::DERIVED_TAG_BUDGET,
            log_periodic_scales: vec![7.0, 30.0, 365.0], // Weekly, monthly, yearly resonance
            auto_compress: true,
            graph_mirror_sync_secs: crate::memory::slow_store::MIRROR_SYNC_TTL.as_secs(),
            bm25_commit_interval_ms:
                crate::memory::bm25_projection::DEFAULT_BACKGROUND_COMMIT_INTERVAL.as_millis()
                    as u64,
            gap_worker_nice: GapWorkerConfig::default().nice,
            gap_worker_chunk_rows: GapWorkerConfig::default().chunk_rows,
            gap_worker_cpu_pct: GapWorkerConfig::default().cpu_pct,
            intent_log_segment_mb: INTENT_LOG_SEGMENT_MB_DEFAULT,
            journal_forget_purge_days: 0,
        }
    }
}

/// Default [`ServerConfig::intent_log_segment_mb`].
pub const INTENT_LOG_SEGMENT_MB_DEFAULT: u64 = 256;

/// Largest [`ServerConfig::bm25_commit_interval_ms`].
pub const BM25_COMMIT_INTERVAL_MS_MAX: u64 = 60_000;

/// Smallest [`ServerConfig::gap_worker_chunk_rows`].
pub const GAP_WORKER_CHUNK_ROWS_MIN: usize = 64;

/// `raw`, the value of `name`, parsed and clamped to `min..=max`; `default`
/// when unset or not a number.
fn env_clamped<T>(name: &str, raw: Result<String, env::VarError>, default: T, min: T, max: T) -> T
where
    T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display,
{
    let Ok(raw) = raw else {
        return default;
    };
    match raw.trim().parse::<T>() {
        Ok(n) if n < min || n > max => {
            let clamped = if n < min { min } else { max };
            tracing::warn!("{name}={n} is outside {min}..={max}, clamped to {clamped}");
            clamped
        }
        Ok(n) => n,
        Err(_) => {
            tracing::warn!("{name}={raw} is not a number, using default {default}");
            default
        }
    }
}

impl ServerConfig {
    /// The graph-mirror TTL as a `Duration`, for `SlowStore::begin_sync`.
    pub fn graph_mirror_sync_ttl(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.graph_mirror_sync_secs)
    }

    /// How each tenant's BM25 projection commits its index.
    pub fn bm25_commit_mode(&self) -> crate::memory::bm25_projection::CommitMode {
        crate::memory::bm25_projection::CommitMode::from_interval_ms(self.bm25_commit_interval_ms)
    }

    /// How the gap analysis worker thread runs.
    pub fn gap_worker_config(&self) -> GapWorkerConfig {
        GapWorkerConfig {
            nice: self.gap_worker_nice,
            chunk_rows: self.gap_worker_chunk_rows,
            cpu_pct: self.gap_worker_cpu_pct,
        }
    }

    /// How gap paths sync a tenant's graph mirror.
    pub fn graph_mirror_options(&self) -> crate::memory::graph_mirror::MirrorSyncOptions {
        crate::memory::graph_mirror::MirrorSyncOptions {
            ttl: self.graph_mirror_sync_ttl(),
            chunk_rows: self.gap_worker_chunk_rows,
        }
    }

    /// How long a `Forget` stays in the journal before its memory's frames are
    /// erased; `None` keeps them.
    pub fn journal_forget_purge_age(&self) -> Option<std::time::Duration> {
        (self.journal_forget_purge_days > 0).then(|| {
            std::time::Duration::from_secs(self.journal_forget_purge_days.saturating_mul(86_400))
        })
    }

    /// The intent-log segment size in bytes, for `IntentLog::set_segment_bytes`.
    pub fn intent_log_segment_bytes(&self) -> Option<u64> {
        (self.intent_log_segment_mb > 0)
            .then(|| self.intent_log_segment_mb.saturating_mul(1024 * 1024))
    }

    /// The TLS listener to start, `None` when it is off, or why the settings refuse.
    pub fn tls_listener(&self) -> anyhow::Result<Option<TlsListenerConfig>> {
        self.tls.validate(&self.host, self.port)
    }

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

        config.tls = TlsSettings::from_lookup(|name| env::var(name).ok());

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
        // A TLS listener on a reachable interface makes the server not local.
        if local_dev_disables_rate_limit(
            config.is_production,
            &config.host,
            &config.tls,
            rate_limit_explicit,
            config.rate_limit_per_second,
        ) {
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
        if let Ok(val) = env::var("VELD_GAP_TIMEOUT") {
            if let Ok(n) = val.parse::<u64>() {
                config.gap_request_timeout_secs = n.max(1);
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

        if let Ok(val) = env::var("VELD_AUTO_COMPRESS") {
            match val.trim().to_ascii_lowercase().as_str() {
                "true" | "1" => config.auto_compress = true,
                "false" | "0" => config.auto_compress = false,
                _ => tracing::warn!(
                    "VELD_AUTO_COMPRESS={} is not a boolean, using default {}",
                    val,
                    config.auto_compress
                ),
            }
        }

        if let Ok(val) = env::var("VELD_GRAPH_MIRROR_SYNC_SECS") {
            let min = crate::memory::slow_store::MIRROR_SYNC_TTL_MIN.as_secs();
            match val.trim().parse::<u64>() {
                Ok(n) => {
                    config.graph_mirror_sync_secs = n.max(min);
                    if n < min {
                        tracing::warn!(
                            "VELD_GRAPH_MIRROR_SYNC_SECS={} is below minimum, clamped to {}",
                            n,
                            min
                        );
                    }
                }
                Err(_) => tracing::warn!(
                    "VELD_GRAPH_MIRROR_SYNC_SECS={} is not a number, using default {}",
                    val,
                    config.graph_mirror_sync_secs
                ),
            }
        }

        config.bm25_commit_interval_ms = env_clamped(
            "VELD_BM25_COMMIT_INTERVAL_MS",
            env::var("VELD_BM25_COMMIT_INTERVAL_MS"),
            config.bm25_commit_interval_ms,
            0,
            BM25_COMMIT_INTERVAL_MS_MAX,
        );
        config.gap_worker_nice = env_clamped(
            "VELD_GAP_WORKER_NICE",
            env::var("VELD_GAP_WORKER_NICE"),
            config.gap_worker_nice,
            0,
            19,
        );
        config.gap_worker_chunk_rows = env_clamped(
            "VELD_GAP_WORKER_CHUNK_ROWS",
            env::var("VELD_GAP_WORKER_CHUNK_ROWS"),
            config.gap_worker_chunk_rows,
            GAP_WORKER_CHUNK_ROWS_MIN,
            usize::MAX,
        );
        config.gap_worker_cpu_pct = env_clamped(
            "VELD_GAP_WORKER_CPU_PCT",
            env::var("VELD_GAP_WORKER_CPU_PCT"),
            config.gap_worker_cpu_pct,
            1,
            100,
        );

        if let Ok(val) = env::var("VELD_JOURNAL_FORGET_PURGE_DAYS") {
            match val.trim().parse::<u64>() {
                Ok(n) => config.journal_forget_purge_days = n,
                Err(_) => tracing::warn!(
                    "VELD_JOURNAL_FORGET_PURGE_DAYS={} is not a number, keeping forgotten frames",
                    val
                ),
            }
        }

        if let Ok(val) = env::var("VELD_INTENT_LOG_SEGMENT_MB") {
            match val.trim().parse::<u64>() {
                Ok(n) => config.intent_log_segment_mb = n,
                Err(_) => tracing::warn!(
                    "VELD_INTENT_LOG_SEGMENT_MB={} is not a number, using default {}",
                    val,
                    config.intent_log_segment_mb
                ),
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

        if let Ok(val) = env::var("VELD_DERIVED_TAG_BUDGET") {
            if let Ok(n) = val.parse::<usize>() {
                let clamped = n.clamp(1, 50);
                if clamped != n {
                    tracing::warn!(
                        "VELD_DERIVED_TAG_BUDGET={} clamped to {} (valid range: 1–50)",
                        n,
                        clamped
                    );
                }
                config.derived_tag_budget = clamped;
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
                self.requested_storage_backend, self.effective_storage_backend
            );
        }
        let tuning = StorageTuning::process();
        info!(
            "   Storage durability: {} (redb cache {}MB per database, RocksDB write buffer {}MB)",
            tuning.durability,
            tuning.redb_cache_bytes / (1024 * 1024),
            tuning.rocksdb_write_buffer_bytes / (1024 * 1024)
        );
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
        info!(
            "   Request timeout: {}s (/api/gap/*: {}s)",
            self.request_timeout_secs, self.gap_request_timeout_secs
        );
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
    println!("  VELD_HOST             - Bind address (default: 127.0.0.1, use 0.0.0.0 for Docker)");
    println!("  VELD_PORT             - Server port (default: 3030)");
    println!("  VELD_TLS_PORT         - Port of an HTTPS listener serving the same API beside the plain one (default: off)");
    println!("  VELD_TLS_CERT         - PEM certificate chain for the HTTPS listener (required with VELD_TLS_PORT)");
    println!("  VELD_TLS_KEY          - PEM private key for it, PKCS#8, RSA or SEC1 (required with VELD_TLS_PORT)");
    println!("                          Cert and key reload when either file changes (checked every 30s) or on SIGHUP");
    println!("  VELD_TLS_HOST         - Bind address of the HTTPS listener (default: 0.0.0.0)");
    println!("  VELD_TLS_CLIENT_CA    - CA a client certificate must chain to on the HTTPS listener (mTLS; default: none)");
    println!("  VELD_MEMORY_PATH     - Storage directory (default: platform data dir, e.g. ~/.local/share/veld/)");
    println!("  VELD_STORAGE_BACKEND  - Storage engine: rocksdb (default, ADR 0007) or redb");
    println!(
        "  VELD_REPLICATE_FROM   - Base URL of a veld instance to follow (live memory replication)"
    );
    println!("  VELD_REPLICATE_ADMIN_KEY - That instance's admin key (its VELD_ADMIN_API_KEY)");
    println!(
        "  VELD_REPLICATE_API_KEY - That instance's API key (default: this server's own first key)"
    );
    println!("  VELD_REPLICATE_INTERVAL_SECS - Poll interval while following (default: 2)");
    println!("  VELD_API_KEYS         - Comma-separated API keys (required in production)");
    println!(
        "  VELD_DEV_API_KEY      - Development API key (required in dev if VELD_API_KEYS not set)"
    );
    println!("  VELD_KEYSTORE_PATH    - Keystore file enabling encryption-at-rest (required for encrypted production)");
    println!("  VELD_MASTER_PASSPHRASE - Passphrase that unseals the keystore (required when VELD_KEYSTORE_PATH is set)");
    println!("  VELD_SLOW_STORE_ENCRYPT_GRAPH / _EMBEDDINGS - Also encrypt the slow-store graph content / embeddings (default off; payload on)");
    println!("  VELD_ALLOW_KEYSTORE_ROLLBACK - Accept a restored older keystore.json once, resetting the rollback sentinel (default off)");
    println!("  VELD_MAX_USERS        - Max users in memory LRU (default: 1000)");
    println!("  VELD_AUTO_COMPRESS    - Compress old memories during maintenance (default: true)");
    println!("  VELD_GRAPH_MIRROR_SYNC_SECS - Seconds a rebuilt graph mirror is served before the next rebuild (default: 600, min: 30)");
    println!("  VELD_GAP_WORKER_NICE  - Nice value of the gap analysis worker thread (default: 10, 0-19)");
    println!("  VELD_GAP_WORKER_CHUNK_ROWS - Rows of gap work between pauses and per graph-lock hold (default: 4096, min: 64)");
    println!("  VELD_GAP_WORKER_CPU_PCT - Share of one core the gap analysis worker may take (default: 50, 1-100)");
    println!("  VELD_RATE_LIMIT       - Requests per second (default: 0 on localhost/dev, otherwise 4000)");
    println!("  VELD_RATE_BURST       - Burst size (default: 8000)");
    println!("  VELD_MAX_CONCURRENT   - Max concurrent requests (default: 200)");
    println!("  VELD_REQUEST_TIMEOUT  - Request timeout in seconds (default: 60)");
    println!("  VELD_GAP_TIMEOUT      - Request timeout for /api/gap/* in seconds (default: 300)");
    println!("  VELD_AUDIT_MAX_ENTRIES    - Max audit entries per user (default: 10000)");
    println!("  VELD_AUDIT_RETENTION_DAYS - Audit log retention days (default: 30)");
    println!();
    println!("Security — secure-by-default overrides:");
    println!("  VELD_ALLOW_UNSIGNED_WEBHOOKS - Allow webhooks when no *_WEBHOOK_SECRET is set (default: false).");
    println!("                                 False = reject unsigned webhooks with 503. True = warn and process.");
    println!("  VELD_PUBLIC_RATE_LIMIT       - Apply rate limiting to non-probe public routes (default: true).");
    println!("                                 Set to false to exempt public routes (probe routes are never limited).");
    println!(
        "  VELD_METRICS_PUBLIC          - Expose /metrics without authentication (default: false)."
    );
    println!("                                 True = /metrics is public (for unauthenticated Prometheus scrapers).");
    println!("  VELD_ADMIN_API_KEY           - API key for /api/admin/* endpoints (separate from VELD_API_KEYS).");
    println!("                                 Required to use admin operations such as rate-limit reset.");
    println!("  VELD_ENFORCE_HTTPS           - Reject insecure HTTP overrides for integration API URLs (default: true).");
    println!("                                 False = honour a non-localhost http:// override (token in cleartext).");
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
    println!("Storage Engine Tuning:");
    println!("  VELD_STORAGE_DURABILITY         - immediate or eventual, for both engines (default: {DEFAULT_STORAGE_DURABILITY}).");
    println!("                                    immediate = every commit synced to disk (redb Immediate, RocksDB synced WAL);");
    println!("                                    eventual = survives a process crash, a power loss can lose the last commits.");
    println!("                                    Supersedes VELD_WRITE_MODE (sync = immediate, async = eventual).");
    println!("  VELD_REDB_CACHE_BYTES           - redb page cache per database file, in bytes (default: {DEFAULT_REDB_CACHE_BYTES})");
    println!(
        "  VELD_ROCKSDB_WRITE_BUFFER_BYTES - RocksDB memtable size per column family, in bytes (default: {})",
        crate::constants::ROCKSDB_MEMORY_WRITE_BUFFER_BYTES
    );
    println!(
        "  VELD_ROCKSDB_BLOCK_CACHE_BYTES  - RocksDB block cache, in bytes: the server's shared cache (default: {})",
        crate::constants::ROCKSDB_SHARED_CACHE_BYTES
    );
    println!("                                    or a standalone store's own (default: {DEFAULT_ROCKSDB_STANDALONE_BLOCK_CACHE_BYTES})");
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
#[derive(Default)]
pub enum SleepTimeProfile {
    /// Off. `enabled=false`, conservative caps preserved.
    #[default]
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
        // requested must EQUAL effective by default — a default that asks for a backend the build
        // cannot serve publishes an unhonoured request beside status:healthy.
        let expected = if cfg!(feature = "rocksdb") || !cfg!(feature = "storage-redb") {
            StorageBackend::RocksDb
        } else {
            StorageBackend::Redb
        };
        assert_eq!(config.requested_storage_backend, expected);
        assert_eq!(config.effective_storage_backend, expected);
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
    fn test_auto_compress_and_mirror_ttl_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::remove_var("VELD_AUTO_COMPRESS");
        env::remove_var("VELD_GRAPH_MIRROR_SYNC_SECS");

        let config = ServerConfig::from_env();
        assert!(config.auto_compress);
        assert_eq!(config.graph_mirror_sync_secs, 600);
        assert_eq!(
            config.graph_mirror_sync_ttl(),
            std::time::Duration::from_secs(600)
        );
    }

    #[test]
    fn test_auto_compress_env_parsing() {
        let _guard = ENV_LOCK.lock().unwrap();
        for (raw, expected) in [("false", false), ("0", false), ("TRUE", true), ("1", true)] {
            env::set_var("VELD_AUTO_COMPRESS", raw);
            assert_eq!(ServerConfig::from_env().auto_compress, expected, "{raw}");
        }
        env::set_var("VELD_AUTO_COMPRESS", "maybe");
        assert!(ServerConfig::from_env().auto_compress);
        env::remove_var("VELD_AUTO_COMPRESS");
    }

    #[test]
    fn test_journal_forget_purge_days_env_parsing() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::remove_var("VELD_JOURNAL_FORGET_PURGE_DAYS");
        assert_eq!(ServerConfig::from_env().journal_forget_purge_age(), None);
        env::set_var("VELD_JOURNAL_FORGET_PURGE_DAYS", "30");
        assert_eq!(
            ServerConfig::from_env().journal_forget_purge_age(),
            Some(std::time::Duration::from_secs(30 * 86_400))
        );
        env::set_var("VELD_JOURNAL_FORGET_PURGE_DAYS", "soon");
        assert_eq!(ServerConfig::from_env().journal_forget_purge_age(), None);
        env::remove_var("VELD_JOURNAL_FORGET_PURGE_DAYS");
    }

    #[test]
    fn test_intent_log_segment_mb_env_parsing() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::remove_var("VELD_INTENT_LOG_SEGMENT_MB");
        assert_eq!(
            ServerConfig::from_env().intent_log_segment_bytes(),
            Some(INTENT_LOG_SEGMENT_MB_DEFAULT * 1024 * 1024)
        );
        env::set_var("VELD_INTENT_LOG_SEGMENT_MB", "64");
        assert_eq!(
            ServerConfig::from_env().intent_log_segment_bytes(),
            Some(64 * 1024 * 1024)
        );
        env::set_var("VELD_INTENT_LOG_SEGMENT_MB", "0");
        assert_eq!(ServerConfig::from_env().intent_log_segment_bytes(), None);
        env::set_var("VELD_INTENT_LOG_SEGMENT_MB", "big");
        assert_eq!(
            ServerConfig::from_env().intent_log_segment_mb,
            INTENT_LOG_SEGMENT_MB_DEFAULT
        );
        env::remove_var("VELD_INTENT_LOG_SEGMENT_MB");
    }

    #[test]
    fn test_graph_mirror_sync_secs_env_parsing_and_clamp() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("VELD_GRAPH_MIRROR_SYNC_SECS", "1200");
        assert_eq!(ServerConfig::from_env().graph_mirror_sync_secs, 1200);
        env::set_var("VELD_GRAPH_MIRROR_SYNC_SECS", "5");
        assert_eq!(ServerConfig::from_env().graph_mirror_sync_secs, 30);
        env::set_var("VELD_GRAPH_MIRROR_SYNC_SECS", "0");
        assert_eq!(ServerConfig::from_env().graph_mirror_sync_secs, 30);
        env::set_var("VELD_GRAPH_MIRROR_SYNC_SECS", "-3");
        assert_eq!(ServerConfig::from_env().graph_mirror_sync_secs, 600);
        env::set_var("VELD_GRAPH_MIRROR_SYNC_SECS", "soon");
        assert_eq!(ServerConfig::from_env().graph_mirror_sync_secs, 600);
        env::remove_var("VELD_GRAPH_MIRROR_SYNC_SECS");
    }

    #[test]
    fn gap_worker_knobs_default_parse_and_clamp() {
        let _guard = ENV_LOCK.lock().unwrap();
        for name in [
            "VELD_GAP_WORKER_NICE",
            "VELD_GAP_WORKER_CHUNK_ROWS",
            "VELD_GAP_WORKER_CPU_PCT",
        ] {
            env::remove_var(name);
        }
        assert_eq!(
            ServerConfig::from_env().gap_worker_config(),
            GapWorkerConfig::default()
        );
        env::set_var("VELD_GAP_WORKER_NICE", "40");
        env::set_var("VELD_GAP_WORKER_CHUNK_ROWS", "8");
        env::set_var("VELD_GAP_WORKER_CPU_PCT", "25");
        let config = ServerConfig::from_env();
        assert_eq!(config.gap_worker_nice, 19);
        assert_eq!(config.gap_worker_chunk_rows, GAP_WORKER_CHUNK_ROWS_MIN);
        assert_eq!(config.gap_worker_cpu_pct, 25);
        env::set_var("VELD_GAP_WORKER_CPU_PCT", "most");
        assert_eq!(ServerConfig::from_env().gap_worker_cpu_pct, 50);
        for name in [
            "VELD_GAP_WORKER_NICE",
            "VELD_GAP_WORKER_CHUNK_ROWS",
            "VELD_GAP_WORKER_CPU_PCT",
        ] {
            env::remove_var(name);
        }
    }

    #[test]
    fn test_config_file_bridges_new_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("config.toml");
        std::fs::write(
            &file,
            "auto_compress = false\ngraph_mirror_sync_secs = 900\n",
        )
        .unwrap();
        env::remove_var("VELD_AUTO_COMPRESS");
        env::remove_var("VELD_GRAPH_MIRROR_SYNC_SECS");

        load_config_path_into_env(&file);
        let config = ServerConfig::from_env();
        assert!(!config.auto_compress);
        assert_eq!(config.graph_mirror_sync_secs, 900);

        // An explicit environment variable outranks the file.
        env::set_var("VELD_GRAPH_MIRROR_SYNC_SECS", "45");
        load_config_path_into_env(&file);
        assert_eq!(ServerConfig::from_env().graph_mirror_sync_secs, 45);

        env::remove_var("VELD_AUTO_COMPRESS");
        env::remove_var("VELD_GRAPH_MIRROR_SYNC_SECS");
    }

    #[test]
    fn test_local_dev_disables_rate_limit_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::remove_var("VELD_RATE_LIMIT");
        env::remove_var("VELD_HOST");
        env::remove_var("VELD_ENV");
        for (name, _) in TlsSettings::default().vars() {
            env::remove_var(name);
        }

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

    fn tls(pairs: &[(&str, &str)]) -> TlsSettings {
        let vars: std::collections::HashMap<&str, &str> = pairs.iter().copied().collect();
        TlsSettings::from_lookup(|name| vars.get(name).map(|v| v.to_string()))
    }

    const TLS_COMPLETE: [(&str, &str); 3] = [
        ("VELD_TLS_CERT", "/etc/veld/chain.pem"),
        ("VELD_TLS_KEY", "/etc/veld/key.pem"),
        ("VELD_TLS_PORT", "3443"),
    ];

    #[test]
    fn tls_is_off_when_no_variable_is_set() {
        assert_eq!(tls(&[]).validate("127.0.0.1", 3030).unwrap(), None);
        assert_eq!(
            tls(&[("VELD_TLS_PORT", "  ")])
                .validate("127.0.0.1", 3030)
                .unwrap(),
            None,
            "a blank value is unset"
        );
    }

    #[test]
    fn a_half_configured_tls_listener_names_what_is_missing() {
        let err = tls(&[("VELD_TLS_CERT", "/etc/veld/chain.pem")])
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELD_TLS_CERT set"), "{err}");
        assert!(err.contains("VELD_TLS_KEY, VELD_TLS_PORT missing"), "{err}");

        let err = tls(&TLS_COMPLETE[..2])
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELD_TLS_CERT, VELD_TLS_KEY set"), "{err}");
        assert!(err.contains(", VELD_TLS_PORT missing"), "{err}");

        // A client CA without the listener would guard nothing, so it asks for the listener.
        let err = tls(&[("VELD_TLS_CLIENT_CA", "/etc/veld/ca.pem")])
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("VELD_TLS_CERT, VELD_TLS_KEY, VELD_TLS_PORT missing"),
            "{err}"
        );
    }

    #[test]
    fn a_complete_tls_listener_binds_where_asked() {
        let config = tls(&TLS_COMPLETE)
            .validate("127.0.0.1", 3030)
            .unwrap()
            .expect("all three set");
        assert_eq!(config.addr, "0.0.0.0:3443".parse().unwrap());
        assert_eq!(config.cert, PathBuf::from("/etc/veld/chain.pem"));
        assert_eq!(config.key, PathBuf::from("/etc/veld/key.pem"));
        assert_eq!(config.client_ca, None);

        let mut more = TLS_COMPLETE.to_vec();
        more.extend([
            ("VELD_TLS_HOST", "[::1]"),
            ("VELD_TLS_CLIENT_CA", "/etc/veld/ca.pem"),
        ]);
        let config = tls(&more).validate("127.0.0.1", 3030).unwrap().unwrap();
        assert_eq!(config.addr, "[::1]:3443".parse().unwrap());
        assert_eq!(config.client_ca, Some(PathBuf::from("/etc/veld/ca.pem")));
    }

    #[test]
    fn tls_refuses_a_port_or_host_it_cannot_bind() {
        let mut bad_port = TLS_COMPLETE.to_vec();
        bad_port[2] = ("VELD_TLS_PORT", "https");
        let err = tls(&bad_port)
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELD_TLS_PORT='https'"), "{err}");

        let mut bad_host = TLS_COMPLETE.to_vec();
        bad_host.push(("VELD_TLS_HOST", "veld.example.com"));
        let err = tls(&bad_host)
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELD_TLS_HOST='veld.example.com'"), "{err}");
    }

    #[test]
    fn a_blank_or_padded_tls_port_as_the_command_line_gives_it_is_trimmed() {
        let given = |port: &str| TlsSettings {
            cert: Some(PathBuf::from("/etc/veld/chain.pem")),
            key: Some(PathBuf::from("/etc/veld/key.pem")),
            port: Some(port.to_string()),
            ..TlsSettings::default()
        };

        let config = given(" 3443 ")
            .validate("127.0.0.1", 3030)
            .unwrap()
            .expect("a space-padded port is a port");
        assert_eq!(config.addr, "0.0.0.0:3443".parse().unwrap());

        for blank in ["", "   "] {
            let err = given(blank)
                .validate("127.0.0.1", 3030)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("VELD_TLS_CERT, VELD_TLS_KEY set, VELD_TLS_PORT missing"),
                "{blank:?}: {err}"
            );
        }

        let err = given(" https ")
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELD_TLS_PORT='https'"), "{err}");

        let blank_alone = TlsSettings {
            port: Some(" ".to_string()),
            ..TlsSettings::default()
        };
        assert_eq!(blank_alone.validate("127.0.0.1", 3030).unwrap(), None);
        assert!(!blank_alone.exposes_network());
    }

    #[test]
    fn tls_refuses_the_plain_port_on_an_overlapping_host() {
        let mut same_port = TLS_COMPLETE.to_vec();
        same_port[2] = ("VELD_TLS_PORT", "3030");
        let err = tls(&same_port)
            .validate("127.0.0.1", 3030)
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELD_TLS_PORT=3030"), "{err}");
        assert!(tls(&same_port).validate("0.0.0.0", 3030).is_err());

        let mut localhost = same_port.clone();
        localhost.push(("VELD_TLS_HOST", "localhost"));
        assert!(tls(&localhost).validate("127.0.0.1", 3030).is_err());

        let mut disjoint = same_port.clone();
        disjoint.push(("VELD_TLS_HOST", "::1"));
        assert!(tls(&disjoint)
            .validate("127.0.0.1", 3030)
            .unwrap()
            .is_some());
        assert!(tls(&same_port)
            .validate("127.0.0.1", 3031)
            .unwrap()
            .is_some());
    }

    #[test]
    fn the_local_dev_default_disables_rate_limiting_only_when_nothing_is_reachable() {
        let no_tls = TlsSettings::default();
        let tls_on = |host: Option<&str>| {
            let mut pairs = TLS_COMPLETE.to_vec();
            pairs.extend(host.map(|h| ("VELD_TLS_HOST", h)));
            tls(&pairs)
        };
        let dev = |host, tls: &TlsSettings| {
            local_dev_disables_rate_limit(false, host, tls, false, RATE_LIMIT_DEFAULT)
        };

        assert!(dev("127.0.0.1", &no_tls));
        assert!(dev("::1", &no_tls));
        assert!(
            dev("localhost", &tls_on(Some("127.0.0.1"))),
            "a TLS listener on loopback leaves the server local"
        );
        assert!(
            !dev("127.0.0.1", &tls_on(None)),
            "TLS on a non-loopback host keeps it on: the default TLS host is 0.0.0.0"
        );
        assert!(
            !dev("127.0.0.1", &tls_on(Some("10.0.0.2"))),
            "TLS on a non-loopback host keeps it on"
        );
        assert!(!dev("0.0.0.0", &no_tls));

        let local = "127.0.0.1";
        assert!(!local_dev_disables_rate_limit(
            true,
            local,
            &no_tls,
            false,
            RATE_LIMIT_DEFAULT
        ));
        assert!(!local_dev_disables_rate_limit(
            false,
            local,
            &no_tls,
            true,
            RATE_LIMIT_DEFAULT
        ));
        assert!(!local_dev_disables_rate_limit(
            false, local, &no_tls, false, 100
        ));
    }

    #[test]
    fn a_tls_listener_on_the_network_keeps_rate_limiting_on() {
        let _guard = ENV_LOCK.lock().unwrap();
        for name in ["VELD_RATE_LIMIT", "VELD_HOST", "VELD_ENV", "VELD_TLS_HOST"] {
            env::remove_var(name);
        }
        for (name, value) in TLS_COMPLETE {
            env::set_var(name, value);
        }

        assert_eq!(ServerConfig::from_env().rate_limit_per_second, 4000);
        env::set_var("VELD_TLS_HOST", "127.0.0.1");
        assert_eq!(ServerConfig::from_env().rate_limit_per_second, 0);

        for (name, _) in TlsSettings::default().vars() {
            env::remove_var(name);
        }
    }

    #[test]
    fn a_padded_tls_port_in_config_toml_is_trimmed_like_the_variable() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("config.toml");
        std::fs::write(
            &file,
            "tls_cert = \"/etc/veld/chain.pem\"\ntls_key = \"/etc/veld/key.pem\"\n\
             tls_port = \" 3443 \"\n",
        )
        .unwrap();
        for (name, _) in TlsSettings::default().vars() {
            env::remove_var(name);
        }

        load_config_path_into_env(&file);
        assert_eq!(env::var("VELD_TLS_PORT").unwrap(), " 3443 ");
        let listener = ServerConfig::from_env()
            .tls_listener()
            .unwrap()
            .expect("bridged from the file");
        assert_eq!(listener.addr.port(), 3443);

        for (name, _) in TlsSettings::default().vars() {
            env::remove_var(name);
        }
    }

    #[test]
    fn test_config_file_bridges_the_tls_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("config.toml");
        std::fs::write(
            &file,
            "tls_cert = \"/etc/veld/chain.pem\"\ntls_key = \"/etc/veld/key.pem\"\n\
             tls_port = 3443\ntls_host = \"10.0.0.2\"\ntls_client_ca = \"/etc/veld/ca.pem\"\n",
        )
        .unwrap();
        for (name, _) in TlsSettings::default().vars() {
            env::remove_var(name);
        }

        load_config_path_into_env(&file);
        let config = ServerConfig::from_env();
        let listener = config
            .tls_listener()
            .unwrap()
            .expect("bridged from the file");
        assert_eq!(listener.addr, "10.0.0.2:3443".parse().unwrap());
        assert_eq!(listener.cert, PathBuf::from("/etc/veld/chain.pem"));
        assert_eq!(listener.key, PathBuf::from("/etc/veld/key.pem"));
        assert_eq!(listener.client_ca, Some(PathBuf::from("/etc/veld/ca.pem")));

        for (name, _) in TlsSettings::default().vars() {
            env::remove_var(name);
        }
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
        assert_eq!(
            cors.allowed_origins,
            vec!["https://app.example.com".to_string()]
        );

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

    fn tuning_from(vars: &[(&str, &str)]) -> StorageTuning {
        let vars: std::collections::HashMap<String, String> = vars
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        StorageTuning::from_lookup(|name| vars.get(name).cloned())
    }

    #[test]
    fn storage_tuning_without_variables_is_the_default() {
        let tuning = tuning_from(&[]);
        assert_eq!(tuning, StorageTuning::default());
        assert_eq!(tuning.durability, StorageDurability::Eventual);
        assert!(!tuning.rocksdb_sync_writes());
        assert_eq!(tuning.redb_cache_bytes, DEFAULT_REDB_CACHE_BYTES);
        assert_eq!(
            tuning.rocksdb_write_buffer_bytes,
            crate::constants::ROCKSDB_MEMORY_WRITE_BUFFER_BYTES
        );
        assert_eq!(
            tuning.rocksdb_standalone_block_cache_bytes(),
            DEFAULT_ROCKSDB_STANDALONE_BLOCK_CACHE_BYTES
        );
        assert_eq!(
            tuning.rocksdb_shared_block_cache_bytes(),
            crate::constants::ROCKSDB_SHARED_CACHE_BYTES
        );
    }

    #[test]
    fn storage_durability_parses_each_class() {
        let immediate = tuning_from(&[("VELD_STORAGE_DURABILITY", "immediate")]);
        assert_eq!(immediate.durability, StorageDurability::Immediate);
        assert!(immediate.rocksdb_sync_writes());

        let eventual = tuning_from(&[("VELD_STORAGE_DURABILITY", " Eventual ")]);
        assert_eq!(eventual.durability, StorageDurability::Eventual);
        assert!(!eventual.rocksdb_sync_writes());
    }

    #[test]
    fn an_invalid_storage_durability_keeps_the_default() {
        for raw in ["fsync", "", "sync", "1"] {
            assert_eq!(
                tuning_from(&[("VELD_STORAGE_DURABILITY", raw)]).durability,
                DEFAULT_STORAGE_DURABILITY,
                "VELD_STORAGE_DURABILITY={raw:?}"
            );
        }
    }

    #[test]
    fn the_legacy_write_mode_selects_durability_only_when_storage_durability_is_unset() {
        assert_eq!(
            tuning_from(&[("VELD_WRITE_MODE", "sync")]).durability,
            StorageDurability::Immediate
        );
        assert_eq!(
            tuning_from(&[("VELD_WRITE_MODE", "ASYNC")]).durability,
            StorageDurability::Eventual
        );
        assert_eq!(
            tuning_from(&[("VELD_WRITE_MODE", "fast")]).durability,
            DEFAULT_STORAGE_DURABILITY
        );
        assert_eq!(
            tuning_from(&[
                ("VELD_WRITE_MODE", "sync"),
                ("VELD_STORAGE_DURABILITY", "eventual"),
            ])
            .durability,
            StorageDurability::Eventual,
            "VELD_STORAGE_DURABILITY must win over the legacy variable"
        );
    }

    #[test]
    fn byte_count_variables_parse_and_fall_back_on_invalid_values() {
        let tuned = tuning_from(&[
            ("VELD_REDB_CACHE_BYTES", "67108864"),
            ("VELD_ROCKSDB_WRITE_BUFFER_BYTES", " 4194304 "),
            ("VELD_ROCKSDB_BLOCK_CACHE_BYTES", "33554432"),
        ]);
        assert_eq!(tuned.redb_cache_bytes, 64 * 1024 * 1024);
        assert_eq!(tuned.rocksdb_write_buffer_bytes, 4 * 1024 * 1024);
        assert_eq!(tuned.rocksdb_block_cache_bytes, Some(32 * 1024 * 1024));
        assert_eq!(
            tuned.rocksdb_standalone_block_cache_bytes(),
            32 * 1024 * 1024
        );
        assert_eq!(tuned.rocksdb_shared_block_cache_bytes(), 32 * 1024 * 1024);

        let defaults = StorageTuning::default();
        for raw in ["0", "-1", "64MB", "", "1.5"] {
            let tuning = tuning_from(&[
                ("VELD_REDB_CACHE_BYTES", raw),
                ("VELD_ROCKSDB_WRITE_BUFFER_BYTES", raw),
                ("VELD_ROCKSDB_BLOCK_CACHE_BYTES", raw),
            ]);
            assert_eq!(tuning, defaults, "byte count {raw:?} was not rejected");
        }
    }

    #[test]
    fn storage_tuning_from_env_reads_the_process_environment() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::set_var("VELD_STORAGE_DURABILITY", "immediate");
        env::set_var("VELD_REDB_CACHE_BYTES", "1048576");
        env::set_var("VELD_ROCKSDB_WRITE_BUFFER_BYTES", "2097152");
        env::set_var("VELD_ROCKSDB_BLOCK_CACHE_BYTES", "not-a-number");

        let tuning = StorageTuning::from_env();

        env::remove_var("VELD_STORAGE_DURABILITY");
        env::remove_var("VELD_REDB_CACHE_BYTES");
        env::remove_var("VELD_ROCKSDB_WRITE_BUFFER_BYTES");
        env::remove_var("VELD_ROCKSDB_BLOCK_CACHE_BYTES");

        assert_eq!(
            tuning,
            StorageTuning {
                durability: StorageDurability::Immediate,
                redb_cache_bytes: 1024 * 1024,
                rocksdb_write_buffer_bytes: 2 * 1024 * 1024,
                rocksdb_block_cache_bytes: None,
            }
        );
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
