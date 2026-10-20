//! Read-side of the veld instance registry.
//!
//! `veld server` writes a heartbeated JSON file per instance under
//! `<config>/veld/instances/` (see `src/instance_registry.rs` in the veld crate).
//! Here we read that directory to learn the live fleet — no port guessing. The
//! JSON shape is the contract; we deserialize only the fields the monitor renders.

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Deserialize;

/// An entry whose heartbeat is older than this is treated as dead. Matches the
/// server's `STALE_TTL` in `instance_registry.rs`.
const STALE_TTL: Duration = Duration::from_secs(30);

/// A discovered veld instance (subset of the server's `InstanceInfo`).
#[derive(Debug, Clone, Deserialize)]
pub struct DiscoveredInstance {
    /// Ready-to-use base URL, e.g. `http://127.0.0.1:3030`.
    pub base_url: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub pid: u32,
    /// The host the listener bound to (`VELD_HOST`), as distinct from `base_url`'s
    /// connectable host.
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub port: u16,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub data_path: String,
    /// When the server process started: its own uptime, unlike a watcher's
    /// first successful probe.
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    pub last_heartbeat: DateTime<Utc>,
    /// How the server was started; absent for entries written by older servers.
    #[serde(default)]
    pub launch: Option<LaunchRecipe>,
}

/// Mirror of the server's `instance_registry::LaunchRecipe`: command line and
/// working directory, plus the names (never values) of its `VELD_*` variables.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LaunchRecipe {
    pub exe: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub env_names: Vec<String>,
}

/// The shared registry directory: `<config>/veld/instances`, sibling of
/// `config.toml` (so it resolves identically across instances regardless of each
/// one's `VELD_MEMORY_PATH`).
pub fn instances_dir() -> PathBuf {
    crate::default_config_path()
        .parent()
        .map(|p| p.join("instances"))
        .unwrap_or_else(|| PathBuf::from("instances"))
}

/// Return every instance with a fresh heartbeat, sorted by port then base_url for
/// a stable tab order. Empty when the registry dir is absent or all entries stale.
pub fn discover_instances() -> Vec<DiscoveredInstance> {
    let now = Utc::now();
    let dir = instances_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<DiscoveredInstance> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(inst) = serde_json::from_str::<DiscoveredInstance>(&text) else {
            continue;
        };
        let fresh = now
            .signed_duration_since(inst.last_heartbeat)
            .to_std()
            .map(|age| age < STALE_TTL)
            .unwrap_or(true); // heartbeat in the future (clock skew) ⇒ treat as live
        if fresh {
            out.push(inst);
        }
    }
    out.sort_by(|a, b| a.port.cmp(&b.port).then_with(|| a.base_url.cmp(&b.base_url)));
    out
}
