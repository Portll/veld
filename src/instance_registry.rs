//! Instance registry — how a running `veld server` makes itself discoverable.
//!
//! Rather than have clients *guess* which ports a veld might be listening on, each
//! server **declares itself**: after a successful port bind it atomically writes a
//! small JSON file to a shared directory, heartbeats it, and removes it on graceful
//! shutdown. Consumers (the Veld Monitor via `status-core`, the Claude statusline)
//! read that directory to learn the live fleet — no port lists, no probing.
//!
//! ## Why the filesystem
//! It is zero-dependency, cross-platform, works fully offline, needs no broker or
//! daemon, and matches veld's "single binary, no external services" ethos. The
//! registry lives under the **config** dir (`<config>/veld/instances/`), which is
//! shared across instances even when each runs with a different `VELD_MEMORY_PATH`.
//!
//! ## Liveness without pid-probing
//! Cross-platform "is pid N alive?" is awkward, so liveness is decided purely by
//! heartbeat freshness: an entry is live iff `now - last_heartbeat < STALE_TTL`.
//! Servers reap stale entries on startup; **readers never delete** (no reader races).
//!
//! ## Extension seam (multi-host future)
//! [`InstanceDiscovery`] abstracts "list the live fleet". [`FilesystemDiscovery`]
//! ships today; a `ZenohDiscovery` / mDNS provider can be added later and merged
//! into the same consumer code without changes here.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

/// On-disk schema version for [`InstanceInfo`]. Bump on breaking changes.
pub const REGISTRY_SCHEMA_VERSION: u32 = 1;

/// How often a live server refreshes its `last_heartbeat`.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// An entry whose `last_heartbeat` is older than this is considered dead. Set well
/// above [`HEARTBEAT_INTERVAL`] so a momentarily-busy server is never flapped offline.
pub const STALE_TTL: Duration = Duration::from_secs(30);

/// One discoverable veld instance, as written to `<config>/veld/instances/<pid>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// On-disk schema version (see [`REGISTRY_SCHEMA_VERSION`]).
    #[serde(default)]
    pub v: u32,
    /// Stable id for this *run* (pid plus start time), robust to pid reuse.
    pub instance_id: String,
    pub pid: u32,
    /// The host the listener bound to (may be `0.0.0.0` / `::`).
    pub host: String,
    /// A host a local consumer can actually connect to (`0.0.0.0`/`::` → loopback).
    pub connect_host: String,
    pub port: u16,
    /// Ready-to-use base URL built from `connect_host` + `port`.
    pub base_url: String,
    /// Storage directory this instance is serving (distinguishes scratch vs real).
    pub data_path: String,
    /// Crate version of the running binary.
    pub version: String,
    /// Friendly label (`VELD_INSTANCE_LABEL`, else derived from the data path).
    pub label: String,
    pub started_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    /// How this process was started, so a relauncher (the Veld Monitor's "Restart
    /// server") can start it the same way. Absent on entries from older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<LaunchRecipe>,
}

/// The command line and working directory a server was started with. Environment
/// values are never written here (several are secrets); only the names of the
/// `VELD_*` variables the process was started with (not those it set on itself,
/// from `config.toml` or its own arguments), so a relauncher that cannot read the
/// live environment can say which ones the new process will lack.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRecipe {
    pub exe: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub env_names: Vec<String>,
}

fn capture_launch() -> Option<LaunchRecipe> {
    let exe = std::env::current_exe().ok()?;
    let mut env_names: Vec<String> = match crate::config::veld_env_names_at_start() {
        Some(names) => names.to_vec(),
        None => std::env::vars_os()
            .filter_map(|(k, _)| k.into_string().ok())
            .filter(|k| k.starts_with("VELD_"))
            .collect(),
    };
    env_names.sort();
    Some(LaunchRecipe {
        exe: exe.to_string_lossy().to_string(),
        args: std::env::args().skip(1).collect(),
        cwd: std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().to_string()),
        env_names,
    })
}

impl InstanceInfo {
    /// True when the last heartbeat is fresh enough to consider the instance alive.
    pub fn is_live(&self, now: DateTime<Utc>, ttl: Duration) -> bool {
        match now.signed_duration_since(self.last_heartbeat).to_std() {
            Ok(age) => age < ttl,
            // Negative age (heartbeat in the future, e.g. clock skew) → treat as live.
            Err(_) => true,
        }
    }
}

/// The shared directory holding instance files: `<config dir>/veld/instances/`.
///
/// Mirrors [`crate::config::config_file_path`] resolution (same `<config>/veld` root),
/// falling back to `./.veld/instances` when the platform config dir is unknown.
pub fn instances_dir() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join("veld"))
        .unwrap_or_else(|| PathBuf::from(".veld"))
        .join("instances")
}

fn connectable_host(host: &str) -> String {
    match host {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
        other => other.to_string(),
    }
}

fn derive_label(data_path: &Path) -> String {
    if let Ok(label) = std::env::var("VELD_INSTANCE_LABEL") {
        let label = label.trim();
        if !label.is_empty() {
            return label.to_string();
        }
    }
    data_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "veld".to_string())
}

/// Atomically write `info` to its `<pid>.json` (temp file + rename; `rename`
/// replaces an existing destination on both Unix and Windows).
fn write_atomic(dir: &Path, info: &InstanceInfo) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let final_path = dir.join(format!("{}.json", info.pid));
    let tmp_path = dir.join(format!("{}.json.tmp", info.pid));
    let json = serde_json::to_vec_pretty(info)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp_path, &json)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(final_path)
}

/// Read every parseable `*.json` entry in the registry dir. Unparseable or `.tmp`
/// files are skipped (never deleted by readers).
pub fn read_all(dir: &Path) -> Vec<InstanceInfo> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out, // dir absent ⇒ no instances
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(info) = serde_json::from_str::<InstanceInfo>(&text) {
                out.push(info);
            }
        }
    }
    out
}

/// Remove entries staler than [`STALE_TTL`]. Intended to be called by a *server*
/// at startup (writers, not readers, do the reaping). Best-effort; errors ignored.
pub fn reap_stale(dir: &Path, now: DateTime<Utc>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let dead = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<InstanceInfo>(&text) {
                Ok(info) => !info.is_live(now, STALE_TTL),
                Err(_) => false, // unparseable: leave it (could be a torn write elsewhere)
            },
            Err(_) => false,
        };
        if dead {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Provider abstraction for listing the live fleet. Filesystem today; Zenoh/mDNS
/// can be added later and merged by consumers without touching call sites.
pub trait InstanceDiscovery: Send + Sync {
    fn list_live(&self) -> Vec<InstanceInfo>;
}

/// Discovery backed by the shared registry directory.
pub struct FilesystemDiscovery {
    dir: PathBuf,
    ttl: Duration,
}

impl Default for FilesystemDiscovery {
    fn default() -> Self {
        Self {
            dir: instances_dir(),
            ttl: STALE_TTL,
        }
    }
}

impl FilesystemDiscovery {
    pub fn new(dir: PathBuf, ttl: Duration) -> Self {
        Self { dir, ttl }
    }
}

impl InstanceDiscovery for FilesystemDiscovery {
    fn list_live(&self) -> Vec<InstanceInfo> {
        let now = Utc::now();
        let mut live: Vec<InstanceInfo> = read_all(&self.dir)
            .into_iter()
            .filter(|i| i.is_live(now, self.ttl))
            .collect();
        // Deterministic ordering for stable tab order: by port then start time.
        live.sort_by(|a, b| a.port.cmp(&b.port).then(a.started_at.cmp(&b.started_at)));
        live
    }
}

/// RAII handle for this process's registration. Holds the heartbeat task; removing
/// the registration on `deregister()` (or drop) makes the instance disappear promptly.
pub struct RegistryGuard {
    path: PathBuf,
    stop: Arc<Notify>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl RegistryGuard {
    /// Stop heartbeating and remove this instance's file. Call on graceful shutdown.
    pub async fn deregister(mut self) {
        self.stop.notify_one();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        // Best-effort if `deregister` wasn't awaited: signal the task and unlink.
        self.stop.notify_one();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Register this running server in the shared directory and start heartbeating.
///
/// Reaps stale peers first, writes our `<pid>.json`, then spawns a task that bumps
/// `last_heartbeat` every [`HEARTBEAT_INTERVAL`] until [`RegistryGuard::deregister`]
/// (or drop). On any I/O failure this logs and returns `None` — discovery is a
/// convenience and must never take the server down.
pub fn register(host: &str, port: u16, data_path: &Path) -> Option<RegistryGuard> {
    let dir = instances_dir();
    let now = Utc::now();
    reap_stale(&dir, now);

    let pid = std::process::id();
    let connect_host = connectable_host(host);
    let info = InstanceInfo {
        v: REGISTRY_SCHEMA_VERSION,
        instance_id: format!("{}-{}", pid, now.timestamp_millis()),
        pid,
        host: host.to_string(),
        connect_host: connect_host.clone(),
        port,
        base_url: format!("http://{}:{}", connect_host, port),
        data_path: data_path.to_string_lossy().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        label: derive_label(data_path),
        started_at: now,
        last_heartbeat: now,
        launch: capture_launch(),
    };

    let path = match write_atomic(&dir, &info) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("instance registry: could not write registration: {e}");
            return None;
        }
    };
    tracing::info!(
        "instance registry: registered {} at {}",
        info.instance_id,
        path.display()
    );

    let stop = Arc::new(Notify::new());
    let task = {
        let stop = Arc::clone(&stop);
        let dir = dir.clone();
        let mut info = info;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = stop.notified() => break,
                    _ = tokio::time::sleep(HEARTBEAT_INTERVAL) => {
                        info.last_heartbeat = Utc::now();
                        if let Err(e) = write_atomic(&dir, &info) {
                            tracing::debug!("instance registry: heartbeat write failed: {e}");
                        }
                    }
                }
            }
        })
    };

    Some(RegistryGuard {
        path,
        stop,
        task: Some(task),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(pid: u32, port: u16, heartbeat: DateTime<Utc>) -> InstanceInfo {
        InstanceInfo {
            v: REGISTRY_SCHEMA_VERSION,
            instance_id: format!("{pid}-x"),
            pid,
            host: "127.0.0.1".to_string(),
            connect_host: "127.0.0.1".to_string(),
            port,
            base_url: format!("http://127.0.0.1:{port}"),
            data_path: "/tmp/veld".to_string(),
            version: "test".to_string(),
            label: "veld".to_string(),
            started_at: heartbeat,
            last_heartbeat: heartbeat,
            launch: None,
        }
    }

    #[test]
    fn entries_without_a_launch_recipe_still_deserialize() {
        let mut info = sample(7, 3030, Utc::now());
        info.launch = Some(LaunchRecipe {
            exe: "/usr/local/bin/veld".into(),
            args: vec!["server".into(), "--port".into(), "3030".into()],
            cwd: Some("/home/me".into()),
            env_names: vec!["VELD_PORT".into()],
        });
        let json = serde_json::to_string(&info).unwrap();
        let back: InstanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.launch, info.launch);

        let legacy: InstanceInfo =
            serde_json::from_str(&serde_json::to_string(&sample(8, 3031, Utc::now())).unwrap())
                .unwrap();
        assert_eq!(legacy.launch, None);
    }

    #[test]
    fn launch_recipe_names_veld_variables_without_their_values() {
        std::env::set_var("VELD_REGISTRY_TEST_SECRET", "hunter2");
        let recipe = capture_launch().expect("current_exe is known in tests");
        std::env::remove_var("VELD_REGISTRY_TEST_SECRET");
        assert!(recipe.env_names.iter().any(|n| n == "VELD_REGISTRY_TEST_SECRET"));
        let json = serde_json::to_string(&recipe).unwrap();
        assert!(!json.contains("hunter2"));
        assert!(!recipe.exe.is_empty());
    }

    #[test]
    fn liveness_is_decided_by_heartbeat_freshness() {
        let now = Utc::now();
        let fresh = sample(1, 3030, now);
        let stale = sample(2, 3031, now - chrono::Duration::seconds(120));
        assert!(fresh.is_live(now, STALE_TTL));
        assert!(!stale.is_live(now, STALE_TTL));
    }

    #[test]
    fn connectable_host_maps_wildcards_to_loopback() {
        assert_eq!(connectable_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(connectable_host("::"), "127.0.0.1");
        assert_eq!(connectable_host("192.168.1.5"), "192.168.1.5");
    }

    #[test]
    fn read_all_and_reap_roundtrip() {
        let dir = std::env::temp_dir().join(format!("veld-reg-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = Utc::now();
        write_atomic(&dir, &sample(101, 3030, now)).unwrap();
        write_atomic(
            &dir,
            &sample(102, 3031, now - chrono::Duration::seconds(300)),
        )
        .unwrap();

        assert_eq!(read_all(&dir).len(), 2);
        let live: Vec<_> = read_all(&dir)
            .into_iter()
            .filter(|i| i.is_live(now, STALE_TTL))
            .collect();
        assert_eq!(live.len(), 1);

        reap_stale(&dir, now);
        assert_eq!(read_all(&dir).len(), 1, "stale entry should be reaped");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
