//! Tauri application bootstrap: manages a *fleet* of veld status clients (one per
//! discovered instance), registers commands, manages window visibility, and emits
//! live snapshot events for the pinned instance.
//!
//! Multi-instance: the configured server is the "pinned" instance and is always
//! present. A background task reconciles the fleet against the filesystem instance
//! registry (`veld_status_core::discover_instances`) — spawning a poller when a new
//! instance comes online and dropping it (which aborts its tasks) when it goes away.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use veld_status_core::{discover_instances, StatusClient, StatusClientConfig, StatusSnapshot};

use crate::tray;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:3030";
const DEFAULT_DEV_API_KEY: &str = "sk-veld-dev-local-testing-key";
const DEFAULT_USER_ID: &str = "claude-code";
const RECONCILE_INTERVAL: Duration = Duration::from_secs(3);

/// One member of the fleet: a running status client plus its snapshot handle.
struct FleetEntry {
    /// Held so its background tasks live; dropped (→ aborted) on removal.
    _client: StatusClient,
    snapshot: Arc<RwLock<StatusSnapshot>>,
    label: String,
    pinned: bool,
}

/// Tab metadata handed to the frontend.
#[derive(Serialize, Clone)]
struct InstanceMeta {
    base_url: String,
    label: String,
    pinned: bool,
    online: bool,
}

/// App-wide state: connection params + the live fleet.
pub struct MonitorState {
    api_key: String,
    user_id: String,
    request_timeout: Duration,
    /// Normalized base_url of the configured (always-present) instance.
    pinned_base_url: String,
    fleet: Arc<Mutex<HashMap<String, FleetEntry>>>,
}

/// Strip a trailing slash so registry/discovery/config URLs key consistently.
fn norm(base: &str) -> String {
    base.trim_end_matches('/').to_string()
}

/// Short tab label derived from a base URL: ":3030" from http://127.0.0.1:3030.
fn url_short(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    match trimmed.rsplit_once(':') {
        Some((_, port)) if port.chars().all(|c| c.is_ascii_digit()) && !port.is_empty() => {
            format!(":{port}")
        }
        _ => trimmed.to_string(),
    }
}

/// Spawn a status client for `base_url`. Returns its snapshot handle alongside.
/// Must be called within a tokio runtime (StatusClient::spawn spawns tasks).
fn spawn_client(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    timeout: Duration,
) -> Option<(StatusClient, Arc<RwLock<StatusSnapshot>>)> {
    let cfg = StatusClientConfig::new(base_url.to_string(), api_key.to_string(), user_id.to_string())
        .with_refresh_interval(Duration::from_secs(2))
        .with_request_timeout(timeout);
    match StatusClient::spawn(cfg) {
        Ok(client) => {
            let snap = client.snapshot();
            Some((client, snap))
        }
        Err(e) => {
            tracing::warn!("fleet: failed to spawn client for {base_url}: {e}");
            None
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let base_url = std::env::var("VELD_SERVER_URL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
            let api_key = std::env::var("VELD_API_KEY")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .or_else(|| {
                    let path = veld_status_core::default_config_path();
                    veld_status_core::load_api_key_from(&path).ok()
                })
                .unwrap_or_else(|| DEFAULT_DEV_API_KEY.to_string());
            let user_id = std::env::var("VELD_USER_ID")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_USER_ID.to_string());
            let timeout = Duration::from_secs(5);
            let pinned_key = norm(&base_url);

            // Spawn the pinned client (inside a runtime context).
            let (client, snapshot) = tauri::async_runtime::block_on(async {
                spawn_client(&base_url, &api_key, &user_id, timeout)
            })
            .ok_or_else(|| {
                Box::new(std::io::Error::other("failed to spawn pinned status client"))
                    as Box<dyn std::error::Error>
            })?;

            let mut map: HashMap<String, FleetEntry> = HashMap::new();
            map.insert(
                pinned_key.clone(),
                FleetEntry {
                    _client: client,
                    snapshot: Arc::clone(&snapshot),
                    label: url_short(&base_url),
                    pinned: true,
                },
            );
            let fleet = Arc::new(Mutex::new(map));

            app.manage(MonitorState {
                api_key: api_key.clone(),
                user_id: user_id.clone(),
                request_timeout: timeout,
                pinned_base_url: pinned_key.clone(),
                fleet: Arc::clone(&fleet),
            });

            // Tray + live event stream follow the pinned instance.
            tray::build(app, Arc::clone(&snapshot))?;
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(emit_snapshots(handle, snapshot));

            // Reconcile the fleet against the registry in the background.
            tauri::async_runtime::spawn(reconcile_fleet(
                fleet, pinned_key, api_key, user_id, timeout,
            ));

            Ok(())
        })
        .on_window_event(|window, event| {
            // Minimize to tray: the X button HIDES the window (keeping the app +
            // tray icon alive) instead of quitting. Quit is on the tray menu.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            get_snapshot_for,
            list_instances,
            show_dashboard,
            quit_app,
            get_dials_for,
            set_dials_for,
            first_run_check,
            mark_initialized,
            seed_project,
            claude_hook_status
        ])
        .run(tauri::generate_context!())
        .expect("failed to run tauri application");
}

/// Pinned instance's snapshot (back-compat; used by the default tab + fallback).
#[tauri::command]
fn get_snapshot(state: State<'_, MonitorState>) -> StatusSnapshot {
    let fleet = state.fleet.lock();
    fleet
        .get(&state.pinned_base_url)
        .map(|e| e.snapshot.read().clone())
        .unwrap_or_default()
}

/// Snapshot for a specific instance (the active tab). Falls back to the pinned
/// instance, then an empty snapshot, if the base_url isn't in the fleet.
#[tauri::command]
fn get_snapshot_for(state: State<'_, MonitorState>, base_url: String) -> StatusSnapshot {
    let key = norm(&base_url);
    let fleet = state.fleet.lock();
    fleet
        .get(&key)
        .or_else(|| fleet.get(&state.pinned_base_url))
        .map(|e| e.snapshot.read().clone())
        .unwrap_or_default()
}

/// All instances currently in the fleet (pinned first), with live online status.
#[tauri::command]
fn list_instances(state: State<'_, MonitorState>) -> Vec<InstanceMeta> {
    let fleet = state.fleet.lock();
    let mut metas: Vec<InstanceMeta> = fleet
        .values()
        .map(|e| {
            let snap = e.snapshot.read();
            InstanceMeta {
                base_url: snap.base_url.clone(),
                label: e.label.clone(),
                pinned: e.pinned,
                online: snap.server.state.is_reachable(),
            }
        })
        .collect();
    // Pinned first, then by base_url for a stable tab order.
    metas.sort_by(|a, b| b.pinned.cmp(&a.pinned).then_with(|| a.base_url.cmp(&b.base_url)));
    metas
}

#[tauri::command]
fn show_dashboard(app: AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    app.exit(0);
}

/// Read the cognitive dials from a specific instance (`GET /api/dials`).
#[tauri::command]
async fn get_dials_for(
    state: State<'_, MonitorState>,
    base_url: String,
) -> Result<serde_json::Value, String> {
    let (key, user, timeout) = (
        state.api_key.clone(),
        state.user_id.clone(),
        state.request_timeout,
    );
    StatusClient::get_dials(&base_url, &key, &user, timeout)
        .await
        .map_err(|e| e.to_string())
}

/// Replace the cognitive dials on a specific instance (`PUT /api/dials`).
#[tauri::command]
async fn set_dials_for(
    state: State<'_, MonitorState>,
    base_url: String,
    dials: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let (key, user, timeout) = (
        state.api_key.clone(),
        state.user_id.clone(),
        state.request_timeout,
    );
    StatusClient::set_dials(&base_url, &key, &user, dials, timeout)
        .await
        .map_err(|e| e.to_string())
}

/// Marker written after first-run setup completes: `<config>/veld/monitor-initialized`.
fn monitor_marker_path() -> std::path::PathBuf {
    veld_status_core::default_config_path()
        .parent()
        .map(|p| p.join("monitor-initialized"))
        .unwrap_or_else(|| std::path::PathBuf::from("monitor-initialized"))
}

/// True on the monitor's first run (the marker file is absent).
#[tauri::command]
fn first_run_check() -> bool {
    !monitor_marker_path().exists()
}

/// Record that first-run setup is done so the welcome panel doesn't reappear.
#[tauri::command]
fn mark_initialized() -> Result<(), String> {
    let path = monitor_marker_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&path, b"initialized\n").map_err(|e| e.to_string())
}

/// Seed a project into a veld instance (`POST /api/seed`). The server validates the
/// path (must be under the home dir or /tmp) and scans it into foundational memories.
#[tauri::command]
async fn seed_project(
    state: State<'_, MonitorState>,
    base_url: String,
    project_path: String,
) -> Result<serde_json::Value, String> {
    let (key, user, timeout) = (
        state.api_key.clone(),
        state.user_id.clone(),
        state.request_timeout,
    );
    StatusClient::seed_project(&base_url, &key, &user, &project_path, timeout)
        .await
        .map_err(|e| e.to_string())
}

/// Whether a project has the veld Claude Code hooks wired (read-only — never edits
/// the user's settings). Looks for the memory hook / statusLine in its settings.json.
#[derive(Serialize)]
struct HookStatus {
    configured: bool,
    path: String,
}

#[tauri::command]
fn claude_hook_status(project_dir: String) -> HookStatus {
    let path = std::path::Path::new(&project_dir)
        .join(".claude")
        .join("settings.json");
    let configured = std::fs::read_to_string(&path)
        .map(|s| s.contains("memory-hook.ts") || s.contains("statusLine"))
        .unwrap_or(false);
    HookStatus {
        configured,
        path: path.display().to_string(),
    }
}

/// Keep the fleet in sync with the registry: add pollers for newly-online
/// instances, drop them when they go stale. The pinned instance is never removed.
async fn reconcile_fleet(
    fleet: Arc<Mutex<HashMap<String, FleetEntry>>>,
    pinned_key: String,
    api_key: String,
    user_id: String,
    timeout: Duration,
) {
    loop {
        // Desired set: the pinned instance plus every freshly-heartbeating one.
        let mut desired: HashMap<String, String> = HashMap::new();
        desired.insert(pinned_key.clone(), String::new());
        for inst in discover_instances() {
            let key = norm(&inst.base_url);
            let label = if inst.label.trim().is_empty() {
                url_short(&inst.base_url)
            } else {
                inst.label.clone()
            };
            desired.insert(key, label);
        }

        // Which desired instances are not yet in the fleet?
        let missing: Vec<(String, String)> = {
            let f = fleet.lock();
            desired
                .iter()
                .filter(|(k, _)| !f.contains_key(*k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        for (key, label) in missing {
            if let Some((client, snap)) = spawn_client(&key, &api_key, &user_id, timeout) {
                let pinned = key == pinned_key;
                let lbl = if label.is_empty() { url_short(&key) } else { label };
                fleet.lock().insert(
                    key,
                    FleetEntry {
                        _client: client,
                        snapshot: snap,
                        label: lbl,
                        pinned,
                    },
                );
            }
        }

        // Drop non-pinned instances that are no longer discovered (abort on drop).
        let stale: Vec<String> = {
            let f = fleet.lock();
            f.iter()
                .filter(|(k, e)| !e.pinned && !desired.contains_key(*k))
                .map(|(k, _)| k.clone())
                .collect()
        };
        if !stale.is_empty() {
            let mut f = fleet.lock();
            for key in stale {
                f.remove(&key);
            }
        }

        // Refresh discovered labels on existing non-pinned entries.
        {
            let mut f = fleet.lock();
            for (key, label) in &desired {
                if label.is_empty() {
                    continue;
                }
                if let Some(e) = f.get_mut(key) {
                    if !e.pinned {
                        e.label = label.clone();
                    }
                }
            }
        }

        tokio::time::sleep(RECONCILE_INTERVAL).await;
    }
}

async fn emit_snapshots(handle: AppHandle, snapshot: Arc<RwLock<StatusSnapshot>>) {
    // Cadence: 1s while the main window is visible, 5s while hidden. Emits the
    // pinned instance; non-pinned tabs poll get_snapshot_for on demand.
    let mut last_visible = false;
    loop {
        let visible = handle
            .get_webview_window("main")
            .and_then(|w| w.is_visible().ok())
            .unwrap_or(false);
        let interval = if visible {
            Duration::from_secs(1)
        } else {
            Duration::from_secs(5)
        };
        if visible != last_visible {
            tracing::info!(visible, "monitor window visibility changed");
            last_visible = visible;
        }
        let snap = snapshot.read().clone();
        if let Err(err) = handle.emit("snapshot-updated", &snap) {
            tracing::warn!(?err, "could not emit snapshot-updated");
        }
        tokio::time::sleep(interval).await;
    }
}
