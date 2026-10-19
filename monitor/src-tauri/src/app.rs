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

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use std::collections::BTreeMap;

use veld_status_core::{
    discover_instances, inspect_process, launch_recipes_path, load_launch_recipes, parse_env_file,
    restart_instance, restart_request, rewrite_env_var, server_env_path, LaunchSpec, StatusClient,
    StatusClientConfig, StatusSnapshot,
};

use crate::{replica, tray};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:3030";
const DEFAULT_DEV_API_KEY: &str = "sk-veld-dev-local-testing-key";
const DEFAULT_USER_ID: &str = "claude-code";
const RECONCILE_INTERVAL: Duration = Duration::from_secs(3);
const REPLICATION_POLL_INTERVAL: Duration = Duration::from_secs(10);
/// Exports, restores and migrations walk whole stores.
const DATA_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const CONFIG_BACKEND_KEY: &str = "storage_backend";

/// One member of the fleet: a running status client plus its snapshot handle.
struct FleetEntry {
    /// Held so its background tasks live; dropped (→ aborted) on removal.
    _client: StatusClient,
    snapshot: Arc<RwLock<StatusSnapshot>>,
    label: String,
    pinned: bool,
    /// The registry's `started_at`; None while the instance is not in the registry.
    started_at: Option<DateTime<Utc>>,
}

/// Tab metadata handed to the frontend.
#[derive(Serialize, Clone)]
struct InstanceMeta {
    base_url: String,
    label: String,
    pinned: bool,
    online: bool,
    started_at: Option<DateTime<Utc>>,
}

/// App-wide state: connection params + the live fleet.
pub struct MonitorState {
    api_key: String,
    user_id: String,
    request_timeout: Duration,
    /// `VELD_ADMIN_API_KEY` of this process, if set; the Data panel asks otherwise.
    admin_key: Option<String>,
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

fn encode_url_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
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
        .plugin(tauri_plugin_dialog::init())
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
            let admin_key = std::env::var("VELD_ADMIN_API_KEY")
                .ok()
                .filter(|s| !s.trim().is_empty());
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
                    started_at: None,
                },
            );
            let fleet = Arc::new(Mutex::new(map));

            app.manage(MonitorState {
                api_key: api_key.clone(),
                user_id: user_id.clone(),
                request_timeout: timeout,
                admin_key: admin_key.clone(),
                pinned_base_url: pinned_key.clone(),
                fleet: Arc::clone(&fleet),
            });

            // Tray + live event stream follow the pinned instance; the tray's
            // replication line covers every instance that follows a leader.
            let replication = Arc::new(RwLock::new(None));
            tray::build(app, Arc::clone(&snapshot), Arc::clone(&replication))?;
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(emit_snapshots(handle, snapshot));
            tauri::async_runtime::spawn(poll_replication(
                Arc::clone(&fleet),
                api_key.clone(),
                admin_key,
                timeout,
                replication,
            ));

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
            graph_view_url_for,
            first_run_check,
            mark_initialized,
            seed_project,
            claude_hook_status,
            pick_directory,
            minimize_window,
            toggle_maximize,
            hide_window,
            security_request,
            enclave,
            save_secret_file,
            open_in_browser,
            data_request,
            admin_key_configured,
            save_export_file,
            pick_import_file,
            config_backend,
            set_config_backend,
            restart_server,
            launch_recipe,
            set_launch_recipe,
            capture_launch_recipe
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
                started_at: e.started_at,
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

// ---- Custom title-bar window controls (frameless window) ----------------
#[tauri::command]
fn minimize_window(window: tauri::WebviewWindow) {
    let _ = window.minimize();
}

#[tauri::command]
fn toggle_maximize(window: tauri::WebviewWindow) {
    if window.is_maximized().unwrap_or(false) {
        let _ = window.unmaximize();
    } else {
        let _ = window.maximize();
    }
}

/// Close button → hide to tray (matches the CloseRequested handler).
#[tauri::command]
fn hide_window(window: tauri::WebviewWindow) {
    let _ = window.hide();
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

/// Build an authenticated graph-view URL for an iframe.
///
/// The API key is carried in the URL fragment, not the query string, so it is
/// not sent with the public `/graph/view` request. The graph page reads it
/// locally and uses it as an `X-API-Key` header for protected JSON fetches.
#[tauri::command]
fn graph_view_url_for(state: State<'_, MonitorState>, base_url: String) -> Result<String, String> {
    let base = norm(&base_url);
    if base.is_empty() {
        return Err("base_url is empty".to_string());
    }

    let user = encode_url_component(&state.user_id);
    let key = encode_url_component(&state.api_key);
    let bust = chrono::Utc::now().timestamp_millis();
    Ok(format!(
        "{base}/graph/view?user_id={user}&_t={bust}#api_key={key}"
    ))
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

/// Open a native folder picker and return the chosen directory (or None if the
/// user cancels). Backs the "Browse…" buttons on the first-run panel's path inputs.
#[tauri::command]
async fn pick_directory(app: AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |picked| {
        let _ = tx.send(picked);
    });
    rx.await
        .ok()
        .flatten()
        .and_then(|fp| fp.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

/// A call to `/api/user_auth/*` (logins and keys) for the Security tab. The
/// session token lives only in the webview's memory and rides each call. Key
/// operations run Argon2 over every keystore, hence the long timeout.
#[tauri::command]
async fn security_request(
    base_url: String,
    method: String,
    path: String,
    session: Option<String>,
    step_up: Option<String>,
    body: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let (status, body) = StatusClient::security_request(
        &base_url,
        &method,
        &path,
        session.as_deref(),
        step_up.as_deref(),
        body,
        Duration::from_secs(180),
    )
    .await
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "status": status, "body": body }))
}

/// `VELD_ENCLAVE_HELPER`, else `veld-enclave` beside this app, else on PATH.
fn enclave_helper() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("VELD_ENCLAVE_HELPER") {
        if !path.trim().is_empty() {
            return path.into();
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("veld-enclave")))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| "veld-enclave".into())
}

/// Run the Touch ID helper on this Mac: `probe`, `list`, `enrol` or `unlock`. A
/// released secret goes back to the webview, which posts it to veld over the same
/// connection as the rest of the Security tab.
#[tauri::command]
async fn enclave(
    action: String,
    id: Option<String>,
    label: Option<String>,
) -> Result<serde_json::Value, String> {
    let mut args = vec![action.clone()];
    match action.as_str() {
        "probe" | "list" => {}
        "enrol" => {
            if let Some(label) = label {
                args.extend(["--label".into(), label]);
            }
        }
        "unlock" => {
            args.extend(["--id".into(), id.ok_or("unlock needs an id")?]);
            args.extend(["--reason".into(), "authorise a change to the veld keyring".into()]);
        }
        other => return Err(format!("unknown enclave action {other}")),
    }
    let helper = enclave_helper();
    let out = tokio::task::spawn_blocking(move || {
        std::process::Command::new(&helper)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("run {}: {e}", helper.display()))
    })
    .await
    .map_err(|e| e.to_string())??;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("helper output: {e}"))
}

/// Native save dialog seeded with `suggested_name`; None when the user cancels.
async fn pick_save_path(app: &AppHandle, suggested_name: &str) -> Option<std::path::PathBuf> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_file_name(suggested_name)
        .save_file(move |picked| {
            let _ = tx.send(picked);
        });
    rx.await
        .ok()
        .flatten()
        .and_then(|fp| fp.into_path().ok())
}

/// Write `contents` readable by its owner only, fsynced before returning.
fn write_private_file(path: &std::path::Path, contents: &str) -> Result<(), String> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    use std::io::Write;
    let mut file = options.open(path).map_err(|e| e.to_string())?;
    file.write_all(contents.as_bytes()).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Save a recovery kit where the user chooses, readable by its owner only.
#[tauri::command]
async fn save_secret_file(
    app: AppHandle,
    suggested_name: String,
    contents: String,
) -> Result<Option<String>, String> {
    let Some(path) = pick_save_path(&app, &suggested_name).await else {
        return Ok(None);
    };
    write_private_file(&path, &contents)?;
    Ok(Some(path.to_string_lossy().to_string()))
}

/// Save a memory export where the user chooses. An export carries every memory of
/// the user, so it gets the same owner-only mode as a recovery kit.
#[tauri::command]
async fn save_export_file(
    app: AppHandle,
    suggested_name: String,
    contents: String,
) -> Result<Option<String>, String> {
    let Some(path) = pick_save_path(&app, &suggested_name).await else {
        return Ok(None);
    };
    write_private_file(&path, &contents)?;
    Ok(Some(path.to_string_lossy().to_string()))
}

/// A file picked for import, with its text.
#[derive(Serialize)]
struct PickedFile {
    path: String,
    contents: String,
}

/// Native open dialog, then the file's text; None when the user cancels.
#[tauri::command]
async fn pick_import_file(app: AppHandle) -> Result<Option<PickedFile>, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_file(move |picked| {
        let _ = tx.send(picked);
    });
    let Some(path) = rx
        .await
        .ok()
        .flatten()
        .and_then(|fp| fp.into_path().ok())
    else {
        return Ok(None);
    };
    let display = path.to_string_lossy().to_string();
    let contents = tokio::task::spawn_blocking(move || std::fs::read_to_string(&path))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("read {display}: {e}"))?;
    Ok(Some(PickedFile {
        path: display,
        contents,
    }))
}

/// A call to the data-management surface (backups, MIF export/import, storage
/// migration, health) for the Data panel. The admin key for `/api/admin/storage/`
/// is the one the panel was given, else this process's `VELD_ADMIN_API_KEY`.
#[tauri::command]
async fn data_request(
    state: State<'_, MonitorState>,
    base_url: String,
    method: String,
    path: String,
    admin_key: Option<String>,
    body: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let api_key = state.api_key.clone();
    let admin = admin_key
        .filter(|k| !k.trim().is_empty())
        .or_else(|| state.admin_key.clone());
    let (status, body) = StatusClient::data_request(
        &base_url,
        &api_key,
        &method,
        &path,
        admin.as_deref(),
        body,
        DATA_REQUEST_TIMEOUT,
    )
    .await
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "status": status, "body": body }))
}

/// Whether this process carries an admin key, so the panel knows not to ask.
#[tauri::command]
fn admin_key_configured(state: State<'_, MonitorState>) -> bool {
    state.admin_key.is_some()
}

/// Where the next start's backend is decided: `config.toml`'s `storage_backend`,
/// and above it `VELD_STORAGE_BACKEND` in the operator's `server.env` when Veld
/// Desktop spawns the server (the environment beats the file).
#[derive(Serialize)]
struct ConfigBackend {
    path: String,
    backend: Option<String>,
    server_env_path: String,
    server_env_backend: Option<String>,
}

fn server_env_vars() -> Vec<(String, String)> {
    std::fs::read_to_string(server_env_path())
        .map(|t| parse_env_file(&t))
        .unwrap_or_default()
}

fn server_env_backend() -> Option<String> {
    server_env_vars()
        .into_iter()
        .find(|(k, _)| k == "VELD_STORAGE_BACKEND")
        .map(|(_, v)| v.to_ascii_lowercase())
}

fn config_backend_line(line: &str) -> Option<String> {
    let (key, value) = line.trim().split_once('=')?;
    if key.trim() != CONFIG_BACKEND_KEY {
        return None;
    }
    let cleaned = value
        .split('#')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('"')
        .trim_matches('\'');
    Some(cleaned.to_string())
}

#[tauri::command]
fn config_backend() -> ConfigBackend {
    let path = veld_status_core::default_config_path();
    let backend = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| {
            text.lines()
                .filter(|l| !l.trim().starts_with('#'))
                .find_map(config_backend_line)
        })
        .filter(|b| !b.is_empty());
    ConfigBackend {
        path: path.display().to_string(),
        backend,
        server_env_path: server_env_path().display().to_string(),
        server_env_backend: server_env_backend(),
    }
}

/// Name the backend the server serves on its next start: rewrites the
/// `storage_backend` line of `config.toml`, or appends one, and when the
/// operator's `server.env` pins `VELD_STORAGE_BACKEND` rewrites that line too,
/// since it would otherwise win. Every other line of both files is kept byte for
/// byte, the API key included.
#[tauri::command]
fn set_config_backend(backend: String) -> Result<ConfigBackend, String> {
    let backend = backend.trim().to_ascii_lowercase();
    if backend != "redb" && backend != "rocksdb" {
        return Err(format!("unknown backend '{backend}' (expected redb or rocksdb)"));
    }
    let path = veld_status_core::default_config_path();
    let existing = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    let setting = format!("{CONFIG_BACKEND_KEY} = \"{backend}\"");
    let mut replaced = false;
    let mut lines: Vec<String> = existing
        .lines()
        .map(|line| {
            let names_key =
                !line.trim().starts_with('#') && config_backend_line(line).is_some();
            if names_key && !replaced {
                replaced = true;
                setting.clone()
            } else {
                line.to_string()
            }
        })
        .collect();
    if !replaced {
        lines.push(setting);
    }
    let mut text = lines.join("\n");
    text.push('\n');
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    write_private_file(&path, &text)?;

    let env_path = server_env_path();
    let mut server_env_backend = server_env_backend();
    if server_env_backend.is_some() {
        let current = std::fs::read_to_string(&env_path)
            .map_err(|e| format!("read {}: {e}", env_path.display()))?;
        write_private_file(&env_path, &rewrite_env_var(&current, "VELD_STORAGE_BACKEND", &backend))?;
        server_env_backend = Some(backend.clone());
    }
    Ok(ConfigBackend {
        path: path.display().to_string(),
        backend: Some(backend),
        server_env_path: env_path.display().to_string(),
        server_env_backend,
    })
}

/// Open a veld page in the system browser: passkeys need one, a webview cannot
/// reach the platform authenticator without entitlements.
#[tauri::command]
fn open_in_browser(url: String) -> Result<(), String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("only http(s) URLs open".into());
    }
    let (program, args): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
        ("open", vec![url.as_str()])
    } else if cfg!(target_os = "windows") {
        ("cmd", vec!["/C", "start", "", url.as_str()])
    } else {
        ("xdg-open", vec![url.as_str()])
    };
    std::process::Command::new(program)
        .args(args)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn load_recipes() -> Result<BTreeMap<String, LaunchSpec>, String> {
    load_launch_recipes().map_err(|e| e.to_string())
}

fn save_recipes(recipes: &BTreeMap<String, LaunchSpec>) -> Result<(), String> {
    let path = launch_recipes_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(recipes).map_err(|e| e.to_string())?;
    write_private_file(&path, &text)
}

fn find_instance(base_url: &str) -> Option<veld_status_core::DiscoveredInstance> {
    let key = norm(base_url);
    discover_instances()
        .into_iter()
        .find(|i| norm(&i.base_url) == key)
}

/// What a restart of `base_url` would use, for the Launch card.
#[derive(Serialize)]
struct LaunchInfo {
    running: bool,
    pid: Option<u32>,
    /// The registry's recipe: `exe`, `args`, `cwd`, `env_names`.
    registry: Option<serde_json::Value>,
    saved: Option<LaunchSpec>,
    recipes_path: String,
    server_env_path: String,
    server_env_names: Vec<String>,
}

#[tauri::command]
fn launch_recipe(base_url: String) -> Result<LaunchInfo, String> {
    let instance = find_instance(&base_url);
    let registry = instance.as_ref().and_then(|i| i.launch.as_ref()).map(|r| {
        serde_json::json!({
            "exe": r.exe, "args": r.args, "cwd": r.cwd, "env_names": r.env_names,
        })
    });
    let saved = load_recipes()?.remove(&norm(&base_url));
    Ok(LaunchInfo {
        running: instance.is_some(),
        pid: instance.as_ref().map(|i| i.pid).filter(|p| *p != 0),
        registry,
        saved,
        recipes_path: launch_recipes_path().display().to_string(),
        server_env_path: server_env_path().display().to_string(),
        server_env_names: server_env_vars().into_iter().map(|(k, _)| k).collect(),
    })
}

/// Save (or with None, forget) the recipe a restart of `base_url` uses.
#[tauri::command]
fn set_launch_recipe(base_url: String, spec: Option<LaunchSpec>) -> Result<(), String> {
    let key = norm(&base_url);
    let mut recipes = load_recipes()?;
    match spec {
        Some(spec) if !spec.exe.trim().is_empty() => {
            recipes.insert(key, spec);
        }
        Some(_) => return Err("the recipe needs an executable".into()),
        None => {
            recipes.remove(&key);
        }
    }
    save_recipes(&recipes)
}

/// The running server's command line, directory and `VELD_*` variables, read from
/// the OS, as a recipe the user can save. Nothing is written.
#[tauri::command]
async fn capture_launch_recipe(base_url: String) -> Result<LaunchSpec, String> {
    let instance = find_instance(&base_url)
        .ok_or_else(|| format!("{} is not in the instance registry", norm(&base_url)))?;
    let pid = instance.pid;
    if pid == 0 {
        return Err("the registry entry carries no pid".into());
    }
    tokio::task::spawn_blocking(move || inspect_process(pid))
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("could not read process {pid}: it may be gone, or this platform does not expose it"))
}

/// Stop the instance at `base_url`, if it runs, and bring it back the way
/// `veld restart` does (`restart_request`). `storage_backend`, when given, is set
/// as `VELD_STORAGE_BACKEND` on the new process (a swap just performed).
async fn restart_at(
    base_url: &str,
    storage_backend: Option<String>,
) -> Result<veld_status_core::RestartOutcome, String> {
    let key = norm(base_url);
    let instance = find_instance(&key);
    let recipes = load_recipes()?;
    let mut env_overrides = Vec::new();
    if let Some(backend) = storage_backend.filter(|b| !b.trim().is_empty()) {
        env_overrides.push(("VELD_STORAGE_BACKEND".to_string(), backend));
    }
    restart_instance(restart_request(&key, instance, &recipes, env_overrides))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn restart_server(
    base_url: String,
    storage_backend: Option<String>,
) -> Result<veld_status_core::RestartOutcome, String> {
    restart_at(&base_url, storage_backend).await
}

/// Tray entry point: restart the pinned instance in the background and tell the
/// webview how it went (`server-restarted` carries the outcome or the error).
pub fn restart_pinned(app: &AppHandle) {
    let base_url = app.state::<MonitorState>().pinned_base_url.clone();
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let payload = match restart_at(&base_url, None).await {
            Ok(outcome) => {
                tracing::info!(?outcome, "server restarted from the tray");
                serde_json::json!({ "ok": outcome })
            }
            Err(err) => {
                tracing::warn!(error = %err, "server restart from the tray failed");
                serde_json::json!({ "error": err, "base_url": base_url })
            }
        };
        let _ = handle.emit("server-restarted", payload);
    });
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
        let mut started: HashMap<String, DateTime<Utc>> = HashMap::new();
        for inst in discover_instances() {
            let key = norm(&inst.base_url);
            let label = if inst.label.trim().is_empty() {
                url_short(&inst.base_url)
            } else {
                inst.label.clone()
            };
            // A killed server's entry outlives it until its heartbeat goes stale,
            // so beside its successor on the same port the newer start wins.
            if let Some(at) = inst.started_at {
                started
                    .entry(key.clone())
                    .and_modify(|t| *t = (*t).max(at))
                    .or_insert(at);
            }
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
                let started_at = started.get(&key).copied();
                fleet.lock().insert(
                    key,
                    FleetEntry {
                        _client: client,
                        snapshot: snap,
                        label: lbl,
                        pinned,
                        started_at,
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

        // Refresh discovered labels on existing non-pinned entries, and every
        // entry's start time (a restart writes a new one).
        {
            let mut f = fleet.lock();
            for (key, e) in f.iter_mut() {
                e.started_at = started.get(key).copied();
            }
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

fn is_loopback(base_url: &str) -> bool {
    let Some(url) = tauri::Url::parse(base_url).ok() else {
        return false;
    };
    let host = url.host_str().unwrap_or("");
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Ask every reachable instance in the fleet whether it follows a leader and keep
/// the tray's line on those that do. The status route is admin-gated: the key is
/// this process's `VELD_ADMIN_API_KEY`, else the one in the operator's
/// `server.env`, which the servers on this host were started with and which is
/// therefore sent to loopback instances only.
async fn poll_replication(
    fleet: Arc<Mutex<HashMap<String, FleetEntry>>>,
    api_key: String,
    admin_key: Option<String>,
    timeout: Duration,
    line: Arc<RwLock<Option<String>>>,
) {
    loop {
        let targets: Vec<(String, String)> = fleet
            .lock()
            .iter()
            .filter(|(_, e)| e.snapshot.read().server.state.is_reachable())
            .map(|(key, e)| (key.clone(), e.label.clone()))
            .collect();
        let env_key = server_env_vars()
            .into_iter()
            .find(|(k, v)| k == "VELD_ADMIN_API_KEY" && !v.trim().is_empty())
            .map(|(_, v)| v);
        let mut replicas = Vec::new();
        for (base_url, label) in targets {
            let key = admin_key
                .as_deref()
                .or_else(|| env_key.as_deref().filter(|_| is_loopback(&base_url)));
            let Some(key) = key else { continue };
            match StatusClient::data_request(
                &base_url,
                &api_key,
                "GET",
                "/api/admin/replication/status",
                Some(key),
                None,
                timeout,
            )
            .await
            {
                Ok((200, body)) => {
                    if let Some(summary) = replica::summarize(&body) {
                        replicas.push((label, summary));
                    }
                }
                Ok((status, _)) => {
                    tracing::debug!(%base_url, status, "replication status refused");
                }
                Err(e) => tracing::debug!(%base_url, error = %e, "replication status unavailable"),
            }
        }
        replicas.sort_by(|a, b| a.0.cmp(&b.0));
        *line.write() = replica::tray_line(&replicas);
        tokio::time::sleep(REPLICATION_POLL_INTERVAL).await;
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
