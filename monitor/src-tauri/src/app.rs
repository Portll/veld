//! Tauri application bootstrap: spawns the status client, registers commands,
//! manages window visibility, emits live snapshot events.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use veld_status_core::{StatusClient, StatusClientConfig, StatusSnapshot};

use crate::tray;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:3030";
const DEFAULT_DEV_API_KEY: &str = "sk-veld-dev-local-testing-key";
const DEFAULT_USER_ID: &str = "claude-code";

/// Anchors the background poller so it lives as long as the app does.
pub struct MonitorState {
    pub snapshot: Arc<RwLock<StatusSnapshot>>,
    _client: StatusClient,
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

            let config = StatusClientConfig::new(base_url, api_key, user_id)
                .with_refresh_interval(Duration::from_secs(2))
                .with_request_timeout(Duration::from_secs(5));

            let client = tauri::async_runtime::block_on(async move {
                StatusClient::spawn(config)
            })
            .map_err(|e| Box::new(std::io::Error::other(e.to_string())) as Box<dyn std::error::Error>)?;

            let snapshot = client.snapshot();
            app.manage(MonitorState {
                snapshot: Arc::clone(&snapshot),
                _client: client,
            });

            tray::build(app, Arc::clone(&snapshot))?;

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(emit_snapshots(handle, snapshot));

            Ok(())
        })
        .on_window_event(|window, event| {
            // X-button closes the whole app, matching the standard desktop
            // expectation. The tray menu still has Show / Quit; the tray
            // icon stays available while the window is hidden via the
            // menu "Hide" path, but pressing X = exit.
            if let WindowEvent::CloseRequested { .. } = event {
                window.app_handle().exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            show_dashboard,
            quit_app
        ])
        .run(tauri::generate_context!())
        .expect("failed to run tauri application");
}

#[tauri::command]
fn get_snapshot(state: State<'_, MonitorState>) -> StatusSnapshot {
    state.snapshot.read().clone()
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

async fn emit_snapshots(handle: AppHandle, snapshot: Arc<RwLock<StatusSnapshot>>) {
    // Cadence: 1s while the main window is visible, 5s while hidden. Cheap because
    // emit is in-process and the consumer is a single webview.
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
