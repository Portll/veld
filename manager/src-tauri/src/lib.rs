//! Veld Manager — desktop console for workspaces, folders, git repositories,
//! GitHub accounts, and Veld tenant scopes, and the per-install credential
//! broker for the Veld memory system.
//!
//! This module wires the Tauri application. View logic (the Account → Org →
//! Workspace → Project → Folder tree, workspace builder, scope/key management,
//! hook unification, and scoped recall) and the IPC commands that back them are
//! added incrementally per build phase; secrets are always held Rust-side and
//! never returned to the webview.

mod discovery;
mod state;

use std::sync::Arc;

use state::ManagerState;

/// Minimal app identity surfaced to the webview at startup (P0 skeleton).
#[derive(serde::Serialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
}

/// `app_info` — returns the app name + version for the webview's title bar.
#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        name: "Veld Manager".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// `list_gh_accounts` — GitHub accounts known to the gh CLI (from hosts.yml).
#[tauri::command]
fn list_gh_accounts() -> Vec<discovery::GhAccount> {
    discovery::list_gh_accounts()
}

/// `scan_repos` — repositories + worktrees under ~/Repositories, classified by
/// org/account/boundary, with the hook-matching project hash.
#[tauri::command]
fn scan_repos() -> Vec<discovery::RepoNode> {
    discovery::scan_repos()
}

/// `project_hash` — the `CLAUDE_PROJECT_HASH` a hook would compute for a path.
#[tauri::command]
fn project_hash(path: String) -> String {
    discovery::project_hash(&path)
}

/// `server_health` — reachability + RTT of the configured Veld endpoint.
#[tauri::command]
async fn server_health(
    state: tauri::State<'_, Arc<ManagerState>>,
) -> Result<discovery::ServerHealth, String> {
    let host = state.veld_host.read().clone();
    Ok(discovery::server_health(&host).await)
}

/// Launch the Veld Manager desktop application.
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let state = Arc::new(ManagerState::new());

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            app_info,
            list_gh_accounts,
            scan_repos,
            project_hash,
            server_health
        ])
        .run(tauri::generate_context!())
        .expect("error while running Veld Manager");
}
