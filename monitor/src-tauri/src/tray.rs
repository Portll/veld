//! System-tray icon + menu. The menu surface deliberately reads from a *snapshot*
//! of the latest data so that hovering the tray shows useful numbers without
//! opening the dashboard.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, Manager,
};
use veld_status_core::{ReachState, StatusSnapshot};

/// Where the replication line sits while shown: after "Sessions".
const REPLICATION_POSITION: usize = 5;

/// `replication` is the one-line summary of the instances that follow a leader
/// (see [`crate::replica`]); the menu shows it only while there is one.
pub fn build(
    app: &mut App,
    snapshot: Arc<RwLock<StatusSnapshot>>,
    replication: Arc<RwLock<Option<String>>>,
) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open Dashboard", true, None::<&str>)?;
    let status_item = MenuItem::with_id(app, "status", "Server: probing…", false, None::<&str>)?;
    let memory_item = MenuItem::with_id(app, "memory", "Memories: —", false, None::<&str>)?;
    let sessions_item =
        MenuItem::with_id(app, "sessions", "Sessions: —", false, None::<&str>)?;
    let replication_item =
        MenuItem::with_id(app, "replication", "Replica: —", false, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let restart = MenuItem::with_id(app, "restart", "Restart Server", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Veld Monitor", true, None::<&str>)?;

    let menu = Menu::with_items(
        app,
        &[
            &open,
            &separator,
            &status_item,
            &memory_item,
            &sessions_item,
            &separator,
            &restart,
            &quit,
        ],
    )?;

    let mut builder = TrayIconBuilder::with_id("main")
        .menu(&menu)
        .show_menu_on_left_click(false);
    if let Some(icon) = app.default_window_icon().cloned() {
        builder = builder.icon(icon);
    }
    let tray = builder
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize();
                    let _ = window.set_focus();
                }
            }
            "restart" => crate::app::restart_pinned(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Only LEFT click + button release opens the window. Right click
            // must fall through to the OS so the context menu can appear; if
            // we match all clicks here we swallow that event and the menu
            // (which contains the only Quit option) never shows.
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                if let Some(window) = tray.app_handle().get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    // Background task: refresh tray title + menu labels from the live snapshot.
    let handle = app.handle().clone();
    let snapshot_for_refresh = Arc::clone(&snapshot);
    let menu_for_refresh = menu.clone();
    tauri::async_runtime::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(2));
        let mut replication_shown = false;
        loop {
            ticker.tick().await;
            let snap = snapshot_for_refresh.read().clone();
            update_menu(&snap, &status_item, &memory_item, &sessions_item);
            let line = replication.read().clone();
            update_replication(&menu_for_refresh, &replication_item, line, &mut replication_shown);
            update_tray_title(&handle, &snap);
        }
    });

    drop(tray); // keep alive via the AppHandle's registry
    Ok(())
}

fn update_menu(
    snap: &StatusSnapshot,
    status_item: &MenuItem<tauri::Wry>,
    memory_item: &MenuItem<tauri::Wry>,
    sessions_item: &MenuItem<tauri::Wry>,
) {
    let server_label = match &snap.server.state {
        ReachState::Reachable => {
            let rtt = snap.server.rtt_ms.unwrap_or(0);
            format!("Server: {} · {}ms", snap.base_url, rtt)
        }
        ReachState::Unhealthy(msg) => format!("Server: unhealthy ({})", short(msg)),
        ReachState::Unreachable(_) => format!("Server: offline ({})", snap.base_url),
        ReachState::Unknown => "Server: probing…".to_string(),
    };
    let _ = status_item.set_text(server_label);
    let _ = memory_item.set_text(format!("Memories: {}", thousands(snap.memory.total)));
    let _ = sessions_item.set_text(format!("Sessions: {}", snap.sessions.len()));
}

fn update_replication(
    menu: &Menu<tauri::Wry>,
    item: &MenuItem<tauri::Wry>,
    line: Option<String>,
    shown: &mut bool,
) {
    match line {
        Some(text) => {
            let _ = item.set_text(text);
            if !*shown {
                *shown = menu.insert(item, REPLICATION_POSITION).is_ok();
            }
        }
        None if *shown => {
            *shown = menu.remove(item).is_err();
        }
        None => {}
    }
}

fn update_tray_title(handle: &tauri::AppHandle, snap: &StatusSnapshot) {
    if let Some(tray) = handle.tray_by_id("main") {
        // macOS shows tray title text next to the icon; other platforms ignore.
        let title = match &snap.server.state {
            ReachState::Reachable => format!("● {}", thousands(snap.memory.total)),
            ReachState::Unhealthy(_) => "▲ unhealthy".to_string(),
            ReachState::Unreachable(_) => "○ offline".to_string(),
            ReachState::Unknown => String::new(),
        };
        let _ = tray.set_title(Some(title.as_str()));
    }
}

fn short(s: &str) -> String {
    if s.chars().count() > 32 {
        let pre: String = s.chars().take(31).collect();
        format!("{}…", pre)
    } else {
        s.to_string()
    }
}

pub(crate) fn thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}
