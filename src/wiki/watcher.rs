//! Live wiki edit-back watcher (W-003/W-004 — optional layer).
//!
//! The robust edit-back path is the on-demand scan
//! ([`crate::handlers::state::MultiUserMemoryManager::scan_wiki_edits`],
//! triggered by `POST /api/wiki/render`). This module adds an *optional* live
//! layer: a per-user filesystem watcher (the `notify` crate) that debounces
//! events on the user's wiki directory and runs that same scan, so an Obsidian
//! edit flows back within the debounce window instead of on the next render
//! call.
//!
//! Gated by `VELD_WIKI_WATCH` (default off), separate from `VELD_WIKI_RENDER`,
//! because it is the most fragile contract in the design (the synthesis's
//! warning): a long-lived task + a filesystem watcher + a new dependency.
//! Self-write safety is inherited from the scan, not re-invented here — a page
//! veld just rendered hashes equal to its recorded `last_rendered_hash` and is
//! skipped, so even an event fired by veld's own write is a no-op and the loop
//! converges.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use crate::handlers::state::MultiUserMemoryManager;

/// Debounce window — a burst of events (editors write-temp-then-rename,
/// producing several) collapses into a single scan.
const DEBOUNCE: Duration = Duration::from_millis(750);

/// Is the live watcher enabled? Default OFF; separate from
/// `VELD_WIKI_RENDER` so the live layer is opt-in even when render is on.
pub fn wiki_watch_enabled() -> bool {
    std::env::var("VELD_WIKI_WATCH")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

/// Spawn a debounced filesystem watcher on `dir` that runs the edit-back scan
/// for `user_id`. Returns the watcher handle — the caller MUST keep it alive
/// (dropping it stops watching and lets the debounce thread exit cleanly).
///
/// The debounce thread holds a strong `Arc<MultiUserMemoryManager>` (the
/// global server state, process-lifetime anyway) and exits when the watcher's
/// event sender is dropped.
pub fn spawn_wiki_watcher(
    manager: Arc<MultiUserMemoryManager>,
    user_id: String,
    dir: &Path,
) -> notify::Result<RecommendedWatcher> {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let mut watcher = RecommendedWatcher::new(
        move |res: notify::Result<notify::Event>| {
            // Any change triggers a debounced scan; which event it was does
            // not matter (the scan re-reads every page and is the source of
            // truth). Watcher errors are dropped.
            if res.is_ok() {
                let _ = tx.send(());
            }
        },
        notify::Config::default(),
    )?;
    watcher.watch(dir, RecursiveMode::NonRecursive)?;

    let thread_user = user_id.clone();
    let spawned = std::thread::Builder::new()
        .name(format!("wiki-watch-{user_id}"))
        .spawn(move || {
            // Block until the watcher's sender is dropped (Err => exit).
            while rx.recv().is_ok() {
                // Debounce: collapse a burst of events into one scan.
                std::thread::sleep(DEBOUNCE);
                while rx.try_recv().is_ok() {}
                match manager.scan_wiki_edits(&thread_user) {
                    Ok(n) if n > 0 => tracing::info!(
                        user_id = %thread_user,
                        applied = n,
                        "live wiki watcher applied edit-back",
                    ),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(
                        user_id = %thread_user,
                        error = %e,
                        "live wiki watcher scan failed (non-fatal)",
                    ),
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(user_id = %user_id, error = %e, "failed to spawn wiki watcher debounce thread");
    }

    Ok(watcher)
}
