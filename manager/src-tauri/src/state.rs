//! Shared, mutable application state for Veld Manager.
//!
//! Mirrors the Sleight Monitor idiom: `parking_lot::RwLock` fields behind an
//! `Arc`, secrets held only Rust-side. This struct grows per build phase (a
//! registry cache, the resolved Veld connection, an epoch fence for stale
//! events, and — in the credential-broker phase — a zeroize-on-drop admin key).

use parking_lot::RwLock;

/// Manager runtime state, `manage()`d by Tauri and shared across IPC commands.
pub struct ManagerState {
    /// The Veld HTTP endpoint the Manager talks to (loopback by default).
    pub veld_host: RwLock<String>,
}

impl ManagerState {
    pub fn new() -> Self {
        Self {
            veld_host: RwLock::new("http://127.0.0.1:3030".to_string()),
        }
    }
}

impl Default for ManagerState {
    fn default() -> Self {
        Self::new()
    }
}
