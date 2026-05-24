//! Veld status core: shared data model and IO for the lightweight status TUI and GUI.
//!
//! Holds a single [`StatusSnapshot`] under an [`Arc<parking_lot::RwLock>`] and keeps it
//! fresh in the background. Consumers (TUI, Tauri commands) acquire a short read lock
//! and clone the fields they need — never hold the lock across an `.await`.

mod client;
mod dto;
mod snapshot;
mod sse;

pub use client::{StatusClient, StatusClientConfig};
pub use snapshot::{
    ActivityEntry, ContextSession, GraphStats, ReachState, ServerHealth, StatusSnapshot,
    TierStats, TodoStats,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum StatusError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("unexpected http status {status} for {url}: {body}")]
    BadStatus {
        status: u16,
        url: String,
        body: String,
    },
    #[error("invalid base url '{0}': {1}")]
    InvalidBaseUrl(String, String),
    #[error("no users available from /api/users")]
    NoUsers,
    #[error("config file '{path}' could not be read: {source}")]
    ConfigRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("config file '{path}' contained no api key")]
    ConfigEmpty { path: String },
}

pub type Result<T> = std::result::Result<T, StatusError>;

/// Load an API key from `$XDG_CONFIG_HOME/veld/config.toml` (or platform-equivalent),
/// mirroring the lookup logic the rich TUI uses.
///
/// Accepts both `api_key = "..."` lines and a single bare-line file.
pub fn load_api_key_from_default_config() -> Result<String> {
    let path = default_config_path();
    load_api_key_from(&path)
}

/// Load an API key from an arbitrary path. Same parsing rules as the default config.
pub fn load_api_key_from(path: &std::path::Path) -> Result<String> {
    let contents = std::fs::read_to_string(path).map_err(|source| StatusError::ConfigRead {
        path: path.display().to_string(),
        source,
    })?;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once('=') {
            if key.trim() == "api_key" {
                let cleaned = value
                    .split('#')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                if !cleaned.is_empty() {
                    return Ok(cleaned.to_string());
                }
            }
        }
    }

    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .ok_or(StatusError::ConfigEmpty {
            path: path.display().to_string(),
        })
}

/// Default location of the veld config file.
///
/// Resolved via the `dirs` crate's `config_dir` semantics, replicated inline to avoid
/// pulling `dirs` into every dependent crate.
pub fn default_config_path() -> std::path::PathBuf {
    use std::path::PathBuf;

    let base = if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg)
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    } else if cfg!(target_os = "macos") {
        std::env::var("HOME")
            .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
            .unwrap_or_else(|_| PathBuf::from("."))
    } else {
        std::env::var("HOME")
            .map(|h| PathBuf::from(h).join(".config"))
            .unwrap_or_else(|_| PathBuf::from("."))
    };

    base.join("veld").join("config.toml")
}
