//! Transitional Roots orchestration API.
//!
//! `roots` is the intended orchestration layer above the `earth` substrate.
//! This module exposes the current runtime and HTTP wiring behind that name so
//! the architectural split can proceed incrementally.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;

pub use crate::config::{ServerConfig, StorageBackend};
pub use crate::handlers::{
    build_protected_routes, build_public_routes, build_router, AppState, MultiUserMemoryManager,
};
pub use crate::server::ServerRunConfig;

/// Blocking server bootstrap for the orchestration layer.
pub fn run(config: ServerRunConfig) -> Result<()> {
    crate::server::run(config)
}

/// Runtime wrapper around the current multi-user orchestrator state.
pub struct RootsRuntime {
    manager: Arc<MultiUserMemoryManager>,
}

impl RootsRuntime {
    /// Create a new orchestration runtime rooted at `base_path`.
    pub fn new(base_path: PathBuf, server_config: ServerConfig) -> Result<Self> {
        Ok(Self {
            manager: Arc::new(MultiUserMemoryManager::new(base_path, server_config)?),
        })
    }

    /// Wrap an existing manager without changing ownership semantics.
    pub fn from_manager(manager: Arc<MultiUserMemoryManager>) -> Self {
        Self { manager }
    }

    /// Return the shared application state used by the HTTP handlers.
    pub fn state(&self) -> AppState {
        Arc::clone(&self.manager)
    }

    /// Borrow the shared runtime manager.
    pub fn manager(&self) -> &Arc<MultiUserMemoryManager> {
        &self.manager
    }

    /// Consume the wrapper and return the shared runtime manager.
    pub fn into_manager(self) -> Arc<MultiUserMemoryManager> {
        self.manager
    }

    /// Build the public routes for this runtime.
    pub fn public_routes(&self) -> Router {
        build_public_routes(self.state())
    }

    /// Build the protected routes for this runtime.
    pub fn protected_routes(&self) -> Router {
        build_protected_routes(self.state())
    }

    /// Build the full router for this runtime.
    pub fn router(&self) -> Router {
        build_router(self.state())
    }
}