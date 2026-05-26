//! HTTP API Handlers - Modular organization of the REST API
//!
//! This module contains all HTTP handlers extracted from the monolithic main.rs.
//! Each submodule handles a specific domain of functionality.

// Core modules
pub mod router;
pub mod state;
pub mod types;

// Health and utilities
pub mod health;
pub mod utils;

// Memory core operations
pub mod crud;
pub mod recall;
pub mod remember;

// Advanced memory operations
pub mod compression;
pub mod facts;
pub mod lineage;
pub mod search;

// Knowledge graph
pub mod gap_analysis;
pub mod graph;
pub mod visualization;

// Task management
pub mod todos;

// MCP and webhooks
pub mod mif;
pub mod webhooks;

// External integrations
pub mod integrations;

// Multi-format text ingestion
pub mod ingest;

// Cold-start project seeding
pub mod seed;

// Session and user management
pub mod sessions;
pub mod users;

// File and codebase memory
pub mod files;

// Background processing
pub mod consolidation;

// Admin operational endpoints (rate-limit reset, etc.)
pub mod admin;

// Context blocks (Letta-style mutable agent state)
pub mod context_blocks;

// A/B testing
pub mod ab_testing;

// External dimension push (graph topological health — Sleight integration)
pub mod external_dimensions;

// Prompt generation (end-to-end context assembly) and entity resolution
pub mod prompt_gen;

// User auth (Phase C) — password + TOTP + recovery codes behind a flag
pub mod user_auth;

// Test utilities (compiled only in test builds)
#[cfg(test)]
pub mod test_helpers;

// Re-export commonly used items
pub use router::{
    build_probe_routes, build_protected_routes, build_public_routes, build_router, AppState,
};
pub use state::MultiUserMemoryManager;
pub use types::*;
