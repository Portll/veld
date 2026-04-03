//! Veld Library
//!
//! Edge-native AI memory system for autonomous agents.
//! Optimized for deployment on resource-constrained devices.
//!
//! # Key Features
//! - Tiered memory (working/session/long-term) based on cognitive science
//! - Local vector search (Vamana/DiskANN)
//! - Local embeddings (MiniLM-L6 via ONNX)
//! - Knowledge graph for entity relationships
//!
//! # Edge Optimizations
//! - Lazy model loading (reduces startup RAM by ~200MB)
//! - Configurable thread count for power efficiency
//! - Backend-selectable embedded storage (legacy RocksDB compatibility today)
//! - Full offline operation

pub mod ab_testing;
pub mod auth;
pub mod backup;
pub mod config;
pub mod constants;
pub mod decay;
pub mod decay_scales;
pub mod earth;
pub mod embeddings;
pub mod encryption;
pub mod errors;
#[cfg(feature = "multi-tenant")]
pub mod extensions;
pub mod graph_memory;
pub mod handlers;
pub mod ingest;
pub mod integrations;
pub mod memory;
pub mod metrics;
pub mod middleware;
pub mod mif;
pub mod query_parsing;
pub mod relevance;
pub mod roots;
pub mod server;
pub mod similarity;
pub mod storage;
pub mod streaming;
pub mod tracing_setup;
pub mod validation;
pub mod vector_db;

pub mod mcp;

// Re-export dependencies to ensure tests/benchmarks use the same version
pub use chrono;
pub use parking_lot;
pub use uuid;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "zenoh")]
pub mod zenoh_transport;

#[cfg(feature = "fortress")]
pub mod fortress;
