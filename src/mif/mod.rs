//! Memory Interchange Format (MIF) v2
//!
//! Vendor-neutral schema for exporting, importing, and converting memory data
//! across different AI memory systems (veld-memory, mem0, generic JSON, markdown).
//!
//! Architecture:
//! - `schema` — MIF v2 types (vendor-neutral core + vendor extensions)
//! - `export` — streaming export from veld internals → MifDocument
//! - `import` — reference-preserving import from MifDocument → veld internals
//! - `pii` — PII detection and redaction
//! - `adapters` — format converters (veld JSON, mem0, generic, markdown)

pub mod adapters;
pub mod export;
pub mod import;
pub mod pii;
pub mod schema;

pub use schema::MifDocument;
