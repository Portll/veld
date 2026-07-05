//! Server-side registry: the GitHub-account → org → workspace → project(= repo
//! = tenant `user_id`) → folder graph the Veld Manager curates, plus per-key ACL
//! grants.
//!
//! This is the **R2 foundation** — the backing store (5 shared-DB column
//! families, atomic `WriteBatch` writes, prefix scans) and the entity types.
//! The admin CRUD + ACL-filtered `list_tree` + `resolve_scope` + server-side
//! `scoped_recall` endpoints (R3–R7) build on top of it.
//!
//! Reconciliation note: a project's `scope` is the flat veld `user_id`
//! (dot-encoded `{boundary}.{org}.{repo}`), NOT `ScopePath`,
//! which is an orthogonal intra-tenant memory-attribution dimension.

pub mod entities;
pub mod store;

pub use entities::{derive_scope, Account, Boundary, Folder, Grant, Org, Project, Workspace};
pub use store::{RegistryStore, CF_ACL, CF_ENTITIES, CF_INDEX, CF_LINKS, CF_META, REGISTRY_LOCK};
