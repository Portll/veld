//! A change of embedder as a resumable drain, not a pass at start.
//!
//! When the active embedder writes another space than the corpus markers name, start-up records
//! a `SpaceMigration` in the tenant's own store and moves on. The backlog drain then walks the
//! memories in id order, a batch per tick: each memory whose primary vector is not in `to` is
//! embedded into `to`, keeps the vector it had, and is re-indexed; the cursor is saved after
//! every batch. A kill, a full disk or a restart mid-way resumes at the cursor. The same record
//! carries the one-time stamp of spaces onto vectors written before spaces were recorded, so an
//! ordinary start rewrites nothing.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::types::MemoryId;
use crate::embeddings::space::SpaceId;
use crate::storage::KeyValueStore;

const KEY: &[u8] = b"space_migration";

/// What the drain is doing to the corpus's spaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationKind {
    /// Name the space of every vector that has none, without re-embedding: the corpus markers
    /// name it. Nothing is unreachable while it runs, since a vector with no space is read as
    /// the index's at the index's width.
    Stamp,
    /// Re-embed every primary vector not in `to`.
    Reembed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpaceMigration {
    pub kind: MigrationKind,
    /// The space the corpus markers named when the migration began; `None` when they named
    /// none. A stamp records it on vectors with no space.
    pub from: Option<SpaceId>,
    pub to: SpaceId,
    /// The last memory id the drain finished; the next batch starts after it.
    pub cursor: Option<MemoryId>,
    pub started_at: DateTime<Utc>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
    /// Memories the drain has walked past.
    #[serde(default)]
    pub walked: usize,
    /// Memories it re-embedded, or stamped.
    #[serde(default)]
    pub changed: usize,
    /// Memories it could not finish: unreadable, or the embedder refused them. They are left
    /// to the backlog.
    #[serde(default)]
    pub failed: usize,
    /// Memories in the store when the migration began.
    #[serde(default)]
    pub total: usize,
    #[serde(default)]
    pub last_error: Option<String>,
    /// Whether the drain has tried to map the new space back into the old one.
    #[serde(default)]
    pub bridge_tried: bool,
}

impl SpaceMigration {
    pub fn new(kind: MigrationKind, from: Option<SpaceId>, to: SpaceId, total: usize) -> Self {
        Self {
            kind,
            from,
            to,
            cursor: None,
            started_at: Utc::now(),
            finished_at: None,
            walked: 0,
            changed: 0,
            failed: 0,
            total,
            last_error: None,
            bridge_tried: false,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished_at.is_some()
    }
}

/// The record, in the tenant's own key-value store.
pub struct SpaceMigrationStore {
    kv: Arc<dyn KeyValueStore>,
}

impl SpaceMigrationStore {
    pub fn with_kv(kv: Arc<dyn KeyValueStore>) -> Self {
        Self { kv }
    }

    pub fn load(&self) -> Result<Option<SpaceMigration>> {
        match self.kv.get_bytes(KEY).context("read the space migration")? {
            Some(bytes) => Ok(Some(
                serde_json::from_slice(&bytes).context("decode the space migration")?,
            )),
            None => Ok(None),
        }
    }

    pub fn save(&self, migration: &SpaceMigration) -> Result<()> {
        self.kv
            .put_bytes(KEY, &serde_json::to_vec(migration)?)
            .context("write the space migration")
    }

    pub fn clear(&self) -> Result<()> {
        self.kv
            .delete_bytes(KEY)
            .context("delete the space migration")
    }
}

/// A migration's progress as the unauthenticated `/health` may show it: no cursor, no error
/// text, nothing that names a memory or a host.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MigrationProgress {
    pub kind: MigrationKind,
    pub from: Option<SpaceId>,
    pub to: SpaceId,
    pub walked: usize,
    pub total: usize,
    pub changed: usize,
    pub failed: usize,
    pub started_at: DateTime<Utc>,
    /// Whether the last batch stopped for want of an embedder.
    pub stalled: bool,
}

impl From<&SpaceMigration> for MigrationProgress {
    fn from(m: &SpaceMigration) -> Self {
        Self {
            kind: m.kind.clone(),
            from: m.from.clone(),
            to: m.to.clone(),
            walked: m.walked,
            total: m.total,
            changed: m.changed,
            failed: m.failed,
            started_at: m.started_at,
            stalled: m.last_error.is_some(),
        }
    }
}

/// What one drain tick of a migration did.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct MigrationStep {
    pub walked: usize,
    pub changed: usize,
    pub failed: usize,
    pub finished: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
}
