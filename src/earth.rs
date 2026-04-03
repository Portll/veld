//! Transitional Earth substrate API.
//!
//! This module establishes the intended `earth` boundary inside the existing
//! `shodh-memory` crate so new code can depend on the substrate API without
//! waiting for a repo or crate rename.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use anyhow::Result;

pub use crate::config::StorageBackend;
pub use crate::graph_memory::{EntityNode, EpisodicNode, GraphMemory, GraphStats, RelationshipEdge};
pub use crate::memory::storage::MemoryStorage;
pub use crate::memory::{Experience, Memory, MemoryConfig, MemoryId, MemoryStats};
pub use crate::storage::{
    AuditLogEntry, AuditStore, BackupStore, GraphStore, KeyValueStore, MigrationStore,
    PrimaryMemoryStore, StorageCapabilities, StorageFactory,
};

use crate::memory::MemorySystem;

/// Shared substrate handle used by the orchestration layer.
pub type SharedEarth = Arc<parking_lot::RwLock<Earth>>;

/// Stable substrate wrapper over [`MemorySystem`].
///
/// `earth` is the intended database and memory substrate boundary. During the
/// transition, this wrapper gives higher layers a clear type to depend on while
/// preserving the current implementation internally.
pub struct Earth {
    inner: MemorySystem,
}

impl Earth {
    /// Create a substrate using the current default storage bootstrap path.
    pub fn new(config: MemoryConfig, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        Ok(Self {
            inner: MemorySystem::new(config, shared_cache)?,
        })
    }

    /// Create a substrate with an already-opened primary store.
    pub fn with_storage(config: MemoryConfig, storage: Arc<MemoryStorage>) -> Result<Self> {
        Ok(Self {
            inner: MemorySystem::with_storage(config, storage)?,
        })
    }

    /// Return the wrapped implementation.
    pub fn into_inner(self) -> MemorySystem {
        self.inner
    }

    /// Borrow the wrapped implementation explicitly.
    pub fn as_memory_system(&self) -> &MemorySystem {
        &self.inner
    }

    /// Mutably borrow the wrapped implementation explicitly.
    pub fn as_memory_system_mut(&mut self) -> &mut MemorySystem {
        &mut self.inner
    }
}

impl Deref for Earth {
    type Target = MemorySystem;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Earth {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}