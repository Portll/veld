//! User memory substrate API.
//!
//! This module establishes the user-memory boundary inside the existing
//! `veld` crate so new code can depend on the substrate API without
//! waiting for a repo or crate rename.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use anyhow::Result;

pub use crate::config::StorageBackend;
pub use crate::graph_memory::{
    EntityNode, EpisodicNode, GraphMemory, GraphStats, RelationshipEdge,
};
pub use crate::memory::hybrid_search::BM25Index;
pub use crate::memory::storage::MemoryStorage;
pub use crate::memory::{Experience, Memory, MemoryConfig, MemoryId, MemoryStats};
pub use crate::storage::{
    AuditLogEntry, AuditStore, BackupStore, GraphStore, KeyValueStore, MigrationStore,
    PrimaryMemoryStore, StorageCapabilities, StorageFactory,
};

use crate::memory::MemorySystem;

/// Shared user-memory handle used by the orchestration layer.
pub type SharedUserMemory = Arc<parking_lot::RwLock<UserMemory>>;

/// Stable substrate wrapper over [`MemorySystem`].
///
/// `UserMemory` is the database and memory substrate boundary. During the
/// transition, this wrapper gives higher layers a clear type to depend on while
/// preserving the current implementation internally.
pub struct UserMemory {
    inner: MemorySystem,
}

impl UserMemory {
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

    /// Create a substrate with an already-opened primary store *and* an
    /// externally-owned `Arc<BM25Index>`.
    ///
    /// Used by the multi-user runtime so the substrate's
    /// retrieval-side BM25 reader shares the same tantivy handle the
    /// per-tenant `Bm25Projection` writes through from the intent log.
    /// Without this seam the read side would open a second on-disk index
    /// at a parallel path and diverge from the writer until the next
    /// process restart triggered a replay catch-up.
    pub fn with_storage_and_bm25_index(
        config: MemoryConfig,
        storage: Arc<MemoryStorage>,
        bm25_index: Arc<BM25Index>,
    ) -> Result<Self> {
        Ok(Self {
            inner: MemorySystem::with_storage_and_bm25_index(config, storage, bm25_index)?,
        })
    }

    /// Backend-agnostic construction — see
    /// [`MemorySystem::with_primary_store_and_bm25_index`].
    pub fn with_primary_store_and_bm25_index(
        config: MemoryConfig,
        primary_store: Arc<dyn crate::storage::PrimaryMemoryStore>,
        rocksdb_backup_db: Option<Arc<rocksdb::DB>>,
        bm25_index: Arc<BM25Index>,
    ) -> Result<Self> {
        Ok(Self {
            inner: MemorySystem::with_primary_store_and_bm25_index(
                config,
                primary_store,
                rocksdb_backup_db,
                bm25_index,
            )?,
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

impl Deref for UserMemory {
    type Target = MemorySystem;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for UserMemory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
