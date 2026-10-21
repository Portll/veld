//! Multi-User Runtime Manager - Core State Management
//!
//! This module contains the central state manager for the Veld server and server runtime.
//! It handles per-user UserMemory handles, graph stores, audit logs, and all
//! subsidiary stores (todos, reminders, files, etc.).

use anyhow::{Context, Result};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use tracing::info;

use crate::user_memory::{SharedUserMemory, UserMemory};
/// Static regex for extracting all-caps terms (API, TUI, NER, REST, etc.)
fn allcaps_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\b[A-Z]{2,}[A-Z0-9]*\b").unwrap())
}

/// Static regex for extracting issue IDs (SHO-XX, JIRA-123, etc.)
fn issue_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\b([A-Z]{2,10}-\d+)\b").unwrap())
}

use crate::ab_testing;
use crate::backup;
use crate::backup::BackupMetadata;
use crate::config::{ServerConfig, StorageBackend};
use crate::declared_entities::DeclaredEntity;
use crate::embeddings::{
    are_ner_models_downloaded, auto_download_models_enabled, download_ner_models,
    get_ner_models_dir, ner::NerEntityType, neural_ner_enabled, KeywordExtractor, NerConfig,
    NeuralNer,
};
#[cfg(feature = "multi-tenant")]
use crate::extensions::maintenance;
use crate::graph_memory::{
    EdgeTier, EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, GraphStats,
    LtpStatus, RelationType, RelationshipEdge,
};
use crate::memory::{
    query_parser, Experience, FeedbackStore, FileMemoryStore, MemoryConfig, MemoryId, MemoryStats,
    ProspectiveStore, SessionStore, TodoStore,
};
use crate::relevance::RelevanceEngine;
#[cfg(feature = "rocksdb")]
use crate::storage::legacy_rocksdb::{
    import_separate_db_into_cf, RocksDbAuditStore, RocksDbColumnFamilyStore, RocksDbColumnStore,
    RocksDbGraphStore, RocksDbPrimaryMemoryStore,
};
use crate::storage::{AuditLogEntry, AuditStore, StorageCapabilities};
use crate::streaming;

use super::types::{AuditEvent, ContextStatus, MemoryEvent};

/// Type alias for context sessions map
pub type ContextSessions = DashMap<String, ContextStatus>;

struct SharedStoreBootstrap {
    shared_db: crate::storage::SharedEngineDb,
    audit_store: Arc<dyn AuditStore>,
    prospective_store: Arc<ProspectiveStore>,
    todo_store: Arc<TodoStore>,
    context_block_store: Arc<crate::memory::ContextBlockStore>,
    file_store: Arc<FileMemoryStore>,
    feedback_store: Arc<parking_lot::RwLock<FeedbackStore>>,
    #[cfg(feature = "storage-redb")]
    audit_redb: Option<Arc<::redb::Database>>,
}

/// Under `<base>/backups/`; a leading dot is not a valid user id, so no rocksdb user's archive can share it.
#[cfg(feature = "storage-redb")]
const REDB_BACKUP_DIR: &str = ".redb";

/// Helper struct for audit log rotation (allows spawn_blocking with minimal clone)
struct MultiUserMemoryManagerRotationHelper {
    audit_store: Arc<dyn AuditStore>,
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,
    audit_retention_days: i64,
    audit_max_entries: usize,
}

#[cfg_attr(not(feature = "rocksdb"), allow(dead_code))]
const CF_AUDIT: &str = "audit";

impl From<&AuditEvent> for AuditLogEntry {
    fn from(event: &AuditEvent) -> Self {
        Self {
            timestamp: event.timestamp,
            event_type: event.event_type.clone(),
            memory_id: event.memory_id.clone(),
            details: event.details.clone(),
        }
    }
}

impl From<AuditLogEntry> for AuditEvent {
    fn from(event: AuditLogEntry) -> Self {
        Self {
            timestamp: event.timestamp,
            event_type: event.event_type,
            memory_id: event.memory_id,
            details: event.details,
        }
    }
}

impl MultiUserMemoryManagerRotationHelper {
    /// Rotate audit logs for a user - delete old entries and enforce max count.
    ///
    /// Keys are `{user_id}:{timestamp_nanos:020}` so RocksDB returns them in
    /// ascending timestamp order. Two strategies depending on scale:
    /// - ≤100K keys: collect all, compute excess, batch delete
    /// - >100K keys: streaming 2-pass (count, then delete) to avoid OOM
    fn rotate_user_audit_logs(&self, user_id: &str) -> Result<usize> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(self.audit_retention_days);
        let cutoff_nanos = cutoff_time.timestamp_nanos_opt().unwrap_or_else(|| {
            tracing::warn!("audit cutoff timestamp outside i64 nanos range, using 0");
            0
        });
        let removed_count =
            self.audit_store
                .rotate_events(user_id, self.audit_max_entries, cutoff_time)?;

        // Sync in-memory cache
        if removed_count > 0 {
            if let Some(log) = self.audit_logs.get(user_id) {
                let mut log_guard = log.write();

                log_guard.retain(|event| {
                    let event_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    event_nanos >= cutoff_nanos
                });

                while log_guard.len() > self.audit_max_entries {
                    log_guard.pop_front();
                }
            }
        }

        Ok(removed_count)
    }
}

/// Multi-user Veld manager for the server runtime.
pub struct MultiUserMemoryManager {
    /// Per-user UserMemory handles with LRU eviction.
    pub user_memories: moka::sync::Cache<String, SharedUserMemory>,

    /// Per-user audit logs (in-memory cache)
    pub audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,

    /// Shared engine database for all global stores (todos, reminders, files,
    /// feedback, audit), carried by ENGINE. Engine-typed consumers (user-auth,
    /// GDPR purge, comprehensive backup) go through
    /// `require_rocksdb(..)` and refuse on the other engine by name.
    pub shared_db: crate::storage::SharedEngineDb,

    /// Tenants with data on each backend, measured by the boot guard.
    pub backend_census: crate::storage::boot_guard::BackendCensus,

    /// Audit log storage routed through the backend abstraction layer.
    audit_store: Arc<dyn AuditStore>,

    /// Base storage path
    pub base_path: std::path::PathBuf,

    /// Default config
    pub default_config: MemoryConfig,

    /// Counter for audit log rotation checks
    pub audit_log_counter: Arc<std::sync::atomic::AtomicUsize>,

    /// Per-user graph stores
    pub graph_memories: moka::sync::Cache<String, Arc<parking_lot::RwLock<GraphMemory>>>,

    /// Neural NER for automatic entity extraction
    pub neural_ner: Arc<NeuralNer>,

    /// Statistical keyword extraction for graph population
    pub keyword_extractor: Arc<KeywordExtractor>,

    /// User eviction counter for metrics
    pub user_evictions: Arc<std::sync::atomic::AtomicUsize>,

    /// Server configuration
    pub server_config: ServerConfig,

    /// SSE event broadcaster for real-time dashboard updates
    pub event_broadcaster: tokio::sync::broadcast::Sender<MemoryEvent>,

    /// Streaming memory extractor for implicit learning
    pub streaming_extractor: Arc<streaming::StreamingMemoryExtractor>,

    /// Prospective memory store for reminders/intentions
    pub prospective_store: Arc<ProspectiveStore>,

    /// GTD-style todo store
    pub todo_store: Arc<TodoStore>,

    /// Agent context block store (Letta-style mutable key-value blocks)
    pub context_block_store: Arc<crate::memory::ContextBlockStore>,

    /// File memory store for codebase integration
    pub file_store: Arc<FileMemoryStore>,

    /// Implicit feedback store for memory reinforcement
    pub feedback_store: Arc<parking_lot::RwLock<FeedbackStore>>,

    /// Backup engine for automated and manual backups
    /// None when the backend is not rocksdb: the BackupEngine is a rocksdb
    /// facility, and every consumer goes through `require_backup_engine(..)`
    /// so an absent engine refuses by operation name instead of existing in a
    /// state where any call would be wrong.
    pub backup_engine: Option<Arc<backup::VeldBackupEngine>>,

    /// None on rocksdb.
    #[cfg(feature = "storage-redb")]
    pub redb_backup_engine: Option<Arc<crate::storage::migrate::backup::RedbBackupEngine>>,

    /// The redb backup reads audit.redb through this: a second opener fails on its lock.
    #[cfg(feature = "storage-redb")]
    audit_redb: Option<Arc<::redb::Database>>,

    /// Context status from Claude Code sessions
    pub context_sessions: Arc<ContextSessions>,

    /// SSE broadcaster for context status updates
    pub context_broadcaster: tokio::sync::broadcast::Sender<ContextStatus>,

    /// A/B testing manager for relevance scoring experiments
    pub ab_test_manager: Arc<ab_testing::ABTestManager>,

    /// Session tracking store
    pub session_store: Arc<SessionStore>,

    /// Shared relevance engine for proactive memory surfacing (entity cache + learned weights persist)
    pub relevance_engine: Arc<RelevanceEngine>,

    /// Maintenance cycle counter: cycles 0..5 are lightweight (in-memory only),
    /// cycle 0 (mod 6) is heavyweight (graph decay, fact extraction, flush).
    /// At 300s intervals, heavy cycles fire every 30 minutes.
    maintenance_cycle: std::sync::atomic::AtomicU64,

    /// Per-user creation locks to prevent TOCTOU races in get_user_memory.
    /// Without this, concurrent first-access requests for the same user_id can both
    /// miss the cache check, both try to open RocksDB, and the second open fails
    /// because RocksDB holds an exclusive file lock.
    user_memory_init_locks: DashMap<String, Arc<parking_lot::Mutex<()>>>,

    /// Separate per-user creation locks for graph memory.
    /// Must be separate from user_memory_init_locks because get_user_memory()
    /// calls get_user_graph() while holding its lock, and parking_lot::Mutex
    /// is not re-entrant — sharing a single lock map would deadlock.
    user_graph_init_locks: DashMap<String, Arc<parking_lot::Mutex<()>>>,

    #[cfg_attr(not(feature = "rocksdb"), allow(dead_code))]
    /// Shared RocksDB block cache across all per-user DB instances.
    /// Single LRU cache provides a hard memory ceiling regardless of user count.
    /// Without this, each user's MemoryStorage + GraphMemory allocates ~96MB in
    /// independent caches — 6 users = 576MB just in block caches alone.
    shared_rocksdb_cache: Option<crate::storage::SharedBlockCache>,

    /// Per-user SlowStore cache. Avoids reopening SQLite on every gap analysis call,
    /// and ensures the sync TTL actually works across requests.
    pub slow_stores: DashMap<String, std::sync::Arc<crate::memory::slow_store::SlowStore>>,

    /// Per-user BM25 (tantivy) index cache. Shared between the live
    /// [`crate::memory::Bm25Projection`] (which writes through it from
    /// the intent log) and any read-side hybrid-search reader that
    /// needs to query the same documents. Wrapped in `Arc` because the
    /// index's interior writer and reader are already lock-protected;
    /// the outer pointer just amortises the open cost.
    pub bm25_indices: DashMap<String, std::sync::Arc<crate::memory::BM25Index>>,

    /// Per-user [`JournaledWriter`]. Created lazily on first journaled
    /// CRUD for that user; holds the open intent log, the per-user
    /// `CheckpointStore`, and the live `SqliteProjection` attached for
    /// dispatch. Wrapped in `Mutex` because `record_and_apply` mutates
    /// the underlying log + projections — only one writer at a time per
    /// tenant.
    ///
    /// Wired up here (and not in the per-user UserMemory) because the
    /// projection list is a runtime construct that may grow over time
    /// (Vamana, BM25, Postgres) and we want one place to register them.
    pub journaled_writers:
        DashMap<String, std::sync::Arc<parking_lot::Mutex<crate::intent_log::JournaledWriter>>>,

    /// One init-lock per tenant for the lazy `JournaledWriter` creation.
    /// Without this, two concurrent first-touch requests for the same
    /// user could both miss the cache, both call `IntentLog::open`, and
    /// the second open would race with the first writer's append.
    journaled_writer_init_locks: DashMap<String, Arc<parking_lot::Mutex<()>>>,

    /// C2 (W-001): set of users whose store↔log reconcile has already run
    /// in this process. The reconcile is a one-time-per-boot scan on first
    /// access; this guards against re-running it on every cache miss.
    reconciled_users: DashMap<String, ()>,

    /// Set by [`Self::close_all_writes`]; a tenant opened after it starts closed.
    writes_closed: std::sync::atomic::AtomicBool,

    /// Callers that missed the tenant cache and do not have their tenant yet: the one
    /// opening it and any queued on its init lock. Non-zero means a tenant is opening,
    /// and `/health/ready` answers 503 until it has.
    tenant_open_waiters: std::sync::atomic::AtomicUsize,

    /// (b) Live wiki edit-back watchers, one per user, kept alive here so the
    /// `notify` watcher keeps running (dropping the handle stops it). The
    /// `Mutex` only satisfies `Sync` — the value is a keep-alive handle, never
    /// locked. Populated lazily when both `VELD_WIKI_RENDER` and
    /// `VELD_WIKI_WATCH` are on.
    wiki_watchers: DashMap<String, parking_lot::Mutex<notify::RecommendedWatcher>>,

    /// Capabilities for the effective backend in the current runtime.
    storage_capabilities: StorageCapabilities,

    /// Phase C user-auth runtime. `Some` iff `VELD_USER_AUTH_ENABLED` was
    /// truthy at server startup. Handlers gate on this — when `None`, the
    /// /api/user_auth/* surface returns 404.
    pub user_auth_runtime: Option<crate::user_auth::runtime::UserAuthRuntime>,

    /// Passkey enrolments waiting for the browser (`handlers::keys`).
    pub passkey_tickets: super::keys::PasskeyTickets,

    /// Optional relational-backed dataset store (W7). `None` when no
    /// relational backend has been wired (the v0.9 redb-cutover work will
    /// flip this to mandatory). Handlers behind `/api/datasets/*` return
    /// `503 Service Unavailable` when this is `None`.
    pub dataset_store: Option<std::sync::Arc<dyn crate::datasets::DatasetStore>>,

    /// Optional row-link store backed by the same relational engine as
    /// [`Self::dataset_store`]. Independently optional: present only when
    /// the relational store is wired and link persistence is enabled.
    pub link_store: Option<std::sync::Arc<dyn crate::datasets::LinkStore>>,

    /// Optional raw relational executor shared with [`Self::dataset_store`]
    /// and [`Self::link_store`]. The `/api/datasets/{name}/query` endpoint
    /// uses this to run parametrised SELECTs directly against the same
    /// database, since the [`crate::datasets::DatasetStore`] trait does not
    /// expose raw SELECT.
    pub dataset_executor: Option<
        std::sync::Arc<
            dyn crate::storage::relational::RelationalStore<
                Error = crate::storage::relational::BoxError,
            >,
        >,
    >,

    /// Sleep-time / observational memory orchestrator (V1). `Some` only
    /// when [`crate::config::SleepTimeConfig::enabled`] is true at startup
    /// AND the API key environment variable resolves. Handlers gate on
    /// this — when `None`, the /api/sleep_time/* surface returns 503
    /// Service Unavailable with a clear reason.
    pub sleep_time_orchestrator:
        parking_lot::RwLock<Option<Arc<crate::memory::sleep_time::SleepTimeOrchestrator>>>,

    /// Fire-and-forget writes spawned on behalf of requests (graph processing, deferred
    /// embedding, audit persistence). Shutdown closes it and waits, so the stores are not
    /// flushed or released under a write still in flight.
    pub task_tracker: tokio_util::task::TaskTracker,
}

/// Written into a tenant's directory by a backup restore and removed by the C2
/// reconcile that follows. A backup stages the journal before checkpointing the
/// primary store, so a memory deleted between the two is live in the restored log
/// but absent from the restored store; for a restored tenant that is an
/// acknowledged delete, not a lost write.
const RESTORED_FROM_BACKUP_MARKER: &str = ".restored-from-backup";

/// User directories under the data root. An unreadable entry is an error, not a
/// skipped user: the boot guard must see every tenant or refuse.
pub(crate) fn list_user_dirs(base_path: &std::path::Path) -> std::io::Result<Vec<String>> {
    let mut users = Vec::new();
    for entry in std::fs::read_dir(base_path)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            if !crate::validation::is_reserved_data_dir(name) {
                users.push(name.to_string());
            }
        }
    }
    users.sort();
    Ok(users)
}

/// One count in `MultiUserMemoryManager::tenant_open_waiters` for as long as it lives, so
/// every way out of `get_user_memory`'s slow path, errors included, gives it back.
struct TenantOpenWaiter<'a>(&'a std::sync::atomic::AtomicUsize);

impl<'a> TenantOpenWaiter<'a> {
    fn enter(waiters: &'a std::sync::atomic::AtomicUsize) -> Self {
        waiters.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self(waiters)
    }
}

impl Drop for TenantOpenWaiter<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl MultiUserMemoryManager {
    pub fn new(base_path: std::path::PathBuf, server_config: ServerConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;
        let backend_census = crate::storage::boot_guard::refuse_stranded_data(
            &base_path,
            server_config.effective_storage_backend,
            &list_user_dirs(&base_path)?,
        )?;

        let (event_broadcaster, _) = tokio::sync::broadcast::channel(1024);

        let ner_dir = get_ner_models_dir();
        let ner_config = NerConfig::from_env();
        let local_ner_available =
            ner_config.model_path.exists() && ner_config.tokenizer_path.exists();
        let neural_ner = if !neural_ner_enabled() {
            tracing::info!(
                "Neural NER disabled (set VELD_NEURAL_NER=true to enable local TinyBERT NER)"
            );
            Arc::new(NeuralNer::new_fallback(NerConfig::default()))
        } else if local_ner_available || are_ner_models_downloaded() {
            tracing::debug!("NER models found locally at {:?}", ner_dir);
            match NeuralNer::new(ner_config.clone()) {
                Ok(ner) => {
                    info!(
                        "Neural NER initialized (TinyBERT model at {:?})",
                        ner_config.model_path
                    );
                    Arc::new(ner)
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize neural NER: {}. Using fallback.", e);
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        } else if auto_download_models_enabled() {
            tracing::info!("Downloading NER models (TinyBERT-NER, ~15MB)...");
            match download_ner_models(Some(std::sync::Arc::new(|downloaded, total| {
                if total > 0 {
                    let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                    if percent.is_multiple_of(20) {
                        tracing::info!("NER model download: {}%", percent);
                    }
                }
            }))) {
                Ok(ner_dir) => {
                    let downloaded_config = NerConfig {
                        model_path: ner_dir.join("model.onnx"),
                        tokenizer_path: ner_dir.join("tokenizer.json"),
                        max_length: ner_config.max_length,
                        confidence_threshold: ner_config.confidence_threshold,
                    };
                    match NeuralNer::new(downloaded_config) {
                        Ok(ner) => {
                            info!("Neural NER initialized after download");
                            Arc::new(ner)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to initialize downloaded NER: {}. Using fallback.",
                                e
                            );
                            Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to download NER models: {}. Using rule-based fallback.",
                        e
                    );
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        } else {
            tracing::info!(
                "Neural NER enabled but no local models found and VELD_AUTO_DOWNLOAD_MODELS is not enabled. Using rule-based fallback."
            );
            Arc::new(NeuralNer::new_fallback(NerConfig::default()))
        };

        let user_evictions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let evictions_clone = user_evictions.clone();
        let max_cache = server_config.max_users_in_memory;
        let eviction_base_path = base_path.clone();
        #[cfg(feature = "multi-tenant")]
        let eviction_multi_tenant = server_config.multi_tenant_mode;

        let user_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .time_to_idle(std::time::Duration::from_secs(3600))
            .eviction_listener(move |key: Arc<String>, value: SharedUserMemory, cause| {
                if matches!(cause, moka::notification::RemovalCause::Size | moka::notification::RemovalCause::Expired) {
                    evictions_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    let cause_label = if cause == moka::notification::RemovalCause::Expired { "idle-timeout" } else { "LRU" };

                    // Spawn blocking task to persist vector index without holding the lock
                    // during I/O. The eviction listener runs synchronously inside moka,
                    // so we must not block here for disk writes.
                    //
                    // CRITICAL: We must drop the Arc<RwLock<UserMemory>> as soon as
                    // possible after saving, otherwise the RocksDB file lock is held
                    // until the thread exits. If a new request arrives for the same user
                    // while the lock is held, MemorySystem::new() fails with a lock error.
                    let index_path = eviction_base_path.join(key.as_str()).join("vector_index");
                        #[cfg(feature = "multi-tenant")]
                        let user_path = eviction_base_path.join(key.as_str());
                    let user_key = key.clone();
                    std::thread::spawn(move || {
                        // Scope the read guard so it drops before we drop the Arc.
                        // This ensures the RocksDB file lock is released promptly.
                        let save_result = {
                            if let Some(guard) = value.try_read() {
                                let result = guard.save_vector_index(&index_path);
                                #[cfg(feature = "multi-tenant")]
                                if eviction_multi_tenant {
                                    let weights = guard.learned_weight_state();
                                    let _ = maintenance::persist_learned_weights(
                                        &user_path,
                                        weights.bm25,
                                        weights.vector,
                                        weights.graph,
                                        weights.update_count,
                                    );
                                }
                                Some(result)
                            } else {
                                None
                            }
                        };
                        // Arc dropped here — releases MemorySystem and RocksDB handle
                        drop(value);
                        match save_result {
                            Some(Ok(())) => {
                                info!(
                                    "Evicted user '{}' from memory cache ({}, cache_size={}) - vector index saved",
                                    user_key, cause_label, max_cache
                                );
                            }
                            Some(Err(e)) => {
                                tracing::warn!(
                                    "Evicted user '{}' from memory cache ({}) - failed to save vector index: {}",
                                    user_key, cause_label, e
                                );
                            }
                            None => {
                                tracing::warn!(
                                    "Evicted user '{}' from memory cache ({}) - could not acquire lock to save index",
                                    user_key, cause_label
                                );
                            }
                        }
                    });
                }
            })
            .build();

        let graph_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .time_to_idle(std::time::Duration::from_secs(3600))
            .eviction_listener(move |key: Arc<String>, _value, cause| {
                let cause_label = if cause == moka::notification::RemovalCause::Expired {
                    "idle-timeout"
                } else {
                    "LRU"
                };
                info!(
                    "Evicted graph for user '{}' from memory cache ({})",
                    key, cause_label
                );
            })
            .build();

        // Single shared LRU block cache for ALL RocksDB instances (per-user memory DBs,
        // per-user graph DBs, and the global shared DB). Provides a hard memory ceiling
        // regardless of how many users are active. Without this, each user allocates
        // ~96MB in independent caches — the shared cache collapses that to a single
        // 256MB pool with LRU eviction of the coldest blocks across all users.
        #[cfg(feature = "rocksdb")]
        let shared_rocksdb_cache = {
            let cache_bytes =
                crate::config::StorageTuning::process().rocksdb_shared_block_cache_bytes();
            let cache = rocksdb::Cache::new_lru_cache(cache_bytes);
            info!(
                "Shared RocksDB block cache initialized ({}MB)",
                cache_bytes / (1024 * 1024)
            );
            Some(cache)
        };
        #[cfg(not(feature = "rocksdb"))]
        let shared_rocksdb_cache: Option<crate::storage::SharedBlockCache> = None;

        let shared_stores = Self::bootstrap_shared_stores(
            &base_path,
            &server_config,
            shared_rocksdb_cache.as_ref(),
        )?;

        info!("Todo store initialized");

        let backup_engine = Self::open_backup_engine(&base_path, &server_config)?;
        #[cfg(feature = "storage-redb")]
        let redb_backup_engine = match server_config.effective_storage_backend {
            StorageBackend::Redb => Some(Arc::new(
                crate::storage::migrate::backup::RedbBackupEngine::new(
                    base_path.join("backups").join(REDB_BACKUP_DIR),
                )?,
            )),
            StorageBackend::RocksDb => None,
        };
        if server_config.backup_enabled {
            info!(
                "Backup engine initialized (interval: {}h, keep: {})",
                server_config.backup_interval_secs / 3600,
                server_config.backup_max_count
            );
        } else {
            info!("Backup engine initialized (auto-backup disabled)");
        }

        // PIPE-9: StreamingMemoryExtractor no longer needs FeedbackStore
        // Feedback momentum is now applied in the MemorySystem pipeline
        let task_tracker = tokio_util::task::TaskTracker::new();
        let streaming_extractor = Arc::new(
            streaming::StreamingMemoryExtractor::new(neural_ner.clone())
                .with_task_tracker(task_tracker.clone()),
        );
        info!("Streaming memory extractor initialized");

        let keyword_extractor = Arc::new(KeywordExtractor::new());
        info!("Keyword extractor initialized (YAKE)");

        let relevance_engine = Arc::new(RelevanceEngine::new(neural_ner.clone()));
        info!("Relevance engine initialized (entity cache + learned weights)");

        let broadcast_capacity = (server_config.max_users_in_memory * 4).max(64);
        let storage_capabilities =
            StorageCapabilities::for_backend(server_config.effective_storage_backend);

        // Phase C — initialise the user-auth runtime only when the feature
        // flag was truthy at startup. On rocksdb the shared DB declared the
        // user_auth CF at open under the same flag; on redb the table is created.
        let user_auth_runtime = if crate::user_auth::feature_enabled() {
            match crate::user_auth::store::UserAuthStore::with_store(
                shared_stores.shared_db.column_store(),
            ) {
                Ok(store) => {
                    // A configured keystore that cannot be used refuses the start, as the
                    // record store does: starting without an encryptor stored TOTP secrets
                    // in plaintext outside production mode.
                    let totp_encryptor = crate::keystore::SecretCryptor::from_env("totp")
                        .context(
                            "keystore (VELD_KEYSTORE_PATH/VELD_MASTER_PASSPHRASE) is configured but \
                             unusable; refusing to start user-auth without an encryptor",
                        )?
                        .or_else(|| crate::keyring::installed_cryptor("totp"));
                    info!(
                        encryption = totp_encryptor.is_some(),
                        "user-auth runtime initialised (VELD_USER_AUTH_ENABLED=true)"
                    );
                    Some(crate::user_auth::runtime::UserAuthRuntime::new(
                        store,
                        totp_encryptor,
                    ))
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "VELD_USER_AUTH_ENABLED=true but UserAuthStore failed to open; surface remains disabled"
                    );
                    None
                }
            }
        } else {
            None
        };

        let manager = Self {
            user_memories,
            audit_logs: Arc::new(DashMap::new()),
            shared_db: shared_stores.shared_db,
            backend_census,
            audit_store: shared_stores.audit_store,
            base_path,
            default_config: MemoryConfig {
                collective_store_dir: if server_config.multi_tenant_mode {
                    Some(server_config.collective_store_dir.clone())
                } else {
                    None
                },
                auto_compress: server_config.auto_compress,
                ..MemoryConfig::default()
            },
            audit_log_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            graph_memories,
            neural_ner,
            keyword_extractor,
            user_evictions,
            server_config,
            event_broadcaster,
            streaming_extractor,
            prospective_store: shared_stores.prospective_store,
            todo_store: shared_stores.todo_store,
            context_block_store: shared_stores.context_block_store,
            file_store: shared_stores.file_store,
            feedback_store: shared_stores.feedback_store,
            backup_engine,
            #[cfg(feature = "storage-redb")]
            redb_backup_engine,
            #[cfg(feature = "storage-redb")]
            audit_redb: shared_stores.audit_redb,
            context_sessions: Arc::new(DashMap::new()),
            context_broadcaster: {
                let (tx, _) = tokio::sync::broadcast::channel(broadcast_capacity);
                tx
            },
            ab_test_manager: Arc::new(ab_testing::ABTestManager::new()),
            session_store: Arc::new(SessionStore::new()),
            relevance_engine,
            maintenance_cycle: std::sync::atomic::AtomicU64::new(0),
            user_memory_init_locks: DashMap::new(),
            user_graph_init_locks: DashMap::new(),
            shared_rocksdb_cache,
            slow_stores: DashMap::new(),
            bm25_indices: DashMap::new(),
            journaled_writers: DashMap::new(),
            journaled_writer_init_locks: DashMap::new(),
            reconciled_users: DashMap::new(),
            writes_closed: std::sync::atomic::AtomicBool::new(false),
            tenant_open_waiters: std::sync::atomic::AtomicUsize::new(0),
            wiki_watchers: DashMap::new(),
            storage_capabilities,
            user_auth_runtime,
            passkey_tickets: Default::default(),
            // W7: dataset / link stores are opt-in until the relational
            // backend is wired into server bootstrap. Tests construct
            // these in-place via `with_dataset_stores`.
            dataset_store: None,
            link_store: None,
            dataset_executor: None,
            // Sleep-time orchestrator is wired post-construction so its
            // startup (cold-start purge + spawn workers) can run after the
            // rest of state is settled and a tokio runtime is available.
            // See `set_sleep_time_orchestrator()`.
            sleep_time_orchestrator: parking_lot::RwLock::new(None),
            task_tracker,
        };

        info!("Running initial audit log rotation...");
        if let Err(e) = manager.rotate_all_audit_logs() {
            tracing::warn!("Failed to rotate audit logs on startup: {}", e);
        }

        Ok(manager)
    }

    /// Builder-style helper for tests and bootstrap code that has already
    /// constructed the relational [`crate::datasets::DatasetStore`] and
    /// [`crate::datasets::LinkStore`] and wants to attach them to the
    /// running manager.
    ///
    /// Production bootstrap will set both at construction time once the
    /// relational engine is wired in (W4 → W7 follow-up); until then the
    /// fields default to `None` and the dataset HTTP surface returns
    /// `503 Service Unavailable`.
    pub fn with_dataset_stores(
        mut self,
        dataset_store: std::sync::Arc<dyn crate::datasets::DatasetStore>,
        link_store: std::sync::Arc<dyn crate::datasets::LinkStore>,
        dataset_executor: std::sync::Arc<
            dyn crate::storage::relational::RelationalStore<
                Error = crate::storage::relational::BoxError,
            >,
        >,
    ) -> Self {
        self.dataset_store = Some(dataset_store);
        self.link_store = Some(link_store);
        self.dataset_executor = Some(dataset_executor);
        self
    }

    fn bootstrap_shared_stores(
        base_path: &std::path::Path,
        server_config: &ServerConfig,
        #[cfg_attr(not(feature = "rocksdb"), allow(unused_variables))] shared_rocksdb_cache: Option<
            &crate::storage::SharedBlockCache,
        >,
    ) -> Result<SharedStoreBootstrap> {
        match server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => {
                let shared_db = Self::open_legacy_shared_db(base_path, shared_rocksdb_cache)?;

                Self::migrate_audit_db(base_path, &shared_db)?;

                let audit_store: Arc<dyn AuditStore> =
                    Arc::new(RocksDbAuditStore::new(shared_db.clone(), CF_AUDIT));

                // Kept on boot: nothing shows every install already imported its pre-column-family DBs.
                for (dir, cf) in ProspectiveStore::LEGACY_SEPARATE_DBS
                    .iter()
                    .chain(TodoStore::LEGACY_SEPARATE_DBS)
                    .chain(FileMemoryStore::LEGACY_SEPARATE_DBS)
                {
                    import_separate_db_into_cf(&shared_db, &base_path.join(dir), cf)?;
                }

                let column: Arc<dyn crate::storage::ColumnStore> =
                    Arc::new(RocksDbColumnStore::new(Arc::clone(&shared_db)));

                let prospective_store =
                    Arc::new(ProspectiveStore::with_store(Arc::clone(&column))?);
                info!("Prospective memory store initialized");

                let todo_store = Arc::new(TodoStore::with_store(Arc::clone(&column), base_path)?);

                let file_store = Arc::new(FileMemoryStore::with_store(Arc::clone(&column))?);
                info!("File memory store initialized");

                let context_block_store = Arc::new(crate::memory::ContextBlockStore::with_store(
                    Arc::new(RocksDbColumnFamilyStore::new(
                        shared_db.clone(),
                        crate::memory::context_blocks::CF_CONTEXT_BLOCKS,
                    )),
                ));
                info!("Context block store initialized");

                // Feedback is advisory: an import or load failure degrades to in-memory rather than refusing boot.
                let feedback_store = Arc::new(parking_lot::RwLock::new(
                    FeedbackStore::LEGACY_SEPARATE_DBS
                        .iter()
                        .try_for_each(|(dir, cf)| {
                            import_separate_db_into_cf(&shared_db, &base_path.join(dir), cf)
                        })
                        .and_then(|()| {
                            FeedbackStore::with_store(
                                Arc::new(RocksDbColumnFamilyStore::new(
                                    shared_db.clone(),
                                    crate::memory::feedback::CF_FEEDBACK,
                                )),
                                base_path,
                            )
                        })
                        .unwrap_or_else(|e| {
                            tracing::warn!("Failed to load feedback store: {}, using in-memory", e);
                            FeedbackStore::new()
                        }),
                ));
                info!("Feedback store initialized");

                Ok(SharedStoreBootstrap {
                    shared_db: crate::storage::SharedEngineDb::RocksDb(shared_db),
                    audit_store,
                    prospective_store,
                    todo_store,
                    context_block_store,
                    file_store,
                    feedback_store,
                    #[cfg(feature = "storage-redb")]
                    audit_redb: None,
                })
            }
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("the shared stores")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => {
                let database = crate::storage::redb::open_shared_database(base_path)?;

                // Audit is its own audit.redb, mirroring the rocksdb layout where
                // audit rows live apart from user data.
                let audit = crate::storage::redb::RedbAuditStore::open(base_path)?;
                let audit_redb = Some(audit.database());
                let audit_store: Arc<dyn AuditStore> = Arc::new(audit);

                // ONE ColumnStore over ONE database handle — every multi-family
                // store shares it (redb permits one opener per file), and each
                // with_store declares its families via ensure_cf.
                let column: Arc<dyn crate::storage::ColumnStore> = Arc::new(
                    crate::storage::redb::RedbColumnStore::new(Arc::clone(&database)),
                );

                let prospective_store =
                    Arc::new(ProspectiveStore::with_store(Arc::clone(&column))?);
                info!("Prospective memory store initialized (redb)");

                let todo_store = Arc::new(TodoStore::with_store(Arc::clone(&column), base_path)?);

                let file_store = Arc::new(FileMemoryStore::with_store(Arc::clone(&column))?);
                info!("File memory store initialized (redb)");

                let context_block_store = Arc::new(crate::memory::ContextBlockStore::with_store(
                    Arc::new(crate::storage::redb::RedbTableKeyValueStore::new(
                        Arc::clone(&database),
                        "context_blocks",
                    )),
                ));
                info!("Context block store initialized (redb)");

                // Same degrade-to-in-memory shape as the rocksdb arm: feedback is
                // advisory state, and the rocksdb arm already treats a load
                // failure as warn-and-continue rather than boot failure.
                let feedback_store = Arc::new(parking_lot::RwLock::new(
                    FeedbackStore::with_store(
                        Arc::new(crate::storage::redb::RedbTableKeyValueStore::new(
                            Arc::clone(&database),
                            "feedback",
                        )),
                        base_path,
                    )
                    .unwrap_or_else(|e| {
                        tracing::warn!("Failed to load feedback store: {}, using in-memory", e);
                        FeedbackStore::new()
                    }),
                ));
                info!("Feedback store initialized (redb)");

                Ok(SharedStoreBootstrap {
                    shared_db: crate::storage::SharedEngineDb::Redb(database),
                    audit_store,
                    prospective_store,
                    todo_store,
                    context_block_store,
                    file_store,
                    feedback_store,
                    audit_redb,
                })
            }
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' requires this build to enable the storage-redb feature",
                server_config.effective_storage_backend
            )),
        }
    }

    #[cfg(feature = "rocksdb")]
    fn open_legacy_shared_db(
        base_path: &std::path::Path,
        shared_rocksdb_cache: Option<&rocksdb::Cache>,
    ) -> Result<Arc<rocksdb::DB>> {
        use rocksdb::{BlockBasedOptions, ColumnFamilyDescriptor, Options as RocksOptions};

        let shared_db_path = base_path.join("shared");
        std::fs::create_dir_all(&shared_db_path)?;

        let mut db_opts = RocksOptions::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        db_opts.set_max_write_buffer_number(2);
        db_opts.set_write_buffer_size(8 * 1024 * 1024);

        match shared_rocksdb_cache {
            Some(cache) => {
                db_opts.set_block_based_table_factory(
                    &crate::memory::storage::block_based_options(cache),
                );
                db_opts.set_write_buffer_manager(
                    crate::memory::storage::shared_write_buffer_manager(cache),
                );
            }
            None => {
                let mut block_opts = BlockBasedOptions::default();
                block_opts.set_cache_index_and_filter_blocks(true);
                db_opts.set_block_based_table_factory(&block_opts);
            }
        }

        let mut cfs = vec![ColumnFamilyDescriptor::new("default", {
            let mut o = RocksOptions::default();
            o.create_if_missing(true);
            if let Some(cache) = shared_rocksdb_cache {
                o.set_block_based_table_factory(&crate::memory::storage::block_based_options(
                    cache,
                ));
            }
            o
        })];
        cfs.extend(TodoStore::cf_descriptors());
        cfs.extend(ProspectiveStore::column_family_descriptors());
        cfs.extend(FileMemoryStore::cf_descriptors());
        cfs.extend(crate::memory::ContextBlockStore::cf_descriptors());
        // Sleep-time / observational memory CFs (V1 + V2). Always created
        // so the orchestrator can be enabled later without an offline migration.
        for cf in [
            crate::memory::sleep_time::CF_SLEEP_TIME_BUDGET,
            crate::memory::sleep_time::CF_SLEEP_TIME_GLOBAL_BUDGET,
            crate::memory::sleep_time::CF_SLEEP_TIME_QUEUE,
            crate::memory::sleep_time::CF_SLEEP_TIME_GRADUATIONS,
            crate::memory::sleep_time::CF_SLEEP_TIME_SUPERSESSIONS,
        ] {
            cfs.push(ColumnFamilyDescriptor::new(
                cf,
                Self::shared_store_cf_options(shared_rocksdb_cache),
            ));
        }
        cfs.push(ColumnFamilyDescriptor::new(
            crate::memory::feedback::CF_FEEDBACK,
            Self::shared_store_cf_options(shared_rocksdb_cache),
        ));
        cfs.push(ColumnFamilyDescriptor::new(
            CF_AUDIT,
            Self::shared_store_cf_options(shared_rocksdb_cache),
        ));

        // Phase C user auth: conditionally declare the user_auth CF. When
        // VELD_USER_AUTH_ENABLED is unset, the CF is never created on disk —
        // satisfies the spec requirement that disabling the flag leaves no
        // footprint.
        if crate::user_auth::feature_enabled() {
            cfs.push(crate::user_auth::store::cf_descriptor());
        }

        Ok(Arc::new(
            rocksdb::DB::open_cf_descriptors(&db_opts, &shared_db_path, cfs)
                .context("Failed to open shared DB with column families")?,
        ))
    }

    #[cfg(feature = "rocksdb")]
    fn shared_store_cf_options(
        cache: Option<&crate::storage::SharedBlockCache>,
    ) -> rocksdb::Options {
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        options.set_compression_type(rocksdb::DBCompressionType::Lz4);
        if let Some(cache) = cache {
            options
                .set_block_based_table_factory(&crate::memory::storage::block_based_options(cache));
        }
        options
    }

    fn open_backup_engine(
        base_path: &std::path::Path,
        server_config: &ServerConfig,
    ) -> Result<Option<Arc<backup::VeldBackupEngine>>> {
        let backup_path = base_path.join("backups");

        match server_config.effective_storage_backend {
            StorageBackend::RocksDb => {
                Ok(Some(Arc::new(backup::VeldBackupEngine::new(backup_path)?)))
            }
            // None: redb deployments route to redb_backup_engine instead.
            StorageBackend::Redb => Ok(None),
        }
    }

    /// The rocksdb backup engine, or a refusal that names the operation.
    fn require_backup_engine(&self, what: &str) -> Result<&Arc<backup::VeldBackupEngine>> {
        self.backup_engine.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "{what} requires the rocksdb backup engine; this deployment's backend is '{}' \
                 and its backups go through the redb backup engine",
                self.server_config.effective_storage_backend
            )
        })
    }

    /// The redb backup engine, or a refusal that names the operation.
    #[cfg(feature = "storage-redb")]
    fn require_redb_backup_engine(
        &self,
        what: &str,
    ) -> Result<&Arc<crate::storage::migrate::backup::RedbBackupEngine>> {
        self.redb_backup_engine.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "{what} requires the redb backup engine; this deployment's backend is '{}'",
                self.server_config.effective_storage_backend
            )
        })
    }

    #[cfg(feature = "rocksdb")]
    /// Migrate old standalone audit_logs DB into the shared DB's audit CF.
    /// Old directory is renamed to `audit_logs.pre_cf_migration` for rollback safety.
    fn migrate_audit_db(base_path: &std::path::Path, shared_db: &rocksdb::DB) -> Result<()> {
        let old_dir = base_path.join("audit_logs");
        if !old_dir.exists() {
            return Ok(());
        }

        let audit_cf = shared_db
            .cf_handle(CF_AUDIT)
            .expect("audit CF must exist in shared DB");

        // Check if CF already has data (migration already done)
        let mut has_data = false;
        let mut iter = shared_db.raw_iterator_cf(audit_cf);
        iter.seek_to_first();
        if iter.valid() {
            has_data = true;
        }
        if has_data {
            tracing::info!(
                "Audit CF already has data, skipping migration from {:?}",
                old_dir
            );
            return Ok(());
        }

        tracing::info!("Migrating audit_logs from standalone DB to shared DB audit CF...");

        let old_opts = rocksdb::Options::default();
        let old_db = rocksdb::DB::open_for_read_only(&old_opts, &old_dir, false)
            .context("Failed to open old audit_logs DB for migration")?;

        let write_opts = crate::storage::legacy_rocksdb::write_options(
            crate::config::StorageTuning::process().rocksdb_sync_writes(),
        );
        let mut batch = rocksdb::WriteBatch::default();
        let mut count = 0usize;
        const BATCH_SIZE: usize = 10_000;

        let iter = old_db.iterator(rocksdb::IteratorMode::Start);
        for item in iter {
            let (key, value) =
                item.map_err(|e| anyhow::anyhow!("audit migration iter error: {e}"))?;
            batch.put_cf(audit_cf, &key, &value);
            count += 1;

            if count.is_multiple_of(BATCH_SIZE) {
                shared_db
                    .write_opt(std::mem::take(&mut batch), &write_opts)
                    .map_err(|e| anyhow::anyhow!("audit migration batch write error: {e}"))?;
            }
        }

        if !count.is_multiple_of(BATCH_SIZE) {
            shared_db
                .write_opt(batch, &write_opts)
                .map_err(|e| anyhow::anyhow!("audit migration final batch error: {e}"))?;
        }

        drop(old_db);

        let renamed = old_dir.with_file_name("audit_logs.pre_cf_migration");
        if renamed.exists() {
            let _ = std::fs::remove_dir_all(&renamed);
        }
        std::fs::rename(&old_dir, &renamed)
            .context("Failed to rename old audit_logs dir after migration")?;

        tracing::info!(
            "Migrated {} audit entries from standalone DB to shared CF, old dir renamed to {:?}",
            count,
            renamed
        );

        Ok(())
    }

    /// Closes the task tracker and waits up to `timeout` for every tracked task. Returns false
    /// when the wait timed out. Tasks spawned after the close are still tracked and waited for.
    pub async fn drain_background_tasks(&self, timeout: std::time::Duration) -> bool {
        self.task_tracker.close();
        tokio::time::timeout(timeout, self.task_tracker.wait())
            .await
            .is_ok()
    }

    /// Log audit event (non-blocking with background persistence)
    pub fn log_event(&self, user_id: &str, event_type: &str, memory_id: &str, details: &str) {
        let event = AuditEvent {
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            memory_id: memory_id.to_string(),
            details: details.to_string(),
        };

        let audit_store = self.audit_store.clone();
        let storage_event = AuditLogEntry::from(&event);
        let user_id = user_id.to_string();
        let persisted_user_id = user_id.clone();

        // Tracked, not a bare spawn_blocking: the task holds its own handle on the audit store,
        // so an untracked write outlives a drain and keeps the store open after the manager drops.
        self.task_tracker.spawn_blocking(move || {
            if let Err(e) = audit_store.append_event(&persisted_user_id, &storage_event) {
                tracing::error!("Failed to persist audit log: {}", e);
            }
        });

        let max_entries = self.server_config.audit_max_entries_per_user;
        let log = self
            .audit_logs
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::RwLock::new(VecDeque::new())))
            .clone();
        {
            let mut entries = log.write();
            entries.push_back(event);
            while entries.len() > max_entries {
                entries.pop_front();
            }
        }

        let count = self
            .audit_log_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count.is_multiple_of(self.server_config.audit_rotation_check_interval) && count > 0 {
            let audit_store = self.audit_store.clone();
            let audit_logs = self.audit_logs.clone();
            let user_id_clone = user_id.to_string();

            let audit_retention_days = self.server_config.audit_retention_days as i64;
            let audit_max_entries = self.server_config.audit_max_entries_per_user;

            self.task_tracker.spawn_blocking(move || {
                let manager = MultiUserMemoryManagerRotationHelper {
                    audit_store,
                    audit_logs,
                    audit_retention_days,
                    audit_max_entries,
                };
                if let Err(e) = manager.rotate_user_audit_logs(&user_id_clone) {
                    tracing::debug!("Audit log rotation check for user {}: {}", user_id_clone, e);
                }
            });
        }
    }

    /// Emit SSE event to all connected dashboard clients
    pub fn emit_event(&self, event: MemoryEvent) {
        let _ = self.event_broadcaster.send(event);
    }

    /// Subscribe to SSE events
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<MemoryEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get audit history for user
    pub fn get_history(&self, user_id: &str, memory_id: Option<&str>) -> Vec<AuditEvent> {
        if let Some(log) = self.audit_logs.get(user_id) {
            let events = log.read();
            if !events.is_empty() {
                return if let Some(mid) = memory_id {
                    events
                        .iter()
                        .filter(|e| e.memory_id == mid)
                        .cloned()
                        .collect()
                } else {
                    events.iter().cloned().collect()
                };
            }
        }

        let events = self
            .audit_store
            .list_events(user_id, usize::MAX)
            .unwrap_or_default()
            .into_iter()
            .map(AuditEvent::from)
            .collect::<Vec<_>>();

        if !events.is_empty() {
            self.audit_logs
                .entry(user_id.to_string())
                .or_insert_with(|| {
                    Arc::new(parking_lot::RwLock::new(VecDeque::from(events.clone())))
                });
        }

        if let Some(mid) = memory_id {
            events.into_iter().filter(|e| e.memory_id == mid).collect()
        } else {
            events
        }
    }

    /// Where the tenant's record store keeps its keystore: beside `storage.redb`,
    /// or inside the RocksDB `storage/` directory.
    fn record_keystore_path(&self, user_id: &str) -> Result<std::path::PathBuf> {
        Ok(crate::keystore::store_keystore_path(
            &self.user_dir(user_id)?,
            self.server_config.effective_storage_backend,
        ))
    }

    /// The key a tenant's `purpose` data (journal, slow store) is sealed under: the
    /// global keystore when VELD_KEYSTORE_PATH is set, otherwise the tenant's store
    /// keystore. A passphrase alone seals the records under the store keystore, and
    /// the journal and slow store hold the same memories as full snapshots, so they
    /// are sealed under it too: one keystore, one decision.
    fn purpose_cryptor(
        &self,
        user_id: &str,
        purpose: &str,
    ) -> Result<Option<crate::keystore::SecretCryptor>> {
        match crate::keystore::SecretCryptor::from_env(purpose)
            .with_context(|| format!("derive {purpose} encryptor for user {user_id}"))?
        {
            Some(cryptor) => Ok(Some(cryptor)),
            None => crate::keystore::SecretCryptor::from_record_keystore(
                &self.record_keystore_path(user_id)?,
                purpose,
            )
            .with_context(|| {
                format!("derive {purpose} encryptor for user {user_id} from its store keystore")
            }),
        }
    }

    /// Whether this tenant's data is sealed at rest: a global keystore is set, a
    /// passphrase is set (the store creates its keystore from it), or the store
    /// already has one (a KMS-only unseal).
    fn tenant_encrypted(&self, user_id: &str) -> Result<bool> {
        let set = |k: &str| {
            std::env::var(k)
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
        };
        Ok(set("VELD_KEYSTORE_PATH")
            || set("VELD_MASTER_PASSPHRASE")
            || self.record_keystore_path(user_id)?.exists())
    }

    /// The per-user state directory, with the identifier validated as a single path component
    /// BEFORE it is joined — `user_id` arrives from HTTP/MCP/CLI callers, and `../` or an
    /// absolute path in it would escape `base_path` at every join below. Every user-scoped path
    /// in this manager goes through here.
    pub(crate) fn user_dir(&self, user_id: &str) -> Result<std::path::PathBuf> {
        if crate::validation::is_reserved_data_dir(user_id) {
            anyhow::bail!("user id '{user_id}' names a reserved data directory, not a tenant");
        }
        Ok(self
            .base_path
            .join(crate::path_guard::sanitize_component(user_id, "user id")?))
    }

    fn open_user_memory(&self, user_id: &str, config: MemoryConfig) -> Result<UserMemory> {
        match self.server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => {
                let primary_store = RocksDbPrimaryMemoryStore::open(
                    &config.storage_path,
                    self.shared_rocksdb_cache.as_ref(),
                )
                .with_context(|| {
                    format!(
                        "Failed to open primary memory store at {:?}",
                        config.storage_path
                    )
                })?;

                // Share the tenant's `Arc<BM25Index>` with the
                // retrieval engine so reads see exactly the same tantivy
                // segments the `Bm25Projection` commits to from the
                // intent log. Without this the substrate would open its
                // own `{storage_path}/bm25_index` directory and diverge
                // from the projection until the next restart-time
                // replay reconverged them.
                let bm25_index = self.get_user_bm25_index(user_id)?;

                UserMemory::with_storage_and_bm25_index(
                    config,
                    primary_store.into_inner(),
                    bm25_index,
                )
            }
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("opening a user memory store")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => {
                let primary =
                    crate::storage::redb::RedbPrimaryMemoryStore::open(&config.storage_path)
                        .with_context(|| {
                            format!(
                                "Failed to open redb primary memory store at {:?}",
                                config.storage_path
                            )
                        })?;
                let database = primary.database();
                let primary_store: Arc<dyn crate::storage::PrimaryMemoryStore> = Arc::new(primary);

                let bm25_index = self.get_user_bm25_index(user_id)?;

                // None: there is no rocksdb engine behind this store, and the
                // comprehensive-backup path refuses by name on it.
                let mut memory = UserMemory::with_primary_store_and_bm25_index(
                    config,
                    primary_store,
                    None,
                    bm25_index,
                )?;
                memory.attach_redb_backup_db(database);
                Ok(memory)
            }
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' requires this build to enable the storage-redb feature",
                self.server_config.effective_storage_backend
            )),
        }
    }

    fn open_user_graph_memory(&self, user_id: &str) -> Result<GraphMemory> {
        let graph_path = self.user_dir(user_id)?.join("graph");

        match self.server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => {
                let graph_store =
                    RocksDbGraphStore::open(&graph_path, self.shared_rocksdb_cache.as_ref())
                        .with_context(|| {
                            format!(
                                "Failed to open graph store for user '{user_id}' at {:?}",
                                graph_path
                            )
                        })?;

                Ok(graph_store.into_inner())
            }
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("opening a user graph store")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => {
                std::fs::create_dir_all(&graph_path)?;
                let database = crate::storage::redb::open_graph_database(&graph_path)?;
                let mut graph = GraphMemory::with_store(Arc::new(
                    crate::storage::redb::RedbColumnStore::new(Arc::clone(&database)),
                ))
                .with_context(|| {
                    format!(
                        "Failed to open redb graph store for user '{user_id}' at {graph_path:?}"
                    )
                })?;
                graph.attach_redb_backup_db(database);
                Ok(graph)
            }
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' requires this build to enable the storage-redb feature",
                self.server_config.effective_storage_backend
            )),
        }
    }

    /// Get or create a UserMemory substrate for a user.
    ///
    /// Uses double-checked locking to prevent TOCTOU races where concurrent
    /// first-access requests both miss the cache and try to open RocksDB.
    /// RocksDB holds exclusive file locks, so the second open would fail.
    ///
    /// A miss opens the tenant on the calling thread, and every concurrent caller for
    /// that tenant waits on its init lock until the open is done: 11.7 s for the live
    /// server's largest tenant and 10 min for the redb replica's on 2026-10-19. Async
    /// code calls [`Self::get_user_memory_async`] instead; each tokio worker parked here
    /// is one fewer to run `/health` on, and enough of them stop it answering at all.
    pub fn get_user_memory(&self, user_id: &str) -> Result<SharedUserMemory> {
        // Fast path: already cached
        if let Some(memory) = self.user_memories.get(user_id) {
            return Ok(memory);
        }

        let _waiter = TenantOpenWaiter::enter(&self.tenant_open_waiters);

        // Acquire per-user creation lock to serialize initialization
        let lock = self
            .user_memory_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        // Re-check after acquiring lock (another thread may have created it)
        if let Some(memory) = self.user_memories.get(user_id) {
            return Ok(memory);
        }

        let user_path = self.user_dir(user_id)?;
        let config = MemoryConfig {
            storage_path: user_path,
            collective_store_dir: if self.server_config.multi_tenant_mode {
                Some(self.server_config.collective_store_dir.clone())
            } else {
                None
            },
            ..self.default_config.clone()
        };

        // Retry with backoff for RocksDB lock contention. This can happen when a
        // moka eviction thread is still saving the vector index for this user (the
        // old MemorySystem holds the DB lock until the save thread drops its Arc).
        let mut user_mem = {
            let mut last_err = None;
            let mut created = None;
            for attempt in 0..4u32 {
                match self.open_user_memory(user_id, config.clone()) {
                    Ok(user_mem) => {
                        if attempt > 0 {
                            info!(
                                "UserMemory for user '{}' created after {} retries (lock contention resolved)",
                                user_id, attempt
                            );
                        }
                        created = Some(user_mem);
                        break;
                    }
                    Err(e) => {
                        // Retry EVERY error, not just ones whose Display
                        // mentions "lock". The observed post-eviction 500s
                        // (2026-07-31..08-03, always the tenant being lazily
                        // idle-evicted by its own request) failed immediately
                        // through the old non-lock branch — the transient
                        // races here are not limited to the RocksDB LOCK
                        // file, and a genuinely-broken tenant only pays
                        // ~750ms extra before the same error surfaces.
                        let delay = std::time::Duration::from_millis(50 * 2u64.pow(attempt));
                        tracing::warn!(
                            error = format!("{:#}", e),
                            "UserMemory open failed for user '{}' (attempt {}/4), retrying in {:?} (post-eviction races are expected to clear)",
                            user_id, attempt + 1, delay
                        );
                        std::thread::sleep(delay);
                        last_err = Some(e);
                    }
                }
            }
            match created {
                Some(user_mem) => user_mem,
                None => {
                    return Err(
                        last_err.unwrap_or_else(|| anyhow::anyhow!("all retry attempts failed"))
                    )
                    .with_context(|| {
                        format!(
                            "Failed to initialize UserMemory for user '{}' after 4 attempts",
                            user_id
                        )
                    });
                }
            }
        };
        // Wire up GraphMemory for Layer 2 (spreading activation) and Layer 5 (Hebbian learning)
        let graph = self.get_user_graph(user_id)?;
        user_mem.set_graph_memory(graph);
        // Wire up FeedbackStore for PIPE-9 (feedback momentum in all retrieval paths)
        user_mem.set_feedback_store(self.feedback_store.clone());

        let memory_arc: SharedUserMemory = Arc::new(parking_lot::RwLock::new(user_mem));

        self.user_memories
            .insert(user_id.to_string(), memory_arc.clone());
        // Checked after the insert, not before: a close that runs between a check and
        // the insert would miss this tenant in its cache walk.
        if self.writes_closed.load(std::sync::atomic::Ordering::SeqCst) {
            memory_arc.read().close_writes();
        }

        info!("Created memory system for user: {}", user_id);

        // C2 (W-001): one-time-per-process store↔log reconcile. Runs after
        // the user memory is cached so the writer-open path resolves it via the
        // fast cache path without re-entering this per-user init lock.
        // Non-fatal — a reconcile failure must not block user memory creation.
        if self
            .reconciled_users
            .insert(user_id.to_string(), ())
            .is_none()
        {
            if let Err(e) = self.reconcile_store_with_intent_log(user_id, &memory_arc) {
                tracing::warn!(
                    user_id = %user_id,
                    error = %e,
                    "C2 store↔log reconcile failed (non-fatal)",
                );
            }
        }

        Ok(memory_arc)
    }

    /// [`Self::get_user_memory`] for async code. A cached tenant comes back at once; an
    /// open, or a wait behind one, runs on the blocking pool so the worker stays free for
    /// other requests and the probes. A request dropped mid-open leaves the open to finish
    /// and cache the tenant for the next one.
    pub async fn get_user_memory_async(
        self: &Arc<Self>,
        user_id: &str,
    ) -> Result<SharedUserMemory> {
        if let Some(memory) = self.user_memories.get(user_id) {
            return Ok(memory);
        }
        let manager = Arc::clone(self);
        let user_id = user_id.to_string();
        tokio::task::spawn_blocking(move || manager.get_user_memory(&user_id))
            .await
            .context("tenant open task did not complete")?
    }

    /// Callers in `get_user_memory`'s slow path right now; see `tenant_open_waiters`.
    pub fn tenant_open_waiters(&self) -> usize {
        self.tenant_open_waiters
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The lock `get_user_memory` opens `user_id` under. Holding it stands in for an
    /// open that is taking as long as a large tenant's does.
    #[cfg(test)]
    pub(crate) fn user_memory_init_lock(&self, user_id: &str) -> Arc<parking_lot::Mutex<()>> {
        self.user_memory_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone()
    }

    /// Evict a user's memory and graph from in-memory caches (releases DB handles).
    /// Does NOT delete data — used before restore to release file locks.
    /// Evict the tenant and close its memory and graph stores before files are restored under
    /// them. Work spawned by a recent write can hold them for seconds; past `within` the
    /// restore is refused rather than run under an open store.
    fn evict_user_and_wait_for_release(
        &self,
        user_id: &str,
        within: std::time::Duration,
    ) -> Result<()> {
        let memory = self.user_memories.get(user_id);
        let graph = self.graph_memories.get(user_id);
        self.evict_user(user_id);
        let deadline = std::time::Instant::now() + within;
        // A store closes in whichever thread drops the last reference, after the count has
        // already reached zero, and a closing RocksDB deletes the files it does not know. So
        // the restore waits to hold the last reference and drops it here, closing the store
        // before any file is replaced.
        if let Some(memory) = memory {
            wait_until_sole_owner(&memory, deadline, user_id, within)?;
            drop(memory);
        }
        if let Some(graph) = graph {
            wait_until_sole_owner(&graph, deadline, user_id, within)?;
            drop(graph);
        }
        Ok(())
    }

    pub fn evict_user(&self, user_id: &str) {
        self.user_memories.invalidate(user_id);
        self.graph_memories.invalidate(user_id);
        self.user_memories.run_pending_tasks();
        self.graph_memories.run_pending_tasks();

        #[cfg(target_os = "windows")]
        {
            // Windows needs extra time to release file handles
            std::thread::sleep(std::time::Duration::from_millis(200));
            self.user_memories.run_pending_tasks();
            self.graph_memories.run_pending_tasks();
        }

        tracing::info!(user_id = user_id, "Evicted user caches for restore");
    }

    /// Delete user data (GDPR compliance)
    ///
    /// Cleans up:
    /// 1. In-memory caches (user_memories, graph_memories)
    /// 2. Shared RocksDB: todos, projects, todo indices, reminders, files, feedback, audit
    /// 3. Per-user filesystem: per-user RocksDB, graph DB, vector indices
    pub fn forget_user(&self, user_id: &str) -> Result<()> {
        self.user_memories.invalidate(user_id);
        self.graph_memories.invalidate(user_id);

        self.user_memories.run_pending_tasks();
        self.graph_memories.run_pending_tasks();

        #[cfg(target_os = "windows")]
        {
            std::thread::sleep(std::time::Duration::from_millis(200));
            self.user_memories.run_pending_tasks();
            self.graph_memories.run_pending_tasks();
        }

        // Clean up all user data from shared RocksDB column families
        self.purge_user_from_shared_db(user_id)?;

        // Clean up todo vector indices
        self.todo_store.purge_user_vectors(user_id);

        // Clean up in-memory feedback state
        {
            let mut fb = self.feedback_store.write();
            fb.take_pending(user_id);
        }

        // Delete per-user filesystem (memories DB, graph DB, vector index files)
        let user_path = self.user_dir(user_id)?;
        if user_path.exists() {
            let mut attempts = 0;
            let max_attempts = 10;
            while attempts < max_attempts {
                match std::fs::remove_dir_all(&user_path) {
                    Ok(_) => break,
                    Err(e) if attempts < max_attempts - 1 => {
                        let delay = 100 * (1 << attempts.min(4));
                        tracing::debug!(
                            "Delete retry {} for {} (waiting {}ms): {}",
                            attempts + 1,
                            user_id,
                            delay,
                            e
                        );
                        std::thread::sleep(std::time::Duration::from_millis(delay));
                        attempts += 1;
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to delete user data after {max_attempts} retries: {e}"
                        ))
                    }
                }
            }
        }

        info!("Deleted all data for user: {}", user_id);
        Ok(())
    }

    fn purge_user_from_shared_db(&self, user_id: &str) -> Result<()> {
        let n = purge_user_from_shared_store(self.shared_db.column_store().as_ref(), user_id)?;
        tracing::debug!("GDPR: purged {n} shared-store entries for {user_id}");
        self.audit_logs.remove(user_id);
        Ok(())
    }

    /// Get statistics for a user
    pub fn get_stats(&self, user_id: &str) -> Result<MemoryStats> {
        let memory = self.get_user_memory(user_id)?;
        let memory_guard = memory.read();
        let mut stats = memory_guard.stats();

        if let Ok(graph) = self.get_user_graph(user_id) {
            let graph_guard = graph.read();
            if let Ok(graph_stats) = graph_guard.get_stats() {
                stats.graph_nodes = graph_stats.entity_count;
                stats.graph_edges = graph_stats.relationship_count;
            }
        }

        Ok(stats)
    }

    /// List all users
    pub fn list_users(&self) -> Vec<String> {
        list_user_dirs(&self.base_path).unwrap_or_default()
    }

    /// List users currently loaded in the Moka cache (no filesystem scan)
    pub fn list_cached_users(&self) -> Vec<String> {
        self.user_memories
            .iter()
            .map(|(id, _)| id.to_string())
            .collect()
    }

    /// Get audit logs for a user
    pub fn get_audit_logs(&self, user_id: &str, limit: usize) -> Vec<AuditEvent> {
        let mut events = self
            .audit_store
            .list_events(user_id, usize::MAX)
            .unwrap_or_default()
            .into_iter()
            .map(AuditEvent::from)
            .collect::<Vec<_>>();
        events.reverse();
        events.truncate(limit);
        events
    }

    /// Flush all RocksDB databases
    /// Close every tenant's write gate, waiting for the write in progress in each. Queued
    /// and later creates and upserts fail with `WritesClosed` (503 to HTTP callers)
    /// instead of running past the flush. Blocking: call it from the blocking pool.
    /// Returns how many tenants were closed. Reaches the cached tenants only: one the
    /// cache evicted while a request still held it keeps its gate open.
    pub fn close_all_writes(&self) -> usize {
        self.writes_closed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let tenants: Vec<SharedUserMemory> = self.user_memories.iter().map(|(_, v)| v).collect();
        for memory in &tenants {
            memory.read().close_writes();
        }
        tenants.len()
    }

    pub fn flush_all_databases(&self) -> Result<()> {
        info!("Flushing all databases to disk...");

        self.drain_deferred_projections(crate::intent_log::journal::DEFERRED_DRAIN_TIMEOUT);

        // Single flush covers all shared stores (todos, prospective, files, feedback, audit)
        match &self.shared_db {
            #[cfg(feature = "rocksdb")]
            crate::storage::SharedEngineDb::RocksDb(db) => {
                crate::storage::write_latch::guard(db, || {
                    db.flush()
                        .map_err(|e| anyhow::anyhow!("Failed to flush shared database: {e}"))
                })?;
                info!("  Shared database flushed (todos, prospective, files, feedback, audit)");
            }
            #[cfg(feature = "storage-redb")]
            crate::storage::SharedEngineDb::Redb(_) => {
                // redb commits are durable per transaction — nothing buffered.
                info!("  Shared redb database: durable per transaction, nothing to flush");
            }
        }

        let user_entries: Vec<(String, SharedUserMemory)> = self
            .user_memories
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let mut flushed = 0;
        for (user_id, memory_system) in user_entries {
            if let Some(guard) = memory_system.try_read() {
                if let Err(e) = guard.flush_storage() {
                    tracing::warn!("  Failed to flush database for user {}: {}", user_id, e);
                } else {
                    flushed += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!(
            "All databases flushed: shared (5 stores), {} user memories",
            flushed
        );

        Ok(())
    }

    /// Wait, bounded by `timeout` across all tenants, for every journaled
    /// writer's deferred projection queue to drain, then persist those
    /// projections' checkpoints. Each writer's mutex is held only long
    /// enough to take a drain handle, so journaled writes keep flowing while
    /// the drain waits. Returns `false` if any queue was still non-empty at
    /// the deadline; replay on the next open applies the remainder.
    pub fn drain_deferred_projections(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let writers: Vec<(String, _)> = self
            .journaled_writers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        let mut drained = true;
        for (user_id, writer) in writers {
            let handle = writer.lock().deferred_drain_handle();
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if !handle.drain(remaining) {
                drained = false;
                tracing::warn!(
                    user_id = %user_id,
                    "deferred projection drain timed out; replay on next open applies the remainder",
                );
            }
        }
        drained
    }

    /// Save all vector indices to disk
    pub fn save_all_vector_indices(&self) -> Result<()> {
        info!("Saving vector indices to disk...");

        let user_entries: Vec<(String, SharedUserMemory)> = self
            .user_memories
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let mut saved = 0;
        for (user_id, memory_system) in user_entries {
            if let Some(guard) = memory_system.try_read() {
                let index_path = self.user_dir(&user_id)?.join("vector_index");
                if let Err(e) = guard.save_vector_index(&index_path) {
                    tracing::warn!("  Failed to save vector index for user {}: {}", user_id, e);
                } else {
                    info!("  Saved vector index for user: {}", user_id);
                    saved += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!("Saved {} vector indices", saved);
        Ok(())
    }

    /// Rotate audit logs for all users
    fn rotate_all_audit_logs(&self) -> Result<()> {
        let mut total_removed = 0;

        // Enumeration through the trait — the raw keyspace scan this replaces was
        // the last reason rotation reached past the audit abstraction.
        let user_ids: std::collections::HashSet<String> =
            self.audit_store.list_user_ids()?.into_iter().collect();

        let helper = MultiUserMemoryManagerRotationHelper {
            audit_store: self.audit_store.clone(),
            audit_logs: self.audit_logs.clone(),
            audit_retention_days: self.server_config.audit_retention_days as i64,
            audit_max_entries: self.server_config.audit_max_entries_per_user,
        };

        for user_id in user_ids {
            match helper.rotate_user_audit_logs(&user_id) {
                Ok(removed) => {
                    if removed > 0 {
                        info!(
                            "  Rotated audit logs for user {}: removed {} old entries",
                            user_id, removed
                        );
                        total_removed += removed;
                    }
                }
                Err(e) => {
                    tracing::warn!("  Failed to rotate audit logs for user {}: {}", user_id, e);
                }
            }
        }

        if total_removed > 0 {
            info!(
                "Audit log rotation complete: removed {} total entries",
                total_removed
            );
        }

        Ok(())
    }

    /// Get neural NER for entity extraction
    pub fn get_neural_ner(&self) -> Arc<NeuralNer> {
        self.neural_ner.clone()
    }

    /// Get keyword extractor for statistical term extraction
    /// Fill `query.ner_entities` from the neural NER, so a harness can hold
    /// entity extraction constant while varying the retrieval stages under
    /// test. Delegates to [`annotate_query_ner_with`].
    pub fn annotate_query_ner(&self, query: &mut crate::memory::types::Query) {
        annotate_query_ner_with(&self.neural_ner, query)
    }

    pub fn get_keyword_extractor(&self) -> Arc<KeywordExtractor> {
        self.keyword_extractor.clone()
    }

    /// Get or create graph memory for a user
    ///
    /// Uses the same per-user creation lock as get_user_memory to prevent
    /// concurrent RocksDB open races on the graph directory.
    pub fn get_user_graph(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<GraphMemory>>> {
        // Fast path: already cached
        if let Some(graph) = self.graph_memories.get(user_id) {
            return Ok(graph);
        }

        // Acquire per-user graph creation lock (separate from UserMemory lock
        // to avoid deadlock when get_user_memory() calls get_user_graph())
        let lock = self
            .user_graph_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        // Re-check after acquiring lock
        if let Some(graph) = self.graph_memories.get(user_id) {
            return Ok(graph);
        }

        // ── the self-lock fix ──────────────────────────────────────────────
        //
        // `user_memories` and `graph_memories` are separate moka caches with
        // separate time_to_idle, but a cached UserMemory HOLDS an Arc to its
        // GraphMemory (set at creation). So the graph cache can expire an entry
        // whose GraphMemory is still very much alive, still owning the RocksDB
        // handle. Opening again then failed against ourselves:
        //
        //   IO error: lock hold by current process … /graph/LOCK: No locks available
        //
        // Retrying could never clear that — the holder is a live cache entry,
        // not a dying thread. So before opening, ask whether anyone already has
        // it, and re-adopt that Arc instead of racing our own handle.
        //
        // try_read, never read: a caller may already hold a read guard on this
        // same UserMemory while calling us (the backup path does), and
        // parking_lot's RwLock is write-preferring, so a recursive read can
        // deadlock if a writer queues between the two. Failing to look is safe —
        // we simply fall through to the open below, which is today's behaviour.
        // `.get()` on the cache never creates, so this cannot recurse into
        // get_user_memory().
        if let Some(user_mem) = self.user_memories.get(user_id) {
            if let Some(guard) = user_mem.try_read() {
                if let Some(existing) = guard.graph_memory() {
                    let adopted = Arc::clone(existing);
                    drop(guard);
                    self.graph_memories
                        .insert(user_id.to_string(), Arc::clone(&adopted));
                    tracing::debug!(
                        user_id = %user_id,
                        "graph cache miss re-adopted the live handle held by user memory \
                         (avoided a self-lock on reopen)",
                    );
                    return Ok(adopted);
                }
            }
        }

        // Retry with backoff for RocksDB lock contention (same pattern as get_user_memory).
        // Graph eviction drops synchronously so contention is rare, but possible on Windows
        // where file handle release can lag.
        let graph_memory = {
            let mut last_err = None;
            let mut created = None;
            for attempt in 0..4u32 {
                match self.open_user_graph_memory(user_id) {
                    Ok(gm) => {
                        if attempt > 0 {
                            info!(
                                "Graph memory for user '{}' created after {} retries",
                                user_id, attempt
                            );
                        }
                        created = Some(gm);
                        break;
                    }
                    Err(e) => {
                        // Retry every error — same reasoning as the
                        // UserMemory open loop above: the post-eviction
                        // races that produced the 2026-07/08 500s bypassed
                        // the old "lock"-string gate.
                        let delay = std::time::Duration::from_millis(50 * 2u64.pow(attempt));
                        tracing::warn!(
                            error = format!("{:#}", e),
                            "graph memory open failed for user '{}' (attempt {}/4), retrying in {:?}",
                            user_id, attempt + 1, delay
                        );
                        std::thread::sleep(delay);
                        last_err = Some(e);
                    }
                }
            }
            match created {
                Some(gm) => gm,
                None => {
                    return Err(
                        last_err.unwrap_or_else(|| anyhow::anyhow!("all retry attempts failed"))
                    )
                    .with_context(|| {
                        format!(
                            "Failed to initialize graph memory for user '{}' after 4 attempts",
                            user_id
                        )
                    });
                }
            }
        };
        let graph_arc = Arc::new(parking_lot::RwLock::new(graph_memory));

        self.graph_memories
            .insert(user_id.to_string(), graph_arc.clone());

        info!("Created graph memory for user: {}", user_id);

        Ok(graph_arc)
    }

    /// Get or create a cached SlowStore for a user.
    ///
    /// SlowStores are cached per-user to avoid reopening SQLite on every request
    /// and to ensure the sync TTL actually works across calls.
    pub fn get_user_slow_store(
        &self,
        user_id: &str,
    ) -> Result<std::sync::Arc<crate::memory::slow_store::SlowStore>> {
        if let Some(store) = self.slow_stores.get(user_id) {
            return Ok(store.clone());
        }

        let user_path = self.user_dir(user_id)?;
        std::fs::create_dir_all(&user_path)?;
        let db_path = user_path.join("slow_store.db");
        // Full memory snapshots, like the journal: sealed under the same keystore.
        let store = std::sync::Arc::new(crate::memory::slow_store::SlowStore::open_with_encryptor(
            &db_path,
            self.purpose_cryptor(user_id, "slow-store")?,
        )?);
        self.slow_stores.insert(user_id.to_string(), store.clone());
        Ok(store)
    }

    /// Get or create the per-tenant `Arc<BM25Index>`.
    ///
    /// One tantivy directory per user at
    /// `{base_path}/{user_id}/bm25_projection_index`, shared by:
    ///
    /// 1. the `Bm25Projection` driven off the user's intent log (writes,
    ///    via `JournaledWriter::record_and_apply`), and
    /// 2. the retrieval-side `HybridSearchEngine` living inside the
    ///    user's `UserMemory` (reads, via `MemorySystem::with_storage_and_bm25_index`).
    ///
    /// Sharing the `Arc` means tantivy's single-writer / multi-reader
    /// constraint is respected (the projection holds the only
    /// `IndexWriter`) and reads see fresh segments as soon as the
    /// projection's `commit()` + `reload()` returns. The legacy
    /// `{base_path}/{user_id}/bm25_index` directory is no longer
    /// created by any code path; operators can delete it from old
    /// tenants at any time.
    pub fn get_user_bm25_index(
        &self,
        user_id: &str,
    ) -> Result<std::sync::Arc<crate::memory::BM25Index>> {
        if let Some(index) = self.bm25_indices.get(user_id) {
            return Ok(index.clone());
        }
        let user_path = self.user_dir(user_id)?;
        let bm25_path = user_path.join(crate::memory::BM25_INDEX_DIR);
        // An encrypted tenant's index stays in memory: on disk it would be a
        // plaintext copy of the content its sealed records and journal protect.
        let index = if self.tenant_encrypted(user_id)? {
            std::sync::Arc::new(
                crate::memory::BM25Index::in_memory(&bm25_path)
                    .with_context(|| format!("open in-memory BM25 index for user {user_id}"))?,
            )
        } else {
            std::fs::create_dir_all(&bm25_path)?;
            std::sync::Arc::new(
                crate::memory::BM25Index::new(&bm25_path)
                    .with_context(|| format!("open BM25 projection index for user {user_id}"))?,
            )
        };
        self.bm25_indices.insert(user_id.to_string(), index.clone());
        Ok(index)
    }

    /// Get or create a [`JournaledWriter`] for `user_id`.
    ///
    /// On first call for a tenant, this:
    ///   1. opens (or creates) the per-user intent log at
    ///      `{base_path}/{user_id}/intent.log`,
    ///   2. opens (or creates) the per-user checkpoint store at
    ///      `{base_path}/{user_id}/projection_checkpoints.bin`,
    ///   3. instantiates a [`SqliteProjection`] wrapped around the user's
    ///      `SlowStore` and a [`Bm25Projection`] wrapped around the
    ///      user's BM25 (tantivy) index,
    ///   4. **runs replay** so both projections catch up to the head of
    ///      the log before any live write lands, and
    ///   5. attaches the projections to the writer so subsequent live
    ///      `record_and_apply` calls dispatch through them in order.
    ///
    /// On any subsequent call this returns the cached writer.
    ///
    /// Replay runs synchronously (inline with the call) because:
    ///   - the log is small (one frame per CRUD operation, bincoded),
    ///   - we MUST close the W5 catch-up gap before live writes can race
    ///     against a half-replayed projection, and
    ///   - the caller can spawn this on a blocking task pool if it needs
    ///     a non-blocking startup. Doing the replay inside the lazy-open
    ///     keeps the "logged ↔ projected" invariant local and provable.
    pub fn get_user_journaled_writer(
        &self,
        user_id: &str,
    ) -> Result<std::sync::Arc<parking_lot::Mutex<crate::intent_log::JournaledWriter>>> {
        // Fast path — cached.
        if let Some(w) = self.journaled_writers.get(user_id) {
            return Ok(w.clone());
        }

        // Slow path — single-flight init under a per-user lock.
        let init_lock = self
            .journaled_writer_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = init_lock.lock();

        // Re-check after acquiring the lock — another thread may have
        // populated the writer while we were waiting.
        if let Some(w) = self.journaled_writers.get(user_id) {
            return Ok(w.clone());
        }

        let user_path = self.user_dir(user_id)?;
        std::fs::create_dir_all(&user_path)?;
        let log_path = user_path.join("intent.log");
        let checkpoint_path = user_path.join("projection_checkpoints.bin");

        // Seal the journal's frame payloads at rest (encryption-v2) using a
        // keystore-derived "journal" purpose key — so the WAL never holds
        // plaintext memory snapshots on disk (or in backups). `None` (no keystore
        // configured) keeps the journal plaintext, backward compatible.
        // Without VELD_KEYSTORE_PATH the key comes from the store's own keystore:
        // a passphrase alone seals the records, and the journal beside them holds
        // the same memories, so it is sealed under the same keystore.
        let journal_cryptor = self.purpose_cryptor(user_id, "journal")?;
        let checkpoint_store = std::sync::Arc::new(parking_lot::Mutex::new(
            crate::intent_log::CheckpointStore::open(&checkpoint_path)
                .with_context(|| format!("open checkpoint store for user {user_id}"))?,
        ));
        // The offset hints the last writer persisted let the open scan only
        // the frames appended after them; a hint that does not verify
        // against the file is ignored.
        let frame_index_hint = checkpoint_store.lock().frame_index();
        let log = crate::intent_log::IntentLog::open_with_index(
            &log_path,
            journal_cryptor,
            frame_index_hint,
        )
        .with_context(|| format!("open intent log for user {user_id}"))?;

        // SQLite slow store projection — first in the replay pass.
        let slow_store = self.get_user_slow_store(user_id)?;
        let mut sqlite_projection = match &self.dataset_executor {
            // W4 cutover: when a relational backend is configured, the
            // memories projection writes through that store (bridged
            // sync→async) — the *same* store the W6 query planner reads and
            // the W7 dataset surface uses — instead of the local rusqlite
            // slow store. The slow store is still passed for checkpoint
            // bookkeeping and the gap-analysis tables it also owns. With no
            // relational backend configured (the default), this is the
            // original rusqlite path, byte-for-byte.
            Some(store) => crate::memory::slow_store::SqliteProjection::with_relational(
                slow_store.clone(),
                checkpoint_store.clone(),
                crate::memory::slow_store::RelationalSlowStoreAdapter::new(store.clone()),
            ),
            None => crate::memory::slow_store::SqliteProjection::new(
                slow_store.clone(),
                checkpoint_store.clone(),
            ),
        };

        // Second family of projections: Vamana, one per embedding kind.
        // Each user gets up to five dedicated vector indices that are
        // *derived from the log*. Each kind lives under
        // `{base}/{user_id}/vamana_projection/{kind}/` so a
        // wipe-and-replay operator workflow can blow away one kind's
        // directory (e.g. re-embed all images) without touching the
        // others, the SQLite slow store, or RocksDB.
        //
        // We use the *lazy* variant: each kind's index is materialised
        // on the first memory whose embedding-for-that-kind is present
        // (during replay or live writes). This avoids coupling the
        // projection's open to the embedder cache (the embedder is
        // constructed lazily per-tenant via `get_user_memory`, which can
        // be a different lifecycle than `get_user_journaled_writer`).
        // The first embedded memory tells us the dim — that's the same
        // dim every subsequent memory of the same kind must match (each
        // embedder is stable for a tenant's corpus), so this is correct
        // by construction.
        //
        // All five kinds are registered eagerly. The cost of an empty,
        // un-materialised projection is one in-memory `HashMap` and one
        // checkpoint-store row — negligible. Kinds that never see a
        // matching field stay pending and never allocate a Vamana
        // graph, so there's no I/O penalty for tenants that only ever
        // store text memories.
        let vamana_root = user_path.join("vamana_projection");
        std::fs::create_dir_all(&vamana_root).with_context(|| {
            format!(
                "create vamana projection root for user {user_id}: {:?}",
                vamana_root
            )
        })?;
        let mut vamana_projections: Vec<crate::vector_db::VamanaProjection> = Vec::new();
        for &kind in crate::vector_db::VamanaEmbeddingKind::all() {
            let kind_dir = vamana_root.join(kind.dir_component());
            std::fs::create_dir_all(&kind_dir).with_context(|| {
                format!(
                    "create vamana projection dir for user {user_id} kind {}: {:?}",
                    kind.projection_name(),
                    kind_dir
                )
            })?;
            let bootstrap = crate::vector_db::VamanaProjectionBootstrap {
                storage_path: Some(kind_dir),
                config_template: crate::vector_db::VamanaConfig {
                    // dimension is rewritten on first embedded apply
                    // for this kind.
                    dimension: 0,
                    max_degree: 32,
                    search_list_size: 100,
                    alpha: 1.2,
                    use_mmap: true,
                    distance_metric: crate::vector_db::DistanceMetric::default(),
                },
            };
            // The primary text index holds the tenant's active space: the embedder this process
            // runs, else what the corpus markers name. Bound before replay, so a log that
            // spans an embedder change does not build it in the old space.
            let active_space = (kind == crate::vector_db::VamanaEmbeddingKind::TextPrimary)
                .then(|| {
                    crate::embeddings::primary_embedder_info()
                        .map(|info| {
                            crate::embeddings::space::SpaceId::new(&info.model, info.dimension)
                        })
                        .or_else(|| crate::embeddings::space::corpus_space(&user_path))
                })
                .flatten();
            vamana_projections.push(
                crate::vector_db::VamanaProjection::lazy(kind, bootstrap, checkpoint_store.clone())
                    .bind_space(active_space),
            );
        }

        // Third projection: BM25 inverted index.
        let bm25_index = self.get_user_bm25_index(user_id)?;
        // An in-memory index starts empty every time, so its projection replays
        // the whole log rather than resuming from the checkpoint a past index wrote.
        let mut bm25_projection = if bm25_index.is_in_memory() {
            crate::memory::Bm25Projection::new_from_start(
                bm25_index.clone(),
                checkpoint_store.clone(),
            )
        } else {
            crate::memory::Bm25Projection::new(bm25_index.clone(), checkpoint_store.clone())
        };
        // Fourth projection (default-OFF, gated by VELD_WIKI_RENDER): the
        // wiki render. Init or replay failure is non-fatal — the other
        // projections and the write path are unaffected; the wiki is simply
        // not registered for this writer's lifetime.
        let mut wiki_projection = if crate::wiki::wiki_render_enabled() {
            let wiki_dir = crate::wiki::wiki_target_dir(&user_path, user_id);
            match crate::wiki::WikiProjection::new(wiki_dir, checkpoint_store.clone()) {
                Ok(wp) => Some(wp),
                Err(e) => {
                    tracing::warn!(
                        user_id = %user_id,
                        error = %e,
                        "wiki projection init failed (non-fatal; wiki render disabled this session)",
                    );
                    None
                }
            }
        } else {
            None
        };

        // Catch every projection up to the head of the log before any goes
        // live, in one pass: each frame is read once from the lowest
        // checkpoint and handed to the projections still below it.
        // Checkpoints stay independent — a projection that fails stops at
        // its failed record while the others replay to the head.
        let persist_every = 100
            .min(crate::memory::bm25_projection::COMMIT_EVERY)
            .min(crate::wiki::COMMIT_EVERY);
        let replay_results = {
            let mut targets: Vec<&mut dyn crate::intent_log::ReplayTarget> =
                Vec::with_capacity(vamana_projections.len() + 3);
            targets.push(&mut sqlite_projection);
            for vp in vamana_projections.iter_mut() {
                targets.push(vp);
            }
            targets.push(&mut bm25_projection);
            if let Some(wp) = wiki_projection.as_mut() {
                targets.push(wp);
            }
            let names: Vec<String> = targets
                .iter()
                .map(|t| t.target_name().to_string())
                .collect();
            let results = crate::intent_log::replay_all(&log, &mut targets, Some(persist_every))
                .with_context(|| format!("replay intent log for user {user_id}"))?;
            names.into_iter().zip(results).collect::<Vec<_>>()
        };
        let wiki_name = wiki_projection
            .as_ref()
            .map(|wp| crate::intent_log::ReplayTarget::target_name(wp).to_string());
        let mut first_failure = None;
        for (name, result) in replay_results {
            match result {
                Ok(applied) if applied > 0 => tracing::info!(
                    user_id = %user_id,
                    projection = %name,
                    applied,
                    "replayed intent log records into projection on first writer open",
                ),
                Ok(_) => {}
                Err(e) if Some(&name) == wiki_name.as_ref() => {
                    tracing::warn!(
                        user_id = %user_id,
                        error = %e,
                        "wiki projection replay failed (non-fatal; wiki render disabled this session)",
                    );
                    wiki_projection = None;
                }
                Err(e) => {
                    tracing::error!(
                        user_id = %user_id,
                        projection = %name,
                        error = %e,
                        "intent log replay failed on first writer open",
                    );
                    first_failure.get_or_insert(e);
                }
            }
        }
        if let Some(e) = first_failure {
            return Err(
                anyhow::Error::new(e).context(format!("replay intent log for user {user_id}"))
            );
        }

        // Hand the log handle into the writer so subsequent appends
        // continue against the same file. Replay above used a read-only
        // borrow; the same handle is fine for the writer's appends.
        let mut writer = crate::intent_log::JournaledWriter::new(log);
        writer.attach_checkpoint_store(checkpoint_store.clone());
        if let Err(e) = writer.persist_frame_index() {
            tracing::warn!(
                user_id = %user_id,
                error = %e,
                "could not persist intent log offset hints (the next open scans further back)",
            );
        }
        // The SQLite mirror is derived and non-authoritative, and a busy
        // competing SQLite writer can stall its apply for seconds; applied
        // inline that stall serialized every write for the tenant behind the
        // writer mutex. Deferred, it applies on its own worker after the ack.
        writer.add_deferred_projection(Box::new(sqlite_projection));
        for vp in vamana_projections {
            writer.add_projection(Box::new(vp));
        }
        writer.add_projection(Box::new(bm25_projection));
        // Register the wiki render projection (built + replayed above) if it
        // was enabled and opened cleanly. Live writes now also fan out to it.
        if let Some(wiki_projection) = wiki_projection {
            writer.add_projection(Box::new(wiki_projection));
        }

        let writer = std::sync::Arc::new(parking_lot::Mutex::new(writer));
        self.journaled_writers
            .insert(user_id.to_string(), writer.clone());
        Ok(writer)
    }

    /// Convenience: journal a typed payload through the user's
    /// `JournaledWriter`. Calls `record_and_apply` under the writer's
    /// mutex and returns the assigned LSN. Per-projection apply errors
    /// are *not* propagated as failures — they are logged at warn level
    /// (the log frame is durable, replay will retry on restart). The
    /// SQLite projection is deferred: the returned LSN is durable and
    /// applied to the inline projections, but may not be in SQLite yet.
    pub fn journal_and_apply(
        &self,
        user_id: &str,
        payload: &crate::intent_log::IntentPayload,
    ) -> Result<crate::intent_log::Lsn> {
        let writer = self.get_user_journaled_writer(user_id)?;
        let outcome = {
            let mut writer = writer.lock();
            writer
                .record_and_apply(payload)
                .map_err(|e| anyhow::anyhow!("journal record_and_apply: {e}"))?
        };
        for err in &outcome.apply_errors {
            tracing::warn!(
                user_id = %user_id,
                projection = %err.projection,
                lsn = outcome.lsn.0,
                error = %err.source,
                "projection apply failed during journaled write (log durable; replay will retry)",
            );
        }
        Ok(outcome.lsn)
    }

    /// [`Self::journal_and_apply`] for many records: one writer lock, one append
    /// batch and one fsync, then each record applied to the projections in order. The
    /// batch is durable whole or not at all.
    pub fn journal_and_apply_batch(
        &self,
        user_id: &str,
        payloads: &[crate::intent_log::IntentPayload],
    ) -> Result<Vec<crate::intent_log::Lsn>> {
        let writer = self.get_user_journaled_writer(user_id)?;
        let outcomes = writer
            .lock()
            .record_and_apply_batch(payloads)
            .map_err(|e| anyhow::anyhow!("journal record_and_apply_batch: {e}"))?;
        for outcome in &outcomes {
            for err in &outcome.apply_errors {
                tracing::warn!(
                    user_id = %user_id,
                    projection = %err.projection,
                    lsn = outcome.lsn.0,
                    error = %err.source,
                    "projection apply failed during journaled write (log durable; replay will retry)",
                );
            }
        }
        Ok(outcomes.into_iter().map(|outcome| outcome.lsn).collect())
    }

    /// W-018 support: ensure the wiki projection is materialised for
    /// `user_id` and pull any out-of-band page edits back into the canonical
    /// store, then report `(target_dir, pages_on_disk, edits_applied)`.
    /// Opening the journaled writer replays the intent log into the wiki
    /// projection on first open (full backfill); the edit-back scan folds in
    /// user edits. Callers gate on [`crate::wiki::wiki_render_enabled`].
    pub fn render_wiki(&self, user_id: &str) -> Result<(String, usize, usize)> {
        // Opening the writer triggers the on-open replay/backfill of the
        // wiki projection (when the feature is enabled).
        let _ = self.get_user_journaled_writer(user_id)?;
        // Fold in user edits first — each edit journals an Update that
        // re-renders its page and syncs the other projections.
        let edits = self.scan_wiki_edits(user_id)?;
        let user_path = self.user_dir(user_id)?;
        let dir = crate::wiki::wiki_target_dir(&user_path, user_id);
        let pages = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
                    .count()
            })
            .unwrap_or(0);
        Ok((dir.display().to_string(), pages, edits))
    }

    /// (b) Ensure a live filesystem watcher is running for `user_id`'s wiki
    /// dir, so out-of-band edits flow back without an explicit
    /// `POST /api/wiki/render`. No-op unless both `VELD_WIKI_RENDER` and
    /// `VELD_WIKI_WATCH` are on, and idempotent (one watcher per user). The
    /// watcher debounces and calls [`Self::scan_wiki_edits`] — the same
    /// hash-guarded edit-back path — so it never re-ingests veld's own writes.
    /// `manager` is the shared `Arc<Self>` the spawned thread calls back into.
    pub fn ensure_wiki_watcher(&self, manager: std::sync::Arc<Self>, user_id: &str) {
        if !crate::wiki::wiki_render_enabled() || !crate::wiki::watcher::wiki_watch_enabled() {
            return;
        }
        if self.wiki_watchers.contains_key(user_id) {
            return;
        }
        // An id that fails the guard has never had a directory created for it, so there is
        // nothing to watch — refuse quietly rather than joining an unvalidated component.
        let Ok(user_path) = self.user_dir(user_id) else {
            tracing::warn!(user_id = %user_id, "wiki watcher: refusing invalid user id");
            return;
        };
        let dir = crate::wiki::wiki_target_dir(&user_path, user_id);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(user_id = %user_id, error = %e, "wiki watcher: could not create wiki dir");
            return;
        }
        match crate::wiki::watcher::spawn_wiki_watcher(manager, user_id.to_string(), &dir) {
            Ok(w) => {
                self.wiki_watchers
                    .insert(user_id.to_string(), parking_lot::Mutex::new(w));
                tracing::info!(user_id = %user_id, dir = ?dir, "started live wiki edit-back watcher");
            }
            Err(e) => tracing::warn!(
                user_id = %user_id,
                error = %e,
                "failed to start live wiki watcher (non-fatal)"
            ),
        }
    }

    /// W-003/W-004: scan a user's wiki directory for body edits and apply
    /// them back to the canonical store. **Self-write suppression (W-004)**
    /// is the `last_rendered_hash` match — a page veld just rendered hashes
    /// equal to its recorded hash and is skipped, so veld never re-ingests
    /// its own output and the render→edit loop cannot run away. Redacted
    /// pages are skipped (a redacted body can't be round-tripped). Only files
    /// carrying valid veld frontmatter are eligible, so a stray markdown file
    /// dropped into the directory is ignored (ingestion-sink guard, W-007).
    /// Returns the number of edits applied.
    ///
    /// Scan-based rather than a live `notify` watcher by design: the watcher
    /// was the synthesis's "most fragile contract" (debounce races, a
    /// long-lived task, a new dependency). A scan triggered on demand (via
    /// `POST /api/wiki/render`) is the robust equivalent — a live watcher can
    /// layer on top later without changing this contract.
    pub fn scan_wiki_edits(&self, user_id: &str) -> Result<usize> {
        if !crate::wiki::wiki_render_enabled() {
            return Ok(0);
        }
        let user_path = self.user_dir(user_id)?;
        let dir = crate::wiki::wiki_target_dir(&user_path, user_id);
        let read = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => return Ok(0),
        };
        let mut applied = 0usize;
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("md") {
                continue;
            }
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let Some(page) = crate::wiki::parse_rendered_page(&text) else {
                continue; // not a veld-rendered page (ingestion-sink guard)
            };
            if page.redacted {
                continue; // redacted body can't round-trip; render is one-way
            }
            if crate::wiki::body_sha256_hex(&page.body) == page.last_rendered_hash {
                continue; // unchanged since veld rendered it (self-write)
            }
            match self.apply_wiki_edit(user_id, &page.id, &page.body) {
                Ok(true) => applied += 1,
                Ok(false) => {}
                Err(e) => tracing::warn!(
                    user_id = %user_id,
                    page = ?path,
                    error = %e,
                    "wiki edit-back failed for page (non-fatal)",
                ),
            }
        }
        if applied > 0 {
            tracing::info!(
                user_id = %user_id,
                applied,
                "applied wiki page edits back to the canonical store",
            );
        }
        Ok(applied)
    }

    /// Apply one wiki-page body edit back to the canonical memory. Journals
    /// an `Update` so the page re-renders (fresh `last_rendered_hash` → the
    /// next scan is a no-op, closing the loop) and the other projections
    /// sync. Returns `true` iff the content actually changed. Unknown /
    /// unparsable ids are ignored (ingestion-sink safety).
    fn apply_wiki_edit(&self, user_id: &str, memory_id: &str, new_body: &str) -> Result<bool> {
        let uuid = match uuid::Uuid::parse_str(memory_id) {
            Ok(u) => u,
            Err(_) => return Ok(false),
        };
        let mid = crate::memory::types::MemoryId(uuid);
        let user_mem = self.get_user_memory(user_id)?;
        let new_content = new_body.trim_end().to_string();
        let updated_bincode = {
            let guard = user_mem.read();
            let mut mem = match guard.get_memory(&mid) {
                Ok(m) => m,
                Err(_) => return Ok(false), // unknown memory — ignore
            };
            if mem.experience.content == new_content {
                return Ok(false); // no real change
            }
            mem.experience.content = new_content;
            mem.experience.clear_primary_vector(); // force re-embed / re-index
            guard.update_memory_journaled(&mem)?;
            bincode::serde::encode_to_vec(&mem, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("wiki edit-back bincode: {e}"))?
        };
        let payload = crate::intent_log::IntentPayload::Update {
            user_id: user_id.to_string(),
            memory_id: memory_id.to_string(),
            memory_bincode: updated_bincode,
            schema_version: Some(crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        self.journal_and_apply(user_id, &payload)?;
        Ok(true)
    }

    /// C2 (W-001): reconcile a user's canonical store against the intent
    /// log. A write's log frame is appended *after* the RocksDB commit, in
    /// a detached task (`handlers::remember`), so a crash in that window
    /// leaves a memory durable in the store with no corresponding log
    /// frame. Left alone, every projection / PITR replay rebuilt from the
    /// log would be missing that memory. On first access per process we
    /// scan the log, find store records the log never recorded, and
    /// backfill a `Remember` frame for each.
    ///
    /// Each direction follows what the client was last told:
    /// - store-ahead-of-log (crashed before a `Remember`/`Update` frame
    ///   landed): backfill a `Remember` frame from the store snapshot.
    /// - log-ahead-of-store: the log holds a live memory the store lacks.
    ///   Either a store write whose log frame was synced was lost (storage
    ///   durability is `eventual` by default, so an OS crash can drop it
    ///   after the remember was acknowledged), or a crash fell between a
    ///   store delete and its `Forget` frame, and every delete path responds
    ///   only after journaling, so that delete was never acknowledged. Both
    ///   ways the memory should exist: restore it into the store from its
    ///   last logged snapshot, with any later anchor applied, and journal
    ///   nothing. Only a memory with no snapshot in the log (stored before
    ///   the journal existed, then anchored) cannot be restored; it gets a
    ///   `Forget` frame so replay agrees with the store. On a tenant just
    ///   restored from a backup ([`RESTORED_FROM_BACKUP_MARKER`]) the store
    ///   is authoritative instead: every log-ahead memory gets a `Forget`.
    ///
    /// Returns the number of memories reconciled. Non-fatal — the caller
    /// logs and continues on error.
    fn reconcile_store_with_intent_log(
        &self,
        user_id: &str,
        user_mem: &SharedUserMemory,
    ) -> Result<usize> {
        // Ids the log already covers (Remember/Update/Anchor present, minus
        // Forget). The writer resumes this from the snapshot the previous
        // reconcile left in the checkpoint store, so only frames appended
        // since are read, and it reads each memory id off the payload prefix
        // without decoding the memory snapshot. Writer lock is released
        // before the backfill so each batch can retake it.
        let writer = self.get_user_journaled_writer(user_id)?;
        let covered = writer
            .lock()
            .live_memory_ids()
            .map_err(|e| anyhow::anyhow!("reconcile: scanning intent log: {e}"))?;

        let store_ids = {
            let guard = user_mem.read();
            guard.all_memory_ids()?
        };

        let store_id_strings: std::collections::HashSet<String> =
            store_ids.iter().map(|id| id.0.to_string()).collect();

        // store-ahead-of-log: snapshot each uncovered memory and backfill a
        // `Remember` frame so the log is a complete record of the store.
        let mut remember_frames = Vec::new();
        for id in &store_ids {
            let id_str = id.0.to_string();
            if covered.contains(&id_str) {
                continue;
            }
            let snapshot = {
                let guard = user_mem.read();
                match guard.get_memory(id) {
                    Ok(mem) => {
                        match bincode::serde::encode_to_vec(&mem, bincode::config::standard()) {
                            Ok(bytes) => Some(bytes),
                            Err(e) => {
                                tracing::warn!(
                                    user_id = %user_id,
                                    memory_id = %id_str,
                                    error = %e,
                                    "C2 reconcile: skipping backfill, bincode encode failed",
                                );
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            user_id = %user_id,
                            memory_id = %id_str,
                            error = %e,
                            "C2 reconcile: skipping backfill, memory not retrievable",
                        );
                        None
                    }
                }
            };
            let Some(bytes) = snapshot else { continue };
            remember_frames.push(crate::intent_log::IntentPayload::Remember {
                user_id: user_id.to_string(),
                memory_id: id_str,
                memory_bincode: bytes,
                schema_version: Some(crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
            });
        }
        let backfilled =
            self.journal_reconcile_backfill(user_id, &writer, remember_frames, "Remember");

        // log-ahead-of-store: the log says live, the store has no such memory. Restore it
        // from its last logged snapshot; a memory the log holds no snapshot for gets a
        // `Forget`, so replay (including restore-from-backup) cannot resurrect it.
        let missing: std::collections::HashSet<String> = covered
            .iter()
            .filter(|id| !store_id_strings.contains(*id))
            .cloned()
            .collect();
        let mut restored = 0usize;
        let mut forget_frames = Vec::new();
        let marker = self.user_dir(user_id)?.join(RESTORED_FROM_BACKUP_MARKER);
        let restored_from_backup = marker.exists();
        let mut snapshots = if missing.is_empty() || restored_from_backup {
            HashMap::new()
        } else {
            self.last_logged_snapshots(user_id, &missing)?
        };
        let mut restore = Vec::new();
        for id_str in &missing {
            match snapshots.remove(id_str) {
                Some(Ok(memory)) => restore.push(memory),
                Some(Err(e)) => tracing::warn!(
                    user_id = %user_id,
                    memory_id = %id_str,
                    error = %e,
                    "C2 reconcile: logged snapshot of a memory missing from the store does not decode; left unrestored",
                ),
                None => forget_frames.push(crate::intent_log::IntentPayload::Forget {
                    user_id: user_id.to_string(),
                    memory_id: id_str.clone(),
                    schema_version: Some(
                        crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION,
                    ),
                }),
            }
        }
        let forget_backfilled =
            self.journal_reconcile_backfill(user_id, &writer, forget_frames, "Forget");
        if !restore.is_empty() {
            for memory in &restore {
                tracing::warn!(
                    user_id = %user_id,
                    memory_id = %memory.id.0,
                    "C2 reconcile: memory live in the intent log but missing from the store; restored from its logged snapshot",
                );
            }
            restored = user_mem.read().upsert_memories(&restore)?;
        }

        if restored_from_backup {
            std::fs::remove_file(&marker).with_context(|| {
                format!("reconcile: removing the backup-restore marker {marker:?}")
            })?;
        }
        if backfilled + restored + forget_backfilled > 0 {
            tracing::warn!(
                user_id = %user_id,
                backfilled,
                restored,
                forget_backfilled,
                "C2 store↔log reconcile: Remember frames for store-ahead, restores and Forget frames for log-ahead",
            );
        }
        Ok(backfilled + restored + forget_backfilled)
    }

    /// The last `Remember`/`Update` snapshot of each of `ids` in `user_id`'s log, with
    /// any later anchor frame applied, decoded. An id the log holds no snapshot for is
    /// absent from the result. Reads the log once, under the writer lock.
    fn last_logged_snapshots(
        &self,
        user_id: &str,
        ids: &std::collections::HashSet<String>,
    ) -> Result<HashMap<String, Result<crate::memory::types::Memory>>> {
        use crate::intent_log::IntentPayload;
        enum Anchoring {
            Inferred(f32),
            Set(bool, f32),
        }
        let mut last: HashMap<String, (Vec<u8>, Option<Anchoring>)> = HashMap::new();
        {
            let writer = self.get_user_journaled_writer(user_id)?;
            let guard = writer.lock();
            for record in guard
                .log()
                .iter()
                .map_err(|e| anyhow::anyhow!("reconcile: reading intent log: {e}"))?
            {
                let record =
                    record.map_err(|e| anyhow::anyhow!("reconcile: reading intent log: {e}"))?;
                let (_, payload) = crate::intent_log::payload::decode_record(&record)
                    .map_err(|e| anyhow::anyhow!("reconcile: decoding intent log: {e}"))?;
                if !ids.contains(payload.memory_id()) {
                    continue;
                }
                match payload {
                    IntentPayload::Remember {
                        memory_id,
                        memory_bincode,
                        ..
                    }
                    | IntentPayload::Update {
                        memory_id,
                        memory_bincode,
                        ..
                    } => {
                        last.insert(memory_id, (memory_bincode, None));
                    }
                    IntentPayload::Forget { memory_id, .. } => {
                        last.remove(&memory_id);
                    }
                    IntentPayload::Anchor {
                        memory_id,
                        importance,
                        ..
                    } => {
                        if let Some(entry) = last.get_mut(&memory_id) {
                            entry.1 = Some(Anchoring::Inferred(importance));
                        }
                    }
                    IntentPayload::SetAnchored {
                        memory_id,
                        anchored,
                        importance,
                        ..
                    } => {
                        if let Some(entry) = last.get_mut(&memory_id) {
                            entry.1 = Some(Anchoring::Set(anchored, importance));
                        }
                    }
                }
            }
        }
        Ok(last
            .into_iter()
            .map(|(id, (bytes, anchoring))| {
                let memory =
                    crate::memory::storage::decode_memory_snapshot(&bytes).inspect(|memory| {
                        match anchoring {
                            // Same reading the replication follower gives an `Anchor` frame,
                            // which carries only the importance after the change.
                            Some(Anchoring::Inferred(importance)) => {
                                if importance > memory.importance() {
                                    memory.set_anchored(true);
                                } else if importance < memory.importance() {
                                    memory.set_anchored(false);
                                }
                                memory.set_importance(importance);
                            }
                            Some(Anchoring::Set(anchored, importance)) => {
                                memory.set_anchored(anchored);
                                memory.set_importance(importance);
                            }
                            None => {}
                        }
                    });
                (id, memory)
            })
            .collect())
    }

    /// Journal reconcile backfill frames in batches, one fsync per batch and
    /// the writer lock held for one batch at a time. A failed batch is logged
    /// and skipped (the next reconcile finds the same gap). Returns the
    /// number of frames journaled.
    fn journal_reconcile_backfill(
        &self,
        user_id: &str,
        writer: &parking_lot::Mutex<crate::intent_log::JournaledWriter>,
        frames: Vec<crate::intent_log::IntentPayload>,
        kind: &str,
    ) -> usize {
        const BATCH: usize = 256;
        let mut journaled = 0usize;
        for batch in frames.chunks(BATCH) {
            let outcomes = writer.lock().record_and_apply_batch(batch);
            match outcomes {
                Ok(outcomes) => {
                    for outcome in &outcomes {
                        for err in &outcome.apply_errors {
                            tracing::warn!(
                                user_id = %user_id,
                                projection = %err.projection,
                                lsn = outcome.lsn.0,
                                error = %err.source,
                                "projection apply failed during reconcile backfill (log durable; replay will retry)",
                            );
                        }
                    }
                    journaled += outcomes.len();
                }
                Err(e) => tracing::warn!(
                    user_id = %user_id,
                    frames = batch.len(),
                    kind,
                    error = %e,
                    "C2 reconcile: backfill batch could not be journaled",
                ),
            }
        }
        journaled
    }

    /// Get graph statistics for a user
    pub fn get_user_graph_stats(&self, user_id: &str) -> Result<GraphStats> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.read();
        graph_guard.get_stats()
    }

    /// Run maintenance on all cached user memories
    pub fn run_maintenance_all_users(&self) -> usize {
        let cycle = self
            .maintenance_cycle
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Heavy cycle every 6th iteration (6 hours at 3600s intervals).
        // Heavy cycles run replay, entity-entity strengthening, fact extraction (full memory scan),
        // and flush databases (triggers compaction). Light cycles only touch in-memory data.
        let is_heavy = cycle.is_multiple_of(6);

        if is_heavy {
            tracing::info!(
                "Maintenance cycle {} (HEAVY — graph decay + fact extraction + flush)",
                cycle
            );
        } else {
            tracing::debug!("Maintenance cycle {} (light — in-memory only)", cycle);
        }

        let decay_factor = self.server_config.activation_decay_factor;
        let mut total_processed = 0;
        #[cfg(feature = "multi-tenant")]
        let mut collective_weights = Vec::new();
        #[cfg(feature = "multi-tenant")]
        let mut total_feedback_events = 0u64;

        let user_ids: Vec<String> = self
            .user_memories
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();

        let user_count = user_ids.len();
        let mut edges_decayed = 0;
        let mut edges_strengthened = 0;
        let mut entity_edges_strengthened = 0;
        let mut total_facts_extracted = 0;
        let mut total_facts_reinforced = 0;

        for user_id in user_ids {
            let maintenance_result = if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                let memory = memory_lock.read();
                #[cfg(feature = "multi-tenant")]
                if self.server_config.multi_tenant_mode {
                    let learned = memory.learned_weight_state();
                    if learned.update_count > 0 {
                        collective_weights.push((learned.bm25, learned.vector, learned.graph));
                        total_feedback_events += learned.update_count;
                    }
                }
                match memory.run_maintenance(decay_factor, &user_id, is_heavy) {
                    Ok(result) => {
                        total_processed += result.decayed_count;
                        total_facts_extracted += result.facts_extracted;
                        total_facts_reinforced += result.facts_reinforced;
                        Some(result)
                    }
                    Err(e) => {
                        tracing::warn!("Maintenance failed for user {}: {}", user_id, e);
                        None
                    }
                }
            } else {
                None
            };

            // Direction 1: Edge strengthening + promotion boost propagation
            if let Some(ref result) = maintenance_result {
                if !result.edge_boosts.is_empty() {
                    if let Ok(graph) = self.get_user_graph(&user_id) {
                        let graph_guard = graph.read();
                        match graph_guard
                            .strengthen_memory_edges(&result.edge_boosts, "maintenance:replay")
                        {
                            Ok((count, promotion_boosts)) => {
                                edges_strengthened += count;

                                // Direction 1: Apply edge promotion boosts to memory importance
                                if !promotion_boosts.is_empty() {
                                    if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                                        let memory = memory_lock.read();
                                        match memory.apply_edge_promotion_boosts(&promotion_boosts)
                                        {
                                            Ok(boosted) => {
                                                tracing::debug!(
                                                    user_id = %user_id,
                                                    boosted,
                                                    promotions = promotion_boosts.len(),
                                                    "Applied edge promotion boosts"
                                                );
                                            }
                                            Err(e) => {
                                                tracing::debug!(
                                                    "Edge promotion boost failed for user {}: {}",
                                                    user_id,
                                                    e
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::debug!(
                                    "Edge boost application failed for user {}: {}",
                                    user_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }

            // Direction 3: Entity-entity Hebbian reinforcement for replayed memories
            // During replay, memories are re-activated — strengthen edges between entities
            // that co-occur in the same episode, reinforcing semantic associations.
            if let Some(ref result) = maintenance_result {
                if !result.replay_memory_ids.is_empty() {
                    if let Ok(graph) = self.get_user_graph(&user_id) {
                        let graph_guard = graph.read();
                        for mem_id_str in &result.replay_memory_ids {
                            if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                                match graph_guard
                                    .strengthen_episode_entity_edges(&uuid, "maintenance:replay")
                                {
                                    Ok(count) => entity_edges_strengthened += count,
                                    Err(e) => {
                                        tracing::debug!(
                                            "Entity edge strengthening failed for memory {}: {}",
                                            mem_id_str,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Direction 2: Lazy decay — flush opportunistic pruning queue
            // Instead of scanning all 34k+ edges (apply_decay), we queue edges found
            // below threshold during normal reads and batch-delete them here.
            // Runs every cycle since it's just targeted deletes, not a full scan.
            if let Ok(graph) = self.get_user_graph(&user_id) {
                let graph_guard = graph.read();
                match graph_guard.flush_pending_maintenance() {
                    Ok(decay_result) => {
                        edges_decayed += decay_result.pruned_count;

                        // Direction 2: Compensate memories that lost all graph edges
                        if !decay_result.orphaned_entity_ids.is_empty() {
                            if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                                let memory = memory_lock.read();
                                match memory
                                    .compensate_orphaned_memories(&decay_result.orphaned_entity_ids)
                                {
                                    Ok(compensated) => {
                                        tracing::debug!(
                                            user_id = %user_id,
                                            compensated,
                                            orphaned = decay_result.orphaned_entity_ids.len(),
                                            "Compensated orphaned memories"
                                        );
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                            "Orphan compensation failed for user {}: {}",
                                            user_id,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Graph lazy pruning failed for user {}: {}", user_id, e);
                    }
                }
            }

            // Direction 4: Full graph decay on heavy cycles
            // Lazy pruning (above) only processes edges found below threshold during reads.
            // Edges that are never read still need decay applied. Run full apply_decay()
            // every heavy cycle (6 hours) to ensure no edge escapes time-based weakening.
            if is_heavy {
                if let Ok(graph) = self.get_user_graph(&user_id) {
                    let graph_guard = graph.read();
                    match graph_guard.apply_decay() {
                        Ok(decay_result) => {
                            if decay_result.pruned_count > 0 {
                                edges_decayed += decay_result.pruned_count;
                                tracing::debug!(
                                    user_id = %user_id,
                                    pruned = decay_result.pruned_count,
                                    orphaned = decay_result.orphaned_entity_ids.len(),
                                    "Full graph decay applied"
                                );
                            }

                            if !decay_result.orphaned_entity_ids.is_empty() {
                                if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                                    let memory = memory_lock.read();
                                    let _ = memory.compensate_orphaned_memories(
                                        &decay_result.orphaned_entity_ids,
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Full graph decay failed for user {}: {}", user_id, e);
                        }
                    }
                }
            }
        }

        #[cfg(feature = "multi-tenant")]
        if self.server_config.multi_tenant_mode {
            if let Err(error) = maintenance::run_maintenance_cycle(
                &self.server_config.collective_store_dir,
                &collective_weights,
                total_feedback_events,
            ) {
                tracing::warn!(error = %error, "Collective maintenance aggregation failed");
            }
        }

        // Heavy cycle: sleep-time graduation pass (R27) + supersession decay
        // (R48). Both run only when the orchestrator is installed; both are
        // best-effort — a failure here does not abort the rest of the
        // maintenance cycle.
        if is_heavy {
            let orch_opt = self.sleep_time_orchestrator.read().clone();
            if let Some(orch) = orch_opt {
                // R48: supersession-confidence decay across every user's
                // edges (the store is global). Single pass, returns
                // (decayed, removed) counts.
                match orch
                    .supersession_store()
                    .decay_all(crate::memory::sleep_time::DEFAULT_SUPERSESSION_DECAY)
                {
                    Ok((decayed, removed)) => {
                        if decayed > 0 || removed > 0 {
                            tracing::info!(
                                decayed,
                                removed,
                                "sleep-time supersession decay applied (R48)"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "sleep-time supersession decay failed");
                    }
                }

                // R27: graduation pass per user. Scans each user's memories,
                // promotes observation memories whose access_count meets the
                // configured threshold.
                let threshold = crate::memory::sleep_time::DEFAULT_GRADUATION_ACCESS_THRESHOLD;
                for (user_id_arc, _) in self.user_memories.iter() {
                    let user_id = user_id_arc.as_ref();
                    if let Ok(user_mem) = self.get_user_memory(user_id) {
                        let user_mem_guard = user_mem.read();
                        match crate::memory::sleep_time::graduate_eligible_observations(
                            user_mem_guard.as_memory_system(),
                            orch.graduation_store(),
                            threshold,
                        ) {
                            Ok(result) => {
                                if result.graduated > 0 {
                                    tracing::info!(
                                        user_id = %user_id,
                                        graduated = result.graduated,
                                        scanned = result.observations_scanned,
                                        threshold,
                                        "sleep-time graduation pass (R27)"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::debug!(
                                    user_id = %user_id,
                                    error = %e,
                                    "sleep-time graduation pass failed"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Heavy cycle: R1 anticipatory prefetch tick (Physarum-anticipation).
        // Per-user, isolated to each user's memory. Warms rhythm-predicted memories
        // into the session tier via add_shared (idempotent, size-bounded). Adds a
        // maintenance STEP; removes none. Best-effort — never aborts the cycle.
        if is_heavy {
            let mut total_warmed = 0usize;
            for (user_id_arc, _) in self.user_memories.iter() {
                let user_id = user_id_arc.as_ref();
                if let Ok(user_mem) = self.get_user_memory(user_id) {
                    let user_mem_guard = user_mem.read();
                    total_warmed += user_mem_guard.run_prefetch_tick(user_id);
                }
            }
            if total_warmed > 0 {
                tracing::info!(
                    warmed = total_warmed,
                    "anticipatory prefetch warmed memories into session tiers (R1)"
                );
            }
        }

        // Heavy cycle: clean up old triggered/dismissed reminders (C4 fix)
        if is_heavy {
            for (user_id_arc, _) in self.user_memories.iter() {
                let user_id = user_id_arc.as_ref();
                match self.prospective_store.cleanup_old_tasks(user_id, 30) {
                    Ok(deleted) if deleted > 0 => {
                        tracing::info!(
                            user_id = %user_id,
                            deleted = deleted,
                            "Cleaned up old prospective tasks (>30 days)"
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            user_id = %user_id,
                            error = %e,
                            "Prospective task cleanup failed"
                        );
                    }
                    _ => {}
                }
            }
        }

        // Flush databases only on heavy cycles — flush triggers RocksDB compaction
        // which allocates significant C++ memory through Windows CRT
        if is_heavy {
            if let Err(e) = self.flush_all_databases() {
                tracing::warn!("Periodic flush failed: {}", e);
            }

            // Prune init locks: remove entries for users no longer in cache.
            // This prevents unbounded growth of the DashMaps over time.
            let active_users: std::collections::HashSet<String> = self
                .user_memories
                .iter()
                .map(|(id, _)| id.to_string())
                .collect();
            self.user_memory_init_locks
                .retain(|user_id, _| active_users.contains(user_id));
            self.user_graph_init_locks
                .retain(|user_id, _| active_users.contains(user_id));
            // Prune SlowStore cache for evicted users to prevent unbounded DashMap growth.
            // Each SlowStore holds an open SQLite connection with 8MB page cache.
            let pre_slow = self.slow_stores.len();
            self.slow_stores
                .retain(|user_id, _| active_users.contains(user_id));
            let pruned_slow = pre_slow.saturating_sub(self.slow_stores.len());
            if pruned_slow > 0 {
                tracing::info!(
                    "Pruned SlowStore connections for {} evicted users ({} active)",
                    pruned_slow,
                    self.slow_stores.len()
                );
            }

            // Prune audit logs for evicted users to prevent unbounded DashMap growth.
            // Each user's log can hold up to audit_max_entries_per_user entries (~2-5MB),
            // and without pruning, entries persist long after the user's memory/graph are evicted.
            let pre_audit = self.audit_logs.len();
            self.audit_logs
                .retain(|user_id, _| active_users.contains(user_id));
            let pruned_audit = pre_audit.saturating_sub(self.audit_logs.len());
            if pruned_audit > 0 {
                tracing::info!(
                    "Pruned audit logs for {} evicted users ({} active)",
                    pruned_audit,
                    self.audit_logs.len()
                );
            }
        }

        tracing::info!(
            "Maintenance complete (cycle {}, {}): {} memories processed, {} edges strengthened, {} entity edges strengthened, {} weak edges pruned, {} facts extracted, {} facts reinforced across {} users",
            cycle,
            if is_heavy { "heavy" } else { "light" },
            total_processed,
            edges_strengthened,
            entity_edges_strengthened,
            edges_decayed,
            total_facts_extracted,
            total_facts_reinforced,
            user_count
        );

        total_processed
    }

    /// Get the streaming extractor
    pub fn streaming_extractor(&self) -> &Arc<streaming::StreamingMemoryExtractor> {
        &self.streaming_extractor
    }

    /// Get the backup engine
    pub fn backup_engine(&self) -> Option<&Arc<backup::VeldBackupEngine>> {
        self.backup_engine.as_ref()
    }

    /// Get the A/B test manager
    pub fn ab_test_manager(&self) -> &Arc<ab_testing::ABTestManager> {
        &self.ab_test_manager
    }

    /// Get the todo store
    pub fn todo_store(&self) -> &Arc<TodoStore> {
        &self.todo_store
    }

    /// Get the prospective store
    pub fn prospective_store(&self) -> &Arc<ProspectiveStore> {
        &self.prospective_store
    }

    /// Get the file store
    pub fn file_store(&self) -> &Arc<FileMemoryStore> {
        &self.file_store
    }

    /// Get the feedback store
    pub fn feedback_store(&self) -> &Arc<parking_lot::RwLock<FeedbackStore>> {
        &self.feedback_store
    }

    /// Get the session store
    pub fn session_store(&self) -> &Arc<SessionStore> {
        &self.session_store
    }

    /// Get context sessions
    pub fn context_sessions(&self) -> &Arc<ContextSessions> {
        &self.context_sessions
    }

    /// Subscribe to context status updates
    pub fn subscribe_context(&self) -> tokio::sync::broadcast::Receiver<ContextStatus> {
        self.context_broadcaster.subscribe()
    }

    /// Broadcast context status update
    pub fn broadcast_context(&self, status: ContextStatus) {
        let _ = self.context_broadcaster.send(status);
    }

    /// Get server config
    pub fn server_config(&self) -> &ServerConfig {
        &self.server_config
    }

    /// Capabilities for the backend currently serving persistence calls.
    pub fn storage_capabilities(&self) -> StorageCapabilities {
        self.storage_capabilities
    }

    /// Get base path
    pub fn base_path(&self) -> &std::path::Path {
        &self.base_path
    }

    /// Get user evictions count
    pub fn user_evictions(&self) -> usize {
        self.user_evictions
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get users in cache count
    pub fn users_in_cache(&self) -> usize {
        self.user_memories.entry_count() as usize
    }

    /// Lightweight readiness check for public probes.
    ///
    /// This intentionally avoids lazy per-user initialization, which can create
    /// state, open RocksDB handles, and trigger model/network work for an
    /// unauthenticated health request.
    pub fn is_ready(&self) -> bool {
        if !self.base_path.exists() {
            return false;
        }

        // These column families are required for the shared stores initialized
        // at startup. Missing handles indicate the shared DB is not usable.
        #[cfg(feature = "rocksdb")]
        const REQUIRED_CFS: &[&str] = &[
            "audit",
            "prospective",
            "todos",
            "projects",
            "files",
            "feedback",
        ];

        match &self.shared_db {
            #[cfg(feature = "rocksdb")]
            crate::storage::SharedEngineDb::RocksDb(db) => {
                REQUIRED_CFS.iter().all(|cf| db.cf_handle(cf).is_some())
            }
            #[cfg(feature = "storage-redb")]
            crate::storage::SharedEngineDb::Redb(_) => {
                // Families are materialized at construction by each store's
                // ensure_cf, or bootstrap already failed. The rocksdb check
                // exists because ITS families come from open-time descriptors
                // that config drift could omit; redb has no equivalent drift.
                true
            }
        }
    }

    /// Active reminder check: scan all users for due reminders, mark them triggered,
    /// and emit `REMINDER_DUE` events to the broadcast channel.
    ///
    /// Called by the dedicated 60-second reminder scheduler in main.rs.
    /// Returns the number of reminders triggered.
    pub fn check_and_emit_due_reminders(&self) -> usize {
        let due_tasks = match self.prospective_store.get_all_due_tasks() {
            Ok(tasks) => tasks,
            Err(e) => {
                tracing::debug!("Active reminder check failed: {}", e);
                return 0;
            }
        };

        let mut triggered = 0;
        for (user_id, task) in &due_tasks {
            match self.prospective_store.mark_triggered(user_id, &task.id) {
                Ok(true) => {} // successfully triggered
                Ok(false) => {
                    // Already triggered by concurrent call — skip event emission
                    tracing::debug!(
                        user_id = %user_id,
                        reminder_id = %task.id.0,
                        "Reminder already triggered (scheduler race)"
                    );
                    continue;
                }
                Err(e) => {
                    tracing::warn!(
                        user_id = %user_id,
                        reminder_id = %task.id.0,
                        error = %e,
                        "Failed to mark reminder triggered in scheduler"
                    );
                    continue;
                }
            }

            self.emit_event(MemoryEvent {
                event_type: "REMINDER_DUE".to_string(),
                timestamp: chrono::Utc::now(),
                user_id: user_id.clone(),
                memory_id: Some(task.id.0.to_string()),
                content_preview: Some(task.content.chars().take(100).collect()),
                memory_type: Some("reminder".to_string()),
                importance: Some(task.priority as f32 / 5.0),
                count: None,
                entities: None,
                results: None,
            });

            tracing::info!(
                user_id = %user_id,
                reminder_id = %task.id.0,
                content = %task.content.chars().take(50).collect::<String>(),
                "Reminder triggered (active)"
            );

            triggered += 1;
        }

        triggered
    }

    #[cfg(feature = "rocksdb")]
    fn collect_legacy_secondary_store_refs(&self) -> Vec<(String, std::sync::Arc<rocksdb::DB>)> {
        // Empty when not rocksdb-backed: these refs feed the rocksdb
        // BackupEngine, which require_backup_engine has already refused by then.
        match self.shared_db.rocksdb() {
            Some(db) => vec![("shared".to_string(), std::sync::Arc::clone(db))],
            None => Vec::new(),
        }
    }

    /// Directory holding `user_id`'s W5 intent log: the journal lives at
    /// `{base_path}/{user_id}/intent.log` next to the tenant's stores
    /// (see `get_user_journaled_writer`). The old `<base_path>/intent_log/`
    /// convention predates the per-user layout — backups pointed at it
    /// captured nothing (`intent_log_files: []` on every archive).
    fn intent_log_user_dir(&self, user_id: &str) -> Result<std::path::PathBuf> {
        self.user_dir(user_id)
    }

    pub fn create_user_backup(&self, user_id: &str) -> Result<BackupMetadata> {
        match self.server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => {
                // Always attempt to include the intent log; the backup
                // engine no-ops cleanly when the dir doesn't exist (W5
                // not yet active for this install).
                let intent_log_spec = crate::backup::IntentLogBackupSpec::per_user(
                    self.intent_log_user_dir(user_id)?,
                );

                // Stage the journal under the tenant's writer mutex so the
                // cut is clean (no mid-frame copy, exact cut LSN), and
                // BEFORE the primary checkpoint so the snapshot is a
                // superset of the journal — the direction C2 reconcile can
                // heal on restore. This runs before any user-memory or
                // graph lock is taken (get_user_journaled_writer may touch
                // user memory on first open, and holding the writer mutex
                // here only blocks journal appends for the duration of a
                // file copy). If the writer can't be opened, fall back to
                // an unquiesced copy rather than failing the backup.
                let staged = match self.get_user_journaled_writer(user_id) {
                    Ok(writer) => {
                        let _writer_guard = writer.lock();
                        Some(
                            self.require_backup_engine("backup intent-log staging")?
                                .stage_intent_log(&intent_log_spec, user_id)?,
                        )
                    }
                    Err(e) => {
                        tracing::warn!(
                            user_id = %user_id,
                            error = %e,
                            "backup: journaled writer unavailable; staging intent log without quiesce",
                        );
                        Some(
                            self.require_backup_engine("backup intent-log staging")?
                                .stage_intent_log(&intent_log_spec, user_id)?,
                        )
                    }
                };

                let memory_sys = match self.get_user_memory(user_id) {
                    Ok(memory_sys) => memory_sys,
                    Err(e) => {
                        if let Some(staged) = &staged {
                            let _ = std::fs::remove_dir_all(&staged.staging_dir);
                        }
                        return Err(e);
                    }
                };
                let memory_guard = memory_sys.read();
                // Same refuse-don't-degrade shape as the graph below: a system
                // that is not rocksdb-backed has no engine DB for the rocksdb
                // BackupEngine, and pretending otherwise writes an archive
                // missing its primary store while reporting success.
                let db = match memory_guard.rocksdb_for_comprehensive_backup() {
                    Some(db) => db,
                    None => {
                        if let Some(staged) = &staged {
                            let _ = std::fs::remove_dir_all(&staged.staging_dir);
                        }
                        return Err(anyhow::anyhow!(
                            "refusing comprehensive backup for '{user_id}': the memory system \
                             is not rocksdb-backed, and the comprehensive backup is a rocksdb \
                             BackupEngine facility. The redb route is export/import."
                        ));
                    }
                };

                let secondary_refs = self.collect_legacy_secondary_store_refs();
                let store_refs: Vec<crate::backup::SecondaryStoreRef<'_>> = secondary_refs
                    .iter()
                    .map(|(name, db)| crate::backup::SecondaryStoreRef { name, db })
                    .collect();

                // REFUSE, do not degrade. This was `.ok()`, which turned a failure
                // to open the graph into `None` — and `None` means "no graph to
                // capture", so the archive was written without it, `secondary_stores`
                // simply lacked "graph", and the response still said success: true.
                // A backup missing a store while reporting success is worse than no
                // backup, because it is the thing a restore is trusted to rely on.
                //
                // Absence is not a case here: GraphMemory::new opens with
                // create_if_missing(true), so a user who has never had a graph gets
                // an empty one rather than an error. An Err is therefore a real
                // failure to read something that exists — exactly what a caller
                // needs to hear about before trusting this archive.
                let graph_lock = self.get_user_graph(user_id).map_err(|e| {
                    if let Some(staged) = &staged {
                        let _ = std::fs::remove_dir_all(&staged.staging_dir);
                    }
                    anyhow::anyhow!(
                        "refusing to write a backup for '{user_id}' without its graph store: {e}. \
                         The archive would have been silently incomplete."
                    )
                })?;
                let graph_guard = graph_lock.read();
                // Same refusal shape as above: a graph without a rocksdb engine
                // cannot feed the rocksdb BackupEngine, and omitting it would be
                // the silently-incomplete archive this block refuses.
                let graph_db_ref = Some(graph_guard.rocksdb_for_backup().ok_or_else(|| {
                    if let Some(staged) = &staged {
                        let _ = std::fs::remove_dir_all(&staged.staging_dir);
                    }
                    anyhow::anyhow!(
                        "refusing comprehensive backup for '{user_id}': the graph store is                          not rocksdb-backed"
                    )
                })?);

                self.require_backup_engine("comprehensive backup creation")?
                    .create_comprehensive_backup_with_staged_intent_log(
                        &db,
                        user_id,
                        &store_refs,
                        graph_db_ref,
                        staged,
                    )
            }
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("backup creation")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => self.create_redb_user_backup(user_id),
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend 'redb' requires this build to enable the storage-redb feature"
            )),
        }
    }

    /// Journal staged under the writer lock before the stores are dumped: the rocksdb path's consistency cut.
    #[cfg(feature = "storage-redb")]
    fn create_redb_user_backup(&self, user_id: &str) -> Result<BackupMetadata> {
        let engine = self.require_redb_backup_engine("backup creation")?;
        let refuse = |store: &str| {
            anyhow::anyhow!(
                "refusing to write a backup for '{user_id}' without its {store}: this process \
                 holds no redb handle for it, and the archive would be silently incomplete"
            )
        };
        let memory = self.get_user_memory(user_id)?;
        let primary = memory
            .read()
            .redb_for_backup()
            .ok_or_else(|| refuse("primary store"))?;
        let graph_lock = self.get_user_graph(user_id).map_err(|e| {
            anyhow::anyhow!(
                "refusing to write a backup for '{user_id}' without its graph store: {e}"
            )
        })?;
        let graph = graph_lock
            .read()
            .redb_for_backup()
            .ok_or_else(|| refuse("graph store"))?;
        let shared = self
            .shared_db
            .redb()
            .cloned()
            .ok_or_else(|| refuse("shared stores"))?;
        let audit = self
            .audit_redb
            .clone()
            .ok_or_else(|| refuse("audit store"))?;

        let mut pending = engine.begin(user_id)?;
        let spec = crate::backup::IntentLogBackupSpec::per_user(self.intent_log_user_dir(user_id)?);
        match self.get_user_journaled_writer(user_id) {
            Ok(writer) => {
                let _writer_guard = writer.lock();
                pending.stage_intent_log(&spec)?;
            }
            Err(e) => {
                tracing::warn!(
                    user_id = %user_id,
                    error = %e,
                    "backup: journaled writer unavailable; staging intent log without quiesce",
                );
                pending.stage_intent_log(&spec)?;
            }
        }
        pending.export(
            &self.base_path,
            &crate::storage::migrate::LiveRedb {
                primary: Some(&primary),
                graph: Some(&graph),
                audit: Some(&audit),
                shared: Some(&shared),
            },
        )?;
        pending.finish()
    }

    pub fn list_backups_for_user(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        match self.server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => self
                .require_backup_engine("backup listing")?
                .list_backups(user_id),
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("backup listing")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => self
                .require_redb_backup_engine("backup listing")?
                .list(user_id),
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend 'redb' requires this build to enable the storage-redb feature"
            )),
        }
    }

    pub fn verify_backup_for_user(&self, user_id: &str, backup_id: u32) -> Result<bool> {
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => self
                .require_backup_engine("backup verification")?
                .verify_backup(user_id, backup_id),
            // Ok(true) only: every mismatch or missing file is an Err naming it.
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => self
                .require_redb_backup_engine("backup verification")?
                .verify(user_id, backup_id)
                .map(|_| true),
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend 'redb' requires this build to enable the storage-redb feature"
            )),
        }
    }

    pub fn purge_backups_for_user(&self, user_id: &str, keep_count: usize) -> Result<usize> {
        match self.server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => self
                .require_backup_engine("backup purging")?
                .purge_old_backups(user_id, keep_count),
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("backup purging")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => self
                .require_redb_backup_engine("backup purging")?
                .purge(user_id, keep_count),
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend 'redb' requires this build to enable the storage-redb feature"
            )),
        }
    }

    pub fn restore_user_backup(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
    ) -> Result<Vec<String>> {
        self.restore_user_backup_with_options(user_id, backup_id, None)
    }

    /// Restore a user's backup with optional point-in-time-restore.
    ///
    /// `max_lsn` (if set) truncates the restored intent log so that the
    /// final on-disk frame has `lsn == max_lsn`. Frames with higher LSNs
    /// in the archive are dropped.
    pub fn restore_user_backup_with_options(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        max_lsn: Option<u64>,
    ) -> Result<Vec<String>> {
        const RELEASE_WAIT: std::time::Duration = std::time::Duration::from_secs(30);
        // Work a recent write started can still hold the tenant and may need its graph to
        // finish, so it is let go before the tenant's open locks are taken.
        self.evict_user_and_wait_for_release(user_id, RELEASE_WAIT)?;
        // get_user_memory and get_user_graph open a tenant under these locks, in this order.
        // Holding them keeps anything from reopening its stores while their files are replaced:
        // a store opened mid-restore reads the old or half-written files and stays cached.
        let memory_lock = self
            .user_memory_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let graph_lock = self
            .user_graph_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _memory_guard = memory_lock.lock();
        let _graph_guard = graph_lock.lock();
        self.evict_user_and_wait_for_release(user_id, RELEASE_WAIT)?;

        let restored = self.restore_user_stores(user_id, backup_id, max_lsn)?;
        std::fs::write(
            self.user_dir(user_id)?.join(RESTORED_FROM_BACKUP_MARKER),
            b"",
        )
        .with_context(|| format!("marking '{user_id}' as restored from a backup"))?;
        self.reconciled_users.remove(user_id);
        Ok(restored)
    }

    /// The store swap of [`Self::restore_user_backup_with_options`], run with the
    /// tenant evicted and its open locks held.
    fn restore_user_stores(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        max_lsn: Option<u64>,
    ) -> Result<Vec<String>> {
        match self.server_config.effective_storage_backend {
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb => {
                let memory_db_path = self.user_dir(user_id)?.join("storage");
                let graph_path = self.user_dir(user_id)?.join("graph").join("graph");

                let secondary_restore_paths: Vec<(&str, &std::path::Path)> = vec![];
                let intent_log_spec = crate::backup::IntentLogBackupSpec::per_user(
                    self.intent_log_user_dir(user_id)?,
                );
                let restore_opts = crate::backup::RestoreOptions { max_lsn };
                let backup_engine = self.require_backup_engine("backup restore")?;
                let restored_stores = backup_engine.restore_comprehensive_backup_with_intent_log(
                    user_id,
                    backup_id,
                    &memory_db_path,
                    &secondary_restore_paths,
                    Some(&intent_log_spec),
                    &restore_opts,
                )?;

                let resolved_backup_id = backup_id.unwrap_or_else(|| {
                    backup_engine
                        .list_backups(user_id)
                        .ok()
                        .and_then(|backups| backups.first().map(|metadata| metadata.backup_id))
                        .unwrap_or(0)
                });
                let graph_checkpoint = backup_engine
                    .backup_path()
                    .join(user_id)
                    .join(format!("secondary_{resolved_backup_id}"))
                    .join("graph");

                let mut all_restored = restored_stores;

                // The main memories DB is what the caller actually came for, and it
                // was never named in this list — so a successful restore reported
                // stores that sound peripheral and stayed silent about the memories,
                // which reads as "my memories did not come back".
                all_restored.insert(0, "memories".to_string());

                if graph_checkpoint.exists() {
                    // STAGE THEN SWAP. This used to delete the live graph first and
                    // then copy — so a failed copy left the user with NO graph at
                    // all, and the failure was a warn!, so the function still
                    // returned Ok and the endpoint still answered success: true.
                    // Destroying data and reporting success is the worst available
                    // outcome; the original is now never removed until its
                    // replacement is on disk beside it.
                    let staging = graph_path.with_extension("restoring");
                    if staging.exists() {
                        std::fs::remove_dir_all(&staging).with_context(|| {
                            format!("clearing stale graph staging dir {staging:?}")
                        })?;
                    }
                    crate::backup::copy_dir_recursive_pub(&graph_checkpoint, &staging).map_err(
                        |e| {
                            let _ = std::fs::remove_dir_all(&staging);
                            anyhow::anyhow!(
                                "restoring the graph for '{user_id}' failed while staging: {e}. \
                                 The existing graph was left untouched."
                            )
                        },
                    )?;

                    // Only now is it safe to move the original aside.
                    let previous = graph_path.with_extension("superseded");
                    if graph_path.exists() {
                        let _ = std::fs::remove_dir_all(&previous);
                        std::fs::rename(&graph_path, &previous).with_context(|| {
                            format!("moving the previous graph aside at {graph_path:?}")
                        })?;
                    }
                    if let Err(e) = std::fs::rename(&staging, &graph_path) {
                        // Put the original back rather than leaving the user with
                        // neither.
                        if previous.exists() {
                            let _ = std::fs::rename(&previous, &graph_path);
                        }
                        let _ = std::fs::remove_dir_all(&staging);
                        return Err(anyhow::anyhow!(
                            "restoring the graph for '{user_id}' failed on swap: {e}. \
                             The previous graph was put back."
                        ));
                    }
                    let _ = std::fs::remove_dir_all(&previous);

                    all_restored.push("graph".to_string());
                    tracing::info!("Graph DB restored from backup");
                }

                Ok(all_restored)
            }
            #[cfg(not(feature = "rocksdb"))]
            StorageBackend::RocksDb => Err(no_rocksdb("backup restore")),
            #[cfg(feature = "storage-redb")]
            StorageBackend::Redb => self.restore_redb_user_backup(user_id, backup_id, max_lsn),
            #[cfg(not(feature = "storage-redb"))]
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend 'redb' requires this build to enable the storage-redb feature"
            )),
        }
    }

    /// Staged import then rename-aside swap, with the tenant evicted and its journal quiesced.
    #[cfg(feature = "storage-redb")]
    fn restore_redb_user_backup(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        max_lsn: Option<u64>,
    ) -> Result<Vec<String>> {
        let engine = self.require_redb_backup_engine("backup restore")?;
        let backup_id = match backup_id {
            Some(id) => id,
            None => engine
                .list(user_id)?
                .first()
                .map(|m| m.backup_id)
                .ok_or_else(|| anyhow::anyhow!("no backups for '{user_id}' to restore"))?,
        };
        let live_user = self.user_dir(user_id)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // Inside the backup root: same filesystem as the data for the renames, and outside any user scan.
        let staging = engine.root().join(format!(
            ".restore-{}-{nanos}",
            live_user
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("user")
        ));
        let result = (|| -> Result<Vec<String>> {
            let report = engine.restore_into(
                user_id,
                backup_id,
                &staging,
                &crate::backup::RestoreOptions { max_lsn },
                false,
            )?;
            let staged_user = staging.join(live_user.file_name().unwrap_or_default());
            let pairs: Vec<(std::path::PathBuf, std::path::PathBuf)> = [
                "storage.redb",
                "graph/graph.redb",
                "intent.log",
                "projection_checkpoints.bin",
            ]
            .iter()
            .map(|rel| (staged_user.join(rel), live_user.join(rel)))
            .collect();

            let writer = self.journaled_writers.get(user_id).map(|w| w.clone());
            let mut writer_guard = writer.as_ref().map(|w| w.lock());
            // The deferred SQLite worker writes the checkpoint file being
            // swapped below, so it is drained and stopped first.
            if let Some(guard) = writer_guard.as_mut() {
                if !guard.shutdown_deferred(crate::intent_log::journal::DEFERRED_DRAIN_TIMEOUT) {
                    tracing::warn!(
                        user_id,
                        "deferred projections not drained before restore; the restored journal replays them",
                    );
                }
            }
            self.evict_user(user_id);
            crate::storage::migrate::backup::swap_files_into_place(&pairs)?;
            // The cached writer appends to the replaced journal; the next access reopens and replays the restored one.
            self.journaled_writers.remove(user_id);
            drop(writer_guard);

            let mut restored = vec!["memories".to_string(), "graph".to_string()];
            if !report.intent_log_files.is_empty() {
                restored.push("intent_log".to_string());
            }
            restored.extend(
                report
                    .not_restored
                    .iter()
                    .map(|n| format!("NOT RESTORED: {n}")),
            );
            tracing::info!(user_id, backup_id, ?restored, "redb backup restored");
            Ok(restored)
        })();
        let _ = std::fs::remove_dir_all(&staging);
        result
    }

    /// Run backups for all active users
    pub fn run_backup_all_users(&self, max_backups: usize) -> usize {
        let mut backed_up = 0;

        let users_path = &self.base_path;
        if let Ok(entries) = std::fs::read_dir(users_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') || name == "audit_logs" || name == "backups" {
                    continue;
                }

                // `storage`, not `memory.db`. The RocksDB store is a DIRECTORY and
                // that file has never existed, so this gate skipped every user on
                // every scheduled run — silently, because the loop just counts what
                // it did back up and nothing asserts that number is plausible. The
                // manual endpoint takes an explicit user_id and bypasses this scan,
                // which is why it kept working and hid the fault. The same path is
                // spelled correctly in restore_user_backup_with_options.
                let db_path = match self.server_config.effective_storage_backend {
                    StorageBackend::RocksDb => path.join("storage"),
                    StorageBackend::Redb => path.join("storage.redb"),
                };
                if !db_path.exists() {
                    tracing::debug!(
                        user_id = name,
                        path = ?db_path,
                        "scheduled backup: skipping, no storage directory",
                    );
                    continue;
                }

                if self.get_user_memory(name).is_ok() {
                    match self.create_user_backup(name) {
                        Ok(metadata) => {
                            tracing::info!(
                                user_id = name,
                                backup_id = metadata.backup_id,
                                size_mb = metadata.size_bytes / 1024 / 1024,
                                "Backup created successfully"
                            );
                            backed_up += 1;

                            if let Err(e) = self.purge_backups_for_user(name, max_backups) {
                                tracing::warn!(
                                    user_id = name,
                                    error = %e,
                                    "Failed to purge old backups"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                user_id = name,
                                error = %e,
                                "Failed to create backup"
                            );
                        }
                    }
                }
            }
        }

        backed_up
    }

    /// Rebuilds a memory's graph episode after its entity set changed: an upsert rewrote it,
    /// or a content-identical duplicate merged in entities or tags. The episode is deleted,
    /// with the edges it sourced, and built again from `experience` by the same pass that
    /// built it, so there is one implementation of edge construction. It scans the
    /// relationships, so callers run it only on a real delta.
    pub fn rebuild_experience_graph(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
    ) -> Result<()> {
        self.rebuild_experience_graph_declared(user_id, experience, memory_id, &[])
    }

    /// [`rebuild_experience_graph`](Self::rebuild_experience_graph) with the writer's declared
    /// entities, as [`process_experience_into_graph_declared`](Self::process_experience_into_graph_declared).
    pub fn rebuild_experience_graph_declared(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
        declared: &[DeclaredEntity],
    ) -> Result<()> {
        if let Ok(graph) = self.get_user_graph(user_id) {
            match graph.read().delete_episode(&memory_id.0) {
                Ok(true) => tracing::debug!(
                    "Cleaned up old episode {} before graph rebuild",
                    &memory_id.0.to_string()[..8]
                ),
                Ok(false) => {}
                Err(e) => tracing::debug!(
                    "Old episode cleanup failed for {} (non-fatal): {}",
                    &memory_id.0.to_string()[..8],
                    e
                ),
            }
        }
        self.process_experience_into_graph_declared(user_id, experience, memory_id, declared)
    }

    /// Process an experience and extract entities/relationships into the graph
    ///
    /// SHO-102: Improved graph building with:
    /// - Neural NER entities
    /// - Tags as Technology/Concept entities
    /// - All-caps terms (API, TUI, NER, etc.)
    /// - Issue IDs (SHO-XX pattern)
    /// - Semantic similarity edges between memories
    pub fn process_experience_into_graph(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
    ) -> Result<()> {
        self.process_experience_into_graph_declared(user_id, experience, memory_id, &[])
    }

    /// [`process_experience_into_graph`](Self::process_experience_into_graph) plus the entities
    /// the writer declared (see [`crate::declared_entities`]).
    ///
    /// Declared entities keep their labels, are appended AFTER the derived list is cut to
    /// `max_entities_per_memory` (so they never compete for a slot), are never stopped by the
    /// generic-entity gate, and their co-occurrence edges carry the declared weight with
    /// `LtpStatus::Weekly` and `EdgeSource::Explicit`, so degree-cap pruning — which removes
    /// unpotentiated edges before potentiated ones — reaches them last. Weekly rather than Burst
    /// because a burst expires to `None` after 48h, and rather than Full because a declaration is
    /// one observation, not ten.
    ///
    /// Derived entities pass a stop list of words measured as hubs on a code-heavy store, and a
    /// derived entity already at `MAX_ENTITY_DEGREE` gets no new edges from this memory: at the
    /// cap, every new edge only bought a prune of an older one.
    pub fn process_experience_into_graph_declared(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
        declared: &[DeclaredEntity],
    ) -> Result<()> {
        let graph = self.get_user_graph(user_id)?;

        // =====================================================================
        // PHASE 1: CPU-INTENSIVE WORK (NO LOCK)
        // All NER, regex, query parsing happens here to minimize lock hold time.
        // Was 100-400ms under lock, now only fast I/O under lock (~10-30ms).
        // =====================================================================

        let now = chrono::Utc::now();

        // Stop words for filtering DERIVED entities. Declared entities never pass through this.
        // The second row is the generic-entity gate: each was a hub at MAX_ENTITY_DEGREE on the
        // live store on 2026-10-07, typed Person/Concept by a news-trained NER reading code-heavy
        // text, and pruned on every new edge. They name no particular thing in that corpus.
        let stop_words: std::collections::HashSet<&str> = [
            "the", "and", "for", "that", "this", "with", "from", "have", "been", "are", "was",
            "were", "will", "would", "could", "should", "may", "might",
        ]
        .iter()
        .chain(GENERIC_DERIVED_ENTITY_WORDS.iter())
        .cloned()
        .collect();
        // A name the writer declared is the declared entity, not also a derived one.
        let declared_names: std::collections::HashSet<String> =
            declared.iter().map(|d| d.name.to_lowercase()).collect();

        // Use pre-extracted NER records for proper entity labels when available
        // This avoids redundant NER inference — the handler already ran NER in Pass 1
        let extracted_entities = if !experience.ner_entities.is_empty() {
            tracing::debug!(
                "Using {} pre-extracted NER entities from handler",
                experience.ner_entities.len()
            );
            experience
                .ner_entities
                .iter()
                .map(|record| crate::embeddings::ner::NerEntity {
                    text: record.text.clone(),
                    entity_type: match record.entity_type.as_str() {
                        "PER" => NerEntityType::Person,
                        "ORG" => NerEntityType::Organization,
                        "LOC" => NerEntityType::Location,
                        _ => NerEntityType::Misc,
                    },
                    confidence: record.confidence,
                    start: record.start_char.unwrap_or(0),
                    end: record.end_char.unwrap_or(record.text.len()),
                })
                .collect()
        } else if !experience.entities.is_empty() {
            tracing::debug!(
                "Using {} pre-extracted entity names (no NER types available)",
                experience.entities.len()
            );
            experience
                .entities
                .iter()
                .map(|name| crate::embeddings::ner::NerEntity {
                    text: name.clone(),
                    entity_type: NerEntityType::Misc,
                    confidence: 0.8,
                    start: 0,
                    end: name.len(),
                })
                .collect()
        } else {
            match self.neural_ner.extract(&experience.content) {
                Ok(entities) => {
                    tracing::debug!(
                        "NER extracted {} entities: {:?}",
                        entities.len(),
                        entities.iter().map(|e| e.text.as_str()).collect::<Vec<_>>()
                    );
                    entities
                }
                Err(e) => {
                    tracing::debug!("NER extraction failed: {}. Continuing without entities.", e);
                    Vec::new()
                }
            }
        };

        // Filter noise entities
        let filtered_entities: Vec<_> = extracted_entities
            .into_iter()
            .filter(|e| {
                let name = e.text.trim();
                if name.len() < 3 {
                    return false;
                }
                if !name.chars().any(|c| c.is_uppercase()) && e.confidence < 0.7 {
                    return false;
                }
                if stop_words.contains(name.to_lowercase().as_str()) {
                    return false;
                }
                if name.len() < 5 && e.confidence < 0.8 {
                    return false;
                }
                if declared_names.contains(&name.to_lowercase()) {
                    return false;
                }
                true
            })
            .collect();

        tracing::debug!(
            "After filtering: {} entities: {:?}",
            filtered_entities.len(),
            filtered_entities
                .iter()
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>()
        );

        // Build NER entity nodes
        let ner_entities: Vec<(String, EntityNode)> = filtered_entities
            .into_iter()
            .map(|ner_entity| {
                let label = match ner_entity.entity_type {
                    NerEntityType::Person => EntityLabel::Person,
                    NerEntityType::Organization => EntityLabel::Organization,
                    NerEntityType::Location => EntityLabel::Location,
                    NerEntityType::Misc => EntityLabel::Other("MISC".to_string()),
                };
                let node = EntityNode {
                    uuid: uuid::Uuid::new_v4(),
                    name: ner_entity.text.clone(),
                    labels: vec![label],
                    created_at: now,
                    last_seen_at: now,
                    mention_count: 1,
                    summary: String::new(),
                    attributes: HashMap::new(),
                    name_embedding: None,
                    salience: ner_entity.confidence,
                    // Only PER, ORG, LOC are proper nouns; MISC includes non-proper
                    // nouns like nationalities, events, etc.
                    is_proper_noun: !matches!(ner_entity.entity_type, NerEntityType::Misc),
                    pii_classification: Default::default(),
                };
                (ner_entity.text, node)
            })
            .collect();

        // Build tag entity nodes
        let tag_entities: Vec<(String, EntityNode)> = experience
            .tags
            .iter()
            .filter_map(|tag| {
                let tag_name = tag.trim();
                let lower = tag_name.to_lowercase();
                if tag_name.len() >= 2
                    && !stop_words.contains(lower.as_str())
                    && !declared_names.contains(&lower)
                {
                    Some((
                        tag_name.to_string(),
                        EntityNode {
                            uuid: uuid::Uuid::new_v4(),
                            name: tag_name.to_string(),
                            labels: vec![EntityLabel::Technology],
                            created_at: now,
                            last_seen_at: now,
                            mention_count: 1,
                            summary: String::new(),
                            attributes: HashMap::new(),
                            name_embedding: None,
                            salience: 0.6,
                            is_proper_noun: false,
                            pii_classification: Default::default(),
                        },
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Collect names already covered (for dedup in regex/verb phases)
        let mut known_names: Vec<String> = ner_entities
            .iter()
            .map(|(name, _)| name.clone())
            .chain(tag_entities.iter().map(|(name, _)| name.clone()))
            .chain(declared.iter().map(|d| d.name.clone()))
            .collect();

        // Extract all-caps terms (API, TUI, NER, REST, etc.)
        let allcaps_entities: Vec<(String, EntityNode)> = allcaps_regex()
            .find_iter(&experience.content)
            .filter_map(|cap| {
                let term = cap.as_str();
                if known_names
                    .iter()
                    .any(|name| name.eq_ignore_ascii_case(term))
                {
                    return None;
                }
                if stop_words.contains(term.to_lowercase().as_str()) {
                    return None;
                }
                known_names.push(term.to_string());
                Some((
                    term.to_string(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: term.to_string(),
                        labels: vec![EntityLabel::Technology],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: 0.5,
                        is_proper_noun: true,
                        pii_classification: Default::default(),
                    },
                ))
            })
            .collect();

        // Extract issue IDs (SHO-XX, JIRA-123, etc.)
        let issue_entities: Vec<(String, EntityNode)> = issue_regex()
            .find_iter(&experience.content)
            .filter_map(|issue| {
                let issue_id = issue.as_str();
                if known_names.iter().any(|name| name == issue_id) {
                    return None;
                }
                known_names.push(issue_id.to_string());
                Some((
                    issue_id.to_string(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: issue_id.to_string(),
                        labels: vec![EntityLabel::Other("Issue".to_string())],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: 0.7,
                        is_proper_noun: true,
                        pii_classification: Default::default(),
                    },
                ))
            })
            .collect();

        // Extract verbs for multi-hop reasoning
        let analysis = query_parser::analyze_query(&experience.content);
        let mut verb_entities: Vec<(String, EntityNode)> = Vec::new();
        for verb in &analysis.relational_context {
            let verb_text = verb.text.as_str();
            let verb_stem = verb.stem.as_str();

            if known_names
                .iter()
                .any(|name| name.eq_ignore_ascii_case(verb_text))
            {
                continue;
            }
            if stop_words.contains(verb_text.to_lowercase().as_str()) {
                continue;
            }
            if verb_text.len() < 3 {
                continue;
            }

            for name in [verb_text, verb_stem] {
                if name.len() < 3 {
                    continue;
                }
                if known_names.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                    continue;
                }
                known_names.push(name.to_string());
                verb_entities.push((
                    name.to_string(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: name.to_string(),
                        labels: vec![EntityLabel::Other("Verb".to_string())],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: 0.4,
                        is_proper_noun: false,
                        pii_classification: Default::default(),
                    },
                ));
            }
        }

        // Combine all entity groups for insertion, capped at 10 to prevent
        // O(n²) edge explosion (10 entities → max 45 edges)
        let mut all_entities: Vec<(String, EntityNode)> = ner_entities
            .into_iter()
            .chain(tag_entities)
            .chain(allcaps_entities)
            .chain(issue_entities)
            .chain(verb_entities)
            .collect();
        all_entities.sort_by(|a, b| b.1.salience.total_cmp(&a.1.salience));
        let entity_cap = self.server_config.max_entities_per_memory;
        all_entities.truncate(entity_cap);

        // Declared entities go in after the cut, so the cap bounds only what veld derived.
        let declared_nodes: Vec<(String, EntityNode, f32)> = declared
            .iter()
            .map(|d| {
                (
                    d.name.clone(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: d.name.clone(),
                        labels: d.labels.clone(),
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: d.weight,
                        // An identifier, not a word: keeps it out of the stemmed merge tier.
                        is_proper_noun: true,
                        pii_classification: Default::default(),
                    },
                    d.weight,
                )
            })
            .collect();

        // =====================================================================
        // PHASE 2: GRAPH INSERTION (WITH LOCK)
        // Only fast I/O operations happen here.
        // =====================================================================

        let graph_guard = graph.read();

        let mut entity_uuids = Vec::new();
        // Per entry of `entity_uuids`: Some(weight) for a declared entity, None for derived.
        let mut declared_weight: Vec<Option<f32>> = Vec::new();
        // Per entry: a derived entity already at the degree cap takes no new edges.
        let mut at_cap: Vec<bool> = Vec::new();

        // Insert all pre-built entities
        for (name, entity) in all_entities {
            match graph_guard.add_entity(entity) {
                Ok(uuid) => {
                    let hub = match graph_guard.entity_degree(&uuid) {
                        Ok(degree) => degree >= crate::constants::MAX_ENTITY_DEGREE,
                        Err(e) => {
                            tracing::debug!("Degree read failed for {}: {}", name, e);
                            false
                        }
                    };
                    entity_uuids.push((name, uuid));
                    declared_weight.push(None);
                    at_cap.push(hub);
                }
                Err(e) => tracing::debug!("Failed to add entity {}: {}", name, e),
            }
        }
        for (name, entity, weight) in declared_nodes {
            match graph_guard.add_entity(entity) {
                Ok(uuid) => {
                    entity_uuids.push((name, uuid));
                    declared_weight.push(Some(weight));
                    at_cap.push(false);
                }
                Err(e) => tracing::debug!("Failed to add declared entity {}: {}", name, e),
            }
        }

        // Create episodic node
        tracing::debug!(
            "Creating episode for memory {} with {} entities: {:?}",
            &memory_id.0.to_string()[..8],
            entity_uuids.len(),
            entity_uuids
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
        );

        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: format!("Memory {}", &memory_id.0.to_string()[..8]),
            content: experience.content.clone(),
            valid_at: now,
            created_at: now,
            entity_refs: entity_uuids.iter().map(|(_, uuid)| *uuid).collect(),
            source: EpisodeSource::Message,
            metadata: experience.metadata.clone(),
        };

        match graph_guard.add_episode(episode) {
            Ok(uuid) => {
                tracing::debug!(
                    "Episode {} added with {} entity refs",
                    &uuid.to_string()[..8],
                    entity_uuids.len()
                );
            }
            Err(e) => {
                tracing::warn!("Failed to add episode: {}", e);
            }
        }

        // Create relationships between co-occurring entities
        // Pre-compute truncated context once (avoids re-allocating per edge)
        let truncated_context: String = experience.content.chars().take(150).collect();

        // Upstream's SHODH_GRAPH_EXTRACTED_PREDICATES gate: on unless set to 0/false.
        let extract_predicates: bool = std::env::var("SHODH_GRAPH_EXTRACTED_PREDICATES")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        // A name contained in a co-mentioned name ("Mor" in "the Morwen incident") is a
        // recognizer fragment; a causal edge on it welds unrelated chains together.
        let fragment_of_comention: Vec<bool> = (0..entity_uuids.len())
            .map(|k| {
                let name_k = entity_uuids[k].0.to_lowercase();
                entity_uuids.iter().enumerate().any(|(m, other)| {
                    m != k && {
                        let name_m = other.0.to_lowercase();
                        name_m.len() > name_k.len() && name_m.contains(name_k.as_str())
                    }
                })
            })
            .collect();

        let mut hub_edges_skipped = 0u64;
        for i in 0..entity_uuids.len() {
            for j in (i + 1)..entity_uuids.len() {
                if at_cap[i] || at_cap[j] {
                    hub_edges_skipped += 1;
                    continue;
                }
                let declared_strength = match (declared_weight[i], declared_weight[j]) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) | (None, Some(a)) => Some(a),
                    (None, None) => None,
                };
                let state_edge_init =
                    declared_strength.unwrap_or_else(|| EdgeTier::L1Working.initial_weight());
                let mut from_entity = entity_uuids[i].1;
                let mut to_entity = entity_uuids[j].1;
                let cue_hit =
                    if extract_predicates && !fragment_of_comention[i] && !fragment_of_comention[j]
                    {
                        crate::graph_memory::extract_directed_predicate(
                            &experience.content,
                            &entity_uuids[i].0,
                            &entity_uuids[j].0,
                        )
                    } else {
                        None
                    };
                let relation_type = match cue_hit {
                    Some((rt, a_is_source)) => {
                        if !a_is_source {
                            std::mem::swap(&mut from_entity, &mut to_entity);
                        }
                        rt
                    }
                    None => RelationType::RelatedTo,
                };
                let edge = RelationshipEdge {
                    uuid: uuid::Uuid::new_v4(),
                    from_entity,
                    to_entity,
                    relation_type,
                    strength: state_edge_init,
                    created_at: now,
                    valid_at: now,
                    invalidated_at: None,
                    source_episode_id: Some(memory_id.0),
                    context: truncated_context.clone(),
                    last_activated: now,
                    activation_count: 1,
                    ltp_status: if declared_strength.is_some() {
                        LtpStatus::Weekly
                    } else {
                        LtpStatus::None
                    },
                    tier: EdgeTier::L1Working,
                    activation_timestamps: None,
                    entity_confidence: None,
                    created_by: if declared_strength.is_some() {
                        crate::graph_memory::EdgeSource::Explicit
                    } else {
                        crate::graph_memory::EdgeSource::CoOccurrence
                    },
                    forward_strength: state_edge_init,
                    backward_strength: state_edge_init,
                    strengthening_sources: Default::default(),
                    last_corroborated: None,
                };

                if let Err(e) = graph_guard.add_relationship(edge) {
                    tracing::debug!("Failed to add relationship: {}", e);
                }
            }
        }
        if hub_edges_skipped > 0 {
            crate::metrics::GRAPH_HUB_EDGES_SKIPPED.inc_by(hub_edges_skipped);
            tracing::debug!(
                memory_id = %memory_id.0,
                skipped = hub_edges_skipped,
                hubs = ?entity_uuids
                    .iter()
                    .zip(&at_cap)
                    .filter(|(_, hub)| **hub)
                    .map(|((name, _), _)| name.as_str())
                    .collect::<Vec<_>>(),
                "Skipped co-occurrence edges on entities already at the degree cap"
            );
        }
        // Lock released here

        Ok(())
    }
}

/// Words that never become DERIVED graph entities (matched case-insensitively). Declared
/// entities are not subject to this list.
pub const GENERIC_DERIVED_ENTITY_WORDS: &[&str] = &[
    "file",
    "files",
    "code",
    "first",
    "moved",
    "line",
    "lines",
    "test",
    "description",
    "operator",
];

/// W6 query-planner integration: surface each tenant's live Vamana
/// indices to the planner's vector adapter (`RealVectorQuerier`).
///
/// The adapter needs two things per `(user_id, kind)`: the shared
/// `Arc<RwLock<VamanaIndex>>` to search, and the reverse
/// `(vector_id → memory_id)` map to resolve hits back to memory ids.
/// Both live on the per-kind [`crate::vector_db::VamanaProjection`],
/// which the write path *moves* into the tenant's
/// [`crate::intent_log::JournaledWriter`] and which then stays there for
/// the tenant's lifetime. Rather than duplicate that state into a
/// parallel map — which would drift on every write and could never
/// capture a *lazy* projection's index (it only materialises on the
/// first embedded memory of that kind) — we reach through the live
/// writer to the projection that owns it.
///
/// `kind` is the projection-name string carried in the `VectorPredicate`
/// (e.g. `"vamana-text-primary"`), which is exactly what
/// [`crate::intent_log::TypedProjection::name`] reports, so resolution is
/// a direct name match. Each call clones the index `Arc` out and releases
/// the writer mutex immediately; the ANN search runs under the index's
/// own `RwLock`, never under the writer lock.
impl crate::query_planner::adapters::VamanaProvider for MultiUserMemoryManager {
    fn index(
        &self,
        user_id: &str,
        kind: &str,
    ) -> Option<std::sync::Arc<parking_lot::RwLock<crate::vector_db::VamanaIndex>>> {
        let writer = self.get_user_journaled_writer(user_id).ok()?;
        let guard = writer.lock();
        let projection = guard.projection_by_name(kind)?;
        let vamana = projection
            .as_any()
            .downcast_ref::<crate::vector_db::VamanaProjection>()?;
        // `index()` is `None` while the projection is still lazy (no
        // embedded memory of this kind yet); that resolves to "no
        // candidates" upstream, which is correct.
        vamana.index().cloned()
    }

    fn memory_id_for(&self, user_id: &str, kind: &str, vector_id: u32) -> Option<String> {
        let writer = self.get_user_journaled_writer(user_id).ok()?;
        let guard = writer.lock();
        let projection = guard.projection_by_name(kind)?;
        let vamana = projection
            .as_any()
            .downcast_ref::<crate::vector_db::VamanaProjection>()?;
        vamana.memory_id_for(vector_id)
    }
}

/// W6 query-planner integration: surface each tenant's entity/episode
/// topology to the planner's graph adapter (`RealGraphQuerier`).
///
/// The memory↔entity link lives on the `Memory` itself (its `entity_refs`
/// set), not in `GraphMemory`, so these primitives read the tenant's
/// `UserMemory`. Each returns owned ids and releases the `UserMemory` read lock
/// before returning — the querier never holds it across an `await`.
///
/// `memory_ids_with_entity` / `memory_ids_in_episode` scan the tenant's
/// live memory set (`get_all_memories`); that is `O(n)` in the corpus.
/// This is the query-planner surface, not the hot recall path, so the
/// linear scan is acceptable for v1 — a reverse `entity → memory_ids`
/// index is a future optimisation if the graph stage becomes a leading
/// scan on large corpora.
impl crate::query_planner::adapters::GraphProvider for MultiUserMemoryManager {
    fn entity_ids_of_memory(&self, user_id: &str, memory_id: &str) -> Vec<uuid::Uuid> {
        let user_mem = match self.get_user_memory(user_id) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let parsed = match uuid::Uuid::parse_str(memory_id) {
            Ok(u) => crate::memory::MemoryId(u),
            Err(_) => return Vec::new(),
        };
        let guard = user_mem.read();
        match guard.get_memory(&parsed) {
            Ok(memory) => memory.entity_ids(),
            Err(_) => Vec::new(),
        }
    }

    fn memory_ids_with_entity(&self, user_id: &str, entity_id: &uuid::Uuid) -> Vec<String> {
        let user_mem = match self.get_user_memory(user_id) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let guard = user_mem.read();
        let memories = match guard.get_all_memories() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        memories
            .iter()
            .filter(|m| m.entity_ids().contains(entity_id))
            .map(|m| m.id.0.to_string())
            .collect()
    }

    fn memory_ids_in_episode(&self, user_id: &str, episode_id: &uuid::Uuid) -> Vec<String> {
        let user_mem = match self.get_user_memory(user_id) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let guard = user_mem.read();
        let memories = match guard.get_all_memories() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let target = episode_id.to_string();
        memories
            .iter()
            .filter(|m| {
                m.experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.episode_id.as_deref())
                    == Some(target.as_str())
            })
            .map(|m| m.id.0.to_string())
            .collect()
    }
}

#[cfg(test)]
mod shared_store_bootstrap_tests {
    use super::*;

    fn boot_backend(
        base: &std::path::Path,
        backend: crate::config::StorageBackend,
    ) -> Result<SharedStoreBootstrap> {
        let config = ServerConfig {
            effective_storage_backend: backend,
            ..ServerConfig::default()
        };
        #[cfg(feature = "rocksdb")]
        let cache = Some(rocksdb::Cache::new_lru_cache(1024));
        #[cfg(not(feature = "rocksdb"))]
        let cache: Option<crate::storage::SharedBlockCache> = None;
        MultiUserMemoryManager::bootstrap_shared_stores(base, &config, cache.as_ref())
    }

    // ADR 0006: a rocksdb arm that binds a different family or key layout serves an empty store over live data.
    #[cfg(feature = "rocksdb")]
    mod rocksdb_on_disk_compat {
        use super::*;
        use crate::memory::types::{
            FileMemory, FileType, ProjectId, ProspectiveTask, ProspectiveTrigger, Todo,
            TodoPriority, TodoStatus,
        };
        use crate::memory::ContextBlockStore;
        use serde_json::{json, Value};

        fn boot(base: &std::path::Path) -> SharedStoreBootstrap {
            boot_backend(base, crate::config::StorageBackend::RocksDb)
                .expect("rocksdb shared-store bootstrap")
        }

        fn to_json<T: serde::Serialize>(v: &T) -> Value {
            serde_json::to_value(v).unwrap()
        }

        // Literal family names on purpose: a renamed constant would move both sides of a round trip.
        fn raw_has(bundle: &SharedStoreBootstrap, cf: &str, key: &str) -> bool {
            let db = bundle.shared_db.rocksdb().expect("rocksdb engine");
            let handle = db
                .cf_handle(cf)
                .unwrap_or_else(|| panic!("deployed family '{cf}' missing"));
            db.get_cf(handle, key.as_bytes()).unwrap().is_some()
        }

        fn write_prospective(store: &ProspectiveStore) -> Vec<ProspectiveTask> {
            let past = chrono::Utc::now() - chrono::Duration::hours(2);
            let mut due = ProspectiveTask::new(
                "alice".into(),
                "renew the certificate".into(),
                ProspectiveTrigger::AtTime { at: past },
            );
            due.tags = vec!["ops".into()];
            due.priority = 4;
            let context = ProspectiveTask::new(
                "alice".into(),
                "mention the migration".into(),
                ProspectiveTrigger::OnContext {
                    keywords: vec!["Deploy".into(), "release".into()],
                    threshold: 0.7,
                },
            );
            let dismissed = ProspectiveTask::new(
                "bob".into(),
                "call back".into(),
                ProspectiveTrigger::AfterDuration {
                    seconds: 86_400,
                    from: chrono::Utc::now(),
                },
            );
            for t in [&due, &context, &dismissed] {
                store.store(t).unwrap();
            }
            assert!(store.mark_dismissed("bob", &dismissed.id).unwrap());
            vec![due, context, dismissed]
        }

        fn read_prospective(store: &ProspectiveStore, tasks: &[ProspectiveTask]) -> Value {
            let gets: Vec<Value> = tasks
                .iter()
                .map(|t| to_json(&store.get(&t.user_id, &t.id).unwrap()))
                .collect();
            let prefix = &tasks[0].id.0.to_string()[..8];
            json!({
                "gets": gets,
                "alice_all": to_json(&store.list_for_user("alice", None).unwrap()),
                "alice_pending": to_json(&store.list_for_user("alice", Some(crate::memory::types::ProspectiveTaskStatus::Pending)).unwrap()),
                "bob_dismissed": to_json(&store.list_for_user("bob", Some(crate::memory::types::ProspectiveTaskStatus::Dismissed)).unwrap()),
                "alice_due": to_json(&store.get_due_tasks("alice").unwrap()),
                "all_due": to_json(&store.get_all_due_tasks().unwrap()),
                "context": to_json(&store.check_context_triggers("alice", "ready to deploy now").unwrap()),
                "pending_count": store.pending_count("alice").unwrap(),
                "by_prefix": to_json(&store.find_by_prefix("alice", prefix).unwrap()),
            })
        }

        #[test]
        fn rocksdb_prospective_rows_survive_reboot_through_with_store() {
            let tmp = tempfile::tempdir().unwrap();
            let (tasks, expected) = {
                let bundle = boot(tmp.path());
                let tasks = write_prospective(&bundle.prospective_store);
                let expected = read_prospective(&bundle.prospective_store, &tasks);
                (tasks, expected)
            };
            assert_eq!(
                expected["gets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|v| v.is_null())
                    .count(),
                0
            );
            assert_eq!(expected["alice_due"].as_array().unwrap().len(), 1);
            assert_eq!(expected["context"].as_array().unwrap().len(), 1);

            let bundle = boot(tmp.path());
            let t = &tasks[0];
            assert!(raw_has(&bundle, "prospective", &format!("alice:{}", t.id)));
            assert!(raw_has(
                &bundle,
                "prospective_index",
                &format!("user:alice:{}", t.id)
            ));
            assert_eq!(
                read_prospective(&bundle.prospective_store, &tasks),
                expected
            );
        }

        struct TodoFixture {
            todos: Vec<Todo>,
            project: crate::memory::types::Project,
        }

        fn write_todos(store: &TodoStore) -> TodoFixture {
            let project = store
                .find_or_create_project("alice", "Backend Rewrite")
                .unwrap();
            let mut parent = Todo::new("alice".into(), "ship the storage traits".into());
            parent.project_id = Some(project.id.clone());
            parent.contexts = vec!["@Computer".into()];
            parent.priority = TodoPriority::High;
            parent.due_date = Some(chrono::Utc::now() - chrono::Duration::days(1));
            parent.tags = vec!["adr-0006".into()];
            let parent = store.store_todo(&parent).unwrap();
            let mut child = Todo::new("alice".into(), "write the compat test".into());
            child.parent_id = Some(parent.id);
            child.status = TodoStatus::InProgress;
            child.blocked_on = Some(parent.id.0.to_string());
            let child = store.store_todo(&child).unwrap();
            let mut external = Todo::new("bob".into(), "imported".into());
            external.external_id = Some("LIN-42".into());
            let external = store.store_todo(&external).unwrap();
            store
                .add_comment("alice", &parent.id, "alice".into(), "halfway".into(), None)
                .unwrap()
                .unwrap();
            TodoFixture {
                todos: vec![parent, child, external],
                project,
            }
        }

        fn read_todos(store: &TodoStore, fx: &TodoFixture) -> Value {
            let parent = &fx.todos[0];
            let gets: Vec<Value> = fx
                .todos
                .iter()
                .map(|t| to_json(&store.get_todo(&t.user_id, &t.id).unwrap()))
                .collect();
            json!({
                "gets": gets,
                "alice_all": to_json(&store.list_todos_for_user("alice", None).unwrap()),
                "alice_in_progress": to_json(&store.list_todos_for_user("alice", Some(&[TodoStatus::InProgress])).unwrap()),
                "by_project": to_json(&store.list_todos_by_project("alice", &fx.project.id).unwrap()),
                "by_context": to_json(&store.list_todos_by_context("alice", "@computer").unwrap()),
                "due": to_json(&store.list_due_todos("alice", true).unwrap()),
                "subtasks": to_json(&store.list_subtasks(&parent.id).unwrap()),
                "external": to_json(&store.find_by_external_id("bob", "LIN-42").unwrap()),
                "short_id": to_json(&store.find_todo_by_prefix("alice", &parent.short_id()).unwrap()),
                "comments": to_json(&store.get_comments("alice", &parent.id).unwrap()),
                "projects": to_json(&store.list_projects("alice").unwrap()),
                "project_by_name": to_json(&store.find_project_by_name("alice", "backend rewrite").unwrap()),
                "project_stats": to_json(&store.get_project_stats("alice", &fx.project.id).unwrap()),
                "user_stats": to_json(&store.get_user_stats("alice").unwrap()),
            })
        }

        #[test]
        fn rocksdb_todos_rows_survive_reboot_through_with_store() {
            let tmp = tempfile::tempdir().unwrap();
            let (fx, expected) = {
                let bundle = boot(tmp.path());
                let fx = write_todos(&bundle.todo_store);
                let expected = read_todos(&bundle.todo_store, &fx);
                (fx, expected)
            };
            for key in [
                "by_project",
                "by_context",
                "due",
                "subtasks",
                "comments",
                "projects",
            ] {
                assert!(
                    !expected[key].as_array().unwrap().is_empty(),
                    "fixture reaches {key}"
                );
            }

            let bundle = boot(tmp.path());
            let parent = &fx.todos[0];
            assert!(raw_has(&bundle, "todos", &format!("alice:{}", parent.id.0)));
            assert!(raw_has(
                &bundle,
                "todo_index",
                &format!("user:alice:{}", parent.id.0)
            ));
            assert!(raw_has(
                &bundle,
                "projects",
                &format!("alice:{}", fx.project.id.0)
            ));
            let store = &bundle.todo_store;
            assert_eq!(read_todos(store, &fx), expected);

            // The sequence counter lives in the index family; a reset would reissue short ids.
            let next = store
                .store_todo(&Todo::new("alice".into(), "after reopen".into()))
                .unwrap();
            let max_standalone = fx.todos[1].seq_num;
            assert_eq!(next.seq_num, max_standalone + 1);
        }

        fn write_files(store: &FileMemoryStore) -> (ProjectId, Vec<FileMemory>) {
            let project = ProjectId::new();
            let mut a = FileMemory::new(
                project.clone(),
                "alice".into(),
                "src/main.rs".into(),
                "/repo/src/main.rs".into(),
                "hash-a".into(),
                FileType::Rust,
                120,
                4096,
            );
            a.summary = "entry point".into();
            let b = FileMemory::new(
                project.clone(),
                "alice".into(),
                "docs/index.md".into(),
                "/repo/docs/index.md".into(),
                "hash-b".into(),
                FileType::Markdown,
                30,
                900,
            );
            for f in [&a, &b] {
                store.store(f).unwrap();
            }
            (project, vec![a, b])
        }

        fn read_files(store: &FileMemoryStore, project: &ProjectId, fs: &[FileMemory]) -> Value {
            let gets: Vec<Value> = fs
                .iter()
                .map(|f| to_json(&store.get(&f.user_id, &f.id).unwrap()))
                .collect();
            json!({
                "gets": gets,
                "by_path": to_json(&store.get_by_path("alice", project, "src/main.rs").unwrap()),
                "by_user": to_json(&store.list_by_user("alice", None).unwrap()),
                "by_project": to_json(&store.list_by_project("alice", project, None).unwrap()),
                "by_type": to_json(&store.list_by_type("alice", project, &FileType::Rust, None).unwrap()),
                "count": store.count_by_project("alice", project).unwrap(),
                "stats": to_json(&store.stats("alice").unwrap()),
            })
        }

        #[test]
        fn rocksdb_files_rows_survive_reboot_through_with_store() {
            let tmp = tempfile::tempdir().unwrap();
            let (project, fs, expected) = {
                let bundle = boot(tmp.path());
                let (project, fs) = write_files(&bundle.file_store);
                let expected = read_files(&bundle.file_store, &project, &fs);
                (project, fs, expected)
            };
            assert!(!expected["by_path"].is_null());
            assert_eq!(expected["by_type"].as_array().unwrap().len(), 1);
            assert_eq!(expected["count"], 2);

            let bundle = boot(tmp.path());
            assert!(raw_has(&bundle, "files", &format!("alice:{}", fs[0].id.0)));
            assert!(raw_has(
                &bundle,
                "file_index",
                &format!("user:alice:{}", fs[0].id.0)
            ));
            assert_eq!(read_files(&bundle.file_store, &project, &fs), expected);
        }

        fn read_blocks(store: &ContextBlockStore) -> Value {
            json!({
                "persona": to_json(&store.get("alice", "persona").unwrap()),
                "state": to_json(&store.get("alice", "project_state").unwrap()),
                "bob": to_json(&store.get("bob", "persona").unwrap()),
                "alice_list": to_json(&store.list("alice").unwrap()),
                "bob_list": to_json(&store.list("bob").unwrap()),
            })
        }

        #[test]
        fn rocksdb_context_blocks_rows_survive_reboot_through_with_store() {
            let tmp = tempfile::tempdir().unwrap();
            let expected = {
                let bundle = boot(tmp.path());
                let store = &bundle.context_block_store;
                store
                    .set("alice", "persona", "terse reviewer", Some(500))
                    .unwrap();
                store
                    .set("alice", "persona", "terse reviewer, v2", None)
                    .unwrap();
                store
                    .set("alice", "project_state", "WP3 in flight", None)
                    .unwrap();
                store.set_locked("alice", "project_state", true).unwrap();
                store.set("bob", "persona", "someone else", None).unwrap();
                read_blocks(store)
            };
            assert_eq!(expected["persona"]["version"], 2);
            assert_eq!(expected["state"]["locked"], true);
            assert_eq!(expected["alice_list"].as_array().unwrap().len(), 2);

            let bundle = boot(tmp.path());
            assert!(raw_has(&bundle, "context_blocks", "alice:persona"));
            assert_eq!(read_blocks(&bundle.context_block_store), expected);
        }

        fn read_feedback(store: &FeedbackStore, memory: &crate::memory::MemoryId) -> Value {
            json!({
                "pending": to_json(&store.get_pending("alice")),
                "prev_ctx": to_json(&store.get_previous_context("alice")),
                "momentum": to_json(&store.get_momentum(memory)),
                "stats": to_json(&store.stats()),
            })
        }

        #[test]
        fn rocksdb_feedback_rows_survive_reboot_through_with_store() {
            let tmp = tempfile::tempdir().unwrap();
            let memory = crate::memory::MemoryId(uuid::Uuid::new_v4());
            let expected = {
                let bundle = boot(tmp.path());
                let mut store = bundle.feedback_store.write();
                let surfaced = crate::memory::feedback::SurfacedMemoryInfo {
                    id: memory.clone(),
                    entities: ["rocksdb".to_string()].into_iter().collect(),
                    content_preview: "column families".into(),
                    score: 0.8,
                    embedding: vec![0.1, 0.2],
                };
                store.set_pending(crate::memory::feedback::PendingFeedback::new(
                    "alice".into(),
                    "how is storage opened".into(),
                    vec![0.3, 0.4],
                    vec![surfaced],
                ));
                store.set_previous_context(
                    "alice",
                    "how is storage opened".into(),
                    vec![0.3, 0.4],
                    vec![memory.clone()],
                );
                let m = store.get_or_create_momentum(
                    memory.clone(),
                    crate::memory::types::ExperienceType::Learning,
                );
                m.ema = 0.42;
                m.signal_count = 3;
                store.mark_dirty(&memory);
                store.flush().unwrap();
                read_feedback(&store, &memory)
            };
            assert!(!expected["pending"].is_null());
            assert!(!expected["prev_ctx"].is_null());
            assert_eq!(expected["stats"]["total_momentum_entries"], 1);

            let bundle = boot(tmp.path());
            assert!(raw_has(&bundle, "feedback", "pending:alice"));
            assert!(raw_has(
                &bundle,
                "feedback",
                &format!("momentum:{}", memory.0)
            ));
            assert!(bundle.feedback_store.read().is_persistent());
            assert_eq!(
                read_feedback(&bundle.feedback_store.read(), &memory),
                expected
            );
        }

        #[test]
        fn rocksdb_bootstrap_imports_pre_column_family_dbs() {
            let tmp = tempfile::tempdir().unwrap();
            let legacy: Vec<(&str, &str)> = ProspectiveStore::LEGACY_SEPARATE_DBS
                .iter()
                .chain(TodoStore::LEGACY_SEPARATE_DBS)
                .chain(FileMemoryStore::LEGACY_SEPARATE_DBS)
                .chain(FeedbackStore::LEGACY_SEPARATE_DBS)
                .copied()
                .collect();
            assert_eq!(legacy.len(), 8);

            let todo = Todo::new("alice".into(), "written before column families".into());
            for (dir, cf) in &legacy {
                let mut rows = vec![(format!("marker:{cf}").into_bytes(), b"1".to_vec())];
                if *cf == "todos" {
                    let key = format!("alice:{}", todo.id.0);
                    rows.push((key.into_bytes(), serde_json::to_vec(&todo).unwrap()));
                }
                if *cf == "todo_index" {
                    rows.push((
                        format!("user:alice:{}", todo.id.0).into_bytes(),
                        b"1".to_vec(),
                    ));
                }
                crate::storage::legacy_rocksdb::create_separate_db_for_test(
                    &tmp.path().join(dir),
                    &rows,
                )
                .unwrap();
            }

            for _ in 0..2 {
                let bundle = boot(tmp.path());
                for (dir, cf) in &legacy {
                    assert!(
                        raw_has(&bundle, cf, &format!("marker:{cf}")),
                        "{dir} -> {cf}"
                    );
                    let old = tmp.path().join(dir);
                    assert!(!old.exists(), "{dir} left in place");
                    let name = old.file_name().unwrap().to_string_lossy().into_owned();
                    assert!(old
                        .with_file_name(format!("{name}.pre_cf_migration"))
                        .is_dir());
                }
                let got = bundle.todo_store.get_todo("alice", &todo.id).unwrap();
                assert_eq!(got.map(|t| t.content), Some(todo.content.clone()));
                assert_eq!(
                    bundle
                        .todo_store
                        .list_todos_for_user("alice", None)
                        .unwrap()
                        .len(),
                    1
                );
            }
        }
    }

    /// The redb arm of bootstrap_shared_stores BOOTS and its stores WORK.
    /// The compile-only version of this claim is form satisfied, nothing fed:
    /// the arm could construct stores over the wrong tables and nothing would
    /// look different until a live cutover.
    #[cfg(feature = "storage-redb")]
    #[test]
    fn redb_shared_store_bootstrap_boots_and_stores_work() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = boot_backend(tmp.path(), crate::config::StorageBackend::Redb)
            .expect("redb shared-store bootstrap");

        // The engine is the one asked for.
        #[cfg(feature = "rocksdb")]
        assert!(bundle.shared_db.rocksdb().is_none());

        // A todo round-trips through the bundle's store (crosses all three
        // todo families via seq_num allocation and indices).
        let todo = crate::memory::types::Todo::new("boot-user".into(), "works".into());
        let stored = bundle.todo_store.store_todo(&todo).unwrap();
        assert!(bundle
            .todo_store
            .get_todo("boot-user", &stored.id)
            .unwrap()
            .is_some());

        // Audit appends and enumerates through the trait.
        bundle
            .audit_store
            .append_event(
                "boot-user",
                &crate::storage::AuditLogEntry {
                    timestamp: chrono::Utc::now(),
                    event_type: "test".into(),
                    memory_id: "m".into(),
                    details: "d".into(),
                },
            )
            .unwrap();
        assert_eq!(
            bundle.audit_store.list_user_ids().unwrap(),
            vec!["boot-user".to_string()]
        );

        // Context blocks and feedback stores answer over their tables.
        assert!(bundle
            .context_block_store
            .get("boot-user", "nope")
            .unwrap()
            .is_none());
    }

    /// The graph half of the boot: arm 1060 over redb, exercised.
    #[cfg(feature = "storage-redb")]
    #[test]
    fn redb_graph_memory_boots_and_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let graph_path = tmp.path().join("graph");
        std::fs::create_dir_all(&graph_path).unwrap();
        let database = crate::storage::redb::open_graph_database(&graph_path).unwrap();
        let graph = crate::graph_memory::GraphMemory::with_store(std::sync::Arc::new(
            crate::storage::redb::RedbColumnStore::new(database),
        ))
        .expect("redb graph boots");

        // No engine DB behind it — the backup accessor must say so rather than
        // hand a caller something that cannot exist.
        #[cfg(feature = "rocksdb")]
        assert!(graph.rocksdb_for_backup().is_none());

        // A round trip across two families: the entity row and its name index.
        let entity = crate::graph_memory::EntityNode {
            uuid: uuid::Uuid::new_v4(),
            name: "Ada Lovelace".to_string(),
            labels: vec![crate::graph_memory::EntityLabel::Person],
            created_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            pii_classification: crate::graph_memory::PiiClassification::PersonalIdentity,
        };
        let uuid = graph.add_entity(entity).expect("add_entity over redb");
        let fetched = graph
            .get_entity(&uuid)
            .expect("get_entity")
            .expect("present");
        assert_eq!(fetched.name, "Ada Lovelace");

        // The in-memory name index was populated from the same write path that
        // a restart would reload, so the two agree by construction.
        let stats = graph.get_stats().expect("stats");
        assert_eq!(stats.entity_count, 1);
    }
}

/// Annotate a query with NER-extracted entity names.
///
/// Off with `VELD_QUERY_NER=0`. Entities below 0.5 confidence or shorter than
/// two characters are dropped, and at most eight are kept — the same filter
/// upstream applies, so a harness run is comparable across both projects.
pub fn annotate_query_ner_with(ner: &NeuralNer, query: &mut crate::memory::types::Query) {
    let enabled = std::env::var("VELD_QUERY_NER")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true);
    if !enabled {
        return;
    }
    let Some(text) = query.query_text.as_deref() else {
        return;
    };
    match ner.extract(text) {
        Ok(entities) => {
            let names: Vec<String> = entities
                .into_iter()
                .filter(|e| e.confidence >= 0.5 && e.text.trim().len() >= 2)
                .map(|e| e.text)
                .take(8)
                .collect();
            if !names.is_empty() {
                query.ner_entities = Some(names);
            }
        }
        Err(e) => tracing::debug!("query NER annotation failed: {e}"),
    }
}

fn wait_until_sole_owner<T: ?Sized>(
    held: &Arc<T>,
    deadline: std::time::Instant,
    user_id: &str,
    within: std::time::Duration,
) -> Result<()> {
    while Arc::strong_count(held) > 1 {
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "refusing to restore '{user_id}': its stores are still in use after {within:?}, \
                 by work a recent write started; retry once it finishes"
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Ok(())
}

#[cfg(not(feature = "rocksdb"))]
fn no_rocksdb(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{what} needs the rocksdb engine, and this build has none; rebuild veld with `--features rocksdb`"
    )
}

/// Delete every row the shared stores keep for `user_id` and return how many went.
pub(crate) fn purge_user_from_shared_store(
    store: &dyn crate::storage::ColumnStore,
    user_id: &str,
) -> Result<usize> {
    let prefix = format!("{user_id}:");
    let mut total = 0;
    for cf in [
        "todos",
        "projects",
        "prospective",
        "context_blocks",
        "files",
        "audit",
    ] {
        total += purge_family(store, cf, user_id, std::slice::from_ref(&prefix), false)?;
    }
    let todo_index: Vec<String> = [
        "user",
        "status:Backlog",
        "status:Todo",
        "status:InProgress",
        "status:Blocked",
        "status:Done",
        "status:Cancelled",
        "vector_id",
        "todo_vector",
    ]
    .iter()
    .map(|p| format!("{p}:{prefix}"))
    .collect();
    total += purge_family(store, "todo_index", user_id, &todo_index, true)?;
    let prospective_index: Vec<String> = [
        "user",
        "status:Pending",
        "status:Triggered",
        "status:Dismissed",
    ]
    .iter()
    .map(|p| format!("{p}:{prefix}"))
    .collect();
    total += purge_family(
        store,
        "prospective_index",
        user_id,
        &prospective_index,
        true,
    )?;
    total += purge_family(
        store,
        "file_index",
        user_id,
        &[format!("file_idx:{prefix}")],
        true,
    )?;

    if store.ensure_cf("feedback").is_ok() {
        let pending = format!("pending:{user_id}");
        if store.get_cf("feedback", pending.as_bytes())?.is_some() {
            store.delete_cf("feedback", pending.as_bytes())?;
            total += 1;
        }
    }
    Ok(total)
}

/// One family's share of the purge. Index families also keep the user id as an inner key
/// segment (priority, due date, context keyword), so `scan_segments` matches it anywhere, anchored
/// on the `:` separators so that purging "al" leaves "sal" alone. A family the store never created
/// holds nothing.
fn purge_family(
    store: &dyn crate::storage::ColumnStore,
    cf: &str,
    user_id: &str,
    prefixes: &[String],
    scan_segments: bool,
) -> Result<usize> {
    if store.ensure_cf(cf).is_err() {
        return Ok(0);
    }
    let mut keys = std::collections::BTreeSet::new();
    for prefix in prefixes {
        for (key, _) in store.scan_cf(cf, Some(prefix.as_bytes()), None)? {
            keys.insert(key);
        }
    }
    if scan_segments {
        let leading = format!("{user_id}:");
        let inner = format!(":{user_id}:");
        for (key, _) in store.scan_cf(cf, None, None)? {
            if std::str::from_utf8(&key)
                .is_ok_and(|k| k.starts_with(&leading) || k.contains(&inner))
            {
                keys.insert(key);
            }
        }
    }
    if keys.is_empty() {
        return Ok(0);
    }
    let mut batch = crate::storage::WriteBatch::new();
    for key in &keys {
        batch.delete_cf(cf, key.clone());
    }
    store.write_batch(&batch)?;
    tracing::debug!(
        "GDPR: purged {} entries from {cf} for {user_id}",
        keys.len()
    );
    Ok(keys.len())
}

#[cfg(test)]
mod shared_purge_tests {
    use super::*;
    use crate::storage::test_support::{assert_both_backends, column_stores};

    const FAMILIES: [&str; 10] = [
        "todos",
        "projects",
        "prospective",
        "context_blocks",
        "files",
        "audit",
        "todo_index",
        "prospective_index",
        "file_index",
        "feedback",
    ];

    fn rows(user: &str) -> Vec<(&'static str, String)> {
        let mut rows: Vec<(&'static str, String)> = [
            "todos",
            "projects",
            "prospective",
            "context_blocks",
            "files",
            "audit",
        ]
        .iter()
        .map(|cf| (*cf, format!("{user}:item1")))
        .collect();
        for key in [
            "user:{u}:t1",
            "status:Backlog:{u}:t2",
            "status:Todo:{u}:t3",
            "status:InProgress:{u}:t4",
            "status:Blocked:{u}:t5",
            "status:Done:{u}:t6",
            "status:Cancelled:{u}:t7",
            "vector_id:{u}:v1",
            "todo_vector:{u}:v2",
            "priority:high:{u}:t8",
        ] {
            rows.push(("todo_index", key.replace("{u}", user)));
        }
        for key in [
            "user:{u}:p1",
            "status:Pending:{u}:p2",
            "status:Triggered:{u}:p3",
            "status:Dismissed:{u}:p4",
            "context:keyword:{u}:p5",
        ] {
            rows.push(("prospective_index", key.replace("{u}", user)));
        }
        rows.push(("file_index", format!("file_idx:{user}:f1")));
        rows.push(("file_index", format!("file_meta:{user}:f3")));
        rows.push(("feedback", format!("pending:{user}")));
        rows
    }

    fn seed(store: &dyn crate::storage::ColumnStore, user: &str) -> usize {
        let rows = rows(user);
        for (cf, key) in &rows {
            store.put_cf(cf, key.as_bytes(), b"v").unwrap();
        }
        rows.len()
    }

    fn present(store: &dyn crate::storage::ColumnStore, user: &str) -> Vec<bool> {
        rows(user)
            .iter()
            .map(|(cf, key)| store.get_cf(cf, key.as_bytes()).unwrap().is_some())
            .collect()
    }

    #[test]
    fn purges_every_shared_row_for_the_user_and_no_other() {
        let stores = column_stores(&FAMILIES);
        assert_both_backends(&stores);
        for (backend, store, _dir) in stores {
            let seeded = seed(store.as_ref(), "alice");
            seed(store.as_ref(), "bob");
            let purged = purge_user_from_shared_store(store.as_ref(), "alice").unwrap();
            assert_eq!(purged, seeded, "{backend:?}");
            assert!(
                present(store.as_ref(), "alice").iter().all(|p| !p),
                "{backend:?}"
            );
            assert!(
                present(store.as_ref(), "bob").iter().all(|p| *p),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn a_user_id_inside_another_is_purged_alone() {
        let stores = column_stores(&FAMILIES);
        assert_both_backends(&stores);
        for (backend, store, _dir) in stores {
            let seeded = seed(store.as_ref(), "al");
            seed(store.as_ref(), "alice");
            seed(store.as_ref(), "sal");
            assert_eq!(
                purge_user_from_shared_store(store.as_ref(), "al").unwrap(),
                seeded,
                "{backend:?}"
            );
            assert!(
                present(store.as_ref(), "alice").iter().all(|p| *p),
                "{backend:?}"
            );
            assert!(
                present(store.as_ref(), "sal").iter().all(|p| *p),
                "{backend:?}"
            );
        }
    }
}

#[cfg(test)]
mod background_task_tests {
    use super::*;
    use std::time::Duration;

    fn manager(
        base: &std::path::Path,
        audit_rotation_check_interval: usize,
    ) -> MultiUserMemoryManager {
        let config = ServerConfig {
            storage_path: base.to_path_buf(),
            backup_enabled: false,
            audit_rotation_check_interval,
            ..ServerConfig::default()
        };
        MultiUserMemoryManager::new(base.to_path_buf(), config).expect("manager")
    }

    #[tokio::test]
    async fn drain_waits_for_tracked_tasks_and_reports_a_timeout() {
        let home = tempfile::tempdir().expect("temp home");
        let state = manager(home.path(), 100);

        let (release, released) = tokio::sync::oneshot::channel::<()>();
        state.task_tracker.spawn(async move {
            let _ = released.await;
        });

        assert!(
            !state
                .drain_background_tasks(Duration::from_millis(50))
                .await,
            "a drain must report the timeout while a tracked task is still running"
        );
        release.send(()).expect("task still waiting");
        assert!(
            state.drain_background_tasks(Duration::from_secs(30)).await,
            "the drain returns once the tracked task finishes"
        );
        assert!(state.task_tracker.is_empty());
    }

    /// The blocking pool is pinned to one thread and that thread is held, so the audit writes are
    /// still queued when the drain starts rather than usually finished. The burst crosses
    /// `audit_rotation_check_interval`, so the rotation task is queued as well as the writes.
    #[test]
    fn a_drained_manager_releases_the_shared_db_despite_queued_audit_writes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let home = tempfile::tempdir().expect("temp home");
            let state = Arc::new(manager(home.path(), 4));
            let audit_store = Arc::downgrade(&state.audit_store);
            #[cfg(feature = "rocksdb")]
            let shared_db = state.shared_db.rocksdb().map(Arc::downgrade);

            // Released by time, not by a signal: a drain that waits for the queued writes must
            // also wait for this.
            tokio::task::spawn_blocking(|| std::thread::sleep(Duration::from_millis(300)));
            let events = state.server_config.audit_rotation_check_interval + 1;
            for i in 0..events {
                state.log_event(
                    "drain-probe",
                    "PROBE",
                    &format!("probe-{i}"),
                    "queued write",
                );
            }

            assert!(state.drain_background_tasks(Duration::from_secs(30)).await);
            drop(state);

            assert!(
                audit_store.upgrade().is_none(),
                "a queued audit task still holds the audit store after drain and drop, so \
                 reopening this data directory fails on the store's lock"
            );
            #[cfg(feature = "rocksdb")]
            if let Some(shared_db) = shared_db {
                assert!(
                    shared_db.upgrade().is_none(),
                    "the shared RocksDB is still open after drain and drop"
                );
            }
        });
    }
}

impl MultiUserMemoryManager {
    /// How this server reads a vector slot of `user_id`'s that records no space: from the
    /// tenant's memory system when it is open, else the space registry beside its store, else
    /// the corpus markers. `None` when none of them says. Opens nothing.
    pub fn slot_spaces(&self, user_id: &str) -> Option<crate::embeddings::space::SlotSpaces> {
        use crate::embeddings::space::{corpus_space, SlotSpaces, SpaceRegistry};
        if let Some(memory) = self.user_memories.get(user_id) {
            return Some(memory.read().slot_spaces());
        }
        let dir = self.user_dir(user_id).ok()?;
        if let Ok(Some(registry)) = SpaceRegistry::read(&dir) {
            return Some(SlotSpaces {
                primary: registry.index_space,
                secondary: registry.secondary_index_space,
            });
        }
        corpus_space(&dir).map(|primary| SlotSpaces {
            primary,
            secondary: None,
        })
    }

    /// Embed one batch of every cached user's embed queue, journaling the records the write
    /// gate changed. Returns how many memories were taken off the queues. Users not in the
    /// cache have nothing queued: the queue lives in the user's memory system.
    pub fn drain_embed_queues(&self, batch: usize) -> usize {
        let mut drained = 0;
        for user_id in self.list_cached_users() {
            let Some(memory) = self.user_memories.get(&user_id) else {
                continue;
            };
            if memory.read().embed_queue_len() == 0 {
                continue;
            }
            let drain = memory.read().drain_embed_queue_journaled(batch);
            crate::handlers::remember::journal_absorptions(
                self,
                &memory,
                &user_id,
                &drain.absorbed,
            );
            drained += drain.taken;
        }
        drained
    }

    /// Drain every cached user's embedding backlog, reconciling first when asked. Only users
    /// already in the cache: loading a cold user is seconds of work that a tick must not do.
    pub fn drain_embedding_backlogs(
        &self,
        limit_per_user: usize,
        reconcile: bool,
    ) -> Vec<(
        String,
        anyhow::Result<crate::memory::embedding_backlog::DrainReport>,
    )> {
        let mut out = Vec::new();
        for user_id in self.list_cached_users() {
            let Some(memory) = self.user_memories.get(&user_id) else {
                continue;
            };
            let guard = memory.read();
            if reconcile {
                if let Err(e) = guard.reconcile_embedding_backlog() {
                    out.push((user_id, Err(e)));
                    continue;
                }
            }
            let result = guard.drain_embedding_backlog(limit_per_user);
            if let Ok(report) = guard.embedding_backlog_report() {
                crate::metrics::EMBEDDING_BACKLOG_PENDING
                    .with_label_values(&[user_id.as_str()])
                    .set(report.pending as i64);
            }
            out.push((user_id, result));
        }
        out
    }
}

#[cfg(test)]
mod reserved_dir_tests {
    use super::*;

    #[test]
    fn the_registry_and_cursor_directories_are_not_tenants() {
        let root = tempfile::TempDir::new().unwrap();
        for dir in ["alice", "instances", "replication", "todos"] {
            std::fs::create_dir(root.path().join(dir)).unwrap();
        }
        assert_eq!(
            list_user_dirs(root.path()).unwrap(),
            vec!["alice".to_string()]
        );
    }

    #[test]
    fn no_tenant_opens_inside_a_reserved_directory() {
        let root = tempfile::TempDir::new().unwrap();
        let config = ServerConfig {
            storage_path: root.path().to_path_buf(),
            backup_enabled: false,
            ..ServerConfig::default()
        };
        let manager = MultiUserMemoryManager::new(root.path().to_path_buf(), config).unwrap();
        assert!(manager.user_dir("instances").is_err());
        assert!(manager.user_dir("Replication").is_err());
        assert!(manager.get_user_memory("instances").is_err());
        assert!(!root.path().join("instances").exists());
        assert!(manager.user_dir("alice").is_ok());
    }
}

#[cfg(test)]
mod reconcile_restore_tests {
    use super::*;
    use crate::intent_log::payload::{live_memory_ids, CURRENT_PAYLOAD_SCHEMA_VERSION};
    use crate::intent_log::IntentPayload;
    use crate::memory::types::Experience;

    fn manager(root: &std::path::Path) -> MultiUserMemoryManager {
        let config = ServerConfig {
            storage_path: root.to_path_buf(),
            backup_enabled: false,
            ..ServerConfig::default()
        };
        MultiUserMemoryManager::new(root.to_path_buf(), config).unwrap()
    }

    /// Remember `content` and journal its snapshot, as the remember handler does.
    fn remember_journaled(manager: &MultiUserMemoryManager, content: &str) -> MemoryId {
        let user_mem = manager.get_user_memory("u").unwrap();
        let id = user_mem
            .read()
            .remember(
                Experience {
                    content: content.into(),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let mem = user_mem.read().get_memory(&id).unwrap();
        let memory_bincode =
            bincode::serde::encode_to_vec(&mem, bincode::config::standard()).unwrap();
        manager
            .journal_and_apply(
                "u",
                &IntentPayload::Remember {
                    user_id: "u".into(),
                    memory_id: id.0.to_string(),
                    memory_bincode,
                    schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
                },
            )
            .unwrap();
        id
    }

    fn live(manager: &MultiUserMemoryManager) -> std::collections::HashSet<String> {
        let writer = manager.get_user_journaled_writer("u").unwrap();
        let guard = writer.lock();
        live_memory_ids(guard.log()).unwrap()
    }

    #[test]
    fn a_memory_the_store_lost_is_restored_from_its_logged_snapshot() {
        let root = tempfile::TempDir::new().unwrap();
        let manager = manager(root.path());
        let id = remember_journaled(&manager, "the backup key lives in the second safe");
        let user_mem = manager.get_user_memory("u").unwrap();
        user_mem
            .read()
            .forget(crate::memory::ForgetCriteria::ById(id.clone()))
            .unwrap();
        assert!(
            user_mem.read().get_memory(&id).is_err(),
            "the store lost the write"
        );

        assert_eq!(
            manager
                .reconcile_store_with_intent_log("u", &user_mem)
                .unwrap(),
            1
        );

        let restored = user_mem
            .read()
            .get_memory(&id)
            .expect("restored into the store");
        assert_eq!(
            restored.experience.content,
            "the backup key lives in the second safe"
        );
        assert!(
            live(&manager).contains(&id.0.to_string()),
            "a restore journals no Forget"
        );
    }

    #[test]
    fn a_restore_applies_the_anchor_logged_after_the_snapshot() {
        let root = tempfile::TempDir::new().unwrap();
        let manager = manager(root.path());
        let id = remember_journaled(&manager, "rotate the signing key every ninety days");
        manager
            .journal_and_apply(
                "u",
                &IntentPayload::SetAnchored {
                    user_id: "u".into(),
                    memory_id: id.0.to_string(),
                    anchored: true,
                    importance: 0.9,
                    schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
                },
            )
            .unwrap();
        let user_mem = manager.get_user_memory("u").unwrap();
        user_mem
            .read()
            .forget(crate::memory::ForgetCriteria::ById(id.clone()))
            .unwrap();

        manager
            .reconcile_store_with_intent_log("u", &user_mem)
            .unwrap();

        let restored = user_mem.read().get_memory(&id).expect("restored");
        assert!(restored.is_anchored());
        assert!((restored.importance() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn a_tenant_restored_from_a_backup_forgets_what_its_store_lacks() {
        let root = tempfile::TempDir::new().unwrap();
        let manager = manager(root.path());
        let id = remember_journaled(&manager, "deleted after the journal was captured");
        let user_mem = manager.get_user_memory("u").unwrap();
        user_mem
            .read()
            .forget(crate::memory::ForgetCriteria::ById(id.clone()))
            .unwrap();
        let marker = manager
            .user_dir("u")
            .unwrap()
            .join(RESTORED_FROM_BACKUP_MARKER);
        std::fs::write(&marker, b"").unwrap();

        assert_eq!(
            manager
                .reconcile_store_with_intent_log("u", &user_mem)
                .unwrap(),
            1
        );

        assert!(
            user_mem.read().get_memory(&id).is_err(),
            "the restored store is authoritative for deletes in the backup window"
        );
        assert!(!live(&manager).contains(&id.0.to_string()));
        assert!(!marker.exists(), "the marker is spent by the reconcile");
    }

    #[test]
    fn a_memory_with_no_logged_snapshot_is_forgotten() {
        let root = tempfile::TempDir::new().unwrap();
        let manager = manager(root.path());
        let user_mem = manager.get_user_memory("u").unwrap();
        let orphan = uuid::Uuid::new_v4().to_string();
        manager
            .journal_and_apply(
                "u",
                &IntentPayload::SetAnchored {
                    user_id: "u".into(),
                    memory_id: orphan.clone(),
                    anchored: true,
                    importance: 0.8,
                    schema_version: Some(CURRENT_PAYLOAD_SCHEMA_VERSION),
                },
            )
            .unwrap();
        assert!(live(&manager).contains(&orphan));

        assert_eq!(
            manager
                .reconcile_store_with_intent_log("u", &user_mem)
                .unwrap(),
            1
        );

        assert!(
            !live(&manager).contains(&orphan),
            "nothing to restore from, so replay is made to agree with the store"
        );
    }
}
