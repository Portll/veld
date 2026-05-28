//! Multi-User Runtime Manager - Core State Management
//!
//! This module contains the central state manager for the Veld server and Roots runtime.
//! It handles per-user Earth handles, graph stores, audit logs, and all
//! subsidiary stores (todos, reminders, files, etc.).

use anyhow::{Context, Result};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use tracing::info;

use crate::earth::{Earth, SharedEarth};
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
#[cfg(feature = "multi-tenant")]
use crate::extensions::maintenance;
use crate::backup;
use crate::backup::BackupMetadata;
use crate::config::{ServerConfig, StorageBackend};
use crate::embeddings::{
    are_ner_models_downloaded, auto_download_models_enabled, download_ner_models,
    get_ner_models_dir, ner::NerEntityType, neural_ner_enabled, KeywordExtractor, NerConfig,
    NeuralNer,
};
use crate::graph_memory::{
    EdgeTier, EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, GraphStats,
    LtpStatus, RelationType, RelationshipEdge,
};
use crate::memory::{
    query_parser, Experience, FeedbackStore, FileMemoryStore, MemoryConfig, MemoryId, MemoryStats,
    ProspectiveStore, SessionStore, TodoStore,
};
use crate::relevance::RelevanceEngine;
use crate::storage::legacy_rocksdb::{RocksDbGraphStore, RocksDbPrimaryMemoryStore};
use crate::storage::{AuditLogEntry, AuditStore, StorageCapabilities};
use crate::streaming;

use super::types::{AuditEvent, ContextStatus, MemoryEvent};

/// Type alias for context sessions map
pub type ContextSessions = DashMap<String, ContextStatus>;

struct SharedStoreBootstrap {
    shared_db: Arc<rocksdb::DB>,
    audit_store: Arc<dyn AuditStore>,
    prospective_store: Arc<ProspectiveStore>,
    todo_store: Arc<TodoStore>,
    context_block_store: Arc<crate::memory::ContextBlockStore>,
    file_store: Arc<FileMemoryStore>,
    feedback_store: Arc<parking_lot::RwLock<FeedbackStore>>,
}

/// Helper struct for audit log rotation (allows spawn_blocking with minimal clone)
struct MultiUserMemoryManagerRotationHelper {
    audit_store: Arc<dyn AuditStore>,
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,
    audit_retention_days: i64,
    audit_max_entries: usize,
}

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
        let removed_count = self.audit_store.rotate_events(
            user_id,
            self.audit_max_entries,
            cutoff_time,
        )?;

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

/// Multi-user Veld manager for the Roots runtime.
pub struct MultiUserMemoryManager {
    /// Per-user Earth handles with LRU eviction.
    pub user_earths: moka::sync::Cache<String, SharedEarth>,

    /// Per-user audit logs (in-memory cache)
    pub audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,

    /// Shared DB for all global stores (todos, reminders, files, feedback, audit)
    pub shared_db: Arc<rocksdb::DB>,

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
    pub backup_engine: Arc<backup::VeldBackupEngine>,

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

    /// Per-user creation locks to prevent TOCTOU races in get_user_earth.
    /// Without this, concurrent first-access requests for the same user_id can both
    /// miss the cache check, both try to open RocksDB, and the second open fails
    /// because RocksDB holds an exclusive file lock.
    user_earth_init_locks: DashMap<String, Arc<parking_lot::Mutex<()>>>,

    /// Separate per-user creation locks for graph memory.
    /// Must be separate from user_earth_init_locks because get_user_earth()
    /// calls get_user_graph() while holding its lock, and parking_lot::Mutex
    /// is not re-entrant — sharing a single lock map would deadlock.
    user_graph_init_locks: DashMap<String, Arc<parking_lot::Mutex<()>>>,

    /// Shared RocksDB block cache across all per-user DB instances.
    /// Single LRU cache provides a hard memory ceiling regardless of user count.
    /// Without this, each user's MemoryStorage + GraphMemory allocates ~96MB in
    /// independent caches — 6 users = 576MB just in block caches alone.
    shared_rocksdb_cache: rocksdb::Cache,

    /// Per-user SlowStore cache. Avoids reopening SQLite on every gap analysis call,
    /// and ensures the sync TTL actually works across requests.
    pub slow_stores: DashMap<String, std::sync::Arc<crate::memory::slow_store::SlowStore>>,

    /// Per-user [`JournaledWriter`]. Created lazily on first journaled
    /// CRUD for that user; holds the open intent log, the per-user
    /// `CheckpointStore`, and the live `SqliteProjection` attached for
    /// dispatch. Wrapped in `Mutex` because `record_and_apply` mutates
    /// the underlying log + projections — only one writer at a time per
    /// tenant.
    ///
    /// Wired up here (and not in the per-user Earth) because the
    /// projection list is a runtime construct that may grow over time
    /// (Vamana, BM25, Postgres) and we want one place to register them.
    pub journaled_writers: DashMap<
        String,
        std::sync::Arc<parking_lot::Mutex<crate::intent_log::JournaledWriter>>,
    >,

    /// One init-lock per tenant for the lazy `JournaledWriter` creation.
    /// Without this, two concurrent first-touch requests for the same
    /// user could both miss the cache, both call `IntentLog::open`, and
    /// the second open would race with the first writer's append.
    journaled_writer_init_locks: DashMap<String, Arc<parking_lot::Mutex<()>>>,

    /// Capabilities for the effective backend in the current runtime.
    storage_capabilities: StorageCapabilities,

    /// Phase C user-auth runtime. `Some` iff `VELD_USER_AUTH_ENABLED` was
    /// truthy at server startup. Handlers gate on this — when `None`, the
    /// /api/user_auth/* surface returns 404.
    pub user_auth_runtime: Option<crate::user_auth::runtime::UserAuthRuntime>,
}

impl MultiUserMemoryManager {
    pub fn new(base_path: std::path::PathBuf, server_config: ServerConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;

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

        let user_earths = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .time_to_idle(std::time::Duration::from_secs(3600))
            .eviction_listener(move |key: Arc<String>, value: SharedEarth, cause| {
                if matches!(cause, moka::notification::RemovalCause::Size | moka::notification::RemovalCause::Expired) {
                    evictions_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    let cause_label = if cause == moka::notification::RemovalCause::Expired { "idle-timeout" } else { "LRU" };

                    // Spawn blocking task to persist vector index without holding the lock
                    // during I/O. The eviction listener runs synchronously inside moka,
                    // so we must not block here for disk writes.
                    //
                    // CRITICAL: We must drop the Arc<RwLock<Earth>> as soon as
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
        let shared_rocksdb_cache =
            rocksdb::Cache::new_lru_cache(crate::constants::ROCKSDB_SHARED_CACHE_BYTES);
        info!(
            "Shared RocksDB block cache initialized ({}MB)",
            crate::constants::ROCKSDB_SHARED_CACHE_BYTES / (1024 * 1024)
        );

        let shared_stores = Self::bootstrap_shared_stores(
            &base_path,
            &server_config,
            &shared_rocksdb_cache,
        )?;

        if let Err(e) = shared_stores.todo_store.load_vector_indices() {
            tracing::warn!("Failed to load todo vector indices: {}, semantic todo search will rebuild on first use", e);
        }
        info!("Todo store initialized");

        let backup_engine = Self::open_backup_engine(&base_path, &server_config)?;
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
        let streaming_extractor =
            Arc::new(streaming::StreamingMemoryExtractor::new(neural_ner.clone()));
        info!("Streaming memory extractor initialized");

        let keyword_extractor = Arc::new(KeywordExtractor::new());
        info!("Keyword extractor initialized (YAKE)");

        let relevance_engine = Arc::new(RelevanceEngine::new(neural_ner.clone()));
        info!("Relevance engine initialized (entity cache + learned weights)");

        let broadcast_capacity = (server_config.max_users_in_memory * 4).max(64);
        let storage_capabilities =
            StorageCapabilities::for_backend(server_config.effective_storage_backend);

        // Phase C — initialise the user-auth runtime only when the feature
        // flag was truthy at startup. The shared DB already declared the
        // user_auth CF in `open_legacy_shared_db` under the same flag, so
        // construction here just wraps the existing handle.
        let user_auth_runtime = if crate::user_auth::feature_enabled() {
            match crate::user_auth::store::UserAuthStore::new(shared_stores.shared_db.clone()) {
                Ok(store) => {
                    let field_encryptor = match crate::encryption::FieldEncryptor::from_env() {
                        Ok(encryptor) => encryptor,
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "VELD_ENCRYPTION_KEY present but malformed; user-auth runtime starts without an encryptor (2FA enrollment will be refused in production mode)"
                            );
                            None
                        }
                    };
                    info!(
                        encryption = field_encryptor.is_some(),
                        "user-auth runtime initialised (VELD_USER_AUTH_ENABLED=true)"
                    );
                    Some(crate::user_auth::runtime::UserAuthRuntime::new(
                        store,
                        field_encryptor,
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
            user_earths,
            audit_logs: Arc::new(DashMap::new()),
            shared_db: shared_stores.shared_db,
            audit_store: shared_stores.audit_store,
            base_path,
            default_config: MemoryConfig {
                collective_store_dir: if server_config.multi_tenant_mode {
                    Some(server_config.collective_store_dir.clone())
                } else {
                    None
                },
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
            context_sessions: Arc::new(DashMap::new()),
            context_broadcaster: {
                let (tx, _) = tokio::sync::broadcast::channel(broadcast_capacity);
                tx
            },
            ab_test_manager: Arc::new(ab_testing::ABTestManager::new()),
            session_store: Arc::new(SessionStore::new()),
            relevance_engine,
            maintenance_cycle: std::sync::atomic::AtomicU64::new(0),
            user_earth_init_locks: DashMap::new(),
            user_graph_init_locks: DashMap::new(),
            shared_rocksdb_cache,
            slow_stores: DashMap::new(),
            journaled_writers: DashMap::new(),
            journaled_writer_init_locks: DashMap::new(),
            storage_capabilities,
            user_auth_runtime,
        };

        info!("Running initial audit log rotation...");
        if let Err(e) = manager.rotate_all_audit_logs() {
            tracing::warn!("Failed to rotate audit logs on startup: {}", e);
        }

        Ok(manager)
    }

    fn bootstrap_shared_stores(
        base_path: &std::path::Path,
        server_config: &ServerConfig,
        shared_rocksdb_cache: &rocksdb::Cache,
    ) -> Result<SharedStoreBootstrap> {
        match server_config.effective_storage_backend {
            StorageBackend::RocksDb => {
                let shared_db = Self::open_legacy_shared_db(base_path, shared_rocksdb_cache)?;

                Self::migrate_audit_db(base_path, &shared_db)?;

                let audit_store: Arc<dyn AuditStore> = Arc::new(
                    crate::storage::legacy_rocksdb::RocksDbAuditStore::new(
                        shared_db.clone(),
                        CF_AUDIT,
                    ),
                );

                let prospective_store =
                    Arc::new(ProspectiveStore::new(shared_db.clone(), base_path)?);
                info!("Prospective memory store initialized");

                let todo_store = Arc::new(TodoStore::new(shared_db.clone(), base_path)?);

                let file_store = Arc::new(FileMemoryStore::new(shared_db.clone(), base_path)?);
                info!("File memory store initialized");

                let context_block_store =
                    Arc::new(crate::memory::ContextBlockStore::new(shared_db.clone()));
                info!("Context block store initialized");

                let feedback_store = Arc::new(parking_lot::RwLock::new(
                    FeedbackStore::with_shared_db(shared_db.clone(), base_path).unwrap_or_else(
                        |e| {
                            tracing::warn!(
                                "Failed to load feedback store: {}, using in-memory",
                                e
                            );
                            FeedbackStore::new()
                        },
                    ),
                ));
                info!("Feedback store initialized");

                Ok(SharedStoreBootstrap {
                    shared_db,
                    audit_store,
                    prospective_store,
                    todo_store,
                    context_block_store,
                    file_store,
                    feedback_store,
                })
            }
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for shared-store bootstrap yet",
                server_config.effective_storage_backend
            )),
        }
    }

    fn open_legacy_shared_db(
        base_path: &std::path::Path,
        shared_rocksdb_cache: &rocksdb::Cache,
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

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(shared_rocksdb_cache);
        block_opts.set_cache_index_and_filter_blocks(true);
        db_opts.set_block_based_table_factory(&block_opts);

        let mut cfs = vec![ColumnFamilyDescriptor::new("default", {
            let mut o = RocksOptions::default();
            o.create_if_missing(true);
            o
        })];
        cfs.extend(TodoStore::cf_descriptors());
        cfs.extend(ProspectiveStore::column_family_descriptors());
        cfs.extend(FileMemoryStore::cf_descriptors());
        cfs.extend(crate::memory::ContextBlockStore::cf_descriptors());
        cfs.push(ColumnFamilyDescriptor::new(
            crate::memory::feedback::CF_FEEDBACK,
            Self::shared_store_cf_options(),
        ));
        cfs.push(ColumnFamilyDescriptor::new(
            CF_AUDIT,
            Self::shared_store_cf_options(),
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

    fn shared_store_cf_options() -> rocksdb::Options {
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        options.set_compression_type(rocksdb::DBCompressionType::Lz4);
        options
    }

    fn open_backup_engine(
        base_path: &std::path::Path,
        server_config: &ServerConfig,
    ) -> Result<Arc<backup::VeldBackupEngine>> {
        let backup_path = base_path.join("backups");

        match server_config.effective_storage_backend {
            StorageBackend::RocksDb => Ok(Arc::new(backup::VeldBackupEngine::new(backup_path)?)),
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for backup engine bootstrap yet",
                server_config.effective_storage_backend
            )),
        }
    }

    /// Get the audit column family handle from the shared DB
    fn audit_cf(&self) -> &rocksdb::ColumnFamily {
        self.shared_db
            .cf_handle(CF_AUDIT)
            .expect("audit CF must exist in shared DB")
    }

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
                    .write(std::mem::take(&mut batch))
                    .map_err(|e| anyhow::anyhow!("audit migration batch write error: {e}"))?;
                batch = rocksdb::WriteBatch::default();
            }
        }

        if !count.is_multiple_of(BATCH_SIZE) {
            shared_db
                .write(batch)
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

        tokio::task::spawn_blocking(move || {
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

            tokio::task::spawn_blocking(move || {
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

    fn open_user_earth(&self, config: MemoryConfig) -> Result<Earth> {
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => {
                let primary_store = RocksDbPrimaryMemoryStore::open(
                    &config.storage_path,
                    Some(&self.shared_rocksdb_cache),
                )
                .with_context(|| {
                    format!("Failed to open primary memory store at {:?}", config.storage_path)
                })?;

                Earth::with_storage(config, Arc::new(primary_store.into_inner()))
            }
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for MemorySystem construction yet",
                self.server_config.effective_storage_backend
            )),
        }
    }

    fn open_user_graph_memory(&self, user_id: &str) -> Result<GraphMemory> {
        let graph_path = self.base_path.join(user_id).join("graph");

        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => {
                let graph_store = RocksDbGraphStore::open(
                    &graph_path,
                    Some(&self.shared_rocksdb_cache),
                )
                .with_context(|| {
                    format!("Failed to open graph store for user '{user_id}' at {:?}", graph_path)
                })?;

                Ok(graph_store.into_inner())
            }
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for GraphMemory construction yet",
                self.server_config.effective_storage_backend
            )),
        }
    }

    /// Get or create an Earth substrate for a user.
    ///
    /// Uses double-checked locking to prevent TOCTOU races where concurrent
    /// first-access requests both miss the cache and try to open RocksDB.
    /// RocksDB holds exclusive file locks, so the second open would fail.
    pub fn get_user_earth(&self, user_id: &str) -> Result<SharedEarth> {
        // Fast path: already cached
        if let Some(memory) = self.user_earths.get(user_id) {
            return Ok(memory);
        }

        // Acquire per-user creation lock to serialize initialization
        let lock = self
            .user_earth_init_locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        // Re-check after acquiring lock (another thread may have created it)
        if let Some(memory) = self.user_earths.get(user_id) {
            return Ok(memory);
        }

        let user_path = self.base_path.join(user_id);
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
        let mut earth = {
            let mut last_err = None;
            let mut created = None;
            for attempt in 0..4u32 {
                match self.open_user_earth(config.clone()) {
                    Ok(earth) => {
                        if attempt > 0 {
                            info!(
                                "Earth for user '{}' created after {} retries (lock contention resolved)",
                                user_id, attempt
                            );
                        }
                        created = Some(earth);
                        break;
                    }
                    Err(e) => {
                        let err_str = e.to_string();
                        if err_str.contains("lock") || err_str.contains("LOCK") {
                            let delay = std::time::Duration::from_millis(50 * 2u64.pow(attempt));
                            tracing::warn!(
                                "RocksDB lock contention for user '{}' (attempt {}/4), retrying in {:?}",
                                user_id, attempt + 1, delay
                            );
                            std::thread::sleep(delay);
                            last_err = Some(e);
                        } else {
                            // Non-lock error, fail immediately
                            return Err(e).with_context(|| {
                                format!("Failed to initialize Earth for user '{user_id}'")
                            });
                        }
                    }
                }
            }
            match created {
                Some(earth) => earth,
                None => {
                    return Err(last_err.unwrap_or_else(|| anyhow::anyhow!("all retry attempts failed"))).with_context(|| {
                        format!(
                            "Failed to initialize Earth for user '{}' after 4 attempts (RocksDB lock held by eviction thread)",
                            user_id
                        )
                    });
                }
            }
        };
        // Wire up GraphMemory for Layer 2 (spreading activation) and Layer 5 (Hebbian learning)
        let graph = self.get_user_graph(user_id)?;
        earth.set_graph_memory(graph);
        // Wire up FeedbackStore for PIPE-9 (feedback momentum in all retrieval paths)
        earth.set_feedback_store(self.feedback_store.clone());

        let memory_arc: SharedEarth = Arc::new(parking_lot::RwLock::new(earth));

        self.user_earths
            .insert(user_id.to_string(), memory_arc.clone());

        info!("Created memory system for user: {}", user_id);

        Ok(memory_arc)
    }

    /// Evict a user's memory and graph from in-memory caches (releases DB handles).
    /// Does NOT delete data — used before restore to release file locks.
    pub fn evict_user(&self, user_id: &str) {
        self.user_earths.invalidate(user_id);
        self.graph_memories.invalidate(user_id);
        self.user_earths.run_pending_tasks();
        self.graph_memories.run_pending_tasks();

        #[cfg(target_os = "windows")]
        {
            // Windows needs extra time to release file handles
            std::thread::sleep(std::time::Duration::from_millis(200));
            self.user_earths.run_pending_tasks();
            self.graph_memories.run_pending_tasks();
        }

        tracing::info!(user_id = user_id, "Evicted user caches for restore");
    }

    /// Delete user data (GDPR compliance)
    ///
    /// Cleans up:
    /// 1. In-memory caches (user_earths, graph_memories)
    /// 2. Shared RocksDB: todos, projects, todo indices, reminders, files, feedback, audit
    /// 3. Per-user filesystem: per-user RocksDB, graph DB, vector indices
    pub fn forget_user(&self, user_id: &str) -> Result<()> {
        self.user_earths.invalidate(user_id);
        self.graph_memories.invalidate(user_id);

        self.user_earths.run_pending_tasks();
        self.graph_memories.run_pending_tasks();

        #[cfg(target_os = "windows")]
        {
            std::thread::sleep(std::time::Duration::from_millis(200));
            self.user_earths.run_pending_tasks();
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
        let user_path = self.base_path.join(user_id);
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

    /// Prefix-scan and batch-delete all keys starting with `{user_id}:` from a column family
    fn delete_by_prefix(db: &rocksdb::DB, cf: &rocksdb::ColumnFamily, prefix: &[u8]) -> usize {
        let mut batch = rocksdb::WriteBatch::default();
        let mut count = 0;
        let iter = db.prefix_iterator_cf(cf, prefix);
        for item in iter.flatten() {
            let (key, _) = item;
            if !key.starts_with(prefix) {
                break;
            }
            batch.delete_cf(cf, &key);
            count += 1;
        }
        if count > 0 {
            let _ = db.write(batch);
        }
        count
    }

    /// Purge all user data from shared RocksDB (todos, reminders, files, feedback, audit)
    fn purge_user_from_shared_db(&self, user_id: &str) -> Result<()> {
        let prefix = format!("{user_id}:");
        let prefix_bytes = prefix.as_bytes();

        // Shared CF names that use `{user_id}:` as key prefix
        let cf_names = ["todos", "projects", "prospective", "context_blocks"];
        for name in &cf_names {
            if let Some(cf) = self.shared_db.cf_handle(name) {
                let n = Self::delete_by_prefix(&self.shared_db, cf, prefix_bytes);
                if n > 0 {
                    tracing::debug!("GDPR: purged {n} entries from {name} CF for {user_id}");
                }
            }
        }

        // Index CFs use varied key prefixes — scan all relevant patterns
        if let Some(cf) = self.shared_db.cf_handle("todo_index") {
            let prefixes = [
                format!("user:{user_id}:"),
                format!("status:Backlog:{user_id}:"),
                format!("status:Todo:{user_id}:"),
                format!("status:InProgress:{user_id}:"),
                format!("status:Blocked:{user_id}:"),
                format!("status:Done:{user_id}:"),
                format!("status:Cancelled:{user_id}:"),
                format!("vector_id:{user_id}:"),
                format!("todo_vector:{user_id}:"),
            ];
            for p in &prefixes {
                Self::delete_by_prefix(&self.shared_db, cf, p.as_bytes());
            }
            // Priority and due/context keys also contain user_id but at varying positions.
            // Full scan of index CF to catch them all.
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.shared_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter.flatten() {
                let (key, _) = item;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.contains(&prefix) {
                        batch.delete_cf(cf, &key);
                    }
                }
            }
            let _ = self.shared_db.write(batch);
        }

        if let Some(cf) = self.shared_db.cf_handle("prospective_index") {
            let prefixes = [
                format!("user:{user_id}:"),
                format!("status:Pending:{user_id}:"),
                format!("status:Triggered:{user_id}:"),
                format!("status:Dismissed:{user_id}:"),
            ];
            for p in &prefixes {
                Self::delete_by_prefix(&self.shared_db, cf, p.as_bytes());
            }
            // Context keyword indices: `context:{keyword}:{user_id}:{id}`
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.shared_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter.flatten() {
                let (key, _) = item;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.contains(&prefix) {
                        batch.delete_cf(cf, &key);
                    }
                }
            }
            let _ = self.shared_db.write(batch);
        }

        // Files
        if let Some(cf) = self.shared_db.cf_handle("files") {
            Self::delete_by_prefix(&self.shared_db, cf, prefix_bytes);
        }
        if let Some(cf) = self.shared_db.cf_handle("file_index") {
            let idx_prefix = format!("file_idx:{user_id}:");
            Self::delete_by_prefix(&self.shared_db, cf, idx_prefix.as_bytes());
            // Also catch other patterns
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.shared_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter.flatten() {
                let (key, _) = item;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.contains(&prefix) {
                        batch.delete_cf(cf, &key);
                    }
                }
            }
            let _ = self.shared_db.write(batch);
        }

        // Feedback: `pending:{user_id}`
        if let Some(cf) = self.shared_db.cf_handle("feedback") {
            let pending_key = format!("pending:{user_id}");
            let _ = self.shared_db.delete_cf(cf, pending_key.as_bytes());
        }

        // Audit logs
        if let Some(cf) = self.shared_db.cf_handle("audit") {
            Self::delete_by_prefix(&self.shared_db, cf, prefix_bytes);
        }

        // Clear in-memory audit log cache
        self.audit_logs.remove(user_id);

        Ok(())
    }

    /// Get statistics for a user
    pub fn get_stats(&self, user_id: &str) -> Result<MemoryStats> {
        let memory = self.get_user_earth(user_id)?;
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
        let mut users = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            // Filter out system directories
                            if name != "audit_logs"
                                && name != "audit_logs.pre_cf_migration"
                                && name != "backups"
                                && name != "feedback"
                                && name != "feedback.pre_cf_migration"
                                && name != "semantic_facts"
                                && name != "files"
                                && name != "files.pre_cf_migration"
                                && name != "prospective"
                                && name != "prospective.pre_cf_migration"
                                && name != "todos"
                                && name != "todos.pre_cf_migration"
                                && name != "shared"
                            {
                                users.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }
        users.sort();
        users
    }

    /// List users currently loaded in the Moka cache (no filesystem scan)
    pub fn list_cached_users(&self) -> Vec<String> {
        self.user_earths
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
    pub fn flush_all_databases(&self) -> Result<()> {
        info!("Flushing all databases to disk...");

        // Single flush covers all shared stores (todos, prospective, files, feedback, audit)
        self.shared_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush shared database: {e}"))?;
        info!("  Shared database flushed (todos, prospective, files, feedback, audit)");

        let user_entries: Vec<(String, SharedEarth)> = self
            .user_earths
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

    /// Save all vector indices to disk
    pub fn save_all_vector_indices(&self) -> Result<()> {
        info!("Saving vector indices to disk...");

        let user_entries: Vec<(String, SharedEarth)> = self
            .user_earths
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let mut saved = 0;
        for (user_id, memory_system) in user_entries {
            if let Some(guard) = memory_system.try_read() {
                let index_path = self.base_path.join(&user_id).join("vector_index");
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

        let mut user_ids = std::collections::HashSet::new();
        let audit = self.audit_cf();
        let iter = self
            .shared_db
            .iterator_cf(audit, rocksdb::IteratorMode::Start);

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(user_id) = key_str.split(':').next() {
                    user_ids.insert(user_id.to_string());
                }
            }
        }

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
    pub fn get_keyword_extractor(&self) -> Arc<KeywordExtractor> {
        self.keyword_extractor.clone()
    }

    /// Get or create graph memory for a user
    ///
    /// Uses the same per-user creation lock as get_user_earth to prevent
    /// concurrent RocksDB open races on the graph directory.
    pub fn get_user_graph(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<GraphMemory>>> {
        // Fast path: already cached
        if let Some(graph) = self.graph_memories.get(user_id) {
            return Ok(graph);
        }

        // Acquire per-user graph creation lock (separate from Earth lock
        // to avoid deadlock when get_user_earth() calls get_user_graph())
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

        // Retry with backoff for RocksDB lock contention (same pattern as get_user_earth).
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
                        let err_str = e.to_string();
                        if err_str.contains("lock") || err_str.contains("LOCK") {
                            let delay = std::time::Duration::from_millis(50 * 2u64.pow(attempt));
                            tracing::warn!(
                                "RocksDB lock contention on graph for user '{}' (attempt {}/4), retrying in {:?}",
                                user_id, attempt + 1, delay
                            );
                            std::thread::sleep(delay);
                            last_err = Some(e);
                        } else {
                            return Err(e).with_context(|| {
                                format!("Failed to initialize graph memory for user '{user_id}'")
                            });
                        }
                    }
                }
            }
            match created {
                Some(gm) => gm,
                None => {
                    return Err(last_err.unwrap_or_else(|| anyhow::anyhow!("all retry attempts failed"))).with_context(|| {
                        format!(
                            "Failed to initialize graph memory for user '{}' after 4 attempts (RocksDB lock contention)",
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

        let user_path = self.base_path.join(user_id);
        std::fs::create_dir_all(&user_path)?;
        let db_path = user_path.join("slow_store.db");
        let store = std::sync::Arc::new(crate::memory::slow_store::SlowStore::open(&db_path)?);
        self.slow_stores.insert(user_id.to_string(), store.clone());
        Ok(store)
    }

    /// Get or create a [`JournaledWriter`] for `user_id`.
    ///
    /// On first call for a tenant, this:
    ///   1. opens (or creates) the per-user intent log at
    ///      `{base_path}/{user_id}/intent.log`,
    ///   2. opens (or creates) the per-user checkpoint store at
    ///      `{base_path}/{user_id}/projection_checkpoints.bin`,
    ///   3. instantiates a [`SqliteProjection`] wrapped around the user's
    ///      `SlowStore`,
    ///   4. **runs replay** so the SQLite slow store catches up to the
    ///      head of the log before any live write lands, and
    ///   5. attaches the projection to the writer so subsequent live
    ///      `record_and_apply` calls dispatch through it.
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
    ) -> Result<
        std::sync::Arc<parking_lot::Mutex<crate::intent_log::JournaledWriter>>,
    > {
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

        let user_path = self.base_path.join(user_id);
        std::fs::create_dir_all(&user_path)?;
        let log_path = user_path.join("intent.log");
        let checkpoint_path = user_path.join("projection_checkpoints.bin");

        let log = crate::intent_log::IntentLog::open(&log_path)
            .with_context(|| format!("open intent log for user {user_id}"))?;
        let checkpoint_store = std::sync::Arc::new(parking_lot::Mutex::new(
            crate::intent_log::CheckpointStore::open(&checkpoint_path)
                .with_context(|| format!("open checkpoint store for user {user_id}"))?,
        ));

        let slow_store = self.get_user_slow_store(user_id)?;
        let mut sqlite_projection = crate::memory::slow_store::SqliteProjection::new(
            slow_store.clone(),
            checkpoint_store.clone(),
        );

        // Catch the SQLite projection up to the head of the log before
        // it goes live. The `Some(100)` flushes the checkpoint every 100
        // applied records so a crash during a long replay survives.
        let applied_sqlite =
            crate::intent_log::replay(&log, &mut sqlite_projection, Some(100))
                .with_context(|| {
                    format!("replay intent log for user {user_id} (sqlite projection)")
                })?;
        if applied_sqlite > 0 {
            tracing::info!(
                user_id = %user_id,
                applied = applied_sqlite,
                "replayed intent log records into SQLite projection on first writer open",
            );
        }

        // Second projection: Vamana. Each user gets a dedicated vector
        // index that is *derived from the log*. The index lives under
        // `{base}/{user_id}/vamana_projection/` so a wipe-and-replay
        // operator workflow can blow away the directory without
        // touching the SQLite slow store or RocksDB.
        //
        // We use the *lazy* variant: the index is materialised on the
        // first embedded `Memory` we see during replay or live writes.
        // This avoids coupling the projection's open to the embedder
        // cache (the embedder is constructed lazily per-tenant via
        // `get_user_earth`, which can be a different lifecycle than
        // `get_user_journaled_writer`). The first embedded memory tells
        // us the dim — that's the same dim every subsequent memory in
        // the log must match (the embedder is stable for a tenant's
        // corpus), so this is correct by construction.
        let vamana_index_dir = user_path.join("vamana_projection");
        std::fs::create_dir_all(&vamana_index_dir).with_context(|| {
            format!(
                "create vamana projection dir for user {user_id}: {:?}",
                vamana_index_dir
            )
        })?;
        let bootstrap = crate::vector_db::vamana_projection::VamanaProjectionBootstrap {
            storage_path: Some(vamana_index_dir),
            config_template: crate::vector_db::VamanaConfig {
                // dimension is rewritten on first embedded apply.
                dimension: 0,
                max_degree: 32,
                search_list_size: 100,
                alpha: 1.2,
                use_mmap: true,
                distance_metric: crate::vector_db::DistanceMetric::default(),
            },
        };
        let mut vamana_projection = crate::vector_db::VamanaProjection::lazy(
            bootstrap,
            checkpoint_store.clone(),
        );

        let applied_vamana =
            crate::intent_log::replay(&log, &mut vamana_projection, Some(100))
                .with_context(|| {
                    format!("replay intent log for user {user_id} (vamana projection)")
                })?;
        if applied_vamana > 0 {
            tracing::info!(
                user_id = %user_id,
                applied = applied_vamana,
                "replayed intent log records into Vamana projection on first writer open",
            );
        }

        // Hand the log handle into the writer so subsequent appends
        // continue against the same file. Replay above used a read-only
        // borrow; the same handle is fine for the writer's appends.
        let mut writer = crate::intent_log::JournaledWriter::new(log);
        writer.add_projection(Box::new(sqlite_projection));
        writer.add_projection(Box::new(vamana_projection));

        let writer = std::sync::Arc::new(parking_lot::Mutex::new(writer));
        self.journaled_writers
            .insert(user_id.to_string(), writer.clone());
        Ok(writer)
    }

    /// Convenience: journal a typed payload through the user's
    /// `JournaledWriter`. Calls `record_and_apply` under the writer's
    /// mutex and returns the assigned LSN. Per-projection apply errors
    /// are *not* propagated as failures — they are logged at warn level
    /// (the log frame is durable, replay will retry on restart).
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
            .user_earths
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
            let maintenance_result = if let Ok(memory_lock) = self.get_user_earth(&user_id) {
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
                        match graph_guard.strengthen_memory_edges(&result.edge_boosts) {
                            Ok((count, promotion_boosts)) => {
                                edges_strengthened += count;

                                // Direction 1: Apply edge promotion boosts to memory importance
                                if !promotion_boosts.is_empty() {
                                    if let Ok(memory_lock) = self.get_user_earth(&user_id) {
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
                                match graph_guard.strengthen_episode_entity_edges(&uuid) {
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
                            if let Ok(memory_lock) = self.get_user_earth(&user_id) {
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
                                if let Ok(memory_lock) = self.get_user_earth(&user_id) {
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

        // Heavy cycle: clean up old triggered/dismissed reminders (C4 fix)
        if is_heavy {
            for (user_id_arc, _) in self.user_earths.iter() {
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
                .user_earths
                .iter()
                .map(|(id, _)| id.to_string())
                .collect();
            self.user_earth_init_locks
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
    pub fn backup_engine(&self) -> &Arc<backup::VeldBackupEngine> {
        &self.backup_engine
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
        self.user_earths.entry_count() as usize
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
        const REQUIRED_CFS: &[&str] = &[
            "audit",
            "prospective",
            "todos",
            "projects",
            "files",
            "feedback",
        ];

        REQUIRED_CFS
            .iter()
            .all(|cf| self.shared_db.cf_handle(cf).is_some())
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

    fn collect_legacy_secondary_store_refs(&self) -> Vec<(String, std::sync::Arc<rocksdb::DB>)> {
        vec![("shared".to_string(), std::sync::Arc::clone(&self.shared_db))]
    }

    /// Default location of the W5 intent log directory for this server
    /// instance: `<base_path>/intent_log/`. The path is conventional —
    /// callers that need a different layout can call the lower-level
    /// `VeldBackupEngine` methods directly.
    fn intent_log_default_dir(&self) -> std::path::PathBuf {
        self.base_path.join("intent_log")
    }

    pub fn create_user_backup(&self, user_id: &str) -> Result<BackupMetadata> {
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => {
                let memory_sys = self.get_user_earth(user_id)?;
                let memory_guard = memory_sys.read();
                let db = memory_guard.get_db();

                let secondary_refs = self.collect_legacy_secondary_store_refs();
                let store_refs: Vec<crate::backup::SecondaryStoreRef<'_>> = secondary_refs
                    .iter()
                    .map(|(name, db)| crate::backup::SecondaryStoreRef { name, db })
                    .collect();

                let graph_lock = self.get_user_graph(user_id).ok();
                let graph_guard = graph_lock.as_ref().map(|graph| graph.read());
                let graph_db_ref = graph_guard.as_ref().map(|graph| graph.get_db());

                // Always attempt to include the intent log; the backup
                // engine no-ops cleanly when the dir doesn't exist (W5
                // not yet active for this install).
                let intent_log_dir = self.intent_log_default_dir();
                let intent_log_spec =
                    crate::backup::IntentLogBackupSpec::with_default_log_name(intent_log_dir);

                self.backup_engine.create_comprehensive_backup_with_intent_log(
                    &db,
                    user_id,
                    &store_refs,
                    graph_db_ref,
                    Some(&intent_log_spec),
                )
            }
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for backup creation yet",
                self.server_config.effective_storage_backend
            )),
        }
    }

    pub fn list_backups_for_user(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => self.backup_engine.list_backups(user_id),
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for backup listing yet",
                self.server_config.effective_storage_backend
            )),
        }
    }

    pub fn verify_backup_for_user(&self, user_id: &str, backup_id: u32) -> Result<bool> {
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => self.backup_engine.verify_backup(user_id, backup_id),
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for backup verification yet",
                self.server_config.effective_storage_backend
            )),
        }
    }

    pub fn purge_backups_for_user(&self, user_id: &str, keep_count: usize) -> Result<usize> {
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => self.backup_engine.purge_old_backups(user_id, keep_count),
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for backup purging yet",
                self.server_config.effective_storage_backend
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
        match self.server_config.effective_storage_backend {
            StorageBackend::RocksDb => {
                let memory_db_path = self.base_path.join(user_id).join("storage");
                let graph_path = self.base_path.join(user_id).join("graph").join("graph");
                let intent_log_dir = self.intent_log_default_dir();

                self.evict_user(user_id);

                let secondary_restore_paths: Vec<(&str, &std::path::Path)> = vec![];
                let intent_log_spec =
                    crate::backup::IntentLogBackupSpec::with_default_log_name(intent_log_dir);
                let restore_opts = crate::backup::RestoreOptions { max_lsn };
                let restored_stores = self
                    .backup_engine
                    .restore_comprehensive_backup_with_intent_log(
                        user_id,
                        backup_id,
                        &memory_db_path,
                        &secondary_restore_paths,
                        Some(&intent_log_spec),
                        &restore_opts,
                    )?;

                let resolved_backup_id = backup_id.unwrap_or_else(|| {
                    self.backup_engine
                        .list_backups(user_id)
                        .ok()
                        .and_then(|backups| backups.last().map(|metadata| metadata.backup_id))
                        .unwrap_or(0)
                });
                let graph_checkpoint = self
                    .backup_engine
                    .backup_path()
                    .join(user_id)
                    .join(format!("secondary_{resolved_backup_id}"))
                    .join("graph");

                let mut all_restored = restored_stores;
                if graph_checkpoint.exists() {
                    if graph_path.exists() {
                        let _ = std::fs::remove_dir_all(&graph_path);
                    }
                    if let Err(e) = crate::backup::copy_dir_recursive_pub(&graph_checkpoint, &graph_path) {
                        tracing::warn!(error = %e, "Failed to restore graph DB from backup");
                    } else {
                        all_restored.push("graph".to_string());
                        tracing::info!("Graph DB restored from backup");
                    }
                }

                Ok(all_restored)
            }
            StorageBackend::Redb => Err(anyhow::anyhow!(
                "storage backend '{}' is not wired for backup restore yet",
                self.server_config.effective_storage_backend
            )),
        }
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

                let db_path = path.join("memory.db");
                if !db_path.exists() {
                    continue;
                }

                if self.get_user_earth(name).is_ok() {
                    match self.create_user_backup(name) {
                        Ok(metadata) => {
                            tracing::info!(
                                user_id = name,
                                backup_id = metadata.backup_id,
                                size_mb = metadata.size_bytes / 1024 / 1024,
                                "Backup created successfully"
                            );
                            backed_up += 1;

                            if let Err(e) = self.purge_backups_for_user(name, max_backups)
                            {
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
        let graph = self.get_user_graph(user_id)?;

        // =====================================================================
        // PHASE 1: CPU-INTENSIVE WORK (NO LOCK)
        // All NER, regex, query parsing happens here to minimize lock hold time.
        // Was 100-400ms under lock, now only fast I/O under lock (~10-30ms).
        // =====================================================================

        let now = chrono::Utc::now();

        // Stop words for filtering
        let stop_words: std::collections::HashSet<&str> = [
            "the", "and", "for", "that", "this", "with", "from", "have", "been", "are", "was",
            "were", "will", "would", "could", "should", "may", "might",
        ]
        .iter()
        .cloned()
        .collect();

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
                if tag_name.len() >= 2 && !stop_words.contains(tag_name.to_lowercase().as_str()) {
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

        // =====================================================================
        // PHASE 2: GRAPH INSERTION (WITH LOCK)
        // Only fast I/O operations happen here.
        // =====================================================================

        let graph_guard = graph.read();

        let mut entity_uuids = Vec::new();

        // Insert all pre-built entities
        for (name, entity) in all_entities {
            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((name, uuid)),
                Err(e) => tracing::debug!("Failed to add entity {}: {}", name, e),
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
        for i in 0..entity_uuids.len() {
            for j in (i + 1)..entity_uuids.len() {
                let state_edge_init = EdgeTier::L1Working.initial_weight();
                let edge = RelationshipEdge {
                    uuid: uuid::Uuid::new_v4(),
                    from_entity: entity_uuids[i].1,
                    to_entity: entity_uuids[j].1,
                    relation_type: RelationType::RelatedTo,
                    strength: state_edge_init,
                    created_at: now,
                    valid_at: now,
                    invalidated_at: None,
                    source_episode_id: Some(memory_id.0),
                    context: truncated_context.clone(),
                    last_activated: now,
                    activation_count: 1,
                    ltp_status: LtpStatus::None,
                    tier: EdgeTier::L1Working,
                    activation_timestamps: None,
                    entity_confidence: None,
                    created_by: crate::graph_memory::EdgeSource::CoOccurrence,
                    forward_strength: state_edge_init,
                    backward_strength: state_edge_init,
                };

                if let Err(e) = graph_guard.add_relationship(edge) {
                    tracing::debug!("Failed to add relationship: {}", e);
                }
            }
        }
        // Lock released here

        Ok(())
    }
}
