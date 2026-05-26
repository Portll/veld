//! P2: Backup & Restore System
//!
//! Provides production-grade backup and restore capabilities:
//! - Incremental backups using RocksDB checkpoints
//! - Point-in-time recovery (PITR)
//! - Export to JSON/Parquet formats
//! - Backup verification and integrity checks
//! - Automated scheduling support
//!
//! ## W5 / Phase C extensions
//!
//! - Intent-log files (the append-only journal + the checkpoint store) live
//!   outside RocksDB; they are copied into the archive under an
//!   `intent_log/` subdirectory, hashed with SHA-256, and described in the
//!   manifest with their LSN range.
//! - The `cf_user_auth` column family (Phase C) is included automatically
//!   when present in the database. The CF set actually captured is
//!   recorded in the manifest under `column_families`. Restore is a
//!   no-op for an absent CF and a full restore for a present one — the
//!   RocksDB `BackupEngine` snapshots SSTs and the MANIFEST, which carry
//!   every CF that exists in the source DB.
//! - Restore supports a `--max-lsn` point-in-time-restore option. With it
//!   set, the intent log is materialised on disk in its archived form and
//!   then truncated *down to and including* the requested LSN using
//!   [`crate::intent_log::IntentLog::truncate_to_lsn`]. Each frame is
//!   CRC-verified during the scan; a corrupt frame anywhere up to the
//!   target LSN aborts the restore.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{
    backup::{BackupEngine, BackupEngineOptions},
    checkpoint::Checkpoint,
    Env, DB,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::intent_log::{IntentLog, IntentLogIter, Lsn};

/// Column family that holds Phase C user authentication records. It is
/// future-known: if the database does not contain it, the backup proceeds
/// without it. If it exists, it is recorded in the manifest and restored
/// automatically because the RocksDB `BackupEngine` snapshots SSTs and the
/// MANIFEST as one unit.
pub const CF_USER_AUTH: &str = "cf_user_auth";

/// Other column families known to live on the *main memories* database.
/// Used only for enumeration into the manifest; the backup engine itself
/// captures whatever the source DB actually has, regardless of this list.
const KNOWN_MAIN_DB_COLUMN_FAMILIES: &[&str] = &["default", "memory_index"];

/// Backup metadata for tracking and verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup ID
    pub backup_id: u32,
    /// Timestamp when backup was created
    pub created_at: DateTime<Utc>,
    /// User ID (if single-user backup) or "all" for full backup
    pub user_id: String,
    /// Backup type: "full" or "incremental"
    pub backup_type: BackupType,
    /// Size in bytes (compressed)
    pub size_bytes: u64,
    /// SHA-256 checksum for integrity verification
    pub checksum: String,
    /// Number of memories included in backup
    pub memory_count: usize,
    /// RocksDB sequence number (for PITR)
    pub sequence_number: u64,
    /// Secondary stores included in this backup
    #[serde(default)]
    pub secondary_stores: Vec<String>,
    /// Total size of secondary store backups in bytes
    #[serde(default)]
    pub secondary_size_bytes: u64,
    /// Intent-log files captured under `intent_log/` in the archive (W5).
    /// Includes the journal log + the checkpoint store + any other files
    /// found in the configured `intent_log/` source directory.
    #[serde(default)]
    pub intent_log_files: Vec<IntentLogFileMeta>,
    /// Column families that the main memories DB actually contained at
    /// backup time. `cf_user_auth` appears here when Phase C has been
    /// initialised; otherwise it is omitted. Older backups written before
    /// this field existed deserialise with an empty vector.
    #[serde(default)]
    pub column_families: Vec<String>,
}

/// One file copied into the backup under `intent_log/`. Carries enough
/// information to (a) verify the file survived round-trip via SHA-256,
/// and (b) decide on PITR whether the LSN the operator asked for is in
/// range. The `lsn_range` is `None` for non-journal files (e.g. the
/// checkpoint store) since those don't carry LSNs in their own framing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentLogFileMeta {
    /// File name as it appears in the source directory (and as it is
    /// re-created in the restore-target directory).
    pub name: String,
    /// SHA-256 of the file contents, lower-case hex.
    pub sha256: String,
    /// Raw byte length of the file at backup time.
    pub size_bytes: u64,
    /// `Some((lo, hi))` for intent-log files where `lo` is the first LSN
    /// and `hi` is the last LSN present, both inclusive. `None` for
    /// auxiliary files (checkpoint store, future tombstone manifests).
    #[serde(default)]
    pub lsn_range: Option<(u64, u64)>,
}

/// Configuration describing where intent-log files live and where they
/// should be restored.
///
/// The same source dir is used for backup, and the same target dir for
/// restore. By default the W5 wiring puts the journal + checkpoint store
/// under `~/.veld/intent_log/`, but the path is configurable and the
/// backup engine accepts whatever the caller hands it.
#[derive(Debug, Clone)]
pub struct IntentLogBackupSpec {
    /// Directory whose files are copied into the backup archive.
    pub source_dir: PathBuf,
    /// File name (within `source_dir`) of the main journal. Used to know
    /// which file to scan for LSN range information and to know which
    /// file to verify-CRC on restore. Other files in `source_dir` are
    /// copied without CRC scanning.
    pub log_filename: String,
}

impl IntentLogBackupSpec {
    /// Build a spec from `source_dir` with the conventional log filename
    /// (`intent.log`). For tests and call sites that want full control,
    /// construct the struct directly.
    pub fn with_default_log_name(source_dir: PathBuf) -> Self {
        Self {
            source_dir,
            log_filename: "intent.log".to_string(),
        }
    }
}

/// Options applied to a single restore operation.
#[derive(Debug, Clone, Default)]
pub struct RestoreOptions {
    /// If `Some(lsn)`, the restored intent log is truncated so that the
    /// last surviving frame has `frame.lsn == lsn`. Any frames with a
    /// higher LSN in the archive are dropped on disk.
    pub max_lsn: Option<u64>,
}

/// Named reference to a RocksDB database for backup
pub struct SecondaryStoreRef<'a> {
    pub name: &'a str,
    pub db: &'a Arc<DB>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackupType {
    Full,
    Incremental,
}

/// Backup engine for creating and managing backups
pub struct VeldBackupEngine {
    backup_path: PathBuf,
}

impl VeldBackupEngine {
    /// Create a new backup engine
    ///
    /// # Arguments
    /// * `backup_path` - Directory to store backups
    pub fn new(backup_path: PathBuf) -> Result<Self> {
        fs::create_dir_all(&backup_path)?;
        Ok(Self { backup_path })
    }

    /// Get the backup storage path.
    pub fn backup_path(&self) -> &Path {
        &self.backup_path
    }

    /// Create a full backup of a RocksDB database
    ///
    /// # Arguments
    /// * `db` - Reference to the RocksDB database
    /// * `user_id` - User ID for the backup (or "all" for full system backup)
    ///
    /// # Returns
    /// BackupMetadata with backup details
    pub fn create_backup(&self, db: &DB, user_id: &str) -> Result<BackupMetadata> {
        let backup_dir = self.backup_path.join(user_id);
        fs::create_dir_all(&backup_dir)?;

        // Create RocksDB backup engine
        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &env)?;

        // Create backup
        let before_count = backup_engine.get_backup_info().len();
        backup_engine.create_new_backup(db)?;

        let backup_info = backup_engine.get_backup_info();
        let latest_backup = backup_info
            .last()
            .ok_or_else(|| anyhow!("No backup created"))?;

        let backup_id = latest_backup.backup_id;
        let size_bytes = latest_backup.size;

        // Get latest sequence number from DB
        let sequence_number = db.latest_sequence_number();

        // Count memories (estimate from DB size)
        let memory_count = self.estimate_memory_count(db)?;

        // Calculate checksum of backup directory
        let checksum = self.calculate_backup_checksum(&backup_dir, backup_id)?;

        // Determine backup type
        let backup_type = if before_count == 0 {
            BackupType::Full
        } else {
            BackupType::Incremental
        };

        // Enumerate column families actually present in the source DB.
        // The RocksDB BackupEngine captures all CFs in the snapshot
        // regardless — this list is for the manifest so a verifier can
        // assert presence of `cf_user_auth` etc.
        let column_families = enumerate_present_cfs(db);

        let metadata = BackupMetadata {
            backup_id,
            created_at: Utc::now(),
            user_id: user_id.to_string(),
            backup_type,
            size_bytes,
            checksum,
            memory_count,
            sequence_number,
            secondary_stores: Vec::new(),
            secondary_size_bytes: 0,
            intent_log_files: Vec::new(),
            column_families,
        };

        // Save metadata
        self.save_metadata(&metadata)?;

        tracing::info!(
            backup_id = backup_id,
            user_id = user_id,
            size_mb = size_bytes / 1024 / 1024,
            "Backup created successfully"
        );

        Ok(metadata)
    }

    /// Create a comprehensive backup of the main database, secondary stores, and graph.
    ///
    /// Uses RocksDB BackupEngine for the main memories DB and Checkpoint API
    /// for secondary stores (todos, reminders, facts, files, feedback, audit)
    /// and the knowledge graph database.
    pub fn create_comprehensive_backup(
        &self,
        db: &DB,
        user_id: &str,
        secondary_stores: &[SecondaryStoreRef<'_>],
    ) -> Result<BackupMetadata> {
        self.create_comprehensive_backup_with_graph(db, user_id, secondary_stores, None)
    }

    /// Create a comprehensive backup including the knowledge graph DB.
    pub fn create_comprehensive_backup_with_graph(
        &self,
        db: &DB,
        user_id: &str,
        secondary_stores: &[SecondaryStoreRef<'_>],
        graph_db: Option<&DB>,
    ) -> Result<BackupMetadata> {
        // Step 1: Create main memories backup (existing logic)
        let mut metadata = self.create_backup(db, user_id)?;

        // Step 2: Checkpoint each secondary store alongside the backup
        let secondary_dir = self
            .backup_path
            .join(user_id)
            .join(format!("secondary_{}", metadata.backup_id));
        fs::create_dir_all(&secondary_dir)?;

        // Step 2a: Checkpoint graph DB if provided
        if let Some(graph) = graph_db {
            let graph_checkpoint_dir = secondary_dir.join("graph");
            let checkpoint = Checkpoint::new(graph)
                .map_err(|e| anyhow!("Failed to create checkpoint handle for graph DB: {}", e))?;
            checkpoint
                .create_checkpoint(&graph_checkpoint_dir)
                .map_err(|e| {
                    let _ = fs::remove_dir_all(&graph_checkpoint_dir);
                    anyhow!("Failed to checkpoint graph DB: {}", e)
                })?;
            let graph_size = dir_size(&graph_checkpoint_dir).unwrap_or(0);
            tracing::debug!(size_kb = graph_size / 1024, "Graph DB checkpointed");
        }

        let mut backed_up_stores = Vec::new();
        let mut total_secondary_bytes: u64 = 0;

        for store_ref in secondary_stores {
            let store_checkpoint_dir = secondary_dir.join(store_ref.name);

            // Skip if checkpoint directory already exists (shouldn't happen, but be safe)
            if store_checkpoint_dir.exists() {
                tracing::warn!(
                    store = store_ref.name,
                    "Checkpoint directory already exists, skipping"
                );
                continue;
            }

            let checkpoint = Checkpoint::new(store_ref.db).map_err(|e| {
                anyhow!(
                    "Failed to create checkpoint handle for secondary store '{}': {}",
                    store_ref.name,
                    e
                )
            })?;

            if let Err(e) = checkpoint.create_checkpoint(&store_checkpoint_dir) {
                // Clean up partial checkpoint before returning error
                let _ = fs::remove_dir_all(&store_checkpoint_dir);
                return Err(anyhow!(
                    "Failed to checkpoint secondary store '{}': {}",
                    store_ref.name,
                    e
                ));
            }

            let store_size = dir_size(&store_checkpoint_dir).unwrap_or(0);
            total_secondary_bytes += store_size;
            backed_up_stores.push(store_ref.name.to_string());

            tracing::debug!(
                store = store_ref.name,
                size_kb = store_size / 1024,
                "Secondary store checkpointed"
            );
        }

        // Track graph in metadata if it was checkpointed
        if graph_db.is_some() {
            backed_up_stores.push("graph".to_string());
        }

        // Step 3: Update metadata with secondary store info
        metadata.secondary_stores = backed_up_stores;
        metadata.secondary_size_bytes = total_secondary_bytes;

        // Recompute checksum now that secondary stores are included.
        // The initial checksum from create_backup() only covered the main DB.
        let backup_dir = self.backup_path.join(user_id);
        metadata.checksum = self.calculate_backup_checksum(&backup_dir, metadata.backup_id)?;
        self.save_metadata(&metadata)?;

        tracing::info!(
            backup_id = metadata.backup_id,
            user_id = user_id,
            secondary_stores = metadata.secondary_stores.len(),
            secondary_size_kb = total_secondary_bytes / 1024,
            "Comprehensive backup created"
        );

        Ok(metadata)
    }

    /// Create a comprehensive backup that ALSO captures the W5 intent log
    /// directory. Behaviour is identical to
    /// [`create_comprehensive_backup_with_graph`] for the main DB, secondary
    /// stores, and graph; in addition every file under
    /// `intent_log_spec.source_dir` is copied into the archive at
    /// `<backup_dir>/intent_log_<backup_id>/`, SHA-256-hashed, and the main
    /// journal is scanned to derive its LSN range. The manifest's
    /// `intent_log_files` field is populated and the archive checksum is
    /// recomputed to cover the new tree.
    pub fn create_comprehensive_backup_with_intent_log(
        &self,
        db: &DB,
        user_id: &str,
        secondary_stores: &[SecondaryStoreRef<'_>],
        graph_db: Option<&DB>,
        intent_log_spec: Option<&IntentLogBackupSpec>,
    ) -> Result<BackupMetadata> {
        let mut metadata =
            self.create_comprehensive_backup_with_graph(db, user_id, secondary_stores, graph_db)?;

        if let Some(spec) = intent_log_spec {
            let backup_dir = self.backup_path.join(user_id);
            let intent_log_dir =
                backup_dir.join(format!("intent_log_{}", metadata.backup_id));
            fs::create_dir_all(&intent_log_dir)?;

            metadata.intent_log_files =
                self.copy_intent_log_into_archive(spec, &intent_log_dir)?;

            // Recompute checksum now that the intent-log tree is in place.
            metadata.checksum =
                self.calculate_backup_checksum(&backup_dir, metadata.backup_id)?;
            self.save_metadata(&metadata)?;

            tracing::info!(
                backup_id = metadata.backup_id,
                user_id = user_id,
                files = metadata.intent_log_files.len(),
                "Intent log copied into backup archive"
            );
        }

        Ok(metadata)
    }

    /// Copy every file in `spec.source_dir` (non-recursive — the intent log
    /// is a flat directory) into `target_dir`, hashing as we go. Returns
    /// one [`IntentLogFileMeta`] per file.
    fn copy_intent_log_into_archive(
        &self,
        spec: &IntentLogBackupSpec,
        target_dir: &Path,
    ) -> Result<Vec<IntentLogFileMeta>> {
        if !spec.source_dir.exists() {
            // Nothing to back up. This is legal — W5 may not be active
            // yet on a given installation. Record an empty list.
            return Ok(Vec::new());
        }

        let mut metas = Vec::new();
        let mut entries: Vec<_> = fs::read_dir(&spec.source_dir)
            .with_context(|| {
                format!(
                    "failed to read intent log directory {:?}",
                    spec.source_dir
                )
            })?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let dst = target_dir.join(&file_name);

            // Stream-copy + hash + size so a huge log doesn't load fully
            // into memory.
            let (sha256, size_bytes) = copy_and_hash(&path, &dst)?;

            // For the main journal file, scan for the LSN range so PITR
            // can decide whether the operator-requested LSN is in range.
            let lsn_range = if file_name == spec.log_filename {
                scan_lsn_range(&dst)
                    .with_context(|| {
                        format!(
                            "failed to scan LSN range for intent log {:?}",
                            dst
                        )
                    })?
            } else {
                None
            };

            metas.push(IntentLogFileMeta {
                name: file_name,
                sha256,
                size_bytes,
                lsn_range,
            });
        }

        Ok(metas)
    }

    /// Restore from a specific backup
    ///
    /// # Arguments
    /// * `user_id` - User ID to restore
    /// * `backup_id` - Backup ID to restore from (None = latest)
    /// * `restore_path` - Path to restore the database to
    pub fn restore_backup(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        restore_path: &Path,
    ) -> Result<()> {
        let backup_dir = self.backup_path.join(user_id);

        if !backup_dir.exists() {
            return Err(anyhow!("No backups found for user: {user_id}"));
        }

        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &env)?;

        // Restore from specific backup or latest
        match backup_id {
            Some(id) => {
                tracing::info!(backup_id = id, "Restoring from specific backup");
                backup_engine.restore_from_backup(
                    restore_path,
                    restore_path,
                    &rocksdb::backup::RestoreOptions::default(),
                    id,
                )?;
            }
            None => {
                tracing::info!("Restoring from latest backup");
                backup_engine.restore_from_latest_backup(
                    restore_path,
                    restore_path,
                    &rocksdb::backup::RestoreOptions::default(),
                )?;
            }
        }

        tracing::info!(
            user_id = user_id,
            restore_path = ?restore_path,
            "Restore completed successfully"
        );

        Ok(())
    }

    /// List all available backups for a user
    pub fn list_backups(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        let backup_dir = self.backup_path.join(user_id);

        if !backup_dir.exists() {
            return Ok(Vec::new());
        }

        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let backup_engine = BackupEngine::open(&backup_opts, &env)?;

        let backup_info = backup_engine.get_backup_info();
        let mut metadata_list = Vec::new();

        for info in backup_info {
            if let Ok(metadata) = self.load_metadata(user_id, info.backup_id) {
                metadata_list.push(metadata);
            }
        }

        Ok(metadata_list)
    }

    /// Restore from a comprehensive backup, including secondary stores.
    ///
    /// The `secondary_restore_paths` map store names to their target restore directories.
    /// Secondary stores are restored by copying the checkpoint directory to the target path.
    pub fn restore_comprehensive_backup(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        restore_path: &Path,
        secondary_restore_paths: &[(&str, &Path)],
    ) -> Result<Vec<String>> {
        // Step 1: Restore main memories DB
        self.restore_backup(user_id, backup_id, restore_path)?;

        // Step 2: Determine which backup_id was restored
        let resolved_backup_id = match backup_id {
            Some(id) => id,
            None => {
                let backup_dir = self.backup_path.join(user_id);
                let backup_opts = BackupEngineOptions::new(&backup_dir)?;
                let env = Env::new()?;
                let backup_engine = BackupEngine::open(&backup_opts, &env)?;
                let info = backup_engine.get_backup_info();
                info.last()
                    .map(|i| i.backup_id)
                    .ok_or_else(|| anyhow!("No backups available"))?
            }
        };

        // Step 3: Restore secondary stores from checkpoints
        let secondary_dir = self
            .backup_path
            .join(user_id)
            .join(format!("secondary_{resolved_backup_id}"));

        let mut restored_stores = Vec::new();

        if secondary_dir.exists() {
            for (store_name, target_path) in secondary_restore_paths {
                let checkpoint_dir = secondary_dir.join(store_name);
                if !checkpoint_dir.exists() {
                    tracing::debug!(
                        store = *store_name,
                        "No checkpoint found in backup, skipping"
                    );
                    continue;
                }

                // Safe restore: copy to temp dir first, then atomic swap.
                // This prevents data loss if copy fails midway.
                let mut tmp_os = target_path.as_os_str().to_os_string();
                tmp_os.push(".restore_tmp");
                let temp_path = PathBuf::from(tmp_os);
                if temp_path.exists() {
                    fs::remove_dir_all(&temp_path).map_err(|e| {
                        anyhow!(
                            "Failed to clean up stale temp dir for {}: {}",
                            store_name,
                            e
                        )
                    })?;
                }

                if let Err(e) = copy_dir_recursive(&checkpoint_dir, &temp_path) {
                    // Copy failed — clean up temp, leave original intact
                    let _ = fs::remove_dir_all(&temp_path);
                    tracing::warn!(
                        store = *store_name,
                        error = %e,
                        "Failed to copy checkpoint for restore, skipping (original data preserved)"
                    );
                    continue;
                }

                // Copy succeeded — now swap: remove original, rename temp to target
                if target_path.exists() {
                    if let Err(e) = fs::remove_dir_all(target_path) {
                        // Can't remove original — roll back by removing temp
                        let _ = fs::remove_dir_all(&temp_path);
                        return Err(anyhow!(
                            "Failed to remove existing {} directory at {:?}: {}",
                            store_name,
                            target_path,
                            e
                        ));
                    }
                }

                if let Err(e) = fs::rename(&temp_path, target_path) {
                    // Rename failed (cross-device?), fall back to copy + remove temp
                    if let Err(copy_err) = copy_dir_recursive(&temp_path, target_path) {
                        let _ = fs::remove_dir_all(&temp_path);
                        return Err(anyhow!(
                            "Failed to finalize restore for {}: rename={}, copy={}",
                            store_name,
                            e,
                            copy_err
                        ));
                    }
                    let _ = fs::remove_dir_all(&temp_path);
                }

                restored_stores.push(store_name.to_string());
                tracing::info!(
                    store = *store_name,
                    target = ?target_path,
                    "Secondary store restored from checkpoint"
                );
            }
        }

        tracing::info!(
            user_id = user_id,
            backup_id = resolved_backup_id,
            restored_secondary = restored_stores.len(),
            "Comprehensive restore completed"
        );

        Ok(restored_stores)
    }

    /// Restore from a comprehensive backup AND materialise the intent log
    /// into `intent_log_target.source_dir`.
    ///
    /// Behaviour:
    /// 1. Run [`restore_comprehensive_backup`] for the main DB + secondary
    ///    stores.
    /// 2. Locate the archive's `intent_log_<backup_id>` subfolder.
    /// 3. For every file recorded in the manifest, copy it into
    ///    `intent_log_target.source_dir`, verifying its SHA-256 against
    ///    the manifest. A mismatch aborts the restore (no partial state
    ///    on disk — files written so far are removed).
    /// 4. Open the journal file via [`IntentLog::open`] and **iterate
    ///    every frame**. A CRC mismatch or torn frame aborts the restore
    ///    with a clear error — half-restoring a corrupt log is
    ///    catastrophic and we refuse to do it.
    /// 5. If `options.max_lsn` is set, call
    ///    [`IntentLog::truncate_to_lsn`] on the freshly-restored journal
    ///    so that the final on-disk log ends at exactly that LSN.
    ///
    /// Returns the list of stores that were restored (compatible with
    /// [`restore_comprehensive_backup`]), with `"intent_log"` appended if
    /// the intent-log materialisation succeeded.
    pub fn restore_comprehensive_backup_with_intent_log(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        restore_path: &Path,
        secondary_restore_paths: &[(&str, &Path)],
        intent_log_target: Option<&IntentLogBackupSpec>,
        options: &RestoreOptions,
    ) -> Result<Vec<String>> {
        let mut restored = self.restore_comprehensive_backup(
            user_id,
            backup_id,
            restore_path,
            secondary_restore_paths,
        )?;

        let Some(target) = intent_log_target else {
            return Ok(restored);
        };

        // Resolve backup id (may have been None = latest).
        let resolved_id = match backup_id {
            Some(id) => id,
            None => {
                let backup_dir = self.backup_path.join(user_id);
                let backup_opts = BackupEngineOptions::new(&backup_dir)?;
                let env = Env::new()?;
                let engine = BackupEngine::open(&backup_opts, &env)?;
                engine
                    .get_backup_info()
                    .last()
                    .map(|i| i.backup_id)
                    .ok_or_else(|| anyhow!("No backups available for intent log restore"))?
            }
        };

        let archive_intent_dir = self
            .backup_path
            .join(user_id)
            .join(format!("intent_log_{resolved_id}"));
        if !archive_intent_dir.exists() {
            // No intent log was captured for this backup. Not an error —
            // pre-W5 backups will hit this path.
            tracing::debug!(
                backup_id = resolved_id,
                "Backup contains no intent_log subdir, skipping"
            );
            return Ok(restored);
        }

        let metadata = self.load_metadata(user_id, resolved_id)?;

        // Materialise files into a staging dir, hash-verify each, then
        // swap into place. Staging keeps the existing intent log dir
        // intact until we know the restored copy is good.
        let mut staging_os = target.source_dir.as_os_str().to_os_string();
        staging_os.push(".restore_tmp");
        let staging = PathBuf::from(staging_os);
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        let mut newly_written: Vec<PathBuf> = Vec::new();
        for file_meta in &metadata.intent_log_files {
            let src = archive_intent_dir.join(&file_meta.name);
            if !src.exists() {
                // Manifest references a file that's missing from the
                // archive — refuse to half-restore.
                let _ = fs::remove_dir_all(&staging);
                return Err(anyhow!(
                    "intent log file {:?} listed in manifest is missing from archive",
                    file_meta.name
                ));
            }
            let dst = staging.join(&file_meta.name);
            let (sha256, size_bytes) = copy_and_hash(&src, &dst)?;
            if sha256 != file_meta.sha256 || size_bytes != file_meta.size_bytes {
                let _ = fs::remove_dir_all(&staging);
                return Err(anyhow!(
                    "intent log file {:?} failed integrity check on restore (archive SHA-256={} expected={}; archive size={} expected={})",
                    file_meta.name,
                    sha256,
                    file_meta.sha256,
                    size_bytes,
                    file_meta.size_bytes,
                ));
            }
            newly_written.push(dst);
        }

        // Frame-level CRC verification on the journal: open it as an
        // IntentLog and walk every frame. Any error means the archive
        // contains a corrupt log — fail loudly rather than handing the
        // operator a broken log.
        let journal_path = staging.join(&target.log_filename);
        if journal_path.exists() {
            let walk = IntentLog::open(&journal_path)?;
            let iter = walk.iter()?;
            for frame in iter {
                if let Err(e) = frame {
                    let _ = fs::remove_dir_all(&staging);
                    return Err(anyhow!(
                        "intent log frame failed CRC verification during restore: {}",
                        e
                    ));
                }
            }
        }

        // PITR: if a max_lsn was requested, truncate the staged log down
        // to it before swapping into place. truncate_to_lsn aborts on
        // mid-log corruption, so the half-restore concern is addressed
        // there too.
        if let Some(max_lsn_u64) = options.max_lsn {
            if !journal_path.exists() {
                let _ = fs::remove_dir_all(&staging);
                return Err(anyhow!(
                    "max_lsn requested but no journal file found in archive"
                ));
            }
            let mut log = IntentLog::open(&journal_path)?;
            log.truncate_to_lsn(Lsn(max_lsn_u64)).map_err(|e| {
                let _ = fs::remove_dir_all(&staging);
                anyhow!(
                    "PITR truncate_to_lsn({}) failed: {}",
                    max_lsn_u64,
                    e
                )
            })?;
            log.sync().context("failed to sync truncated intent log")?;
        }

        // Swap staging into place. We delete any pre-existing target
        // first, since the operator explicitly asked for the restored
        // state to be on disk. (Note: this is destructive — that is the
        // whole point of restore.)
        if target.source_dir.exists() {
            fs::remove_dir_all(&target.source_dir).map_err(|e| {
                // Leave staging in place so the operator can recover.
                anyhow!(
                    "failed to remove existing intent log dir {:?}: {}",
                    target.source_dir,
                    e
                )
            })?;
        }
        if let Some(parent) = target.source_dir.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::rename(&staging, &target.source_dir) {
            Ok(()) => {}
            Err(_) => {
                // Fall back to copy + remove for cross-device renames.
                copy_dir_recursive(&staging, &target.source_dir)?;
                let _ = fs::remove_dir_all(&staging);
            }
        }

        restored.push("intent_log".to_string());
        tracing::info!(
            user_id = user_id,
            backup_id = resolved_id,
            files = newly_written.len(),
            max_lsn = ?options.max_lsn,
            "Intent log restored from backup"
        );

        Ok(restored)
    }

    /// Delete old backups, keeping only the most recent N backups.
    /// `keep_count` must be >= 1 to prevent accidental deletion of all backups.
    pub fn purge_old_backups(&self, user_id: &str, keep_count: usize) -> Result<usize> {
        if keep_count == 0 {
            return Err(anyhow!(
                "keep_count must be >= 1 to prevent deleting all backups"
            ));
        }

        let backup_dir = self.backup_path.join(user_id);

        if !backup_dir.exists() {
            return Ok(0);
        }

        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &env)?;

        let backup_info = backup_engine.get_backup_info();
        let total_backups = backup_info.len();

        if total_backups <= keep_count {
            return Ok(0);
        }

        let to_delete = total_backups - keep_count;

        // Collect IDs of backups that will be purged (oldest ones)
        let mut purge_ids: Vec<u32> = backup_info.iter().map(|b| b.backup_id).collect();
        purge_ids.sort();
        let purge_ids: Vec<u32> = purge_ids.into_iter().take(to_delete).collect();

        // Delete oldest backups (purge keeps the most recent N backups)
        backup_engine.purge_old_backups(keep_count)?;

        // Clean up secondary store checkpoints + intent-log captures for purged backups
        for purged_id in &purge_ids {
            let secondary_dir = backup_dir.join(format!("secondary_{purged_id}"));
            if secondary_dir.exists() {
                if let Err(e) = fs::remove_dir_all(&secondary_dir) {
                    tracing::warn!(
                        backup_id = purged_id,
                        error = %e,
                        "Failed to clean up secondary store checkpoint"
                    );
                }
            }
            let intent_log_dir = backup_dir.join(format!("intent_log_{purged_id}"));
            if intent_log_dir.exists() {
                if let Err(e) = fs::remove_dir_all(&intent_log_dir) {
                    tracing::warn!(
                        backup_id = purged_id,
                        error = %e,
                        "Failed to clean up intent log archive"
                    );
                }
            }
            // Clean up metadata file
            let metadata_path = backup_dir.join(format!("backup_{purged_id}.json"));
            if let Err(e) = fs::remove_file(&metadata_path) {
                tracing::warn!(
                    backup_id = purged_id,
                    error = %e,
                    "Failed to remove backup metadata file"
                );
            }
        }

        tracing::info!(
            purged_count = to_delete,
            kept_count = keep_count,
            user_id = user_id,
            "Purged old backups"
        );

        Ok(to_delete)
    }

    /// Verify backup integrity using checksum
    pub fn verify_backup(&self, user_id: &str, backup_id: u32) -> Result<bool> {
        let metadata = self.load_metadata(user_id, backup_id)?;
        let backup_dir = self.backup_path.join(user_id);

        let current_checksum = self.calculate_backup_checksum(&backup_dir, backup_id)?;

        Ok(current_checksum == metadata.checksum)
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    fn save_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let metadata_path = self
            .backup_path
            .join(&metadata.user_id)
            .join(format!("backup_{}.json", metadata.backup_id));

        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(metadata_path, json)?;

        Ok(())
    }

    fn load_metadata(&self, user_id: &str, backup_id: u32) -> Result<BackupMetadata> {
        let metadata_path = self
            .backup_path
            .join(user_id)
            .join(format!("backup_{backup_id}.json"));

        let json = fs::read_to_string(metadata_path)?;
        let metadata = serde_json::from_str(&json)?;

        Ok(metadata)
    }

    fn calculate_backup_checksum(&self, backup_dir: &Path, backup_id: u32) -> Result<String> {
        let mut hasher = Sha256::new();

        // Hash main backup directory (sorted by filename for deterministic ordering)
        let backup_path = backup_dir.join(format!("private/{backup_id}"));
        self.hash_directory_sorted(&backup_path, &mut hasher)?;

        // Hash secondary store directory (B5: was previously excluded)
        let secondary_path = backup_dir.join(format!("secondary_{backup_id}"));
        self.hash_directory_sorted(&secondary_path, &mut hasher)?;

        // Hash intent-log directory if present (W5: included by
        // create_comprehensive_backup_with_intent_log). Missing dir hashes
        // to the empty input — pre-W5 backups produce the same digest as
        // before this field existed.
        let intent_log_path = backup_dir.join(format!("intent_log_{backup_id}"));
        self.hash_directory_sorted(&intent_log_path, &mut hasher)?;

        let result = hasher.finalize();
        Ok(format!("{result:x}"))
    }

    fn hash_directory_sorted(&self, dir: &Path, hasher: &mut Sha256) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        let mut entries: Vec<_> = fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            // Hash filename for rename detection
            hasher.update(entry.file_name().to_string_lossy().as_bytes());
            if path.is_dir() {
                // Recurse into subdirectories (secondary stores have nested structure)
                self.hash_directory_sorted(&path, hasher)?;
            } else {
                let file_contents = fs::read(&path)?;
                hasher.update(&file_contents);
            }
        }
        Ok(())
    }

    fn estimate_memory_count(&self, db: &DB) -> Result<usize> {
        // Estimate by counting keys (this is a rough estimate)
        let mut count = 0;
        let iter = db.iterator(rocksdb::IteratorMode::Start);

        for _ in iter {
            count += 1;
        }

        Ok(count)
    }
}

/// Public wrapper for copy_dir_recursive (used by restore handler).
pub fn copy_dir_recursive_pub(src: &Path, dst: &Path) -> Result<()> {
    copy_dir_recursive(src, dst)
}

/// Calculate total size of a directory recursively
fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                total += dir_size(&entry_path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    }
    Ok(total)
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Enumerate the column families that the current DB instance actually
/// contains. We probe a known set including Phase C's `cf_user_auth`. CFs
/// that are not present are simply omitted from the result.
///
/// This is for the manifest only — the RocksDB BackupEngine snapshots
/// whatever CFs exist regardless of this list.
fn enumerate_present_cfs(db: &DB) -> Vec<String> {
    let candidates: Vec<&str> = KNOWN_MAIN_DB_COLUMN_FAMILIES
        .iter()
        .copied()
        .chain(std::iter::once(CF_USER_AUTH))
        .collect();

    let mut out = Vec::new();
    for name in candidates {
        // `default` always exists; for it, skip the cf_handle probe so we
        // don't depend on RocksDB returning a handle for the implicit
        // default CF (some crate versions do, some don't).
        if name == "default" {
            out.push(name.to_string());
            continue;
        }
        if db.cf_handle(name).is_some() {
            out.push(name.to_string());
        }
    }
    out
}

/// Stream-copy `src` to `dst` while computing its SHA-256 and byte count.
/// Used for intent-log archive captures (potentially many MiB) so we
/// don't load the whole file into memory.
fn copy_and_hash(src: &Path, dst: &Path) -> Result<(String, u64)> {
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Write};

    let in_file = File::open(src)
        .with_context(|| format!("open source intent log file {src:?}"))?;
    let mut reader = BufReader::new(in_file);

    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    let out_file =
        File::create(dst).with_context(|| format!("create dest intent log file {dst:?}"))?;
    let mut writer = BufWriter::new(out_file);

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        writer.write_all(&buf[..n])?;
        total += n as u64;
    }
    writer.flush()?;
    let digest = hasher.finalize();
    Ok((format!("{digest:x}"), total))
}

/// Scan an intent log file at `path` and return `Some((min_lsn, max_lsn))`
/// if any frames are present, `None` if the file is empty. Returns an
/// error if any frame is corrupt — the journal isn't supposed to be
/// captured mid-corruption, and silently widening the LSN window would
/// mislead PITR callers.
fn scan_lsn_range(path: &Path) -> Result<Option<(u64, u64)>> {
    use std::fs::File;
    let file = File::open(path).with_context(|| format!("open intent log {path:?}"))?;
    if file.metadata()?.len() == 0 {
        return Ok(None);
    }
    drop(file);

    let mut iter = IntentLogIter::open_for_scan(path)?;
    let mut min_lsn: Option<u64> = None;
    let mut max_lsn: Option<u64> = None;
    loop {
        match iter.next() {
            None => break,
            Some(Ok(rec)) => {
                if min_lsn.is_none() {
                    min_lsn = Some(rec.lsn.0);
                }
                max_lsn = Some(rec.lsn.0);
            }
            Some(Err(e)) => {
                return Err(anyhow!("intent log frame error during scan: {e}"));
            }
        }
    }

    Ok(match (min_lsn, max_lsn) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocksdb::Options;
    use serde_json::Value;
    use tempfile::TempDir;

    #[test]
    fn test_backup_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let backup_engine = VeldBackupEngine::new(temp_dir.path().to_path_buf());
        assert!(backup_engine.is_ok());
    }

    #[test]
    fn test_backup_metadata_serialization() {
        let metadata = BackupMetadata {
            backup_id: 1,
            created_at: Utc::now(),
            user_id: "test_user".to_string(),
            backup_type: BackupType::Full,
            size_bytes: 1024,
            checksum: "abc123".to_string(),
            memory_count: 100,
            sequence_number: 42,
            secondary_stores: vec!["todo_items".to_string(), "prospective_tasks".to_string()],
            secondary_size_bytes: 2048,
            intent_log_files: vec![IntentLogFileMeta {
                name: "intent.log".to_string(),
                sha256: "deadbeef".to_string(),
                size_bytes: 512,
                lsn_range: Some((0, 9)),
            }],
            column_families: vec![
                "default".to_string(),
                "memory_index".to_string(),
                CF_USER_AUTH.to_string(),
            ],
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: BackupMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata.backup_id, deserialized.backup_id);
        assert_eq!(metadata.user_id, deserialized.user_id);
        assert_eq!(metadata.intent_log_files.len(), 1);
        assert_eq!(metadata.intent_log_files[0].name, "intent.log");
        assert_eq!(metadata.intent_log_files[0].lsn_range, Some((0, 9)));
        assert!(metadata.column_families.contains(&CF_USER_AUTH.to_string()));
    }

    #[test]
    fn test_backup_metadata_backward_compatible_default_fields() {
        // Old serialized form without intent_log_files / column_families
        // must deserialise (serde(default) → empty vectors).
        let old_json = r#"{
            "backup_id": 7,
            "created_at": "2024-01-01T00:00:00Z",
            "user_id": "u",
            "backup_type": "Full",
            "size_bytes": 0,
            "checksum": "x",
            "memory_count": 0,
            "sequence_number": 0
        }"#;
        let parsed: BackupMetadata = serde_json::from_str(old_json).unwrap();
        assert!(parsed.intent_log_files.is_empty());
        assert!(parsed.column_families.is_empty());
        assert!(parsed.secondary_stores.is_empty());
    }

    #[test]
    fn test_dir_size_counts_nested_files() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();

        fs::write(root.join("a.txt"), b"12345").unwrap();
        fs::write(nested.join("b.txt"), b"1234567890").unwrap();

        let size = dir_size(root).unwrap();
        assert_eq!(size, 15);
    }

    #[test]
    fn test_copy_dir_recursive_pub_copies_files() {
        let temp_dir = TempDir::new().unwrap();
        let src = temp_dir.path().join("src");
        let dst = temp_dir.path().join("dst");
        fs::create_dir_all(src.join("deep")).unwrap();
        fs::write(src.join("file1.txt"), b"alpha").unwrap();
        fs::write(src.join("deep").join("file2.txt"), b"beta").unwrap();

        copy_dir_recursive_pub(&src, &dst).unwrap();

        assert_eq!(fs::read(dst.join("file1.txt")).unwrap(), b"alpha");
        assert_eq!(
            fs::read(dst.join("deep").join("file2.txt")).unwrap(),
            b"beta"
        );
    }

    #[test]
    fn test_list_backups_empty_when_user_missing() {
        let temp_dir = TempDir::new().unwrap();
        let engine = VeldBackupEngine::new(temp_dir.path().to_path_buf()).unwrap();
        let backups = engine.list_backups("missing-user").unwrap();
        assert!(backups.is_empty());
    }

    #[test]
    fn test_verify_backup_round_trip() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let user_id = "user1";

        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, &db_path).unwrap();
        db.put(b"k1", b"v1").unwrap();
        db.put(b"k2", b"v2").unwrap();

        let engine = VeldBackupEngine::new(backup_root.clone()).unwrap();
        let metadata = engine.create_backup(&db, user_id).unwrap();

        let verified = engine.verify_backup(user_id, metadata.backup_id).unwrap();
        assert!(verified);
    }

    #[test]
    fn test_verify_backup_detects_checksum_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let user_id = "user2";

        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, &db_path).unwrap();
        db.put(b"k", b"v").unwrap();

        let engine = VeldBackupEngine::new(backup_root.clone()).unwrap();
        let metadata = engine.create_backup(&db, user_id).unwrap();

        let metadata_path = backup_root
            .join(user_id)
            .join(format!("backup_{}.json", metadata.backup_id));
        let json = fs::read_to_string(&metadata_path).unwrap();
        let mut parsed: Value = serde_json::from_str(&json).unwrap();
        parsed["checksum"] = Value::String("0000badchecksum".to_string());
        fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&parsed).unwrap(),
        )
        .unwrap();

        let verified = engine.verify_backup(user_id, metadata.backup_id).unwrap();
        assert!(!verified);
    }

    #[test]
    fn test_purge_old_backups_validates_keep_count() {
        let temp_dir = TempDir::new().unwrap();
        let engine = VeldBackupEngine::new(temp_dir.path().to_path_buf()).unwrap();
        let err = engine.purge_old_backups("user", 0).unwrap_err();
        assert!(err.to_string().contains("keep_count must be >= 1"));
    }

    #[test]
    fn test_purge_old_backups_removes_old_entries() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let user_id = "purge-user";

        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, &db_path).unwrap();

        let engine = VeldBackupEngine::new(backup_root.clone()).unwrap();
        db.put(b"k1", b"v1").unwrap();
        let first = engine.create_backup(&db, user_id).unwrap();
        db.put(b"k2", b"v2").unwrap();
        let second = engine.create_backup(&db, user_id).unwrap();

        let purged = engine.purge_old_backups(user_id, 1).unwrap();
        assert_eq!(purged, 1);

        let remaining = engine.list_backups(user_id).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].backup_id, second.backup_id);

        let first_metadata_path = backup_root
            .join(user_id)
            .join(format!("backup_{}.json", first.backup_id));
        assert!(!first_metadata_path.exists());
    }

    // ========================================================================
    // W5 / Phase C: intent log + cf_user_auth + PITR coverage
    // ========================================================================

    /// Helper: build a tiny intent log dir with N frames + a checkpoint
    /// store file, return the spec and the LSN values written.
    fn build_intent_log_dir(
        dir: &Path,
        frame_payloads: &[&[u8]],
    ) -> (IntentLogBackupSpec, Vec<u64>) {
        fs::create_dir_all(dir).unwrap();
        let log_path = dir.join("intent.log");
        let mut log = IntentLog::open(&log_path).unwrap();
        let mut lsns = Vec::new();
        for p in frame_payloads {
            let lsn = log.append(p).unwrap();
            lsns.push(lsn.0);
        }
        log.sync().unwrap();
        drop(log);

        // Add a sibling checkpoint-store-shaped file so the backup
        // includes more than just the journal.
        let checkpoint_path = dir.join("checkpoints.bin");
        fs::write(&checkpoint_path, b"checkpoint-store-bytes").unwrap();

        (
            IntentLogBackupSpec::with_default_log_name(dir.to_path_buf()),
            lsns,
        )
    }

    /// Helper: create a DB that has `cf_user_auth` and write a row into it.
    fn open_db_with_user_auth(path: &Path) -> rocksdb::DB {
        use rocksdb::{ColumnFamilyDescriptor, Options};
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new("memory_index", Options::default()),
            ColumnFamilyDescriptor::new(CF_USER_AUTH, Options::default()),
        ];
        let db = rocksdb::DB::open_cf_descriptors(&opts, path, cfs).unwrap();
        let cf = db.cf_handle(CF_USER_AUTH).unwrap();
        db.put_cf(cf, b"user-row-key", b"argon2id-hash-bytes")
            .unwrap();
        db.put(b"mem-key", b"mem-value").unwrap();
        db
    }

    #[test]
    fn manifest_lists_cf_user_auth_when_present() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let user_id = "auth-user";

        let db = open_db_with_user_auth(&db_path);

        let engine = VeldBackupEngine::new(backup_root.clone()).unwrap();
        let metadata = engine.create_backup(&db, user_id).unwrap();

        assert!(metadata
            .column_families
            .iter()
            .any(|c| c == CF_USER_AUTH));
        assert!(metadata
            .column_families
            .iter()
            .any(|c| c == "default"));
    }

    #[test]
    fn manifest_omits_cf_user_auth_when_absent() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let user_id = "noauth-user";

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, &db_path).unwrap();
        db.put(b"k", b"v").unwrap();

        let engine = VeldBackupEngine::new(backup_root).unwrap();
        let metadata = engine.create_backup(&db, user_id).unwrap();

        assert!(!metadata
            .column_families
            .iter()
            .any(|c| c == CF_USER_AUTH));
        assert!(metadata
            .column_families
            .iter()
            .any(|c| c == "default"));
    }

    #[test]
    fn backup_and_restore_round_trips_intent_log_and_cf_user_auth() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let restored_db_path = temp_dir.path().join("restored_db");
        let intent_source = temp_dir.path().join("intent_log");
        let intent_restore = temp_dir.path().join("intent_log_restored");
        let user_id = "e2e";

        let db = open_db_with_user_auth(&db_path);

        let (spec, lsns) =
            build_intent_log_dir(&intent_source, &[b"frame-A", b"frame-B", b"frame-C"]);
        assert_eq!(lsns, vec![0, 1, 2]);

        let engine = VeldBackupEngine::new(backup_root.clone()).unwrap();
        let metadata = engine
            .create_comprehensive_backup_with_intent_log(
                &db,
                user_id,
                &[],
                None,
                Some(&spec),
            )
            .unwrap();

        // Manifest carries both new fields.
        assert!(metadata
            .column_families
            .iter()
            .any(|c| c == CF_USER_AUTH));
        assert_eq!(metadata.intent_log_files.len(), 2);
        let journal_meta = metadata
            .intent_log_files
            .iter()
            .find(|m| m.name == "intent.log")
            .unwrap();
        assert_eq!(journal_meta.lsn_range, Some((0, 2)));
        let checkpoint_meta = metadata
            .intent_log_files
            .iter()
            .find(|m| m.name == "checkpoints.bin")
            .unwrap();
        assert!(checkpoint_meta.lsn_range.is_none());

        // Verify checksum still holds with intent log included.
        assert!(engine.verify_backup(user_id, metadata.backup_id).unwrap());

        // Close source DB before restore (Windows file locking).
        drop(db);

        let restore_spec = IntentLogBackupSpec::with_default_log_name(intent_restore.clone());
        let restored = engine
            .restore_comprehensive_backup_with_intent_log(
                user_id,
                Some(metadata.backup_id),
                &restored_db_path,
                &[],
                Some(&restore_spec),
                &RestoreOptions::default(),
            )
            .unwrap();

        assert!(restored.iter().any(|s| s == "intent_log"));

        // Intent log files exist with expected content.
        let restored_journal = intent_restore.join("intent.log");
        assert!(restored_journal.exists());
        let restored_cp = intent_restore.join("checkpoints.bin");
        assert!(restored_cp.exists());
        assert_eq!(fs::read(&restored_cp).unwrap(), b"checkpoint-store-bytes");

        // Walking the restored journal yields the same frames in order.
        let log = IntentLog::open(&restored_journal).unwrap();
        let recs: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].payload, b"frame-A");
        assert_eq!(recs[1].payload, b"frame-B");
        assert_eq!(recs[2].payload, b"frame-C");
        drop(log);

        // cf_user_auth row survived the round-trip.
        use rocksdb::{ColumnFamilyDescriptor, Options as RocksOptions};
        let mut opts = RocksOptions::default();
        opts.create_if_missing(false);
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", RocksOptions::default()),
            ColumnFamilyDescriptor::new("memory_index", RocksOptions::default()),
            ColumnFamilyDescriptor::new(CF_USER_AUTH, RocksOptions::default()),
        ];
        let restored_db =
            rocksdb::DB::open_cf_descriptors(&opts, &restored_db_path, cfs).unwrap();
        let cf = restored_db.cf_handle(CF_USER_AUTH).unwrap();
        let val = restored_db.get_cf(cf, b"user-row-key").unwrap();
        assert_eq!(val.unwrap(), b"argon2id-hash-bytes");
    }

    #[test]
    fn restore_with_max_lsn_truncates_log_inclusive() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let restored_db_path = temp_dir.path().join("restored_db");
        let intent_source = temp_dir.path().join("intent_log");
        let intent_restore = temp_dir.path().join("intent_log_restored");
        let user_id = "pitr";

        // Five frames written (LSNs 0..=4).
        let db = open_db_with_user_auth(&db_path);
        let (spec, _) = build_intent_log_dir(
            &intent_source,
            &[b"f0", b"f1", b"f2", b"f3", b"f4"],
        );

        let engine = VeldBackupEngine::new(backup_root).unwrap();
        let metadata = engine
            .create_comprehensive_backup_with_intent_log(
                &db,
                user_id,
                &[],
                None,
                Some(&spec),
            )
            .unwrap();
        drop(db);

        // Restore with --max-lsn=2 → final log must have exactly N+1 = 3
        // frames (LSNs 0, 1, 2).
        let restore_spec =
            IntentLogBackupSpec::with_default_log_name(intent_restore.clone());
        let options = RestoreOptions { max_lsn: Some(2) };
        engine
            .restore_comprehensive_backup_with_intent_log(
                user_id,
                Some(metadata.backup_id),
                &restored_db_path,
                &[],
                Some(&restore_spec),
                &options,
            )
            .unwrap();

        let restored_journal = intent_restore.join("intent.log");
        let log = IntentLog::open(&restored_journal).unwrap();
        let recs: Vec<_> = log.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs.last().unwrap().lsn, Lsn(2));
    }

    #[test]
    fn restore_rejects_archive_with_corrupt_intent_log_frame() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let restored_db_path = temp_dir.path().join("restored_db");
        let intent_source = temp_dir.path().join("intent_log");
        let intent_restore = temp_dir.path().join("intent_log_restored");
        let user_id = "corrupt";

        let db = open_db_with_user_auth(&db_path);
        let (spec, _) =
            build_intent_log_dir(&intent_source, &[b"good-0", b"good-1", b"good-2"]);

        let engine = VeldBackupEngine::new(backup_root.clone()).unwrap();
        let metadata = engine
            .create_comprehensive_backup_with_intent_log(
                &db,
                user_id,
                &[],
                None,
                Some(&spec),
            )
            .unwrap();
        drop(db);

        // Flip a byte inside the archive copy of the journal so its CRC
        // no longer matches. The manifest's SHA-256 will catch the
        // mismatch before we even open the journal.
        let archive_journal = backup_root
            .join(user_id)
            .join(format!("intent_log_{}", metadata.backup_id))
            .join("intent.log");
        let mut bytes = fs::read(&archive_journal).unwrap();
        // Flip a payload-region byte (header is the first 12 bytes).
        let target_idx = bytes.len() / 2;
        bytes[target_idx] ^= 0xff;
        fs::write(&archive_journal, &bytes).unwrap();

        let restore_spec =
            IntentLogBackupSpec::with_default_log_name(intent_restore.clone());
        let err = engine
            .restore_comprehensive_backup_with_intent_log(
                user_id,
                Some(metadata.backup_id),
                &restored_db_path,
                &[],
                Some(&restore_spec),
                &RestoreOptions::default(),
            )
            .unwrap_err();
        // Must surface a clear failure — either SHA mismatch (preferred,
        // catches it earliest) or a frame-level CRC error.
        let msg = err.to_string();
        assert!(
            msg.contains("integrity check") || msg.contains("CRC") || msg.contains("frame"),
            "unexpected error message: {msg}"
        );
        // No partial state should exist at the restore target.
        assert!(!intent_restore.exists());
    }

    #[test]
    fn intent_log_spec_default_log_name() {
        let spec = IntentLogBackupSpec::with_default_log_name(PathBuf::from("/tmp/x"));
        assert_eq!(spec.log_filename, "intent.log");
        assert_eq!(spec.source_dir, PathBuf::from("/tmp/x"));
    }

    #[test]
    fn missing_intent_log_source_dir_yields_empty_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let backup_root = temp_dir.path().join("backups");
        let db_path = temp_dir.path().join("db");
        let missing_intent = temp_dir.path().join("does-not-exist");
        let user_id = "no-intent";

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, &db_path).unwrap();
        db.put(b"k", b"v").unwrap();

        let engine = VeldBackupEngine::new(backup_root).unwrap();
        let spec = IntentLogBackupSpec::with_default_log_name(missing_intent);
        let metadata = engine
            .create_comprehensive_backup_with_intent_log(
                &db,
                user_id,
                &[],
                None,
                Some(&spec),
            )
            .unwrap();
        assert!(metadata.intent_log_files.is_empty());
    }
}
